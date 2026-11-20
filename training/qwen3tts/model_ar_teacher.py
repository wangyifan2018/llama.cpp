"""The autoregressive code predictor, used as the distillation teacher.

This is the *original* 15-step chain, implemented faithfully so the soft targets match what the
deployed runtime computes: 16 positions (the hidden bridge, then code0..code14), 5 layers with GQA
attention, NEOX rope at theta 1e6, per head q/k RMSNorm, causal over the prefix.

Only the teacher's distributions are needed for training; at inference the 1-pass student replaces it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_predictor import (EPS, FF, HEAD_DIM, N_EMBD, N_HEAD_KV, N_HEAD_Q, N_LAYERS, N_BOOKS,
                             RMSNorm, VOCAB)


def _rope_neox(x: torch.Tensor, pos: torch.Tensor, theta: float = 1e6) -> torch.Tensor:
    """x [B, H, T, D], pos [T]; rotate pairs across the two halves (NEOX, as ggml_rope_ext)."""
    d = x.shape[-1]
    half = d // 2
    inv = theta ** (-torch.arange(half, dtype=torch.float32, device=x.device) / half)
    ang = pos.float()[:, None] * inv[None, :]                 # [T, half]
    cos = ang.cos()[None, None, :, :]                          # [1, 1, T, half]
    sin = ang.sin()[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ARLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm()
        self.q = nn.Linear(N_EMBD, N_HEAD_Q * HEAD_DIM, bias=False)
        self.k = nn.Linear(N_EMBD, N_HEAD_KV * HEAD_DIM, bias=False)
        self.v = nn.Linear(N_EMBD, N_HEAD_KV * HEAD_DIM, bias=False)
        self.o = nn.Linear(N_HEAD_Q * HEAD_DIM, N_EMBD, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        self.ln2 = RMSNorm()
        self.gate = nn.Linear(N_EMBD, FF, bias=False)
        self.up = nn.Linear(N_EMBD, FF, bias=False)
        self.down = nn.Linear(FF, N_EMBD, bias=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, t, N_HEAD_Q, HEAD_DIM).transpose(1, 2)
        k = self.k(h).view(b, t, N_HEAD_KV, HEAD_DIM).transpose(1, 2)
        v = self.v(h).view(b, t, N_HEAD_KV, HEAD_DIM).transpose(1, 2)
        q = _rope_neox(self.q_norm(q), pos)
        k = _rope_neox(self.k_norm(k), pos)
        # GQA: each KV head serves two query heads, same grouping as the runtime (i//2)
        k = k.repeat_interleave(N_HEAD_Q // N_HEAD_KV, dim=1)
        v = v.repeat_interleave(N_HEAD_Q // N_HEAD_KV, dim=1)

        attn = q @ k.transpose(-1, -2) / (HEAD_DIM ** 0.5)       # [B, HQ, T, T]
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        attn = attn.masked_fill(~mask, float("-inf")).softmax(-1)
        out = attn @ v
        x = x + self.o(out.transpose(1, 2).reshape(b, t, -1))

        h2 = self.ln2(x)
        return x + self.down(F.silu(self.gate(h2)) * self.up(h2))


class ARTeacher(nn.Module):
    """[B, 1024] talker hidden + [B, 16] true codes -> per book logits [B, 15, 2048]."""

    def __init__(self):
        super().__init__()
        self.proj_in = None                      # optional, only on the 1.7B checkpoint
        self.layers = nn.ModuleList(ARLayer() for _ in range(N_LAYERS))
        self.norm = RMSNorm()
        self.out_embd = nn.Embedding(3072, N_EMBD)          # code0 lives in the talker's table
        self.book_embd = nn.ModuleList(nn.Embedding(VOCAB, N_EMBD) for _ in range(N_BOOKS))
        # near zero so a randomly initialised teacher still produces a usable (non uniform) signal;
        # from_ar_state_dict() overwrites this with the real head weights
        self.heads = nn.Parameter(torch.randn(N_BOOKS, VOCAB, N_EMBD) * 0.02)

    @classmethod
    def from_ar_state_dict(cls, sd: dict[str, torch.Tensor], prefix: str = "talker.code_predictor.") -> "ARTeacher":
        m = cls()
        has_proj = (prefix + "small_to_mtp_projection.weight") in sd
        if has_proj:
            m.proj_in = nn.Linear(N_EMBD, N_EMBD, bias=True)
        with torch.no_grad():
            if has_proj:
                m.proj_in.weight.copy_(sd[prefix + "small_to_mtp_projection.weight"])
                m.proj_in.bias.copy_(sd[prefix + "small_to_mtp_projection.bias"])
            for i, layer in enumerate(m.layers):
                p = f"{prefix}model.layers.{i}."
                layer.ln1.weight.copy_(sd[p + "input_layernorm.weight"])
                layer.ln2.weight.copy_(sd[p + "post_attention_layernorm.weight"])
                layer.q.weight.copy_(sd[p + "self_attn.q_proj.weight"])
                layer.k.weight.copy_(sd[p + "self_attn.k_proj.weight"])
                layer.v.weight.copy_(sd[p + "self_attn.v_proj.weight"])
                layer.o.weight.copy_(sd[p + "self_attn.o_proj.weight"])
                layer.q_norm.weight.copy_(sd[p + "self_attn.q_norm.weight"])
                layer.k_norm.weight.copy_(sd[p + "self_attn.k_norm.weight"])
                layer.gate.weight.copy_(sd[p + "mlp.gate_proj.weight"])
                layer.up.weight.copy_(sd[p + "mlp.up_proj.weight"])
                layer.down.weight.copy_(sd[p + "mlp.down_proj.weight"])
            m.norm.weight.copy_(sd[prefix + "model.norm.weight"])
            m.out_embd.weight.copy_(sd["talker.model.codec_embedding.weight"])
            for g in range(N_BOOKS):
                m.book_embd[g].weight.copy_(sd[f"{prefix}model.codec_embedding.{g}.weight"])
                m.heads[g].copy_(sd[f"{prefix}lm_head.{g}.weight"])
        return m

    def forward(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        x = h.new_zeros(b, 16, N_EMBD)
        x[:, 0] = h
        x[:, 1] = self.out_embd(codes[:, 0])
        for t in range(2, 16):
            x[:, t] = self.book_embd[t - 2](codes[:, t - 1])
        if self.proj_in is not None:
            x = self.proj_in(x)
        pos = torch.arange(16, device=h.device)  # one position per sequence slot, same for every row
        for layer in self.layers:
            x = layer(x, pos)
        x = self.norm(x)
        # book g (0..14) is predicted at position g+1 by head[g]
        hh = x[:, 1:16]                          # [B, 15, 1024]
        return torch.einsum("b t d, t v d -> b t v", hh, self.heads)


def kd_loss(student: torch.Tensor, teacher: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """per book KL(teacher || student) at temperature T, averaged with the same weights as the CE."""
    total = student.new_zeros(())
    wsum = 0.0
    for g in range(N_BOOKS):
        w = 1.0 if g < 4 else 0.5
        p_t = (teacher[:, g, :] / T).log_softmax(-1)
        p_s = (student[:, g, :] / T).log_softmax(-1)
        total = total + w * F.kl_div(p_s, p_t, log_target=True, reduction="batchmean") * (T * T)
        wsum += w
    return total / wsum
