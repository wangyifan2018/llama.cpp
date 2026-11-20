"""1-pass code predictor: 5 FFN layers + 15 parallel heads.

This is the P2a target shape. It mirrors exactly what the runtime builds in the TTS_FAKE_NAR branch
of tools/mtmd/models/qwen3tts-gen.cpp, so the trained tensors can be written back without changing
the graph:

  cur = proj_in(h)                                   # 1024 -> 1024, with bias
  for layer in 5:
      cur += o_proj(tile(v_proj(rmsnorm(cur) * ln1))) # tile: each KV head serves 2 query heads
      cur += down(silu(gate(rmsnorm(cur) * ln2)) * up(...))
  cur = rmsnorm(cur) * norm
  logits[g] = head[g] @ cur        for g in 0..14

the attention in the runtime is over a single position, so softmax is a constant 1 and the output
per query head is just v. q/k/rope are computed by the runtime but cannot affect the result, which is
why this model has no q/k at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EMBD = 1024
N_HEAD_KV = 8
HEAD_DIM = 128
N_HEAD_Q = 16
N_LAYERS = 5
N_BOOKS = 15
VOCAB = 2048
FF = 3072
EPS = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, n: int = N_EMBD):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.weight


class TowerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm()
        self.v_proj = nn.Linear(N_EMBD, N_HEAD_KV * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEAD_Q * HEAD_DIM, N_EMBD, bias=False)
        self.ln2 = RMSNorm()
        self.gate = nn.Linear(N_EMBD, FF, bias=False)
        self.up = nn.Linear(N_EMBD, FF, bias=False)
        self.down = nn.Linear(FF, N_EMBD, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v_proj(self.ln1(x))                                  # [B, 1024]
        v = v.view(*v.shape[:-1], N_HEAD_KV, HEAD_DIM)
        v = v.repeat_interleave(N_HEAD_Q // N_HEAD_KV, dim=-2)        # GQA 2:1, same as the runtime tiling
        x = x + self.o_proj(v.flatten(-2))
        h = self.ln2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class MTPStylePredictor(nn.Module):
    def __init__(self, has_proj_in: bool = False):
        super().__init__()
        # the 0.6B checkpoint has no small_to_mtp_projection (predictor hidden == talker hidden),
        # the 1.7B one does; the runtime treats a missing gen_code_proj_in_w as identity
        self.proj_in = nn.Linear(N_EMBD, N_EMBD, bias=True) if has_proj_in else None
        self.layers = nn.ModuleList(TowerLayer() for _ in range(N_LAYERS))
        self.norm = RMSNorm()
        # same small init as the teacher: from_ar_state_dict() replaces it with the real weights
        self.heads = nn.Parameter(torch.randn(N_BOOKS, VOCAB, N_EMBD) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 1024] talker hidden -> logits [B, 15, 2048]"""
        x = self.proj_in(h) if self.proj_in is not None else h
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return torch.einsum("b d, g v d -> b g v", x, self.heads)

    @classmethod
    def from_ar_state_dict(cls, sd: dict[str, torch.Tensor], prefix: str = "talker.code_predictor.") -> "MTPStylePredictor":
        """Initialize from the autoregressive checkpoint.

        Every tensor keeps its identity: the AR layer's v_proj/o_proj/mlp/norms are exactly the ones
        the 1-pass tower needs, only q/k (and their norms) are dropped. Only position 0's weights are
        used -- the other positions existed to carry the AR chain.
        """
        has_proj_in = (prefix + "small_to_mtp_projection.weight") in sd
        m = cls(has_proj_in=has_proj_in)
        with torch.no_grad():
            if has_proj_in:
                m.proj_in.weight.copy_(sd[prefix + "small_to_mtp_projection.weight"])
                m.proj_in.bias.copy_(sd[prefix + "small_to_mtp_projection.bias"])
            for i, layer in enumerate(m.layers):
                p = f"{prefix}model.layers.{i}."
                layer.ln1.weight.copy_(sd[p + "input_layernorm.weight"])
                layer.ln2.weight.copy_(sd[p + "post_attention_layernorm.weight"])
                layer.v_proj.weight.copy_(sd[p + "self_attn.v_proj.weight"])
                layer.o_proj.weight.copy_(sd[p + "self_attn.o_proj.weight"])
                layer.gate.weight.copy_(sd[p + "mlp.gate_proj.weight"])
                layer.up.weight.copy_(sd[p + "mlp.up_proj.weight"])
                layer.down.weight.copy_(sd[p + "mlp.down_proj.weight"])
            m.norm.weight.copy_(sd[prefix + "model.norm.weight"])
            for g in range(N_BOOKS):
                m.heads[g].copy_(sd[f"{prefix}lm_head.{g}.weight"])
        return m


# ---- metrics -------------------------------------------------------------------------------

def per_book_loss(logits: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """weighted CE, one term per codebook. codes[:, 1:] are the 15 acoustic books."""
    total = logits.new_zeros(())
    weight_sum = 0.0
    for g in range(N_BOOKS):
        w = 1.0 if g < 4 else 0.5
        total = total + w * F.cross_entropy(logits[:, g, :], codes[:, g + 1])
        weight_sum += w
    return total / weight_sum


@torch.no_grad()
def agreement(logits: torch.Tensor, codes: torch.Tensor) -> list[float]:
    """top-1 match rate per codebook against the teacher codes."""
    pred = logits.argmax(-1)
    return [(pred[:, g] == codes[:, g + 1]).float().mean().item() for g in range(N_BOOKS)]


@torch.no_grad()
def book_entropy(logits: torch.Tensor) -> list[float]:
    p = logits.softmax(-1)
    return [(-(p[:, g] * p[:, g].clamp_min(1e-12).log()).sum(-1).mean().item()) for g in range(N_BOOKS)]
