# ONNX reference analysis: X-ASR 480 ms streaming Zipformer2, int8 export

Host analysis of the ONNX models that ship with the sherpa-onnx X-ASR package, plus a
same-host CPU A/B against this fork's GGML encoder. No Android device was touched. Every
number below is from macOS on an Apple M2 (8 cores), python 3.9, onnx 1.15.0,
onnxruntime 1.18.0, numpy 1.23.0.

Device numbers are quoted only from the checked-in docs (examples/x-asr/BENCHMARK.md,
examples/x-asr/OPTIMIZATION.md) and are marked as such. Host CPU numbers are marked HOST
and are used only to rank work and compare lowerings, not to predict device latency.

Paths used in every command below:

    MODEL=/Users/wangyifan/Library/Caches/qwen3-asr-htp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
    GGUF=/Users/wangyifan/Library/Caches/qwen3-asr-htp/xasr-multistream/x-asr-zh-en-q8_0.gguf
    REPO=/Users/wangyifan/Desktop/llama.cpp

All commands are reproduced in full in section 8.

---

## 0. Headline: the same-host A/B, and what it says about the device gap

HOST, same 10.053 s fixture (test_wavs/0.wav), same 61-frame / 48-frame-advance
chunking, same greedy decode, one fresh process per run:

| Path | Threads | Wall | RTF | Encoder | Predictor+joiner | Per chunk |
|---|---:|---:|---:|---:|---:|---:|
| ONNX Runtime 1.18, CPU EP | 1 | 1694.0 ms | 0.169 | 1620.0 ms (22 chunks, 73.6 ms/chunk) | 59.1 ms (264 joiner + 27 decoder calls) | 77.0 ms |
| ONNX Runtime 1.18, CPU EP | 2 | 1769.3 ms | 0.176 | 1682.1 ms (22 chunks, 76.5 ms/chunk) | 80.3 ms | 79.7 ms |
| this fork, CPU backend, llama-x-asr | 2 | 3221.8 ms | 0.321 | 2306.4 ms (23 chunks, 100.3 ms/chunk) | 897.7 ms (264 frames) | 139.3 ms |

Gap decomposition (fork minus ORT-1-thread, 3221.8 - 1694.0 = 1527.8 ms): 686 ms
encoder + 839 ms predictor/joiner + 12 ms of fork-side fbank - 9 ms of harness overhead
(the ONNX loop times feature extraction outside its measured window). So on this host
55 percent of the gap is the host-side predictor/joiner loop and 45 percent is the
encoder.

Per frame the predictor+joiner costs 3.40 ms in the fork versus 0.21 ms in ORT (16x).
Per chunk the encoder costs 100.3 ms in the fork on 2 threads versus 73.6 ms in ORT on
1 thread, so ORT's single thread does 36 percent more encoder work per second than the
fork gets out of two threads.

Two measured causes for the joiner gap:

1. Bytes per call. xasr.cpp materialises join.output_linear.weight (512 x 5000, GGUF
   Q8_0, 2,720,000 B) and dec.emb.weight into host float vectors with to_f32(), so each
   frame streams 10.24 MB of f32 instead of 2.72 MB of q8_0. ONNX keeps the weight int8
   (2,560,000 B) and reads 2.56 MB per call.
2. Working-set size. 2.56 MB int8 stays resident across the 264 calls of a stream;
   10.24 MB f32 does not. Measured weight-byte throughput: ORT 12.2 GB/s of joiner time,
   fork 3.0 GB/s.

The encoder gap is not plumbing: the fork's graph is 2877 nodes (printed by the binary)
against 6112 ORT kernels for the same chunk, so the fork issues half as many kernels and
is still slower. ORT's fused DynamicQuantizeMatMul runs at 19.7 GMAC/s while its fp32
Conv runs at 3.7 GMAC/s (section 2.4). On the CPU path the fork has no operator-level
int8 matmul: it relies on ggml_mul_mat with q8_0 weights, whose per-32-block
dequantise-and-dot is a different kernel from an integer dot product.

---

## 1. Graph inventory

Command: section 8, C1.

| Model | File bytes | Initialiser bytes | Parameters | Nodes | Initialisers |
|---|---:|---:|---:|---:|---:|
| encoder.int8.onnx | 155,278,641 | 152,603,040 | 147,852,138 | 12,261 | 1,686 |
| decoder.onnx | 11,309,084 | 11,307,008 | 2,826,752 | 17 | 4 |
| joiner.int8.onnx | 2,581,422 | 2,580,005 | 2,565,002 | 8 | 4 |

Bytes by dtype:

| Model | float32 | int8 |
|---|---:|---:|
| encoder.int8.onnx | 6,334,536 B (998 tensors) | 146,268,504 B (688 tensors) |
| decoder.onnx | 11,307,008 B (4 tensors) | 0 |
| joiner.int8.onnx | 20,004 B (2 tensors) | 2,560,001 B (2 tensors) |

Encoder bytes outside initialisers: 155,278,641 - 152,603,040 = 2,675,601 B of protobuf
(12,261 nodes, 10,800 value_info entries).

IR version 7, opset 13 for all three. Producers: onnx.quantize 0.1.0 (encoder, joiner),
pytorch 2.6.0 (decoder).

### 1.1 Encoder metadata_props (verbatim)

    onnx.infer=onnxruntime.quant   model_type=zipformer2   version=1
    model_author=k2-fsa            comment=streaming zipformer2
    decode_chunk_len=48            T=61
    num_encoder_layers=2,2,4,5,4,2
    encoder_dims=192,256,512,768,512,256
    cnn_module_kernels=31,31,15,15,15,31
    left_context_len=256,128,64,32,64,128
    query_head_dims=32,32,32,32,32,32
    value_head_dims=12,12,12,12,12,12
    num_heads=4,4,4,8,4,4

decoder: context_size=2, vocab_size=5000. joiner: joiner_dim=512.

Chunk arithmetic implied by these numbers and confirmed by running the encoder (C1):
61 input frames of 80 mel bins -> Conv2dSubsampling yields 27 time frames x 19 frequency
bins (measured, see 3c) -> ConvNeXt consumes 3 cached plus 27 new frames and emits 24 ->
output downsample 2 -> encoder_out = (1, 12, 512). One 480 ms chunk therefore emits 12
joiner frames, not 24.

### 1.2 Encoder inputs and outputs

117 inputs, 117 outputs. Input 0 is x, float32 [N, 61, 80]; the last two are
embed_states, float32 [N, 128, 3, 19], and processed_lens, int64 [N]. The other 114 are
19 layers x 6 streaming caches:

| Input | Shape | Meaning |
|---|---|---|
| cached_key_i | [left_i, N, 32 x H] | attention key cache; left_i is left_context_len_i (256/128/64/32), last axis is 128 at H=4 heads and 256 at H=8 |
| cached_val1_i | [left_i, N, 12 x H] | value cache 1, 48 at H=4 and 96 at H=8 |
| cached_val2_i | [left_i, N, 12 x H] | value cache 2 |
| cached_nonlin_attn_i | [1, N, left_i, 144/192/384/576] | nonlinear-attention cache; last axis per stage, as measured on the input list |
| cached_conv1_i | [N, dim_i, 15 or 7] | causal depthwise conv history |
| cached_conv2_i | [N, dim_i, 15 or 7] | chunkwise depthwise conv history |

Outputs are encoder_out [N, A, 512], new_cached_* (the same 114, axis 0 symbolic),
new_embed_states [N, 128, A, 19] and new_processed_lens int64 [N]. Cache bytes per
direction: 4,148,736 B, matching what the fork's prepare_graph prints.

Only the batch axis N is symbolic on the inputs. Every time axis is static (61, 256,
15, 3). That is worth noting against section 3e: the static time axes did not remove the
per-inference shape recomputation.

---

## 2. Operator census and where the time goes

Commands: section 8, C1 (census), C5 and C6 (profile).

### 2.1 Full census, encoder.int8.onnx (12,261 nodes)

    Constant 4073  Mul 1211  Add 916  Unsqueeze 844  Shape 574  Cast 556
    Gather 511  Slice 463  Concat 427  Sub 418  MatMulInteger 344
    DynamicQuantizeLinear 312  Transpose 269  Reshape 265  Where 158  Exp 119
    Equal 101  Log 99  MatMul 95  Conv 82  Expand 63  Div 63  Neg 42  Pow 40
    Range 38  Sigmoid 38  ConstantOfShape 29  ReduceMean 20  Tile 19
    GatherElements 19  Softmax 19  Tanh 19  ReduceSum 6  Max 4  Abs 4  LessOrEqual 1

decoder.onnx (17 nodes): Constant 4, Cast 2, Transpose 2, Clip 1, Gather 1,
GreaterOrEqual 1, Unsqueeze 1, Mul 1, Conv 1, Relu 1, Squeeze 1, Gemm 1.

joiner.int8.onnx (8 nodes): Add 2, Mul 2, Tanh 1, DynamicQuantizeLinear 1,
MatMulInteger 1, Cast 1.

### 2.2 Grouped view of the encoder

| Group | Count | Detail |
|---|---:|---|
| Conv | 82 | 77 depthwise (group>1), 3 subsampling 3x3, 2 pointwise 1x1 |
| MatMul family | 439 | 344 MatMulInteger (int8 weights) + 95 MatMul (fp32 attention) |
| Gemm | 0 | the only Gemm is in the decoder |
| LSTM / GRU | 0 | none; the conv module is explicit depthwise convolution |
| Attention | 19 Softmax + 95 MatMul + 19 Tanh + 38 Sigmoid + 19 GatherElements | no fused Attention op |
| Transpose / Reshape / Concat / Slice | 1,424 | plus 844 Unsqueeze, 0 Squeeze |
| Shape plumbing | 574 Shape + 511 Gather + 556 Cast | 4,073 Constant nodes feed these |
| Normalisation | 20 norms | 20 ReduceMean + 40 Pow + 20 Sub + 60 Mul |
| Activations | 99 Swoosh (99 Exp + 99 Log), 38 Sigmoid, 19 Tanh, 19 Softmax | see 3f |

The dominant structural fact is plumbing: 4,073 Constant + 574 Shape + 511 Gather + 463
Slice + 427 Concat + 844 Unsqueeze + 265 Reshape + 269 Transpose + 158 Where + 101 Equal
= 7,685 of 12,261 nodes (63 percent) compute or move indices, shapes and masks rather
than arithmetic.

### 2.3 HOST profile, one encoder inference, intra_op_num_threads=2

Command: section 8, C5. Inputs are zeros built from the declared shapes (x [1,61,80],
all caches 0). The profiler inflates absolute time: the same call takes 76.5 ms of wall
time unprofiled at 2 threads and 73.6 ms at 1 thread, while the profiled run sums to
144.0 ms over 6,112 kernels. Three profiled runs summed to 163.8, 212.5 and 144.0 ms, so
inflation is 2-3x and noisy. Use the shares and the ranking only.

| Category | ms | share | kernels |
|---|---:|---:|---:|
| int8 MatMul, dynamic-quant (DynamicQuantizeMatMul + MatMulIntegerToFloat) | 41.52 | 28.8% | 344 |
| Shape/layout movement (Shape, Gather, Slice, Concat, Unsqueeze, Reshape, Transpose, Cast, Range, Expand, Tile, GatherElements, ReduceMean, ReduceSum, ConstantOfShape, LessOrEqual) | 37.69 | 26.2% | 3,086 |
| Conv fp32 | 19.79 | 13.7% | 82 |
| Add | 12.35 | 8.6% | 916 |
| Sub | 5.78 | 4.0% | 418 |
| Where (masking) | 5.48 | 3.8% | 158 |
| MatMul fp32 | 4.79 | 3.3% | 95 |
| Mul | 4.67 | 3.3% | 460 |
| Log (Swoosh softplus tail) | 2.79 | 1.9% | 99 |
| Exp | 2.60 | 1.8% | 99 |
| Equal (mask predicate) | 1.79 | 1.2% | 101 |
| Sigmoid (GLU in the 38 conv modules) | 1.00 | 0.7% | 38 |
| Pow (LayerNorm variance) | 0.99 | 0.7% | 40 |
| Div | 0.56 | 0.4% | 63 |
| Neg | 0.55 | 0.4% | 42 |
| DynamicQuantizeLinear (unfused remainder) | 0.31 | 0.2% | 25 |
| Tanh (nonlinear attention) | 0.27 | 0.2% | 19 |
| Softmax | 0.27 | 0.2% | 19 |
| Max 0.12 + Abs 0.66 | 0.78 | 0.5% | 8 |

Top 15 nodes by profiled time:

| ms | share | ORT kernel | node |
|---:|---:|---|---|
| 2.71 | 1.88% | Conv | /Conv (ConvNeXt 7x7 depthwise) |
| 2.17 | 1.50% | Conv | /pointwise_conv1/Conv |
| 1.68 | 1.17% | Conv | /conv/conv.7/Conv (subsampling 3x3, 32->128) |
| 1.61 | 1.12% | DynamicQuantizeMatMul | /feed_forward2/in_proj_4/MatMul_quant |
| 1.50 | 1.04% | DynamicQuantizeMatMul | /in_proj_63/MatMul_quant |
| 1.29 | 0.90% | Conv | /pointwise_conv2/Conv |
| 1.24 | 0.86% | MatMulIntegerToFloat | /feed_forward1/in_proj_11/MatMul_quant |
| 1.24 | 0.86% | MatMul | /MatMul_83 |
| 1.11 | 0.77% | Gather | /Gather_114 (shape scalar) |
| 0.95 | 0.66% | Where | /feed_forward2/out_proj_18/Where |
| 0.92 | 0.64% | MatMulIntegerToFloat | /feed_forward1/in_proj_4/MatMul_quant |
| 0.87 | 0.60% | DynamicQuantizeMatMul | /feed_forward2/in_proj_5/MatMul_quant |
| 0.86 | 0.60% | Equal | /feed_forward1/out_proj_3/Equal |
| 0.84 | 0.58% | Conv | /causal_conv_10/Conv |
| 0.84 | 0.58% | DynamicQuantizeMatMul | /in_proj_33/MatMul_quant |

No single node is a hot spot: the maximum is 1.9 percent. Cost is spread over 6,112
small kernels.

### 2.4 Arithmetic actually performed per chunk

Command: section 8, C6 (MAC counts derived from the input/output shapes that ORT records
per kernel).

| Kernel class | GMAC | profiled ms | GMAC/s |
|---|---:|---:|---:|
| DynamicQuantizeMatMul (int8) | 0.663 | 33.57 | 19.74 |
| MatMulIntegerToFloat (int8) | 0.077 | 7.95 | 9.70 |
| Conv (fp32, 82) | 0.073 | 19.79 | 3.70 |
| MatMul (fp32, 95) | 0.011 | 4.79 | 2.24 |
| Total | 0.824 | 144.01 | 11.44 GFLOP/s |

0.824 GMAC = 1.65 GFLOP per 480 ms chunk, i.e. 3.4 GFLOP/s for real time. The int8
matmuls carry 90 percent of the MACs at 5x the throughput of the fp32 convs. The 82
convs carry 8.9 percent of the MACs but 13.7 percent of the time. The depthwise convs are
the worst class: one MAC per weight, so they are latency bound, not throughput bound.

The other 5,591 kernels (91.5 percent of all kernels) perform no MACs at all and take
77.9 ms, 54.1 percent of the profiled time; 37.7 ms of that is shape/layout work alone.

---

## 3. Layout and lowering evidence

Commands: section 8, C3 (Conv attributes and weights) and C4.

### 3a. ConvNeXt 7x7 depthwise convolution

ONNX has no DepthwiseConv. The export lowers it to a single Conv with group = channels:

    node    /Concat  embed_states [N,128,3,19] + conv.9 out [N,128,27,19] -> [N,128,30,19]
    node    /Conv
    input   /Concat_output_0                              [N, 128, 30, 19]
    weight  encoder_embed.convnext.depthwise_conv.weight  [128, 1, 7, 7] float32
    attrs   group=128  kernel_shape=[7,7]  strides=[1,1]  pads=[0,3,0,3]  dilations=[1,1]
    output  /Conv_output_0                                [N, 128, 24, 19]

For a 4-D input, pads holds the spatial only form [H_begin, W_begin, H_end, W_end], so
[0,3,0,3] is zero padding of 3 on each side of W (frequency) and none on H (time). The
measured shapes confirm it: H 30 - 7 + 1 = 24, W 19 + 6 - 7 + 1 = 19. The 7x7 kernel
therefore spans 7 consecutive time frames and 7 of the 19 frequency bins, and the time
axis is valid (no padding), so the receptive field of output frame k is input frames
k..k+6. The 3-frame cache supplies the left part for the first outputs, and the 3 extra
frames that the 61-frame input window produces (27 for a 24-frame chunk) supply the right
part. This is exactly the call ggml_conv_2d_dw_direct(..., s=1,1, p_freq=3, p_time=0,
d=1,1) at xasr.cpp:634.

### 3b. The two pointwise projections

    /pointwise_conv1/Conv  weight [384, 128, 1, 1]  group=1  128 -> 384  (GLU halves it back to 192)
    /pointwise_conv2/Conv  weight [128, 384, 1, 1]  group=1  384 -> 128

They are plain fp32 Conv with a 1x1 kernel, not MatMul: the exporter kept the NCHW
layout so it could use the Conv kernel directly. The fork reshapes the same weights to
2-D and calls ggml_mul_mat (xasr.cpp:638 and 640), which is why it needs the
permute + cont pair at xasr.cpp:636 and 642.

### 3c. Conv2d subsampling at the front

    /conv/conv.0/Conv  weight [8, 1, 3, 3]    stride [1,1]  pads [0,1,0,1]  [1,1,61,80] -> [1,8,59,80]
    /conv/conv.4/Conv  weight [32, 8, 3, 3]   stride [2,2]  pads [0,0,0,0]               -> [1,32,29,39]
    /conv/conv.7/Conv  weight [128, 32, 3, 3] stride [1,2]  pads [0,0,0,0]               -> [1,128,27,19]

All three are dense fp32 Conv with fused bias (the bias is the third input), each
followed by a SwooshR chain (3f). The stride pattern is (time, freq) = (1,1), (2,2),
(1,2), so time goes 61 -> 59 -> 29 -> 27 (floor division on the strided conv) and
frequency 80 -> 80 -> 39 -> 19, which is the F' = 19 in embed_states [N, 128, 3, 19].
The 27 time frames are chunk + 3 = 27, matching the fork's Tc == G.chunk + 3 assertion at
xasr.cpp:626: 24 frames of the chunk plus 3 lookahead frames that the ConvNeXt 7x7 valid
convolution consumes.

### 3d. Attention projection

Every attention projection is int8 MatMulInteger, never Conv, never Gemm:

    /in_proj_63/MatMul_quant    A [1, 3, 768]   B int8 [768, 1728]
    /out_proj_.../MatMul_quant  A [1, 3, 768]   B int8 [1728, 768]
    /linear_pos/MatMul_quant    A [1, 37, 48]   B int8 [48, 32]
    /encoder_proj/MatMul_quant  A [1, 24, 2432] B int8 [2432, 192]
    /out/MatMul_quant           A [1, 24, 2432] B int8 [2432, 192]

The projection widths are the concatenated per-head blocks. At the dim-768 stack
(8 heads, query head dim 32, value head dim 12, position head dim 4) the measured
in-projections are [768,544] (= 8 x (32 + 32 + 4), query/key/position),
[768,96] (= 8 x 12, value) and [768,1728] (= 3 x 576, the nonlinear-attention
in-projection). The only fp32 MatMul nodes are the 95 that form the scores and apply the
attention weights, for example /MatMul_48: [8,1,3,35] x [8,1,35,12] -> [8,1,3,12]
(8 heads, 3 query frames, 35 cached+current keys, 12 value head dim).

### 3e. Transpose / Reshape / Unsqueeze / Squeeze

Counts: Transpose 269, Reshape 265, Unsqueeze 844, Squeeze 0. Measured producers and
consumers of Transpose:

    perms: (2,1,0,3) 152 | (1,2,0) 38 | (2,0,1) 38 | (2,1,3,0) 19 | (2,0,3,1) 19 | (0,2,1,3) 1 | (1,0,2) 2
    producers: Reshape 133 | Add 39 | Mul 39 | MatMul 57 | ReduceSum 1
    consumers: MatMul 76 | Reshape 58 | Concat 19 | MatMul+Shape 38 | Shape+Where 38 | Mul+Sub 38 | Slice 1 | DynamicQuantizeLinear 1

All 269 are real data movement, not views: ONNX has no transpose view, and ORT spends
3.90 ms (2.7 percent) in its Transpose kernel. The classes:

- 133 Reshape -> Transpose -> Reshape: per-layer layout round trips. The one that matters
  is the ConvNeXt output: /Transpose perm=[0,2,1,3] converts NCHW [1,128,24,19] to
  [1,24,128,19] (NCW -> NWC) and /Reshape flattens to [24, 2432] for the int8 out/MatMul.
  That is a layout conversion, and the fork performs the same one with ggml_permute +
  ggml_cont at xasr.cpp:645-646.
- 76 Transposes feeding MatMul: attention score layout ([N,T,H,d] -> [N,H,T,d] style),
  again a heads-as-batch layout conversion for GEMM.
- 38 + 38 + 38 Transposes whose consumers are Shape/Where or Mul/Sub: mask construction
  and the nonlinear-attention branch, pure layout.
- 2 leading (1,0,2) transposes: cache axis swap for the streaming slice.

Unsqueeze is 844 nodes, 689 of which feed Concat and 96 feed Slice: they build 1-D
shape/start/end vectors because every Slice and Concat in the streaming path takes a
dynamic start index. 511 Gather inputs come from 574 Shape nodes, i.e. the graph
recomputes the current chunk length from tensor shapes on every inference. Reshape is
mostly shape-vector application (265 nodes, 133 immediately before a Transpose).

Measured consequence in 2.3: this plumbing is 26 percent of ORT's kernel time. The fork
does not pay it (static graph, 2877 nodes), so the fork's deficit is entirely in the
arithmetic kernels, not in shape handling.

### 3f. Where the Swoosh activations are

Each activation is 12 primitive ops: Sub, Mul, Max, Sub, Abs, Neg, Exp, Add, Log, Add,
Sub, Sub. Constants for the subsampling chain (/conv/conv.3/): offset 1.0, slope 0.08,
subtract 0.3132616877555847, i.e. SwooshR(x) = softplus(x - 1) - 0.08x - 0.3132617.
For the ConvNeXt chain (/activation/): offset 4.0, slope 0.08, subtract 0.035, i.e.
SwooshL. 99 Exp/Log pairs exist, distributed as 20 norm modules, 57 feed-forward
modules, 19 attention out-projections, 3 subsampling stages and 1 ConvNeXt block.
The fork calls fused helpers for these (swoosh_r, swoosh_l at xasr.cpp:383-392), so this
is a case where the fork is already ahead: 99 x 12 = 1,188 ONNX nodes versus 99 ggml
nodes.

LayerNorm is likewise primitive: /out_norm is Sub(x, bias), Pow 2, ReduceMean axis 2,
Pow -0.5, Mul(exp(log_scale)), Mul, with a learned scale exp(log_scale) and no mean
subtraction (the mean is folded into bias). 20 such norms exist (19 layers plus the
embed out_norm). The fork's bias_norm (xasr.cpp:395) is the same formula.

---

## 4. Quantization format

Command: section 8, C4.

| Op | encoder | joiner | decoder |
|---|---:|---:|---:|
| QuantizeLinear | 0 | 0 | 0 |
| DequantizeLinear | 0 | 0 | 0 |
| QLinearConv | 0 | 0 | 0 |
| QLinearMatMul | 0 | 0 | 0 |
| ConvInteger | 0 | 0 | 0 |
| MatMulInteger | 344 | 1 | 0 |
| DynamicQuantizeLinear | 312 | 1 | 0 |
| MatMul / Gemm / Conv | 95 / 0 / 82 | 0 / 0 / 0 | 0 / 1 / 1 |

This export is operator-level integer MatMul only. There is no QDQ pair anywhere, no
QLinearConv, no ConvInteger. The pattern per projection is:

    DynamicQuantizeLinear(x) -> x_u8, x_scale (scalar f32), x_zero (scalar u8)
    MatMulInteger(x_u8, W_int8, x_zero, W_zero=0) -> int32
    Cast(int32 -> f32); Mul(., x_scale * W_scale); Add(., bias)

ORT fuses that 5-node pattern into one kernel: 287 DynamicQuantizeMatMul plus 57
MatMulIntegerToFloat = 344, and only 25 DynamicQuantizeLinear kernels survive outside a
fusion (section 2.3).

Granularity, measured over all 344 MatMulInteger nodes:

- Weight scale: float32, per-tensor, shape (), 344 of 344. Values are scalars, e.g.
  0.024594994261860847 for the joiner.
- Weight zero point: int8, shape (), value 0 for all 344 (symmetric).
- Weight data: int8 for all 344 (onnx::MatMul_*_quantized).
- Activation: uint8, per-tensor dynamic quantisation recomputed inside the graph on every
  inference (scalar scale, scalar uint8 zero point). 312 DynamicQuantizeLinear nodes for
  344 matmuls means 32 groups share one quantised activation: one quantised tensor feeds
  several MatMuls, e.g. /Slice_3_output_0_quantized feeds both /in_proj/MatMul_quant and
  /feed_forward1/in_proj/MatMul_quant.
- 43 further float32 scales exist with 1-D shapes ((192,), (256,), (512,), (768,)). None
  is referenced by a MatMulInteger zero point and no Conv is quantised, so they are
  unused leftovers of the quantiser.

Conv stays fp32: 82 Conv nodes, 2,717,632 B of fp32 weight+bias, 42.9 percent of the
encoder's 6,334,536 fp32 bytes. The encoder is therefore a hybrid: int8 for every linear
projection (146.3 MB), fp32 for every convolution (2.7 MB) and for norm/bias/scale vectors
(3.4 MB).

Per-model precision, and what dominates the bytes:

| Model | Precision | Bytes dominated by |
|---|---|---|
| decoder.onnx | fp32 throughout | decoder.embedding.weight [5000,512] = 10,240,000 B = 90.6 percent of 11.3 MB; decoder_proj.weight [512,512] = 1,048,576 B = 9.3 percent; conv 16,384 B and bias 2,048 B are noise |
| joiner.int8.onnx | int8 weight, fp32 bias/scale | output_linear.weight_quantized [512,5000] = 2,560,000 B = 99.2 percent of 2.58 MB; bias 20,000 B |
| encoder.int8.onnx | int8 linear, fp32 conv/norm | 146,268,504 B int8 (95.9 percent) versus 6,334,536 B fp32 |

GGUF counterpart (section 8, C2): 966 tensors, 168,064,520 B total; Q8_0 160,488,704 B in
298 tensors, F32 6,371,592 B in 619, F16 1,204,224 B in 49. join.output_linear.weight is
Q8_0 [512,5000] = 2,720,000 B and dec.emb.weight is Q8_0 [512,5000] = 2,720,000 B, so the
GGUF stores the same 8-bit granularity as the ONNX export, and finer: Q8_0 has one scale
per 32 weights, ONNX has one scale per tensor. The fork then calls to_f32() on both
(xasr.cpp:366 and 370), expanding each to 10.24 MB of host f32. That single decision is
the mechanical cause of the 16x joiner gap in section 0.

---

## 5. Predictor and joiner

### 5.1 Decoder graph, exactly as expressed

    y int64 [N,2]
     0 Cast            y -> int64 (no-op)
     1 Constant        0
     2 Clip            y, min=0            -> y_clamped
     3 Gather          embedding.weight [5000,512] x y_clamped -> [N,2,512]
     4 Constant        0
     5 GreaterOrEqual  y >= 0              -> bool [N,2]
     6 Constant        axis -1
     7 Unsqueeze       -> [N,2,1]
     8 Cast            to float            -> [N,2,1]
     9 Mul             embedding * mask    -> [N,2,512]  (only y<0 rows are zeroed)
    10 Transpose       perm [0,2,1]        -> [N,512,2]
    11 Conv            weight [512,4,2], group=128, kernel [2], stride [1], pads [0,0], no bias
                                           -> [N,512,1]
    12 Transpose       perm [0,2,1]        -> [N,1,512]
    13 Relu
    14 Constant        axis 1
    15 Squeeze         -> [N,512]
    16 Gemm            weight [512,512], bias [512], transB=1, alpha=1, beta=1
                                           -> decoder_out [N,512]

So the predictor is: embedding lookup (zeroed for negative context slots), grouped 1-D
convolution with kernel 2 and 128 groups (4 input channels per group), ReLU, dense 512x512
projection with bias. No hidden layer, no normalisation.

### 5.2 Joiner graph

     0 Add              encoder_out [N,512] + decoder_out [N,512] -> [N,512]
     1 Tanh
     2 DynamicQuantizeLinear -> u8 + scalar scale + u8 zero point
     3 Mul              activation_scale * weight_scale (0.02459499)
     4 MatMulInteger    t_u8 x output_linear.weight_quantized [512,5000] int8, W_zero=0 -> int32
     5 Cast             to f32
     6 Mul              by the combined scale
     7 Add              output_linear.bias [5000] -> logit [N,5000]

Add, tanh, one 5000-wide projection, bias. No hidden layer, no ReLU, no second projection;
joiner_dim 512 is both the input and the output width, matching xasr.cpp
(t = tanh(e + d), then V dot products of length J).

Verification (section 8, C9): a numpy re-implementation of the graph above, with the int8
weights dequantised as w_int8 * 0.024594994261860847, reproduces ORT's joiner output to
8.6e-2 max absolute error on N(0,1) inputs. The residual is the activation-side uint8
quantisation (the emulation used fp32 activations), which confirms the graph is exactly
add/tanh/linear/bias with nothing hidden. The same check on the decoder, with a numpy
grouped conv in f32, gives max absolute error 4.8e-7 for tokens (0,0), 4.8e-7 for (5,7)
and 6.0e-7 for (-1,3), i.e. float-level agreement, so the decoder graph is exactly the
grouped-conv formula.

### 5.3 Comparison with xasr.cpp

decoder_out (xasr.cpp:809) copies dec_emb rows for tok >= 0 (leaves zeros otherwise),
runs the grouped conv with gs = dec_conv.size() / (D*Kc) = 4 input channels per group,

    h[o] = sum_i sum_k dec_conv[(o*gs + i)*Kc + k] * e[k*D + g0 + i]

then ReLU, then dec_proj + bias. The GGUF tensor is dec.conv.weight [2,4,512], whose ggml
linear order (k + 2i + 8o) equals the ONNX [512,4,2] order (o, i, k), so the indexing
agrees. The ONNX mask is y >= 0 (Constant_1 is 0, not 1; the Clip minimum is also 0),
which is the same convention as tok >= 0: negative slots are zeroed, the blank id 0 keeps
its embedding row. Measured to be sure: dec([0,0]) has norm 11.68 while dec([-1,-1]) has
norm 0.33, so blank is not masked. Conclusion: mathematically the same computation, same
shapes, same weight layout, same negative-index convention, same ReLU, same projection.
The only differences are that ONNX recomputes it in fp32 per token on a tiny graph while
the fork keeps f32 tables in host RAM.

greedy_run (xasr.cpp:856) computes t[j] = tanh(e[j] + d[j]), then
logits[v] = join_out_b[v] + sum_j join_out_w[v*J + j] * t[j], then argmax, then re-runs
the predictor when a token is accepted. That is the same computation as 5.2, with two
differences: the weights are f32 in RAM instead of int8 (better accuracy, 4x the bytes),
and the ONNX path applies per-tensor int8 quantisation to t on every call, so the ONNX
joiner is slightly less accurate than the fork's. There is no algorithmic difference in
the predictor or joiner; the 16x gap in section 0 is memory layout and kernel choice.

---

## 6. Measured HOST profile

All numbers in this section are macOS Apple M2 CPU, onnxruntime CPUExecutionProvider,
intra_op_num_threads=2, inter_op_num_threads=1, zeros input built from the declared
shapes. They rank work; they are not device numbers.

6.1 Encoder, one inference: 144.01 ms of profiled kernel time over 6,112 kernels, of
which 28.8 percent int8 matmul, 26.2 percent shape/layout, 13.7 percent fp32 Conv
(sections 2.3 and 2.4). Unprofiled wall time of the same call is 76.5 ms at 2 threads and
73.6 ms at 1 thread, so the profiler inflates absolute values by roughly 1.9x.

6.2 Same-host A/B against this fork: section 0; commands in section 8, C7 and C8.

6.3 On the device, quoted from examples/x-asr/BENCHMARK.md and not measured here: SM6650,
Hexagon v73, 2 HVX, 2 MB VTCM. This fork's CPU backend scored RTF 0.321 and a 275 ms tail
on the 2.65 s Chinese clip; its HTP path scored RTF 0.398 and a 404 ms tail. The host CPU
RTF of the same fork is 0.3205 on the 10 s clip, the same order, which is consistent with
the encoder being kernel-bound rather than backend-bound. No ONNX device number is
reproduced here because the device was off limits; if the observed device gap is ONNX
faster than this fork, the host A/B says roughly half of it should be the
predictor/joiner and half the encoder.

6.4 Why the fork's 2 threads do not beat ORT's 1: the fork's encoder graph is 2877 nodes
with 1 scheduler split and ggml_backend_sched barriers between passes. ORT schedules
6,112 kernels per inference, and its profile records, per kernel, a thread pool named
session-1-intra-op plus a work distribution and run/wait count (the
thread_scheduling_stats field printed by C5), so its parallel granularity is one kernel
with a shared pool. Both are latency-bound at sequence lengths 3 to 24, so ORT's
per-kernel efficiency wins over the fork's more compact graph.


---

## 7. Portable A/B experiments

Each is runnable inside this fork and reuses existing ggml infrastructure. Measure with
XASR_TIMING=1 (compute_ms, greedy_ms, chunk_ms) plus wall time and RTF on the same
fixtures, and keep the transcript identical as the correctness gate.

### E1. Keep the joiner weight quantised on the host

- Hypothesis: the joiner is memory-bound on a 10.24 MB f32 matrix that misses cache; the
  same weights as 2.56 MB int8 fit and are 4x less traffic, so per-frame cost should drop
  by close to 4x.
- Change: in load(), stop calling to_f32() on join.output_linear.weight; keep the
  ggml_tensor (Q8_0) and in greedy_run() compute the 5000 logits with ggml_mul_mat on the
  CPU backend, or with a dequant-and-dot loop built on
  ggml_get_type_traits(type)->vec_dot. Keep the argmax in C++.
- Measure: greedy_ms per stream and per chunk, RTF, transcript equality, and process RSS.
- ONNX evidence: joiner.int8 uses one int8 MatMulInteger per frame with a per-tensor scale
  (section 4) and costs 0.21 ms per frame against the fork's 3.40 ms (section 0). The GGUF
  already stores the tensor as Q8_0, so accuracy cannot get worse than ONNX, whose scale
  is per-tensor.

### E2. Batch joiner frames that share a predictor state

- Hypothesis: for k consecutive encoder frames that emit no token, d is constant, so one
  [k,512] x [512,5000] product does the same work with one pass over the weight matrix
  instead of k passes.
- Change: in greedy_run(), collect the run of frames until a token is accepted, run one
  matmul for the whole run, then argmax per row. The 10 s fixture produced 264 frames and
  27 accepted tokens, so the mean run is about 9.8 frames.
- Measure: greedy_ms, RTF, and the token sequence, which must be identical because argmax
  per row is independent.
- ONNX evidence: the ONNX joiner is shaped [N,512] -> [N,5000]; the reference python driver
  calls it with N=1, but the graph and the kernel support N>1 for free, and each chunk
  already delivers 12 frames (section 1.1). The 10 s timing run called the joiner 264
  times and accepted 27 tokens (section 8, C7).

### E3. Stop unpacking the predictor tables to f32

- Hypothesis: dec.emb.weight and join.decoder_proj.weight expanded to f32 cost 10.24 MB +
  1 MB of host RAM, a measurable part of the 745 ms init on this host, and the predictor
  re-reads the whole expanded table on every accepted token.
- Change: keep both as ggml_tensor Q8_0, do the 512x512 projection with ggml_mul_mat, and
  for the embedding dequantise one row on demand (or read the Q8_0 block directly at
  tok * 512).
- Measure: the xasr: init line, greedy_ms, RSS, transcript equality.
- ONNX evidence: the decoder keeps the embedding as a 10.24 MB fp32 matrix and gathers one
  row per token; the whole decoder is 3.6 ms for 27 calls, 0.2 percent of ORT's time
  (section 0). The fork's cost is the full-table streaming, not the gather.

### E4. Per-stage timing inside the fork's encoder

- Hypothesis: the fork's 100.3 ms/chunk splits differently from ORT's 73.6 ms/chunk, and
  the subsampling block, the ConvNeXt block and the 19 stacks need separating before
  anything is optimised.
- Change: add env-gated timing around the embed block, each stack and the final
  projection in build_chunk_graph / run_chunks. No numerical change.
- Measure: ms per stage per chunk over three fixtures; the sum must equal compute_ms.
- ONNX evidence: the same split is measured for ORT in section 2.3: 82 fp32 convs are
  13.7 percent of ORT's time (embed block), the int8 matmuls 28.8 percent (stacks), and
  convolution throughput is 5x worse than matmul throughput. If the fork's embed block
  exceeds roughly 20 percent of compute_ms, the subsampling/ConvNeXt lowering is the
  target; otherwise the stacks are.

### E5. Replace the two-step depthwise conv with one fused 1-D conv

- Hypothesis: the fork runs ggml_ssm_conv twice per conv module (causal 16-tap and
  chunkwise 31-tap) on two different buffers, with ggml_concat and ggml_pad_ext around
  them (xasr.cpp:568-580); the ONNX graph runs two Conv nodes on differently padded
  inputs with no extra copies.
- Change: build one [pad+seq] buffer, run a single depthwise conv with the union kernel,
  or keep two convs but read them from wider views so the pad and concat disappear.
- Measure: compute_ms, and the conv-module stage time once E4 exists; transcript equality.
- ONNX evidence: 77 of the 82 Convs are group>1 depthwise with kernel 16/31 or 8/15
  (sections 3a and 3e), and ORT spends 13.7 percent of its time on all 82 at 3.7 GMAC/s,
  the worst throughput of any class.

### E6. Thread-count and scheduler-split sweep

- Hypothesis: at sequence length 3 to 24 the fork is latency-bound, so 2 threads may be
  slower than 1 and 4 may not scale; ORT's single thread already beats the fork's two.
- Change: expose n_threads on the CLI (xasr_context_params.n_threads already exists) and
  sweep 1/2/4 on CPU for the encoder and the joiner separately.
- Measure: compute_ms, greedy_ms, cpu_ms, RTF.
- ONNX evidence: section 0, ORT 1 thread 73.6 ms/chunk versus the fork 2 threads 100.3
  ms/chunk for the same 0.824 GMAC.

### E7. Quantise the 82 fp32 convolutions

- Hypothesis: the 82 convs are 2.7 MB of weights, 8.9 percent of the MACs but 13.7 percent
  of ORT's time at 3.7 GMAC/s. A quantised depthwise conv (per-channel scale, int8 weights,
  int32 accumulate) should cut that stage on both CPU and HTP.
- Change: add a Q8_0 or per-channel int8 depthwise conv path next to
  ggml_conv_2d_dw_direct and ggml_ssm_conv; keep the three subsampling convs fp32, they are
  only 1 percent of the bytes.
- Measure: compute_ms plus encoder-output cosine similarity against the fp32 path, with
  the tolerance anchored on the int8 activation error the reference already has (8.6e-2 on
  logits, section 5.2).
- ONNX evidence: the reference does not quantise its convs, which is exactly why ORT's conv
  class is its slowest. This is the one place where the fork can beat the reference rather
  than match it. Treat the accuracy gate as the risk, not the speed.

### E8. Build the attention mask once per stream instead of once per chunk

- Hypothesis: 101 Equal + 158 Where + 463 Slice per inference exist only to zero padded
  keys, and the fork rebuilds and re-uploads its mask vector every chunk
  (xasr.cpp:768-781) even though only the frames that cross from -inf to 0 change.
- Change: build mask_s once, and per chunk upload only the newly unmasked prefix.
- Measure: mask_build_ms, mask_up_ms, chunk_ms, transcript equality.
- ONNX evidence: ORT's mask path is 101 Equal + 158 Where = 7.27 ms, 5.0 percent of its
  profiled time, recomputed on every inference. The fork's static graph can avoid it
  entirely, so this is a case where the fork's design is strictly better and only the
  implementation is wasteful.

---

## 8. Exact commands

Shell prefix used by all of them:

    MODEL=/Users/wangyifan/Library/Caches/qwen3-asr-htp/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
    GGUF=/Users/wangyifan/Library/Caches/qwen3-asr-htp/xasr-multistream/x-asr-zh-en-q8_0.gguf
    REPO=/Users/wangyifan/Desktop/llama.cpp

### C1. Inventory, metadata, IO, op census, encoder output shapes

    cd "$MODEL" && python3 - <<'PY'
    import onnx, collections
    from onnx import numpy_helper
    DT={1:"float32",2:"uint8",3:"int8",6:"int32",7:"int64",9:"bool"}
    for fn in ["encoder.int8.onnx","decoder.onnx","joiner.int8.onnx"]:
        m=onnx.load(fn, load_external_data=False); g=m.graph
        print("="*60, fn, "ir", m.ir_version, "opset",
              [(o.domain,o.version) for o in m.opset_import],
              "producer", m.producer_name, m.producer_version)
        print("metadata:", [(p.key,p.value) for p in m.metadata_props])
        for tag,seq in (("in",g.input),("out",g.output)):
            for t in seq:
                tt=t.type.tensor_type
                dims=[d.dim_value if d.HasField("dim_value") else d.dim_param
                      for d in tt.shape.dim]
                print(" ",tag,t.name,DT.get(tt.elem_type,tt.elem_type),dims)
        print("nodes",len(g.node),"inits",len(g.initializer))
        print("ops", dict(sorted(collections.Counter(n.op_type for n in g.node).items(),
                                key=lambda kv:-kv[1])))
        tot=collections.Counter(); cnt=collections.Counter(); s=0
        for i in g.initializer:
            a=numpy_helper.to_array(i); tot[DT.get(i.data_type)]+=a.nbytes
            cnt[DT.get(i.data_type)]+=1; s+=a.nbytes
        print("init bytes",s,{k:(v,cnt[k]) for k,v in tot.items()})
        print("params",sum(numpy_helper.to_array(i).size for i in g.initializer))
    PY

    cd "$MODEL" && python3 - <<'PY'
    import onnxruntime as ort, numpy as np
    so=ort.SessionOptions(); so.intra_op_num_threads=2; so.inter_op_num_threads=1
    s=ort.InferenceSession("encoder.int8.onnx", so, providers=["CPUExecutionProvider"])
    feeds={i.name: np.zeros([1 if isinstance(d,str) else d for d in i.shape],
             dtype=np.int64 if i.type=="tensor(int64)" else np.float32)
           for i in s.get_inputs()}
    o=s.run(None,feeds)
    for n,v in zip([x.name for x in s.get_outputs()],o):
        if n in ("encoder_out","new_embed_states","new_processed_lens",
                 "new_cached_key_0","new_cached_conv1_0"):
            print(n, v.shape, v.dtype)
    PY

### C2. GGUF census

    cd "$REPO" && PYTHONPATH="$REPO/gguf-py" python3 - <<'PY'
    import gguf, collections
    rd=gguf.GGUFReader("/Users/wangyifan/Library/Caches/qwen3-asr-htp/xasr-multistream/x-asr-zh-en-q8_0.gguf")
    by=collections.Counter(); cnt=collections.Counter()
    for t in rd.tensors:
        by[t.tensor_type.name]+=len(t.data.tobytes()); cnt[t.tensor_type.name]+=1
    print("n_tensors",len(rd.tensors),"bytes",sum(by.values()))
    for k in by: print(k,by[k],cnt[k])
    PY

### C3. Conv census

    cd "$MODEL" && python3 - <<'PY'
    import onnx
    m=onnx.load("encoder.int8.onnx", load_external_data=False); g=m.graph
    init={i.name:i for i in g.initializer}
    for i,n in enumerate(g.node):
        if n.op_type!="Conv": continue
        a={x.name:x for x in n.attribute}
        at=lambda k,d=None: (list(a[k].ints) if a[k].ints else a[k].i) if k in a else d
        print(i,n.name,list(init[n.input[1]].dims),"group",at("group",1),
              "k",at("kernel_shape"),"s",at("strides"),"p",at("pads"),"in",n.input[0])
    PY

### C4. Quantisation census

    cd "$MODEL" && python3 - <<'PY'
    import onnx, collections
    from onnx import numpy_helper
    for fn in ["encoder.int8.onnx","joiner.int8.onnx"]:
        m=onnx.load(fn, load_external_data=False); g=m.graph
        c=collections.Counter(n.op_type for n in g.node)
        for op in ("QuantizeLinear","DequantizeLinear","QLinearConv","QLinearMatMul",
                   "ConvInteger","MatMulInteger","DynamicQuantizeLinear","Gemm","MatMul","Conv"):
            print(fn,op,c.get(op,0))
        sh=collections.Counter(); zp=collections.Counter()
        for i in g.initializer:
            if i.name.endswith("_scale"): sh[tuple(numpy_helper.to_array(i).shape)]+=1
            if i.name.endswith("_zero_point"):
                a=numpy_helper.to_array(i); zp[(str(a.dtype),tuple(a.shape))]+=1
        print(" weight scale shapes",dict(sh))
        print(" weight zero points",dict(zp))
    PY

### C5. HOST per-op profile, one encoder inference, 2 threads

    cd "$MODEL" && python3 - <<'PY'
    import onnxruntime as ort, numpy as np, json, collections, time
    so=ort.SessionOptions(); so.intra_op_num_threads=2; so.inter_op_num_threads=1
    so.enable_profiling=True; so.profile_file_prefix="/tmp/xasr_enc_prof"
    s=ort.InferenceSession("encoder.int8.onnx", so, providers=["CPUExecutionProvider"])
    feeds={i.name: np.zeros([1 if isinstance(d,str) else d for d in i.shape],
             dtype=np.int64 if i.type=="tensor(int64)" else np.float32)
           for i in s.get_inputs()}
    for k in range(3):
        t=time.perf_counter(); s.run(None,feeds)
        print("run",k,"wall_ms",round((time.perf_counter()-t)*1e3,2))
    p=s.end_profiling(); print("profile",p)
    ev=[e for e in json.load(open(p))
        if e.get("cat")=="Node" and e["name"].endswith("_kernel_time") and e["dur"]>0]
    ev.sort(key=lambda e:e["ts"])
    names=[e["name"] for e in ev]
    b=[i for i,n in enumerate(names) if n==names[0]]+[len(ev)]
    for r in range(len(b)-1):
        print("run",r,"kernels",b[r+1]-b[r],
              "sum_ms %.2f"%(sum(e["dur"] for e in ev[b[r]:b[r+1]])/1000))
    last=ev[b[-2]:]
    tot=sum(e["dur"] for e in last)
    print("last run kernels",len(last),"sum_ms %.2f"%(tot/1000))
    per=collections.Counter(); cnt=collections.Counter()
    for e in last:
        per[e["args"]["op_name"]]+=e["dur"]; cnt[e["args"]["op_name"]]+=1
    for k,v in per.most_common():
        print("%8.2fms %6.2f%% n=%-5d %s"%(v/1000,100*v/tot,cnt[k],k))
    for e in sorted(last,key=lambda x:-x["dur"])[:15]:
        print("%8.2fms %5.2f%% %-22s %s"%(e["dur"]/1000,100*e["dur"]/tot,
              e["args"]["op_name"],e["name"][:-len("_kernel_time")]))
    PY

### C6. MAC counts per kernel class

    python3 - <<'PY'
    import json,glob,collections,math
    p=sorted(glob.glob("/tmp/xasr_enc_prof_*.json"))[-1]
    ev=[e for e in json.load(open(p))
        if e.get("cat")=="Node" and e["name"].endswith("_kernel_time") and e["dur"]>0]
    ev.sort(key=lambda e:e["ts"]); names=[e["name"] for e in ev]
    b=[i for i,n in enumerate(names) if n==names[0]]; last=ev[b[-1]:]
    def shp(x):
        if isinstance(x,dict):
            for v in x.values(): return v
        if isinstance(x,list) and x: return shp(x[0])
    macs=collections.Counter(); tm=collections.Counter(); cnt=collections.Counter(); T=0
    for e in last:
        k=e["args"]["op_name"]; o=shp(e["args"].get("output_type_shape"))
        ins=e["args"].get("input_type_shape"); i0=shp(ins[0]) if ins else None
        m=0
        if k in ("DynamicQuantizeMatMul","MatMulIntegerToFloat","MatMul") and o and i0:
            m=math.prod(o)*i0[-1]
        elif k=="Conv" and o and ins and len(ins)>1:
            w=shp(ins[1]); m=math.prod(o)*w[1]*w[2]*(w[3] if len(w)==4 else 1)
        macs[k]+=m; tm[k]+=e["dur"]; cnt[k]+=1; T+=m
    print("TOTAL %.3f GMAC %.3f GFLOP"%(T/1e9,2*T/1e9))
    for k,v in macs.most_common():
        if v: print("%-22s %8.3f GMAC %8.2f ms n=%d"%(k,v/1e9,tm[k]/1000,cnt[k]))
    PY

### C7. ORT streaming timing, encoder / joiner / decoder split

    cd "$MODEL" && python3 - <<'PY'
    import numpy as np, onnxruntime as ort, time, kaldi_native_fbank as knf, soundfile as sf
    def feat(path):
        d,sr=sf.read(path, always_2d=True, dtype="float32"); d=d[:,0]
        o=knf.FbankOptions(); o.frame_opts.dither=0; o.frame_opts.snip_edges=False
        o.frame_opts.window_type="povey"; o.frame_opts.samp_freq=sr; o.mel_opts.num_bins=80
        f=knf.OnlineFbank(o); f.accept_waveform(sr,d.tolist()); f.input_finished()
        x=np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])
        return np.ascontiguousarray(np.pad(x,((0,100),(0,0)),mode="constant"))
    for nth in (1,2):
        so=ort.SessionOptions(); so.intra_op_num_threads=nth; so.inter_op_num_threads=1
        enc=ort.InferenceSession("encoder.int8.onnx",so,providers=["CPUExecutionProvider"])
        dec=ort.InferenceSession("decoder.onnx",so,providers=["CPUExecutionProvider"])
        joi=ort.InferenceSession("joiner.int8.onnx",so,providers=["CPUExecutionProvider"])
        meta=enc.get_modelmeta().custom_metadata_map
        T=int(meta["T"]); dcl=int(meta["decode_chunk_len"])
        st=[np.zeros([1 if isinstance(d,str) else d for d in n.shape],
             dtype=np.int64 if n.type=="tensor(int64)" else np.float32)
            for n in enc.get_inputs()[1:]]
        d=dec.run(None,{dec.get_inputs()[0].name:np.array([[0,0]],dtype=np.int64)})[0]
        x=feat("test_wavs/0.wav")
        te=tj=td=0.0; ne=nj=nd=0; nf=0; start=0; t0=time.perf_counter()
        while start+T < x.shape[0]:
            a=time.perf_counter()
            out=enc.run(None,{**{enc.get_inputs()[0].name:x[start:start+T][None]},
                              **{n.name:s for n,s in zip(enc.get_inputs()[1:],st)}})
            te+=time.perf_counter()-a; ne+=1; start+=dcl
            eo, st = out[0], out[1:]; nf+=eo.shape[1]
            for k in range(eo.shape[1]):
                a=time.perf_counter()
                lg=joi.run(None,{"encoder_out":eo[0,k:k+1],"decoder_out":d})[0]
                tj+=time.perf_counter()-a; nj+=1
                if int(lg.argmax())!=0:
                    a=time.perf_counter()
                    d=dec.run(None,{dec.get_inputs()[0].name:
                                     np.array([[0,0]],dtype=np.int64)})[0]
                    td+=time.perf_counter()-a; nd+=1
        tot=time.perf_counter()-t0
        print(f"threads={nth} chunks={ne} frames={nf} joiner={nj} decoder={nd}")
        print(f"  wall={tot*1e3:.1f} enc={te*1e3:.1f} joiner={tj*1e3:.1f} decoder={td*1e3:.1f}")
        print(f"  per_chunk_enc={te/ne*1e3:.1f} per_frame_joiner={tj/nj*1e3:.3f}")
    PY

The decoder call in that harness always passes [0,0]; it is a timing harness only, and
no claim above depends on the text it produces. The per-frame joiner cost is measured
directly; the decoder cost is measured as 27 calls.

### C8. Build and run the fork on the host, out of tree, no repo change

    cmake -S "$REPO" -B /tmp/xasr-build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
          -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DLLAMA_BUILD_TESTS=OFF \
          -DLLAMA_BUILD_SERVER=OFF
    cmake --build /tmp/xasr-build --target llama-x-asr -j 8
    cd "$MODEL/test_wavs" && XASR_TIMING=1 /tmp/xasr-build/bin/llama-x-asr "$GGUF" CPU 0.wav

### C9. Decoder and joiner numeric checks

    cd "$MODEL" && python3 - <<'PY'
    import onnx, numpy as np, onnxruntime as ort
    from onnx import numpy_helper
    so=ort.SessionOptions(); so.intra_op_num_threads=2; so.inter_op_num_threads=1
    J=ort.InferenceSession("joiner.int8.onnx",so,providers=["CPUExecutionProvider"])
    D=ort.InferenceSession("decoder.onnx",so,providers=["CPUExecutionProvider"])
    ini={i.name:numpy_helper.to_array(i) for i in
         onnx.load("joiner.int8.onnx",load_external_data=False).graph.initializer}
    W=ini["output_linear.weight_quantized"].astype(np.float32)*float(ini["output_linear.weight_scale"])
    b=ini["output_linear.bias"]
    rng=np.random.default_rng(0); e=rng.standard_normal((1,512),dtype=np.float32)
    d=rng.standard_normal((1,512),dtype=np.float32)
    print("joiner max|onnx-ref|",
          float(np.abs(J.run(None,{"encoder_out":e,"decoder_out":d})[0]
                      -(np.tanh(e+d)@W+b)).max()))
    di={i.name:numpy_helper.to_array(i) for i in
        onnx.load("decoder.onnx",load_external_data=False).graph.initializer}
    emb,cw,pw,pb=(di["decoder.embedding.weight"],di["decoder.conv.weight"],
                  di["decoder_proj.weight"],di["decoder_proj.bias"])
    def ref(y):
        x=np.stack([emb[t] if t>=0 else np.zeros(512,np.float32) for t in y])
        H=np.zeros(512,np.float32)
        for o in range(512):
            g0=(o//4)*4
            H[o]=sum(cw[o,i,k]*x[k,g0+i] for i in range(4) for k in range(2))
        return np.maximum(H,0)@pw.T+pb
    for y in ([0,0],[5,7],[-1,3]):
        print(y, float(np.abs(D.run(None,{"y":np.array([y],dtype=np.int64)})[0][0]-ref(y)).max()))
    PY

### C10. Swoosh constants

    cd "$MODEL" && python3 - <<'PY'
    import onnx, numpy as np
    from onnx import numpy_helper
    g=onnx.load("encoder.int8.onnx",load_external_data=False).graph
    for pref in ("/conv/conv.3/","/activation/"):
        for n in g.node:
            if n.op_type=="Constant" and n.name.startswith(pref):
                v=numpy_helper.to_array(n.attribute[0].t)
                if v.size==1: print(n.name, float(v.ravel()[0]))
    PY

---

## 9. Claims checked and rejected

- The encoder is not all int8. 82 Convs and 20 norms are fp32, 6,334,536 B of fp32
  weights. Only the linear projections are int8 (sections 3 and 4).
- There is no QDQ pair, no QLinearConv, no ConvInteger, no LSTM, no GRU, no fused
  Attention and no Gemm in the encoder (sections 2.1 and 4).
- The decoder does not zero the blank embedding: the mask is y >= 0, and dec([0,0])
  differs from dec([-1,-1]) by 1.71 in max absolute value (section 5.3). The fork's
  tok >= 0 matches.
- The joiner is not a 512 -> 512 -> 5000 two-layer MLP. It is add, tanh, one 5000-wide
  projection, bias (section 5.2).
- The ConvNeXt padding is not on the time axis. pads [0,3,0,3] pads the frequency axis
  by 3 on each side and time by nothing; the 7x7 kernel is valid along time, so output
  frame k reads input frames k..k+6, with the 3-frame cache and 3 lookahead frames
  supplying the ends (section 3a, measured shapes [1,128,30,19] -> [1,128,24,19]).
- ONNX Runtime's graph is not smaller. It runs 6,112 kernels for one chunk against the
  fork's 2,877 nodes, so the fork loses on kernel throughput, not on graph size or shape
  plumbing (sections 0 and 2.3).
- The host profile is not a device measurement. Profiling inflates the encoder by 2-3x
  (144.0 ms profiled versus 76.5 ms unprofiled at 2 threads), so only the shares and the
  kernel ranking are used.

