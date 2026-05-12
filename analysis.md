# GLM-5.1 â€” Kernel Map, AMD MI355X (SGLang, decode)

## Setup

| Field | Value |
|---|---|
| Model | `amd/GLM-5.1-MXFP4` (`GlmMoeDsaForCausalLM`, same backbone as `nvidia/GLM-5-NVFP4`) |
| Quantization | MXFP4 weights (block-scaled FP4 over 32-elem groups, e8m0 scales â€” see `dynamic_per_group_scaled_quant_kernel<â€¦fp4_tâ€¦Li32Eâ€¦>` and the `ck::f4x2_pk_t, ck::e8m0_bexp_t` template args in the MoE GEMM); KV-cache FP8 (paged MLA) |
| Hardware | **MI355X** (CDNA4, ISA `gfx950` â€” `ISA950` token in Cijk tile names), runner `alka-mi355x-p02-g57` |
| Engine | SGLang `v0.5.10rc0-rocm720-mi35x-20260415` (aiter + ck_tile + Triton) |
| Tensor parallel | TP=4 (`cross_device_reduce_2stage<DF16b, Li4, Lb0>` template arg `Li4` â‡’ 4-rank AR) |
| Phase | decode |
| Batch (M) | **128 tokens** (concurrency=128, ISL=1024, OSL=1024) |
| Trace | `Profile_Traces/1778592510.4342594-TP-1-DECODE.trace.json.gz` |
| Single-layer window | 239.07 ms span / 78 layers â‰ˆ 3.06 ms per layer |
| Total kernels in layer-filtered slice | 3,541 launches across 36 unique kernels |

## Architecture (HF config â†’ counts)

| Property | Value |
|---|---|
| Layers | **78** (3 dense MLP + 75 MoE; `first_k_dense_replace=3`) |
| hidden_size H | **6144** |
| num_attention_heads | **64** (TP=4 â‡’ **16 local heads**) |
| MLA latents | q_a (H=6144 â†’ q_lora=2048) â†’ q_a_layernorm â†’ q_b (2048 â†’ local 16Â·256=4096); kv_a_proj_with_mqa (6144 â†’ kv_lora+rope=512+64=576) â†’ kv_a_layernorm â†’ kv_b (512 â†’ local 16Â·(192+256)=7168) |
| Heads | qk_head_dim=256 (qk_nope=192 + qk_rope=64), v_head_dim=256 |
| RoPE | default, theta=1e6, `rope_interleave=true` |
| DSA indexer | `index_n_heads=32`, `index_head_dim=128`, `index_topk=2048`, `indexer_rope_interleave=true` (sparse-attention indexer over kv_lora) |
| MoE | n_routed=**256**, top_k=**8**, n_groups=1, topk_group=1, scoring=`sigmoid`/`noaux_tc`, moe_inter=**2048**, n_shared=**1** (shared_inter=2048), routed_scaling=2.5 |
| Dense MLP (first 3 layers) | intermediate_size=**12288** (gate_up = 2Â·12288 = 24576) |

---

## Execution order (recovered from launch timestamps, decode iter)

```
[per-iter setup: arange/neg/index_put/argmax/clamp_position/...]
  â†’ for L = 0..2  (DENSE):  [ATTN] â†’ AR â†’ [DENSE-MLP gate_up + SwiGLU + down] â†’ AR
  â†’ for L = 3..77 (MoE):    [ATTN] â†’ AR â†’ [MoE: norm â†’ router â†’ topk â†’ sort â†’ shared-exp + 2 grouped FP4 GEMMs] â†’ AR
[per-iter sampling tail: argmax + ncclDevKernel + bf16â†’fp32 copy]
```

Per layer the AMD launch sequence I recovered (from `decode_events.csv`, first MoE layer post-warmup, ts-deltas in Âµs):

```
0.0    AR(prev)                   â”€â”€ (folded as setup)
17.5   add_rmsnorm_quant (H=6144) â”€â”€ input_layernorm + act-quant
21.8   GEMM MT64x128x128          â”€â”€ q_a_proj | kv_a_proj_with_mqa (fused 6144â†’2624)
44.8   add_rmsnorm_quant (kv_lora=512)  â”€â”€ kv_a_layernorm + quant
49.2   add_rmsnorm_quant (q_lora=2048)  â”€â”€ q_a_layernorm + quant
53.4   GEMM MT64x32x256 (Ã—1)      â”€â”€ q_b_proj (2048 â†’ 4096 local)
65.3   GEMM MT64x32x256 (Ã—1)      â”€â”€ kv_b_proj absorb / wq_b indexer (2048 â†’ small N)
76.4   GEMM MT16x16x1024          â”€â”€ DSA indexer wk (H=6144 â†’ head_dim=128) or projection
87.1   ck_tile Layernorm2dFwd     â”€â”€ DSA indexer k_norm (head_dim=128)
91.3   aiter::kn_entry_2c_sbhd_cached_indirect_inplace  â”€â”€ KV-cache write (paged MLA, indirect)
101.1  fast_hadamard_transform (Ã—2)  â”€â”€ Hadamard rotation on Q/K (DSA logits prep)
109.0  act_quant_kernel (Ã—2)         â”€â”€ per-group activation quant for indexer GEMM
118.5  _set_k_and_s_triton_kernel    â”€â”€ set k & scale buffers for paged FP8 logits
122.7  triton_poi_fused__to_copy_gemm_a16w16_0  â”€â”€ A16W16 indexer GEMM stage
127.2  GEMM MT16x16x64 (S_B)         â”€â”€ DSA indexer wq_b/inner indexer GEMM
134.8  Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4  â”€â”€ post-GEMM scaling/bias for indexer
141.2  triton_poi_fused_mul_unsqueeze_1  â”€â”€ indexer score scaling
145.8  fill_kernel<float>            â”€â”€ workspace init
150.2  _gluon_deepgemm_fp8_paged_mqa_logits  â”€â”€ DSA paged-MLA logits (FP8 paged GEMM)
165.3  topk_transform_decode_kernel  â”€â”€ DSA top-2048 selection over indexer logits
174.2  GEMM MT128x32x64              â”€â”€ q-absorb / kv_b_proj absorb (small N decode)
179.8  _fused_qk_rope_cat_and_cache_mla_kernel  â”€â”€ RoPE on Q/K + cache write into MLA buffer
184.1  CatArrayBatchedCopy           â”€â”€ concat nope+rope along head dim
191.0  main_kernel (Ã—2)              â”€â”€ flash-attention MLA forward (sparse, top-2048)
253.9  GEMM MT256x128x64             â”€â”€ o_proj (16Â·256=4096 â†’ H=6144)
269.8  GEMM MT64x64x256              â”€â”€ (paired GEMM) o_proj epilogue / merge
290.9  cross_device_reduce_2stage    â”€â”€ post-attn AllReduce (TP=4)
â”€â”€â”€â”€â”€â”€â”€ MoE block â”€â”€â”€â”€â”€â”€â”€
310.1  add_rmsnorm_quant (H=6144)    â”€â”€ post_attention_layernorm
314.6  GEMM MT16x16x1024             â”€â”€ router (H=6144 â†’ 256 experts)
325.0  grouped_topk_kernel           â”€â”€ top-8 expert selection (sigmoid + noaux_tc)
330.4  _fused_append_shared_experts  â”€â”€ shared-expert path append
334.3  MoeSortingMultiPhase_P0_v2    â”€â”€ expert sort phase 0
338.2  MoeSortingMultiPhase_P23      â”€â”€ expert sort phases 2/3
342.9  dynamic_per_group_scaled_quant<â€¦fp4_tâ€¦Li32>  â”€â”€ activation FP4 quant (group=32)
347.3  mxfp4_moe_sort_kernel<â€¦32,24,32>  â”€â”€ MoE token-permute (gate_up shuffle)
351.8  ck::kernel_moe_mxgemm_2lds<â€¦MulABScaleShuffledâ€¦>  â”€â”€ MoE GEMM1 (gate_up, FP4 â†’ fp32)
380.9  dynamic_per_group_scaled_quant<â€¦fp4_tâ€¦Li32>  â”€â”€ activation FP4 quant for GEMM2
385.2  mxfp4_moe_sort_kernel<â€¦64,4,32>  â”€â”€ MoE token-permute (down)
389.3  ck::kernel_moe_mxgemm_2lds<â€¦MulABScaleExpertWeightShuffledâ€¦>  â”€â”€ MoE GEMM2 (down, FP4 + expert-weight scatter)
420.7  cross_device_reduce_2stage    â”€â”€ post-MoE AllReduce
```

---

## 1. ATTENTION block (78 layers, decode, MLA + DSA indexer)

Workhorse Tensile (`Cijk_â€¦`) GEMMs are **split by launch position** â€” the same kernel name fires multiple times per layer at different per-call durations for different roles. Per-call Âµs is the kernel-mean from the layer-filtered CSV (counts are total launches across the 78-layer window).

<table style="table-layout:fixed;width:100%;font-size:12px;word-break:break-word;">
<colgroup>
<col style="width:4%"><col style="width:30%"><col style="width:46%"><col style="width:5%"><col style="width:5%"><col style="width:5%"><col style="width:5%"><col style="width:5%">
</colgroup>
<thead>
<tr><th>pos</th><th>Role</th><th>AMD kernel</th><th>cnt</th><th>Âµs</th><th>M</th><th>N</th><th>K</th></tr>
</thead>
<tbody>

<tr>
<td>1</td>
<td><b>input_layernorm + add + act-quant (H=6144)</b><br><i>fused residual-add + RMSNorm + per-token activation quant; aiter template `Li256ELi24` (256-thread block, vec=24) matches H=6144=256Â·24. <br>`add_rmsnorm_quant` is the aiter fused norm+quant; one per layer.</i></td>
<td><code>_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4_</code></td>
<td>156</td><td>4.47</td><td>128</td><td>â€“</td><td>6144</td>
</tr>

<tr>
<td>2</td>
<td><b>q_a_proj + kv_a_proj_with_mqa (fused QKV-A GEMM, bf16)</b><br><i>Role from <code>modeling_glm_moe_dsa.py</code>: `q_a_proj: Hâ†’q_lora` and `kv_a_proj_with_mqa: Hâ†’kv_lora+qk_rope`. Output N = q_lora + kv_lora + qk_rope = 2048 + 512 + 64 = 2624. The Tensile MT64x128x128 tile is the dominant 6144-K GEMM in the slice.</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_â€¦</code></td>
<td>78</td><td>22.57</td><td>128</td><td>2624</td><td>6144</td>
</tr>

<tr>
<td>3</td>
<td><b>kv_a_layernorm + act-quant (kv_lora=512)</b><br><i>Per `modeling_glm_moe_dsa.py`: `kv_a_layernorm: GlmMoeDsaRMSNorm(kv_lora_rank=512)`. aiter template `Li64ELi8` â‡’ 64-thread Ã— vec=8 = 512. AMD fires kv_a-norm BEFORE q_a-norm (smaller-N kernel dispatched first).</i></td>
<td><code>_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_</code></td>
<td>78</td><td>4.11</td><td>128</td><td>â€“</td><td>512</td>
</tr>

<tr>
<td>4</td>
<td><b>q_a_layernorm + act-quant (q_lora=2048)</b><br><i>Per `modeling_glm_moe_dsa.py`: `q_a_layernorm: GlmMoeDsaRMSNorm(q_lora_rank=2048)`. aiter template `Li256ELi8` â‡’ 256Â·8 = 2048. Fires after kv_a-norm in this build.</i></td>
<td><code>_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_</code></td>
<td>78</td><td>4.26</td><td>128</td><td>â€“</td><td>2048</td>
</tr>

<tr>
<td>5</td>
<td><b>q_b_proj GEMM (2048 â†’ local 16Â·256=4096), 1st of 2 launches/L</b><br><i>Tensile MT64x32x256 fires twice per layer at this point in the launch sequence â€” first call drives q_b_proj (q_loraâ†’local heads Â· qk_head_dim).</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_â€¦</code></td>
<td>156</td><td>11.65</td><td>128</td><td>4096</td><td>2048</td>
</tr>

<tr>
<td>6</td>
<td><b>kv_b_proj / indexer wq_b absorb GEMM, 2nd of 2 launches/L</b><br><i>Same Tensile MT64x32x256 kernel as pos 5. Per `modeling_glm_moe_dsa.py`, `indexer.wq_b: q_loraâ†’n_headsÂ·head_dim = 32Â·128 = 4096`; both fan-out the q_lora=2048 latent.</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_â€¦</code></td>
<td>(see pos 5: 156 total)</td><td>11.65</td><td>128</td><td>4096</td><td>2048</td>
</tr>

<tr>
<td>7</td>
<td><b>DSA indexer wk GEMM (H=6144 â†’ head_dim=128)</b><br><i>Tensile MT16x16x1024 has K-tile 1024 â€” large-K small-N shape consistent with `indexer.wk: Linear(hidden_size, head_dim)` (6144â†’128). Fires twice per layer (also reused as router GEMM in MoE block â€” see MoE pos 2).</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_â€¦</code></td>
<td>153</td><td>10.36</td><td>128</td><td>128</td><td>6144</td>
</tr>

<tr>
<td>8</td>
<td><b>DSA indexer k_norm (LayerNorm head_dim=128)</b><br><i>Per `modeling_glm_moe_dsa.py`: `indexer.k_norm: LayerNorm(head_dim)`. ck_tile `Layernorm2dFwd` with `BlockShape sequence&lt;4,128&gt;` (block-tile 4Ã—128) matches head_dim=128.</i></td>
<td><code>_ZN7ck_tile6kentryILi1ENS_14Layernorm2dFwdINS_29Layernorm2dFwdPipelineOnePassINS_29Layern</code></td>
<td>78</td><td>4.19</td><td>128</td><td>â€“</td><td>128</td>
</tr>

<tr>
<td>9</td>
<td><b>KV-cache write (paged MLA, indirect store, bf16â†’bf16)</b><br><i>aiter `kn_entry_2c_sbhd_cached_indirect_inplace<OpCachedFwd, 1, true, false, true, true, 1, true, BFloat16, BFloat16>` â€” paged KV-cache write into the MLA latent cache (kv_lora=512 + rope=64 = 576 per token).</i></td>
<td><code>void aiter::kn_entry_2c_sbhd_cached_indirect_inplace&lt;aiter::OpCachedFwd, 1, true, false, â€¦</code></td>
<td>78</td><td>9.57</td><td>128</td><td>â€“</td><td>576</td>
</tr>

<tr>
<td>10</td>
<td><b>Hadamard rotation on Q/K (DSA logits prep)</b><br><i>`fast_hadamard_transform_kernel<â€¦16, 7â€¦>` applies a learned/Hadamard rotation; fires twice per layer (Q-side and K-side of the indexer) â€” DSA logits-path preprocessing.</i></td>
<td><code>void fast_hadamard_transform_kernel&lt;fast_hadamard_transform_kernel_traits&lt;16, 7, c10::BFlo</code></td>
<td>156</td><td>4.18</td><td>128</td><td>â€“</td><td>128</td>
</tr>

<tr>
<td>11</td>
<td><b>Per-group activation FP8 quant for indexer (Ã—2 launches)</b><br><i>`act_quant_kernel__kernel` â€” Triton per-token/per-group activation quantization feeding the FP8 paged-logits GEMM at pos 17.</i></td>
<td><code>act_quant_kernel__kernel</code></td>
<td>156</td><td>4.85</td><td>128</td><td>â€“</td><td>128</td>
</tr>

<tr>
<td>12</td>
<td><b>Set k & scale buffers for paged FP8 logits (Triton)</b><br><i>`_set_k_and_s_triton_kernel` â€” names match SGLang/DeepGEMM helper that stages the K-tensor and per-block scales for the paged MQA logits kernel (pos 17).</i></td>
<td><code>_set_k_and_s_triton_kernel</code></td>
<td>78</td><td>4.36</td><td>128</td><td>â€“</td><td>â€“</td>
</tr>

<tr>
<td>13</td>
<td><b>A16W16 indexer GEMM staging (Inductor/Triton)</b><br><i>`triton_poi_fused__to_copy_gemm_a16w16_0` â€” torch.compile/Inductor-emitted GEMM staging for the bf16-act Ã— bf16-weight indexer projection.</i></td>
<td><code>triton_poi_fused__to_copy_gemm_a16w16_0</code></td>
<td>78</td><td>4.51</td><td>128</td><td>â€“</td><td>â€“</td>
</tr>

<tr>
<td>14</td>
<td><b>DSA inner indexer GEMM (small-N MFMA tile)</b><br><i>Tensile MT16x16x64 single-buffer (S_B) tile â€” small-M small-N small-K shape consistent with the indexer's per-token score reduction over `index_n_heads=32` heads.</i></td>
<td><code>Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_â€¦</code></td>
<td>78</td><td>7.82</td><td>128</td><td>32</td><td>128</td>
</tr>

<tr>
<td>15</td>
<td><b>Indexer GEMM post-scale + bias (Cijk SS-stream)</b><br><i>`Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` â€” Tensile post-GEMM stream-K reduction with bias add and scale-alpha-vec; epilogue for the indexer GEMM (pos 14).</i></td>
<td><code>Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4</code></td>
<td>78</td><td>7.37</td><td>128</td><td>32</td><td>â€“</td>
</tr>

<tr>
<td>16</td>
<td><b>Indexer score scaling (Inductor mul+unsqueeze)</b><br><i>`triton_poi_fused_mul_unsqueeze_1` â€” final per-head scaling and shape unsqueeze on the indexer logits before paged-MQA logits.</i></td>
<td><code>triton_poi_fused_mul_unsqueeze_1</code></td>
<td>78</td><td>4.06</td><td>128</td><td>32</td><td>â€“</td>
</tr>

<tr>
<td>17</td>
<td><b>DSA paged-MLA logits (FP8 paged MQA GEMM, gluon DeepGEMM)</b><br><i>`_gluon_deepgemm_fp8_paged_mqa_logits` â€” gluon-Triton DeepGEMM kernel that computes the sparse-attention indexer logits over the paged FP8 KV-cache (paged MQA over 32 indexer-heads).</i></td>
<td><code>_gluon_deepgemm_fp8_paged_mqa_logits</code></td>
<td>78</td><td>15.72</td><td>128</td><td>cache_len</td><td>128</td>
</tr>

<tr>
<td>18</td>
<td><b>DSA top-K selection (`index_topk=2048`)</b><br><i>Anonymous `topk_transform_decode_kernel(FastTopKParams, â€¦)` â€” selects top `index_topk=2048` cache positions per query for the sparse attention path.</i></td>
<td><code>(anonymous namespace)::topk_transform_decode_kernel((anonymous namespace)::FastTopKParams,</code></td>
<td>78</td><td>4.21</td><td>128</td><td>2048</td><td>cache_len</td>
</tr>

<tr>
<td>19</td>
<td><b>q_b / kv_b absorb GEMM (post-DSA, decode-shape MFMA tile)</b><br><i>Tensile MT128x32x64 â€” N=32, K=64 small-N tile fired after DSA top-K and before flash-attn; consistent with the kv_b-absorb path GEMM that materializes nope/v components from the kv_lora latent for the heads selected by DSA.</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_â€¦</code></td>
<td>78</td><td>6.56</td><td>128</td><td>32</td><td>64</td>
</tr>

<tr>
<td>20</td>
<td><b>RoPE apply on Q/K + write into MLA cache (fused)</b><br><i>`_fused_qk_rope_cat_and_cache_mla_kernel` â€” name token-by-token: applies RoPE to Q/K, concats with the nope half, and writes into the MLA cache. Fires once per layer.</i></td>
<td><code>_fused_qk_rope_cat_and_cache_mla_kernel</code></td>
<td>78</td><td>4.30</td><td>128</td><td>â€“</td><td>64</td>
</tr>

<tr>
<td>21</td>
<td><b>Concat nope+rope along head dim (CatArrayBatchedCopy)</b><br><i>`at::native::CatArrayBatchedCopy<â€¦OpaqueType<1u>, â€¦, 3, 64, 64>` â€” torch concat of the qk_nope_head_dim=192 and qk_rope_head_dim=64 halves into qk_head_dim=256 per head.</i></td>
<td><code>void at::native::(anonymous namespace)::CatArrayBatchedCopy&lt;at::native::(anonymous namespa</code></td>
<td>78</td><td>6.94</td><td>128</td><td>â€“</td><td>256</td>
</tr>

<tr>
<td>22</td>
<td><b>Flash-Attention MLA forward (sparse, top-2048) â€” `main_kernel`</b><br><i>`main_kernel` is SGLang/aiter's single-name attention dispatch (the aiter mha/MLA fwd compiled symbol). Fires twice per layer (likely chunked over the 2048-key DSA window). Workhorse cost â€” 29.1 Âµs per call mean, dur range 4.2 â†’ 56.0 Âµs.</i></td>
<td><code>main_kernel</code></td>
<td>156</td><td>29.10</td><td>128</td><td>â€“</td><td>2048</td>
</tr>

<tr>
<td>23</td>
<td><b>Direct-copy bf16 staging post-attention</b><br><i>`elementwise_kernel_manual_unroll<128,8,â€¦direct_copy_kernel_cudaâ€¦BFloat16>` â€” bf16 staging copy between the flash-attn output buffer and the o_proj input (head-major â†’ seq-major reshape).</i></td>
<td><code>void at::native::elementwise_kernel_manual_unroll&lt;128, 8, at::native::gpu_kernel_impl_noca</code></td>
<td>79</td><td>5.72</td><td>128</td><td>â€“</td><td>4096</td>
</tr>

<tr>
<td>24</td>
<td><b>o_proj GEMM (local 16Â·256=4096 â†’ H=6144), 1st of 2 launches/L</b><br><i>Tensile MT256x128x64 â€” large-M large-N tile; the `o_proj: Linear(num_headsÂ·v_head_dim, hidden_size)` weight is 4096â†’6144. Per-call mean 12.5 Âµs.</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_â€¦</code></td>
<td>78</td><td>12.49</td><td>128</td><td>6144</td><td>4096</td>
</tr>

<tr>
<td>25</td>
<td><b>kv_b_proj heavy GEMM (decode tail, 2nd MFMA mul of 64x64x256 tile)</b><br><i>Tensile MT64x64x256 fires 81 times in the slice (78 layers + 3 dense extras). N=64 K=256 mid-tile shape; fits the residual `kv_b_proj` materialization path on the small-batch decode side. Mean 21.5 Âµs.</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_â€¦</code></td>
<td>81</td><td>21.48</td><td>128</td><td>64</td><td>256</td>
</tr>

<tr>
<td>26</td>
<td><b>TP AllReduce post-attn (cross_device_reduce_2stage, TP=4)</b><br><i>aiter `cross_device_reduce_2stage<DF16b, Li4, Lb0>` â€” bf16 2-stage AR over 4 ranks. Fires 157 times across the slice (â‰ˆ2 per layer: post-attn + post-MoE). The slice mean of 21.07 Âµs is across both ATTN-AR and MoE-AR launches.</i></td>
<td><code>_ZN5aiter26cross_device_reduce_2stageIDF16bLi4ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPN</code></td>
<td>78 (attn-AR)</td><td>21.07</td><td>128</td><td>6144</td><td>â€“</td>
</tr>

</tbody>
</table>

---

## 2. MoE block (75 layers, FP4 grouped GEMM)

<table style="table-layout:fixed;width:100%;font-size:12px;word-break:break-word;">
<colgroup>
<col style="width:4%"><col style="width:30%"><col style="width:46%"><col style="width:5%"><col style="width:5%"><col style="width:5%"><col style="width:5%"><col style="width:5%">
</colgroup>
<thead>
<tr><th>pos</th><th>Role</th><th>AMD kernel</th><th>cnt</th><th>Âµs</th><th>M</th><th>N</th><th>K</th></tr>
</thead>
<tbody>

<tr>
<td>1</td>
<td><b>post_attention_layernorm + add + act-quant (H=6144)</b><br><i>Same fused `add_rmsnorm_quant<â€¦Li256ELi24â€¦>` as ATTN-row 1 (count is shared, see ATTN pos 1 â€” 156 total launches = 78 input_layernorm + 78 post_attention_layernorm).</i></td>
<td><code>_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4_</code></td>
<td>(see ATTN pos 1)</td><td>4.47</td><td>128</td><td>â€“</td><td>6144</td>
</tr>

<tr>
<td>2</td>
<td><b>Router GEMM (H=6144 â†’ 256 experts), bf16 replicated</b><br><i>Same Tensile MT16x16x1024 kernel as ATTN pos 7 â€” large-K (=H=6144) small-N tile is the router projection (`gate.weight: n_routed_experts Ã— hidden_size = 256 Ã— 6144`). 153 total launches = 78 indexer-wk + 75 router (75 MoE layers).</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_â€¦</code></td>
<td>(see ATTN pos 7)</td><td>10.36</td><td>128</td><td>256</td><td>6144</td>
</tr>

<tr>
<td>3</td>
<td><b>Grouped top-K routing (sigmoid, noaux_tc, top-8 / 256)</b><br><i>aiter `grouped_topk_kernel<BFloat16, float __vector(4), 1, true, true, false>` â€” selects top-8 of 256 experts with sigmoid scoring + noaux_tc bias correction; matches the HF config (`topk_method=noaux_tc`, `scoring_func=sigmoid`).</i></td>
<td><code>void aiter::grouped_topk_kernel&lt;c10::BFloat16, float __vector(4), 1, true, true, false&gt;(c1</code></td>
<td>75</td><td>5.42</td><td>128</td><td>8</td><td>256</td>
</tr>

<tr>
<td>4</td>
<td><b>Append shared-experts to expert list (n_shared=1)</b><br><i>`_fused_append_shared_experts_kernel` â€” concats the n_shared=1 shared expert slot onto the routed expert dispatch list so the same grouped-GEMM path serves both routed and shared experts.</i></td>
<td><code>_fused_append_shared_experts_kernel</code></td>
<td>75</td><td>3.94</td><td>128</td><td>9</td><td>â€“</td>
</tr>

<tr>
<td>5</td>
<td><b>Expert sort phase 0 (token-permute setup)</b><br><i>`ck_tile::MoeSortingMultiPhaseKernel_P0_v2<MoeSortingProblemMp<int, float, int, 1, false, false, true>>` â€” phase-0 of the multi-phase MoE sort that builds per-expert token-id lists.</i></td>
<td><code>void ck_tile::kentry&lt;2, ck_tile::MoeSortingMultiPhaseKernel_P0_v2&lt;ck_tile::MoeSortingProbl</code></td>
<td>75</td><td>3.88</td><td>128Â·8</td><td>â€“</td><td>â€“</td>
</tr>

<tr>
<td>6</td>
<td><b>Expert sort phases 2/3 (finalize permutation)</b><br><i>`ck_tile::MoeSortingMultiPhaseKernel_P23<â€¦>` â€” phases 2/3 finalize the tokenâ†’expert permutation (counts, offsets, scatter indices).</i></td>
<td><code>void ck_tile::kentry&lt;2, ck_tile::MoeSortingMultiPhaseKernel_P23&lt;ck_tile::MoeSortingProblem</code></td>
<td>75</td><td>4.66</td><td>128Â·8</td><td>â€“</td><td>â€“</td>
</tr>

<tr>
<td>7</td>
<td><b>Per-group activation FP4 quant for MoE GEMM1 (group=32)</b><br><i>aiter `dynamic_per_group_scaled_quant_kernel<DF16b, opus::fp4_t, Li32E>` â€” bf16 â†’ MXFP4 with e8m0 group scales over 32-elem groups (matches MXFP4 spec). Fires 150 times = 2 quant ops Ã— 75 MoE layers (one before each grouped GEMM).</i></td>
<td><code>_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bN4opus5fp4_tELi32EEEvPT0_PfPKT_PKfil</code></td>
<td>150</td><td>4.37</td><td>128</td><td>â€“</td><td>6144</td>
</tr>

<tr>
<td>8</td>
<td><b>MoE token permute for gate_up (mxfp4 sort, 32-bit perm)</b><br><i>aiter `mxfp4_moe_sort_kernel<256, 32, 24, 32>` â€” packs MXFP4 activations + scales into the per-expert token order required by the gate_up grouped GEMM. Template params (256 threads, tile 32, 24, 32) match the gate_up layout.</i></td>
<td><code>void aiter::mxfp4_moe_sort_kernel&lt;256, 32, 24, 32&gt;(unsigned char*, unsigned char*, int con</code></td>
<td>75</td><td>4.67</td><td>128Â·8</td><td>â€“</td><td>â€“</td>
</tr>

<tr>
<td>9</td>
<td><b>MoE GEMM1 â€” gate_up FP4 grouped GEMM (H â†’ 2Â·moe_inter = 4096 per expert)</b> âš <br><i>ck `kernel_moe_mxgemm_2lds<â€¦GridwiseMoeGemmMX_BPreshuffleâ€¦f4x2_pk_t, e8m0_bexp_tâ€¦MulABScaleShuffledâ€¦>` â€” MXFP4 grouped GEMM with per-block e8m0 A/B scales, shuffled-scale variant. `gate_up_proj: Parameter(num_experts Ã— 2Â·moe_intermediate_size Ã— hidden_size) = 256Ã—4096Ã—6144`. Largest single AMD kernel cost in this slice (132.7 Âµs/call mean, 146 Âµs max).</i></td>
<td><code>void ck::kernel_moe_mxgemm_2lds&lt;ck::GridwiseMoeGemmMX_BPreshuffle&lt;â€¦MulABScaleShuffled, â€¦&gt;</code></td>
<td>75</td><td><b>132.72</b></td><td>M_e</td><td>4096</td><td>6144</td>
</tr>

<tr>
<td>10</td>
<td><b>SwiGLU activation in MoE FFN (NV-style act_and_mul, hipified)</b><br><i>The MoE GEMM1 is followed by SwiGLU on its 2Â·I=4096 output before GEMM2. The SGLang `act_and_mul_kernel<__hip_bfloat16, ...silu...>` (count=3, dense layers only â€” see Dense-MLP block) is **not** fired for MoE layers in this build; SwiGLU is folded into MoE GEMM1's epilogue (`MulABScaleShuffled` already incorporates the gateÂ·up multiplication via expert-weight scatter pattern). No separate kernel observed in MoE iter â€” kept here only as a marker.</i></td>
<td colspan="5" align="center">â€” (folded into MoE GEMM1 epilogue / GEMM2 input quant; no standalone act kernel in MoE iters)</td>
</tr>

<tr>
<td>11</td>
<td><b>Per-group activation FP4 quant for MoE GEMM2 (group=32)</b><br><i>Second invocation of `dynamic_per_group_scaled_quant<â€¦fp4_tâ€¦Li32>` per layer (cnt 150 = 2 Ã— 75 â€” see pos 7). Quantizes the post-SwiGLU intermediate (moe_inter=2048) before the down GEMM.</i></td>
<td><code>_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bN4opus5fp4_tELi32EEEvPT0_PfPKT_PKfil</code></td>
<td>(see pos 7: 150 total)</td><td>4.37</td><td>128</td><td>â€“</td><td>2048</td>
</tr>

<tr>
<td>12</td>
<td><b>MoE token permute for down (mxfp4 sort, 64-bit perm)</b><br><i>aiter `mxfp4_moe_sort_kernel<256, 64, 4, 32>` â€” second permute variant (different tile shape) feeding the down grouped GEMM.</i></td>
<td><code>void aiter::mxfp4_moe_sort_kernel&lt;256, 64, 4, 32&gt;(unsigned char*, unsigned char*, int cons</code></td>
<td>75</td><td>4.15</td><td>128Â·8</td><td>â€“</td><td>â€“</td>
</tr>

<tr>
<td>13</td>
<td><b>MoE GEMM2 â€” down FP4 grouped GEMM with expert-weight scatter (moe_inter â†’ H)</b> âš <br><i>ck `kernel_moe_mxgemm_2lds<â€¦GridwiseMoeGemmMX_BPreshuffleâ€¦MulABScaleExpertWeightShuffledâ€¦>` â€” MXFP4 grouped GEMM with `MulABScaleExpertWeightShuffled` epilogue (folds top-k expert-weight scatter into the GEMM finalize). `down_proj: Parameter(num_experts Ã— hidden_size Ã— moe_intermediate_size) = 256Ã—6144Ã—2048`. Second-largest cost (67.7 Âµs/call mean).</i></td>
<td><code>void ck::kernel_moe_mxgemm_2lds&lt;ck::GridwiseMoeGemmMX_BPreshuffle&lt;â€¦MulABScaleExpertWeightShuffled, â€¦&gt;</code></td>
<td>75</td><td><b>67.68</b></td><td>M_e</td><td>6144</td><td>2048</td>
</tr>

<tr>
<td>14</td>
<td><b>TP AllReduce post-MoE (cross_device_reduce_2stage, TP=4)</b><br><i>Same aiter AR kernel as ATTN pos 26. Of the 157 total launches, 75 are post-MoE (75 MoE layers); the other 82 split across post-attn (78) + dense-MLP-AR (3) + a 1-launch warmup.</i></td>
<td><code>_ZN5aiter26cross_device_reduce_2stageIDF16bLi4ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPN</code></td>
<td>75 (moe-AR)</td><td>21.07</td><td>128</td><td>6144</td><td>â€“</td>
</tr>

</tbody>
</table>

> **MoE shape glossary:** `M`=128 batch tokens, `H`=6144, `E`=256 routed experts, `I`=2048 moe_intermediate_size, `K_top`=8 experts/token, `M_e`â‰ˆMÂ·K_top/E = 128Â·8/256 = 4 tokens-per-expert (balanced). Real per-expert M is variable; the grouped-GEMM kernel processes all 256 experts with padded/skipped tiles for empty experts.

---

## 3. Dense MLP block (3 layers, gate_up + SwiGLU + down)

For `first_k_dense_replace=3`, layers 0â€“2 use a standard MLP instead of MoE. The trace's distinct dense-MLP kernels appear with very low counts (3â€“4 launches in the slice) because the layer-filter threshold (cnt â‰¥ 62) drops them; they are present in `decode_summary.csv`:

<table style="table-layout:fixed;width:100%;font-size:12px;word-break:break-word;">
<colgroup>
<col style="width:4%"><col style="width:30%"><col style="width:46%"><col style="width:5%"><col style="width:5%"><col style="width:5%"><col style="width:5%"><col style="width:5%">
</colgroup>
<thead>
<tr><th>pos</th><th>Role</th><th>AMD kernel</th><th>cnt</th><th>Âµs</th><th>M</th><th>N</th><th>K</th></tr>
</thead>
<tbody>

<tr>
<td>1</td>
<td><b>Dense MLP gate_up GEMM (H=6144 â†’ 2Â·intermediate=24576)</b><br><i>Tensile MT192x128x128 â€” only fires 4 times in the slice (3 dense layers + 1 warmup). Per-call mean 59.4 Âµs, with first-call max 117 Âµs.</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x128x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_â€¦</code></td>
<td>4</td><td>59.45</td><td>128</td><td>24576</td><td>6144</td>
</tr>

<tr>
<td>2</td>
<td><b>SwiGLU activation (`act_and_mul`)</b><br><i>SGLang `sgl_hip::activation::act_and_mul_kernel<__hip_bfloat16, â€¦siluâ€¦>` â€” applies `silu(gate)Â·up` to the dense gate_up output. Fires 3 times (one per dense layer); MoE layers fold SwiGLU into the grouped-GEMM epilogue (see MoE pos 10).</i></td>
<td><code>_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_ES3_S5_EEEEvPS3_PS4_i</code></td>
<td>3</td><td>4.29</td><td>128</td><td>â€“</td><td>12288</td>
</tr>

<tr>
<td>3</td>
<td><b>Dense MLP down GEMM (intermediate=12288 â†’ H=6144)</b><br><i>Reuses the ATTN pos 25 Tensile MT64x64x256 tile (the 81-launch count = 78 attn + 3 dense-down).</i></td>
<td><code>Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_â€¦</code></td>
<td>(see ATTN pos 25)</td><td>21.48</td><td>128</td><td>6144</td><td>12288</td>
</tr>

</tbody>
</table>

---

## Per-iter wall-time totals

Single-layer kernel-time sum (compute-only, layer-filtered slice â€” sum of per-layer per-call duration Ã— per-layer count, divided by 78 layers):

| | per-iter total kernel time | per-iter wall window | per-token (M=128) |
|---|---:|---:|---:|
| AMD MI355X | â‰ˆ 2.94 ms / layer Ã— 78 = **229 ms** kernel-active (compute-only sum from layer-filtered) | **239.07 ms** wall (full-trace fallback) | **1.87 ms/tok** Ã· M=128 â‰ˆ **14.6 Âµs/tok/layer Ã— 78 = 1140 Âµs/tok** end-to-end (decode iter) |

> Parser used full-trace span fallback (no `execute_context` decode-window marker detected). Treat the wall total as an upper bound that includes per-iter setup (arange/argmax/sampling tail).

### Top single-layer cost drivers (sorted by per-call mean Ã— per-layer launches)

| Rank | Kernel | Per-layer time (Âµs) |
|---:|---|---:|
| 1 | MoE GEMM1 (`kernel_moe_mxgemm_2lds<â€¦MulABScaleShuffledâ€¦>`, FP4 gate_up) | **132.7** |
| 2 | MoE GEMM2 (`kernel_moe_mxgemm_2lds<â€¦MulABScaleExpertWeightShuffledâ€¦>`, FP4 down) | **67.7** |
| 3 | TP AllReduce (`cross_device_reduce_2stage`, Ã—2/L) | **42.1** (21.07 Ã— 2) |
| 4 | flash-attn `main_kernel` (Ã—2/L, MLA sparse top-2048) | **58.2** (29.10 Ã— 2) |
| 5 | qkv_a fused GEMM (Tensile MT64x128x128, K=6144) | **22.6** |
| 6 | kv_b heavy GEMM (Tensile MT64x64x256) | **21.5** |
| 7 | DSA paged-MLA logits (`_gluon_deepgemm_fp8_paged_mqa_logits`) | **15.7** |
| 8 | o_proj GEMM (Tensile MT256x128x64) | **12.5** |
| 9 | q_b / kv_b absorb GEMMs (Tensile MT64x32x256, Ã—2/L) | **23.3** (11.65 Ã— 2) |

---

## Inputs / outputs

- AMD layer-filtered kernels CSV: `/wrk/dcgmktg_bench_xhd/ajith/ALKA_DATABASE/DATABASE/GLM5.1/MI355X/SGLANG/TP-4_EP-DISABLED_FP4_DP-1_CONC-128_ISL-1024_OSL-1024/20260512_190057/single_layer_kernels.csv`
- Parser metadata: `/wrk/dcgmktg_bench_xhd/ajith/ALKA_DATABASE/DATABASE/GLM5.1/MI355X/SGLANG/TP-4_EP-DISABLED_FP4_DP-1_CONC-128_ISL-1024_OSL-1024/20260512_190057/parse_metadata.json`
- Per-launch events CSV (used for ts-ordering): `/wrk/dcgmktg_bench_xhd/ajith/ALKA_DATABASE/DATABASE/GLM5.1/MI355X/SGLANG/TP-4_EP-DISABLED_FP4_DP-1_CONC-128_ISL-1024_OSL-1024/20260512_190057/decode_events.csv`
- HF config: `/home/asirra/.cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6/config.json`
- Modeling source: `transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` (`GlmMoeDsaForCausalLM`, `GlmMoeDsaAttention`, `GlmMoeDsaMoE`, `GlmMoeDsaIndexer`)
- This map: `/wrk/dcgmktg_bench_xhd/ajith/ALKA_DATABASE/DATABASE/GLM5.1/MI355X/SGLANG/TP-4_EP-DISABLED_FP4_DP-1_CONC-128_ISL-1024_OSL-1024/20260512_190057/kernel_map_GLM5.1_MI355X_SGLang_TP4_FP4_DECODE.md`
