// Copyright 2026 Eric Malloy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//------------------------------------------------------------------------------
// CAIF - C++ AI Framework
// Attention-support CUDA kernels: transpose 0213 (+strided), causal and
// prefix mask fill/grad, attention softmax forward/backward, RoPE
// (full, partial, offset, backward), GQA repeat/reduce, KV-cache
// append, fill_fp32, and the softmax block-size selection helpers.
// Launcher declarations for src/caif_cuda_kernels_attention_support.cu
//. CPU-only builds link the no-op stubs in
// legacy/src/caif_cuda_kernels_cpu.cpp, same contract as the old
// caif_cuda_kernels.h umbrella header.
//------------------------------------------------------------------------------

#pragma once

#include <cstdint>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#else
// Placeholder for non-CUDA builds
typedef void *cudaStream_t;
#endif

//------------------------------------------------------------------------------
// RoPE dimension pairing style (C linkage, plain enum)
// Interleaved: pairs (0,1),(2,3),... — GPT-NeoX style.
// HalfSplit:   pairs (0,half),(1,half+1),... — LLaMA/Qwen/Gemma style.
//------------------------------------------------------------------------------
enum CAIF_RoPEStyle
{
  CAIF_ROPE_INTERLEAVED=0,
  CAIF_ROPE_HALF_SPLIT=1
};

// Note: this header used to wrap all declarations in extern "C". Dtype-aware
// launchers are function templates, which require C++ linkage, so the wrapper
// was removed. All callers are C++; no external C consumer depends on this.

#ifdef USE_CAIF_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

//------------------------------------------------------------------------------
// Multi-Head Attention kernels
//------------------------------------------------------------------------------

/**
 * Transpose dims 1 and 2 of a 4D tensor:
 * [batch, dim0, dim1, dim2] -> [batch, dim1, dim0, dim2]
 * Used for split/merge heads.
 * Templated on T ∈ {float, __half, __nv_bfloat16}. Pure data shuffle —
 * no arithmetic, so no fp32 reduction needed.
 */
template<typename T>
void launch_transpose_0213(const T *input,
                           T *output,
                           int batch,
                           int dim0,
                           int dim1,
                           int dim2,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_transpose_0213<float>(const float *,
                                                  float *,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  cudaStream_t);
extern template void launch_transpose_0213<__half>(const __half *,
                                                   __half *,
                                                   int,
                                                   int,
                                                   int,
                                                   int,
                                                   cudaStream_t);
extern template void launch_transpose_0213<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          __nv_bfloat16 *,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          cudaStream_t);
#endif

/**
 * Strided variant of transpose_0213. Reads a width-slice of a wider packed
 * row buffer. Input logical view is [batch, dim0, dim1, dim2] where the
 * stride along dim0 is input_d0_stride (>= dim1*dim2). Output layout is
 * contiguous [batch, dim1, dim0, dim2]. Used to split a fused QKV MatMul
 * output into Q/K/V transposed tensors without a separate split kernel.
 */
template<typename T>
void launch_transpose_0213_strided(const T *input,
                                   T *output,
                                   int batch,
                                   int dim0,
                                   int dim1,
                                   int dim2,
                                   int input_d0_stride,
                                   cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_transpose_0213_strided<float>(const float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          cudaStream_t);
extern template void launch_transpose_0213_strided<__half>(const __half *,
                                                           __half *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           cudaStream_t);
extern template void launch_transpose_0213_strided<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  __nv_bfloat16 *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  cudaStream_t);
#endif

// Fill an fp32 buffer with a scalar value (device-side; no host staging).
void launch_fill_fp32(float *data,const float value,const int64_t n,cudaStream_t stream);

// Attention logit soft-cap (Gemma-2/3): scores = cap*tanh(scores/cap), applied
// in place after the score scale in the explicit (non-flash) attention path.
template<typename T>
void launch_attn_logit_softcap(T *scores,const float softcap,const int64_t n,cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_attn_logit_softcap<float>(float *,const float,const int64_t,cudaStream_t);
extern template void launch_attn_logit_softcap<__half>(__half *,const float,const int64_t,cudaStream_t);
extern template void launch_attn_logit_softcap<__nv_bfloat16>(__nv_bfloat16 *,
                                                              const float,
                                                              const int64_t,
                                                              cudaStream_t);
#endif

// Soft-cap backward (explicit path): grad *= 1 - tanh^2(scores_val/cap), where
// scores_val is the pre-cap scaled score (scale * Q.K^T), recomputed by caller.
template<typename T>
void launch_attn_logit_softcap_backward(T *grad,
                                        const T *scores_val,
                                        const float softcap,
                                        const int64_t n,
                                        cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_attn_logit_softcap_backward<float>(float *,
                                                               const float *,
                                                               const float,
                                                               const int64_t,
                                                               cudaStream_t);
extern template void launch_attn_logit_softcap_backward<__half>(__half *,
                                                                const __half *,
                                                                const float,
                                                                const int64_t,
                                                                cudaStream_t);
extern template void launch_attn_logit_softcap_backward<__nv_bfloat16>(__nv_bfloat16 *,
                                                                       const __nv_bfloat16 *,
                                                                       const float,
                                                                       const int64_t,
                                                                       cudaStream_t);
#endif

// Sliding-window mask (Mistral / Gemma-2): fill scores where q - k >= window
// with -inf, applied after the causal mask. Layout [num_matrices, s, s].
template<typename T>
void launch_sliding_window_mask(T *scores,int num_matrices,int seq_len,int window,cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sliding_window_mask<float>(float *,int,int,int,cudaStream_t);
extern template void launch_sliding_window_mask<__half>(__half *,int,int,int,cudaStream_t);
extern template void launch_sliding_window_mask<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,cudaStream_t);
#endif

// Sliding-window mask gradient: zero the score gradient where q - k >= window.
template<typename T>
void launch_sliding_window_mask_grad(T *grad_scores,int num_matrices,int seq_len,int window,cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sliding_window_mask_grad<float>(float *,int,int,int,cudaStream_t);
extern template void launch_sliding_window_mask_grad<__half>(__half *,int,int,int,cudaStream_t);
extern template void launch_sliding_window_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,cudaStream_t);
#endif

// ALiBi linear position bias (MPT / BLOOM): add slopes[head]*(k-q) to each
// score in place, applied after the score scale and soft-cap and before the
// mask in the explicit (non-flash) path. slopes is a device array of length
// num_heads; the head is matrix_index % num_heads. Layout [num_matrices,s,s].
// The bias is an additive constant, so the backward has no derivative term;
// the backward re-applies this same forward bias when it recomputes scores.
template<typename T>
void launch_alibi_bias(T *scores,
                       const float *slopes,
                       int num_matrices,
                       int num_heads,
                       int seq_len,
                       cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_alibi_bias<float>(float *,const float *,int,int,int,cudaStream_t);
extern template void launch_alibi_bias<__half>(__half *,const float *,int,int,int,cudaStream_t);
extern template void launch_alibi_bias<__nv_bfloat16>(__nv_bfloat16 *,const float *,int,int,int,cudaStream_t);
#endif

/**
 * Fill upper triangle (j > i) of score matrices with -1e9.
 * Scores layout: [num_matrices, seq_len, seq_len].
 * Templated on T ∈ {float, __half, __nv_bfloat16}.
 */
template<typename T>
void launch_causal_mask_fill(T *scores,
                             int num_matrices,
                             int seq_len,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_causal_mask_fill<float>(float *,int,int,cudaStream_t);
extern template void launch_causal_mask_fill<__half>(__half *,int,int,cudaStream_t);
extern template void launch_causal_mask_fill<__nv_bfloat16>(__nv_bfloat16 *,int,int,cudaStream_t);
#endif

/**
 * Fill causal mask for rectangular matrices with offset (for KV-cache).
 * Scores layout: [num_matrices, query_len, key_len].
 * Masks positions where col > (row + offset).
 * offset = previous cache length (so row 0 corresponds to position offset).
 */
template<typename T>
void launch_causal_mask_fill_offset(T *scores,
                                    int num_matrices,
                                    int query_len,
                                    int key_len,
                                    int offset,
                                    cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_causal_mask_fill_offset<float>(float *,int,int,int,int,cudaStream_t);
extern template void launch_causal_mask_fill_offset<__half>(__half *,int,int,int,int,cudaStream_t);
extern template void launch_causal_mask_fill_offset<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,cudaStream_t);
#endif

/**
 * Row-wise softmax with numerical stability (max subtraction).
 * Input/output: [num_rows, row_len]. One block per row.
 * Templated on T ∈ {float, __half, __nv_bfloat16}. Reductions always fp32.
 */
template<typename T>
void launch_attention_softmax(const T *input,
                              T *output,
                              int num_rows,
                              int row_len,
                              cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_attention_softmax<float>(const float *,
                                                     float *,
                                                     int,
                                                     int,
                                                     cudaStream_t);
extern template void launch_attention_softmax<__half>(const __half *,
                                                      __half *,
                                                      int,
                                                      int,
                                                      cudaStream_t);
extern template void launch_attention_softmax<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             int,
                                                             int,
                                                             cudaStream_t);
#endif

/**
 * Softmax backward for attention scores.
 * grad_input[i] = output[i] * (grad_output[i] - dot(grad_output, output))
 * Layout: [num_rows, row_len].
 */
template<typename T>
void launch_attention_softmax_backward(const T *grad_output,
                                       const T *output,
                                       T *grad_input,
                                       int num_rows,
                                       int row_len,
                                       cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_attention_softmax_backward<float>(const float *,
                                                              const float *,
                                                              float *,
                                                              int,
                                                              int,
                                                              cudaStream_t);
extern template void launch_attention_softmax_backward<__half>(const __half *,
                                                               const __half *,
                                                               __half *,
                                                               int,
                                                               int,
                                                               cudaStream_t);
extern template void launch_attention_softmax_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                      const __nv_bfloat16 *,
                                                                      __nv_bfloat16 *,
                                                                      int,
                                                                      int,
                                                                      cudaStream_t);
#endif

/**
 * Zero upper triangle (j > i) of gradient scores.
 * Layout: [num_matrices, seq_len, seq_len].
 */
template<typename T>
void launch_causal_mask_grad(T *grad_scores,
                             int num_matrices,
                             int seq_len,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_causal_mask_grad<float>(float *,int,int,cudaStream_t);
extern template void launch_causal_mask_grad<__half>(__half *,int,int,cudaStream_t);
extern template void launch_causal_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,
                                                            int,int,
                                                            cudaStream_t);
#endif

/**
 * Prefix-LM mask fill.
 * Allowed iff (col <= row) OR (col < prefix_lengths[batch]).
 * Disallowed positions set to -1e9.
 * Scores layout: [batch*num_heads, seq_len, seq_len].
 * prefix_lengths: device pointer, length=batch (int32).
 */
template<typename T>
void launch_prefix_mask_fill(T *scores,
                             const uint32_t *prefix_lengths,
                             int batch_size,
                             int num_heads,
                             int seq_len,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_prefix_mask_fill<float>(float *,
                                                    const uint32_t *,
                                                    int,int,int,
                                                    cudaStream_t);
extern template void launch_prefix_mask_fill<__half>(__half *,
                                                     const uint32_t *,
                                                     int,int,int,
                                                     cudaStream_t);
extern template void launch_prefix_mask_fill<__nv_bfloat16>(__nv_bfloat16 *,
                                                            const uint32_t *,
                                                            int,int,int,
                                                            cudaStream_t);
#endif

/**
 * Prefix-LM mask gradient.
 * Zeros gradient at disallowed positions.
 */
template<typename T>
void launch_prefix_mask_grad(T *grad_scores,
                             const uint32_t *prefix_lengths,
                             int batch_size,
                             int num_heads,
                             int seq_len,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_prefix_mask_grad<float>(float *,
                                                    const uint32_t *,
                                                    int,int,int,
                                                    cudaStream_t);
extern template void launch_prefix_mask_grad<__half>(__half *,
                                                     const uint32_t *,
                                                     int,int,int,
                                                     cudaStream_t);
extern template void launch_prefix_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,
                                                            const uint32_t *,
                                                            int,int,int,
                                                            cudaStream_t);
#endif

//------------------------------------------------------------------------------
// RoPE (Rotary Position Embeddings) kernels
//------------------------------------------------------------------------------

/**
 * Apply rotary position embeddings in-place (forward).
 * data: [batch_heads, seq_len, head_dim], head_dim must be even.
 * style: CAIF_ROPE_INTERLEAVED or CAIF_ROPE_HALF_SPLIT.
 * Internal trig stays fp32; load/store casts through T.
 * Full-rotation form: rotates every dim of every head. For partial
 * rotary (Glm4Moe-style), use launch_rope_forward_partial below.
 */
template<typename T>
void launch_rope_forward(T *data,
                         int batch_heads,
                         int seq_len,
                         int head_dim,
                         float base,
                         int style,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_rope_forward<float>(float *,
                                                int,int,int,float,int,
                                                cudaStream_t);
extern template void launch_rope_forward<__half>(__half *,
                                                 int,int,int,float,int,
                                                 cudaStream_t);
extern template void launch_rope_forward<__nv_bfloat16>(__nv_bfloat16 *,
                                                        int,int,int,float,
                                                        int,cudaStream_t);
#endif

/**
 * Partial-rotary forward variant.
 * Rotates only the first `rope_dim` dims of each head row (rope_dim
 * must be even, rope_dim<=head_dim). The remaining (head_dim-rope_dim)
 * dims at the end of each head pass through untouched. The half-split
 * pivot is rope_dim/2 (not head_dim/2). Frequencies use rope_dim as
 * the divisor (matches HF rotate_half(x[..., :rope_dim])).
 * Used by Glm4Moe (partial_rotary_factor=0.5 -> rope_dim=head_dim/2).
 */
template<typename T>
void launch_rope_forward_partial(T *data,
                                 int batch_heads,
                                 int seq_len,
                                 int head_dim,
                                 int rope_dim,
                                 float base,
                                 int style,
                                 cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_rope_forward_partial<float>(float *,
                                                        int,int,int,int,
                                                        float,int,cudaStream_t);
extern template void launch_rope_forward_partial<__half>(__half *,
                                                         int,int,int,int,
                                                         float,int,cudaStream_t);
extern template void launch_rope_forward_partial<__nv_bfloat16>(
    __nv_bfloat16 *,int,int,int,int,float,int,cudaStream_t);
#endif

/**
 * Apply rotary position embeddings with position offset (for KV-cache).
 * Positions are computed as (local_pos + pos_offset) for each row.
 * style: CAIF_ROPE_INTERLEAVED or CAIF_ROPE_HALF_SPLIT.
 */
template<typename T>
void launch_rope_forward_offset(T *data,
                                int batch_heads,
                                int seq_len,
                                int head_dim,
                                float base,
                                int pos_offset,
                                int style,
                                cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_rope_forward_offset<float>(float *,
                                                       int,int,int,float,
                                                       int,int,cudaStream_t);
extern template void launch_rope_forward_offset<__half>(__half *,
                                                        int,int,int,float,
                                                        int,int,cudaStream_t);
extern template void launch_rope_forward_offset<__nv_bfloat16>(
    __nv_bfloat16 *,int,int,int,float,int,int,cudaStream_t);
#endif

/**
 * Apply inverse rotary position embeddings in-place (backward).
 * data: [batch_heads, seq_len, head_dim], head_dim must be even.
 * style: CAIF_ROPE_INTERLEAVED or CAIF_ROPE_HALF_SPLIT.
 * Full-rotation form. For partial, use launch_rope_backward_partial.
 */
template<typename T>
void launch_rope_backward(T *data,
                          int batch_heads,
                          int seq_len,
                          int head_dim,
                          float base,
                          int style,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_rope_backward<float>(float *,
                                                 int,int,int,float,int,
                                                 cudaStream_t);
extern template void launch_rope_backward<__half>(__half *,
                                                  int,int,int,float,int,
                                                  cudaStream_t);
extern template void launch_rope_backward<__nv_bfloat16>(__nv_bfloat16 *,
                                                         int,int,int,float,
                                                         int,cudaStream_t);
#endif

/**
 * Partial-rotary backward variant. Pairs with launch_rope_forward_partial;
 * rope_dim must match the forward call.
 */
template<typename T>
void launch_rope_backward_partial(T *data,
                                  int batch_heads,
                                  int seq_len,
                                  int head_dim,
                                  int rope_dim,
                                  float base,
                                  int style,
                                  cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_rope_backward_partial<float>(float *,
                                                         int,int,int,int,
                                                         float,int,cudaStream_t);
extern template void launch_rope_backward_partial<__half>(__half *,
                                                          int,int,int,int,
                                                          float,int,cudaStream_t);
extern template void launch_rope_backward_partial<__nv_bfloat16>(
    __nv_bfloat16 *,int,int,int,int,float,int,cudaStream_t);
#endif

//------------------------------------------------------------------------------
// GQA (Grouped-Query Attention) kernels
//------------------------------------------------------------------------------

/**
 * Repeat KV heads for GQA expansion.
 * input: [batch * num_kv_heads, seq_len, head_dim]
 * output: [batch * num_heads, seq_len, head_dim]
 * where num_heads = num_kv_heads * repeat_factor
 * Pure data shuffle — no arithmetic.
 */
template<typename T>
void launch_gqa_repeat_kv(const T *input,
                          T *output,
                          int batch,
                          int num_kv_heads,
                          int repeat_factor,
                          int seq_len,
                          int head_dim,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gqa_repeat_kv<float>(const float *,float *,
                                                 int,int,int,int,int,
                                                 cudaStream_t);
extern template void launch_gqa_repeat_kv<__half>(const __half *,__half *,
                                                  int,int,int,int,int,
                                                  cudaStream_t);
extern template void launch_gqa_repeat_kv<__nv_bfloat16>(
    const __nv_bfloat16 *,__nv_bfloat16 *,int,int,int,int,int,cudaStream_t);
#endif

/**
 * Sum-reduce expanded gradients back to kv_heads for GQA backward.
 * input: [batch * num_heads, seq_len, head_dim]
 * output: [batch * num_kv_heads, seq_len, head_dim]
 * where num_heads = num_kv_heads * repeat_factor
 * Reduction accumulates in fp32; stores T.
 */
template<typename T>
void launch_gqa_reduce_kv(const T *input,
                          T *output,
                          int batch,
                          int num_kv_heads,
                          int repeat_factor,
                          int seq_len,
                          int head_dim,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gqa_reduce_kv<float>(const float *,float *,
                                                 int,int,int,int,int,
                                                 cudaStream_t);
extern template void launch_gqa_reduce_kv<__half>(const __half *,__half *,
                                                  int,int,int,int,int,
                                                  cudaStream_t);
extern template void launch_gqa_reduce_kv<__nv_bfloat16>(
    const __nv_bfloat16 *,__nv_bfloat16 *,int,int,int,int,int,cudaStream_t);
#endif

//------------------------------------------------------------------------------
// KV-Cache kernels
//------------------------------------------------------------------------------

/**
 * Append new K/V rows to the KV cache (original layout).
 * Copies new_kv[batch, new_len, num_kv_heads, head_dim] to
 * cache[batch, cache_pos:cache_pos+new_len, num_kv_heads, head_dim].
 * Pure data copy.
 */
template<typename T>
void launch_kv_cache_append(const T *new_kv,
                            T *cache,
                            int batch,
                            int new_len,
                            int cache_pos,
                            int max_seq_len,
                            int num_kv_heads,
                            int head_dim,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_kv_cache_append<float>(const float *,float *,
                                                   int,int,int,int,int,int,
                                                   cudaStream_t);
extern template void launch_kv_cache_append<__half>(const __half *,__half *,
                                                    int,int,int,int,int,int,
                                                    cudaStream_t);
extern template void launch_kv_cache_append<__nv_bfloat16>(
    const __nv_bfloat16 *,__nv_bfloat16 *,int,int,int,int,int,int,
    cudaStream_t);
#endif

/**
 * Append new K/V to cache (transposed layout).
 * Copies new_kv[batch_kv_heads, new_len, head_dim] to
 * cache[batch_kv_heads, cache_pos:cache_pos+new_len, head_dim].
 * This layout matches the attention format after head splitting.
 */
template<typename T>
void launch_kv_cache_append_transposed(const T *new_kv,
                                       T *cache,
                                       int batch_kv_heads,
                                       int new_len,
                                       int cache_pos,
                                       int max_seq_len,
                                       int head_dim,
                                       cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_kv_cache_append_transposed<float>(const float *,
                                                              float *,
                                                              int,int,int,
                                                              int,int,
                                                              cudaStream_t);
extern template void launch_kv_cache_append_transposed<__half>(const __half *,
                                                               __half *,
                                                               int,int,int,
                                                               int,int,
                                                               cudaStream_t);
extern template void launch_kv_cache_append_transposed<__nv_bfloat16>(
    const __nv_bfloat16 *,__nv_bfloat16 *,int,int,int,int,int,cudaStream_t);
#endif

