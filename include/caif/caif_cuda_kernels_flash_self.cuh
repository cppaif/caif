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
// FlashAttention-2 self-attention launcher declarations. Definitions live in
// src/caif_cuda_kernels_flash_self.cu. CPU-only builds
// link the no-op stubs in legacy/src/caif_cuda_kernels_cpu.cpp, same contract
// as caif_cuda_kernels.h.
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
// FlashAttention kernels
//------------------------------------------------------------------------------

/**
 * FlashAttention-2 forward pass.
 * Computes scaled dot-product attention without materializing the full attention matrix.
 * Uses online softmax and tiled computation for memory efficiency.
 *
 * Q, K, V: [batch_heads, seq_len, head_dim]
 * O: [batch_heads, seq_len, head_dim] - output
 * L: [batch_heads, seq_len] - logsumexp values (needed for backward)
 *
 * Supports causal masking when causal=true.
 */
template<typename T>
void launch_flash_attention_forward(const T *Q,
                                    const T *K,
                                    const T *V,
                                    T *O,
                                    float *L,
                                    int batch_heads,
                                    int seq_len,
                                    int head_dim,
                                    float scale,
                                    float softcap,
                                    int causal,
                                    int window,
                                    int num_heads,
                                    int num_kv_heads,
                                    const float *alibi_slopes,
                                    cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_flash_attention_forward<float>(const float *,
                                                           const float *,
                                                           const float *,
                                                           float *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           float,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           const float *,
                                                           cudaStream_t);
extern template void launch_flash_attention_forward<__half>(const __half *,
                                                            const __half *,
                                                            const __half *,
                                                            __half *,
                                                            float *,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            float,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            const float *,
                                                            cudaStream_t);
extern template void launch_flash_attention_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   __nv_bfloat16 *,
                                                                   float *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   float,
                                                                   float,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   const float *,
                                                                   cudaStream_t);
#endif

/**
 * FlashAttention-2 forward pass with prefix-LM mask.
 * Allowed iff (k<=q) OR (k<prefix_lens[batch]); otherwise masked (-INFINITY).
 * prefix_lens: device pointer [batch_size]; per-row prefix length in tokens.
 */
template<typename T>
void launch_flash_attention_forward_prefix(const T *Q,
                                           const T *K,
                                           const T *V,
                                           T *O,
                                           float *L,
                                           const uint32_t *prefix_lens,
                                           int batch_size,
                                           int num_heads,
                                           int num_kv_heads,
                                           int seq_len,
                                           int head_dim,
                                           float scale,
                                           float softcap,
                                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_flash_attention_forward_prefix<float>(const float *,
                                                                  const float *,
                                                                  const float *,
                                                                  float *,
                                                                  float *,
                                                                  const uint32_t *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  float,
                                                                  float,
                                                                  cudaStream_t);
extern template void launch_flash_attention_forward_prefix<__half>(const __half *,
                                                                   const __half *,
                                                                   const __half *,
                                                                   __half *,
                                                                   float *,
                                                                   const uint32_t *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   float,
                                                                   float,
                                                                   cudaStream_t);
extern template void launch_flash_attention_forward_prefix<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                          const __nv_bfloat16 *,
                                                                          const __nv_bfloat16 *,
                                                                          __nv_bfloat16 *,
                                                                          float *,
                                                                          const uint32_t *,
                                                                          int,
                                                                          int,
                                                                          int,
                                                                          int,
                                                                          int,
                                                                          float,
                                                                          float,
                                                                          cudaStream_t);
#endif

/**
 * FlashAttention-2 backward pass.
 * Computes gradients dQ, dK, dV without materializing the full attention matrix.
 * Recomputes attention on-the-fly using stored logsumexp values.
 *
 * Q, K, V, O: [batch_heads, seq_len, head_dim] - from forward
 * dO: [batch_heads, seq_len, head_dim] - gradient of output
 * L: [batch_heads, seq_len] - logsumexp from forward
 * dQ, dK, dV: [batch_heads, seq_len, head_dim] - output gradients
 */
template<typename T>
void launch_flash_attention_backward(const T *Q,
                                     const T *K,
                                     const T *V,
                                     const T *O,
                                     const T *dO,
                                     const float *L,
                                     T *dQ,
                                     T *dK,
                                     T *dV,
                                     int batch_heads,
                                     int seq_len,
                                     int head_dim,
                                     float scale,
                                     float softcap,
                                     int causal,
                                     int window,
                                     int num_heads,
                                     const float *alibi_slopes,
                                     cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_flash_attention_backward<float>(const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            float *,
                                                            float *,
                                                            float *,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            float,
                                                            int,
                                                            int,
                                                            int,
                                                            const float *,
                                                            cudaStream_t);
extern template void launch_flash_attention_backward<__half>(const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const float *,
                                                             __half *,
                                                             __half *,
                                                             __half *,
                                                             int,
                                                             int,
                                                             int,
                                                             float,
                                                             float,
                                                             int,
                                                             int,
                                                             int,
                                                             const float *,
                                                             cudaStream_t);
extern template void launch_flash_attention_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const float *,
                                                                    __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    float,
                                                                    float,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    const float *,
                                                                    cudaStream_t);
#endif

/**
 * FlashAttention-2 backward pass with prefix-LM mask.
 * Mask semantics identical to launch_flash_attention_forward_prefix.
 */
template<typename T>
void launch_flash_attention_backward_prefix(const T *Q,
                                            const T *K,
                                            const T *V,
                                            const T *O,
                                            const T *dO,
                                            const float *L,
                                            T *dQ,
                                            T *dK,
                                            T *dV,
                                            const uint32_t *prefix_lens,
                                            int batch_size,
                                            int num_heads,
                                            int seq_len,
                                            int head_dim,
                                            float scale,
                                            float softcap,
                                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_flash_attention_backward_prefix<float>(const float *,
                                                                   const float *,
                                                                   const float *,
                                                                   const float *,
                                                                   const float *,
                                                                   const float *,
                                                                   float *,
                                                                   float *,
                                                                   float *,
                                                                   const uint32_t *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   float,
                                                                   float,
                                                                   cudaStream_t);
extern template void launch_flash_attention_backward_prefix<__half>(const __half *,
                                                                    const __half *,
                                                                    const __half *,
                                                                    const __half *,
                                                                    const __half *,
                                                                    const float *,
                                                                    __half *,
                                                                    __half *,
                                                                    __half *,
                                                                    const uint32_t *,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    float,
                                                                    float,
                                                                    cudaStream_t);
extern template void launch_flash_attention_backward_prefix<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                           const __nv_bfloat16 *,
                                                                           const __nv_bfloat16 *,
                                                                           const __nv_bfloat16 *,
                                                                           const __nv_bfloat16 *,
                                                                           const float *,
                                                                           __nv_bfloat16 *,
                                                                           __nv_bfloat16 *,
                                                                           __nv_bfloat16 *,
                                                                           const uint32_t *,
                                                                           int,
                                                                           int,
                                                                           int,
                                                                           int,
                                                                           float,
                                                                           float,
                                                                           cudaStream_t);
#endif
