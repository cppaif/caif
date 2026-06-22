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
// FlashAttention-2 cross-attention launcher declarations. Definitions live in
// src/caif_cuda_kernels_flash_cross.cu. CPU-only builds
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
// FlashAttention Cross-Attention kernels
// Same algorithm as self-attention but Q has q_seq_len, K/V have kv_seq_len.
// No causal mask (decoder attends to all encoder positions).
//------------------------------------------------------------------------------

/**
 * FlashAttention-2 cross-attention forward pass.
 * Q: [batch_heads, q_seq_len, head_dim]
 * K, V: [batch_heads, kv_seq_len, head_dim]
 * O: [batch_heads, q_seq_len, head_dim] - output
 * L: [batch_heads, q_seq_len] - logsumexp values (needed for backward)
 */
template<typename T>
void launch_flash_attention_forward_cross(const T *Q,
                                          const T *K,
                                          const T *V,
                                          T *O,
                                          float *L,
                                          int batch_heads,
                                          int q_seq_len,
                                          int kv_seq_len,
                                          int head_dim,
                                          float scale,
                                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_flash_attention_forward_cross<float>(const float *,
                                                                 const float *,
                                                                 const float *,
                                                                 float *,
                                                                 float *,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 float,
                                                                 cudaStream_t);
extern template void launch_flash_attention_forward_cross<__half>(const __half *,
                                                                  const __half *,
                                                                  const __half *,
                                                                  __half *,
                                                                  float *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  float,
                                                                  cudaStream_t);
extern template void launch_flash_attention_forward_cross<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                         const __nv_bfloat16 *,
                                                                         const __nv_bfloat16 *,
                                                                         __nv_bfloat16 *,
                                                                         float *,
                                                                         int,
                                                                         int,
                                                                         int,
                                                                         int,
                                                                         float,
                                                                         cudaStream_t);
#endif

/**
 * FlashAttention-2 cross-attention backward pass.
 * Q, O: [batch_heads, q_seq_len, head_dim]
 * K, V: [batch_heads, kv_seq_len, head_dim]
 * dO: [batch_heads, q_seq_len, head_dim]
 * L: [batch_heads, q_seq_len] - logsumexp from forward
 * dQ: [batch_heads, q_seq_len, head_dim]
 * dK, dV: [batch_heads, kv_seq_len, head_dim]
 */
template<typename T>
void launch_flash_attention_backward_cross(const T *Q,
                                           const T *K,
                                           const T *V,
                                           const T *O,
                                           const T *dO,
                                           const float *L,
                                           T *dQ,
                                           T *dK,
                                           T *dV,
                                           int batch_heads,
                                           int q_seq_len,
                                           int kv_seq_len,
                                           int head_dim,
                                           float scale,
                                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_flash_attention_backward_cross<float>(const float *,
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
                                                                  int,
                                                                  float,
                                                                  cudaStream_t);
extern template void launch_flash_attention_backward_cross<__half>(const __half *,
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
                                                                   int,
                                                                   float,
                                                                   cudaStream_t);
extern template void launch_flash_attention_backward_cross<__nv_bfloat16>(const __nv_bfloat16 *,
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
                                                                          int,
                                                                          float,
                                                                          cudaStream_t);
#endif
