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
// Embedding CUDA kernels: token embedding lookup/backward, float-id
// conversion, patch embedding (extract/backward, CLS prepend/grad),
// positional encoding add + table backward.
// Launcher declarations for src/caif_cuda_kernels_embeddings.cu
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
// Embedding kernels
//------------------------------------------------------------------------------

/**
 * Gather rows from embedding table by uint32 token ID.
 * output[token_idx * dim + d] = table[token_ids[token_idx] * dim + d]
 *
 * Templated on T = float / __half / __nv_bfloat16. The table and the
 * output share T; token IDs are always uint32.
 */
template<typename T>
void launch_embedding_lookup(const T *table,
                             const unsigned int *token_ids,
                             T *output,
                             int num_tokens,
                             int dim,
                             cudaStream_t stream);

extern template void launch_embedding_lookup<float>(const float *,
                                                    const unsigned int *,
                                                    float *,
                                                    int,
                                                    int,
                                                    cudaStream_t);
extern template void launch_embedding_lookup<__half>(const __half *,
                                                     const unsigned int *,
                                                     __half *,
                                                     int,
                                                     int,
                                                     cudaStream_t);
extern template void launch_embedding_lookup<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const unsigned int *,
                                                            __nv_bfloat16 *,
                                                            int,
                                                            int,
                                                            cudaStream_t);

/**
 * Gather rows from embedding table by float-encoded token ID.
 * token_id = (uint32_t)float_ids[token_idx]
 */
template<typename T>
void launch_embedding_lookup_float(const T *table,
                                   const float *float_ids,
                                   T *output,
                                   int num_tokens,
                                   int dim,
                                   cudaStream_t stream);

extern template void launch_embedding_lookup_float<float>(const float *,
                                                          const float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          cudaStream_t);
extern template void launch_embedding_lookup_float<__half>(const __half *,
                                                           const float *,
                                                           __half *,
                                                           int,
                                                           int,
                                                           cudaStream_t);
extern template void launch_embedding_lookup_float<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  const float *,
                                                                  __nv_bfloat16 *,
                                                                  int,
                                                                  int,
                                                                  cudaStream_t);

/**
 * Convert float-encoded token IDs to uint32 on GPU.
 * Eliminates host roundtrip for training path.
 *
 * fp32-only by contract: this IS the fp32→uint dtype-conversion utility.
 */
void launch_float_to_uint(const float *float_ids,
                          unsigned int *uint_ids,
                          int64_t n,
                          cudaStream_t stream);

/**
 * Scatter-add gradients back to embedding table.
 * grad_table must be pre-zeroed. Uses atomicAdd.
 *
 * Templated on the storage dtype T of grad_output. The gradient table is
 * always fp32 and accumulated via atomicAdd in fp32 regardless of T, so
 * repeated tokens (common in LM training) keep full precision in bf16/fp16.
 */
template<typename T>
void launch_embedding_backward(const T *grad_output,
                               const unsigned int *token_ids,
                               float *grad_table,
                               int num_tokens,
                               int dim,
                               cudaStream_t stream);

extern template void launch_embedding_backward<float>(const float *,
                                                      const unsigned int *,
                                                      float *,
                                                      int,
                                                      int,
                                                      cudaStream_t);
extern template void launch_embedding_backward<__half>(const __half *,
                                                       const unsigned int *,
                                                       float *,
                                                       int,
                                                       int,
                                                       cudaStream_t);
extern template void launch_embedding_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const unsigned int *,
                                                              float *,
                                                              int,
                                                              int,
                                                              cudaStream_t);

//------------------------------------------------------------------------------
// Patch embedding kernels
//------------------------------------------------------------------------------

/**
 * Extract non-overlapping patches from BHWC images.
 * input:  [batch, height, width, channels]
 * output: [batch * num_patches, patch_flat_dim]
 */
template<typename T>
void launch_extract_patches(const T *input,
                            T *output,
                            int batch,
                            int height,
                            int width,
                            int channels,
                            int patch_size,
                            int num_patches_h,
                            int num_patches_w,
                            int patch_flat_dim,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_extract_patches<float>(const float *,float *,
                                                   int,int,int,int,
                                                   int,int,int,int,cudaStream_t);
extern template void launch_extract_patches<__half>(const __half *,__half *,
                                                    int,int,int,int,
                                                    int,int,int,int,cudaStream_t);
extern template void launch_extract_patches<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,int,int,int,
                                                           int,int,int,int,cudaStream_t);
#endif

/**
 * Scatter-add patch gradients back to image layout (col2im).
 * grad_input must be pre-zeroed. Uses atomicAdd.
 */
template<typename T>
void launch_extract_patches_backward(const T *grad_patches,
                                     T *grad_input,
                                     int batch,
                                     int height,
                                     int width,
                                     int channels,
                                     int patch_size,
                                     int num_patches_h,
                                     int num_patches_w,
                                     int patch_flat_dim,
                                     cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_extract_patches_backward<float>(const float *,float *,
                                                            int,int,int,int,
                                                            int,int,int,int,cudaStream_t);
extern template void launch_extract_patches_backward<__half>(const __half *,__half *,
                                                             int,int,int,int,
                                                             int,int,int,int,cudaStream_t);
extern template void launch_extract_patches_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    int,int,int,int,
                                                                    int,int,int,int,cudaStream_t);
#endif

/**
 * Prepend CLS token at position 0 of each sequence.
 * patches: [batch, num_patches, dim], cls: [1, dim]
 * output:  [batch, num_patches+1, dim]
 */
template<typename T>
void launch_cls_prepend(const T *patches,
                        const T *cls_token,
                        T *output,
                        int batch,
                        int num_patches,
                        int dim,
                        cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cls_prepend<float>(const float *,const float *,float *,
                                               int,int,int,cudaStream_t);
extern template void launch_cls_prepend<__half>(const __half *,const __half *,__half *,
                                                int,int,int,cudaStream_t);
extern template void launch_cls_prepend<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const __nv_bfloat16 *,
                                                       __nv_bfloat16 *,
                                                       int,int,int,cudaStream_t);
#endif

/**
 * Split CLS gradient from patch gradients.
 * grad_output: [batch, num_patches+1, dim]
 * grad_cls: [1, dim] (summed over batch), grad_patches: [batch, num_patches, dim]
 */
template<typename T>
void launch_cls_grad_extract(const T *grad_output,
                             T *grad_cls,
                             T *grad_patches,
                             int batch,
                             int num_patches,
                             int dim,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cls_grad_extract<float>(const float *,float *,float *,
                                                    int,int,int,cudaStream_t);
extern template void launch_cls_grad_extract<__half>(const __half *,__half *,__half *,
                                                     int,int,int,cudaStream_t);
extern template void launch_cls_grad_extract<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int,int,int,cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Positional encoding kernels
//------------------------------------------------------------------------------

/**
 * Add positional encoding table rows to input (broadcast over batch).
 * output[b,s,d] = input[b,s,d] + pe_table[s,d]
 */
template<typename T>
void launch_add_positional_encoding(const T *input,
                                    const T *pe_table,
                                    T *output,
                                    int batch,
                                    int seq_len,
                                    int dim,
                                    cudaStream_t stream);

/**
 * Accumulate PE gradient by summing over batch dimension.
 * grad_table[s,d] = sum_b grad_output[b,s,d]
 * Accumulates in float for numerical stability regardless of T.
 */
template<typename T>
void launch_pe_table_backward(const T *grad_output,
                              T *grad_table,
                              int batch,
                              int seq_len,
                              int dim,
                              cudaStream_t stream);

