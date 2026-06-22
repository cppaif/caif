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
// Loss CUDA kernels: cross-entropy (basic, index-target, logits,
// fused) and MSE, forward + backward, plus loss reductions.
// Launcher declarations for src/caif_cuda_kernels_loss.cu
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

template<typename T>
void launch_mse_loss(const T *predictions,
                     const T *targets,
                     float *loss,
                     int n,
                     cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_mse_loss<float>(const float *,
                                            const float *,
                                            float *,
                                            int,
                                            cudaStream_t);
extern template void launch_mse_loss<__half>(const __half *,
                                             const __half *,
                                             float *,
                                             int,
                                             cudaStream_t);
extern template void launch_mse_loss<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    float *,
                                                    int,
                                                    cudaStream_t);
#endif

template<typename T>
void launch_mse_gradient(const T *predictions,
                         const T *targets,
                         T *gradient,
                         int n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_mse_gradient<float>(const float *,
                                                const float *,
                                                float *,
                                                int,
                                                cudaStream_t);
extern template void launch_mse_gradient<__half>(const __half *,
                                                 const __half *,
                                                 __half *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_mse_gradient<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int,
                                                        cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Cross-entropy loss from logits (numerically stable for language modeling)
//------------------------------------------------------------------------------

/**
 * Compute per-position cross-entropy loss from raw logits.
 * Uses log-softmax formulation: loss[i] = -logits[target[i]] + log(sum(exp(logits)))
 * With numerical stability: subtracts max(logits) before exp.
 * logits: [N, vocab_size], targets: [N] (float-encoded token IDs)
 * losses: [N] (per-position loss)
 */
template<typename T>
void launch_cross_entropy_logits_forward(const T *logits,
                                         const float *targets,
                                         float *losses,
                                         int n,
                                         int vocab_size,
                                         int ignore_index,
                                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cross_entropy_logits_forward<float>(const float *,
                                                                const float *,
                                                                float *,
                                                                int,
                                                                int,
                                                                int,
                                                                cudaStream_t);
extern template void launch_cross_entropy_logits_forward<__half>(const __half *,
                                                                 const float *,
                                                                 float *,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 cudaStream_t);
extern template void launch_cross_entropy_logits_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                        const float *,
                                                                        float *,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        cudaStream_t);
#endif

/**
 * Compute gradient of cross-entropy loss w.r.t. logits.
 * grad[i,j] = softmax(logits)[i,j] - (j == target[i] ? 1 : 0)
 * Divided by valid_count for mean reduction.
 * logits: [N, vocab_size], targets: [N]
 * grad: [N, vocab_size]
 */
template<typename T>
void launch_cross_entropy_logits_backward(const T *logits,
                                          const float *targets,
                                          T *grad,
                                          int n,
                                          int vocab_size,
                                          int ignore_index,
                                          float scale,
                                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cross_entropy_logits_backward<float>(const float *,
                                                                 const float *,
                                                                 float *,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 float,
                                                                 cudaStream_t);
extern template void launch_cross_entropy_logits_backward<__half>(const __half *,
                                                                  const float *,
                                                                  __half *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  float,
                                                                  cudaStream_t);
extern template void launch_cross_entropy_logits_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                         const float *,
                                                                         __nv_bfloat16 *,
                                                                         int,
                                                                         int,
                                                                         int,
                                                                         float,
                                                                         cudaStream_t);
#endif

/**
 * Reduce per-position losses to scalar mean (excluding ignored positions).
 * losses: [N], output: scalar
 *
 * fp32-only by contract: CE reduction accumulator is fp32 regardless of
 * logits StorageT — operates on already-reduced per-position losses and
 * float-encoded index targets.
 */
void launch_cross_entropy_reduce_mean(const float *losses,
                                      const float *targets,
                                      float *output,
                                      int n,
                                      int ignore_index,
                                      cudaStream_t stream);

/**
 * Fused cross-entropy forward+backward. Computes loss AND gradient in one pass.
 * Eliminates host roundtrip for valid_count and halves logits memory reads.
 * logits: [N, vocab_size], targets: [N], losses: [N], grad: [N, vocab_size]
 * result: [2] floats (must be pre-zeroed) — result[0]=valid_count, result[1]=loss_sum
 */
template<typename T>
void launch_cross_entropy_fused(const T *logits,
                                const float *targets,
                                float *losses,
                                T *grad,
                                float *result,
                                int n,
                                int vocab_size,
                                int ignore_index,
                                cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cross_entropy_fused<float>(const float *,
                                                       const float *,
                                                       float *,
                                                       float *,
                                                       float *,
                                                       int,
                                                       int,
                                                       int,
                                                       cudaStream_t);
extern template void launch_cross_entropy_fused<__half>(const __half *,
                                                        const float *,
                                                        float *,
                                                        __half *,
                                                        float *,
                                                        int,
                                                        int,
                                                        int,
                                                        cudaStream_t);
extern template void launch_cross_entropy_fused<__nv_bfloat16>(const __nv_bfloat16 *,
                                                               const float *,
                                                               float *,
                                                               __nv_bfloat16 *,
                                                               float *,
                                                               int,
                                                               int,
                                                               int,
                                                               cudaStream_t);
#endif

