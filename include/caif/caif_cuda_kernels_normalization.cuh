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
// RMSNorm / LayerNorm launcher declarations. Definitions live in
// src/caif_cuda_kernels_normalization.cu. CPU-only
// builds link the no-op stubs in legacy/src/caif_cuda_kernels_cpu.cpp, same
// contract as caif_cuda_kernels.h.
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
// RMSNorm kernels
// Templated on activation type T ∈ {float, __half, __nv_bfloat16}.
// gamma / rms_cache / grad_gamma stay fp32 (standard autocast convention:
// weights and per-row statistics kept at high precision).
//------------------------------------------------------------------------------
template<typename T>
void launch_rmsnorm_forward(const T *input,
                            const float *gamma,
                            T *output,
                            float *rms_cache,
                            float epsilon,
                            int rows,
                            int dim,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_rmsnorm_forward<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   float *,
                                                   float,
                                                   int,
                                                   int,
                                                   cudaStream_t);
extern template void launch_rmsnorm_forward<__half>(const __half *,
                                                    const float *,
                                                    __half *,
                                                    float *,
                                                    float,
                                                    int,
                                                    int,
                                                    cudaStream_t);
extern template void launch_rmsnorm_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const float *,
                                                           __nv_bfloat16 *,
                                                           float *,
                                                           float,
                                                           int,
                                                           int,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_rmsnorm_backward(const T *grad_output,
                             const T *input,
                             const float *gamma,
                             const float *rms_cache,
                             T *grad_input,
                             float *grad_gamma,
                             float epsilon,
                             int rows,
                             int dim,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_rmsnorm_backward<float>(const float *,
                                                    const float *,
                                                    const float *,
                                                    const float *,
                                                    float *,
                                                    float *,
                                                    float,
                                                    int,
                                                    int,
                                                    cudaStream_t);
extern template void launch_rmsnorm_backward<__half>(const __half *,
                                                     const __half *,
                                                     const float *,
                                                     const float *,
                                                     __half *,
                                                     float *,
                                                     float,
                                                     int,
                                                     int,
                                                     cudaStream_t);
extern template void launch_rmsnorm_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            const float *,
                                                            const float *,
                                                            __nv_bfloat16 *,
                                                            float *,
                                                            float,
                                                            int,
                                                            int,
                                                            cudaStream_t);
#endif

//------------------------------------------------------------------------------
// LayerNorm kernels
// gamma / beta / mean_cache / rstd_cache / grad_gamma / grad_beta stay fp32.
//------------------------------------------------------------------------------
template<typename T>
void launch_layernorm_forward(const T *input,
                              const float *gamma,
                              const float *beta,
                              T *output,
                              float *mean_cache,
                              float *rstd_cache,
                              float epsilon,
                              int rows,
                              int dim,
                              cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_layernorm_forward<float>(const float *,
                                                     const float *,
                                                     const float *,
                                                     float *,
                                                     float *,
                                                     float *,
                                                     float,
                                                     int,
                                                     int,
                                                     cudaStream_t);
extern template void launch_layernorm_forward<__half>(const __half *,
                                                      const float *,
                                                      const float *,
                                                      __half *,
                                                      float *,
                                                      float *,
                                                      float,
                                                      int,
                                                      int,
                                                      cudaStream_t);
extern template void launch_layernorm_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             const float *,
                                                             const float *,
                                                             __nv_bfloat16 *,
                                                             float *,
                                                             float *,
                                                             float,
                                                             int,
                                                             int,
                                                             cudaStream_t);
#endif

template<typename T>
void launch_layernorm_backward(const T *grad_output,
                               const T *input,
                               const float *gamma,
                               const float *mean_cache,
                               const float *rstd_cache,
                               T *grad_input,
                               float *grad_gamma,
                               float *grad_beta,
                               int rows,
                               int dim,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_layernorm_backward<float>(const float *,
                                                      const float *,
                                                      const float *,
                                                      const float *,
                                                      const float *,
                                                      float *,
                                                      float *,
                                                      float *,
                                                      int,
                                                      int,
                                                      cudaStream_t);
extern template void launch_layernorm_backward<__half>(const __half *,
                                                       const __half *,
                                                       const float *,
                                                       const float *,
                                                       const float *,
                                                       __half *,
                                                       float *,
                                                       float *,
                                                       int,
                                                       int,
                                                       cudaStream_t);
extern template void launch_layernorm_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              const float *,
                                                              const float *,
                                                              const float *,
                                                              __nv_bfloat16 *,
                                                              float *,
                                                              float *,
                                                              int,
                                                              int,
                                                              cudaStream_t);
#endif
