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
// Element-wise + generic-reduction launcher declarations. Definitions live in
// src/caif_cuda_kernels_elementwise.cu. CPU-only
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
// Element-wise operation kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_elementwise_add(const T *a,
                            const T *b,
                            T *result,
                            int64_t n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_add<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int64_t,
                                                   cudaStream_t);
extern template void launch_elementwise_add<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_elementwise_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_add_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int64_t n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_add_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int64_t,
                                                          cudaStream_t);
extern template void launch_elementwise_add_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int64_t,
                                                           cudaStream_t);
extern template void launch_elementwise_add_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int64_t,
                                                                  cudaStream_t);
#endif

// Bias add (broadcast bias over rows for 2D tensors)
template<typename T>
void launch_bias_add_2d(const T *input,
                        const T *bias,
                        T *output,
                        int batch,
                        int units,
                        cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_bias_add_2d<float>(const float *,
                                               const float *,
                                               float *,
                                               int,
                                               int,
                                               cudaStream_t);
extern template void launch_bias_add_2d<__half>(const __half *,
                                                const __half *,
                                                __half *,
                                                int,
                                                int,
                                                cudaStream_t);
extern template void launch_bias_add_2d<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const __nv_bfloat16 *,
                                                       __nv_bfloat16 *,
                                                       int,
                                                       int,
                                                       cudaStream_t);
#endif

// Bias gradient (sum over batch rows)
template<typename T>
void launch_bias_grad_2d(const T *grad_output,
                         T *bias_grad,
                         int batch,
                         int units,
                         cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_bias_grad_2d<float>(const float *,
                                                float *,
                                                int,
                                                int,
                                                cudaStream_t);
extern template void launch_bias_grad_2d<__half>(const __half *,
                                                 __half *,
                                                 int,
                                                 int,
                                                 cudaStream_t);
extern template void launch_bias_grad_2d<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int,
                                                        int,
                                                        cudaStream_t);
#endif

template<typename T>
void launch_elementwise_sub(const T *a,
                            const T *b,
                            T *result,
                            int64_t n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_sub<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int64_t,
                                                   cudaStream_t);
extern template void launch_elementwise_sub<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_elementwise_sub<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_sub_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int64_t n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_sub_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int64_t,
                                                          cudaStream_t);
extern template void launch_elementwise_sub_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int64_t,
                                                           cudaStream_t);
extern template void launch_elementwise_sub_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int64_t,
                                                                  cudaStream_t);
#endif

template<typename T>
void launch_elementwise_mul(const T *a,
                            const T *b,
                            T *result,
                            int64_t n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_mul<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int64_t,
                                                   cudaStream_t);
extern template void launch_elementwise_mul<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_elementwise_mul<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_mul_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int64_t n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_mul_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int64_t,
                                                          cudaStream_t);
extern template void launch_elementwise_mul_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int64_t,
                                                           cudaStream_t);
extern template void launch_elementwise_mul_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int64_t,
                                                                  cudaStream_t);
#endif

template<typename T>
void launch_elementwise_div(const T *a,
                            const T *b,
                            T *result,
                            int64_t n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_div<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int64_t,
                                                   cudaStream_t);
extern template void launch_elementwise_div<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_elementwise_div<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_div_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int64_t n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_div_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int64_t,
                                                          cudaStream_t);
extern template void launch_elementwise_div_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int64_t,
                                                           cudaStream_t);
extern template void launch_elementwise_div_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int64_t,
                                                                  cudaStream_t);
#endif

template<typename T>
void launch_elementwise_sqrt(const T *a,
                             T *result,
                             int64_t n,
                             cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_sqrt<float>(const float *,
                                                    float *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_elementwise_sqrt<__half>(const __half *,
                                                     __half *,
                                                     int64_t,
                                                     cudaStream_t);
extern template void launch_elementwise_sqrt<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int64_t,
                                                            cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Reduction kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_reduction_sum(const T *input,
                          float *output,
                          int64_t n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_reduction_sum<float>(const float *,
                                                 float *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_reduction_sum<__half>(const __half *,
                                                  float *,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_reduction_sum<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         float *,
                                                         int64_t,
                                                         cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Generic reductions (sum_axis0 / sum_axis1 / sum_of_squares / logsumexp)
//------------------------------------------------------------------------------
template<typename T>
void launch_sum_axis0(const T *input,
                      float *output,
                      int batch,
                      int dim,
                      cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sum_axis0<float>(const float *,
                                             float *,
                                             int,
                                             int,
                                             cudaStream_t);
extern template void launch_sum_axis0<__half>(const __half *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);
extern template void launch_sum_axis0<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     float *,
                                                     int,
                                                     int,
                                                     cudaStream_t);
#endif

/**
 * Sum along axis 1 (over dim): output[b] = sum_d input[b,d]
 */
template<typename T>
void launch_sum_axis1(const T *input,
                      float *output,
                      int batch,
                      int dim,
                      cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sum_axis1<float>(const float *,
                                             float *,
                                             int,
                                             int,
                                             cudaStream_t);
extern template void launch_sum_axis1<__half>(const __half *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);
extern template void launch_sum_axis1<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     float *,
                                                     int,
                                                     int,
                                                     cudaStream_t);
#endif

/**
 * Sum of squares: output[0] += sum(input[i]^2).
 * Caller must zero output before launch.
 */
template<typename T>
void launch_sum_of_squares(const T *input,
                           float *output,
                           int64_t n,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sum_of_squares<float>(const float *,
                                                  float *,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_sum_of_squares<__half>(const __half *,
                                                   float *,
                                                   int64_t,
                                                   cudaStream_t);
extern template void launch_sum_of_squares<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          float *,
                                                          int64_t,
                                                          cudaStream_t);
#endif

/**
 * Log-sum-exp along last axis: output[b] = log(sum_d exp(input[b,d]))
 * With numerical stability (max subtraction).
 */
template<typename T>
void launch_logsumexp(const T *input,
                      float *output,
                      int batch,
                      int dim,
                      cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_logsumexp<float>(const float *,
                                             float *,
                                             int,
                                             int,
                                             cudaStream_t);
extern template void launch_logsumexp<__half>(const __half *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);
extern template void launch_logsumexp<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     float *,
                                                     int,
                                                     int,
                                                     cudaStream_t);
#endif
