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
// Fused optimizer CUDA kernels: Adam / AdamW, clipped Adam, SGD,
// SGD+momentum, RMSprop, AdaGrad.
// Launcher declarations for src/caif_cuda_kernels_optimizers.cu
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
// Optimizer kernels
//------------------------------------------------------------------------------

/**
 * @brief Fused Adam optimizer update
 * Combines all Adam operations into a single kernel pass.
 * param/grad in storage T; m/v always fp32 master state. Math is fp32.
 */
template<typename T>
void launch_fused_adam(T *param,
                       const T *grad,
                       float *m,
                       float *v,
                       float lr,
                       float beta1,
                       float beta2,
                       float epsilon,
                       float weight_decay,
                       float bias_correction1,
                       float bias_correction2,
                       int64_t n,
                       cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_adam<float>(float *,
                                              const float *,
                                              float *,
                                              float *,
                                              float,
                                              float,
                                              float,
                                              float,
                                              float,
                                              float,
                                              float,
                                              int64_t,
                                              cudaStream_t);
extern template void launch_fused_adam<__half>(__half *,
                                               const __half *,
                                               float *,
                                               float *,
                                               float,
                                               float,
                                               float,
                                               float,
                                               float,
                                               float,
                                               float,
                                               int64_t,
                                               cudaStream_t);
extern template void launch_fused_adam<__nv_bfloat16>(__nv_bfloat16 *,
                                                      const __nv_bfloat16 *,
                                                      float *,
                                                      float *,
                                                      float,
                                                      float,
                                                      float,
                                                      float,
                                                      float,
                                                      float,
                                                      float,
                                                      int64_t,
                                                      cudaStream_t);
#endif

/**
 * @brief Mixed-precision loss-scaler unscale + overflow check.
 * grad *= inv_scale in place; if any element is non-finite, found_inf[0] is
 * set to 1.0f (race-free constant store, no clear). fp32 math so half/bf16
 * overflow is caught. See CAIF_LossScaler.
 */
template<typename T>
void launch_unscale_check_inf(T *grad,
                              float inv_scale,
                              float *found_inf,
                              int64_t n,
                              cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_unscale_check_inf<float>(float *,
                                                     float,
                                                     float *,
                                                     int64_t,
                                                     cudaStream_t);
extern template void launch_unscale_check_inf<__half>(__half *,
                                                      float,
                                                      float *,
                                                      int64_t,
                                                      cudaStream_t);
extern template void launch_unscale_check_inf<__nv_bfloat16>(__nv_bfloat16 *,
                                                             float,
                                                             float *,
                                                             int64_t,
                                                             cudaStream_t);
#endif

/**
 * @brief Multi-tensor ("foreach") fused Adam — ONE launch updates every
 * parameter, replacing the one-launch-per-parameter path. params/grads/ms/vs
 * are device arrays of device pointers; offsets is the element-count prefix sum
 * (length num_tensors+1). Only float is instantiated: the optimizer always
 * updates an fp32 target (param when fp32, else its fp32 master). Math matches
 * launch_fused_adam exactly. See CAIF_AdamOptimizer.
 */
template<typename T>
void launch_multi_tensor_adam(T *const *params,
                              const T *const *grads,
                              float *const *ms,
                              float *const *vs,
                              const int64_t *offsets,
                              int num_tensors,
                              int64_t total_elements,
                              float lr,
                              float beta1,
                              float beta2,
                              float epsilon,
                              float weight_decay,
                              float bias_correction1,
                              float bias_correction2,
                              cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_multi_tensor_adam<float>(float *const *,
                                                     const float *const *,
                                                     float *const *,
                                                     float *const *,
                                                     const int64_t *,
                                                     int,
                                                     int64_t,
                                                     float,
                                                     float,
                                                     float,
                                                     float,
                                                     float,
                                                     float,
                                                     float,
                                                     cudaStream_t);
#endif

// Multi-tensor counterparts of the other four optimizers (one launch each).
// Only float instantiated; state pointer arrays are fp32. See CAIF_Optimizer.
template<typename T>
void launch_multi_tensor_sgd(T *const *params,
                             const T *const *grads,
                             const int64_t *offsets,
                             int num_tensors,
                             int64_t total_elements,
                             float lr,
                             float weight_decay,
                             cudaStream_t stream);

template<typename T>
void launch_multi_tensor_momentum(T *const *params,
                                  const T *const *grads,
                                  float *const *velocities,
                                  const int64_t *offsets,
                                  int num_tensors,
                                  int64_t total_elements,
                                  float lr,
                                  float momentum,
                                  float weight_decay,
                                  cudaStream_t stream);

template<typename T>
void launch_multi_tensor_rmsprop(T *const *params,
                                 const T *const *grads,
                                 float *const *avg_sqs,
                                 const int64_t *offsets,
                                 int num_tensors,
                                 int64_t total_elements,
                                 float lr,
                                 float alpha,
                                 float epsilon,
                                 float weight_decay,
                                 cudaStream_t stream);

template<typename T>
void launch_multi_tensor_adagrad(T *const *params,
                                 const T *const *grads,
                                 float *const *accums,
                                 const int64_t *offsets,
                                 int num_tensors,
                                 int64_t total_elements,
                                 float lr,
                                 float epsilon,
                                 float weight_decay,
                                 cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_multi_tensor_sgd<float>(float *const *,
                                                    const float *const *,
                                                    const int64_t *,
                                                    int,
                                                    int64_t,
                                                    float,
                                                    float,
                                                    cudaStream_t);
extern template void launch_multi_tensor_momentum<float>(float *const *,
                                                         const float *const *,
                                                         float *const *,
                                                         const int64_t *,
                                                         int,
                                                         int64_t,
                                                         float,
                                                         float,
                                                         float,
                                                         cudaStream_t);
extern template void launch_multi_tensor_rmsprop<float>(float *const *,
                                                        const float *const *,
                                                        float *const *,
                                                        const int64_t *,
                                                        int,
                                                        int64_t,
                                                        float,
                                                        float,
                                                        float,
                                                        float,
                                                        cudaStream_t);
extern template void launch_multi_tensor_adagrad<float>(float *const *,
                                                        const float *const *,
                                                        float *const *,
                                                        const int64_t *,
                                                        int,
                                                        int64_t,
                                                        float,
                                                        float,
                                                        float,
                                                        cudaStream_t);
#endif

/**
 * @brief Fused SGD with momentum
 */
template<typename T>
void launch_fused_sgd_momentum(T *param,
                               const T *grad,
                               T *velocity,
                               float lr,
                               float momentum,
                               float weight_decay,
                               int64_t n,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_sgd_momentum<float>(float *,
                                                      const float *,
                                                      float *,
                                                      float,
                                                      float,
                                                      float,
                                                      int64_t,
                                                      cudaStream_t);
extern template void launch_fused_sgd_momentum<__half>(__half *,
                                                       const __half *,
                                                       __half *,
                                                       float,
                                                       float,
                                                       float,
                                                       int64_t,
                                                       cudaStream_t);
extern template void launch_fused_sgd_momentum<__nv_bfloat16>(__nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              float,
                                                              float,
                                                              float,
                                                              int64_t,
                                                              cudaStream_t);
#endif

/**
 * @brief Fused plain SGD (no momentum, no velocity buffer)
 */
template<typename T>
void launch_fused_sgd(T *param,
                      const T *grad,
                      float lr,
                      float weight_decay,
                      int64_t n,
                      cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_sgd<float>(float *,
                                             const float *,
                                             float,
                                             float,
                                             int64_t,
                                             cudaStream_t);
extern template void launch_fused_sgd<__half>(__half *,
                                              const __half *,
                                              float,
                                              float,
                                              int64_t,
                                              cudaStream_t);
extern template void launch_fused_sgd<__nv_bfloat16>(__nv_bfloat16 *,
                                                     const __nv_bfloat16 *,
                                                     float,
                                                     float,
                                                     int64_t,
                                                     cudaStream_t);
#endif

/**
 * @brief Fused RMSprop optimizer update
 *
 * avg_sq = alpha * avg_sq + (1 - alpha) * grad^2
 * param  = param - lr * (grad + weight_decay * param) / (sqrt(avg_sq) + epsilon)
 */
template<typename T>
void launch_fused_rmsprop(T *param,
                          const T *grad,
                          T *avg_sq,
                          float lr,
                          float alpha,
                          float epsilon,
                          float weight_decay,
                          int64_t n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_rmsprop<float>(float *,
                                                 const float *,
                                                 float *,
                                                 float,
                                                 float,
                                                 float,
                                                 float,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_fused_rmsprop<__half>(__half *,
                                                  const __half *,
                                                  __half *,
                                                  float,
                                                  float,
                                                  float,
                                                  float,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_fused_rmsprop<__nv_bfloat16>(__nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         float,
                                                         float,
                                                         float,
                                                         float,
                                                         int64_t,
                                                         cudaStream_t);
#endif

/**
 * @brief Fused AdaGrad optimizer update
 *
 * accum = accum + grad^2
 * param = param - lr * (grad + weight_decay * param) / (sqrt(accum) + epsilon)
 */
template<typename T>
void launch_fused_adagrad(T *param,
                          const T *grad,
                          T *accum,
                          float lr,
                          float epsilon,
                          float weight_decay,
                          int64_t n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_adagrad<float>(float *,
                                                 const float *,
                                                 float *,
                                                 float,
                                                 float,
                                                 float,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_fused_adagrad<__half>(__half *,
                                                  const __half *,
                                                  __half *,
                                                  float,
                                                  float,
                                                  float,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_fused_adagrad<__nv_bfloat16>(__nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         float,
                                                         float,
                                                         float,
                                                         int64_t,
                                                         cudaStream_t);
#endif

