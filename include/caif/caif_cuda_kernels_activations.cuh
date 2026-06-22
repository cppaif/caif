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
// Pointwise + gated activation launcher declarations. Definitions live in
// src/caif_cuda_kernels_activations.cu. CPU-only
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
// Gated activation op enum (C linkage, plain enum)
//------------------------------------------------------------------------------
enum CAIF_GatedActivationOp
{
  CAIF_GATED_OP_SWISH=0,
  CAIF_GATED_OP_GELU=1,
  CAIF_GATED_OP_RELU=2,
  CAIF_GATED_OP_SIGMOID=3,
  CAIF_GATED_OP_LINEAR=4
};

//------------------------------------------------------------------------------
// ReLU kernels (vectorized, replaces cuDNN).
// Templated on element type T ∈ {float, __half, __nv_bfloat16}.
// Explicit instantiations live in caif_cuda_kernels.cu (GPU) and
// caif_cuda_kernels_cpu.cpp (no-op stubs for CPU-only builds).
//------------------------------------------------------------------------------
template<typename T>
void launch_relu_forward(const T *input,
                         T *output,
                         int64_t n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_relu_forward<float>(const float *,
                                                float *,
                                                int64_t,
                                                cudaStream_t);
extern template void launch_relu_forward<__half>(const __half *,
                                                 __half *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_relu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int64_t,
                                                        cudaStream_t);
#endif

template<typename T>
void launch_relu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          int64_t n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_relu_backward<float>(const float *,
                                                 const float *,
                                                 float *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_relu_backward<__half>(const __half *,
                                                  const __half *,
                                                  __half *,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int64_t,
                                                         cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Sigmoid kernels (vectorized, replaces cuDNN)
//------------------------------------------------------------------------------
template<typename T>
void launch_sigmoid_forward(const T *input,
                            T *output,
                            int64_t n,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sigmoid_forward<float>(const float *,
                                                   float *,
                                                   int64_t,
                                                   cudaStream_t);
extern template void launch_sigmoid_forward<__half>(const __half *,
                                                    __half *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_sigmoid_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_sigmoid_backward(const T *grad_output,
                             const T *output,
                             T *grad_input,
                             int64_t n,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sigmoid_backward<float>(const float *,
                                                    const float *,
                                                    float *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_sigmoid_backward<__half>(const __half *,
                                                     const __half *,
                                                     __half *,
                                                     int64_t,
                                                     cudaStream_t);
extern template void launch_sigmoid_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int64_t,
                                                            cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Tanh kernels (vectorized, replaces cuDNN)
//------------------------------------------------------------------------------
template<typename T>
void launch_tanh_forward(const T *input,
                         T *output,
                         int64_t n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_tanh_forward<float>(const float *,
                                                float *,
                                                int64_t,
                                                cudaStream_t);
extern template void launch_tanh_forward<__half>(const __half *,
                                                 __half *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_tanh_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int64_t,
                                                        cudaStream_t);
#endif

template<typename T>
void launch_tanh_backward(const T *grad_output,
                          const T *output,
                          T *grad_input,
                          int64_t n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_tanh_backward<float>(const float *,
                                                 const float *,
                                                 float *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_tanh_backward<__half>(const __half *,
                                                  const __half *,
                                                  __half *,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_tanh_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int64_t,
                                                         cudaStream_t);
#endif

//------------------------------------------------------------------------------
// LeakyReLU kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_leaky_relu_forward(const T *input,
                               T *output,
                               float alpha,
                               int64_t n,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_leaky_relu_forward<float>(const float *,
                                                      float *,
                                                      float,
                                                      int64_t,
                                                      cudaStream_t);
extern template void launch_leaky_relu_forward<__half>(const __half *,
                                                       __half *,
                                                       float,
                                                       int64_t,
                                                       cudaStream_t);
extern template void launch_leaky_relu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              float,
                                                              int64_t,
                                                              cudaStream_t);
#endif

template<typename T>
void launch_leaky_relu_backward(const T *grad_output,
                                const T *input,
                                T *grad_input,
                                float alpha,
                                int64_t n,
                                cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_leaky_relu_backward<float>(const float *,
                                                       const float *,
                                                       float *,
                                                       float,
                                                       int64_t,
                                                       cudaStream_t);
extern template void launch_leaky_relu_backward<__half>(const __half *,
                                                        const __half *,
                                                        __half *,
                                                        float,
                                                        int64_t,
                                                        cudaStream_t);
extern template void launch_leaky_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                               const __nv_bfloat16 *,
                                                               __nv_bfloat16 *,
                                                               float,
                                                               int64_t,
                                                               cudaStream_t);
#endif

//------------------------------------------------------------------------------
// ELU kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_elu_forward(const T *input,
                        T *output,
                        float alpha,
                        int64_t n,
                        cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_elu_forward<float>(const float *,
                                               float *,
                                               float,
                                               int64_t,
                                               cudaStream_t);
extern template void launch_elu_forward<__half>(const __half *,
                                                __half *,
                                                float,
                                                int64_t,
                                                cudaStream_t);
extern template void launch_elu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       __nv_bfloat16 *,
                                                       float,
                                                       int64_t,
                                                       cudaStream_t);
#endif

template<typename T>
void launch_elu_backward(const T *grad_output,
                         const T *input,
                         const T *output,
                         T *grad_input,
                         float alpha,
                         int64_t n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_elu_backward<float>(const float *,
                                                const float *,
                                                const float *,
                                                float *,
                                                float,
                                                int64_t,
                                                cudaStream_t);
extern template void launch_elu_backward<__half>(const __half *,
                                                 const __half *,
                                                 const __half *,
                                                 __half *,
                                                 float,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_elu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        float,
                                                        int64_t,
                                                        cudaStream_t);
#endif

//------------------------------------------------------------------------------
// GELU kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_gelu_forward(const T *input,
                         T *output,
                         int64_t n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gelu_forward<float>(const float *,
                                                float *,
                                                int64_t,
                                                cudaStream_t);
extern template void launch_gelu_forward<__half>(const __half *,
                                                 __half *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_gelu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int64_t,
                                                        cudaStream_t);
#endif

template<typename T>
void launch_gelu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          int64_t n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gelu_backward<float>(const float *,
                                                 const float *,
                                                 float *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_gelu_backward<__half>(const __half *,
                                                  const __half *,
                                                  __half *,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_gelu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int64_t,
                                                         cudaStream_t);
#endif

template<typename T>
void launch_gelu_forward_erf(const T *input,
                             T *output,
                             int64_t n,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gelu_forward_erf<float>(const float *,
                                                    float *,
                                                    int64_t,
                                                    cudaStream_t);
extern template void launch_gelu_forward_erf<__half>(const __half *,
                                                     __half *,
                                                     int64_t,
                                                     cudaStream_t);
extern template void launch_gelu_forward_erf<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int64_t,
                                                            cudaStream_t);
#endif

template<typename T>
void launch_gelu_backward_erf(const T *grad_output,
                              const T *input,
                              T *grad_input,
                              int64_t n,
                              cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gelu_backward_erf<float>(const float *,
                                                     const float *,
                                                     float *,
                                                     int64_t,
                                                     cudaStream_t);
extern template void launch_gelu_backward_erf<__half>(const __half *,
                                                      const __half *,
                                                      __half *,
                                                      int64_t,
                                                      cudaStream_t);
extern template void launch_gelu_backward_erf<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             int64_t,
                                                             cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Swish kernels (fallback for cuDNN < 8.0)
//------------------------------------------------------------------------------
template<typename T>
void launch_swish_forward(const T *input,
                          T *output,
                          int64_t n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_swish_forward<float>(const float *,
                                                 float *,
                                                 int64_t,
                                                 cudaStream_t);
extern template void launch_swish_forward<__half>(const __half *,
                                                  __half *,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_swish_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int64_t,
                                                         cudaStream_t);
#endif

template<typename T>
void launch_swish_backward(const T *grad_output,
                           const T *input,
                           const T *output,
                           T *grad_input,
                           int64_t n,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_swish_backward<float>(const float *,
                                                  const float *,
                                                  const float *,
                                                  float *,
                                                  int64_t,
                                                  cudaStream_t);
extern template void launch_swish_backward<__half>(const __half *,
                                                   const __half *,
                                                   const __half *,
                                                   __half *,
                                                   int64_t,
                                                   cudaStream_t);
extern template void launch_swish_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          const __nv_bfloat16 *,
                                                          const __nv_bfloat16 *,
                                                          __nv_bfloat16 *,
                                                          int64_t,
                                                          cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Gated activation kernels (SwiGLU, GeGLU, ReGLU, GLU, Bilinear)
//------------------------------------------------------------------------------

/**
 * Fused gated activation forward: output[i] = apply_op(gate[i]) * up[i]
 * The op parameter selects which gate activation to apply.
 *
 * Templated on T ∈ {float, __half, __nv_bfloat16}. The activation math
 * runs in fp32 internally; load/store cast through T.
 */
template<typename T>
void launch_gated_activation_forward(const T *gate_input,
                                     const T *up_input,
                                     T *output,
                                     int op,
                                     int64_t n,
                                     cudaStream_t stream);

extern template void launch_gated_activation_forward<float>(const float *,
                                                            const float *,
                                                            float *,
                                                            int,
                                                            int64_t,
                                                            cudaStream_t);
extern template void launch_gated_activation_forward<__half>(const __half *,
                                                             const __half *,
                                                             __half *,
                                                             int,
                                                             int64_t,
                                                             cudaStream_t);
extern template void launch_gated_activation_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    int,
                                                                    int64_t,
                                                                    cudaStream_t);

/**
 * Fused gated activation backward:
 *   grad_gate[i] = grad_output[i] * up[i] * d_activate(gate[i])
 *   grad_up[i]   = grad_output[i] * activate(gate[i])
 */
template<typename T>
void launch_gated_activation_backward(const T *grad_output,
                                      const T *cached_gate_input,
                                      const T *cached_up_input,
                                      T *grad_gate,
                                      T *grad_up,
                                      int op,
                                      int64_t n,
                                      cudaStream_t stream);

extern template void launch_gated_activation_backward<float>(const float *,
                                                             const float *,
                                                             const float *,
                                                             float *,
                                                             float *,
                                                             int,
                                                             int64_t,
                                                             cudaStream_t);
extern template void launch_gated_activation_backward<__half>(const __half *,
                                                              const __half *,
                                                              const __half *,
                                                              __half *,
                                                              __half *,
                                                              int,
                                                              int64_t,
                                                              cudaStream_t);
extern template void launch_gated_activation_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                     const __nv_bfloat16 *,
                                                                     const __nv_bfloat16 *,
                                                                     __nv_bfloat16 *,
                                                                     __nv_bfloat16 *,
                                                                     int,
                                                                     int64_t,
                                                                     cudaStream_t);
