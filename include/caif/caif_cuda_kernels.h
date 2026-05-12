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
// CUDA kernel declarations for operations not supported by cuDNN
//
// IMPORTANT: Every function declared in this header has a GPU implementation
// in caif_cuda_kernels.cu and a CPU no-op stub in caif_cuda_kernels_cpu.cpp.
// When adding a new launch_* function here, you MUST also add a matching
// stub to caif_cuda_kernels_cpu.cpp or CPU-only builds will fail to link.
//
// dtype-dispatch convention (TYPE_DISPATCH_FULL_PLAN Phase 5.6):
//   Every `launch_*` either:
//     (a) is a `template<typename T>` with explicit instantiations for
//         {float, __half, __nv_bfloat16}, OR
//     (b) carries a per-launcher `fp32-only by contract: <reason>` comment
//         immediately above its declaration naming why the body cannot be
//         templated (e.g. dtype-conversion utility, fp32-master-to-quantized
//         pack/unpack, integer-only operating data).
//   The Phase 9 grep gate verifies (a)|(b) holds for every entry.
//------------------------------------------------------------------------------
#ifndef CAIF_CUDA_KERNELS_H
#define CAIF_CUDA_KERNELS_H

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#else
typedef void *cudaStream_t;  // Placeholder for non-CUDA builds
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
// ReLU kernels (vectorized, replaces cuDNN).
// Templated on element type T ∈ {float, __half, __nv_bfloat16}.
// Explicit instantiations live in caif_cuda_kernels.cu (GPU) and
// caif_cuda_kernels_cpu.cpp (no-op stubs for CPU-only builds).
//------------------------------------------------------------------------------
template<typename T>
void launch_relu_forward(const T *input,
                         T *output,
                         int n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_relu_forward<float>(const float *,
                                                float *,
                                                int,
                                                cudaStream_t);
extern template void launch_relu_forward<__half>(const __half *,
                                                 __half *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_relu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int,
                                                        cudaStream_t);
#endif

template<typename T>
void launch_relu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          int n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_relu_backward<float>(const float *,
                                                 const float *,
                                                 float *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_relu_backward<__half>(const __half *,
                                                  const __half *,
                                                  __half *,
                                                  int,
                                                  cudaStream_t);
extern template void launch_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int,
                                                         cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Sigmoid kernels (vectorized, replaces cuDNN)
//------------------------------------------------------------------------------
template<typename T>
void launch_sigmoid_forward(const T *input,
                            T *output,
                            int n,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sigmoid_forward<float>(const float *,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
extern template void launch_sigmoid_forward<__half>(const __half *,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
extern template void launch_sigmoid_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_sigmoid_backward(const T *grad_output,
                             const T *output,
                             T *grad_input,
                             int n,
                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sigmoid_backward<float>(const float *,
                                                    const float *,
                                                    float *,
                                                    int,
                                                    cudaStream_t);
extern template void launch_sigmoid_backward<__half>(const __half *,
                                                     const __half *,
                                                     __half *,
                                                     int,
                                                     cudaStream_t);
extern template void launch_sigmoid_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int,
                                                            cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Tanh kernels (vectorized, replaces cuDNN)
//------------------------------------------------------------------------------
template<typename T>
void launch_tanh_forward(const T *input,
                         T *output,
                         int n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_tanh_forward<float>(const float *,
                                                float *,
                                                int,
                                                cudaStream_t);
extern template void launch_tanh_forward<__half>(const __half *,
                                                 __half *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_tanh_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int,
                                                        cudaStream_t);
#endif

template<typename T>
void launch_tanh_backward(const T *grad_output,
                          const T *output,
                          T *grad_input,
                          int n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_tanh_backward<float>(const float *,
                                                 const float *,
                                                 float *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_tanh_backward<__half>(const __half *,
                                                  const __half *,
                                                  __half *,
                                                  int,
                                                  cudaStream_t);
extern template void launch_tanh_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int,
                                                         cudaStream_t);
#endif

//------------------------------------------------------------------------------
// LeakyReLU kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_leaky_relu_forward(const T *input,
                               T *output,
                               float alpha,
                               int n,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_leaky_relu_forward<float>(const float *,
                                                      float *,
                                                      float,
                                                      int,
                                                      cudaStream_t);
extern template void launch_leaky_relu_forward<__half>(const __half *,
                                                       __half *,
                                                       float,
                                                       int,
                                                       cudaStream_t);
extern template void launch_leaky_relu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              float,
                                                              int,
                                                              cudaStream_t);
#endif

template<typename T>
void launch_leaky_relu_backward(const T *grad_output,
                                const T *input,
                                T *grad_input,
                                float alpha,
                                int n,
                                cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_leaky_relu_backward<float>(const float *,
                                                       const float *,
                                                       float *,
                                                       float,
                                                       int,
                                                       cudaStream_t);
extern template void launch_leaky_relu_backward<__half>(const __half *,
                                                        const __half *,
                                                        __half *,
                                                        float,
                                                        int,
                                                        cudaStream_t);
extern template void launch_leaky_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                               const __nv_bfloat16 *,
                                                               __nv_bfloat16 *,
                                                               float,
                                                               int,
                                                               cudaStream_t);
#endif

//------------------------------------------------------------------------------
// ELU kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_elu_forward(const T *input,
                        T *output,
                        float alpha,
                        int n,
                        cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_elu_forward<float>(const float *,
                                               float *,
                                               float,
                                               int,
                                               cudaStream_t);
extern template void launch_elu_forward<__half>(const __half *,
                                                __half *,
                                                float,
                                                int,
                                                cudaStream_t);
extern template void launch_elu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       __nv_bfloat16 *,
                                                       float,
                                                       int,
                                                       cudaStream_t);
#endif

template<typename T>
void launch_elu_backward(const T *grad_output,
                         const T *input,
                         const T *output,
                         T *grad_input,
                         float alpha,
                         int n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_elu_backward<float>(const float *,
                                                const float *,
                                                const float *,
                                                float *,
                                                float,
                                                int,
                                                cudaStream_t);
extern template void launch_elu_backward<__half>(const __half *,
                                                 const __half *,
                                                 const __half *,
                                                 __half *,
                                                 float,
                                                 int,
                                                 cudaStream_t);
extern template void launch_elu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        float,
                                                        int,
                                                        cudaStream_t);
#endif

//------------------------------------------------------------------------------
// GELU kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_gelu_forward(const T *input,
                         T *output,
                         int n,
                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gelu_forward<float>(const float *,
                                                float *,
                                                int,
                                                cudaStream_t);
extern template void launch_gelu_forward<__half>(const __half *,
                                                 __half *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_gelu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        int,
                                                        cudaStream_t);
#endif

template<typename T>
void launch_gelu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          int n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gelu_backward<float>(const float *,
                                                 const float *,
                                                 float *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_gelu_backward<__half>(const __half *,
                                                  const __half *,
                                                  __half *,
                                                  int,
                                                  cudaStream_t);
extern template void launch_gelu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int,
                                                         cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Swish kernels (fallback for cuDNN < 8.0)
//------------------------------------------------------------------------------
template<typename T>
void launch_swish_forward(const T *input,
                          T *output,
                          int n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_swish_forward<float>(const float *,
                                                 float *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_swish_forward<__half>(const __half *,
                                                  __half *,
                                                  int,
                                                  cudaStream_t);
extern template void launch_swish_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         int,
                                                         cudaStream_t);
#endif

template<typename T>
void launch_swish_backward(const T *grad_output,
                           const T *input,
                           const T *output,
                           T *grad_input,
                           int n,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_swish_backward<float>(const float *,
                                                  const float *,
                                                  const float *,
                                                  float *,
                                                  int,
                                                  cudaStream_t);
extern template void launch_swish_backward<__half>(const __half *,
                                                   const __half *,
                                                   const __half *,
                                                   __half *,
                                                   int,
                                                   cudaStream_t);
extern template void launch_swish_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          const __nv_bfloat16 *,
                                                          const __nv_bfloat16 *,
                                                          __nv_bfloat16 *,
                                                          int,
                                                          cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Element-wise operation kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_elementwise_add(const T *a,
                            const T *b,
                            T *result,
                            int n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_add<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
extern template void launch_elementwise_add<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
extern template void launch_elementwise_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_add_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_add_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int,
                                                          cudaStream_t);
extern template void launch_elementwise_add_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int,
                                                           cudaStream_t);
extern template void launch_elementwise_add_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int,
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
                            int n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_sub<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
extern template void launch_elementwise_sub<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
extern template void launch_elementwise_sub<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_sub_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_sub_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int,
                                                          cudaStream_t);
extern template void launch_elementwise_sub_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int,
                                                           cudaStream_t);
extern template void launch_elementwise_sub_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int,
                                                                  cudaStream_t);
#endif

template<typename T>
void launch_elementwise_mul(const T *a,
                            const T *b,
                            T *result,
                            int n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_mul<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
extern template void launch_elementwise_mul<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
extern template void launch_elementwise_mul<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_mul_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_mul_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int,
                                                          cudaStream_t);
extern template void launch_elementwise_mul_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int,
                                                           cudaStream_t);
extern template void launch_elementwise_mul_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int,
                                                                  cudaStream_t);
#endif

template<typename T>
void launch_elementwise_div(const T *a,
                            const T *b,
                            T *result,
                            int n,
                            cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_div<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
extern template void launch_elementwise_div<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
extern template void launch_elementwise_div<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);
#endif

template<typename T>
void launch_elementwise_div_scalar(const T *a,
                                   float scalar,
                                   T *result,
                                   int n,
                                   cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_div_scalar<float>(const float *,
                                                          float,
                                                          float *,
                                                          int,
                                                          cudaStream_t);
extern template void launch_elementwise_div_scalar<__half>(const __half *,
                                                           float,
                                                           __half *,
                                                           int,
                                                           cudaStream_t);
extern template void launch_elementwise_div_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  float,
                                                                  __nv_bfloat16 *,
                                                                  int,
                                                                  cudaStream_t);
#endif

template<typename T>
void launch_elementwise_sqrt(const T *a,
                             T *result,
                             int n,
                             cudaStream_t stream);
#ifdef USE_CAIF_CUDA
extern template void launch_elementwise_sqrt<float>(const float *,
                                                    float *,
                                                    int,
                                                    cudaStream_t);
extern template void launch_elementwise_sqrt<__half>(const __half *,
                                                     __half *,
                                                     int,
                                                     cudaStream_t);
extern template void launch_elementwise_sqrt<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int,
                                                            cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Reduction kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_reduction_sum(const T *input,
                          float *output,
                          int n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_reduction_sum<float>(const float *,
                                                 float *,
                                                 int,
                                                 cudaStream_t);
extern template void launch_reduction_sum<__half>(const __half *,
                                                  float *,
                                                  int,
                                                  cudaStream_t);
extern template void launch_reduction_sum<__nv_bfloat16>(const __nv_bfloat16 *,
                                                         float *,
                                                         int,
                                                         cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Loss function kernels
//------------------------------------------------------------------------------
template<typename T>
void launch_cross_entropy_loss(const T *predictions,
                               const T *targets,
                               float *loss,
                               float epsilon,
                               int batch_size,
                               int num_classes,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cross_entropy_loss<float>(const float *,
                                                      const float *,
                                                      float *,
                                                      float,
                                                      int,
                                                      int,
                                                      cudaStream_t);
extern template void launch_cross_entropy_loss<__half>(const __half *,
                                                       const __half *,
                                                       float *,
                                                       float,
                                                       int,
                                                       int,
                                                       cudaStream_t);
extern template void launch_cross_entropy_loss<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              float *,
                                                              float,
                                                              int,
                                                              int,
                                                              cudaStream_t);
#endif

template<typename T>
void launch_cross_entropy_gradient(const T *predictions,
                                   const T *targets,
                                   T *gradient,
                                   float epsilon,
                                   int batch_size,
                                   int n,
                                   cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cross_entropy_gradient<float>(const float *,
                                                          const float *,
                                                          float *,
                                                          float,
                                                          int,
                                                          int,
                                                          cudaStream_t);
extern template void launch_cross_entropy_gradient<__half>(const __half *,
                                                           const __half *,
                                                           __half *,
                                                           float,
                                                           int,
                                                           int,
                                                           cudaStream_t);
extern template void launch_cross_entropy_gradient<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  const __nv_bfloat16 *,
                                                                  __nv_bfloat16 *,
                                                                  float,
                                                                  int,
                                                                  int,
                                                                  cudaStream_t);
#endif

// Cross-entropy loss/grad with index targets
template<typename T>
void launch_cross_entropy_loss_index(const T *predictions,
                                     const int *target_indices,
                                     float *loss,
                                     float epsilon,
                                     int batch_size,
                                     int num_classes,
                                     cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cross_entropy_loss_index<float>(const float *,
                                                            const int *,
                                                            float *,
                                                            float,
                                                            int,
                                                            int,
                                                            cudaStream_t);
extern template void launch_cross_entropy_loss_index<__half>(const __half *,
                                                             const int *,
                                                             float *,
                                                             float,
                                                             int,
                                                             int,
                                                             cudaStream_t);
extern template void launch_cross_entropy_loss_index<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    const int *,
                                                                    float *,
                                                                    float,
                                                                    int,
                                                                    int,
                                                                    cudaStream_t);
#endif

template<typename T>
void launch_cross_entropy_gradient_index(const T *predictions,
                                         const int *target_indices,
                                         T *gradient,
                                         float epsilon,
                                         int batch_size,
                                         int num_classes,
                                         cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_cross_entropy_gradient_index<float>(const float *,
                                                                const int *,
                                                                float *,
                                                                float,
                                                                int,
                                                                int,
                                                                cudaStream_t);
extern template void launch_cross_entropy_gradient_index<__half>(const __half *,
                                                                 const int *,
                                                                 __half *,
                                                                 float,
                                                                 int,
                                                                 int,
                                                                 cudaStream_t);
extern template void launch_cross_entropy_gradient_index<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                        const int *,
                                                                        __nv_bfloat16 *,
                                                                        float,
                                                                        int,
                                                                        int,
                                                                        cudaStream_t);
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
                       int n,
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
                                              int,
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
                                               int,
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
                                                      int,
                                                      cudaStream_t);
#endif

/**
 * @brief Fused Adam with gradient clipping
 */
template<typename T>
void launch_fused_adam_clipped(T *param,
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
                               float grad_scale,
                               int n,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_adam_clipped<float>(float *,
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
                                                      float,
                                                      int,
                                                      cudaStream_t);
extern template void launch_fused_adam_clipped<__half>(__half *,
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
                                                       float,
                                                       int,
                                                       cudaStream_t);
extern template void launch_fused_adam_clipped<__nv_bfloat16>(__nv_bfloat16 *,
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
                                                              float,
                                                              int,
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
                               int n,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_sgd_momentum<float>(float *,
                                                      const float *,
                                                      float *,
                                                      float,
                                                      float,
                                                      float,
                                                      int,
                                                      cudaStream_t);
extern template void launch_fused_sgd_momentum<__half>(__half *,
                                                       const __half *,
                                                       __half *,
                                                       float,
                                                       float,
                                                       float,
                                                       int,
                                                       cudaStream_t);
extern template void launch_fused_sgd_momentum<__nv_bfloat16>(__nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              float,
                                                              float,
                                                              float,
                                                              int,
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
                      int n,
                      cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_sgd<float>(float *,
                                             const float *,
                                             float,
                                             float,
                                             int,
                                             cudaStream_t);
extern template void launch_fused_sgd<__half>(__half *,
                                              const __half *,
                                              float,
                                              float,
                                              int,
                                              cudaStream_t);
extern template void launch_fused_sgd<__nv_bfloat16>(__nv_bfloat16 *,
                                                     const __nv_bfloat16 *,
                                                     float,
                                                     float,
                                                     int,
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
                          int n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_rmsprop<float>(float *,
                                                 const float *,
                                                 float *,
                                                 float,
                                                 float,
                                                 float,
                                                 float,
                                                 int,
                                                 cudaStream_t);
extern template void launch_fused_rmsprop<__half>(__half *,
                                                  const __half *,
                                                  __half *,
                                                  float,
                                                  float,
                                                  float,
                                                  float,
                                                  int,
                                                  cudaStream_t);
extern template void launch_fused_rmsprop<__nv_bfloat16>(__nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         float,
                                                         float,
                                                         float,
                                                         float,
                                                         int,
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
                          int n,
                          cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_fused_adagrad<float>(float *,
                                                 const float *,
                                                 float *,
                                                 float,
                                                 float,
                                                 float,
                                                 int,
                                                 cudaStream_t);
extern template void launch_fused_adagrad<__half>(__half *,
                                                  const __half *,
                                                  __half *,
                                                  float,
                                                  float,
                                                  float,
                                                  int,
                                                  cudaStream_t);
extern template void launch_fused_adagrad<__nv_bfloat16>(__nv_bfloat16 *,
                                                         const __nv_bfloat16 *,
                                                         __nv_bfloat16 *,
                                                         float,
                                                         float,
                                                         float,
                                                         int,
                                                         cudaStream_t);
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
 * Prefix-LM mask fill with KV-cache offset.
 * Query row r corresponds to absolute position (offset + r).
 * Allowed iff col <= (offset + r) OR col < prefix_lengths[batch].
 */
template<typename T>
void launch_prefix_mask_fill_offset(T *scores,
                                    const uint32_t *prefix_lengths,
                                    int batch_size,
                                    int num_heads,
                                    int query_len,
                                    int key_len,
                                    int offset,
                                    cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_prefix_mask_fill_offset<float>(float *,
                                                           const uint32_t *,
                                                           int,int,int,int,
                                                           int,cudaStream_t);
extern template void launch_prefix_mask_fill_offset<__half>(__half *,
                                                            const uint32_t *,
                                                            int,int,int,int,
                                                            int,cudaStream_t);
extern template void launch_prefix_mask_fill_offset<__nv_bfloat16>(
    __nv_bfloat16 *,const uint32_t *,int,int,int,int,int,cudaStream_t);
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
                                     int n,
                                     cudaStream_t stream);

extern template void launch_gated_activation_forward<float>(const float *,
                                                            const float *,
                                                            float *,
                                                            int,
                                                            int,
                                                            cudaStream_t);
extern template void launch_gated_activation_forward<__half>(const __half *,
                                                             const __half *,
                                                             __half *,
                                                             int,
                                                             int,
                                                             cudaStream_t);
extern template void launch_gated_activation_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    int,
                                                                    int,
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
                                      int n,
                                      cudaStream_t stream);

extern template void launch_gated_activation_backward<float>(const float *,
                                                             const float *,
                                                             const float *,
                                                             float *,
                                                             float *,
                                                             int,
                                                             int,
                                                             cudaStream_t);
extern template void launch_gated_activation_backward<__half>(const __half *,
                                                              const __half *,
                                                              const __half *,
                                                              __half *,
                                                              __half *,
                                                              int,
                                                              int,
                                                              cudaStream_t);
extern template void launch_gated_activation_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                     const __nv_bfloat16 *,
                                                                     const __nv_bfloat16 *,
                                                                     __nv_bfloat16 *,
                                                                     __nv_bfloat16 *,
                                                                     int,
                                                                     int,
                                                                     cudaStream_t);

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
                           int n,
                           cudaStream_t stream);

/**
 * Scatter-add gradients back to embedding table.
 * grad_table must be pre-zeroed. Uses atomicAdd.
 *
 * Templated on T. fp16/bf16 use the native atomicAdd overloads on
 * sm_70+ (compute capability >= 7.0); accumulation precision matches
 * the storage dtype (the standard fp16/bf16 grad-accumulation choice).
 */
template<typename T>
void launch_embedding_backward(const T *grad_output,
                               const unsigned int *token_ids,
                               T *grad_table,
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
                                                       __half *,
                                                       int,
                                                       int,
                                                       cudaStream_t);
extern template void launch_embedding_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const unsigned int *,
                                                              __nv_bfloat16 *,
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

//------------------------------------------------------------------------------
// Reduction and normalization kernels
//------------------------------------------------------------------------------

/**
 * Sum along axis 0 (over batch): output[d] = sum_b input[b,d]
 */
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
                           int n,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_sum_of_squares<float>(const float *,
                                                  float *,
                                                  int,
                                                  cudaStream_t);
extern template void launch_sum_of_squares<__half>(const __half *,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
extern template void launch_sum_of_squares<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          float *,
                                                          int,
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

/**
 * Normalize each row to sum to 1: output[b,d] = input[b,d] / sum_d(input[b,:])
 */
template<typename T>
void launch_normalize_rows(const T *input,
                           T *output,
                           int batch,
                           int dim,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_normalize_rows<float>(const float *,
                                                  float *,
                                                  int,
                                                  int,
                                                  cudaStream_t);
extern template void launch_normalize_rows<__half>(const __half *,
                                                   __half *,
                                                   int,
                                                   int,
                                                   cudaStream_t);
extern template void launch_normalize_rows<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          __nv_bfloat16 *,
                                                          int,
                                                          int,
                                                          cudaStream_t);
#endif

/**
 * NormalizeRows backward Jacobian on top-k rows, gather variant.
 * Gathers p[k] = probs[t, indices[t,k]] from the full softmax cache,
 * computes s, w, dot, and grad_p_topk inline — no forward-side caches
 * beyond the existing probs/indices are needed.
 *   grad_p_topk[t,k] = (grad_w[t,k] - dot(t)) / s(t),
 *   s(t) = sum_k p[k],  w[k] = p[k]/s,  dot = sum_k w[k]*grad_w[t,k]
 * One warp per token; top_k must be <= 32.
 */
template<typename T>
void launch_normalize_rows_backward_topk_gather(const T *grad_w,
                                                const T *probs,
                                                const int *indices,
                                                T *grad_p_topk,
                                                int num_tokens,
                                                int num_experts,
                                                int top_k,
                                                cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_normalize_rows_backward_topk_gather<float>(const float *,
                                                                       const float *,
                                                                       const int *,
                                                                       float *,
                                                                       int,
                                                                       int,
                                                                       int,
                                                                       cudaStream_t);
extern template void launch_normalize_rows_backward_topk_gather<__half>(const __half *,
                                                                        const __half *,
                                                                        const int *,
                                                                        __half *,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        cudaStream_t);
extern template void launch_normalize_rows_backward_topk_gather<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                               const __nv_bfloat16 *,
                                                                               const int *,
                                                                               __nv_bfloat16 *,
                                                                               int,
                                                                               int,
                                                                               int,
                                                                               cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Top-K and scatter operations
//------------------------------------------------------------------------------

/**
 * Select top-k values and indices per row.
 * input: [batch, dim], indices: [batch, k] (int32), values: [batch, k]
 */
template<typename T>
void launch_topk(const T *input,
                 int *indices,
                 T *values,
                 int batch,
                 int dim,
                 int k,
                 cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_topk<float>(const float *,
                                        int *,
                                        float *,
                                        int,
                                        int,
                                        int,
                                        cudaStream_t);
extern template void launch_topk<__half>(const __half *,
                                         int *,
                                         __half *,
                                         int,
                                         int,
                                         int,
                                         cudaStream_t);
extern template void launch_topk<__nv_bfloat16>(const __nv_bfloat16 *,
                                                int *,
                                                __nv_bfloat16 *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);
#endif

/**
 * Scatter-add values to output using indices.
 * output[b, indices[b,k]] += values[b,k]
 * values: [batch, k], indices: [batch, k], output: [batch, dim]
 */
template<typename T>
void launch_scatter_add(const T *values,
                        const int *indices,
                        T *output,
                        int batch,
                        int k,
                        int dim,
                        cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_scatter_add<float>(const float *,
                                               const int *,
                                               float *,
                                               int,
                                               int,
                                               int,
                                               cudaStream_t);
extern template void launch_scatter_add<__half>(const __half *,
                                                const int *,
                                                __half *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);
extern template void launch_scatter_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const int *,
                                                       __nv_bfloat16 *,
                                                       int,
                                                       int,
                                                       int,
                                                       cudaStream_t);
#endif

/**
 * Gather per-row scores at top-k indices.  Used by SigmoidNoauxTc
 * routing (HF noaux_tc) Phase 1b to read the ORIGINAL (uncorrected)
 * sigmoid scores at the chosen top-k positions while selection
 * happens on bias-corrected scores.
 *
 *   out[t, k] = scores[t, indices[t, k]]
 *
 * scores: [num_tokens, num_experts], indices: [num_tokens, top_k] (int32),
 * out:    [num_tokens, top_k].  top_k must be <= 32.
 */
template<typename T>
void launch_gather_topk_values(const T *scores,
                               const int *indices,
                               T *out,
                               int num_tokens,
                               int num_experts,
                               int top_k,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gather_topk_values<float>(const float *,
                                                      const int *,
                                                      float *,
                                                      int,
                                                      int,
                                                      int,
                                                      cudaStream_t);
extern template void launch_gather_topk_values<__half>(const __half *,
                                                       const int *,
                                                       __half *,
                                                       int,
                                                       int,
                                                       int,
                                                       cudaStream_t);
extern template void launch_gather_topk_values<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const int *,
                                                              __nv_bfloat16 *,
                                                              int,
                                                              int,
                                                              int,
                                                              cudaStream_t);
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
                                    int causal,
                                    int num_heads,
                                    int num_kv_heads,
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
                                                           int,
                                                           int,
                                                           int,
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
                                                            int,
                                                            int,
                                                            int,
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
                                                                   int,
                                                                   int,
                                                                   int,
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
                                     int causal,
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
                                                            int,
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
                                                             int,
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
                                                                    int,
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
                                                                            cudaStream_t);
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

//------------------------------------------------------------------------------
// MoE (Mixture of Experts) kernels
//------------------------------------------------------------------------------

/**
 * Dispatch tokens to expert buffers based on routing indices.
 * input: [num_tokens, dim] - token embeddings
 * expert_indices: [num_tokens, top_k] - which experts each token routes to (as floats)
 * dispatch_map: [num_tokens, top_k] - position within expert buffer for each assignment
 * expert_buffer: [total_assigned_tokens, dim] - contiguous buffer for all experts
 * expert_offsets: [num_experts+1] - cumulative token counts per expert
 */
template<typename T>
void launch_moe_dispatch(const T *input,
                         const int *expert_indices,
                         const int *dispatch_map,
                         T *expert_buffer,
                         const int *expert_offsets,
                         int num_tokens,
                         int dim,
                         int top_k,
                         cudaStream_t stream);

/**
 * Combine expert outputs back to token positions with routing weights.
 * expert_buffer: [total_assigned_tokens, dim] - contiguous buffer of expert outputs
 * expert_indices: [num_tokens, top_k] - which experts each token routed to (as floats)
 * expert_weights: [num_tokens, top_k] - routing weights (normalized)
 * dispatch_map: [num_tokens, top_k] - position within expert buffer for each assignment
 * expert_offsets: [num_experts+1] - cumulative token counts per expert
 * output: [num_tokens, dim] - combined output
 */
template<typename T>
void launch_moe_combine(const T *expert_buffer,
                        const int *expert_indices,
                        const T *expert_weights,
                        const int *dispatch_map,
                        const int *expert_offsets,
                        T *output,
                        int num_tokens,
                        int dim,
                        int top_k,
                        cudaStream_t stream);

/**
 * Backward of MoECombine — grad_expert_buffer path.
 * grad_expert_buffer[(off[e]+pos)*dim+d] = expert_weights[t,k] * grad_output[t,d]
 */
template<typename T>
void launch_moe_combine_backward_grad_expert(const T *grad_output,
                                              const int *expert_indices,
                                              const T *expert_weights,
                                              const int *dispatch_map,
                                              const int *expert_offsets,
                                              T *grad_expert_buffer,
                                              int num_tokens,
                                              int dim,
                                              int top_k,
                                              cudaStream_t stream);

/**
 * Backward of MoECombine — grad_weights path.
 * grad_weights[t,k] = sum_d grad_output[t,d] * expert_buffer[(off[e]+pos)*dim+d]
 */
template<typename T>
void launch_moe_combine_backward_grad_weights(const T *grad_output,
                                               const T *expert_buffer,
                                               const int *expert_indices,
                                               const int *dispatch_map,
                                               const int *expert_offsets,
                                               T *grad_weights,
                                               int num_tokens,
                                               int dim,
                                               int top_k,
                                               cudaStream_t stream);

/**
 * Backward of MoEDispatch — gathers per-assignment gradients back to each token.
 * grad_input[t,d] = sum_k grad_expert_buffer[(off[e(t,k)]+pos(t,k))*dim+d]
 */
template<typename T>
void launch_moe_dispatch_backward(const T *grad_expert_buffer,
                                   const int *expert_indices,
                                   const int *dispatch_map,
                                   const int *expert_offsets,
                                   T *grad_input,
                                   int num_tokens,
                                   int dim,
                                   int top_k,
                                   cudaStream_t stream);

/**
 * Fused softmax + top-k selection for MoE router.
 * router_logits: [num_tokens, num_experts] - raw router output
 * expert_indices: [num_tokens, top_k] - selected expert indices (as floats)
 * expert_weights: [num_tokens, top_k] - normalized routing weights
 * router_probs: [num_tokens, num_experts] - full softmax probabilities (for aux loss)
 */
// router_logits dtype is the layer's StorageT; internal accumulation
// runs in fp32 for numerical stability. Outputs (expert_weights /
// router_probs) are fp32 by router-output contract.
template<typename T>
void launch_moe_topk_gating(const T *router_logits,
                            int *expert_indices,
                            float *expert_weights,
                            float *router_probs,
                            int num_tokens,
                            int num_experts,
                            int top_k,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_moe_topk_gating<float>(const float *,
                                                   int *,
                                                   float *,
                                                   float *,
                                                   int,
                                                   int,
                                                   int,
                                                   cudaStream_t);
extern template void launch_moe_topk_gating<__half>(const __half *,
                                                    int *,
                                                    float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);
extern template void launch_moe_topk_gating<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           int *,
                                                           float *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           cudaStream_t);
#endif

/**
 * Count tokens per expert using atomic operations.
 * expert_indices: [num_tokens, top_k] - which experts each token routes to (Int32)
 * expert_counts: [num_experts] - output counts per expert (must be zeroed first)
 *
 * fp32-only by contract: integer-only kernel (Int32 indices in, Int32
 * counts out). No floating-point data — templating on T is meaningless.
 */
void launch_moe_count_per_expert(const int *expert_indices,
                                 int *expert_counts,
                                 int num_tokens,
                                 int num_experts,
                                 int top_k,
                                 int capacity_per_expert,
                                 cudaStream_t stream);

/**
 * ST-MoE router z-loss gradient contribution.
 * grad_logits[t,e] += logsumexp_scaled[t] * probs[t,e]
 * logsumexp_scaled is pre-multiplied by (2 * z_loss_weight / N) so the kernel
 * only fuses a row-broadcast multiply-add across [N, E].
 */
template<typename T>
void launch_moe_z_loss_grad(const T *logsumexp_scaled,
                            const T *probs,
                            T *grad_logits,
                            int num_tokens,
                            int num_experts,
                            cudaStream_t stream);

//------------------------------------------------------------------------------
// Data type conversion kernels
//------------------------------------------------------------------------------

/**
 * Convert FP32 array to FP16 (__half) on device.
 * input: [n] float, output: [n] half (2 bytes each)
 */
void launch_convert_fp32_to_fp16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream);

/**
 * Convert FP16 (__half) array to FP32 on device.
 * input: [n] half, output: [n] float
 */
void launch_convert_fp16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream);

/**
 * Convert FP32 array to BF16 (__nv_bfloat16) on device.
 * input: [n] float, output: [n] bfloat16 (2 bytes each)
 *
 * fp32-only by contract: dtype-conversion utility — fp32 IN, bf16 OUT
 * is the operation's name. Templating on T is meaningless.
 */
void launch_convert_fp32_to_bf16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream);

/**
 * Convert BF16 (__nv_bfloat16) array to FP32 on device.
 * input: [n] bfloat16, output: [n] float
 *
 * fp32-only by contract: dtype-conversion utility — bf16 IN, fp32 OUT
 * is the operation's name.
 */
void launch_convert_bf16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream);

//------------------------------------------------------------------------------
// INT8 conversion kernels
//------------------------------------------------------------------------------

/**
 * Convert FP32 to INT8 on device. Clamps to [-127,127] and casts.
 * input: [n] float, output: [n] int8_t
 *
 * fp32-only by contract: dtype-conversion utility — fp32 IN, int8 OUT
 * is the operation's name.
 */
void launch_convert_fp32_to_int8(const float *input,
                                  void *output,
                                  int n,
                                  cudaStream_t stream);

/**
 * Convert INT8 to FP32 on device. Simple cast.
 * input: [n] int8_t, output: [n] float
 *
 * fp32-only by contract: dtype-conversion utility — int8 IN, fp32 OUT
 * is the operation's name.
 */
void launch_convert_int8_to_fp32(const void *input,
                                  float *output,
                                  int n,
                                  cudaStream_t stream);

//------------------------------------------------------------------------------
// INT8 scaled quantization kernels (symmetric, per-tensor and per-channel)
//------------------------------------------------------------------------------

/**
 * Symmetric per-tensor INT8 quantization.
 *
 * Computes scale = max(|input|)/127, then q = round(input/scale) clamped to
 * [-127, 127]. Writes quantized bytes into `output` and the scale value into
 * `scale[0]`. Caller allocates `output` as int8_t[n] and `scale` as float[1].
 *
 * If the input is identically zero the scale is set to 1.0f so dequantization
 * returns zero.
 *
 * fp32-only by contract: quantization utility — fp32 master IN, int8
 * packed OUT. The fp32 input is the canonical representation; int8 is
 * the quantized output. Not templated.
 */
void launch_quantize_int8_per_tensor(const float *input,
                                      void *output,
                                      void *scale,
                                      int n,
                                      cudaStream_t stream);

/**
 * Dequantize per-tensor INT8 back to fp32.
 *
 * fp32-only by contract: dequantization utility — int8 packed IN, fp32
 * master OUT.
 */
void launch_dequantize_int8_per_tensor(const void *input,
                                        float *output,
                                        const void *scale,
                                        int n,
                                        cudaStream_t stream);

/**
 * Symmetric per-channel INT8 quantization on a 2D tensor [rows, cols] with
 * scales computed along `cols` (the output-channel axis for weight tensors).
 *
 * scales[c] = max_r |input[r, c]| / 127
 * q[r, c]   = round(input[r, c] / scales[c]) clamped to [-127, 127]
 *
 * `output` is int8_t[rows*cols], `scales` is float[cols].
 *
 * fp32-only by contract: quantization utility — fp32 master IN, int8
 * packed OUT (per-channel scales).
 */
void launch_quantize_int8_per_channel(const float *input,
                                       void *output,
                                       void *scales,
                                       int rows,
                                       int cols,
                                       cudaStream_t stream);

/**
 * Dequantize per-channel INT8 back to fp32.
 *
 * fp32-only by contract: dequantization utility — int8 packed IN, fp32
 * master OUT.
 */
void launch_dequantize_int8_per_channel(const void *input,
                                         float *output,
                                         const void *scales,
                                         int rows,
                                         int cols,
                                         cudaStream_t stream);

//------------------------------------------------------------------------------
// INT4 quantization kernels (symmetric per-group with FP16 scales)
//------------------------------------------------------------------------------

/**
 * Dequantize INT4 packed data to FP32.
 * Packed format: 2 elements per byte (low nibble=even, high nibble=odd).
 * Symmetric per-group: val = int4_val * scale[group_idx]
 *
 * packed_data: [(num_elements+1)/2] bytes of packed INT4
 * scales: [num_groups] FP16 per-group scales
 * output: [num_elements] float
 * num_elements: total number of logical elements
 * group_size: elements per quantization group
 *
 * fp32-only by contract: dequantization utility — int4 packed IN with
 * fp16 scales, fp32 master OUT.
 */
void launch_dequantize_int4(const void *packed_data,
                             const void *scales,
                             float *output,
                             int num_elements,
                             int group_size,
                             cudaStream_t stream);

/**
 * Quantize FP32 to INT4 packed format with per-group FP16 scales.
 * Computes scales from data, packs 2 elements per byte.
 *
 * input: [num_elements] float
 * packed_output: [(num_elements+1)/2] bytes
 * scales_output: [num_groups] FP16
 * num_elements: total number of logical elements
 * group_size: elements per quantization group
 *
 * fp32-only by contract: quantization utility — fp32 master IN, int4
 * packed OUT (with per-group fp16 scales).
 */
void launch_quantize_to_int4(const float *input,
                              void *packed_output,
                              void *scales_output,
                              int num_elements,
                              int group_size,
                              cudaStream_t stream);

//------------------------------------------------------------------------------
// Tensor slice and concatenation kernels
//------------------------------------------------------------------------------

/**
 * Slice along the last dimension of a 2D (logically) tensor.
 * input: [rows, in_cols], output: [rows, out_cols]
 * Copies columns [col_start, col_start + out_cols) from each row.
 */
template<typename T>
void launch_slice_last_dim(const T *input,
                           T *output,
                           int rows,
                           int in_cols,
                           int col_start,
                           int out_cols,
                           cudaStream_t stream);

/**
 * Scatter a slice back into the last dimension (backward of slice).
 * grad_input: [rows, in_cols] (must be pre-zeroed or pre-filled)
 * grad_output: [rows, out_cols]
 * Adds grad_output into grad_input[:, col_start:col_start+out_cols].
 */
template<typename T>
void launch_slice_last_dim_backward(const T *grad_output,
                                    T *grad_input,
                                    int rows,
                                    int in_cols,
                                    int col_start,
                                    int out_cols,
                                    cudaStream_t stream);

/**
 * Concatenate two tensors along the last dimension.
 * a: [rows, cols_a], b: [rows, cols_b], output: [rows, cols_a + cols_b]
 */
template<typename T>
void launch_concat_last_dim(const T *a,
                            const T *b,
                            T *output,
                            int rows,
                            int cols_a,
                            int cols_b,
                            cudaStream_t stream);

/**
 * Compute T5-style relative position bias matrix.
 * embedding: [num_heads, num_buckets] — always fp32 (autocast convention)
 * output: [num_heads, q_len, k_len] — dtype T (matches attention dtype)
 * Uses logarithmic bucketing for distances > num_buckets/2.
 */
template<typename T>
void launch_relative_position_bias_forward(const float *embedding,
                                           T *output,
                                           int num_heads,
                                           int q_len,
                                           int k_len,
                                           int num_buckets,
                                           int max_distance,
                                           int bidirectional,
                                           cudaStream_t stream);

/**
 * Backward for relative position bias — accumulate embedding gradient.
 * grad_output: [num_heads, q_len, k_len] — dtype T
 * grad_embedding: [num_heads, num_buckets] — fp32 (atomicAdd stays on float,
 * sm_75 safe; grads up-cast from T before accumulation).
 */
template<typename T>
void launch_relative_position_bias_backward(const T *grad_output,
                                            float *grad_embedding,
                                            int num_heads,
                                            int q_len,
                                            int k_len,
                                            int num_buckets,
                                            int max_distance,
                                            int bidirectional,
                                            cudaStream_t stream);

#endif  // CAIF_CUDA_KERNELS_H

