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
// CAIF - AI Framework
// Pointwise activation CUDA kernels (ReLU / Sigmoid / Tanh / GELU tanh+erf /
// Swish / LeakyReLU / ELU, forward + backward) and the gated-activation
// kernels (SwiGLU / GeGLU / ReGLU / GLU / Bilinear merge). Carved verbatim
// out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_activations.cuh
//------------------------------------------------------------------------------
// Disable GNU C++ extensions to avoid rsqrt conflict between CUDA and glibc
// This must be set BEFORE any includes
#undef _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "caif_cuda_kernels_common.cuh"
#include "caif_cuda_kernels_activations.cuh"

//------------------------------------------------------------------------------
// ReLU Forward: f(x) = max(0, x)
// 128-bit int4 load → unpack T[lanes] → op → pack → 128-bit store.
//------------------------------------------------------------------------------
template<typename T>
__global__ void relu_forward_kernel(const T *input,
                                    T *output,
                                    const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      if(lane[i]>T(0))
      {
        lane[i]=lane[i];
      }
      else
      {
        lane[i]=T(0);
      }
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void relu_forward_tail_kernel(const T *input,
                                         T *output,
                                         const int64_t offset,
                                         const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    if(input[idx]>T(0))
    {
      output[idx]=input[idx];
    }
    else
    {
      output[idx]=T(0);
    }
  }
}

//------------------------------------------------------------------------------
// ReLU Backward: grad = upstream if input > 0, else 0
//------------------------------------------------------------------------------
template<typename T>
__global__ void relu_backward_kernel(const T *grad_output,
                                     const T *input,
                                     T *grad_input,
                                     const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 x=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *x_lane=reinterpret_cast<const T*>(&x);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      if(x_lane[i]>T(0))
      {
        r_lane[i]=g_lane[i];
      }
      else
      {
        r_lane[i]=T(0);
      }
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void relu_backward_tail_kernel(const T *grad_output,
                                          const T *input,
                                          T *grad_input,
                                          const int64_t offset,
                                          const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    if(input[idx]>T(0))
    {
      grad_input[idx]=grad_output[idx];
    }
    else
    {
      grad_input[idx]=T(0);
    }
  }
}

//------------------------------------------------------------------------------
// Sigmoid Forward: f(x) = 1 / (1 + exp(-x))
// Transcendentals compute in fp32 regardless of storage dtype.
//------------------------------------------------------------------------------
template<typename T>
__global__ void sigmoid_forward_kernel(const T *input,
                                       T *output,
                                       const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      lane[i]=T(1.0f/(1.0f+expf(-x)));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void sigmoid_forward_tail_kernel(const T *input,
                                            T *output,
                                            const int64_t offset,
                                            const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    output[idx]=T(1.0f/(1.0f+expf(-x)));
  }
}

//------------------------------------------------------------------------------
// Sigmoid Backward: grad = upstream * output * (1 - output)
//------------------------------------------------------------------------------
template<typename T>
__global__ void sigmoid_backward_kernel(const T *grad_output,
                                        const T *output,
                                        T *grad_input,
                                        const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 o=reinterpret_cast<const int4 *>(output)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *o_lane=reinterpret_cast<const T*>(&o);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float ov=float(o_lane[i]);
      r_lane[i]=T(gv*ov*(1.0f-ov));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void sigmoid_backward_tail_kernel(const T *grad_output,
                                             const T *output,
                                             T *grad_input,
                                             const int64_t offset,
                                             const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float ov=float(output[idx]);
    grad_input[idx]=T(gv*ov*(1.0f-ov));
  }
}

//------------------------------------------------------------------------------
// Tanh Forward: f(x) = tanh(x)
//------------------------------------------------------------------------------
template<typename T>
__global__ void tanh_forward_kernel(const T *input,
                                    T *output,
                                    const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      lane[i]=T(tanhf(float(lane[i])));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void tanh_forward_tail_kernel(const T *input,
                                         T *output,
                                         const int64_t offset,
                                         const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=T(tanhf(float(input[idx])));
  }
}

//------------------------------------------------------------------------------
// Tanh Backward: grad = upstream * (1 - output^2)
//------------------------------------------------------------------------------
template<typename T>
__global__ void tanh_backward_kernel(const T *grad_output,
                                     const T *output,
                                     T *grad_input,
                                     const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 o=reinterpret_cast<const int4 *>(output)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *o_lane=reinterpret_cast<const T*>(&o);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float ov=float(o_lane[i]);
      r_lane[i]=T(gv*(1.0f-ov*ov));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void tanh_backward_tail_kernel(const T *grad_output,
                                          const T *output,
                                          T *grad_input,
                                          const int64_t offset,
                                          const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float ov=float(output[idx]);
    grad_input[idx]=T(gv*(1.0f-ov*ov));
  }
}

//------------------------------------------------------------------------------
// GELU Forward: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//------------------------------------------------------------------------------

template<typename T>
__global__ void gelu_forward_kernel(const T *input,
                                    T *output,
                                    const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      const float inner=g_cu_gelu_sqrt_2_over_pi*(x+g_cu_gelu_coeff*x*x*x);
      lane[i]=T(g_cu_gelu_half*x*(1.0f+tanhf(inner)));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void gelu_forward_tail_kernel(const T *input,
                                         T *output,
                                         const int64_t offset,
                                         const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    const float inner=g_cu_gelu_sqrt_2_over_pi*(x+g_cu_gelu_coeff*x*x*x);
    output[idx]=T(g_cu_gelu_half*x*(1.0f+tanhf(inner)));
  }
}

//------------------------------------------------------------------------------
// GELU Forward (exact erf): f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//------------------------------------------------------------------------------
template<typename T>
__global__ void gelu_forward_erf_kernel(const T *input,
                                        T *output,
                                        const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      lane[i]=T(g_cu_gelu_half*x*(1.0f+erff(x*g_cu_gelu_inv_sqrt2)));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void gelu_forward_erf_tail_kernel(const T *input,
                                             T *output,
                                             const int64_t offset,
                                             const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    output[idx]=T(g_cu_gelu_half*x*(1.0f+erff(x*g_cu_gelu_inv_sqrt2)));
  }
}

//------------------------------------------------------------------------------
// GELU Backward
//------------------------------------------------------------------------------
template<typename T>
__global__ void gelu_backward_kernel(const T *grad_output,
                                     const T *input,
                                     T *grad_input,
                                     const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *v_lane=reinterpret_cast<const T*>(&v);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float x=float(v_lane[i]);
      const float inner=g_cu_gelu_sqrt_2_over_pi*(x+g_cu_gelu_coeff*x*x*x);
      const float th=tanhf(inner);
      const float di=g_cu_gelu_sqrt_2_over_pi*(1.0f+g_cu_gelu_cubic_factor*g_cu_gelu_coeff*x*x);
      r_lane[i]=T(gv*(g_cu_gelu_half*(1.0f+th)+g_cu_gelu_half*x*(1.0f-th*th)*di));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void gelu_backward_tail_kernel(const T *grad_output,
                                          const T *input,
                                          T *grad_input,
                                          const int64_t offset,
                                          const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float x=float(input[idx]);
    const float inner=g_cu_gelu_sqrt_2_over_pi*(x+g_cu_gelu_coeff*x*x*x);
    const float th=tanhf(inner);
    const float di=g_cu_gelu_sqrt_2_over_pi*(1.0f+g_cu_gelu_cubic_factor*g_cu_gelu_coeff*x*x);
    grad_input[idx]=T(gv*(g_cu_gelu_half*(1.0f+th)+g_cu_gelu_half*x*(1.0f-th*th)*di));
  }
}

//------------------------------------------------------------------------------
// GELU Backward (exact erf): f'(x) = Phi(x) + x*phi(x)
//   Phi(x) = 0.5*(1 + erf(x/sqrt(2)))    (standard-normal cdf)
//   phi(x) = (1/sqrt(2*pi))*exp(-x^2/2)  (standard-normal pdf)
//------------------------------------------------------------------------------
template<typename T>
__global__ void gelu_backward_erf_kernel(const T *grad_output,
                                         const T *input,
                                         T *grad_input,
                                         const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *v_lane=reinterpret_cast<const T*>(&v);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float x=float(v_lane[i]);
      const float cdf=g_cu_gelu_half*(1.0f+erff(x*g_cu_gelu_inv_sqrt2));
      const float pdf=g_cu_gelu_inv_sqrt2pi*expf(-g_cu_gelu_half*x*x);
      r_lane[i]=T(gv*(cdf+x*pdf));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void gelu_backward_erf_tail_kernel(const T *grad_output,
                                              const T *input,
                                              T *grad_input,
                                              const int64_t offset,
                                              const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float x=float(input[idx]);
    const float cdf=g_cu_gelu_half*(1.0f+erff(x*g_cu_gelu_inv_sqrt2));
    const float pdf=g_cu_gelu_inv_sqrt2pi*expf(-g_cu_gelu_half*x*x);
    grad_input[idx]=T(gv*(cdf+x*pdf));
  }
}

//------------------------------------------------------------------------------
// Swish/SiLU Forward: f(x) = x * sigmoid(x)
//------------------------------------------------------------------------------
template<typename T>
__global__ void swish_forward_kernel(const T *input,
                                     T *output,
                                     const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      lane[i]=T(x/(1.0f+expf(-x)));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void swish_forward_tail_kernel(const T *input,
                                          T *output,
                                          const int64_t offset,
                                          const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    output[idx]=T(x/(1.0f+expf(-x)));
  }
}

//------------------------------------------------------------------------------
// Swish/SiLU Backward: d/dx(x*sig(x)) = sig(x) * (1 + x * (1 - sig(x)))
// Note: the *output* argument from the forward pass is unused here — the
// closed-form expression recomputes sig(x) directly, which is cheap.
//------------------------------------------------------------------------------
template<typename T>
__global__ void swish_backward_kernel(const T *grad_output,
                                      const T *input,
                                      T *grad_input,
                                      const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *v_lane=reinterpret_cast<const T*>(&v);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float x=float(v_lane[i]);
      const float s=1.0f/(1.0f+expf(-x));
      r_lane[i]=T(gv*s*(1.0f+x*(1.0f-s)));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void swish_backward_tail_kernel(const T *grad_output,
                                           const T *input,
                                           T *grad_input,
                                           const int64_t offset,
                                           const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float x=float(input[idx]);
    const float s=1.0f/(1.0f+expf(-x));
    grad_input[idx]=T(gv*s*(1.0f+x*(1.0f-s)));
  }
}

//------------------------------------------------------------------------------
// LeakyReLU Forward: f(x) = x if x > 0, else alpha * x
// alpha stays float — scalar kernel arg, cast to T per-lane at use.
//------------------------------------------------------------------------------
template<typename T>
__global__ void leaky_relu_forward_kernel(const T *input,
                                          T *output,
                                          const float alpha,
                                          const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      if(x>0.0f)
      {
        lane[i]=T(x);
      }
      else
      {
        lane[i]=T(alpha*x);
      }
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void leaky_relu_forward_tail_kernel(const T *input,
                                               T *output,
                                               const float alpha,
                                               const int64_t offset,
                                               const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    if(x>0.0f)
    {
      output[idx]=T(x);
    }
    else
    {
      output[idx]=T(alpha*x);
    }
  }
}

//------------------------------------------------------------------------------
// LeakyReLU Backward
//------------------------------------------------------------------------------
template<typename T>
__global__ void leaky_relu_backward_kernel(const T *grad_output,
                                           const T *input,
                                           T *grad_input,
                                           const float alpha,
                                           const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 x=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *x_lane=reinterpret_cast<const T*>(&x);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float xv=float(x_lane[i]);
      if(xv>0.0f)
      {
        r_lane[i]=T(gv);
      }
      else
      {
        r_lane[i]=T(alpha*gv);
      }
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void leaky_relu_backward_tail_kernel(const T *grad_output,
                                                const T *input,
                                                T *grad_input,
                                                const float alpha,
                                                const int64_t offset,
                                                const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float xv=float(input[idx]);
    if(xv>0.0f)
    {
      grad_input[idx]=T(gv);
    }
    else
    {
      grad_input[idx]=T(alpha*gv);
    }
  }
}

//------------------------------------------------------------------------------
// ELU Forward: f(x) = x if x > 0, else alpha * (exp(x) - 1)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elu_forward_kernel(const T *input,
                                   T *output,
                                   const float alpha,
                                   const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      if(x>0.0f)
      {
        lane[i]=T(x);
      }
      else
      {
        lane[i]=T(alpha*(expf(x)-1.0f));
      }
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void elu_forward_tail_kernel(const T *input,
                                        T *output,
                                        const float alpha,
                                        const int64_t offset,
                                        const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    if(x>0.0f)
    {
      output[idx]=T(x);
    }
    else
    {
      output[idx]=T(alpha*(expf(x)-1.0f));
    }
  }
}

//------------------------------------------------------------------------------
// ELU Backward: grad = upstream if input > 0, else upstream * (output + alpha)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elu_backward_kernel(const T *grad_output,
                                    const T *input,
                                    const T *output,
                                    T *grad_input,
                                    const float alpha,
                                    const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 x=reinterpret_cast<const int4 *>(input)[idx];
    int4 o=reinterpret_cast<const int4 *>(output)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *x_lane=reinterpret_cast<const T*>(&x);
    const T *o_lane=reinterpret_cast<const T*>(&o);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float xv=float(x_lane[i]);
      const float ov=float(o_lane[i]);
      if(xv>0.0f)
      {
        r_lane[i]=T(gv);
      }
      else
      {
        r_lane[i]=T(gv*(ov+alpha));
      }
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void elu_backward_tail_kernel(const T *grad_output,
                                         const T *input,
                                         const T *output,
                                         T *grad_input,
                                         const float alpha,
                                         const int64_t offset,
                                         const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float xv=float(input[idx]);
    if(xv>0.0f)
    {
      grad_input[idx]=T(gv);
    }
    else
    {
      const float ov=float(output[idx]);
      grad_input[idx]=T(gv*(ov+alpha));
    }
  }
}

//------------------------------------------------------------------------------
// Kernel launcher functions (callable from C++)
// All launchers use float4 vectorized path with scalar tail handling.
//------------------------------------------------------------------------------

template<typename T>
void launch_relu_forward(const T *input,
                         T *output,
                         const int64_t n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    relu_forward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,n_vec);
  }
  if(tail>0)
  {
    relu_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_relu_forward<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_relu_forward<__half>(const __half *,__half *,int64_t,cudaStream_t);
template void launch_relu_forward<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int64_t,cudaStream_t);

template<typename T>
void launch_relu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          const int64_t n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    relu_backward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,input,grad_input,n_vec);
  }
  if(tail>0)
  {
    relu_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,input,grad_input,n_vec*lanes,n);
  }
}
template void launch_relu_backward<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_relu_backward<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int64_t,
                                                  cudaStream_t);

template<typename T>
void launch_sigmoid_forward(const T *input,
                            T *output,
                            const int64_t n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    sigmoid_forward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,n_vec);
  }
  if(tail>0)
  {
    sigmoid_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_sigmoid_forward<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_sigmoid_forward<__half>(const __half *,__half *,int64_t,cudaStream_t);
template void launch_sigmoid_forward<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int64_t,cudaStream_t);

template<typename T>
void launch_sigmoid_backward(const T *grad_output,
                             const T *output,
                             T *grad_input,
                             const int64_t n,
                             cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    sigmoid_backward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,output,grad_input,n_vec);
  }
  if(tail>0)
  {
    sigmoid_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,output,grad_input,n_vec*lanes,n);
  }
}
template void launch_sigmoid_backward<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_sigmoid_backward<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_sigmoid_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     const __nv_bfloat16 *,
                                                     __nv_bfloat16 *,
                                                     int64_t,
                                                     cudaStream_t);

template<typename T>
void launch_tanh_forward(const T *input,
                         T *output,
                         const int64_t n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    tanh_forward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,n_vec);
  }
  if(tail>0)
  {
    tanh_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_tanh_forward<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_tanh_forward<__half>(const __half *,__half *,int64_t,cudaStream_t);
template void launch_tanh_forward<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int64_t,cudaStream_t);

template<typename T>
void launch_tanh_backward(const T *grad_output,
                          const T *output,
                          T *grad_input,
                          const int64_t n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    tanh_backward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,output,grad_input,n_vec);
  }
  if(tail>0)
  {
    tanh_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,output,grad_input,n_vec*lanes,n);
  }
}
template void launch_tanh_backward<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_tanh_backward<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_tanh_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int64_t,
                                                  cudaStream_t);

template<typename T>
void launch_leaky_relu_forward(const T *input,
                               T *output,
                               const float alpha,
                               const int64_t n,
                               cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    leaky_relu_forward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,alpha,n_vec);
  }
  if(tail>0)
  {
    leaky_relu_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,alpha,n_vec*lanes,n);
  }
}
template void launch_leaky_relu_forward<float>(const float *,float *,float,int64_t,cudaStream_t);
template void launch_leaky_relu_forward<__half>(const __half *,__half *,float,int64_t,cudaStream_t);
template void launch_leaky_relu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       __nv_bfloat16 *,
                                                       float,
                                                       int64_t,
                                                       cudaStream_t);

template<typename T>
void launch_leaky_relu_backward(const T *grad_output,
                                const T *input,
                                T *grad_input,
                                const float alpha,
                                const int64_t n,
                                cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    leaky_relu_backward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,
                                                                           input,
                                                                           grad_input,
                                                                           alpha,
                                                                           n_vec);
  }
  if(tail>0)
  {
    leaky_relu_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,input,grad_input,alpha,n_vec*lanes,n);
  }
}
template void launch_leaky_relu_backward<float>(const float *,const float *,float *,float,int64_t,cudaStream_t);
template void launch_leaky_relu_backward<__half>(const __half *,const __half *,__half *,float,int64_t,cudaStream_t);
template void launch_leaky_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        float,
                                                        int64_t,
                                                        cudaStream_t);

template<typename T>
void launch_gelu_forward(const T *input,
                         T *output,
                         const int64_t n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    gelu_forward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,n_vec);
  }
  if(tail>0)
  {
    gelu_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_gelu_forward<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_gelu_forward<__half>(const __half *,__half *,int64_t,cudaStream_t);
template void launch_gelu_forward<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int64_t,cudaStream_t);

template<typename T>
void launch_gelu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          const int64_t n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    gelu_backward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,input,grad_input,n_vec);
  }
  if(tail>0)
  {
    gelu_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,input,grad_input,n_vec*lanes,n);
  }
}
template void launch_gelu_backward<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_gelu_backward<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_gelu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int64_t,
                                                  cudaStream_t);

template<typename T>
void launch_gelu_forward_erf(const T *input,
                             T *output,
                             const int64_t n,
                             cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    gelu_forward_erf_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,n_vec);
  }
  if(tail>0)
  {
    gelu_forward_erf_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_gelu_forward_erf<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_gelu_forward_erf<__half>(const __half *,__half *,int64_t,cudaStream_t);
template void launch_gelu_forward_erf<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int64_t,cudaStream_t);

template<typename T>
void launch_gelu_backward_erf(const T *grad_output,
                              const T *input,
                              T *grad_input,
                              const int64_t n,
                              cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    gelu_backward_erf_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,input,grad_input,n_vec);
  }
  if(tail>0)
  {
    gelu_backward_erf_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,input,grad_input,n_vec*lanes,n);
  }
}
template void launch_gelu_backward_erf<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_gelu_backward_erf<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_gelu_backward_erf<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      const __nv_bfloat16 *,
                                                      __nv_bfloat16 *,
                                                      int64_t,
                                                      cudaStream_t);

template<typename T>
void launch_swish_forward(const T *input,
                          T *output,
                          const int64_t n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    swish_forward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,n_vec);
  }
  if(tail>0)
  {
    swish_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_swish_forward<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_swish_forward<__half>(const __half *,__half *,int64_t,cudaStream_t);
template void launch_swish_forward<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int64_t,cudaStream_t);

// Note: the `output` pointer is kept in the launcher signature for API
// symmetry with other backward launchers; the closed-form backward
// recomputes sigmoid(x) from input and the kernel does not read it.
template<typename T>
void launch_swish_backward(const T *grad_output,
                           const T *input,
                           const T *output,
                           T *grad_input,
                           const int64_t n,
                           cudaStream_t stream)
{
  (void)output;
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    swish_backward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,input,grad_input,n_vec);
  }
  if(tail>0)
  {
    swish_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,input,grad_input,n_vec*lanes,n);
  }
}
template void launch_swish_backward<float>(const float *,const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_swish_backward<__half>(const __half *,
                                            const __half *,
                                            const __half *,
                                            __half *,
                                            int64_t,
                                            cudaStream_t);
template void launch_swish_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int64_t,
                                                   cudaStream_t);

template<typename T>
void launch_elu_forward(const T *input,
                        T *output,
                        const float alpha,
                        const int64_t n,
                        cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elu_forward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(input,output,alpha,n_vec);
  }
  if(tail>0)
  {
    elu_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,alpha,n_vec*lanes,n);
  }
}
template void launch_elu_forward<float>(const float *,float *,float,int64_t,cudaStream_t);
template void launch_elu_forward<__half>(const __half *,__half *,float,int64_t,cudaStream_t);
template void launch_elu_forward<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,float,int64_t,cudaStream_t);

template<typename T>
void launch_elu_backward(const T *grad_output,
                         const T *input,
                         const T *output,
                         T *grad_input,
                         const float alpha,
                         const int64_t n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elu_backward_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(grad_output,
                                                                    input,
                                                                    output,
                                                                    grad_input,
                                                                    alpha,
                                                                    n_vec);
  }
  if(tail>0)
  {
    elu_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,input,output,grad_input,alpha,n_vec*lanes,n);
  }
}
template void launch_elu_backward<float>(const float *,
                                         const float *,
                                         const float *,
                                         float *,
                                         float,
                                         int64_t,
                                         cudaStream_t);
template void launch_elu_backward<__half>(const __half *,
                                          const __half *,
                                          const __half *,
                                          __half *,
                                          float,
                                          int64_t,
                                          cudaStream_t);
template void launch_elu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                 const __nv_bfloat16 *,
                                                 const __nv_bfloat16 *,
                                                 __nv_bfloat16 *,
                                                 float,
                                                 int64_t,
                                                 cudaStream_t);

//------------------------------------------------------------------------------
// Gated Activation device helpers
//------------------------------------------------------------------------------
__device__ float gated_apply_op(float x,int op)
{
  // Swish: x * sigmoid(x)
  if(op==CAIF_GATED_OP_SWISH)
  {
    return x/(1.0f+expf(-x));
  }
  // GELU approximation
  if(op==CAIF_GATED_OP_GELU)
  {
    const float inner=g_cu_gelu_sqrt_2_over_pi*(x+g_cu_gelu_coeff*x*x*x);
    return g_cu_gelu_half*x*(1.0f+tanhf(inner));
  }
  // ReLU
  if(op==CAIF_GATED_OP_RELU)
  {
    if(x>0.0f)
    {
      return x;
    }
    return 0.0f;
  }
  // Sigmoid
  if(op==CAIF_GATED_OP_SIGMOID)
  {
    return 1.0f/(1.0f+expf(-x));
  }
  // CAIF_GATED_OP_LINEAR: identity
  return x;
}

__device__ float gated_apply_op_derivative(float x,float activated,int op)
{
  // Swish derivative: sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
  //                 = activated/x + activated*(1-activated/x) when x!=0
  if(op==CAIF_GATED_OP_SWISH)
  {
    const float sig=1.0f/(1.0f+expf(-x));
    return sig+x*sig*(1.0f-sig);
  }
  // GELU derivative
  if(op==CAIF_GATED_OP_GELU)
  {
    const float inner=g_cu_gelu_sqrt_2_over_pi*(x+g_cu_gelu_coeff*x*x*x);
    const float tanh_val=tanhf(inner);
    const float sech2=1.0f-tanh_val*tanh_val;
    const float d_inner=g_cu_gelu_sqrt_2_over_pi*(1.0f+g_cu_gelu_cubic_factor*g_cu_gelu_coeff*x*x);
    return g_cu_gelu_half*(1.0f+tanh_val)+g_cu_gelu_half*x*sech2*d_inner;
  }
  // ReLU derivative
  if(op==CAIF_GATED_OP_RELU)
  {
    (void)activated;
    if(x>0.0f)
    {
      return 1.0f;
    }
    return 0.0f;
  }
  // Sigmoid derivative: sigmoid(x)*(1-sigmoid(x))
  if(op==CAIF_GATED_OP_SIGMOID)
  {
    return activated*(1.0f-activated);
  }
  // CAIF_GATED_OP_LINEAR: derivative of identity
  (void)activated;
  return 1.0f;
}

//------------------------------------------------------------------------------
// Gated Activation Forward Kernel
// output[i] = apply_op(gate_input[i], op) * up_input[i]
//------------------------------------------------------------------------------
template<typename T>
__global__ void gated_activation_forward_kernel(const T *gate_input,
                                                const T *up_input,
                                                T *output,
                                                const int op,
                                                const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float g=float(gate_input[idx]);
    const float u=float(up_input[idx]);
    const float activated=gated_apply_op(g,op);
    output[idx]=T(activated*u);
  }
}

//------------------------------------------------------------------------------
// Gated Activation Backward Kernel
// grad_gate[i] = grad_output[i] * up[i] * d_activate(gate[i])
// grad_up[i]   = grad_output[i] * activate(gate[i])
//------------------------------------------------------------------------------
template<typename T>
__global__ void gated_activation_backward_kernel(const T *grad_output,
                                                 const T *cached_gate_input,
                                                 const T *cached_up_input,
                                                 T *grad_gate,
                                                 T *grad_up,
                                                 const int op,
                                                 const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float g=float(cached_gate_input[idx]);
    const float u=float(cached_up_input[idx]);
    const float go=float(grad_output[idx]);
    const float activated=gated_apply_op(g,op);
    const float d_activated=gated_apply_op_derivative(g,activated,op);
    grad_gate[idx]=T(go*u*d_activated);
    grad_up[idx]=T(go*activated);
  }
}

//------------------------------------------------------------------------------
// Gated Activation Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_gated_activation_forward(const T *gate_input,
                                     const T *up_input,
                                     T *output,
                                     const int op,
                                     const int64_t n,
                                     cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  gated_activation_forward_kernel<T><<<num_blocks,block_size,0,stream>>>(gate_input,up_input,output,op,n);
}

template void launch_gated_activation_forward<float>(const float *,const float *,float *,int,int64_t,cudaStream_t);
template void launch_gated_activation_forward<__half>(const __half *,
                                                      const __half *,
                                                      __half *,
                                                      int,
                                                      int64_t,
                                                      cudaStream_t);
template void launch_gated_activation_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             int,
                                                             int64_t,
                                                             cudaStream_t);

template<typename T>
void launch_gated_activation_backward(const T *grad_output,
                                      const T *cached_gate_input,
                                      const T *cached_up_input,
                                      T *grad_gate,
                                      T *grad_up,
                                      const int op,
                                      const int64_t n,
                                      cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  gated_activation_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_output,
                                                                          cached_gate_input,
                                                                          cached_up_input,
                                                                          grad_gate,
                                                                          grad_up,
                                                                          op,
                                                                          n);
}

template void launch_gated_activation_backward<float>(const float *,
                                                      const float *,
                                                      const float *,
                                                      float *,
                                                      float *,
                                                      int,
                                                      int64_t,
                                                      cudaStream_t);
template void launch_gated_activation_backward<__half>(const __half *,
                                                       const __half *,
                                                       const __half *,
                                                       __half *,
                                                       __half *,
                                                       int,
                                                       int64_t,
                                                       cudaStream_t);
template void launch_gated_activation_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              int,
                                                              int64_t,
                                                              cudaStream_t);
