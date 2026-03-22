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
// AIF - AI Framework
// Custom CUDA kernels for operations not supported by cuDNN
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

extern "C"
{

//------------------------------------------------------------------------------
// Vectorized Activation Kernels (float4)
// All activation kernels use float4 loads/stores for 4x memory efficiency.
// Each thread processes 4 elements. Tail elements handled by scalar fallback.
//------------------------------------------------------------------------------
constexpr int g_act_block_size=256;

//------------------------------------------------------------------------------
// ReLU Forward: f(x) = max(0, x)
//------------------------------------------------------------------------------
__global__ void relu_forward_kernel(const float *input,
                                    float *output,
                                    const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    r.x=fmaxf(v.x,0.0f);
    r.y=fmaxf(v.y,0.0f);
    r.z=fmaxf(v.z,0.0f);
    r.w=fmaxf(v.w,0.0f);
    reinterpret_cast<float4 *>(output)[idx]=r;
  }
}

__global__ void relu_forward_tail_kernel(const float *input,
                                         float *output,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=fmaxf(input[idx],0.0f);
  }
}

//------------------------------------------------------------------------------
// ReLU Backward: grad = upstream if input > 0, else 0
//------------------------------------------------------------------------------
__global__ void relu_backward_kernel(const float *grad_output,
                                     const float *input,
                                     float *grad_input,
                                     const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 x=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    if(x.x>0.0f){r.x=g.x;}else{r.x=0.0f;}
    if(x.y>0.0f){r.y=g.y;}else{r.y=0.0f;}
    if(x.z>0.0f){r.z=g.z;}else{r.z=0.0f;}
    if(x.w>0.0f){r.w=g.w;}else{r.w=0.0f;}
    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void relu_backward_tail_kernel(const float *grad_output,
                                          const float *input,
                                          float *grad_input,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    if(input[idx]>0.0f)
    {
      grad_input[idx]=grad_output[idx];
    }
    else
    {
      grad_input[idx]=0.0f;
    }
  }
}

//------------------------------------------------------------------------------
// Sigmoid Forward: f(x) = 1 / (1 + exp(-x))
//------------------------------------------------------------------------------
__global__ void sigmoid_forward_kernel(const float *input,
                                       float *output,
                                       const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    r.x=1.0f/(1.0f+expf(-v.x));
    r.y=1.0f/(1.0f+expf(-v.y));
    r.z=1.0f/(1.0f+expf(-v.z));
    r.w=1.0f/(1.0f+expf(-v.w));
    reinterpret_cast<float4 *>(output)[idx]=r;
  }
}

__global__ void sigmoid_forward_tail_kernel(const float *input,
                                            float *output,
                                            const int offset,
                                            const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=1.0f/(1.0f+expf(-input[idx]));
  }
}

//------------------------------------------------------------------------------
// Sigmoid Backward: grad = upstream * output * (1 - output)
//------------------------------------------------------------------------------
__global__ void sigmoid_backward_kernel(const float *grad_output,
                                        const float *output,
                                        float *grad_input,
                                        const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 o=reinterpret_cast<const float4 *>(output)[idx];
    float4 r;
    r.x=g.x*o.x*(1.0f-o.x);
    r.y=g.y*o.y*(1.0f-o.y);
    r.z=g.z*o.z*(1.0f-o.z);
    r.w=g.w*o.w*(1.0f-o.w);
    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void sigmoid_backward_tail_kernel(const float *grad_output,
                                             const float *output,
                                             float *grad_input,
                                             const int offset,
                                             const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float o=output[idx];
    grad_input[idx]=grad_output[idx]*o*(1.0f-o);
  }
}

//------------------------------------------------------------------------------
// Tanh Forward: f(x) = tanh(x)
//------------------------------------------------------------------------------
__global__ void tanh_forward_kernel(const float *input,
                                    float *output,
                                    const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    r.x=tanhf(v.x);
    r.y=tanhf(v.y);
    r.z=tanhf(v.z);
    r.w=tanhf(v.w);
    reinterpret_cast<float4 *>(output)[idx]=r;
  }
}

__global__ void tanh_forward_tail_kernel(const float *input,
                                         float *output,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=tanhf(input[idx]);
  }
}

//------------------------------------------------------------------------------
// Tanh Backward: grad = upstream * (1 - output^2)
//------------------------------------------------------------------------------
__global__ void tanh_backward_kernel(const float *grad_output,
                                     const float *output,
                                     float *grad_input,
                                     const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 o=reinterpret_cast<const float4 *>(output)[idx];
    float4 r;
    r.x=g.x*(1.0f-o.x*o.x);
    r.y=g.y*(1.0f-o.y*o.y);
    r.z=g.z*(1.0f-o.z*o.z);
    r.w=g.w*(1.0f-o.w*o.w);
    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void tanh_backward_tail_kernel(const float *grad_output,
                                          const float *output,
                                          float *grad_input,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float o=output[idx];
    grad_input[idx]=grad_output[idx]*(1.0f-o*o);
  }
}

//------------------------------------------------------------------------------
// GELU Forward: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//------------------------------------------------------------------------------
constexpr float g_gelu_sqrt_2_over_pi=0.7978845608f;
constexpr float g_gelu_coeff=0.044715f;

__global__ void gelu_forward_kernel(const float *input,
                                    float *output,
                                    const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    r.x=0.5f*v.x*(1.0f+tanhf(g_gelu_sqrt_2_over_pi*
                               (v.x+g_gelu_coeff*v.x*v.x*v.x)));
    r.y=0.5f*v.y*(1.0f+tanhf(g_gelu_sqrt_2_over_pi*
                               (v.y+g_gelu_coeff*v.y*v.y*v.y)));
    r.z=0.5f*v.z*(1.0f+tanhf(g_gelu_sqrt_2_over_pi*
                               (v.z+g_gelu_coeff*v.z*v.z*v.z)));
    r.w=0.5f*v.w*(1.0f+tanhf(g_gelu_sqrt_2_over_pi*
                               (v.w+g_gelu_coeff*v.w*v.w*v.w)));
    reinterpret_cast<float4 *>(output)[idx]=r;
  }
}

__global__ void gelu_forward_tail_kernel(const float *input,
                                         float *output,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=input[idx];
    const float inner=g_gelu_sqrt_2_over_pi*(x+g_gelu_coeff*x*x*x);
    output[idx]=0.5f*x*(1.0f+tanhf(inner));
  }
}

//------------------------------------------------------------------------------
// GELU Backward
//------------------------------------------------------------------------------
__global__ void gelu_backward_kernel(const float *grad_output,
                                     const float *input,
                                     float *grad_input,
                                     const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;

    float inner=g_gelu_sqrt_2_over_pi*(v.x+g_gelu_coeff*v.x*v.x*v.x);
    float th=tanhf(inner);
    float di=g_gelu_sqrt_2_over_pi*(1.0f+3.0f*g_gelu_coeff*v.x*v.x);
    r.x=g.x*(0.5f*(1.0f+th)+0.5f*v.x*(1.0f-th*th)*di);

    inner=g_gelu_sqrt_2_over_pi*(v.y+g_gelu_coeff*v.y*v.y*v.y);
    th=tanhf(inner);
    di=g_gelu_sqrt_2_over_pi*(1.0f+3.0f*g_gelu_coeff*v.y*v.y);
    r.y=g.y*(0.5f*(1.0f+th)+0.5f*v.y*(1.0f-th*th)*di);

    inner=g_gelu_sqrt_2_over_pi*(v.z+g_gelu_coeff*v.z*v.z*v.z);
    th=tanhf(inner);
    di=g_gelu_sqrt_2_over_pi*(1.0f+3.0f*g_gelu_coeff*v.z*v.z);
    r.z=g.z*(0.5f*(1.0f+th)+0.5f*v.z*(1.0f-th*th)*di);

    inner=g_gelu_sqrt_2_over_pi*(v.w+g_gelu_coeff*v.w*v.w*v.w);
    th=tanhf(inner);
    di=g_gelu_sqrt_2_over_pi*(1.0f+3.0f*g_gelu_coeff*v.w*v.w);
    r.w=g.w*(0.5f*(1.0f+th)+0.5f*v.w*(1.0f-th*th)*di);

    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void gelu_backward_tail_kernel(const float *grad_output,
                                          const float *input,
                                          float *grad_input,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=input[idx];
    const float inner=g_gelu_sqrt_2_over_pi*(x+g_gelu_coeff*x*x*x);
    const float th=tanhf(inner);
    const float di=g_gelu_sqrt_2_over_pi*(1.0f+3.0f*g_gelu_coeff*x*x);
    grad_input[idx]=grad_output[idx]*
                    (0.5f*(1.0f+th)+0.5f*x*(1.0f-th*th)*di);
  }
}

//------------------------------------------------------------------------------
// Swish/SiLU Forward: f(x) = x * sigmoid(x)
//------------------------------------------------------------------------------
__global__ void swish_forward_kernel(const float *input,
                                     float *output,
                                     const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    r.x=v.x/(1.0f+expf(-v.x));
    r.y=v.y/(1.0f+expf(-v.y));
    r.z=v.z/(1.0f+expf(-v.z));
    r.w=v.w/(1.0f+expf(-v.w));
    reinterpret_cast<float4 *>(output)[idx]=r;
  }
}

__global__ void swish_forward_tail_kernel(const float *input,
                                          float *output,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=input[idx];
    output[idx]=x/(1.0f+expf(-x));
  }
}

//------------------------------------------------------------------------------
// Swish/SiLU Backward: d/dx(x*sig(x)) = sig(x) * (1 + x * (1 - sig(x)))
//------------------------------------------------------------------------------
__global__ void swish_backward_kernel(const float *grad_output,
                                      const float *input,
                                      const float *output,
                                      float *grad_input,
                                      const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    float s;
    s=1.0f/(1.0f+expf(-v.x));r.x=g.x*s*(1.0f+v.x*(1.0f-s));
    s=1.0f/(1.0f+expf(-v.y));r.y=g.y*s*(1.0f+v.y*(1.0f-s));
    s=1.0f/(1.0f+expf(-v.z));r.z=g.z*s*(1.0f+v.z*(1.0f-s));
    s=1.0f/(1.0f+expf(-v.w));r.w=g.w*s*(1.0f+v.w*(1.0f-s));
    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void swish_backward_tail_kernel(const float *grad_output,
                                           const float *input,
                                           float *grad_input,
                                           const int offset,
                                           const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=input[idx];
    const float s=1.0f/(1.0f+expf(-x));
    grad_input[idx]=grad_output[idx]*s*(1.0f+x*(1.0f-s));
  }
}

//------------------------------------------------------------------------------
// LeakyReLU Forward: f(x) = x if x > 0, else alpha * x
//------------------------------------------------------------------------------
__global__ void leaky_relu_forward_kernel(const float *input,
                                          float *output,
                                          const float alpha,
                                          const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    if(v.x>0.0f){r.x=v.x;}else{r.x=alpha*v.x;}
    if(v.y>0.0f){r.y=v.y;}else{r.y=alpha*v.y;}
    if(v.z>0.0f){r.z=v.z;}else{r.z=alpha*v.z;}
    if(v.w>0.0f){r.w=v.w;}else{r.w=alpha*v.w;}
    reinterpret_cast<float4 *>(output)[idx]=r;
  }
}

__global__ void leaky_relu_forward_tail_kernel(const float *input,
                                               float *output,
                                               const float alpha,
                                               const int offset,
                                               const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=input[idx];
    if(x>0.0f)
    {
      output[idx]=x;
    }
    else
    {
      output[idx]=alpha*x;
    }
  }
}

//------------------------------------------------------------------------------
// LeakyReLU Backward
//------------------------------------------------------------------------------
__global__ void leaky_relu_backward_kernel(const float *grad_output,
                                           const float *input,
                                           float *grad_input,
                                           const float alpha,
                                           const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 x=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    if(x.x>0.0f){r.x=g.x;}else{r.x=alpha*g.x;}
    if(x.y>0.0f){r.y=g.y;}else{r.y=alpha*g.y;}
    if(x.z>0.0f){r.z=g.z;}else{r.z=alpha*g.z;}
    if(x.w>0.0f){r.w=g.w;}else{r.w=alpha*g.w;}
    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void leaky_relu_backward_tail_kernel(const float *grad_output,
                                                const float *input,
                                                float *grad_input,
                                                const float alpha,
                                                const int offset,
                                                const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    if(input[idx]>0.0f)
    {
      grad_input[idx]=grad_output[idx];
    }
    else
    {
      grad_input[idx]=alpha*grad_output[idx];
    }
  }
}

//------------------------------------------------------------------------------
// ELU Forward: f(x) = x if x > 0, else alpha * (exp(x) - 1)
//------------------------------------------------------------------------------
__global__ void elu_forward_kernel(const float *input,
                                   float *output,
                                   const float alpha,
                                   const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    if(v.x>0.0f){r.x=v.x;}else{r.x=alpha*(expf(v.x)-1.0f);}
    if(v.y>0.0f){r.y=v.y;}else{r.y=alpha*(expf(v.y)-1.0f);}
    if(v.z>0.0f){r.z=v.z;}else{r.z=alpha*(expf(v.z)-1.0f);}
    if(v.w>0.0f){r.w=v.w;}else{r.w=alpha*(expf(v.w)-1.0f);}
    reinterpret_cast<float4 *>(output)[idx]=r;
  }
}

__global__ void elu_forward_tail_kernel(const float *input,
                                        float *output,
                                        const float alpha,
                                        const int offset,
                                        const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=input[idx];
    if(x>0.0f)
    {
      output[idx]=x;
    }
    else
    {
      output[idx]=alpha*(expf(x)-1.0f);
    }
  }
}

//------------------------------------------------------------------------------
// ELU Backward: grad = upstream if input > 0, else upstream * (output + alpha)
//------------------------------------------------------------------------------
__global__ void elu_backward_kernel(const float *grad_output,
                                    const float *input,
                                    const float *output,
                                    float *grad_input,
                                    const float alpha,
                                    const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 x=reinterpret_cast<const float4 *>(input)[idx];
    const float4 o=reinterpret_cast<const float4 *>(output)[idx];
    float4 r;
    if(x.x>0.0f){r.x=g.x;}else{r.x=g.x*(o.x+alpha);}
    if(x.y>0.0f){r.y=g.y;}else{r.y=g.y*(o.y+alpha);}
    if(x.z>0.0f){r.z=g.z;}else{r.z=g.z*(o.z+alpha);}
    if(x.w>0.0f){r.w=g.w;}else{r.w=g.w*(o.w+alpha);}
    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void elu_backward_tail_kernel(const float *grad_output,
                                         const float *input,
                                         const float *output,
                                         float *grad_input,
                                         const float alpha,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    if(input[idx]>0.0f)
    {
      grad_input[idx]=grad_output[idx];
    }
    else
    {
      grad_input[idx]=grad_output[idx]*(output[idx]+alpha);
    }
  }
}

//------------------------------------------------------------------------------
// Kernel launcher functions (callable from C++)
// All launchers use float4 vectorized path with scalar tail handling.
//------------------------------------------------------------------------------

void launch_relu_forward(const float *input,
                         float *output,
                         const int n,
                         cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    relu_forward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      input,output,n4);
  }
  if(tail>0)
  {
    relu_forward_tail_kernel<<<1,tail,0,stream>>>(
      input,output,n4*4,n);
  }
}

void launch_relu_backward(const float *grad_output,
                           const float *input,
                           float *grad_input,
                           const int n,
                           cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    relu_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,input,grad_input,n4);
  }
  if(tail>0)
  {
    relu_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,input,grad_input,n4*4,n);
  }
}

void launch_sigmoid_forward(const float *input,
                             float *output,
                             const int n,
                             cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    sigmoid_forward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      input,output,n4);
  }
  if(tail>0)
  {
    sigmoid_forward_tail_kernel<<<1,tail,0,stream>>>(
      input,output,n4*4,n);
  }
}

void launch_sigmoid_backward(const float *grad_output,
                              const float *output,
                              float *grad_input,
                              const int n,
                              cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    sigmoid_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,output,grad_input,n4);
  }
  if(tail>0)
  {
    sigmoid_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,output,grad_input,n4*4,n);
  }
}

void launch_tanh_forward(const float *input,
                          float *output,
                          const int n,
                          cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    tanh_forward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      input,output,n4);
  }
  if(tail>0)
  {
    tanh_forward_tail_kernel<<<1,tail,0,stream>>>(
      input,output,n4*4,n);
  }
}

void launch_tanh_backward(const float *grad_output,
                           const float *output,
                           float *grad_input,
                           const int n,
                           cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    tanh_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,output,grad_input,n4);
  }
  if(tail>0)
  {
    tanh_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,output,grad_input,n4*4,n);
  }
}

void launch_leaky_relu_forward(const float *input,
                               float *output,
                               const float alpha,
                               const int n,
                               cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    leaky_relu_forward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      input,output,alpha,n4);
  }
  if(tail>0)
  {
    leaky_relu_forward_tail_kernel<<<1,tail,0,stream>>>(
      input,output,alpha,n4*4,n);
  }
}

void launch_leaky_relu_backward(const float *grad_output,
                                const float *input,
                                float *grad_input,
                                const float alpha,
                                const int n,
                                cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    leaky_relu_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,input,grad_input,alpha,n4);
  }
  if(tail>0)
  {
    leaky_relu_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,input,grad_input,alpha,n4*4,n);
  }
}

void launch_gelu_forward(const float *input,
                         float *output,
                         const int n,
                         cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    gelu_forward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      input,output,n4);
  }
  if(tail>0)
  {
    gelu_forward_tail_kernel<<<1,tail,0,stream>>>(
      input,output,n4*4,n);
  }
}

void launch_gelu_backward(const float *grad_output,
                          const float *input,
                          float *grad_input,
                          const int n,
                          cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    gelu_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,input,grad_input,n4);
  }
  if(tail>0)
  {
    gelu_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,input,grad_input,n4*4,n);
  }
}

void launch_swish_forward(const float *input,
                          float *output,
                          const int n,
                          cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    swish_forward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      input,output,n4);
  }
  if(tail>0)
  {
    swish_forward_tail_kernel<<<1,tail,0,stream>>>(
      input,output,n4*4,n);
  }
}

void launch_swish_backward(const float *grad_output,
                           const float *input,
                           const float *output,
                           float *grad_input,
                           const int n,
                           cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    swish_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,input,output,grad_input,n4);
  }
  if(tail>0)
  {
    swish_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,input,grad_input,n4*4,n);
  }
}

void launch_elu_forward(const float *input,
                        float *output,
                        const float alpha,
                        const int n,
                        cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    elu_forward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      input,output,alpha,n4);
  }
  if(tail>0)
  {
    elu_forward_tail_kernel<<<1,tail,0,stream>>>(
      input,output,alpha,n4*4,n);
  }
}

void launch_elu_backward(const float *grad_output,
                         const float *input,
                         const float *output,
                         float *grad_input,
                         const float alpha,
                         const int n,
                         cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    elu_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,input,output,grad_input,alpha,n4);
  }
  if(tail>0)
  {
    elu_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,input,output,grad_input,alpha,n4*4,n);
  }
}

//------------------------------------------------------------------------------
// Element-wise Add Kernel (tensor + tensor)
//------------------------------------------------------------------------------
__global__ void elementwise_add_kernel(const float *a,
                                       const float *b,
                                       float *result,
                                       const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]+b[idx];
  }
}

//------------------------------------------------------------------------------
// Element-wise Add Scalar Kernel (tensor + scalar)
//------------------------------------------------------------------------------
__global__ void elementwise_add_scalar_kernel(const float *a,
                                              const float scalar,
                                              float *result,
                                              const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]+scalar;
  }
}

//------------------------------------------------------------------------------
// Bias Add (2D: broadcast bias over batch rows)
//------------------------------------------------------------------------------
__global__ void bias_add_2d_kernel(const float *input,
                                   const float *bias,
                                   float *output,
                                   const int units,
                                   const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int col=idx%units;
    output[idx]=input[idx]+bias[col];
  }
}

//------------------------------------------------------------------------------
// Bias Gradient (sum over batch rows)
//------------------------------------------------------------------------------
__global__ void bias_grad_2d_kernel(const float *grad,
                                    float *bias_grad,
                                    const int batch,
                                    const int units)
{
  const int u=blockIdx.x*blockDim.x+threadIdx.x;
  if(u<units)
  {
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=grad[b*units+u];
    }
    bias_grad[u]=sum;
  }
}

//------------------------------------------------------------------------------
// Element-wise Subtract Kernel (tensor - tensor)
//------------------------------------------------------------------------------
__global__ void elementwise_sub_kernel(const float *a,
                                       const float *b,
                                       float *result,
                                       const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]-b[idx];
  }
}

//------------------------------------------------------------------------------
// Element-wise Subtract Scalar Kernel (tensor - scalar)
//------------------------------------------------------------------------------
__global__ void elementwise_sub_scalar_kernel(const float *a,
                                              const float scalar,
                                              float *result,
                                              const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]-scalar;
  }
}

//------------------------------------------------------------------------------
// Element-wise Multiply Kernel (tensor * tensor)
//------------------------------------------------------------------------------
__global__ void elementwise_mul_kernel(const float *a,
                                       const float *b,
                                       float *result,
                                       const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]*b[idx];
  }
}

//------------------------------------------------------------------------------
// Element-wise Multiply Scalar Kernel (tensor * scalar)
//------------------------------------------------------------------------------
__global__ void elementwise_mul_scalar_kernel(const float *a,
                                              const float scalar,
                                              float *result,
                                              const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]*scalar;
  }
}

//------------------------------------------------------------------------------
// Element-wise Divide Kernel (tensor / tensor)
//------------------------------------------------------------------------------
__global__ void elementwise_div_kernel(const float *a,
                                       const float *b,
                                       float *result,
                                       const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]/b[idx];
  }
}

//------------------------------------------------------------------------------
// Element-wise Divide Scalar Kernel (tensor / scalar)
//------------------------------------------------------------------------------
__global__ void elementwise_div_scalar_kernel(const float *a,
                                              const float scalar,
                                              float *result,
                                              const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=a[idx]/scalar;
  }
}

//------------------------------------------------------------------------------
// Element-wise Sqrt Kernel
//------------------------------------------------------------------------------
__global__ void elementwise_sqrt_kernel(const float *a,
                                        float *result,
                                        const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=sqrtf(a[idx]);
  }
}

//------------------------------------------------------------------------------
// Reduction Sum Kernel (parallel reduction)
// Uses shared memory for efficient intra-block reduction
//------------------------------------------------------------------------------
__global__ void reduction_sum_kernel(const float *input,
                                     float *output,
                                     const int n)
{
  extern __shared__ float sdata[];

  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;

  // Load element into shared memory (or 0 if out of bounds)
  if(idx<n)
  {
    sdata[tid]=input[idx];
  }
  else
  {
    sdata[tid]=0.0f;
  }
  __syncthreads();

  // Parallel reduction in shared memory
  for(int s=blockDim.x/2; s>0; s>>=1)
  {
    if(tid<s)
    {
      sdata[tid]+=sdata[tid+s];
    }
    __syncthreads();
  }

  // Write result for this block
  if(tid==0)
  {
    output[blockIdx.x]=sdata[0];
  }
}

//------------------------------------------------------------------------------
// Element-wise launcher functions
//------------------------------------------------------------------------------
void launch_elementwise_add(const float *a,
                            const float *b,
                            float *result,
                            const int n,
                            cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_add_kernel<<<num_blocks,block_size,0,stream>>>(a,b,result,n);
}

void launch_elementwise_add_scalar(const float *a,
                                   const float scalar,
                                   float *result,
                                   const int n,
                                   cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_add_scalar_kernel<<<num_blocks,block_size,0,stream>>>(a,scalar,result,n);
}

void launch_bias_add_2d(const float *input,
                        const float *bias,
                        float *output,
                        const int batch,
                        const int units,
                        cudaStream_t stream)
{
  const int total=batch*units;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  bias_add_2d_kernel<<<num_blocks,block_size,0,stream>>>(input,bias,output,units,total);
}

void launch_bias_grad_2d(const float *grad_output,
                         float *bias_grad,
                         const int batch,
                         const int units,
                         cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(units+block_size-1)/block_size;
  bias_grad_2d_kernel<<<num_blocks,block_size,0,stream>>>(grad_output,bias_grad,batch,units);
}

void launch_elementwise_sub(const float *a,
                            const float *b,
                            float *result,
                            const int n,
                            cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_sub_kernel<<<num_blocks,block_size,0,stream>>>(a,b,result,n);
}

void launch_elementwise_sub_scalar(const float *a,
                                   const float scalar,
                                   float *result,
                                   const int n,
                                   cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_sub_scalar_kernel<<<num_blocks,block_size,0,stream>>>(a,scalar,result,n);
}

void launch_elementwise_mul(const float *a,
                            const float *b,
                            float *result,
                            const int n,
                            cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_mul_kernel<<<num_blocks,block_size,0,stream>>>(a,b,result,n);
}

void launch_elementwise_mul_scalar(const float *a,
                                   const float scalar,
                                   float *result,
                                   const int n,
                                   cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_mul_scalar_kernel<<<num_blocks,block_size,0,stream>>>(a,scalar,result,n);
}

void launch_elementwise_div(const float *a,
                            const float *b,
                            float *result,
                            const int n,
                            cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_div_kernel<<<num_blocks,block_size,0,stream>>>(a,b,result,n);
}

void launch_elementwise_div_scalar(const float *a,
                                   const float scalar,
                                   float *result,
                                   const int n,
                                   cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_div_scalar_kernel<<<num_blocks,block_size,0,stream>>>(a,scalar,result,n);
}

void launch_elementwise_sqrt(const float *a,
                             float *result,
                             const int n,
                             cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  elementwise_sqrt_kernel<<<num_blocks,block_size,0,stream>>>(a,result,n);
}

void launch_reduction_sum(const float *input,
                          float *output,
                          const int n,
                          cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  const size_t shared_mem_size=block_size*sizeof(float);
  reduction_sum_kernel<<<num_blocks,block_size,shared_mem_size,stream>>>(input,output,n);
}

//------------------------------------------------------------------------------
// Loss Function Kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Cross Entropy Loss Kernel
// loss_i = -target_i * log(max(epsilon, min(1-epsilon, pred_i)))
//------------------------------------------------------------------------------
__global__ void cross_entropy_loss_kernel(const float *predictions,
                                          const float *targets,
                                          float *loss,
                                          const float epsilon,
                                          const int batch_size,
                                          const int num_classes)
{
  const int batch_idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(batch_idx<batch_size)
  {
    float sample_loss=0.0f;
    for(int c=0;c<num_classes;++c)
    {
      const int idx=batch_idx*num_classes+c;
      float pred=predictions[idx];
      // Clamp to avoid log(0)
      if(pred<epsilon)
      {
        pred=epsilon;
      }
      else if(pred>1.0f-epsilon)
      {
        pred=1.0f-epsilon;
      }
      const float target=targets[idx];
      if(target>0.0f)
      {
        sample_loss-=target*logf(pred);
      }
    }
    loss[batch_idx]=sample_loss;
  }
}

//------------------------------------------------------------------------------
// Cross Entropy Gradient Kernel
// grad_i = -target_i / max(epsilon, min(1-epsilon, pred_i)) / batch_size
//------------------------------------------------------------------------------
__global__ void cross_entropy_gradient_kernel(const float *predictions,
                                              const float *targets,
                                              float *gradient,
                                              const float epsilon,
                                              const float batch_size_inv,
                                              const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float pred=predictions[idx];
    // Clamp to avoid division by zero
    if(pred<epsilon)
    {
      pred=epsilon;
    }
    else if(pred>1.0f-epsilon)
    {
      pred=1.0f-epsilon;
    }
    const float target=targets[idx];
    gradient[idx]=-target/(pred*batch_size_inv);
  }
}

//------------------------------------------------------------------------------
// Cross Entropy with index targets
//------------------------------------------------------------------------------
__global__ void cross_entropy_loss_index_kernel(const float *predictions,
                                                const int *target_idx,
                                                float *loss,
                                                const float epsilon,
                                                const int num_classes)
{
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b>=gridDim.x*blockDim.x)
  {
    return;
  }
  const int cls=target_idx[b];
  if(cls<0||cls>=num_classes)
  {
    loss[b]=0.0f;
    return;
  }
  const int idx=b*num_classes+cls;
  const float pred=predictions[idx];
  const float clipped=fminf(1.0f-epsilon,fmaxf(epsilon,pred));
  loss[b]=-logf(clipped);
}

__global__ void cross_entropy_gradient_index_kernel(const float *predictions,
                                                    const int *target_idx,
                                                    float *gradient,
                                                    const float epsilon,
                                                    const float batch_size_inv,
                                                    const int num_classes,
                                                    const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }
  const int b=idx/num_classes;
  const int c=idx%num_classes;
  const int target=target_idx[b];
  if(target==c)
  {
    float pred=predictions[idx];
    const float clipped=fminf(1.0f-epsilon,fmaxf(epsilon,pred));
    gradient[idx]=-batch_size_inv/clipped;
  }
  else
  {
    gradient[idx]=0.0f;
  }
}

//------------------------------------------------------------------------------
// MSE Loss Reduction Kernel
// Computes sum of (pred - target)^2 via parallel reduction to a scalar.
// Output must be pre-zeroed; each block atomically adds its partial sum.
//------------------------------------------------------------------------------
__global__ void mse_loss_reduce_kernel(const float *predictions,
                                       const float *targets,
                                       float *loss,
                                       const int n)
{
  extern __shared__ float sdata[];

  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;

  if(idx<n)
  {
    const float diff=predictions[idx]-targets[idx];
    sdata[tid]=diff*diff;
  }
  else
  {
    sdata[tid]=0.0f;
  }
  __syncthreads();

  // Block-level parallel reduction in shared memory
  for(int s=blockDim.x/2; s>0; s>>=1)
  {
    if(tid<s)
    {
      sdata[tid]+=sdata[tid+s];
    }
    __syncthreads();
  }

  // Atomic add block partial sum to output scalar
  if(tid==0)
  {
    atomicAdd(loss,sdata[0]);
  }
}

//------------------------------------------------------------------------------
// MSE Gradient Kernel
// grad = 2 * (pred - target) / n
//------------------------------------------------------------------------------
__global__ void mse_gradient_kernel(const float *predictions,
                                    const float *targets,
                                    float *gradient,
                                    const float scale,
                                    const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    gradient[idx]=scale*(predictions[idx]-targets[idx]);
  }
}

//------------------------------------------------------------------------------
// Loss Function Launchers
//------------------------------------------------------------------------------

void launch_cross_entropy_loss(const float *predictions,
                               const float *targets,
                               float *loss,
                               const float epsilon,
                               const int batch_size,
                               const int num_classes,
                               cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(batch_size+block_size-1)/block_size;
  cross_entropy_loss_kernel<<<num_blocks,block_size,0,stream>>>(predictions,targets,loss,
                                                                epsilon,batch_size,num_classes);
}

void launch_cross_entropy_gradient(const float *predictions,
                                   const float *targets,
                                   float *gradient,
                                   const float epsilon,
                                   const int batch_size,
                                   const int n,
                                   cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  const float batch_size_inv=1.0f/static_cast<float>(batch_size);
  cross_entropy_gradient_kernel<<<num_blocks,block_size,0,stream>>>(predictions,targets,
                                                                     gradient,epsilon,
                                                                     batch_size_inv,n);
}

void launch_cross_entropy_loss_index(const float *predictions,
                                     const int *target_indices,
                                     float *loss,
                                     const float epsilon,
                                     const int batch_size,
                                     const int num_classes,
                                     cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(batch_size+block_size-1)/block_size;
  cross_entropy_loss_index_kernel<<<num_blocks,block_size,0,stream>>>(predictions,target_indices,loss,epsilon,num_classes);
}

void launch_cross_entropy_gradient_index(const float *predictions,
                                         const int *target_indices,
                                         float *gradient,
                                         const float epsilon,
                                         const int batch_size,
                                         const int num_classes,
                                         cudaStream_t stream)
{
  const int total=batch_size*num_classes;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  const float batch_size_inv=1.0f/static_cast<float>(batch_size);
  cross_entropy_gradient_index_kernel<<<num_blocks,block_size,0,stream>>>(predictions,target_indices,gradient,epsilon,batch_size_inv,num_classes,total);
}

void launch_mse_loss(const float *predictions,
                     const float *targets,
                     float *loss,
                     const int n,
                     cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  const size_t shared_mem_size=block_size*sizeof(float);
  mse_loss_reduce_kernel<<<num_blocks,block_size,shared_mem_size,stream>>>(predictions,
                                                                           targets,
                                                                           loss,
                                                                           n);
}

void launch_mse_gradient(const float *predictions,
                         const float *targets,
                         float *gradient,
                         const int n,
                         cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  const float scale=2.0f/static_cast<float>(n);
  mse_gradient_kernel<<<num_blocks,block_size,0,stream>>>(predictions,targets,gradient,scale,n);
}

//------------------------------------------------------------------------------
// Fused Adam Optimizer Kernel
// Combines all Adam operations into a single kernel for efficiency:
// 1. Apply weight decay: g = grad + weight_decay * param
// 2. Update first moment: m = beta1 * m + (1 - beta1) * g
// 3. Update second moment: v = beta2 * v + (1 - beta2) * g^2
// 4. Compute bias-corrected moments
// 5. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
//------------------------------------------------------------------------------
__global__ void fused_adam_kernel(float *param,
                                  const float *grad,
                                  float *m,
                                  float *v,
                                  const float lr,
                                  const float beta1,
                                  const float beta2,
                                  const float epsilon,
                                  const float weight_decay,
                                  const float bias_correction1,
                                  const float bias_correction2,
                                  const float grad_clip_val,
                                  const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    // Load values
    float p=param[idx];
    float g=grad[idx];
    float m_val=m[idx];
    float v_val=v[idx];

    // Sanitize gradient: replace NaN/Inf with 0, clamp to [-clip_val, clip_val]
    if(isnan(g)||isinf(g))
    {
      g=0.0f;
    }
    else if(g>grad_clip_val)
    {
      g=grad_clip_val;
    }
    else if(g<-grad_clip_val)
    {
      g=-grad_clip_val;
    }

    // Apply weight decay (L2 regularization)
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
    m_val=beta1*m_val+(1.0f-beta1)*g;

    // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
    v_val=beta2*v_val+(1.0f-beta2)*g*g;

    // Compute bias-corrected first moment: m_hat = m / bias_correction1
    const float m_hat=m_val/bias_correction1;

    // Compute bias-corrected second moment: v_hat = v / bias_correction2
    const float v_hat=v_val/bias_correction2;

    // Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    p=p-lr*m_hat/(sqrtf(v_hat)+epsilon);

    // Store updated values
    param[idx]=p;
    m[idx]=m_val;
    v[idx]=v_val;
  }
}

//------------------------------------------------------------------------------
// Fused Adam with Gradient Clipping Kernel
// Same as fused_adam but includes gradient clipping
//------------------------------------------------------------------------------
__global__ void fused_adam_clipped_kernel(float *param,
                                          const float *grad,
                                          float *m,
                                          float *v,
                                          const float lr,
                                          const float beta1,
                                          const float beta2,
                                          const float epsilon,
                                          const float weight_decay,
                                          const float bias_correction1,
                                          const float bias_correction2,
                                          const float grad_scale,
                                          const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    // Load values
    float p=param[idx];
    float g=grad[idx]*grad_scale;  // Apply gradient scaling (for clipping)
    float m_val=m[idx];
    float v_val=v[idx];

    // Apply weight decay (L2 regularization)
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }

    // Update biased first moment estimate
    m_val=beta1*m_val+(1.0f-beta1)*g;

    // Update biased second moment estimate
    v_val=beta2*v_val+(1.0f-beta2)*g*g;

    // Compute bias-corrected moments
    const float m_hat=m_val/bias_correction1;
    const float v_hat=v_val/bias_correction2;

    // Update parameter
    p=p-lr*m_hat/(sqrtf(v_hat)+epsilon);

    // Store updated values
    param[idx]=p;
    m[idx]=m_val;
    v[idx]=v_val;
  }
}

//------------------------------------------------------------------------------
// Fused SGD with Momentum Kernel
// Combines momentum update and parameter update
//------------------------------------------------------------------------------
__global__ void fused_sgd_momentum_kernel(float *param,
                                          const float *grad,
                                          float *velocity,
                                          const float lr,
                                          const float momentum,
                                          const float weight_decay,
                                          const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=param[idx];
    float g=grad[idx];
    float v=velocity[idx];

    // Apply weight decay
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }

    // Update velocity: v = momentum * v + grad
    v=momentum*v+g;

    // Update parameter: param = param - lr * v
    p=p-lr*v;

    // Store
    param[idx]=p;
    velocity[idx]=v;
  }
}

//------------------------------------------------------------------------------
// Optimizer Kernel Launchers
//------------------------------------------------------------------------------

void launch_fused_adam(float *param,
                       const float *grad,
                       float *m,
                       float *v,
                       const float lr,
                       const float beta1,
                       const float beta2,
                       const float epsilon,
                       const float weight_decay,
                       const float bias_correction1,
                       const float bias_correction2,
                       const int n,
                       cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_adam_kernel<<<num_blocks,block_size,0,stream>>>(param,grad,m,v,lr,beta1,beta2,
                                                        epsilon,weight_decay,
                                                        bias_correction1,bias_correction2,
                                                        1e30f,n);
}

void launch_fused_adam_clipped(float *param,
                               const float *grad,
                               float *m,
                               float *v,
                               const float lr,
                               const float beta1,
                               const float beta2,
                               const float epsilon,
                               const float weight_decay,
                               const float bias_correction1,
                               const float bias_correction2,
                               const float grad_scale,
                               const int n,
                               cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_adam_clipped_kernel<<<num_blocks,block_size,0,stream>>>(param,grad,m,v,lr,beta1,beta2,
                                                                 epsilon,weight_decay,
                                                                 bias_correction1,bias_correction2,
                                                                 grad_scale,n);
}

void launch_fused_sgd_momentum(float *param,
                               const float *grad,
                               float *velocity,
                               const float lr,
                               const float momentum,
                               const float weight_decay,
                               const int n,
                               cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_sgd_momentum_kernel<<<num_blocks,block_size,0,stream>>>(param,grad,velocity,
                                                                 lr,momentum,weight_decay,n);
}

//------------------------------------------------------------------------------
// RMSNorm Forward Kernel
// y[row][col] = x[row][col] / rms(x[row]) * gamma[col]
// rms(x) = sqrt(mean(x^2) + epsilon)
// One block per row, shared memory parallel reduction
//------------------------------------------------------------------------------
__global__ void rmsnorm_forward_kernel(const float *input,
                                        const float *gamma,
                                        float *output,
                                        float *rms_cache,
                                        const float epsilon,
                                        const int rows,
                                        const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const float *x=input+row*dim;
  float *y=output+row*dim;

  extern __shared__ float sdata[];

  // Phase 1: Compute sum(x^2) using grid-stride loop
  float local_sum=0.0f;
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    const float val=x[col];
    local_sum+=val*val;
  }
  sdata[threadIdx.x]=local_sum;
  __syncthreads();

  // Parallel reduction in shared memory
  for(int s=blockDim.x/2;s>0;s>>=1)
  {
    if(threadIdx.x<s)
    {
      sdata[threadIdx.x]+=sdata[threadIdx.x+s];
    }
    __syncthreads();
  }

  // Compute RMS value
  const float rms=sqrtf(sdata[0]/static_cast<float>(dim)+epsilon);
  if(threadIdx.x==0)
  {
    rms_cache[row]=rms;
  }

  const float rstd=1.0f/rms;

  // Phase 2: Normalize and scale by gamma
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    y[col]=x[col]*rstd*gamma[col];
  }
}

//------------------------------------------------------------------------------
// RMSNorm Backward Kernel
// Computes grad_input and accumulates grad_gamma via atomicAdd.
// grad_gamma must be pre-zeroed before this kernel is launched.
//------------------------------------------------------------------------------
__global__ void rmsnorm_backward_kernel(const float *grad_output,
                                         const float *input,
                                         const float *gamma,
                                         const float *rms_cache,
                                         float *grad_input,
                                         float *grad_gamma,
                                         const float epsilon,
                                         const int rows,
                                         const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const float *dy=grad_output+row*dim;
  const float *x=input+row*dim;
  float *dx=grad_input+row*dim;

  const float rstd=1.0f/rms_cache[row];

  extern __shared__ float sdata[];

  // Phase 1: Compute sum(dy * gamma * x_hat) where x_hat = x * rstd
  float local_sum=0.0f;
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    const float x_hat=x[col]*rstd;
    local_sum+=dy[col]*gamma[col]*x_hat;
  }
  sdata[threadIdx.x]=local_sum;
  __syncthreads();

  // Parallel reduction
  for(int s=blockDim.x/2;s>0;s>>=1)
  {
    if(threadIdx.x<s)
    {
      sdata[threadIdx.x]+=sdata[threadIdx.x+s];
    }
    __syncthreads();
  }

  const float sum_term=sdata[0]/static_cast<float>(dim);

  // Phase 2: Compute grad_input and accumulate grad_gamma
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    const float x_hat=x[col]*rstd;
    dx[col]=rstd*(dy[col]*gamma[col]-x_hat*sum_term);
    atomicAdd(&grad_gamma[col],dy[col]*x_hat);
  }
}

//------------------------------------------------------------------------------
// LayerNorm Forward Kernel (Block-per-Row)
// y[row][col] = (x[row][col] - mean) / sqrt(var + eps) * gamma[col] + beta[col]
// One block per row, 256 threads, shared memory parallel reduction.
// Mirrors the RMSNorm architecture for maximum memory bandwidth utilization.
//------------------------------------------------------------------------------
__global__ void layernorm_forward_kernel(const float *input,
                                             const float *gamma,
                                             const float *beta,
                                             float *output,
                                             float *mean_cache,
                                             float *rstd_cache,
                                             const float epsilon,
                                             const int rows,
                                             const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const float *x=input+row*dim;
  float *y=output+row*dim;

  extern __shared__ float sdata[];
  float *s_sum=sdata;
  float *s_sum_sq=sdata+blockDim.x;

  // Phase 1: Compute sum(x) and sum(x^2) using grid-stride loop
  float local_sum=0.0f;
  float local_sum_sq=0.0f;
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    const float val=x[col];
    local_sum+=val;
    local_sum_sq+=val*val;
  }
  s_sum[threadIdx.x]=local_sum;
  s_sum_sq[threadIdx.x]=local_sum_sq;
  __syncthreads();

  // Parallel reduction in shared memory
  for(int s=blockDim.x/2;s>0;s>>=1)
  {
    if(threadIdx.x<s)
    {
      s_sum[threadIdx.x]+=s_sum[threadIdx.x+s];
      s_sum_sq[threadIdx.x]+=s_sum_sq[threadIdx.x+s];
    }
    __syncthreads();
  }

  // Compute mean and rstd
  __shared__ float s_mean;
  __shared__ float s_rstd;
  if(threadIdx.x==0)
  {
    const float dim_f=static_cast<float>(dim);
    s_mean=s_sum[0]/dim_f;
    const float variance=s_sum_sq[0]/dim_f-s_mean*s_mean;
    s_rstd=rsqrtf(variance+epsilon);
    mean_cache[row]=s_mean;
    rstd_cache[row]=s_rstd;
  }
  __syncthreads();

  const float mean=s_mean;
  const float rstd=s_rstd;

  // Phase 2: Normalize, scale, and shift
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    const float x_hat=(x[col]-mean)*rstd;
    y[col]=x_hat*gamma[col]+beta[col];
  }
}

//------------------------------------------------------------------------------
// LayerNorm Backward Kernel
// Computes grad_input and accumulates grad_gamma/grad_beta via atomicAdd.
// grad_gamma and grad_beta must be pre-zeroed before this kernel is launched.
//------------------------------------------------------------------------------
__global__ void layernorm_backward_kernel(const float *grad_output,
                                           const float *input,
                                           const float *gamma,
                                           const float *mean_cache,
                                           const float *rstd_cache,
                                           float *grad_input,
                                           float *grad_gamma,
                                           float *grad_beta,
                                           const int rows,
                                           const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const float *dy=grad_output+row*dim;
  const float *x=input+row*dim;
  float *dx=grad_input+row*dim;

  const float mean=mean_cache[row];
  const float rstd=rstd_cache[row];

  extern __shared__ float sdata[];
  float *s_sum1=sdata;
  float *s_sum2=sdata+blockDim.x;

  // Phase 1: Compute S1=sum(dy*gamma) and S2=sum(dy*gamma*x_hat)
  float local_s1=0.0f;
  float local_s2=0.0f;
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    const float x_hat=(x[col]-mean)*rstd;
    const float dy_gamma=dy[col]*gamma[col];
    local_s1+=dy_gamma;
    local_s2+=dy_gamma*x_hat;
  }
  s_sum1[threadIdx.x]=local_s1;
  s_sum2[threadIdx.x]=local_s2;
  __syncthreads();

  // Parallel reduction
  for(int s=blockDim.x/2;s>0;s>>=1)
  {
    if(threadIdx.x<s)
    {
      s_sum1[threadIdx.x]+=s_sum1[threadIdx.x+s];
      s_sum2[threadIdx.x]+=s_sum2[threadIdx.x+s];
    }
    __syncthreads();
  }

  const float dim_f=static_cast<float>(dim);
  const float s1=s_sum1[0]/dim_f;
  const float s2=s_sum2[0]/dim_f;

  // Phase 2: Compute grad_input, accumulate grad_gamma and grad_beta
  for(int col=threadIdx.x;col<dim;col+=blockDim.x)
  {
    const float x_hat=(x[col]-mean)*rstd;
    dx[col]=rstd*(dy[col]*gamma[col]-s1-x_hat*s2);
    atomicAdd(&grad_gamma[col],dy[col]*x_hat);
    atomicAdd(&grad_beta[col],dy[col]);
  }
}

//------------------------------------------------------------------------------
// RMSNorm / LayerNorm Kernel Launchers
//------------------------------------------------------------------------------

void launch_rmsnorm_forward(const float *input,
                            const float *gamma,
                            float *output,
                            float *rms_cache,
                            const float epsilon,
                            const int rows,
                            const int dim,
                            cudaStream_t stream)
{
  const int block_size=256;
  const size_t shared_mem_size=block_size*sizeof(float);
  rmsnorm_forward_kernel<<<rows,block_size,shared_mem_size,stream>>>(input,
                                                                      gamma,
                                                                      output,
                                                                      rms_cache,
                                                                      epsilon,
                                                                      rows,
                                                                      dim);
}

void launch_rmsnorm_backward(const float *grad_output,
                              const float *input,
                              const float *gamma,
                              const float *rms_cache,
                              float *grad_input,
                              float *grad_gamma,
                              const float epsilon,
                              const int rows,
                              const int dim,
                              cudaStream_t stream)
{
  const int block_size=256;
  const size_t shared_mem_size=block_size*sizeof(float);
  rmsnorm_backward_kernel<<<rows,block_size,shared_mem_size,stream>>>(grad_output,
                                                                       input,
                                                                       gamma,
                                                                       rms_cache,
                                                                       grad_input,
                                                                       grad_gamma,
                                                                       epsilon,
                                                                       rows,
                                                                       dim);
}

void launch_layernorm_forward(const float *input,
                              const float *gamma,
                              const float *beta,
                              float *output,
                              float *mean_cache,
                              float *rstd_cache,
                              const float epsilon,
                              const int rows,
                              const int dim,
                              cudaStream_t stream)
{
  const int block_size=256;
  const size_t shared_mem_size=2*block_size*sizeof(float);
  layernorm_forward_kernel<<<rows,block_size,shared_mem_size,stream>>>(
    input,gamma,beta,output,mean_cache,rstd_cache,epsilon,rows,dim);
}

void launch_layernorm_backward(const float *grad_output,
                                const float *input,
                                const float *gamma,
                                const float *mean_cache,
                                const float *rstd_cache,
                                float *grad_input,
                                float *grad_gamma,
                                float *grad_beta,
                                const int rows,
                                const int dim,
                                cudaStream_t stream)
{
  const int block_size=256;
  const size_t shared_mem_size=2*block_size*sizeof(float);
  layernorm_backward_kernel<<<rows,block_size,shared_mem_size,stream>>>(grad_output,
                                                                         input,
                                                                         gamma,
                                                                         mean_cache,
                                                                         rstd_cache,
                                                                         grad_input,
                                                                         grad_gamma,
                                                                         grad_beta,
                                                                         rows,
                                                                         dim);
}

//------------------------------------------------------------------------------
// Transpose 0213 Kernel
// Swaps dims 1 and 2 of a logical 4D tensor [batch, dim0, dim1, dim2]
// -> [batch, dim1, dim0, dim2]
// Element-parallel: one thread per element.
//------------------------------------------------------------------------------
__global__ void transpose_0213_kernel(const float *input,
                                       float *output,
                                       const int dim0,
                                       const int dim1,
                                       const int dim2,
                                       const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    // Decompose flat index into [b, d0, d1, d2]
    const int stride_d0=dim1*dim2;
    const int stride_batch=dim0*stride_d0;
    const int b=idx/stride_batch;
    const int rem=idx%stride_batch;
    const int d0=rem/stride_d0;
    const int rem2=rem%stride_d0;
    const int d1=rem2/dim2;
    const int d2=rem2%dim2;

    // Output index in [b, d1, d0, d2]
    const int out_stride_d1=dim0*dim2;
    const int out_idx=b*dim1*out_stride_d1+
                      d1*out_stride_d1+
                      d0*dim2+
                      d2;
    output[out_idx]=input[idx];
  }
}

//------------------------------------------------------------------------------
// Causal Mask Fill Kernel
// Sets upper triangle (j > i) to -1e9 for each [seq_len, seq_len] matrix.
// Layout: [num_matrices, seq_len, seq_len]
//------------------------------------------------------------------------------
__global__ void causal_mask_fill_kernel(float *scores,
                                        const int seq_len,
                                        const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(col>row)
    {
      scores[idx]=-1e9f;
    }
  }
}

//------------------------------------------------------------------------------
// Causal Mask Fill with Offset Kernel (for KV-cache)
// Sets positions where col > (row + offset) to -1e9 for rectangular matrices.
// Layout: [num_matrices, query_len, key_len]
// offset = previous cached sequence length
//------------------------------------------------------------------------------
__global__ void causal_mask_fill_offset_kernel(float *scores,
                                               const int query_len,
                                               const int key_len,
                                               const int offset,
                                               const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=query_len*key_len;
    const int pos=idx%matrix_size;
    const int row=pos/key_len;
    const int col=pos%key_len;
    // Query at row corresponds to sequence position (offset + row)
    // Can only attend to keys at positions <= (offset + row)
    if(col>(offset+row))
    {
      scores[idx]=-1e9f;
    }
  }
}

//------------------------------------------------------------------------------
// Attention Softmax Forward Kernel
// Row-wise softmax with max subtraction for numerical stability.
// One block per row: shared-memory reduction for max, sum(exp), then normalize.
// Layout: [num_rows, row_len]
//------------------------------------------------------------------------------
__global__ void attention_softmax_kernel(const float *input,
                                          float *output,
                                          const int num_rows,
                                          const int row_len)
{
  const int row=blockIdx.x;
  if(row>=num_rows)
  {
    return;
  }

  const float *x=input+row*row_len;
  float *y=output+row*row_len;

  extern __shared__ float sdata[];

  // Phase 1: Find max value
  float local_max=-1e30f;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    const float val=x[col];
    if(val>local_max)
    {
      local_max=val;
    }
  }
  sdata[threadIdx.x]=local_max;
  __syncthreads();

  for(int s=blockDim.x/2;s>0;s>>=1)
  {
    if(threadIdx.x<s)
    {
      if(sdata[threadIdx.x+s]>sdata[threadIdx.x])
      {
        sdata[threadIdx.x]=sdata[threadIdx.x+s];
      }
    }
    __syncthreads();
  }
  const float row_max=sdata[0];

  // Phase 2: Compute sum(exp(x - max))
  float local_sum=0.0f;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    local_sum+=expf(x[col]-row_max);
  }
  sdata[threadIdx.x]=local_sum;
  __syncthreads();

  for(int s=blockDim.x/2;s>0;s>>=1)
  {
    if(threadIdx.x<s)
    {
      sdata[threadIdx.x]+=sdata[threadIdx.x+s];
    }
    __syncthreads();
  }
  const float row_sum=sdata[0];

  // Phase 3: Normalize
  const float inv_sum=1.0f/row_sum;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    y[col]=expf(x[col]-row_max)*inv_sum;
  }
}

//------------------------------------------------------------------------------
// Attention Softmax Backward Kernel
// For each row: grad_input[i] = output[i] * (grad_output[i] - dot(grad_output, output))
// One block per row: shared-memory reduction for dot product.
//------------------------------------------------------------------------------
__global__ void attention_softmax_backward_kernel(const float *grad_output,
                                                    const float *output,
                                                    float *grad_input,
                                                    const int num_rows,
                                                    const int row_len)
{
  const int row=blockIdx.x;
  if(row>=num_rows)
  {
    return;
  }

  const float *dy=grad_output+row*row_len;
  const float *y=output+row*row_len;
  float *dx=grad_input+row*row_len;

  extern __shared__ float sdata[];

  // Phase 1: Compute dot(grad_output, output)
  float local_dot=0.0f;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    local_dot+=dy[col]*y[col];
  }
  sdata[threadIdx.x]=local_dot;
  __syncthreads();

  for(int s=blockDim.x/2;s>0;s>>=1)
  {
    if(threadIdx.x<s)
    {
      sdata[threadIdx.x]+=sdata[threadIdx.x+s];
    }
    __syncthreads();
  }
  const float dot_val=sdata[0];

  // Phase 2: Compute gradient
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    dx[col]=y[col]*(dy[col]-dot_val);
  }
}

//------------------------------------------------------------------------------
// Causal Mask Gradient Kernel
// Zeros upper triangle (j > i) of gradient scores.
// Layout: [num_matrices, seq_len, seq_len]
//------------------------------------------------------------------------------
__global__ void causal_mask_grad_kernel(float *grad_scores,
                                         const int seq_len,
                                         const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(col>row)
    {
      grad_scores[idx]=0.0f;
    }
  }
}

//------------------------------------------------------------------------------
// Multi-Head Attention Kernel Launchers
//------------------------------------------------------------------------------

void launch_transpose_0213(const float *input,
                           float *output,
                           const int batch,
                           const int dim0,
                           const int dim1,
                           const int dim2,
                           cudaStream_t stream)
{
  const int total=batch*dim0*dim1*dim2;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  transpose_0213_kernel<<<num_blocks,block_size,0,stream>>>(input,output,
                                                             dim0,dim1,dim2,
                                                             total);
}

void launch_causal_mask_fill(float *scores,
                             const int num_matrices,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int total=num_matrices*seq_len*seq_len;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_fill_kernel<<<num_blocks,block_size,0,stream>>>(scores,seq_len,
                                                               total);
}

void launch_causal_mask_fill_offset(float *scores,
                                    const int num_matrices,
                                    const int query_len,
                                    const int key_len,
                                    const int offset,
                                    cudaStream_t stream)
{
  const int total=num_matrices*query_len*key_len;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_fill_offset_kernel<<<num_blocks,block_size,0,stream>>>(scores,
                                                                      query_len,
                                                                      key_len,
                                                                      offset,
                                                                      total);
}

void launch_attention_softmax(const float *input,
                              float *output,
                              const int num_rows,
                              const int row_len,
                              cudaStream_t stream)
{
  const int block_size=256;
  const size_t shared_mem_size=block_size*sizeof(float);
  attention_softmax_kernel<<<num_rows,block_size,shared_mem_size,stream>>>(
    input,output,num_rows,row_len);
}

void launch_attention_softmax_backward(const float *grad_output,
                                       const float *output,
                                       float *grad_input,
                                       const int num_rows,
                                       const int row_len,
                                       cudaStream_t stream)
{
  const int block_size=256;
  const size_t shared_mem_size=block_size*sizeof(float);
  attention_softmax_backward_kernel<<<num_rows,block_size,shared_mem_size,stream>>>(
    grad_output,output,grad_input,num_rows,row_len);
}

void launch_causal_mask_grad(float *grad_scores,
                             const int num_matrices,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int total=num_matrices*seq_len*seq_len;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_grad_kernel<<<num_blocks,block_size,0,stream>>>(grad_scores,
                                                               seq_len,total);
}

//------------------------------------------------------------------------------
// RoPE Forward Kernel
// Applies rotary position embeddings in-place.
// data: [batch_heads, seq_len, head_dim], head_dim must be even.
// One thread per (batch_head, position, pair_index).
//------------------------------------------------------------------------------
__global__ void rope_forward_kernel(float *data,
                                    const int seq_len,
                                    const int head_dim,
                                    const float base,
                                    const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_dim=head_dim/2;
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=row%seq_len;

    const float freq_exp=2.0f*static_cast<float>(pair)/static_cast<float>(head_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int base_idx=row*head_dim+pair*2;
    const float x0=data[base_idx];
    const float x1=data[base_idx+1];
    data[base_idx]=x0*cos_t-x1*sin_t;
    data[base_idx+1]=x0*sin_t+x1*cos_t;
  }
}

//------------------------------------------------------------------------------
// RoPE Forward with Position Offset Kernel (for KV-cache)
// Same as rope_forward_kernel but adds pos_offset to position calculation.
// Used when processing new tokens that continue from cached positions.
//------------------------------------------------------------------------------
__global__ void rope_forward_offset_kernel(float *data,
                                           const int seq_len,
                                           const int head_dim,
                                           const float base,
                                           const int pos_offset,
                                           const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_dim=head_dim/2;
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=(row%seq_len)+pos_offset;

    const float freq_exp=2.0f*static_cast<float>(pair)/static_cast<float>(head_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int base_idx=row*head_dim+pair*2;
    const float x0=data[base_idx];
    const float x1=data[base_idx+1];
    data[base_idx]=x0*cos_t-x1*sin_t;
    data[base_idx+1]=x0*sin_t+x1*cos_t;
  }
}

//------------------------------------------------------------------------------
// RoPE Backward Kernel
// Applies inverse rotation in-place (swap sin sign).
//------------------------------------------------------------------------------
__global__ void rope_backward_kernel(float *data,
                                     const int seq_len,
                                     const int head_dim,
                                     const float base,
                                     const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_dim=head_dim/2;
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=row%seq_len;

    const float freq_exp=2.0f*static_cast<float>(pair)/static_cast<float>(head_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int base_idx=row*head_dim+pair*2;
    const float g0=data[base_idx];
    const float g1=data[base_idx+1];
    data[base_idx]=g0*cos_t+g1*sin_t;
    data[base_idx+1]=-g0*sin_t+g1*cos_t;
  }
}

//------------------------------------------------------------------------------
// GQA Repeat KV Kernel
// input: [batch * num_kv_heads, seq_len, head_dim]
// output: [batch * num_heads, seq_len, head_dim]
// One thread per output element.
//------------------------------------------------------------------------------
__global__ void gqa_repeat_kv_kernel(const float *input,
                                     float *output,
                                     const int num_kv_heads,
                                     const int repeat_factor,
                                     const int seq_len,
                                     const int head_dim,
                                     const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int num_heads=num_kv_heads*repeat_factor;
    const int head_stride=seq_len*head_dim;
    const int batch_stride=num_heads*head_stride;

    const int b=idx/batch_stride;
    const int rem=idx%batch_stride;
    const int h=rem/head_stride;
    const int rem2=rem%head_stride;

    const int kv_h=h/repeat_factor;
    const int input_batch_stride=num_kv_heads*head_stride;
    const int in_idx=b*input_batch_stride+kv_h*head_stride+rem2;

    output[idx]=input[in_idx];
  }
}

//------------------------------------------------------------------------------
// GQA Reduce KV Kernel
// input: [batch * num_heads, seq_len, head_dim]
// output: [batch * num_kv_heads, seq_len, head_dim]
// One thread per output element, loops over repeat_factor.
//------------------------------------------------------------------------------
__global__ void gqa_reduce_kv_kernel(const float *input,
                                     float *output,
                                     const int num_kv_heads,
                                     const int repeat_factor,
                                     const int seq_len,
                                     const int head_dim,
                                     const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int head_stride=seq_len*head_dim;
    const int out_batch_stride=num_kv_heads*head_stride;

    const int b=idx/out_batch_stride;
    const int rem=idx%out_batch_stride;
    const int kv_h=rem/head_stride;
    const int rem2=rem%head_stride;

    const int num_heads=num_kv_heads*repeat_factor;
    const int in_batch_stride=num_heads*head_stride;

    float sum=0.0f;
    for(int r=0;r<repeat_factor;++r)
    {
      const int h=kv_h*repeat_factor+r;
      const int in_idx=b*in_batch_stride+h*head_stride+rem2;
      sum+=input[in_idx];
    }
    output[idx]=sum;
  }
}

//------------------------------------------------------------------------------
// RoPE / GQA Kernel Launchers
//------------------------------------------------------------------------------

void launch_rope_forward(float *data,
                         const int batch_heads,
                         const int seq_len,
                         const int head_dim,
                         const float base,
                         cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(head_dim/2);
  const int block_size=256;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_kernel<<<num_blocks,block_size,0,stream>>>(data,seq_len,
                                                           head_dim,base,
                                                           total_pairs);
}

void launch_rope_forward_offset(float *data,
                                const int batch_heads,
                                const int seq_len,
                                const int head_dim,
                                const float base,
                                const int pos_offset,
                                cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(head_dim/2);
  const int block_size=256;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_offset_kernel<<<num_blocks,block_size,0,stream>>>(data,seq_len,
                                                                  head_dim,base,
                                                                  pos_offset,
                                                                  total_pairs);
}

void launch_rope_backward(float *data,
                          const int batch_heads,
                          const int seq_len,
                          const int head_dim,
                          const float base,
                          cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(head_dim/2);
  const int block_size=256;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_backward_kernel<<<num_blocks,block_size,0,stream>>>(data,seq_len,
                                                            head_dim,base,
                                                            total_pairs);
}

void launch_gqa_repeat_kv(const float *input,
                          float *output,
                          const int batch,
                          const int num_kv_heads,
                          const int repeat_factor,
                          const int seq_len,
                          const int head_dim,
                          cudaStream_t stream)
{
  const int total=batch*num_kv_heads*repeat_factor*seq_len*head_dim;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  gqa_repeat_kv_kernel<<<num_blocks,block_size,0,stream>>>(input,output,
                                                            num_kv_heads,
                                                            repeat_factor,
                                                            seq_len,head_dim,
                                                            total);
}

void launch_gqa_reduce_kv(const float *input,
                          float *output,
                          const int batch,
                          const int num_kv_heads,
                          const int repeat_factor,
                          const int seq_len,
                          const int head_dim,
                          cudaStream_t stream)
{
  const int total=batch*num_kv_heads*seq_len*head_dim;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  gqa_reduce_kv_kernel<<<num_blocks,block_size,0,stream>>>(input,output,
                                                            num_kv_heads,
                                                            repeat_factor,
                                                            seq_len,head_dim,
                                                            total);
}

//------------------------------------------------------------------------------
// KV-Cache kernels
//------------------------------------------------------------------------------

/**
 * Kernel to append new K/V data to the cache.
 * new_kv: [batch, new_len, num_kv_heads, head_dim] - contiguous
 * cache: [batch, max_seq_len, num_kv_heads, head_dim] - contiguous
 * Copies new_kv into cache starting at position cache_pos.
 */
__global__ void kv_cache_append_kernel(const float *new_kv,
                                       float *cache,
                                       const int new_len,
                                       const int cache_pos,
                                       const int max_seq_len,
                                       const int num_kv_heads,
                                       const int head_dim,
                                       const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }

  // Decompose linear index into [batch, new_pos, kv_head, d]
  const int kv_size=num_kv_heads*head_dim;
  const int batch_stride=new_len*kv_size;
  const int b=idx/batch_stride;
  const int rem=idx%batch_stride;
  const int new_pos=rem/kv_size;
  const int kv_idx=rem%kv_size;

  // Compute destination position in cache
  const int cache_batch_stride=max_seq_len*kv_size;
  const int cache_dst=b*cache_batch_stride+(cache_pos+new_pos)*kv_size+kv_idx;

  cache[cache_dst]=new_kv[idx];
}

void launch_kv_cache_append(const float *new_kv,
                            float *cache,
                            const int batch,
                            const int new_len,
                            const int cache_pos,
                            const int max_seq_len,
                            const int num_kv_heads,
                            const int head_dim,
                            cudaStream_t stream)
{
  const int total=batch*new_len*num_kv_heads*head_dim;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  kv_cache_append_kernel<<<num_blocks,block_size,0,stream>>>(new_kv,cache,
                                                              new_len,cache_pos,
                                                              max_seq_len,
                                                              num_kv_heads,
                                                              head_dim,total);
}

//------------------------------------------------------------------------------
// KV-Cache Append Kernel (transposed layout)
// new_kv: [batch_kv_heads, new_len, head_dim]
// cache:  [batch_kv_heads, max_seq_len, head_dim]
//------------------------------------------------------------------------------
__global__ void kv_cache_append_transposed_kernel(const float *new_kv,
                                                  float *cache,
                                                  const int new_len,
                                                  const int cache_pos,
                                                  const int max_seq_len,
                                                  const int head_dim,
                                                  const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }

  // Decompose linear index into [bkv, new_pos, d]
  const int row_size=new_len*head_dim;
  const int bkv=idx/row_size;
  const int rem=idx%row_size;
  const int new_pos=rem/head_dim;
  const int d=rem%head_dim;

  // Compute destination position in cache
  const int cache_row_size=max_seq_len*head_dim;
  const int cache_dst=bkv*cache_row_size+(cache_pos+new_pos)*head_dim+d;

  cache[cache_dst]=new_kv[idx];
}

void launch_kv_cache_append_transposed(const float *new_kv,
                                       float *cache,
                                       const int batch_kv_heads,
                                       const int new_len,
                                       const int cache_pos,
                                       const int max_seq_len,
                                       const int head_dim,
                                       cudaStream_t stream)
{
  const int total=batch_kv_heads*new_len*head_dim;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  kv_cache_append_transposed_kernel<<<num_blocks,block_size,0,stream>>>(
    new_kv,cache,new_len,cache_pos,max_seq_len,head_dim,total);
}

//------------------------------------------------------------------------------
// Gated Activation device helpers
//------------------------------------------------------------------------------
__device__ float gated_apply_op(float x,int op)
{
  if(op==0)  // Swish: x * sigmoid(x)
  {
    return x/(1.0f+expf(-x));
  }
  if(op==1)  // GELU approximation
  {
    const float sqrt_2_over_pi=0.7978845608f;
    const float coeff=0.044715f;
    const float inner=sqrt_2_over_pi*(x+coeff*x*x*x);
    return 0.5f*x*(1.0f+tanhf(inner));
  }
  if(op==2)  // ReLU
  {
    if(x>0.0f)
    {
      return x;
    }
    return 0.0f;
  }
  if(op==3)  // Sigmoid
  {
    return 1.0f/(1.0f+expf(-x));
  }
  // op==4: Linear (identity)
  return x;
}

__device__ float gated_apply_op_derivative(float x,float activated,int op)
{
  if(op==0)  // Swish derivative: sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
  {              //                 = activated/x + activated*(1-activated/x) when x!=0
    const float sig=1.0f/(1.0f+expf(-x));
    return sig+x*sig*(1.0f-sig);
  }
  if(op==1)  // GELU derivative
  {
    const float sqrt_2_over_pi=0.7978845608f;
    const float coeff=0.044715f;
    const float inner=sqrt_2_over_pi*(x+coeff*x*x*x);
    const float tanh_val=tanhf(inner);
    const float sech2=1.0f-tanh_val*tanh_val;
    const float d_inner=sqrt_2_over_pi*(1.0f+3.0f*coeff*x*x);
    return 0.5f*(1.0f+tanh_val)+0.5f*x*sech2*d_inner;
  }
  if(op==2)  // ReLU derivative
  {
    (void)activated;
    if(x>0.0f)
    {
      return 1.0f;
    }
    return 0.0f;
  }
  if(op==3)  // Sigmoid derivative: sigmoid(x)*(1-sigmoid(x))
  {
    return activated*(1.0f-activated);
  }
  // op==4: Linear derivative
  (void)activated;
  return 1.0f;
}

//------------------------------------------------------------------------------
// Gated Activation Forward Kernel
// output[i] = apply_op(gate_input[i], op) * up_input[i]
//------------------------------------------------------------------------------
__global__ void gated_activation_forward_kernel(const float *gate_input,
                                                const float *up_input,
                                                float *output,
                                                const int op,
                                                const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float activated=gated_apply_op(gate_input[idx],op);
    output[idx]=activated*up_input[idx];
  }
}

//------------------------------------------------------------------------------
// Gated Activation Backward Kernel
// grad_gate[i] = grad_output[i] * up[i] * d_activate(gate[i])
// grad_up[i]   = grad_output[i] * activate(gate[i])
//------------------------------------------------------------------------------
__global__ void gated_activation_backward_kernel(const float *grad_output,
                                                 const float *cached_gate_input,
                                                 const float *cached_up_input,
                                                 float *grad_gate,
                                                 float *grad_up,
                                                 const int op,
                                                 const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float g=cached_gate_input[idx];
    const float u=cached_up_input[idx];
    const float go=grad_output[idx];
    const float activated=gated_apply_op(g,op);
    const float d_activated=gated_apply_op_derivative(g,activated,op);
    grad_gate[idx]=go*u*d_activated;
    grad_up[idx]=go*activated;
  }
}

//------------------------------------------------------------------------------
// Gated Activation Launchers
//------------------------------------------------------------------------------
void launch_gated_activation_forward(const float *gate_input,
                                     const float *up_input,
                                     float *output,
                                     const int op,
                                     const int n,
                                     cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  gated_activation_forward_kernel<<<num_blocks,block_size,0,stream>>>(
    gate_input,up_input,output,op,n);
}

void launch_gated_activation_backward(const float *grad_output,
                                      const float *cached_gate_input,
                                      const float *cached_up_input,
                                      float *grad_gate,
                                      float *grad_up,
                                      const int op,
                                      const int n,
                                      cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  gated_activation_backward_kernel<<<num_blocks,block_size,0,stream>>>(
    grad_output,cached_gate_input,cached_up_input,grad_gate,grad_up,op,n);
}

//==============================================================================
// Embedding kernels
//==============================================================================

//------------------------------------------------------------------------------
// Embedding Lookup Kernel - Vectorized 2D Grid (uint32 token IDs)
// Grid: (num_tokens, ceil(dim/THREADS_PER_BLOCK/4))
// Each thread loads a float4 (4 elements) - no div/mod needed
//------------------------------------------------------------------------------
__global__ void embedding_lookup_kernel(const float *table,
                                        const unsigned int *token_ids,
                                        float *output,
                                        const int num_tokens,
                                        const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x*4+threadIdx.x*4;

  if(token_idx>=num_tokens||d>=dim)
  {
    return;
  }

  const unsigned int token_id=token_ids[token_idx];
  const float *src=table+token_id*dim+d;
  float *dst=output+token_idx*dim+d;

  if(d+4<=dim)
  {
    const float4 val=*reinterpret_cast<const float4 *>(src);
    *reinterpret_cast<float4 *>(dst)=val;
  }
  else
  {
    for(int i=0;i<4&&d+i<dim;++i)
    {
      dst[i]=src[i];
    }
  }
}

//------------------------------------------------------------------------------
// Embedding Lookup Kernel - Vectorized 2D Grid (float-encoded token IDs)
//------------------------------------------------------------------------------
__global__ void embedding_lookup_float_kernel(const float *table,
                                              const float *float_ids,
                                              float *output,
                                              const int num_tokens,
                                              const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x*4+threadIdx.x*4;

  if(token_idx>=num_tokens||d>=dim)
  {
    return;
  }

  const unsigned int token_id=static_cast<unsigned int>(float_ids[token_idx]);
  const float *src=table+token_id*dim+d;
  float *dst=output+token_idx*dim+d;

  if(d+4<=dim)
  {
    const float4 val=*reinterpret_cast<const float4 *>(src);
    *reinterpret_cast<float4 *>(dst)=val;
  }
  else
  {
    for(int i=0;i<4&&d+i<dim;++i)
    {
      dst[i]=src[i];
    }
  }
}

//------------------------------------------------------------------------------
// Float-to-uint conversion kernel (runs on GPU, eliminates host roundtrip)
//------------------------------------------------------------------------------
__global__ void float_to_uint_kernel(const float *float_ids,
                                     unsigned int *uint_ids,
                                     const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    uint_ids[idx]=static_cast<unsigned int>(float_ids[idx]);
  }
}

//------------------------------------------------------------------------------
// Embedding Backward Kernel - Vectorized 2D Grid (scatter-add gradients)
// Uses float4 vectorized loads for better bandwidth
//------------------------------------------------------------------------------
__global__ void embedding_backward_kernel(const float *grad_output,
                                          const unsigned int *token_ids,
                                          float *grad_table,
                                          const int num_tokens,
                                          const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x*4+threadIdx.x*4;

  if(token_idx>=num_tokens||d>=dim)
  {
    return;
  }

  const unsigned int token_id=token_ids[token_idx];
  const float *src=grad_output+token_idx*dim+d;
  float *dst=grad_table+token_id*dim+d;

  if(d+4<=dim)
  {
    const float4 val=*reinterpret_cast<const float4 *>(src);
    atomicAdd(&dst[0],val.x);
    atomicAdd(&dst[1],val.y);
    atomicAdd(&dst[2],val.z);
    atomicAdd(&dst[3],val.w);
  }
  else
  {
    for(int i=0;i<4&&d+i<dim;++i)
    {
      atomicAdd(&dst[i],src[i]);
    }
  }
}

//------------------------------------------------------------------------------
// Embedding Launchers
//------------------------------------------------------------------------------
void launch_embedding_lookup(const float *table,
                             const unsigned int *token_ids,
                             float *output,
                             const int num_tokens,
                             const int dim,
                             cudaStream_t stream)
{
  const int block_size=256;
  const int y_blocks=(dim+block_size*4-1)/(block_size*4);
  dim3 grid(num_tokens,y_blocks);
  embedding_lookup_kernel<<<grid,block_size,0,stream>>>(
    table,token_ids,output,num_tokens,dim);
}

void launch_embedding_lookup_float(const float *table,
                                   const float *float_ids,
                                   float *output,
                                   const int num_tokens,
                                   const int dim,
                                   cudaStream_t stream)
{
  const int block_size=256;
  const int y_blocks=(dim+block_size*4-1)/(block_size*4);
  dim3 grid(num_tokens,y_blocks);
  embedding_lookup_float_kernel<<<grid,block_size,0,stream>>>(
    table,float_ids,output,num_tokens,dim);
}

void launch_float_to_uint(const float *float_ids,
                           unsigned int *uint_ids,
                           const int n,
                           cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  float_to_uint_kernel<<<num_blocks,block_size,0,stream>>>(
    float_ids,uint_ids,n);
}

void launch_embedding_backward(const float *grad_output,
                               const unsigned int *token_ids,
                               float *grad_table,
                               const int num_tokens,
                               const int dim,
                               cudaStream_t stream)
{
  const int block_size=256;
  const int y_blocks=(dim+block_size*4-1)/(block_size*4);
  dim3 grid(num_tokens,y_blocks);
  embedding_backward_kernel<<<grid,block_size,0,stream>>>(
    grad_output,token_ids,grad_table,num_tokens,dim);
}

//==============================================================================
// Patch embedding kernels
//==============================================================================

//------------------------------------------------------------------------------
// Extract Patches Kernel (im2col for non-overlapping patches)
// input:  [batch, height, width, channels] (BHWC)
// output: [batch * num_patches, patch_flat_dim]
//------------------------------------------------------------------------------
__global__ void extract_patches_kernel(const float *input,
                                       float *output,
                                       const int batch,
                                       const int height,
                                       const int width,
                                       const int channels,
                                       const int patch_size,
                                       const int num_patches_h,
                                       const int num_patches_w,
                                       const int patch_flat_dim)
{
  const int total=batch*num_patches_h*num_patches_w*patch_flat_dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int num_patches=num_patches_h*num_patches_w;
    const int flat_pos=idx%patch_flat_dim;
    const int patch_idx=(idx/patch_flat_dim)%num_patches;
    const int b=idx/(patch_flat_dim*num_patches);

    const int ph=patch_idx/num_patches_w;
    const int pw=patch_idx%num_patches_w;

    const int c=flat_pos%channels;
    const int local_pixel=flat_pos/channels;
    const int local_h=local_pixel/patch_size;
    const int local_w=local_pixel%patch_size;

    const int global_h=ph*patch_size+local_h;
    const int global_w=pw*patch_size+local_w;

    const int input_idx=((b*height+global_h)*width+global_w)*channels+c;
    output[idx]=input[input_idx];
  }
}

//------------------------------------------------------------------------------
// Extract Patches Backward Kernel (col2im scatter-add)
//------------------------------------------------------------------------------
__global__ void extract_patches_backward_kernel(const float *grad_patches,
                                                float *grad_input,
                                                const int batch,
                                                const int height,
                                                const int width,
                                                const int channels,
                                                const int patch_size,
                                                const int num_patches_h,
                                                const int num_patches_w,
                                                const int patch_flat_dim)
{
  const int num_patches=num_patches_h*num_patches_w;
  const int total=batch*num_patches*patch_flat_dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int flat_pos=idx%patch_flat_dim;
    const int patch_idx=(idx/patch_flat_dim)%num_patches;
    const int b=idx/(patch_flat_dim*num_patches);

    const int ph=patch_idx/num_patches_w;
    const int pw=patch_idx%num_patches_w;

    const int c=flat_pos%channels;
    const int local_pixel=flat_pos/channels;
    const int local_h=local_pixel/patch_size;
    const int local_w=local_pixel%patch_size;

    const int global_h=ph*patch_size+local_h;
    const int global_w=pw*patch_size+local_w;

    const int input_idx=((b*height+global_h)*width+global_w)*channels+c;
    atomicAdd(&grad_input[input_idx],grad_patches[idx]);
  }
}

//------------------------------------------------------------------------------
// CLS Prepend Kernel
// Prepend CLS token at position 0, shift patches to 1..N
// patches: [batch, num_patches, dim], cls: [1, dim]
// output:  [batch, num_patches+1, dim]
//------------------------------------------------------------------------------
__global__ void cls_prepend_kernel(const float *patches,
                                   const float *cls_token,
                                   float *output,
                                   const int batch,
                                   const int num_patches,
                                   const int dim)
{
  const int out_seq=num_patches+1;
  const int total=batch*out_seq*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=idx%dim;
    const int s=(idx/dim)%out_seq;
    const int b=idx/(dim*out_seq);

    if(s==0)
    {
      output[idx]=cls_token[d];
    }
    else
    {
      output[idx]=patches[(b*num_patches+(s-1))*dim+d];
    }
  }
}

//------------------------------------------------------------------------------
// CLS Gradient Extract Kernel
// Split CLS gradient from patch gradients
// grad_output: [batch, num_patches+1, dim]
// grad_cls: [1, dim] (summed over batch), grad_patches: [batch, num_patches, dim]
//------------------------------------------------------------------------------
__global__ void cls_grad_extract_kernel(const float *grad_output,
                                        float *grad_cls,
                                        float *grad_patches,
                                        const int batch,
                                        const int num_patches,
                                        const int dim)
{
  const int out_seq=num_patches+1;
  const int total=batch*out_seq*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=idx%dim;
    const int s=(idx/dim)%out_seq;
    const int b=idx/(dim*out_seq);

    if(s==0)
    {
      atomicAdd(&grad_cls[d],grad_output[idx]);
    }
    else
    {
      grad_patches[(b*num_patches+(s-1))*dim+d]=grad_output[idx];
    }
  }
}

//------------------------------------------------------------------------------
// Patch Embedding Launchers
//------------------------------------------------------------------------------
void launch_extract_patches(const float *input,
                            float *output,
                            const int batch,
                            const int height,
                            const int width,
                            const int channels,
                            const int patch_size,
                            const int num_patches_h,
                            const int num_patches_w,
                            const int patch_flat_dim,
                            cudaStream_t stream)
{
  const int n=batch*num_patches_h*num_patches_w*patch_flat_dim;
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  extract_patches_kernel<<<num_blocks,block_size,0,stream>>>(
    input,output,batch,height,width,channels,
    patch_size,num_patches_h,num_patches_w,patch_flat_dim);
}

void launch_extract_patches_backward(const float *grad_patches,
                                     float *grad_input,
                                     const int batch,
                                     const int height,
                                     const int width,
                                     const int channels,
                                     const int patch_size,
                                     const int num_patches_h,
                                     const int num_patches_w,
                                     const int patch_flat_dim,
                                     cudaStream_t stream)
{
  const int n=batch*num_patches_h*num_patches_w*patch_flat_dim;
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  extract_patches_backward_kernel<<<num_blocks,block_size,0,stream>>>(
    grad_patches,grad_input,batch,height,width,channels,
    patch_size,num_patches_h,num_patches_w,patch_flat_dim);
}

void launch_cls_prepend(const float *patches,
                        const float *cls_token,
                        float *output,
                        const int batch,
                        const int num_patches,
                        const int dim,
                        cudaStream_t stream)
{
  const int out_seq=num_patches+1;
  const int n=batch*out_seq*dim;
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  cls_prepend_kernel<<<num_blocks,block_size,0,stream>>>(
    patches,cls_token,output,batch,num_patches,dim);
}

void launch_cls_grad_extract(const float *grad_output,
                             float *grad_cls,
                             float *grad_patches,
                             const int batch,
                             const int num_patches,
                             const int dim,
                             cudaStream_t stream)
{
  const int out_seq=num_patches+1;
  const int n=batch*out_seq*dim;
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  cls_grad_extract_kernel<<<num_blocks,block_size,0,stream>>>(
    grad_output,grad_cls,grad_patches,batch,num_patches,dim);
}

//==============================================================================
// Positional encoding kernels
//==============================================================================

//------------------------------------------------------------------------------
// Add Positional Encoding Kernel
// output[b,s,d] = input[b,s,d] + pe_table[s,d]
//------------------------------------------------------------------------------
__global__ void add_positional_encoding_kernel(const float *input,
                                               const float *pe_table,
                                               float *output,
                                               const int batch,
                                               const int seq_len,
                                               const int dim)
{
  const int total=batch*seq_len*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=idx%dim;
    const int s=(idx/dim)%seq_len;
    output[idx]=input[idx]+pe_table[s*dim+d];
  }
}

//------------------------------------------------------------------------------
// PE Table Backward Kernel
// grad_table[s,d] = sum_b grad_output[b,s,d]
// One thread per (s,d) pair, loops over batch
//------------------------------------------------------------------------------
__global__ void pe_table_backward_kernel(const float *grad_output,
                                         float *grad_table,
                                         const int batch,
                                         const int seq_len,
                                         const int dim)
{
  const int total=seq_len*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int s=idx/dim;
    const int d=idx%dim;
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=grad_output[(b*seq_len+s)*dim+d];
    }
    grad_table[idx]=sum;
  }
}

//------------------------------------------------------------------------------
// Positional Encoding Launchers
//------------------------------------------------------------------------------
void launch_add_positional_encoding(const float *input,
                                    const float *pe_table,
                                    float *output,
                                    const int batch,
                                    const int seq_len,
                                    const int dim,
                                    cudaStream_t stream)
{
  const int n=batch*seq_len*dim;
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  add_positional_encoding_kernel<<<num_blocks,block_size,0,stream>>>(
    input,pe_table,output,batch,seq_len,dim);
}

void launch_pe_table_backward(const float *grad_output,
                              float *grad_table,
                              const int batch,
                              const int seq_len,
                              const int dim,
                              cudaStream_t stream)
{
  const int n=seq_len*dim;
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  pe_table_backward_kernel<<<num_blocks,block_size,0,stream>>>(
    grad_output,grad_table,batch,seq_len,dim);
}

//==============================================================================
// Cross-entropy loss from logits (numerically stable for language modeling)
//==============================================================================

//------------------------------------------------------------------------------
// Cross-Entropy Logits Forward Kernel
// One block per position. Each block:
// 1. Finds max logit (for stability)
// 2. Computes log-sum-exp
// 3. Returns loss = -logits[target] + log_sum_exp
//------------------------------------------------------------------------------
__global__ void cross_entropy_logits_forward_kernel(const float *logits,
                                                     const float *targets,
                                                     float *losses,
                                                     const int vocab_size,
                                                     const int ignore_index)
{
  const int pos=blockIdx.x;
  const int tid=threadIdx.x;
  const int target_id=static_cast<int>(targets[pos]);

  // Check for ignore index
  if(target_id==ignore_index)
  {
    if(tid==0)
    {
      losses[pos]=0.0f;
    }
    return;
  }

  // Shared memory for reduction
  extern __shared__ float shared[];
  float *s_max=shared;
  float *s_sum=shared+blockDim.x;

  // Pointer to this position's logits
  const float *pos_logits=logits+pos*vocab_size;

  // Step 1: Find max logit (parallel reduction)
  float local_max=-1e30f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=pos_logits[v];
    if(val>local_max)
    {
      local_max=val;
    }
  }
  s_max[tid]=local_max;
  __syncthreads();

  // Reduce to find global max
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      if(s_max[tid+stride]>s_max[tid])
      {
        s_max[tid]=s_max[tid+stride];
      }
    }
    __syncthreads();
  }
  const float max_logit=s_max[0];

  // Step 2: Compute sum of exp(logit - max)
  float local_sum=0.0f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    local_sum+=expf(pos_logits[v]-max_logit);
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  // Reduce to find total sum
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }

  // Step 3: Compute loss = log_sum_exp - target_logit (always >= 0 mathematically)
  if(tid==0)
  {
    const float log_sum_exp=max_logit+logf(fmaxf(1.0f,s_sum[0]));
    const float target_logit=pos_logits[target_id];
    const float raw_loss=log_sum_exp-target_logit;
    // Propagate NaN instead of masking it; clamp tiny negative fp errors to 0
    losses[pos]=isnan(raw_loss)?raw_loss:fmaxf(0.0f,raw_loss);
  }
}

//------------------------------------------------------------------------------
// Cross-Entropy Logits Backward Kernel
// grad[i,j] = softmax(logits)[i,j] - (j == target[i] ? 1 : 0)
// One block per position.
//------------------------------------------------------------------------------
__global__ void cross_entropy_logits_backward_kernel(const float *logits,
                                                      const float *targets,
                                                      float *grad,
                                                      const int vocab_size,
                                                      const int ignore_index,
                                                      const float scale)
{
  const int pos=blockIdx.x;
  const int tid=threadIdx.x;
  const int target_id=static_cast<int>(targets[pos]);

  // Shared memory for reduction
  extern __shared__ float shared[];
  float *s_max=shared;
  float *s_sum=shared+blockDim.x;

  // Pointer to this position's logits and grad
  const float *pos_logits=logits+pos*vocab_size;
  float *pos_grad=grad+pos*vocab_size;

  // Check for ignore index - zero gradient
  if(target_id==ignore_index)
  {
    for(int v=tid;v<vocab_size;v+=blockDim.x)
    {
      pos_grad[v]=0.0f;
    }
    return;
  }

  // Step 1: Find max logit (parallel reduction)
  float local_max=-1e30f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=pos_logits[v];
    if(val>local_max)
    {
      local_max=val;
    }
  }
  s_max[tid]=local_max;
  __syncthreads();

  // Reduce to find global max
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      if(s_max[tid+stride]>s_max[tid])
      {
        s_max[tid]=s_max[tid+stride];
      }
    }
    __syncthreads();
  }
  const float max_logit=s_max[0];

  // Step 2: Compute sum of exp(logit - max)
  float local_sum=0.0f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    local_sum+=expf(pos_logits[v]-max_logit);
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  // Reduce to find total sum
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }
  const float sum_exp=s_sum[0];

  // Step 3: Compute gradient: softmax - one_hot
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float softmax_val=expf(pos_logits[v]-max_logit)/sum_exp;
    float g=softmax_val;
    if(v==target_id)
    {
      g-=1.0f;
    }
    pos_grad[v]=g*scale;
  }
}

//------------------------------------------------------------------------------
// Cross-Entropy Reduce Mean Kernel
// Sums losses and divides by valid count (excluding ignore_index)
//------------------------------------------------------------------------------
__global__ void cross_entropy_reduce_mean_kernel(const float *losses,
                                                  const float *targets,
                                                  float *output,
                                                  const int n,
                                                  const int ignore_index)
{
  extern __shared__ float shared[];
  float *s_sum=shared;
  float *s_count=shared+blockDim.x;

  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+tid;

  float local_sum=0.0f;
  float local_count=0.0f;

  if(idx<n)
  {
    const int target_id=static_cast<int>(targets[idx]);
    if(target_id!=ignore_index)
    {
      local_sum=losses[idx];
      local_count=1.0f;
    }
  }

  s_sum[tid]=local_sum;
  s_count[tid]=local_count;
  __syncthreads();

  // Reduce within block
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
      s_count[tid]+=s_count[tid+stride];
    }
    __syncthreads();
  }

  // Atomic add to global output
  if(tid==0)
  {
    atomicAdd(&output[0],s_sum[0]);
    atomicAdd(&output[1],s_count[0]);
  }
}

//------------------------------------------------------------------------------
// Cross-Entropy Logits Launchers
//------------------------------------------------------------------------------
void launch_cross_entropy_logits_forward(const float *logits,
                                         const float *targets,
                                         float *losses,
                                         const int n,
                                         const int vocab_size,
                                         const int ignore_index,
                                         cudaStream_t stream)
{
  // One block per position, use enough threads to cover vocab
  const int block_size=256;
  const int num_blocks=n;
  const size_t shared_size=2*block_size*sizeof(float);
  cross_entropy_logits_forward_kernel<<<num_blocks,block_size,shared_size,stream>>>(
    logits,targets,losses,vocab_size,ignore_index);
}

void launch_cross_entropy_logits_backward(const float *logits,
                                          const float *targets,
                                          float *grad,
                                          const int n,
                                          const int vocab_size,
                                          const int ignore_index,
                                          cudaStream_t stream)
{
  // Count valid positions for scale factor
  // For simplicity, use 1/n as scale (caller should adjust for ignore_index if needed)
  const float scale=1.0f/static_cast<float>(n);

  const int block_size=256;
  const int num_blocks=n;
  const size_t shared_size=2*block_size*sizeof(float);
  cross_entropy_logits_backward_kernel<<<num_blocks,block_size,shared_size,stream>>>(
    logits,targets,grad,vocab_size,ignore_index,scale);
}

void launch_cross_entropy_reduce_mean(const float *losses,
                                      const float *targets,
                                      float *output,
                                      const int n,
                                      const int ignore_index,
                                      cudaStream_t stream)
{
  // Output should be pre-zeroed (stores sum and count)
  const int block_size=256;
  const int num_blocks=(n+block_size-1)/block_size;
  const size_t shared_size=2*block_size*sizeof(float);
  cross_entropy_reduce_mean_kernel<<<num_blocks,block_size,shared_size,stream>>>(
    losses,targets,output,n,ignore_index);
}

//------------------------------------------------------------------------------
// SiLU Backward Kernel (vectorized float4)
// grad_input = grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
// Same as swish backward but without the output parameter.
//------------------------------------------------------------------------------
__global__ void silu_backward_kernel(const float *grad_output,
                                     const float *input,
                                     float *grad_input,
                                     const int n4)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n4)
  {
    const float4 g=reinterpret_cast<const float4 *>(grad_output)[idx];
    const float4 v=reinterpret_cast<const float4 *>(input)[idx];
    float4 r;
    float s;
    s=1.0f/(1.0f+expf(-v.x));r.x=g.x*s*(1.0f+v.x*(1.0f-s));
    s=1.0f/(1.0f+expf(-v.y));r.y=g.y*s*(1.0f+v.y*(1.0f-s));
    s=1.0f/(1.0f+expf(-v.z));r.z=g.z*s*(1.0f+v.z*(1.0f-s));
    s=1.0f/(1.0f+expf(-v.w));r.w=g.w*s*(1.0f+v.w*(1.0f-s));
    reinterpret_cast<float4 *>(grad_input)[idx]=r;
  }
}

__global__ void silu_backward_tail_kernel(const float *grad_output,
                                          const float *input,
                                          float *grad_input,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=input[idx];
    const float s=1.0f/(1.0f+expf(-x));
    grad_input[idx]=grad_output[idx]*s*(1.0f+x*(1.0f-s));
  }
}

void launch_silu_backward(const float *grad_output,
                          const float *input,
                          float *grad_input,
                          const int n,
                          cudaStream_t stream)
{
  const int n4=n/4;
  const int tail=n-n4*4;
  if(n4>0)
  {
    const int num_blocks=(n4+g_act_block_size-1)/g_act_block_size;
    silu_backward_kernel<<<num_blocks,g_act_block_size,0,stream>>>(
      grad_output,input,grad_input,n4);
  }
  if(tail>0)
  {
    silu_backward_tail_kernel<<<1,tail,0,stream>>>(
      grad_output,input,grad_input,n4*4,n);
  }
}

//------------------------------------------------------------------------------
// Sum Axis 0 Kernel (sum over batch)
// input: [batch, dim], output: [dim]
//------------------------------------------------------------------------------
__global__ void sum_axis0_kernel(const float *input,
                                 float *output,
                                 const int batch,
                                 const int dim)
{
  const int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d<dim)
  {
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=input[b*dim+d];
    }
    output[d]=sum;
  }
}

void launch_sum_axis0(const float *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(dim+block_size-1)/block_size;
  sum_axis0_kernel<<<num_blocks,block_size,0,stream>>>(input,output,batch,dim);
}

//------------------------------------------------------------------------------
// Sum Axis 1 Kernel (sum over dim)
// input: [batch, dim], output: [batch]
//------------------------------------------------------------------------------
__global__ void sum_axis1_kernel(const float *input,
                                 float *output,
                                 const int batch,
                                 const int dim)
{
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b<batch)
  {
    float sum=0.0f;
    for(int d=0;d<dim;++d)
    {
      sum+=input[b*dim+d];
    }
    output[b]=sum;
  }
}

void launch_sum_axis1(const float *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(batch+block_size-1)/block_size;
  sum_axis1_kernel<<<num_blocks,block_size,0,stream>>>(input,output,batch,dim);
}

//------------------------------------------------------------------------------
// Log-Sum-Exp Kernel
// input: [batch, dim], output: [batch]
// output[b] = log(sum_d(exp(input[b,d])))
//------------------------------------------------------------------------------
__global__ void logsumexp_kernel(const float *input,
                                 float *output,
                                 const int batch,
                                 const int dim)
{
  const int b=blockIdx.x;
  if(b>=batch)
  {
    return;
  }

  extern __shared__ float shared[];
  float *s_max=shared;
  float *s_sum=shared+blockDim.x;
  const int tid=threadIdx.x;
  const float *row=input+b*dim;

  // Find max
  float local_max=-1e30f;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_max=fmaxf(local_max,row[d]);
  }
  s_max[tid]=local_max;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_max[tid]=fmaxf(s_max[tid],s_max[tid+stride]);
    }
    __syncthreads();
  }
  const float max_val=s_max[0];

  // Sum exp(x - max)
  float local_sum=0.0f;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_sum+=expf(row[d]-max_val);
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }

  if(tid==0)
  {
    output[b]=max_val+logf(s_sum[0]);
  }
}

void launch_logsumexp(const float *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=batch;
  const size_t shared_size=2*block_size*sizeof(float);
  logsumexp_kernel<<<num_blocks,block_size,shared_size,stream>>>(input,output,
                                                                   batch,dim);
}

//------------------------------------------------------------------------------
// Normalize Rows Kernel
// input: [batch, dim], output: [batch, dim]
// output[b,d] = input[b,d] / sum_d(input[b,:])
//------------------------------------------------------------------------------
__global__ void normalize_rows_kernel(const float *input,
                                      float *output,
                                      const int batch,
                                      const int dim)
{
  const int b=blockIdx.x;
  if(b>=batch)
  {
    return;
  }

  extern __shared__ float s_sum[];
  const int tid=threadIdx.x;
  const float *row_in=input+b*dim;
  float *row_out=output+b*dim;

  // Sum the row
  float local_sum=0.0f;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_sum+=row_in[d];
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }
  const float sum=fmaxf(s_sum[0],1e-10f);

  // Normalize
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    row_out[d]=row_in[d]/sum;
  }
}

void launch_normalize_rows(const float *input,
                           float *output,
                           const int batch,
                           const int dim,
                           cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=batch;
  const size_t shared_size=block_size*sizeof(float);
  normalize_rows_kernel<<<num_blocks,block_size,shared_size,stream>>>(input,output,
                                                                        batch,dim);
}

//------------------------------------------------------------------------------
// Top-K Kernel (simple implementation)
// input: [batch, dim], indices: [batch, k], values: [batch, k]
//------------------------------------------------------------------------------
__global__ void topk_kernel(const float *input,
                            int *indices,
                            float *values,
                            const int batch,
                            const int dim,
                            const int k)
{
  const int b=blockIdx.x;
  if(b>=batch)
  {
    return;
  }

  const float *row=input+b*dim;
  int *out_idx=indices+b*k;
  float *out_val=values+b*k;

  // Simple selection sort for top-k (good enough for small k)
  // Mark selected indices with -inf
  extern __shared__ float s_data[];
  float *temp=s_data;
  const int tid=threadIdx.x;

  // Copy to shared memory
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    temp[d]=row[d];
  }
  __syncthreads();

  // Only thread 0 does the selection (simple but correct)
  if(tid==0)
  {
    for(int i=0;i<k;++i)
    {
      float max_val=-1e30f;
      int max_idx=0;
      for(int d=0;d<dim;++d)
      {
        if(temp[d]>max_val)
        {
          max_val=temp[d];
          max_idx=d;
        }
      }
      out_idx[i]=max_idx;
      out_val[i]=max_val;
      temp[max_idx]=-1e30f;  // Mark as selected
    }
  }
}

void launch_topk(const float *input,
                 int *indices,
                 float *values,
                 const int batch,
                 const int dim,
                 const int k,
                 cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=batch;
  const size_t shared_size=dim*sizeof(float);
  topk_kernel<<<num_blocks,block_size,shared_size,stream>>>(input,indices,values,
                                                              batch,dim,k);
}

//------------------------------------------------------------------------------
// Scatter Add Kernel
// output[b, indices[b,k]] += values[b,k]
// values: [batch, k], indices: [batch, k], output: [batch, dim]
//------------------------------------------------------------------------------
__global__ void scatter_add_kernel(const float *values,
                                   const int *indices,
                                   float *output,
                                   const int batch,
                                   const int k,
                                   const int dim)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=batch*k;
  if(idx<total)
  {
    const int b=idx/k;
    const int target_idx=indices[idx];
    if(target_idx>=0&&target_idx<dim)
    {
      atomicAdd(&output[b*dim+target_idx],values[idx]);
    }
  }
}

void launch_scatter_add(const float *values,
                        const int *indices,
                        float *output,
                        const int batch,
                        const int k,
                        const int dim,
                        cudaStream_t stream)
{
  const int total=batch*k;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  scatter_add_kernel<<<num_blocks,block_size,0,stream>>>(values,indices,output,
                                                           batch,k,dim);
}

//------------------------------------------------------------------------------
// MoE Dispatch Kernel
// Gathers tokens to expert-specific buffers based on routing indices
// expert_indices: [num_tokens, top_k] - which experts each token routes to
// expert_offsets: [num_experts+1] - cumulative token counts per expert
// dispatch_map: [num_tokens, top_k] - position within expert buffer for each assignment
//------------------------------------------------------------------------------
__global__ void moe_dispatch_kernel(const float *input,
                                    const float *expert_indices,
                                    const int *dispatch_map,
                                    float *expert_buffer,
                                    const int *expert_offsets,
                                    const int num_tokens,
                                    const int dim,
                                    const int top_k)
{
  // Each thread handles one dimension of one token-expert assignment
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int total_assignments=num_tokens*top_k*dim;

  if(tid<total_assignments)
  {
    const int d=tid%dim;
    const int assignment_idx=(tid/dim);
    const int token_idx=assignment_idx/top_k;
    const int k_idx=assignment_idx%top_k;

    const int expert_idx=static_cast<int>(expert_indices[token_idx*top_k+k_idx]);
    const int pos_in_expert=dispatch_map[token_idx*top_k+k_idx];

    if(expert_idx>=0&&pos_in_expert>=0)
    {
      const int expert_start=expert_offsets[expert_idx];
      const int dest_idx=(expert_start+pos_in_expert)*dim+d;
      expert_buffer[dest_idx]=input[token_idx*dim+d];
    }
  }
}

void launch_moe_dispatch(const float *input,
                         const float *expert_indices,
                         const int *dispatch_map,
                         float *expert_buffer,
                         const int *expert_offsets,
                         const int num_tokens,
                         const int dim,
                         const int top_k,
                         cudaStream_t stream)
{
  const int total=num_tokens*top_k*dim;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  moe_dispatch_kernel<<<num_blocks,block_size,0,stream>>>(input,
                                                           expert_indices,
                                                           dispatch_map,
                                                           expert_buffer,
                                                           expert_offsets,
                                                           num_tokens,
                                                           dim,
                                                           top_k);
}

//------------------------------------------------------------------------------
// MoE Combine Kernel
// Scatters expert outputs back to token positions with routing weights
//------------------------------------------------------------------------------
__global__ void moe_combine_kernel(const float *expert_buffer,
                                   const float *expert_indices,
                                   const float *expert_weights,
                                   const int *dispatch_map,
                                   const int *expert_offsets,
                                   float *output,
                                   const int num_tokens,
                                   const int dim,
                                   const int top_k)
{
  // Each thread handles one dimension of one token
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=num_tokens*dim;

  if(tid<total)
  {
    const int token_idx=tid/dim;
    const int d=tid%dim;

    float sum=0.0f;

    for(int k=0;k<top_k;++k)
    {
      const int expert_idx=static_cast<int>(expert_indices[token_idx*top_k+k]);
      const float weight=expert_weights[token_idx*top_k+k];
      const int pos_in_expert=dispatch_map[token_idx*top_k+k];

      if(expert_idx>=0&&pos_in_expert>=0)
      {
        const int expert_start=expert_offsets[expert_idx];
        const int src_idx=(expert_start+pos_in_expert)*dim+d;
        sum+=weight*expert_buffer[src_idx];
      }
    }

    output[tid]=sum;
  }
}

void launch_moe_combine(const float *expert_buffer,
                        const float *expert_indices,
                        const float *expert_weights,
                        const int *dispatch_map,
                        const int *expert_offsets,
                        float *output,
                        const int num_tokens,
                        const int dim,
                        const int top_k,
                        cudaStream_t stream)
{
  const int total=num_tokens*dim;
  const int block_size=256;
  const int num_blocks=(total+block_size-1)/block_size;
  moe_combine_kernel<<<num_blocks,block_size,0,stream>>>(expert_buffer,
                                                          expert_indices,
                                                          expert_weights,
                                                          dispatch_map,
                                                          expert_offsets,
                                                          output,
                                                          num_tokens,
                                                          dim,
                                                          top_k);
}

//------------------------------------------------------------------------------
// MoE Top-K Gating Kernel
// Fused softmax + top-k selection for router
// Input: router_logits [num_tokens, num_experts]
// Output: expert_indices [num_tokens, top_k], expert_weights [num_tokens, top_k]
//------------------------------------------------------------------------------
__global__ void moe_topk_gating_kernel(const float *router_logits,
                                       float *expert_indices,
                                       float *expert_weights,
                                       float *router_probs,
                                       const int num_tokens,
                                       const int num_experts,
                                       const int top_k)
{
  // Each block handles one token
  const int token_idx=blockIdx.x;
  if(token_idx>=num_tokens)
  {
    return;
  }

  extern __shared__ float shared[];
  float *logits_shared=shared;
  float *probs_shared=shared+num_experts;
  int *top_indices=reinterpret_cast<int*>(shared+2*num_experts);
  float *top_values=reinterpret_cast<float*>(shared+2*num_experts+top_k);

  const int tid=threadIdx.x;

  // Load logits to shared memory
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    logits_shared[e]=router_logits[token_idx*num_experts+e];
  }
  __syncthreads();

  // Find max for numerical stability (thread 0)
  float max_val=-1e30f;
  if(tid==0)
  {
    for(int e=0;e<num_experts;++e)
    {
      if(logits_shared[e]>max_val)
      {
        max_val=logits_shared[e];
      }
    }
  }
  __syncthreads();

  // Broadcast max_val using shared memory
  if(tid==0)
  {
    probs_shared[0]=max_val;
  }
  __syncthreads();
  max_val=probs_shared[0];

  // Compute exp(logits - max) and sum
  float local_sum=0.0f;
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    float exp_val=expf(logits_shared[e]-max_val);
    probs_shared[e]=exp_val;
    local_sum+=exp_val;
  }

  // Reduce sum across threads (simple reduction for small num_experts)
  __shared__ float sum_shared[256];
  sum_shared[tid]=local_sum;
  __syncthreads();

  if(tid==0)
  {
    float total_sum=0.0f;
    for(int i=0;i<blockDim.x&&i<num_experts;++i)
    {
      total_sum+=sum_shared[i];
    }
    sum_shared[0]=total_sum;
  }
  __syncthreads();

  float total_sum=sum_shared[0];

  // Normalize to get probabilities
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    probs_shared[e]/=total_sum;
    router_probs[token_idx*num_experts+e]=probs_shared[e];
  }
  __syncthreads();

  // Top-k selection (thread 0 does sequential selection for simplicity)
  if(tid==0)
  {
    for(int k=0;k<top_k;++k)
    {
      float max_prob=-1.0f;
      int max_idx=-1;

      for(int e=0;e<num_experts;++e)
      {
        // Check if already selected
        bool already_selected=false;
        for(int j=0;j<k;++j)
        {
          if(top_indices[j]==e)
          {
            already_selected=true;
            break;
          }
        }

        if(already_selected==false&&probs_shared[e]>max_prob)
        {
          max_prob=probs_shared[e];
          max_idx=e;
        }
      }

      top_indices[k]=max_idx;
      top_values[k]=max_prob;
    }

    // Renormalize top-k weights to sum to 1
    float topk_sum=0.0f;
    for(int k=0;k<top_k;++k)
    {
      topk_sum+=top_values[k];
    }

    for(int k=0;k<top_k;++k)
    {
      expert_indices[token_idx*top_k+k]=static_cast<float>(top_indices[k]);
      expert_weights[token_idx*top_k+k]=top_values[k]/topk_sum;
    }
  }
}

void launch_moe_topk_gating(const float *router_logits,
                            float *expert_indices,
                            float *expert_weights,
                            float *router_probs,
                            const int num_tokens,
                            const int num_experts,
                            const int top_k,
                            cudaStream_t stream)
{
  // One block per token, enough threads to cover num_experts
  const int threads_per_block=min(256,((num_experts+31)/32)*32);
  const int shared_size=(2*num_experts)*sizeof(float)+(top_k)*sizeof(int)+(top_k)*sizeof(float);
  moe_topk_gating_kernel<<<num_tokens,threads_per_block,shared_size,stream>>>(router_logits,
                                                                               expert_indices,
                                                                               expert_weights,
                                                                               router_probs,
                                                                               num_tokens,
                                                                               num_experts,
                                                                               top_k);
}

//------------------------------------------------------------------------------
// MoE Build Dispatch Map Kernel
// Builds the dispatch_map that tracks position of each token within expert buffers
// Also computes expert_offsets (cumulative counts)
//------------------------------------------------------------------------------
__global__ void moe_count_per_expert_kernel(const float *expert_indices,
                                            int *expert_counts,
                                            const int num_tokens,
                                            const int num_experts,
                                            const int top_k,
                                            const int capacity_per_expert)
{
  // Simple atomic counting - each thread handles one token
  const int token_idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(token_idx>=num_tokens)
  {
    return;
  }

  for(int k=0;k<top_k;++k)
  {
    const int expert_idx=static_cast<int>(expert_indices[token_idx*top_k+k]);
    if(expert_idx>=0&&expert_idx<num_experts)
    {
      const int old_count=atomicAdd(&expert_counts[expert_idx],1);
      // Capacity enforcement happens at dispatch time
      (void)old_count;
    }
  }
}

void launch_moe_count_per_expert(const float *expert_indices,
                                 int *expert_counts,
                                 const int num_tokens,
                                 const int num_experts,
                                 const int top_k,
                                 const int capacity_per_expert,
                                 cudaStream_t stream)
{
  const int block_size=256;
  const int num_blocks=(num_tokens+block_size-1)/block_size;
  moe_count_per_expert_kernel<<<num_blocks,block_size,0,stream>>>(expert_indices,
                                                                    expert_counts,
                                                                    num_tokens,
                                                                    num_experts,
                                                                    top_k,
                                                                    capacity_per_expert);
}

}  // extern "C" - end of non-template functions

//------------------------------------------------------------------------------
// FlashAttention-2 Forward Kernel
// Implements tiled attention with online softmax to avoid O(n²) memory
// References: https://arxiv.org/abs/2307.08691
//------------------------------------------------------------------------------

// Block sizes for FlashAttention - tuned for modern GPUs (RTX 5090)
// These must fit in shared memory: 2*Br*D + 2*Bc*D + Br*Bc floats
constexpr int FA_BR=64;   // Q block rows
constexpr int FA_BC=64;   // K/V block rows
constexpr int FA_D_MAX=128;  // Max head dimension supported

// Warp reduction for max value
__device__ __forceinline__ float warp_reduce_max(float val)
{
  for(int offset=16;offset>0;offset/=2)
  {
    val=fmaxf(val,__shfl_down_sync(0xffffffff,val,offset));
  }
  return val;
}

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val)
{
  for(int offset=16;offset>0;offset/=2)
  {
    val+=__shfl_down_sync(0xffffffff,val,offset);
  }
  return val;
}

// Block reduction for max (across all warps)
__device__ __forceinline__ float block_reduce_max(float val,float *shared_mem,int tid,int block_size)
{
  const int warp_id=tid/32;
  const int lane_id=tid%32;
  const int num_warps=(block_size+31)/32;

  val=warp_reduce_max(val);

  if(lane_id==0)
  {
    shared_mem[warp_id]=val;
  }
  __syncthreads();

  if(tid<num_warps)
  {
    val=shared_mem[tid];
  }
  else
  {
    val=-INFINITY;
  }

  if(warp_id==0)
  {
    val=warp_reduce_max(val);
  }

  return val;
}

// Block reduction for sum
__device__ __forceinline__ float block_reduce_sum(float val,float *shared_mem,int tid,int block_size)
{
  const int warp_id=tid/32;
  const int lane_id=tid%32;
  const int num_warps=(block_size+31)/32;

  val=warp_reduce_sum(val);

  if(lane_id==0)
  {
    shared_mem[warp_id]=val;
  }
  __syncthreads();

  if(tid<num_warps)
  {
    val=shared_mem[tid];
  }
  else
  {
    val=0.0f;
  }

  if(warp_id==0)
  {
    val=warp_reduce_sum(val);
  }

  return val;
}

// FlashAttention-2 Forward Kernel
// Grid: (batch_heads, ceil(seq_len/Br))
// Block: (Br) threads - each thread handles one row of Q block
template<int D>
__global__ void flash_attention_forward_kernel(const float * __restrict__ Q,
                                                const float * __restrict__ K,
                                                const float * __restrict__ V,
                                                float * __restrict__ O,
                                                float * __restrict__ L,
                                                const int seq_len,
                                                const float scale,
                                                const int causal)
{
  // Batch-head index
  const int bh=blockIdx.x;
  // Q block index
  const int q_block_idx=blockIdx.y;
  // Thread ID within block (corresponds to row within Q block)
  const int tid=threadIdx.x;

  // Global row index for this thread's Q row
  const int q_row=q_block_idx*FA_BR+tid;

  // Early exit if this Q row is beyond sequence length
  if(q_row>=seq_len)
  {
    return;
  }

  // Shared memory layout:
  // [0..FA_BR*D): Q tile
  // [FA_BR*D..FA_BR*D+FA_BC*D): K tile
  // [FA_BR*D+FA_BC*D..FA_BR*D+2*FA_BC*D): V tile
  // [FA_BR*D+2*FA_BC*D..): scratch for reductions
  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *K_tile=smem+FA_BR*D;
  float *V_tile=smem+FA_BR*D+FA_BC*D;

  // Pointers to this batch-head's Q, K, V, O
  const float *Q_bh=Q+bh*seq_len*D;
  const float *K_bh=K+bh*seq_len*D;
  const float *V_bh=V+bh*seq_len*D;
  float *O_bh=O+bh*seq_len*D;
  float *L_bh=L+bh*seq_len;

  // Per-row accumulators (registers)
  float m_i=-INFINITY;  // Running max
  float l_i=0.0f;        // Running sum of exp
  float o_i[FA_D_MAX];   // Running output accumulator
  for(int d=0;d<D;++d)
  {
    o_i[d]=0.0f;
  }

  // Load Q row for this thread into shared memory
  // All threads in block load their Q rows cooperatively
  for(int d=0;d<D;++d)
  {
    if(q_row<seq_len)
    {
      Q_tile[tid*D+d]=Q_bh[q_row*D+d];
    }
    else
    {
      Q_tile[tid*D+d]=0.0f;
    }
  }
  __syncthreads();

  // Number of K/V blocks to process
  int num_kv_blocks=(seq_len+FA_BC-1)/FA_BC;

  // For causal: only process K/V blocks up to and including the block containing q_row
  if(causal)
  {
    num_kv_blocks=min(num_kv_blocks,(q_row/FA_BC)+1);
  }

  // Loop over K/V blocks
  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    // Cooperatively load K and V tiles into shared memory
    // Each thread loads multiple elements
    const int kv_start=kv_block*FA_BC;
    const int elements_per_tile=FA_BC*D;
    const int threads_in_block=FA_BR;

    for(int i=tid;i<elements_per_tile;i+=threads_in_block)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;

      if(global_row<seq_len)
      {
        K_tile[row*D+col]=K_bh[global_row*D+col];
        V_tile[row*D+col]=V_bh[global_row*D+col];
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    // Compute S[tid, :] = Q[tid, :] @ K[:, :]^T * scale
    // This is a row of the attention scores for this Q row
    float S_row[FA_BC];
    float row_max=-INFINITY;

    for(int j=0;j<FA_BC;++j)
    {
      const int k_row=kv_start+j;

      // Apply causal mask: if k_row > q_row, mask with -inf
      if(causal&&k_row>q_row)
      {
        S_row[j]=-INFINITY;
      }
      else if(k_row>=seq_len)
      {
        S_row[j]=-INFINITY;
      }
      else
      {
        // Dot product of Q[tid, :] and K[j, :]
        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_tile[tid*D+d]*K_tile[j*D+d];
        }
        S_row[j]=dot*scale;
      }

      if(S_row[j]>row_max)
      {
        row_max=S_row[j];
      }
    }

    // Online softmax update
    // m_new = max(m_i, row_max)
    const float m_new=fmaxf(m_i,row_max);

    // Compute sum of exp(S_row - row_max) for this block
    float block_sum=0.0f;
    for(int j=0;j<FA_BC;++j)
    {
      if(S_row[j]>-INFINITY)
      {
        block_sum+=expf(S_row[j]-m_new);
      }
    }

    // Rescale previous accumulator and add new contribution
    // l_new = exp(m_i - m_new) * l_i + sum(exp(S_row - m_new))
    const float scale_old=expf(m_i-m_new);
    const float l_new=scale_old*l_i+block_sum;

    // Update output accumulator
    // o_new = (scale_old * l_i * o_i + sum(exp(S_row - m_new) * V)) / l_new
    // But we defer the division until the end

    // Scale old output
    for(int d=0;d<D;++d)
    {
      o_i[d]*=scale_old;
    }

    // Add new contribution: sum over j of exp(S_row[j] - m_new) * V[j, :]
    for(int j=0;j<FA_BC;++j)
    {
      if(S_row[j]>-INFINITY)
      {
        const float p=expf(S_row[j]-m_new);
        for(int d=0;d<D;++d)
        {
          o_i[d]+=p*V_tile[j*D+d];
        }
      }
    }

    // Update running state
    m_i=m_new;
    l_i=l_new;

    __syncthreads();
  }

  // Final normalization and write output
  if(q_row<seq_len)
  {
    float inv_l=0.0f;
    if(l_i>0.0f)
    {
      inv_l=1.0f/l_i;
    }
    for(int d=0;d<D;++d)
    {
      O_bh[q_row*D+d]=o_i[d]*inv_l;
    }

    // Store logsumexp for backward pass: L = m + log(l)
    L_bh[q_row]=m_i+logf(l_i+1e-10f);
  }
}

// Launch wrappers must have C linkage for header declaration
extern "C"
{

// Launch wrapper for FlashAttention forward
void launch_flash_attention_forward(const float *Q,
                                    const float *K,
                                    const float *V,
                                    float *O,
                                    float *L,
                                    const int batch_heads,
                                    const int seq_len,
                                    const int head_dim,
                                    const float scale,
                                    const int causal,
                                    cudaStream_t stream)
{
  // Grid: (batch_heads, num_q_blocks)
  const int num_q_blocks=(seq_len+FA_BR-1)/FA_BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(FA_BR);

  // Shared memory: Q tile + K tile + V tile + scratch
  const size_t smem_size=(FA_BR*head_dim+2*FA_BC*head_dim+32)*sizeof(float);

  // Dispatch based on head dimension
  // Note: head_dim >= 64 needs > 48KB shared memory; opt-in via cudaFuncSetAttribute
  switch(head_dim)
  {
    case 32:
      flash_attention_forward_kernel<32><<<grid,block,smem_size,stream>>>(Q,K,V,O,L,seq_len,scale,causal);
      break;
    case 64:
      cudaFuncSetAttribute((void *)flash_attention_forward_kernel<64>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_size));
      flash_attention_forward_kernel<64><<<grid,block,smem_size,stream>>>(Q,K,V,O,L,seq_len,scale,causal);
      break;
    case 80:
      cudaFuncSetAttribute((void *)flash_attention_forward_kernel<80>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_size));
      flash_attention_forward_kernel<80><<<grid,block,smem_size,stream>>>(Q,K,V,O,L,seq_len,scale,causal);
      break;
    case 96:
      cudaFuncSetAttribute((void *)flash_attention_forward_kernel<96>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_size));
      flash_attention_forward_kernel<96><<<grid,block,smem_size,stream>>>(Q,K,V,O,L,seq_len,scale,causal);
      break;
    case 128:
      cudaFuncSetAttribute((void *)flash_attention_forward_kernel<128>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_size));
      flash_attention_forward_kernel<128><<<grid,block,smem_size,stream>>>(Q,K,V,O,L,seq_len,scale,causal);
      break;
    default:
      fprintf(stderr,
              "FATAL: flash_attention_forward unsupported head_dim=%d"
              " (supported: 32,64,80,96,128). Use standard attention.\n",
              head_dim);
      abort();
  }
}

}  // extern "C" - end of forward launch wrapper

//------------------------------------------------------------------------------
// FlashAttention-2 Backward: Precompute Di = dot(dO, O) per row
//------------------------------------------------------------------------------
template<int D>
__global__ void flash_attention_precompute_di_kernel(const float * __restrict__ dO,
                                                      const float * __restrict__ O,
                                                      float * __restrict__ Di,
                                                      const int seq_len)
{
  const int bh=blockIdx.x;
  const int row=blockIdx.y*blockDim.x+threadIdx.x;

  if(row>=seq_len)
  {
    return;
  }

  const float *dO_row=dO+bh*seq_len*D+row*D;
  const float *O_row=O+bh*seq_len*D+row*D;

  float sum=0.0f;
  for(int d=0;d<D;++d)
  {
    sum+=dO_row[d]*O_row[d];
  }
  Di[bh*seq_len+row]=sum;
}

//------------------------------------------------------------------------------
// FlashAttention-2 Backward Kernel - dK/dV (optimized)
// Uses precomputed Di, template-sized register arrays, 128 thread blocks
//------------------------------------------------------------------------------
constexpr int FA_BWD_BR=64;     // Q tile rows for dK/dV kernel
constexpr int FA_BWD_BC=128;    // K/V block size (threads per block)
constexpr int FA_BWD_DQ_BR=128; // Q block size for dQ kernel
constexpr int FA_BWD_DQ_BC=64;  // K/V tile rows for dQ kernel

template<int D>
__global__ void flash_attention_backward_kernel(const float * __restrict__ Q,
                                                 const float * __restrict__ K,
                                                 const float * __restrict__ V,
                                                 const float * __restrict__ dO,
                                                 const float * __restrict__ L,
                                                 const float * __restrict__ Di,
                                                 float * __restrict__ dK,
                                                 float * __restrict__ dV,
                                                 const int seq_len,
                                                 const float scale,
                                                 const int causal)
{
  const int bh=blockIdx.x;
  const int kv_block_idx=blockIdx.y;
  const int tid=threadIdx.x;

  const int kv_row=kv_block_idx*FA_BWD_BC+tid;

  if(kv_row>=seq_len)
  {
    return;
  }

  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *dO_tile=smem+FA_BWD_BR*D;
  float *L_tile=smem+2*FA_BWD_BR*D;
  float *Di_tile=smem+2*FA_BWD_BR*D+FA_BWD_BR;

  const float *Q_bh=Q+bh*seq_len*D;
  const float *K_bh=K+bh*seq_len*D;
  const float *V_bh=V+bh*seq_len*D;
  const float *dO_bh=dO+bh*seq_len*D;
  const float *L_bh=L+bh*seq_len;
  const float *Di_bh=Di+bh*seq_len;
  float *dK_bh=dK+bh*seq_len*D;
  float *dV_bh=dV+bh*seq_len*D;

  // Load K and V rows into registers (sized exactly to D)
  float K_row[D];
  float V_row[D];
  for(int d=0;d<D;++d)
  {
    K_row[d]=K_bh[kv_row*D+d];
    V_row[d]=V_bh[kv_row*D+d];
  }

  float dK_acc[D];
  float dV_acc[D];
  for(int d=0;d<D;++d)
  {
    dK_acc[d]=0.0f;
    dV_acc[d]=0.0f;
  }

  const int num_q_blocks=(seq_len+FA_BWD_BR-1)/FA_BWD_BR;

  int start_q_block=0;
  if(causal)
  {
    start_q_block=kv_row/FA_BWD_BR;
  }

  for(int q_block=start_q_block;q_block<num_q_blocks;++q_block)
  {
    const int q_start=q_block*FA_BWD_BR;

    // Cooperatively load Q tile and dO tile
    for(int i=tid;i<FA_BWD_BR*D;i+=FA_BWD_BC)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=q_start+row;

      if(global_row<seq_len)
      {
        Q_tile[row*D+col]=Q_bh[global_row*D+col];
        dO_tile[row*D+col]=dO_bh[global_row*D+col];
      }
      else
      {
        Q_tile[row*D+col]=0.0f;
        dO_tile[row*D+col]=0.0f;
      }
    }

    // Load L and precomputed Di
    if(tid<FA_BWD_BR)
    {
      const int global_row=q_start+tid;
      if(global_row<seq_len)
      {
        L_tile[tid]=L_bh[global_row];
        Di_tile[tid]=Di_bh[global_row];
      }
      else
      {
        L_tile[tid]=-INFINITY;
        Di_tile[tid]=0.0f;
      }
    }
    __syncthreads();

    for(int qi=0;qi<FA_BWD_BR;++qi)
    {
      const int q_row=q_start+qi;

      if(causal&&kv_row>q_row)
      {
        continue;
      }

      if(q_row>=seq_len)
      {
        continue;
      }

      float dot=0.0f;
      for(int d=0;d<D;++d)
      {
        dot+=Q_tile[qi*D+d]*K_row[d];
      }
      const float S_val=dot*scale;
      const float P_val=expf(fminf(S_val-L_tile[qi],0.0f));

      for(int d=0;d<D;++d)
      {
        dV_acc[d]+=P_val*dO_tile[qi*D+d];
      }

      float dP=0.0f;
      for(int d=0;d<D;++d)
      {
        dP+=dO_tile[qi*D+d]*V_row[d];
      }

      const float dS=P_val*(dP-Di_tile[qi])*scale;

      for(int d=0;d<D;++d)
      {
        dK_acc[d]+=dS*Q_tile[qi*D+d];
      }
    }
    __syncthreads();
  }

  if(kv_row<seq_len)
  {
    for(int d=0;d<D;++d)
    {
      dK_bh[kv_row*D+d]=dK_acc[d];
      dV_bh[kv_row*D+d]=dV_acc[d];
    }
  }
}

//------------------------------------------------------------------------------
// FlashAttention-2 Backward Kernel - dQ (optimized)
// Uses precomputed Di, template-sized register arrays, 128 thread blocks
//------------------------------------------------------------------------------
template<int D>
__global__ void flash_attention_backward_dq_kernel(const float * __restrict__ Q,
                                                    const float * __restrict__ K,
                                                    const float * __restrict__ V,
                                                    const float * __restrict__ dO,
                                                    const float * __restrict__ L,
                                                    const float * __restrict__ Di,
                                                    float * __restrict__ dQ,
                                                    const int seq_len,
                                                    const float scale,
                                                    const int causal)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;

  const int q_row=q_block_idx*FA_BWD_DQ_BR+tid;

  if(q_row>=seq_len)
  {
    return;
  }

  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+FA_BWD_DQ_BC*D;

  const float *Q_bh=Q+bh*seq_len*D;
  const float *K_bh=K+bh*seq_len*D;
  const float *V_bh=V+bh*seq_len*D;
  const float *dO_bh=dO+bh*seq_len*D;
  const float *L_bh=L+bh*seq_len;
  const float *Di_bh=Di+bh*seq_len;
  float *dQ_bh=dQ+bh*seq_len*D;

  // Load Q row and dO row (sized exactly to D)
  float Q_row[D];
  float dO_row[D];
  for(int d=0;d<D;++d)
  {
    Q_row[d]=Q_bh[q_row*D+d];
    dO_row[d]=dO_bh[q_row*D+d];
  }

  const float L_val=L_bh[q_row];
  const float Di_val=Di_bh[q_row];

  float dQ_acc[D];
  for(int d=0;d<D;++d)
  {
    dQ_acc[d]=0.0f;
  }

  int num_kv_blocks=(seq_len+FA_BWD_DQ_BC-1)/FA_BWD_DQ_BC;
  if(causal)
  {
    num_kv_blocks=min(num_kv_blocks,(q_row/FA_BWD_DQ_BC)+1);
  }

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*FA_BWD_DQ_BC;

    // Load K and V tiles cooperatively
    for(int i=tid;i<FA_BWD_DQ_BC*D;i+=FA_BWD_DQ_BR)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;

      if(global_row<seq_len)
      {
        K_tile[row*D+col]=K_bh[global_row*D+col];
        V_tile[row*D+col]=V_bh[global_row*D+col];
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    for(int j=0;j<FA_BWD_DQ_BC;++j)
    {
      const int k_row=kv_start+j;

      if(causal&&k_row>q_row)
      {
        continue;
      }

      if(k_row>=seq_len)
      {
        continue;
      }

      float dot=0.0f;
      for(int d=0;d<D;++d)
      {
        dot+=Q_row[d]*K_tile[j*D+d];
      }
      const float S_val=dot*scale;
      const float P_val=expf(fminf(S_val-L_val,0.0f));

      float dP=0.0f;
      for(int d=0;d<D;++d)
      {
        dP+=dO_row[d]*V_tile[j*D+d];
      }

      const float dS=P_val*(dP-Di_val)*scale;

      for(int d=0;d<D;++d)
      {
        dQ_acc[d]+=dS*K_tile[j*D+d];
      }
    }
    __syncthreads();
  }

  for(int d=0;d<D;++d)
  {
    dQ_bh[q_row*D+d]=dQ_acc[d];
  }
}

extern "C"
{

// Launch wrapper for FlashAttention backward
void launch_flash_attention_backward(const float *Q,
                                     const float *K,
                                     const float *V,
                                     const float *O,
                                     const float *dO,
                                     const float *L,
                                     float *dQ,
                                     float *dK,
                                     float *dV,
                                     const int batch_heads,
                                     const int seq_len,
                                     const int head_dim,
                                     const float scale,
                                     const int causal,
                                     cudaStream_t stream)
{
  // Allocate temporary Di buffer [batch_heads, seq_len]
  float *Di_buf=nullptr;
  cudaMallocAsync(reinterpret_cast<void **>(&Di_buf),
                  static_cast<size_t>(batch_heads)*seq_len*sizeof(float),stream);

  // Kernel 0: Precompute Di = dot(dO, O) for each row
  {
    const int block_size=256;
    const int rows_per_grid=(seq_len+block_size-1)/block_size;
    dim3 grid(batch_heads,rows_per_grid);

    switch(head_dim)
    {
      case 32:
        flash_attention_precompute_di_kernel<32><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 64:
        flash_attention_precompute_di_kernel<64><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 80:
        flash_attention_precompute_di_kernel<80><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 96:
        flash_attention_precompute_di_kernel<96><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 128:
        flash_attention_precompute_di_kernel<128><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 1: Compute dK and dV (128 threads/block)
  {
    const int num_kv_blocks=(seq_len+FA_BWD_BC-1)/FA_BWD_BC;
    dim3 grid(batch_heads,num_kv_blocks);
    dim3 block(FA_BWD_BC);
    const size_t smem_size=(2*FA_BWD_BR*head_dim+2*FA_BWD_BR)*sizeof(float);

    switch(head_dim)
    {
      case 32:
        flash_attention_backward_kernel<32><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,seq_len,scale,causal);
        break;
      case 64:
        flash_attention_backward_kernel<64><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,seq_len,scale,causal);
        break;
      case 80:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<80><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,seq_len,scale,causal);
        break;
      case 96:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<96><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,seq_len,scale,causal);
        break;
      case 128:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<128><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,seq_len,scale,causal);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 2: Compute dQ (128 threads/block)
  {
    const int num_q_blocks=(seq_len+FA_BWD_DQ_BR-1)/FA_BWD_DQ_BR;
    dim3 grid(batch_heads,num_q_blocks);
    dim3 block(FA_BWD_DQ_BR);
    const size_t smem_size=2*FA_BWD_DQ_BC*head_dim*sizeof(float);

    switch(head_dim)
    {
      case 32:
        flash_attention_backward_dq_kernel<32><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,seq_len,scale,causal);
        break;
      case 64:
        flash_attention_backward_dq_kernel<64><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,seq_len,scale,causal);
        break;
      case 80:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<80><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,seq_len,scale,causal);
        break;
      case 96:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<96><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,seq_len,scale,causal);
        break;
      case 128:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<128><<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,seq_len,scale,causal);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_dq unsupported head_dim=%d\n",
                head_dim);
        abort();
        break;
    }
  }

  // Free temporary Di buffer (enqueue on stream for proper ordering)
  cudaFreeAsync(Di_buf,stream);
}

//------------------------------------------------------------------------------
// Data type conversion kernels (FP32 <-> FP16, FP32 <-> BF16)
//------------------------------------------------------------------------------
constexpr int g_convert_block_size=256;

__global__ void convert_fp32_to_fp16_kernel(const float *input,
                                            __half *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__float2half(input[idx]);
  }
}

void launch_convert_fp32_to_fp16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_convert_block_size-1)/g_convert_block_size;
  convert_fp32_to_fp16_kernel<<<grid,g_convert_block_size,0,stream>>>(input,static_cast<__half*>(output),n);
}

__global__ void convert_fp16_to_fp32_kernel(const __half *input,
                                            float *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__half2float(input[idx]);
  }
}

void launch_convert_fp16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_convert_block_size-1)/g_convert_block_size;
  convert_fp16_to_fp32_kernel<<<grid,g_convert_block_size,0,stream>>>(
      static_cast<const __half*>(input),output,n);
}

__global__ void convert_fp32_to_bf16_kernel(const float *input,
                                            __nv_bfloat16 *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__float2bfloat16(input[idx]);
  }
}

void launch_convert_fp32_to_bf16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_convert_block_size-1)/g_convert_block_size;
  convert_fp32_to_bf16_kernel<<<grid,g_convert_block_size,0,stream>>>(
      input,static_cast<__nv_bfloat16*>(output),n);
}

__global__ void convert_bf16_to_fp32_kernel(const __nv_bfloat16 *input,
                                            float *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__bfloat162float(input[idx]);
  }
}

void launch_convert_bf16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_convert_block_size-1)/g_convert_block_size;
  convert_bf16_to_fp32_kernel<<<grid,g_convert_block_size,0,stream>>>(
      static_cast<const __nv_bfloat16*>(input),output,n);
}

//------------------------------------------------------------------------------
// INT8 conversion kernels
//------------------------------------------------------------------------------

__global__ void convert_fp32_to_int8_kernel(const float *input,
                                            int8_t *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float val=input[idx];
    if(val>127.0f)
    {
      val=127.0f;
    }
    else if(val<-127.0f)
    {
      val=-127.0f;
    }
    output[idx]=static_cast<int8_t>(rintf(val));
  }
}

void launch_convert_fp32_to_int8(const float *input,
                                  void *output,
                                  int n,
                                  cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_convert_block_size-1)/g_convert_block_size;
  convert_fp32_to_int8_kernel<<<grid,g_convert_block_size,0,stream>>>(
      input,static_cast<int8_t*>(output),n);
}

__global__ void convert_int8_to_fp32_kernel(const int8_t *input,
                                            float *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=static_cast<float>(input[idx]);
  }
}

void launch_convert_int8_to_fp32(const void *input,
                                  float *output,
                                  int n,
                                  cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_convert_block_size-1)/g_convert_block_size;
  convert_int8_to_fp32_kernel<<<grid,g_convert_block_size,0,stream>>>(
      static_cast<const int8_t*>(input),output,n);
}

//------------------------------------------------------------------------------
// INT4 quantization kernels (symmetric per-group with FP16 scales)
//------------------------------------------------------------------------------

__global__ void dequantize_int4_kernel(const uint8_t *packed_data,
                                       const __half *scales,
                                       float *output,
                                       const int num_elements,
                                       const int group_size)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<num_elements)
  {
    const int byte_idx=idx/2;
    const uint8_t packed=packed_data[byte_idx];

    int int4_val;
    if((idx&1)==0)
    {
      int4_val=packed&0x0F;
    }
    else
    {
      int4_val=(packed>>4)&0x0F;
    }

    // Sign extend: if bit 3 is set, value is negative
    if((int4_val&0x08)!=0)
    {
      int4_val|=static_cast<int>(0xFFFFFFF0);
    }

    const int group_idx=idx/group_size;
    const float scale=__half2float(scales[group_idx]);
    output[idx]=static_cast<float>(int4_val)*scale;
  }
}

void launch_dequantize_int4(const void *packed_data,
                             const void *scales,
                             float *output,
                             int num_elements,
                             int group_size,
                             cudaStream_t stream)
{
  if(num_elements<=0)
  {
    return;
  }
  const int grid=(num_elements+g_convert_block_size-1)/g_convert_block_size;
  dequantize_int4_kernel<<<grid,g_convert_block_size,0,stream>>>(
      static_cast<const uint8_t*>(packed_data),
      static_cast<const __half*>(scales),
      output,
      num_elements,
      group_size);
}

__global__ void quantize_to_int4_kernel(const float *input,
                                        uint8_t *packed_output,
                                        __half *scales_output,
                                        const int num_elements,
                                        const int group_size)
{
  const int group_idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int num_groups=(num_elements+group_size-1)/group_size;
  if(group_idx>=num_groups)
  {
    return;
  }

  const int group_start=group_idx*group_size;
  int group_end=group_start+group_size;
  if(group_end>num_elements)
  {
    group_end=num_elements;
  }

  // Find max absolute value in group
  float max_abs=0.0f;
  for(int i=group_start;i<group_end;++i)
  {
    float abs_val=fabsf(input[i]);
    if(abs_val>max_abs)
    {
      max_abs=abs_val;
    }
  }

  // Compute scale: scale = max_abs / 7.0f
  float scale;
  if(max_abs==0.0f)
  {
    scale=0.0f;
  }
  else
  {
    scale=max_abs/7.0f;
  }
  scales_output[group_idx]=__float2half(scale);

  // Quantize and pack 2 elements per byte
  float inv_scale;
  if(scale==0.0f)
  {
    inv_scale=0.0f;
  }
  else
  {
    inv_scale=1.0f/scale;
  }

  for(int i=group_start;i<group_end;i+=2)
  {
    float v0=input[i]*inv_scale;
    if(v0>7.0f)
    {
      v0=7.0f;
    }
    else if(v0<-7.0f)
    {
      v0=-7.0f;
    }
    int q0=static_cast<int>(rintf(v0))&0x0F;

    int q1=0;
    if(i+1<group_end)
    {
      float v1=input[i+1]*inv_scale;
      if(v1>7.0f)
      {
        v1=7.0f;
      }
      else if(v1<-7.0f)
      {
        v1=-7.0f;
      }
      q1=static_cast<int>(rintf(v1))&0x0F;
    }

    packed_output[i/2]=static_cast<uint8_t>(q0|(q1<<4));
  }
}

void launch_quantize_to_int4(const float *input,
                              void *packed_output,
                              void *scales_output,
                              int num_elements,
                              int group_size,
                              cudaStream_t stream)
{
  if(num_elements<=0)
  {
    return;
  }
  const int num_groups=(num_elements+group_size-1)/group_size;
  const int grid=(num_groups+g_convert_block_size-1)/g_convert_block_size;
  quantize_to_int4_kernel<<<grid,g_convert_block_size,0,stream>>>(
      input,
      static_cast<uint8_t*>(packed_output),
      static_cast<__half*>(scales_output),
      num_elements,
      group_size);
}

//------------------------------------------------------------------------------
// Tensor slice and concatenation kernels
//------------------------------------------------------------------------------
constexpr int g_slice_block_size=256;

__global__ void slice_last_dim_kernel(const float *input,
                                      float *output,
                                      const int rows,
                                      const int in_cols,
                                      const int col_start,
                                      const int out_cols)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=rows*out_cols;
  if(idx<total)
  {
    const int row=idx/out_cols;
    const int col=idx%out_cols;
    output[idx]=input[row*in_cols+col_start+col];
  }
}

void launch_slice_last_dim(const float *input,
                           float *output,
                           int rows,
                           int in_cols,
                           int col_start,
                           int out_cols,
                           cudaStream_t stream)
{
  const int total=rows*out_cols;
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_slice_block_size-1)/g_slice_block_size;
  slice_last_dim_kernel<<<grid,g_slice_block_size,0,stream>>>(input,output,rows,in_cols,col_start,out_cols);
}

__global__ void slice_last_dim_backward_kernel(const float *grad_output,
                                               float *grad_input,
                                               const int rows,
                                               const int in_cols,
                                               const int col_start,
                                               const int out_cols)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=rows*out_cols;
  if(idx<total)
  {
    const int row=idx/out_cols;
    const int col=idx%out_cols;
    grad_input[row*in_cols+col_start+col]+=grad_output[idx];
  }
}

void launch_slice_last_dim_backward(const float *grad_output,
                                    float *grad_input,
                                    int rows,
                                    int in_cols,
                                    int col_start,
                                    int out_cols,
                                    cudaStream_t stream)
{
  const int total=rows*out_cols;
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_slice_block_size-1)/g_slice_block_size;
  slice_last_dim_backward_kernel<<<grid,g_slice_block_size,0,stream>>>(
      grad_output,grad_input,rows,in_cols,col_start,out_cols);
}

__global__ void concat_last_dim_kernel(const float *a,
                                       const float *b,
                                       float *output,
                                       const int rows,
                                       const int cols_a,
                                       const int cols_b)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total_cols=cols_a+cols_b;
  const int total=rows*total_cols;
  if(idx<total)
  {
    const int row=idx/total_cols;
    const int col=idx%total_cols;
    if(col<cols_a)
    {
      output[idx]=a[row*cols_a+col];
    }
    else
    {
      output[idx]=b[row*cols_b+(col-cols_a)];
    }
  }
}

void launch_concat_last_dim(const float *a,
                            const float *b,
                            float *output,
                            int rows,
                            int cols_a,
                            int cols_b,
                            cudaStream_t stream)
{
  const int total=rows*(cols_a+cols_b);
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_slice_block_size-1)/g_slice_block_size;
  concat_last_dim_kernel<<<grid,g_slice_block_size,0,stream>>>(a,b,output,rows,cols_a,cols_b);
}

}  // extern "C"

