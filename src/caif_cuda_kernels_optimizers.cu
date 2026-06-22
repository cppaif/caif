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
// Fused optimizer CUDA kernels: Adam / AdamW, clipped Adam, SGD,
// SGD+momentum, RMSprop, AdaGrad.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_optimizers.cuh
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

//------------------------------------------------------------------------------
// Fused Adam Optimizer Kernel
// Combines all Adam operations into a single kernel for efficiency:
// AdamW (decoupled weight decay, the Loshchilov & Hutter formulation):
// 1. Update first moment: m = beta1 * m + (1 - beta1) * g
// 2. Update second moment: v = beta2 * v + (1 - beta2) * g^2
// 3. Compute bias-corrected moments
// 4. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
// 5. Apply decoupled weight decay: param = param - lr * weight_decay * param
//------------------------------------------------------------------------------
// param/grad in storage T (float/__half/__nv_bfloat16); m/v always fp32 master.
// All math is fp32 — only the load/store of param and grad cross precisions.
template<typename T>
__global__ void fused_adam_kernel(T *param,
                                  const T *grad,
                                  float *m,
                                  float *v,
                                  const float lr,
                                  const float beta1,
                                  const float beta2,
                                  const float epsilon,
                                  const float weight_decay,
                                  const float bias_correction1,
                                  const float bias_correction2,
                                  const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    const float g=caif_load_f<T>(grad[idx]);
    float m_val=m[idx];
    float v_val=v[idx];

    // Decoupled weight decay (AdamW): shrink the weight BEFORE the adaptive
    // step, matching torch.optim.AdamW. NaN/Inf gradients are intentionally
    // NOT sanitized so divergence stays visible and a mixed-precision loss
    // scaler can detect overflow and skip the step.
    if(weight_decay!=0.0f)
    {
      p=p*(1.0f-lr*weight_decay);
    }

    m_val=beta1*m_val+(1.0f-beta1)*g;
    v_val=beta2*v_val+(1.0f-beta2)*g*g;

    const float m_hat=m_val/bias_correction1;
    const float v_hat=v_val/bias_correction2;

    p=p-lr*m_hat/(sqrtf(v_hat)+epsilon);

    param[idx]=caif_store_f<T>(p);
    m[idx]=m_val;
    v[idx]=v_val;
  }
}

//------------------------------------------------------------------------------
// Fused SGD with Momentum Kernel
// Combines momentum update and parameter update
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_sgd_momentum_kernel(T *param,
                                          const T *grad,
                                          T *velocity,
                                          const float lr,
                                          const float momentum,
                                          const float weight_decay,
                                          const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    float v=caif_load_f<T>(velocity[idx]);

    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }

    v=momentum*v+g;
    p=p-lr*v;

    param[idx]=caif_store_f<T>(p);
    velocity[idx]=caif_store_f<T>(v);
  }
}

//------------------------------------------------------------------------------
// Fused plain-SGD kernel (no momentum, no velocity buffer)
// param = param - lr * (grad + weight_decay * param)
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_sgd_kernel(T *param,
                                 const T *grad,
                                 const float lr,
                                 const float weight_decay,
                                 const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    param[idx]=caif_store_f<T>(p-lr*g);
  }
}

//------------------------------------------------------------------------------
// Fused RMSprop kernel
// avg_sq = alpha * avg_sq + (1 - alpha) * grad^2
// param  = param - lr * (grad + wd * param) / (sqrt(avg_sq) + epsilon)
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_rmsprop_kernel(T *param,
                                     const T *grad,
                                     T *avg_sq,
                                     const float lr,
                                     const float alpha,
                                     const float epsilon,
                                     const float weight_decay,
                                     const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    float a=caif_load_f<T>(avg_sq[idx]);
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    a=alpha*a+(1.0f-alpha)*g*g;
    param[idx]=caif_store_f<T>(p-lr*g/(sqrtf(a)+epsilon));
    avg_sq[idx]=caif_store_f<T>(a);
  }
}

//------------------------------------------------------------------------------
// Fused AdaGrad kernel
// accum = accum + grad^2
// param = param - lr * (grad + wd * param) / (sqrt(accum) + epsilon)
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_adagrad_kernel(T *param,
                                     const T *grad,
                                     T *accum,
                                     const float lr,
                                     const float epsilon,
                                     const float weight_decay,
                                     const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    float a=caif_load_f<T>(accum[idx]);
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    a=a+g*g;
    param[idx]=caif_store_f<T>(p-lr*g/(sqrtf(a)+epsilon));
    accum[idx]=caif_store_f<T>(a);
  }
}

//------------------------------------------------------------------------------
// Optimizer Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_fused_adam(T *param,
                       const T *grad,
                       float *m,
                       float *v,
                       const float lr,
                       const float beta1,
                       const float beta2,
                       const float epsilon,
                       const float weight_decay,
                       const float bias_correction1,
                       const float bias_correction2,
                       const int64_t n,
                       cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  fused_adam_kernel<T><<<num_blocks,block_size,0,stream>>>(param,
                                                           grad,
                                                           m,
                                                           v,
                                                           lr,
                                                           beta1,
                                                           beta2,
                                                           epsilon,
                                                           weight_decay,
                                                           bias_correction1,
                                                           bias_correction2,
                                                           n);
}

template void launch_fused_adam<float>(float *,
                                       const float *,
                                       float *,
                                       float *,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const int64_t,
                                       cudaStream_t);
template void launch_fused_adam<__half>(__half *,
                                        const __half *,
                                        float *,
                                        float *,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const int64_t,
                                        cudaStream_t);
template void launch_fused_adam<__nv_bfloat16>(__nv_bfloat16 *,
                                               const __nv_bfloat16 *,
                                               float *,
                                               float *,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const int64_t,
                                               cudaStream_t);

//------------------------------------------------------------------------------
// Mixed-precision loss-scaler support.
// Unscale a gradient in place (grad *= inv_scale) and flag overflow: if any
// element is non-finite, found_inf[0] is set to 1.0f. The flag write is a plain
// store of a constant, so it is race-free without atomics — every writer stores
// the same value and the launcher only ever sets the flag, never clears it, so
// across the per-parameter passes (serialized on one stream) the flag ends up 1
// iff ANY parameter overflowed. Math is fp32 (load -> scale -> isfinite ->
// store) so the check catches overflow even for half/bf16 grads. The caller
// (CAIF_LossScaler) reads the flag after the sweep and skips the optimizer step
// when it is set, then backs the scale off — the GradScaler protocol.
//------------------------------------------------------------------------------
template<typename T>
__global__ void unscale_check_inf_kernel(T *grad,
                                         const float inv_scale,
                                         float *found_inf,
                                         const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float g=caif_load_f<T>(grad[idx])*inv_scale;
    if(isfinite(g)==false)
    {
      found_inf[0]=1.0f;
    }
    grad[idx]=caif_store_f<T>(g);
  }
}

template<typename T>
void launch_unscale_check_inf(T *grad,
                              const float inv_scale,
                              float *found_inf,
                              const int64_t n,
                              cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  unscale_check_inf_kernel<T><<<num_blocks,block_size,0,stream>>>(grad,
                                                                  inv_scale,
                                                                  found_inf,
                                                                  n);
}

template void launch_unscale_check_inf<float>(float *,
                                              const float,
                                              float *,
                                              const int64_t,
                                              cudaStream_t);
template void launch_unscale_check_inf<__half>(__half *,
                                               const float,
                                               float *,
                                               const int64_t,
                                               cudaStream_t);
template void launch_unscale_check_inf<__nv_bfloat16>(__nv_bfloat16 *,
                                                      const float,
                                                      float *,
                                                      const int64_t,
                                                      cudaStream_t);

//------------------------------------------------------------------------------
// Shared multi-tensor ("foreach") helper: binary-search the element-count
// prefix sum `offsets` (length num_tensors+1) for the tensor t that owns global
// index gidx, i.e. offsets[t] <= gidx < offsets[t+1]. Every multi-tensor
// optimizer kernel below uses this so the search lives in exactly one place.
//------------------------------------------------------------------------------
__device__ __forceinline__ int64_t mt_find_tensor(const int64_t *offsets,
                                                  const int num_tensors,
                                                  const int64_t gidx)
{
  int lo=0;
  int hi=num_tensors;
  while(lo+1<hi)
  {
    const int mid=(lo+hi)>>1;
    if(offsets[mid]<=gidx)
    {
      lo=mid;
    }
    else
    {
      hi=mid;
    }
  }
  return lo;
}

//------------------------------------------------------------------------------
// Multi-tensor ("foreach") fused Adam — one kernel launch updates EVERY
// trainable parameter, replacing the one-launch-per-parameter path whose
// per-launch overhead dominates the optimizer step for models with many small
// tensors. params/grads/ms/vs are arrays of device pointers (one entry per
// parameter tensor); `offsets` is the element-count prefix sum (length
// num_tensors+1, offsets[0]==0, offsets[num_tensors]==total_elements) so each
// thread maps its global index to (tensor t, local index j) with one binary
// search. The per-element update math is identical to fused_adam_kernel —
// reused verbatim — so numerics do not change, only the launch count.
//------------------------------------------------------------------------------
template<typename T>
__global__ void multi_tensor_adam_kernel(T *const *params,
                                         const T *const *grads,
                                         float *const *ms,
                                         float *const *vs,
                                         const int64_t *offsets,
                                         const int num_tensors,
                                         const int64_t total_elements,
                                         const float lr,
                                         const float beta1,
                                         const float beta2,
                                         const float epsilon,
                                         const float weight_decay,
                                         const float bias_correction1,
                                         const float bias_correction2)
{
  const int64_t stride=static_cast<int64_t>(gridDim.x)*blockDim.x;
  for(int64_t gidx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;gidx<total_elements;gidx+=stride)
  {
    const int64_t t=mt_find_tensor(offsets,num_tensors,gidx);
    const int64_t j=gidx-offsets[t];

    T *param=params[t];
    const T *grad=grads[t];
    float *m=ms[t];
    float *v=vs[t];

    float p=caif_load_f<T>(param[j]);
    const float g=caif_load_f<T>(grad[j]);
    float m_val=m[j];
    float v_val=v[j];

    if(weight_decay!=0.0f)
    {
      p=p*(1.0f-lr*weight_decay);
    }

    m_val=beta1*m_val+(1.0f-beta1)*g;
    v_val=beta2*v_val+(1.0f-beta2)*g*g;

    const float m_hat=m_val/bias_correction1;
    const float v_hat=v_val/bias_correction2;

    p=p-lr*m_hat/(sqrtf(v_hat)+epsilon);

    param[j]=caif_store_f<T>(p);
    m[j]=m_val;
    v[j]=v_val;
  }
}

template<typename T>
void launch_multi_tensor_adam(T *const *params,
                              const T *const *grads,
                              float *const *ms,
                              float *const *vs,
                              const int64_t *offsets,
                              const int num_tensors,
                              const int64_t total_elements,
                              const float lr,
                              const float beta1,
                              const float beta2,
                              const float epsilon,
                              const float weight_decay,
                              const float bias_correction1,
                              const float bias_correction2,
                              cudaStream_t stream)
{
  if(total_elements<=0)
  {
    return;
  }
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total_elements+block_size-1)/block_size);
  multi_tensor_adam_kernel<T><<<num_blocks,block_size,0,stream>>>(params,
                                                                  grads,
                                                                  ms,
                                                                  vs,
                                                                  offsets,
                                                                  num_tensors,
                                                                  total_elements,
                                                                  lr,
                                                                  beta1,
                                                                  beta2,
                                                                  epsilon,
                                                                  weight_decay,
                                                                  bias_correction1,
                                                                  bias_correction2);
}

// Only float is instantiated: the optimizer always updates an fp32 target (the
// param itself when it is fp32, otherwise its fp32 master copy) with an fp32
// gradient, so the batched update is always fp32. Storage-dtype conversion is a
// separate cast, exactly as in the per-parameter path.
template void launch_multi_tensor_adam<float>(float *const *,
                                              const float *const *,
                                              float *const *,
                                              float *const *,
                                              const int64_t *,
                                              const int,
                                              const int64_t,
                                              const float,
                                              const float,
                                              const float,
                                              const float,
                                              const float,
                                              const float,
                                              const float,
                                              cudaStream_t);

//------------------------------------------------------------------------------
// Multi-tensor plain SGD: param -= lr*(grad + wd*param). No state. Mirrors
// fused_sgd_kernel.
//------------------------------------------------------------------------------
template<typename T>
__global__ void multi_tensor_sgd_kernel(T *const *params,
                                        const T *const *grads,
                                        const int64_t *offsets,
                                        const int num_tensors,
                                        const int64_t total_elements,
                                        const float lr,
                                        const float weight_decay)
{
  const int64_t stride=static_cast<int64_t>(gridDim.x)*blockDim.x;
  for(int64_t gidx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;gidx<total_elements;gidx+=stride)
  {
    const int64_t t=mt_find_tensor(offsets,num_tensors,gidx);
    const int64_t j=gidx-offsets[t];
    T *param=params[t];
    float p=caif_load_f<T>(param[j]);
    float g=caif_load_f<T>(grads[t][j]);
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    param[j]=caif_store_f<T>(p-lr*g);
  }
}

template<typename T>
void launch_multi_tensor_sgd(T *const *params,
                             const T *const *grads,
                             const int64_t *offsets,
                             const int num_tensors,
                             const int64_t total_elements,
                             const float lr,
                             const float weight_decay,
                             cudaStream_t stream)
{
  if(total_elements<=0)
  {
    return;
  }
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total_elements+block_size-1)/block_size);
  multi_tensor_sgd_kernel<T><<<num_blocks,block_size,0,stream>>>(params,
                                                                 grads,
                                                                 offsets,
                                                                 num_tensors,
                                                                 total_elements,
                                                                 lr,
                                                                 weight_decay);
}

template void launch_multi_tensor_sgd<float>(float *const *,
                                             const float *const *,
                                             const int64_t *,
                                             const int,
                                             const int64_t,
                                             const float,
                                             const float,
                                             cudaStream_t);

//------------------------------------------------------------------------------
// Multi-tensor SGD with momentum. State: velocity (fp32). Mirrors
// fused_sgd_momentum_kernel (velocity is fp32 here, matching AllocateState).
//------------------------------------------------------------------------------
template<typename T>
__global__ void multi_tensor_momentum_kernel(T *const *params,
                                             const T *const *grads,
                                             float *const *velocities,
                                             const int64_t *offsets,
                                             const int num_tensors,
                                             const int64_t total_elements,
                                             const float lr,
                                             const float momentum,
                                             const float weight_decay)
{
  const int64_t stride=static_cast<int64_t>(gridDim.x)*blockDim.x;
  for(int64_t gidx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;gidx<total_elements;gidx+=stride)
  {
    const int64_t t=mt_find_tensor(offsets,num_tensors,gidx);
    const int64_t j=gidx-offsets[t];
    T *param=params[t];
    float *velocity=velocities[t];
    float p=caif_load_f<T>(param[j]);
    float g=caif_load_f<T>(grads[t][j]);
    float v=velocity[j];
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    v=momentum*v+g;
    param[j]=caif_store_f<T>(p-lr*v);
    velocity[j]=v;
  }
}

template<typename T>
void launch_multi_tensor_momentum(T *const *params,
                                  const T *const *grads,
                                  float *const *velocities,
                                  const int64_t *offsets,
                                  const int num_tensors,
                                  const int64_t total_elements,
                                  const float lr,
                                  const float momentum,
                                  const float weight_decay,
                                  cudaStream_t stream)
{
  if(total_elements<=0)
  {
    return;
  }
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total_elements+block_size-1)/block_size);
  multi_tensor_momentum_kernel<T><<<num_blocks,block_size,0,stream>>>(params,
                                                                      grads,
                                                                      velocities,
                                                                      offsets,
                                                                      num_tensors,
                                                                      total_elements,
                                                                      lr,
                                                                      momentum,
                                                                      weight_decay);
}

template void launch_multi_tensor_momentum<float>(float *const *,
                                                  const float *const *,
                                                  float *const *,
                                                  const int64_t *,
                                                  const int,
                                                  const int64_t,
                                                  const float,
                                                  const float,
                                                  const float,
                                                  cudaStream_t);

//------------------------------------------------------------------------------
// Multi-tensor RMSprop. State: avg_sq (fp32). Mirrors fused_rmsprop_kernel.
//------------------------------------------------------------------------------
template<typename T>
__global__ void multi_tensor_rmsprop_kernel(T *const *params,
                                            const T *const *grads,
                                            float *const *avg_sqs,
                                            const int64_t *offsets,
                                            const int num_tensors,
                                            const int64_t total_elements,
                                            const float lr,
                                            const float alpha,
                                            const float epsilon,
                                            const float weight_decay)
{
  const int64_t stride=static_cast<int64_t>(gridDim.x)*blockDim.x;
  for(int64_t gidx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;gidx<total_elements;gidx+=stride)
  {
    const int64_t t=mt_find_tensor(offsets,num_tensors,gidx);
    const int64_t j=gidx-offsets[t];
    T *param=params[t];
    float *avg_sq=avg_sqs[t];
    float p=caif_load_f<T>(param[j]);
    float g=caif_load_f<T>(grads[t][j]);
    float a=avg_sq[j];
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    a=alpha*a+(1.0f-alpha)*g*g;
    param[j]=caif_store_f<T>(p-lr*g/(sqrtf(a)+epsilon));
    avg_sq[j]=a;
  }
}

template<typename T>
void launch_multi_tensor_rmsprop(T *const *params,
                                 const T *const *grads,
                                 float *const *avg_sqs,
                                 const int64_t *offsets,
                                 const int num_tensors,
                                 const int64_t total_elements,
                                 const float lr,
                                 const float alpha,
                                 const float epsilon,
                                 const float weight_decay,
                                 cudaStream_t stream)
{
  if(total_elements<=0)
  {
    return;
  }
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total_elements+block_size-1)/block_size);
  multi_tensor_rmsprop_kernel<T><<<num_blocks,block_size,0,stream>>>(params,
                                                                     grads,
                                                                     avg_sqs,
                                                                     offsets,
                                                                     num_tensors,
                                                                     total_elements,
                                                                     lr,
                                                                     alpha,
                                                                     epsilon,
                                                                     weight_decay);
}

template void launch_multi_tensor_rmsprop<float>(float *const *,
                                                 const float *const *,
                                                 float *const *,
                                                 const int64_t *,
                                                 const int,
                                                 const int64_t,
                                                 const float,
                                                 const float,
                                                 const float,
                                                 const float,
                                                 cudaStream_t);

//------------------------------------------------------------------------------
// Multi-tensor AdaGrad. State: accum (fp32). Mirrors fused_adagrad_kernel.
//------------------------------------------------------------------------------
template<typename T>
__global__ void multi_tensor_adagrad_kernel(T *const *params,
                                            const T *const *grads,
                                            float *const *accums,
                                            const int64_t *offsets,
                                            const int num_tensors,
                                            const int64_t total_elements,
                                            const float lr,
                                            const float epsilon,
                                            const float weight_decay)
{
  const int64_t stride=static_cast<int64_t>(gridDim.x)*blockDim.x;
  for(int64_t gidx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;gidx<total_elements;gidx+=stride)
  {
    const int64_t t=mt_find_tensor(offsets,num_tensors,gidx);
    const int64_t j=gidx-offsets[t];
    T *param=params[t];
    float *accum=accums[t];
    float p=caif_load_f<T>(param[j]);
    float g=caif_load_f<T>(grads[t][j]);
    float a=accum[j];
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    a=a+g*g;
    param[j]=caif_store_f<T>(p-lr*g/(sqrtf(a)+epsilon));
    accum[j]=a;
  }
}

template<typename T>
void launch_multi_tensor_adagrad(T *const *params,
                                 const T *const *grads,
                                 float *const *accums,
                                 const int64_t *offsets,
                                 const int num_tensors,
                                 const int64_t total_elements,
                                 const float lr,
                                 const float epsilon,
                                 const float weight_decay,
                                 cudaStream_t stream)
{
  if(total_elements<=0)
  {
    return;
  }
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total_elements+block_size-1)/block_size);
  multi_tensor_adagrad_kernel<T><<<num_blocks,block_size,0,stream>>>(params,
                                                                     grads,
                                                                     accums,
                                                                     offsets,
                                                                     num_tensors,
                                                                     total_elements,
                                                                     lr,
                                                                     epsilon,
                                                                     weight_decay);
}

template void launch_multi_tensor_adagrad<float>(float *const *,
                                                 const float *const *,
                                                 float *const *,
                                                 const int64_t *,
                                                 const int,
                                                 const int64_t,
                                                 const float,
                                                 const float,
                                                 const float,
                                                 cudaStream_t);

template<typename T>
void launch_fused_sgd_momentum(T *param,
                               const T *grad,
                               T *velocity,
                               const float lr,
                               const float momentum,
                               const float weight_decay,
                               const int64_t n,
                               cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  fused_sgd_momentum_kernel<T><<<num_blocks,block_size,0,stream>>>(param,
                                                                   grad,
                                                                   velocity,
                                                                   lr,
                                                                   momentum,
                                                                   weight_decay,
                                                                   n);
}

template void launch_fused_sgd_momentum<float>(float *,
                                               const float *,
                                               float *,
                                               const float,
                                               const float,
                                               const float,
                                               const int64_t,
                                               cudaStream_t);
template void launch_fused_sgd_momentum<__half>(__half *,
                                                const __half *,
                                                __half *,
                                                const float,
                                                const float,
                                                const float,
                                                const int64_t,
                                                cudaStream_t);
template void launch_fused_sgd_momentum<__nv_bfloat16>(__nv_bfloat16 *,
                                                       const __nv_bfloat16 *,
                                                       __nv_bfloat16 *,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const int64_t,
                                                       cudaStream_t);

template<typename T>
void launch_fused_sgd(T *param,
                      const T *grad,
                      const float lr,
                      const float weight_decay,
                      const int64_t n,
                      cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  fused_sgd_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,lr,weight_decay,n);
}

template void launch_fused_sgd<float>(float *,const float *,const float,const float,const int64_t,cudaStream_t);
template void launch_fused_sgd<__half>(__half *,const __half *,const float,const float,const int64_t,cudaStream_t);
template void launch_fused_sgd<__nv_bfloat16>(__nv_bfloat16 *,
                                              const __nv_bfloat16 *,
                                              const float,
                                              const float,
                                              const int64_t,
                                              cudaStream_t);

template<typename T>
void launch_fused_rmsprop(T *param,
                          const T *grad,
                          T *avg_sq,
                          const float lr,
                          const float alpha,
                          const float epsilon,
                          const float weight_decay,
                          const int64_t n,
                          cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  fused_rmsprop_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,avg_sq,lr,alpha,epsilon,weight_decay,n);
}

template void launch_fused_rmsprop<float>(float *,
                                          const float *,
                                          float *,
                                          const float,
                                          const float,
                                          const float,
                                          const float,
                                          const int64_t,
                                          cudaStream_t);
template void launch_fused_rmsprop<__half>(__half *,
                                           const __half *,
                                           __half *,
                                           const float,
                                           const float,
                                           const float,
                                           const float,
                                           const int64_t,
                                           cudaStream_t);
template void launch_fused_rmsprop<__nv_bfloat16>(__nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  const float,
                                                  const float,
                                                  const float,
                                                  const float,
                                                  const int64_t,
                                                  cudaStream_t);

template<typename T>
void launch_fused_adagrad(T *param,
                          const T *grad,
                          T *accum,
                          const float lr,
                          const float epsilon,
                          const float weight_decay,
                          const int64_t n,
                          cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  fused_adagrad_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,accum,lr,epsilon,weight_decay,n);
}

template void launch_fused_adagrad<float>(float *,
                                          const float *,
                                          float *,
                                          const float,
                                          const float,
                                          const float,
                                          const int64_t,
                                          cudaStream_t);
template void launch_fused_adagrad<__half>(__half *,
                                           const __half *,
                                           __half *,
                                           const float,
                                           const float,
                                           const float,
                                           const int64_t,
                                           cudaStream_t);
template void launch_fused_adagrad<__nv_bfloat16>(__nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  const float,
                                                  const float,
                                                  const float,
                                                  const int64_t,
                                                  cudaStream_t);

