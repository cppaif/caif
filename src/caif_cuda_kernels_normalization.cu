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
// Normalization CUDA kernels: RMSNorm and LayerNorm, forward + backward.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_normalization.cuh
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
// RMSNorm Forward Kernel
// y[row][col] = x[row][col] / rms(x[row]) * gamma[col]
// rms(x) = sqrt(mean(x^2) + epsilon)
// One block per row, shared memory parallel reduction.
// Templated on activation T ∈ {float, __half, __nv_bfloat16}.
// gamma and rms_cache are always fp32 (standard autocast convention).
// All internal reductions are fp32; input loaded as T then cast.
//------------------------------------------------------------------------------
template<typename T>
__global__ void rmsnorm_forward_kernel(const T *input,
                                       const float *gamma,
                                       T *output,
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

  const T *x=input+static_cast<int64_t>(row)*dim;
  T *y=output+static_cast<int64_t>(row)*dim;
  const int tid=threadIdx.x;

  // Phase 1: Compute sum(x^2) using stride loop
  float local_sum=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float val=float(x[col]);
    local_sum+=val*val;
  }

  // Warp shuffle reduction
  local_sum=warp_reduce_sum(local_sum);

  // Per-warp partial sums via shared memory
  __shared__ float warp_sums[g_cu_warp_size];
  const int lane=tid&(g_cu_warp_size-1);
  const int warp_id=tid/g_cu_warp_size;
  const int num_warps=blockDim.x/g_cu_warp_size;

  if(lane==0)
  {
    warp_sums[warp_id]=local_sum;
  }
  __syncthreads();

  // First warp reduces the per-warp sums
  if(warp_id==0)
  {
    local_sum=0.0f;
    if(lane<num_warps)
    {
      local_sum=warp_sums[lane];
    }
    local_sum=warp_reduce_sum(local_sum);
  }

  // Broadcast result to all threads via shared memory
  if(tid==0)
  {
    warp_sums[0]=local_sum;
  }
  __syncthreads();

  // Compute RMS value
  const float rms=sqrtf(warp_sums[0]/static_cast<float>(dim)+epsilon);
  if(tid==0)
  {
    rms_cache[row]=rms;
  }

  const float rstd=1.0f/rms;

  // Phase 2: Normalize and scale by gamma
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    y[col]=T(float(x[col])*rstd*gamma[col]);
  }
}

//------------------------------------------------------------------------------
// RMSNorm Backward Kernel
// Computes grad_input and accumulates grad_gamma via atomicAdd.
// grad_gamma must be pre-zeroed before this kernel is launched.
// Templated on T ∈ {float, __half, __nv_bfloat16}. gamma / rms_cache /
// grad_gamma stay fp32 (standard autocast convention).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rmsnorm_backward_kernel(const T *grad_output,
                                        const T *input,
                                        const float *gamma,
                                        const float *rms_cache,
                                        T *grad_input,
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

  const T *dy=grad_output+static_cast<int64_t>(row)*dim;
  const T *x=input+static_cast<int64_t>(row)*dim;
  T *dx=grad_input+static_cast<int64_t>(row)*dim;
  const int tid=threadIdx.x;

  const float rstd=1.0f/rms_cache[row];

  // Phase 1: Compute sum(dy * gamma * x_hat) where x_hat = x * rstd
  float local_sum=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_hat=float(x[col])*rstd;
    local_sum+=float(dy[col])*gamma[col]*x_hat;
  }

  // Warp shuffle reduction
  local_sum=warp_reduce_sum(local_sum);

  __shared__ float warp_sums[g_cu_warp_size];
  const int lane=tid&(g_cu_warp_size-1);
  const int warp_id=tid/g_cu_warp_size;
  const int num_warps=blockDim.x/g_cu_warp_size;

  if(lane==0)
  {
    warp_sums[warp_id]=local_sum;
  }
  __syncthreads();

  if(warp_id==0)
  {
    local_sum=0.0f;
    if(lane<num_warps)
    {
      local_sum=warp_sums[lane];
    }
    local_sum=warp_reduce_sum(local_sum);
  }

  // Broadcast result
  if(tid==0)
  {
    warp_sums[0]=local_sum;
  }
  __syncthreads();

  const float sum_term=warp_sums[0]/static_cast<float>(dim);

  // Phase 2: Compute grad_input and accumulate grad_gamma
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_val=float(x[col]);
    const float dy_val=float(dy[col]);
    const float x_hat=x_val*rstd;
    dx[col]=T(rstd*(dy_val*gamma[col]-x_hat*sum_term));
    atomicAdd(&grad_gamma[col],dy_val*x_hat);
  }
}

//------------------------------------------------------------------------------
// LayerNorm Forward Kernel (Block-per-Row)
// y[row][col] = (x[row][col] - mean) / sqrt(var + eps) * gamma[col] + beta[col]
// One block per row, g_cu_block_size threads, shared memory parallel reduction.
// Mirrors the RMSNorm architecture for maximum memory bandwidth utilization.
//------------------------------------------------------------------------------
template<typename T>
__global__ void layernorm_forward_kernel(const T *input,
                                         const float *gamma,
                                         const float *beta,
                                         T *output,
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

  const T *x=input+static_cast<int64_t>(row)*dim;
  T *y=output+static_cast<int64_t>(row)*dim;
  const int tid=threadIdx.x;

  const float dim_f=static_cast<float>(dim);

  // Per-warp partial-sum scratch, reused across both reduction passes.
  __shared__ float ws[g_cu_warp_size];
  const int lane=tid&(g_cu_warp_size-1);
  const int warp_id=tid/g_cu_warp_size;
  const int num_warps=blockDim.x/g_cu_warp_size;

  // Pass 1 — mean = sum(x)/dim. A dedicated pass (rather than the one-pass
  // E[x^2]-E[x]^2) is what keeps the variance numerically stable: with a large
  // DC offset, sum(x^2) and mean^2 are both ~offset^2 and their fp32 difference
  // loses the (much smaller) true variance to catastrophic cancellation — it
  // can even go negative, leaving rsqrt on the +epsilon floor. PyTorch/HF
  // compute the variance the same two-pass way.
  float local_sum=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    local_sum+=float(x[col]);
  }
  local_sum=warp_reduce_sum(local_sum);
  if(lane==0)
  {
    ws[warp_id]=local_sum;
  }
  __syncthreads();

  __shared__ float s_mean;
  if(warp_id==0)
  {
    float v=0.0f;
    if(lane<num_warps)
    {
      v=ws[lane];
    }
    v=warp_reduce_sum(v);
    if(lane==0)
    {
      s_mean=v/dim_f;
    }
  }
  __syncthreads();
  const float mean=s_mean;

  // Pass 2 — var = sum((x-mean)^2)/dim from the centered values, so every
  // summand is O(spread^2) and no large-magnitude cancellation occurs. The
  // __syncthreads above also retires every Pass 1 read of ws[] before the
  // writes below reuse it.
  float local_sq=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float d=float(x[col])-mean;
    local_sq+=d*d;
  }
  local_sq=warp_reduce_sum(local_sq);
  if(lane==0)
  {
    ws[warp_id]=local_sq;
  }
  __syncthreads();

  __shared__ float s_rstd;
  if(warp_id==0)
  {
    float v=0.0f;
    if(lane<num_warps)
    {
      v=ws[lane];
    }
    v=warp_reduce_sum(v);
    if(lane==0)
    {
      const float variance=v/dim_f;
      s_rstd=rsqrtf(variance+epsilon);
    }
  }
  __syncthreads();
  const float rstd=s_rstd;

  if(tid==0)
  {
    mean_cache[row]=mean;
    rstd_cache[row]=rstd;
  }

  // Pass 3 — normalize, scale, and shift.
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_hat=(float(x[col])-mean)*rstd;
    y[col]=T(x_hat*gamma[col]+beta[col]);
  }
}

//------------------------------------------------------------------------------
// LayerNorm Backward Kernel
// Computes grad_input and accumulates grad_gamma/grad_beta via atomicAdd.
// grad_gamma and grad_beta must be pre-zeroed before this kernel is launched.
// Templated on T ∈ {float, __half, __nv_bfloat16}. gamma / beta / cache /
// grad_gamma / grad_beta stay fp32.
//------------------------------------------------------------------------------
template<typename T>
__global__ void layernorm_backward_kernel(const T *grad_output,
                                          const T *input,
                                          const float *gamma,
                                          const float *mean_cache,
                                          const float *rstd_cache,
                                          T *grad_input,
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

  const T *dy=grad_output+static_cast<int64_t>(row)*dim;
  const T *x=input+static_cast<int64_t>(row)*dim;
  T *dx=grad_input+static_cast<int64_t>(row)*dim;
  const int tid=threadIdx.x;

  const float mean=mean_cache[row];
  const float rstd=rstd_cache[row];

  // Phase 1: Compute S1=sum(dy*gamma) and S2=sum(dy*gamma*x_hat)
  float local_s1=0.0f;
  float local_s2=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_hat=(float(x[col])-mean)*rstd;
    const float dy_gamma=float(dy[col])*gamma[col];
    local_s1+=dy_gamma;
    local_s2+=dy_gamma*x_hat;
  }

  // Warp shuffle reduction for both sums
  local_s1=warp_reduce_sum(local_s1);
  local_s2=warp_reduce_sum(local_s2);

  __shared__ float ws_s1[g_cu_warp_size];
  __shared__ float ws_s2[g_cu_warp_size];
  const int lane=tid&(g_cu_warp_size-1);
  const int warp_id=tid/g_cu_warp_size;
  const int num_warps=blockDim.x/g_cu_warp_size;

  if(lane==0)
  {
    ws_s1[warp_id]=local_s1;
    ws_s2[warp_id]=local_s2;
  }
  __syncthreads();

  if(warp_id==0)
  {
    local_s1=0.0f;
    local_s2=0.0f;
    if(lane<num_warps)
    {
      local_s1=ws_s1[lane];
      local_s2=ws_s2[lane];
    }
    local_s1=warp_reduce_sum(local_s1);
    local_s2=warp_reduce_sum(local_s2);
  }

  // Broadcast results
  if(tid==0)
  {
    ws_s1[0]=local_s1;
    ws_s2[0]=local_s2;
  }
  __syncthreads();

  const float dim_f=static_cast<float>(dim);
  const float s1=ws_s1[0]/dim_f;
  const float s2=ws_s2[0]/dim_f;

  // Phase 2: Compute grad_input, accumulate grad_gamma and grad_beta
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float dy_val=float(dy[col]);
    const float x_hat=(float(x[col])-mean)*rstd;
    dx[col]=T(rstd*(dy_val*gamma[col]-s1-x_hat*s2));
    atomicAdd(&grad_gamma[col],dy_val*x_hat);
    atomicAdd(&grad_beta[col],dy_val);
  }
}

//------------------------------------------------------------------------------
// RMSNorm / LayerNorm Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_rmsnorm_forward(const T *input,
                            const float *gamma,
                            T *output,
                            float *rms_cache,
                            const float epsilon,
                            const int rows,
                            const int dim,
                            cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  rmsnorm_forward_kernel<T><<<rows,block_size,0,stream>>>(input,gamma,output,rms_cache,epsilon,rows,dim);
}
template void launch_rmsnorm_forward<float>(const float *,
                                            const float *,
                                            float *,
                                            float *,
                                            float,
                                            int,
                                            int,
                                            cudaStream_t);
template void launch_rmsnorm_forward<__half>(const __half *,
                                             const float *,
                                             __half *,
                                             float *,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);
template void launch_rmsnorm_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const float *,
                                                    __nv_bfloat16 *,
                                                    float *,
                                                    float,
                                                    int,
                                                    int,
                                                    cudaStream_t);

template<typename T>
void launch_rmsnorm_backward(const T *grad_output,
                             const T *input,
                             const float *gamma,
                             const float *rms_cache,
                             T *grad_input,
                             float *grad_gamma,
                             const float epsilon,
                             const int rows,
                             const int dim,
                             cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  rmsnorm_backward_kernel<T><<<rows,block_size,0,stream>>>(grad_output,
                                                           input,
                                                           gamma,
                                                           rms_cache,
                                                           grad_input,
                                                           grad_gamma,
                                                           epsilon,
                                                           rows,
                                                           dim);
}
template void launch_rmsnorm_backward<float>(const float *,
                                             const float *,
                                             const float *,
                                             const float *,
                                             float *,
                                             float *,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);
template void launch_rmsnorm_backward<__half>(const __half *,
                                              const __half *,
                                              const float *,
                                              const float *,
                                              __half *,
                                              float *,
                                              float,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_rmsnorm_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     const __nv_bfloat16 *,
                                                     const float *,
                                                     const float *,
                                                     __nv_bfloat16 *,
                                                     float *,
                                                     float,
                                                     int,
                                                     int,
                                                     cudaStream_t);

template<typename T>
void launch_layernorm_forward(const T *input,
                              const float *gamma,
                              const float *beta,
                              T *output,
                              float *mean_cache,
                              float *rstd_cache,
                              const float epsilon,
                              const int rows,
                              const int dim,
                              cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  layernorm_forward_kernel<T><<<rows,block_size,0,stream>>>(input,
                                                            gamma,
                                                            beta,
                                                            output,
                                                            mean_cache,
                                                            rstd_cache,
                                                            epsilon,
                                                            rows,
                                                            dim);
}
template void launch_layernorm_forward<float>(const float *,
                                              const float *,
                                              const float *,
                                              float *,
                                              float *,
                                              float *,
                                              float,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_layernorm_forward<__half>(const __half *,
                                               const float *,
                                               const float *,
                                               __half *,
                                               float *,
                                               float *,
                                               float,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_layernorm_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      const float *,
                                                      const float *,
                                                      __nv_bfloat16 *,
                                                      float *,
                                                      float *,
                                                      float,
                                                      int,
                                                      int,
                                                      cudaStream_t);

template<typename T>
void launch_layernorm_backward(const T *grad_output,
                               const T *input,
                               const float *gamma,
                               const float *mean_cache,
                               const float *rstd_cache,
                               T *grad_input,
                               float *grad_gamma,
                               float *grad_beta,
                               const int rows,
                               const int dim,
                               cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  layernorm_backward_kernel<T><<<rows,block_size,0,stream>>>(grad_output,
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
template void launch_layernorm_backward<float>(const float *,
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
template void launch_layernorm_backward<__half>(const __half *,
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
template void launch_layernorm_backward<__nv_bfloat16>(const __nv_bfloat16 *,
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
