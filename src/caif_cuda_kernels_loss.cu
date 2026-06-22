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
// Loss CUDA kernels: cross-entropy (basic, index-target, logits,
// fused) and MSE, forward + backward, plus loss reductions.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_loss.cuh
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
// Loss Function Kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// MSE Loss Reduction Kernel
// Computes sum of (pred - target)^2 via parallel reduction to a scalar.
// Output must be pre-zeroed; each block atomically adds its partial sum.
//------------------------------------------------------------------------------
template<typename T>
__global__ void mse_loss_reduce_kernel(const T *predictions,
                                       const T *targets,
                                       float *loss,
                                       const int n)
{
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;

  float val=0.0f;
  if(idx<n)
  {
    const float diff=float(predictions[idx])-float(targets[idx]);
    val=diff*diff;
  }

  // Warp shuffle reduction
  val=warp_reduce_sum(val);

  // Per-warp partial sums via shared memory
  __shared__ float warp_sums[g_cu_warp_size];
  const int lane=tid&(g_cu_warp_size-1);
  const int warp_id=tid/g_cu_warp_size;
  const int num_warps=blockDim.x/g_cu_warp_size;

  if(lane==0)
  {
    warp_sums[warp_id]=val;
  }
  __syncthreads();

  // First warp reduces the per-warp sums
  if(warp_id==0)
  {
    val=0.0f;
    if(lane<num_warps)
    {
      val=warp_sums[lane];
    }
    val=warp_reduce_sum(val);
  }

  if(tid==0)
  {
    atomicAdd(loss,val);
  }
}

//------------------------------------------------------------------------------
// MSE Gradient Kernel
// grad = 2 * (pred - target) / n
//------------------------------------------------------------------------------
template<typename T>
__global__ void mse_gradient_kernel(const T *predictions,
                                    const T *targets,
                                    T *gradient,
                                    const float scale,
                                    const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    gradient[idx]=T(scale*(float(predictions[idx])-float(targets[idx])));
  }
}

//------------------------------------------------------------------------------
// Loss Function Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_mse_loss(const T *predictions,
                     const T *targets,
                     float *loss,
                     const int n,
                     cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  mse_loss_reduce_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,targets,loss,n);
}

template void launch_mse_loss<float>(const float *,const float *,float *,int,cudaStream_t);
template void launch_mse_loss<__half>(const __half *,const __half *,float *,int,cudaStream_t);
template void launch_mse_loss<__nv_bfloat16>(const __nv_bfloat16 *,
                                             const __nv_bfloat16 *,
                                             float *,
                                             int,
                                             cudaStream_t);

template<typename T>
void launch_mse_gradient(const T *predictions,
                         const T *targets,
                         T *gradient,
                         const int n,
                         cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  const float scale=g_cu_mse_grad_coeff/static_cast<float>(n);
  mse_gradient_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,targets,gradient,scale,n);
}

template void launch_mse_gradient<float>(const float *,const float *,float *,int,cudaStream_t);
template void launch_mse_gradient<__half>(const __half *,const __half *,__half *,int,cudaStream_t);
template void launch_mse_gradient<__nv_bfloat16>(const __nv_bfloat16 *,
                                                 const __nv_bfloat16 *,
                                                 __nv_bfloat16 *,
                                                 int,
                                                 cudaStream_t);

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
template<typename T>
__global__ void cross_entropy_logits_forward_kernel(const T *logits,
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
  const T *pos_logits=logits+static_cast<size_t>(pos)*vocab_size;

  // Step 1: Find max logit (parallel reduction)
  float local_max=g_cu_neg_sentinel;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=float(pos_logits[v]);
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
    local_sum+=expf(float(pos_logits[v])-max_logit);
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

  // Step 3: loss = log_sum_exp - target_logit. Mathematically >= 0; kept raw
  // (no clamp on the sum or the loss) so it matches the backward path's
  // unclamped sum, and any NaN / negative drift stays visible.
  if(tid==0)
  {
    const float log_sum_exp=max_logit+logf(s_sum[0]);
    const float target_logit=float(pos_logits[target_id]);
    losses[pos]=log_sum_exp-target_logit;
  }
}

//------------------------------------------------------------------------------
// Cross-Entropy Logits Backward Kernel
// grad[i,j] = softmax(logits)[i,j] - (j == target[i] ? 1 : 0)
// One block per position.
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_logits_backward_kernel(const T *logits,
                                                     const float *targets,
                                                     T *grad,
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
  const T *pos_logits=logits+static_cast<size_t>(pos)*vocab_size;
  T *pos_grad=grad+static_cast<size_t>(pos)*vocab_size;

  // Check for ignore index - zero gradient
  if(target_id==ignore_index)
  {
    for(int v=tid;v<vocab_size;v+=blockDim.x)
    {
      pos_grad[v]=T(0.0f);
    }
    return;
  }

  // Step 1: Find max logit (parallel reduction)
  float local_max=g_cu_neg_sentinel;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=float(pos_logits[v]);
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
    local_sum+=expf(float(pos_logits[v])-max_logit);
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
    const float softmax_val=expf(float(pos_logits[v])-max_logit)/sum_exp;
    float g=softmax_val;
    if(v==target_id)
    {
      g-=1.0f;
    }
    pos_grad[v]=T(g*scale);
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
template<typename T>
void launch_cross_entropy_logits_forward(const T *logits,
                                         const float *targets,
                                         float *losses,
                                         const int n,
                                         const int vocab_size,
                                         const int ignore_index,
                                         cudaStream_t stream)
{
  // One block per position, use enough threads to cover vocab
  const int block_size=g_cu_block_size;
  const int num_blocks=n;
  const size_t shared_size=g_cu_softmax_stat_arrays*block_size*sizeof(float);
  cross_entropy_logits_forward_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(logits,
                                                                                       targets,
                                                                                       losses,
                                                                                       vocab_size,
                                                                                       ignore_index);
}

template void launch_cross_entropy_logits_forward<float>(const float *,
                                                         const float *,
                                                         float *,
                                                         int,
                                                         int,
                                                         int,
                                                         cudaStream_t);
template void launch_cross_entropy_logits_forward<__half>(const __half *,
                                                          const float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          int,
                                                          cudaStream_t);
template void launch_cross_entropy_logits_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                 const float *,
                                                                 float *,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 cudaStream_t);

template<typename T>
void launch_cross_entropy_logits_backward(const T *logits,
                                          const float *targets,
                                          T *grad,
                                          const int n,
                                          const int vocab_size,
                                          const int ignore_index,
                                          const float scale,
                                          cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=n;
  const size_t shared_size=g_cu_softmax_stat_arrays*block_size*sizeof(float);
  cross_entropy_logits_backward_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(logits,
                                                                                        targets,
                                                                                        grad,
                                                                                        vocab_size,
                                                                                        ignore_index,
                                                                                        scale);
}

template void launch_cross_entropy_logits_backward<float>(const float *,
                                                          const float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          int,
                                                          float,
                                                          cudaStream_t);
template void launch_cross_entropy_logits_backward<__half>(const __half *,
                                                           const float *,
                                                           __half *,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           cudaStream_t);
template void launch_cross_entropy_logits_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  const float *,
                                                                  __nv_bfloat16 *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  float,
                                                                  cudaStream_t);

void launch_cross_entropy_reduce_mean(const float *losses,
                                      const float *targets,
                                      float *output,
                                      const int n,
                                      const int ignore_index,
                                      cudaStream_t stream)
{
  // Output should be pre-zeroed (stores sum and count)
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  const size_t shared_size=g_cu_ce_mean_stat_arrays*block_size*sizeof(float);
  cross_entropy_reduce_mean_kernel<<<num_blocks,block_size,shared_size,stream>>>(losses,
                                                                                 targets,
                                                                                 output,
                                                                                 n,
                                                                                 ignore_index);
}

//------------------------------------------------------------------------------
// Cross-Entropy Fused Kernels
// Fuses forward loss + backward gradient into a single pass over logits.
// Eliminates host roundtrip for valid_count and halves logits memory reads.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Count valid (non-ignored) positions. Writes count to *count_out (pre-zeroed).
//------------------------------------------------------------------------------
__global__ void cross_entropy_count_valid_kernel(const float *targets,
                                                 float *count_out,
                                                 const int n,
                                                 const int ignore_index)
{
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+tid;

  float local_count=0.0f;
  if(idx<n)
  {
    const int target_id=static_cast<int>(targets[idx]);
    if(target_id!=ignore_index)
    {
      local_count=1.0f;
    }
  }

  local_count=warp_reduce_sum(local_count);

  if((tid%g_cu_warp_size)==0 && local_count>0.0f)
  {
    atomicAdd(count_out,local_count);
  }
}

//------------------------------------------------------------------------------
// Fused forward+backward: one block per position.
// Reads logits once (3 passes: max, sum_exp, gradient).
// Reads valid_count from device memory (set by count kernel on same stream).
// Outputs per-position loss AND scaled gradient.
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_fused_loss_grad_kernel(const T *logits,
                                                     const float *targets,
                                                     float *losses,
                                                     T *grad,
                                                     const float *valid_count,
                                                     const int vocab_size,
                                                     const int ignore_index)
{
  const int pos=blockIdx.x;
  const int tid=threadIdx.x;
  const int target_id=static_cast<int>(targets[pos]);

  const T *pos_logits=logits+static_cast<size_t>(pos)*vocab_size;
  T *pos_grad=grad+static_cast<size_t>(pos)*vocab_size;

  // Ignored position: zero loss and gradient
  if(target_id==ignore_index)
  {
    if(tid==0)
    {
      losses[pos]=0.0f;
    }
    for(int v=tid;v<vocab_size;v+=blockDim.x)
    {
      pos_grad[v]=T(0.0f);
    }
    return;
  }

  // Read valid count from device memory and compute gradient scale
  const float count=*valid_count;
  float scale=0.0f;
  if(count>=1.0f)
  {
    scale=1.0f/count;
  }

  // Pass 1: Find max logit (warp shuffle reduction)
  float local_max=g_cu_neg_sentinel;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=float(pos_logits[v]);
    if(val>local_max)
    {
      local_max=val;
    }
  }
  local_max=warp_reduce_max(local_max);
  __shared__ float ws_max[g_cu_warp_size];
  const int warp_id=tid/g_cu_warp_size;
  const int lane_id=tid%g_cu_warp_size;
  if(lane_id==0)
  {
    ws_max[warp_id]=local_max;
  }
  __syncthreads();
  if(tid<g_cu_warp_size)
  {
    const int num_warps=blockDim.x/g_cu_warp_size;
    float v=g_cu_neg_sentinel;
    if(tid<num_warps)
    {
      v=ws_max[tid];
    }
    v=warp_reduce_max(v);
    ws_max[0]=v;
  }
  __syncthreads();
  const float max_logit=ws_max[0];

  // Pass 2: Compute sum of exp(logit - max) (warp shuffle reduction)
  float local_sum=0.0f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    local_sum+=expf(float(pos_logits[v])-max_logit);
  }
  local_sum=warp_reduce_sum(local_sum);
  __shared__ float ws_sum[g_cu_warp_size];
  if(lane_id==0)
  {
    ws_sum[warp_id]=local_sum;
  }
  __syncthreads();
  if(tid<g_cu_warp_size)
  {
    const int num_warps=blockDim.x/g_cu_warp_size;
    float v=0.0f;
    if(tid<num_warps)
    {
      v=ws_sum[tid];
    }
    v=warp_reduce_sum(v);
    ws_sum[0]=v;
  }
  __syncthreads();
  const float sum_exp=ws_sum[0];

  // Write per-position loss (thread 0 only). Kept raw — no clamp on the sum or
  // the loss — so it matches the unclamped sum used for the gradient below.
  if(tid==0)
  {
    const float log_sum_exp=max_logit+logf(sum_exp);
    const float target_logit=float(pos_logits[target_id]);
    losses[pos]=log_sum_exp-target_logit;
  }

  // Pass 3: Compute scaled gradient = (softmax - one_hot) * scale
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    float g=expf(float(pos_logits[v])-max_logit)/sum_exp;
    if(v==target_id)
    {
      g-=1.0f;
    }
    pos_grad[v]=T(g*scale);
  }
}

//------------------------------------------------------------------------------
// Sum per-position losses (excluding ignored). Writes to *output (pre-zeroed).
//------------------------------------------------------------------------------
__global__ void cross_entropy_sum_losses_kernel(const float *losses,
                                                const float *targets,
                                                float *output,
                                                const int n,
                                                const int ignore_index)
{
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+tid;

  float local_sum=0.0f;
  if(idx<n)
  {
    const int target_id=static_cast<int>(targets[idx]);
    if(target_id!=ignore_index)
    {
      local_sum=losses[idx];
    }
  }

  local_sum=warp_reduce_sum(local_sum);

  if((tid%g_cu_warp_size)==0 && local_sum!=0.0f)
  {
    atomicAdd(output,local_sum);
  }
}

//------------------------------------------------------------------------------
// Fused cross-entropy launcher: count + fused loss/grad + sum losses.
// All 3 kernels on same stream, no host sync between them.
// result[0] = valid_count, result[1] = loss_sum (must be pre-zeroed).
//------------------------------------------------------------------------------
template<typename T>
void launch_cross_entropy_fused(const T *logits,
                                const float *targets,
                                float *losses,
                                T *grad,
                                float *result,
                                const int n,
                                const int vocab_size,
                                const int ignore_index,
                                cudaStream_t stream)
{
  const int block_size=g_cu_block_size;

  // Kernel 1: count valid positions → result[0]
  const int count_blocks=(n+block_size-1)/block_size;
  cross_entropy_count_valid_kernel<<<count_blocks,block_size,0,stream>>>(targets,&result[0],n,ignore_index);

  // Kernel 2: fused forward+backward (reads result[0] for scale)
  cross_entropy_fused_loss_grad_kernel<T><<<n,block_size,0,stream>>>(logits,
                                                                     targets,
                                                                     losses,
                                                                     grad,
                                                                     &result[0],
                                                                     vocab_size,
                                                                     ignore_index);

  // Kernel 3: sum per-position losses → result[1]
  const int sum_blocks=(n+block_size-1)/block_size;
  cross_entropy_sum_losses_kernel<<<sum_blocks,block_size,0,stream>>>(losses,targets,&result[1],n,ignore_index);
}

template void launch_cross_entropy_fused<float>(const float *,
                                                const float *,
                                                float *,
                                                float *,
                                                float *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);
template void launch_cross_entropy_fused<__half>(const __half *,
                                                 const float *,
                                                 float *,
                                                 __half *,
                                                 float *,
                                                 int,
                                                 int,
                                                 int,
                                                 cudaStream_t);
template void launch_cross_entropy_fused<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const float *,
                                                        float *,
                                                        __nv_bfloat16 *,
                                                        float *,
                                                        int,
                                                        int,
                                                        int,
                                                        cudaStream_t);

// SiLU backward kernels removed 2026-05-02: superseded by templated
// `launch_swish_backward<T>` (caif_cuda_kernels_activations.cu). The fp32-only
// `launch_silu_backward` had zero callers in src/tests/benchmarks.

