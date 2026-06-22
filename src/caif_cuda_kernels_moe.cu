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
// MoE CUDA kernels: normalize_rows (+backward/topk-gather), top-k,
// gather_topk_values, scatter-add, dispatch/combine (+backward),
// top-k gating, build-dispatch-map, router z-loss gradient.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_moe.cuh
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
// Normalize Rows Kernel
// input: [batch, dim], output: [batch, dim]
// output[b,d] = input[b,d] / sum_d(input[b,:])
//------------------------------------------------------------------------------
template<typename T>
__global__ void normalize_rows_kernel(const T *input,
                                      T *output,
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
  const T *row_in=input+static_cast<size_t>(b)*dim;
  T *row_out=output+static_cast<size_t>(b)*dim;

  // Sum the row
  float local_sum=0.0f;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_sum+=float(row_in[d]);
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
  const float sum=fmaxf(s_sum[0],g_cu_moe_row_sum_epsilon);
  const float inv_sum=1.0f/sum;

  // Normalize
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    row_out[d]=T(float(row_in[d])*inv_sum);
  }
}

template<typename T>
void launch_normalize_rows(const T *input,
                           T *output,
                           const int batch,
                           const int dim,
                           cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=batch;
  const size_t shared_size=block_size*sizeof(float);
  normalize_rows_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(input,output,batch,dim);
}
template void launch_normalize_rows<float>(const float *,
                                           float *,
                                           int,
                                           int,
                                           cudaStream_t);
template void launch_normalize_rows<__half>(const __half *,
                                            __half *,
                                            int,
                                            int,
                                            cudaStream_t);
template void launch_normalize_rows<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int,
                                                   int,
                                                   cudaStream_t);

//------------------------------------------------------------------------------
// NormalizeRows Backward Jacobian, gather variant (top-k rows)
//   Given full softmax probs [T,E] and topk indices [T,K] (stored float in CAIF):
//     p[k]  = probs[t, indices[t,k]]
//     s     = sum_k p[k]
//     w[k]  = p[k]/s
//     dot   = sum_k w[k]*grad_w[k]
//     grad_p_topk[t,k] = (grad_w[t,k] - dot)/s
// This avoids caching `w_norm` and `row_sum` in forward — all Jacobian work
// runs here from the probs/indices caches that already exist for the softmax
// backward. One warp per token, top_k <= 32.
//------------------------------------------------------------------------------
template<typename T>
__global__ void normalize_rows_backward_topk_gather_kernel(const T *grad_w,
                                                           const T *probs,
                                                           const int *indices,
                                                           T *grad_p_topk,
                                                           const int num_tokens,
                                                           const int num_experts,
                                                           const int top_k)
{
  const int t=blockIdx.x;
  if(t>=num_tokens)
  {
    return;
  }
  const int tid=threadIdx.x;

  float p_k=0.0f;
  float grad_w_k=0.0f;
  if(tid<top_k)
  {
    const int expert_idx=indices[t*top_k+tid];
    p_k=float(probs[static_cast<int64_t>(t)*num_experts+expert_idx]);
    grad_w_k=float(grad_w[t*top_k+tid]);
  }

  float s=p_k;
  for(int offset=g_cu_warp_half_size;offset>0;offset>>=1)
  {
    s+=__shfl_xor_sync(g_cu_warp_full_mask,s,offset);
  }
  s=fmaxf(s,g_cu_moe_topk_sum_epsilon);
  const float inv_s=1.0f/s;
  const float w_k=p_k*inv_s;

  float dot=w_k*grad_w_k;
  for(int offset=g_cu_warp_half_size;offset>0;offset>>=1)
  {
    dot+=__shfl_xor_sync(g_cu_warp_full_mask,dot,offset);
  }

  if(tid<top_k)
  {
    grad_p_topk[t*top_k+tid]=T((grad_w_k-dot)*inv_s);
  }
}

template<typename T>
void launch_normalize_rows_backward_topk_gather(const T *grad_w,
                                                const T *probs,
                                                const int *indices,
                                                T *grad_p_topk,
                                                const int num_tokens,
                                                const int num_experts,
                                                const int top_k,
                                                cudaStream_t stream)
{
  normalize_rows_backward_topk_gather_kernel<T><<<num_tokens,g_cu_warp_size,0,stream>>>(grad_w,
                                                                                        probs,
                                                                                        indices,
                                                                                        grad_p_topk,
                                                                                        num_tokens,
                                                                                        num_experts,
                                                                                        top_k);
}
template void launch_normalize_rows_backward_topk_gather<float>(const float *,
                                                                const float *,
                                                                const int *,
                                                                float *,
                                                                int,
                                                                int,
                                                                int,
                                                                cudaStream_t);
template void launch_normalize_rows_backward_topk_gather<__half>(const __half *,
                                                                 const __half *,
                                                                 const int *,
                                                                 __half *,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 cudaStream_t);
template void launch_normalize_rows_backward_topk_gather<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                        const __nv_bfloat16 *,
                                                                        const int *,
                                                                        __nv_bfloat16 *,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        cudaStream_t);

//------------------------------------------------------------------------------
// GatherTopKValues — out[t,k] = scores[t, indices[t,k]]
//   Phase 1b of the SigmoidNoauxTc gating path: selection happens on
//   bias-corrected sigmoid scores host-side via AddBias + TopK, but the
//   combine weights must be the ORIGINAL (uncorrected) sigmoid scores at
//   the chosen indices to match HF DeepSeek-V2 / GLM-4-MoE
//   `topk_method=noaux_tc` (transformers/.../modeling_glm4_moe.py
//   `router_logits.gather(1, topk_indices)`).
//
//   One block per token, top_k threads (top_k <= 32 — pre-checked
//   host-side).  Each thread reads one (token, k) gather and writes
//   one element.  No reductions needed.
//------------------------------------------------------------------------------
template<typename T>
__global__ void gather_topk_values_kernel(const T *scores,
                                          const int *indices,
                                          T *out,
                                          const int num_tokens,
                                          const int num_experts,
                                          const int top_k)
{
  const int t=blockIdx.x;
  if(t>=num_tokens)
  {
    return;
  }
  const int tid=threadIdx.x;
  if(tid<top_k)
  {
    const int expert_idx=indices[t*top_k+tid];
    out[t*top_k+tid]=scores[static_cast<int64_t>(t)*num_experts+expert_idx];
  }
}

template<typename T>
void launch_gather_topk_values(const T *scores,
                               const int *indices,
                               T *out,
                               const int num_tokens,
                               const int num_experts,
                               const int top_k,
                               cudaStream_t stream)
{
  gather_topk_values_kernel<T><<<num_tokens,g_cu_warp_size,0,stream>>>(scores,
                                                                       indices,
                                                                       out,
                                                                       num_tokens,
                                                                       num_experts,
                                                                       top_k);
}
template void launch_gather_topk_values<float>(const float *,const int *,float *,int,int,int,cudaStream_t);
template void launch_gather_topk_values<__half>(const __half *,const int *,__half *,int,int,int,cudaStream_t);
template void launch_gather_topk_values<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const int *,
                                                       __nv_bfloat16 *,
                                                       int,
                                                       int,
                                                       int,
                                                       cudaStream_t);

//------------------------------------------------------------------------------
// Top-K Kernel (simple implementation)
// input: [batch, dim], indices: [batch, k], values: [batch, k]
//------------------------------------------------------------------------------
template<typename T>
__global__ void topk_kernel(const T *input,
                            int *indices,
                            T *values,
                            const int batch,
                            const int dim,
                            const int k)
{
  const int b=blockIdx.x;
  if(b>=batch)
  {
    return;
  }

  const T *row=input+static_cast<size_t>(b)*dim;
  int *out_idx=indices+static_cast<size_t>(b)*k;
  T *out_val=values+static_cast<size_t>(b)*k;

  // Simple selection sort for top-k (good enough for small k)
  // Mark selected indices with the g_cu_neg_sentinel. Compare in fp32 in
  // shared memory so the sentinel works for fp16 storage (fp16 cannot
  // represent it).
  extern __shared__ float s_data[];
  float *temp=s_data;
  const int tid=threadIdx.x;

  // Copy to shared memory (convert T -> float)
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    temp[d]=float(row[d]);
  }
  __syncthreads();

  // Only thread 0 does the selection (simple but correct)
  if(tid==0)
  {
    for(int i=0;i<k;++i)
    {
      float max_val=g_cu_neg_sentinel;
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
      out_val[i]=T(max_val);
      // Mark as selected
      temp[max_idx]=g_cu_neg_sentinel;
    }
  }
}

template<typename T>
void launch_topk(const T *input,
                 int *indices,
                 T *values,
                 const int batch,
                 const int dim,
                 const int k,
                 cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=batch;
  const size_t shared_size=dim*sizeof(float);
  topk_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(input,indices,values,batch,dim,k);
}
template void launch_topk<float>(const float *,
                                 int *,
                                 float *,
                                 int,
                                 int,
                                 int,
                                 cudaStream_t);
template void launch_topk<__half>(const __half *,
                                  int *,
                                  __half *,
                                  int,
                                  int,
                                  int,
                                  cudaStream_t);
template void launch_topk<__nv_bfloat16>(const __nv_bfloat16 *,
                                         int *,
                                         __nv_bfloat16 *,
                                         int,
                                         int,
                                         int,
                                         cudaStream_t);

//------------------------------------------------------------------------------
// DeepSeek group-limited routing mask. The experts of each token's score row
// are split into n_group equal groups; each group is scored by the sum of its
// top-2 expert scores; the top-topk_group groups are kept and every expert in a
// non-selected group is set to the neg-sentinel so the downstream top-k never
// picks it. In place on the [num_tokens, num_experts] selection scores; one
// block per token. Scores compared in fp32 in shared memory so the sentinel
// works for fp16/bf16 storage.
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_group_mask_kernel(T *selection,
                                      const int num_tokens,
                                      const int num_experts,
                                      const int n_group,
                                      const int topk_group)
{
  const int b=blockIdx.x;
  if(b>=num_tokens)
  {
    return;
  }
  const int experts_per_group=num_experts/n_group;
  T *row=selection+static_cast<size_t>(b)*num_experts;

  extern __shared__ float s_mem[];
  float *s_row=s_mem;
  float *s_group=s_mem+num_experts;
  float *s_selected=s_group+n_group;
  const int tid=threadIdx.x;

  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    s_row[e]=float(row[e]);
  }
  __syncthreads();

  if(tid==0)
  {
    for(int g=0;g<n_group;++g)
    {
      float top1=g_cu_neg_sentinel;
      float top2=g_cu_neg_sentinel;
      const int base=g*experts_per_group;
      for(int j=0;j<experts_per_group;++j)
      {
        const float v=s_row[base+j];
        if(v>top1)
        {
          top2=top1;
          top1=v;
        }
        else if(v>top2)
        {
          top2=v;
        }
      }
      s_group[g]=top1+top2;
      s_selected[g]=0.0f;
    }
    for(int r=0;r<topk_group;++r)
    {
      float best=g_cu_neg_sentinel;
      int best_g=0;
      for(int g=0;g<n_group;++g)
      {
        if(s_group[g]>best)
        {
          best=s_group[g];
          best_g=g;
        }
      }
      s_selected[best_g]=1.0f;
      s_group[best_g]=g_cu_neg_sentinel;
    }
  }
  __syncthreads();

  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    const int g=e/experts_per_group;
    if(s_selected[g]==0.0f)
    {
      row[e]=T(g_cu_neg_sentinel);
    }
  }
}

template<typename T>
void launch_moe_group_mask(T *selection,
                           const int num_tokens,
                           const int num_experts,
                           const int n_group,
                           const int topk_group,
                           cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=num_tokens;
  const size_t shared_size=static_cast<size_t>(num_experts+2*n_group)*sizeof(float);
  moe_group_mask_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(selection,
                                                                         num_tokens,
                                                                         num_experts,
                                                                         n_group,
                                                                         topk_group);
}
template void launch_moe_group_mask<float>(float *,int,int,int,int,cudaStream_t);
template void launch_moe_group_mask<__half>(__half *,int,int,int,int,cudaStream_t);
template void launch_moe_group_mask<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// Aux-loss-free router bias update (DeepSeek-V3). Given each expert's token
// load over the step, nudge its router bias toward balance: bias[e] += rate if
// the expert is below the mean load, -= rate if above (sign of mean - load).
// No gradient, no aux loss. Single block; loads are reduced to a mean in shared
// memory, then every expert's bias is updated in place. block size must be a
// power of two (the tree reduction assumes it).
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_bias_update_kernel(T *bias,
                                       const int *counts,
                                       const int num_experts,
                                       const float rate)
{
  extern __shared__ float s_sum[];
  const int tid=threadIdx.x;

  float local=0.0f;
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    local+=float(counts[e]);
  }
  s_sum[tid]=local;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride/=2)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }
  const float mean=s_sum[0]/float(num_experts);

  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    const float err=mean-float(counts[e]);
    float step=0.0f;
    if(err>0.0f)
    {
      step=rate;
    }
    else if(err<0.0f)
    {
      step=-rate;
    }
    bias[e]=T(float(bias[e])+step);
  }
}

template<typename T>
void launch_moe_bias_update(T *bias,
                            const int *counts,
                            const int num_experts,
                            const float rate,
                            cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const size_t shared_size=static_cast<size_t>(block_size)*sizeof(float);
  moe_bias_update_kernel<T><<<1,block_size,shared_size,stream>>>(bias,counts,num_experts,rate);
}
template void launch_moe_bias_update<float>(float *,const int *,int,float,cudaStream_t);
template void launch_moe_bias_update<__half>(__half *,const int *,int,float,cudaStream_t);
template void launch_moe_bias_update<__nv_bfloat16>(__nv_bfloat16 *,const int *,int,float,cudaStream_t);

//------------------------------------------------------------------------------
// Scatter Add Kernel
// output[b, indices[b,k]] += values[b,k]
// values: [batch, k], indices: [batch, k], output: [batch, dim]
// Uses caif_atomic_add<T> defined above (see Patch embedding kernels section).
//------------------------------------------------------------------------------
template<typename T>
__global__ void scatter_add_kernel(const T *values,
                                   const int *indices,
                                   T *output,
                                   const int batch,
                                   const int k,
                                   const int dim)
{
  const size_t idx=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t total=static_cast<size_t>(batch)*k;
  if(idx<total)
  {
    const int b=static_cast<int>(idx/k);
    const int target_idx=indices[idx];
    if(target_idx>=0 && target_idx<dim)
    {
      caif_atomic_add<T>(&output[static_cast<size_t>(b)*dim+target_idx],values[idx]);
    }
  }
}

template<typename T>
void launch_scatter_add(const T *values,
                        const int *indices,
                        T *output,
                        const int batch,
                        const int k,
                        const int dim,
                        cudaStream_t stream)
{
  const size_t total=static_cast<size_t>(batch)*k;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  scatter_add_kernel<T><<<num_blocks,block_size,0,stream>>>(values,indices,output,batch,k,dim);
}
template void launch_scatter_add<float>(const float *,
                                        const int *,
                                        float *,
                                        int,
                                        int,
                                        int,
                                        cudaStream_t);
template void launch_scatter_add<__half>(const __half *,
                                         const int *,
                                         __half *,
                                         int,
                                         int,
                                         int,
                                         cudaStream_t);
template void launch_scatter_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                const int *,
                                                __nv_bfloat16 *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);

//------------------------------------------------------------------------------
// MoE Dispatch Kernel
// Gathers tokens to expert-specific buffers based on routing indices
// expert_indices: [num_tokens, top_k] - which experts each token routes to
// expert_offsets: [num_experts+1] - cumulative token counts per expert
// dispatch_map: [num_tokens, top_k] - position within expert buffer for each assignment
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_dispatch_kernel(const T *input,
                                    const int *expert_indices,
                                    const int *dispatch_map,
                                    T *expert_buffer,
                                    const int *expert_offsets,
                                    const int num_tokens,
                                    const int dim,
                                    const int top_k)
{
  // Each thread handles one dimension of one token-expert assignment
  const size_t tid=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t total_assignments=static_cast<size_t>(num_tokens)*top_k*dim;

  if(tid<total_assignments)
  {
    const int d=static_cast<int>(tid%dim);
    const int assignment_idx=static_cast<int>(tid/dim);
    const int token_idx=assignment_idx/top_k;
    const int k_idx=assignment_idx%top_k;

    const int expert_idx=expert_indices[token_idx*top_k+k_idx];
    const int pos_in_expert=dispatch_map[token_idx*top_k+k_idx];

    if(expert_idx>=0 && pos_in_expert>=0)
    {
      const int expert_start=expert_offsets[expert_idx];
      const size_t dest_idx=(static_cast<size_t>(expert_start)+pos_in_expert)*dim+d;
      expert_buffer[dest_idx]=input[static_cast<size_t>(token_idx)*dim+d];
    }
  }
}

template<typename T>
void launch_moe_dispatch(const T *input,
                         const int *expert_indices,
                         const int *dispatch_map,
                         T *expert_buffer,
                         const int *expert_offsets,
                         const int num_tokens,
                         const int dim,
                         const int top_k,
                         cudaStream_t stream)
{
  const size_t total=static_cast<size_t>(num_tokens)*top_k*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  moe_dispatch_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                             expert_indices,
                                                             dispatch_map,
                                                             expert_buffer,
                                                             expert_offsets,
                                                             num_tokens,
                                                             dim,
                                                             top_k);
}

template void launch_moe_dispatch<float>(const float *,
                                         const int *,
                                         const int *,
                                         float *,
                                         const int *,
                                         const int,
                                         const int,
                                         const int,
                                         cudaStream_t);
template void launch_moe_dispatch<__half>(const __half *,
                                          const int *,
                                          const int *,
                                          __half *,
                                          const int *,
                                          const int,
                                          const int,
                                          const int,
                                          cudaStream_t);
template void launch_moe_dispatch<__nv_bfloat16>(const __nv_bfloat16 *,
                                                 const int *,
                                                 const int *,
                                                 __nv_bfloat16 *,
                                                 const int *,
                                                 const int,
                                                 const int,
                                                 const int,
                                                 cudaStream_t);

//------------------------------------------------------------------------------
// MoE Combine Kernel
// Scatters expert outputs back to token positions with routing weights
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_combine_kernel(const T *expert_buffer,
                                   const int *expert_indices,
                                   const T *expert_weights,
                                   const int *dispatch_map,
                                   const int *expert_offsets,
                                   T *output,
                                   const int num_tokens,
                                   const int dim,
                                   const int top_k)
{
  // Each thread handles one dimension of one token
  const size_t tid=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t total=static_cast<size_t>(num_tokens)*dim;

  if(tid<total)
  {
    const int token_idx=static_cast<int>(tid/dim);
    const int d=static_cast<int>(tid%dim);

    // fp32 accumulator regardless of T — reduces precision loss when
    // top_k expert outputs are summed at fp16 / bf16.
    float sum=0.0f;

    for(int k=0;k<top_k;++k)
    {
      const int expert_idx=expert_indices[token_idx*top_k+k];
      const float weight=static_cast<float>(expert_weights[token_idx*top_k+k]);
      const int pos_in_expert=dispatch_map[token_idx*top_k+k];

      if(expert_idx>=0 && pos_in_expert>=0)
      {
        const int expert_start=expert_offsets[expert_idx];
        const size_t src_idx=(static_cast<size_t>(expert_start)+pos_in_expert)*dim+d;
        sum+=weight*static_cast<float>(expert_buffer[src_idx]);
      }
    }

    output[tid]=static_cast<T>(sum);
  }
}

template<typename T>
void launch_moe_combine(const T *expert_buffer,
                        const int *expert_indices,
                        const T *expert_weights,
                        const int *dispatch_map,
                        const int *expert_offsets,
                        T *output,
                        const int num_tokens,
                        const int dim,
                        const int top_k,
                        cudaStream_t stream)
{
  const size_t total=static_cast<size_t>(num_tokens)*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  moe_combine_kernel<T><<<num_blocks,block_size,0,stream>>>(expert_buffer,
                                                            expert_indices,
                                                            expert_weights,
                                                            dispatch_map,
                                                            expert_offsets,
                                                            output,
                                                            num_tokens,
                                                            dim,
                                                            top_k);
}

template void launch_moe_combine<float>(const float *,
                                        const int *,
                                        const float *,
                                        const int *,
                                        const int *,
                                        float *,
                                        const int,
                                        const int,
                                        const int,
                                        cudaStream_t);
template void launch_moe_combine<__half>(const __half *,
                                         const int *,
                                         const __half *,
                                         const int *,
                                         const int *,
                                         __half *,
                                         const int,
                                         const int,
                                         const int,
                                         cudaStream_t);
template void launch_moe_combine<__nv_bfloat16>(const __nv_bfloat16 *,
                                                const int *,
                                                const __nv_bfloat16 *,
                                                const int *,
                                                const int *,
                                                __nv_bfloat16 *,
                                                const int,
                                                const int,
                                                const int,
                                                cudaStream_t);

//------------------------------------------------------------------------------
// MoE Combine Backward — grad_expert_buffer path
//
// Forward (for reference):
//   output[t,d] = sum_k W[t,k] * B[(off[e(t,k)] + pos(t,k))*dim + d]
//
// Backward (this kernel) parallelizes over (t, k, d) and writes exactly
// one slot of grad_expert_buffer per (t,k,d). No atomics needed because
// each slot is touched by exactly one (t,k).
//
//   grad_B[(off[e]+pos)*dim + d] = W[t,k] * grad_output[t,d]
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_combine_backward_grad_expert_kernel(const T *grad_output,
                                                        const int *expert_indices,
                                                        const T *expert_weights,
                                                        const int *dispatch_map,
                                                        const int *expert_offsets,
                                                        T *grad_expert_buffer,
                                                        const int num_tokens,
                                                        const int dim,
                                                        const int top_k)
{
  const size_t tid=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t total=static_cast<size_t>(num_tokens)*top_k*dim;

  if(tid<total)
  {
    const int d=static_cast<int>(tid%dim);
    const int assignment_idx=static_cast<int>(tid/dim);
    const int token_idx=assignment_idx/top_k;
    const int k_idx=assignment_idx%top_k;

    const int expert_idx=expert_indices[token_idx*top_k+k_idx];
    const int pos_in_expert=dispatch_map[token_idx*top_k+k_idx];

    if(expert_idx>=0 && pos_in_expert>=0)
    {
      const float w=static_cast<float>(expert_weights[token_idx*top_k+k_idx]);
      const size_t dst_idx=(static_cast<size_t>(expert_offsets[expert_idx])+pos_in_expert)*dim+d;
      const float g=static_cast<float>(grad_output[static_cast<size_t>(token_idx)*dim+d]);
      grad_expert_buffer[dst_idx]=static_cast<T>(w*g);
    }
  }
}

template<typename T>
void launch_moe_combine_backward_grad_expert(const T *grad_output,
                                             const int *expert_indices,
                                             const T *expert_weights,
                                             const int *dispatch_map,
                                             const int *expert_offsets,
                                             T *grad_expert_buffer,
                                             const int num_tokens,
                                             const int dim,
                                             const int top_k,
                                             cudaStream_t stream)
{
  const size_t total=static_cast<size_t>(num_tokens)*top_k*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  moe_combine_backward_grad_expert_kernel<T>
    <<<num_blocks,block_size,0,stream>>>(grad_output,
                                         expert_indices,
                                         expert_weights,
                                         dispatch_map,
                                         expert_offsets,
                                         grad_expert_buffer,
                                         num_tokens,
                                         dim,
                                         top_k);
}

template void launch_moe_combine_backward_grad_expert<float>(const float *,
                                                             const int *,
                                                             const float *,
                                                             const int *,
                                                             const int *,
                                                             float *,
                                                             const int,
                                                             const int,
                                                             const int,
                                                             cudaStream_t);
template void launch_moe_combine_backward_grad_expert<__half>(const __half *,
                                                              const int *,
                                                              const __half *,
                                                              const int *,
                                                              const int *,
                                                              __half *,
                                                              const int,
                                                              const int,
                                                              const int,
                                                              cudaStream_t);
template void launch_moe_combine_backward_grad_expert<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                     const int *,
                                                                     const __nv_bfloat16 *,
                                                                     const int *,
                                                                     const int *,
                                                                     __nv_bfloat16 *,
                                                                     const int,
                                                                     const int,
                                                                     const int,
                                                                     cudaStream_t);

//------------------------------------------------------------------------------
// MoE Combine Backward — grad_weights path
//
// grad_W[t,k] = sum_d grad_output[t,d] * expert_buffer[(off[e]+pos)*dim + d]
//
// One thread per (t, k) pair, serial reduction over d.
//------------------------------------------------------------------------------
// One block per (t,k). Threads in the block cooperatively reduce the
// dot product over dim. Dim ranges from hundreds to several thousand in
// real workloads; serial reduction per (t,k) left the GPU starved at
// prod scale (only num_tokens*top_k active threads). Block reduction
// restores parallelism proportional to dim.
template<typename T>
__global__ void moe_combine_backward_grad_weights_block_kernel(const T *grad_output,
                                                               const T *expert_buffer,
                                                               const int *expert_indices,
                                                               const int *dispatch_map,
                                                               const int *expert_offsets,
                                                               T *grad_weights,
                                                               const int num_tokens,
                                                               const int dim,
                                                               const int top_k)
{
  const int assignment_idx=blockIdx.x;
  const int token_idx=assignment_idx/top_k;
  const int k_idx=assignment_idx%top_k;

  if(token_idx>=num_tokens)
  {
    return;
  }

  const int expert_idx=expert_indices[token_idx*top_k+k_idx];
  const int pos_in_expert=dispatch_map[token_idx*top_k+k_idx];

  float partial=0.0f;
  if(expert_idx>=0 && pos_in_expert>=0)
  {
    const size_t src_base=(static_cast<size_t>(expert_offsets[expert_idx])+pos_in_expert)*dim;
    const size_t grad_base=static_cast<size_t>(token_idx)*dim;
    for(int d=threadIdx.x;d<dim;d+=g_cu_moe_grad_weights_block_size)
    {
      partial+=static_cast<float>(grad_output[grad_base+d])
               *static_cast<float>(expert_buffer[src_base+d]);
    }
  }

  __shared__ float sdata[g_cu_moe_grad_weights_block_size];
  sdata[threadIdx.x]=partial;
  __syncthreads();

  for(int stride=g_cu_moe_grad_weights_block_size/2;stride>0;stride>>=1)
  {
    if(threadIdx.x<stride)
    {
      sdata[threadIdx.x]+=sdata[threadIdx.x+stride];
    }
    __syncthreads();
  }

  if(threadIdx.x==0)
  {
    grad_weights[assignment_idx]=static_cast<T>(sdata[0]);
  }
}

template<typename T>
void launch_moe_combine_backward_grad_weights(const T *grad_output,
                                              const T *expert_buffer,
                                              const int *expert_indices,
                                              const int *dispatch_map,
                                              const int *expert_offsets,
                                              T *grad_weights,
                                              const int num_tokens,
                                              const int dim,
                                              const int top_k,
                                              cudaStream_t stream)
{
  const int num_blocks=num_tokens*top_k;
  moe_combine_backward_grad_weights_block_kernel<T>
    <<<num_blocks,g_cu_moe_grad_weights_block_size,0,stream>>>(grad_output,
                                                               expert_buffer,
                                                               expert_indices,
                                                               dispatch_map,
                                                               expert_offsets,
                                                               grad_weights,
                                                               num_tokens,
                                                               dim,
                                                               top_k);
}

template void launch_moe_combine_backward_grad_weights<float>(const float *,
                                                              const float *,
                                                              const int *,
                                                              const int *,
                                                              const int *,
                                                              float *,
                                                              const int,
                                                              const int,
                                                              const int,
                                                              cudaStream_t);
template void launch_moe_combine_backward_grad_weights<__half>(const __half *,
                                                               const __half *,
                                                               const int *,
                                                               const int *,
                                                               const int *,
                                                               __half *,
                                                               const int,
                                                               const int,
                                                               const int,
                                                               cudaStream_t);
template void launch_moe_combine_backward_grad_weights<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                      const __nv_bfloat16 *,
                                                                      const int *,
                                                                      const int *,
                                                                      const int *,
                                                                      __nv_bfloat16 *,
                                                                      const int,
                                                                      const int,
                                                                      const int,
                                                                      cudaStream_t);

//------------------------------------------------------------------------------
// MoE Dispatch Backward
//
// Forward wrote input[t,d] to slot (off[e(t,k)]+pos(t,k))*dim + d for each k.
// Backward sums contributions from all k:
//
//   grad_input[t,d] = sum_k grad_expert_buffer[(off[e(t,k)]+pos(t,k))*dim + d]
//
// One thread per (t, d). Sums over k serially.
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_dispatch_backward_kernel(const T *grad_expert_buffer,
                                             const int *expert_indices,
                                             const int *dispatch_map,
                                             const int *expert_offsets,
                                             T *grad_input,
                                             const int num_tokens,
                                             const int dim,
                                             const int top_k)
{
  const size_t tid=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t total=static_cast<size_t>(num_tokens)*dim;

  if(tid<total)
  {
    const int token_idx=static_cast<int>(tid/dim);
    const int d=static_cast<int>(tid%dim);

    float sum=0.0f;
    for(int k=0;k<top_k;++k)
    {
      const int expert_idx=expert_indices[token_idx*top_k+k];
      const int pos_in_expert=dispatch_map[token_idx*top_k+k];
      if(expert_idx>=0 && pos_in_expert>=0)
      {
        const size_t src_idx=(static_cast<size_t>(expert_offsets[expert_idx])+pos_in_expert)*dim+d;
        sum+=static_cast<float>(grad_expert_buffer[src_idx]);
      }
    }
    grad_input[tid]=static_cast<T>(sum);
  }
}

template<typename T>
void launch_moe_dispatch_backward(const T *grad_expert_buffer,
                                  const int *expert_indices,
                                  const int *dispatch_map,
                                  const int *expert_offsets,
                                  T *grad_input,
                                  const int num_tokens,
                                  const int dim,
                                  const int top_k,
                                  cudaStream_t stream)
{
  const size_t total=static_cast<size_t>(num_tokens)*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  moe_dispatch_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_expert_buffer,
                                                                      expert_indices,
                                                                      dispatch_map,
                                                                      expert_offsets,
                                                                      grad_input,
                                                                      num_tokens,
                                                                      dim,
                                                                      top_k);
}

template void launch_moe_dispatch_backward<float>(const float *,
                                                  const int *,
                                                  const int *,
                                                  const int *,
                                                  float *,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);
template void launch_moe_dispatch_backward<__half>(const __half *,
                                                   const int *,
                                                   const int *,
                                                   const int *,
                                                   __half *,
                                                   const int,
                                                   const int,
                                                   const int,
                                                   cudaStream_t);
template void launch_moe_dispatch_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          const int *,
                                                          const int *,
                                                          const int *,
                                                          __nv_bfloat16 *,
                                                          const int,
                                                          const int,
                                                          const int,
                                                          cudaStream_t);

//------------------------------------------------------------------------------
// MoE Top-K Gating Kernel
// Fused softmax + top-k selection for router
// Input: router_logits [num_tokens, num_experts] dtype T
// Internal accumulation is fp32 for numerical stability — softmax-over-
// experts wants the same fp32 reduction guarantees as the rest of the
// codebase even when StorageT is fp16/bf16.
// Output: expert_indices Int32 (always); expert_weights / router_probs fp32
// (router output drives expert dispatch which keeps its own StorageT for
// the actual activation pipeline).
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_topk_gating_kernel(const T *router_logits,
                                       int *expert_indices,
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
  int *top_indices=reinterpret_cast<int*>(shared+g_cu_moe_gating_stat_arrays*num_experts);
  float *top_values=reinterpret_cast<float*>(shared+g_cu_moe_gating_stat_arrays*num_experts+top_k);

  const int tid=threadIdx.x;

  // Load logits to shared memory, upcasting to fp32 for the reduction.
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    logits_shared[e]=caif_load_f<T>(router_logits[static_cast<int64_t>(token_idx)*num_experts+e]);
  }
  __syncthreads();

  // Find max for numerical stability (thread 0)
  float max_val=g_cu_neg_sentinel;
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
  __shared__ float sum_shared[g_cu_block_size];
  sum_shared[tid]=local_sum;
  __syncthreads();

  if(tid==0)
  {
    float total_sum=0.0f;
    for(int i=0;i<blockDim.x && i<num_experts;++i)
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
    router_probs[static_cast<int64_t>(token_idx)*num_experts+e]=probs_shared[e];
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

        if(already_selected==false && probs_shared[e]>max_prob)
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
      expert_indices[token_idx*top_k+k]=top_indices[k];
      expert_weights[token_idx*top_k+k]=top_values[k]/topk_sum;
    }
  }
}

template<typename T>
void launch_moe_topk_gating(const T *router_logits,
                            int *expert_indices,
                            float *expert_weights,
                            float *router_probs,
                            const int num_tokens,
                            const int num_experts,
                            const int top_k,
                            cudaStream_t stream)
{
  // One block per token, enough threads to cover num_experts
  const int threads_per_block=min(g_cu_block_size,
                                  ((num_experts+g_cu_warp_size-1)/g_cu_warp_size)*g_cu_warp_size);
  const int shared_size=(g_cu_moe_gating_stat_arrays*num_experts)*sizeof(float)+
                        (top_k)*sizeof(int)+
                        (top_k)*sizeof(float);
  moe_topk_gating_kernel<T><<<num_tokens,threads_per_block,shared_size,stream>>>(router_logits,
                                                                                 expert_indices,
                                                                                 expert_weights,
                                                                                 router_probs,
                                                                                 num_tokens,
                                                                                 num_experts,
                                                                                 top_k);
}

template void launch_moe_topk_gating<float>(const float *,
                                            int *,
                                            float *,
                                            float *,
                                            int,
                                            int,
                                            int,
                                            cudaStream_t);
template void launch_moe_topk_gating<__half>(const __half *,
                                             int *,
                                             float *,
                                             float *,
                                             int,
                                             int,
                                             int,
                                             cudaStream_t);
template void launch_moe_topk_gating<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    int *,
                                                    float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);

//------------------------------------------------------------------------------
// MoE Build Dispatch Map Kernel
// Builds the dispatch_map that tracks position of each token within expert buffers
// Also computes expert_offsets (cumulative counts)
//------------------------------------------------------------------------------
__global__ void moe_count_per_expert_kernel(const int *expert_indices,
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
    const int expert_idx=expert_indices[token_idx*top_k+k];
    if(expert_idx>=0 && expert_idx<num_experts)
    {
      const int old_count=atomicAdd(&expert_counts[expert_idx],1);
      // Capacity enforcement happens at dispatch time
      (void)old_count;
    }
  }
}

void launch_moe_count_per_expert(const int *expert_indices,
                                 int *expert_counts,
                                 const int num_tokens,
                                 const int num_experts,
                                 const int top_k,
                                 const int capacity_per_expert,
                                 cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(num_tokens+block_size-1)/block_size;
  moe_count_per_expert_kernel<<<num_blocks,block_size,0,stream>>>(expert_indices,
                                                                  expert_counts,
                                                                  num_tokens,
                                                                  num_experts,
                                                                  top_k,
                                                                  capacity_per_expert);
}

//------------------------------------------------------------------------------
// ST-MoE router z-loss gradient add kernel
// grad_logits[t,e] += logsumexp_scaled[t] * probs[t,e]
// logsumexp_scaled is pre-multiplied host-side by (2*z_loss_weight/N).
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_z_loss_grad_kernel(const T *logsumexp_scaled,
                                       const T *probs,
                                       T *grad_logits,
                                       const int num_tokens,
                                       const int num_experts)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t total=static_cast<int64_t>(num_tokens)*num_experts;
  if(idx>=total)
  {
    return;
  }
  const int t=static_cast<int>(idx/num_experts);
  // fp32 accumulate; cast on store keeps fp16/bf16 from rounding inside the
  // multiply-add.
  const float lse=static_cast<float>(logsumexp_scaled[t]);
  const float p=static_cast<float>(probs[idx]);
  const float prev=static_cast<float>(grad_logits[idx]);
  grad_logits[idx]=static_cast<T>(prev+lse*p);
}

template<typename T>
void launch_moe_z_loss_grad(const T *logsumexp_scaled,
                            const T *probs,
                            T *grad_logits,
                            const int num_tokens,
                            const int num_experts,
                            cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_tokens)*num_experts;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  moe_z_loss_grad_kernel<T><<<num_blocks,block_size,0,stream>>>(logsumexp_scaled,
                                                                probs,
                                                                grad_logits,
                                                                num_tokens,
                                                                num_experts);
}

template void launch_moe_z_loss_grad<float>(const float *,const float *,float *,const int,const int,cudaStream_t);
template void launch_moe_z_loss_grad<__half>(const __half *,
                                             const __half *,
                                             __half *,
                                             const int,
                                             const int,
                                             cudaStream_t);
template void launch_moe_z_loss_grad<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    const int,
                                                    const int,
                                                    cudaStream_t);

// end of former extern "C" block — functions now have C++ linkage for dtype templates

// (former extern "C" block — C++ linkage used for dtype templates)

