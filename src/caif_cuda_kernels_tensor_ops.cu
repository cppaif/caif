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
// Tensor-op CUDA kernels: slice/concat on the last dim (+backward)
// and T5-style relative position bias forward/backward.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_tensor_ops.cuh
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
// Tensor slice and concatenation kernels
//------------------------------------------------------------------------------

template<typename T>
__global__ void slice_last_dim_kernel(const T *input,
                                      T *output,
                                      const int rows,
                                      const int in_cols,
                                      const int col_start,
                                      const int out_cols)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t total=static_cast<int64_t>(rows)*out_cols;
  if(idx<total)
  {
    const int64_t row=idx/out_cols;
    const int col=static_cast<int>(idx%out_cols);
    output[idx]=input[row*in_cols+col_start+col];
  }
}

template<typename T>
void launch_slice_last_dim(const T *input,
                           T *output,
                           int rows,
                           int in_cols,
                           int col_start,
                           int out_cols,
                           cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(rows)*out_cols;
  if(total<=0)
  {
    return;
  }
  const int grid=static_cast<int>((total+g_cu_block_size-1)/g_cu_block_size);
  slice_last_dim_kernel<T><<<grid,g_cu_block_size,0,stream>>>(input,output,rows,in_cols,col_start,out_cols);
}

template void launch_slice_last_dim<float>(const float *input,
                                           float *output,
                                           int rows,
                                           int in_cols,
                                           int col_start,
                                           int out_cols,
                                           cudaStream_t stream);
template void launch_slice_last_dim<__half>(const __half *input,
                                            __half *output,
                                            int rows,
                                            int in_cols,
                                            int col_start,
                                            int out_cols,
                                            cudaStream_t stream);
template void launch_slice_last_dim<__nv_bfloat16>(const __nv_bfloat16 *input,
                                                   __nv_bfloat16 *output,
                                                   int rows,
                                                   int in_cols,
                                                   int col_start,
                                                   int out_cols,
                                                   cudaStream_t stream);

template<typename T>
__global__ void slice_last_dim_backward_kernel(const T *grad_output,
                                               T *grad_input,
                                               const int rows,
                                               const int in_cols,
                                               const int col_start,
                                               const int out_cols)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t total=static_cast<int64_t>(rows)*out_cols;
  if(idx<total)
  {
    const int64_t row=idx/out_cols;
    const int col=static_cast<int>(idx%out_cols);
    grad_input[row*in_cols+col_start+col]+=grad_output[idx];
  }
}

template<typename T>
void launch_slice_last_dim_backward(const T *grad_output,
                                    T *grad_input,
                                    int rows,
                                    int in_cols,
                                    int col_start,
                                    int out_cols,
                                    cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(rows)*out_cols;
  if(total<=0)
  {
    return;
  }
  const int grid=static_cast<int>((total+g_cu_block_size-1)/g_cu_block_size);
  slice_last_dim_backward_kernel<T><<<grid,g_cu_block_size,0,stream>>>(grad_output,
                                                                       grad_input,
                                                                       rows,
                                                                       in_cols,
                                                                       col_start,
                                                                       out_cols);
}

template void launch_slice_last_dim_backward<float>(const float *grad_output,
                                                    float *grad_input,
                                                    int rows,
                                                    int in_cols,
                                                    int col_start,
                                                    int out_cols,
                                                    cudaStream_t stream);
template void launch_slice_last_dim_backward<__half>(const __half *grad_output,
                                                     __half *grad_input,
                                                     int rows,
                                                     int in_cols,
                                                     int col_start,
                                                     int out_cols,
                                                     cudaStream_t stream);
template void launch_slice_last_dim_backward<__nv_bfloat16>(const __nv_bfloat16 *grad_output,
                                                            __nv_bfloat16 *grad_input,
                                                            int rows,
                                                            int in_cols,
                                                            int col_start,
                                                            int out_cols,
                                                            cudaStream_t stream);

template<typename T>
__global__ void concat_last_dim_kernel(const T *a,
                                       const T *b,
                                       T *output,
                                       const int rows,
                                       const int cols_a,
                                       const int cols_b)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int total_cols=cols_a+cols_b;
  const int64_t total=static_cast<int64_t>(rows)*total_cols;
  if(idx<total)
  {
    const int64_t row=idx/total_cols;
    const int col=static_cast<int>(idx%total_cols);
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

template<typename T>
void launch_concat_last_dim(const T *a,
                            const T *b,
                            T *output,
                            int rows,
                            int cols_a,
                            int cols_b,
                            cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(rows)*(cols_a+cols_b);
  if(total<=0)
  {
    return;
  }
  const int grid=static_cast<int>((total+g_cu_block_size-1)/g_cu_block_size);
  concat_last_dim_kernel<T><<<grid,g_cu_block_size,0,stream>>>(a,b,output,rows,cols_a,cols_b);
}

template void launch_concat_last_dim<float>(const float *a,
                                            const float *b,
                                            float *output,
                                            int rows,
                                            int cols_a,
                                            int cols_b,
                                            cudaStream_t stream);
template void launch_concat_last_dim<__half>(const __half *a,
                                             const __half *b,
                                             __half *output,
                                             int rows,
                                             int cols_a,
                                             int cols_b,
                                             cudaStream_t stream);
template void launch_concat_last_dim<__nv_bfloat16>(const __nv_bfloat16 *a,
                                                    const __nv_bfloat16 *b,
                                                    __nv_bfloat16 *output,
                                                    int rows,
                                                    int cols_a,
                                                    int cols_b,
                                                    cudaStream_t stream);

//------------------------------------------------------------------------------
// Relative Position Bias (T5-style)
//------------------------------------------------------------------------------

__device__ int relative_position_bucket(int relative_position,
                                        int bidirectional,
                                        int num_buckets,
                                        int max_distance)
{
  int ret=0;
  int n=-relative_position;

  if(bidirectional!=0)
  {
    num_buckets=num_buckets/2;
    if(n<0)
    {
      ret=num_buckets;
      n=-n;
    }
  }
  else
  {
    if(n<0)
    {
      n=0;
    }
  }

  const int max_exact=num_buckets/2;
  int val;

  if(n<max_exact)
  {
    val=n;
  }
  else
  {
    const float log_ratio=logf(static_cast<float>(n)/static_cast<float>(max_exact))/
                          logf(static_cast<float>(max_distance)/static_cast<float>(max_exact));
    val=max_exact+static_cast<int>(log_ratio*static_cast<float>(num_buckets-max_exact));
    if(val>num_buckets-1)
    {
      val=num_buckets-1;
    }
  }

  ret=ret+val;
  return ret;
}

template<typename T>
__global__ void relative_position_bias_forward_kernel(const float *embedding,
                                                      T *output,
                                                      const int num_heads,
                                                      const int q_len,
                                                      const int k_len,
                                                      const int num_buckets,
                                                      const int max_distance,
                                                      const int bidirectional)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t total=static_cast<int64_t>(num_heads)*q_len*k_len;
  if(idx>=total)
  {
    return;
  }

  const int h=static_cast<int>(idx/(static_cast<int64_t>(q_len)*k_len));
  const int64_t rem=idx%(static_cast<int64_t>(q_len)*k_len);
  const int q=static_cast<int>(rem/k_len);
  const int k=static_cast<int>(rem%k_len);

  const int rel_pos=k-q;
  const int bucket=relative_position_bucket(rel_pos,bidirectional,num_buckets,max_distance);

  output[idx]=static_cast<T>(embedding[h*num_buckets+bucket]);
}

template<typename T>
__global__ void relative_position_bias_backward_kernel(const T *grad_output,
                                                       float *grad_embedding,
                                                       const int num_heads,
                                                       const int q_len,
                                                       const int k_len,
                                                       const int num_buckets,
                                                       const int max_distance,
                                                       const int bidirectional)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t total=static_cast<int64_t>(num_heads)*q_len*k_len;
  if(idx>=total)
  {
    return;
  }

  const int h=static_cast<int>(idx/(static_cast<int64_t>(q_len)*k_len));
  const int64_t rem=idx%(static_cast<int64_t>(q_len)*k_len);
  const int q=static_cast<int>(rem/k_len);
  const int k=static_cast<int>(rem%k_len);

  const int rel_pos=k-q;
  const int bucket=relative_position_bucket(rel_pos,bidirectional,num_buckets,max_distance);

  atomicAdd(&grad_embedding[h*num_buckets+bucket],static_cast<float>(grad_output[idx]));
}

template<typename T>
void launch_relative_position_bias_forward(const float *embedding,
                                           T *output,
                                           int num_heads,
                                           int q_len,
                                           int k_len,
                                           int num_buckets,
                                           int max_distance,
                                           int bidirectional,
                                           cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_heads)*q_len*k_len;
  if(total<=0)
  {
    return;
  }
  const int grid=static_cast<int>((total+g_cu_block_size-1)/g_cu_block_size);
  relative_position_bias_forward_kernel<T>
    <<<grid,g_cu_block_size,0,stream>>>(embedding,
                                        output,
                                        num_heads,
                                        q_len,
                                        k_len,
                                        num_buckets,
                                        max_distance,
                                        bidirectional);
}

template void launch_relative_position_bias_forward<float>(const float *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           cudaStream_t);
template void launch_relative_position_bias_forward<__half>(const float *,
                                                            __half *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            cudaStream_t);
template void launch_relative_position_bias_forward<__nv_bfloat16>(const float *,
                                                                   __nv_bfloat16 *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   cudaStream_t);

template<typename T>
void launch_relative_position_bias_backward(const T *grad_output,
                                            float *grad_embedding,
                                            int num_heads,
                                            int q_len,
                                            int k_len,
                                            int num_buckets,
                                            int max_distance,
                                            int bidirectional,
                                            cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_heads)*q_len*k_len;
  if(total<=0)
  {
    return;
  }
  const int grid=static_cast<int>((total+g_cu_block_size-1)/g_cu_block_size);
  relative_position_bias_backward_kernel<T>
    <<<grid,g_cu_block_size,0,stream>>>(grad_output,
                                        grad_embedding,
                                        num_heads,
                                        q_len,
                                        k_len,
                                        num_buckets,
                                        max_distance,
                                        bidirectional);
}

template void launch_relative_position_bias_backward<float>(const float *,
                                                            float *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            cudaStream_t);
template void launch_relative_position_bias_backward<__half>(const __half *,
                                                             float *,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             cudaStream_t);
template void launch_relative_position_bias_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    float *,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    cudaStream_t);

// end of former extern "C" block

