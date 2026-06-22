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
// Element-wise CUDA kernels (add / sub / mul / div / sqrt, scalar variants,
// 2D bias add + bias gradient, parallel reduction sum) and the generic
// reductions (sum_axis0, sum_axis1, sum_of_squares, logsumexp). Carved
// verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_elementwise.cuh
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
// Element-wise Add Kernel (tensor + tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_add_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])+float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_add_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int64_t offset,
                                            const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])+float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Add Scalar Kernel (tensor + scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_add_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])+scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_add_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int64_t offset,
                                                   const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])+scalar);
  }
}

//------------------------------------------------------------------------------
// Bias Add (2D: broadcast bias over batch rows)
// Scalar-indexed template (col=idx%units), no vectorization.
//------------------------------------------------------------------------------
template<typename T>
__global__ void bias_add_2d_kernel(const T *input,
                                   const T *bias,
                                   T *output,
                                   const int units,
                                   const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int col=idx%units;
    output[idx]=T(float(input[idx])+float(bias[col]));
  }
}

//------------------------------------------------------------------------------
// Bias Gradient (sum over batch rows)
// fp32 accumulator, final cast to T on store.
//------------------------------------------------------------------------------
template<typename T>
__global__ void bias_grad_2d_kernel(const T *grad,
                                    T *bias_grad,
                                    const int batch,
                                    const int units)
{
  const int u=blockIdx.x*blockDim.x+threadIdx.x;
  if(u<units)
  {
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=float(grad[static_cast<int64_t>(b)*units+u]);
    }
    bias_grad[u]=T(sum);
  }
}

//------------------------------------------------------------------------------
// Element-wise Subtract Kernel (tensor - tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_sub_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])-float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_sub_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int64_t offset,
                                            const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])-float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Subtract Scalar Kernel (tensor - scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_sub_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])-scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_sub_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int64_t offset,
                                                   const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])-scalar);
  }
}

//------------------------------------------------------------------------------
// Element-wise Multiply Kernel (tensor * tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_mul_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])*float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_mul_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int64_t offset,
                                            const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])*float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Multiply Scalar Kernel (tensor * scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_mul_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])*scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_mul_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int64_t offset,
                                                   const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])*scalar);
  }
}

//------------------------------------------------------------------------------
// Element-wise Divide Kernel (tensor / tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_div_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])/float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_div_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int64_t offset,
                                            const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])/float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Divide Scalar Kernel (tensor / scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_div_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])/scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_div_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int64_t offset,
                                                   const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])/scalar);
  }
}

//------------------------------------------------------------------------------
// Element-wise Sqrt Kernel
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_sqrt_kernel(const T *a,
                                        T *result,
                                        const int64_t n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(sqrtf(float(a_lane[i])));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_sqrt_tail_kernel(const T *a,
                                             T *result,
                                             const int64_t offset,
                                             const int64_t n)
{
  const int64_t idx=offset+static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(sqrtf(float(a[idx])));
  }
}

//------------------------------------------------------------------------------
// Reduction Sum Kernel (parallel reduction)
// Uses shared memory for efficient intra-block reduction
//------------------------------------------------------------------------------
template<typename T>
__global__ void reduction_sum_kernel(const T *input,
                                     float *output,
                                     const int64_t n)
{
  const int tid=threadIdx.x;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;

  float val=0.0f;
  if(idx<n)
  {
    val=float(input[idx]);
  }

  // Warp shuffle reduction (no shared memory for intra-warp)
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
    output[blockIdx.x]=val;
  }
}

//------------------------------------------------------------------------------
// Element-wise launcher functions
//------------------------------------------------------------------------------
template<typename T>
void launch_elementwise_add(const T *a,
                            const T *b,
                            T *result,
                            const int64_t n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_add_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_add_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_add<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_elementwise_add<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_elementwise_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int64_t,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_add_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int64_t n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_add_scalar_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_add_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_add_scalar<float>(const float *,float,float *,int64_t,cudaStream_t);
template void launch_elementwise_add_scalar<__half>(const __half *,float,__half *,int64_t,cudaStream_t);
template void launch_elementwise_add_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);

template<typename T>
void launch_bias_add_2d(const T *input,
                        const T *bias,
                        T *output,
                        const int batch,
                        const int units,
                        cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch)*units;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  bias_add_2d_kernel<T><<<num_blocks,block_size,0,stream>>>(input,bias,output,units,total);
}
template void launch_bias_add_2d<float>(const float *,const float *,float *,int,int,cudaStream_t);
template void launch_bias_add_2d<__half>(const __half *,const __half *,__half *,int,int,cudaStream_t);
template void launch_bias_add_2d<__nv_bfloat16>(const __nv_bfloat16 *,
                                                const __nv_bfloat16 *,
                                                __nv_bfloat16 *,
                                                int,
                                                int,
                                                cudaStream_t);

template<typename T>
void launch_bias_grad_2d(const T *grad_output,
                         T *bias_grad,
                         const int batch,
                         const int units,
                         cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(units+block_size-1)/block_size;
  bias_grad_2d_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_output,bias_grad,batch,units);
}
template void launch_bias_grad_2d<float>(const float *,float *,int,int,cudaStream_t);
template void launch_bias_grad_2d<__half>(const __half *,__half *,int,int,cudaStream_t);
template void launch_bias_grad_2d<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int,int,cudaStream_t);

template<typename T>
void launch_elementwise_sub(const T *a,
                            const T *b,
                            T *result,
                            const int64_t n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_sub_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_sub_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_sub<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_elementwise_sub<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_elementwise_sub<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int64_t,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_sub_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int64_t n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_sub_scalar_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_sub_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_sub_scalar<float>(const float *,float,float *,int64_t,cudaStream_t);
template void launch_elementwise_sub_scalar<__half>(const __half *,float,__half *,int64_t,cudaStream_t);
template void launch_elementwise_sub_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);

template<typename T>
void launch_elementwise_mul(const T *a,
                            const T *b,
                            T *result,
                            const int64_t n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_mul_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_mul_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_mul<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_elementwise_mul<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_elementwise_mul<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int64_t,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_mul_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int64_t n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_mul_scalar_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_mul_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_mul_scalar<float>(const float *,float,float *,int64_t,cudaStream_t);
template void launch_elementwise_mul_scalar<__half>(const __half *,float,__half *,int64_t,cudaStream_t);
template void launch_elementwise_mul_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);

template<typename T>
void launch_elementwise_div(const T *a,
                            const T *b,
                            T *result,
                            const int64_t n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_div_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_div_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_div<float>(const float *,const float *,float *,int64_t,cudaStream_t);
template void launch_elementwise_div<__half>(const __half *,const __half *,__half *,int64_t,cudaStream_t);
template void launch_elementwise_div<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int64_t,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_div_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int64_t n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_div_scalar_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_div_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_div_scalar<float>(const float *,float,float *,int64_t,cudaStream_t);
template void launch_elementwise_div_scalar<__half>(const __half *,float,__half *,int64_t,cudaStream_t);
template void launch_elementwise_div_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int64_t,
                                                           cudaStream_t);

template<typename T>
void launch_elementwise_sqrt(const T *a,
                             T *result,
                             const int64_t n,
                             cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int64_t n_vec=n/lanes;
  const int tail=static_cast<int>(n-n_vec*lanes);
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_cu_block_size-1)/g_cu_block_size;
    elementwise_sqrt_kernel<T><<<num_blocks,g_cu_block_size,0,stream>>>(a,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_sqrt_tail_kernel<T><<<1,tail,0,stream>>>(a,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_sqrt<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_elementwise_sqrt<__half>(const __half *,__half *,int64_t,cudaStream_t);
template void launch_elementwise_sqrt<__nv_bfloat16>(const __nv_bfloat16 *,__nv_bfloat16 *,int64_t,cudaStream_t);

template<typename T>
void launch_reduction_sum(const T *input,
                          float *output,
                          const int64_t n,
                          cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  reduction_sum_kernel<T><<<num_blocks,block_size,0,stream>>>(input,output,n);
}
template void launch_reduction_sum<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_reduction_sum<__half>(const __half *,float *,int64_t,cudaStream_t);
template void launch_reduction_sum<__nv_bfloat16>(const __nv_bfloat16 *,float *,int64_t,cudaStream_t);

//------------------------------------------------------------------------------
// Sum Axis 0 Kernel (sum over batch)
// input: [batch, dim], output: [dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void sum_axis0_kernel(const T *input,
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
      sum+=float(input[static_cast<int64_t>(b)*dim+d]);
    }
    output[d]=sum;
  }
}

template<typename T>
void launch_sum_axis0(const T *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(dim+block_size-1)/block_size;
  sum_axis0_kernel<T><<<num_blocks,block_size,0,stream>>>(input,output,batch,dim);
}
template void launch_sum_axis0<float>(const float *,float *,int,int,cudaStream_t);
template void launch_sum_axis0<__half>(const __half *,float *,int,int,cudaStream_t);
template void launch_sum_axis0<__nv_bfloat16>(const __nv_bfloat16 *,float *,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// Sum Axis 1 Kernel (sum over dim)
// input: [batch, dim], output: [batch]
//------------------------------------------------------------------------------
template<typename T>
__global__ void sum_axis1_kernel(const T *input,
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
      sum+=float(input[static_cast<int64_t>(b)*dim+d]);
    }
    output[b]=sum;
  }
}

template<typename T>
void launch_sum_axis1(const T *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(batch+block_size-1)/block_size;
  sum_axis1_kernel<T><<<num_blocks,block_size,0,stream>>>(input,output,batch,dim);
}
template void launch_sum_axis1<float>(const float *,float *,int,int,cudaStream_t);
template void launch_sum_axis1<__half>(const __half *,float *,int,int,cudaStream_t);
template void launch_sum_axis1<__nv_bfloat16>(const __nv_bfloat16 *,float *,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// Sum of Squares Kernel
// Computes sum(x[i]^2) over all n elements.
// Uses block-level reduction with atomicAdd to a single output float.
// Caller must zero the output before launch.
//------------------------------------------------------------------------------
template<typename T>
__global__ void sum_of_squares_kernel(const T *input,
                                      float *output,
                                      const int64_t n)
{
  extern __shared__ float shared[];
  const int tid=threadIdx.x;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+tid;

  float local_sum=0.0f;
  if(idx<n)
  {
    const float val=float(input[idx]);
    if(isnan(val)==false && isinf(val)==false)
    {
      local_sum=val*val;
    }
  }
  shared[tid]=local_sum;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      shared[tid]+=shared[tid+stride];
    }
    __syncthreads();
  }

  if(tid==0)
  {
    atomicAdd(output,shared[0]);
  }
}

template<typename T>
void launch_sum_of_squares(const T *input,
                           float *output,
                           const int64_t n,
                           cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  sum_of_squares_kernel<T><<<num_blocks,block_size,block_size*sizeof(float),stream>>>(input,output,n);
}
template void launch_sum_of_squares<float>(const float *,float *,int64_t,cudaStream_t);
template void launch_sum_of_squares<__half>(const __half *,float *,int64_t,cudaStream_t);
template void launch_sum_of_squares<__nv_bfloat16>(const __nv_bfloat16 *,float *,int64_t,cudaStream_t);

//------------------------------------------------------------------------------
// Log-Sum-Exp Kernel
// input: [batch, dim], output: [batch]
// output[b] = log(sum_d(exp(input[b,d])))
//------------------------------------------------------------------------------
template<typename T>
__global__ void logsumexp_kernel(const T *input,
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
  const T *row=input+static_cast<int64_t>(b)*dim;

  // Find max
  float local_max=g_cu_neg_sentinel;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_max=fmaxf(local_max,float(row[d]));
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
    local_sum+=expf(float(row[d])-max_val);
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

template<typename T>
void launch_logsumexp(const T *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=batch;
  const size_t shared_size=g_cu_softmax_stat_arrays*block_size*sizeof(float);
  logsumexp_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(input,output,batch,dim);
}
template void launch_logsumexp<float>(const float *,float *,int,int,cudaStream_t);
template void launch_logsumexp<__half>(const __half *,float *,int,int,cudaStream_t);
template void launch_logsumexp<__nv_bfloat16>(const __nv_bfloat16 *,float *,int,int,cudaStream_t);
