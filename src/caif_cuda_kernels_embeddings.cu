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
// Embedding CUDA kernels: token embedding lookup/backward, float-id
// conversion, patch embedding (extract/backward, CLS prepend/grad),
// positional encoding add + table backward.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_embeddings.cuh
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

//==============================================================================
// Embedding kernels
//==============================================================================

//------------------------------------------------------------------------------
// Embedding Lookup Kernel - Vectorized 2D Grid (uint32 token IDs)
// Grid: (num_tokens, ceil(dim/THREADS_PER_BLOCK/4))
// Each thread loads a float4 (4 elements) - no div/mod needed
//------------------------------------------------------------------------------
template<typename T>
__global__ void embedding_lookup_kernel(const T *__restrict__ table,
                                        const unsigned int *__restrict__ token_ids,
                                        T *__restrict__ output,
                                        const int num_tokens,
                                        const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x+threadIdx.x;

  if(token_idx>=num_tokens || d>=dim)
  {
    return;
  }

  const unsigned int token_id=token_ids[token_idx];
  output[static_cast<int64_t>(token_idx)*dim+d]=table[static_cast<int64_t>(token_id)*dim+d];
}

//------------------------------------------------------------------------------
// Embedding Lookup Kernel - 2D Grid (float-encoded token IDs)
//------------------------------------------------------------------------------
template<typename T>
__global__ void embedding_lookup_float_kernel(const T *__restrict__ table,
                                              const float *__restrict__ float_ids,
                                              T *__restrict__ output,
                                              const int num_tokens,
                                              const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x+threadIdx.x;

  if(token_idx>=num_tokens || d>=dim)
  {
    return;
  }

  const unsigned int token_id=static_cast<unsigned int>(float_ids[token_idx]);
  output[static_cast<int64_t>(token_idx)*dim+d]=table[static_cast<int64_t>(token_id)*dim+d];
}

//------------------------------------------------------------------------------
// Float-to-uint conversion kernel (runs on GPU, eliminates host roundtrip)
//------------------------------------------------------------------------------
__global__ void float_to_uint_kernel(const float *float_ids,
                                     unsigned int *uint_ids,
                                     const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    uint_ids[idx]=static_cast<unsigned int>(float_ids[idx]);
  }
}

//------------------------------------------------------------------------------
// Embedding Backward Kernel - Vectorized 2D Grid (scatter-add gradients)
// Uses float4 vectorized loads for better bandwidth
//------------------------------------------------------------------------------
template<typename T>
__global__ void embedding_backward_kernel(const T *grad_output,
                                          const unsigned int *token_ids,
                                          float *grad_table,
                                          const int num_tokens,
                                          const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x+threadIdx.x;

  if(token_idx>=num_tokens || d>=dim)
  {
    return;
  }

  const unsigned int token_id=token_ids[token_idx];
  atomicAdd(&grad_table[static_cast<int64_t>(token_id)*dim+d],caif_load_f<T>(grad_output[static_cast<int64_t>(token_idx)*dim+d]));
}

//------------------------------------------------------------------------------
// Embedding Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_embedding_lookup(const T *table,
                             const unsigned int *token_ids,
                             T *output,
                             const int num_tokens,
                             const int dim,
                             cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int y_blocks=(dim+block_size-1)/block_size;
  dim3 grid(num_tokens,y_blocks);
  embedding_lookup_kernel<T><<<grid,block_size,0,stream>>>(table,token_ids,output,num_tokens,dim);
}

template void launch_embedding_lookup<float>(const float *,const unsigned int *,float *,int,int,cudaStream_t);
template void launch_embedding_lookup<__half>(const __half *,const unsigned int *,__half *,int,int,cudaStream_t);
template void launch_embedding_lookup<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     const unsigned int *,
                                                     __nv_bfloat16 *,
                                                     int,
                                                     int,
                                                     cudaStream_t);

template<typename T>
void launch_embedding_lookup_float(const T *table,
                                   const float *float_ids,
                                   T *output,
                                   const int num_tokens,
                                   const int dim,
                                   cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int y_blocks=(dim+block_size-1)/block_size;
  dim3 grid(num_tokens,y_blocks);
  embedding_lookup_float_kernel<T><<<grid,block_size,0,stream>>>(table,float_ids,output,num_tokens,dim);
}

template void launch_embedding_lookup_float<float>(const float *,const float *,float *,int,int,cudaStream_t);
template void launch_embedding_lookup_float<__half>(const __half *,const float *,__half *,int,int,cudaStream_t);
template void launch_embedding_lookup_float<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const float *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           int,
                                                           cudaStream_t);

void launch_float_to_uint(const float *float_ids,
                          unsigned int *uint_ids,
                          const int64_t n,
                          cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  float_to_uint_kernel<<<num_blocks,block_size,0,stream>>>(float_ids,uint_ids,n);
}

template<typename T>
void launch_embedding_backward(const T *grad_output,
                               const unsigned int *token_ids,
                               float *grad_table,
                               const int num_tokens,
                               const int dim,
                               cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int y_blocks=(dim+block_size-1)/block_size;
  dim3 grid(num_tokens,y_blocks);
  embedding_backward_kernel<T><<<grid,block_size,0,stream>>>(grad_output,token_ids,grad_table,num_tokens,dim);
}

template void launch_embedding_backward<float>(const float *,const unsigned int *,float *,int,int,cudaStream_t);
template void launch_embedding_backward<__half>(const __half *,
                                                const unsigned int *,
                                                float *,
                                                int,
                                                int,
                                                cudaStream_t);
template void launch_embedding_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const unsigned int *,
                                                       float *,
                                                       int,
                                                       int,
                                                       cudaStream_t);

//==============================================================================
// Patch embedding kernels
// Uses caif_atomic_add<T>, caif_load_f<T>, caif_store_f<T> defined at top of file.
//==============================================================================

//------------------------------------------------------------------------------
// Extract Patches Kernel (im2col for non-overlapping patches)
// input:  [batch, height, width, channels] (BHWC)
// output: [batch * num_patches, patch_flat_dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void extract_patches_kernel(const T *input,
                                       T *output,
                                       const int batch,
                                       const int height,
                                       const int width,
                                       const int channels,
                                       const int patch_size,
                                       const int num_patches_h,
                                       const int num_patches_w,
                                       const int patch_flat_dim)
{
  const int64_t total=static_cast<int64_t>(batch)*num_patches_h*num_patches_w*patch_flat_dim;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int num_patches=num_patches_h*num_patches_w;
    const int flat_pos=static_cast<int>(idx%patch_flat_dim);
    const int patch_idx=static_cast<int>((idx/patch_flat_dim)%num_patches);
    const int b=static_cast<int>(idx/(static_cast<int64_t>(patch_flat_dim)*num_patches));

    const int ph=patch_idx/num_patches_w;
    const int pw=patch_idx%num_patches_w;

    const int c=flat_pos%channels;
    const int local_pixel=flat_pos/channels;
    const int local_h=local_pixel/patch_size;
    const int local_w=local_pixel%patch_size;

    const int global_h=ph*patch_size+local_h;
    const int global_w=pw*patch_size+local_w;

    const int64_t input_idx=((static_cast<int64_t>(b)*height+global_h)*width+global_w)*channels+c;
    output[idx]=input[input_idx];
  }
}

//------------------------------------------------------------------------------
// Extract Patches Backward Kernel (col2im scatter-add)
//------------------------------------------------------------------------------
template<typename T>
__global__ void extract_patches_backward_kernel(const T *grad_patches,
                                                T *grad_input,
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
  const int64_t total=static_cast<int64_t>(batch)*num_patches*patch_flat_dim;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int flat_pos=static_cast<int>(idx%patch_flat_dim);
    const int patch_idx=static_cast<int>((idx/patch_flat_dim)%num_patches);
    const int b=static_cast<int>(idx/(static_cast<int64_t>(patch_flat_dim)*num_patches));

    const int ph=patch_idx/num_patches_w;
    const int pw=patch_idx%num_patches_w;

    const int c=flat_pos%channels;
    const int local_pixel=flat_pos/channels;
    const int local_h=local_pixel/patch_size;
    const int local_w=local_pixel%patch_size;

    const int global_h=ph*patch_size+local_h;
    const int global_w=pw*patch_size+local_w;

    const int64_t input_idx=((static_cast<int64_t>(b)*height+global_h)*width+global_w)*channels+c;
    caif_atomic_add<T>(&grad_input[input_idx],grad_patches[idx]);
  }
}

//------------------------------------------------------------------------------
// CLS Prepend Kernel
// Prepend CLS token at position 0, shift patches to 1..N
// patches: [batch, num_patches, dim], cls: [1, dim]
// output:  [batch, num_patches+1, dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void cls_prepend_kernel(const T *patches,
                                   const T *cls_token,
                                   T *output,
                                   const int batch,
                                   const int num_patches,
                                   const int dim)
{
  const int out_seq=num_patches+1;
  const int64_t total=static_cast<int64_t>(batch)*out_seq*dim;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=static_cast<int>(idx%dim);
    const int s=static_cast<int>((idx/dim)%out_seq);
    const int b=static_cast<int>(idx/(static_cast<int64_t>(dim)*out_seq));

    if(s==0)
    {
      output[idx]=cls_token[d];
    }
    else
    {
      output[idx]=patches[(static_cast<int64_t>(b)*num_patches+(s-1))*dim+d];
    }
  }
}

//------------------------------------------------------------------------------
// CLS Gradient Extract Kernel
// Split CLS gradient from patch gradients
// grad_output: [batch, num_patches+1, dim]
// grad_cls: [1, dim] (summed over batch), grad_patches: [batch, num_patches, dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void cls_grad_extract_kernel(const T *grad_output,
                                        T *grad_cls,
                                        T *grad_patches,
                                        const int batch,
                                        const int num_patches,
                                        const int dim)
{
  const int out_seq=num_patches+1;
  const int64_t total=static_cast<int64_t>(batch)*out_seq*dim;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=static_cast<int>(idx%dim);
    const int s=static_cast<int>((idx/dim)%out_seq);
    const int b=static_cast<int>(idx/(static_cast<int64_t>(dim)*out_seq));

    if(s==0)
    {
      caif_atomic_add<T>(&grad_cls[d],grad_output[idx]);
    }
    else
    {
      grad_patches[(static_cast<int64_t>(b)*num_patches+(s-1))*dim+d]=grad_output[idx];
    }
  }
}

//------------------------------------------------------------------------------
// Patch Embedding Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_extract_patches(const T *input,
                            T *output,
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
  const int64_t n=static_cast<int64_t>(batch)*num_patches_h*num_patches_w*patch_flat_dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  extract_patches_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                                output,
                                                                batch,
                                                                height,
                                                                width,
                                                                channels,
                                                                patch_size,
                                                                num_patches_h,
                                                                num_patches_w,
                                                                patch_flat_dim);
}

template void launch_extract_patches<float>(const float *,
                                            float *,
                                            const int,
                                            const int,
                                            const int,
                                            const int,
                                            const int,
                                            const int,
                                            const int,
                                            const int,
                                            cudaStream_t);
template void launch_extract_patches<__half>(const __half *,
                                             __half *,
                                             const int,
                                             const int,
                                             const int,
                                             const int,
                                             const int,
                                             const int,
                                             const int,
                                             const int,
                                             cudaStream_t);
template void launch_extract_patches<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    const int,
                                                    const int,
                                                    const int,
                                                    const int,
                                                    const int,
                                                    const int,
                                                    const int,
                                                    const int,
                                                    cudaStream_t);

template<typename T>
void launch_extract_patches_backward(const T *grad_patches,
                                     T *grad_input,
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
  const int64_t n=static_cast<int64_t>(batch)*num_patches_h*num_patches_w*patch_flat_dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  extract_patches_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_patches,
                                                                         grad_input,
                                                                         batch,
                                                                         height,
                                                                         width,
                                                                         channels,
                                                                         patch_size,
                                                                         num_patches_h,
                                                                         num_patches_w,
                                                                         patch_flat_dim);
}

template void launch_extract_patches_backward<float>(const float *,
                                                     float *,
                                                     const int,
                                                     const int,
                                                     const int,
                                                     const int,
                                                     const int,
                                                     const int,
                                                     const int,
                                                     const int,
                                                     cudaStream_t);
template void launch_extract_patches_backward<__half>(const __half *,
                                                      __half *,
                                                      const int,
                                                      const int,
                                                      const int,
                                                      const int,
                                                      const int,
                                                      const int,
                                                      const int,
                                                      const int,
                                                      cudaStream_t);
template void launch_extract_patches_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             const int,
                                                             const int,
                                                             const int,
                                                             const int,
                                                             const int,
                                                             const int,
                                                             const int,
                                                             const int,
                                                             cudaStream_t);

template<typename T>
void launch_cls_prepend(const T *patches,
                        const T *cls_token,
                        T *output,
                        const int batch,
                        const int num_patches,
                        const int dim,
                        cudaStream_t stream)
{
  const int out_seq=num_patches+1;
  const int64_t n=static_cast<int64_t>(batch)*out_seq*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  cls_prepend_kernel<T><<<num_blocks,block_size,0,stream>>>(patches,cls_token,output,batch,num_patches,dim);
}

template void launch_cls_prepend<float>(const float *,
                                        const float *,
                                        float *,
                                        const int,
                                        const int,
                                        const int,
                                        cudaStream_t);
template void launch_cls_prepend<__half>(const __half *,
                                         const __half *,
                                         __half *,
                                         const int,
                                         const int,
                                         const int,
                                         cudaStream_t);
template void launch_cls_prepend<__nv_bfloat16>(const __nv_bfloat16 *,
                                                const __nv_bfloat16 *,
                                                __nv_bfloat16 *,
                                                const int,
                                                const int,
                                                const int,
                                                cudaStream_t);

template<typename T>
void launch_cls_grad_extract(const T *grad_output,
                             T *grad_cls,
                             T *grad_patches,
                             const int batch,
                             const int num_patches,
                             const int dim,
                             cudaStream_t stream)
{
  const int out_seq=num_patches+1;
  const int64_t n=static_cast<int64_t>(batch)*out_seq*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  cls_grad_extract_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_output,
                                                                 grad_cls,
                                                                 grad_patches,
                                                                 batch,
                                                                 num_patches,
                                                                 dim);
}

template void launch_cls_grad_extract<float>(const float *,
                                             float *,
                                             float *,
                                             const int,
                                             const int,
                                             const int,
                                             cudaStream_t);
template void launch_cls_grad_extract<__half>(const __half *,
                                              __half *,
                                              __half *,
                                              const int,
                                              const int,
                                              const int,
                                              cudaStream_t);
template void launch_cls_grad_extract<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     __nv_bfloat16 *,
                                                     __nv_bfloat16 *,
                                                     const int,
                                                     const int,
                                                     const int,
                                                     cudaStream_t);

//==============================================================================
// Positional encoding kernels
//==============================================================================

//------------------------------------------------------------------------------
// Add Positional Encoding Kernel
// output[b,s,d] = input[b,s,d] + pe_table[s,d]
//------------------------------------------------------------------------------
template<typename T>
__global__ void add_positional_encoding_kernel(const T *input,
                                               const T *pe_table,
                                               T *output,
                                               const int batch,
                                               const int seq_len,
                                               const int dim)
{
  const int64_t total=static_cast<int64_t>(batch)*seq_len*dim;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=static_cast<int>(idx%dim);
    const int s=static_cast<int>((idx/dim)%seq_len);
    const float v=static_cast<float>(input[idx])+static_cast<float>(pe_table[static_cast<int64_t>(s)*dim+d]);
    output[idx]=static_cast<T>(v);
  }
}

//------------------------------------------------------------------------------
// PE Table Backward Kernel
// grad_table[s,d] = sum_b grad_output[b,s,d]
// One thread per (s,d) pair, loops over batch. Sum accumulates in float
// for numerical stability at fp16/bf16.
//------------------------------------------------------------------------------
template<typename T>
__global__ void pe_table_backward_kernel(const T *grad_output,
                                         T *grad_table,
                                         const int batch,
                                         const int seq_len,
                                         const int dim)
{
  const int64_t total=static_cast<int64_t>(seq_len)*dim;
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int s=static_cast<int>(idx/dim);
    const int d=static_cast<int>(idx%dim);
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=static_cast<float>(grad_output[(static_cast<int64_t>(b)*seq_len+s)*dim+d]);
    }
    grad_table[idx]=static_cast<T>(sum);
  }
}

//------------------------------------------------------------------------------
// Positional Encoding Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_add_positional_encoding(const T *input,
                                    const T *pe_table,
                                    T *output,
                                    const int batch,
                                    const int seq_len,
                                    const int dim,
                                    cudaStream_t stream)
{
  const int64_t n=static_cast<int64_t>(batch)*seq_len*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  add_positional_encoding_kernel<T><<<num_blocks,block_size,0,stream>>>(input,pe_table,output,batch,seq_len,dim);
}
template void launch_add_positional_encoding<float>(const float *,
                                                    const float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);
template void launch_add_positional_encoding<__half>(const __half *,
                                                     const __half *,
                                                     __half *,
                                                     int,
                                                     int,
                                                     int,
                                                     cudaStream_t);
template void launch_add_positional_encoding<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int,
                                                            int,
                                                            int,
                                                            cudaStream_t);

template<typename T>
void launch_pe_table_backward(const T *grad_output,
                              T *grad_table,
                              const int batch,
                              const int seq_len,
                              const int dim,
                              cudaStream_t stream)
{
  const int64_t n=static_cast<int64_t>(seq_len)*dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  pe_table_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_output,grad_table,batch,seq_len,dim);
}
template void launch_pe_table_backward<float>(const float *,float *,int,int,int,cudaStream_t);
template void launch_pe_table_backward<__half>(const __half *,__half *,int,int,int,cudaStream_t);
template void launch_pe_table_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      __nv_bfloat16 *,
                                                      int,
                                                      int,
                                                      int,
                                                      cudaStream_t);

