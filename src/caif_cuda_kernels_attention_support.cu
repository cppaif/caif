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
// Attention-support CUDA kernels: transpose 0213 (+strided), causal and
// prefix mask fill/grad, attention softmax forward/backward, RoPE
// (full, partial, offset, backward), GQA repeat/reduce, KV-cache
// append, fill_fp32, and the softmax block-size selection helpers.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_attention_support.cuh
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
// RoPE dimension pairing style — must match CAIF_RoPEStyle in caif_cuda_kernels.h
//------------------------------------------------------------------------------
enum CAIF_RoPEStyle
{
  CAIF_ROPE_INTERLEAVED=0,
  CAIF_ROPE_HALF_SPLIT=1
};

//------------------------------------------------------------------------------
// Per-arch softmax block-size selection. Sizes live in
// caif_cuda_kernels_constants.cuh (g_cu_softmax_block_size_*). The row-reduce
// kernels no longer use inter-warp shared-memory tree reduction; intra-warp
// is __shfl_xor_sync only, cross-warp combine stages (num_warps) floats.
//------------------------------------------------------------------------------
constexpr int SelectSoftmaxBlockSize(const int cc_major,
                                     const int cc_minor)
{
  if(cc_major==7&&cc_minor==5)
  {
    return g_cu_softmax_block_size_sm75;
  }
  if(cc_major==8&&cc_minor==0)
  {
    return g_cu_softmax_block_size_sm80;
  }
  if(cc_major==8&&cc_minor==6)
  {
    return g_cu_softmax_block_size_sm86;
  }
  if(cc_major==8&&cc_minor==9)
  {
    return g_cu_softmax_block_size_sm89;
  }
  if(cc_major==9)
  {
    return g_cu_softmax_block_size_sm90;
  }
  if(cc_major==12)
  {
    return g_cu_softmax_block_size_sm120;
  }
  return g_cu_softmax_block_size_default;
}

// Cached per-process softmax block size — queries compute capability once
// on first call, reuses thereafter. Single-device; multi-GPU would key by
// device id. Used by launch_attention_softmax{,_backward}.
// Reads the current device's compute capability and selects its block size.
inline int DeviceSoftmaxBlockSize()
{
  int dev=0;
  cudaGetDevice(&dev);
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props,dev);
  return SelectSoftmaxBlockSize(props.major,props.minor);
}

// Cached per-process softmax block size — queries compute capability once
// on first call, reuses thereafter. Single-device; multi-GPU would key by
// device id. Used by launch_attention_softmax{,_backward}.
inline int SoftmaxBlockSize()
{
  static const int block_size=DeviceSoftmaxBlockSize();
  return block_size;
}

//------------------------------------------------------------------------------
// Transpose 0213 Kernel
// Swaps dims 1 and 2 of a logical 4D tensor [batch, dim0, dim1, dim2]
// -> [batch, dim1, dim0, dim2]
// Element-parallel: one thread per element.
//------------------------------------------------------------------------------
template<typename T>
__global__ void transpose_0213_kernel(const T *input,
                                      T *output,
                                      const int dim0,
                                      const int dim1,
                                      const int dim2,
                                      const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
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
    const int64_t out_idx=static_cast<int64_t>(b)*dim1*out_stride_d1+
                          d1*out_stride_d1+
                          d0*dim2+
                          d2;
    output[out_idx]=input[idx];
  }
}

//------------------------------------------------------------------------------
// Transpose 0213 Strided Kernel
// Same logical reshuffle as transpose_0213_kernel, but reads from an input
// whose stride along dim0 may be LARGER than dim1*dim2 (i.e. the input is a
// width-slice of a wider packed row). Caller supplies the base pointer
// positioned at column offset 0 of the slice, and input_d0_stride = full row
// width of the packed buffer.
// Input logical view: [batch, dim0, dim1, dim2] with row stride input_d0_stride
// Output layout:      [batch, dim1, dim0, dim2] contiguous
//------------------------------------------------------------------------------
template<typename T>
__global__ void transpose_0213_strided_kernel(const T *input,
                                              T *output,
                                              const int dim0,
                                              const int dim1,
                                              const int dim2,
                                              const int input_d0_stride,
                                              const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    // Decompose flat output index into [b, d0, d1, d2]
    const int stride_d0=dim1*dim2;
    const int stride_batch=dim0*stride_d0;
    const int b=idx/stride_batch;
    const int rem=idx%stride_batch;
    const int d0=rem/stride_d0;
    const int rem2=rem%stride_d0;
    const int d1=rem2/dim2;
    const int d2=rem2%dim2;

    // Input index uses input_d0_stride (> dim1*dim2) between consecutive d0
    const int input_batch_stride=dim0*input_d0_stride;
    const int64_t in_idx=static_cast<int64_t>(b)*input_batch_stride+
                         d0*input_d0_stride+
                         d1*dim2+
                         d2;

    // Output index in [b, d1, d0, d2]
    const int out_stride_d1=dim0*dim2;
    const int64_t out_idx=static_cast<int64_t>(b)*dim1*out_stride_d1+
                          d1*out_stride_d1+
                          d0*dim2+
                          d2;
    output[out_idx]=input[in_idx];
  }
}

//------------------------------------------------------------------------------
// Fill an fp32 buffer with a scalar value (device-side; no host staging).
//------------------------------------------------------------------------------
__global__ void fill_fp32_kernel(float *data,const float value,const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    data[idx]=value;
  }
}

void launch_fill_fp32(float *data,const float value,const int64_t n,cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  fill_fp32_kernel<<<num_blocks,block_size,0,stream>>>(data,value,n);
}

//------------------------------------------------------------------------------
// Attention logit soft-cap (Gemma-2/3): scores = cap*tanh(scores/cap), applied
// in place to the materialized [bh, q, k] score tensor in the explicit
// (non-flash) attention path, after the 1/sqrt(head_dim) scale. The flash path
// applies the same cap inside its kernel (see caif_cuda_kernels_flash_self.cu).
//------------------------------------------------------------------------------
template<typename T>
__global__ void attn_logit_softcap_kernel(T *scores,const float softcap,const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float s=float(scores[idx]);
    scores[idx]=T(softcap*tanhf(s/softcap));
  }
}

template<typename T>
void launch_attn_logit_softcap(T *scores,const float softcap,const int64_t n,cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  attn_logit_softcap_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,softcap,n);
}

template void launch_attn_logit_softcap<float>(float *,const float,const int64_t,cudaStream_t);
template void launch_attn_logit_softcap<__half>(__half *,const float,const int64_t,cudaStream_t);
template void launch_attn_logit_softcap<__nv_bfloat16>(__nv_bfloat16 *,const float,const int64_t,cudaStream_t);

//------------------------------------------------------------------------------
// Soft-cap backward for the explicit path: multiplies the gradient w.r.t. the
// capped scores by the cap's derivative 1 - tanh^2(S_val/cap), where S_val is
// the pre-cap scaled score (scale * Q.K^T), recomputed by the caller.
//------------------------------------------------------------------------------
template<typename T>
__global__ void attn_logit_softcap_backward_kernel(T *grad,
                                                   const T *scores_val,
                                                   const float softcap,
                                                   const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float t=tanhf(float(scores_val[idx])/softcap);
    grad[idx]=T(float(grad[idx])*(1.0f-t*t));
  }
}

template<typename T>
void launch_attn_logit_softcap_backward(T *grad,
                                        const T *scores_val,
                                        const float softcap,
                                        const int64_t n,
                                        cudaStream_t stream)
{
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((n+block_size-1)/block_size);
  attn_logit_softcap_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(grad,scores_val,softcap,n);
}

template void launch_attn_logit_softcap_backward<float>(float *,
                                                        const float *,
                                                        const float,
                                                        const int64_t,
                                                        cudaStream_t);
template void launch_attn_logit_softcap_backward<__half>(__half *,
                                                         const __half *,
                                                         const float,
                                                         const int64_t,
                                                         cudaStream_t);
template void launch_attn_logit_softcap_backward<__nv_bfloat16>(__nv_bfloat16 *,
                                                                const __nv_bfloat16 *,
                                                                const float,
                                                                const int64_t,
                                                                cudaStream_t);

//------------------------------------------------------------------------------
// Causal Mask Fill Kernel
// Sets upper triangle (j > i) to -1e9 for each [seq_len, seq_len] matrix.
// Layout: [num_matrices, seq_len, seq_len]
//------------------------------------------------------------------------------
template<typename T>
__global__ void causal_mask_fill_kernel(T *scores,
                                        const int seq_len,
                                        const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(col>row)
    {
      scores[idx]=T(g_cu_attn_mask_fill);
    }
  }
}

//------------------------------------------------------------------------------
// Causal Mask Fill with Offset Kernel (for KV-cache)
// Sets positions where col > (row + offset) to -1e9 for rectangular matrices.
// Layout: [num_matrices, query_len, key_len]
// offset = previous cached sequence length
//------------------------------------------------------------------------------
template<typename T>
__global__ void causal_mask_fill_offset_kernel(T *scores,
                                               const int query_len,
                                               const int key_len,
                                               const int offset,
                                               const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=query_len*key_len;
    const int pos=idx%matrix_size;
    const int row=pos/key_len;
    const int col=pos%key_len;
    if(col>(offset+row))
    {
      scores[idx]=T(g_cu_attn_mask_fill);
    }
  }
}

//------------------------------------------------------------------------------
// Attention Softmax Forward Kernel
// Row-wise softmax with max subtraction for numerical stability.
// One block per row: multi-warp online reduction, normalize.
// Layout: [num_rows, row_len]
//------------------------------------------------------------------------------
// Multi-warp block per row. Intra-warp uses __shfl_xor_sync only. Cross-warp
// combine writes num_warps floats each for (m, s) into shared memory, then
// warp 0 reduces those num_warps values using __shfl_xor_sync again (still
// a pure warp-shuffle reduction — the shared memory is just a staging area
// for warp leaders, not the inter-warp reduction pattern that triggers the
// sm_120 hazard documented in caif_constants.h).
template<typename T>
__global__ void attention_softmax_kernel(const T *input,
                                         T *output,
                                         const int num_rows,
                                         const int row_len)
{
  const int row=blockIdx.x;
  if(row>=num_rows)
  {
    return;
  }

  const T *x=input+static_cast<int64_t>(row)*row_len;
  T *y=output+static_cast<int64_t>(row)*row_len;
  const unsigned mask=g_cu_warp_full_mask;
  const int lane=threadIdx.x%g_cu_warp_size;
  const int warp_id=threadIdx.x/g_cu_warp_size;
  const int num_warps=blockDim.x/g_cu_warp_size;

  extern __shared__ float smem[];
  // Layout: smem[0..num_warps)          = per-warp max
  //         smem[num_warps..2*num_warps) = per-warp sum
  //         smem[2*num_warps]            = final row_max
  //         smem[2*num_warps+1]          = final row_sum

  // Phase 1: per-thread online (max, sum) scan.
  // Reference: Milakov & Gimelshein, "Online normalizer calculation for
  // softmax" (2018). Invariant per thread: s = sum_{seen} exp(x_i - m).
  float m=g_cu_neg_sentinel;
  float s=0.0f;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    const float v=float(x[col]);
    if(v>m)
    {
      s=s*expf(m-v)+1.0f;
      m=v;
    }
    else
    {
      s+=expf(v-m);
    }
  }

  // Intra-warp combine.
  for(int off=g_cu_warp_half_size;off>0;off>>=1)
  {
    const float m_other=__shfl_xor_sync(mask,m,off);
    const float s_other=__shfl_xor_sync(mask,s,off);
    if(m_other>m)
    {
      s=s*expf(m-m_other)+s_other;
      m=m_other;
    }
    else
    {
      s+=s_other*expf(m_other-m);
    }
  }

  // Cross-warp combine.
  if(lane==0)
  {
    smem[warp_id]=m;
    smem[num_warps+warp_id]=s;
  }
  __syncthreads();

  if(warp_id==0)
  {
    float fm=g_cu_neg_sentinel;
    float fs=0.0f;
    if(lane<num_warps)
    {
      fm=smem[lane];
      fs=smem[num_warps+lane];
    }
    for(int off=num_warps>>1;off>0;off>>=1)
    {
      const float m_other=__shfl_xor_sync(mask,fm,off);
      const float s_other=__shfl_xor_sync(mask,fs,off);
      if(m_other>fm)
      {
        fs=fs*expf(fm-m_other)+s_other;
        fm=m_other;
      }
      else
      {
        fs+=s_other*expf(m_other-fm);
      }
    }
    if(lane==0)
    {
      smem[g_cu_softmax_stat_arrays*num_warps]=fm;
      smem[g_cu_softmax_stat_arrays*num_warps+1]=fs;
    }
  }
  __syncthreads();

  const float row_max=smem[g_cu_softmax_stat_arrays*num_warps];
  const float inv_sum=1.0f/smem[g_cu_softmax_stat_arrays*num_warps+1];

  // Phase 2: normalize (second and final pass over input).
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    y[col]=T(expf(float(x[col])-row_max)*inv_sum);
  }
}

//------------------------------------------------------------------------------
// Attention Softmax Backward Kernel
// For each row: grad_input[i] = output[i] * (grad_output[i] - dot(grad_output, output))
// One block per row, multi-warp, shuffle-only reductions.
//------------------------------------------------------------------------------
template<typename T>
__global__ void attention_softmax_backward_kernel(const T *grad_output,
                                                  const T *output,
                                                  T *grad_input,
                                                  const int num_rows,
                                                  const int row_len)
{
  const int row=blockIdx.x;
  if(row>=num_rows)
  {
    return;
  }

  const T *dy=grad_output+static_cast<int64_t>(row)*row_len;
  const T *y=output+static_cast<int64_t>(row)*row_len;
  T *dx=grad_input+static_cast<int64_t>(row)*row_len;
  const unsigned mask=g_cu_warp_full_mask;
  const int lane=threadIdx.x%g_cu_warp_size;
  const int warp_id=threadIdx.x/g_cu_warp_size;
  const int num_warps=blockDim.x/g_cu_warp_size;

  extern __shared__ float smem[];
  // Layout: smem[0..num_warps) = per-warp dot partial
  //         smem[num_warps]    = final row dot

  // Phase 1: dot(grad_output, output) — per-thread partial.
  float local_dot=0.0f;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    local_dot+=float(dy[col])*float(y[col]);
  }
  // Intra-warp reduce.
  for(int off=g_cu_warp_half_size;off>0;off>>=1)
  {
    local_dot+=__shfl_xor_sync(mask,local_dot,off);
  }
  // Cross-warp combine.
  if(lane==0)
  {
    smem[warp_id]=local_dot;
  }
  __syncthreads();

  if(warp_id==0)
  {
    float fd=0.0f;
    if(lane<num_warps)
    {
      fd=smem[lane];
    }
    for(int off=num_warps>>1;off>0;off>>=1)
    {
      fd+=__shfl_xor_sync(mask,fd,off);
    }
    if(lane==0)
    {
      smem[num_warps]=fd;
    }
  }
  __syncthreads();
  const float dot_val=smem[num_warps];

  // Phase 2: gradient.
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    const float y_val=float(y[col]);
    const float dy_val=float(dy[col]);
    dx[col]=T(y_val*(dy_val-dot_val));
  }
}

//------------------------------------------------------------------------------
// Causal Mask Gradient Kernel
// Zeros upper triangle (j > i) of gradient scores.
// Layout: [num_matrices, seq_len, seq_len]
//------------------------------------------------------------------------------
template<typename T>
__global__ void causal_mask_grad_kernel(T *grad_scores,
                                        const int seq_len,
                                        const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(col>row)
    {
      grad_scores[idx]=T(0.0f);
    }
  }
}

//------------------------------------------------------------------------------
// Prefix Mask Fill Kernel
// Allowed if (k <= q) OR (k < prefix_lengths[b]); disallowed positions get -1e9.
// Layout: [num_matrices, seq_len, seq_len] where num_matrices = batch*num_heads.
// prefix_lengths: device array of length batch (int32 per batch element).
//------------------------------------------------------------------------------
template<typename T>
__global__ void prefix_mask_fill_kernel(T *scores,
                                        const uint32_t *prefix_lengths,
                                        const int num_heads,
                                        const int seq_len,
                                        const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=seq_len*seq_len;
    const int matrix_idx=idx/matrix_size;
    const int pos=idx%matrix_size;
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    const int batch_idx=matrix_idx/num_heads;
    const int pfx=prefix_lengths[batch_idx];
    if(col>row && col>=pfx)
    {
      scores[idx]=T(g_cu_attn_mask_fill);
    }
  }
}

//------------------------------------------------------------------------------
// Prefix Mask Gradient Kernel
// Zeros disallowed positions (col > row AND col >= prefix_lengths[b]).
//------------------------------------------------------------------------------
template<typename T>
__global__ void prefix_mask_grad_kernel(T *grad_scores,
                                        const uint32_t *prefix_lengths,
                                        const int num_heads,
                                        const int seq_len,
                                        const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=seq_len*seq_len;
    const int matrix_idx=idx/matrix_size;
    const int pos=idx%matrix_size;
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    const int batch_idx=matrix_idx/num_heads;
    const int pfx=prefix_lengths[batch_idx];
    if(col>row && col>=pfx)
    {
      grad_scores[idx]=T(0.0f);
    }
  }
}

//------------------------------------------------------------------------------
// Multi-Head Attention Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_transpose_0213(const T *input,
                           T *output,
                           const int batch,
                           const int dim0,
                           const int dim1,
                           const int dim2,
                           cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch)*dim0*dim1*dim2;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  transpose_0213_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                               output,
                                                               dim0,
                                                               dim1,
                                                               dim2,
                                                               total);
}

template void launch_transpose_0213<float>(const float *,float *,int,int,int,int,cudaStream_t);
template void launch_transpose_0213<__half>(const __half *,__half *,int,int,int,int,cudaStream_t);
template void launch_transpose_0213<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int,
                                                   int,
                                                   int,
                                                   int,
                                                   cudaStream_t);

template<typename T>
void launch_transpose_0213_strided(const T *input,
                                   T *output,
                                   const int batch,
                                   const int dim0,
                                   const int dim1,
                                   const int dim2,
                                   const int input_d0_stride,
                                   cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch)*dim0*dim1*dim2;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  transpose_0213_strided_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                                       output,
                                                                       dim0,
                                                                       dim1,
                                                                       dim2,
                                                                       input_d0_stride,
                                                                       total);
}

template void launch_transpose_0213_strided<float>(const float *,
                                                   float *,
                                                   int,
                                                   int,
                                                   int,
                                                   int,
                                                   int,
                                                   cudaStream_t);
template void launch_transpose_0213_strided<__half>(const __half *,
                                                    __half *,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);
template void launch_transpose_0213_strided<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           cudaStream_t);

template<typename T>
void launch_causal_mask_fill(T *scores,
                             const int num_matrices,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_matrices)*seq_len*seq_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_fill_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,
                                                                 seq_len,
                                                                 total);
}

template void launch_causal_mask_fill<float>(float *,int,int,cudaStream_t);
template void launch_causal_mask_fill<__half>(__half *,int,int,cudaStream_t);
template void launch_causal_mask_fill<__nv_bfloat16>(__nv_bfloat16 *,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// Sliding-window mask (Mistral / Gemma-2): fill scores[matrix, q, k] where the
// key is more than `window` positions before the query (q - k >= window) with
// -inf. Applied after the causal mask in the explicit attention path; together
// they leave only the keys in (q-window, q]. Scores layout [num_matrices, s, s].
//------------------------------------------------------------------------------
template<typename T>
__global__ void sliding_window_mask_fill_kernel(T *scores,
                                                const int seq_len,
                                                const int window,
                                                const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(row-col>=window)
    {
      scores[idx]=T(g_cu_attn_mask_fill);
    }
  }
}

template<typename T>
void launch_sliding_window_mask(T *scores,
                                const int num_matrices,
                                const int seq_len,
                                const int window,
                                cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_matrices)*seq_len*seq_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  sliding_window_mask_fill_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,seq_len,window,total);
}

template void launch_sliding_window_mask<float>(float *,int,int,int,cudaStream_t);
template void launch_sliding_window_mask<__half>(__half *,int,int,int,cudaStream_t);
template void launch_sliding_window_mask<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// Sliding-window mask gradient: zero the score gradient where q - k >= window
// (the position was masked out of the softmax in the forward). Applied after
// the causal mask gradient in the explicit backward path.
//------------------------------------------------------------------------------
template<typename T>
__global__ void sliding_window_mask_grad_kernel(T *grad_scores,
                                                const int seq_len,
                                                const int window,
                                                const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(row-col>=window)
    {
      grad_scores[idx]=T(0.0f);
    }
  }
}

template<typename T>
void launch_sliding_window_mask_grad(T *grad_scores,
                                     const int num_matrices,
                                     const int seq_len,
                                     const int window,
                                     cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_matrices)*seq_len*seq_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  sliding_window_mask_grad_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_scores,seq_len,window,total);
}

template void launch_sliding_window_mask_grad<float>(float *,int,int,int,cudaStream_t);
template void launch_sliding_window_mask_grad<__half>(__half *,int,int,int,cudaStream_t);
template void launch_sliding_window_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// ALiBi linear position bias (MPT / BLOOM): add slopes[head]*(k-q) to each
// score in place. The per-head slope is read from a device array of length
// num_heads, indexed by matrix % num_heads. Applied after the score scale and
// soft-cap and before the mask. The additive constant has no backward
// derivative; the backward re-applies this same forward bias on the recomputed
// scores so its softmax matches the forward. Scores layout [num_matrices,s,s].
//------------------------------------------------------------------------------
template<typename T>
__global__ void alibi_bias_kernel(T *scores,
                                  const float *__restrict__ slopes,
                                  const int num_heads,
                                  const int seq_len,
                                  const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int64_t mat_size=static_cast<int64_t>(seq_len)*seq_len;
    const int matrix=static_cast<int>(idx/mat_size);
    const int pos=static_cast<int>(idx%mat_size);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    const float bias=slopes[matrix%num_heads]*static_cast<float>(col-row);
    scores[idx]=T(static_cast<float>(scores[idx])+bias);
  }
}

template<typename T>
void launch_alibi_bias(T *scores,
                       const float *slopes,
                       const int num_matrices,
                       const int num_heads,
                       const int seq_len,
                       cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_matrices)*seq_len*seq_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=static_cast<int>((total+block_size-1)/block_size);
  alibi_bias_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,slopes,num_heads,seq_len,total);
}

template void launch_alibi_bias<float>(float *,const float *,int,int,int,cudaStream_t);
template void launch_alibi_bias<__half>(__half *,const float *,int,int,int,cudaStream_t);
template void launch_alibi_bias<__nv_bfloat16>(__nv_bfloat16 *,const float *,int,int,int,cudaStream_t);

template<typename T>
void launch_causal_mask_fill_offset(T *scores,
                                    const int num_matrices,
                                    const int query_len,
                                    const int key_len,
                                    const int offset,
                                    cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_matrices)*query_len*key_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_fill_offset_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,
                                                                        query_len,
                                                                        key_len,
                                                                        offset,
                                                                        total);
}

template void launch_causal_mask_fill_offset<float>(float *,int,int,int,int,cudaStream_t);
template void launch_causal_mask_fill_offset<__half>(__half *,int,int,int,int,cudaStream_t);
template void launch_causal_mask_fill_offset<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,cudaStream_t);

template<typename T>
void launch_attention_softmax(const T *input,
                              T *output,
                              const int num_rows,
                              const int row_len,
                              cudaStream_t stream)
{
  // Multi-warp block per row. Intra-warp reductions use __shfl_xor_sync only;
  // cross-warp combine writes one float per warp to shared memory, reduces
  // those num_warps values in warp 0 via warp-shuffle, and broadcasts the
  // result back through shared memory. Shared-mem layout:
  //   [0..num_warps)       = per-warp max
  //   [num_warps..2*num_warps) = per-warp sum
  //   [2*num_warps]        = final row max
  //   [2*num_warps+1]      = final row sum
  const int block_size=SoftmaxBlockSize();
  const int num_warps=block_size/g_cu_warp_size;
  const size_t shared_mem_size=g_cu_softmax_stat_arrays*(num_warps+1)*sizeof(float);
  attention_softmax_kernel<T><<<num_rows,block_size,shared_mem_size,stream>>>(input,output,num_rows,row_len);
}
template void launch_attention_softmax<float>(const float *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_attention_softmax<__half>(const __half *,
                                               __half *,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_attention_softmax<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      __nv_bfloat16 *,
                                                      int,
                                                      int,
                                                      cudaStream_t);

template<typename T>
void launch_attention_softmax_backward(const T *grad_output,
                                       const T *output,
                                       T *grad_input,
                                       const int num_rows,
                                       const int row_len,
                                       cudaStream_t stream)
{
  // Multi-warp block per row. See launch_attention_softmax for the layout
  // rationale; backward needs only (num_warps + 1) staging floats (per-warp
  // dot partials + final).
  const int block_size=SoftmaxBlockSize();
  const int num_warps=block_size/g_cu_warp_size;
  const size_t shared_mem_size=(num_warps+1)*sizeof(float);
  attention_softmax_backward_kernel<T><<<num_rows,block_size,shared_mem_size,stream>>>(grad_output,
                                                                                       output,
                                                                                       grad_input,
                                                                                       num_rows,
                                                                                       row_len);
}
template void launch_attention_softmax_backward<float>(const float *,
                                                       const float *,
                                                       float *,
                                                       int,
                                                       int,
                                                       cudaStream_t);
template void launch_attention_softmax_backward<__half>(const __half *,
                                                        const __half *,
                                                        __half *,
                                                        int,
                                                        int,
                                                        cudaStream_t);
template void launch_attention_softmax_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                               const __nv_bfloat16 *,
                                                               __nv_bfloat16 *,
                                                               int,
                                                               int,
                                                               cudaStream_t);

template<typename T>
void launch_causal_mask_grad(T *grad_scores,
                             const int num_matrices,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(num_matrices)*seq_len*seq_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_grad_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_scores,
                                                                 seq_len,
                                                                 total);
}

template void launch_causal_mask_grad<float>(float *,int,int,cudaStream_t);
template void launch_causal_mask_grad<__half>(__half *,int,int,cudaStream_t);
template void launch_causal_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,int,int,cudaStream_t);

template<typename T>
void launch_prefix_mask_fill(T *scores,
                             const uint32_t *prefix_lengths,
                             const int batch_size,
                             const int num_heads,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch_size)*num_heads*seq_len*seq_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  prefix_mask_fill_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,
                                                                 prefix_lengths,
                                                                 num_heads,
                                                                 seq_len,
                                                                 total);
}

template void launch_prefix_mask_fill<float>(float *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_fill<__half>(__half *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_fill<__nv_bfloat16>(__nv_bfloat16 *,const uint32_t *,int,int,int,cudaStream_t);

template<typename T>
void launch_prefix_mask_grad(T *grad_scores,
                             const uint32_t *prefix_lengths,
                             const int batch_size,
                             const int num_heads,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch_size)*num_heads*seq_len*seq_len;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  prefix_mask_grad_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_scores,
                                                                 prefix_lengths,
                                                                 num_heads,
                                                                 seq_len,
                                                                 total);
}

template void launch_prefix_mask_grad<float>(float *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_grad<__half>(__half *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,const uint32_t *,int,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// RoPE Forward Kernel
// Applies rotary position embeddings in-place.
// data: [batch_heads, seq_len, head_dim], head_dim must be even.
// One thread per (batch_head, position, pair_index).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_forward_kernel(T *data,
                                    const int seq_len,
                                    const int head_dim,
                                    const float base,
                                    const int style,
                                    const int64_t total_pairs)
{
  // Precompute the per-pair inverse frequencies once per block in shared memory
  // (theta = pos * inv_freq[pair]) instead of a powf per element. All threads
  // fill + sync before the work guard so none returns early.
  __shared__ float inv_freq[g_cu_rope_max_half_dim];
  const int half_dim=head_dim/g_cu_rope_dims_per_pair;
  for(int p=threadIdx.x;p<half_dim;p+=blockDim.x)
  {
    const float freq_exp=static_cast<float>(g_cu_rope_dims_per_pair*p)/static_cast<float>(head_dim);
    inv_freq[p]=1.0f/powf(base,freq_exp);
  }
  __syncthreads();

  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=row%seq_len;

    const float theta=static_cast<float>(pos)*inv_freq[pair];
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int64_t row_base=static_cast<int64_t>(row)*head_dim;
    int64_t idx0;
    int64_t idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_dim;
    }
    else
    {
      idx0=row_base+pair*g_cu_rope_dims_per_pair;
      idx1=row_base+pair*g_cu_rope_dims_per_pair+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t-x1*sin_t);
    data[idx1]=T(x0*sin_t+x1*cos_t);
  }
}

//------------------------------------------------------------------------------
// RoPE Forward with Position Offset Kernel (for KV-cache)
// Same as rope_forward_kernel but adds pos_offset to position calculation.
// Used when processing new tokens that continue from cached positions.
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_forward_offset_kernel(T *data,
                                           const int seq_len,
                                           const int head_dim,
                                           const float base,
                                           const int pos_offset,
                                           const int style,
                                           const int64_t total_pairs)
{
  // Per-pair inverse frequencies precomputed once per block (see
  // rope_forward_kernel); avoids a powf per element.
  __shared__ float inv_freq[g_cu_rope_max_half_dim];
  const int half_dim=head_dim/g_cu_rope_dims_per_pair;
  for(int p=threadIdx.x;p<half_dim;p+=blockDim.x)
  {
    const float freq_exp=static_cast<float>(g_cu_rope_dims_per_pair*p)/static_cast<float>(head_dim);
    inv_freq[p]=1.0f/powf(base,freq_exp);
  }
  __syncthreads();

  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=(row%seq_len)+pos_offset;

    const float theta=static_cast<float>(pos)*inv_freq[pair];
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int64_t row_base=static_cast<int64_t>(row)*head_dim;
    int64_t idx0;
    int64_t idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_dim;
    }
    else
    {
      idx0=row_base+pair*g_cu_rope_dims_per_pair;
      idx1=row_base+pair*g_cu_rope_dims_per_pair+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t-x1*sin_t);
    data[idx1]=T(x0*sin_t+x1*cos_t);
  }
}

//------------------------------------------------------------------------------
// RoPE Backward Kernel
// Applies inverse rotation in-place (swap sin sign).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_backward_kernel(T *data,
                                     const int seq_len,
                                     const int head_dim,
                                     const float base,
                                     const int style,
                                     const int64_t total_pairs)
{
  // Per-pair inverse frequencies precomputed once per block (see
  // rope_forward_kernel); avoids a powf per element.
  __shared__ float inv_freq[g_cu_rope_max_half_dim];
  const int half_dim=head_dim/g_cu_rope_dims_per_pair;
  for(int p=threadIdx.x;p<half_dim;p+=blockDim.x)
  {
    const float freq_exp=static_cast<float>(g_cu_rope_dims_per_pair*p)/static_cast<float>(head_dim);
    inv_freq[p]=1.0f/powf(base,freq_exp);
  }
  __syncthreads();

  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=row%seq_len;

    const float theta=static_cast<float>(pos)*inv_freq[pair];
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int64_t row_base=static_cast<int64_t>(row)*head_dim;
    int64_t idx0;
    int64_t idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_dim;
    }
    else
    {
      idx0=row_base+pair*g_cu_rope_dims_per_pair;
      idx1=row_base+pair*g_cu_rope_dims_per_pair+1;
    }

    const float g0=float(data[idx0]);
    const float g1=float(data[idx1]);
    data[idx0]=T(g0*cos_t+g1*sin_t);
    data[idx1]=T(-g0*sin_t+g1*cos_t);
  }
}

//------------------------------------------------------------------------------
// GQA Repeat KV Kernel
// input: [batch * num_kv_heads, seq_len, head_dim]
// output: [batch * num_heads, seq_len, head_dim]
// One thread per output element.
//------------------------------------------------------------------------------
template<typename T>
__global__ void gqa_repeat_kv_kernel(const T *input,
                                     T *output,
                                     const int num_kv_heads,
                                     const int repeat_factor,
                                     const int seq_len,
                                     const int head_dim,
                                     const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
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
    const int64_t in_idx=static_cast<int64_t>(b)*input_batch_stride+kv_h*head_stride+rem2;

    output[idx]=input[in_idx];
  }
}

//------------------------------------------------------------------------------
// GQA Reduce KV Kernel
// input: [batch * num_heads, seq_len, head_dim]
// output: [batch * num_kv_heads, seq_len, head_dim]
// One thread per output element, loops over repeat_factor.
//------------------------------------------------------------------------------
template<typename T>
__global__ void gqa_reduce_kv_kernel(const T *input,
                                     T *output,
                                     const int num_kv_heads,
                                     const int repeat_factor,
                                     const int seq_len,
                                     const int head_dim,
                                     const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
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
      const int64_t in_idx=static_cast<int64_t>(b)*in_batch_stride+h*head_stride+rem2;
      sum+=float(input[in_idx]);
    }
    output[idx]=T(sum);
  }
}

//------------------------------------------------------------------------------
// RoPE / GQA Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_rope_forward(T *data,
                         const int batch_heads,
                         const int seq_len,
                         const int head_dim,
                         const float base,
                         const int style,
                         cudaStream_t stream)
{
  const int64_t total_pairs=static_cast<int64_t>(batch_heads)*seq_len*(head_dim/g_cu_rope_dims_per_pair);
  const int block_size=g_cu_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_kernel<T><<<num_blocks,block_size,0,stream>>>(data,seq_len,head_dim,base,style,total_pairs);
}

template void launch_rope_forward<float>(float *,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward<__half>(__half *,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,float,int,cudaStream_t);

//------------------------------------------------------------------------------
// Partial-rotary RoPE Forward Kernel
// Rotates only the first `rope_dim` dims of each head row. Stride within
// a head is still `head_dim` (the unrotated tail at indices [rope_dim,
// head_dim) is left untouched and stays in place).  Frequencies use
// rope_dim as the divisor (matches HF rotate_half(x[..., :rope_dim])).
// One thread per (batch_head, position, pair_index in [0, rope_dim/2)).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_forward_partial_kernel(T *data,
                                            const int seq_len,
                                            const int head_dim,
                                            const int rope_dim,
                                            const float base,
                                            const int style,
                                            const int64_t total_pairs)
{
  // Per-pair inverse frequencies precomputed once per block (see
  // rope_forward_kernel); avoids a powf per element. Frequencies use rope_dim.
  __shared__ float inv_freq[g_cu_rope_max_half_dim];
  const int half_rope=rope_dim/g_cu_rope_dims_per_pair;
  for(int p=threadIdx.x;p<half_rope;p+=blockDim.x)
  {
    const float freq_exp=static_cast<float>(g_cu_rope_dims_per_pair*p)/static_cast<float>(rope_dim);
    inv_freq[p]=1.0f/powf(base,freq_exp);
  }
  __syncthreads();

  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int row=idx/half_rope;
    const int pair=idx%half_rope;

    const int pos=row%seq_len;

    const float theta=static_cast<float>(pos)*inv_freq[pair];
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int64_t row_base=static_cast<int64_t>(row)*head_dim;
    int64_t idx0;
    int64_t idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_rope;
    }
    else
    {
      idx0=row_base+pair*g_cu_rope_dims_per_pair;
      idx1=row_base+pair*g_cu_rope_dims_per_pair+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t-x1*sin_t);
    data[idx1]=T(x0*sin_t+x1*cos_t);
  }
}

template<typename T>
void launch_rope_forward_partial(T *data,
                                 const int batch_heads,
                                 const int seq_len,
                                 const int head_dim,
                                 const int rope_dim,
                                 const float base,
                                 const int style,
                                 cudaStream_t stream)
{
  const int64_t total_pairs=static_cast<int64_t>(batch_heads)*seq_len*(rope_dim/g_cu_rope_dims_per_pair);
  const int block_size=g_cu_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_partial_kernel<T><<<num_blocks,block_size,0,stream>>>(data,
                                                                     seq_len,
                                                                     head_dim,
                                                                     rope_dim,
                                                                     base,
                                                                     style,
                                                                     total_pairs);
}

template void launch_rope_forward_partial<float>(float *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward_partial<__half>(__half *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward_partial<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,float,int,cudaStream_t);

template<typename T>
void launch_rope_forward_offset(T *data,
                                const int batch_heads,
                                const int seq_len,
                                const int head_dim,
                                const float base,
                                const int pos_offset,
                                const int style,
                                cudaStream_t stream)
{
  const int64_t total_pairs=static_cast<int64_t>(batch_heads)*seq_len*(head_dim/g_cu_rope_dims_per_pair);
  const int block_size=g_cu_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_offset_kernel<T><<<num_blocks,block_size,0,stream>>>(data,
                                                                    seq_len,
                                                                    head_dim,
                                                                    base,
                                                                    pos_offset,
                                                                    style,
                                                                    total_pairs);
}

template void launch_rope_forward_offset<float>(float *,int,int,int,float,int,int,cudaStream_t);
template void launch_rope_forward_offset<__half>(__half *,int,int,int,float,int,int,cudaStream_t);
template void launch_rope_forward_offset<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,float,int,int,cudaStream_t);

template<typename T>
void launch_rope_backward(T *data,
                          const int batch_heads,
                          const int seq_len,
                          const int head_dim,
                          const float base,
                          const int style,
                          cudaStream_t stream)
{
  const int64_t total_pairs=static_cast<int64_t>(batch_heads)*seq_len*(head_dim/g_cu_rope_dims_per_pair);
  const int block_size=g_cu_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(data,seq_len,head_dim,base,style,total_pairs);
}

template void launch_rope_backward<float>(float *,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward<__half>(__half *,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,float,int,cudaStream_t);

//------------------------------------------------------------------------------
// Partial-rotary RoPE Backward Kernel — inverse rotation on first
// `rope_dim` dims of each head row, untouched tail.
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_backward_partial_kernel(T *data,
                                             const int seq_len,
                                             const int head_dim,
                                             const int rope_dim,
                                             const float base,
                                             const int style,
                                             const int64_t total_pairs)
{
  // Per-pair inverse frequencies precomputed once per block (see
  // rope_forward_kernel); avoids a powf per element. Frequencies use rope_dim.
  __shared__ float inv_freq[g_cu_rope_max_half_dim];
  const int half_rope=rope_dim/g_cu_rope_dims_per_pair;
  for(int p=threadIdx.x;p<half_rope;p+=blockDim.x)
  {
    const float freq_exp=static_cast<float>(g_cu_rope_dims_per_pair*p)/static_cast<float>(rope_dim);
    inv_freq[p]=1.0f/powf(base,freq_exp);
  }
  __syncthreads();

  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int row=idx/half_rope;
    const int pair=idx%half_rope;

    const int pos=row%seq_len;

    const float theta=static_cast<float>(pos)*inv_freq[pair];
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int64_t row_base=static_cast<int64_t>(row)*head_dim;
    int64_t idx0;
    int64_t idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_rope;
    }
    else
    {
      idx0=row_base+pair*g_cu_rope_dims_per_pair;
      idx1=row_base+pair*g_cu_rope_dims_per_pair+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t+x1*sin_t);
    data[idx1]=T(-x0*sin_t+x1*cos_t);
  }
}

template<typename T>
void launch_rope_backward_partial(T *data,
                                  const int batch_heads,
                                  const int seq_len,
                                  const int head_dim,
                                  const int rope_dim,
                                  const float base,
                                  const int style,
                                  cudaStream_t stream)
{
  const int64_t total_pairs=static_cast<int64_t>(batch_heads)*seq_len*(rope_dim/g_cu_rope_dims_per_pair);
  const int block_size=g_cu_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_backward_partial_kernel<T><<<num_blocks,block_size,0,stream>>>(data,
                                                                      seq_len,
                                                                      head_dim,
                                                                      rope_dim,
                                                                      base,
                                                                      style,
                                                                      total_pairs);
}

template void launch_rope_backward_partial<float>(float *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward_partial<__half>(__half *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward_partial<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,float,int,cudaStream_t);

template<typename T>
void launch_gqa_repeat_kv(const T *input,
                          T *output,
                          const int batch,
                          const int num_kv_heads,
                          const int repeat_factor,
                          const int seq_len,
                          const int head_dim,
                          cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch)*num_kv_heads*repeat_factor*seq_len*head_dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  gqa_repeat_kv_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                              output,
                                                              num_kv_heads,
                                                              repeat_factor,
                                                              seq_len,
                                                              head_dim,
                                                              total);
}

template void launch_gqa_repeat_kv<float>(const float *,float *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_repeat_kv<__half>(const __half *,__half *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_repeat_kv<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  cudaStream_t);

template<typename T>
void launch_gqa_reduce_kv(const T *input,
                          T *output,
                          const int batch,
                          const int num_kv_heads,
                          const int repeat_factor,
                          const int seq_len,
                          const int head_dim,
                          cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch)*num_kv_heads*seq_len*head_dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  gqa_reduce_kv_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                              output,
                                                              num_kv_heads,
                                                              repeat_factor,
                                                              seq_len,
                                                              head_dim,
                                                              total);
}

template void launch_gqa_reduce_kv<float>(const float *,float *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_reduce_kv<__half>(const __half *,__half *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_reduce_kv<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  cudaStream_t);

//------------------------------------------------------------------------------
// KV-Cache kernels
//------------------------------------------------------------------------------

/**
 * Kernel to append new K/V data to the cache.
 * new_kv: [batch, new_len, num_kv_heads, head_dim] - contiguous
 * cache: [batch, max_seq_len, num_kv_heads, head_dim] - contiguous
 * Copies new_kv into cache starting at position cache_pos.
 */
template<typename T>
__global__ void kv_cache_append_kernel(const T *new_kv,
                                       T *cache,
                                       const int new_len,
                                       const int cache_pos,
                                       const int max_seq_len,
                                       const int num_kv_heads,
                                       const int head_dim,
                                       const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }

  const int kv_size=num_kv_heads*head_dim;
  const int batch_stride=new_len*kv_size;
  const int b=idx/batch_stride;
  const int rem=idx%batch_stride;
  const int new_pos=rem/kv_size;
  const int kv_idx=rem%kv_size;

  const int cache_batch_stride=max_seq_len*kv_size;
  const int64_t cache_dst=static_cast<int64_t>(b)*cache_batch_stride+(cache_pos+new_pos)*kv_size+kv_idx;

  cache[cache_dst]=new_kv[idx];
}

template<typename T>
void launch_kv_cache_append(const T *new_kv,
                            T *cache,
                            const int batch,
                            const int new_len,
                            const int cache_pos,
                            const int max_seq_len,
                            const int num_kv_heads,
                            const int head_dim,
                            cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch)*new_len*num_kv_heads*head_dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  kv_cache_append_kernel<T><<<num_blocks,block_size,0,stream>>>(new_kv,
                                                                cache,
                                                                new_len,
                                                                cache_pos,
                                                                max_seq_len,
                                                                num_kv_heads,
                                                                head_dim,
                                                                total);
}

template void launch_kv_cache_append<float>(const float *,float *,int,int,int,int,int,int,cudaStream_t);
template void launch_kv_cache_append<__half>(const __half *,__half *,int,int,int,int,int,int,cudaStream_t);
template void launch_kv_cache_append<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);

//------------------------------------------------------------------------------
// KV-Cache Append Kernel (transposed layout)
// new_kv: [batch_kv_heads, new_len, head_dim]
// cache:  [batch_kv_heads, max_seq_len, head_dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void kv_cache_append_transposed_kernel(const T *new_kv,
                                                  T *cache,
                                                  const int new_len,
                                                  const int cache_pos,
                                                  const int max_seq_len,
                                                  const int head_dim,
                                                  const int64_t total)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }

  const int row_size=new_len*head_dim;
  const int bkv=idx/row_size;
  const int rem=idx%row_size;
  const int new_pos=rem/head_dim;
  const int d=rem%head_dim;

  const int cache_row_size=max_seq_len*head_dim;
  const int64_t cache_dst=static_cast<int64_t>(bkv)*cache_row_size+(cache_pos+new_pos)*head_dim+d;

  cache[cache_dst]=new_kv[idx];
}

template<typename T>
void launch_kv_cache_append_transposed(const T *new_kv,
                                       T *cache,
                                       const int batch_kv_heads,
                                       const int new_len,
                                       const int cache_pos,
                                       const int max_seq_len,
                                       const int head_dim,
                                       cudaStream_t stream)
{
  const int64_t total=static_cast<int64_t>(batch_kv_heads)*new_len*head_dim;
  const int block_size=g_cu_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  kv_cache_append_transposed_kernel<T><<<num_blocks,block_size,0,stream>>>(new_kv,
                                                                           cache,
                                                                           new_len,
                                                                           cache_pos,
                                                                           max_seq_len,
                                                                           head_dim,
                                                                           total);
}

template void launch_kv_cache_append_transposed<float>(const float *,float *,int,int,int,int,int,cudaStream_t);
template void launch_kv_cache_append_transposed<__half>(const __half *,__half *,int,int,int,int,int,cudaStream_t);
template void launch_kv_cache_append_transposed<__nv_bfloat16>(const __nv_bfloat16 *,
                                                               __nv_bfloat16 *,
                                                               int,
                                                               int,
                                                               int,
                                                               int,
                                                               int,
                                                               cudaStream_t);

