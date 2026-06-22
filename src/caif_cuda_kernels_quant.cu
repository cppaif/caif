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
// Quantization / dtype conversion CUDA kernels: fp32<->fp16/bf16,
// int8 (plain + scaled per-tensor/per-channel), int4 per-group
// quantize/dequantize.
// Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_quant.cuh
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
// Data type conversion kernels (FP32 <-> FP16, FP32 <-> BF16)
//------------------------------------------------------------------------------

__global__ void convert_fp32_to_fp16_kernel(const float *input,
                                            __half *output,
                                            const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__float2half(input[idx]);
  }
}

void launch_convert_fp32_to_fp16(const float *input,
                                 void *output,
                                 int64_t n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_cu_block_size-1)/g_cu_block_size;
  convert_fp32_to_fp16_kernel<<<grid,g_cu_block_size,0,stream>>>(input,static_cast<__half*>(output),n);
}

__global__ void convert_fp16_to_fp32_kernel(const __half *input,
                                            float *output,
                                            const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__half2float(input[idx]);
  }
}

void launch_convert_fp16_to_fp32(const void *input,
                                 float *output,
                                 int64_t n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_cu_block_size-1)/g_cu_block_size;
  convert_fp16_to_fp32_kernel<<<grid,g_cu_block_size,0,stream>>>(static_cast<const __half*>(input),output,n);
}

__global__ void convert_fp32_to_bf16_kernel(const float *input,
                                            __nv_bfloat16 *output,
                                            const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__float2bfloat16(input[idx]);
  }
}

void launch_convert_fp32_to_bf16(const float *input,
                                 void *output,
                                 int64_t n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_cu_block_size-1)/g_cu_block_size;
  convert_fp32_to_bf16_kernel<<<grid,g_cu_block_size,0,stream>>>(input,static_cast<__nv_bfloat16*>(output),n);
}

__global__ void convert_bf16_to_fp32_kernel(const __nv_bfloat16 *input,
                                            float *output,
                                            const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__bfloat162float(input[idx]);
  }
}

void launch_convert_bf16_to_fp32(const void *input,
                                 float *output,
                                 int64_t n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_cu_block_size-1)/g_cu_block_size;
  convert_bf16_to_fp32_kernel<<<grid,g_cu_block_size,0,stream>>>(static_cast<const __nv_bfloat16*>(input),
                                                                 output,
                                                                 n);
}

//------------------------------------------------------------------------------
// INT8 conversion kernels
//------------------------------------------------------------------------------

__global__ void convert_fp32_to_int8_kernel(const float *input,
                                            int8_t *output,
                                            const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float val=input[idx];
    if(val>g_cu_quant_int8_max)
    {
      val=g_cu_quant_int8_max;
    }
    else if(val<-g_cu_quant_int8_max)
    {
      val=-g_cu_quant_int8_max;
    }
    output[idx]=static_cast<int8_t>(rintf(val));
  }
}

void launch_convert_fp32_to_int8(const float *input,
                                 void *output,
                                 int64_t n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_cu_block_size-1)/g_cu_block_size;
  convert_fp32_to_int8_kernel<<<grid,g_cu_block_size,0,stream>>>(input,static_cast<int8_t*>(output),n);
}

__global__ void convert_int8_to_fp32_kernel(const int8_t *input,
                                            float *output,
                                            const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=static_cast<float>(input[idx]);
  }
}

void launch_convert_int8_to_fp32(const void *input,
                                 float *output,
                                 int64_t n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_cu_block_size-1)/g_cu_block_size;
  convert_int8_to_fp32_kernel<<<grid,g_cu_block_size,0,stream>>>(static_cast<const int8_t*>(input),output,n);
}

//------------------------------------------------------------------------------
// INT4 quantization kernels (symmetric per-group with FP16 scales)
//------------------------------------------------------------------------------

__global__ void dequantize_int4_kernel(const uint8_t *packed_data,
                                       const __half *scales,
                                       float *output,
                                       const int64_t num_elements,
                                       const int group_size)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<num_elements)
  {
    const int64_t byte_idx=idx/g_cu_quant_int4_per_byte;
    const uint8_t packed=packed_data[byte_idx];

    int int4_val;
    if((idx&1)==0)
    {
      int4_val=packed&g_cu_quant_int4_nibble_mask;
    }
    else
    {
      int4_val=(packed>>g_cu_quant_int4_nibble_bits)&g_cu_quant_int4_nibble_mask;
    }

    // Sign extend: if bit 3 is set, value is negative
    if((int4_val&g_cu_quant_int4_sign_bit)!=0)
    {
      int4_val|=g_cu_quant_int4_sign_extend;
    }

    const int64_t group_idx=idx/group_size;
    const float scale=__half2float(scales[group_idx]);
    output[idx]=static_cast<float>(int4_val)*scale;
  }
}

void launch_dequantize_int4(const void *packed_data,
                            const void *scales,
                            float *output,
                            int64_t num_elements,
                            int group_size,
                            cudaStream_t stream)
{
  if(num_elements<=0)
  {
    return;
  }
  const int grid=(num_elements+g_cu_block_size-1)/g_cu_block_size;
  dequantize_int4_kernel<<<grid,g_cu_block_size,0,stream>>>(static_cast<const uint8_t*>(packed_data),
                                                            static_cast<const __half*>(scales),
                                                            output,
                                                            num_elements,
                                                            group_size);
}

__global__ void quantize_to_int4_kernel(const float *input,
                                        uint8_t *packed_output,
                                        __half *scales_output,
                                        const int64_t num_elements,
                                        const int group_size)
{
  const int64_t group_idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t num_groups=(num_elements+group_size-1)/group_size;
  if(group_idx>=num_groups)
  {
    return;
  }

  const int64_t group_start=group_idx*group_size;
  int64_t group_end=group_start+group_size;
  if(group_end>num_elements)
  {
    group_end=num_elements;
  }

  // Find max absolute value in group
  float max_abs=0.0f;
  for(int64_t i=group_start;i<group_end;++i)
  {
    float abs_val=fabsf(input[i]);
    if(abs_val>max_abs)
    {
      max_abs=abs_val;
    }
  }

  // Compute scale: scale = max_abs / g_cu_quant_int4_max
  float scale;
  if(max_abs==0.0f)
  {
    scale=0.0f;
  }
  else
  {
    scale=max_abs/g_cu_quant_int4_max;
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

  for(int64_t i=group_start;i<group_end;i+=g_cu_quant_int4_per_byte)
  {
    float v0=input[i]*inv_scale;
    if(v0>g_cu_quant_int4_max)
    {
      v0=g_cu_quant_int4_max;
    }
    else if(v0<-g_cu_quant_int4_max)
    {
      v0=-g_cu_quant_int4_max;
    }
    int q0=static_cast<int>(rintf(v0))&g_cu_quant_int4_nibble_mask;

    int q1=0;
    if(i+1<group_end)
    {
      float v1=input[i+1]*inv_scale;
      if(v1>g_cu_quant_int4_max)
      {
        v1=g_cu_quant_int4_max;
      }
      else if(v1<-g_cu_quant_int4_max)
      {
        v1=-g_cu_quant_int4_max;
      }
      q1=static_cast<int>(rintf(v1))&g_cu_quant_int4_nibble_mask;
    }

    packed_output[i/g_cu_quant_int4_per_byte]=static_cast<uint8_t>(q0|(q1<<g_cu_quant_int4_nibble_bits));
  }
}

void launch_quantize_to_int4(const float *input,
                             void *packed_output,
                             void *scales_output,
                             int64_t num_elements,
                             int group_size,
                             cudaStream_t stream)
{
  if(num_elements<=0)
  {
    return;
  }
  const int64_t num_groups=(num_elements+group_size-1)/group_size;
  const int grid=(num_groups+g_cu_block_size-1)/g_cu_block_size;
  quantize_to_int4_kernel<<<grid,g_cu_block_size,0,stream>>>(input,
                                                             static_cast<uint8_t*>(packed_output),
                                                             static_cast<__half*>(scales_output),
                                                             num_elements,
                                                             group_size);
}

//------------------------------------------------------------------------------
// INT8 scaled quantization kernels (symmetric, per-tensor and per-channel)
//
// Per-tensor scheme:
//   scale = max(abs(x)) / 127.0f, stored as a single float
//   q[i]  = round(x[i] / scale), clamped to [-127, 127]
//   x'[i] = q[i] * scale
//
// Per-channel scheme (on last dim, interpreted as the output-channel axis):
//   scale[c] = max over rows of abs(x[:, c]) / 127.0f
//   q[r, c]  = round(x[r, c] / scale[c])
//   x'[r, c] = q[r, c] * scale[c]
//
// Weight tensors stored as [in_features, out_features] get per-channel on the
// out axis; activation tensors typically use per-tensor.
//------------------------------------------------------------------------------

// Accumulator buffer layout: scale_out[0] holds max(|x|) as a non-negative
// float. We atomicMax on its int-reinterpretation — valid because IEEE 754
// positive-float bit patterns preserve ordering when compared as int.
__global__ void compute_int8_per_tensor_scale_kernel(const float *input,
                                                     float *scale_out,
                                                     const int64_t n)
{
  float local_max=0.0f;
  for(int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<int64_t>(gridDim.x)*blockDim.x)
  {
    const float v=fabsf(input[i]);
    if(v>local_max)
    {
      local_max=v;
    }
  }
  if(local_max>0.0f)
  {
    atomicMax(reinterpret_cast<int*>(scale_out),__float_as_int(local_max));
  }
}

__global__ void finalize_int8_per_tensor_scale_kernel(float *scale_out)
{
  if(threadIdx.x==0&&blockIdx.x==0)
  {
    const float max_abs=scale_out[0];
    if(max_abs>0.0f)
    {
      scale_out[0]=max_abs/g_cu_quant_int8_max;
    }
    else
    {
      scale_out[0]=1.0f;
    }
  }
}

__global__ void quantize_int8_per_tensor_kernel(const float *input,
                                                int8_t *output,
                                                const float *scale,
                                                const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float s=scale[0];
    float inv_s=0.0f;
    if(s>0.0f)
    {
      inv_s=1.0f/s;
    }
    float v=input[idx]*inv_s;
    if(v>g_cu_quant_int8_max)
    {
      v=g_cu_quant_int8_max;
    }
    else if(v<-g_cu_quant_int8_max)
    {
      v=-g_cu_quant_int8_max;
    }
    output[idx]=static_cast<int8_t>(rintf(v));
  }
}

__global__ void dequantize_int8_per_tensor_kernel(const int8_t *input,
                                                  float *output,
                                                  const float *scale,
                                                  const int64_t n)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=static_cast<float>(input[idx])*scale[0];
  }
}

void launch_quantize_int8_per_tensor(const float *input,
                                     void *output,
                                     void *scale,
                                     int64_t n,
                                     cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  // Zero the scale buffer (used as an atomicMax accumulator).
  cudaMemsetAsync(scale,0,sizeof(float),stream);
  const int block=g_cu_block_size;
  const int grid=(n+block-1)/block;
  int cap_grid=grid;
  if(cap_grid>g_cu_quant_reduce_max_grid)
  {
    cap_grid=g_cu_quant_reduce_max_grid;
  }
  compute_int8_per_tensor_scale_kernel<<<cap_grid,block,0,stream>>>(input,static_cast<float*>(scale),n);
  finalize_int8_per_tensor_scale_kernel<<<1,1,0,stream>>>(static_cast<float*>(scale));
  quantize_int8_per_tensor_kernel<<<grid,block,0,stream>>>(input,
                                                           static_cast<int8_t*>(output),
                                                           static_cast<const float*>(scale),
                                                           n);
}

void launch_dequantize_int8_per_tensor(const void *input,
                                       float *output,
                                       const void *scale,
                                       int64_t n,
                                       cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_cu_block_size-1)/g_cu_block_size;
  dequantize_int8_per_tensor_kernel<<<grid,g_cu_block_size,0,stream>>>(static_cast<const int8_t*>(input),
                                                                       output,
                                                                       static_cast<const float*>(scale),
                                                                       n);
}

__global__ void compute_int8_per_channel_scale_kernel(const float *input,
                                                      float *scales,
                                                      const int rows,
                                                      const int cols)
{
  const int col=blockIdx.x*blockDim.x+threadIdx.x;
  if(col>=cols)
  {
    return;
  }
  float max_abs=0.0f;
  for(int r=0;r<rows;++r)
  {
    const float v=fabsf(input[static_cast<int64_t>(r)*cols+col]);
    if(v>max_abs)
    {
      max_abs=v;
    }
  }
  if(max_abs>0.0f)
  {
    scales[col]=max_abs/g_cu_quant_int8_max;
  }
  else
  {
    scales[col]=1.0f;
  }
}

__global__ void quantize_int8_per_channel_kernel(const float *input,
                                                 int8_t *output,
                                                 const float *scales,
                                                 const int rows,
                                                 const int cols)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t total=static_cast<int64_t>(rows)*cols;
  if(idx<total)
  {
    const int col=idx%cols;
    const float s=scales[col];
    float inv_s=0.0f;
    if(s>0.0f)
    {
      inv_s=1.0f/s;
    }
    float v=input[idx]*inv_s;
    if(v>g_cu_quant_int8_max)
    {
      v=g_cu_quant_int8_max;
    }
    else if(v<-g_cu_quant_int8_max)
    {
      v=-g_cu_quant_int8_max;
    }
    output[idx]=static_cast<int8_t>(rintf(v));
  }
}

__global__ void dequantize_int8_per_channel_kernel(const int8_t *input,
                                                   float *output,
                                                   const float *scales,
                                                   const int rows,
                                                   const int cols)
{
  const int64_t idx=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t total=static_cast<int64_t>(rows)*cols;
  if(idx<total)
  {
    const int col=idx%cols;
    output[idx]=static_cast<float>(input[idx])*scales[col];
  }
}

void launch_quantize_int8_per_channel(const float *input,
                                      void *output,
                                      void *scales,
                                      int rows,
                                      int cols,
                                      cudaStream_t stream)
{
  if(rows<=0||cols<=0)
  {
    return;
  }
  const int block=g_cu_block_size;
  const int scale_grid=(cols+block-1)/block;
  compute_int8_per_channel_scale_kernel<<<scale_grid,block,0,stream>>>(input,
                                                                       static_cast<float*>(scales),
                                                                       rows,
                                                                       cols);
  const int64_t total=static_cast<int64_t>(rows)*cols;
  const int grid=(total+block-1)/block;
  quantize_int8_per_channel_kernel<<<grid,block,0,stream>>>(input,
                                                            static_cast<int8_t*>(output),
                                                            static_cast<const float*>(scales),
                                                            rows,
                                                            cols);
}

void launch_dequantize_int8_per_channel(const void *input,
                                        float *output,
                                        const void *scales,
                                        int rows,
                                        int cols,
                                        cudaStream_t stream)
{
  if(rows<=0||cols<=0)
  {
    return;
  }
  const int64_t total=static_cast<int64_t>(rows)*cols;
  const int grid=(total+g_cu_block_size-1)/g_cu_block_size;
  dequantize_int8_per_channel_kernel<<<grid,g_cu_block_size,0,stream>>>(static_cast<const int8_t*>(input),
                                                                        output,
                                                                        static_cast<const float*>(scales),
                                                                        rows,
                                                                        cols);
}

