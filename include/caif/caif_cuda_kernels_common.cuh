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
// Shared device-inline helpers and constants for the caif_cuda_kernels_* CUDA
// translation units. Device-internal — lives in src/, not part of the public
// API. Every helper here stays __forceinline__/inline (never static) so each
// including TU gets its own definition with no multiple-definition errors;
// template specializations stay in this header with their primary template.
//------------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "caif_cuda_kernels_constants.cuh"

//------------------------------------------------------------------------------
// 128-bit vectorized access via int4 blob. int4 = 16 bytes = 4 fp32 / 8 fp16
// / 8 bf16. Templated kernels load as int4, reinterpret as T[lanes], op, store.
//------------------------------------------------------------------------------
template<typename T>
__host__ __device__ constexpr int caif_vec_lanes(){return sizeof(int4)/sizeof(T);}

//------------------------------------------------------------------------------
// caif_load_f / caif_store_f: dtype-agnostic fp32 load/store helpers.
// Used by kernels that compute in fp32 for numerical stability while
// reading/writing storage tensors of any dtype (float/__half/__nv_bfloat16).
//------------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ float caif_load_f(const T &v)
{
  return static_cast<float>(v);
}
template<>
__device__ __forceinline__ float caif_load_f<__half>(const __half &v)
{
  return __half2float(v);
}
template<>
__device__ __forceinline__ float caif_load_f<__nv_bfloat16>(const __nv_bfloat16 &v)
{
  return __bfloat162float(v);
}

template<typename T>
__device__ __forceinline__ T caif_store_f(float v)
{
  return static_cast<T>(v);
}
template<>
__device__ __forceinline__ __half caif_store_f<__half>(float v)
{
  return __float2half(v);
}
template<>
__device__ __forceinline__ __nv_bfloat16 caif_store_f<__nv_bfloat16>(float v)
{
  return __float2bfloat16(v);
}

//------------------------------------------------------------------------------
// caif_atomic_add: dispatch atomicAdd on storage T, with a CAS-loop fallback
// for bf16 below sm_90 (CUDA only added native bf16 atomicAdd on sm_90).
//------------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ void caif_atomic_add(T *addr,T val)
{
  atomicAdd(addr,val);
}

template<>
__device__ __forceinline__ void caif_atomic_add<__nv_bfloat16>(__nv_bfloat16 *addr,
                                                               __nv_bfloat16 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  atomicAdd(addr,val);
#else
  // Word-aligned CAS emulation: pack/unpack the bf16 in the high or low half
  // of the containing 32-bit word.
  uintptr_t addr_int=reinterpret_cast<uintptr_t>(addr);
  unsigned int *base=reinterpret_cast<unsigned int *>(addr_int & ~uintptr_t(g_cu_word_align_mask));
  unsigned int shift=0u;
  if((addr_int & uintptr_t(g_cu_word_align_mask))!=0)
  {
    shift=g_cu_half_word_bits;
  }
  const float val_f=__bfloat162float(val);
  unsigned int old=*base;
  unsigned int assumed;
  do
  {
    assumed=old;
    const unsigned short cur_bits=static_cast<unsigned short>((assumed>>shift)&g_cu_half_word_mask);
    __nv_bfloat16 cur;
    *reinterpret_cast<unsigned short *>(&cur)=cur_bits;
    const __nv_bfloat16 sum=__float2bfloat16(__bfloat162float(cur)+val_f);
    const unsigned short sum_bits=*reinterpret_cast<const unsigned short *>(&sum);
    const unsigned int new_word=(assumed & ~(g_cu_half_word_mask<<shift))|
                                (static_cast<unsigned int>(sum_bits)<<shift);
    old=atomicCAS(base,assumed,new_word);
  }while(assumed!=old);
#endif
}

//------------------------------------------------------------------------------
// Warp-level sum reduction using shuffle intrinsics.
// Avoids shared memory entirely. Returns the sum in lane 0.
// All 32 lanes in the warp must call this with their value.
//------------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val)
{
  for(int offset=g_cu_warp_half_size;offset>0;offset/=2)
  {
    val+=__shfl_down_sync(g_cu_warp_full_mask,val,offset);
  }
  return val;
}

//------------------------------------------------------------------------------
// Warp-level max reduction using shuffle intrinsics.
// Returns the max in lane 0.
//------------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val)
{
  for(int offset=g_cu_warp_half_size;offset>0;offset/=2)
  {
    val=fmaxf(val,__shfl_down_sync(g_cu_warp_full_mask,val,offset));
  }
  return val;
}
