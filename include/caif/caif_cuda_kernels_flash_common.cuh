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
// Shared device helpers for the two flash-attention CUDA translation units
// (caif_cuda_kernels_flash_self.cu and the cross-attention kernels). Device-
// internal -- lives in src/, not part of the public API. Helpers stay
// __forceinline__/inline (never static-ified).
//------------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "caif_cuda_kernels_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Only compile TC kernel for sm_80+
#define CAIF_HAS_TC_FLASH 1
#else
#define CAIF_HAS_TC_FLASH 0
#endif

__device__ __forceinline__ void cp_async_f2(void *dst_shared,
                                            const void *src_global)
{
  const uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
               ::"r"(dst),"l"(src_global));
}

__device__ __forceinline__ void cp_async_commit()
{
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait()
{
  asm volatile("cp.async.wait_group 0;\n");
}


//------------------------------------------------------------------------------
// Adaptive TC/scalar dispatch for flash attention forward.
// Computes smem from formula for each (BR,BC) candidate and picks the
// largest BC that fits in the GPU's optin shared memory limit.
//------------------------------------------------------------------------------
constexpr size_t fa_tc_smem(int d,int br,int bc)
{
  // Padded strides eliminate smem bank conflicts for wmma loads.
  // Q_tile[BR*(D+pad)] + KV_buf[BC*(D+pad)] + S_tile[BR*(BC+pad)] + row_max/sum[2*BR]
  const int d_pad=d+g_cu_fa_smem_pad;
  const int bc_pad=bc+g_cu_fa_smem_pad;
  return static_cast<size_t>(br*d_pad+bc*d_pad+br*bc_pad+g_cu_fa_smem_stat_arrays*br)*sizeof(float);
}

constexpr size_t fa_scalar_smem(int d)
{
  return static_cast<size_t>(g_cu_fa_kv_tiles*g_cu_fa_fwd_bc*d)*sizeof(float);
}

// Compute optimal number of warps for a given TC tile configuration.
// max_tiles: largest of S-tiles and O-tiles that warps must cover.
// blocks_from_smem: how many blocks the SM can hold from shared memory alone.
// If 2+ blocks fit, halve warps per block to double occupancy.
inline int fa_tc_optimal_nw(int max_tiles,
                            int blocks_from_smem,
                            int tiles_m,
                            int tiles_n_s,
                            int tiles_n_o)
{
  int nw;
  if(blocks_from_smem>=g_cu_fa_tc_occupancy_blocks && max_tiles>=g_cu_fa_tc_halve_min_tiles)
  {
    nw=max_tiles/2;
    if(nw<g_cu_fa_tc_warps_min)
    {
      nw=g_cu_fa_tc_warps_min;
    }
  }
  else if(max_tiles>g_cu_fa_tc_warps_large)
  {
    nw=g_cu_fa_tc_warps_large;
  }
  else if(max_tiles<g_cu_fa_tc_warps_min)
  {
    nw=g_cu_fa_tc_warps_min;
  }
  else
  {
    nw=max_tiles;
  }

  // Ensure register softmax constraints:
  // nw % tiles_m == 0, tiles_n_s and tiles_n_o divisible by warps_per_m
  while(nw>g_cu_fa_tc_warps_min)
  {
    if(nw%tiles_m==0)
    {
      const int wpm=nw/tiles_m;
      if(tiles_n_s%wpm==0 && tiles_n_o%wpm==0)
      {
        break;
      }
    }
    nw/=2;
  }

  return nw;
}
