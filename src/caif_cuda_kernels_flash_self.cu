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
// FlashAttention-2 self-attention CUDA kernels (forward + backward, TC and
// scalar paths). Carved verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_flash_self.cuh
//------------------------------------------------------------------------------
// Disable GNU C++ extensions to avoid rsqrt conflict between CUDA and glibc
// This must be set BEFORE any includes
#undef _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "caif_cuda_kernels_common.cuh"
#include "caif_cuda_kernels_flash_common.cuh"

//------------------------------------------------------------------------------
// FlashAttention-2 Forward Kernel
// Implements tiled attention with online softmax to avoid O(n²) memory
// References: https://arxiv.org/abs/2307.08691
//------------------------------------------------------------------------------

// Legacy block sizes removed — forward uses g_cu_fa_fwd_bc, backward uses g_cu_fa_bwd_*

//------------------------------------------------------------------------------
// FlashAttention-2 Forward — TF32 Tensor Core Kernel (sm_80+)
//
// Uses nvcuda::wmma 16x16x8 TF32 tiles for the two matmuls per KV block:
//   S = Q @ K^T  (scores)
//   O += softmax(S) @ V  (output accumulation)
//
// Template: D=head_dim, BR=Q rows/block, BC=KV tile cols
// Grid: (batch_heads, ceil(seq_len/BR))
// Block: n_warps*g_cu_warp_size where
//        n_warps = (BR/g_cu_wmma_tile_m) * (BC/g_cu_wmma_tile_n) / tiles_per_warp_s
//
// Shared memory: Q_tile[BR*(D+2)] + KV_buf[BC*(D+2)] + S_tile[BR*(BC+2)]
//                + row_max[BR] + row_sum[BR]  (padded strides for bank conflicts)
// O accumulator and S scores live in wmma register fragments.
// Softmax computed in-register with cross-warp reduce via S_tile[0..NW*16].
//------------------------------------------------------------------------------

// NO __launch_bounds__ here — tested and removed 2026-06-10: with it, every
// fa_kernel shape measured slower (Qwen-1.5B b=4 s=2048: 1.794/1.780 ms vs
// 1.762 ms without).
template<typename T,int D,int BR,int BC,int NW>
__global__ void flash_attention_forward_tc_kernel(const T *__restrict__ Q,
                                                  const T *__restrict__ K,
                                                  const T *__restrict__ V,
                                                  T *__restrict__ O,
                                                  float *__restrict__ L,
                                                  const int seq_len,
                                                  const float scale,
                                                  const float softcap,
                                                  const int causal,
                                                  const int window,
                                                  const uint32_t *__restrict__ prefix_lens,
                                                  const int num_heads,
                                                  const int num_kv_heads,
                                                  const float *__restrict__ alibi_slopes)
{
#if CAIF_HAS_TC_FLASH
  const int bh=blockIdx.x;
  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }
  // ALiBi linear position bias: per-head slope (geometric in the head index),
  // applied as slope*(k-q) on the logits below. Off (slope 0) when no slopes
  // are supplied.
  float alibi_slope=0.0f;
  if(alibi_slopes!=nullptr)
  {
    alibi_slope=alibi_slopes[bh%num_heads];
  }
  // Native GQA: Q/O index by the full bh, K/V index by the KV head group.
  // For MHA (num_kv_heads == num_heads) this is the identity map.
  const int bh_kv=bh*num_kv_heads/num_heads;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int lane_id=tid%g_cu_warp_size;
  const int warp_id=tid/g_cu_warp_size;

  constexpr int n_warps=NW;
  constexpr int tiles_m=BR/g_cu_wmma_tile_m;
  constexpr int tiles_n_s=BC/g_cu_wmma_tile_n;
  constexpr int tiles_n_o=D/g_cu_wmma_tile_n;
  constexpr int block_threads=n_warps*g_cu_warp_size;

  // Smem stride padding: +2 eliminates bank conflicts for wmma loads.
  // stride % 32 == 2 → each of 16 rows within a wmma tile maps to a
  // distinct pair of banks, giving zero conflicts on column-parallel access.
  constexpr int d_pad=D+g_cu_fa_smem_pad;
  constexpr int bc_pad=BC+g_cu_fa_smem_pad;
  constexpr int d_f2=D/g_cu_float2_lanes;
  constexpr int d_pad_f2=d_pad/g_cu_float2_lanes;

  // Shared memory layout (O is in registers, not smem)
  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *KV_buf=Q_tile+BR*d_pad;
  float *S_tile=KV_buf+BC*d_pad;
  float *row_max_arr=S_tile+BR*bc_pad;
  float *row_sum_arr=row_max_arr+BR;

  // Batch-head pointers. Q/O follow the full bh (one per query head);
  // K/V follow bh_kv so native GQA avoids materializing a repeat-expanded
  // KV tensor. For MHA this reduces to bh_kv == bh.
  const T *Q_bh=Q+static_cast<size_t>(bh)*seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh_kv)*seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh_kv)*seq_len*D;
  T *O_bh=O+static_cast<size_t>(bh)*seq_len*D;
  float *L_bh=L+static_cast<size_t>(bh)*seq_len;

  const int q_start=q_block_idx*BR;

  // Warp grouping for register-based softmax.
  // Each M-tile group's warps collectively cover all N S-tiles, enabling
  // in-register softmax with cross-warp reduce (no S_tile smem round-trip).
  constexpr int warps_per_m=n_warps/tiles_m;
  constexpr int s_tiles_pw=(tiles_n_s>=warps_per_m)*(tiles_n_s/warps_per_m);
  constexpr int o_tiles_pw=(tiles_n_o>=warps_per_m)*(tiles_n_o/warps_per_m);
  // Array size must be >=1 for CUDA; loops guard on tile count
  constexpr int s_arr=s_tiles_pw+(s_tiles_pw==0);
  constexpr int o_arr=o_tiles_pw+(o_tiles_pw==0);
  const int m_idx=warp_id/warps_per_m;
  const int group_warp=warp_id%warps_per_m;
  const int n_start_s=group_warp*s_tiles_pw;
  const int n_start_o=group_warp*o_tiles_pw;
  const int group_base=m_idx*warps_per_m;

  // Persistent O accumulators in wmma registers
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                         g_cu_wmma_tile_m,
                         g_cu_wmma_tile_n,
                         g_cu_wmma_tile_k,
                         float> o_frags[o_arr];
  for(int t=0;t<o_tiles_pw;++t)
  {
    nvcuda::wmma::fill_fragment(o_frags[t],0.0f);
  }

  // Cooperative load Q_tile[BR, d_pad] from global memory (padded stride)
  const int valid_q_rows=min(BR,seq_len-q_start);
  {
    if constexpr(sizeof(T)==sizeof(float))
    {
      const int valid_q_f2=max(valid_q_rows,0)*d_f2;
      constexpr int total_q_f2=BR*d_f2;
      const float2 *Q_src2=reinterpret_cast<const float2 *>(Q_bh+q_start*D);
      float2 *Q_dst2=reinterpret_cast<float2 *>(Q_tile);
      for(int i=tid;i<valid_q_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        Q_dst2[row*d_pad_f2+f2c]=Q_src2[i];
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=valid_q_f2+tid;i<total_q_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        Q_dst2[row*d_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int valid_q=max(valid_q_rows,0)*D;
      constexpr int total_q=BR*D;
      for(int i=tid;i<valid_q;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        Q_tile[row*d_pad+col]=float(Q_bh[(q_start+row)*D+col]);
      }
      for(int i=valid_q+tid;i<total_q;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        Q_tile[row*d_pad+col]=0.0f;
      }
    }
  }

  // Init row_max/row_sum
  for(int i=tid;i<BR;i+=block_threads)
  {
    row_max_arr[i]=-INFINITY;
    row_sum_arr[i]=0.0f;
  }
  __syncthreads();

  // Number of KV blocks
  int num_kv_blocks=(seq_len+BC-1)/BC;
  if(causal==1 && prefix_lens==nullptr)
  {
    int max_q=q_start+BR-1;
    if(max_q>=seq_len)
    {
      max_q=seq_len-1;
    }
    num_kv_blocks=min(num_kv_blocks,(max_q/BC)+1);
  }

  constexpr int kv_f2=BC*d_f2;

  // Pipeline: prefetch K[0]
  float2 *KV_dst2=reinterpret_cast<float2 *>(KV_buf);
  if(num_kv_blocks>0)
  {
    if constexpr(sizeof(T)==sizeof(float))
    {
      const int kv0_valid=min(BC,seq_len)*d_f2;
      const float2 *K0_src2=reinterpret_cast<const float2 *>(K_bh);
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&K0_src2[i]);
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=kv0_valid+tid;i<kv_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        KV_dst2[row*d_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int kv0_valid=min(BC,seq_len)*D;
      constexpr int kv_total=BC*D;
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        KV_buf[row*d_pad+col]=float(K_bh[row*D+col]);
      }
      for(int i=kv0_valid+tid;i<kv_total;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        KV_buf[row*d_pad+col]=0.0f;
      }
    }
    cp_async_commit();
  }

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*BC;
    const int valid_kv_rows=min(BC,seq_len-kv_start);
    const int valid_kv_f2=valid_kv_rows*d_f2;

    // PHASE 1: Wait for K, compute S = Q @ K^T in wmma registers
    cp_async_wait();
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                           g_cu_wmma_tile_m,
                           g_cu_wmma_tile_n,
                           g_cu_wmma_tile_k,
                           float> s_accs[s_arr];
    for(int t=0;t<s_tiles_pw;++t)
    {
      const int n=n_start_s+t;
      nvcuda::wmma::fill_fragment(s_accs[t],0.0f);
      for(int k=0;k<D/g_cu_wmma_tile_k;++k)
      {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                               g_cu_wmma_tile_m,
                               g_cu_wmma_tile_n,
                               g_cu_wmma_tile_k,
                               nvcuda::wmma::precision::tf32,
                               nvcuda::wmma::row_major> q_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                               g_cu_wmma_tile_m,
                               g_cu_wmma_tile_n,
                               g_cu_wmma_tile_k,
                               nvcuda::wmma::precision::tf32,
                               nvcuda::wmma::col_major> k_frag;
        nvcuda::wmma::load_matrix_sync(q_frag,&Q_tile[m_idx*g_cu_wmma_tile_m*d_pad+k*g_cu_wmma_tile_k],d_pad);
        nvcuda::wmma::load_matrix_sync(k_frag,&KV_buf[n*g_cu_wmma_tile_n*d_pad+k*g_cu_wmma_tile_k],d_pad);
        nvcuda::wmma::mma_sync(s_accs[t],q_frag,k_frag,s_accs[t]);
      }
      for(int i=0;i<s_accs[t].num_elements;++i)
      {
        s_accs[t].x[i]*=scale;
      }
      // Gemma-2/3 logit soft-cap: squash scores into (-cap, cap) before the
      // mask/softmax. Uniform branch (softcap is a kernel arg) — the no-cap
      // path predicts through it at no cost.
      if(softcap>0.0f)
      {
        const float inv_softcap=1.0f/softcap;
        for(int i=0;i<s_accs[t].num_elements;++i)
        {
          s_accs[t].x[i]=softcap*tanhf(s_accs[t].x[i]*inv_softcap);
        }
      }
    }

    // Sync: all warps done reading KV_buf before V overwrites it
    __syncthreads();

    // Async V load into KV_buf (overlapped with softmax)
    {
      if constexpr(sizeof(T)==sizeof(float))
      {
        const float2 *V_src2=reinterpret_cast<const float2 *>(V_bh+kv_start*D);
        for(int i=tid;i<valid_kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&V_src2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=valid_kv_f2+tid;i<kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          KV_dst2[row*d_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int valid_kv=valid_kv_rows*D;
        constexpr int kv_total=BC*D;
        for(int i=tid;i<valid_kv;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=float(V_bh[(kv_start+row)*D+col]);
        }
        for(int i=valid_kv+tid;i<kv_total;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }

    // PHASE 2: Register-based online softmax on s_accs
    // Fragment layout (stable sm_80-sm_120):
    //   elements {0,1,4,5} → local_row = lane_id/4       (row_lo)
    //   elements {2,3,6,7} → local_row = lane_id/4 + 8   (row_hi)
    //   elem col: 0→(lane_id%g_cu_wmma_acc_lanes_per_row)*2, 1→+1, 4→+8, 5→+9 (same for 2/3/6/7)
    {
      const int row_lo=m_idx*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row);
      const int row_hi=m_idx*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row)+g_cu_wmma_acc_half_tile;
      const int global_q_lo=q_start+row_lo;
      const int global_q_hi=q_start+row_hi;

      // Apply causal + boundary masks in registers
      for(int t=0;t<s_tiles_pw;++t)
      {
        const int n=n_start_s+t;
        const int bc0=kv_start+
                      n*g_cu_wmma_tile_n+
                      (lane_id%g_cu_wmma_acc_lanes_per_row)*g_cu_wmma_acc_cols_per_lane;
        const int bc1=bc0+1;
        const int bc2=kv_start+
                      n*g_cu_wmma_tile_n+
                      (lane_id%g_cu_wmma_acc_lanes_per_row)*g_cu_wmma_acc_cols_per_lane+
                      g_cu_wmma_acc_half_tile;
        const int bc3=bc2+1;

        // ALiBi linear position bias: add slope*(k-q) before the mask. Same
        // per-element layout as the masks below (x[0,1,4,5]->q_lo,
        // x[2,3,6,7]->q_hi; cols bc0..bc3). The additive constant does not
        // enter the backward's score gradient, but the backward recompute adds
        // the same bias so its softmax matches the forward.
        if(alibi_slope!=0.0f)
        {
          s_accs[t].x[0]+=alibi_slope*static_cast<float>(bc0-global_q_lo);
          s_accs[t].x[1]+=alibi_slope*static_cast<float>(bc1-global_q_lo);
          s_accs[t].x[2]+=alibi_slope*static_cast<float>(bc0-global_q_hi);
          s_accs[t].x[3]+=alibi_slope*static_cast<float>(bc1-global_q_hi);
          s_accs[t].x[4]+=alibi_slope*static_cast<float>(bc2-global_q_lo);
          s_accs[t].x[5]+=alibi_slope*static_cast<float>(bc3-global_q_lo);
          s_accs[t].x[6]+=alibi_slope*static_cast<float>(bc2-global_q_hi);
          s_accs[t].x[7]+=alibi_slope*static_cast<float>(bc3-global_q_hi);
        }
        // Prefix-LM: allowed iff (k<=q) OR (k<pfx). Plain causal: k<=q.
        // With prefix_lens==nullptr we reduce exactly to the causal-only rule.
        if(causal==1)
        {
          if(bc0>global_q_lo && bc0>=pfx)
          {
            s_accs[t].x[0]=-INFINITY;
          }
          if(bc1>global_q_lo && bc1>=pfx)
          {
            s_accs[t].x[1]=-INFINITY;
          }
          if(bc0>global_q_hi && bc0>=pfx)
          {
            s_accs[t].x[2]=-INFINITY;
          }
          if(bc1>global_q_hi && bc1>=pfx)
          {
            s_accs[t].x[3]=-INFINITY;
          }
          if(bc2>global_q_lo && bc2>=pfx)
          {
            s_accs[t].x[4]=-INFINITY;
          }
          if(bc3>global_q_lo && bc3>=pfx)
          {
            s_accs[t].x[5]=-INFINITY;
          }
          if(bc2>global_q_hi && bc2>=pfx)
          {
            s_accs[t].x[6]=-INFINITY;
          }
          if(bc3>global_q_hi && bc3>=pfx)
          {
            s_accs[t].x[7]=-INFINITY;
          }
        }
        // Sliding-window mask: a key more than `window` positions before the
        // query is out of the window. Same per-element layout as the causal
        // mask above (x[0,1,4,5]->q_lo, x[2,3,6,7]->q_hi; cols bc0..bc3).
        if(window>0)
        {
          if(global_q_lo-bc0>=window){s_accs[t].x[0]=-INFINITY;}
          if(global_q_lo-bc1>=window){s_accs[t].x[1]=-INFINITY;}
          if(global_q_hi-bc0>=window){s_accs[t].x[2]=-INFINITY;}
          if(global_q_hi-bc1>=window){s_accs[t].x[3]=-INFINITY;}
          if(global_q_lo-bc2>=window){s_accs[t].x[4]=-INFINITY;}
          if(global_q_lo-bc3>=window){s_accs[t].x[5]=-INFINITY;}
          if(global_q_hi-bc2>=window){s_accs[t].x[6]=-INFINITY;}
          if(global_q_hi-bc3>=window){s_accs[t].x[7]=-INFINITY;}
        }
        // K boundary mask: zero-padded K positions beyond seq_len
        // must be masked to -inf so they don't participate in
        // softmax. Causal mask catches these implicitly; non-causal
        // does not.
        if(bc0>=seq_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[2]=-INFINITY;
        }
        if(bc1>=seq_len)
        {
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
        }
        if(bc2>=seq_len)
        {
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
        }
        if(bc3>=seq_len)
        {
          s_accs[t].x[5]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }
        if(global_q_lo>=seq_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[5]=-INFINITY;
        }
        if(global_q_hi>=seq_len)
        {
          s_accs[t].x[2]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }
      }

      // Local max across this warp's S tiles
      float max_lo=-INFINITY;
      float max_hi=-INFINITY;
      for(int t=0;t<s_tiles_pw;++t)
      {
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[0],s_accs[t].x[1]));
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[4],s_accs[t].x[5]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[2],s_accs[t].x[3]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[6],s_accs[t].x[7]));
      }

      // Reduce max within 4-thread row group (lane_id%g_cu_wmma_acc_lanes_per_row groups)
      max_lo=fmaxf(max_lo,__shfl_xor_sync(g_cu_warp_full_mask,max_lo,1));
      max_lo=fmaxf(max_lo,__shfl_xor_sync(g_cu_warp_full_mask,max_lo,2));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(g_cu_warp_full_mask,max_hi,1));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(g_cu_warp_full_mask,max_hi,2));

      // Cross-warp max reduce via S_tile[0..NW*16] temporary
      float *reduce_buf=S_tile;
      if(lane_id%g_cu_wmma_acc_lanes_per_row==0)
      {
        reduce_buf[warp_id*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row)]=max_lo;
        reduce_buf[warp_id*g_cu_wmma_tile_m+
                   (lane_id/g_cu_wmma_acc_lanes_per_row)+
                   g_cu_wmma_acc_half_tile]=max_hi;
      }
      __syncthreads();

      float full_max_lo=-INFINITY;
      float full_max_hi=-INFINITY;
      for(int w=group_base;w<group_base+warps_per_m;++w)
      {
        full_max_lo=fmaxf(full_max_lo,reduce_buf[w*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row)]);
        full_max_hi=fmaxf(full_max_hi,
                          reduce_buf[w*g_cu_wmma_tile_m+
                                     (lane_id/g_cu_wmma_acc_lanes_per_row)+
                                     g_cu_wmma_acc_half_tile]);
      }

      // Online correction factor
      const float old_max_lo=row_max_arr[row_lo];
      const float old_max_hi=row_max_arr[row_hi];
      const float new_max_lo=fmaxf(old_max_lo,full_max_lo);
      const float new_max_hi=fmaxf(old_max_hi,full_max_hi);
      const float corr_lo=__expf(old_max_lo-new_max_lo);
      const float corr_hi=__expf(old_max_hi-new_max_hi);

      // Compute exp(S - new_max) in place, accumulate local sum
      float sum_lo=0.0f;
      float sum_hi=0.0f;
      for(int t=0;t<s_tiles_pw;++t)
      {
        s_accs[t].x[0]=__expf(s_accs[t].x[0]-new_max_lo);
        sum_lo+=s_accs[t].x[0];
        s_accs[t].x[1]=__expf(s_accs[t].x[1]-new_max_lo);
        sum_lo+=s_accs[t].x[1];
        s_accs[t].x[4]=__expf(s_accs[t].x[4]-new_max_lo);
        sum_lo+=s_accs[t].x[4];
        s_accs[t].x[5]=__expf(s_accs[t].x[5]-new_max_lo);
        sum_lo+=s_accs[t].x[5];
        s_accs[t].x[2]=__expf(s_accs[t].x[2]-new_max_hi);
        sum_hi+=s_accs[t].x[2];
        s_accs[t].x[3]=__expf(s_accs[t].x[3]-new_max_hi);
        sum_hi+=s_accs[t].x[3];
        s_accs[t].x[6]=__expf(s_accs[t].x[6]-new_max_hi);
        sum_hi+=s_accs[t].x[6];
        s_accs[t].x[7]=__expf(s_accs[t].x[7]-new_max_hi);
        sum_hi+=s_accs[t].x[7];
      }

      // Reduce sum within 4-thread row group
      sum_lo+=__shfl_xor_sync(g_cu_warp_full_mask,sum_lo,1);
      sum_lo+=__shfl_xor_sync(g_cu_warp_full_mask,sum_lo,2);
      sum_hi+=__shfl_xor_sync(g_cu_warp_full_mask,sum_hi,1);
      sum_hi+=__shfl_xor_sync(g_cu_warp_full_mask,sum_hi,2);

      // Cross-warp sum reduce
      if(lane_id%g_cu_wmma_acc_lanes_per_row==0)
      {
        reduce_buf[warp_id*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row)]=sum_lo;
        reduce_buf[warp_id*g_cu_wmma_tile_m+
                   (lane_id/g_cu_wmma_acc_lanes_per_row)+
                   g_cu_wmma_acc_half_tile]=sum_hi;
      }
      __syncthreads();

      float full_sum_lo=0.0f;
      float full_sum_hi=0.0f;
      for(int w=group_base;w<group_base+warps_per_m;++w)
      {
        full_sum_lo+=reduce_buf[w*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row)];
        full_sum_hi+=reduce_buf[w*g_cu_wmma_tile_m+
                                (lane_id/g_cu_wmma_acc_lanes_per_row)+
                                g_cu_wmma_acc_half_tile];
      }

      // Update row state (one warp per group writes)
      if(group_warp==0 && lane_id%g_cu_wmma_acc_lanes_per_row==0)
      {
        row_sum_arr[row_lo]=corr_lo*row_sum_arr[row_lo]+full_sum_lo;
        row_sum_arr[row_hi]=corr_hi*row_sum_arr[row_hi]+full_sum_hi;
        row_max_arr[row_lo]=new_max_lo;
        row_max_arr[row_hi]=new_max_hi;
      }

      // Rescale O fragments by correction (all O tiles share this warp's m_idx)
      for(int t=0;t<o_tiles_pw;++t)
      {
        o_frags[t].x[0]*=corr_lo;
        o_frags[t].x[1]*=corr_lo;
        o_frags[t].x[2]*=corr_hi;
        o_frags[t].x[3]*=corr_hi;
        o_frags[t].x[4]*=corr_lo;
        o_frags[t].x[5]*=corr_lo;
        o_frags[t].x[6]*=corr_hi;
        o_frags[t].x[7]*=corr_hi;
      }
    }

    // Barrier: S_tile doubles as the cross-warp sum/max reduce buffer
    // above; without this sync a fast warp can begin the wmma store below
    // and overwrite reduce_buf slots that a slower warp in the same
    // m_idx group is still reading. Latent at NW=2/4 (warps near
    // lockstep), reliably corrupts output at NW=8.
    __syncthreads();

    // Store exp(S) to S_tile for Phase 3 (single write, padded stride)
    for(int t=0;t<s_tiles_pw;++t)
    {
      const int n=n_start_s+t;
      nvcuda::wmma::store_matrix_sync(&S_tile[m_idx*g_cu_wmma_tile_m*bc_pad+n*g_cu_wmma_tile_n],
                                      s_accs[t],
                                      bc_pad,
                                      nvcuda::wmma::mem_row_major);
    }

    // Wait for V + ensure S_tile writes visible
    cp_async_wait();
    __syncthreads();

    // PHASE 3: Accumulate O += softmax(S) @ V using tensor cores (padded strides)
    for(int t=0;t<o_tiles_pw;++t)
    {
      const int n=n_start_o+t;
      for(int k=0;k<BC/g_cu_wmma_tile_k;++k)
      {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                               g_cu_wmma_tile_m,
                               g_cu_wmma_tile_n,
                               g_cu_wmma_tile_k,
                               nvcuda::wmma::precision::tf32,
                               nvcuda::wmma::row_major> s_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                               g_cu_wmma_tile_m,
                               g_cu_wmma_tile_n,
                               g_cu_wmma_tile_k,
                               nvcuda::wmma::precision::tf32,
                               nvcuda::wmma::row_major> v_frag;
        nvcuda::wmma::load_matrix_sync(s_frag,&S_tile[m_idx*g_cu_wmma_tile_m*bc_pad+k*g_cu_wmma_tile_k],bc_pad);
        nvcuda::wmma::load_matrix_sync(v_frag,&KV_buf[k*g_cu_wmma_tile_k*d_pad+n*g_cu_wmma_tile_n],d_pad);
        nvcuda::wmma::mma_sync(o_frags[t],s_frag,v_frag,o_frags[t]);
      }
    }
    __syncthreads();

    // Pipeline: prefetch K[next] into KV_buf
    if(kv_block+1<num_kv_blocks)
    {
      const int next_start=(kv_block+1)*BC;
      if constexpr(sizeof(T)==sizeof(float))
      {
        const int next_valid=min(BC,seq_len-next_start)*d_f2;
        const float2 *K_next2=reinterpret_cast<const float2 *>(K_bh+next_start*D);
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&K_next2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=next_valid+tid;i<kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          KV_dst2[row*d_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int next_valid=min(BC,seq_len-next_start)*D;
        constexpr int kv_total=BC*D;
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=float(K_bh[(next_start+row)*D+col]);
        }
        for(int i=next_valid+tid;i<kv_total;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }
  }

  // Final: store O fragments to smem (reuse Q_tile, padded stride), normalize, write
  __syncthreads();
  float *O_smem=Q_tile;
  for(int t=0;t<o_tiles_pw;++t)
  {
    const int n=n_start_o+t;
    nvcuda::wmma::store_matrix_sync(&O_smem[m_idx*g_cu_wmma_tile_m*d_pad+n*g_cu_wmma_tile_n],
                                    o_frags[t],
                                    d_pad,
                                    nvcuda::wmma::mem_row_major);
  }
  __syncthreads();

  for(int i=tid;i<BR*D;i+=block_threads)
  {
    const int row=i/D;
    const int col=i-row*D;
    const int global_row=q_start+row;
    if(global_row<seq_len)
    {
      float inv_l=0.0f;
      if(row_sum_arr[row]>0.0f)
      {
        inv_l=1.0f/row_sum_arr[row];
      }
      O_bh[global_row*D+col]=T(O_smem[row*d_pad+col]*inv_l);
    }
  }

  // Write logsumexp (one per Q row)
  for(int r=tid;r<BR;r+=block_threads)
  {
    const int global_row=q_start+r;
    if(global_row<seq_len)
    {
      L_bh[global_row]=row_max_arr[r]+logf(row_sum_arr[r]+g_cu_fa_logsumexp_epsilon);
    }
  }
#endif  // CAIF_HAS_TC_FLASH
}

template<typename T,int D,int BR,int BC,int NW>
static void launch_fa_fwd_tc(const T *Q,
                             const T *K,
                             const T *V,
                             T *O,
                             float *L,
                             const int batch_heads,
                             const int seq_len,
                             const float scale,
                             const float softcap,
                             const int causal,
                             const int window,
                             const uint32_t *prefix_lens,
                             const int num_heads,
                             const int num_kv_heads,
                             const float *alibi_slopes,
                             cudaStream_t stream)
{
  const int num_q_blocks=(seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(NW*g_cu_warp_size);
  constexpr size_t smem_size=fa_tc_smem(D,BR,BC);

  if(smem_size>g_cu_default_shared_memory)
  {
    cudaFuncSetAttribute((void *)flash_attention_forward_tc_kernel<T,D,BR,BC,NW>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_size));
  }
  flash_attention_forward_tc_kernel<T,D,BR,BC,NW>
    <<<grid,block,smem_size,stream>>>(Q,
                                      K,
                                      V,
                                      O,
                                      L,
                                      seq_len,
                                      scale,
                                      softcap,
                                      causal,
                                      window,
                                      prefix_lens,
                                      num_heads,
                                      num_kv_heads,
                                      alibi_slopes);
}

//------------------------------------------------------------------------------
// FlashAttention-2 Forward Kernel — Warp-Per-Row (Memory-Efficient Fallback)
//
// Each warp (32 threads) processes one Q row. Lanes parallelize across
// head_dim D. BR warps per block = BR Q rows per block.
//
// Grid: (batch_heads, ceil(seq_len / BR))
// Block: (BR * 32) threads
//
// Template: D = head_dim, BR = Q rows per block
// KV tile size is g_cu_fa_fwd_bc (caif_cuda_kernels_flash_common.cuh).
//
// Q lives in registers (no Q tile in shared memory).
// Two-pass score computation eliminates S_local register array:
//   Pass 1: dot products via warp reduce, find row_max
//   Pass 2: recompute dots, compute exp, accumulate V

template<typename T,int D,int BR>
__global__ void flash_attention_forward_kernel(const T *__restrict__ Q,
                                               const T *__restrict__ K,
                                               const T *__restrict__ V,
                                               T *__restrict__ O,
                                               float *__restrict__ L,
                                               const int seq_len,
                                               const float scale,
                                               const float softcap,
                                               const int causal,
                                               const int window,
                                               const uint32_t *__restrict__ prefix_lens,
                                               const int num_heads,
                                               const int num_kv_heads,
                                               const float *__restrict__ alibi_slopes)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int warp_id=tid/g_cu_warp_size;
  const int lane_id=tid%g_cu_warp_size;

  const int q_row=q_block_idx*BR+warp_id;
  const bool q_valid=(q_row<seq_len);

  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }

  // ALiBi linear position bias: per-head slope, applied as slope*(k-q) on the
  // logits below. Off (slope 0) when no slopes are supplied.
  float alibi_slope=0.0f;
  if(alibi_slopes!=nullptr)
  {
    alibi_slope=alibi_slopes[bh%num_heads];
  }

  // Native GQA: map the Q head index onto its KV head group so the kernel
  // can attend against a [batch*num_kv_heads, seq, D] KV tensor without the
  // caller materializing a repeat-expanded copy. When num_kv_heads equals
  // num_heads this collapses to the MHA identity (bh_kv == bh).
  const int bh_kv=bh*num_kv_heads/num_heads;

  // Shared memory: K tile + V tile (Q is in registers)
  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_cu_fa_fwd_bc*D;

  // Batch-head pointers
  const T *Q_bh=Q+static_cast<size_t>(bh)*seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh_kv)*seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh_kv)*seq_len*D;
  T *O_bh=O+static_cast<size_t>(bh)*seq_len*D;
  float *L_bh=L+static_cast<size_t>(bh)*seq_len;

  // Load Q row into registers — each lane holds ceil(D/g_cu_warp_size) elements
  constexpr int elems=(D+g_cu_warp_size-1)/g_cu_warp_size;
  float q_reg[elems];
  for(int dd=lane_id,e=0;dd<D;dd+=g_cu_warp_size,++e)
  {
    if(q_valid==true)
    {
      q_reg[e]=float(Q_bh[q_row*D+dd]);
    }
    else
    {
      q_reg[e]=0.0f;
    }
  }

  // Output accumulators in registers
  float m_i=-INFINITY;
  float l_i=0.0f;
  float o_reg[elems];
  for(int e=0;e<elems;++e)
  {
    o_reg[e]=0.0f;
  }

  // Number of KV blocks — uniform across block for cooperative loads
  int num_kv_blocks=(seq_len+g_cu_fa_fwd_bc-1)/g_cu_fa_fwd_bc;
  if(causal==1 && prefix_lens==nullptr)
  {
    int max_q=q_block_idx*BR+BR-1;
    if(max_q>=seq_len)
    {
      max_q=seq_len-1;
    }
    num_kv_blocks=min(num_kv_blocks,(max_q/g_cu_fa_fwd_bc)+1);
  }

  const int block_threads=BR*g_cu_warp_size;

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_cu_fa_fwd_bc;

    // Cooperative K/V tile load — all threads in block participate
    const int tile_elems=g_cu_fa_fwd_bc*D;
    for(int i=tid;i<tile_elems;i+=block_threads)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;
      if(global_row<seq_len)
      {
        K_tile[row*D+col]=float(K_bh[global_row*D+col]);
        V_tile[row*D+col]=float(V_bh[global_row*D+col]);
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    // Single pass: compute scores, find max, then exp + accumulate V
    float s_local[g_cu_fa_fwd_bc];
    float row_max=-INFINITY;
    int num_valid=0;

    for(int j=0;j<g_cu_fa_fwd_bc;++j)
    {
      const int k_row=kv_start+j;
      if(k_row>=seq_len)
      {
        break;
      }
      // Prefix-LM: allowed iff (k<=q) OR (k<pfx). Plain causal: k<=q.
      bool masked=false;
      if(prefix_lens!=nullptr)
      {
        masked=(k_row>q_row) && (k_row>=pfx);
      }
      else if(causal==1)
      {
        masked=(k_row>q_row);
        if(masked==true)
        {
          break;
        }
      }
      // Sliding-window mask: a key more than `window` positions before the
      // query is out of the window (does not break — later keys are in-window).
      if(window>0 && (q_row-k_row)>=window)
      {
        masked=true;
      }

      float dot=0.0f;
      for(int dd=lane_id,e=0;dd<D;dd+=g_cu_warp_size,++e)
      {
        dot+=q_reg[e]*K_tile[j*D+dd];
      }
      // Warp reduce sum
      for(int offset=g_cu_warp_half_size;offset>0;offset/=2)
      {
        dot+=__shfl_down_sync(g_cu_warp_full_mask,dot,offset);
      }
      // Broadcast from lane 0
      dot=__shfl_sync(g_cu_warp_full_mask,dot,0);
      if(masked==true)
      {
        s_local[j]=-INFINITY;
      }
      else
      {
        s_local[j]=dot*scale;
        // Gemma-2/3 logit soft-cap, before the running max (uniform branch).
        if(softcap>0.0f)
        {
          s_local[j]=softcap*tanhf(s_local[j]/softcap);
        }
        // ALiBi linear position bias on the logit, before the running max.
        if(alibi_slope!=0.0f)
        {
          s_local[j]+=alibi_slope*static_cast<float>(k_row-q_row);
        }
        if(s_local[j]>row_max)
        {
          row_max=s_local[j];
        }
      }
      num_valid=j+1;
    }

    // Online softmax rescale
    const float m_new=fmaxf(m_i,row_max);
    const float scale_old=expf(m_i-m_new);
    for(int e=0;e<elems;++e)
    {
      o_reg[e]*=scale_old;
    }

    // Exp + V accumulation from stored scores
    float block_sum=0.0f;
    for(int j=0;j<num_valid;++j)
    {
      const float p=expf(s_local[j]-m_new);
      block_sum+=p;
      for(int dd=lane_id,e=0;dd<D;dd+=g_cu_warp_size,++e)
      {
        o_reg[e]+=p*V_tile[j*D+dd];
      }
    }

    l_i=scale_old*l_i+block_sum;
    m_i=m_new;
    __syncthreads();
  }

  // Final normalization and output
  if(q_valid==true)
  {
    float inv_l=0.0f;
    if(l_i>0.0f)
    {
      inv_l=1.0f/l_i;
    }
    for(int dd=lane_id,e=0;dd<D;dd+=g_cu_warp_size,++e)
    {
      O_bh[q_row*D+dd]=T(o_reg[e]*inv_l);
    }
    // Only lane 0 writes logsumexp (one value per Q row)
    if(lane_id==0)
    {
      L_bh[q_row]=m_i+logf(l_i+g_cu_fa_logsumexp_epsilon);
    }
  }
}

// Helper: launch a specific <D,BR> instantiation with opt-in shared memory
template<typename T,int D,int BR>
static void launch_fa_fwd(const T *Q,
                          const T *K,
                          const T *V,
                          T *O,
                          float *L,
                          const int batch_heads,
                          const int seq_len,
                          const float scale,
                          const float softcap,
                          const int causal,
                          const int window,
                          const uint32_t *prefix_lens,
                          const int num_heads,
                          const int num_kv_heads,
                          const float *alibi_slopes,
                          cudaStream_t stream)
{
  const int num_q_blocks=(seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(BR*g_cu_warp_size);
  const size_t smem_size=fa_scalar_smem(D);

  if(smem_size>g_cu_default_shared_memory)
  {
    cudaFuncSetAttribute((void *)flash_attention_forward_kernel<T,D,BR>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_size));
  }
  flash_attention_forward_kernel<T,D,BR>
    <<<grid,block,smem_size,stream>>>(Q,
                                      K,
                                      V,
                                      O,
                                      L,
                                      seq_len,
                                      scale,
                                      softcap,
                                      causal,
                                      window,
                                      prefix_lens,
                                      num_heads,
                                      num_kv_heads,
                                      alibi_slopes);
}

// Dispatch helper that selects NW from the computed optimal value.
// Instantiates NW=2, NW=4, and NW=8; runtime picks the closest one.
template<typename T,int D,int BR,int BC>
static void dispatch_fa_fwd_tc_nw(const T *Q,
                                  const T *K,
                                  const T *V,
                                  T *O,
                                  float *L,
                                  const int batch_heads,
                                  const int seq_len,
                                  const float scale,
                                  const float softcap,
                                  const int causal,
                                  const int window,
                                  const uint32_t *prefix_lens,
                                  const int num_heads,
                                  const int num_kv_heads,
                                  const float *alibi_slopes,
                                  cudaStream_t stream,
                                  const int nw)
{
  if(nw<=g_cu_fa_tc_warps_min)
  {
    launch_fa_fwd_tc<T,D,BR,BC,g_cu_fa_tc_warps_min>(Q,
                                                     K,
                                                     V,
                                                     O,
                                                     L,
                                                     batch_heads,
                                                     seq_len,
                                                     scale,
                                                     softcap,
                                                     causal,
                                                     window,
                                                     prefix_lens,
                                                     num_heads,
                                                     num_kv_heads,
                                                     alibi_slopes,
                                                     stream);
  }
  else if(nw<=g_cu_fa_tc_warps_small)
  {
    launch_fa_fwd_tc<T,D,BR,BC,g_cu_fa_tc_warps_small>(Q,
                                                       K,
                                                       V,
                                                       O,
                                                       L,
                                                       batch_heads,
                                                       seq_len,
                                                       scale,
                                                       softcap,
                                                       causal,
                                                       window,
                                                       prefix_lens,
                                                       num_heads,
                                                       num_kv_heads,
                                                       alibi_slopes,
                                                       stream);
  }
  else
  {
    launch_fa_fwd_tc<T,D,BR,BC,g_cu_fa_tc_warps_large>(Q,
                                                       K,
                                                       V,
                                                       O,
                                                       L,
                                                       batch_heads,
                                                       seq_len,
                                                       scale,
                                                       softcap,
                                                       causal,
                                                       window,
                                                       prefix_lens,
                                                       num_heads,
                                                       num_kv_heads,
                                                       alibi_slopes,
                                                       stream);
  }
}

template<typename T,int D>
static void dispatch_fa_fwd_tc(const T *Q,
                               const T *K,
                               const T *V,
                               T *O,
                               float *L,
                               const int batch_heads,
                               const int seq_len,
                               const float scale,
                               const float softcap,
                               const int causal,
                               const int window,
                               const uint32_t *prefix_lens,
                               const int num_heads,
                               const int num_kv_heads,
                               const float *alibi_slopes,
                               cudaStream_t stream,
                               const int cc_major,
                               const size_t smem_limit,
                               const size_t smem_per_sm,
                               const int max_threads)
{
  // TC path (sm_80+): try largest BC first, then largest BR.
  // For each candidate, compute smem and optimal NW from GPU properties.
  if(cc_major>=g_cu_fa_tc_min_cc_major)
  {
    // Candidate tile sizes in priority order (largest BC first)
    constexpr int br_opts[]={g_cu_fa_fwd_tc_br_large,
                             g_cu_fa_fwd_tc_br_small,
                             g_cu_fa_fwd_tc_br_large,
                             g_cu_fa_fwd_tc_br_small};
    constexpr int bc_opts[]={g_cu_fa_fwd_tc_bc_large,
                             g_cu_fa_fwd_tc_bc_large,
                             g_cu_fa_fwd_tc_bc_small,
                             g_cu_fa_fwd_tc_bc_small};

    for(int c=0;c<g_cu_fa_fwd_tc_option_count;++c)
    {
      const int br=br_opts[c];
      const int bc=bc_opts[c];
      const size_t smem=fa_tc_smem(D,br,bc);
      if(smem>smem_limit)
      {
        continue;
      }

      const int tiles_s=(br/g_cu_wmma_tile_m)*(bc/g_cu_wmma_tile_n);
      const int tiles_o=(br/g_cu_wmma_tile_m)*(D/g_cu_wmma_tile_n);
      int max_tiles=tiles_s;
      if(tiles_o>tiles_s)
      {
        max_tiles=tiles_o;
      }
      const int blocks_from_smem=static_cast<int>(smem_per_sm/smem);
      const int tiles_m=br/g_cu_wmma_tile_m;
      const int tiles_n_s=bc/g_cu_wmma_tile_n;
      const int tiles_n_o=D/g_cu_wmma_tile_n;
      const int nw=fa_tc_optimal_nw(max_tiles,blocks_from_smem,tiles_m,tiles_n_s,tiles_n_o);

      // Dispatch to the matching (BR,BC) template with computed NW
      if(br==g_cu_fa_fwd_tc_br_large && bc==g_cu_fa_fwd_tc_bc_large)
      {
        dispatch_fa_fwd_tc_nw<T,D,g_cu_fa_fwd_tc_br_large,g_cu_fa_fwd_tc_bc_large>(Q,
                                                                                   K,
                                                                                   V,
                                                                                   O,
                                                                                   L,
                                                                                   batch_heads,
                                                                                   seq_len,
                                                                                   scale,
                                                                                   softcap,
                                                                                   causal,
                                                                                   window,
                                                                                   prefix_lens,
                                                                                   num_heads,
                                                                                   num_kv_heads,
                                                                                   alibi_slopes,
                                                                                   stream,
                                                                                   nw);
        return;
      }
      if(br==g_cu_fa_fwd_tc_br_small && bc==g_cu_fa_fwd_tc_bc_large)
      {
        dispatch_fa_fwd_tc_nw<T,D,g_cu_fa_fwd_tc_br_small,g_cu_fa_fwd_tc_bc_large>(Q,
                                                                                   K,
                                                                                   V,
                                                                                   O,
                                                                                   L,
                                                                                   batch_heads,
                                                                                   seq_len,
                                                                                   scale,
                                                                                   softcap,
                                                                                   causal,
                                                                                   window,
                                                                                   prefix_lens,
                                                                                   num_heads,
                                                                                   num_kv_heads,
                                                                                   alibi_slopes,
                                                                                   stream,
                                                                                   nw);
        return;
      }
      if(br==g_cu_fa_fwd_tc_br_large && bc==g_cu_fa_fwd_tc_bc_small)
      {
        dispatch_fa_fwd_tc_nw<T,D,g_cu_fa_fwd_tc_br_large,g_cu_fa_fwd_tc_bc_small>(Q,
                                                                                   K,
                                                                                   V,
                                                                                   O,
                                                                                   L,
                                                                                   batch_heads,
                                                                                   seq_len,
                                                                                   scale,
                                                                                   softcap,
                                                                                   causal,
                                                                                   window,
                                                                                   prefix_lens,
                                                                                   num_heads,
                                                                                   num_kv_heads,
                                                                                   alibi_slopes,
                                                                                   stream,
                                                                                   nw);
        return;
      }
      if(br==g_cu_fa_fwd_tc_br_small && bc==g_cu_fa_fwd_tc_bc_small)
      {
        dispatch_fa_fwd_tc_nw<T,D,g_cu_fa_fwd_tc_br_small,g_cu_fa_fwd_tc_bc_small>(Q,
                                                                                   K,
                                                                                   V,
                                                                                   O,
                                                                                   L,
                                                                                   batch_heads,
                                                                                   seq_len,
                                                                                   scale,
                                                                                   softcap,
                                                                                   causal,
                                                                                   window,
                                                                                   prefix_lens,
                                                                                   num_heads,
                                                                                   num_kv_heads,
                                                                                   alibi_slopes,
                                                                                   stream,
                                                                                   nw);
        return;
      }
    }
  }

  // Scalar warp-per-row fallback (all architectures)
  if(fa_scalar_smem(D)<=smem_limit && max_threads>=g_cu_fa_scalar_warps_large*g_cu_warp_size)
  {
    launch_fa_fwd<T,D,g_cu_fa_scalar_warps_large>(Q,
                                                  K,
                                                  V,
                                                  O,
                                                  L,
                                                  batch_heads,
                                                  seq_len,
                                                  scale,
                                                  softcap,
                                                  causal,
                                                  window,
                                                  prefix_lens,
                                                  num_heads,
                                                  num_kv_heads,
                                                  alibi_slopes,
                                                  stream);
  }
  else if(fa_scalar_smem(D)<=smem_limit && max_threads>=g_cu_fa_scalar_warps_small*g_cu_warp_size)
  {
    launch_fa_fwd<T,D,g_cu_fa_scalar_warps_small>(Q,
                                                  K,
                                                  V,
                                                  O,
                                                  L,
                                                  batch_heads,
                                                  seq_len,
                                                  scale,
                                                  softcap,
                                                  causal,
                                                  window,
                                                  prefix_lens,
                                                  num_heads,
                                                  num_kv_heads,
                                                  alibi_slopes,
                                                  stream);
  }
  else
  {
    launch_fa_fwd<T,D,g_cu_fa_scalar_warps_min>(Q,
                                                K,
                                                V,
                                                O,
                                                L,
                                                batch_heads,
                                                seq_len,
                                                scale,
                                                softcap,
                                                causal,
                                                window,
                                                prefix_lens,
                                                num_heads,
                                                num_kv_heads,
                                                alibi_slopes,
                                                stream);
  }
}

// Shared dispatch body for forward launchers (causal + prefix variants).
template<typename T>
static void dispatch_flash_fwd(const T *Q,
                               const T *K,
                               const T *V,
                               T *O,
                               float *L,
                               const int batch_heads,
                               const int seq_len,
                               const int head_dim,
                               const float scale,
                               const float softcap,
                               const int causal,
                               const int window,
                               const uint32_t *prefix_lens,
                               const int num_heads,
                               const int num_kv_heads,
                               const float *alibi_slopes,
                               cudaStream_t stream)
{
  int device_id=0;
  cudaGetDevice(&device_id);
  int max_smem_optin=g_cu_default_shared_memory;
  cudaDeviceGetAttribute(&max_smem_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device_id);
  int smem_per_sm_int=g_cu_default_shared_memory;
  cudaDeviceGetAttribute(&smem_per_sm_int,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                         device_id);
  int max_threads=g_cu_max_threads_fallback;
  cudaDeviceGetAttribute(&max_threads,
                         cudaDevAttrMaxThreadsPerBlock,
                         device_id);
  int cc_major=0;
  cudaDeviceGetAttribute(&cc_major,
                         cudaDevAttrComputeCapabilityMajor,
                         device_id);

  const size_t smem_limit=static_cast<size_t>(max_smem_optin);
  const size_t smem_per_sm=static_cast<size_t>(smem_per_sm_int);

  switch(head_dim)
  {
    case g_cu_fa_head_dim_32:
      dispatch_fa_fwd_tc<T,g_cu_fa_head_dim_32>(Q,
                                                K,
                                                V,
                                                O,
                                                L,
                                                batch_heads,
                                                seq_len,
                                                scale,
                                                softcap,
                                                causal,
                                                window,
                                                prefix_lens,
                                                num_heads,
                                                num_kv_heads,
                                                alibi_slopes,
                                                stream,
                                                cc_major,
                                                smem_limit,
                                                smem_per_sm,
                                                max_threads);
      break;
    case g_cu_fa_head_dim_64:
      dispatch_fa_fwd_tc<T,g_cu_fa_head_dim_64>(Q,
                                                K,
                                                V,
                                                O,
                                                L,
                                                batch_heads,
                                                seq_len,
                                                scale,
                                                softcap,
                                                causal,
                                                window,
                                                prefix_lens,
                                                num_heads,
                                                num_kv_heads,
                                                alibi_slopes,
                                                stream,
                                                cc_major,
                                                smem_limit,
                                                smem_per_sm,
                                                max_threads);
      break;
    case g_cu_fa_head_dim_80:
      dispatch_fa_fwd_tc<T,g_cu_fa_head_dim_80>(Q,
                                                K,
                                                V,
                                                O,
                                                L,
                                                batch_heads,
                                                seq_len,
                                                scale,
                                                softcap,
                                                causal,
                                                window,
                                                prefix_lens,
                                                num_heads,
                                                num_kv_heads,
                                                alibi_slopes,
                                                stream,
                                                cc_major,
                                                smem_limit,
                                                smem_per_sm,
                                                max_threads);
      break;
    case g_cu_fa_head_dim_96:
      dispatch_fa_fwd_tc<T,g_cu_fa_head_dim_96>(Q,
                                                K,
                                                V,
                                                O,
                                                L,
                                                batch_heads,
                                                seq_len,
                                                scale,
                                                softcap,
                                                causal,
                                                window,
                                                prefix_lens,
                                                num_heads,
                                                num_kv_heads,
                                                alibi_slopes,
                                                stream,
                                                cc_major,
                                                smem_limit,
                                                smem_per_sm,
                                                max_threads);
      break;
    case g_cu_fa_head_dim_128:
      dispatch_fa_fwd_tc<T,g_cu_fa_head_dim_128>(Q,
                                                 K,
                                                 V,
                                                 O,
                                                 L,
                                                 batch_heads,
                                                 seq_len,
                                                 scale,
                                                 softcap,
                                                 causal,
                                                 window,
                                                 prefix_lens,
                                                 num_heads,
                                                 num_kv_heads,
                                                 alibi_slopes,
                                                 stream,
                                                 cc_major,
                                                 smem_limit,
                                                 smem_per_sm,
                                                 max_threads);
      break;
    default:
      fprintf(stderr,
              "FATAL: flash_attention_forward unsupported head_dim=%d"
              " (supported: 32,64,80,96,128). Use standard attention.\n",
              head_dim);
      abort();
  }
}

// Launch wrappers must have C linkage for header declaration
// (former extern "C" block — C++ linkage used for dtype templates)

// Launch wrapper for FlashAttention forward (causal / no-prefix path).
template<typename T>
void launch_flash_attention_forward(const T *Q,
                                    const T *K,
                                    const T *V,
                                    T *O,
                                    float *L,
                                    const int batch_heads,
                                    const int seq_len,
                                    const int head_dim,
                                    const float scale,
                                    const float softcap,
                                    const int causal,
                                    const int window,
                                    const int num_heads,
                                    const int num_kv_heads,
                                    const float *alibi_slopes,
                                    cudaStream_t stream)
{
  dispatch_flash_fwd<T>(Q,
                        K,
                        V,
                        O,
                        L,
                        batch_heads,
                        seq_len,
                        head_dim,
                        scale,
                        softcap,
                        causal,
                        window,
                        nullptr,
                        num_heads,
                        num_kv_heads,
                        alibi_slopes,
                        stream);
}
template void launch_flash_attention_forward<float>(const float *,
                                                    const float *,
                                                    const float *,
                                                    float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    float,
                                                    float,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    const float *,
                                                    cudaStream_t);
template void launch_flash_attention_forward<__half>(const __half *,
                                                     const __half *,
                                                     const __half *,
                                                     __half *,
                                                     float *,
                                                     int,
                                                     int,
                                                     int,
                                                     float,
                                                     float,
                                                     int,
                                                     int,
                                                     int,
                                                     int,
                                                     const float *,
                                                     cudaStream_t);
template void launch_flash_attention_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            float *,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            float,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            const float *,
                                                            cudaStream_t);

// Launch wrapper for FlashAttention forward with prefix-LM mask.
// prefix_lens: device pointer, length = batch_size (int32 per batch).
// num_heads: used to map blockIdx.x (batch_heads) back to the batch index.
// Allowed iff (k<=q) OR (k<prefix_lens[batch]).
template<typename T>
void launch_flash_attention_forward_prefix(const T *Q,
                                           const T *K,
                                           const T *V,
                                           T *O,
                                           float *L,
                                           const uint32_t *prefix_lens,
                                           const int batch_size,
                                           const int num_heads,
                                           const int num_kv_heads,
                                           const int seq_len,
                                           const int head_dim,
                                           const float scale,
                                           const float softcap,
                                           cudaStream_t stream)
{
  // Prefix-LM always uses causal+prefix masking internally (causal=1).
  dispatch_flash_fwd<T>(Q,
                        K,
                        V,
                        O,
                        L,
                        batch_size*num_heads,
                        seq_len,
                        head_dim,
                        scale,
                        softcap,
                        1,
                        0,
                        prefix_lens,
                        num_heads,
                        num_kv_heads,
                        nullptr,
                        stream);
}
template void launch_flash_attention_forward_prefix<float>(const float *,
                                                           const float *,
                                                           const float *,
                                                           float *,
                                                           float *,
                                                           const uint32_t *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           float,
                                                           cudaStream_t);
template void launch_flash_attention_forward_prefix<__half>(const __half *,
                                                            const __half *,
                                                            const __half *,
                                                            __half *,
                                                            float *,
                                                            const uint32_t *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            float,
                                                            cudaStream_t);
template void launch_flash_attention_forward_prefix<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   __nv_bfloat16 *,
                                                                   float *,
                                                                   const uint32_t *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   float,
                                                                   float,
                                                                   cudaStream_t);

// end of former extern "C" block

//------------------------------------------------------------------------------
// FlashAttention-2 Backward: Precompute Di = dot(dO, O) per row
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_precompute_di_kernel(const T *__restrict__ dO,
                                                     const T *__restrict__ O,
                                                     float *__restrict__ Di,
                                                     const int seq_len)
{
  const int bh=blockIdx.x;
  const int row=blockIdx.y*blockDim.x+threadIdx.x;

  if(row>=seq_len)
  {
    return;
  }

  const T *dO_row=dO+static_cast<size_t>(bh)*seq_len*D+row*D;
  const T *O_row=O+static_cast<size_t>(bh)*seq_len*D+row*D;

  float sum=0.0f;
  for(int d=0;d<D;++d)
  {
    sum+=float(dO_row[d])*float(O_row[d]);
  }
  Di[static_cast<size_t>(bh)*seq_len+row]=sum;
}

//------------------------------------------------------------------------------
// FlashAttention-2 Backward Kernel - dK/dV (optimized)
// Uses precomputed Di, template-sized register arrays, 128 thread blocks
//------------------------------------------------------------------------------

template<typename T,int D>
__global__ void flash_attention_backward_kernel(const T *__restrict__ Q,
                                                const T *__restrict__ K,
                                                const T *__restrict__ V,
                                                const T *__restrict__ dO,
                                                const float *__restrict__ L,
                                                const float *__restrict__ Di,
                                                T *__restrict__ dK,
                                                T *__restrict__ dV,
                                                const int seq_len,
                                                const float scale,
                                                const float softcap,
                                                const int causal,
                                                const int window,
                                                const uint32_t *__restrict__ prefix_lens,
                                                const int num_heads,
                                                const float *__restrict__ alibi_slopes)
{
  const int bh=blockIdx.x;
  const int kv_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }
  // ALiBi linear position bias: per-head slope. The forward's softmax (and L)
  // saw the biased logit, so the recompute below adds the same slope*(k-q).
  float alibi_slope=0.0f;
  if(alibi_slopes!=nullptr)
  {
    alibi_slope=alibi_slopes[bh%num_heads];
  }

  const int kv_row=kv_block_idx*g_cu_fa_bwd_bc+tid;
  const int active=(kv_row<seq_len);

  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *dO_tile=smem+g_cu_fa_bwd_br*D;
  float *L_tile=smem+g_cu_fa_bwd_q_do_tiles*g_cu_fa_bwd_br*D;
  float *Di_tile=smem+g_cu_fa_bwd_q_do_tiles*g_cu_fa_bwd_br*D+g_cu_fa_bwd_br;

  const T *Q_bh=Q+static_cast<size_t>(bh)*seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh)*seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh)*seq_len*D;
  const T *dO_bh=dO+static_cast<size_t>(bh)*seq_len*D;
  const float *L_bh=L+static_cast<size_t>(bh)*seq_len;
  const float *Di_bh=Di+static_cast<size_t>(bh)*seq_len;
  T *dK_bh=dK+static_cast<size_t>(bh)*seq_len*D;
  T *dV_bh=dV+static_cast<size_t>(bh)*seq_len*D;

  // Load K and V rows into registers (sized exactly to D)
  float K_row[D];
  float V_row[D];
  if(active!=0)
  {
    for(int d=0;d<D;++d)
    {
      K_row[d]=float(K_bh[kv_row*D+d]);
      V_row[d]=float(V_bh[kv_row*D+d]);
    }
  }

  float dK_acc[D];
  float dV_acc[D];
  for(int d=0;d<D;++d)
  {
    dK_acc[d]=0.0f;
    dV_acc[d]=0.0f;
  }

  const int num_q_blocks=(seq_len+g_cu_fa_bwd_br-1)/g_cu_fa_bwd_br;

  int start_q_block=0;
  if(causal!=0 && active!=0 && prefix_lens==nullptr)
  {
    start_q_block=kv_row/g_cu_fa_bwd_br;
  }
  // Prefix mode: a KV row with k<pfx is attended to by every Q row, so we
  // can't skip early Q blocks. Iterate all blocks, mask per-pair below.

  for(int q_block=start_q_block;q_block<num_q_blocks;++q_block)
  {
    const int q_start=q_block*g_cu_fa_bwd_br;

    // Cooperatively load Q tile and dO tile (all threads participate)
    for(int i=tid;i<g_cu_fa_bwd_br*D;i+=g_cu_fa_bwd_bc)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=q_start+row;

      if(global_row<seq_len)
      {
        Q_tile[row*D+col]=float(Q_bh[global_row*D+col]);
        dO_tile[row*D+col]=float(dO_bh[global_row*D+col]);
      }
      else
      {
        Q_tile[row*D+col]=0.0f;
        dO_tile[row*D+col]=0.0f;
      }
    }

    // Load L and precomputed Di
    if(tid<g_cu_fa_bwd_br)
    {
      const int global_row=q_start+tid;
      if(global_row<seq_len)
      {
        L_tile[tid]=L_bh[global_row];
        Di_tile[tid]=Di_bh[global_row];
      }
      else
      {
        L_tile[tid]=-INFINITY;
        Di_tile[tid]=0.0f;
      }
    }
    __syncthreads();

    if(active!=0)
    {
      for(int qi=0;qi<g_cu_fa_bwd_br;++qi)
      {
        const int q_row=q_start+qi;

        // Prefix-LM: allowed iff (kv<=q) OR (kv<pfx). Plain causal: kv<=q.
        // pfx==0 when prefix_lens is null, reducing exactly to causal.
        if(causal!=0 && kv_row>q_row && kv_row>=pfx)
        {
          continue;
        }
        // Sliding-window: key more than `window` positions before the query.
        if(window>0 && (q_row-kv_row)>=window)
        {
          continue;
        }

        if(q_row>=seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_tile[qi*D+d]*K_row[d];
        }
        const float S_val=dot*scale;
        // Soft-cap backward: the softmax saw the capped score, and the score
        // gradient picks up the cap's derivative (1 - tanh^2). Uniform branch,
        // no-op when off.
        float S_eff=S_val;
        float softcap_deriv=1.0f;
        if(softcap>0.0f)
        {
          const float t=tanhf(S_val/softcap);
          S_eff=softcap*t;
          softcap_deriv=1.0f-t*t;
        }
        // ALiBi bias on the logit (additive constant: enters the recomputed
        // softmax via P_val, but not the score gradient dS below).
        if(alibi_slope!=0.0f)
        {
          S_eff+=alibi_slope*static_cast<float>(kv_row-q_row);
        }
        const float P_val=expf(fminf(S_eff-L_tile[qi],0.0f));

        for(int d=0;d<D;++d)
        {
          dV_acc[d]+=P_val*dO_tile[qi*D+d];
        }

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_tile[qi*D+d]*V_row[d];
        }

        const float dS=P_val*(dP-Di_tile[qi])*scale*softcap_deriv;

        for(int d=0;d<D;++d)
        {
          dK_acc[d]+=dS*Q_tile[qi*D+d];
        }
      }
    }
    __syncthreads();
  }

  if(active!=0)
  {
    for(int d=0;d<D;++d)
    {
      dK_bh[kv_row*D+d]=T(dK_acc[d]);
      dV_bh[kv_row*D+d]=T(dV_acc[d]);
    }
  }
}

//------------------------------------------------------------------------------
// FlashAttention-2 Backward Kernel - dQ (optimized)
// Uses precomputed Di, template-sized register arrays, 128 thread blocks
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_backward_dq_kernel(const T *__restrict__ Q,
                                                   const T *__restrict__ K,
                                                   const T *__restrict__ V,
                                                   const T *__restrict__ dO,
                                                   const float *__restrict__ L,
                                                   const float *__restrict__ Di,
                                                   T *__restrict__ dQ,
                                                   const int seq_len,
                                                   const float scale,
                                                   const float softcap,
                                                   const int causal,
                                                   const int window,
                                                   const uint32_t *__restrict__ prefix_lens,
                                                   const int num_heads,
                                                   const float *__restrict__ alibi_slopes)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }
  // ALiBi linear position bias: per-head slope. The forward's softmax (and L)
  // saw the biased logit, so the recompute below adds the same slope*(k-q).
  float alibi_slope=0.0f;
  if(alibi_slopes!=nullptr)
  {
    alibi_slope=alibi_slopes[bh%num_heads];
  }

  const int q_row=q_block_idx*g_cu_fa_bwd_dq_br+tid;
  const int active=(q_row<seq_len);

  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_cu_fa_bwd_dq_bc*D;

  const T *Q_bh=Q+static_cast<size_t>(bh)*seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh)*seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh)*seq_len*D;
  const T *dO_bh=dO+static_cast<size_t>(bh)*seq_len*D;
  const float *L_bh=L+static_cast<size_t>(bh)*seq_len;
  const float *Di_bh=Di+static_cast<size_t>(bh)*seq_len;
  T *dQ_bh=dQ+static_cast<size_t>(bh)*seq_len*D;

  // Load Q row and dO row (sized exactly to D)
  float Q_row_reg[D];
  float dO_row[D];
  float L_val=0.0f;
  float Di_val=0.0f;
  if(active!=0)
  {
    for(int d=0;d<D;++d)
    {
      Q_row_reg[d]=float(Q_bh[q_row*D+d]);
      dO_row[d]=float(dO_bh[q_row*D+d]);
    }
    L_val=L_bh[q_row];
    Di_val=Di_bh[q_row];
  }

  float dQ_acc[D];
  for(int d=0;d<D;++d)
  {
    dQ_acc[d]=0.0f;
  }

  int num_kv_blocks=(seq_len+g_cu_fa_bwd_dq_bc-1)/g_cu_fa_bwd_dq_bc;
  if(causal!=0 && active!=0 && prefix_lens==nullptr)
  {
    num_kv_blocks=min(num_kv_blocks,(q_row/g_cu_fa_bwd_dq_bc)+1);
  }

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_cu_fa_bwd_dq_bc;

    // Load K and V tiles cooperatively (all threads participate)
    for(int i=tid;i<g_cu_fa_bwd_dq_bc*D;i+=g_cu_fa_bwd_dq_br)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;

      if(global_row<seq_len)
      {
        K_tile[row*D+col]=float(K_bh[global_row*D+col]);
        V_tile[row*D+col]=float(V_bh[global_row*D+col]);
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    if(active!=0)
    {
      for(int j=0;j<g_cu_fa_bwd_dq_bc;++j)
      {
        const int k_row=kv_start+j;

        if(causal!=0 && k_row>q_row && k_row>=pfx)
        {
          continue;
        }
        // Sliding-window: key more than `window` positions before the query.
        if(window>0 && (q_row-k_row)>=window)
        {
          continue;
        }

        if(k_row>=seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_row_reg[d]*K_tile[j*D+d];
        }
        const float S_val=dot*scale;
        // Soft-cap backward (see dK/dV kernel): capped score for the softmax
        // recompute, (1 - tanh^2) factor into the score gradient.
        float S_eff=S_val;
        float softcap_deriv=1.0f;
        if(softcap>0.0f)
        {
          const float t=tanhf(S_val/softcap);
          S_eff=softcap*t;
          softcap_deriv=1.0f-t*t;
        }
        // ALiBi bias on the logit (additive constant: enters the recomputed
        // softmax via P_val, but not the score gradient dS below).
        if(alibi_slope!=0.0f)
        {
          S_eff+=alibi_slope*static_cast<float>(k_row-q_row);
        }
        const float P_val=expf(fminf(S_eff-L_val,0.0f));

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_row[d]*V_tile[j*D+d];
        }

        const float dS=P_val*(dP-Di_val)*scale*softcap_deriv;

        for(int d=0;d<D;++d)
        {
          dQ_acc[d]+=dS*K_tile[j*D+d];
        }
      }
    }
    __syncthreads();
  }

  if(active!=0)
  {
    for(int d=0;d<D;++d)
    {
      dQ_bh[q_row*D+d]=T(dQ_acc[d]);
    }
  }
}

template<typename T>
static void dispatch_flash_bwd(const T *Q,
                               const T *K,
                               const T *V,
                               const T *O,
                               const T *dO,
                               const float *L,
                               T *dQ,
                               T *dK,
                               T *dV,
                               const int batch_heads,
                               const int seq_len,
                               const int head_dim,
                               const float scale,
                               const float softcap,
                               const int causal,
                               const int window,
                               const uint32_t *prefix_lens,
                               const int num_heads,
                               const float *alibi_slopes,
                               cudaStream_t stream)
{
  // Allocate temporary Di buffer [batch_heads, seq_len]
  float *Di_buf=nullptr;
  cudaMallocAsync(reinterpret_cast<void **>(&Di_buf),
                  static_cast<size_t>(batch_heads)*seq_len*sizeof(float),stream);

  // Kernel 0: Precompute Di = dot(dO, O) for each row
  {
    const int block_size=g_cu_block_size;
    const int rows_per_grid=(seq_len+block_size-1)/block_size;
    dim3 grid(batch_heads,rows_per_grid);

    switch(head_dim)
    {
      case g_cu_fa_head_dim_32:
        flash_attention_precompute_di_kernel<T,g_cu_fa_head_dim_32>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case g_cu_fa_head_dim_64:
        flash_attention_precompute_di_kernel<T,g_cu_fa_head_dim_64>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case g_cu_fa_head_dim_80:
        flash_attention_precompute_di_kernel<T,g_cu_fa_head_dim_80>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case g_cu_fa_head_dim_96:
        flash_attention_precompute_di_kernel<T,g_cu_fa_head_dim_96>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case g_cu_fa_head_dim_128:
        flash_attention_precompute_di_kernel<T,g_cu_fa_head_dim_128>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 1: Compute dK and dV (128 threads/block)
  {
    const int num_kv_blocks=(seq_len+g_cu_fa_bwd_bc-1)/g_cu_fa_bwd_bc;
    dim3 grid(batch_heads,num_kv_blocks);
    dim3 block(g_cu_fa_bwd_bc);
    const size_t smem_size=(g_cu_fa_bwd_q_do_tiles*
                            g_cu_fa_bwd_br*
                            head_dim+
                            g_cu_fa_smem_stat_arrays*
                            g_cu_fa_bwd_br)*
                            sizeof(float);

    switch(head_dim)
    {
      case g_cu_fa_head_dim_32:
        flash_attention_backward_kernel<T,g_cu_fa_head_dim_32>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_64:
        flash_attention_backward_kernel<T,g_cu_fa_head_dim_64>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_80:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<T,g_cu_fa_head_dim_80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<T,g_cu_fa_head_dim_80>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_96:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<T,g_cu_fa_head_dim_96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<T,g_cu_fa_head_dim_96>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_128:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<T,g_cu_fa_head_dim_128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<T,g_cu_fa_head_dim_128>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 2: Compute dQ (128 threads/block)
  {
    const int num_q_blocks=(seq_len+g_cu_fa_bwd_dq_br-1)/g_cu_fa_bwd_dq_br;
    dim3 grid(batch_heads,num_q_blocks);
    dim3 block(g_cu_fa_bwd_dq_br);
    const size_t smem_size=g_cu_fa_kv_tiles*g_cu_fa_bwd_dq_bc*head_dim*sizeof(float);

    switch(head_dim)
    {
      case g_cu_fa_head_dim_32:
        flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_32>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_64:
        flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_64>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_80:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_80>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_96:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_96>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      case g_cu_fa_head_dim_128:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<T,g_cu_fa_head_dim_128>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            softcap,
                                            causal,
                                            window,
                                            prefix_lens,
                                            num_heads,
                                            alibi_slopes);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_dq unsupported head_dim=%d\n",
                head_dim);
        abort();
        break;
    }
  }

  // Free temporary Di buffer (enqueue on stream for proper ordering)
  cudaFreeAsync(Di_buf,stream);
}

// (former extern "C" block — C++ linkage used for dtype templates)

// Launch wrapper for FlashAttention backward (causal / non-causal)
template<typename T>
void launch_flash_attention_backward(const T *Q,
                                     const T *K,
                                     const T *V,
                                     const T *O,
                                     const T *dO,
                                     const float *L,
                                     T *dQ,
                                     T *dK,
                                     T *dV,
                                     const int batch_heads,
                                     const int seq_len,
                                     const int head_dim,
                                     const float scale,
                                     const float softcap,
                                     const int causal,
                                     const int window,
                                     const int num_heads,
                                     const float *alibi_slopes,
                                     cudaStream_t stream)
{
  dispatch_flash_bwd<T>(Q,
                        K,
                        V,
                        O,
                        dO,
                        L,
                        dQ,
                        dK,
                        dV,
                        batch_heads,
                        seq_len,
                        head_dim,
                        scale,
                        softcap,
                        causal,
                        window,
                        nullptr,
                        num_heads,
                        alibi_slopes,
                        stream);
}
template void launch_flash_attention_backward<float>(const float *,
                                                     const float *,
                                                     const float *,
                                                     const float *,
                                                     const float *,
                                                     const float *,
                                                     float *,
                                                     float *,
                                                     float *,
                                                     int,
                                                     int,
                                                     int,
                                                     float,
                                                     float,
                                                     int,
                                                     int,
                                                     int,
                                                     const float *,
                                                     cudaStream_t);
template void launch_flash_attention_backward<__half>(const __half *,
                                                      const __half *,
                                                      const __half *,
                                                      const __half *,
                                                      const __half *,
                                                      const float *,
                                                      __half *,
                                                      __half *,
                                                      __half *,
                                                      int,
                                                      int,
                                                      int,
                                                      float,
                                                      float,
                                                      int,
                                                      int,
                                                      int,
                                                      const float *,
                                                      cudaStream_t);
template void launch_flash_attention_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const float *,
                                                             __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             int,
                                                             int,
                                                             int,
                                                             float,
                                                             float,
                                                             int,
                                                             int,
                                                             int,
                                                             const float *,
                                                             cudaStream_t);

// Launch wrapper for FlashAttention backward with prefix-LM mask
template<typename T>
void launch_flash_attention_backward_prefix(const T *Q,
                                            const T *K,
                                            const T *V,
                                            const T *O,
                                            const T *dO,
                                            const float *L,
                                            T *dQ,
                                            T *dK,
                                            T *dV,
                                            const uint32_t *prefix_lens,
                                            const int batch_size,
                                            const int num_heads,
                                            const int seq_len,
                                            const int head_dim,
                                            const float scale,
                                            const float softcap,
                                            cudaStream_t stream)
{
  dispatch_flash_bwd<T>(Q,
                        K,
                        V,
                        O,
                        dO,
                        L,
                        dQ,
                        dK,
                        dV,
                        batch_size*num_heads,
                        seq_len,
                        head_dim,
                        scale,
                        softcap,
                        1,
                        0,
                        prefix_lens,
                        num_heads,
                        nullptr,
                        stream);
}
template void launch_flash_attention_backward_prefix<float>(const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            float *,
                                                            float *,
                                                            float *,
                                                            const uint32_t *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            float,
                                                            cudaStream_t);
template void launch_flash_attention_backward_prefix<__half>(const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const float *,
                                                             __half *,
                                                             __half *,
                                                             __half *,
                                                             const uint32_t *,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             float,
                                                             float,
                                                             cudaStream_t);
template void launch_flash_attention_backward_prefix<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const float *,
                                                                    __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    const uint32_t *,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    float,
                                                                    float,
                                                                    cudaStream_t);

// end of former extern "C" block (flash attention backward)
