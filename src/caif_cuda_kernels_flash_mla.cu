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
// CAIF - C++ AI Framework
// Fused tensor-core FlashAttention forward for MLA prefill (declarations in
// caif_cuda_kernels_flash_mla.cuh).
//
// Adapted from the self-attention forward (caif_cuda_kernels_flash_self) with
// two changes; the online softmax, cp.async pipeline, and register fragment
// layout (incl. the mask / cross-warp reduce, which are dimension-independent)
// are reused unchanged:
//   * Q/K (scores) head dim D_QK is decoupled from V/output head dim D_V
//     (DeepSeek-V2-Lite is (192, 128)). KV_buf holds K (D_QK wide) then V (D_V
//     wide); since D_QK >= D_V its row stride is d_qk_pad and V fits inside it,
//     so fa_tc_smem(D_QK,BR,BC) is the exact smem. O_smem aliases Q_tile (D_V <=
//     D_QK by construction). Only tiles_n_o, the V-load width, and the O-store
//     stride track D_V.
//   * q_len / kv_len / q_offset are first-class — Q/O index q_len, K/V index
//     kv_len, and the causal mask compares key bc against q_offset + global_q,
//     so chunked prefill into a warm KV cache works.
//
// Forward inference only — no logsumexp output, no prefix-LM mask, and no GQA
// remap (MLA decompresses K/V per head, so num_kv_heads == num_heads).
// Supported configs: (D_QK,D_V) in {(128,128), (192,128)} on the (32,64) tile;
// per-device tile selection over the full fitting set lands in Step 3.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

#include "caif_cuda_kernels_flash_mla.cuh"
#include "caif_cuda_kernels_common.cuh"
#include "caif_cuda_kernels_flash_common.cuh"

//------------------------------------------------------------------------------
// Fused MLA flash-prefill forward kernel.
// One block per (batch_head, q_block). O accumulator and S scores live in wmma
// register fragments; softmax is computed in-register with a cross-warp reduce
// via S_tile[0..NW*16].
//------------------------------------------------------------------------------
template<typename T,int D_QK,int D_V,int BR,int BC,int NW>
__global__ void flash_attention_forward_mla_tc_kernel(const T *__restrict__ Q,
                                                      const T *__restrict__ K,
                                                      const T *__restrict__ V,
                                                      T *__restrict__ O,
                                                      const int q_len,
                                                      const int kv_len,
                                                      const int q_offset,
                                                      const float scale,
                                                      const int causal)
{
#if CAIF_HAS_TC_FLASH
  // O_smem aliases Q_tile, so the output (D_V wide) must fit the Q tile (D_QK
  // wide); this also makes max(D_QK,D_V) == D_QK, so KV_buf strides on d_qk_pad.
  // Invariant D_QK >= D_V holds by construction: the only instantiated configs
  // are (128,128) and (192,128), both gated by mla_flash_prefill_available.
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int lane_id=tid%g_cu_warp_size;
  const int warp_id=tid/g_cu_warp_size;

  constexpr int n_warps=NW;
  constexpr int tiles_m=BR/g_cu_wmma_tile_m;
  constexpr int tiles_n_s=BC/g_cu_wmma_tile_n;
  constexpr int tiles_n_o=D_V/g_cu_wmma_tile_n;
  constexpr int block_threads=n_warps*g_cu_warp_size;

  // Smem stride padding: +2 eliminates bank conflicts for wmma loads. KV_buf
  // and Q_tile stride on d_qk_pad; O_smem (alias of Q_tile) strides on d_v_pad.
  constexpr int d_qk_pad=D_QK+g_cu_fa_smem_pad;
  constexpr int d_v_pad=D_V+g_cu_fa_smem_pad;
  constexpr int bc_pad=BC+g_cu_fa_smem_pad;
  constexpr int d_qk_f2=D_QK/g_cu_float2_lanes;
  constexpr int d_v_f2=D_V/g_cu_float2_lanes;
  constexpr int d_qk_pad_f2=d_qk_pad/g_cu_float2_lanes;

  // Shared memory layout (O is in registers, not smem)
  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *KV_buf=Q_tile+BR*d_qk_pad;
  float *S_tile=KV_buf+BC*d_qk_pad;
  float *row_max_arr=S_tile+BR*bc_pad;
  float *row_sum_arr=row_max_arr+BR;

  // Q/O follow q_len at width D_QK / D_V; K/V follow kv_len at width D_QK / D_V.
  // No GQA remap: MLA decompresses K/V per head, so num_kv_heads == num_heads.
  const T *Q_bh=Q+static_cast<size_t>(bh)*q_len*D_QK;
  const T *K_bh=K+static_cast<size_t>(bh)*kv_len*D_QK;
  const T *V_bh=V+static_cast<size_t>(bh)*kv_len*D_V;
  T *O_bh=O+static_cast<size_t>(bh)*q_len*D_V;

  const int q_start=q_block_idx*BR;

  // Warp grouping for register-based softmax: each M-tile group's warps
  // collectively cover all N S-tiles, enabling in-register softmax with a
  // cross-warp reduce (no S_tile smem round-trip).
  // Tile-validity invariant (guaranteed by construction — only warp-group-valid
  // tiles are listed in the g_cu_fa_mla_tile_* table, and correctness is gated
  // by the runtime parity test): the warp grouping splits NW warps into tiles_m
  // groups of warps_per_m warps, and each group must cover the S and O n-tiles
  // exactly. So NW must be a multiple of BR/16 and warps_per_m must divide both
  // BC/16 and D_V/16. NW=4 with BR=48 (tiles_m=3) would truncate warps_per_m and
  // read past smem, so BR=48 is excluded from the table.
  constexpr int warps_per_m=n_warps/tiles_m;
  constexpr int s_tiles_pw=(tiles_n_s>=warps_per_m)*(tiles_n_s/warps_per_m);
  constexpr int o_tiles_pw=(tiles_n_o>=warps_per_m)*(tiles_n_o/warps_per_m);
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

  // Cooperative load Q_tile[BR, d_qk_pad] from global memory (padded stride)
  const int valid_q_rows=min(BR,q_len-q_start);
  {
    if constexpr(sizeof(T)==sizeof(float))
    {
      const int valid_q_f2=max(valid_q_rows,0)*d_qk_f2;
      constexpr int total_q_f2=BR*d_qk_f2;
      const float2 *Q_src2=reinterpret_cast<const float2 *>(Q_bh+q_start*D_QK);
      float2 *Q_dst2=reinterpret_cast<float2 *>(Q_tile);
      for(int i=tid;i<valid_q_f2;i+=block_threads)
      {
        const int row=i/d_qk_f2;
        const int f2c=i-row*d_qk_f2;
        Q_dst2[row*d_qk_pad_f2+f2c]=Q_src2[i];
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=valid_q_f2+tid;i<total_q_f2;i+=block_threads)
      {
        const int row=i/d_qk_f2;
        const int f2c=i-row*d_qk_f2;
        Q_dst2[row*d_qk_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int valid_q=max(valid_q_rows,0)*D_QK;
      constexpr int total_q=BR*D_QK;
      for(int i=tid;i<valid_q;i+=block_threads)
      {
        const int row=i/D_QK;
        const int col=i-row*D_QK;
        Q_tile[row*d_qk_pad+col]=float(Q_bh[(q_start+row)*D_QK+col]);
      }
      for(int i=valid_q+tid;i<total_q;i+=block_threads)
      {
        const int row=i/D_QK;
        const int col=i-row*D_QK;
        Q_tile[row*d_qk_pad+col]=0.0f;
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

  // Number of KV blocks. The causal limit uses the ABSOLUTE last query position
  // q_offset + q_start + BR - 1 (capped at kv_len-1), so key blocks beyond what
  // any query in this row block can attend are skipped.
  int num_kv_blocks=(kv_len+BC-1)/BC;
  if(causal==1)
  {
    int max_q=q_offset+q_start+BR-1;
    if(max_q>=kv_len)
    {
      max_q=kv_len-1;
    }
    num_kv_blocks=min(num_kv_blocks,(max_q/BC)+1);
  }

  constexpr int k_f2=BC*d_qk_f2;
  constexpr int v_f2=BC*d_v_f2;

  // Pipeline: prefetch K[0] (D_QK wide) into KV_buf
  float2 *KV_dst2=reinterpret_cast<float2 *>(KV_buf);
  if(num_kv_blocks>0)
  {
    if constexpr(sizeof(T)==sizeof(float))
    {
      const int kv0_valid=min(BC,kv_len)*d_qk_f2;
      const float2 *K0_src2=reinterpret_cast<const float2 *>(K_bh);
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/d_qk_f2;
        const int f2c=i-row*d_qk_f2;
        cp_async_f2(&KV_dst2[row*d_qk_pad_f2+f2c],&K0_src2[i]);
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=kv0_valid+tid;i<k_f2;i+=block_threads)
      {
        const int row=i/d_qk_f2;
        const int f2c=i-row*d_qk_f2;
        KV_dst2[row*d_qk_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int kv0_valid=min(BC,kv_len)*D_QK;
      constexpr int k_total=BC*D_QK;
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/D_QK;
        const int col=i-row*D_QK;
        KV_buf[row*d_qk_pad+col]=float(K_bh[row*D_QK+col]);
      }
      for(int i=kv0_valid+tid;i<k_total;i+=block_threads)
      {
        const int row=i/D_QK;
        const int col=i-row*D_QK;
        KV_buf[row*d_qk_pad+col]=0.0f;
      }
    }
    cp_async_commit();
  }

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*BC;
    const int valid_kv_rows=min(BC,kv_len-kv_start);

    // PHASE 1: Wait for K, compute S = Q @ K^T in wmma registers (contract D_QK)
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
      for(int k=0;k<D_QK/g_cu_wmma_tile_k;++k)
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
        nvcuda::wmma::load_matrix_sync(q_frag,
                                       &Q_tile[m_idx*g_cu_wmma_tile_m*d_qk_pad+k*g_cu_wmma_tile_k],
                                       d_qk_pad);
        nvcuda::wmma::load_matrix_sync(k_frag,&KV_buf[n*g_cu_wmma_tile_n*d_qk_pad+k*g_cu_wmma_tile_k],d_qk_pad);
        nvcuda::wmma::mma_sync(s_accs[t],q_frag,k_frag,s_accs[t]);
      }
      for(int i=0;i<s_accs[t].num_elements;++i)
      {
        s_accs[t].x[i]*=scale;
      }
    }

    // Sync: all warps done reading KV_buf before V overwrites it
    __syncthreads();

    // Async V load (D_V wide) into KV_buf (overlapped with softmax). V data
    // occupies cols [0, D_V) of each d_qk_pad-strided row; the stale K cols
    // [D_V, D_QK) are never read by v_frag. Padding rows are zeroed so the
    // P @ V tiles never read garbage (0 * NaN would poison the output).
    {
      if constexpr(sizeof(T)==sizeof(float))
      {
        const int valid_v_f2=valid_kv_rows*d_v_f2;
        const float2 *V_src2=reinterpret_cast<const float2 *>(V_bh+kv_start*D_V);
        for(int i=tid;i<valid_v_f2;i+=block_threads)
        {
          const int row=i/d_v_f2;
          const int f2c=i-row*d_v_f2;
          cp_async_f2(&KV_dst2[row*d_qk_pad_f2+f2c],&V_src2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=valid_v_f2+tid;i<v_f2;i+=block_threads)
        {
          const int row=i/d_v_f2;
          const int f2c=i-row*d_v_f2;
          KV_dst2[row*d_qk_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int valid_v=valid_kv_rows*D_V;
        constexpr int v_total=BC*D_V;
        for(int i=tid;i<valid_v;i+=block_threads)
        {
          const int row=i/D_V;
          const int col=i-row*D_V;
          KV_buf[row*d_qk_pad+col]=float(V_bh[(kv_start+row)*D_V+col]);
        }
        for(int i=valid_v+tid;i<v_total;i+=block_threads)
        {
          const int row=i/D_V;
          const int col=i-row*D_V;
          KV_buf[row*d_qk_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }

    // PHASE 2: Register-based online softmax on s_accs. The mask and reduce are
    // dimension-independent (act on the [BR, BC] score fragments), reused as-is.
    // Fragment layout (stable sm_80-sm_120):
    //   elements {0,1,4,5} → local_row = lane_id/4       (row_lo)
    //   elements {2,3,6,7} → local_row = lane_id/4 + 8   (row_hi)
    //   elem col: 0→(lane_id%g_cu_wmma_acc_lanes_per_row)*2, 1→+1, 4→+8, 5→+9 (same for 2/3/6/7)
    {
      const int row_lo=m_idx*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row);
      const int row_hi=m_idx*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row)+g_cu_wmma_acc_half_tile;
      const int global_q_lo=q_start+row_lo;
      const int global_q_hi=q_start+row_hi;

      // Apply offset-causal + boundary masks in registers. Offset-causal: the
      // query at absolute position (q_offset + global_q) attends key bc iff
      // bc <= q_offset + global_q. q_offset=0 is whole-prompt prefill;
      // q_offset=cache_len is chunked prefill into a warm KV cache.
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

        if(causal==1)
        {
          if(bc0>q_offset+global_q_lo)
          {
            s_accs[t].x[0]=-INFINITY;
          }
          if(bc1>q_offset+global_q_lo)
          {
            s_accs[t].x[1]=-INFINITY;
          }
          if(bc0>q_offset+global_q_hi)
          {
            s_accs[t].x[2]=-INFINITY;
          }
          if(bc1>q_offset+global_q_hi)
          {
            s_accs[t].x[3]=-INFINITY;
          }
          if(bc2>q_offset+global_q_lo)
          {
            s_accs[t].x[4]=-INFINITY;
          }
          if(bc3>q_offset+global_q_lo)
          {
            s_accs[t].x[5]=-INFINITY;
          }
          if(bc2>q_offset+global_q_hi)
          {
            s_accs[t].x[6]=-INFINITY;
          }
          if(bc3>q_offset+global_q_hi)
          {
            s_accs[t].x[7]=-INFINITY;
          }
        }
        // K boundary mask: zero-padded K positions beyond kv_len must be masked
        // to -inf so they do not participate in softmax. The causal mask catches
        // these implicitly; the non-causal path does not.
        if(bc0>=kv_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[2]=-INFINITY;
        }
        if(bc1>=kv_len)
        {
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
        }
        if(bc2>=kv_len)
        {
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
        }
        if(bc3>=kv_len)
        {
          s_accs[t].x[5]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }
        if(global_q_lo>=q_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[5]=-INFINITY;
        }
        if(global_q_hi>=q_len)
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

      // Rescale O fragments by correction (all O tiles share this warp's m_idx).
      // Element layout is the accumulator's; only the o_tiles_pw count tracks D_V.
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

    // Barrier: S_tile doubles as the cross-warp sum/max reduce buffer above;
    // without this sync a fast warp can begin the wmma store below and overwrite
    // reduce_buf slots a slower warp in the same m_idx group is still reading.
    // Latent at NW=2/4, reliably corrupts output at NW=8.
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

    // PHASE 3: Accumulate O += softmax(S) @ V using tensor cores. Contraction is
    // over BC; the output spans D_V (tiles_n_o tiles). V is read from KV_buf at
    // d_qk_pad stride, cols [0, D_V).
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
        nvcuda::wmma::load_matrix_sync(v_frag,&KV_buf[k*g_cu_wmma_tile_k*d_qk_pad+n*g_cu_wmma_tile_n],d_qk_pad);
        nvcuda::wmma::mma_sync(o_frags[t],s_frag,v_frag,o_frags[t]);
      }
    }
    __syncthreads();

    // Pipeline: prefetch K[next] (D_QK wide) into KV_buf
    if(kv_block+1<num_kv_blocks)
    {
      const int next_start=(kv_block+1)*BC;
      if constexpr(sizeof(T)==sizeof(float))
      {
        const int next_valid=min(BC,kv_len-next_start)*d_qk_f2;
        const float2 *K_next2=reinterpret_cast<const float2 *>(K_bh+next_start*D_QK);
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/d_qk_f2;
          const int f2c=i-row*d_qk_f2;
          cp_async_f2(&KV_dst2[row*d_qk_pad_f2+f2c],&K_next2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=next_valid+tid;i<k_f2;i+=block_threads)
        {
          const int row=i/d_qk_f2;
          const int f2c=i-row*d_qk_f2;
          KV_dst2[row*d_qk_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int next_valid=min(BC,kv_len-next_start)*D_QK;
        constexpr int k_total=BC*D_QK;
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/D_QK;
          const int col=i-row*D_QK;
          KV_buf[row*d_qk_pad+col]=float(K_bh[(next_start+row)*D_QK+col]);
        }
        for(int i=next_valid+tid;i<k_total;i+=block_threads)
        {
          const int row=i/D_QK;
          const int col=i-row*D_QK;
          KV_buf[row*d_qk_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }
  }

  // Final: store O fragments to smem (alias Q_tile, d_v_pad stride), normalize,
  // write. D_V <= D_QK so the D_V-wide O_smem fits the D_QK-wide Q_tile.
  __syncthreads();
  float *O_smem=Q_tile;
  for(int t=0;t<o_tiles_pw;++t)
  {
    const int n=n_start_o+t;
    nvcuda::wmma::store_matrix_sync(&O_smem[m_idx*g_cu_wmma_tile_m*d_v_pad+n*g_cu_wmma_tile_n],
                                    o_frags[t],
                                    d_v_pad,
                                    nvcuda::wmma::mem_row_major);
  }
  __syncthreads();

  for(int i=tid;i<BR*D_V;i+=block_threads)
  {
    const int row=i/D_V;
    const int col=i-row*D_V;
    const int global_row=q_start+row;
    if(global_row<q_len)
    {
      float inv_l=0.0f;
      if(row_sum_arr[row]>0.0f)
      {
        inv_l=1.0f/row_sum_arr[row];
      }
      O_bh[global_row*D_V+col]=T(O_smem[row*d_v_pad+col]*inv_l);
    }
  }
#endif
}

//------------------------------------------------------------------------------
// Per-config launch helper. Returns false (never throws — nvcc TU) when the
// launch cannot run; the .cpp caller turns that into a THROW_CAIFE after a true
// availability check.
//------------------------------------------------------------------------------
template<typename T,int D_QK,int D_V,int BR,int BC,int NW>
bool launch_mla_config(const T *q,
                       const T *k,
                       const T *v,
                       T *out,
                       const int batch_heads,
                       const int q_len,
                       const int kv_len,
                       const int q_offset,
                       const float scale,
                       const int causal,
                       cudaStream_t stream)
{
  constexpr size_t smem_size=fa_tc_smem(D_QK,BR,BC);

  int device_id=0;
  cudaGetDevice(&device_id);
  int optin_smem=0;
  cudaDeviceGetAttribute(&optin_smem,cudaDevAttrMaxSharedMemoryPerBlockOptin,device_id);
  if(smem_size>static_cast<size_t>(optin_smem))
  {
    printf("flash_mla: tile smem %zu B exceeds device %d opt-in %d B\n",
           smem_size,device_id,optin_smem);
    return false;
  }

  if(smem_size>g_cu_default_shared_memory)
  {
    cudaError_t attr_status=cudaFuncSetAttribute(
                              (void *)flash_attention_forward_mla_tc_kernel<T,D_QK,D_V,BR,BC,NW>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem_size));
    if(attr_status!=cudaSuccess)
    {
      printf("flash_mla: cudaFuncSetAttribute(%zu B) failed on device %d: %s\n",
             smem_size,device_id,cudaGetErrorString(attr_status));
      return false;
    }
  }

  const int num_q_blocks=(q_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(NW*g_cu_warp_size);
  flash_attention_forward_mla_tc_kernel<T,D_QK,D_V,BR,BC,NW>
    <<<grid,block,smem_size,stream>>>(q,
                                      k,
                                      v,
                                      out,
                                      q_len,
                                      kv_len,
                                      q_offset,
                                      scale,
                                      causal);
  cudaError_t launch_status=cudaGetLastError();
  if(launch_status!=cudaSuccess)
  {
    printf("flash_mla: kernel launch failed on device %d: %s\n",
           device_id,cudaGetErrorString(launch_status));
    return false;
  }
  return true;
}

//------------------------------------------------------------------------------
// Compile-time tile dispatch. Runs candidate tile_index (a row of the
// g_cu_fa_mla_tile_* table) through launch_mla_config, whose own smem-fit +
// cudaFuncSetAttribute + cudaGetLastError checks gate the launch. Returns false
// for an out-of-range index. NW is shared across tiles (g_cu_fa_mla_nw).
//------------------------------------------------------------------------------
template<typename T,int D_QK,int D_V>
bool launch_mla_indexed(const T *q,
                        const T *k,
                        const T *v,
                        T *out,
                        const int batch_heads,
                        const int q_len,
                        const int kv_len,
                        const int q_offset,
                        const float scale,
                        const int causal,
                        const int tile_index,
                        cudaStream_t stream)
{
  constexpr int NW=g_cu_fa_mla_nw;
  if(tile_index==0)
  {
    constexpr int BR=g_cu_fa_mla_tile_br[0];
    constexpr int BC=g_cu_fa_mla_tile_bc[0];
    return launch_mla_config<T,D_QK,D_V,BR,BC,NW>(q,
                                                  k,
                                                  v,
                                                  out,
                                                  batch_heads,
                                                  q_len,
                                                  kv_len,
                                                  q_offset,
                                                  scale,
                                                  causal,
                                                  stream);
  }
  if(tile_index==1)
  {
    constexpr int BR=g_cu_fa_mla_tile_br[1];
    constexpr int BC=g_cu_fa_mla_tile_bc[1];
    return launch_mla_config<T,D_QK,D_V,BR,BC,NW>(q,
                                                  k,
                                                  v,
                                                  out,
                                                  batch_heads,
                                                  q_len,
                                                  kv_len,
                                                  q_offset,
                                                  scale,
                                                  causal,
                                                  stream);
  }
  return false;
}

//------------------------------------------------------------------------------
// Per-device tile selection. Walks the candidate table in priority order and
// dispatches the first tile whose decoupled smem fits the launch device's
// per-block opt-in. Returns false if no candidate fits (the caller's predicate
// gates this out up front, so a false here is a genuine device-mismatch).
//------------------------------------------------------------------------------
template<typename T,int D_QK,int D_V>
bool launch_mla_select_tile(const T *q,
                            const T *k,
                            const T *v,
                            T *out,
                            const int batch_heads,
                            const int q_len,
                            const int kv_len,
                            const int q_offset,
                            const float scale,
                            const int causal,
                            const int optin_smem,
                            cudaStream_t stream)
{
  for(int c=0;c<g_cu_fa_mla_tile_count;++c)
  {
    const size_t smem=fa_tc_smem(D_QK,g_cu_fa_mla_tile_br[c],g_cu_fa_mla_tile_bc[c]);
    if(smem>static_cast<size_t>(optin_smem))
    {
      continue;
    }
    return launch_mla_indexed<T,D_QK,D_V>(q,k,v,out,batch_heads,q_len,kv_len,q_offset,scale,causal,c,stream);
  }
  return false;
}

//------------------------------------------------------------------------------
// Public launcher. Queries the launch device's per-block opt-in smem once, then
// dispatches the supported (D_QK, D_V) config to the best fitting tile (P3:
// per-device, never a build-time constant).
//------------------------------------------------------------------------------
template<typename T>
bool launch_flash_attention_forward_mla(const T *q,
                                        const T *k,
                                        const T *v_heads,
                                        T *out,
                                        int batch_heads,
                                        int q_len,
                                        int kv_len,
                                        int qk_dim,
                                        int v_dim,
                                        float scale,
                                        int causal,
                                        int q_offset,
                                        cudaStream_t stream)
{
  constexpr int DV=g_cu_fa_mla_v_dim;
  if(v_dim!=DV)
  {
    return false;
  }
  int device_id=0;
  cudaGetDevice(&device_id);
  int optin_smem=0;
  cudaDeviceGetAttribute(&optin_smem,cudaDevAttrMaxSharedMemoryPerBlockOptin,device_id);
  if(qk_dim==g_cu_fa_mla_qk_dim_identity)
  {
    return launch_mla_select_tile<T,g_cu_fa_mla_qk_dim_identity,DV>(q,
                                                                    k,
                                                                    v_heads,
                                                                    out,
                                                                    batch_heads,
                                                                    q_len,
                                                                    kv_len,
                                                                    q_offset,
                                                                    scale,
                                                                    causal,
                                                                    optin_smem,
                                                                    stream);
  }
  if(qk_dim==g_cu_fa_mla_qk_dim_decoupled)
  {
    return launch_mla_select_tile<T,g_cu_fa_mla_qk_dim_decoupled,DV>(q,
                                                                     k,
                                                                     v_heads,
                                                                     out,
                                                                     batch_heads,
                                                                     q_len,
                                                                     kv_len,
                                                                     q_offset,
                                                                     scale,
                                                                     causal,
                                                                     optin_smem,
                                                                     stream);
  }
  return false;
}

//------------------------------------------------------------------------------
// Diagnostic launcher. Forces a specific candidate tile_index regardless of
// priority order (per-arch tile ranking / the Step-3 sweep); the tile's own
// smem-fit check still gates the launch. Production uses the auto-selecting
// launcher above; this exists so the tile order can be re-ranked on a new arch.
//------------------------------------------------------------------------------
template<typename T>
bool launch_flash_attention_forward_mla_tile(const T *q,
                                             const T *k,
                                             const T *v_heads,
                                             T *out,
                                             int batch_heads,
                                             int q_len,
                                             int kv_len,
                                             int qk_dim,
                                             int v_dim,
                                             float scale,
                                             int causal,
                                             int q_offset,
                                             int tile_index,
                                             cudaStream_t stream)
{
  constexpr int DV=g_cu_fa_mla_v_dim;
  if(v_dim!=DV)
  {
    return false;
  }
  if(tile_index<0 || tile_index>=g_cu_fa_mla_tile_count)
  {
    return false;
  }
  if(qk_dim==g_cu_fa_mla_qk_dim_identity)
  {
    return launch_mla_indexed<T,g_cu_fa_mla_qk_dim_identity,DV>(q,
                                                                k,
                                                                v_heads,
                                                                out,
                                                                batch_heads,
                                                                q_len,
                                                                kv_len,
                                                                q_offset,
                                                                scale,
                                                                causal,
                                                                tile_index,
                                                                stream);
  }
  if(qk_dim==g_cu_fa_mla_qk_dim_decoupled)
  {
    return launch_mla_indexed<T,g_cu_fa_mla_qk_dim_decoupled,DV>(q,
                                                                 k,
                                                                 v_heads,
                                                                 out,
                                                                 batch_heads,
                                                                 q_len,
                                                                 kv_len,
                                                                 q_offset,
                                                                 scale,
                                                                 causal,
                                                                 tile_index,
                                                                 stream);
  }
  return false;
}

//------------------------------------------------------------------------------
// Availability predicate (host). True iff a fused instantiation exists for
// (qk_dim, v_dim) and device_id can run it: cc-major >= 8 and at least one
// candidate tile fits the device's opt-in smem. No side effects, never aborts.
//------------------------------------------------------------------------------
bool mla_flash_prefill_available(int qk_dim,int v_dim,int device_id)
{
  if(v_dim!=g_cu_fa_mla_v_dim)
  {
    return false;
  }
  const bool qk_ok=(qk_dim==g_cu_fa_mla_qk_dim_identity ||
                    qk_dim==g_cu_fa_mla_qk_dim_decoupled);
  if(qk_ok==false)
  {
    return false;
  }
  int cc_major=0;
  cudaDeviceGetAttribute(&cc_major,cudaDevAttrComputeCapabilityMajor,device_id);
  if(cc_major<g_cu_fa_tc_min_cc_major)
  {
    return false;
  }
  int optin_smem=0;
  cudaDeviceGetAttribute(&optin_smem,cudaDevAttrMaxSharedMemoryPerBlockOptin,device_id);
  bool any_fits=false;
  for(int c=0;c<g_cu_fa_mla_tile_count;++c)
  {
    const size_t smem=fa_tc_smem(qk_dim,g_cu_fa_mla_tile_br[c],g_cu_fa_mla_tile_bc[c]);
    if(smem<=static_cast<size_t>(optin_smem))
    {
      any_fits=true;
    }
  }
  return any_fits;
}

template bool launch_flash_attention_forward_mla<float>(const float *,
                                                        const float *,
                                                        const float *,
                                                        float *,
                                                        int,
                                                        int,
                                                        int,
                                                        int,
                                                        int,
                                                        float,
                                                        int,
                                                        int,
                                                        cudaStream_t);
template bool launch_flash_attention_forward_mla<__half>(const __half *,
                                                         const __half *,
                                                         const __half *,
                                                         __half *,
                                                         int,
                                                         int,
                                                         int,
                                                         int,
                                                         int,
                                                         float,
                                                         int,
                                                         int,
                                                         cudaStream_t);
template bool launch_flash_attention_forward_mla<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                const __nv_bfloat16 *,
                                                                const __nv_bfloat16 *,
                                                                __nv_bfloat16 *,
                                                                int,
                                                                int,
                                                                int,
                                                                int,
                                                                int,
                                                                float,
                                                                int,
                                                                int,
                                                                cudaStream_t);

template bool launch_flash_attention_forward_mla_tile<float>(const float *,
                                                             const float *,
                                                             const float *,
                                                             float *,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             float,
                                                             int,
                                                             int,
                                                             int,
                                                             cudaStream_t);
template bool launch_flash_attention_forward_mla_tile<__half>(const __half *,
                                                              const __half *,
                                                              const __half *,
                                                              __half *,
                                                              int,
                                                              int,
                                                              int,
                                                              int,
                                                              int,
                                                              float,
                                                              int,
                                                              int,
                                                              int,
                                                              cudaStream_t);
template bool launch_flash_attention_forward_mla_tile<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                     const __nv_bfloat16 *,
                                                                     const __nv_bfloat16 *,
                                                                     __nv_bfloat16 *,
                                                                     int,
                                                                     int,
                                                                     int,
                                                                     int,
                                                                     int,
                                                                     float,
                                                                     int,
                                                                     int,
                                                                     int,
                                                                     cudaStream_t);
