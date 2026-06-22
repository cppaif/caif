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
// FlashAttention-2 cross-attention CUDA kernels (forward + backward, TC and
// scalar paths; Q has q_seq_len, K/V have kv_seq_len, no causal mask). Carved
// verbatim out of caif_cuda_kernels.cu.
// Declarations: include/caif/caif_cuda_kernels_flash_cross.cuh
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

//==============================================================================
// FlashAttention-2 Cross-Attention Kernels
//
// Identical algorithm to self-attention but Q has q_seq_len while K/V have
// kv_seq_len (different lengths). No causal mask — decoder attends to all
// encoder positions.
//==============================================================================

//------------------------------------------------------------------------------
// Cross-Attention Forward — Tensor Core Kernel
//------------------------------------------------------------------------------
template<typename T,int D,int BR,int BC,int NW>
__global__ void flash_attention_forward_cross_tc_kernel(const T *__restrict__ Q,
                                                        const T *__restrict__ K,
                                                        const T *__restrict__ V,
                                                        T *__restrict__ O,
                                                        float *__restrict__ L,
                                                        const int q_seq_len,
                                                        const int kv_seq_len,
                                                        const float scale)
{
#if CAIF_HAS_TC_FLASH
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int lane_id=tid%g_cu_warp_size;
  const int warp_id=tid/g_cu_warp_size;

  constexpr int n_warps=NW;
  constexpr int tiles_m=BR/g_cu_wmma_tile_m;
  constexpr int tiles_n_s=BC/g_cu_wmma_tile_n;
  constexpr int tiles_n_o=D/g_cu_wmma_tile_n;
  constexpr int block_threads=n_warps*g_cu_warp_size;

  constexpr int d_pad=D+g_cu_fa_smem_pad;
  constexpr int bc_pad=BC+g_cu_fa_smem_pad;
  constexpr int d_f2=D/g_cu_float2_lanes;
  constexpr int d_pad_f2=d_pad/g_cu_float2_lanes;

  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *KV_buf=Q_tile+BR*d_pad;
  float *S_tile=KV_buf+BC*d_pad;
  float *row_max_arr=S_tile+BR*bc_pad;
  float *row_sum_arr=row_max_arr+BR;

  const T *Q_bh=Q+static_cast<size_t>(bh)*q_seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh)*kv_seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh)*kv_seq_len*D;
  T *O_bh=O+static_cast<size_t>(bh)*q_seq_len*D;
  float *L_bh=L+static_cast<size_t>(bh)*q_seq_len;

  const int q_start=q_block_idx*BR;

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

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                         g_cu_wmma_tile_m,
                         g_cu_wmma_tile_n,
                         g_cu_wmma_tile_k,
                         float> o_frags[o_arr];
  for(int t=0;t<o_tiles_pw;++t)
  {
    nvcuda::wmma::fill_fragment(o_frags[t],0.0f);
  }

  // Load Q_tile from Q (q_seq_len bounded)
  const int valid_q_rows=min(BR,q_seq_len-q_start);
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

  for(int i=tid;i<BR;i+=block_threads)
  {
    row_max_arr[i]=-INFINITY;
    row_sum_arr[i]=0.0f;
  }
  __syncthreads();

  // KV blocks iterate over kv_seq_len (no causal limit)
  const int num_kv_blocks=(kv_seq_len+BC-1)/BC;
  constexpr int kv_f2=BC*d_f2;

  // Pipeline: prefetch K[0]
  float2 *KV_dst2=reinterpret_cast<float2 *>(KV_buf);
  if(num_kv_blocks>0)
  {
    if constexpr(sizeof(T)==sizeof(float))
    {
      const int kv0_valid=min(BC,kv_seq_len)*d_f2;
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
      const int kv0_valid=min(BC,kv_seq_len)*D;
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
    const int valid_kv_rows=min(BC,kv_seq_len-kv_start);
    const int valid_kv_f2=valid_kv_rows*d_f2;

    // PHASE 1: Wait for K, compute S = Q @ K^T
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
    }

    __syncthreads();

    // Async V load (overlapped with softmax)
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

    // PHASE 2: Register-based online softmax — no causal mask, only boundary masks
    {
      const int row_lo=m_idx*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row);
      const int row_hi=m_idx*g_cu_wmma_tile_m+(lane_id/g_cu_wmma_acc_lanes_per_row)+g_cu_wmma_acc_half_tile;
      const int global_q_lo=q_start+row_lo;
      const int global_q_hi=q_start+row_hi;

      // Boundary masks only (no causal)
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

        // Mask out-of-bounds KV positions
        if(bc0>=kv_seq_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[2]=-INFINITY;
        }
        if(bc1>=kv_seq_len)
        {
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
        }
        if(bc2>=kv_seq_len)
        {
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
        }
        if(bc3>=kv_seq_len)
        {
          s_accs[t].x[5]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }

        // Mask out-of-bounds Q positions
        if(global_q_lo>=q_seq_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[5]=-INFINITY;
        }
        if(global_q_hi>=q_seq_len)
        {
          s_accs[t].x[2]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }
      }

      float max_lo=-INFINITY;
      float max_hi=-INFINITY;
      for(int t=0;t<s_tiles_pw;++t)
      {
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[0],s_accs[t].x[1]));
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[4],s_accs[t].x[5]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[2],s_accs[t].x[3]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[6],s_accs[t].x[7]));
      }

      max_lo=fmaxf(max_lo,__shfl_xor_sync(g_cu_warp_full_mask,max_lo,1));
      max_lo=fmaxf(max_lo,__shfl_xor_sync(g_cu_warp_full_mask,max_lo,2));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(g_cu_warp_full_mask,max_hi,1));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(g_cu_warp_full_mask,max_hi,2));

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

      const float old_max_lo=row_max_arr[row_lo];
      const float old_max_hi=row_max_arr[row_hi];
      const float new_max_lo=fmaxf(old_max_lo,full_max_lo);
      const float new_max_hi=fmaxf(old_max_hi,full_max_hi);
      const float corr_lo=__expf(old_max_lo-new_max_lo);
      const float corr_hi=__expf(old_max_hi-new_max_hi);

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

      sum_lo+=__shfl_xor_sync(g_cu_warp_full_mask,sum_lo,1);
      sum_lo+=__shfl_xor_sync(g_cu_warp_full_mask,sum_lo,2);
      sum_hi+=__shfl_xor_sync(g_cu_warp_full_mask,sum_hi,1);
      sum_hi+=__shfl_xor_sync(g_cu_warp_full_mask,sum_hi,2);

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

      if(group_warp==0 && lane_id%g_cu_wmma_acc_lanes_per_row==0)
      {
        row_sum_arr[row_lo]=corr_lo*row_sum_arr[row_lo]+full_sum_lo;
        row_sum_arr[row_hi]=corr_hi*row_sum_arr[row_hi]+full_sum_hi;
        row_max_arr[row_lo]=new_max_lo;
        row_max_arr[row_hi]=new_max_hi;
      }

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

    // Store exp(S) to S_tile for Phase 3
    for(int t=0;t<s_tiles_pw;++t)
    {
      const int n=n_start_s+t;
      nvcuda::wmma::store_matrix_sync(&S_tile[m_idx*g_cu_wmma_tile_m*bc_pad+n*g_cu_wmma_tile_n],
                                      s_accs[t],
                                      bc_pad,
                                      nvcuda::wmma::mem_row_major);
    }

    cp_async_wait();
    __syncthreads();

    // PHASE 3: O += softmax(S) @ V
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

    // Pipeline: prefetch K[next]
    if(kv_block+1<num_kv_blocks)
    {
      const int next_start=(kv_block+1)*BC;
      if constexpr(sizeof(T)==sizeof(float))
      {
        const int next_valid=min(BC,kv_seq_len-next_start)*d_f2;
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
        const int next_valid=min(BC,kv_seq_len-next_start)*D;
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

  // Final: normalize and write O
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
    if(global_row<q_seq_len)
    {
      float inv_l=0.0f;
      if(row_sum_arr[row]>0.0f)
      {
        inv_l=1.0f/row_sum_arr[row];
      }
      O_bh[global_row*D+col]=T(O_smem[row*d_pad+col]*inv_l);
    }
  }

  for(int r=tid;r<BR;r+=block_threads)
  {
    const int global_row=q_start+r;
    if(global_row<q_seq_len)
    {
      L_bh[global_row]=row_max_arr[r]+logf(row_sum_arr[r]+g_cu_fa_logsumexp_epsilon);
    }
  }
#endif  // CAIF_HAS_TC_FLASH
}

template<typename T,int D,int BR,int BC,int NW>
static void launch_fa_fwd_cross_tc(const T *Q,
                                   const T *K,
                                   const T *V,
                                   T *O,
                                   float *L,
                                   const int batch_heads,
                                   const int q_seq_len,
                                   const int kv_seq_len,
                                   const float scale,
                                   cudaStream_t stream)
{
  const int num_q_blocks=(q_seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(NW*g_cu_warp_size);
  constexpr size_t smem_size=fa_tc_smem(D,BR,BC);

  if(smem_size>g_cu_default_shared_memory)
  {
    cudaFuncSetAttribute((void *)flash_attention_forward_cross_tc_kernel<T,D,BR,BC,NW>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_size));
  }
  flash_attention_forward_cross_tc_kernel<T,D,BR,BC,NW>
    <<<grid,block,smem_size,stream>>>(Q,
                                      K,
                                      V,
                                      O,
                                      L,
                                      q_seq_len,
                                      kv_seq_len,
                                      scale);
}

//------------------------------------------------------------------------------
// Cross-Attention Forward — Warp-Per-Row Scalar Fallback
//------------------------------------------------------------------------------
template<typename T,int D,int BR>
__global__ void flash_attention_forward_cross_kernel(const T *__restrict__ Q,
                                                     const T *__restrict__ K,
                                                     const T *__restrict__ V,
                                                     T *__restrict__ O,
                                                     float *__restrict__ L,
                                                     const int q_seq_len,
                                                     const int kv_seq_len,
                                                     const float scale)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int warp_id=tid/g_cu_warp_size;
  const int lane_id=tid%g_cu_warp_size;

  const int q_row=q_block_idx*BR+warp_id;
  const bool q_valid=(q_row<q_seq_len);

  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_cu_fa_fwd_bc*D;

  const T *Q_bh=Q+static_cast<size_t>(bh)*q_seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh)*kv_seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh)*kv_seq_len*D;
  T *O_bh=O+static_cast<size_t>(bh)*q_seq_len*D;
  float *L_bh=L+static_cast<size_t>(bh)*q_seq_len;

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

  float m_i=-INFINITY;
  float l_i=0.0f;
  float o_reg[elems];
  for(int e=0;e<elems;++e)
  {
    o_reg[e]=0.0f;
  }

  // No causal limit — iterate all KV blocks
  const int num_kv_blocks=(kv_seq_len+g_cu_fa_fwd_bc-1)/g_cu_fa_fwd_bc;
  const int block_threads=BR*g_cu_warp_size;

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_cu_fa_fwd_bc;

    const int tile_elems=g_cu_fa_fwd_bc*D;
    for(int i=tid;i<tile_elems;i+=block_threads)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;
      if(global_row<kv_seq_len)
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

    float s_local[g_cu_fa_fwd_bc];
    float row_max=-INFINITY;
    int num_valid=0;

    for(int j=0;j<g_cu_fa_fwd_bc;++j)
    {
      const int k_row=kv_start+j;
      if(k_row>=kv_seq_len)
      {
        break;
      }

      float dot=0.0f;
      for(int dd=lane_id,e=0;dd<D;dd+=g_cu_warp_size,++e)
      {
        dot+=q_reg[e]*K_tile[j*D+dd];
      }
      for(int offset=g_cu_warp_half_size;offset>0;offset/=2)
      {
        dot+=__shfl_down_sync(g_cu_warp_full_mask,dot,offset);
      }
      dot=__shfl_sync(g_cu_warp_full_mask,dot,0);
      s_local[j]=dot*scale;
      if(s_local[j]>row_max)
      {
        row_max=s_local[j];
      }
      num_valid=j+1;
    }

    const float m_new=fmaxf(m_i,row_max);
    const float scale_old=expf(m_i-m_new);
    for(int e=0;e<elems;++e)
    {
      o_reg[e]*=scale_old;
    }

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
    if(lane_id==0)
    {
      L_bh[q_row]=m_i+logf(l_i+g_cu_fa_logsumexp_epsilon);
    }
  }
}

template<typename T,int D,int BR>
static void launch_fa_fwd_cross(const T *Q,
                                const T *K,
                                const T *V,
                                T *O,
                                float *L,
                                const int batch_heads,
                                const int q_seq_len,
                                const int kv_seq_len,
                                const float scale,
                                cudaStream_t stream)
{
  const int num_q_blocks=(q_seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(BR*g_cu_warp_size);
  const size_t smem_size=fa_scalar_smem(D);

  if(smem_size>g_cu_default_shared_memory)
  {
    cudaFuncSetAttribute((void *)flash_attention_forward_cross_kernel<T,D,BR>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_size));
  }
  flash_attention_forward_cross_kernel<T,D,BR>
    <<<grid,block,smem_size,stream>>>(Q,
                                      K,
                                      V,
                                      O,
                                      L,
                                      q_seq_len,
                                      kv_seq_len,
                                      scale);
}

//------------------------------------------------------------------------------
// Cross-Attention Forward — Adaptive TC/Scalar Dispatch
//------------------------------------------------------------------------------
template<typename T,int D,int BR,int BC>
static void dispatch_fa_fwd_cross_tc_nw(const T *Q,
                                        const T *K,
                                        const T *V,
                                        T *O,
                                        float *L,
                                        const int batch_heads,
                                        const int q_seq_len,
                                        const int kv_seq_len,
                                        const float scale,
                                        cudaStream_t stream,
                                        const int nw)
{
  if(nw<=g_cu_fa_tc_warps_min)
  {
    launch_fa_fwd_cross_tc<T,D,BR,BC,g_cu_fa_tc_warps_min>(Q,
                                                           K,
                                                           V,
                                                           O,
                                                           L,
                                                           batch_heads,
                                                           q_seq_len,
                                                           kv_seq_len,
                                                           scale,
                                                           stream);
  }
  else if(nw<=g_cu_fa_tc_warps_small)
  {
    launch_fa_fwd_cross_tc<T,D,BR,BC,g_cu_fa_tc_warps_small>(Q,
                                                             K,
                                                             V,
                                                             O,
                                                             L,
                                                             batch_heads,
                                                             q_seq_len,
                                                             kv_seq_len,
                                                             scale,
                                                             stream);
  }
  else
  {
    launch_fa_fwd_cross_tc<T,D,BR,BC,g_cu_fa_tc_warps_large>(Q,
                                                             K,
                                                             V,
                                                             O,
                                                             L,
                                                             batch_heads,
                                                             q_seq_len,
                                                             kv_seq_len,
                                                             scale,
                                                             stream);
  }
}

template<typename T,int D>
static void dispatch_fa_fwd_cross(const T *Q,
                                  const T *K,
                                  const T *V,
                                  T *O,
                                  float *L,
                                  const int batch_heads,
                                  const int q_seq_len,
                                  const int kv_seq_len,
                                  const float scale,
                                  cudaStream_t stream,
                                  const int cc_major,
                                  const size_t smem_limit,
                                  const size_t smem_per_sm,
                                  const int max_threads)
{
  if(cc_major>=g_cu_fa_tc_min_cc_major)
  {
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

      if(br==g_cu_fa_fwd_tc_br_large && bc==g_cu_fa_fwd_tc_bc_large)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,g_cu_fa_fwd_tc_br_large,g_cu_fa_fwd_tc_bc_large>(Q,
                                                                                         K,
                                                                                         V,
                                                                                         O,
                                                                                         L,
                                                                                         batch_heads,
                                                                                         q_seq_len,
                                                                                         kv_seq_len,
                                                                                         scale,
                                                                                         stream,
                                                                                         nw);
        return;
      }
      if(br==g_cu_fa_fwd_tc_br_small && bc==g_cu_fa_fwd_tc_bc_large)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,g_cu_fa_fwd_tc_br_small,g_cu_fa_fwd_tc_bc_large>(Q,
                                                                                         K,
                                                                                         V,
                                                                                         O,
                                                                                         L,
                                                                                         batch_heads,
                                                                                         q_seq_len,
                                                                                         kv_seq_len,
                                                                                         scale,
                                                                                         stream,
                                                                                         nw);
        return;
      }
      if(br==g_cu_fa_fwd_tc_br_large && bc==g_cu_fa_fwd_tc_bc_small)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,g_cu_fa_fwd_tc_br_large,g_cu_fa_fwd_tc_bc_small>(Q,
                                                                                         K,
                                                                                         V,
                                                                                         O,
                                                                                         L,
                                                                                         batch_heads,
                                                                                         q_seq_len,
                                                                                         kv_seq_len,
                                                                                         scale,
                                                                                         stream,
                                                                                         nw);
        return;
      }
      if(br==g_cu_fa_fwd_tc_br_small && bc==g_cu_fa_fwd_tc_bc_small)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,g_cu_fa_fwd_tc_br_small,g_cu_fa_fwd_tc_bc_small>(Q,
                                                                                         K,
                                                                                         V,
                                                                                         O,
                                                                                         L,
                                                                                         batch_heads,
                                                                                         q_seq_len,
                                                                                         kv_seq_len,
                                                                                         scale,
                                                                                         stream,
                                                                                         nw);
        return;
      }
    }
  }

  if(fa_scalar_smem(D)<=smem_limit && max_threads>=g_cu_fa_scalar_warps_large*g_cu_warp_size)
  {
    launch_fa_fwd_cross<T,D,g_cu_fa_scalar_warps_large>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
  else if(fa_scalar_smem(D)<=smem_limit && max_threads>=g_cu_fa_scalar_warps_small*g_cu_warp_size)
  {
    launch_fa_fwd_cross<T,D,g_cu_fa_scalar_warps_small>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
  else
  {
    launch_fa_fwd_cross<T,D,g_cu_fa_scalar_warps_min>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
}

// (former extern "C" block — C++ linkage used for dtype templates)

template<typename T>
void launch_flash_attention_forward_cross(const T *Q,
                                          const T *K,
                                          const T *V,
                                          T *O,
                                          float *L,
                                          const int batch_heads,
                                          const int q_seq_len,
                                          const int kv_seq_len,
                                          const int head_dim,
                                          const float scale,
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
      dispatch_fa_fwd_cross<T,g_cu_fa_head_dim_32>(Q,
                                                   K,
                                                   V,
                                                   O,
                                                   L,
                                                   batch_heads,
                                                   q_seq_len,
                                                   kv_seq_len,
                                                   scale,
                                                   stream,
                                                   cc_major,
                                                   smem_limit,
                                                   smem_per_sm,
                                                   max_threads);
      break;
    case g_cu_fa_head_dim_64:
      dispatch_fa_fwd_cross<T,g_cu_fa_head_dim_64>(Q,
                                                   K,
                                                   V,
                                                   O,
                                                   L,
                                                   batch_heads,
                                                   q_seq_len,
                                                   kv_seq_len,
                                                   scale,
                                                   stream,
                                                   cc_major,
                                                   smem_limit,
                                                   smem_per_sm,
                                                   max_threads);
      break;
    case g_cu_fa_head_dim_80:
      dispatch_fa_fwd_cross<T,g_cu_fa_head_dim_80>(Q,
                                                   K,
                                                   V,
                                                   O,
                                                   L,
                                                   batch_heads,
                                                   q_seq_len,
                                                   kv_seq_len,
                                                   scale,
                                                   stream,
                                                   cc_major,
                                                   smem_limit,
                                                   smem_per_sm,
                                                   max_threads);
      break;
    case g_cu_fa_head_dim_96:
      dispatch_fa_fwd_cross<T,g_cu_fa_head_dim_96>(Q,
                                                   K,
                                                   V,
                                                   O,
                                                   L,
                                                   batch_heads,
                                                   q_seq_len,
                                                   kv_seq_len,
                                                   scale,
                                                   stream,
                                                   cc_major,
                                                   smem_limit,
                                                   smem_per_sm,
                                                   max_threads);
      break;
    case g_cu_fa_head_dim_128:
      dispatch_fa_fwd_cross<T,g_cu_fa_head_dim_128>(Q,
                                                    K,
                                                    V,
                                                    O,
                                                    L,
                                                    batch_heads,
                                                    q_seq_len,
                                                    kv_seq_len,
                                                    scale,
                                                    stream,
                                                    cc_major,
                                                    smem_limit,
                                                    smem_per_sm,
                                                    max_threads);
      break;
    default:
      fprintf(stderr,
              "FATAL: flash_attention_forward_cross unsupported head_dim=%d"
              " (supported: 32,64,80,96,128). Use standard attention.\n",
              head_dim);
      abort();
  }
}
template void launch_flash_attention_forward_cross<float>(const float *,
                                                          const float *,
                                                          const float *,
                                                          float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          float,
                                                          cudaStream_t);
template void launch_flash_attention_forward_cross<__half>(const __half *,
                                                           const __half *,
                                                           const __half *,
                                                           __half *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           cudaStream_t);
template void launch_flash_attention_forward_cross<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  const __nv_bfloat16 *,
                                                                  const __nv_bfloat16 *,
                                                                  __nv_bfloat16 *,
                                                                  float *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  float,
                                                                  cudaStream_t);

// end of former extern "C" block (cross-attention forward)

//------------------------------------------------------------------------------
// Cross-Attention Backward — Precompute Di = dot(dO, O) per Q row
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_precompute_di_cross_kernel(const T *__restrict__ dO,
                                                           const T *__restrict__ O,
                                                           float *__restrict__ Di,
                                                           const int q_seq_len)
{
  const int bh=blockIdx.x;
  const int row=blockIdx.y*blockDim.x+threadIdx.x;

  if(row>=q_seq_len)
  {
    return;
  }

  const T *dO_row=dO+static_cast<size_t>(bh)*q_seq_len*D+row*D;
  const T *O_row=O+static_cast<size_t>(bh)*q_seq_len*D+row*D;

  float sum=0.0f;
  for(int d=0;d<D;++d)
  {
    sum+=float(dO_row[d])*float(O_row[d]);
  }
  Di[static_cast<size_t>(bh)*q_seq_len+row]=sum;
}

//------------------------------------------------------------------------------
// Cross-Attention Backward — dK/dV kernel
// Each thread owns one K/V row (kv_seq_len), tiles over Q rows (q_seq_len).
// Tiles are the g_cu_fa_bwd_* set shared with the self-attention backward
// (cross was derived from self; same tiles by construction).
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_backward_cross_kernel(const T *__restrict__ Q,
                                                      const T *__restrict__ K,
                                                      const T *__restrict__ V,
                                                      const T *__restrict__ dO,
                                                      const float *__restrict__ L,
                                                      const float *__restrict__ Di,
                                                      T *__restrict__ dK,
                                                      T *__restrict__ dV,
                                                      const int q_seq_len,
                                                      const int kv_seq_len,
                                                      const float scale)
{
  const int bh=blockIdx.x;
  const int kv_block_idx=blockIdx.y;
  const int tid=threadIdx.x;

  const int kv_row=kv_block_idx*g_cu_fa_bwd_bc+tid;
  const int active=(kv_row<kv_seq_len);

  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *dO_tile=smem+g_cu_fa_bwd_br*D;
  float *L_tile=smem+g_cu_fa_bwd_q_do_tiles*g_cu_fa_bwd_br*D;
  float *Di_tile=smem+g_cu_fa_bwd_q_do_tiles*g_cu_fa_bwd_br*D+g_cu_fa_bwd_br;

  const T *Q_bh=Q+static_cast<size_t>(bh)*q_seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh)*kv_seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh)*kv_seq_len*D;
  const T *dO_bh=dO+static_cast<size_t>(bh)*q_seq_len*D;
  const float *L_bh=L+static_cast<size_t>(bh)*q_seq_len;
  const float *Di_bh=Di+static_cast<size_t>(bh)*q_seq_len;
  T *dK_bh=dK+static_cast<size_t>(bh)*kv_seq_len*D;
  T *dV_bh=dV+static_cast<size_t>(bh)*kv_seq_len*D;

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

  // Iterate all Q blocks (no causal skip)
  const int num_q_blocks=(q_seq_len+g_cu_fa_bwd_br-1)/g_cu_fa_bwd_br;

  for(int q_block=0;q_block<num_q_blocks;++q_block)
  {
    const int q_start=q_block*g_cu_fa_bwd_br;

    for(int i=tid;i<g_cu_fa_bwd_br*D;i+=g_cu_fa_bwd_bc)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=q_start+row;

      if(global_row<q_seq_len)
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

    if(tid<g_cu_fa_bwd_br)
    {
      const int global_row=q_start+tid;
      if(global_row<q_seq_len)
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

        if(q_row>=q_seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_tile[qi*D+d]*K_row[d];
        }
        const float S_val=dot*scale;
        const float P_val=expf(fminf(S_val-L_tile[qi],0.0f));

        for(int d=0;d<D;++d)
        {
          dV_acc[d]+=P_val*dO_tile[qi*D+d];
        }

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_tile[qi*D+d]*V_row[d];
        }

        const float dS=P_val*(dP-Di_tile[qi])*scale;

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
// Cross-Attention Backward — dQ kernel
// Each thread owns one Q row (q_seq_len), tiles over K/V rows (kv_seq_len)
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_backward_cross_dq_kernel(const T *__restrict__ Q,
                                                         const T *__restrict__ K,
                                                         const T *__restrict__ V,
                                                         const T *__restrict__ dO,
                                                         const float *__restrict__ L,
                                                         const float *__restrict__ Di,
                                                         T *__restrict__ dQ,
                                                         const int q_seq_len,
                                                         const int kv_seq_len,
                                                         const float scale)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;

  const int q_row=q_block_idx*g_cu_fa_bwd_dq_br+tid;
  const int active=(q_row<q_seq_len);

  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_cu_fa_bwd_dq_bc*D;

  const T *Q_bh=Q+static_cast<size_t>(bh)*q_seq_len*D;
  const T *K_bh=K+static_cast<size_t>(bh)*kv_seq_len*D;
  const T *V_bh=V+static_cast<size_t>(bh)*kv_seq_len*D;
  const T *dO_bh=dO+static_cast<size_t>(bh)*q_seq_len*D;
  const float *L_bh=L+static_cast<size_t>(bh)*q_seq_len;
  const float *Di_bh=Di+static_cast<size_t>(bh)*q_seq_len;
  T *dQ_bh=dQ+static_cast<size_t>(bh)*q_seq_len*D;

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

  // Iterate all KV blocks (no causal limit)
  const int num_kv_blocks=(kv_seq_len+g_cu_fa_bwd_dq_bc-1)/g_cu_fa_bwd_dq_bc;

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_cu_fa_bwd_dq_bc;

    for(int i=tid;i<g_cu_fa_bwd_dq_bc*D;i+=g_cu_fa_bwd_dq_br)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;

      if(global_row<kv_seq_len)
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

        if(k_row>=kv_seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_row_reg[d]*K_tile[j*D+d];
        }
        const float S_val=dot*scale;
        const float P_val=expf(fminf(S_val-L_val,0.0f));

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_row[d]*V_tile[j*D+d];
        }

        const float dS=P_val*(dP-Di_val)*scale;

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

// (former extern "C" block — C++ linkage used for dtype templates)

template<typename T>
void launch_flash_attention_backward_cross(const T *Q,
                                           const T *K,
                                           const T *V,
                                           const T *O,
                                           const T *dO,
                                           const float *L,
                                           T *dQ,
                                           T *dK,
                                           T *dV,
                                           const int batch_heads,
                                           const int q_seq_len,
                                           const int kv_seq_len,
                                           const int head_dim,
                                           const float scale,
                                           cudaStream_t stream)
{
  // Allocate temporary Di buffer [batch_heads, q_seq_len]
  float *Di_buf=nullptr;
  cudaMallocAsync(reinterpret_cast<void **>(&Di_buf),
                  static_cast<size_t>(batch_heads)*q_seq_len*sizeof(float),stream);

  // Kernel 0: Precompute Di = dot(dO, O) for each Q row
  {
    const int block_size=g_cu_block_size;
    const int rows_per_grid=(q_seq_len+block_size-1)/block_size;
    dim3 grid(batch_heads,rows_per_grid);

    switch(head_dim)
    {
      case g_cu_fa_head_dim_32:
        flash_attention_precompute_di_cross_kernel<T,g_cu_fa_head_dim_32>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case g_cu_fa_head_dim_64:
        flash_attention_precompute_di_cross_kernel<T,g_cu_fa_head_dim_64>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case g_cu_fa_head_dim_80:
        flash_attention_precompute_di_cross_kernel<T,g_cu_fa_head_dim_80>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case g_cu_fa_head_dim_96:
        flash_attention_precompute_di_cross_kernel<T,g_cu_fa_head_dim_96>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case g_cu_fa_head_dim_128:
        flash_attention_precompute_di_cross_kernel<T,g_cu_fa_head_dim_128>
          <<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_cross unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 1: Compute dK and dV (128 threads/block, tiles over q_seq_len)
  {
    const int num_kv_blocks=(kv_seq_len+g_cu_fa_bwd_bc-1)/g_cu_fa_bwd_bc;
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
        flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_32>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_64:
        flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_64>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_80:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_80>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_96:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_96>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_128:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_kernel<T,g_cu_fa_head_dim_128>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_cross unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 2: Compute dQ (128 threads/block, tiles over kv_seq_len)
  {
    const int num_q_blocks=(q_seq_len+g_cu_fa_bwd_dq_br-1)/g_cu_fa_bwd_dq_br;
    dim3 grid(batch_heads,num_q_blocks);
    dim3 block(g_cu_fa_bwd_dq_br);
    const size_t smem_size=g_cu_fa_kv_tiles*g_cu_fa_bwd_dq_bc*head_dim*sizeof(float);

    switch(head_dim)
    {
      case g_cu_fa_head_dim_32:
        flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_32>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_64:
        flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_64>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_80:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_80>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_96:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_96>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      case g_cu_fa_head_dim_128:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_dq_kernel<T,g_cu_fa_head_dim_128>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            q_seq_len,
                                            kv_seq_len,
                                            scale);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_cross_dq unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  cudaFreeAsync(Di_buf,stream);
}
template void launch_flash_attention_backward_cross<float>(const float *,
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
                                                           int,
                                                           float,
                                                           cudaStream_t);
template void launch_flash_attention_backward_cross<__half>(const __half *,
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
                                                            int,
                                                            float,
                                                            cudaStream_t);
template void launch_flash_attention_backward_cross<__nv_bfloat16>(const __nv_bfloat16 *,
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
                                                                   int,
                                                                   float,
                                                                   cudaStream_t);

// end of former extern "C" block (cross-attention backward)
