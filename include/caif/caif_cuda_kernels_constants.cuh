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
// Constants for the CUDA kernel modules (caif_cuda_kernels_*.cu). Single home
// for every named kernel constant — infra mirrors of caif_constants.h values
// (the kernel TUs do not include host-only headers), tile geometry, quant
// ranges, and math-formula constants. Included by every kernel TU via
// caif_cuda_kernels_common.cuh.
//------------------------------------------------------------------------------

#pragma once

// -- infra (must match caif_constants.h) --
constexpr int g_cu_block_size=256;
constexpr int g_cu_warp_size=32;
// shuffle-reduction start offset: half a warp
constexpr int g_cu_warp_half_size=g_cu_warp_size/2;
constexpr unsigned g_cu_warp_full_mask=0xffffffffu;
constexpr int g_cu_default_shared_memory=49152;
constexpr int g_cu_max_threads_fallback=1024;
// CAS word-alignment mask for sub-word atomics (bf16 emulation)
constexpr unsigned long g_cu_word_align_mask=3ul;
// sub-word atomics: half-word geometry for the bf16 CAS emulation
constexpr unsigned g_cu_half_word_bits=16u;
constexpr unsigned g_cu_half_word_mask=0xFFFFu;
// effectively -infinity for fp32 max-tracking (topk select, flash row max);
// chosen so the fp16-storage path cannot represent-and-collide with it
constexpr float g_cu_neg_sentinel=-1e30f;
// fp32 lanes in a float2 vectorized load/store
constexpr int g_cu_float2_lanes=2;

// -- per-arch softmax block sizes (mirror of caif_constants.h) --
// Turing
constexpr int g_cu_softmax_block_size_sm75=128;
// Ampere (A100)
constexpr int g_cu_softmax_block_size_sm80=128;
// Ampere (GA10x)
constexpr int g_cu_softmax_block_size_sm86=128;
// Ada Lovelace
constexpr int g_cu_softmax_block_size_sm89=128;
// Hopper
constexpr int g_cu_softmax_block_size_sm90=128;
// Blackwell
constexpr int g_cu_softmax_block_size_sm120=128;
constexpr int g_cu_softmax_block_size_default=128;
// staging stat arrays in softmax/logsumexp smem layouts (max + sum)
constexpr int g_cu_softmax_stat_arrays=2;
// additive causal/prefix mask fill for disallowed attention scores
constexpr float g_cu_attn_mask_fill=-1e9f;
// dims rotated together per RoPE pair
constexpr int g_cu_rope_dims_per_pair=2;
// upper bound on per-pair entries (head_dim/2) the RoPE kernels stage in shared
// memory for the inv-frequency table; covers head_dim up to 1024 (all real
// attention head dims are <=256, so half_dim <=128).
constexpr int g_cu_rope_max_half_dim=512;

// -- wmma tensor-core fragment shape (TF32 16x16x8) --
constexpr int g_cu_wmma_tile_m=16;
constexpr int g_cu_wmma_tile_n=16;
constexpr int g_cu_wmma_tile_k=8;

// TF32 wmma accumulator-fragment lane mapping (m16n16k8): each accumulator
// row group spans 4 lanes, each lane covers an adjacent column pair, and the
// fragment's second element group sits half a tile (8 rows/cols) away
constexpr int g_cu_wmma_acc_lanes_per_row=4;
constexpr int g_cu_wmma_acc_cols_per_lane=2;
constexpr int g_cu_wmma_acc_half_tile=8;

// -- flash-attention tile geometry (must match caif_constants.h) --
// K/V block size for the scalar forward kernel
constexpr int g_cu_fa_fwd_bc=64;
// backward tiles — SHARED by the self- and cross-attention backward kernels
// (cross was derived from self; the tiles are the same by construction)
// Q tile rows for dK/dV kernel
constexpr int g_cu_fa_bwd_br=64;
// K/V block size (threads per block)
constexpr int g_cu_fa_bwd_bc=128;
// Q block size for dQ kernel
constexpr int g_cu_fa_bwd_dq_br=128;
// K/V tile rows for dQ kernel
constexpr int g_cu_fa_bwd_dq_bc=64;
// TC forward tile candidates (BR x BC), shared by self + cross dispatch
constexpr int g_cu_fa_fwd_tc_br_large=32;
constexpr int g_cu_fa_fwd_tc_br_small=16;
constexpr int g_cu_fa_fwd_tc_bc_large=128;
constexpr int g_cu_fa_fwd_tc_bc_small=64;
// MLA flash-prefill supported head dims.
// Both supported configs share V dim 128 (DSv2-Lite v_head_dim); they differ in
// D_qk: 128 is the identity bring-up (D_qk == D_v), 192 is DSv2-Lite
// (qk_nope+qk_rope).
constexpr int g_cu_fa_mla_v_dim=128;
constexpr int g_cu_fa_mla_qk_dim_identity=128;
constexpr int g_cu_fa_mla_qk_dim_decoupled=192;
constexpr int g_cu_fa_mla_nw=4;
// Per-device tile candidates (Step 3). The launcher picks the first tile whose
// decoupled smem fits the launch device's per-block opt-in, so the order is the
// priority. Ranked by MEASURED (192,128) prefill throughput on the dev RTX 5090
// (Step 3 sweep, bench_flash_mla_tile): 64x48 (98 KB) is ~13-15% faster than
// 32x64 (81 KB) at 8K-16K prefill — the long-context regime this kernel exists
// for — so it leads; it fits every supported fused device's >=99 KB opt-in
// (A100/A10G/L4/H100/5090). 32x64 follows as the fallback for a smaller opt-in
// budget. (64x32 was dropped: dominated by 64x48 and no unique opt-in coverage.)
// Every tile must satisfy the kernel's warp-grouping invariant NW % (BR/16) == 0
// (documented in the kernel) — e.g. BR=48 is excluded at NW=4. H100's BC=128
// tiles (e.g. 32x128 = 138 KB) need the 227 KB opt-in and the hardware to
// validate, so they are a future entry.
constexpr int g_cu_fa_mla_tile_count=2;
constexpr int g_cu_fa_mla_tile_br[g_cu_fa_mla_tile_count]={64,32};
constexpr int g_cu_fa_mla_tile_bc[g_cu_fa_mla_tile_count]={48,64};
// scalar flash kernel warp-count options (block = warps * warp size)
constexpr int g_cu_fa_scalar_warps_large=8;
constexpr int g_cu_fa_scalar_warps_small=4;
// log-sum-exp guard epsilon for empty/fully-masked rows
constexpr float g_cu_fa_logsumexp_epsilon=1e-10f;
// TC kernel warp-count instantiations (runtime picks the closer)
constexpr int g_cu_fa_tc_warps_large=8;
constexpr int g_cu_fa_tc_warps_small=4;
// minimum compute capability for the TC path (sm_80, Ampere)
constexpr int g_cu_fa_tc_min_cc_major=8;
// rows in the (BR,BC) forward tile-candidate table
constexpr int g_cu_fa_fwd_tc_option_count=4;
// smem tile stride padding (in floats) that breaks wmma bank conflicts
constexpr int g_cu_fa_smem_pad=2;
// softmax statistics arrays in flash smem layouts (row max + row sum, or L + D)
constexpr int g_cu_fa_smem_stat_arrays=2;
// K tile + V tile pair held together in smem (scalar forward, backward dQ)
constexpr int g_cu_fa_kv_tiles=2;
// Q tile + dO tile pair in the backward dK/dV smem layout
constexpr int g_cu_fa_bwd_q_do_tiles=2;
// minimum warps per block for the scalar (warp-per-row) fallback
constexpr int g_cu_fa_scalar_warps_min=2;
// minimum warps per block for the TC kernels
constexpr int g_cu_fa_tc_warps_min=2;
// resident-block count at which warps are halved to double occupancy
constexpr int g_cu_fa_tc_occupancy_blocks=2;
// minimum S/O tile count required before halving warps pays off
constexpr int g_cu_fa_tc_halve_min_tiles=4;
// supported flash head dims (dispatch labels)
constexpr int g_cu_fa_head_dim_32=32;
constexpr int g_cu_fa_head_dim_64=64;
constexpr int g_cu_fa_head_dim_80=80;
constexpr int g_cu_fa_head_dim_96=96;
constexpr int g_cu_fa_head_dim_128=128;

// -- GELU math (must match caif_constants.h) --
constexpr float g_cu_gelu_sqrt_2_over_pi=0.7978845608f;
constexpr float g_cu_gelu_coeff=0.044715f;
// 1/sqrt(2)
constexpr float g_cu_gelu_inv_sqrt2=0.7071067812f;
// 1/sqrt(2*pi)
constexpr float g_cu_gelu_inv_sqrt2pi=0.3989422804f;
// the 1/2 in 0.5*x*(1+...) and the cubic-term derivative factor
constexpr float g_cu_gelu_half=0.5f;
constexpr float g_cu_gelu_cubic_factor=3.0f;

// -- loss --
// d(diff^2)/d(diff) coefficient in the MSE gradient
constexpr float g_cu_mse_grad_coeff=2.0f;
// staging arrays in the cross-entropy mean reduction (sum + count)
constexpr int g_cu_ce_mean_stat_arrays=2;

// -- MoE --
// divide-by-zero guard for row normalization
constexpr float g_cu_moe_row_sum_epsilon=1e-10f;
// divide-by-zero guard for top-k weight normalization
constexpr float g_cu_moe_topk_sum_epsilon=1e-12f;
// block size for the combine-backward grad-weights dot-product reduction
constexpr int g_cu_moe_grad_weights_block_size=256;
// fp32 staging arrays in the gating smem layout (logits + probs)
constexpr int g_cu_moe_gating_stat_arrays=2;

// -- quantization ranges / packing --
constexpr float g_cu_quant_int8_max=127.0f;
constexpr float g_cu_quant_int4_max=7.0f;
constexpr int g_cu_quant_int4_nibble_mask=0x0F;
// int4 values packed per byte (two nibbles)
constexpr int g_cu_quant_int4_per_byte=2;
// bits per int4 nibble (pack/unpack shift)
constexpr int g_cu_quant_int4_nibble_bits=4;
constexpr int g_cu_quant_int4_sign_bit=0x08;
// int4 sign extension fill for the high bits of a negative nibble
constexpr int g_cu_quant_int4_sign_extend=static_cast<int>(0xFFFFFFF0);
// per-tensor quant reduction grid cap
constexpr int g_cu_quant_reduce_max_grid=1024;
