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
// Fused tensor-core FlashAttention forward for MLA (Multi-head Latent Attention)
// prefill. Online softmax keeps the [q_len, kv_len] score block in registers /
// SRAM, so prompt encoding is O(seq) memory instead of the explicit path's
// O(seq^2) (which OOMs past ~16K context). Forward inference only.
//
// Differs from the self-attention forward (caif_cuda_kernels_flash_self) in two
// ways:
//   * Q/K (scores) head dim D_qk is decoupled from V/output head dim D_v
//     (DeepSeek-V2-Lite is (192, 128)).
//   * q_len / kv_len / q_offset are first-class: query row r attends keys
//     <= q_offset + r, so chunked prefill into a warm KV cache works
//     (q_offset = cache_len, q_len = new_len, kv_len = total_len).
//
// Definitions live in src/caif_cuda_kernels_flash_mla.cu. CPU-only builds link
// the no-op stubs in legacy/src/caif_cuda_kernels_cpu.cpp (mla_flash_prefill_
// available returns false there, so callers fall back to the explicit path).
//------------------------------------------------------------------------------

#pragma once

#include <cstdint>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#else
// Placeholder for non-CUDA builds
typedef void *cudaStream_t;
#endif

//------------------------------------------------------------------------------
// Dispatch predicate
//------------------------------------------------------------------------------

// Returns true IFF a fused tensor-core instantiation exists for (qk_dim, v_dim)
// AND the device at device_id can run it: compute capability major >= 8 (TF32
// WMMA + cp.async), and at least one candidate tile fits the device's per-block
// opt-in shared memory. When false, the caller MUST use the explicit attention
// path. This NEVER aborts and has no side effects — it is the graceful-fallback
// gate the MLA forward checks before committing to the fused kernel.
bool mla_flash_prefill_available(int qk_dim,int v_dim,int device_id);

//------------------------------------------------------------------------------
// Fused MLA flash-prefill forward
//------------------------------------------------------------------------------

// Scaled dot-product attention without materializing the score matrix.
//
//   q       : [batch_heads, q_len,  qk_dim]   post-RoPE, pre-scale, StorageDtype
//   k       : [batch_heads, kv_len, qk_dim]   = ConcatLastDim(k_nope, k_pe)
//   v_heads : [batch_heads, kv_len, v_dim]
//   out     : [batch_heads, q_len,  v_dim]    (pre-allocated by the caller)
//
// scale is applied to the raw Q.K^T scores BY THE KERNEL — pass the exact value
// the explicit path uses (1/sqrt(qk_dim) for MLA, where qk_dim = qk_nope+qk_rope);
// the kernel does NOT recompute it. causal != 0 applies the offset-causal mask
// (key bc masked when bc > q_offset + global_q). q_offset = 0 for whole-prompt
// prefill; q_offset = cache_len for chunked prefill into a warm cache.
//
// Returns true IFF the fused kernel launched cleanly. PRECONDITION:
// mla_flash_prefill_available(qk_dim, v_dim, <launch device>) returned true.
// The launcher checks cudaFuncSetAttribute's return and cudaGetLastError; on any
// failure it skips/abandons the launch (never runs on garbage) and returns
// false. This is an nvcc TU and CANNOT throw — the .cpp caller turns a false
// return after a true availability check into a THROW_CAIFE (with the cuda
// error). A "no fitting tile / unsupported arch" device is gated out by the
// predicate, so a false here means a genuine post-check failure.
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
                                        cudaStream_t stream);

// Diagnostic / tile-ranking launcher: forces a specific candidate tile_index
// (0 .. g_cu_fa_mla_tile_count-1) instead of the per-device auto-selection, so
// the tile order can be measured and re-ranked on a new arch. The tile's own
// smem-fit check still gates the launch; returns false for an out-of-range
// index, a non-fitting tile, or an unsupported (qk_dim, v_dim). Production code
// uses launch_flash_attention_forward_mla (auto-selecting); benchmarks use this.
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
                                             cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template bool launch_flash_attention_forward_mla<float>(const float *,
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
extern template bool launch_flash_attention_forward_mla<__half>(const __half *,
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
extern template bool launch_flash_attention_forward_mla<__nv_bfloat16>(const __nv_bfloat16 *,
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
extern template bool launch_flash_attention_forward_mla_tile<float>(const float *,
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
extern template bool launch_flash_attention_forward_mla_tile<__half>(const __half *,
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
extern template bool launch_flash_attention_forward_mla_tile<__nv_bfloat16>(const __nv_bfloat16 *,
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
#endif
