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
// MoE CUDA kernels: normalize_rows (+backward/topk-gather), top-k,
// gather_topk_values, scatter-add, dispatch/combine (+backward),
// top-k gating, build-dispatch-map, router z-loss gradient.
// Launcher declarations for src/caif_cuda_kernels_moe.cu
//. CPU-only builds link the no-op stubs in
// legacy/src/caif_cuda_kernels_cpu.cpp, same contract as the old
// caif_cuda_kernels.h umbrella header.
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
// Reduction and normalization kernels
//------------------------------------------------------------------------------

/**
 * Sum along axis 0 (over batch): output[d] = sum_b input[b,d]
 */

/**
 * Normalize each row to sum to 1: output[b,d] = input[b,d] / sum_d(input[b,:])
 */
template<typename T>
void launch_normalize_rows(const T *input,
                           T *output,
                           int batch,
                           int dim,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_normalize_rows<float>(const float *,
                                                  float *,
                                                  int,
                                                  int,
                                                  cudaStream_t);
extern template void launch_normalize_rows<__half>(const __half *,
                                                   __half *,
                                                   int,
                                                   int,
                                                   cudaStream_t);
extern template void launch_normalize_rows<__nv_bfloat16>(const __nv_bfloat16 *,
                                                          __nv_bfloat16 *,
                                                          int,
                                                          int,
                                                          cudaStream_t);
#endif

/**
 * NormalizeRows backward Jacobian on top-k rows, gather variant.
 * Gathers p[k] = probs[t, indices[t,k]] from the full softmax cache,
 * computes s, w, dot, and grad_p_topk inline — no forward-side caches
 * beyond the existing probs/indices are needed.
 *   grad_p_topk[t,k] = (grad_w[t,k] - dot(t)) / s(t),
 *   s(t) = sum_k p[k],  w[k] = p[k]/s,  dot = sum_k w[k]*grad_w[t,k]
 * One warp per token; top_k must be <= 32.
 */
template<typename T>
void launch_normalize_rows_backward_topk_gather(const T *grad_w,
                                                const T *probs,
                                                const int *indices,
                                                T *grad_p_topk,
                                                int num_tokens,
                                                int num_experts,
                                                int top_k,
                                                cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_normalize_rows_backward_topk_gather<float>(const float *,
                                                                       const float *,
                                                                       const int *,
                                                                       float *,
                                                                       int,
                                                                       int,
                                                                       int,
                                                                       cudaStream_t);
extern template void launch_normalize_rows_backward_topk_gather<__half>(const __half *,
                                                                        const __half *,
                                                                        const int *,
                                                                        __half *,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        cudaStream_t);
extern template void launch_normalize_rows_backward_topk_gather<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                               const __nv_bfloat16 *,
                                                                               const int *,
                                                                               __nv_bfloat16 *,
                                                                               int,
                                                                               int,
                                                                               int,
                                                                               cudaStream_t);
#endif

//------------------------------------------------------------------------------
// Top-K and scatter operations
//------------------------------------------------------------------------------

/**
 * Select top-k values and indices per row.
 * input: [batch, dim], indices: [batch, k] (int32), values: [batch, k]
 */
template<typename T>
void launch_topk(const T *input,
                 int *indices,
                 T *values,
                 int batch,
                 int dim,
                 int k,
                 cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_topk<float>(const float *,
                                        int *,
                                        float *,
                                        int,
                                        int,
                                        int,
                                        cudaStream_t);
extern template void launch_topk<__half>(const __half *,
                                         int *,
                                         __half *,
                                         int,
                                         int,
                                         int,
                                         cudaStream_t);
extern template void launch_topk<__nv_bfloat16>(const __nv_bfloat16 *,
                                                int *,
                                                __nv_bfloat16 *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);
#endif

/**
 * DeepSeek group-limited routing mask (in place on [num_tokens, num_experts]
 * selection scores). Splits each token's experts into n_group equal groups,
 * scores a group by the sum of its top-2 experts, keeps the top-topk_group
 * groups, and sets every expert in a non-selected group to the neg-sentinel so
 * the downstream top-k never picks it. num_experts must be divisible by
 * n_group; topk_group in [1, n_group].
 */
template<typename T>
void launch_moe_group_mask(T *selection,
                           int num_tokens,
                           int num_experts,
                           int n_group,
                           int topk_group,
                           cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_moe_group_mask<float>(float *,int,int,int,int,cudaStream_t);
extern template void launch_moe_group_mask<__half>(__half *,int,int,int,int,cudaStream_t);
extern template void launch_moe_group_mask<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,cudaStream_t);
#endif

/**
 * Aux-loss-free router bias update (DeepSeek-V3). Nudges each expert's router
 * bias toward balance from its observed token load: bias[e] += rate*sign(mean
 * load - load[e]). No gradient, no aux loss. counts is [num_experts] int.
 */
template<typename T>
void launch_moe_bias_update(T *bias,
                            const int *counts,
                            int num_experts,
                            float rate,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_moe_bias_update<float>(float *,const int *,int,float,cudaStream_t);
extern template void launch_moe_bias_update<__half>(__half *,const int *,int,float,cudaStream_t);
extern template void launch_moe_bias_update<__nv_bfloat16>(__nv_bfloat16 *,const int *,int,float,cudaStream_t);
#endif

/**
 * Scatter-add values to output using indices.
 * output[b, indices[b,k]] += values[b,k]
 * values: [batch, k], indices: [batch, k], output: [batch, dim]
 */
template<typename T>
void launch_scatter_add(const T *values,
                        const int *indices,
                        T *output,
                        int batch,
                        int k,
                        int dim,
                        cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_scatter_add<float>(const float *,
                                               const int *,
                                               float *,
                                               int,
                                               int,
                                               int,
                                               cudaStream_t);
extern template void launch_scatter_add<__half>(const __half *,
                                                const int *,
                                                __half *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);
extern template void launch_scatter_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const int *,
                                                       __nv_bfloat16 *,
                                                       int,
                                                       int,
                                                       int,
                                                       cudaStream_t);
#endif

/**
 * Gather per-row scores at top-k indices.  Used by SigmoidNoauxTc
 * routing (HF noaux_tc) Phase 1b to read the ORIGINAL (uncorrected)
 * sigmoid scores at the chosen top-k positions while selection
 * happens on bias-corrected scores.
 *
 *   out[t, k] = scores[t, indices[t, k]]
 *
 * scores: [num_tokens, num_experts], indices: [num_tokens, top_k] (int32),
 * out:    [num_tokens, top_k].  top_k must be <= 32.
 */
template<typename T>
void launch_gather_topk_values(const T *scores,
                               const int *indices,
                               T *out,
                               int num_tokens,
                               int num_experts,
                               int top_k,
                               cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_gather_topk_values<float>(const float *,
                                                      const int *,
                                                      float *,
                                                      int,
                                                      int,
                                                      int,
                                                      cudaStream_t);
extern template void launch_gather_topk_values<__half>(const __half *,
                                                       const int *,
                                                       __half *,
                                                       int,
                                                       int,
                                                       int,
                                                       cudaStream_t);
extern template void launch_gather_topk_values<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const int *,
                                                              __nv_bfloat16 *,
                                                              int,
                                                              int,
                                                              int,
                                                              cudaStream_t);
#endif

//------------------------------------------------------------------------------
// MoE (Mixture of Experts) kernels
//------------------------------------------------------------------------------

/**
 * Dispatch tokens to expert buffers based on routing indices.
 * input: [num_tokens, dim] - token embeddings
 * expert_indices: [num_tokens, top_k] - which experts each token routes to (as floats)
 * dispatch_map: [num_tokens, top_k] - position within expert buffer for each assignment
 * expert_buffer: [total_assigned_tokens, dim] - contiguous buffer for all experts
 * expert_offsets: [num_experts+1] - cumulative token counts per expert
 */
template<typename T>
void launch_moe_dispatch(const T *input,
                         const int *expert_indices,
                         const int *dispatch_map,
                         T *expert_buffer,
                         const int *expert_offsets,
                         int num_tokens,
                         int dim,
                         int top_k,
                         cudaStream_t stream);

/**
 * Combine expert outputs back to token positions with routing weights.
 * expert_buffer: [total_assigned_tokens, dim] - contiguous buffer of expert outputs
 * expert_indices: [num_tokens, top_k] - which experts each token routed to (as floats)
 * expert_weights: [num_tokens, top_k] - routing weights (normalized)
 * dispatch_map: [num_tokens, top_k] - position within expert buffer for each assignment
 * expert_offsets: [num_experts+1] - cumulative token counts per expert
 * output: [num_tokens, dim] - combined output
 */
template<typename T>
void launch_moe_combine(const T *expert_buffer,
                        const int *expert_indices,
                        const T *expert_weights,
                        const int *dispatch_map,
                        const int *expert_offsets,
                        T *output,
                        int num_tokens,
                        int dim,
                        int top_k,
                        cudaStream_t stream);

/**
 * Backward of MoECombine — grad_expert_buffer path.
 * grad_expert_buffer[(off[e]+pos)*dim+d] = expert_weights[t,k] * grad_output[t,d]
 */
template<typename T>
void launch_moe_combine_backward_grad_expert(const T *grad_output,
                                             const int *expert_indices,
                                             const T *expert_weights,
                                             const int *dispatch_map,
                                             const int *expert_offsets,
                                             T *grad_expert_buffer,
                                             int num_tokens,
                                             int dim,
                                             int top_k,
                                             cudaStream_t stream);

/**
 * Backward of MoECombine — grad_weights path.
 * grad_weights[t,k] = sum_d grad_output[t,d] * expert_buffer[(off[e]+pos)*dim+d]
 */
template<typename T>
void launch_moe_combine_backward_grad_weights(const T *grad_output,
                                              const T *expert_buffer,
                                              const int *expert_indices,
                                              const int *dispatch_map,
                                              const int *expert_offsets,
                                              T *grad_weights,
                                              int num_tokens,
                                              int dim,
                                              int top_k,
                                              cudaStream_t stream);

/**
 * Backward of MoEDispatch — gathers per-assignment gradients back to each token.
 * grad_input[t,d] = sum_k grad_expert_buffer[(off[e(t,k)]+pos(t,k))*dim+d]
 */
template<typename T>
void launch_moe_dispatch_backward(const T *grad_expert_buffer,
                                  const int *expert_indices,
                                  const int *dispatch_map,
                                  const int *expert_offsets,
                                  T *grad_input,
                                  int num_tokens,
                                  int dim,
                                  int top_k,
                                  cudaStream_t stream);

/**
 * Fused softmax + top-k selection for MoE router.
 * router_logits: [num_tokens, num_experts] - raw router output
 * expert_indices: [num_tokens, top_k] - selected expert indices (as floats)
 * expert_weights: [num_tokens, top_k] - normalized routing weights
 * router_probs: [num_tokens, num_experts] - full softmax probabilities (for aux loss)
 */
// router_logits dtype is the layer's StorageT; internal accumulation
// runs in fp32 for numerical stability. Outputs (expert_weights /
// router_probs) are fp32 by router-output contract.
template<typename T>
void launch_moe_topk_gating(const T *router_logits,
                            int *expert_indices,
                            float *expert_weights,
                            float *router_probs,
                            int num_tokens,
                            int num_experts,
                            int top_k,
                            cudaStream_t stream);

#ifdef USE_CAIF_CUDA
extern template void launch_moe_topk_gating<float>(const float *,
                                                   int *,
                                                   float *,
                                                   float *,
                                                   int,
                                                   int,
                                                   int,
                                                   cudaStream_t);
extern template void launch_moe_topk_gating<__half>(const __half *,
                                                    int *,
                                                    float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);
extern template void launch_moe_topk_gating<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           int *,
                                                           float *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           cudaStream_t);
#endif

/**
 * Count tokens per expert using atomic operations.
 * expert_indices: [num_tokens, top_k] - which experts each token routes to (Int32)
 * expert_counts: [num_experts] - output counts per expert (must be zeroed first)
 *
 * fp32-only by contract: integer-only kernel (Int32 indices in, Int32
 * counts out). No floating-point data — templating on T is meaningless.
 */
void launch_moe_count_per_expert(const int *expert_indices,
                                 int *expert_counts,
                                 int num_tokens,
                                 int num_experts,
                                 int top_k,
                                 int capacity_per_expert,
                                 cudaStream_t stream);

/**
 * ST-MoE router z-loss gradient contribution.
 * grad_logits[t,e] += logsumexp_scaled[t] * probs[t,e]
 * logsumexp_scaled is pre-multiplied by (2 * z_loss_weight / N) so the kernel
 * only fuses a row-broadcast multiply-add across [N, E].
 */
template<typename T>
void launch_moe_z_loss_grad(const T *logsumexp_scaled,
                            const T *probs,
                            T *grad_logits,
                            int num_tokens,
                            int num_experts,
                            cudaStream_t stream);

