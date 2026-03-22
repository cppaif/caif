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
// CUDA kernel declarations for operations not supported by cuDNN
//
// IMPORTANT: Every function declared in this header has a GPU implementation
// in caif_cuda_kernels.cu and a CPU no-op stub in caif_cuda_kernels_cpu.cpp.
// When adding a new launch_* function here, you MUST also add a matching
// stub to caif_cuda_kernels_cpu.cpp or CPU-only builds will fail to link.
//------------------------------------------------------------------------------
#ifndef CAIF_CUDA_KERNELS_H
#define CAIF_CUDA_KERNELS_H

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#else
typedef void *cudaStream_t;  // Placeholder for non-CUDA builds
#endif

//------------------------------------------------------------------------------
// Gated activation op enum (C linkage, plain enum)
//------------------------------------------------------------------------------
enum CAIF_GatedActivationOp
{
  CAIF_GATED_OP_SWISH=0,
  CAIF_GATED_OP_GELU=1,
  CAIF_GATED_OP_RELU=2,
  CAIF_GATED_OP_SIGMOID=3,
  CAIF_GATED_OP_LINEAR=4
};

#ifdef __cplusplus
extern "C"
{
#endif

//------------------------------------------------------------------------------
// ReLU kernels (vectorized, replaces cuDNN)
//------------------------------------------------------------------------------
void launch_relu_forward(const float *input,
                         float *output,
                         int n,
                         cudaStream_t stream);

void launch_relu_backward(const float *grad_output,
                           const float *input,
                           float *grad_input,
                           int n,
                           cudaStream_t stream);

//------------------------------------------------------------------------------
// Sigmoid kernels (vectorized, replaces cuDNN)
//------------------------------------------------------------------------------
void launch_sigmoid_forward(const float *input,
                             float *output,
                             int n,
                             cudaStream_t stream);

void launch_sigmoid_backward(const float *grad_output,
                              const float *output,
                              float *grad_input,
                              int n,
                              cudaStream_t stream);

//------------------------------------------------------------------------------
// Tanh kernels (vectorized, replaces cuDNN)
//------------------------------------------------------------------------------
void launch_tanh_forward(const float *input,
                          float *output,
                          int n,
                          cudaStream_t stream);

void launch_tanh_backward(const float *grad_output,
                           const float *output,
                           float *grad_input,
                           int n,
                           cudaStream_t stream);

//------------------------------------------------------------------------------
// LeakyReLU kernels
//------------------------------------------------------------------------------
void launch_leaky_relu_forward(const float *input,
                               float *output,
                               float alpha,
                               int n,
                               cudaStream_t stream);

void launch_leaky_relu_backward(const float *grad_output,
                                const float *input,
                                float *grad_input,
                                float alpha,
                                int n,
                                cudaStream_t stream);

//------------------------------------------------------------------------------
// ELU kernels
//------------------------------------------------------------------------------
void launch_elu_forward(const float *input,
                        float *output,
                        float alpha,
                        int n,
                        cudaStream_t stream);

void launch_elu_backward(const float *grad_output,
                         const float *input,
                         const float *output,
                         float *grad_input,
                         float alpha,
                         int n,
                         cudaStream_t stream);

//------------------------------------------------------------------------------
// GELU kernels
//------------------------------------------------------------------------------
void launch_gelu_forward(const float *input,
                         float *output,
                         int n,
                         cudaStream_t stream);

void launch_gelu_backward(const float *grad_output,
                          const float *input,
                          float *grad_input,
                          int n,
                          cudaStream_t stream);

//------------------------------------------------------------------------------
// Swish kernels (fallback for cuDNN < 8.0)
//------------------------------------------------------------------------------
void launch_swish_forward(const float *input,
                          float *output,
                          int n,
                          cudaStream_t stream);

void launch_swish_backward(const float *grad_output,
                           const float *input,
                           const float *output,
                           float *grad_input,
                           int n,
                           cudaStream_t stream);

//------------------------------------------------------------------------------
// Element-wise operation kernels
//------------------------------------------------------------------------------
void launch_elementwise_add(const float *a,
                            const float *b,
                            float *result,
                            int n,
                            cudaStream_t stream);

void launch_elementwise_add_scalar(const float *a,
                                   float scalar,
                                   float *result,
                                   int n,
                                   cudaStream_t stream);

// Bias add (broadcast bias over rows for 2D tensors)
void launch_bias_add_2d(const float *input,
                        const float *bias,
                        float *output,
                        int batch,
                        int units,
                        cudaStream_t stream);

// Bias gradient (sum over batch rows)
void launch_bias_grad_2d(const float *grad_output,
                         float *bias_grad,
                         int batch,
                         int units,
                         cudaStream_t stream);

void launch_elementwise_sub(const float *a,
                            const float *b,
                            float *result,
                            int n,
                            cudaStream_t stream);

void launch_elementwise_sub_scalar(const float *a,
                                   float scalar,
                                   float *result,
                                   int n,
                                   cudaStream_t stream);

void launch_elementwise_mul(const float *a,
                            const float *b,
                            float *result,
                            int n,
                            cudaStream_t stream);

void launch_elementwise_mul_scalar(const float *a,
                                   float scalar,
                                   float *result,
                                   int n,
                                   cudaStream_t stream);

void launch_elementwise_div(const float *a,
                            const float *b,
                            float *result,
                            int n,
                            cudaStream_t stream);

void launch_elementwise_div_scalar(const float *a,
                                   float scalar,
                                   float *result,
                                   int n,
                                   cudaStream_t stream);

void launch_elementwise_sqrt(const float *a,
                             float *result,
                             int n,
                             cudaStream_t stream);

//------------------------------------------------------------------------------
// Reduction kernels
//------------------------------------------------------------------------------
void launch_reduction_sum(const float *input,
                          float *output,
                          int n,
                          cudaStream_t stream);

//------------------------------------------------------------------------------
// Loss function kernels
//------------------------------------------------------------------------------
void launch_cross_entropy_loss(const float *predictions,
                               const float *targets,
                               float *loss,
                               float epsilon,
                               int batch_size,
                               int num_classes,
                               cudaStream_t stream);

void launch_cross_entropy_gradient(const float *predictions,
                                   const float *targets,
                                   float *gradient,
                                   float epsilon,
                                   int batch_size,
                                   int n,
                                   cudaStream_t stream);
// Cross-entropy loss/grad with index targets
void launch_cross_entropy_loss_index(const float *predictions,
                                     const int *target_indices,
                                     float *loss,
                                     float epsilon,
                                     int batch_size,
                                     int num_classes,
                                     cudaStream_t stream);

void launch_cross_entropy_gradient_index(const float *predictions,
                                         const int *target_indices,
                                         float *gradient,
                                         float epsilon,
                                         int batch_size,
                                         int num_classes,
                                         cudaStream_t stream);

void launch_mse_loss(const float *predictions,
                     const float *targets,
                     float *loss,
                     int n,
                     cudaStream_t stream);

void launch_mse_gradient(const float *predictions,
                         const float *targets,
                         float *gradient,
                         int n,
                         cudaStream_t stream);

//------------------------------------------------------------------------------
// Optimizer kernels
//------------------------------------------------------------------------------

/**
 * @brief Fused Adam optimizer update
 * Combines all Adam operations into a single kernel pass
 */
void launch_fused_adam(float *param,
                       const float *grad,
                       float *m,
                       float *v,
                       float lr,
                       float beta1,
                       float beta2,
                       float epsilon,
                       float weight_decay,
                       float bias_correction1,
                       float bias_correction2,
                       int n,
                       cudaStream_t stream);

/**
 * @brief Fused Adam with gradient clipping
 */
void launch_fused_adam_clipped(float *param,
                               const float *grad,
                               float *m,
                               float *v,
                               float lr,
                               float beta1,
                               float beta2,
                               float epsilon,
                               float weight_decay,
                               float bias_correction1,
                               float bias_correction2,
                               float grad_scale,
                               int n,
                               cudaStream_t stream);

/**
 * @brief Fused SGD with momentum
 */
void launch_fused_sgd_momentum(float *param,
                               const float *grad,
                               float *velocity,
                               float lr,
                               float momentum,
                               float weight_decay,
                               int n,
                               cudaStream_t stream);

//------------------------------------------------------------------------------
// RMSNorm kernels
//------------------------------------------------------------------------------
void launch_rmsnorm_forward(const float *input,
                            const float *gamma,
                            float *output,
                            float *rms_cache,
                            float epsilon,
                            int rows,
                            int dim,
                            cudaStream_t stream);

void launch_rmsnorm_backward(const float *grad_output,
                              const float *input,
                              const float *gamma,
                              const float *rms_cache,
                              float *grad_input,
                              float *grad_gamma,
                              float epsilon,
                              int rows,
                              int dim,
                              cudaStream_t stream);

//------------------------------------------------------------------------------
// LayerNorm kernels
//------------------------------------------------------------------------------
void launch_layernorm_forward(const float *input,
                              const float *gamma,
                              const float *beta,
                              float *output,
                              float *mean_cache,
                              float *rstd_cache,
                              float epsilon,
                              int rows,
                              int dim,
                              cudaStream_t stream);

void launch_layernorm_backward(const float *grad_output,
                                const float *input,
                                const float *gamma,
                                const float *mean_cache,
                                const float *rstd_cache,
                                float *grad_input,
                                float *grad_gamma,
                                float *grad_beta,
                                int rows,
                                int dim,
                                cudaStream_t stream);

//------------------------------------------------------------------------------
// Multi-Head Attention kernels
//------------------------------------------------------------------------------

/**
 * Transpose dims 1 and 2 of a 4D tensor:
 * [batch, dim0, dim1, dim2] -> [batch, dim1, dim0, dim2]
 * Used for split/merge heads.
 */
void launch_transpose_0213(const float *input,
                           float *output,
                           int batch,
                           int dim0,
                           int dim1,
                           int dim2,
                           cudaStream_t stream);

/**
 * Fill upper triangle (j > i) of score matrices with -1e9.
 * Scores layout: [num_matrices, seq_len, seq_len].
 */
void launch_causal_mask_fill(float *scores,
                             int num_matrices,
                             int seq_len,
                             cudaStream_t stream);

/**
 * Fill causal mask for rectangular matrices with offset (for KV-cache).
 * Scores layout: [num_matrices, query_len, key_len].
 * Masks positions where col > (row + offset).
 * offset = previous cache length (so row 0 corresponds to position offset).
 */
void launch_causal_mask_fill_offset(float *scores,
                                    int num_matrices,
                                    int query_len,
                                    int key_len,
                                    int offset,
                                    cudaStream_t stream);

/**
 * Row-wise softmax with numerical stability (max subtraction).
 * Input/output: [num_rows, row_len]. One block per row.
 */
void launch_attention_softmax(const float *input,
                              float *output,
                              int num_rows,
                              int row_len,
                              cudaStream_t stream);

/**
 * Softmax backward for attention scores.
 * grad_input[i] = output[i] * (grad_output[i] - dot(grad_output, output))
 * Layout: [num_rows, row_len].
 */
void launch_attention_softmax_backward(const float *grad_output,
                                       const float *output,
                                       float *grad_input,
                                       int num_rows,
                                       int row_len,
                                       cudaStream_t stream);

/**
 * Zero upper triangle (j > i) of gradient scores.
 * Layout: [num_matrices, seq_len, seq_len].
 */
void launch_causal_mask_grad(float *grad_scores,
                             int num_matrices,
                             int seq_len,
                             cudaStream_t stream);

//------------------------------------------------------------------------------
// RoPE (Rotary Position Embeddings) kernels
//------------------------------------------------------------------------------

/**
 * Apply rotary position embeddings in-place (forward).
 * data: [batch_heads, seq_len, head_dim], head_dim must be even.
 */
void launch_rope_forward(float *data,
                         int batch_heads,
                         int seq_len,
                         int head_dim,
                         float base,
                         cudaStream_t stream);

/**
 * Apply rotary position embeddings with position offset (for KV-cache).
 * Positions are computed as (local_pos + pos_offset) for each row.
 */
void launch_rope_forward_offset(float *data,
                                int batch_heads,
                                int seq_len,
                                int head_dim,
                                float base,
                                int pos_offset,
                                cudaStream_t stream);

/**
 * Apply inverse rotary position embeddings in-place (backward).
 * data: [batch_heads, seq_len, head_dim], head_dim must be even.
 */
void launch_rope_backward(float *data,
                          int batch_heads,
                          int seq_len,
                          int head_dim,
                          float base,
                          cudaStream_t stream);

//------------------------------------------------------------------------------
// GQA (Grouped-Query Attention) kernels
//------------------------------------------------------------------------------

/**
 * Repeat KV heads for GQA expansion.
 * input: [batch * num_kv_heads, seq_len, head_dim]
 * output: [batch * num_heads, seq_len, head_dim]
 * where num_heads = num_kv_heads * repeat_factor
 */
void launch_gqa_repeat_kv(const float *input,
                          float *output,
                          int batch,
                          int num_kv_heads,
                          int repeat_factor,
                          int seq_len,
                          int head_dim,
                          cudaStream_t stream);

/**
 * Sum-reduce expanded gradients back to kv_heads for GQA backward.
 * input: [batch * num_heads, seq_len, head_dim]
 * output: [batch * num_kv_heads, seq_len, head_dim]
 * where num_heads = num_kv_heads * repeat_factor
 */
void launch_gqa_reduce_kv(const float *input,
                          float *output,
                          int batch,
                          int num_kv_heads,
                          int repeat_factor,
                          int seq_len,
                          int head_dim,
                          cudaStream_t stream);

//------------------------------------------------------------------------------
// KV-Cache kernels
//------------------------------------------------------------------------------

/**
 * Append new K/V rows to the KV cache (original layout).
 * Copies new_kv[batch, new_len, num_kv_heads, head_dim] to
 * cache[batch, cache_pos:cache_pos+new_len, num_kv_heads, head_dim].
 */
void launch_kv_cache_append(const float *new_kv,
                            float *cache,
                            int batch,
                            int new_len,
                            int cache_pos,
                            int max_seq_len,
                            int num_kv_heads,
                            int head_dim,
                            cudaStream_t stream);

/**
 * Append new K/V to cache (transposed layout).
 * Copies new_kv[batch_kv_heads, new_len, head_dim] to
 * cache[batch_kv_heads, cache_pos:cache_pos+new_len, head_dim].
 * This layout matches the attention format after head splitting.
 */
void launch_kv_cache_append_transposed(const float *new_kv,
                                       float *cache,
                                       int batch_kv_heads,
                                       int new_len,
                                       int cache_pos,
                                       int max_seq_len,
                                       int head_dim,
                                       cudaStream_t stream);

//------------------------------------------------------------------------------
// Gated activation kernels (SwiGLU, GeGLU, ReGLU, GLU, Bilinear)
//------------------------------------------------------------------------------

/**
 * Fused gated activation forward: output[i] = apply_op(gate[i]) * up[i]
 * The op parameter selects which gate activation to apply.
 */
void launch_gated_activation_forward(const float *gate_input,
                                     const float *up_input,
                                     float *output,
                                     int op,
                                     int n,
                                     cudaStream_t stream);

/**
 * Fused gated activation backward:
 *   grad_gate[i] = grad_output[i] * up[i] * d_activate(gate[i])
 *   grad_up[i]   = grad_output[i] * activate(gate[i])
 */
void launch_gated_activation_backward(const float *grad_output,
                                      const float *cached_gate_input,
                                      const float *cached_up_input,
                                      float *grad_gate,
                                      float *grad_up,
                                      int op,
                                      int n,
                                      cudaStream_t stream);

//------------------------------------------------------------------------------
// Embedding kernels
//------------------------------------------------------------------------------

/**
 * Gather rows from embedding table by uint32 token ID.
 * output[token_idx * dim + d] = table[token_ids[token_idx] * dim + d]
 */
void launch_embedding_lookup(const float *table,
                             const unsigned int *token_ids,
                             float *output,
                             int num_tokens,
                             int dim,
                             cudaStream_t stream);

/**
 * Gather rows from embedding table by float-encoded token ID.
 * token_id = (uint32_t)float_ids[token_idx]
 */
void launch_embedding_lookup_float(const float *table,
                                   const float *float_ids,
                                   float *output,
                                   int num_tokens,
                                   int dim,
                                   cudaStream_t stream);

/**
 * Convert float-encoded token IDs to uint32 on GPU.
 * Eliminates host roundtrip for training path.
 */
void launch_float_to_uint(const float *float_ids,
                           unsigned int *uint_ids,
                           int n,
                           cudaStream_t stream);

/**
 * Scatter-add gradients back to embedding table.
 * grad_table must be pre-zeroed. Uses atomicAdd.
 */
void launch_embedding_backward(const float *grad_output,
                               const unsigned int *token_ids,
                               float *grad_table,
                               int num_tokens,
                               int dim,
                               cudaStream_t stream);

//------------------------------------------------------------------------------
// Patch embedding kernels
//------------------------------------------------------------------------------

/**
 * Extract non-overlapping patches from BHWC images.
 * input:  [batch, height, width, channels]
 * output: [batch * num_patches, patch_flat_dim]
 */
void launch_extract_patches(const float *input,
                            float *output,
                            int batch,
                            int height,
                            int width,
                            int channels,
                            int patch_size,
                            int num_patches_h,
                            int num_patches_w,
                            int patch_flat_dim,
                            cudaStream_t stream);

/**
 * Scatter-add patch gradients back to image layout (col2im).
 * grad_input must be pre-zeroed. Uses atomicAdd.
 */
void launch_extract_patches_backward(const float *grad_patches,
                                     float *grad_input,
                                     int batch,
                                     int height,
                                     int width,
                                     int channels,
                                     int patch_size,
                                     int num_patches_h,
                                     int num_patches_w,
                                     int patch_flat_dim,
                                     cudaStream_t stream);

/**
 * Prepend CLS token at position 0 of each sequence.
 * patches: [batch, num_patches, dim], cls: [1, dim]
 * output:  [batch, num_patches+1, dim]
 */
void launch_cls_prepend(const float *patches,
                        const float *cls_token,
                        float *output,
                        int batch,
                        int num_patches,
                        int dim,
                        cudaStream_t stream);

/**
 * Split CLS gradient from patch gradients.
 * grad_output: [batch, num_patches+1, dim]
 * grad_cls: [1, dim] (summed over batch), grad_patches: [batch, num_patches, dim]
 */
void launch_cls_grad_extract(const float *grad_output,
                             float *grad_cls,
                             float *grad_patches,
                             int batch,
                             int num_patches,
                             int dim,
                             cudaStream_t stream);

//------------------------------------------------------------------------------
// Positional encoding kernels
//------------------------------------------------------------------------------

/**
 * Add positional encoding table rows to input (broadcast over batch).
 * output[b,s,d] = input[b,s,d] + pe_table[s,d]
 */
void launch_add_positional_encoding(const float *input,
                                    const float *pe_table,
                                    float *output,
                                    int batch,
                                    int seq_len,
                                    int dim,
                                    cudaStream_t stream);

/**
 * Accumulate PE gradient by summing over batch dimension.
 * grad_table[s,d] = sum_b grad_output[b,s,d]
 */
void launch_pe_table_backward(const float *grad_output,
                              float *grad_table,
                              int batch,
                              int seq_len,
                              int dim,
                              cudaStream_t stream);

//------------------------------------------------------------------------------
// Cross-entropy loss from logits (numerically stable for language modeling)
//------------------------------------------------------------------------------

/**
 * Compute per-position cross-entropy loss from raw logits.
 * Uses log-softmax formulation: loss[i] = -logits[target[i]] + log(sum(exp(logits)))
 * With numerical stability: subtracts max(logits) before exp.
 * logits: [N, vocab_size], targets: [N] (float-encoded token IDs)
 * losses: [N] (per-position loss)
 */
void launch_cross_entropy_logits_forward(const float *logits,
                                         const float *targets,
                                         float *losses,
                                         int n,
                                         int vocab_size,
                                         int ignore_index,
                                         cudaStream_t stream);

/**
 * Compute gradient of cross-entropy loss w.r.t. logits.
 * grad[i,j] = softmax(logits)[i,j] - (j == target[i] ? 1 : 0)
 * Divided by valid_count for mean reduction.
 * logits: [N, vocab_size], targets: [N]
 * grad: [N, vocab_size]
 */
void launch_cross_entropy_logits_backward(const float *logits,
                                          const float *targets,
                                          float *grad,
                                          int n,
                                          int vocab_size,
                                          int ignore_index,
                                          cudaStream_t stream);

/**
 * Reduce per-position losses to scalar mean (excluding ignored positions).
 * losses: [N], output: scalar
 */
void launch_cross_entropy_reduce_mean(const float *losses,
                                      const float *targets,
                                      float *output,
                                      int n,
                                      int ignore_index,
                                      cudaStream_t stream);

//------------------------------------------------------------------------------
// SiLU backward kernel
//------------------------------------------------------------------------------

/**
 * SiLU backward: grad_input = grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
 */
void launch_silu_backward(const float *grad_output,
                          const float *input,
                          float *grad_input,
                          int n,
                          cudaStream_t stream);

//------------------------------------------------------------------------------
// Reduction and normalization kernels
//------------------------------------------------------------------------------

/**
 * Sum along axis 0 (over batch): output[d] = sum_b input[b,d]
 */
void launch_sum_axis0(const float *input,
                      float *output,
                      int batch,
                      int dim,
                      cudaStream_t stream);

/**
 * Sum along axis 1 (over dim): output[b] = sum_d input[b,d]
 */
void launch_sum_axis1(const float *input,
                      float *output,
                      int batch,
                      int dim,
                      cudaStream_t stream);

/**
 * Log-sum-exp along last axis: output[b] = log(sum_d exp(input[b,d]))
 * With numerical stability (max subtraction).
 */
void launch_logsumexp(const float *input,
                      float *output,
                      int batch,
                      int dim,
                      cudaStream_t stream);

/**
 * Normalize each row to sum to 1: output[b,d] = input[b,d] / sum_d(input[b,:])
 */
void launch_normalize_rows(const float *input,
                           float *output,
                           int batch,
                           int dim,
                           cudaStream_t stream);

//------------------------------------------------------------------------------
// Top-K and scatter operations
//------------------------------------------------------------------------------

/**
 * Select top-k values and indices per row.
 * input: [batch, dim], indices: [batch, k] (int32), values: [batch, k]
 */
void launch_topk(const float *input,
                 int *indices,
                 float *values,
                 int batch,
                 int dim,
                 int k,
                 cudaStream_t stream);

/**
 * Scatter-add values to output using indices.
 * output[b, indices[b,k]] += values[b,k]
 * values: [batch, k], indices: [batch, k], output: [batch, dim]
 */
void launch_scatter_add(const float *values,
                        const int *indices,
                        float *output,
                        int batch,
                        int k,
                        int dim,
                        cudaStream_t stream);

//------------------------------------------------------------------------------
// FlashAttention kernels
//------------------------------------------------------------------------------

/**
 * FlashAttention-2 forward pass.
 * Computes scaled dot-product attention without materializing the full attention matrix.
 * Uses online softmax and tiled computation for memory efficiency.
 *
 * Q, K, V: [batch_heads, seq_len, head_dim]
 * O: [batch_heads, seq_len, head_dim] - output
 * L: [batch_heads, seq_len] - logsumexp values (needed for backward)
 *
 * Supports causal masking when causal=true.
 */
void launch_flash_attention_forward(const float *Q,
                                    const float *K,
                                    const float *V,
                                    float *O,
                                    float *L,
                                    int batch_heads,
                                    int seq_len,
                                    int head_dim,
                                    float scale,
                                    int causal,
                                    cudaStream_t stream);

/**
 * FlashAttention-2 backward pass.
 * Computes gradients dQ, dK, dV without materializing the full attention matrix.
 * Recomputes attention on-the-fly using stored logsumexp values.
 *
 * Q, K, V, O: [batch_heads, seq_len, head_dim] - from forward
 * dO: [batch_heads, seq_len, head_dim] - gradient of output
 * L: [batch_heads, seq_len] - logsumexp from forward
 * dQ, dK, dV: [batch_heads, seq_len, head_dim] - output gradients
 */
void launch_flash_attention_backward(const float *Q,
                                     const float *K,
                                     const float *V,
                                     const float *O,
                                     const float *dO,
                                     const float *L,
                                     float *dQ,
                                     float *dK,
                                     float *dV,
                                     int batch_heads,
                                     int seq_len,
                                     int head_dim,
                                     float scale,
                                     int causal,
                                     cudaStream_t stream);

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
void launch_moe_dispatch(const float *input,
                         const float *expert_indices,
                         const int *dispatch_map,
                         float *expert_buffer,
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
void launch_moe_combine(const float *expert_buffer,
                        const float *expert_indices,
                        const float *expert_weights,
                        const int *dispatch_map,
                        const int *expert_offsets,
                        float *output,
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
void launch_moe_topk_gating(const float *router_logits,
                            float *expert_indices,
                            float *expert_weights,
                            float *router_probs,
                            int num_tokens,
                            int num_experts,
                            int top_k,
                            cudaStream_t stream);

/**
 * Count tokens per expert using atomic operations.
 * expert_indices: [num_tokens, top_k] - which experts each token routes to (as floats)
 * expert_counts: [num_experts] - output counts per expert (must be zeroed first)
 */
void launch_moe_count_per_expert(const float *expert_indices,
                                 int *expert_counts,
                                 int num_tokens,
                                 int num_experts,
                                 int top_k,
                                 int capacity_per_expert,
                                 cudaStream_t stream);

//------------------------------------------------------------------------------
// Data type conversion kernels
//------------------------------------------------------------------------------

/**
 * Convert FP32 array to FP16 (__half) on device.
 * input: [n] float, output: [n] half (2 bytes each)
 */
void launch_convert_fp32_to_fp16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream);

/**
 * Convert FP16 (__half) array to FP32 on device.
 * input: [n] half, output: [n] float
 */
void launch_convert_fp16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream);

/**
 * Convert FP32 array to BF16 (__nv_bfloat16) on device.
 * input: [n] float, output: [n] bfloat16 (2 bytes each)
 */
void launch_convert_fp32_to_bf16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream);

/**
 * Convert BF16 (__nv_bfloat16) array to FP32 on device.
 * input: [n] bfloat16, output: [n] float
 */
void launch_convert_bf16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream);

//------------------------------------------------------------------------------
// INT8 conversion kernels
//------------------------------------------------------------------------------

/**
 * Convert FP32 to INT8 on device. Clamps to [-127,127] and casts.
 * input: [n] float, output: [n] int8_t
 */
void launch_convert_fp32_to_int8(const float *input,
                                  void *output,
                                  int n,
                                  cudaStream_t stream);

/**
 * Convert INT8 to FP32 on device. Simple cast.
 * input: [n] int8_t, output: [n] float
 */
void launch_convert_int8_to_fp32(const void *input,
                                  float *output,
                                  int n,
                                  cudaStream_t stream);

//------------------------------------------------------------------------------
// INT4 quantization kernels (symmetric per-group with FP16 scales)
//------------------------------------------------------------------------------

/**
 * Dequantize INT4 packed data to FP32.
 * Packed format: 2 elements per byte (low nibble=even, high nibble=odd).
 * Symmetric per-group: val = int4_val * scale[group_idx]
 *
 * packed_data: [(num_elements+1)/2] bytes of packed INT4
 * scales: [num_groups] FP16 per-group scales
 * output: [num_elements] float
 * num_elements: total number of logical elements
 * group_size: elements per quantization group
 */
void launch_dequantize_int4(const void *packed_data,
                             const void *scales,
                             float *output,
                             int num_elements,
                             int group_size,
                             cudaStream_t stream);

/**
 * Quantize FP32 to INT4 packed format with per-group FP16 scales.
 * Computes scales from data, packs 2 elements per byte.
 *
 * input: [num_elements] float
 * packed_output: [(num_elements+1)/2] bytes
 * scales_output: [num_groups] FP16
 * num_elements: total number of logical elements
 * group_size: elements per quantization group
 */
void launch_quantize_to_int4(const float *input,
                              void *packed_output,
                              void *scales_output,
                              int num_elements,
                              int group_size,
                              cudaStream_t stream);

//------------------------------------------------------------------------------
// Tensor slice and concatenation kernels
//------------------------------------------------------------------------------

/**
 * Slice along the last dimension of a 2D (logically) tensor.
 * input: [rows, in_cols], output: [rows, out_cols]
 * Copies columns [col_start, col_start + out_cols) from each row.
 */
void launch_slice_last_dim(const float *input,
                           float *output,
                           int rows,
                           int in_cols,
                           int col_start,
                           int out_cols,
                           cudaStream_t stream);

/**
 * Scatter a slice back into the last dimension (backward of slice).
 * grad_input: [rows, in_cols] (must be pre-zeroed or pre-filled)
 * grad_output: [rows, out_cols]
 * Adds grad_output into grad_input[:, col_start:col_start+out_cols].
 */
void launch_slice_last_dim_backward(const float *grad_output,
                                    float *grad_input,
                                    int rows,
                                    int in_cols,
                                    int col_start,
                                    int out_cols,
                                    cudaStream_t stream);

/**
 * Concatenate two tensors along the last dimension.
 * a: [rows, cols_a], b: [rows, cols_b], output: [rows, cols_a + cols_b]
 */
void launch_concat_last_dim(const float *a,
                            const float *b,
                            float *output,
                            int rows,
                            int cols_a,
                            int cols_b,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // CAIF_CUDA_KERNELS_H

