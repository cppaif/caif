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
// CAIF - CPU stub implementations for CUDA kernel launchers
//
// Compiled only in non-CUDA builds. Provides empty no-op implementations so
// the linker can resolve launch_* symbols from caif_cuda_kernels.h.
//
// IMPORTANT: Every function declared in caif_cuda_kernels.h must have a
// corresponding stub in this file. When adding a new launch_* function to
// the header, add a matching empty stub here or CPU-only builds will fail
// with undefined reference errors at link time.
//------------------------------------------------------------------------------

#ifndef USE_CAIF_CUDA

#include "caif_cuda_kernels.h"

extern "C"
{

void launch_relu_forward(const float *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_relu_backward(const float *grad_output, const float *input, float *grad_input, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)grad_input;
  (void)n;
  (void)stream;
}

void launch_sigmoid_forward(const float *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_sigmoid_backward(const float *grad_output, const float *output, float *grad_input, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)output;
  (void)grad_input;
  (void)n;
  (void)stream;
}

void launch_tanh_forward(const float *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_tanh_backward(const float *grad_output, const float *output, float *grad_input, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)output;
  (void)grad_input;
  (void)n;
  (void)stream;
}

void launch_leaky_relu_forward(const float *input, float *output, float alpha, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)alpha;
  (void)n;
  (void)stream;
}

void launch_leaky_relu_backward(const float *grad_output, const float *input, float *grad_input, float alpha, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)grad_input;
  (void)alpha;
  (void)n;
  (void)stream;
}

void launch_elu_forward(const float *input, float *output, float alpha, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)alpha;
  (void)n;
  (void)stream;
}

void launch_elu_backward(const float *grad_output, const float *input, const float *output, float *grad_input, float alpha, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)output;
  (void)grad_input;
  (void)alpha;
  (void)n;
  (void)stream;
}

void launch_gelu_forward(const float *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_gelu_backward(const float *grad_output, const float *input, float *grad_input, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)grad_input;
  (void)n;
  (void)stream;
}

void launch_swish_forward(const float *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_swish_backward(const float *grad_output, const float *input, const float *output, float *grad_input, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)output;
  (void)grad_input;
  (void)n;
  (void)stream;
}

void launch_elementwise_add(const float *a, const float *b, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)b;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_elementwise_add_scalar(const float *a, float scalar, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)scalar;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_bias_add_2d(const float *input, const float *bias, float *output, int batch, int units, cudaStream_t stream)
{
  (void)input;
  (void)bias;
  (void)output;
  (void)batch;
  (void)units;
  (void)stream;
}

void launch_bias_grad_2d(const float *grad_output, float *bias_grad, int batch, int units, cudaStream_t stream)
{
  (void)grad_output;
  (void)bias_grad;
  (void)batch;
  (void)units;
  (void)stream;
}

void launch_elementwise_sub(const float *a, const float *b, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)b;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_elementwise_sub_scalar(const float *a, float scalar, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)scalar;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_elementwise_mul(const float *a, const float *b, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)b;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_elementwise_mul_scalar(const float *a, float scalar, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)scalar;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_elementwise_div(const float *a, const float *b, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)b;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_elementwise_div_scalar(const float *a, float scalar, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)scalar;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_elementwise_sqrt(const float *a, float *result, int n, cudaStream_t stream)
{
  (void)a;
  (void)result;
  (void)n;
  (void)stream;
}

void launch_reduction_sum(const float *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_cross_entropy_loss(const float *predictions, const float *targets, float *loss, float epsilon, int batch_size, int num_classes, cudaStream_t stream)
{
  (void)predictions;
  (void)targets;
  (void)loss;
  (void)epsilon;
  (void)batch_size;
  (void)num_classes;
  (void)stream;
}

void launch_cross_entropy_gradient(const float *predictions, const float *targets, float *gradient, float epsilon, int batch_size, int n, cudaStream_t stream)
{
  (void)predictions;
  (void)targets;
  (void)gradient;
  (void)epsilon;
  (void)batch_size;
  (void)n;
  (void)stream;
}

void launch_cross_entropy_loss_index(const float *predictions, const int *target_indices, float *loss, float epsilon, int batch_size, int num_classes, cudaStream_t stream)
{
  (void)predictions;
  (void)target_indices;
  (void)loss;
  (void)epsilon;
  (void)batch_size;
  (void)num_classes;
  (void)stream;
}

void launch_cross_entropy_gradient_index(const float *predictions, const int *target_indices, float *gradient, float epsilon, int batch_size, int num_classes, cudaStream_t stream)
{
  (void)predictions;
  (void)target_indices;
  (void)gradient;
  (void)epsilon;
  (void)batch_size;
  (void)num_classes;
  (void)stream;
}

void launch_mse_loss(const float *predictions, const float *targets, float *loss, int n, cudaStream_t stream)
{
  (void)predictions;
  (void)targets;
  (void)loss;
  (void)n;
  (void)stream;
}

void launch_mse_gradient(const float *predictions, const float *targets, float *gradient, int n, cudaStream_t stream)
{
  (void)predictions;
  (void)targets;
  (void)gradient;
  (void)n;
  (void)stream;
}

void launch_fused_adam(float *param, const float *grad, float *m, float *v, float lr, float beta1, float beta2, float epsilon, float weight_decay, float bias_correction1, float bias_correction2, int n, cudaStream_t stream)
{
  (void)param;
  (void)grad;
  (void)m;
  (void)v;
  (void)lr;
  (void)beta1;
  (void)beta2;
  (void)epsilon;
  (void)weight_decay;
  (void)bias_correction1;
  (void)bias_correction2;
  (void)n;
  (void)stream;
}

void launch_fused_adam_clipped(float *param, const float *grad, float *m, float *v, float lr, float beta1, float beta2, float epsilon, float weight_decay, float bias_correction1, float bias_correction2, float grad_scale, int n, cudaStream_t stream)
{
  (void)param;
  (void)grad;
  (void)m;
  (void)v;
  (void)lr;
  (void)beta1;
  (void)beta2;
  (void)epsilon;
  (void)weight_decay;
  (void)bias_correction1;
  (void)bias_correction2;
  (void)grad_scale;
  (void)n;
  (void)stream;
}

void launch_fused_sgd_momentum(float *param, const float *grad, float *velocity, float lr, float momentum, float weight_decay, int n, cudaStream_t stream)
{
  (void)param;
  (void)grad;
  (void)velocity;
  (void)lr;
  (void)momentum;
  (void)weight_decay;
  (void)n;
  (void)stream;
}

void launch_rmsnorm_forward(const float *input, const float *gamma, float *output, float *rms_cache, float epsilon, int rows, int dim, cudaStream_t stream)
{
  (void)input;
  (void)gamma;
  (void)output;
  (void)rms_cache;
  (void)epsilon;
  (void)rows;
  (void)dim;
  (void)stream;
}

void launch_rmsnorm_backward(const float *grad_output, const float *input, const float *gamma, const float *rms_cache, float *grad_input, float *grad_gamma, float epsilon, int rows, int dim, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)gamma;
  (void)rms_cache;
  (void)grad_input;
  (void)grad_gamma;
  (void)epsilon;
  (void)rows;
  (void)dim;
  (void)stream;
}

void launch_layernorm_forward(const float *input, const float *gamma, const float *beta, float *output, float *mean_cache, float *rstd_cache, float epsilon, int rows, int dim, cudaStream_t stream)
{
  (void)input;
  (void)gamma;
  (void)beta;
  (void)output;
  (void)mean_cache;
  (void)rstd_cache;
  (void)epsilon;
  (void)rows;
  (void)dim;
  (void)stream;
}

void launch_layernorm_backward(const float *grad_output, const float *input, const float *gamma, const float *mean_cache, const float *rstd_cache, float *grad_input, float *grad_gamma, float *grad_beta, int rows, int dim, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)gamma;
  (void)mean_cache;
  (void)rstd_cache;
  (void)grad_input;
  (void)grad_gamma;
  (void)grad_beta;
  (void)rows;
  (void)dim;
  (void)stream;
}

void launch_transpose_0213(const float *input, float *output, int batch, int dim0, int dim1, int dim2, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)dim0;
  (void)dim1;
  (void)dim2;
  (void)stream;
}

void launch_causal_mask_fill(float *scores, int num_matrices, int seq_len, cudaStream_t stream)
{
  (void)scores;
  (void)num_matrices;
  (void)seq_len;
  (void)stream;
}

void launch_causal_mask_fill_offset(float *scores, int num_matrices, int query_len, int key_len, int offset, cudaStream_t stream)
{
  (void)scores;
  (void)num_matrices;
  (void)query_len;
  (void)key_len;
  (void)offset;
  (void)stream;
}

void launch_attention_softmax(const float *input, float *output, int num_rows, int row_len, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)num_rows;
  (void)row_len;
  (void)stream;
}

void launch_attention_softmax_backward(const float *grad_output, const float *output, float *grad_input, int num_rows, int row_len, cudaStream_t stream)
{
  (void)grad_output;
  (void)output;
  (void)grad_input;
  (void)num_rows;
  (void)row_len;
  (void)stream;
}

void launch_causal_mask_grad(float *grad_scores, int num_matrices, int seq_len, cudaStream_t stream)
{
  (void)grad_scores;
  (void)num_matrices;
  (void)seq_len;
  (void)stream;
}

void launch_rope_forward(float *data, int batch_heads, int seq_len, int head_dim, float base, cudaStream_t stream)
{
  (void)data;
  (void)batch_heads;
  (void)seq_len;
  (void)head_dim;
  (void)base;
  (void)stream;
}

void launch_rope_forward_offset(float *data, int batch_heads, int seq_len, int head_dim, float base, int pos_offset, cudaStream_t stream)
{
  (void)data;
  (void)batch_heads;
  (void)seq_len;
  (void)head_dim;
  (void)base;
  (void)pos_offset;
  (void)stream;
}

void launch_rope_backward(float *data, int batch_heads, int seq_len, int head_dim, float base, cudaStream_t stream)
{
  (void)data;
  (void)batch_heads;
  (void)seq_len;
  (void)head_dim;
  (void)base;
  (void)stream;
}

void launch_gqa_repeat_kv(const float *input, float *output, int batch, int num_kv_heads, int repeat_factor, int seq_len, int head_dim, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)num_kv_heads;
  (void)repeat_factor;
  (void)seq_len;
  (void)head_dim;
  (void)stream;
}

void launch_gqa_reduce_kv(const float *input, float *output, int batch, int num_kv_heads, int repeat_factor, int seq_len, int head_dim, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)num_kv_heads;
  (void)repeat_factor;
  (void)seq_len;
  (void)head_dim;
  (void)stream;
}

void launch_kv_cache_append(const float *new_kv, float *cache, int batch, int new_len, int cache_pos, int max_seq_len, int num_kv_heads, int head_dim, cudaStream_t stream)
{
  (void)new_kv;
  (void)cache;
  (void)batch;
  (void)new_len;
  (void)cache_pos;
  (void)max_seq_len;
  (void)num_kv_heads;
  (void)head_dim;
  (void)stream;
}

void launch_kv_cache_append_transposed(const float *new_kv, float *cache, int batch_kv_heads, int new_len, int cache_pos, int max_seq_len, int head_dim, cudaStream_t stream)
{
  (void)new_kv;
  (void)cache;
  (void)batch_kv_heads;
  (void)new_len;
  (void)cache_pos;
  (void)max_seq_len;
  (void)head_dim;
  (void)stream;
}

void launch_gated_activation_forward(const float *gate_input, const float *up_input, float *output, int op, int n, cudaStream_t stream)
{
  (void)gate_input;
  (void)up_input;
  (void)output;
  (void)op;
  (void)n;
  (void)stream;
}

void launch_gated_activation_backward(const float *grad_output, const float *cached_gate_input, const float *cached_up_input, float *grad_gate, float *grad_up, int op, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)cached_gate_input;
  (void)cached_up_input;
  (void)grad_gate;
  (void)grad_up;
  (void)op;
  (void)n;
  (void)stream;
}

void launch_embedding_lookup(const float *table, const unsigned int *token_ids, float *output, int num_tokens, int dim, cudaStream_t stream)
{
  (void)table;
  (void)token_ids;
  (void)output;
  (void)num_tokens;
  (void)dim;
  (void)stream;
}

void launch_embedding_lookup_float(const float *table, const float *float_ids, float *output, int num_tokens, int dim, cudaStream_t stream)
{
  (void)table;
  (void)float_ids;
  (void)output;
  (void)num_tokens;
  (void)dim;
  (void)stream;
}

void launch_float_to_uint(const float *float_ids, unsigned int *uint_ids, int n, cudaStream_t stream)
{
  (void)float_ids;
  (void)uint_ids;
  (void)n;
  (void)stream;
}

void launch_embedding_backward(const float *grad_output, const unsigned int *token_ids, float *grad_table, int num_tokens, int dim, cudaStream_t stream)
{
  (void)grad_output;
  (void)token_ids;
  (void)grad_table;
  (void)num_tokens;
  (void)dim;
  (void)stream;
}

void launch_extract_patches(const float *input, float *output, int batch, int height, int width, int channels, int patch_size, int num_patches_h, int num_patches_w, int patch_flat_dim, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)height;
  (void)width;
  (void)channels;
  (void)patch_size;
  (void)num_patches_h;
  (void)num_patches_w;
  (void)patch_flat_dim;
  (void)stream;
}

void launch_extract_patches_backward(const float *grad_patches, float *grad_input, int batch, int height, int width, int channels, int patch_size, int num_patches_h, int num_patches_w, int patch_flat_dim, cudaStream_t stream)
{
  (void)grad_patches;
  (void)grad_input;
  (void)batch;
  (void)height;
  (void)width;
  (void)channels;
  (void)patch_size;
  (void)num_patches_h;
  (void)num_patches_w;
  (void)patch_flat_dim;
  (void)stream;
}

void launch_cls_prepend(const float *patches, const float *cls_token, float *output, int batch, int num_patches, int dim, cudaStream_t stream)
{
  (void)patches;
  (void)cls_token;
  (void)output;
  (void)batch;
  (void)num_patches;
  (void)dim;
  (void)stream;
}

void launch_cls_grad_extract(const float *grad_output, float *grad_cls, float *grad_patches, int batch, int num_patches, int dim, cudaStream_t stream)
{
  (void)grad_output;
  (void)grad_cls;
  (void)grad_patches;
  (void)batch;
  (void)num_patches;
  (void)dim;
  (void)stream;
}

void launch_add_positional_encoding(const float *input, const float *pe_table, float *output, int batch, int seq_len, int dim, cudaStream_t stream)
{
  (void)input;
  (void)pe_table;
  (void)output;
  (void)batch;
  (void)seq_len;
  (void)dim;
  (void)stream;
}

void launch_pe_table_backward(const float *grad_output, float *grad_table, int batch, int seq_len, int dim, cudaStream_t stream)
{
  (void)grad_output;
  (void)grad_table;
  (void)batch;
  (void)seq_len;
  (void)dim;
  (void)stream;
}

void launch_cross_entropy_logits_forward(const float *logits, const float *targets, float *losses, int n, int vocab_size, int ignore_index, cudaStream_t stream)
{
  (void)logits;
  (void)targets;
  (void)losses;
  (void)n;
  (void)vocab_size;
  (void)ignore_index;
  (void)stream;
}

void launch_cross_entropy_logits_backward(const float *logits, const float *targets, float *grad, int n, int vocab_size, int ignore_index, float scale, cudaStream_t stream)
{
  (void)logits;
  (void)targets;
  (void)grad;
  (void)n;
  (void)vocab_size;
  (void)ignore_index;
  (void)scale;
  (void)stream;
}

void launch_cross_entropy_reduce_mean(const float *losses, const float *targets, float *output, int n, int ignore_index, cudaStream_t stream)
{
  (void)losses;
  (void)targets;
  (void)output;
  (void)n;
  (void)ignore_index;
  (void)stream;
}

void launch_cross_entropy_fused(const float *logits,
                                const float *targets,
                                float *losses,
                                float *grad,
                                float *result,
                                int n,
                                int vocab_size,
                                int ignore_index,
                                cudaStream_t stream)
{
  (void)logits;
  (void)targets;
  (void)losses;
  (void)grad;
  (void)result;
  (void)n;
  (void)vocab_size;
  (void)ignore_index;
  (void)stream;
}

void launch_silu_backward(const float *grad_output, const float *input, float *grad_input, int n, cudaStream_t stream)
{
  (void)grad_output;
  (void)input;
  (void)grad_input;
  (void)n;
  (void)stream;
}

void launch_sum_axis0(const float *input, float *output, int batch, int dim, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)dim;
  (void)stream;
}

void launch_sum_axis1(const float *input, float *output, int batch, int dim, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)dim;
  (void)stream;
}

void launch_sum_of_squares(const float *input,
                           float *output,
                           int n,
                           cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_logsumexp(const float *input, float *output, int batch, int dim, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)dim;
  (void)stream;
}

void launch_normalize_rows(const float *input, float *output, int batch, int dim, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)batch;
  (void)dim;
  (void)stream;
}

void launch_topk(const float *input, int *indices, float *values, int batch, int dim, int k, cudaStream_t stream)
{
  (void)input;
  (void)indices;
  (void)values;
  (void)batch;
  (void)dim;
  (void)k;
  (void)stream;
}

void launch_scatter_add(const float *values, const int *indices, float *output, int batch, int k, int dim, cudaStream_t stream)
{
  (void)values;
  (void)indices;
  (void)output;
  (void)batch;
  (void)k;
  (void)dim;
  (void)stream;
}

void launch_flash_attention_forward(const float *Q, const float *K, const float *V, float *O, float *L, int batch_heads, int seq_len, int head_dim, float scale, int causal, cudaStream_t stream)
{
  (void)Q;
  (void)K;
  (void)V;
  (void)O;
  (void)L;
  (void)batch_heads;
  (void)seq_len;
  (void)head_dim;
  (void)scale;
  (void)causal;
  (void)stream;
}

void launch_flash_attention_backward(const float *Q, const float *K, const float *V, const float *O, const float *dO, const float *L, float *dQ, float *dK, float *dV, int batch_heads, int seq_len, int head_dim, float scale, int causal, cudaStream_t stream)
{
  (void)Q;
  (void)K;
  (void)V;
  (void)O;
  (void)dO;
  (void)L;
  (void)dQ;
  (void)dK;
  (void)dV;
  (void)batch_heads;
  (void)seq_len;
  (void)head_dim;
  (void)scale;
  (void)causal;
  (void)stream;
}

void launch_moe_dispatch(const float *input, const float *expert_indices, const int *dispatch_map, float *expert_buffer, const int *expert_offsets, int num_tokens, int dim, int top_k, cudaStream_t stream)
{
  (void)input;
  (void)expert_indices;
  (void)dispatch_map;
  (void)expert_buffer;
  (void)expert_offsets;
  (void)num_tokens;
  (void)dim;
  (void)top_k;
  (void)stream;
}

void launch_moe_combine(const float *expert_buffer, const float *expert_indices, const float *expert_weights, const int *dispatch_map, const int *expert_offsets, float *output, int num_tokens, int dim, int top_k, cudaStream_t stream)
{
  (void)expert_buffer;
  (void)expert_indices;
  (void)expert_weights;
  (void)dispatch_map;
  (void)expert_offsets;
  (void)output;
  (void)num_tokens;
  (void)dim;
  (void)top_k;
  (void)stream;
}

void launch_moe_topk_gating(const float *router_logits, float *expert_indices, float *expert_weights, float *router_probs, int num_tokens, int num_experts, int top_k, cudaStream_t stream)
{
  (void)router_logits;
  (void)expert_indices;
  (void)expert_weights;
  (void)router_probs;
  (void)num_tokens;
  (void)num_experts;
  (void)top_k;
  (void)stream;
}

void launch_moe_count_per_expert(const float *expert_indices, int *expert_counts, int num_tokens, int num_experts, int top_k, int capacity_per_expert, cudaStream_t stream)
{
  (void)expert_indices;
  (void)expert_counts;
  (void)num_tokens;
  (void)num_experts;
  (void)top_k;
  (void)capacity_per_expert;
  (void)stream;
}

void launch_convert_fp32_to_fp16(const float *input, void *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_convert_fp16_to_fp32(const void *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_convert_fp32_to_bf16(const float *input, void *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_convert_bf16_to_fp32(const void *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_convert_fp32_to_int8(const float *input, void *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_convert_int8_to_fp32(const void *input, float *output, int n, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)n;
  (void)stream;
}

void launch_dequantize_int4(const void *packed_data, const void *scales, float *output, int num_elements, int group_size, cudaStream_t stream)
{
  (void)packed_data;
  (void)scales;
  (void)output;
  (void)num_elements;
  (void)group_size;
  (void)stream;
}

void launch_quantize_to_int4(const float *input, void *packed_output, void *scales_output, int num_elements, int group_size, cudaStream_t stream)
{
  (void)input;
  (void)packed_output;
  (void)scales_output;
  (void)num_elements;
  (void)group_size;
  (void)stream;
}

void launch_slice_last_dim(const float *input, float *output, int rows, int in_cols, int col_start, int out_cols, cudaStream_t stream)
{
  (void)input;
  (void)output;
  (void)rows;
  (void)in_cols;
  (void)col_start;
  (void)out_cols;
  (void)stream;
}

void launch_slice_last_dim_backward(const float *grad_output, float *grad_input, int rows, int in_cols, int col_start, int out_cols, cudaStream_t stream)
{
  (void)grad_output;
  (void)grad_input;
  (void)rows;
  (void)in_cols;
  (void)col_start;
  (void)out_cols;
  (void)stream;
}

void launch_concat_last_dim(const float *a, const float *b, float *output, int rows, int cols_a, int cols_b, cudaStream_t stream)
{
  (void)a;
  (void)b;
  (void)output;
  (void)rows;
  (void)cols_a;
  (void)cols_b;
  (void)stream;
}

}  // extern "C"

#endif  // !USE_CAIF_CUDA

