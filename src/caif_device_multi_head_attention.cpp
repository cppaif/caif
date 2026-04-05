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

#include "caif_device_multi_head_attention.h"
#include "caif_device_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <random>

namespace instance
{

CAIF_DeviceMultiHeadAttention::CAIF_DeviceMultiHeadAttention(
                               const AttentionConfig_t &config,
                               CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                        _config(config),
                                                        _use_projections(false),
                                                        _w_q(),
                                                        _w_k(),
                                                        _w_v(),
                                                        _w_o(),
                                                        _grad_w_q(),
                                                        _grad_w_k(),
                                                        _grad_w_v(),
                                                        _grad_w_o(),
                                                        _cached_input(),
                                                        _cached_q_heads(),
                                                        _cached_k_heads(),
                                                        _cached_v_heads(),
                                                        _cached_attn(),
                                                        _cached_concat(),
                                                        _cached_logsumexp(),
                                                        _cached_output(),
                                                        _cached_batch(0),
                                                        _cached_seq_len(0),
                                                        _kv_cache_k(),
                                                        _kv_cache_v(),
                                                        _kv_cache_len(0),
                                                        _kv_cache_max_len(0),
                                                        _kv_cache_batch(0),
                                                        _kv_cache_enabled(false),
                                                        _use_flash_attention(true)
{
  try
  {
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: dim must be > 0");
    }
    if(config.num_heads==0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: num_heads must be > 0");
    }
    if(config.num_kv_heads==0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: num_kv_heads must be > 0");
    }
    if(config.dim%config.num_heads!=0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: dim must be divisible by num_heads");
    }
    if(config.num_heads%config.num_kv_heads!=0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: num_heads must be divisible by num_kv_heads");
    }
    if(config.use_rope==true&&config.head_dim%2!=0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: head_dim must be even when use_rope is true");
    }

    const uint32_t qk_dim=_config.num_heads*_config.head_dim;
    const uint32_t kv_dim=_config.num_kv_heads*_config.head_dim;

    // Allocate weight matrices
    _w_q=CAIF_DeviceTensor::Zeros({_config.dim,qk_dim},stream);
    _w_k=CAIF_DeviceTensor::Zeros({_config.dim,kv_dim},stream);
    _w_v=CAIF_DeviceTensor::Zeros({_config.dim,kv_dim},stream);
    _w_o=CAIF_DeviceTensor::Zeros({qk_dim,_config.dim},stream);

    // Allocate gradient matrices
    _grad_w_q=CAIF_DeviceTensor::Zeros({_config.dim,qk_dim},stream);
    _grad_w_k=CAIF_DeviceTensor::Zeros({_config.dim,kv_dim},stream);
    _grad_w_v=CAIF_DeviceTensor::Zeros({_config.dim,kv_dim},stream);
    _grad_w_o=CAIF_DeviceTensor::Zeros({qk_dim,_config.dim},stream);

    // Initialize weights
    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceMultiHeadAttention::CAIF_DeviceMultiHeadAttention(
  const AttentionConfig_t &config,
  MHAProjections_t projections,
  CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                           _config(config),
                           _projections(std::move(projections)),
                           _use_projections(true),
                           _w_q(),
                           _w_k(),
                           _w_v(),
                           _w_o(),
                           _grad_w_q(),
                           _grad_w_k(),
                           _grad_w_v(),
                           _grad_w_o(),
                           _cached_input(),
                           _cached_q_heads(),
                           _cached_k_heads(),
                           _cached_v_heads(),
                           _cached_attn(),
                           _cached_concat(),
                           _cached_logsumexp(),
                           _cached_output(),
                           _cached_batch(0),
                           _cached_seq_len(0),
                           _kv_cache_k(),
                           _kv_cache_v(),
                           _kv_cache_len(0),
                           _kv_cache_max_len(0),
                           _kv_cache_batch(0),
                           _kv_cache_enabled(false),
                           _use_flash_attention(true)
{
  try
  {
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: dim must be > 0");
    }
    if(config.num_heads==0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: num_heads must be > 0");
    }
    if(config.num_kv_heads==0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: num_kv_heads must be > 0");
    }
    if(config.num_heads%config.num_kv_heads!=0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: num_heads must be divisible by num_kv_heads");
    }
    if(config.use_rope==true&&config.head_dim%2!=0)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: head_dim must be even when use_rope is true");
    }

    // Projections mode: weight allocation and initialization are handled
    // by the projection layers (FrozenLinear/LoRAAdapter). No internal
    // weight matrices needed.
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceMultiHeadAttention::CAIF_DeviceMultiHeadAttention(
  CAIF_DeviceMultiHeadAttention &&other):CAIF_DeviceLayer(std::move(other)),
                                         _config(other._config),
                                         _projections(std::move(other._projections)),
                                         _use_projections(other._use_projections),
                                         _w_q(std::move(other._w_q)),
                                         _w_k(std::move(other._w_k)),
                                         _w_v(std::move(other._w_v)),
                                         _w_o(std::move(other._w_o)),
                                         _grad_w_q(std::move(other._grad_w_q)),
                                         _grad_w_k(std::move(other._grad_w_k)),
                                         _grad_w_v(std::move(other._grad_w_v)),
                                         _grad_w_o(std::move(other._grad_w_o)),
                                         _cached_input(std::move(other._cached_input)),
                                         _cached_q_heads(std::move(other._cached_q_heads)),
                                         _cached_k_heads(std::move(other._cached_k_heads)),
                                         _cached_v_heads(std::move(other._cached_v_heads)),
                                         _cached_attn(std::move(other._cached_attn)),
                                         _cached_concat(std::move(other._cached_concat)),
                                         _cached_logsumexp(std::move(other._cached_logsumexp)),
                                         _cached_output(std::move(other._cached_output)),
                                         _cached_batch(other._cached_batch),
                                         _cached_seq_len(other._cached_seq_len),
                                         _kv_cache_k(std::move(other._kv_cache_k)),
                                         _kv_cache_v(std::move(other._kv_cache_v)),
                                         _kv_cache_len(other._kv_cache_len),
                                         _kv_cache_max_len(other._kv_cache_max_len),
                                         _kv_cache_batch(other._kv_cache_batch),
                                         _kv_cache_enabled(other._kv_cache_enabled),
                                         _use_flash_attention(other._use_flash_attention)
{
}

CAIF_DeviceMultiHeadAttention &CAIF_DeviceMultiHeadAttention::operator=(
                               CAIF_DeviceMultiHeadAttention &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _config=other._config;
      _projections=std::move(other._projections);
      _use_projections=other._use_projections;
      _w_q=std::move(other._w_q);
      _w_k=std::move(other._w_k);
      _w_v=std::move(other._w_v);
      _w_o=std::move(other._w_o);
      _grad_w_q=std::move(other._grad_w_q);
      _grad_w_k=std::move(other._grad_w_k);
      _grad_w_v=std::move(other._grad_w_v);
      _grad_w_o=std::move(other._grad_w_o);
      _cached_input=std::move(other._cached_input);
      _cached_q_heads=std::move(other._cached_q_heads);
      _cached_k_heads=std::move(other._cached_k_heads);
      _cached_v_heads=std::move(other._cached_v_heads);
      _cached_attn=std::move(other._cached_attn);
      _cached_concat=std::move(other._cached_concat);
      _cached_logsumexp=std::move(other._cached_logsumexp);
      _cached_output=std::move(other._cached_output);
      _cached_batch=other._cached_batch;
      _cached_seq_len=other._cached_seq_len;
      _kv_cache_k=std::move(other._kv_cache_k);
      _kv_cache_v=std::move(other._kv_cache_v);
      _kv_cache_len=other._kv_cache_len;
      _kv_cache_max_len=other._kv_cache_max_len;
      _kv_cache_batch=other._kv_cache_batch;
      _kv_cache_enabled=other._kv_cache_enabled;
      _use_flash_attention=other._use_flash_attention;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMultiHeadAttention::Forward(
                   const CAIF_DeviceTensor &input,
                   bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: layer has been moved from");
    }

    // Step 1: Validate input shape [batch, seq_len, dim]
    const auto &shape=input.Shape();
    if(shape.size()!=3)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::Forward: input must be 3D [batch,seq_len,dim]");
    }
    if(shape[2]!=_config.dim)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::Forward: last dim must match config dim");
    }

    const uint32_t batch=shape[0];
    const uint32_t seq_len=shape[1];
    const uint32_t dim=_config.dim;
    const uint32_t num_heads=_config.num_heads;
    const uint32_t num_kv_heads=_config.num_kv_heads;
    const uint32_t head_dim=_config.head_dim;
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*num_heads;
    const uint32_t bkv=batch*num_kv_heads;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;

    // Step 2: Flatten input to [batch*seq_len, dim]
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,dim});

    // Step 3: Project Q, K, V
    CAIF_DeviceTensor q_proj;
    CAIF_DeviceTensor k_proj;
    CAIF_DeviceTensor v_proj;

    if(_use_projections==true)
    {
      q_proj=_projections.q_proj->Forward(flat_input,training);
      k_proj=_projections.k_proj->Forward(flat_input,training);
      v_proj=_projections.v_proj->Forward(flat_input,training);
      if(_projections.q_bias.IsEmpty()==false)
      {
        launch_bias_add_2d(q_proj.DevicePtr(),
                           _projections.q_bias.DevicePtr(),
                           q_proj.DevicePtr(),
                           static_cast<int>(bs),
                           static_cast<int>(qk_dim),
                           _stream->Handle());
      }
      if(_projections.k_bias.IsEmpty()==false)
      {
        launch_bias_add_2d(k_proj.DevicePtr(),
                           _projections.k_bias.DevicePtr(),
                           k_proj.DevicePtr(),
                           static_cast<int>(bs),
                           static_cast<int>(kv_dim),
                           _stream->Handle());
      }
      if(_projections.v_bias.IsEmpty()==false)
      {
        launch_bias_add_2d(v_proj.DevicePtr(),
                           _projections.v_bias.DevicePtr(),
                           v_proj.DevicePtr(),
                           static_cast<int>(bs),
                           static_cast<int>(kv_dim),
                           _stream->Handle());
      }
    }
    else
    {
      q_proj=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},*_stream);
      k_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},*_stream);
      v_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},*_stream);
      CAIF_DeviceOps::MatMul(flat_input,_w_q,q_proj);
      CAIF_DeviceOps::MatMul(flat_input,_w_k,k_proj);
      CAIF_DeviceOps::MatMul(flat_input,_w_v,v_proj);
    }

    // Step 4: Split heads via transpose_0213
    // Q: [batch, seq_len, num_heads, head_dim] -> [batch*num_heads, seq_len, head_dim]
    // K/V: [batch, seq_len, num_kv_heads, head_dim] -> [batch*num_kv_heads, seq_len, head_dim]
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bh*seq_len*head_dim},*_stream);
    CAIF_DeviceTensor k_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv*seq_len*head_dim},*_stream);
    CAIF_DeviceTensor v_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv*seq_len*head_dim},*_stream);

    launch_transpose_0213(q_proj.DevicePtr(),q_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(seq_len),
                          static_cast<int>(num_heads),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    launch_transpose_0213(k_proj.DevicePtr(),k_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(seq_len),
                          static_cast<int>(num_kv_heads),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    launch_transpose_0213(v_proj.DevicePtr(),v_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(seq_len),
                          static_cast<int>(num_kv_heads),
                          static_cast<int>(head_dim),
                          _stream->Handle());

    q_transposed.Reshape({bh,seq_len,head_dim});
    k_transposed.Reshape({bkv,seq_len,head_dim});
    v_transposed.Reshape({bkv,seq_len,head_dim});

    // Step 4.5: RoPE (applied in-place on Q and K after head split)
    if(_config.use_rope==true)
    {
      launch_rope_forward(q_transposed.DevicePtr(),
                          static_cast<int>(bh),
                          static_cast<int>(seq_len),
                          static_cast<int>(head_dim),
                          _config.rope_base,
                          _stream->Handle());
      launch_rope_forward(k_transposed.DevicePtr(),
                          static_cast<int>(bkv),
                          static_cast<int>(seq_len),
                          static_cast<int>(head_dim),
                          _config.rope_base,
                          _stream->Handle());
    }

    // Step 4.6: GQA expand (repeat KV heads to match Q heads)
    CAIF_DeviceTensor k_expanded;
    CAIF_DeviceTensor v_expanded;
    if(num_kv_heads!=num_heads)
    {
      const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);
      k_expanded=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      v_expanded=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      launch_gqa_repeat_kv(k_transposed.DevicePtr(),k_expanded.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
      launch_gqa_repeat_kv(v_transposed.DevicePtr(),v_expanded.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
    }
    else
    {
      k_expanded=std::move(k_transposed);
      v_expanded=std::move(v_transposed);
    }

    // Steps 5-9: Attention computation
    // Use FlashAttention for fused, memory-efficient attention
    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_DeviceTensor context=CAIF_DeviceTensor::Uninitialized(
                               {bh,seq_len,head_dim},*_stream);
    CAIF_DeviceTensor logsumexp;
    CAIF_DeviceTensor attn;

    if(_use_flash_attention==true)
    {
      // FlashAttention: fused QKV attention with online softmax
      // No attention matrix materialization - O(n) memory instead of O(n²)
      logsumexp=CAIF_DeviceTensor::Uninitialized({bh,seq_len},*_stream);

      int causal_flag=0;
      if(_config.causal==true)
      {
        causal_flag=1;
      }
      launch_flash_attention_forward(q_transposed.DevicePtr(),
                                     k_expanded.DevicePtr(),
                                     v_expanded.DevicePtr(),
                                     context.DevicePtr(),
                                     logsumexp.DevicePtr(),
                                     static_cast<int>(bh),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(head_dim),
                                     scale,
                                     causal_flag,
                                     _stream->Handle());
    }

    if(_use_flash_attention==false)
    {
      // Naive attention: explicit materialization of attention matrix
      // Step 5: scores = Q_heads @ K_expanded^T -> [bh, seq_len, seq_len]
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized(
                                {bh,seq_len,seq_len},*_stream);
      CAIF_DeviceOps::BatchedMatMulTransposeB(q_transposed,k_expanded,scores,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(bh));

      // Step 6: Scale by 1/sqrt(head_dim)
      CAIF_DeviceOps::Scale(scores,scale);

      // Step 7: Causal mask
      if(_config.causal==true)
      {
        launch_causal_mask_fill(scores.DevicePtr(),
                                static_cast<int>(bh),
                                static_cast<int>(seq_len),
                                _stream->Handle());
      }

      // Step 8: Softmax
      attn=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},*_stream);
      launch_attention_softmax(scores.DevicePtr(),attn.DevicePtr(),
                               static_cast<int>(bh*seq_len),
                               static_cast<int>(seq_len),
                               _stream->Handle());

      // Step 9: context = attn @ V_expanded -> [bh, seq_len, head_dim]
      CAIF_DeviceOps::BatchedMatMul(attn,v_expanded,context,
                                    static_cast<int>(seq_len),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(head_dim),
                                    static_cast<int>(bh));
    }

    // Step 10: Merge heads via reverse transpose_0213
    // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    CAIF_DeviceTensor merged=CAIF_DeviceTensor::Uninitialized(
                              {bs*qk_dim},*_stream);
    launch_transpose_0213(context.DevicePtr(),merged.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(num_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    merged.Reshape({bs,qk_dim});

    // Step 11: Output projection: output = concat @ W_o
    CAIF_DeviceTensor output_flat;
    if(_use_projections==true)
    {
      output_flat=_projections.o_proj->Forward(merged,training);
    }
    else
    {
      output_flat=CAIF_DeviceTensor::Uninitialized({bs,dim},*_stream);
      CAIF_DeviceOps::MatMul(merged,_w_o,output_flat);
    }

    // Step 12: Reshape to [batch, seq_len, dim]
    output_flat.Reshape({batch,seq_len,dim});

    // Step 13: Cache for backward (store non-expanded K/V for GQA)
    if(training==true)
    {
      _cached_input=std::move(flat_input);
      _cached_q_heads=std::move(q_transposed);
      if(num_kv_heads!=num_heads)
      {
        _cached_k_heads=std::move(k_transposed);
        _cached_v_heads=std::move(v_transposed);
      }
      else
      {
        _cached_k_heads=std::move(k_expanded);
        _cached_v_heads=std::move(v_expanded);
      }
      if(_use_flash_attention==true)
      {
        _cached_logsumexp=std::move(logsumexp);
        _cached_output=std::move(context);  // Safe: context not used after merge transpose
      }
      else
      {
        _cached_attn=std::move(attn);
      }
      _cached_concat=std::move(merged);
      _cached_batch=batch;
      _cached_seq_len=seq_len;
    }

    return output_flat;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMultiHeadAttention::Backward(
                   const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: layer has been moved from");
    }
    if(_cached_input.IsEmpty()==true)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::Backward: "
                  "must call Forward with training=true first");
    }

    const uint32_t batch=_cached_batch;
    const uint32_t seq_len=_cached_seq_len;
    const uint32_t dim=_config.dim;
    const uint32_t num_heads=_config.num_heads;
    const uint32_t num_kv_heads=_config.num_kv_heads;
    const uint32_t head_dim=_config.head_dim;
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*num_heads;
    const uint32_t bkv=batch*num_kv_heads;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;
    const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);

    // Step 1: Flatten grad_output to [bs, dim]
    CAIF_DeviceTensor grad_out_flat=grad_output.Clone();
    grad_out_flat.Reshape({bs,dim});

    // Step 2: Output projection gradients
    CAIF_DeviceTensor grad_concat;
    if(_use_projections==true)
    {
      grad_concat=_projections.o_proj->Backward(grad_out_flat);
    }
    else
    {
      // grad_concat = grad_out_flat @ W_o^T  -> [bs, qk_dim]
      grad_concat=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_out_flat,_w_o,grad_concat);
      // grad_W_o += cached_concat^T @ grad_out_flat  -> [qk_dim, dim]
      CAIF_DeviceTensor grad_w_o_delta=CAIF_DeviceTensor::Uninitialized(
                                        {qk_dim,dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_concat,grad_out_flat,grad_w_o_delta);
      CAIF_DeviceOps::Add(_grad_w_o,grad_w_o_delta,_grad_w_o);
    }

    // Step 3: Split heads of grad_concat
    // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    CAIF_DeviceTensor grad_context=CAIF_DeviceTensor::Uninitialized(
                                    {bh*seq_len*head_dim},*_stream);
    launch_transpose_0213(grad_concat.DevicePtr(),grad_context.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(seq_len),
                          static_cast<int>(num_heads),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    grad_context.Reshape({bh,seq_len,head_dim});

    // Step 4: Re-expand cached K/V if GQA active
    CAIF_DeviceTensor k_expanded_storage;
    CAIF_DeviceTensor v_expanded_storage;
    if(num_kv_heads!=num_heads)
    {
      k_expanded_storage=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      v_expanded_storage=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      launch_gqa_repeat_kv(_cached_k_heads.DevicePtr(),k_expanded_storage.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
      launch_gqa_repeat_kv(_cached_v_heads.DevicePtr(),v_expanded_storage.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
    }
    const CAIF_DeviceTensor &k_expanded=(num_kv_heads!=num_heads)?
                                        k_expanded_storage:_cached_k_heads;
    const CAIF_DeviceTensor &v_expanded=(num_kv_heads!=num_heads)?
                                        v_expanded_storage:_cached_v_heads;

    // Steps 5-9: Attention backward
    // Auto-select: use cuBLAS (naive) when attention matrix fits in memory,
    // flash backward for long sequences. cuBLAS is ~3x faster for short
    // sequences due to flash backward's poor GPU occupancy (~12.5%).
    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_DeviceTensor grad_q_heads;
    CAIF_DeviceTensor grad_k_heads;
    CAIF_DeviceTensor grad_v_heads;

    const size_t attn_matrix_bytes=static_cast<size_t>(bh)*seq_len*seq_len*sizeof(float);
    size_t free_mem=0;
    size_t total_mem=0;
    cudaMemGetInfo(&free_mem,&total_mem);
    const bool use_naive_backward=(attn_matrix_bytes*2<=free_mem);

    if(use_naive_backward==true)
    {
      // cuBLAS attention backward: recompute attention matrix, then use
      // optimized batched matmul for gradients. ~3x faster than flash
      // backward for seq_len <= ~1024.

      // Recompute attention matrix from cached Q, K
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized(
                                {bh,seq_len,seq_len},*_stream);
      CAIF_DeviceOps::BatchedMatMulTransposeB(_cached_q_heads,k_expanded,scores,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(bh));
      CAIF_DeviceOps::Scale(scores,scale);
      if(_config.causal==true)
      {
        launch_causal_mask_fill(scores.DevicePtr(),
                                static_cast<int>(bh),
                                static_cast<int>(seq_len),
                                _stream->Handle());
      }
      CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized(
                               {bh,seq_len,seq_len},*_stream);
      launch_attention_softmax(scores.DevicePtr(),attn.DevicePtr(),
                               static_cast<int>(bh*seq_len),
                               static_cast<int>(seq_len),
                               _stream->Handle());

      // Gradient computation using cuBLAS batched matmul
      // grad_attn = grad_context @ V_expanded^T -> [bh, seq_len, seq_len]
      CAIF_DeviceTensor grad_attn=CAIF_DeviceTensor::Uninitialized(
                                   {bh,seq_len,seq_len},*_stream);
      CAIF_DeviceOps::BatchedMatMulTransposeB(grad_context,v_expanded,grad_attn,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(bh));

      // grad_V_heads = attn^T @ grad_context -> [bh, seq_len, head_dim]
      grad_v_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      CAIF_DeviceOps::BatchedMatMulTransposeA(attn,grad_context,grad_v_heads,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(bh));

      // Softmax backward
      CAIF_DeviceTensor grad_scores=CAIF_DeviceTensor::Uninitialized(
                                     {bh,seq_len,seq_len},*_stream);
      launch_attention_softmax_backward(grad_attn.DevicePtr(),
                                        attn.DevicePtr(),
                                        grad_scores.DevicePtr(),
                                        static_cast<int>(bh*seq_len),
                                        static_cast<int>(seq_len),
                                        _stream->Handle());

      // Causal mask gradient
      if(_config.causal==true)
      {
        launch_causal_mask_grad(grad_scores.DevicePtr(),
                                static_cast<int>(bh),
                                static_cast<int>(seq_len),
                                _stream->Handle());
      }

      // Scale gradient
      CAIF_DeviceOps::Scale(grad_scores,scale);

      // grad_Q_heads = grad_scores @ K_expanded -> [bh, seq_len, head_dim]
      grad_q_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      CAIF_DeviceOps::BatchedMatMul(grad_scores,k_expanded,grad_q_heads,
                                    static_cast<int>(seq_len),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(head_dim),
                                    static_cast<int>(bh));

      // grad_K_heads = grad_scores^T @ Q_heads -> [bh, seq_len, head_dim]
      grad_k_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      CAIF_DeviceOps::BatchedMatMulTransposeA(grad_scores,_cached_q_heads,
                                              grad_k_heads,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(bh));
    }
    else
    {
      // FlashAttention backward: needed for long sequences where
      // O(n²) attention matrix doesn't fit in memory
      grad_q_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      grad_k_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);
      grad_v_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},*_stream);

      launch_flash_attention_backward(_cached_q_heads.DevicePtr(),
                                      k_expanded.DevicePtr(),
                                      v_expanded.DevicePtr(),
                                      _cached_output.DevicePtr(),
                                      grad_context.DevicePtr(),
                                      _cached_logsumexp.DevicePtr(),
                                      grad_q_heads.DevicePtr(),
                                      grad_k_heads.DevicePtr(),
                                      grad_v_heads.DevicePtr(),
                                      static_cast<int>(bh),
                                      static_cast<int>(seq_len),
                                      static_cast<int>(head_dim),
                                      scale,
                                      _config.causal?1:0,
                                      _stream->Handle());
    }

    // Step 10: RoPE backward (inverse rotation on grad_Q and grad_K)
    if(_config.use_rope==true)
    {
      launch_rope_backward(grad_q_heads.DevicePtr(),
                           static_cast<int>(bh),
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _config.rope_base,
                           _stream->Handle());
      launch_rope_backward(grad_k_heads.DevicePtr(),
                           static_cast<int>(bh),
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _config.rope_base,
                           _stream->Handle());
    }

    // Step 11: GQA reduce (sum expanded grads back to kv_heads)
    CAIF_DeviceTensor grad_k_reduced;
    CAIF_DeviceTensor grad_v_reduced;
    if(num_kv_heads!=num_heads)
    {
      grad_k_reduced=CAIF_DeviceTensor::Uninitialized(
                       {bkv,seq_len,head_dim},*_stream);
      grad_v_reduced=CAIF_DeviceTensor::Uninitialized(
                       {bkv,seq_len,head_dim},*_stream);
      launch_gqa_reduce_kv(grad_k_heads.DevicePtr(),
                           grad_k_reduced.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
      launch_gqa_reduce_kv(grad_v_heads.DevicePtr(),
                           grad_v_reduced.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(seq_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
    }
    else
    {
      grad_k_reduced=std::move(grad_k_heads);
      grad_v_reduced=std::move(grad_v_heads);
    }

    // Step 12: Merge heads of grad_Q -> [bs, qk_dim]
    CAIF_DeviceTensor grad_q_flat=CAIF_DeviceTensor::Uninitialized(
                                   {bs*qk_dim},*_stream);
    launch_transpose_0213(grad_q_heads.DevicePtr(),grad_q_flat.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(num_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    grad_q_flat.Reshape({bs,qk_dim});

    // Merge heads of grad_K/V -> [bs, kv_dim]
    CAIF_DeviceTensor grad_k_flat=CAIF_DeviceTensor::Uninitialized(
                                   {bs*kv_dim},*_stream);
    CAIF_DeviceTensor grad_v_flat=CAIF_DeviceTensor::Uninitialized(
                                   {bs*kv_dim},*_stream);
    launch_transpose_0213(grad_k_reduced.DevicePtr(),grad_k_flat.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(num_kv_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    launch_transpose_0213(grad_v_reduced.DevicePtr(),grad_v_flat.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(num_kv_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    grad_k_flat.Reshape({bs,kv_dim});
    grad_v_flat.Reshape({bs,kv_dim});

    // Steps 13-14: Projection weight gradients and input gradient
    CAIF_DeviceTensor grad_input;
    if(_use_projections==true)
    {
      CAIF_DeviceTensor gi_q=_projections.q_proj->Backward(grad_q_flat);
      CAIF_DeviceTensor gi_k=_projections.k_proj->Backward(grad_k_flat);
      CAIF_DeviceTensor gi_v=_projections.v_proj->Backward(grad_v_flat);
      grad_input=CAIF_DeviceTensor::Uninitialized({bs,dim},*_stream);
      CAIF_DeviceOps::Add(gi_q,gi_k,grad_input);
      CAIF_DeviceOps::Add(grad_input,gi_v,grad_input);
    }
    else
    {
      // grad_W_q += cached_input^T @ grad_Q_flat -> [dim, qk_dim]
      CAIF_DeviceTensor grad_wq_delta=CAIF_DeviceTensor::Uninitialized(
                                       {dim,qk_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_q_flat,grad_wq_delta);
      CAIF_DeviceOps::Add(_grad_w_q,grad_wq_delta,_grad_w_q);
      // grad_W_k += cached_input^T @ grad_K_flat -> [dim, kv_dim]
      CAIF_DeviceTensor grad_wk_delta=CAIF_DeviceTensor::Uninitialized(
                                       {dim,kv_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_k_flat,grad_wk_delta);
      CAIF_DeviceOps::Add(_grad_w_k,grad_wk_delta,_grad_w_k);
      // grad_W_v += cached_input^T @ grad_V_flat -> [dim, kv_dim]
      CAIF_DeviceTensor grad_wv_delta=CAIF_DeviceTensor::Uninitialized(
                                       {dim,kv_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_v_flat,grad_wv_delta);
      CAIF_DeviceOps::Add(_grad_w_v,grad_wv_delta,_grad_w_v);
      CAIF_DeviceTensor gi_q=CAIF_DeviceTensor::Uninitialized({bs,dim},*_stream);
      CAIF_DeviceTensor gi_k=CAIF_DeviceTensor::Uninitialized({bs,dim},*_stream);
      CAIF_DeviceTensor gi_v=CAIF_DeviceTensor::Uninitialized({bs,dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_q_flat,_w_q,gi_q);
      CAIF_DeviceOps::MatMulTransposeB(grad_k_flat,_w_k,gi_k);
      CAIF_DeviceOps::MatMulTransposeB(grad_v_flat,_w_v,gi_v);
      grad_input=CAIF_DeviceTensor::Uninitialized({bs,dim},*_stream);
      CAIF_DeviceOps::Add(gi_q,gi_k,grad_input);
      CAIF_DeviceOps::Add(grad_input,gi_v,grad_input);
    }

    // Step 15: Reshape to [batch, seq_len, dim]
    grad_input.Reshape({batch,seq_len,dim});

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMultiHeadAttention::ZeroGradients()
{
  try
  {
    if(_use_projections==true)
    {
      _projections.q_proj->ZeroGradients();
      _projections.k_proj->ZeroGradients();
      _projections.v_proj->ZeroGradients();
      _projections.o_proj->ZeroGradients();
    }
    else
    {
      _grad_w_q.Fill(0.0f);
      _grad_w_k.Fill(0.0f);
      _grad_w_v.Fill(0.0f);
      _grad_w_o.Fill(0.0f);
    }
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMultiHeadAttention::ParameterTensorCount()const
{
  try
  {
    if(_use_projections==true)
    {
      return _projections.q_proj->ParameterTensorCount()+
             _projections.k_proj->ParameterTensorCount()+
             _projections.v_proj->ParameterTensorCount()+
             _projections.o_proj->ParameterTensorCount();
    }
    return g_caif_attention_weight_count;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention::ParameterTensor(size_t index)
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      CAIF_DeviceLayer *projs[]={_projections.q_proj.get(),
                                _projections.k_proj.get(),
                                _projections.v_proj.get(),
                                _projections.o_proj.get()};
      for(size_t p=0;p<4;++p)
      {
        const size_t count=projs[p]->ParameterTensorCount();
        if(index<offset+count)
        {
          return projs[p]->ParameterTensor(index-offset);
        }
        offset+=count;
      }
      THROW_CAIFE("DeviceMultiHeadAttention::ParameterTensor: index out of range");
    }
    if(index==0)
    {
      return _w_q;
    }
    if(index==1)
    {
      return _w_k;
    }
    if(index==2)
    {
      return _w_v;
    }
    if(index==3)
    {
      return _w_o;
    }
    THROW_CAIFE("DeviceMultiHeadAttention::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention::ParameterTensor(
                          size_t index)const
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      const CAIF_DeviceLayer *projs[]={_projections.q_proj.get(),
                                      _projections.k_proj.get(),
                                      _projections.v_proj.get(),
                                      _projections.o_proj.get()};
      for(size_t p=0;p<4;++p)
      {
        const size_t count=projs[p]->ParameterTensorCount();
        if(index<offset+count)
        {
          return projs[p]->ParameterTensor(index-offset);
        }
        offset+=count;
      }
      THROW_CAIFE("DeviceMultiHeadAttention::ParameterTensor: index out of range");
    }
    if(index==0)
    {
      return _w_q;
    }
    if(index==1)
    {
      return _w_k;
    }
    if(index==2)
    {
      return _w_v;
    }
    if(index==3)
    {
      return _w_o;
    }
    THROW_CAIFE("DeviceMultiHeadAttention::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention::GradientTensor(size_t index)
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      CAIF_DeviceLayer *projs[]={_projections.q_proj.get(),
                                _projections.k_proj.get(),
                                _projections.v_proj.get(),
                                _projections.o_proj.get()};
      for(size_t p=0;p<4;++p)
      {
        const size_t count=projs[p]->ParameterTensorCount();
        if(index<offset+count)
        {
          return projs[p]->GradientTensor(index-offset);
        }
        offset+=count;
      }
      THROW_CAIFE("DeviceMultiHeadAttention::GradientTensor: index out of range");
    }
    if(index==0)
    {
      return _grad_w_q;
    }
    if(index==1)
    {
      return _grad_w_k;
    }
    if(index==2)
    {
      return _grad_w_v;
    }
    if(index==3)
    {
      return _grad_w_o;
    }
    THROW_CAIFE("DeviceMultiHeadAttention::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention::GradientTensor(
                          size_t index)const
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      const CAIF_DeviceLayer *projs[]={_projections.q_proj.get(),
                                      _projections.k_proj.get(),
                                      _projections.v_proj.get(),
                                      _projections.o_proj.get()};
      for(size_t p=0;p<4;++p)
      {
        const size_t count=projs[p]->ParameterTensorCount();
        if(index<offset+count)
        {
          return projs[p]->GradientTensor(index-offset);
        }
        offset+=count;
      }
      THROW_CAIFE("DeviceMultiHeadAttention::GradientTensor: index out of range");
    }
    if(index==0)
    {
      return _grad_w_q;
    }
    if(index==1)
    {
      return _grad_w_k;
    }
    if(index==2)
    {
      return _grad_w_v;
    }
    if(index==3)
    {
      return _grad_w_o;
    }
    THROW_CAIFE("DeviceMultiHeadAttention::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMultiHeadAttention::TotalParameterCount()const
{
  try
  {
    if(_use_projections==true)
    {
      return _projections.q_proj->TotalParameterCount()+
             _projections.k_proj->TotalParameterCount()+
             _projections.v_proj->TotalParameterCount()+
             _projections.o_proj->TotalParameterCount();
    }
    return _w_q.TotalElements()+
           _w_k.TotalElements()+
           _w_v.TotalElements()+
           _w_o.TotalElements();
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceMultiHeadAttention::Description()const
{
  try
  {
    std::string causal_str;
    if(_config.causal==true)
    {
      causal_str="true";
    }
    else
    {
      causal_str="false";
    }
    std::string desc="MultiHeadAttention(dim="+std::to_string(_config.dim)+
                     ",heads="+std::to_string(_config.num_heads)+
                     ",head_dim="+std::to_string(_config.head_dim)+
                     ",causal="+causal_str;
    if(_config.num_kv_heads!=_config.num_heads)
    {
      desc+=",kv_heads="+std::to_string(_config.num_kv_heads);
    }
    if(_config.use_rope==true)
    {
      desc+=",rope=true";
    }
    if(_use_projections==true)
    {
      desc+=",projections";
    }
    desc+=")";
    return desc;
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceMultiHeadAttention::ParameterNames(
                           const std::string &prefix)const
{
  try
  {
    if(_use_projections==true)
    {
      std::vector<std::string> names;
      std::vector<std::string> sub;
      sub=_projections.q_proj->ParameterNames(prefix+"q_proj.");
      names.insert(names.end(),sub.begin(),sub.end());
      sub=_projections.k_proj->ParameterNames(prefix+"k_proj.");
      names.insert(names.end(),sub.begin(),sub.end());
      sub=_projections.v_proj->ParameterNames(prefix+"v_proj.");
      names.insert(names.end(),sub.begin(),sub.end());
      sub=_projections.o_proj->ParameterNames(prefix+"o_proj.");
      names.insert(names.end(),sub.begin(),sub.end());
      return names;
    }
    std::vector<std::string> names;
    names.push_back(prefix+"q_proj.weight");
    names.push_back(prefix+"k_proj.weight");
    names.push_back(prefix+"v_proj.weight");
    names.push_back(prefix+"o_proj.weight");
    return names;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMultiHeadAttention::InitializeWeights(uint32_t seed)
{
  try
  {
    if(_use_projections==true)
    {
      return;
    }

    // Xavier uniform initialization
    std::mt19937 rng(seed);

    const uint32_t qk_dim=_config.num_heads*_config.head_dim;
    const uint32_t kv_dim=_config.num_kv_heads*_config.head_dim;
    const uint32_t dim=_config.dim;

    // W_q: [dim, qk_dim]
    const float limit_q=std::sqrt(6.0f/static_cast<float>(dim+qk_dim));
    std::uniform_real_distribution<float> dist_q(-limit_q,limit_q);
    std::vector<float> wq_data(dim*qk_dim);
    for(size_t i=0;i<wq_data.size();++i)
    {
      wq_data[i]=dist_q(rng);
    }
    _w_q.CopyFromHost(wq_data.data(),wq_data.size());

    // W_k: [dim, kv_dim]
    const float limit_kv=std::sqrt(6.0f/static_cast<float>(dim+kv_dim));
    std::uniform_real_distribution<float> dist_kv(-limit_kv,limit_kv);
    std::vector<float> wk_data(dim*kv_dim);
    for(size_t i=0;i<wk_data.size();++i)
    {
      wk_data[i]=dist_kv(rng);
    }
    _w_k.CopyFromHost(wk_data.data(),wk_data.size());

    // W_v: [dim, kv_dim]
    std::vector<float> wv_data(dim*kv_dim);
    for(size_t i=0;i<wv_data.size();++i)
    {
      wv_data[i]=dist_kv(rng);
    }
    _w_v.CopyFromHost(wv_data.data(),wv_data.size());

    // W_o: [qk_dim, dim]
    const float limit_o=std::sqrt(6.0f/static_cast<float>(qk_dim+dim));
    std::uniform_real_distribution<float> dist_o(-limit_o,limit_o);
    std::vector<float> wo_data(qk_dim*dim);
    for(size_t i=0;i<wo_data.size();++i)
    {
      wo_data[i]=dist_o(rng);
    }
    _w_o.CopyFromHost(wo_data.data(),wo_data.size());
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMultiHeadAttention::EnableKVCache(uint32_t batch_size,
                                                  uint32_t max_seq_len)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: layer has been moved from");
    }

    const uint32_t num_kv_heads=_config.num_kv_heads;
    const uint32_t head_dim=_config.head_dim;
    const uint32_t bkv=batch_size*num_kv_heads;

    // Allocate cache tensors: [batch*num_kv_heads, max_seq_len, head_dim]
    // This matches the attention layout for efficient extraction
    _kv_cache_k=CAIF_DeviceTensor::Zeros({bkv,max_seq_len,head_dim},*_stream);
    _kv_cache_v=CAIF_DeviceTensor::Zeros({bkv,max_seq_len,head_dim},*_stream);
    _kv_cache_len=0;
    _kv_cache_max_len=max_seq_len;
    _kv_cache_batch=batch_size;
    _kv_cache_enabled=true;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMultiHeadAttention::DisableKVCache()
{
  try
  {
    _kv_cache_k=CAIF_DeviceTensor();
    _kv_cache_v=CAIF_DeviceTensor();
    _kv_cache_len=0;
    _kv_cache_max_len=0;
    _kv_cache_batch=0;
    _kv_cache_enabled=false;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMultiHeadAttention::ResetKVCache()
{
  try
  {
    if(_kv_cache_enabled==false)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ResetKVCache: cache not enabled");
    }
    _kv_cache_len=0;
    // Keep allocated tensors, just reset length
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMultiHeadAttention::ForwardCached(
                   const CAIF_DeviceTensor &input)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: layer has been moved from");
    }
    if(_kv_cache_enabled==false)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: cache not enabled");
    }

    // Validate input shape [batch, seq_len, dim]
    const auto &shape=input.Shape();
    if(shape.size()!=3)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: input must be 3D");
    }
    if(shape[2]!=_config.dim)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: last dim must match config dim");
    }
    if(shape[0]!=_kv_cache_batch)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: batch size must match cache");
    }

    const uint32_t batch=shape[0];
    const uint32_t new_len=shape[1];
    const uint32_t dim=_config.dim;
    const uint32_t num_heads=_config.num_heads;
    const uint32_t num_kv_heads=_config.num_kv_heads;
    const uint32_t head_dim=_config.head_dim;
    const uint32_t bs=batch*new_len;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;

    // Check cache capacity
    if(_kv_cache_len+new_len>_kv_cache_max_len)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: cache overflow");
    }

    // Step 1: Flatten input to [batch*new_len, dim]
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,dim});

    // Step 2: Project Q, K, V for new tokens
    CAIF_DeviceTensor q_proj;
    CAIF_DeviceTensor k_proj;
    CAIF_DeviceTensor v_proj;

    if(_use_projections==true)
    {
      q_proj=_projections.q_proj->Forward(flat_input,false);
      k_proj=_projections.k_proj->Forward(flat_input,false);
      v_proj=_projections.v_proj->Forward(flat_input,false);
      if(_projections.q_bias.IsEmpty()==false)
      {
        launch_bias_add_2d(q_proj.DevicePtr(),
                           _projections.q_bias.DevicePtr(),
                           q_proj.DevicePtr(),
                           static_cast<int>(bs),
                           static_cast<int>(qk_dim),
                           _stream->Handle());
      }
      if(_projections.k_bias.IsEmpty()==false)
      {
        launch_bias_add_2d(k_proj.DevicePtr(),
                           _projections.k_bias.DevicePtr(),
                           k_proj.DevicePtr(),
                           static_cast<int>(bs),
                           static_cast<int>(kv_dim),
                           _stream->Handle());
      }
      if(_projections.v_bias.IsEmpty()==false)
      {
        launch_bias_add_2d(v_proj.DevicePtr(),
                           _projections.v_bias.DevicePtr(),
                           v_proj.DevicePtr(),
                           static_cast<int>(bs),
                           static_cast<int>(kv_dim),
                           _stream->Handle());
      }
    }
    else
    {
      q_proj=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},*_stream);
      k_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},*_stream);
      v_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},*_stream);
      CAIF_DeviceOps::MatMul(flat_input,_w_q,q_proj);
      CAIF_DeviceOps::MatMul(flat_input,_w_k,k_proj);
      CAIF_DeviceOps::MatMul(flat_input,_w_v,v_proj);
    }

    // Step 3: Split heads for new K/V
    // Transpose to [batch*num_kv_heads, new_len, head_dim] for RoPE and cache
    const uint32_t bkv=batch*num_kv_heads;
    CAIF_DeviceTensor k_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv,new_len,head_dim},*_stream);
    CAIF_DeviceTensor v_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv,new_len,head_dim},*_stream);

    launch_transpose_0213(k_proj.DevicePtr(),k_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(new_len),
                          static_cast<int>(num_kv_heads),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    launch_transpose_0213(v_proj.DevicePtr(),v_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(new_len),
                          static_cast<int>(num_kv_heads),
                          static_cast<int>(head_dim),
                          _stream->Handle());

    // Save old cache length for RoPE position offset
    const uint32_t old_cache_len=_kv_cache_len;

    // Step 4: Apply RoPE to new K (position offset by cache_len)
    if(_config.use_rope==true)
    {
      // Apply RoPE with position offset = old cache length
      launch_rope_forward_offset(k_transposed.DevicePtr(),
                                 static_cast<int>(bkv),
                                 static_cast<int>(new_len),
                                 static_cast<int>(head_dim),
                                 _config.rope_base,
                                 static_cast<int>(old_cache_len),
                                 _stream->Handle());
    }

    // Step 5: Append new K/V to cache
    // Cache layout: [batch*num_kv_heads, max_seq_len, head_dim]
    launch_kv_cache_append_transposed(k_transposed.DevicePtr(),
                                      _kv_cache_k.DevicePtr(),
                                      static_cast<int>(bkv),
                                      static_cast<int>(new_len),
                                      static_cast<int>(_kv_cache_len),
                                      static_cast<int>(_kv_cache_max_len),
                                      static_cast<int>(head_dim),
                                      _stream->Handle());
    launch_kv_cache_append_transposed(v_transposed.DevicePtr(),
                                      _kv_cache_v.DevicePtr(),
                                      static_cast<int>(bkv),
                                      static_cast<int>(new_len),
                                      static_cast<int>(_kv_cache_len),
                                      static_cast<int>(_kv_cache_max_len),
                                      static_cast<int>(head_dim),
                                      _stream->Handle());

    // Update cache length
    const uint32_t total_len=_kv_cache_len+new_len;
    _kv_cache_len=total_len;

    // Step 6: Split heads for Q
    const uint32_t bh=batch*num_heads;
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bh,new_len,head_dim},*_stream);
    launch_transpose_0213(q_proj.DevicePtr(),q_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(new_len),
                          static_cast<int>(num_heads),
                          static_cast<int>(head_dim),
                          _stream->Handle());

    // Step 7: Apply RoPE to Q (position offset by old cache_len)
    if(_config.use_rope==true)
    {
      // Apply RoPE with same position offset as K (old cache length)
      launch_rope_forward_offset(q_transposed.DevicePtr(),
                                 static_cast<int>(bh),
                                 static_cast<int>(new_len),
                                 static_cast<int>(head_dim),
                                 _config.rope_base,
                                 static_cast<int>(old_cache_len),
                                 _stream->Handle());
    }

    // Step 8: Extract K/V from cache
    // Cache is [bkv, max_seq_len, head_dim], we need [bkv, total_len, head_dim]
    // Copy valid portion from each row
    CAIF_DeviceTensor k_full=CAIF_DeviceTensor::Uninitialized(
                              {bkv,total_len,head_dim},*_stream);
    CAIF_DeviceTensor v_full=CAIF_DeviceTensor::Uninitialized(
                              {bkv,total_len,head_dim},*_stream);

    // Copy valid portion from cache to k_full/v_full
    // For each bkv row, copy [0:total_len] from cache[bkv, :, :]
    for(uint32_t row=0;row<bkv;++row)
    {
      const size_t src_offset=row*_kv_cache_max_len*head_dim;
      const size_t dst_offset=row*total_len*head_dim;
      const size_t copy_size=total_len*head_dim*sizeof(float);
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(k_full.DevicePtr()+dst_offset,
                      _kv_cache_k.DevicePtr()+src_offset,
                      copy_size,cudaMemcpyDeviceToDevice,_stream->Handle());
      cudaMemcpyAsync(v_full.DevicePtr()+dst_offset,
                      _kv_cache_v.DevicePtr()+src_offset,
                      copy_size,cudaMemcpyDeviceToDevice,_stream->Handle());
#else
      std::memcpy(k_full.DevicePtr()+dst_offset,
                  _kv_cache_k.DevicePtr()+src_offset,
                  copy_size);
      std::memcpy(v_full.DevicePtr()+dst_offset,
                  _kv_cache_v.DevicePtr()+src_offset,
                  copy_size);
#endif
    }

    // Step 9: GQA expand K/V if needed
    CAIF_DeviceTensor k_expanded;
    CAIF_DeviceTensor v_expanded;
    if(num_kv_heads!=num_heads)
    {
      const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);
      k_expanded=CAIF_DeviceTensor::Uninitialized({bh,total_len,head_dim},*_stream);
      v_expanded=CAIF_DeviceTensor::Uninitialized({bh,total_len,head_dim},*_stream);
      launch_gqa_repeat_kv(k_full.DevicePtr(),k_expanded.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(total_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
      launch_gqa_repeat_kv(v_full.DevicePtr(),v_expanded.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(num_kv_heads),
                           repeat_factor,
                           static_cast<int>(total_len),
                           static_cast<int>(head_dim),
                           _stream->Handle());
    }
    else
    {
      k_expanded=std::move(k_full);
      v_expanded=std::move(v_full);
    }

    // Step 10: Compute attention scores
    // Q: [bh, new_len, head_dim], K: [bh, total_len, head_dim]
    // scores = Q @ K^T -> [bh, new_len, total_len]
    CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized(
                              {bh,new_len,total_len},*_stream);
    CAIF_DeviceOps::BatchedMatMulTransposeB(q_transposed,k_expanded,scores,
                                            static_cast<int>(new_len),
                                            static_cast<int>(head_dim),
                                            static_cast<int>(total_len),
                                            static_cast<int>(bh));

    // Step 11: Scale
    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_DeviceOps::Scale(scores,scale);

    // Step 12: Causal mask (if enabled)
    // For cached mode, mask should account for position offset
    if(_config.causal==true)
    {
      // The causal mask for cached decoding:
      // Query positions [0:new_len] correspond to sequence positions [offset:offset+new_len]
      // Key positions [0:total_len] correspond to sequence positions [0:total_len]
      // Query at row can attend to keys at col <= (offset + row)
      const uint32_t offset=total_len-new_len;
      launch_causal_mask_fill_offset(scores.DevicePtr(),
                                     static_cast<int>(bh),
                                     static_cast<int>(new_len),
                                     static_cast<int>(total_len),
                                     static_cast<int>(offset),
                                     _stream->Handle());
    }

    // Step 13: Softmax
    CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized(
                            {bh,new_len,total_len},*_stream);
    launch_attention_softmax(scores.DevicePtr(),attn.DevicePtr(),
                             static_cast<int>(bh*new_len),
                             static_cast<int>(total_len),
                             _stream->Handle());

    // Step 14: context = attn @ V -> [bh, new_len, head_dim]
    CAIF_DeviceTensor context=CAIF_DeviceTensor::Uninitialized(
                               {bh,new_len,head_dim},*_stream);
    CAIF_DeviceOps::BatchedMatMul(attn,v_expanded,context,
                                  static_cast<int>(new_len),
                                  static_cast<int>(total_len),
                                  static_cast<int>(head_dim),
                                  static_cast<int>(bh));

    // Step 15: Merge heads
    CAIF_DeviceTensor merged=CAIF_DeviceTensor::Uninitialized(
                              {bs*qk_dim},*_stream);
    launch_transpose_0213(context.DevicePtr(),merged.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(num_heads),
                          static_cast<int>(new_len),
                          static_cast<int>(head_dim),
                          _stream->Handle());
    merged.Reshape({bs,qk_dim});

    // Step 16: Output projection
    CAIF_DeviceTensor output_flat;
    if(_use_projections==true)
    {
      output_flat=_projections.o_proj->Forward(merged,false);
    }
    else
    {
      output_flat=CAIF_DeviceTensor::Uninitialized({bs,dim},*_stream);
      CAIF_DeviceOps::MatMul(merged,_w_o,output_flat);
    }

    // Step 17: Reshape to [batch, new_len, dim]
    output_flat.Reshape({batch,new_len,dim});

    return output_flat;
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
