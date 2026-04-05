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
// Multi-head Latent Attention (MLA) layer implementation
//------------------------------------------------------------------------------
#include "caif_device_ml_attention.h"
#include "caif_device_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <cmath>
#include <random>
#include <sstream>

using namespace instance;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

CAIF_DeviceMLAttention::CAIF_DeviceMLAttention(const MLAConfig_t &config,
                                             CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                                      _config(config),
                                                                      _use_projections(false),
                                                                      _cached_batch(0),
                                                                      _cached_seq_len(0),
                                                                      _kv_cache_len(0),
                                                                      _kv_cache_max_len(0),
                                                                      _kv_cache_batch(0),
                                                                      _kv_cache_enabled(false)
{
  try
  {
    // Compute derived dimensions
    _qk_head_dim=_config.qk_nope_head_dim+_config.qk_rope_head_dim;
    _q_proj_dim=_config.num_heads*_qk_head_dim;
    _kv_compress_dim=_config.kv_lora_rank+_config.qk_rope_head_dim;
    _kv_decomp_dim=_config.num_heads*(_config.qk_nope_head_dim+_config.v_head_dim);
    _o_input_dim=_config.num_heads*_config.v_head_dim;

    // Allocate parameters
    _w_q_compress=CAIF_DeviceTensor::Zeros({_config.dim,_config.q_lora_rank},stream);
    _q_norm_gamma=CAIF_DeviceTensor::Zeros({_config.q_lora_rank},stream);
    _w_q_decompress=CAIF_DeviceTensor::Zeros({_config.q_lora_rank,_q_proj_dim},stream);
    _w_kv_compress=CAIF_DeviceTensor::Zeros({_config.dim,_kv_compress_dim},stream);
    _kv_norm_gamma=CAIF_DeviceTensor::Zeros({_config.kv_lora_rank},stream);
    _w_kv_decompress=CAIF_DeviceTensor::Zeros({_config.kv_lora_rank,_kv_decomp_dim},stream);
    _w_o=CAIF_DeviceTensor::Zeros({_o_input_dim,_config.dim},stream);

    // Allocate gradients (same shapes)
    _grad_w_q_compress=CAIF_DeviceTensor::Zeros({_config.dim,_config.q_lora_rank},stream);
    _grad_q_norm_gamma=CAIF_DeviceTensor::Zeros({_config.q_lora_rank},stream);
    _grad_w_q_decompress=CAIF_DeviceTensor::Zeros({_config.q_lora_rank,_q_proj_dim},stream);
    _grad_w_kv_compress=CAIF_DeviceTensor::Zeros({_config.dim,_kv_compress_dim},stream);
    _grad_kv_norm_gamma=CAIF_DeviceTensor::Zeros({_config.kv_lora_rank},stream);
    _grad_w_kv_decompress=CAIF_DeviceTensor::Zeros({_config.kv_lora_rank,_kv_decomp_dim},stream);
    _grad_w_o=CAIF_DeviceTensor::Zeros({_o_input_dim,_config.dim},stream);

    // Initialize gamma to 1.0 (RMSNorm convention)
    _q_norm_gamma.Fill(1.0f);
    _kv_norm_gamma.Fill(1.0f);

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Constructor (projections-based)
//------------------------------------------------------------------------------

CAIF_DeviceMLAttention::CAIF_DeviceMLAttention(const MLAConfig_t &config,
                                             MLAProjections_t projections,
                                             CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                                      _config(config),
                                                                      _projections(std::move(projections)),
                                                                      _use_projections(true),
                                                                      _cached_batch(0),
                                                                      _cached_seq_len(0),
                                                                      _kv_cache_len(0),
                                                                      _kv_cache_max_len(0),
                                                                      _kv_cache_batch(0),
                                                                      _kv_cache_enabled(false)
{
  try
  {
    // Compute derived dimensions
    _qk_head_dim=_config.qk_nope_head_dim+_config.qk_rope_head_dim;
    _q_proj_dim=_config.num_heads*_qk_head_dim;
    _kv_compress_dim=_config.kv_lora_rank+_config.qk_rope_head_dim;
    _kv_decomp_dim=_config.num_heads*(_config.qk_nope_head_dim+_config.v_head_dim);
    _o_input_dim=_config.num_heads*_config.v_head_dim;

    // Allocate internal norm parameters (not owned by projections)
    _q_norm_gamma=CAIF_DeviceTensor::Zeros({_config.q_lora_rank},stream);
    _kv_norm_gamma=CAIF_DeviceTensor::Zeros({_config.kv_lora_rank},stream);
    _grad_q_norm_gamma=CAIF_DeviceTensor::Zeros({_config.q_lora_rank},stream);
    _grad_kv_norm_gamma=CAIF_DeviceTensor::Zeros({_config.kv_lora_rank},stream);

    // Initialize gamma to 1.0 (RMSNorm convention)
    _q_norm_gamma.Fill(1.0f);
    _kv_norm_gamma.Fill(1.0f);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Move semantics
//------------------------------------------------------------------------------

CAIF_DeviceMLAttention::CAIF_DeviceMLAttention(
                     CAIF_DeviceMLAttention &&other):CAIF_DeviceLayer(std::move(other)),
                                                    _config(other._config),
                                                    _projections(std::move(other._projections)),
                                                    _use_projections(other._use_projections),
                                                    _qk_head_dim(other._qk_head_dim),
                                                    _q_proj_dim(other._q_proj_dim),
                                                    _kv_compress_dim(other._kv_compress_dim),
                                                    _kv_decomp_dim(other._kv_decomp_dim),
                                                    _o_input_dim(other._o_input_dim),
                                                    _w_q_compress(std::move(other._w_q_compress)),
                                                    _q_norm_gamma(std::move(other._q_norm_gamma)),
                                                    _w_q_decompress(std::move(other._w_q_decompress)),
                                                    _w_kv_compress(std::move(other._w_kv_compress)),
                                                    _kv_norm_gamma(std::move(other._kv_norm_gamma)),
                                                    _w_kv_decompress(std::move(other._w_kv_decompress)),
                                                    _w_o(std::move(other._w_o)),
                                                    _grad_w_q_compress(std::move(other._grad_w_q_compress)),
                                                    _grad_q_norm_gamma(std::move(other._grad_q_norm_gamma)),
                                                    _grad_w_q_decompress(std::move(other._grad_w_q_decompress)),
                                                    _grad_w_kv_compress(std::move(other._grad_w_kv_compress)),
                                                    _grad_kv_norm_gamma(std::move(other._grad_kv_norm_gamma)),
                                                    _grad_w_kv_decompress(std::move(other._grad_w_kv_decompress)),
                                                    _grad_w_o(std::move(other._grad_w_o)),
                                                    _cached_input(std::move(other._cached_input)),
                                                    _cached_q_compressed(std::move(other._cached_q_compressed)),
                                                    _cached_q_rms(std::move(other._cached_q_rms)),
                                                    _cached_q_normed(std::move(other._cached_q_normed)),
                                                    _cached_kv_compressed(std::move(other._cached_kv_compressed)),
                                                    _cached_kv_rms(std::move(other._cached_kv_rms)),
                                                    _cached_kv_normed(std::move(other._cached_kv_normed)),
                                                    _cached_q(std::move(other._cached_q)),
                                                    _cached_k(std::move(other._cached_k)),
                                                    _cached_v(std::move(other._cached_v)),
                                                    _cached_attn_output(std::move(other._cached_attn_output)),
                                                    _cached_logsumexp(std::move(other._cached_logsumexp)),
                                                    _cached_merged(std::move(other._cached_merged)),
                                                    _cached_batch(other._cached_batch),
                                                    _cached_seq_len(other._cached_seq_len),
                                                    _kv_cache_compressed(std::move(other._kv_cache_compressed)),
                                                    _kv_cache_k_pe(std::move(other._kv_cache_k_pe)),
                                                    _kv_cache_len(other._kv_cache_len),
                                                    _kv_cache_max_len(other._kv_cache_max_len),
                                                    _kv_cache_batch(other._kv_cache_batch),
                                                    _kv_cache_enabled(other._kv_cache_enabled)
{
}

CAIF_DeviceMLAttention &CAIF_DeviceMLAttention::operator=(CAIF_DeviceMLAttention &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _projections=std::move(other._projections);
    _use_projections=other._use_projections;
    _qk_head_dim=other._qk_head_dim;
    _q_proj_dim=other._q_proj_dim;
    _kv_compress_dim=other._kv_compress_dim;
    _kv_decomp_dim=other._kv_decomp_dim;
    _o_input_dim=other._o_input_dim;

    _w_q_compress=std::move(other._w_q_compress);
    _q_norm_gamma=std::move(other._q_norm_gamma);
    _w_q_decompress=std::move(other._w_q_decompress);
    _w_kv_compress=std::move(other._w_kv_compress);
    _kv_norm_gamma=std::move(other._kv_norm_gamma);
    _w_kv_decompress=std::move(other._w_kv_decompress);
    _w_o=std::move(other._w_o);

    _grad_w_q_compress=std::move(other._grad_w_q_compress);
    _grad_q_norm_gamma=std::move(other._grad_q_norm_gamma);
    _grad_w_q_decompress=std::move(other._grad_w_q_decompress);
    _grad_w_kv_compress=std::move(other._grad_w_kv_compress);
    _grad_kv_norm_gamma=std::move(other._grad_kv_norm_gamma);
    _grad_w_kv_decompress=std::move(other._grad_w_kv_decompress);
    _grad_w_o=std::move(other._grad_w_o);

    _cached_input=std::move(other._cached_input);
    _cached_q_compressed=std::move(other._cached_q_compressed);
    _cached_q_rms=std::move(other._cached_q_rms);
    _cached_q_normed=std::move(other._cached_q_normed);
    _cached_kv_compressed=std::move(other._cached_kv_compressed);
    _cached_kv_rms=std::move(other._cached_kv_rms);
    _cached_kv_normed=std::move(other._cached_kv_normed);
    _cached_q=std::move(other._cached_q);
    _cached_k=std::move(other._cached_k);
    _cached_v=std::move(other._cached_v);
    _cached_attn_output=std::move(other._cached_attn_output);
    _cached_logsumexp=std::move(other._cached_logsumexp);
    _cached_merged=std::move(other._cached_merged);
    _cached_batch=other._cached_batch;
    _cached_seq_len=other._cached_seq_len;

    _kv_cache_compressed=std::move(other._kv_cache_compressed);
    _kv_cache_k_pe=std::move(other._kv_cache_k_pe);
    _kv_cache_len=other._kv_cache_len;
    _kv_cache_max_len=other._kv_cache_max_len;
    _kv_cache_batch=other._kv_cache_batch;
    _kv_cache_enabled=other._kv_cache_enabled;
  }
  return *this;
}

//------------------------------------------------------------------------------
// Forward pass
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_DeviceMLAttention::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    const auto &shape=input.Shape();
    if(shape.size()!=3||shape[2]!=_config.dim)
    {
      THROW_CAIFE("MLA Forward: input must be [batch, seq_len, dim]");
    }

    const uint32_t batch=shape[0];
    const uint32_t seq_len=shape[1];
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*_config.num_heads;
    const uint32_t nope=_config.qk_nope_head_dim;
    const uint32_t rope=_config.qk_rope_head_dim;
    const uint32_t v_dim=_config.v_head_dim;
    const float scale=1.0f/std::sqrt(static_cast<float>(_qk_head_dim));

    // Flatten input to [bs, dim]
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,_config.dim});

    //------------------------------------------------------------------
    // Q path
    //------------------------------------------------------------------

    // 1. Q compress: [bs, dim] -> [bs, q_lora_rank]
    CAIF_DeviceTensor q_compressed;
    if(_use_projections==true)
    {
      q_compressed=_projections.q_compress->Forward(flat_input,training);
    }
    else
    {
      q_compressed=CAIF_DeviceTensor::Uninitialized({bs,_config.q_lora_rank},*_stream);
      CAIF_DeviceOps::MatMul(flat_input,_w_q_compress,q_compressed);
    }

    // 2. Q RMSNorm
    CAIF_DeviceTensor q_normed=CAIF_DeviceTensor::Uninitialized({bs,_config.q_lora_rank},*_stream);
    CAIF_DeviceTensor q_rms=CAIF_DeviceTensor::Uninitialized({bs},*_stream);
    launch_rmsnorm_forward(q_compressed.DevicePtr(),
                           _q_norm_gamma.DevicePtr(),
                           q_normed.DevicePtr(),
                           q_rms.DevicePtr(),
                           _config.rms_norm_eps,
                           static_cast<int>(bs),
                           static_cast<int>(_config.q_lora_rank),
                           _stream->Handle());

    // 3. Q decompress: [bs, q_lora_rank] -> [bs, q_proj_dim]
    CAIF_DeviceTensor q_full;
    if(_use_projections==true)
    {
      q_full=_projections.q_decompress->Forward(q_normed,training);
    }
    else
    {
      q_full=CAIF_DeviceTensor::Uninitialized({bs,_q_proj_dim},*_stream);
      CAIF_DeviceOps::MatMul(q_normed,_w_q_decompress,q_full);
    }

    // 4. Reshape to [batch, seq, heads, qk_head_dim] and transpose to [bh, seq, qk_head_dim]
    q_full.Reshape({batch,seq_len,_config.num_heads,_qk_head_dim});
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized({bh,seq_len,_qk_head_dim},*_stream);
    launch_transpose_0213(q_full.DevicePtr(),
                          q_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(seq_len),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(_qk_head_dim),
                          _stream->Handle());

    // 5. Slice Q into nope and rope portions: [bh, seq, nope] + [bh, seq, rope]
    const int q_rows=static_cast<int>(bh*seq_len);
    CAIF_DeviceTensor q_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},*_stream);
    CAIF_DeviceTensor q_rope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},*_stream);
    launch_slice_last_dim(q_transposed.DevicePtr(),q_nope.DevicePtr(),
                          q_rows,static_cast<int>(_qk_head_dim),0,static_cast<int>(nope),
                          _stream->Handle());
    launch_slice_last_dim(q_transposed.DevicePtr(),q_rope.DevicePtr(),
                          q_rows,static_cast<int>(_qk_head_dim),
                          static_cast<int>(nope),static_cast<int>(rope),
                          _stream->Handle());

    // 6. Apply RoPE to q_rope only
    launch_rope_forward(q_rope.DevicePtr(),
                        static_cast<int>(bh),
                        static_cast<int>(seq_len),
                        static_cast<int>(rope),
                        _config.rope_base,
                        _stream->Handle());

    //------------------------------------------------------------------
    // KV path
    //------------------------------------------------------------------

    // 7. KV compress: [bs, dim] -> [bs, kv_compress_dim]
    CAIF_DeviceTensor kv_out;
    if(_use_projections==true)
    {
      kv_out=_projections.kv_compress->Forward(flat_input,training);
    }
    else
    {
      kv_out=CAIF_DeviceTensor::Uninitialized({bs,_kv_compress_dim},*_stream);
      CAIF_DeviceOps::MatMul(flat_input,_w_kv_compress,kv_out);
    }

    // 8. Slice into compressed_kv [bs, kv_lora_rank] and k_pe_flat [bs, rope]
    CAIF_DeviceTensor kv_compressed=CAIF_DeviceTensor::Uninitialized({bs,_config.kv_lora_rank},*_stream);
    CAIF_DeviceTensor k_pe_flat=CAIF_DeviceTensor::Uninitialized({bs,rope},*_stream);
    launch_slice_last_dim(kv_out.DevicePtr(),kv_compressed.DevicePtr(),
                          static_cast<int>(bs),static_cast<int>(_kv_compress_dim),
                          0,static_cast<int>(_config.kv_lora_rank),
                          _stream->Handle());
    launch_slice_last_dim(kv_out.DevicePtr(),k_pe_flat.DevicePtr(),
                          static_cast<int>(bs),static_cast<int>(_kv_compress_dim),
                          static_cast<int>(_config.kv_lora_rank),static_cast<int>(rope),
                          _stream->Handle());

    // 9. KV RMSNorm on compressed_kv
    CAIF_DeviceTensor kv_normed=CAIF_DeviceTensor::Uninitialized({bs,_config.kv_lora_rank},*_stream);
    CAIF_DeviceTensor kv_rms=CAIF_DeviceTensor::Uninitialized({bs},*_stream);
    launch_rmsnorm_forward(kv_compressed.DevicePtr(),
                           _kv_norm_gamma.DevicePtr(),
                           kv_normed.DevicePtr(),
                           kv_rms.DevicePtr(),
                           _config.rms_norm_eps,
                           static_cast<int>(bs),
                           static_cast<int>(_config.kv_lora_rank),
                           _stream->Handle());

    // 10. KV decompress: [bs, kv_lora_rank] -> [bs, kv_decomp_dim]
    CAIF_DeviceTensor kv_full;
    if(_use_projections==true)
    {
      kv_full=_projections.kv_decompress->Forward(kv_normed,training);
    }
    else
    {
      kv_full=CAIF_DeviceTensor::Uninitialized({bs,_kv_decomp_dim},*_stream);
      CAIF_DeviceOps::MatMul(kv_normed,_w_kv_decompress,kv_full);
    }

    // 11. Reshape to [batch, seq, heads, nope+v_dim] and transpose to [bh, seq, nope+v_dim]
    const uint32_t kv_per_head=nope+v_dim;
    kv_full.Reshape({batch,seq_len,_config.num_heads,kv_per_head});
    CAIF_DeviceTensor kv_transposed=CAIF_DeviceTensor::Uninitialized({bh,seq_len,kv_per_head},*_stream);
    launch_transpose_0213(kv_full.DevicePtr(),
                          kv_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(seq_len),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(kv_per_head),
                          _stream->Handle());

    // 12. Split KV into k_nope [bh, seq, nope] and v [bh, seq, v_dim]
    const int kv_rows=static_cast<int>(bh*seq_len);
    CAIF_DeviceTensor k_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},*_stream);
    CAIF_DeviceTensor v=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},*_stream);
    launch_slice_last_dim(kv_transposed.DevicePtr(),k_nope.DevicePtr(),
                          kv_rows,static_cast<int>(kv_per_head),0,static_cast<int>(nope),
                          _stream->Handle());
    launch_slice_last_dim(kv_transposed.DevicePtr(),v.DevicePtr(),
                          kv_rows,static_cast<int>(kv_per_head),
                          static_cast<int>(nope),static_cast<int>(v_dim),
                          _stream->Handle());

    // 13. Broadcast k_pe from [batch, seq, rope] to [bh, seq, rope]
    k_pe_flat.Reshape({batch,seq_len,rope});
    CAIF_DeviceTensor k_pe=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},*_stream);
    launch_gqa_repeat_kv(k_pe_flat.DevicePtr(),
                         k_pe.DevicePtr(),
                         static_cast<int>(batch),
                         1,
                         static_cast<int>(_config.num_heads),
                         static_cast<int>(seq_len),
                         static_cast<int>(rope),
                         _stream->Handle());

    // 14. Apply RoPE to k_pe
    launch_rope_forward(k_pe.DevicePtr(),
                        static_cast<int>(bh),
                        static_cast<int>(seq_len),
                        static_cast<int>(rope),
                        _config.rope_base,
                        _stream->Handle());

    //------------------------------------------------------------------
    // Assemble Q, K and run attention
    //------------------------------------------------------------------

    // 15. Concat Q: [q_nope | q_rope] -> [bh, seq, qk_head_dim]
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized({bh,seq_len,_qk_head_dim},*_stream);
    launch_concat_last_dim(q_nope.DevicePtr(),q_rope.DevicePtr(),q.DevicePtr(),
                           q_rows,static_cast<int>(nope),static_cast<int>(rope),
                           _stream->Handle());

    // 16. Concat K: [k_nope | k_pe] -> [bh, seq, qk_head_dim]
    CAIF_DeviceTensor k=CAIF_DeviceTensor::Uninitialized({bh,seq_len,_qk_head_dim},*_stream);
    launch_concat_last_dim(k_nope.DevicePtr(),k_pe.DevicePtr(),k.DevicePtr(),
                           kv_rows,static_cast<int>(nope),static_cast<int>(rope),
                           _stream->Handle());

    // 17. Standard attention: scores=Q@K^T, softmax, output=attn@V
    //     Flash attention kernel only supports head_dim<=128; GLM uses 256.
    CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},*_stream);
    CAIF_DeviceOps::BatchedMatMulTransposeB(q,
                                           k,
                                           scores,
                                           static_cast<int>(seq_len),
                                           static_cast<int>(_qk_head_dim),
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
    CAIF_DeviceTensor attn_weights=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},*_stream);
    launch_attention_softmax(scores.DevicePtr(),
                             attn_weights.DevicePtr(),
                             static_cast<int>(bh*seq_len),
                             static_cast<int>(seq_len),
                             _stream->Handle());
    CAIF_DeviceTensor attn_output=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},*_stream);
    CAIF_DeviceOps::BatchedMatMul(attn_weights,
                                 v,
                                 attn_output,
                                 static_cast<int>(seq_len),
                                 static_cast<int>(seq_len),
                                 static_cast<int>(v_dim),
                                 static_cast<int>(bh));

    //------------------------------------------------------------------
    // Output projection
    //------------------------------------------------------------------

    // 18. Merge heads: transpose [bh, seq, v_dim] -> [batch, seq, heads, v_dim] -> [bs, o_input_dim]
    CAIF_DeviceTensor attn_merged=CAIF_DeviceTensor::Uninitialized(
        {batch,seq_len,_config.num_heads,v_dim},*_stream);
    launch_transpose_0213(attn_output.DevicePtr(),
                          attn_merged.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(v_dim),
                          _stream->Handle());
    attn_merged.Reshape({bs,_o_input_dim});

    // 19. Output projection: [bs, o_input_dim] -> [bs, dim]
    CAIF_DeviceTensor output;
    if(_use_projections==true)
    {
      output=_projections.o_proj->Forward(attn_merged,training);
    }
    else
    {
      output=CAIF_DeviceTensor::Uninitialized({bs,_config.dim},*_stream);
      CAIF_DeviceOps::MatMul(attn_merged,_w_o,output);
    }

    //------------------------------------------------------------------
    // Cache for backward
    //------------------------------------------------------------------
    if(training==true)
    {
      _cached_input=std::move(flat_input);
      _cached_q_compressed=std::move(q_compressed);
      _cached_q_rms=std::move(q_rms);
      _cached_q_normed=std::move(q_normed);
      _cached_kv_compressed=std::move(kv_compressed);
      _cached_kv_rms=std::move(kv_rms);
      _cached_kv_normed=std::move(kv_normed);
      _cached_q=std::move(q);
      _cached_k=std::move(k);
      _cached_v=std::move(v);
      _cached_attn_output=std::move(attn_output);
      _cached_merged=std::move(attn_merged);
      _cached_batch=batch;
      _cached_seq_len=seq_len;
    }

    output.Reshape({batch,seq_len,_config.dim});
    return output;
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Backward pass
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_DeviceMLAttention::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    const uint32_t batch=_cached_batch;
    const uint32_t seq_len=_cached_seq_len;
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*_config.num_heads;
    const uint32_t nope=_config.qk_nope_head_dim;
    const uint32_t rope=_config.qk_rope_head_dim;
    const uint32_t v_dim=_config.v_head_dim;
    const float scale=1.0f/std::sqrt(static_cast<float>(_qk_head_dim));

    // Flatten grad_output to [bs, dim]
    CAIF_DeviceTensor grad_flat=grad_output.Clone();
    grad_flat.Reshape({bs,_config.dim});

    //------------------------------------------------------------------
    // Output projection backward
    //------------------------------------------------------------------
    CAIF_DeviceTensor grad_merged;
    if(_use_projections==true)
    {
      grad_merged=_projections.o_proj->Backward(grad_flat);
    }
    else
    {
      // grad_merged = grad_flat @ W_o^T -> [bs, o_input_dim]
      grad_merged=CAIF_DeviceTensor::Uninitialized({bs,_o_input_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_flat,_w_o,grad_merged);

      // grad_w_o += merged^T @ grad_flat
      CAIF_DeviceTensor grad_w_o_delta=CAIF_DeviceTensor::Uninitialized(
          {_o_input_dim,_config.dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_merged,grad_flat,grad_w_o_delta);
      CAIF_DeviceOps::Add(_grad_w_o,grad_w_o_delta,_grad_w_o);
    }

    //------------------------------------------------------------------
    // Reverse merge heads: [bs, o_input_dim] -> [bh, seq, v_dim]
    //------------------------------------------------------------------
    grad_merged.Reshape({batch,seq_len,_config.num_heads,v_dim});
    CAIF_DeviceTensor grad_attn_output=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},*_stream);
    launch_transpose_0213(grad_merged.DevicePtr(),
                          grad_attn_output.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(seq_len),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(v_dim),
                          _stream->Handle());

    //------------------------------------------------------------------
    // Standard attention backward: recompute attn, then matmul gradients
    //------------------------------------------------------------------

    // Recompute attention weights from cached Q, K
    CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},*_stream);
    CAIF_DeviceOps::BatchedMatMulTransposeB(_cached_q,
                                           _cached_k,
                                           scores,
                                           static_cast<int>(seq_len),
                                           static_cast<int>(_qk_head_dim),
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
    CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},*_stream);
    launch_attention_softmax(scores.DevicePtr(),
                             attn.DevicePtr(),
                             static_cast<int>(bh*seq_len),
                             static_cast<int>(seq_len),
                             _stream->Handle());

    // grad_attn = grad_attn_output @ V^T
    CAIF_DeviceTensor grad_attn=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},*_stream);
    CAIF_DeviceOps::BatchedMatMulTransposeB(grad_attn_output,
                                           _cached_v,
                                           grad_attn,
                                           static_cast<int>(seq_len),
                                           static_cast<int>(v_dim),
                                           static_cast<int>(seq_len),
                                           static_cast<int>(bh));

    // grad_v = attn^T @ grad_attn_output
    CAIF_DeviceTensor grad_v=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},*_stream);
    CAIF_DeviceOps::BatchedMatMulTransposeA(attn,
                                           grad_attn_output,
                                           grad_v,
                                           static_cast<int>(seq_len),
                                           static_cast<int>(seq_len),
                                           static_cast<int>(v_dim),
                                           static_cast<int>(bh));

    // Softmax backward
    CAIF_DeviceTensor grad_scores=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},*_stream);
    launch_attention_softmax_backward(grad_attn.DevicePtr(),
                                      attn.DevicePtr(),
                                      grad_scores.DevicePtr(),
                                      static_cast<int>(bh*seq_len),
                                      static_cast<int>(seq_len),
                                      _stream->Handle());
    if(_config.causal==true)
    {
      launch_causal_mask_grad(grad_scores.DevicePtr(),
                              static_cast<int>(bh),
                              static_cast<int>(seq_len),
                              _stream->Handle());
    }
    CAIF_DeviceOps::Scale(grad_scores,scale);

    // grad_q = grad_scores @ K
    CAIF_DeviceTensor grad_q=CAIF_DeviceTensor::Uninitialized({bh,seq_len,_qk_head_dim},*_stream);
    CAIF_DeviceOps::BatchedMatMul(grad_scores,
                                 _cached_k,
                                 grad_q,
                                 static_cast<int>(seq_len),
                                 static_cast<int>(seq_len),
                                 static_cast<int>(_qk_head_dim),
                                 static_cast<int>(bh));

    // grad_k = grad_scores^T @ Q
    CAIF_DeviceTensor grad_k=CAIF_DeviceTensor::Uninitialized({bh,seq_len,_qk_head_dim},*_stream);
    CAIF_DeviceOps::BatchedMatMulTransposeA(grad_scores,
                                           _cached_q,
                                           grad_k,
                                           static_cast<int>(seq_len),
                                           static_cast<int>(seq_len),
                                           static_cast<int>(_qk_head_dim),
                                           static_cast<int>(bh));

    //------------------------------------------------------------------
    // Q path backward
    //------------------------------------------------------------------
    const int q_rows=static_cast<int>(bh*seq_len);

    // Split grad_q into grad_q_nope [bh,seq,nope] + grad_q_rope [bh,seq,rope]
    CAIF_DeviceTensor grad_q_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},*_stream);
    CAIF_DeviceTensor grad_q_rope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},*_stream);
    launch_slice_last_dim(grad_q.DevicePtr(),grad_q_nope.DevicePtr(),
                          q_rows,static_cast<int>(_qk_head_dim),0,static_cast<int>(nope),
                          _stream->Handle());
    launch_slice_last_dim(grad_q.DevicePtr(),grad_q_rope.DevicePtr(),
                          q_rows,static_cast<int>(_qk_head_dim),
                          static_cast<int>(nope),static_cast<int>(rope),
                          _stream->Handle());

    // Inverse RoPE on grad_q_rope
    launch_rope_backward(grad_q_rope.DevicePtr(),
                         static_cast<int>(bh),
                         static_cast<int>(seq_len),
                         static_cast<int>(rope),
                         _config.rope_base,
                         _stream->Handle());

    // Reassemble grad_q_full [bh, seq, qk_head_dim]
    CAIF_DeviceTensor grad_q_full=CAIF_DeviceTensor::Uninitialized({bh,seq_len,_qk_head_dim},*_stream);
    launch_concat_last_dim(grad_q_nope.DevicePtr(),grad_q_rope.DevicePtr(),
                           grad_q_full.DevicePtr(),
                           q_rows,static_cast<int>(nope),static_cast<int>(rope),
                           _stream->Handle());

    // Reverse transpose: [bh, seq, qk_head_dim] -> [batch, seq, heads, qk_head_dim] -> [bs, q_proj_dim]
    CAIF_DeviceTensor grad_q_merged=CAIF_DeviceTensor::Uninitialized(
        {batch,seq_len,_config.num_heads,_qk_head_dim},*_stream);
    launch_transpose_0213(grad_q_full.DevicePtr(),
                          grad_q_merged.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(_qk_head_dim),
                          _stream->Handle());
    grad_q_merged.Reshape({bs,_q_proj_dim});

    // Q decompress backward: grad_q_merged -> grad_q_normed
    CAIF_DeviceTensor grad_q_normed;
    if(_use_projections==true)
    {
      grad_q_normed=_projections.q_decompress->Backward(grad_q_merged);
    }
    else
    {
      grad_q_normed=CAIF_DeviceTensor::Uninitialized({bs,_config.q_lora_rank},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_q_merged,_w_q_decompress,grad_q_normed);

      CAIF_DeviceTensor grad_w_qd=CAIF_DeviceTensor::Uninitialized(
          {_config.q_lora_rank,_q_proj_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_q_normed,grad_q_merged,grad_w_qd);
      CAIF_DeviceOps::Add(_grad_w_q_decompress,grad_w_qd,_grad_w_q_decompress);
    }

    // Q RMSNorm backward
    CAIF_DeviceTensor grad_q_compressed=CAIF_DeviceTensor::Uninitialized({bs,_config.q_lora_rank},*_stream);
    launch_rmsnorm_backward(grad_q_normed.DevicePtr(),
                            _cached_q_compressed.DevicePtr(),
                            _q_norm_gamma.DevicePtr(),
                            _cached_q_rms.DevicePtr(),
                            grad_q_compressed.DevicePtr(),
                            _grad_q_norm_gamma.DevicePtr(),
                            _config.rms_norm_eps,
                            static_cast<int>(bs),
                            static_cast<int>(_config.q_lora_rank),
                            _stream->Handle());

    // Q compress backward: grad_q_compressed -> grad_input_q
    CAIF_DeviceTensor grad_input_q;
    if(_use_projections==true)
    {
      grad_input_q=_projections.q_compress->Backward(grad_q_compressed);
    }
    else
    {
      grad_input_q=CAIF_DeviceTensor::Uninitialized({bs,_config.dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_q_compressed,_w_q_compress,grad_input_q);

      CAIF_DeviceTensor grad_w_qc=CAIF_DeviceTensor::Uninitialized(
          {_config.dim,_config.q_lora_rank},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_q_compressed,grad_w_qc);
      CAIF_DeviceOps::Add(_grad_w_q_compress,grad_w_qc,_grad_w_q_compress);
    }

    //------------------------------------------------------------------
    // KV path backward
    //------------------------------------------------------------------
    const int kv_rows=static_cast<int>(bh*seq_len);

    // Split grad_k into grad_k_nope [bh,seq,nope] + grad_k_pe [bh,seq,rope]
    CAIF_DeviceTensor grad_k_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},*_stream);
    CAIF_DeviceTensor grad_k_pe=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},*_stream);
    launch_slice_last_dim(grad_k.DevicePtr(),grad_k_nope.DevicePtr(),
                          kv_rows,static_cast<int>(_qk_head_dim),0,static_cast<int>(nope),
                          _stream->Handle());
    launch_slice_last_dim(grad_k.DevicePtr(),grad_k_pe.DevicePtr(),
                          kv_rows,static_cast<int>(_qk_head_dim),
                          static_cast<int>(nope),static_cast<int>(rope),
                          _stream->Handle());

    // Inverse RoPE on grad_k_pe
    launch_rope_backward(grad_k_pe.DevicePtr(),
                         static_cast<int>(bh),
                         static_cast<int>(seq_len),
                         static_cast<int>(rope),
                         _config.rope_base,
                         _stream->Handle());

    // Reduce k_pe broadcast: [bh, seq, rope] -> [batch, seq, rope]
    CAIF_DeviceTensor grad_k_pe_flat=CAIF_DeviceTensor::Zeros({batch,seq_len,rope},*_stream);
    launch_gqa_reduce_kv(grad_k_pe.DevicePtr(),
                         grad_k_pe_flat.DevicePtr(),
                         static_cast<int>(batch),
                         1,
                         static_cast<int>(_config.num_heads),
                         static_cast<int>(seq_len),
                         static_cast<int>(rope),
                         _stream->Handle());
    grad_k_pe_flat.Reshape({bs,rope});

    // Reassemble grad_kv_decomp: concat [k_nope | v] per head -> [bh, seq, nope+v_dim]
    const uint32_t kv_per_head=nope+v_dim;
    CAIF_DeviceTensor grad_kv_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,kv_per_head},*_stream);
    launch_concat_last_dim(grad_k_nope.DevicePtr(),grad_v.DevicePtr(),
                           grad_kv_heads.DevicePtr(),
                           kv_rows,static_cast<int>(nope),static_cast<int>(v_dim),
                           _stream->Handle());

    // Reverse transpose: [bh, seq, kv_per_head] -> [batch, seq, heads, kv_per_head] -> [bs, kv_decomp_dim]
    CAIF_DeviceTensor grad_kv_merged=CAIF_DeviceTensor::Uninitialized(
        {batch,seq_len,_config.num_heads,kv_per_head},*_stream);
    launch_transpose_0213(grad_kv_heads.DevicePtr(),
                          grad_kv_merged.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(kv_per_head),
                          _stream->Handle());
    grad_kv_merged.Reshape({bs,_kv_decomp_dim});

    // KV decompress backward: grad_kv_merged -> grad_kv_normed
    CAIF_DeviceTensor grad_kv_normed;
    if(_use_projections==true)
    {
      grad_kv_normed=_projections.kv_decompress->Backward(grad_kv_merged);
    }
    else
    {
      grad_kv_normed=CAIF_DeviceTensor::Uninitialized({bs,_config.kv_lora_rank},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_kv_merged,_w_kv_decompress,grad_kv_normed);

      CAIF_DeviceTensor grad_w_kvd=CAIF_DeviceTensor::Uninitialized(
          {_config.kv_lora_rank,_kv_decomp_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_kv_normed,grad_kv_merged,grad_w_kvd);
      CAIF_DeviceOps::Add(_grad_w_kv_decompress,grad_w_kvd,_grad_w_kv_decompress);
    }

    // KV RMSNorm backward
    CAIF_DeviceTensor grad_kv_compressed=CAIF_DeviceTensor::Uninitialized(
        {bs,_config.kv_lora_rank},*_stream);
    launch_rmsnorm_backward(grad_kv_normed.DevicePtr(),
                            _cached_kv_compressed.DevicePtr(),
                            _kv_norm_gamma.DevicePtr(),
                            _cached_kv_rms.DevicePtr(),
                            grad_kv_compressed.DevicePtr(),
                            _grad_kv_norm_gamma.DevicePtr(),
                            _config.rms_norm_eps,
                            static_cast<int>(bs),
                            static_cast<int>(_config.kv_lora_rank),
                            _stream->Handle());

    // Reassemble grad_kv_out: concat [kv_compressed | k_pe_flat] -> [bs, kv_compress_dim]
    CAIF_DeviceTensor grad_kv_out=CAIF_DeviceTensor::Uninitialized({bs,_kv_compress_dim},*_stream);
    launch_concat_last_dim(grad_kv_compressed.DevicePtr(),grad_k_pe_flat.DevicePtr(),
                           grad_kv_out.DevicePtr(),
                           static_cast<int>(bs),
                           static_cast<int>(_config.kv_lora_rank),static_cast<int>(rope),
                           _stream->Handle());

    // KV compress backward: grad_kv_out -> grad_input_kv
    CAIF_DeviceTensor grad_input_kv;
    if(_use_projections==true)
    {
      grad_input_kv=_projections.kv_compress->Backward(grad_kv_out);
    }
    else
    {
      grad_input_kv=CAIF_DeviceTensor::Uninitialized({bs,_config.dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_kv_out,_w_kv_compress,grad_input_kv);

      CAIF_DeviceTensor grad_w_kvc=CAIF_DeviceTensor::Uninitialized(
          {_config.dim,_kv_compress_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_kv_out,grad_w_kvc);
      CAIF_DeviceOps::Add(_grad_w_kv_compress,grad_w_kvc,_grad_w_kv_compress);
    }

    //------------------------------------------------------------------
    // Combine input gradients
    //------------------------------------------------------------------
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({bs,_config.dim},*_stream);
    CAIF_DeviceOps::Add(grad_input_q,grad_input_kv,grad_input);
    grad_input.Reshape({batch,seq_len,_config.dim});

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Parameter management
//------------------------------------------------------------------------------

void CAIF_DeviceMLAttention::ZeroGradients()
{
  if(_use_projections==true)
  {
    _projections.q_compress->ZeroGradients();
    _projections.q_decompress->ZeroGradients();
    _projections.kv_compress->ZeroGradients();
    _projections.kv_decompress->ZeroGradients();
    _projections.o_proj->ZeroGradients();
    _grad_q_norm_gamma.Fill(0.0f);
    _grad_kv_norm_gamma.Fill(0.0f);
  }
  else
  {
    _grad_w_q_compress.Fill(0.0f);
    _grad_q_norm_gamma.Fill(0.0f);
    _grad_w_q_decompress.Fill(0.0f);
    _grad_w_kv_compress.Fill(0.0f);
    _grad_kv_norm_gamma.Fill(0.0f);
    _grad_w_kv_decompress.Fill(0.0f);
    _grad_w_o.Fill(0.0f);
  }
}

size_t CAIF_DeviceMLAttention::ParameterTensorCount()const
{
  if(_use_projections==true)
  {
    return _projections.q_compress->ParameterTensorCount()+
           _projections.q_decompress->ParameterTensorCount()+
           _projections.kv_compress->ParameterTensorCount()+
           _projections.kv_decompress->ParameterTensorCount()+
           _projections.o_proj->ParameterTensorCount()+
           2;
  }
  return 7;
}

CAIF_DeviceTensor &CAIF_DeviceMLAttention::ParameterTensor(size_t index)
{
  if(_use_projections==true)
  {
    size_t offset=0;
    CAIF_DeviceLayer *projs[]={_projections.q_compress.get(),
                              _projections.q_decompress.get(),
                              _projections.kv_compress.get(),
                              _projections.kv_decompress.get(),
                              _projections.o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return _q_norm_gamma;}
    if(index==offset+1){return _kv_norm_gamma;}
    THROW_CAIFE("MLA ParameterTensor: index out of range");
  }
  if(index==0){return _w_q_compress;}
  if(index==1){return _q_norm_gamma;}
  if(index==2){return _w_q_decompress;}
  if(index==3){return _w_kv_compress;}
  if(index==4){return _kv_norm_gamma;}
  if(index==5){return _w_kv_decompress;}
  if(index==6){return _w_o;}
  THROW_CAIFE("MLA ParameterTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMLAttention::ParameterTensor(size_t index)const
{
  if(_use_projections==true)
  {
    size_t offset=0;
    const CAIF_DeviceLayer *projs[]={_projections.q_compress.get(),
                                    _projections.q_decompress.get(),
                                    _projections.kv_compress.get(),
                                    _projections.kv_decompress.get(),
                                    _projections.o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return _q_norm_gamma;}
    if(index==offset+1){return _kv_norm_gamma;}
    THROW_CAIFE("MLA ParameterTensor: index out of range");
  }
  if(index==0){return _w_q_compress;}
  if(index==1){return _q_norm_gamma;}
  if(index==2){return _w_q_decompress;}
  if(index==3){return _w_kv_compress;}
  if(index==4){return _kv_norm_gamma;}
  if(index==5){return _w_kv_decompress;}
  if(index==6){return _w_o;}
  THROW_CAIFE("MLA ParameterTensor: index out of range");
}

CAIF_DeviceTensor &CAIF_DeviceMLAttention::GradientTensor(size_t index)
{
  if(_use_projections==true)
  {
    size_t offset=0;
    CAIF_DeviceLayer *projs[]={_projections.q_compress.get(),
                              _projections.q_decompress.get(),
                              _projections.kv_compress.get(),
                              _projections.kv_decompress.get(),
                              _projections.o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->GradientTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return _grad_q_norm_gamma;}
    if(index==offset+1){return _grad_kv_norm_gamma;}
    THROW_CAIFE("MLA GradientTensor: index out of range");
  }
  if(index==0){return _grad_w_q_compress;}
  if(index==1){return _grad_q_norm_gamma;}
  if(index==2){return _grad_w_q_decompress;}
  if(index==3){return _grad_w_kv_compress;}
  if(index==4){return _grad_kv_norm_gamma;}
  if(index==5){return _grad_w_kv_decompress;}
  if(index==6){return _grad_w_o;}
  THROW_CAIFE("MLA GradientTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMLAttention::GradientTensor(size_t index)const
{
  if(_use_projections==true)
  {
    size_t offset=0;
    const CAIF_DeviceLayer *projs[]={_projections.q_compress.get(),
                                    _projections.q_decompress.get(),
                                    _projections.kv_compress.get(),
                                    _projections.kv_decompress.get(),
                                    _projections.o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->GradientTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return _grad_q_norm_gamma;}
    if(index==offset+1){return _grad_kv_norm_gamma;}
    THROW_CAIFE("MLA GradientTensor: index out of range");
  }
  if(index==0){return _grad_w_q_compress;}
  if(index==1){return _grad_q_norm_gamma;}
  if(index==2){return _grad_w_q_decompress;}
  if(index==3){return _grad_w_kv_compress;}
  if(index==4){return _grad_kv_norm_gamma;}
  if(index==5){return _grad_w_kv_decompress;}
  if(index==6){return _grad_w_o;}
  THROW_CAIFE("MLA GradientTensor: index out of range");
}

size_t CAIF_DeviceMLAttention::TotalParameterCount()const
{
  if(_use_projections==true)
  {
    return _projections.q_compress->TotalParameterCount()+
           _projections.q_decompress->TotalParameterCount()+
           _projections.kv_compress->TotalParameterCount()+
           _projections.kv_decompress->TotalParameterCount()+
           _projections.o_proj->TotalParameterCount()+
           _config.q_lora_rank+
           _config.kv_lora_rank;
  }
  size_t total=0;
  for(size_t i=0;i<ParameterTensorCount();++i)
  {
    total+=ParameterTensor(i).TotalElements();
  }
  return total;
}

std::string CAIF_DeviceMLAttention::Description()const
{
  std::ostringstream ss;
  ss<<"MLA(dim="<<_config.dim
    <<",heads="<<_config.num_heads
    <<",q_lora="<<_config.q_lora_rank
    <<",kv_lora="<<_config.kv_lora_rank
    <<",nope="<<_config.qk_nope_head_dim
    <<",rope="<<_config.qk_rope_head_dim
    <<",v="<<_config.v_head_dim
    <<",params="<<TotalParameterCount();
  if(_use_projections==true)
  {
    ss<<",projections";
  }
  ss<<")";
  return ss.str();
}

std::vector<std::string> CAIF_DeviceMLAttention::ParameterNames(const std::string &prefix)const
{
  if(_use_projections==true)
  {
    std::vector<std::string> names;
    std::vector<std::string> sub;

    sub=_projections.q_compress->ParameterNames(prefix+"q_a_proj.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=_projections.q_decompress->ParameterNames(prefix+"q_b_proj.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=_projections.kv_compress->ParameterNames(prefix+"kv_a_proj_with_mqa.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=_projections.kv_decompress->ParameterNames(prefix+"kv_b_proj.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=_projections.o_proj->ParameterNames(prefix+"o_proj.");
    names.insert(names.end(),sub.begin(),sub.end());

    names.push_back(prefix+"q_a_layernorm.weight");
    names.push_back(prefix+"kv_a_layernorm.weight");
    return names;
  }
  std::vector<std::string> names;
  names.push_back(prefix+"q_a_proj.weight");
  names.push_back(prefix+"q_a_layernorm.weight");
  names.push_back(prefix+"q_b_proj.weight");
  names.push_back(prefix+"kv_a_proj_with_mqa.weight");
  names.push_back(prefix+"kv_a_layernorm.weight");
  names.push_back(prefix+"kv_b_proj.weight");
  names.push_back(prefix+"o_proj.weight");
  return names;
}

//------------------------------------------------------------------------------
// Weight initialization
//------------------------------------------------------------------------------

void CAIF_DeviceMLAttention::InitializeWeights(uint32_t seed)
{
  try
  {
    std::mt19937 rng(seed);

    // Xavier uniform for weight matrices
    const float limit_qc=std::sqrt(6.0f/static_cast<float>(_config.dim+_config.q_lora_rank));
    const float limit_qd=std::sqrt(6.0f/static_cast<float>(_config.q_lora_rank+_q_proj_dim));
    const float limit_kvc=std::sqrt(6.0f/static_cast<float>(_config.dim+_kv_compress_dim));
    const float limit_kvd=std::sqrt(6.0f/static_cast<float>(_config.kv_lora_rank+_kv_decomp_dim));
    const float limit_o=std::sqrt(6.0f/static_cast<float>(_o_input_dim+_config.dim));

    // Helper to fill a tensor with uniform random values
    auto fill_uniform=[&](CAIF_DeviceTensor &tensor,float limit)
                      {
                        std::uniform_real_distribution<float> dist(-limit,limit);
                        std::vector<float> data(tensor.TotalElements());
                        for(size_t i=0;i<data.size();++i)
                        {
                          data[i]=dist(rng);
                        }
                        tensor.CopyFromHost(data.data(),data.size());
                      };

    fill_uniform(_w_q_compress,limit_qc);
    fill_uniform(_w_q_decompress,limit_qd);
    fill_uniform(_w_kv_compress,limit_kvc);
    fill_uniform(_w_kv_decompress,limit_kvd);
    fill_uniform(_w_o,limit_o);

    // Gamma stays at 1.0 (set in constructor)
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// KV-Cache management
//------------------------------------------------------------------------------

void CAIF_DeviceMLAttention::EnableKVCache(uint32_t batch_size,uint32_t max_seq_len)
{
  try
  {
    _kv_cache_compressed=CAIF_DeviceTensor::Zeros(
        {batch_size,max_seq_len,_config.kv_lora_rank},*_stream);
    _kv_cache_k_pe=CAIF_DeviceTensor::Zeros(
        {batch_size,max_seq_len,_config.qk_rope_head_dim},*_stream);
    _kv_cache_len=0;
    _kv_cache_max_len=max_seq_len;
    _kv_cache_batch=batch_size;
    _kv_cache_enabled=true;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMLAttention::DisableKVCache()
{
  _kv_cache_compressed=CAIF_DeviceTensor();
  _kv_cache_k_pe=CAIF_DeviceTensor();
  _kv_cache_len=0;
  _kv_cache_max_len=0;
  _kv_cache_batch=0;
  _kv_cache_enabled=false;
}

void CAIF_DeviceMLAttention::ResetKVCache()
{
  if(_kv_cache_enabled==true)
  {
    _kv_cache_compressed.Fill(0.0f);
    _kv_cache_k_pe.Fill(0.0f);
    _kv_cache_len=0;
  }
}

//------------------------------------------------------------------------------
// Cached forward (autoregressive inference)
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_DeviceMLAttention::ForwardCached(const CAIF_DeviceTensor &input)
{
  try
  {
    if(_kv_cache_enabled==false)
    {
      THROW_CAIFE("MLA ForwardCached: KV cache not enabled");
    }

    const auto &shape=input.Shape();
    if(shape.size()!=3||shape[2]!=_config.dim)
    {
      THROW_CAIFE("MLA ForwardCached: input must be [batch, seq_len, dim]");
    }

    const uint32_t batch=shape[0];
    const uint32_t new_len=shape[1];
    const uint32_t bs=batch*new_len;
    const uint32_t bh=batch*_config.num_heads;
    const uint32_t nope=_config.qk_nope_head_dim;
    const uint32_t rope=_config.qk_rope_head_dim;
    const uint32_t v_dim=_config.v_head_dim;
    const uint32_t total_len=_kv_cache_len+new_len;
    const float scale=1.0f/std::sqrt(static_cast<float>(_qk_head_dim));

    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,_config.dim});

    //------------------------------------------------------------------
    // Q path (same as full forward)
    //------------------------------------------------------------------
    CAIF_DeviceTensor q_compressed=CAIF_DeviceTensor::Uninitialized({bs,_config.q_lora_rank},*_stream);
    CAIF_DeviceOps::MatMul(flat_input,_w_q_compress,q_compressed);

    CAIF_DeviceTensor q_normed=CAIF_DeviceTensor::Uninitialized({bs,_config.q_lora_rank},*_stream);
    CAIF_DeviceTensor q_rms=CAIF_DeviceTensor::Uninitialized({bs},*_stream);
    launch_rmsnorm_forward(q_compressed.DevicePtr(),
                           _q_norm_gamma.DevicePtr(),
                           q_normed.DevicePtr(),
                           q_rms.DevicePtr(),
                           _config.rms_norm_eps,
                           static_cast<int>(bs),
                           static_cast<int>(_config.q_lora_rank),
                           _stream->Handle());

    CAIF_DeviceTensor q_full=CAIF_DeviceTensor::Uninitialized({bs,_q_proj_dim},*_stream);
    CAIF_DeviceOps::MatMul(q_normed,_w_q_decompress,q_full);

    q_full.Reshape({batch,new_len,_config.num_heads,_qk_head_dim});
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized({bh,new_len,_qk_head_dim},*_stream);
    launch_transpose_0213(q_full.DevicePtr(),
                          q_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(new_len),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(_qk_head_dim),
                          _stream->Handle());

    const int q_rows=static_cast<int>(bh*new_len);
    CAIF_DeviceTensor q_nope=CAIF_DeviceTensor::Uninitialized({bh,new_len,nope},*_stream);
    CAIF_DeviceTensor q_rope_t=CAIF_DeviceTensor::Uninitialized({bh,new_len,rope},*_stream);
    launch_slice_last_dim(q_transposed.DevicePtr(),q_nope.DevicePtr(),
                          q_rows,static_cast<int>(_qk_head_dim),0,static_cast<int>(nope),
                          _stream->Handle());
    launch_slice_last_dim(q_transposed.DevicePtr(),q_rope_t.DevicePtr(),
                          q_rows,static_cast<int>(_qk_head_dim),
                          static_cast<int>(nope),static_cast<int>(rope),
                          _stream->Handle());

    launch_rope_forward_offset(q_rope_t.DevicePtr(),
                               static_cast<int>(bh),
                               static_cast<int>(new_len),
                               static_cast<int>(rope),
                               _config.rope_base,
                               static_cast<int>(_kv_cache_len),
                               _stream->Handle());

    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized({bh,new_len,_qk_head_dim},*_stream);
    launch_concat_last_dim(q_nope.DevicePtr(),q_rope_t.DevicePtr(),q.DevicePtr(),
                           q_rows,static_cast<int>(nope),static_cast<int>(rope),
                           _stream->Handle());

    //------------------------------------------------------------------
    // KV path: compress new tokens, append to cache, decompress full cache
    //------------------------------------------------------------------
    CAIF_DeviceTensor kv_out=CAIF_DeviceTensor::Uninitialized({bs,_kv_compress_dim},*_stream);
    CAIF_DeviceOps::MatMul(flat_input,_w_kv_compress,kv_out);

    CAIF_DeviceTensor new_kv_compressed=CAIF_DeviceTensor::Uninitialized({bs,_config.kv_lora_rank},*_stream);
    CAIF_DeviceTensor new_k_pe=CAIF_DeviceTensor::Uninitialized({bs,rope},*_stream);
    launch_slice_last_dim(kv_out.DevicePtr(),new_kv_compressed.DevicePtr(),
                          static_cast<int>(bs),static_cast<int>(_kv_compress_dim),
                          0,static_cast<int>(_config.kv_lora_rank),
                          _stream->Handle());
    launch_slice_last_dim(kv_out.DevicePtr(),new_k_pe.DevicePtr(),
                          static_cast<int>(bs),static_cast<int>(_kv_compress_dim),
                          static_cast<int>(_config.kv_lora_rank),static_cast<int>(rope),
                          _stream->Handle());

    // Append to KV cache
    new_kv_compressed.Reshape({batch,new_len,_config.kv_lora_rank});
    launch_kv_cache_append(new_kv_compressed.DevicePtr(),
                           _kv_cache_compressed.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(new_len),
                           static_cast<int>(_kv_cache_len),
                           static_cast<int>(_kv_cache_max_len),
                           1,
                           static_cast<int>(_config.kv_lora_rank),
                           _stream->Handle());

    new_k_pe.Reshape({batch,new_len,rope});
    launch_kv_cache_append(new_k_pe.DevicePtr(),
                           _kv_cache_k_pe.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(new_len),
                           static_cast<int>(_kv_cache_len),
                           static_cast<int>(_kv_cache_max_len),
                           1,
                           static_cast<int>(rope),
                           _stream->Handle());

    // Decompress full cached KV: [batch, total_len, kv_lora_rank]
    const uint32_t total_bs=batch*total_len;
    CAIF_DeviceTensor cached_kv=CAIF_DeviceTensor::Uninitialized(
        {batch,total_len,_config.kv_lora_rank},*_stream);

    // Copy the valid portion of the cache
    CAIF_DeviceTensor cache_view=_kv_cache_compressed.Clone();
    cache_view.Reshape({batch*_kv_cache_max_len,_config.kv_lora_rank});

    // RMSNorm on full cached compressed KV
    cached_kv.Reshape({total_bs,_config.kv_lora_rank});
    CAIF_DeviceTensor kv_normed=CAIF_DeviceTensor::Uninitialized({total_bs,_config.kv_lora_rank},*_stream);
    CAIF_DeviceTensor kv_rms=CAIF_DeviceTensor::Uninitialized({total_bs},*_stream);

    // Extract valid cache range for RMSNorm
    // We need to extract [batch, 0:total_len, kv_lora_rank] from cache [batch, max_len, kv_lora_rank]
    // Use slice per batch or reshape approach
    // For simplicity, copy valid portion row by row
    for(uint32_t b=0;b<batch;++b)
    {
      const float *src=_kv_cache_compressed.DevicePtr()+
                       b*_kv_cache_max_len*_config.kv_lora_rank;
      float *dst=cached_kv.DevicePtr()+b*total_len*_config.kv_lora_rank;
      const size_t copy_bytes=total_len*_config.kv_lora_rank*sizeof(float);
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(dst,src,copy_bytes,cudaMemcpyDeviceToDevice,_stream->Handle());
#endif
    }

    launch_rmsnorm_forward(cached_kv.DevicePtr(),
                           _kv_norm_gamma.DevicePtr(),
                           kv_normed.DevicePtr(),
                           kv_rms.DevicePtr(),
                           _config.rms_norm_eps,
                           static_cast<int>(total_bs),
                           static_cast<int>(_config.kv_lora_rank),
                           _stream->Handle());

    // KV decompress full sequence
    CAIF_DeviceTensor kv_full=CAIF_DeviceTensor::Uninitialized({total_bs,_kv_decomp_dim},*_stream);
    CAIF_DeviceOps::MatMul(kv_normed,_w_kv_decompress,kv_full);

    const uint32_t kv_per_head=nope+v_dim;
    kv_full.Reshape({batch,total_len,_config.num_heads,kv_per_head});
    CAIF_DeviceTensor kv_transposed=CAIF_DeviceTensor::Uninitialized(
        {bh,total_len,kv_per_head},*_stream);
    launch_transpose_0213(kv_full.DevicePtr(),
                          kv_transposed.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(total_len),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(kv_per_head),
                          _stream->Handle());

    const int full_kv_rows=static_cast<int>(bh*total_len);
    CAIF_DeviceTensor k_nope=CAIF_DeviceTensor::Uninitialized({bh,total_len,nope},*_stream);
    CAIF_DeviceTensor v_heads=CAIF_DeviceTensor::Uninitialized({bh,total_len,v_dim},*_stream);
    launch_slice_last_dim(kv_transposed.DevicePtr(),k_nope.DevicePtr(),
                          full_kv_rows,static_cast<int>(kv_per_head),
                          0,static_cast<int>(nope),
                          _stream->Handle());
    launch_slice_last_dim(kv_transposed.DevicePtr(),v_heads.DevicePtr(),
                          full_kv_rows,static_cast<int>(kv_per_head),
                          static_cast<int>(nope),static_cast<int>(v_dim),
                          _stream->Handle());

    // Extract and broadcast cached k_pe for full sequence
    CAIF_DeviceTensor cached_k_pe=CAIF_DeviceTensor::Uninitialized({batch,total_len,rope},*_stream);
    for(uint32_t b=0;b<batch;++b)
    {
      const float *src=_kv_cache_k_pe.DevicePtr()+b*_kv_cache_max_len*rope;
      float *dst=cached_k_pe.DevicePtr()+b*total_len*rope;
      const size_t copy_bytes=total_len*rope*sizeof(float);
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(dst,src,copy_bytes,cudaMemcpyDeviceToDevice,_stream->Handle());
#endif
    }

    CAIF_DeviceTensor k_pe_expanded=CAIF_DeviceTensor::Uninitialized({bh,total_len,rope},*_stream);
    launch_gqa_repeat_kv(cached_k_pe.DevicePtr(),
                         k_pe_expanded.DevicePtr(),
                         static_cast<int>(batch),
                         1,
                         static_cast<int>(_config.num_heads),
                         static_cast<int>(total_len),
                         static_cast<int>(rope),
                         _stream->Handle());

    // Apply RoPE to k_pe (full sequence positions)
    launch_rope_forward(k_pe_expanded.DevicePtr(),
                        static_cast<int>(bh),
                        static_cast<int>(total_len),
                        static_cast<int>(rope),
                        _config.rope_base,
                        _stream->Handle());

    // Assemble K: [k_nope | k_pe] -> [bh, total_len, qk_head_dim]
    CAIF_DeviceTensor k=CAIF_DeviceTensor::Uninitialized({bh,total_len,_qk_head_dim},*_stream);
    launch_concat_last_dim(k_nope.DevicePtr(),k_pe_expanded.DevicePtr(),k.DevicePtr(),
                           full_kv_rows,static_cast<int>(nope),static_cast<int>(rope),
                           _stream->Handle());

    //------------------------------------------------------------------
    // Attention: Q [bh, new_len, qk_head_dim] x K [bh, total_len, qk_head_dim]
    //------------------------------------------------------------------
    // Use batched matmul for non-square attention (query_len != key_len)
    CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},*_stream);
    CAIF_DeviceOps::BatchedMatMulTransposeB(q,k,scores,
                                           static_cast<int>(new_len),
                                           static_cast<int>(_qk_head_dim),
                                           static_cast<int>(total_len),
                                           static_cast<int>(bh));
    CAIF_DeviceOps::Scale(scores,scale);

    if(_config.causal==true)
    {
      launch_causal_mask_fill_offset(scores.DevicePtr(),
                                     static_cast<int>(bh),
                                     static_cast<int>(new_len),
                                     static_cast<int>(total_len),
                                     static_cast<int>(_kv_cache_len),
                                     _stream->Handle());
    }

    CAIF_DeviceTensor attn_weights=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},*_stream);
    launch_attention_softmax(scores.DevicePtr(),
                             attn_weights.DevicePtr(),
                             static_cast<int>(bh*new_len),
                             static_cast<int>(total_len),
                             _stream->Handle());

    CAIF_DeviceTensor attn_output=CAIF_DeviceTensor::Uninitialized({bh,new_len,v_dim},*_stream);
    CAIF_DeviceOps::BatchedMatMul(attn_weights,v_heads,attn_output,
                                 static_cast<int>(new_len),
                                 static_cast<int>(total_len),
                                 static_cast<int>(v_dim),
                                 static_cast<int>(bh));

    //------------------------------------------------------------------
    // Output projection
    //------------------------------------------------------------------
    CAIF_DeviceTensor attn_merged=CAIF_DeviceTensor::Uninitialized(
        {batch,new_len,_config.num_heads,v_dim},*_stream);
    launch_transpose_0213(attn_output.DevicePtr(),
                          attn_merged.DevicePtr(),
                          static_cast<int>(batch),
                          static_cast<int>(_config.num_heads),
                          static_cast<int>(new_len),
                          static_cast<int>(v_dim),
                          _stream->Handle());
    attn_merged.Reshape({bs,_o_input_dim});

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({bs,_config.dim},*_stream);
    CAIF_DeviceOps::MatMul(attn_merged,_w_o,output);

    _kv_cache_len=total_len;

    output.Reshape({batch,new_len,_config.dim});
    return output;
  }
  CAIF_CATCH_BLOCK()
}
