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
// Multi-head Latent Attention (MLA) layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_ML_ATTENTION_H
#define CAIF_DEVICE_ML_ATTENTION_H

#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Multi-head Latent Attention (MLA) layer.
 *
 * MLA compresses Q/K/V through low-rank bottlenecks with split RoPE:
 *
 * Q path: input -> W_q_compress -> RMSNorm -> W_q_decompress -> split heads
 *         -> split nope/rope -> RoPE on rope portion only
 *
 * KV path: input -> W_kv_compress -> split [compressed_kv | k_pe]
 *          compressed_kv -> RMSNorm -> W_kv_decompress -> split heads
 *          -> split k_nope/v; k_pe broadcast to all heads -> RoPE
 *
 * Attention: Q=[q_nope|q_rope], K=[k_nope|k_pe], V=v
 * Output: merge_heads -> W_o
 *
 * KV-cache stores compressed_kv + k_pe (576 floats/position vs 10240 for MHA).
 * Used by GLM-4.7-Flash and DeepSeek-V2/V3 architectures.
 */
class CAIF_DeviceMLAttention:public CAIF_DeviceLayer
{
  public:
    struct MLAConfig_t
    {
      uint32_t dim;
      uint32_t num_heads;
      uint32_t q_lora_rank;
      uint32_t kv_lora_rank;
      uint32_t qk_rope_head_dim;
      uint32_t qk_nope_head_dim;
      uint32_t v_head_dim;
      bool causal;
      float rope_base;
      float rms_norm_eps;
    };

    struct MLAProjections_t
    {
      std::unique_ptr<CAIF_DeviceLayer> q_compress;
      std::unique_ptr<CAIF_DeviceLayer> q_decompress;
      std::unique_ptr<CAIF_DeviceLayer> kv_compress;
      std::unique_ptr<CAIF_DeviceLayer> kv_decompress;
      std::unique_ptr<CAIF_DeviceLayer> o_proj;
    };

    CAIF_DeviceMLAttention(const MLAConfig_t &config,CAIF_CudaStream &stream);
    CAIF_DeviceMLAttention(const MLAConfig_t &config,
                           MLAProjections_t projections,
                           CAIF_CudaStream &stream);
    ~CAIF_DeviceMLAttention() override=default;

    // Move semantics
    CAIF_DeviceMLAttention(CAIF_DeviceMLAttention &&other);
    CAIF_DeviceMLAttention &operator=(CAIF_DeviceMLAttention &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training) override;
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output) override;
    void ZeroGradients() override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index) override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index) override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    const MLAConfig_t &Config()const{return _config;}

    // Weight initialization (Xavier uniform)
    void InitializeWeights(uint32_t seed=0);

    // KV-Cache management for inference
    void EnableKVCache(uint32_t batch_size,uint32_t max_seq_len);
    void DisableKVCache();
    void ResetKVCache();
    bool IsKVCacheEnabled()const{return _kv_cache_enabled;}
    uint32_t KVCacheLength()const{return _kv_cache_len;}

    // Cached forward for autoregressive inference
    CAIF_DeviceTensor ForwardCached(const CAIF_DeviceTensor &input);

  protected:

  private:
    MLAConfig_t _config;
    MLAProjections_t _projections;
    bool _use_projections;

    // Derived dimensions (computed once in constructor)
    uint32_t _qk_head_dim;     // qk_nope_head_dim + qk_rope_head_dim
    uint32_t _q_proj_dim;      // num_heads * _qk_head_dim
    uint32_t _kv_compress_dim; // kv_lora_rank + qk_rope_head_dim
    uint32_t _kv_decomp_dim;   // num_heads * (qk_nope_head_dim + v_head_dim)
    uint32_t _o_input_dim;     // num_heads * v_head_dim

    // Parameters: 7 tensors
    CAIF_DeviceTensor _w_q_compress;    // [dim, q_lora_rank]
    CAIF_DeviceTensor _q_norm_gamma;    // [q_lora_rank]
    CAIF_DeviceTensor _w_q_decompress;  // [q_lora_rank, q_proj_dim]
    CAIF_DeviceTensor _w_kv_compress;   // [dim, kv_compress_dim]
    CAIF_DeviceTensor _kv_norm_gamma;   // [kv_lora_rank]
    CAIF_DeviceTensor _w_kv_decompress; // [kv_lora_rank, kv_decomp_dim]
    CAIF_DeviceTensor _w_o;             // [o_input_dim, dim]

    // Gradients: 7 tensors
    CAIF_DeviceTensor _grad_w_q_compress;
    CAIF_DeviceTensor _grad_q_norm_gamma;
    CAIF_DeviceTensor _grad_w_q_decompress;
    CAIF_DeviceTensor _grad_w_kv_compress;
    CAIF_DeviceTensor _grad_kv_norm_gamma;
    CAIF_DeviceTensor _grad_w_kv_decompress;
    CAIF_DeviceTensor _grad_w_o;

    // Cached for backward (populated when training==true)
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_q_compressed;
    CAIF_DeviceTensor _cached_q_rms;
    CAIF_DeviceTensor _cached_q_normed;
    CAIF_DeviceTensor _cached_kv_compressed;
    CAIF_DeviceTensor _cached_kv_rms;
    CAIF_DeviceTensor _cached_kv_normed;
    CAIF_DeviceTensor _cached_q;
    CAIF_DeviceTensor _cached_k;
    CAIF_DeviceTensor _cached_v;
    CAIF_DeviceTensor _cached_attn_output;
    CAIF_DeviceTensor _cached_logsumexp;
    CAIF_DeviceTensor _cached_merged;
    uint32_t _cached_batch;
    uint32_t _cached_seq_len;

    // KV-cache for inference (stores compressed_kv + k_pe per position)
    CAIF_DeviceTensor _kv_cache_compressed; // [batch, max_seq, kv_lora_rank]
    CAIF_DeviceTensor _kv_cache_k_pe;       // [batch, max_seq, qk_rope_head_dim]
    uint32_t _kv_cache_len;
    uint32_t _kv_cache_max_len;
    uint32_t _kv_cache_batch;
    bool _kv_cache_enabled;
};

}//end instance namespace

#endif  // CAIF_DEVICE_ML_ATTENTION_H
