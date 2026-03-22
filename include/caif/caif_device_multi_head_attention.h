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
// Device-resident Multi-Head Attention layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_MULTI_HEAD_ATTENTION_H
#define CAIF_DEVICE_MULTI_HEAD_ATTENTION_H

#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_constants.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Multi-Head Attention layer (device-resident)
 *
 * Implements scaled dot-product multi-head attention:
 *   Q = input @ W_q
 *   K = input @ W_k
 *   V = input @ W_v
 *   scores = (Q_heads @ K_heads^T) / sqrt(head_dim)
 *   attn = softmax(scores)   [with optional causal mask]
 *   context = attn @ V_heads
 *   output = merge_heads(context) @ W_o
 *
 * Phase 3: standard MHA (num_kv_heads == num_heads), no RoPE, no GQA,
 * no dropout. Input [batch, seq_len, dim] -> Output [batch, seq_len, dim].
 *
 * Parameters: W_q, W_k, W_v, W_o (4 weight matrices, no biases).
 * Initialized with Xavier uniform by default.
 */
class CAIF_DeviceMultiHeadAttention:public CAIF_DeviceLayer
{
  public:
    struct AttentionConfig_t
    {
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      uint32_t head_dim;
      bool causal;
      bool use_rope;
      float rope_base;
      float dropout_rate;
    };

    struct MHAProjections_t
    {
      std::unique_ptr<CAIF_DeviceLayer> q_proj;
      std::unique_ptr<CAIF_DeviceLayer> k_proj;
      std::unique_ptr<CAIF_DeviceLayer> v_proj;
      std::unique_ptr<CAIF_DeviceLayer> o_proj;
      CAIF_DeviceTensor q_bias;
      CAIF_DeviceTensor k_bias;
      CAIF_DeviceTensor v_bias;
    };

    CAIF_DeviceMultiHeadAttention(const AttentionConfig_t &config,
                                 CAIF_CudaStream &stream);
    CAIF_DeviceMultiHeadAttention(const AttentionConfig_t &config,
                                 MHAProjections_t projections,
                                 CAIF_CudaStream &stream);
    ~CAIF_DeviceMultiHeadAttention()override=default;

    // Move
    CAIF_DeviceMultiHeadAttention(CAIF_DeviceMultiHeadAttention &&other);
    CAIF_DeviceMultiHeadAttention &operator=(CAIF_DeviceMultiHeadAttention &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training)override;
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)override;
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors
    const AttentionConfig_t &Config()const{return _config;}
    CAIF_DeviceTensor &QBias(){return _projections.q_bias;}
    CAIF_DeviceTensor &KBias(){return _projections.k_bias;}
    CAIF_DeviceTensor &VBias(){return _projections.v_bias;}

    // Weight initialization (Xavier uniform)
    void InitializeWeights(uint32_t seed=0);

    // KV-Cache management for inference
    void EnableKVCache(uint32_t batch_size,uint32_t max_seq_len);
    void DisableKVCache();
    void ResetKVCache();
    bool IsKVCacheEnabled()const{return _kv_cache_enabled;}
    uint32_t KVCacheLength()const{return _kv_cache_len;}

    // Cached forward for autoregressive inference (no backward support)
    // Input: [batch, seq_len, dim] where seq_len can be full prompt or 1 token
    // Returns: [batch, seq_len, dim]
    CAIF_DeviceTensor ForwardCached(const CAIF_DeviceTensor &input);

  protected:

  private:
    AttentionConfig_t _config;

    // External projection layers (when _use_projections==true)
    MHAProjections_t _projections;
    bool _use_projections;

    // Parameters: 4 weight matrices, no biases (when _use_projections==false)
    CAIF_DeviceTensor _w_q;      // [dim, num_heads * head_dim]
    CAIF_DeviceTensor _w_k;      // [dim, num_kv_heads * head_dim]
    CAIF_DeviceTensor _w_v;      // [dim, num_kv_heads * head_dim]
    CAIF_DeviceTensor _w_o;      // [num_heads * head_dim, dim]

    // Gradients
    CAIF_DeviceTensor _grad_w_q;
    CAIF_DeviceTensor _grad_w_k;
    CAIF_DeviceTensor _grad_w_v;
    CAIF_DeviceTensor _grad_w_o;

    // Cached for backward (populated when training==true)
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_q_heads;
    CAIF_DeviceTensor _cached_k_heads;
    CAIF_DeviceTensor _cached_v_heads;
    CAIF_DeviceTensor _cached_attn;
    CAIF_DeviceTensor _cached_concat;
    CAIF_DeviceTensor _cached_logsumexp;   // [batch*heads, seq_len] for FlashAttention backward
    CAIF_DeviceTensor _cached_output;       // [batch*heads, seq_len, head_dim] for FlashAttention backward
    uint32_t _cached_batch;
    uint32_t _cached_seq_len;

    // KV-cache for inference
    CAIF_DeviceTensor _kv_cache_k;    // [batch, max_seq_len, num_kv_heads, head_dim]
    CAIF_DeviceTensor _kv_cache_v;    // [batch, max_seq_len, num_kv_heads, head_dim]
    uint32_t _kv_cache_len;          // Current cached sequence length
    uint32_t _kv_cache_max_len;      // Maximum sequence length
    uint32_t _kv_cache_batch;        // Batch size for cache
    bool _kv_cache_enabled;          // Whether caching is active

    // FlashAttention flag — auto-selected per Forward() call based on
    // attention matrix size. Naive (cuBLAS) is faster for short sequences;
    // flash is needed for long sequences where O(n²) memory is prohibitive.
    // Threshold: 256MB for the attention matrix.
    bool _use_flash_attention;
    static constexpr size_t FLASH_ATTN_THRESHOLD_BYTES=256ULL*1024*1024;
};

}//end instance namespace

#endif  // CAIF_DEVICE_MULTI_HEAD_ATTENTION_H
