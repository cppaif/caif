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
// Device-resident Multi-Head Attention layer (templated on
// <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_tensor.h"
#include "caif_constants.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceMultiHeadAttention:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    // Index of each per-MHA parameter / gradient tensor when MHA does
    // not delegate to sub-projection layers. Matches the historical
    // (W_q, W_k, W_v, W_o) ordering. ParameterTensor / GradientTensor
    // dispatch on this enum instead of bare integer literals.
    enum class ParamSlot_e:size_t
    {
      WQ_e=0,
      WK_e=1,
      WV_e=2,
      WO_e=3
    };

    struct AttentionConfig_t
    {
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      uint32_t head_dim;
      bool causal;
      bool use_rope;
      float rope_base;
      int rope_style=0;
      // Number of leading dims of each head to rotate. 0 means
      // "same as head_dim" (full rotation, the legacy behavior).
      // For partial-rotary models like Glm4Moe (partial_rotary_factor
      // 0.5) set rope_dim = head_dim/2; the kernel rotates only the
      // first rope_dim dims and passes the trailing (head_dim - rope_dim)
      // dims through untouched.  Must be even when nonzero.
      int rope_dim=0;
      float dropout_rate;
      // RMSNorm epsilon used when QK-norm gammas are loaded. Default
      // matches HF's conventional 1e-5 (OLMoE / Olmo2 / Qwen3). No
      // effect when q_norm_gamma / k_norm_gamma are empty.
      float qk_norm_eps=1.0e-5f;
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
      // Optional QK-norm gammas: RMSNorm applied to Q and K after the
      // projection (+ bias) and before RoPE / GQA-expand. Shape is the
      // full projection output width (num_heads*head_dim for q,
      // num_kv_heads*head_dim for k). Empty = no norm (legacy behavior
      // for models without QK-norm; the load methods below populate
      // these only when the caller-side naming profile exposes the
      // tensors). Used by OLMoE, Qwen3, OLMo2 — every recent model
      // with the "QK-norm" trick.
      CAIF_DeviceTensor q_norm_gamma;
      CAIF_DeviceTensor k_norm_gamma;
    };

    CAIF_DeviceMultiHeadAttention(const AttentionConfig_t &config,
                                  CAIF_CudaStream &stream);
    CAIF_DeviceMultiHeadAttention(const AttentionConfig_t &config,
                                  MHAProjections_t projections,
                                  CAIF_CudaStream &stream);
    ~CAIF_DeviceMultiHeadAttention()override=default;

    CAIF_DeviceMultiHeadAttention(CAIF_DeviceMultiHeadAttention &&other);
    CAIF_DeviceMultiHeadAttention &operator=(CAIF_DeviceMultiHeadAttention &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::MHA_e;
    }
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;
    size_t FrozenTensorCount()const override;
    CAIF_DeviceTensor FrozenTensorFP32(size_t index)const override;
    std::vector<std::string> FrozenTensorNames(const std::string &prefix="")const override;

    const AttentionConfig_t &Config()const{return _config;}
    CAIF_DeviceTensor &QBias(){return _projections.q_bias;}
    const CAIF_DeviceTensor &QBias()const{return _projections.q_bias;}
    CAIF_DeviceTensor &KBias(){return _projections.k_bias;}
    const CAIF_DeviceTensor &KBias()const{return _projections.k_bias;}
    CAIF_DeviceTensor &VBias(){return _projections.v_bias;}
    const CAIF_DeviceTensor &VBias()const{return _projections.v_bias;}
    const CAIF_DeviceTensor &QNormGamma()const{return _projections.q_norm_gamma;}
    const CAIF_DeviceTensor &KNormGamma()const{return _projections.k_norm_gamma;}

    bool HasProjections()const{return _use_projections;}

    CAIF_DeviceLayer &QProj()
    {
      if(_projections.q_proj==nullptr)
      {
        THROW_CAIFE("MHA: q_proj is null");
      }
      return *_projections.q_proj;
    }
    const CAIF_DeviceLayer &QProj()const
    {
      if(_projections.q_proj==nullptr)
      {
        THROW_CAIFE("MHA: q_proj is null");
      }
      return *_projections.q_proj;
    }
    CAIF_DeviceLayer &KProj()
    {
      if(_projections.k_proj==nullptr)
      {
        THROW_CAIFE("MHA: k_proj is null");
      }
      return *_projections.k_proj;
    }
    const CAIF_DeviceLayer &KProj()const
    {
      if(_projections.k_proj==nullptr)
      {
        THROW_CAIFE("MHA: k_proj is null");
      }
      return *_projections.k_proj;
    }
    CAIF_DeviceLayer &VProj()
    {
      if(_projections.v_proj==nullptr)
      {
        THROW_CAIFE("MHA: v_proj is null");
      }
      return *_projections.v_proj;
    }
    const CAIF_DeviceLayer &VProj()const
    {
      if(_projections.v_proj==nullptr)
      {
        THROW_CAIFE("MHA: v_proj is null");
      }
      return *_projections.v_proj;
    }
    CAIF_DeviceLayer &OProj()
    {
      if(_projections.o_proj==nullptr)
      {
        THROW_CAIFE("MHA: o_proj is null");
      }
      return *_projections.o_proj;
    }
    const CAIF_DeviceLayer &OProj()const
    {
      if(_projections.o_proj==nullptr)
      {
        THROW_CAIFE("MHA: o_proj is null");
      }
      return *_projections.o_proj;
    }

    void InitializeWeights(uint32_t seed=0)override;

    void LoadWQ(CAIF_DeviceTensor &&w_q);
    void LoadWK(CAIF_DeviceTensor &&w_k);
    void LoadWV(CAIF_DeviceTensor &&w_v);
    void LoadWO(CAIF_DeviceTensor &&w_o);
    void LoadQBias(CAIF_DeviceTensor &&q_bias);
    void LoadKBias(CAIF_DeviceTensor &&k_bias);
    void LoadVBias(CAIF_DeviceTensor &&v_bias);
    // QK-norm gamma loaders. Shape [num_heads*head_dim] for q,
    // [num_kv_heads*head_dim] for k. Always fp32. When loaded, MHA's
    // forward applies RMSNorm(gamma, eps=1e-5) to the projection
    // output (post-bias-add) before RoPE / GQA-expand. No-op when
    // not loaded.
    void LoadQNormGamma(CAIF_DeviceTensor &&q_norm_gamma);
    void LoadKNormGamma(CAIF_DeviceTensor &&k_norm_gamma);
    bool HasQNormGamma()const{return _projections.q_norm_gamma.IsEmpty()==false;}
    bool HasKNormGamma()const{return _projections.k_norm_gamma.IsEmpty()==false;}

    void EnableKVCache(uint32_t batch_size,uint32_t max_seq_len);
    void DisableKVCache();
    void ResetKVCache();
    bool IsKVCacheEnabled()const{return _kv_cache_enabled;}
    uint32_t KVCacheLength()const{return _kv_cache_len;}

    CAIF_DeviceTensor ForwardCached(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx);

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

    virtual bool RequiresExplicitScores()const
    {
      return false;
    }

    virtual void ApplyScoreBias(CAIF_DeviceTensor &scores,uint32_t batch,
                                uint32_t seq_len,CAIF_RunContext &ctx)
    {
      (void)scores;
      (void)batch;
      (void)seq_len;
      (void)ctx;
    }

    virtual void BackwardScoreBias(const CAIF_DeviceTensor &grad_scores,
                                   uint32_t batch,uint32_t seq_len,
                                   CAIF_RunContext &ctx)
    {
      (void)grad_scores;
      (void)batch;
      (void)seq_len;
      (void)ctx;
    }

    void XavierInit(CAIF_DeviceTensor &tensor,std::mt19937 &gen,
                    uint32_t fan_in,uint32_t fan_out);

  private:
    void BuildFusedWqkv();

    // Internal accessors — single point of access for every member, even
    // from inside this class's own methods. See CODING_GUIDELINES.md
    // §Member Access. *Mut() forms hand out a non-const reference for
    // in-place mutation (Reshape, Add, std::move-target on the LHS of an
    // assignment is forbidden — use Set*() for rebinds). Most accessors
    // are private because nothing outside this class needs them.
    const MHAProjections_t &Projections()const{return _projections;}
    MHAProjections_t &ProjectionsMut(){return _projections;}
    bool WQkvDirty()const{return _w_qkv_dirty;}
    void SetWQkvDirty(const bool v){_w_qkv_dirty=v;}
    const CAIF_DeviceTensor &WQ()const{return _w_q;}
    CAIF_DeviceTensor &WQMut(){return _w_q;}
    const CAIF_DeviceTensor &WK()const{return _w_k;}
    CAIF_DeviceTensor &WKMut(){return _w_k;}
    const CAIF_DeviceTensor &WV()const{return _w_v;}
    CAIF_DeviceTensor &WVMut(){return _w_v;}
    const CAIF_DeviceTensor &WO()const{return _w_o;}
    CAIF_DeviceTensor &WOMut(){return _w_o;}
    const CAIF_DeviceTensor &WQkv()const{return _w_qkv;}
    CAIF_DeviceTensor &WQkvMut(){return _w_qkv;}
    const CAIF_DeviceTensor &GradWQ()const{return _grad_w_q;}
    CAIF_DeviceTensor &GradWQMut(){return _grad_w_q;}
    const CAIF_DeviceTensor &GradWK()const{return _grad_w_k;}
    CAIF_DeviceTensor &GradWKMut(){return _grad_w_k;}
    const CAIF_DeviceTensor &GradWV()const{return _grad_w_v;}
    CAIF_DeviceTensor &GradWVMut(){return _grad_w_v;}
    const CAIF_DeviceTensor &GradWO()const{return _grad_w_o;}
    CAIF_DeviceTensor &GradWOMut(){return _grad_w_o;}
    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    CAIF_DeviceTensor &CachedInputMut(){return _cached_input;}
    const CAIF_DeviceTensor &CachedQHeads()const{return _cached_q_heads;}
    CAIF_DeviceTensor &CachedQHeadsMut(){return _cached_q_heads;}
    const CAIF_DeviceTensor &CachedKHeads()const{return _cached_k_heads;}
    CAIF_DeviceTensor &CachedKHeadsMut(){return _cached_k_heads;}
    const CAIF_DeviceTensor &CachedVHeads()const{return _cached_v_heads;}
    CAIF_DeviceTensor &CachedVHeadsMut(){return _cached_v_heads;}
    const CAIF_DeviceTensor &CachedAttn()const{return _cached_attn;}
    CAIF_DeviceTensor &CachedAttnMut(){return _cached_attn;}
    const CAIF_DeviceTensor &CachedConcat()const{return _cached_concat;}
    CAIF_DeviceTensor &CachedConcatMut(){return _cached_concat;}
    const CAIF_DeviceTensor &CachedLogsumexp()const{return _cached_logsumexp;}
    CAIF_DeviceTensor &CachedLogsumexpMut(){return _cached_logsumexp;}
    const CAIF_DeviceTensor &CachedOutput()const{return _cached_output;}
    CAIF_DeviceTensor &CachedOutputMut(){return _cached_output;}
    const CAIF_DeviceTensor &CachedQPreNorm()const{return _cached_q_pre_norm;}
    CAIF_DeviceTensor &CachedQPreNormMut(){return _cached_q_pre_norm;}
    const CAIF_DeviceTensor &CachedKPreNorm()const{return _cached_k_pre_norm;}
    CAIF_DeviceTensor &CachedKPreNormMut(){return _cached_k_pre_norm;}
    const CAIF_DeviceTensor &CachedQRms()const{return _cached_q_rms;}
    CAIF_DeviceTensor &CachedQRmsMut(){return _cached_q_rms;}
    const CAIF_DeviceTensor &CachedKRms()const{return _cached_k_rms;}
    CAIF_DeviceTensor &CachedKRmsMut(){return _cached_k_rms;}
    uint32_t CachedBatch()const{return _cached_batch;}
    void SetCachedBatch(const uint32_t v){_cached_batch=v;}
    uint32_t CachedSeqLen()const{return _cached_seq_len;}
    void SetCachedSeqLen(const uint32_t v){_cached_seq_len=v;}
    const CAIF_DeviceTensor &KVCacheK()const{return _kv_cache_k;}
    CAIF_DeviceTensor &KVCacheKMut(){return _kv_cache_k;}
    const CAIF_DeviceTensor &KVCacheV()const{return _kv_cache_v;}
    CAIF_DeviceTensor &KVCacheVMut(){return _kv_cache_v;}
    uint32_t KVCacheMaxLen()const{return _kv_cache_max_len;}
    void SetKVCacheMaxLen(const uint32_t v){_kv_cache_max_len=v;}
    uint32_t KVCacheBatch()const{return _kv_cache_batch;}
    void SetKVCacheBatch(const uint32_t v){_kv_cache_batch=v;}
    void SetKVCacheLen(const uint32_t v){_kv_cache_len=v;}
    void SetKVCacheEnabled(const bool v){_kv_cache_enabled=v;}
    bool UseFlashAttention()const{return _use_flash_attention;}
    void SetUseFlashAttention(const bool v){_use_flash_attention=v;}

    // Tensor setters: rvalue-only rebind. Pair with the const/Mut
    // accessors above. Public LoadWQ / LoadWK / ... still wrap these
    // with shape validation; internal call sites (BuildFusedWqkv,
    // InitializeWeights, ForwardImpl caching, etc.) use these
    // raw setters since the shape is computed inline.
    void SetWQ(CAIF_DeviceTensor &&v){_w_q=std::move(v);}
    void SetWK(CAIF_DeviceTensor &&v){_w_k=std::move(v);}
    void SetWV(CAIF_DeviceTensor &&v){_w_v=std::move(v);}
    void SetWO(CAIF_DeviceTensor &&v){_w_o=std::move(v);}
    void SetWQkv(CAIF_DeviceTensor &&v){_w_qkv=std::move(v);}
    void SetGradWQ(CAIF_DeviceTensor &&v){_grad_w_q=std::move(v);}
    void SetGradWK(CAIF_DeviceTensor &&v){_grad_w_k=std::move(v);}
    void SetGradWV(CAIF_DeviceTensor &&v){_grad_w_v=std::move(v);}
    void SetGradWO(CAIF_DeviceTensor &&v){_grad_w_o=std::move(v);}
    void SetCachedInput(CAIF_DeviceTensor &&v){_cached_input=std::move(v);}
    void SetCachedQHeads(CAIF_DeviceTensor &&v){_cached_q_heads=std::move(v);}
    void SetCachedKHeads(CAIF_DeviceTensor &&v){_cached_k_heads=std::move(v);}
    void SetCachedVHeads(CAIF_DeviceTensor &&v){_cached_v_heads=std::move(v);}
    void SetCachedAttn(CAIF_DeviceTensor &&v){_cached_attn=std::move(v);}
    void SetCachedConcat(CAIF_DeviceTensor &&v){_cached_concat=std::move(v);}
    void SetCachedLogsumexp(CAIF_DeviceTensor &&v){_cached_logsumexp=std::move(v);}
    void SetCachedOutput(CAIF_DeviceTensor &&v){_cached_output=std::move(v);}
    void SetCachedQPreNorm(CAIF_DeviceTensor &&v){_cached_q_pre_norm=std::move(v);}
    void SetCachedKPreNorm(CAIF_DeviceTensor &&v){_cached_k_pre_norm=std::move(v);}
    void SetCachedQRms(CAIF_DeviceTensor &&v){_cached_q_rms=std::move(v);}
    void SetCachedKRms(CAIF_DeviceTensor &&v){_cached_k_rms=std::move(v);}
    void SetKVCacheK(CAIF_DeviceTensor &&v){_kv_cache_k=std::move(v);}
    void SetKVCacheV(CAIF_DeviceTensor &&v){_kv_cache_v=std::move(v);}

    AttentionConfig_t _config;

    MHAProjections_t _projections;
    bool _use_projections;
    bool _w_qkv_dirty;

    CAIF_DeviceTensor _w_q;
    CAIF_DeviceTensor _w_k;
    CAIF_DeviceTensor _w_v;
    CAIF_DeviceTensor _w_o;
    CAIF_DeviceTensor _w_qkv;

    CAIF_DeviceTensor _grad_w_q;
    CAIF_DeviceTensor _grad_w_k;
    CAIF_DeviceTensor _grad_w_v;
    CAIF_DeviceTensor _grad_w_o;

    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_q_heads;
    CAIF_DeviceTensor _cached_k_heads;
    CAIF_DeviceTensor _cached_v_heads;
    CAIF_DeviceTensor _cached_attn;
    CAIF_DeviceTensor _cached_concat;
    CAIF_DeviceTensor _cached_logsumexp;
    CAIF_DeviceTensor _cached_output;
    // QK-norm forward state cached for backward: pre-norm projection
    // outputs and the per-row RMS denominators returned by
    // launch_rmsnorm_forward. Populated only when q_norm_gamma /
    // k_norm_gamma are loaded and the forward was a training pass.
    CAIF_DeviceTensor _cached_q_pre_norm;
    CAIF_DeviceTensor _cached_k_pre_norm;
    CAIF_DeviceTensor _cached_q_rms;
    CAIF_DeviceTensor _cached_k_rms;
    uint32_t _cached_batch;
    uint32_t _cached_seq_len;

    CAIF_DeviceTensor _kv_cache_k;
    CAIF_DeviceTensor _kv_cache_v;
    uint32_t _kv_cache_len;
    uint32_t _kv_cache_max_len;
    uint32_t _kv_cache_batch;
    bool _kv_cache_enabled;

    bool _use_flash_attention;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceMultiHeadAttention<float,float>;
extern template class CAIF_DeviceMultiHeadAttention<float,__half>;
extern template class CAIF_DeviceMultiHeadAttention<float,__nv_bfloat16>;
extern template class CAIF_DeviceMultiHeadAttention<__half,float>;
extern template class CAIF_DeviceMultiHeadAttention<__half,__half>;
extern template class CAIF_DeviceMultiHeadAttention<__half,__nv_bfloat16>;
extern template class CAIF_DeviceMultiHeadAttention<__nv_bfloat16,float>;
extern template class CAIF_DeviceMultiHeadAttention<__nv_bfloat16,__half>;
extern template class CAIF_DeviceMultiHeadAttention<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceMultiHeadAttention<float,float>;
#endif

}//end instance namespace
