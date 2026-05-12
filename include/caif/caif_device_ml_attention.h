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
// Multi-head Latent Attention (MLA) layer (templated on <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
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
class CAIF_DeviceMLAttention:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
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
      int rope_style=0;
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
    ~CAIF_DeviceMLAttention()override=default;

    CAIF_DeviceMLAttention(CAIF_DeviceMLAttention &&other);
    CAIF_DeviceMLAttention &operator=(CAIF_DeviceMLAttention &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::MLA_e;
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

    const MLAConfig_t &Config()const{return _config;}
    const MLAProjections_t &Projections()const{return _projections;}
    bool UsesProjections()const{return _use_projections;}
    bool UsesQLoRA()const{return _use_q_lora;}
    uint32_t QKHeadDim()const{return _qk_head_dim;}
    uint32_t QProjDim()const{return _q_proj_dim;}
    uint32_t KVCompressDim()const{return _kv_compress_dim;}
    uint32_t KVDecompDim()const{return _kv_decomp_dim;}
    uint32_t OInputDim()const{return _o_input_dim;}

    const CAIF_DeviceTensor &WQ()const{return _w_q;}
    const CAIF_DeviceTensor &WQCompress()const{return _w_q_compress;}
    const CAIF_DeviceTensor &QNormGamma()const{return _q_norm_gamma;}
    const CAIF_DeviceTensor &WQDecompress()const{return _w_q_decompress;}
    const CAIF_DeviceTensor &WKVCompress()const{return _w_kv_compress;}
    const CAIF_DeviceTensor &KVNormGamma()const{return _kv_norm_gamma;}
    const CAIF_DeviceTensor &WKVDecompress()const{return _w_kv_decompress;}
    const CAIF_DeviceTensor &WO()const{return _w_o;}
    const CAIF_DeviceTensor &GradWQ()const{return _grad_w_q;}
    const CAIF_DeviceTensor &GradWQCompress()const{return _grad_w_q_compress;}
    const CAIF_DeviceTensor &GradQNormGamma()const{return _grad_q_norm_gamma;}
    const CAIF_DeviceTensor &GradWQDecompress()const{return _grad_w_q_decompress;}
    const CAIF_DeviceTensor &GradWKVCompress()const{return _grad_w_kv_compress;}
    const CAIF_DeviceTensor &GradKVNormGamma()const{return _grad_kv_norm_gamma;}
    const CAIF_DeviceTensor &GradWKVDecompress()const{return _grad_w_kv_decompress;}
    const CAIF_DeviceTensor &GradWO()const{return _grad_w_o;}

    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    const CAIF_DeviceTensor &CachedQCompressed()const{return _cached_q_compressed;}
    const CAIF_DeviceTensor &CachedQRMS()const{return _cached_q_rms;}
    const CAIF_DeviceTensor &CachedQNormed()const{return _cached_q_normed;}
    const CAIF_DeviceTensor &CachedKVCompressed()const{return _cached_kv_compressed;}
    const CAIF_DeviceTensor &CachedKVRMS()const{return _cached_kv_rms;}
    const CAIF_DeviceTensor &CachedKVNormed()const{return _cached_kv_normed;}
    const CAIF_DeviceTensor &CachedQ()const{return _cached_q;}
    const CAIF_DeviceTensor &CachedK()const{return _cached_k;}
    const CAIF_DeviceTensor &CachedV()const{return _cached_v;}
    const CAIF_DeviceTensor &CachedAttnOutput()const{return _cached_attn_output;}
    const CAIF_DeviceTensor &CachedLogsumexp()const{return _cached_logsumexp;}
    const CAIF_DeviceTensor &CachedMerged()const{return _cached_merged;}
    uint32_t CachedBatch()const{return _cached_batch;}
    uint32_t CachedSeqLen()const{return _cached_seq_len;}

    const CAIF_DeviceTensor &KVCacheCompressed()const{return _kv_cache_compressed;}
    const CAIF_DeviceTensor &KVCacheKPE()const{return _kv_cache_k_pe;}
    bool IsKVCacheEnabled()const{return _kv_cache_enabled;}
    uint32_t KVCacheLength()const{return _kv_cache_len;}
    uint32_t KVCacheMaxLen()const{return _kv_cache_max_len;}
    uint32_t KVCacheBatch()const{return _kv_cache_batch;}

    // Mutable accessors — used inside Forward/Backward for in-place ops
    // (FillZero, CopyFromHost, CAIF_Ops::Add accumulator updates). A
    // subclass can override these to intercept the mutation point.
    CAIF_DeviceTensor &MutableWQ(){return _w_q;}
    CAIF_DeviceTensor &MutableWQCompress(){return _w_q_compress;}
    CAIF_DeviceTensor &MutableQNormGamma(){return _q_norm_gamma;}
    CAIF_DeviceTensor &MutableWQDecompress(){return _w_q_decompress;}
    CAIF_DeviceTensor &MutableWKVCompress(){return _w_kv_compress;}
    CAIF_DeviceTensor &MutableKVNormGamma(){return _kv_norm_gamma;}
    CAIF_DeviceTensor &MutableWKVDecompress(){return _w_kv_decompress;}
    CAIF_DeviceTensor &MutableWO(){return _w_o;}
    CAIF_DeviceTensor &MutableGradWQ(){return _grad_w_q;}
    CAIF_DeviceTensor &MutableGradWQCompress(){return _grad_w_q_compress;}
    CAIF_DeviceTensor &MutableGradQNormGamma(){return _grad_q_norm_gamma;}
    CAIF_DeviceTensor &MutableGradWQDecompress(){return _grad_w_q_decompress;}
    CAIF_DeviceTensor &MutableGradWKVCompress(){return _grad_w_kv_compress;}
    CAIF_DeviceTensor &MutableGradKVNormGamma(){return _grad_kv_norm_gamma;}
    CAIF_DeviceTensor &MutableGradWKVDecompress(){return _grad_w_kv_decompress;}
    CAIF_DeviceTensor &MutableGradWO(){return _grad_w_o;}
    CAIF_DeviceTensor &MutableKVCacheCompressed(){return _kv_cache_compressed;}
    CAIF_DeviceTensor &MutableKVCacheKPE(){return _kv_cache_k_pe;}

    // Take (move-out) — destroys the field on this side; used in the
    // move-ctor / op= helper to move state from the source instance.
    MLAProjections_t TakeProjections(){return std::move(_projections);}
    CAIF_DeviceTensor TakeWQ(){return std::move(_w_q);}
    CAIF_DeviceTensor TakeWQCompress(){return std::move(_w_q_compress);}
    CAIF_DeviceTensor TakeQNormGamma(){return std::move(_q_norm_gamma);}
    CAIF_DeviceTensor TakeWQDecompress(){return std::move(_w_q_decompress);}
    CAIF_DeviceTensor TakeWKVCompress(){return std::move(_w_kv_compress);}
    CAIF_DeviceTensor TakeKVNormGamma(){return std::move(_kv_norm_gamma);}
    CAIF_DeviceTensor TakeWKVDecompress(){return std::move(_w_kv_decompress);}
    CAIF_DeviceTensor TakeWO(){return std::move(_w_o);}
    CAIF_DeviceTensor TakeGradWQ(){return std::move(_grad_w_q);}
    CAIF_DeviceTensor TakeGradWQCompress(){return std::move(_grad_w_q_compress);}
    CAIF_DeviceTensor TakeGradQNormGamma(){return std::move(_grad_q_norm_gamma);}
    CAIF_DeviceTensor TakeGradWQDecompress(){return std::move(_grad_w_q_decompress);}
    CAIF_DeviceTensor TakeGradWKVCompress(){return std::move(_grad_w_kv_compress);}
    CAIF_DeviceTensor TakeGradKVNormGamma(){return std::move(_grad_kv_norm_gamma);}
    CAIF_DeviceTensor TakeGradWKVDecompress(){return std::move(_grad_w_kv_decompress);}
    CAIF_DeviceTensor TakeGradWO(){return std::move(_grad_w_o);}
    CAIF_DeviceTensor TakeCachedInput(){return std::move(_cached_input);}
    CAIF_DeviceTensor TakeCachedQCompressed(){return std::move(_cached_q_compressed);}
    CAIF_DeviceTensor TakeCachedQRMS(){return std::move(_cached_q_rms);}
    CAIF_DeviceTensor TakeCachedQNormed(){return std::move(_cached_q_normed);}
    CAIF_DeviceTensor TakeCachedKVCompressed(){return std::move(_cached_kv_compressed);}
    CAIF_DeviceTensor TakeCachedKVRMS(){return std::move(_cached_kv_rms);}
    CAIF_DeviceTensor TakeCachedKVNormed(){return std::move(_cached_kv_normed);}
    CAIF_DeviceTensor TakeCachedQ(){return std::move(_cached_q);}
    CAIF_DeviceTensor TakeCachedK(){return std::move(_cached_k);}
    CAIF_DeviceTensor TakeCachedV(){return std::move(_cached_v);}
    CAIF_DeviceTensor TakeCachedAttnOutput(){return std::move(_cached_attn_output);}
    CAIF_DeviceTensor TakeCachedLogsumexp(){return std::move(_cached_logsumexp);}
    CAIF_DeviceTensor TakeCachedMerged(){return std::move(_cached_merged);}
    CAIF_DeviceTensor TakeKVCacheCompressed(){return std::move(_kv_cache_compressed);}
    CAIF_DeviceTensor TakeKVCacheKPE(){return std::move(_kv_cache_k_pe);}

    // Setters (move-in for tensors, value for primitives).
    void SetConfig(const MLAConfig_t &c){_config=c;}
    void SetProjections(MLAProjections_t p){_projections=std::move(p);}
    void SetUseProjections(bool v){_use_projections=v;}
    void SetUseQLoRA(bool v){_use_q_lora=v;}
    void SetQKHeadDim(uint32_t v){_qk_head_dim=v;}
    void SetQProjDim(uint32_t v){_q_proj_dim=v;}
    void SetKVCompressDim(uint32_t v){_kv_compress_dim=v;}
    void SetKVDecompDim(uint32_t v){_kv_decomp_dim=v;}
    void SetOInputDim(uint32_t v){_o_input_dim=v;}

    void SetWQ(CAIF_DeviceTensor &&t){_w_q=std::move(t);}
    void SetWQCompress(CAIF_DeviceTensor &&t){_w_q_compress=std::move(t);}
    void SetQNormGamma(CAIF_DeviceTensor &&t){_q_norm_gamma=std::move(t);}
    void SetWQDecompress(CAIF_DeviceTensor &&t){_w_q_decompress=std::move(t);}
    void SetWKVCompress(CAIF_DeviceTensor &&t){_w_kv_compress=std::move(t);}
    void SetKVNormGamma(CAIF_DeviceTensor &&t){_kv_norm_gamma=std::move(t);}
    void SetWKVDecompress(CAIF_DeviceTensor &&t){_w_kv_decompress=std::move(t);}
    void SetWO(CAIF_DeviceTensor &&t){_w_o=std::move(t);}
    void SetGradWQ(CAIF_DeviceTensor &&t){_grad_w_q=std::move(t);}
    void SetGradWQCompress(CAIF_DeviceTensor &&t){_grad_w_q_compress=std::move(t);}
    void SetGradQNormGamma(CAIF_DeviceTensor &&t){_grad_q_norm_gamma=std::move(t);}
    void SetGradWQDecompress(CAIF_DeviceTensor &&t){_grad_w_q_decompress=std::move(t);}
    void SetGradWKVCompress(CAIF_DeviceTensor &&t){_grad_w_kv_compress=std::move(t);}
    void SetGradKVNormGamma(CAIF_DeviceTensor &&t){_grad_kv_norm_gamma=std::move(t);}
    void SetGradWKVDecompress(CAIF_DeviceTensor &&t){_grad_w_kv_decompress=std::move(t);}
    void SetGradWO(CAIF_DeviceTensor &&t){_grad_w_o=std::move(t);}
    void SetCachedInput(CAIF_DeviceTensor &&t){_cached_input=std::move(t);}
    void SetCachedQCompressed(CAIF_DeviceTensor &&t){_cached_q_compressed=std::move(t);}
    void SetCachedQRMS(CAIF_DeviceTensor &&t){_cached_q_rms=std::move(t);}
    void SetCachedQNormed(CAIF_DeviceTensor &&t){_cached_q_normed=std::move(t);}
    void SetCachedKVCompressed(CAIF_DeviceTensor &&t){_cached_kv_compressed=std::move(t);}
    void SetCachedKVRMS(CAIF_DeviceTensor &&t){_cached_kv_rms=std::move(t);}
    void SetCachedKVNormed(CAIF_DeviceTensor &&t){_cached_kv_normed=std::move(t);}
    void SetCachedQ(CAIF_DeviceTensor &&t){_cached_q=std::move(t);}
    void SetCachedK(CAIF_DeviceTensor &&t){_cached_k=std::move(t);}
    void SetCachedV(CAIF_DeviceTensor &&t){_cached_v=std::move(t);}
    void SetCachedAttnOutput(CAIF_DeviceTensor &&t){_cached_attn_output=std::move(t);}
    void SetCachedLogsumexp(CAIF_DeviceTensor &&t){_cached_logsumexp=std::move(t);}
    void SetCachedMerged(CAIF_DeviceTensor &&t){_cached_merged=std::move(t);}
    void SetCachedBatch(uint32_t v){_cached_batch=v;}
    void SetCachedSeqLen(uint32_t v){_cached_seq_len=v;}
    void SetKVCacheCompressed(CAIF_DeviceTensor &&t){_kv_cache_compressed=std::move(t);}
    void SetKVCacheKPE(CAIF_DeviceTensor &&t){_kv_cache_k_pe=std::move(t);}
    void SetKVCacheLen(uint32_t v){_kv_cache_len=v;}
    void SetKVCacheMaxLen(uint32_t v){_kv_cache_max_len=v;}
    void SetKVCacheBatch(uint32_t v){_kv_cache_batch=v;}
    void SetKVCacheEnabled(bool v){_kv_cache_enabled=v;}

    void InitializeWeights(uint32_t seed=0)override;

    // Q LoRA path (q_lora_rank > 0): compress -> RMSNorm -> decompress.
    // Each Load* validates shape/dtype then routes through the matching
    // Set* setter so subclasses can override the mutation site.
    void LoadWQCompress(CAIF_DeviceTensor &&w);
    void LoadQNormGamma(CAIF_DeviceTensor &&gamma);
    void LoadWQDecompress(CAIF_DeviceTensor &&w);
    // Direct Q projection (q_lora_rank == 0, e.g. DeepSeek-V2-Lite):
    // a single [dim, q_proj_dim] matmul with no LoRA decomposition.
    void LoadWQ(CAIF_DeviceTensor &&w);
    void LoadWKVCompress(CAIF_DeviceTensor &&w);
    void LoadKVNormGamma(CAIF_DeviceTensor &&gamma);
    void LoadWKVDecompress(CAIF_DeviceTensor &&w);
    void LoadWO(CAIF_DeviceTensor &&w_o);

    void EnableKVCache(uint32_t batch_size,uint32_t max_seq_len);
    void DisableKVCache();
    void ResetKVCache();

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

    void XavierInit(CAIF_DeviceTensor &tensor,std::mt19937 &gen,
                    uint32_t fan_in,uint32_t fan_out);

    // Move helper — invoked from move-ctor body and op= so that
    // every field assignment routes through a SetXxx setter, keeping
    // direct member writes confined to the inline setters themselves.
    void MoveAssignFrom(CAIF_DeviceMLAttention &&other);

  private:
    MLAConfig_t _config;
    MLAProjections_t _projections;
    bool _use_projections;
    bool _use_q_lora;

    uint32_t _qk_head_dim;
    uint32_t _q_proj_dim;
    uint32_t _kv_compress_dim;
    uint32_t _kv_decomp_dim;
    uint32_t _o_input_dim;

    CAIF_DeviceTensor _w_q;
    CAIF_DeviceTensor _w_q_compress;
    CAIF_DeviceTensor _q_norm_gamma;
    CAIF_DeviceTensor _w_q_decompress;
    CAIF_DeviceTensor _w_kv_compress;
    CAIF_DeviceTensor _kv_norm_gamma;
    CAIF_DeviceTensor _w_kv_decompress;
    CAIF_DeviceTensor _w_o;

    CAIF_DeviceTensor _grad_w_q;
    CAIF_DeviceTensor _grad_w_q_compress;
    CAIF_DeviceTensor _grad_q_norm_gamma;
    CAIF_DeviceTensor _grad_w_q_decompress;
    CAIF_DeviceTensor _grad_w_kv_compress;
    CAIF_DeviceTensor _grad_kv_norm_gamma;
    CAIF_DeviceTensor _grad_w_kv_decompress;
    CAIF_DeviceTensor _grad_w_o;

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

    CAIF_DeviceTensor _kv_cache_compressed;
    CAIF_DeviceTensor _kv_cache_k_pe;
    uint32_t _kv_cache_len;
    uint32_t _kv_cache_max_len;
    uint32_t _kv_cache_batch;
    bool _kv_cache_enabled;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceMLAttention<float,float>;
extern template class CAIF_DeviceMLAttention<float,__half>;
extern template class CAIF_DeviceMLAttention<float,__nv_bfloat16>;
extern template class CAIF_DeviceMLAttention<__half,float>;
extern template class CAIF_DeviceMLAttention<__half,__half>;
extern template class CAIF_DeviceMLAttention<__half,__nv_bfloat16>;
extern template class CAIF_DeviceMLAttention<__nv_bfloat16,float>;
extern template class CAIF_DeviceMLAttention<__nv_bfloat16,__half>;
extern template class CAIF_DeviceMLAttention<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceMLAttention<float,float>;
#endif

}//end instance namespace
