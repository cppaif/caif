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
//
// Per-site dispositions:
//   - `_q_norm_gamma`, `_kv_norm_gamma`, `_grad_q_norm_gamma`,
//     `_grad_kv_norm_gamma`, `q_rms` / `kv_rms`, `_cached_q_rms` /
//     `_cached_kv_rms` are fp32 RMSNorm scale/grad/cache —
//     `DevicePtr<float>()`.
//   - `ctx.PrefixLengths()` is uint32_t — passed directly via
//     `DevicePtr<uint32_t>()`. Phase 3 changed the kernel signatures
//     from `const int *` to `const uint32_t *`, so no reinterpret_cast
//     remains at the call sites.
//------------------------------------------------------------------------------
#include "caif_device_ml_attention.h"
#include "caif_ops.h"
#include "caif_cuda_kernels_attention_support.cuh"
#include "caif_cuda_kernels_normalization.cuh"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include "caif_mla_decode_threshold.h"
#include "caif_cuda_kernels_flash_mla.cuh"
#include "caif_settings.h"
#include <cmath>
#include <random>
#include <sstream>

namespace instance
{

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
CAIF_DeviceMLAttention<ComputeT,StorageT>::CAIF_DeviceMLAttention(const CAIF_DeviceMLAttentionConfig &config,
                                             CAIF_CudaStream &stream):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                                                      _config(config),
                                                                      _use_projections(false),
                                                                      _use_q_lora(config.QLoraRank()>0),
                                                                      _cached_batch(0),
                                                                      _cached_seq_len(0),
                                                                      _kv_cache_len(0),
                                                                      _kv_cache_max_len(0),
                                                                      _kv_cache_batch(0),
                                                                      _kv_cache_enabled(false),
                                                                      _absorb_ready(false)
{
  try
  {
    const CAIF_DeviceMLAttentionConfig &cfg=Config();
    const uint32_t qk_head_dim=cfg.QkNopeHeadDim()+cfg.QkRopeHeadDim();
    SetQKHeadDim(qk_head_dim);
    SetQProjDim(cfg.NumHeads()*qk_head_dim);
    SetKVCompressDim(cfg.KvLoraRank()+cfg.QkRopeHeadDim());
    SetKVDecompDim(cfg.NumHeads()*(cfg.QkNopeHeadDim()+cfg.VHeadDim()));
    SetOInputDim(cfg.NumHeads()*cfg.VHeadDim());

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    // Q parameters: LoRA path (compress + norm + decompress) when
    // q_lora_rank>0; otherwise a single direct projection (DeepSeek-V2-Lite
    // and other configs that omit Q-LoRA).
    if(UsesQLoRA()==true)
    {
      SetWQCompress(CAIF_DeviceTensor::Uninitialized({cfg.Dim(),cfg.QLoraRank()},stream,sdt));
      // Norm gammas always live at fp32 — the launch_rmsnorm_backward
      // kernel reads/writes them via `DevicePtr<float>()`. Allocating at
      // a smaller dtype (fp16/bf16) overruns the buffer on backward
      // (writes 4 bytes into 2-byte slots) and corrupts the gamma grad.
      SetQNormGamma(CAIF_DeviceTensor::Zeros({cfg.QLoraRank()},stream,
                                              CAIF_DataType::CAIF_DataType_e::Float32));
      SetWQDecompress(CAIF_DeviceTensor::Uninitialized({cfg.QLoraRank(),QProjDim()},stream,sdt));
      SetGradWQCompress(CAIF_DeviceTensor::Zeros({cfg.Dim(),cfg.QLoraRank()},stream,sdt));
      SetGradQNormGamma(CAIF_DeviceTensor::Zeros({cfg.QLoraRank()},stream,
                                                  CAIF_DataType::CAIF_DataType_e::Float32));
      SetGradWQDecompress(CAIF_DeviceTensor::Zeros({cfg.QLoraRank(),QProjDim()},stream,sdt));
    }
    else
    {
      SetWQ(CAIF_DeviceTensor::Uninitialized({cfg.Dim(),QProjDim()},stream,sdt));
      SetGradWQ(CAIF_DeviceTensor::Zeros({cfg.Dim(),QProjDim()},stream,sdt));
    }

    // KV path (always LoRA) and output projection.
    SetWKVCompress(CAIF_DeviceTensor::Uninitialized({cfg.Dim(),KVCompressDim()},stream,sdt));
    // KV norm gamma always at fp32 — see Q-norm comment above.
    SetKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.KvLoraRank()},stream,
                                             CAIF_DataType::CAIF_DataType_e::Float32));
    SetWKVDecompress(CAIF_DeviceTensor::Uninitialized({cfg.KvLoraRank(),KVDecompDim()},stream,sdt));
    SetWO(CAIF_DeviceTensor::Uninitialized({OInputDim(),cfg.Dim()},stream,sdt));

    SetGradWKVCompress(CAIF_DeviceTensor::Zeros({cfg.Dim(),KVCompressDim()},stream,sdt));
    SetGradKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.KvLoraRank()},stream,
                                                 CAIF_DataType::CAIF_DataType_e::Float32));
    SetGradWKVDecompress(CAIF_DeviceTensor::Zeros({cfg.KvLoraRank(),KVDecompDim()},stream,sdt));
    SetGradWO(CAIF_DeviceTensor::Zeros({OInputDim(),cfg.Dim()},stream,sdt));

    // Initialize gamma to 1.0 (RMSNorm convention). Norm gammas are
    // always fp32 — Fill(float) works directly with no stage-and-cast.
    if(UsesQLoRA()==true)
    {
      MutableQNormGamma().Fill(1.0f);
    }
    MutableKVNormGamma().Fill(1.0f);

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Constructor (projections-based)
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
CAIF_DeviceMLAttention<ComputeT,StorageT>::CAIF_DeviceMLAttention(const CAIF_DeviceMLAttentionConfig &config,
                                             MLAProjections_t projections,
                                             CAIF_CudaStream &stream):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                                                      _config(config),
                                                                      _projections(std::move(projections)),
                                                                      _use_projections(true),
                                                                      _use_q_lora(config.QLoraRank()>0),
                                                                      _cached_batch(0),
                                                                      _cached_seq_len(0),
                                                                      _kv_cache_len(0),
                                                                      _kv_cache_max_len(0),
                                                                      _kv_cache_batch(0),
                                                                      _kv_cache_enabled(false),
                                                                      _absorb_ready(false)
{
  try
  {
    if(UsesQLoRA()==false)
    {
      THROW_CAIFE("MLA projections constructor requires q_lora_rank>0; the no-LoRA"
                  " Q path uses the basic-tensor constructor with LoadWQ()");
    }
    const CAIF_DeviceMLAttentionConfig &cfg=Config();
    const uint32_t qk_head_dim=cfg.QkNopeHeadDim()+cfg.QkRopeHeadDim();
    SetQKHeadDim(qk_head_dim);
    SetQProjDim(cfg.NumHeads()*qk_head_dim);
    SetKVCompressDim(cfg.KvLoraRank()+cfg.QkRopeHeadDim());
    SetKVDecompDim(cfg.NumHeads()*(cfg.QkNopeHeadDim()+cfg.VHeadDim()));
    SetOInputDim(cfg.NumHeads()*cfg.VHeadDim());

    // Norm parameters (not owned by projections).
    // Norm gammas always live at fp32 — see no-projections ctor comment.
    SetQNormGamma(CAIF_DeviceTensor::Zeros({cfg.QLoraRank()},stream,
                                            CAIF_DataType::CAIF_DataType_e::Float32));
    SetKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.KvLoraRank()},stream,
                                             CAIF_DataType::CAIF_DataType_e::Float32));
    SetGradQNormGamma(CAIF_DeviceTensor::Zeros({cfg.QLoraRank()},stream,
                                                CAIF_DataType::CAIF_DataType_e::Float32));
    SetGradKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.KvLoraRank()},stream,
                                                 CAIF_DataType::CAIF_DataType_e::Float32));

    // Initialize gamma to 1.0 (RMSNorm convention).
    MutableQNormGamma().Fill(1.0f);
    MutableKVNormGamma().Fill(1.0f);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Move semantics — init list zero-inits the primitive scalars (the language
// requires they be initialized before any method body, including
// MoveAssignFrom, runs); MoveAssignFrom then routes every state transfer
// through SetXxx setters and TakeXxx move-outs so subclasses can override
// the mutation site.
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
CAIF_DeviceMLAttention<ComputeT,StorageT>::CAIF_DeviceMLAttention(
                     CAIF_DeviceMLAttention &&other):CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                                                    _config(other.Config()),
                                                    _use_projections(false),
                                                    _use_q_lora(false),
                                                    _qk_head_dim(0),
                                                    _q_proj_dim(0),
                                                    _kv_compress_dim(0),
                                                    _kv_decomp_dim(0),
                                                    _o_input_dim(0),
                                                    _cached_batch(0),
                                                    _cached_seq_len(0),
                                                    _kv_cache_len(0),
                                                    _kv_cache_max_len(0),
                                                    _kv_cache_batch(0),
                                                    _kv_cache_enabled(false),
                                                    _absorb_ready(false)
{
  MoveAssignFrom(std::move(other));
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMLAttention<ComputeT,StorageT> &CAIF_DeviceMLAttention<ComputeT,StorageT>::operator=(CAIF_DeviceMLAttention<ComputeT,StorageT> &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    MoveAssignFrom(std::move(other));
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::MoveAssignFrom(CAIF_DeviceMLAttention &&other)
{
  SetConfig(other.Config());
  SetProjections(other.TakeProjections());
  SetUseProjections(other.UsesProjections());
  SetUseQLoRA(other.UsesQLoRA());
  SetQKHeadDim(other.QKHeadDim());
  SetQProjDim(other.QProjDim());
  SetKVCompressDim(other.KVCompressDim());
  SetKVDecompDim(other.KVDecompDim());
  SetOInputDim(other.OInputDim());

  SetWQ(other.TakeWQ());
  SetWQCompress(other.TakeWQCompress());
  SetQNormGamma(other.TakeQNormGamma());
  SetWQDecompress(other.TakeWQDecompress());
  SetWKVCompress(other.TakeWKVCompress());
  SetKVNormGamma(other.TakeKVNormGamma());
  SetWKVDecompress(other.TakeWKVDecompress());
  SetWO(other.TakeWO());

  SetGradWQ(other.TakeGradWQ());
  SetGradWQCompress(other.TakeGradWQCompress());
  SetGradQNormGamma(other.TakeGradQNormGamma());
  SetGradWQDecompress(other.TakeGradWQDecompress());
  SetGradWKVCompress(other.TakeGradWKVCompress());
  SetGradKVNormGamma(other.TakeGradKVNormGamma());
  SetGradWKVDecompress(other.TakeGradWKVDecompress());
  SetGradWO(other.TakeGradWO());

  SetCachedInput(other.TakeCachedInput());
  SetCachedQCompressed(other.TakeCachedQCompressed());
  SetCachedQRMS(other.TakeCachedQRMS());
  SetCachedQNormed(other.TakeCachedQNormed());
  SetCachedKVCompressed(other.TakeCachedKVCompressed());
  SetCachedKVRMS(other.TakeCachedKVRMS());
  SetCachedKVNormed(other.TakeCachedKVNormed());
  SetCachedQ(other.TakeCachedQ());
  SetCachedK(other.TakeCachedK());
  SetCachedV(other.TakeCachedV());
  SetCachedAttnOutput(other.TakeCachedAttnOutput());
  SetCachedLogsumexp(other.TakeCachedLogsumexp());
  SetCachedMerged(other.TakeCachedMerged());
  SetCachedBatch(other.CachedBatch());
  SetCachedSeqLen(other.CachedSeqLen());

  SetKVCacheCompressed(other.TakeKVCacheCompressed());
  SetKVCacheKPE(other.TakeKVCacheKPE());
  SetKVCacheLen(other.KVCacheLength());
  SetKVCacheMaxLen(other.KVCacheMaxLen());
  SetKVCacheBatch(other.KVCacheBatch());
  SetKVCacheEnabled(other.IsKVCacheEnabled());
}

//------------------------------------------------------------------------------
// Forward pass
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMLAttention<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)
{
  try
  {
    const auto &shape=input.Shape();
    const CAIF_DeviceMLAttentionConfig &cfg=Config();
    if(shape.size()!=3||shape[2]!=cfg.Dim())
    {
      THROW_CAIFE("MLA Forward: input must be [batch, seq_len, dim]");
    }
    if(input.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("MLA Forward: input dtype must match config.storage_dtype (caller must Cast upstream)");
    }

    const bool has_prefix=ctx.HasPrefixLengths();

    const uint32_t batch=shape[0];
    const uint32_t seq_len=shape[1];
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*cfg.NumHeads();
    const uint32_t nope=cfg.QkNopeHeadDim();
    const uint32_t rope=cfg.QkRopeHeadDim();
    const uint32_t v_dim=cfg.VHeadDim();
    const float scale=1.0f/std::sqrt(static_cast<float>(QKHeadDim()));

    // Flatten input to [bs, dim]
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,cfg.Dim()});

    //------------------------------------------------------------------
    // Q path
    //
    // q_lora_rank>0: input -> compress -> RMSNorm -> decompress -> q_full
    // q_lora_rank==0 (DeepSeek-V2-Lite): input -> direct projection -> q_full
    //------------------------------------------------------------------
    CAIF_DeviceTensor q_compressed;
    CAIF_DeviceTensor q_normed;
    CAIF_DeviceTensor q_rms;
    CAIF_DeviceTensor q_full;

    if(UsesQLoRA()==true)
    {
      if(UsesProjections()==true)
      {
        q_compressed=Projections().q_compress->Forward(flat_input,ctx);
      }
      else
      {
        q_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.QLoraRank()},Stream(),StorageDtype());
        CAIF_Ops::MatMul(flat_input,WQCompress(),q_compressed,ctx,ComputeDtype());
      }

      q_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.QLoraRank()},Stream(),StorageDtype());
      q_rms=CAIF_DeviceTensor::Uninitialized({bs},Stream());
      launch_rmsnorm_forward<StorageT>(q_compressed.template DevicePtr<StorageT>(),
                                        QNormGamma().template DevicePtr<float>(),
                                        q_normed.template DevicePtr<StorageT>(),
                                        q_rms.template DevicePtr<float>(),
                                        cfg.RmsNormEps(),
                                        static_cast<int>(bs),
                                        static_cast<int>(cfg.QLoraRank()),
                                        Stream().Handle());

      if(UsesProjections()==true)
      {
        q_full=Projections().q_decompress->Forward(q_normed,ctx);
      }
      else
      {
        q_full=CAIF_DeviceTensor::Uninitialized({bs,QProjDim()},Stream(),StorageDtype());
        CAIF_Ops::MatMul(q_normed,WQDecompress(),q_full,ctx,ComputeDtype());
      }
    }
    else
    {
      // q_lora_rank==0: a single [dim, q_proj_dim] matmul produces q_full
      // directly. No compress/norm/decompress chain.
      q_full=CAIF_DeviceTensor::Uninitialized({bs,QProjDim()},Stream(),StorageDtype());
      CAIF_Ops::MatMul(flat_input,WQ(),q_full,ctx,ComputeDtype());
    }

    // Reshape to [batch, seq, heads, qk_head_dim] and transpose to [bh, seq, qk_head_dim]
    q_full.Reshape({batch,seq_len,cfg.NumHeads(),QKHeadDim()});
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},
                                                                    Stream(),
                                                                    StorageDtype());
    launch_transpose_0213<StorageT>(q_full.template DevicePtr<StorageT>(),
                                     q_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(QKHeadDim()),
                                     Stream().Handle());

    // Slice Q into nope and rope portions: [bh, seq, nope] + [bh, seq, rope]
    CAIF_DeviceTensor q_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},Stream(),StorageDtype());
    CAIF_DeviceTensor q_rope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(q_transposed,q_nope,0);
    CAIF_Ops::SliceLastDim(q_transposed,q_rope,static_cast<uint32_t>(nope));

    // Apply RoPE to q_rope only
    launch_rope_forward<StorageT>(q_rope.template DevicePtr<StorageT>(),
                                   static_cast<int>(bh),
                                   static_cast<int>(seq_len),
                                   static_cast<int>(rope),
                                   cfg.RopeBase(),
                                   cfg.RopeStyle(),
                                   Stream().Handle());

    //------------------------------------------------------------------
    // KV path
    //------------------------------------------------------------------
    CAIF_DeviceTensor kv_out;
    if(UsesProjections()==true)
    {
      kv_out=Projections().kv_compress->Forward(flat_input,ctx);
    }
    else
    {
      kv_out=CAIF_DeviceTensor::Uninitialized({bs,KVCompressDim()},Stream(),StorageDtype());
      CAIF_Ops::MatMul(flat_input,WKVCompress(),kv_out,ctx,ComputeDtype());
    }

    // Slice into compressed_kv [bs, kv_lora_rank] and k_pe_flat [bs, rope]
    CAIF_DeviceTensor kv_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.KvLoraRank()},
                                                                     Stream(),
                                                                     StorageDtype());
    CAIF_DeviceTensor k_pe_flat=CAIF_DeviceTensor::Uninitialized({bs,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(kv_out,kv_compressed,0);
    CAIF_Ops::SliceLastDim(kv_out,k_pe_flat,cfg.KvLoraRank());

    // KV RMSNorm on compressed_kv
    CAIF_DeviceTensor kv_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.KvLoraRank()},Stream(),StorageDtype());
    CAIF_DeviceTensor kv_rms=CAIF_DeviceTensor::Uninitialized({bs},Stream());
    launch_rmsnorm_forward<StorageT>(kv_compressed.template DevicePtr<StorageT>(),
                                      KVNormGamma().template DevicePtr<float>(),
                                      kv_normed.template DevicePtr<StorageT>(),
                                      kv_rms.template DevicePtr<float>(),
                                      cfg.RmsNormEps(),
                                      static_cast<int>(bs),
                                      static_cast<int>(cfg.KvLoraRank()),
                                      Stream().Handle());

    // KV decompress: [bs, kv_lora_rank] -> [bs, kv_decomp_dim]
    CAIF_DeviceTensor kv_full;
    if(UsesProjections()==true)
    {
      kv_full=Projections().kv_decompress->Forward(kv_normed,ctx);
    }
    else
    {
      kv_full=CAIF_DeviceTensor::Uninitialized({bs,KVDecompDim()},Stream(),StorageDtype());
      CAIF_Ops::MatMul(kv_normed,WKVDecompress(),kv_full,ctx,ComputeDtype());
    }

    // Reshape to [batch, seq, heads, nope+v_dim] and transpose
    const uint32_t kv_per_head=nope+v_dim;
    kv_full.Reshape({batch,seq_len,cfg.NumHeads(),kv_per_head});
    CAIF_DeviceTensor kv_transposed=CAIF_DeviceTensor::Uninitialized({bh,seq_len,kv_per_head},
                                                                     Stream(),
                                                                     StorageDtype());
    launch_transpose_0213<StorageT>(kv_full.template DevicePtr<StorageT>(),
                                     kv_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(kv_per_head),
                                     Stream().Handle());

    // Split KV into k_nope [bh, seq, nope] and v [bh, seq, v_dim]
    CAIF_DeviceTensor k_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},Stream(),StorageDtype());
    CAIF_DeviceTensor v=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(kv_transposed,k_nope,0);
    CAIF_Ops::SliceLastDim(kv_transposed,v,nope);

    // Broadcast k_pe from [batch, seq, rope] to [bh, seq, rope]
    k_pe_flat.Reshape({batch,seq_len,rope});
    CAIF_DeviceTensor k_pe=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},Stream(),StorageDtype());
    launch_gqa_repeat_kv<StorageT>(k_pe_flat.template DevicePtr<StorageT>(),
                                    k_pe.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    1,
                                    static_cast<int>(cfg.NumHeads()),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(rope),
                                    Stream().Handle());

    // Apply RoPE to k_pe
    launch_rope_forward<StorageT>(k_pe.template DevicePtr<StorageT>(),
                                   static_cast<int>(bh),
                                   static_cast<int>(seq_len),
                                   static_cast<int>(rope),
                                   cfg.RopeBase(),
                                   cfg.RopeStyle(),
                                   Stream().Handle());

    //------------------------------------------------------------------
    // Assemble Q, K and run attention
    //------------------------------------------------------------------
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(q_nope,q_rope,q);

    CAIF_DeviceTensor k=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(k_nope,k_pe,k);

    CAIF_DeviceTensor attn_output=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},Stream(),StorageDtype());

    // Fused tensor-core flash prefill (O(seq) memory) for inference without a
    // prefix-LM mask, when the device supports the (QKHeadDim, v_dim) config and
    // the path is enabled. Otherwise fall through to the explicit O(seq^2) path.
    // Training keeps the explicit path so the backward caches are produced.
    bool used_flash=false;
    if(ctx.Training()==false && has_prefix==false && CAIF_Settings::FlashMlaPrefill()==true)
    {
      int flash_device=0;
      cudaGetDevice(&flash_device);
      if(mla_flash_prefill_available(static_cast<int>(QKHeadDim()),
                                     static_cast<int>(v_dim),
                                     flash_device)==true)
      {
        int causal_flag=0;
        if(cfg.Causal()==true)
        {
          causal_flag=1;
        }
        const bool launched=launch_flash_attention_forward_mla<StorageT>(
                            q.template DevicePtr<StorageT>(),
                            k.template DevicePtr<StorageT>(),
                            v.template DevicePtr<StorageT>(),
                            attn_output.template DevicePtr<StorageT>(),
                            static_cast<int>(bh),
                            static_cast<int>(seq_len),
                            static_cast<int>(seq_len),
                            static_cast<int>(QKHeadDim()),
                            static_cast<int>(v_dim),
                            scale,
                            causal_flag,
                            0,
                            Stream().Handle());
        if(launched==false)
        {
          THROW_CAIFE("MLA flash prefill launch failed after availability check");
        }
        used_flash=true;
      }
    }

    if(used_flash==false)
    {
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},Stream(),StorageDtype());
      CAIF_Ops::BatchedMatMulTransposeB(q,
                                        k,
                                        scores,
                                        static_cast<int>(seq_len),
                                        static_cast<int>(QKHeadDim()),
                                        static_cast<int>(seq_len),
                                        static_cast<int>(bh),
                                        ctx);
      CAIF_Ops::Scale(scores,scale);
      if(has_prefix==true)
      {
        launch_prefix_mask_fill<StorageT>(scores.template DevicePtr<StorageT>(),
                                          ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                          static_cast<int>(batch),
                                          static_cast<int>(cfg.NumHeads()),
                                          static_cast<int>(seq_len),
                                          Stream().Handle());
      }
      else if(cfg.Causal()==true)
      {
        launch_causal_mask_fill<StorageT>(scores.template DevicePtr<StorageT>(),
                                          static_cast<int>(bh),
                                          static_cast<int>(seq_len),
                                          Stream().Handle());
      }
      // Templated softmax dispatch on StorageT (Phase 9 sweep extending the
      // 5.1 MHA fix to MLA; MLA's constructor rejects Int8/Int4 storage so
      // reaching this site with a non-fp32/fp16/bf16 StorageT is impossible).
      CAIF_DeviceTensor attn_weights=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},
                                                                      Stream(),
                                                                      scores.Dtype());
      launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                         attn_weights.template DevicePtr<StorageT>(),
                                         static_cast<int>(bh*seq_len),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
      CAIF_Ops::BatchedMatMul(attn_weights,
                              v,
                              attn_output,
                              static_cast<int>(seq_len),
                              static_cast<int>(seq_len),
                              static_cast<int>(v_dim),
                              static_cast<int>(bh),
                              ctx);
    }

    //------------------------------------------------------------------
    // Output projection
    //------------------------------------------------------------------
    const std::vector<uint32_t> merged_shape={batch,seq_len,cfg.NumHeads(),v_dim};
    CAIF_DeviceTensor attn_merged=CAIF_DeviceTensor::Uninitialized(merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(attn_output.template DevicePtr<StorageT>(),
                                     attn_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(v_dim),
                                     Stream().Handle());
    attn_merged.Reshape({bs,OInputDim()});

    CAIF_DeviceTensor output;
    if(UsesProjections()==true)
    {
      output=Projections().o_proj->Forward(attn_merged,ctx);
    }
    else
    {
      output=CAIF_DeviceTensor::Uninitialized({bs,cfg.Dim()},Stream(),StorageDtype());
      CAIF_Ops::MatMul(attn_merged,WO(),output,ctx,ComputeDtype());
    }

    //------------------------------------------------------------------
    // Cache for backward
    //------------------------------------------------------------------
    if(ctx.Training()==true)
    {
      SetCachedInput(std::move(flat_input));
      SetCachedQCompressed(std::move(q_compressed));
      SetCachedQRMS(std::move(q_rms));
      SetCachedQNormed(std::move(q_normed));
      SetCachedKVCompressed(std::move(kv_compressed));
      SetCachedKVRMS(std::move(kv_rms));
      SetCachedKVNormed(std::move(kv_normed));
      SetCachedQ(std::move(q));
      SetCachedK(std::move(k));
      SetCachedV(std::move(v));
      SetCachedAttnOutput(std::move(attn_output));
      SetCachedMerged(std::move(attn_merged));
      SetCachedBatch(batch);
      SetCachedSeqLen(seq_len);
    }

    output.Reshape({batch,seq_len,cfg.Dim()});
    return output;
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Backward pass
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMLAttention<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)
{
  try
  {
    const bool has_prefix=ctx.HasPrefixLengths();
    const CAIF_DeviceMLAttentionConfig &cfg=Config();

    const uint32_t batch=CachedBatch();
    const uint32_t seq_len=CachedSeqLen();
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*cfg.NumHeads();
    const uint32_t nope=cfg.QkNopeHeadDim();
    const uint32_t rope=cfg.QkRopeHeadDim();
    const uint32_t v_dim=cfg.VHeadDim();
    const float scale=1.0f/std::sqrt(static_cast<float>(QKHeadDim()));

    // Flatten grad_output to [bs, dim] — zero-copy view
    const std::vector<uint32_t> flat_shape={bs,cfg.Dim()};
    CAIF_DeviceTensor grad_flat=CAIF_DeviceTensor::WrapView(const_cast<void *>(grad_output.DeviceDataRaw()),
                                                            flat_shape,
                                                            Stream(),
                                                            grad_output.Dtype());

    //------------------------------------------------------------------
    // Output projection backward
    //------------------------------------------------------------------
    CAIF_DeviceTensor grad_merged;
    if(UsesProjections()==true)
    {
      grad_merged=Projections().o_proj->Backward(grad_flat,ctx);
    }
    else
    {
      grad_merged=CAIF_DeviceTensor::Uninitialized({bs,OInputDim()},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_flat,WO(),grad_merged,ctx,ComputeDtype());

      const std::vector<uint32_t> w_o_shape={OInputDim(),cfg.Dim()};
      CAIF_DeviceTensor grad_w_o_delta=CAIF_DeviceTensor::Uninitialized(w_o_shape,Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedMerged(),grad_flat,grad_w_o_delta,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWO(),grad_w_o_delta,MutableGradWO());
    }

    //------------------------------------------------------------------
    // Reverse merge heads: [bs, o_input_dim] -> [bh, seq, v_dim]
    //------------------------------------------------------------------
    grad_merged.Reshape({batch,seq_len,cfg.NumHeads(),v_dim});
    CAIF_DeviceTensor grad_attn_output=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},
                                                                        Stream(),
                                                                        StorageDtype());
    launch_transpose_0213<StorageT>(grad_merged.template DevicePtr<StorageT>(),
                                     grad_attn_output.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(v_dim),
                                     Stream().Handle());

    //------------------------------------------------------------------
    // Standard attention backward: recompute attn, then matmul gradients
    //------------------------------------------------------------------
    CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},Stream(),StorageDtype());
    CAIF_Ops::BatchedMatMulTransposeB(CachedQ(),
                                       CachedK(),
                                       scores,
                                       static_cast<int>(seq_len),
                                       static_cast<int>(QKHeadDim()),
                                       static_cast<int>(seq_len),
                                       static_cast<int>(bh),
                                       ctx);
    CAIF_Ops::Scale(scores,scale);
    if(has_prefix==true)
    {
      launch_prefix_mask_fill<StorageT>(scores.template DevicePtr<StorageT>(),
                                         ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                         static_cast<int>(batch),
                                         static_cast<int>(cfg.NumHeads()),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    else if(cfg.Causal()==true)
    {
      launch_causal_mask_fill<StorageT>(scores.template DevicePtr<StorageT>(),
                                         static_cast<int>(bh),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},Stream(),scores.Dtype());
    launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                       attn.template DevicePtr<StorageT>(),
                                       static_cast<int>(bh*seq_len),
                                       static_cast<int>(seq_len),
                                       Stream().Handle());

    CAIF_DeviceTensor grad_attn=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},Stream(),scores.Dtype());
    CAIF_Ops::BatchedMatMulTransposeB(grad_attn_output,
                                       CachedV(),
                                       grad_attn,
                                       static_cast<int>(seq_len),
                                       static_cast<int>(v_dim),
                                       static_cast<int>(seq_len),
                                       static_cast<int>(bh),
                                       ctx);

    CAIF_DeviceTensor grad_v=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},Stream(),StorageDtype());
    CAIF_Ops::BatchedMatMulTransposeA(attn,
                                       grad_attn_output,
                                       grad_v,
                                       static_cast<int>(seq_len),
                                       static_cast<int>(seq_len),
                                       static_cast<int>(v_dim),
                                       static_cast<int>(bh),
                                       ctx);

    CAIF_DeviceTensor grad_scores=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},Stream(),attn.Dtype());
    launch_attention_softmax_backward<StorageT>(grad_attn.template DevicePtr<StorageT>(),
                                                attn.template DevicePtr<StorageT>(),
                                                grad_scores.template DevicePtr<StorageT>(),
                                                static_cast<int>(bh*seq_len),
                                                static_cast<int>(seq_len),
                                                Stream().Handle());
    if(has_prefix==true)
    {
      launch_prefix_mask_grad<StorageT>(grad_scores.template DevicePtr<StorageT>(),
                                         ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                         static_cast<int>(batch),
                                         static_cast<int>(cfg.NumHeads()),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    else if(cfg.Causal()==true)
    {
      launch_causal_mask_grad<StorageT>(grad_scores.template DevicePtr<StorageT>(),
                                         static_cast<int>(bh),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    CAIF_Ops::Scale(grad_scores,scale);

    CAIF_DeviceTensor grad_q=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::BatchedMatMul(grad_scores,
                             CachedK(),
                             grad_q,
                             static_cast<int>(seq_len),
                             static_cast<int>(seq_len),
                             static_cast<int>(QKHeadDim()),
                             static_cast<int>(bh),
                             ctx);

    CAIF_DeviceTensor grad_k=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::BatchedMatMulTransposeA(grad_scores,
                                       CachedQ(),
                                       grad_k,
                                       static_cast<int>(seq_len),
                                       static_cast<int>(seq_len),
                                       static_cast<int>(QKHeadDim()),
                                       static_cast<int>(bh),
                                       ctx);

    //------------------------------------------------------------------
    // Q path backward
    //------------------------------------------------------------------
    CAIF_DeviceTensor grad_q_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},Stream(),StorageDtype());
    CAIF_DeviceTensor grad_q_rope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(grad_q,grad_q_nope,0);
    CAIF_Ops::SliceLastDim(grad_q,grad_q_rope,static_cast<uint32_t>(nope));

    launch_rope_backward<StorageT>(grad_q_rope.template DevicePtr<StorageT>(),
                                    static_cast<int>(bh),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(rope),
                                    cfg.RopeBase(),
                                    cfg.RopeStyle(),
                                    Stream().Handle());

    CAIF_DeviceTensor grad_q_full=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},
                                                                   Stream(),
                                                                   StorageDtype());
    CAIF_Ops::ConcatLastDim(grad_q_nope,grad_q_rope,grad_q_full);

    const std::vector<uint32_t> q_merged_shape={batch,seq_len,cfg.NumHeads(),QKHeadDim()};
    CAIF_DeviceTensor grad_q_merged=CAIF_DeviceTensor::Uninitialized(q_merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(grad_q_full.template DevicePtr<StorageT>(),
                                     grad_q_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(QKHeadDim()),
                                     Stream().Handle());
    grad_q_merged.Reshape({bs,QProjDim()});

    CAIF_DeviceTensor grad_input_q;
    if(UsesQLoRA()==true)
    {
      // grad_q_merged -> grad_q_normed -> grad_q_compressed -> grad_input_q
      CAIF_DeviceTensor grad_q_normed;
      if(UsesProjections()==true)
      {
        grad_q_normed=Projections().q_decompress->Backward(grad_q_merged,ctx);
      }
      else
      {
        grad_q_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.QLoraRank()},Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeB(grad_q_merged,WQDecompress(),grad_q_normed,ctx,ComputeDtype());

        const std::vector<uint32_t> w_qd_shape={cfg.QLoraRank(),QProjDim()};
        CAIF_DeviceTensor grad_w_qd=CAIF_DeviceTensor::Uninitialized(w_qd_shape,Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeA(CachedQNormed(),grad_q_merged,grad_w_qd,ctx,ComputeDtype());
        CAIF_Ops::Add(GradWQDecompress(),grad_w_qd,MutableGradWQDecompress());
      }

      CAIF_DeviceTensor grad_q_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.QLoraRank()},
                                                                           Stream(),
                                                                           StorageDtype());
      launch_rmsnorm_backward<StorageT>(grad_q_normed.template DevicePtr<StorageT>(),
                                         CachedQCompressed().template DevicePtr<StorageT>(),
                                         QNormGamma().template DevicePtr<float>(),
                                         CachedQRMS().template DevicePtr<float>(),
                                         grad_q_compressed.template DevicePtr<StorageT>(),
                                         MutableGradQNormGamma().template DevicePtr<float>(),
                                         cfg.RmsNormEps(),
                                         static_cast<int>(bs),
                                         static_cast<int>(cfg.QLoraRank()),
                                         Stream().Handle());

      if(UsesProjections()==true)
      {
        grad_input_q=Projections().q_compress->Backward(grad_q_compressed,ctx);
      }
      else
      {
        grad_input_q=CAIF_DeviceTensor::Uninitialized({bs,cfg.Dim()},Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeB(grad_q_compressed,WQCompress(),grad_input_q,ctx,ComputeDtype());

        const std::vector<uint32_t> w_qc_shape={cfg.Dim(),cfg.QLoraRank()};
        CAIF_DeviceTensor grad_w_qc=CAIF_DeviceTensor::Uninitialized(w_qc_shape,Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_q_compressed,grad_w_qc,ctx,ComputeDtype());
        CAIF_Ops::Add(GradWQCompress(),grad_w_qc,MutableGradWQCompress());
      }
    }
    else
    {
      // q_lora_rank==0: direct grad_q_merged -> grad_input_q via _w_q.
      grad_input_q=CAIF_DeviceTensor::Uninitialized({bs,cfg.Dim()},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_q_merged,WQ(),grad_input_q,ctx,ComputeDtype());

      const std::vector<uint32_t> w_q_shape={cfg.Dim(),QProjDim()};
      CAIF_DeviceTensor grad_w_q_delta=CAIF_DeviceTensor::Uninitialized(w_q_shape,Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedInput(),grad_q_merged,grad_w_q_delta,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWQ(),grad_w_q_delta,MutableGradWQ());
    }

    //------------------------------------------------------------------
    // KV path backward
    //------------------------------------------------------------------
    CAIF_DeviceTensor grad_k_nope=CAIF_DeviceTensor::Uninitialized({bh,seq_len,nope},Stream(),StorageDtype());
    CAIF_DeviceTensor grad_k_pe=CAIF_DeviceTensor::Uninitialized({bh,seq_len,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(grad_k,grad_k_nope,0);
    CAIF_Ops::SliceLastDim(grad_k,grad_k_pe,static_cast<uint32_t>(nope));

    launch_rope_backward<StorageT>(grad_k_pe.template DevicePtr<StorageT>(),
                                    static_cast<int>(bh),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(rope),
                                    cfg.RopeBase(),
                                    cfg.RopeStyle(),
                                    Stream().Handle());

    CAIF_DeviceTensor grad_k_pe_flat=CAIF_DeviceTensor::Zeros({batch,seq_len,rope},Stream(),StorageDtype());
    launch_gqa_reduce_kv<StorageT>(grad_k_pe.template DevicePtr<StorageT>(),
                                    grad_k_pe_flat.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    1,
                                    static_cast<int>(cfg.NumHeads()),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(rope),
                                    Stream().Handle());
    grad_k_pe_flat.Reshape({bs,rope});

    const uint32_t kv_per_head=nope+v_dim;
    CAIF_DeviceTensor grad_kv_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,kv_per_head},
                                                                     Stream(),
                                                                     StorageDtype());
    CAIF_Ops::ConcatLastDim(grad_k_nope,grad_v,grad_kv_heads);

    const std::vector<uint32_t> kv_merged_shape={batch,seq_len,cfg.NumHeads(),kv_per_head};
    CAIF_DeviceTensor grad_kv_merged=CAIF_DeviceTensor::Uninitialized(kv_merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(grad_kv_heads.template DevicePtr<StorageT>(),
                                     grad_kv_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(kv_per_head),
                                     Stream().Handle());
    grad_kv_merged.Reshape({bs,KVDecompDim()});

    CAIF_DeviceTensor grad_kv_normed;
    if(UsesProjections()==true)
    {
      grad_kv_normed=Projections().kv_decompress->Backward(grad_kv_merged,ctx);
    }
    else
    {
      grad_kv_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.KvLoraRank()},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_kv_merged,WKVDecompress(),grad_kv_normed,ctx,ComputeDtype());

      const std::vector<uint32_t> w_kvd_shape={cfg.KvLoraRank(),KVDecompDim()};
      CAIF_DeviceTensor grad_w_kvd=CAIF_DeviceTensor::Uninitialized(w_kvd_shape,Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedKVNormed(),grad_kv_merged,grad_w_kvd,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWKVDecompress(),grad_w_kvd,MutableGradWKVDecompress());
    }

    CAIF_DeviceTensor grad_kv_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.KvLoraRank()},
                                                                          Stream(),
                                                                          StorageDtype());
    launch_rmsnorm_backward<StorageT>(grad_kv_normed.template DevicePtr<StorageT>(),
                                       CachedKVCompressed().template DevicePtr<StorageT>(),
                                       KVNormGamma().template DevicePtr<float>(),
                                       CachedKVRMS().template DevicePtr<float>(),
                                       grad_kv_compressed.template DevicePtr<StorageT>(),
                                       MutableGradKVNormGamma().template DevicePtr<float>(),
                                       cfg.RmsNormEps(),
                                       static_cast<int>(bs),
                                       static_cast<int>(cfg.KvLoraRank()),
                                       Stream().Handle());

    CAIF_DeviceTensor grad_kv_out=CAIF_DeviceTensor::Uninitialized({bs,KVCompressDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(grad_kv_compressed,grad_k_pe_flat,grad_kv_out);

    CAIF_DeviceTensor grad_input_kv;
    if(UsesProjections()==true)
    {
      grad_input_kv=Projections().kv_compress->Backward(grad_kv_out,ctx);
    }
    else
    {
      grad_input_kv=CAIF_DeviceTensor::Uninitialized({bs,cfg.Dim()},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_kv_out,WKVCompress(),grad_input_kv,ctx,ComputeDtype());

      const std::vector<uint32_t> w_kvc_shape={cfg.Dim(),KVCompressDim()};
      CAIF_DeviceTensor grad_w_kvc=CAIF_DeviceTensor::Uninitialized(w_kvc_shape,Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedInput(),grad_kv_out,grad_w_kvc,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWKVCompress(),grad_w_kvc,MutableGradWKVCompress());
    }

    //------------------------------------------------------------------
    // Combine input gradients
    //------------------------------------------------------------------
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({bs,cfg.Dim()},Stream(),StorageDtype());
    CAIF_Ops::Add(grad_input_q,grad_input_kv,grad_input);
    grad_input.Reshape({batch,seq_len,cfg.Dim()});

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Parameter management
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::ZeroGradients()
{
  if(UsesProjections()==true)
  {
    Projections().q_compress->ZeroGradients();
    Projections().q_decompress->ZeroGradients();
    Projections().kv_compress->ZeroGradients();
    Projections().kv_decompress->ZeroGradients();
    Projections().o_proj->ZeroGradients();
    MutableGradQNormGamma().FillZero();
    MutableGradKVNormGamma().FillZero();
    return;
  }
  if(UsesQLoRA()==true)
  {
    MutableGradWQCompress().FillZero();
    MutableGradQNormGamma().FillZero();
    MutableGradWQDecompress().FillZero();
  }
  else
  {
    MutableGradWQ().FillZero();
  }
  MutableGradWKVCompress().FillZero();
  MutableGradKVNormGamma().FillZero();
  MutableGradWKVDecompress().FillZero();
  MutableGradWO().FillZero();
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMLAttention<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(UsesProjections()==true)
  {
    return Projections().q_compress->ParameterTensorCount()+
           Projections().q_decompress->ParameterTensorCount()+
           Projections().kv_compress->ParameterTensorCount()+
           Projections().kv_decompress->ParameterTensorCount()+
           Projections().o_proj->ParameterTensorCount()+
           2;
  }
  // q_lora_rank>0: WQC, QNorm, WQD, WKVC, KVNorm, WKVD, WO = 7
  // q_lora_rank==0: WQ, WKVC, KVNorm, WKVD, WO = 5
  return UsesQLoRA()==true?7:5;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMLAttention<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  if(UsesProjections()==true)
  {
    size_t offset=0;
    CAIF_DeviceLayer *projs[]={Projections().q_compress.get(),
                               Projections().q_decompress.get(),
                               Projections().kv_compress.get(),
                               Projections().kv_decompress.get(),
                               Projections().o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return MutableQNormGamma();}
    if(index==offset+1){return MutableKVNormGamma();}
    THROW_CAIFE("MLA ParameterTensor: index out of range");
  }
  if(UsesQLoRA()==true)
  {
    if(index==0){return MutableWQCompress();}
    if(index==1){return MutableQNormGamma();}
    if(index==2){return MutableWQDecompress();}
    if(index==3){return MutableWKVCompress();}
    if(index==4){return MutableKVNormGamma();}
    if(index==5){return MutableWKVDecompress();}
    if(index==6){return MutableWO();}
  }
  else
  {
    if(index==0){return MutableWQ();}
    if(index==1){return MutableWKVCompress();}
    if(index==2){return MutableKVNormGamma();}
    if(index==3){return MutableWKVDecompress();}
    if(index==4){return MutableWO();}
  }
  THROW_CAIFE("MLA ParameterTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMLAttention<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  if(UsesProjections()==true)
  {
    size_t offset=0;
    const CAIF_DeviceLayer *projs[]={Projections().q_compress.get(),
                                     Projections().q_decompress.get(),
                                     Projections().kv_compress.get(),
                                     Projections().kv_decompress.get(),
                                     Projections().o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return QNormGamma();}
    if(index==offset+1){return KVNormGamma();}
    THROW_CAIFE("MLA ParameterTensor: index out of range");
  }
  if(UsesQLoRA()==true)
  {
    if(index==0){return WQCompress();}
    if(index==1){return QNormGamma();}
    if(index==2){return WQDecompress();}
    if(index==3){return WKVCompress();}
    if(index==4){return KVNormGamma();}
    if(index==5){return WKVDecompress();}
    if(index==6){return WO();}
  }
  else
  {
    if(index==0){return WQ();}
    if(index==1){return WKVCompress();}
    if(index==2){return KVNormGamma();}
    if(index==3){return WKVDecompress();}
    if(index==4){return WO();}
  }
  THROW_CAIFE("MLA ParameterTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMLAttention<ComputeT,StorageT>::GradientTensor(size_t index)
{
  if(UsesProjections()==true)
  {
    size_t offset=0;
    CAIF_DeviceLayer *projs[]={Projections().q_compress.get(),
                               Projections().q_decompress.get(),
                               Projections().kv_compress.get(),
                               Projections().kv_decompress.get(),
                               Projections().o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->GradientTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return MutableGradQNormGamma();}
    if(index==offset+1){return MutableGradKVNormGamma();}
    THROW_CAIFE("MLA GradientTensor: index out of range");
  }
  if(UsesQLoRA()==true)
  {
    if(index==0){return MutableGradWQCompress();}
    if(index==1){return MutableGradQNormGamma();}
    if(index==2){return MutableGradWQDecompress();}
    if(index==3){return MutableGradWKVCompress();}
    if(index==4){return MutableGradKVNormGamma();}
    if(index==5){return MutableGradWKVDecompress();}
    if(index==6){return MutableGradWO();}
  }
  else
  {
    if(index==0){return MutableGradWQ();}
    if(index==1){return MutableGradWKVCompress();}
    if(index==2){return MutableGradKVNormGamma();}
    if(index==3){return MutableGradWKVDecompress();}
    if(index==4){return MutableGradWO();}
  }
  THROW_CAIFE("MLA GradientTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMLAttention<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  if(UsesProjections()==true)
  {
    size_t offset=0;
    const CAIF_DeviceLayer *projs[]={Projections().q_compress.get(),
                                     Projections().q_decompress.get(),
                                     Projections().kv_compress.get(),
                                     Projections().kv_decompress.get(),
                                     Projections().o_proj.get()};
    for(size_t p=0;p<5;++p)
    {
      const size_t count=projs[p]->ParameterTensorCount();
      if(index<offset+count)
      {
        return projs[p]->GradientTensor(index-offset);
      }
      offset+=count;
    }
    if(index==offset){return GradQNormGamma();}
    if(index==offset+1){return GradKVNormGamma();}
    THROW_CAIFE("MLA GradientTensor: index out of range");
  }
  if(UsesQLoRA()==true)
  {
    if(index==0){return GradWQCompress();}
    if(index==1){return GradQNormGamma();}
    if(index==2){return GradWQDecompress();}
    if(index==3){return GradWKVCompress();}
    if(index==4){return GradKVNormGamma();}
    if(index==5){return GradWKVDecompress();}
    if(index==6){return GradWO();}
  }
  else
  {
    if(index==0){return GradWQ();}
    if(index==1){return GradWKVCompress();}
    if(index==2){return GradKVNormGamma();}
    if(index==3){return GradWKVDecompress();}
    if(index==4){return GradWO();}
  }
  THROW_CAIFE("MLA GradientTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMLAttention<ComputeT,StorageT>::TotalParameterCount()const
{
  if(UsesProjections()==true)
  {
    return Projections().q_compress->TotalParameterCount()+
           Projections().q_decompress->TotalParameterCount()+
           Projections().kv_compress->TotalParameterCount()+
           Projections().kv_decompress->TotalParameterCount()+
           Projections().o_proj->TotalParameterCount()+
           Config().QLoraRank()+
           Config().KvLoraRank();
  }
  size_t total=0;
  for(size_t i=0;i<ParameterTensorCount();++i)
  {
    total+=ParameterTensor(i).TotalElements();
  }
  return total;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceMLAttention<ComputeT,StorageT>::Description()const
{
  std::ostringstream ss;
  ss<<g_serial_tag_mla
    <<g_serial_open_paren
    <<g_serial_kv_dim
    <<Config().Dim()
    <<g_serial_comma
    <<g_serial_kv_heads
    <<Config().NumHeads()
    <<g_serial_comma
    <<g_serial_kv_q_lora
    <<Config().QLoraRank()
    <<g_serial_comma
    <<g_serial_kv_kv_lora
    <<Config().KvLoraRank()
    <<g_serial_comma
    <<g_serial_kv_nope
    <<Config().QkNopeHeadDim()
    <<g_serial_comma
    <<g_serial_kv_rope
    <<Config().QkRopeHeadDim()
    <<g_serial_comma
    <<g_serial_kv_v
    <<Config().VHeadDim()
    <<g_serial_comma
    <<g_serial_kv_params
    <<TotalParameterCount();
  if(UsesProjections()==true)
  {
    ss<<g_serial_flag_projections;
  }
  if(UsesQLoRA()==false)
  {
    ss<<g_serial_flag_direct_q;
  }
  ss<<g_serial_close_paren;
  return ss.str();
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceMLAttention<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  // CAIF-neutral role names. External naming conventions (HuggingFace
  // safetensors, third-party checkpoint formats) are the caller's
  // responsibility — naming profiles, weight loaders, and safetensors
  // mappers translate these role tags to whatever their upstream system
  // expects.
  const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
  if(UsesProjections()==true)
  {
    std::vector<std::string> names;
    std::vector<std::string> sub;

    sub=Projections().q_compress->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWQCompress_e)+".");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().q_decompress->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWQDecompress_e)+".");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().kv_compress->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWKVCompress_e)+".");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().kv_decompress->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWKVDecompress_e)+".");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().o_proj->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWO_e)+".");
    names.insert(names.end(),sub.begin(),sub.end());

    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAQNormGamma_e));
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAKVNormGamma_e));
    return names;
  }
  std::vector<std::string> names;
  if(UsesQLoRA()==true)
  {
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWQCompress_e));
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAQNormGamma_e));
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWQDecompress_e));
  }
  else
  {
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::AttnWQ_e));
  }
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWKVCompress_e));
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAKVNormGamma_e));
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWKVDecompress_e));
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MLAWO_e));
  return names;
}

//------------------------------------------------------------------------------
// Weight initialization
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    std::mt19937 rng(seed);
    const CAIF_DeviceMLAttentionConfig &cfg=Config();

    const float limit_q_direct=std::sqrt(6.0f/static_cast<float>(cfg.Dim()+QProjDim()));
    const float limit_qc=std::sqrt(6.0f/static_cast<float>(cfg.Dim()+cfg.QLoraRank()));
    const float limit_qd=std::sqrt(6.0f/static_cast<float>(cfg.QLoraRank()+QProjDim()));
    const float limit_kvc=std::sqrt(6.0f/static_cast<float>(cfg.Dim()+KVCompressDim()));
    const float limit_kvd=std::sqrt(6.0f/static_cast<float>(cfg.KvLoraRank()+KVDecompDim()));
    const float limit_o=std::sqrt(6.0f/static_cast<float>(OInputDim()+cfg.Dim()));

    // Helper: fills a tensor with Xavier-uniform random values, casting to
    // the layer's storage dtype for non-fp32 cells.
    const CAIF_DataType::CAIF_DataType_e storage_dtype=StorageDtype();
    auto fill_uniform=[&rng,storage_dtype](CAIF_DeviceTensor &tensor,float limit)
    {
      std::uniform_real_distribution<float> dist(-limit,limit);
      const size_t n=tensor.TotalElements();
      std::vector<float> data(n);
      for(size_t i=0;i<n;++i)
      {
        data[i]=dist(rng);
      }
      tensor.CopyFromHostFp32(data.data(),n);
    };

    if(UsesQLoRA()==true)
    {
      fill_uniform(MutableWQCompress(),limit_qc);
      fill_uniform(MutableWQDecompress(),limit_qd);
    }
    else
    {
      fill_uniform(MutableWQ(),limit_q_direct);
    }
    fill_uniform(MutableWKVCompress(),limit_kvc);
    fill_uniform(MutableWKVDecompress(),limit_kvd);
    fill_uniform(MutableWO(),limit_o);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// KV-Cache management
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::EnableKVCache(uint32_t batch_size,uint32_t max_seq_len)
{
  try
  {
    const std::vector<uint32_t> kvc_shape={batch_size,max_seq_len,Config().KvLoraRank()};
    const std::vector<uint32_t> kpe_shape={batch_size,max_seq_len,Config().QkRopeHeadDim()};
    SetKVCacheCompressed(CAIF_DeviceTensor::Zeros(kvc_shape,Stream(),StorageDtype()));
    SetKVCacheKPE(CAIF_DeviceTensor::Zeros(kpe_shape,Stream(),StorageDtype()));
    SetKVCacheLen(0);
    SetKVCacheMaxLen(max_seq_len);
    SetKVCacheBatch(batch_size);
    SetKVCacheEnabled(true);

    // When the config left the decode-dispatch threshold on auto (0), resolve it
    // from the model shape and this GPU's compute:bandwidth ratio (device specs).
    if(Config().DecodeAbsorbThreshold()==0u)
    {
      const uint32_t resolved=CAIF_MlaDecodeThreshold::For(Config().Dim(),
                                                           Config().QkNopeHeadDim(),
                                                           Config().VHeadDim());
      if(resolved>0u)
      {
        CAIF_DeviceMLAttentionConfig resolved_cfg=Config();
        resolved_cfg.SetDecodeAbsorbThreshold(resolved);
        SetConfig(resolved_cfg);
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::DisableKVCache()
{
  SetKVCacheCompressed(CAIF_DeviceTensor());
  SetKVCacheKPE(CAIF_DeviceTensor());
  SetKVCacheLen(0);
  SetKVCacheMaxLen(0);
  SetKVCacheBatch(0);
  SetKVCacheEnabled(false);
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::ResetKVCache()
{
  if(IsKVCacheEnabled()==true)
  {
    MutableKVCacheCompressed().FillZero();
    MutableKVCacheKPE().FillZero();
    SetKVCacheLen(0);
  }
}

// Build the absorbed decode weights once. Folds W_k_nope^T into the Q (nope)
// projection so a decode query lands in the normed-latent space, and folds
// W_v then W_o so the attention-weighted latent maps straight to the output —
// letting incremental decode attend in the compressed latent space without ever
// decompressing the KV cache.
template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::PrecomputeAbsorbedWeights()
{
  try
  {
    if(AbsorbReady()==true||UsesProjections()==true)
    {
      return;
    }
    const CAIF_DeviceMLAttentionConfig &cfg=Config();
    const uint32_t heads=cfg.NumHeads();
    const uint32_t nope=cfg.QkNopeHeadDim();
    const uint32_t rope=cfg.QkRopeHeadDim();
    const uint32_t vdim=cfg.VHeadDim();
    const uint32_t kvr=cfg.KvLoraRank();
    const uint32_t qk=QKHeadDim();
    const uint32_t kv_block=nope+vdim;
    const uint32_t dim=cfg.Dim();
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const size_t elem=CAIF_DataType(sdt).ElementSizeBytes();

    const CAIF_DeviceTensor *w_q_src_ptr=&WQ();
    uint32_t q_src_dim=dim;
    if(UsesQLoRA()==true)
    {
      w_q_src_ptr=&WQDecompress();
      q_src_dim=cfg.QLoraRank();
    }
    const CAIF_DeviceTensor &w_q_src=*w_q_src_ptr;

    CAIF_RunContext ctx;
    ctx.SetStream(Stream());

    // _w_vo [heads*kvr, dim]: row block h (contiguous) = W_v_h @ W_o_h.
    CAIF_DeviceTensor wvo=CAIF_DeviceTensor::Uninitialized({heads*kvr,dim},Stream(),sdt);
    // _w_q_absorbed [q_src_dim, heads*kvr]: column block h = W_q_nope_h @ W_k_nope_h^T,
    // assembled by concatenating the per-head [q_src_dim, kvr] blocks along last dim.
    CAIF_DeviceTensor wqa;
    // _w_q_rope [q_src_dim, heads*rope]: column block h = the rope slice of
    // head h's Q projection (columns h*qk+nope .. h*qk+qk). Gathered once so
    // decode never pays the full W_q read just to obtain the rope part.
    CAIF_DeviceTensor wqr;

    for(uint32_t h=0;h<heads;++h)
    {
      CAIF_DeviceTensor w_q_nope_h=CAIF_DeviceTensor::Uninitialized({q_src_dim,nope},Stream(),sdt);
      CAIF_Ops::SliceLastDim(w_q_src,w_q_nope_h,h*qk);
      CAIF_DeviceTensor w_k_nope_h=CAIF_DeviceTensor::Uninitialized({kvr,nope},Stream(),sdt);
      CAIF_Ops::SliceLastDim(WKVDecompress(),w_k_nope_h,h*kv_block);
      CAIF_DeviceTensor wqa_h=CAIF_DeviceTensor::Uninitialized({q_src_dim,kvr},Stream(),sdt);
      CAIF_Ops::MatMulTransposeB(w_q_nope_h,w_k_nope_h,wqa_h,ctx,ComputeDtype());
      CAIF_DeviceTensor wqr_h=CAIF_DeviceTensor::Uninitialized({q_src_dim,rope},Stream(),sdt);
      CAIF_Ops::SliceLastDim(w_q_src,wqr_h,h*qk+nope);
      if(h==0)
      {
        wqa=std::move(wqa_h);
        wqr=std::move(wqr_h);
      }
      else
      {
        CAIF_DeviceTensor cat=CAIF_DeviceTensor::Uninitialized({q_src_dim,(h+1)*kvr},
                                                               Stream(),
                                                               sdt);
        CAIF_Ops::ConcatLastDim(wqa,wqa_h,cat);
        wqa=std::move(cat);
        CAIF_DeviceTensor cat_r=CAIF_DeviceTensor::Uninitialized({q_src_dim,(h+1)*rope},
                                                                 Stream(),
                                                                 sdt);
        CAIF_Ops::ConcatLastDim(wqr,wqr_h,cat_r);
        wqr=std::move(cat_r);
      }

      CAIF_DeviceTensor w_v_h=CAIF_DeviceTensor::Uninitialized({kvr,vdim},Stream(),sdt);
      CAIF_Ops::SliceLastDim(WKVDecompress(),w_v_h,h*kv_block+nope);
      uint8_t *wo_h_ptr=static_cast<uint8_t*>(MutableWO().DeviceDataRaw())+static_cast<size_t>(h)*vdim*dim*elem;
      CAIF_DeviceTensor w_o_h=CAIF_DeviceTensor::WrapView(wo_h_ptr,{vdim,dim},Stream(),sdt);
      uint8_t *wvo_h_ptr=static_cast<uint8_t*>(wvo.DeviceDataRaw())+static_cast<size_t>(h)*kvr*dim*elem;
      CAIF_DeviceTensor wvo_h=CAIF_DeviceTensor::WrapView(wvo_h_ptr,{kvr,dim},Stream(),sdt);
      CAIF_Ops::MatMul(w_v_h,w_o_h,wvo_h,ctx,ComputeDtype());
    }

    // The decode GEMVs are M=1, so their cost is the folded-weight read.
    // Store the folds in AbsorbedDtype() (bf16 for fp32 models, halving the
    // per-step read); the GEMVs still accumulate at ComputeDtype.
    const CAIF_DataType::CAIF_DataType_e adt=AbsorbedDtype();
    if(adt!=sdt)
    {
      CAIF_DeviceTensor wqa_a=CAIF_DeviceTensor::Uninitialized({q_src_dim,heads*kvr},Stream(),adt);
      CAIF_Ops::Cast(wqa,wqa_a,ctx);
      wqa=std::move(wqa_a);
      CAIF_DeviceTensor wvo_a=CAIF_DeviceTensor::Uninitialized({heads*kvr,dim},Stream(),adt);
      CAIF_Ops::Cast(wvo,wvo_a,ctx);
      wvo=std::move(wvo_a);
      CAIF_DeviceTensor wqr_a=CAIF_DeviceTensor::Uninitialized({q_src_dim,heads*rope},Stream(),adt);
      CAIF_Ops::Cast(wqr,wqr_a,ctx);
      wqr=std::move(wqr_a);
    }

    SetWQAbsorbed(std::move(wqa));
    SetWVO(std::move(wvo));
    SetWQRope(std::move(wqr));
    SetAbsorbReady(true);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Cached forward (autoregressive inference)
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMLAttention<ComputeT,StorageT>::ForwardCached(const CAIF_DeviceTensor &input,
                                                        CAIF_RunContext &ctx)
{
  try
  {
    if(IsKVCacheEnabled()==false)
    {
      THROW_CAIFE("MLA ForwardCached: KV cache not enabled");
    }

    const CAIF_DeviceMLAttentionConfig &cfg=Config();
    const auto &shape=input.Shape();
    if(shape.size()!=3||shape[2]!=cfg.Dim())
    {
      THROW_CAIFE("MLA ForwardCached: input must be [batch, seq_len, dim]");
    }

    const uint32_t batch=shape[0];
    const uint32_t new_len=shape[1];
    const uint32_t bs=batch*new_len;
    const uint32_t bh=batch*cfg.NumHeads();
    const uint32_t nope=cfg.QkNopeHeadDim();
    const uint32_t rope=cfg.QkRopeHeadDim();
    const uint32_t v_dim=cfg.VHeadDim();
    const uint32_t cache_len=KVCacheLength();
    const uint32_t cache_max_len=KVCacheMaxLen();
    const uint32_t total_len=cache_len+new_len;
    const float scale=1.0f/std::sqrt(static_cast<float>(QKHeadDim()));

    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,cfg.Dim()});

    //------------------------------------------------------------------
    // Matrix-absorption decode applies to batch==1, new_len==1, a cache
    // at/above the configured absorb threshold, and raw-weight no-q-LoRA
    // MLA. Decided up front: the absorbed path computes its query from the
    // folded weights and never pays the full W_q projection.
    //------------------------------------------------------------------
    const bool use_absorbed=(batch==1
                             &&new_len==1
                             &&cache_len>0
                             &&cache_len>=cfg.DecodeAbsorbThreshold()
                             &&UsesProjections()==false
                             &&UsesQLoRA()==false);

    //------------------------------------------------------------------
    // Q path — branches on UsesQLoRA(): LoRA chain when q_lora_rank>0,
    // single direct matmul (DeepSeek-V2-Lite path) when ==0. Skipped
    // entirely on the absorbed path, which builds its query below from
    // _w_q_absorbed / _w_q_rope instead.
    //------------------------------------------------------------------
    CAIF_DeviceTensor q_nope;
    CAIF_DeviceTensor q_rope_t;
    CAIF_DeviceTensor q;
    if(use_absorbed==false)
    {
      CAIF_DeviceTensor q_full;
      if(UsesQLoRA()==true)
      {
        CAIF_DeviceTensor q_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.QLoraRank()},
                                                                        Stream(),
                                                                        StorageDtype());
        CAIF_Ops::MatMul(flat_input,WQCompress(),q_compressed,ctx,ComputeDtype());

        CAIF_DeviceTensor q_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.QLoraRank()},Stream(),StorageDtype());
        CAIF_DeviceTensor q_rms=CAIF_DeviceTensor::Uninitialized({bs},Stream());
        launch_rmsnorm_forward<StorageT>(q_compressed.template DevicePtr<StorageT>(),
                                         QNormGamma().template DevicePtr<float>(),
                                         q_normed.template DevicePtr<StorageT>(),
                                         q_rms.template DevicePtr<float>(),
                                         cfg.RmsNormEps(),
                                         static_cast<int>(bs),
                                         static_cast<int>(cfg.QLoraRank()),
                                         Stream().Handle());

        q_full=CAIF_DeviceTensor::Uninitialized({bs,QProjDim()},Stream(),StorageDtype());
        CAIF_Ops::MatMul(q_normed,WQDecompress(),q_full,ctx,ComputeDtype());
      }
      else
      {
        q_full=CAIF_DeviceTensor::Uninitialized({bs,QProjDim()},Stream(),StorageDtype());
        CAIF_Ops::MatMul(flat_input,WQ(),q_full,ctx,ComputeDtype());
      }

      q_full.Reshape({batch,new_len,cfg.NumHeads(),QKHeadDim()});
      CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized({bh,new_len,QKHeadDim()},
                                                                      Stream(),
                                                                      StorageDtype());
      launch_transpose_0213<StorageT>(q_full.template DevicePtr<StorageT>(),
                                      q_transposed.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(new_len),
                                      static_cast<int>(cfg.NumHeads()),
                                      static_cast<int>(QKHeadDim()),
                                      Stream().Handle());

      q_nope=CAIF_DeviceTensor::Uninitialized({bh,new_len,nope},Stream(),StorageDtype());
      q_rope_t=CAIF_DeviceTensor::Uninitialized({bh,new_len,rope},Stream(),StorageDtype());
      CAIF_Ops::SliceLastDim(q_transposed,q_nope,0);
      CAIF_Ops::SliceLastDim(q_transposed,q_rope_t,static_cast<uint32_t>(nope));

      launch_rope_forward_offset<StorageT>(q_rope_t.template DevicePtr<StorageT>(),
                                           static_cast<int>(bh),
                                           static_cast<int>(new_len),
                                           static_cast<int>(rope),
                                           cfg.RopeBase(),
                                           static_cast<int>(cache_len),
                                           cfg.RopeStyle(),
                                           Stream().Handle());

      q=CAIF_DeviceTensor::Uninitialized({bh,new_len,QKHeadDim()},Stream(),StorageDtype());
      CAIF_Ops::ConcatLastDim(q_nope,q_rope_t,q);
    }

    //------------------------------------------------------------------
    // KV path: compress the new tokens, RMSNorm + RoPE them, and append the
    // normed/roped latent to the cache. Caching post-norm/post-rope is what
    // lets the absorbed decode below skip re-normalizing and re-roping the
    // whole cache every step (the old O(n^2) recompute).
    //------------------------------------------------------------------
    CAIF_DeviceTensor kv_out=CAIF_DeviceTensor::Uninitialized({bs,KVCompressDim()},Stream(),StorageDtype());
    CAIF_Ops::MatMul(flat_input,WKVCompress(),kv_out,ctx,ComputeDtype());

    CAIF_DeviceTensor new_kv_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.KvLoraRank()},
                                                                         Stream(),
                                                                         StorageDtype());
    CAIF_DeviceTensor new_k_pe=CAIF_DeviceTensor::Uninitialized({bs,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(kv_out,new_kv_compressed,0);
    CAIF_Ops::SliceLastDim(kv_out,new_k_pe,cfg.KvLoraRank());

    CAIF_DeviceTensor new_kv_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.KvLoraRank()},
                                                                     Stream(),
                                                                     StorageDtype());
    CAIF_DeviceTensor new_kv_rms=CAIF_DeviceTensor::Uninitialized({bs},Stream());
    launch_rmsnorm_forward<StorageT>(new_kv_compressed.template DevicePtr<StorageT>(),
                                      KVNormGamma().template DevicePtr<float>(),
                                      new_kv_normed.template DevicePtr<StorageT>(),
                                      new_kv_rms.template DevicePtr<float>(),
                                      cfg.RmsNormEps(),
                                      static_cast<int>(bs),
                                      static_cast<int>(cfg.KvLoraRank()),
                                      Stream().Handle());

    // RoPE the new k_pe at its absolute positions (offset cache_len) before any
    // head expansion (RoPE is shared across heads); cache stores it roped.
    new_k_pe.Reshape({batch,new_len,rope});
    launch_rope_forward_offset<StorageT>(new_k_pe.template DevicePtr<StorageT>(),
                                          static_cast<int>(batch),
                                          static_cast<int>(new_len),
                                          static_cast<int>(rope),
                                          cfg.RopeBase(),
                                          static_cast<int>(cache_len),
                                          cfg.RopeStyle(),
                                          Stream().Handle());

    new_kv_normed.Reshape({batch,new_len,cfg.KvLoraRank()});
    launch_kv_cache_append<StorageT>(new_kv_normed.template DevicePtr<StorageT>(),
                                      MutableKVCacheCompressed().template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(new_len),
                                      static_cast<int>(cache_len),
                                      static_cast<int>(cache_max_len),
                                      1,
                                      static_cast<int>(cfg.KvLoraRank()),
                                      Stream().Handle());
    launch_kv_cache_append<StorageT>(new_k_pe.template DevicePtr<StorageT>(),
                                      MutableKVCacheKPE().template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(new_len),
                                      static_cast<int>(cache_len),
                                      static_cast<int>(cache_max_len),
                                      1,
                                      static_cast<int>(rope),
                                      Stream().Handle());

    SetKVCacheLen(total_len);

    //------------------------------------------------------------------
    // Matrix-absorption decode fast path. Attends directly in the normed-latent
    // space (no cache decompression -> O(n) per step); the gating decision was
    // made before the Q path above. The folded weights live at AbsorbedDtype()
    // (bf16 for fp32 models), so the two big M=1 GEMVs read half the bytes;
    // their tiny activations are cast at the boundaries and everything still
    // accumulates at ComputeDtype.
    //------------------------------------------------------------------
    if(use_absorbed==true)
    {
      PrecomputeAbsorbedWeights();
      const uint32_t heads=cfg.NumHeads();
      const uint32_t kvr=cfg.KvLoraRank();
      const CAIF_DataType::CAIF_DataType_e adt=AbsorbedDtype();

      // Decode input at the folded width ([1, dim] cast; identity copy when
      // AbsorbedDtype matches storage).
      CAIF_DeviceTensor flat_a=CAIF_DeviceTensor::Uninitialized({1,cfg.Dim()},Stream(),adt);
      CAIF_Ops::Cast(flat_input,flat_a,ctx);

      // q_absorbed [heads, kvr] = flat_a [1, dim] @ W_q_absorbed [dim, heads*kvr],
      // brought back to StorageT for the cache-side matmuls.
      CAIF_DeviceTensor q_absorbed_a=CAIF_DeviceTensor::Uninitialized({1,heads*kvr},Stream(),adt);
      CAIF_Ops::MatMul(flat_a,WQAbsorbed(),q_absorbed_a,ctx,ComputeDtype());
      CAIF_DeviceTensor q_absorbed=CAIF_DeviceTensor::Uninitialized({1,heads*kvr},Stream(),StorageDtype());
      CAIF_Ops::Cast(q_absorbed_a,q_absorbed,ctx);
      q_absorbed.Reshape({heads,kvr});

      // q_rope [heads, rope] from the gathered rope slice of W_q — decode
      // never reads the full W_q. Roped at the new token's absolute position.
      CAIF_DeviceTensor q_rope_a=CAIF_DeviceTensor::Uninitialized({1,heads*rope},Stream(),adt);
      CAIF_Ops::MatMul(flat_a,WQRope(),q_rope_a,ctx,ComputeDtype());
      CAIF_DeviceTensor q_rope_dec=CAIF_DeviceTensor::Uninitialized({1,heads*rope},Stream(),StorageDtype());
      CAIF_Ops::Cast(q_rope_a,q_rope_dec,ctx);
      q_rope_dec.Reshape({heads,1,rope});
      launch_rope_forward_offset<StorageT>(q_rope_dec.template DevicePtr<StorageT>(),
                                           static_cast<int>(heads),
                                           1,
                                           static_cast<int>(rope),
                                           cfg.RopeBase(),
                                           static_cast<int>(cache_len),
                                           cfg.RopeStyle(),
                                           Stream().Handle());

      // Views of the valid (normed/roped) cache prefix; batch==1 -> contiguous.
      CAIF_DeviceTensor c_kv=CAIF_DeviceTensor::WrapView(MutableKVCacheCompressed().DeviceDataRaw(),
                                                         {total_len,kvr},
                                                         Stream(),
                                                         StorageDtype());
      CAIF_DeviceTensor c_kpe=CAIF_DeviceTensor::WrapView(MutableKVCacheKPE().DeviceDataRaw(),
                                                          {total_len,rope},
                                                          Stream(),
                                                          StorageDtype());
      CAIF_DeviceTensor q_rope2=CAIF_DeviceTensor::WrapView(q_rope_dec.DeviceDataRaw(),
                                                            {heads,rope},
                                                            Stream(),
                                                            StorageDtype());

      // scores [heads, total] = q_absorbed·c_kv^T + q_rope·c_kpe^T (shared cache
      // -> heads are rows; no per-head broadcast, no decompression).
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({heads,total_len},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(q_absorbed,c_kv,scores,ctx,ComputeDtype());
      CAIF_DeviceTensor scores_rope=CAIF_DeviceTensor::Uninitialized({heads,total_len},
                                                                     Stream(),
                                                                     StorageDtype());
      CAIF_Ops::MatMulTransposeB(q_rope2,c_kpe,scores_rope,ctx,ComputeDtype());
      CAIF_Ops::Add(scores,scores_rope,scores);
      CAIF_Ops::Scale(scores,scale);

      // The one new token sees every cached position (causal-safe), so no mask.
      CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized({heads,total_len},Stream(),scores.Dtype());
      launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                         attn.template DevicePtr<StorageT>(),
                                         static_cast<int>(heads),
                                         static_cast<int>(total_len),
                                         Stream().Handle());

      // ctx_latent [heads, kvr] = attn·c_kv; output = ctx_latent_flat·W_vo at
      // the folded width (the W_vo read is the other M=1 heavy hitter).
      CAIF_DeviceTensor ctx_latent=CAIF_DeviceTensor::Uninitialized({heads,kvr},Stream(),StorageDtype());
      CAIF_Ops::MatMul(attn,c_kv,ctx_latent,ctx,ComputeDtype());
      ctx_latent.Reshape({1,heads*kvr});
      CAIF_DeviceTensor ctx_latent_a=CAIF_DeviceTensor::Uninitialized({1,heads*kvr},Stream(),adt);
      CAIF_Ops::Cast(ctx_latent,ctx_latent_a,ctx);
      CAIF_DeviceTensor output_a=CAIF_DeviceTensor::Uninitialized({1,cfg.Dim()},Stream(),adt);
      CAIF_Ops::MatMul(ctx_latent_a,WVO(),output_a,ctx,ComputeDtype());
      CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({1,cfg.Dim()},Stream(),StorageDtype());
      CAIF_Ops::Cast(output_a,output,ctx);
      output.Reshape({batch,new_len,cfg.Dim()});
      return output;
    }

    //------------------------------------------------------------------
    // Standard decompress path (prefill + fallbacks). The cache now holds the
    // normed latent, so decompress it directly with no re-RMSNorm.
    //------------------------------------------------------------------
    const uint32_t total_bs=batch*total_len;
    CAIF_DeviceTensor cached_kv=CAIF_DeviceTensor::Uninitialized({batch,total_len,cfg.KvLoraRank()},
                                                                 Stream(),
                                                                 StorageDtype());
    cached_kv.Reshape({total_bs,cfg.KvLoraRank()});

    // One strided copy instead of `batch` launches: lift the [total_len*
    // kv_lora_rank] normed-latent prefix out of the [cache_max_len*kv_lora_rank]
    // cache stride.
#ifdef USE_CAIF_CUDA
    const StorageT *cached_compressed_ptr=KVCacheCompressed().template DevicePtr<StorageT>();
    StorageT *cached_kv_ptr=cached_kv.template DevicePtr<StorageT>();
    const size_t kv_row_bytes=static_cast<size_t>(total_len)*cfg.KvLoraRank()*sizeof(StorageT);
    const size_t kv_src_pitch=static_cast<size_t>(cache_max_len)*cfg.KvLoraRank()*sizeof(StorageT);
    cudaError_t rc=cudaMemcpy2DAsync(cached_kv_ptr,
                                     kv_row_bytes,
                                     cached_compressed_ptr,
                                     kv_src_pitch,
                                     kv_row_bytes,
                                     batch,
                                     cudaMemcpyDeviceToDevice,
                                     Stream().Handle());
    if(rc!=cudaSuccess)
    {
      THROW_CAIFE("MLA KV-cache extract: cudaMemcpy2DAsync(latent) failed");
    }
#endif

    CAIF_DeviceTensor kv_full=CAIF_DeviceTensor::Uninitialized({total_bs,KVDecompDim()},Stream(),StorageDtype());
    CAIF_Ops::MatMul(cached_kv,WKVDecompress(),kv_full,ctx,ComputeDtype());

    const uint32_t kv_per_head=nope+v_dim;
    kv_full.Reshape({batch,total_len,cfg.NumHeads(),kv_per_head});
    CAIF_DeviceTensor kv_transposed=CAIF_DeviceTensor::Uninitialized({bh,total_len,kv_per_head},
                                                                     Stream(),
                                                                     StorageDtype());
    launch_transpose_0213<StorageT>(kv_full.template DevicePtr<StorageT>(),
                                     kv_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(total_len),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(kv_per_head),
                                     Stream().Handle());

    CAIF_DeviceTensor k_nope=CAIF_DeviceTensor::Uninitialized({bh,total_len,nope},Stream(),StorageDtype());
    CAIF_DeviceTensor v_heads=CAIF_DeviceTensor::Uninitialized({bh,total_len,v_dim},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(kv_transposed,k_nope,0);
    CAIF_Ops::SliceLastDim(kv_transposed,v_heads,nope);

    //------------------------------------------------------------------
    // Attention -> attn_weights [bh, new_len, total_len]. Two routes:
    //  - decode (batch==1, new_len==1): score the nope part per head and the
    //    rope part as one shared-k_pe matmul, skipping the k_pe head broadcast,
    //    the key concat, and the per-step k_pe copy. The single new token sees
    //    every cached position, so no causal mask is needed.
    //  - prefill / batched: build the full key (broadcast + concat) and score.
    //------------------------------------------------------------------
    CAIF_DeviceTensor attn_weights=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},
                                                                    Stream(),
                                                                    StorageDtype());
    CAIF_DeviceTensor attn_output=CAIF_DeviceTensor::Uninitialized({bh,new_len,v_dim},Stream(),StorageDtype());
    bool used_flash_cached=false;
    if(batch==1&&new_len==1)
    {
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,total_len},Stream(),StorageDtype());
      CAIF_Ops::BatchedMatMulTransposeB(q_nope,
                                         k_nope,
                                         scores,
                                         static_cast<int>(new_len),
                                         static_cast<int>(nope),
                                         static_cast<int>(total_len),
                                         static_cast<int>(bh),
                                         ctx);
      CAIF_DeviceTensor q_rope_2d=CAIF_DeviceTensor::WrapView(q_rope_t.DeviceDataRaw(),
                                                             {bh,rope},
                                                             Stream(),
                                                             StorageDtype());
      CAIF_DeviceTensor k_pe_view=CAIF_DeviceTensor::WrapView(MutableKVCacheKPE().DeviceDataRaw(),
                                                              {total_len,rope},
                                                              Stream(),
                                                              StorageDtype());
      CAIF_DeviceTensor scores_rope=CAIF_DeviceTensor::Uninitialized({bh,total_len},
                                                                     Stream(),
                                                                     StorageDtype());
      CAIF_Ops::MatMulTransposeB(q_rope_2d,k_pe_view,scores_rope,ctx,ComputeDtype());
      CAIF_Ops::Add(scores,scores_rope,scores);
      CAIF_Ops::Scale(scores,scale);
      launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                         attn_weights.template DevicePtr<StorageT>(),
                                         static_cast<int>(bh),
                                         static_cast<int>(total_len),
                                         Stream().Handle());
    }
    else
    {
      // Extract and broadcast cached k_pe for full sequence (typed ptr arith).
      CAIF_DeviceTensor cached_k_pe=CAIF_DeviceTensor::Uninitialized({batch,total_len,rope},
                                                                      Stream(),
                                                                      StorageDtype());
      // One strided copy instead of `batch` launches: lift the [total_len*rope]
      // cached k_pe prefix out of the [cache_max_len*rope] cache stride.
#ifdef USE_CAIF_CUDA
      const StorageT *cache_kpe_ptr=KVCacheKPE().template DevicePtr<StorageT>();
      StorageT *cached_kpe_ptr=cached_k_pe.template DevicePtr<StorageT>();
      const size_t kpe_row_bytes=static_cast<size_t>(total_len)*rope*sizeof(StorageT);
      const size_t kpe_src_pitch=static_cast<size_t>(cache_max_len)*rope*sizeof(StorageT);
      cudaError_t rc=cudaMemcpy2DAsync(cached_kpe_ptr,
                                       kpe_row_bytes,
                                       cache_kpe_ptr,
                                       kpe_src_pitch,
                                       kpe_row_bytes,
                                       batch,
                                       cudaMemcpyDeviceToDevice,
                                       Stream().Handle());
      if(rc!=cudaSuccess)
      {
        THROW_CAIFE("MLA KV-cache extract: cudaMemcpy2DAsync(k_pe) failed");
      }
#endif

      CAIF_DeviceTensor k_pe_expanded=CAIF_DeviceTensor::Uninitialized({bh,total_len,rope},
                                                                       Stream(),
                                                                       StorageDtype());
      launch_gqa_repeat_kv<StorageT>(cached_k_pe.template DevicePtr<StorageT>(),
                                      k_pe_expanded.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      1,
                                      static_cast<int>(cfg.NumHeads()),
                                      static_cast<int>(total_len),
                                      static_cast<int>(rope),
                                      Stream().Handle());

      // cached_k_pe is already roped (applied once at append time), so the
      // head-expanded copy needs no further RoPE here.
      CAIF_DeviceTensor k=CAIF_DeviceTensor::Uninitialized({bh,total_len,QKHeadDim()},
                                                           Stream(),
                                                           StorageDtype());
      CAIF_Ops::ConcatLastDim(k_nope,k_pe_expanded,k);

      // Fused tensor-core flash prefill into a warm KV cache: q_len=new_len,
      // kv_len=total_len, q_offset=cache_len. Only true prefill (new_len>1) —
      // batched single-token decode (new_len==1) keeps the explicit path.
      if(new_len>1 && CAIF_Settings::FlashMlaPrefill()==true)
      {
        int flash_device=0;
        cudaGetDevice(&flash_device);
        if(mla_flash_prefill_available(static_cast<int>(QKHeadDim()),
                                       static_cast<int>(v_dim),
                                       flash_device)==true)
        {
          int causal_flag=0;
          if(cfg.Causal()==true)
          {
            causal_flag=1;
          }
          const bool launched=launch_flash_attention_forward_mla<StorageT>(
                              q.template DevicePtr<StorageT>(),
                              k.template DevicePtr<StorageT>(),
                              v_heads.template DevicePtr<StorageT>(),
                              attn_output.template DevicePtr<StorageT>(),
                              static_cast<int>(bh),
                              static_cast<int>(new_len),
                              static_cast<int>(total_len),
                              static_cast<int>(QKHeadDim()),
                              static_cast<int>(v_dim),
                              scale,
                              causal_flag,
                              static_cast<int>(cache_len),
                              Stream().Handle());
          if(launched==false)
          {
            THROW_CAIFE("MLA flash prefill (cached) launch failed after availability check");
          }
          used_flash_cached=true;
        }
      }
      if(used_flash_cached==false)
      {
        CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},
                                                                  Stream(),
                                                                  StorageDtype());
        CAIF_Ops::BatchedMatMulTransposeB(q,
                                          k,
                                          scores,
                                          static_cast<int>(new_len),
                                          static_cast<int>(QKHeadDim()),
                                          static_cast<int>(total_len),
                                          static_cast<int>(bh),
                                          ctx);
        CAIF_Ops::Scale(scores,scale);

        if(cfg.Causal()==true)
        {
          launch_causal_mask_fill_offset<StorageT>(scores.template DevicePtr<StorageT>(),
                                                   static_cast<int>(bh),
                                                   static_cast<int>(new_len),
                                                   static_cast<int>(total_len),
                                                   static_cast<int>(cache_len),
                                                   Stream().Handle());
        }

        launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                           attn_weights.template DevicePtr<StorageT>(),
                                           static_cast<int>(bh*new_len),
                                           static_cast<int>(total_len),
                                           Stream().Handle());
      }
    }

    if(used_flash_cached==false)
    {
      CAIF_Ops::BatchedMatMul(attn_weights,
                              v_heads,
                              attn_output,
                              static_cast<int>(new_len),
                              static_cast<int>(total_len),
                              static_cast<int>(v_dim),
                              static_cast<int>(bh),
                              ctx);
    }

    //------------------------------------------------------------------
    // Output projection
    //------------------------------------------------------------------
    const std::vector<uint32_t> merged_shape={batch,new_len,cfg.NumHeads(),v_dim};
    CAIF_DeviceTensor attn_merged=CAIF_DeviceTensor::Uninitialized(merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(attn_output.template DevicePtr<StorageT>(),
                                     attn_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.NumHeads()),
                                     static_cast<int>(new_len),
                                     static_cast<int>(v_dim),
                                     Stream().Handle());
    attn_merged.Reshape({bs,OInputDim()});

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({bs,cfg.Dim()},Stream(),StorageDtype());
    CAIF_Ops::MatMul(attn_merged,WO(),output,ctx,ComputeDtype());

    SetKVCacheLen(total_len);

    output.Reshape({batch,new_len,cfg.Dim()});
    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadWQCompress(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UsesProjections()==true)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQCompress: not valid when using sub-projections");
    }
    if(UsesQLoRA()==false)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQCompress: not valid when q_lora_rank==0; use LoadWQ");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().Dim() ||
       shape[1]!=Config().QLoraRank())
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQCompress: shape mismatch, expected [dim, q_lora_rank]");
    }
    SetWQCompress(std::move(w));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadQNormGamma(CAIF_DeviceTensor &&gamma)
{
  try
  {
    if(UsesQLoRA()==false)
    {
      THROW_CAIFE("DeviceMLAttention::LoadQNormGamma: not valid when q_lora_rank==0");
    }
    const std::vector<uint32_t> &shape=gamma.Shape();
    if(shape.size()!=1 || shape[0]!=Config().QLoraRank())
    {
      THROW_CAIFE("DeviceMLAttention::LoadQNormGamma: shape mismatch, expected [q_lora_rank]");
    }
    SetQNormGamma(std::move(gamma));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadWQDecompress(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UsesProjections()==true)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQDecompress: not valid when using sub-projections");
    }
    if(UsesQLoRA()==false)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQDecompress: not valid when q_lora_rank==0; use LoadWQ");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().QLoraRank() ||
       shape[1]!=QProjDim())
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQDecompress: shape mismatch, expected [q_lora_rank, q_proj_dim]");
    }
    SetWQDecompress(std::move(w));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadWQ(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UsesProjections()==true)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQ: not valid when using sub-projections");
    }
    if(UsesQLoRA()==true)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQ: not valid when q_lora_rank>0; use LoadWQCompress/LoadWQDecompress");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().Dim() ||
       shape[1]!=QProjDim())
    {
      THROW_CAIFE("DeviceMLAttention::LoadWQ: shape mismatch, expected [dim, q_proj_dim]");
    }
    SetWQ(std::move(w));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadWKVCompress(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UsesProjections()==true)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWKVCompress: not valid when using sub-projections");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().Dim() ||
       shape[1]!=KVCompressDim())
    {
      THROW_CAIFE("DeviceMLAttention::LoadWKVCompress: shape mismatch, expected [dim, kv_compress_dim]");
    }
    SetWKVCompress(std::move(w));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadKVNormGamma(CAIF_DeviceTensor &&gamma)
{
  try
  {
    const std::vector<uint32_t> &shape=gamma.Shape();
    if(shape.size()!=1 || shape[0]!=Config().KvLoraRank())
    {
      THROW_CAIFE("DeviceMLAttention::LoadKVNormGamma: shape mismatch, expected [kv_lora_rank]");
    }
    SetKVNormGamma(std::move(gamma));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadWKVDecompress(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UsesProjections()==true)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWKVDecompress: not valid when using sub-projections");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().KvLoraRank() ||
       shape[1]!=KVDecompDim())
    {
      THROW_CAIFE("DeviceMLAttention::LoadWKVDecompress: shape mismatch, expected [kv_lora_rank, kv_decomp_dim]");
    }
    SetWKVDecompress(std::move(w));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMLAttention<ComputeT,StorageT>::LoadWO(CAIF_DeviceTensor &&w_o)
{
  try
  {
    if(UsesProjections()==true)
    {
      THROW_CAIFE("DeviceMLAttention::LoadWO: not valid when using sub-projections");
    }
    const std::vector<uint32_t> &shape=w_o.Shape();
    if(shape.size()!=2 ||
       shape[0]!=OInputDim() ||
       shape[1]!=Config().Dim())
    {
      THROW_CAIFE("DeviceMLAttention::LoadWO: shape mismatch, expected [o_input_dim, dim]");
    }
    SetWO(std::move(w_o));
  }
  CAIF_CATCH_BLOCK()
}
// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceMLAttention<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceMLAttention<float,__half>;
template class CAIF_DeviceMLAttention<float,__nv_bfloat16>;
template class CAIF_DeviceMLAttention<__half,float>;
template class CAIF_DeviceMLAttention<__half,__half>;
template class CAIF_DeviceMLAttention<__half,__nv_bfloat16>;
template class CAIF_DeviceMLAttention<__nv_bfloat16,float>;
template class CAIF_DeviceMLAttention<__nv_bfloat16,__half>;
template class CAIF_DeviceMLAttention<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
