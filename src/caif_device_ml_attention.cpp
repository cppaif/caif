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
// Per-site dispositions per the type-dispatch full plan (Phases 2+3):
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
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <cmath>
#include <random>
#include <sstream>

namespace instance
{

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

template<typename ComputeT,typename StorageT>
CAIF_DeviceMLAttention<ComputeT,StorageT>::CAIF_DeviceMLAttention(const MLAConfig_t &config,
                                             CAIF_CudaStream &stream):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                                                      _config(config),
                                                                      _use_projections(false),
                                                                      _use_q_lora(config.q_lora_rank>0),
                                                                      _cached_batch(0),
                                                                      _cached_seq_len(0),
                                                                      _kv_cache_len(0),
                                                                      _kv_cache_max_len(0),
                                                                      _kv_cache_batch(0),
                                                                      _kv_cache_enabled(false)
{
  try
  {
    const MLAConfig_t &cfg=Config();
    const uint32_t qk_head_dim=cfg.qk_nope_head_dim+cfg.qk_rope_head_dim;
    SetQKHeadDim(qk_head_dim);
    SetQProjDim(cfg.num_heads*qk_head_dim);
    SetKVCompressDim(cfg.kv_lora_rank+cfg.qk_rope_head_dim);
    SetKVDecompDim(cfg.num_heads*(cfg.qk_nope_head_dim+cfg.v_head_dim));
    SetOInputDim(cfg.num_heads*cfg.v_head_dim);

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    // Q parameters: LoRA path (compress + norm + decompress) when
    // q_lora_rank>0; otherwise a single direct projection (DeepSeek-V2-Lite
    // and other configs that omit Q-LoRA).
    if(UsesQLoRA()==true)
    {
      SetWQCompress(CAIF_DeviceTensor::Uninitialized({cfg.dim,cfg.q_lora_rank},stream,sdt));
      // Norm gammas always live at fp32 — the launch_rmsnorm_backward
      // kernel reads/writes them via `DevicePtr<float>()`. Allocating at
      // a smaller dtype (fp16/bf16) overruns the buffer on backward
      // (writes 4 bytes into 2-byte slots) and corrupts the gamma grad.
      SetQNormGamma(CAIF_DeviceTensor::Zeros({cfg.q_lora_rank},stream,
                                              CAIF_DataType::CAIF_DataType_e::Float32));
      SetWQDecompress(CAIF_DeviceTensor::Uninitialized({cfg.q_lora_rank,QProjDim()},stream,sdt));
      SetGradWQCompress(CAIF_DeviceTensor::Zeros({cfg.dim,cfg.q_lora_rank},stream,sdt));
      SetGradQNormGamma(CAIF_DeviceTensor::Zeros({cfg.q_lora_rank},stream,
                                                  CAIF_DataType::CAIF_DataType_e::Float32));
      SetGradWQDecompress(CAIF_DeviceTensor::Zeros({cfg.q_lora_rank,QProjDim()},stream,sdt));
    }
    else
    {
      SetWQ(CAIF_DeviceTensor::Uninitialized({cfg.dim,QProjDim()},stream,sdt));
      SetGradWQ(CAIF_DeviceTensor::Zeros({cfg.dim,QProjDim()},stream,sdt));
    }

    // KV path (always LoRA) and output projection.
    SetWKVCompress(CAIF_DeviceTensor::Uninitialized({cfg.dim,KVCompressDim()},stream,sdt));
    // KV norm gamma always at fp32 — see Q-norm comment above.
    SetKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.kv_lora_rank},stream,
                                             CAIF_DataType::CAIF_DataType_e::Float32));
    SetWKVDecompress(CAIF_DeviceTensor::Uninitialized({cfg.kv_lora_rank,KVDecompDim()},stream,sdt));
    SetWO(CAIF_DeviceTensor::Uninitialized({OInputDim(),cfg.dim},stream,sdt));

    SetGradWKVCompress(CAIF_DeviceTensor::Zeros({cfg.dim,KVCompressDim()},stream,sdt));
    SetGradKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.kv_lora_rank},stream,
                                                 CAIF_DataType::CAIF_DataType_e::Float32));
    SetGradWKVDecompress(CAIF_DeviceTensor::Zeros({cfg.kv_lora_rank,KVDecompDim()},stream,sdt));
    SetGradWO(CAIF_DeviceTensor::Zeros({OInputDim(),cfg.dim},stream,sdt));

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
CAIF_DeviceMLAttention<ComputeT,StorageT>::CAIF_DeviceMLAttention(const MLAConfig_t &config,
                                             MLAProjections_t projections,
                                             CAIF_CudaStream &stream):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                                                      _config(config),
                                                                      _projections(std::move(projections)),
                                                                      _use_projections(true),
                                                                      _use_q_lora(config.q_lora_rank>0),
                                                                      _cached_batch(0),
                                                                      _cached_seq_len(0),
                                                                      _kv_cache_len(0),
                                                                      _kv_cache_max_len(0),
                                                                      _kv_cache_batch(0),
                                                                      _kv_cache_enabled(false)
{
  try
  {
    if(UsesQLoRA()==false)
    {
      THROW_CAIFE("MLA projections constructor requires q_lora_rank>0; the no-LoRA"
                  " Q path uses the basic-tensor constructor with LoadWQ()");
    }
    const MLAConfig_t &cfg=Config();
    const uint32_t qk_head_dim=cfg.qk_nope_head_dim+cfg.qk_rope_head_dim;
    SetQKHeadDim(qk_head_dim);
    SetQProjDim(cfg.num_heads*qk_head_dim);
    SetKVCompressDim(cfg.kv_lora_rank+cfg.qk_rope_head_dim);
    SetKVDecompDim(cfg.num_heads*(cfg.qk_nope_head_dim+cfg.v_head_dim));
    SetOInputDim(cfg.num_heads*cfg.v_head_dim);

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    // Norm parameters (not owned by projections).
    // Norm gammas always live at fp32 — see no-projections ctor comment.
    SetQNormGamma(CAIF_DeviceTensor::Zeros({cfg.q_lora_rank},stream,
                                            CAIF_DataType::CAIF_DataType_e::Float32));
    SetKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.kv_lora_rank},stream,
                                             CAIF_DataType::CAIF_DataType_e::Float32));
    SetGradQNormGamma(CAIF_DeviceTensor::Zeros({cfg.q_lora_rank},stream,
                                                CAIF_DataType::CAIF_DataType_e::Float32));
    SetGradKVNormGamma(CAIF_DeviceTensor::Zeros({cfg.kv_lora_rank},stream,
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
                                                    _kv_cache_enabled(false)
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
    const MLAConfig_t &cfg=Config();
    if(shape.size()!=3||shape[2]!=cfg.dim)
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
    const uint32_t bh=batch*cfg.num_heads;
    const uint32_t nope=cfg.qk_nope_head_dim;
    const uint32_t rope=cfg.qk_rope_head_dim;
    const uint32_t v_dim=cfg.v_head_dim;
    const float scale=1.0f/std::sqrt(static_cast<float>(QKHeadDim()));

    // Flatten input to [bs, dim]
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,cfg.dim});

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
        q_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.q_lora_rank},Stream(),StorageDtype());
        CAIF_Ops::MatMul(flat_input,WQCompress(),q_compressed,ctx,ComputeDtype());
      }

      q_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.q_lora_rank},Stream(),StorageDtype());
      q_rms=CAIF_DeviceTensor::Uninitialized({bs},Stream());
      launch_rmsnorm_forward<StorageT>(q_compressed.template DevicePtr<StorageT>(),
                                        QNormGamma().template DevicePtr<float>(),
                                        q_normed.template DevicePtr<StorageT>(),
                                        q_rms.template DevicePtr<float>(),
                                        cfg.rms_norm_eps,
                                        static_cast<int>(bs),
                                        static_cast<int>(cfg.q_lora_rank),
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
    q_full.Reshape({batch,seq_len,cfg.num_heads,QKHeadDim()});
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(q_full.template DevicePtr<StorageT>(),
                                     q_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(cfg.num_heads),
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
                                   cfg.rope_base,
                                   cfg.rope_style,
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
    CAIF_DeviceTensor kv_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.kv_lora_rank},Stream(),StorageDtype());
    CAIF_DeviceTensor k_pe_flat=CAIF_DeviceTensor::Uninitialized({bs,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(kv_out,kv_compressed,0);
    CAIF_Ops::SliceLastDim(kv_out,k_pe_flat,cfg.kv_lora_rank);

    // KV RMSNorm on compressed_kv
    CAIF_DeviceTensor kv_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.kv_lora_rank},Stream(),StorageDtype());
    CAIF_DeviceTensor kv_rms=CAIF_DeviceTensor::Uninitialized({bs},Stream());
    launch_rmsnorm_forward<StorageT>(kv_compressed.template DevicePtr<StorageT>(),
                                      KVNormGamma().template DevicePtr<float>(),
                                      kv_normed.template DevicePtr<StorageT>(),
                                      kv_rms.template DevicePtr<float>(),
                                      cfg.rms_norm_eps,
                                      static_cast<int>(bs),
                                      static_cast<int>(cfg.kv_lora_rank),
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
    kv_full.Reshape({batch,seq_len,cfg.num_heads,kv_per_head});
    CAIF_DeviceTensor kv_transposed=CAIF_DeviceTensor::Uninitialized({bh,seq_len,kv_per_head},Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(kv_full.template DevicePtr<StorageT>(),
                                     kv_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(cfg.num_heads),
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
                                    static_cast<int>(cfg.num_heads),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(rope),
                                    Stream().Handle());

    // Apply RoPE to k_pe
    launch_rope_forward<StorageT>(k_pe.template DevicePtr<StorageT>(),
                                   static_cast<int>(bh),
                                   static_cast<int>(seq_len),
                                   static_cast<int>(rope),
                                   cfg.rope_base,
                                   cfg.rope_style,
                                   Stream().Handle());

    //------------------------------------------------------------------
    // Assemble Q, K and run attention
    //------------------------------------------------------------------
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(q_nope,q_rope,q);

    CAIF_DeviceTensor k=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(k_nope,k_pe,k);

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
                                         static_cast<int>(cfg.num_heads),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    else if(cfg.causal==true)
    {
      launch_causal_mask_fill<StorageT>(scores.template DevicePtr<StorageT>(),
                                         static_cast<int>(bh),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    // Templated softmax dispatch on StorageT (Phase 9 sweep extending the
    // 5.1 MHA fix to MLA; MLA's constructor rejects Int8/Int4 storage so
    // reaching this site with a non-fp32/fp16/bf16 StorageT is impossible).
    CAIF_DeviceTensor attn_weights=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},Stream(),scores.Dtype());
    launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                       attn_weights.template DevicePtr<StorageT>(),
                                       static_cast<int>(bh*seq_len),
                                       static_cast<int>(seq_len),
                                       Stream().Handle());
    CAIF_DeviceTensor attn_output=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},Stream(),StorageDtype());
    CAIF_Ops::BatchedMatMul(attn_weights,
                             v,
                             attn_output,
                             static_cast<int>(seq_len),
                             static_cast<int>(seq_len),
                             static_cast<int>(v_dim),
                             static_cast<int>(bh),
                             ctx);

    //------------------------------------------------------------------
    // Output projection
    //------------------------------------------------------------------
    const std::vector<uint32_t> merged_shape={batch,seq_len,cfg.num_heads,v_dim};
    CAIF_DeviceTensor attn_merged=CAIF_DeviceTensor::Uninitialized(merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(attn_output.template DevicePtr<StorageT>(),
                                     attn_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.num_heads),
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
      output=CAIF_DeviceTensor::Uninitialized({bs,cfg.dim},Stream(),StorageDtype());
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

    output.Reshape({batch,seq_len,cfg.dim});
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
    const MLAConfig_t &cfg=Config();

    const uint32_t batch=CachedBatch();
    const uint32_t seq_len=CachedSeqLen();
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*cfg.num_heads;
    const uint32_t nope=cfg.qk_nope_head_dim;
    const uint32_t rope=cfg.qk_rope_head_dim;
    const uint32_t v_dim=cfg.v_head_dim;
    const float scale=1.0f/std::sqrt(static_cast<float>(QKHeadDim()));

    // Flatten grad_output to [bs, dim] — zero-copy view
    const std::vector<uint32_t> flat_shape={bs,cfg.dim};
    CAIF_DeviceTensor grad_flat=CAIF_DeviceTensor::WrapView(
                                 const_cast<void *>(grad_output.DeviceDataRaw()),
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

      const std::vector<uint32_t> w_o_shape={OInputDim(),cfg.dim};
      CAIF_DeviceTensor grad_w_o_delta=CAIF_DeviceTensor::Uninitialized(w_o_shape,Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedMerged(),grad_flat,grad_w_o_delta,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWO(),grad_w_o_delta,MutableGradWO());
    }

    //------------------------------------------------------------------
    // Reverse merge heads: [bs, o_input_dim] -> [bh, seq, v_dim]
    //------------------------------------------------------------------
    grad_merged.Reshape({batch,seq_len,cfg.num_heads,v_dim});
    CAIF_DeviceTensor grad_attn_output=CAIF_DeviceTensor::Uninitialized({bh,seq_len,v_dim},Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(grad_merged.template DevicePtr<StorageT>(),
                                     grad_attn_output.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(seq_len),
                                     static_cast<int>(cfg.num_heads),
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
                                         static_cast<int>(cfg.num_heads),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    else if(cfg.causal==true)
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
                                         static_cast<int>(cfg.num_heads),
                                         static_cast<int>(seq_len),
                                         Stream().Handle());
    }
    else if(cfg.causal==true)
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
                                    cfg.rope_base,
                                    cfg.rope_style,
                                    Stream().Handle());

    CAIF_DeviceTensor grad_q_full=CAIF_DeviceTensor::Uninitialized({bh,seq_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(grad_q_nope,grad_q_rope,grad_q_full);

    const std::vector<uint32_t> q_merged_shape={batch,seq_len,cfg.num_heads,QKHeadDim()};
    CAIF_DeviceTensor grad_q_merged=CAIF_DeviceTensor::Uninitialized(q_merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(grad_q_full.template DevicePtr<StorageT>(),
                                     grad_q_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.num_heads),
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
        grad_q_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.q_lora_rank},Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeB(grad_q_merged,WQDecompress(),grad_q_normed,ctx,ComputeDtype());

        const std::vector<uint32_t> w_qd_shape={cfg.q_lora_rank,QProjDim()};
        CAIF_DeviceTensor grad_w_qd=CAIF_DeviceTensor::Uninitialized(w_qd_shape,Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeA(CachedQNormed(),grad_q_merged,grad_w_qd,ctx,ComputeDtype());
        CAIF_Ops::Add(GradWQDecompress(),grad_w_qd,MutableGradWQDecompress());
      }

      CAIF_DeviceTensor grad_q_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.q_lora_rank},Stream(),StorageDtype());
      launch_rmsnorm_backward<StorageT>(grad_q_normed.template DevicePtr<StorageT>(),
                                         CachedQCompressed().template DevicePtr<StorageT>(),
                                         QNormGamma().template DevicePtr<float>(),
                                         CachedQRMS().template DevicePtr<float>(),
                                         grad_q_compressed.template DevicePtr<StorageT>(),
                                         MutableGradQNormGamma().template DevicePtr<float>(),
                                         cfg.rms_norm_eps,
                                         static_cast<int>(bs),
                                         static_cast<int>(cfg.q_lora_rank),
                                         Stream().Handle());

      if(UsesProjections()==true)
      {
        grad_input_q=Projections().q_compress->Backward(grad_q_compressed,ctx);
      }
      else
      {
        grad_input_q=CAIF_DeviceTensor::Uninitialized({bs,cfg.dim},Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeB(grad_q_compressed,WQCompress(),grad_input_q,ctx,ComputeDtype());

        const std::vector<uint32_t> w_qc_shape={cfg.dim,cfg.q_lora_rank};
        CAIF_DeviceTensor grad_w_qc=CAIF_DeviceTensor::Uninitialized(w_qc_shape,Stream(),StorageDtype());
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_q_compressed,grad_w_qc,ctx,ComputeDtype());
        CAIF_Ops::Add(GradWQCompress(),grad_w_qc,MutableGradWQCompress());
      }
    }
    else
    {
      // q_lora_rank==0: direct grad_q_merged -> grad_input_q via _w_q.
      grad_input_q=CAIF_DeviceTensor::Uninitialized({bs,cfg.dim},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_q_merged,WQ(),grad_input_q,ctx,ComputeDtype());

      const std::vector<uint32_t> w_q_shape={cfg.dim,QProjDim()};
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
                                    cfg.rope_base,
                                    cfg.rope_style,
                                    Stream().Handle());

    CAIF_DeviceTensor grad_k_pe_flat=CAIF_DeviceTensor::Zeros({batch,seq_len,rope},Stream(),StorageDtype());
    launch_gqa_reduce_kv<StorageT>(grad_k_pe.template DevicePtr<StorageT>(),
                                    grad_k_pe_flat.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    1,
                                    static_cast<int>(cfg.num_heads),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(rope),
                                    Stream().Handle());
    grad_k_pe_flat.Reshape({bs,rope});

    const uint32_t kv_per_head=nope+v_dim;
    CAIF_DeviceTensor grad_kv_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,kv_per_head},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(grad_k_nope,grad_v,grad_kv_heads);

    const std::vector<uint32_t> kv_merged_shape={batch,seq_len,cfg.num_heads,kv_per_head};
    CAIF_DeviceTensor grad_kv_merged=CAIF_DeviceTensor::Uninitialized(kv_merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(grad_kv_heads.template DevicePtr<StorageT>(),
                                     grad_kv_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.num_heads),
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
      grad_kv_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.kv_lora_rank},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_kv_merged,WKVDecompress(),grad_kv_normed,ctx,ComputeDtype());

      const std::vector<uint32_t> w_kvd_shape={cfg.kv_lora_rank,KVDecompDim()};
      CAIF_DeviceTensor grad_w_kvd=CAIF_DeviceTensor::Uninitialized(w_kvd_shape,Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedKVNormed(),grad_kv_merged,grad_w_kvd,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWKVDecompress(),grad_w_kvd,MutableGradWKVDecompress());
    }

    CAIF_DeviceTensor grad_kv_compressed=CAIF_DeviceTensor::Uninitialized(
        {bs,cfg.kv_lora_rank},Stream(),StorageDtype());
    launch_rmsnorm_backward<StorageT>(grad_kv_normed.template DevicePtr<StorageT>(),
                                       CachedKVCompressed().template DevicePtr<StorageT>(),
                                       KVNormGamma().template DevicePtr<float>(),
                                       CachedKVRMS().template DevicePtr<float>(),
                                       grad_kv_compressed.template DevicePtr<StorageT>(),
                                       MutableGradKVNormGamma().template DevicePtr<float>(),
                                       cfg.rms_norm_eps,
                                       static_cast<int>(bs),
                                       static_cast<int>(cfg.kv_lora_rank),
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
      grad_input_kv=CAIF_DeviceTensor::Uninitialized({bs,cfg.dim},Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_kv_out,WKVCompress(),grad_input_kv,ctx,ComputeDtype());

      const std::vector<uint32_t> w_kvc_shape={cfg.dim,KVCompressDim()};
      CAIF_DeviceTensor grad_w_kvc=CAIF_DeviceTensor::Uninitialized(w_kvc_shape,Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedInput(),grad_kv_out,grad_w_kvc,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWKVCompress(),grad_w_kvc,MutableGradWKVCompress());
    }

    //------------------------------------------------------------------
    // Combine input gradients
    //------------------------------------------------------------------
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({bs,cfg.dim},Stream(),StorageDtype());
    CAIF_Ops::Add(grad_input_q,grad_input_kv,grad_input);
    grad_input.Reshape({batch,seq_len,cfg.dim});

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
           Config().q_lora_rank+
           Config().kv_lora_rank;
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
  ss<<"MLA(dim="<<Config().dim
    <<",heads="<<Config().num_heads
    <<",q_lora="<<Config().q_lora_rank
    <<",kv_lora="<<Config().kv_lora_rank
    <<",nope="<<Config().qk_nope_head_dim
    <<",rope="<<Config().qk_rope_head_dim
    <<",v="<<Config().v_head_dim
    <<",params="<<TotalParameterCount();
  if(UsesProjections()==true)
  {
    ss<<",projections";
  }
  if(UsesQLoRA()==false)
  {
    ss<<",direct_q";
  }
  ss<<")";
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
  if(UsesProjections()==true)
  {
    std::vector<std::string> names;
    std::vector<std::string> sub;

    sub=Projections().q_compress->ParameterNames(prefix+"w_q_compress.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().q_decompress->ParameterNames(prefix+"w_q_decompress.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().kv_compress->ParameterNames(prefix+"w_kv_compress.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().kv_decompress->ParameterNames(prefix+"w_kv_decompress.");
    names.insert(names.end(),sub.begin(),sub.end());

    sub=Projections().o_proj->ParameterNames(prefix+"w_o.");
    names.insert(names.end(),sub.begin(),sub.end());

    names.push_back(prefix+"q_norm_gamma");
    names.push_back(prefix+"kv_norm_gamma");
    return names;
  }
  std::vector<std::string> names;
  if(UsesQLoRA()==true)
  {
    names.push_back(prefix+"w_q_compress");
    names.push_back(prefix+"q_norm_gamma");
    names.push_back(prefix+"w_q_decompress");
  }
  else
  {
    names.push_back(prefix+"w_q");
  }
  names.push_back(prefix+"w_kv_compress");
  names.push_back(prefix+"kv_norm_gamma");
  names.push_back(prefix+"w_kv_decompress");
  names.push_back(prefix+"w_o");
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
    const MLAConfig_t &cfg=Config();

    const float limit_q_direct=std::sqrt(6.0f/static_cast<float>(cfg.dim+QProjDim()));
    const float limit_qc=std::sqrt(6.0f/static_cast<float>(cfg.dim+cfg.q_lora_rank));
    const float limit_qd=std::sqrt(6.0f/static_cast<float>(cfg.q_lora_rank+QProjDim()));
    const float limit_kvc=std::sqrt(6.0f/static_cast<float>(cfg.dim+KVCompressDim()));
    const float limit_kvd=std::sqrt(6.0f/static_cast<float>(cfg.kv_lora_rank+KVDecompDim()));
    const float limit_o=std::sqrt(6.0f/static_cast<float>(OInputDim()+cfg.dim));

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
    const std::vector<uint32_t> kvc_shape={batch_size,max_seq_len,Config().kv_lora_rank};
    const std::vector<uint32_t> kpe_shape={batch_size,max_seq_len,Config().qk_rope_head_dim};
    SetKVCacheCompressed(CAIF_DeviceTensor::Zeros(kvc_shape,Stream(),StorageDtype()));
    SetKVCacheKPE(CAIF_DeviceTensor::Zeros(kpe_shape,Stream(),StorageDtype()));
    SetKVCacheLen(0);
    SetKVCacheMaxLen(max_seq_len);
    SetKVCacheBatch(batch_size);
    SetKVCacheEnabled(true);
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

    const MLAConfig_t &cfg=Config();
    const auto &shape=input.Shape();
    if(shape.size()!=3||shape[2]!=cfg.dim)
    {
      THROW_CAIFE("MLA ForwardCached: input must be [batch, seq_len, dim]");
    }

    const uint32_t batch=shape[0];
    const uint32_t new_len=shape[1];
    const uint32_t bs=batch*new_len;
    const uint32_t bh=batch*cfg.num_heads;
    const uint32_t nope=cfg.qk_nope_head_dim;
    const uint32_t rope=cfg.qk_rope_head_dim;
    const uint32_t v_dim=cfg.v_head_dim;
    const uint32_t cache_len=KVCacheLength();
    const uint32_t cache_max_len=KVCacheMaxLen();
    const uint32_t total_len=cache_len+new_len;
    const float scale=1.0f/std::sqrt(static_cast<float>(QKHeadDim()));

    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,cfg.dim});

    //------------------------------------------------------------------
    // Q path — branches on UsesQLoRA(): LoRA chain when q_lora_rank>0,
    // single direct matmul (DeepSeek-V2-Lite path) when ==0.
    //------------------------------------------------------------------
    CAIF_DeviceTensor q_full;
    if(UsesQLoRA()==true)
    {
      CAIF_DeviceTensor q_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.q_lora_rank},Stream(),StorageDtype());
      CAIF_Ops::MatMul(flat_input,WQCompress(),q_compressed,ctx,ComputeDtype());

      CAIF_DeviceTensor q_normed=CAIF_DeviceTensor::Uninitialized({bs,cfg.q_lora_rank},Stream(),StorageDtype());
      CAIF_DeviceTensor q_rms=CAIF_DeviceTensor::Uninitialized({bs},Stream());
      launch_rmsnorm_forward<StorageT>(q_compressed.template DevicePtr<StorageT>(),
                                        QNormGamma().template DevicePtr<float>(),
                                        q_normed.template DevicePtr<StorageT>(),
                                        q_rms.template DevicePtr<float>(),
                                        cfg.rms_norm_eps,
                                        static_cast<int>(bs),
                                        static_cast<int>(cfg.q_lora_rank),
                                        Stream().Handle());

      q_full=CAIF_DeviceTensor::Uninitialized({bs,QProjDim()},Stream(),StorageDtype());
      CAIF_Ops::MatMul(q_normed,WQDecompress(),q_full,ctx,ComputeDtype());
    }
    else
    {
      q_full=CAIF_DeviceTensor::Uninitialized({bs,QProjDim()},Stream(),StorageDtype());
      CAIF_Ops::MatMul(flat_input,WQ(),q_full,ctx,ComputeDtype());
    }

    q_full.Reshape({batch,new_len,cfg.num_heads,QKHeadDim()});
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized({bh,new_len,QKHeadDim()},Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(q_full.template DevicePtr<StorageT>(),
                                     q_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(new_len),
                                     static_cast<int>(cfg.num_heads),
                                     static_cast<int>(QKHeadDim()),
                                     Stream().Handle());

    CAIF_DeviceTensor q_nope=CAIF_DeviceTensor::Uninitialized({bh,new_len,nope},Stream(),StorageDtype());
    CAIF_DeviceTensor q_rope_t=CAIF_DeviceTensor::Uninitialized({bh,new_len,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(q_transposed,q_nope,0);
    CAIF_Ops::SliceLastDim(q_transposed,q_rope_t,static_cast<uint32_t>(nope));

    launch_rope_forward_offset<StorageT>(q_rope_t.template DevicePtr<StorageT>(),
                                          static_cast<int>(bh),
                                          static_cast<int>(new_len),
                                          static_cast<int>(rope),
                                          cfg.rope_base,
                                          static_cast<int>(cache_len),
                                          cfg.rope_style,
                                          Stream().Handle());

    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized({bh,new_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(q_nope,q_rope_t,q);

    //------------------------------------------------------------------
    // KV path: compress new tokens, append to cache, decompress full cache
    //------------------------------------------------------------------
    CAIF_DeviceTensor kv_out=CAIF_DeviceTensor::Uninitialized({bs,KVCompressDim()},Stream(),StorageDtype());
    CAIF_Ops::MatMul(flat_input,WKVCompress(),kv_out,ctx,ComputeDtype());

    CAIF_DeviceTensor new_kv_compressed=CAIF_DeviceTensor::Uninitialized({bs,cfg.kv_lora_rank},Stream(),StorageDtype());
    CAIF_DeviceTensor new_k_pe=CAIF_DeviceTensor::Uninitialized({bs,rope},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(kv_out,new_kv_compressed,0);
    CAIF_Ops::SliceLastDim(kv_out,new_k_pe,cfg.kv_lora_rank);

    new_kv_compressed.Reshape({batch,new_len,cfg.kv_lora_rank});
    launch_kv_cache_append<StorageT>(new_kv_compressed.template DevicePtr<StorageT>(),
                                      MutableKVCacheCompressed().template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(new_len),
                                      static_cast<int>(cache_len),
                                      static_cast<int>(cache_max_len),
                                      1,
                                      static_cast<int>(cfg.kv_lora_rank),
                                      Stream().Handle());

    new_k_pe.Reshape({batch,new_len,rope});
    launch_kv_cache_append<StorageT>(new_k_pe.template DevicePtr<StorageT>(),
                                      MutableKVCacheKPE().template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(new_len),
                                      static_cast<int>(cache_len),
                                      static_cast<int>(cache_max_len),
                                      1,
                                      static_cast<int>(rope),
                                      Stream().Handle());

    // Decompress full cached KV: [batch, total_len, kv_lora_rank]
    const uint32_t total_bs=batch*total_len;
    CAIF_DeviceTensor cached_kv=CAIF_DeviceTensor::Uninitialized(
        {batch,total_len,cfg.kv_lora_rank},Stream(),StorageDtype());

    cached_kv.Reshape({total_bs,cfg.kv_lora_rank});
    CAIF_DeviceTensor kv_normed=CAIF_DeviceTensor::Uninitialized({total_bs,cfg.kv_lora_rank},Stream(),StorageDtype());
    CAIF_DeviceTensor kv_rms=CAIF_DeviceTensor::Uninitialized({total_bs},Stream());

    // Copy the [batch, 0:total_len, kv_lora_rank] valid prefix out of the
    // [batch, max_len, kv_lora_rank] cache, batch by batch — typed StorageT
    // pointer arithmetic + sizeof(StorageT) for the byte count.
    const StorageT *cached_compressed_ptr=KVCacheCompressed().template DevicePtr<StorageT>();
    StorageT *cached_kv_ptr=cached_kv.template DevicePtr<StorageT>();
    for(uint32_t b=0;b<batch;++b)
    {
      const StorageT *src=cached_compressed_ptr+b*cache_max_len*cfg.kv_lora_rank;
      StorageT *dst=cached_kv_ptr+b*total_len*cfg.kv_lora_rank;
      const size_t copy_bytes=total_len*cfg.kv_lora_rank*sizeof(StorageT);
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(dst,src,copy_bytes,cudaMemcpyDeviceToDevice,Stream().Handle());
#endif
    }

    launch_rmsnorm_forward<StorageT>(cached_kv.template DevicePtr<StorageT>(),
                                      KVNormGamma().template DevicePtr<float>(),
                                      kv_normed.template DevicePtr<StorageT>(),
                                      kv_rms.template DevicePtr<float>(),
                                      cfg.rms_norm_eps,
                                      static_cast<int>(total_bs),
                                      static_cast<int>(cfg.kv_lora_rank),
                                      Stream().Handle());

    CAIF_DeviceTensor kv_full=CAIF_DeviceTensor::Uninitialized({total_bs,KVDecompDim()},Stream(),StorageDtype());
    CAIF_Ops::MatMul(kv_normed,WKVDecompress(),kv_full,ctx,ComputeDtype());

    const uint32_t kv_per_head=nope+v_dim;
    kv_full.Reshape({batch,total_len,cfg.num_heads,kv_per_head});
    CAIF_DeviceTensor kv_transposed=CAIF_DeviceTensor::Uninitialized(
        {bh,total_len,kv_per_head},Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(kv_full.template DevicePtr<StorageT>(),
                                     kv_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(total_len),
                                     static_cast<int>(cfg.num_heads),
                                     static_cast<int>(kv_per_head),
                                     Stream().Handle());

    CAIF_DeviceTensor k_nope=CAIF_DeviceTensor::Uninitialized({bh,total_len,nope},Stream(),StorageDtype());
    CAIF_DeviceTensor v_heads=CAIF_DeviceTensor::Uninitialized({bh,total_len,v_dim},Stream(),StorageDtype());
    CAIF_Ops::SliceLastDim(kv_transposed,k_nope,0);
    CAIF_Ops::SliceLastDim(kv_transposed,v_heads,nope);

    // Extract and broadcast cached k_pe for full sequence (typed ptr arith).
    CAIF_DeviceTensor cached_k_pe=CAIF_DeviceTensor::Uninitialized({batch,total_len,rope},
                                                                    Stream(),
                                                                    StorageDtype());
    const StorageT *cache_kpe_ptr=KVCacheKPE().template DevicePtr<StorageT>();
    StorageT *cached_kpe_ptr=cached_k_pe.template DevicePtr<StorageT>();
    for(uint32_t b=0;b<batch;++b)
    {
      const StorageT *src=cache_kpe_ptr+b*cache_max_len*rope;
      StorageT *dst=cached_kpe_ptr+b*total_len*rope;
      const size_t copy_bytes=total_len*rope*sizeof(StorageT);
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(dst,src,copy_bytes,cudaMemcpyDeviceToDevice,Stream().Handle());
#endif
    }

    CAIF_DeviceTensor k_pe_expanded=CAIF_DeviceTensor::Uninitialized({bh,total_len,rope},Stream(),StorageDtype());
    launch_gqa_repeat_kv<StorageT>(cached_k_pe.template DevicePtr<StorageT>(),
                                    k_pe_expanded.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    1,
                                    static_cast<int>(cfg.num_heads),
                                    static_cast<int>(total_len),
                                    static_cast<int>(rope),
                                    Stream().Handle());

    launch_rope_forward<StorageT>(k_pe_expanded.template DevicePtr<StorageT>(),
                                   static_cast<int>(bh),
                                   static_cast<int>(total_len),
                                   static_cast<int>(rope),
                                   cfg.rope_base,
                                   cfg.rope_style,
                                   Stream().Handle());

    CAIF_DeviceTensor k=CAIF_DeviceTensor::Uninitialized({bh,total_len,QKHeadDim()},Stream(),StorageDtype());
    CAIF_Ops::ConcatLastDim(k_nope,k_pe_expanded,k);

    //------------------------------------------------------------------
    // Attention: Q [bh, new_len, qk_head_dim] x K [bh, total_len, qk_head_dim]
    //------------------------------------------------------------------
    CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},Stream(),StorageDtype());
    CAIF_Ops::BatchedMatMulTransposeB(q,
                                       k,
                                       scores,
                                       static_cast<int>(new_len),
                                       static_cast<int>(QKHeadDim()),
                                       static_cast<int>(total_len),
                                       static_cast<int>(bh),
                                       ctx);
    CAIF_Ops::Scale(scores,scale);

    if(cfg.causal==true)
    {
      launch_causal_mask_fill_offset<StorageT>(scores.template DevicePtr<StorageT>(),
                                                static_cast<int>(bh),
                                                static_cast<int>(new_len),
                                                static_cast<int>(total_len),
                                                static_cast<int>(cache_len),
                                                Stream().Handle());
    }

    CAIF_DeviceTensor attn_weights=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},Stream(),scores.Dtype());
    launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                       attn_weights.template DevicePtr<StorageT>(),
                                       static_cast<int>(bh*new_len),
                                       static_cast<int>(total_len),
                                       Stream().Handle());

    CAIF_DeviceTensor attn_output=CAIF_DeviceTensor::Uninitialized({bh,new_len,v_dim},Stream(),StorageDtype());
    CAIF_Ops::BatchedMatMul(attn_weights,
                             v_heads,
                             attn_output,
                             static_cast<int>(new_len),
                             static_cast<int>(total_len),
                             static_cast<int>(v_dim),
                             static_cast<int>(bh),
                             ctx);

    //------------------------------------------------------------------
    // Output projection
    //------------------------------------------------------------------
    const std::vector<uint32_t> merged_shape={batch,new_len,cfg.num_heads,v_dim};
    CAIF_DeviceTensor attn_merged=CAIF_DeviceTensor::Uninitialized(merged_shape,Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(attn_output.template DevicePtr<StorageT>(),
                                     attn_merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(cfg.num_heads),
                                     static_cast<int>(new_len),
                                     static_cast<int>(v_dim),
                                     Stream().Handle());
    attn_merged.Reshape({bs,OInputDim()});

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({bs,cfg.dim},Stream(),StorageDtype());
    CAIF_Ops::MatMul(attn_merged,WO(),output,ctx,ComputeDtype());

    SetKVCacheLen(total_len);

    output.Reshape({batch,new_len,cfg.dim});
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
       shape[0]!=Config().dim ||
       shape[1]!=Config().q_lora_rank)
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
    if(shape.size()!=1 || shape[0]!=Config().q_lora_rank)
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
       shape[0]!=Config().q_lora_rank ||
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
       shape[0]!=Config().dim ||
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
       shape[0]!=Config().dim ||
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
    if(shape.size()!=1 || shape[0]!=Config().kv_lora_rank)
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
       shape[0]!=Config().kv_lora_rank ||
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
       shape[1]!=Config().dim)
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
