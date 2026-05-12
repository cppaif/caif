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

// Per-site dispositions per the type-dispatch full plan (Phases 2+3):
//   - `logsumexp` / `_cached_logsumexp` are fp32 SDPA reduction caches —
//     `DevicePtr<float>()`, with `// fp32: SDPA reduction cache` comment.
//   - `ctx.PrefixLengths()` is a uint32_t tensor — passed directly via
//     `DevicePtr<uint32_t>()`. Phase 3 changed the kernel signatures
//     from `const int *` to `const uint32_t *`, so no reinterpret_cast
//     remains at the call sites.

#include "caif_device_multi_head_attention.h"
#include "caif_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <random>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::CAIF_DeviceMultiHeadAttention(
                               const AttentionConfig_t &config,
                               CAIF_CudaStream &stream):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                                        _config(config),
                                                        _use_projections(false),
                                                        _w_qkv_dirty(false),
                                                        _w_q(),
                                                        _w_k(),
                                                        _w_v(),
                                                        _w_o(),
                                                        _w_qkv(),
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
                                                        _cached_q_pre_norm(),
                                                        _cached_k_pre_norm(),
                                                        _cached_q_rms(),
                                                        _cached_k_rms(),
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
    if(config.use_rope==true&&config.rope_dim!=0)
    {
      if(config.rope_dim<=0||config.rope_dim>static_cast<int>(config.head_dim))
      {
        THROW_CAIFE("DeviceMultiHeadAttention: rope_dim must be in (0, head_dim] when nonzero");
      }
      if((config.rope_dim%2)!=0)
      {
        THROW_CAIFE("DeviceMultiHeadAttention: rope_dim must be even");
      }
    }
    if(StorageDtype()==CAIF_DataType::CAIF_DataType_e::Int8||
       StorageDtype()==CAIF_DataType::CAIF_DataType_e::Int4)
    {
      THROW_CAIFE("DeviceMultiHeadAttention: Int8/Int4 storage_dtype is not"
                  " trainable; use FrozenLinear or QAT path");
    }
    const uint32_t qk_dim=Config().num_heads*Config().head_dim;
    const uint32_t kv_dim=Config().num_kv_heads*Config().head_dim;

    SetWQ(CAIF_DeviceTensor::Uninitialized({Config().dim,qk_dim},stream,StorageDtype()));
    SetWK(CAIF_DeviceTensor::Uninitialized({Config().dim,kv_dim},stream,StorageDtype()));
    SetWV(CAIF_DeviceTensor::Uninitialized({Config().dim,kv_dim},stream,StorageDtype()));
    SetWO(CAIF_DeviceTensor::Uninitialized({qk_dim,Config().dim},stream,StorageDtype()));

    SetGradWQ(CAIF_DeviceTensor::Zeros({Config().dim,qk_dim},stream,StorageDtype()));
    SetGradWK(CAIF_DeviceTensor::Zeros({Config().dim,kv_dim},stream,StorageDtype()));
    SetGradWV(CAIF_DeviceTensor::Zeros({Config().dim,kv_dim},stream,StorageDtype()));
    SetGradWO(CAIF_DeviceTensor::Zeros({qk_dim,Config().dim},stream,StorageDtype()));

    // Initialize weights
    InitializeWeights(0);

    // Build fused [Q|K|V] weight for the inference MatMul fast-path.
    BuildFusedWqkv();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::CAIF_DeviceMultiHeadAttention(
  const AttentionConfig_t &config,
  MHAProjections_t projections,
  CAIF_CudaStream &stream):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                           _config(config),
                           _projections(std::move(projections)),
                           _use_projections(true),
                           _w_qkv_dirty(false),
                           _w_q(),
                           _w_k(),
                           _w_v(),
                           _w_o(),
                           _w_qkv(),
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
                           _cached_q_pre_norm(),
                           _cached_k_pre_norm(),
                           _cached_q_rms(),
                           _cached_k_rms(),
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
    if(config.use_rope==true&&config.rope_dim!=0)
    {
      if(config.rope_dim<=0||config.rope_dim>static_cast<int>(config.head_dim))
      {
        THROW_CAIFE("DeviceMultiHeadAttention: rope_dim must be in (0, head_dim] when nonzero");
      }
      if((config.rope_dim%2)!=0)
      {
        THROW_CAIFE("DeviceMultiHeadAttention: rope_dim must be even");
      }
    }

    // Projections mode: weight allocation and initialization are handled
    // by the projection layers (FrozenLinear/LoRAAdapter). No internal
    // weight matrices needed.
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::CAIF_DeviceMultiHeadAttention(
  CAIF_DeviceMultiHeadAttention &&other):CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                                         _config(other._config),
                                         _projections(std::move(other._projections)),
                                         _use_projections(other._use_projections),
                                         _w_qkv_dirty(other._w_qkv_dirty),
                                         _w_q(std::move(other._w_q)),
                                         _w_k(std::move(other._w_k)),
                                         _w_v(std::move(other._w_v)),
                                         _w_o(std::move(other._w_o)),
                                         _w_qkv(std::move(other._w_qkv)),
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
                                         _cached_q_pre_norm(std::move(other._cached_q_pre_norm)),
                                         _cached_k_pre_norm(std::move(other._cached_k_pre_norm)),
                                         _cached_q_rms(std::move(other._cached_q_rms)),
                                         _cached_k_rms(std::move(other._cached_k_rms)),
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceMultiHeadAttention<ComputeT,StorageT> &CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::operator=(
                               CAIF_DeviceMultiHeadAttention &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      _config=other._config;
      _projections=std::move(other._projections);
      _use_projections=other._use_projections;
      _w_qkv_dirty=other._w_qkv_dirty;
      _w_q=std::move(other._w_q);
      _w_k=std::move(other._w_k);
      _w_v=std::move(other._w_v);
      _w_o=std::move(other._w_o);
      _w_qkv=std::move(other._w_qkv);
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
      _cached_q_pre_norm=std::move(other._cached_q_pre_norm);
      _cached_k_pre_norm=std::move(other._cached_k_pre_norm);
      _cached_q_rms=std::move(other._cached_q_rms);
      _cached_k_rms=std::move(other._cached_k_rms);
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ForwardImpl(
                   const CAIF_DeviceTensor &input,
                   CAIF_RunContext &ctx)
{
  try
  {
    const bool has_prefix=ctx.HasPrefixLengths();

    // Step 1: Validate input shape [batch, seq_len, dim]
    const auto &shape=input.Shape();
    if(shape.size()!=3)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::Forward: input must be 3D [batch,seq_len,dim]");
    }
    if(shape[2]!=Config().dim)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::Forward: last dim must match config dim");
    }
    if(HasProjections()==false&&input.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("DeviceMultiHeadAttention::Forward: input dtype must match"
                  " config.storage_dtype (caller must Cast upstream)");
    }

    const uint32_t batch=shape[0];
    const uint32_t seq_len=shape[1];
    const uint32_t dim=Config().dim;
    const uint32_t num_heads=Config().num_heads;
    const uint32_t num_kv_heads=Config().num_kv_heads;
    const uint32_t head_dim=Config().head_dim;
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*num_heads;
    const uint32_t bkv=batch*num_kv_heads;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;

    // Step 2: Flatten input to [batch*seq_len, dim]
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({bs,dim});

    // Step 3: Project Q, K, V + split heads into q/k/v_transposed.
    // Inference fast-path (no sub-projections, not training, fused weight
    // is built) collapses 3 MatMuls into 1 and uses strided transposes
    // that write directly into q/k/v_transposed — skipping q/k/v_proj
    // allocation altogether.
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bh*seq_len*head_dim},ctx.Stream(),StorageDtype());
    CAIF_DeviceTensor k_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv*seq_len*head_dim},ctx.Stream(),StorageDtype());
    CAIF_DeviceTensor v_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv*seq_len*head_dim},ctx.Stream(),StorageDtype());

    // QK-norm needs separate q_proj / k_proj buffers so RMSNorm can run
    // before the head reshape — the fused-QKV path emits one strided
    // buffer and is incompatible. Fall back to the un-fused path when
    // a gamma is loaded.
    const bool qk_norm_active=(HasQNormGamma()==true
                               || HasKNormGamma()==true);
    const bool use_fused_qkv=(HasProjections()==false
                              && ctx.Training()==false
                              && WQkv().IsAllocated()==true
                              && qk_norm_active==false);

    if(use_fused_qkv==true && WQkvDirty()==true)
    {
      BuildFusedWqkv();
    }

    if(use_fused_qkv==true)
    {
      const uint32_t total_dim=qk_dim+2u*kv_dim;
      CAIF_DeviceTensor qkv_proj=CAIF_DeviceTensor::Uninitialized({bs,total_dim},
                                                                   ctx.Stream(),
                                                                   StorageDtype());
      CAIF_Ops::MatMul(flat_input,WQkv(),qkv_proj,ctx,ComputeDtype());

      launch_transpose_0213_strided<StorageT>(qkv_proj.template DevicePtr<StorageT>()+0u,
                                              q_transposed.template DevicePtr<StorageT>(),
                                              static_cast<int>(batch),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(num_heads),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(total_dim),
                                              ctx.Stream().Handle());
      launch_transpose_0213_strided<StorageT>(qkv_proj.template DevicePtr<StorageT>()
                                                +static_cast<size_t>(qk_dim),
                                              k_transposed.template DevicePtr<StorageT>(),
                                              static_cast<int>(batch),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(num_kv_heads),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(total_dim),
                                              ctx.Stream().Handle());
      launch_transpose_0213_strided<StorageT>(qkv_proj.template DevicePtr<StorageT>()
                                                +static_cast<size_t>(qk_dim+kv_dim),
                                              v_transposed.template DevicePtr<StorageT>(),
                                              static_cast<int>(batch),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(num_kv_heads),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(total_dim),
                                              ctx.Stream().Handle());
    }
    else
    {
      CAIF_DeviceTensor q_proj;
      CAIF_DeviceTensor k_proj;
      CAIF_DeviceTensor v_proj;

      if(HasProjections()==true)
      {
        q_proj=QProj().Forward(flat_input,ctx);
        k_proj=KProj().Forward(flat_input,ctx);
        v_proj=VProj().Forward(flat_input,ctx);
        if(QBias().IsEmpty()==false)
        {
          CAIF_Ops::BiasAdd(q_proj,QBias(),q_proj);
        }
        if(KBias().IsEmpty()==false)
        {
          CAIF_Ops::BiasAdd(k_proj,KBias(),k_proj);
        }
        if(VBias().IsEmpty()==false)
        {
          CAIF_Ops::BiasAdd(v_proj,VBias(),v_proj);
        }
      }
      else
      {
        q_proj=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},ctx.Stream(),StorageDtype());
        k_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},ctx.Stream(),StorageDtype());
        v_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},ctx.Stream(),StorageDtype());
        CAIF_Ops::MatMul(flat_input,WQ(),q_proj,ctx,ComputeDtype());
        CAIF_Ops::MatMul(flat_input,WK(),k_proj,ctx,ComputeDtype());
        CAIF_Ops::MatMul(flat_input,WV(),v_proj,ctx,ComputeDtype());
      }

      // Optional QK-norm: RMSNorm applied across the full per-token
      // projection width (qk_dim for Q, kv_dim for K). Lives between
      // bias-add and the head reshape, matching HF OLMoE / Olmo2 /
      // Qwen3 forward order. No-op when gammas are empty. When
      // training, the pre-norm Q/K and the per-row rms denominator
      // are cached for the backward rmsnorm_backward call.
      if(HasQNormGamma()==true)
      {
        CAIF_DeviceTensor q_normed=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},
                                                                    ctx.Stream(),
                                                                    StorageDtype());
        CAIF_DeviceTensor q_rms=CAIF_DeviceTensor::Uninitialized({bs},ctx.Stream());
        launch_rmsnorm_forward<StorageT>(q_proj.template DevicePtr<StorageT>(),
                                          QNormGamma().template DevicePtr<float>(),
                                          q_normed.template DevicePtr<StorageT>(),
                                          q_rms.template DevicePtr<float>(),
                                          Config().qk_norm_eps,
                                          static_cast<int>(bs),
                                          static_cast<int>(qk_dim),
                                          ctx.Stream().Handle());
        if(ctx.Training()==true)
        {
          SetCachedQPreNorm(std::move(q_proj));
          SetCachedQRms(std::move(q_rms));
        }
        q_proj=std::move(q_normed);
      }
      if(HasKNormGamma()==true)
      {
        CAIF_DeviceTensor k_normed=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},
                                                                    ctx.Stream(),
                                                                    StorageDtype());
        CAIF_DeviceTensor k_rms=CAIF_DeviceTensor::Uninitialized({bs},ctx.Stream());
        launch_rmsnorm_forward<StorageT>(k_proj.template DevicePtr<StorageT>(),
                                          KNormGamma().template DevicePtr<float>(),
                                          k_normed.template DevicePtr<StorageT>(),
                                          k_rms.template DevicePtr<float>(),
                                          Config().qk_norm_eps,
                                          static_cast<int>(bs),
                                          static_cast<int>(kv_dim),
                                          ctx.Stream().Handle());
        if(ctx.Training()==true)
        {
          SetCachedKPreNorm(std::move(k_proj));
          SetCachedKRms(std::move(k_rms));
        }
        k_proj=std::move(k_normed);
      }

      launch_transpose_0213<StorageT>(q_proj.template DevicePtr<StorageT>(),
                                      q_transposed.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(seq_len),
                                      static_cast<int>(num_heads),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
      launch_transpose_0213<StorageT>(k_proj.template DevicePtr<StorageT>(),
                                      k_transposed.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(seq_len),
                                      static_cast<int>(num_kv_heads),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
      launch_transpose_0213<StorageT>(v_proj.template DevicePtr<StorageT>(),
                                      v_transposed.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(seq_len),
                                      static_cast<int>(num_kv_heads),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
    }

    q_transposed.Reshape({bh,seq_len,head_dim});
    k_transposed.Reshape({bkv,seq_len,head_dim});
    v_transposed.Reshape({bkv,seq_len,head_dim});

    // Step 4.5: RoPE (applied in-place on Q and K after head split).
    // Config().rope_dim==0 means "full rotation" (legacy behavior, all
    // head_dim dims rotated). Nonzero rope_dim < head_dim means partial
    // rotary (Glm4Moe-style): rotate first rope_dim dims of each head,
    // pass the rest through. Dispatch to the matching launcher.
    if(Config().use_rope==true)
    {
      int rope_dim_eff=static_cast<int>(head_dim);
      if(Config().rope_dim!=0)
      {
        rope_dim_eff=Config().rope_dim;
      }
      if(rope_dim_eff==static_cast<int>(head_dim))
      {
        launch_rope_forward<StorageT>(q_transposed.template DevicePtr<StorageT>(),
                                      static_cast<int>(bh),
                                      static_cast<int>(seq_len),
                                      static_cast<int>(head_dim),
                                      Config().rope_base,
                                      Config().rope_style,
                                      ctx.Stream().Handle());
        launch_rope_forward<StorageT>(k_transposed.template DevicePtr<StorageT>(),
                                      static_cast<int>(bkv),
                                      static_cast<int>(seq_len),
                                      static_cast<int>(head_dim),
                                      Config().rope_base,
                                      Config().rope_style,
                                      ctx.Stream().Handle());
      }
      else
      {
        launch_rope_forward_partial<StorageT>(q_transposed.template DevicePtr<StorageT>(),
                                              static_cast<int>(bh),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              rope_dim_eff,
                                              Config().rope_base,
                                              Config().rope_style,
                                              ctx.Stream().Handle());
        launch_rope_forward_partial<StorageT>(k_transposed.template DevicePtr<StorageT>(),
                                              static_cast<int>(bkv),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              rope_dim_eff,
                                              Config().rope_base,
                                              Config().rope_style,
                                              ctx.Stream().Handle());
      }
    }

    // Flash gating decision (hoisted above GQA expand — flash path uses
    // native KV indexing (bh_kv = bh*num_kv_heads/num_heads) so it reads
    // the unexpanded K/V directly; only the naive path needs materialized
    // expansion).
    const bool flash_head_dim_supported=(head_dim==32||head_dim==64
                                        ||head_dim==80||head_dim==96
                                        ||head_dim==128);
    const bool use_flash=UseFlashAttention()==true
                         && RequiresExplicitScores()==false
                         && flash_head_dim_supported==true;

    // Step 4.6: GQA expand (repeat KV heads to match Q heads) — naive path only
    CAIF_DeviceTensor k_expanded;
    CAIF_DeviceTensor v_expanded;
    if(num_kv_heads!=num_heads && use_flash==false)
    {
      const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);
      k_expanded=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                  ctx.Stream(),
                                                  StorageDtype());
      v_expanded=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                  ctx.Stream(),
                                                  StorageDtype());
      launch_gqa_repeat_kv<StorageT>(k_transposed.template DevicePtr<StorageT>(),
                                     k_expanded.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
      launch_gqa_repeat_kv<StorageT>(v_transposed.template DevicePtr<StorageT>(),
                                     v_expanded.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    }
    else
    {
      // Flash path keeps [bkv, seq_len, head_dim] shape; naive-path MHA
      // (num_kv_heads==num_heads) takes this branch too, shape equals [bh,...].
      k_expanded=std::move(k_transposed);
      v_expanded=std::move(v_transposed);
    }

    // Steps 5-9: Attention computation
    // Use FlashAttention for fused, memory-efficient attention
    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_DeviceTensor context=CAIF_DeviceTensor::Uninitialized(
                               {bh,seq_len,head_dim},ctx.Stream(),StorageDtype());
    CAIF_DeviceTensor logsumexp;
    CAIF_DeviceTensor attn;

    if(use_flash==true)
    {
      // FlashAttention: fused QKV attention with online softmax
      // No attention matrix materialization - O(n) memory instead of O(n²)
      logsumexp=CAIF_DeviceTensor::Uninitialized({bh,seq_len},ctx.Stream());

      if(has_prefix==true)
      {
        launch_flash_attention_forward_prefix<StorageT>(
                                            q_transposed.template DevicePtr<StorageT>(),
                                            k_expanded.template DevicePtr<StorageT>(),
                                            v_expanded.template DevicePtr<StorageT>(),
                                            context.template DevicePtr<StorageT>(),
                                            logsumexp.DevicePtr<float>(),
                                            ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                            static_cast<int>(batch),
                                            static_cast<int>(num_heads),
                                            static_cast<int>(num_kv_heads),
                                            static_cast<int>(seq_len),
                                            static_cast<int>(head_dim),
                                            scale,
                                            ctx.Stream().Handle());
      }
      else
      {
        int causal_flag=0;
        if(Config().causal==true)
        {
          causal_flag=1;
        }
        launch_flash_attention_forward<StorageT>(q_transposed.template DevicePtr<StorageT>(),
                                                 k_expanded.template DevicePtr<StorageT>(),
                                                 v_expanded.template DevicePtr<StorageT>(),
                                                 context.template DevicePtr<StorageT>(),
                                                 logsumexp.DevicePtr<float>(),
                                                 static_cast<int>(bh),
                                                 static_cast<int>(seq_len),
                                                 static_cast<int>(head_dim),
                                                 scale,
                                                 causal_flag,
                                                 static_cast<int>(num_heads),
                                                 static_cast<int>(num_kv_heads),
                                                 ctx.Stream().Handle());
      }
    }

    if(use_flash==false)
    {
      // Naive attention: explicit materialization of attention matrix
      // Step 5: scores = Q_heads @ K_expanded^T -> [bh, seq_len, seq_len]
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized(
                                {bh,seq_len,seq_len},ctx.Stream(),StorageDtype());
      CAIF_Ops::BatchedMatMulTransposeB(q_transposed,k_expanded,scores,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(bh),
                                              ctx);

      // Step 6: Scale by 1/sqrt(head_dim)
      CAIF_Ops::Scale(scores,scale);

      // Step 6b: Subclass hook for score modification (e.g., position bias)
      ApplyScoreBias(scores,batch,seq_len,ctx);

      // Step 7: Causal or prefix-LM mask
      if(has_prefix==true)
      {
        launch_prefix_mask_fill<StorageT>(scores.template DevicePtr<StorageT>(),
                                          ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                          static_cast<int>(batch),
                                          static_cast<int>(Config().num_heads),
                                          static_cast<int>(seq_len),
                                          ctx.Stream().Handle());
      }
      else if(Config().causal==true)
      {
        launch_causal_mask_fill<StorageT>(scores.template DevicePtr<StorageT>(),
                                          static_cast<int>(bh),
                                          static_cast<int>(seq_len),
                                          ctx.Stream().Handle());
      }

      // Step 8: Softmax. Templated dispatch on StorageT — kernel has 3
      // explicit instantiations (fp32/fp16/bf16). MHA's constructor
      // rejects Int8/Int4 storage at construction (trainable weight
      // matrices), so reaching here with a non-fp32/fp16/bf16 StorageT
      // is impossible.
      attn=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},
                                            ctx.Stream(),
                                            scores.Dtype());
      launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                         attn.template DevicePtr<StorageT>(),
                                         static_cast<int>(bh*seq_len),
                                         static_cast<int>(seq_len),
                                         ctx.Stream().Handle());

      // Step 9: context = attn @ V_expanded -> [bh, seq_len, head_dim]
      CAIF_Ops::BatchedMatMul(attn,v_expanded,context,
                                    static_cast<int>(seq_len),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(head_dim),
                                    static_cast<int>(bh),
                                    ctx);
    }

    // Step 10: Merge heads via reverse transpose_0213
    // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    CAIF_DeviceTensor merged=CAIF_DeviceTensor::Uninitialized(
                              {bs*qk_dim},ctx.Stream(),StorageDtype());
    launch_transpose_0213<StorageT>(context.template DevicePtr<StorageT>(),
                                    merged.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(num_heads),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());
    merged.Reshape({bs,qk_dim});

    CAIF_DeviceTensor output_flat;
    if(HasProjections()==true)
    {
      output_flat=OProj().Forward(merged,ctx);
    }
    else
    {
      output_flat=CAIF_DeviceTensor::Uninitialized({bs,dim},ctx.Stream(),StorageDtype());
      CAIF_Ops::MatMul(merged,WO(),output_flat,ctx,ComputeDtype());
    }

    // Step 12: Reshape to [batch, seq_len, dim]
    output_flat.Reshape({batch,seq_len,dim});

    // Step 13: Cache for backward (store non-expanded K/V for GQA).
    // Mirror the Step 4.6 expansion condition: only the GQA+naive branch
    // leaves k_transposed alive (the freshly-allocated k_expanded sits
    // alongside it). The GQA+flash and MHA branches both std::move
    // k_transposed into k_expanded, so we must cache k_expanded there.
    // For GQA+flash, k_expanded retains the [bkv, seq, head_dim] layout
    // backward expects (the flash path never materialized the repeat).
    if(ctx.Training()==true)
    {
      SetCachedInput(std::move(flat_input));
      SetCachedQHeads(std::move(q_transposed));
      if(num_kv_heads!=num_heads && use_flash==false)
      {
        SetCachedKHeads(std::move(k_transposed));
        SetCachedVHeads(std::move(v_transposed));
      }
      else
      {
        SetCachedKHeads(std::move(k_expanded));
        SetCachedVHeads(std::move(v_expanded));
      }
      if(use_flash==true)
      {
        SetCachedLogsumexp(std::move(logsumexp));
        SetCachedOutput(std::move(context));
      }
      else
      {
        SetCachedAttn(std::move(attn));
      }
      SetCachedConcat(std::move(merged));
      SetCachedBatch(batch);
      SetCachedSeqLen(seq_len);
    }

    return output_flat;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::BackwardImpl(
                   const CAIF_DeviceTensor &grad_output,
                   CAIF_RunContext &ctx)
{
  try
  {
    if(CachedInput().IsEmpty()==true)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::Backward: "
                  "must call Forward with training=true first");
    }

    const bool has_prefix=ctx.HasPrefixLengths();

    const uint32_t batch=CachedBatch();
    const uint32_t seq_len=CachedSeqLen();
    const uint32_t dim=Config().dim;
    const uint32_t num_heads=Config().num_heads;
    const uint32_t num_kv_heads=Config().num_kv_heads;
    const uint32_t head_dim=Config().head_dim;
    const uint32_t bs=batch*seq_len;
    const uint32_t bh=batch*num_heads;
    const uint32_t bkv=batch*num_kv_heads;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;
    const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);

    // Step 1: Flatten grad_output to [bs, dim] — zero-copy view
    CAIF_DeviceTensor grad_out_flat=CAIF_DeviceTensor::WrapView(
                                     const_cast<void *>(grad_output.DeviceDataRaw()),
                                     {bs,dim},
                                     ctx.Stream(),
                                     grad_output.Dtype());

    // Step 2: Output projection gradients
    CAIF_DeviceTensor grad_concat;
    if(HasProjections()==true)
    {
      grad_concat=OProj().Backward(grad_out_flat,ctx);
    }
    else
    {
      // grad_concat = grad_out_flat @ W_o^T  -> [bs, qk_dim]
      grad_concat=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},ctx.Stream(),StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_out_flat,WO(),grad_concat,ctx,ComputeDtype());
      CAIF_DeviceTensor grad_w_o_delta=CAIF_DeviceTensor::Uninitialized({qk_dim,dim},
                                                                       ctx.Stream(),
                                                                       StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedConcat(),grad_out_flat,grad_w_o_delta,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWO(),grad_w_o_delta,GradWOMut());
    }

    CAIF_DeviceTensor grad_context=CAIF_DeviceTensor::Uninitialized({bh*seq_len*head_dim},
                                                                    ctx.Stream(),
                                                                    StorageDtype());
    launch_transpose_0213<StorageT>(grad_concat.template DevicePtr<StorageT>(),
                                    grad_context.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(num_heads),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());
    grad_context.Reshape({bh,seq_len,head_dim});

    CAIF_DeviceTensor k_expanded_storage;
    CAIF_DeviceTensor v_expanded_storage;
    if(num_kv_heads!=num_heads)
    {
      k_expanded_storage=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                          ctx.Stream(),
                                                          StorageDtype());
      v_expanded_storage=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                          ctx.Stream(),
                                                          StorageDtype());
      launch_gqa_repeat_kv<StorageT>(CachedKHeads().template DevicePtr<StorageT>(),
                                     k_expanded_storage.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
      launch_gqa_repeat_kv<StorageT>(CachedVHeads().template DevicePtr<StorageT>(),
                                     v_expanded_storage.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    }
    const CAIF_DeviceTensor *k_expanded_ptr=&CachedKHeads();
    const CAIF_DeviceTensor *v_expanded_ptr=&CachedVHeads();
    if(num_kv_heads!=num_heads)
    {
      k_expanded_ptr=&k_expanded_storage;
      v_expanded_ptr=&v_expanded_storage;
    }
    const CAIF_DeviceTensor &k_expanded=*k_expanded_ptr;
    const CAIF_DeviceTensor &v_expanded=*v_expanded_ptr;

    // Steps 5-9: Attention backward
    // Auto-select: use cuBLAS (naive) when attention matrix fits in memory,
    // flash backward for long sequences. cuBLAS wins for short sequences
    // due to flash backward's lower GPU occupancy in that regime.
    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_DeviceTensor grad_q_heads;
    CAIF_DeviceTensor grad_k_heads;
    CAIF_DeviceTensor grad_v_heads;

    const size_t attn_matrix_bytes=static_cast<size_t>(bh)*seq_len*seq_len*sizeof(float);
    size_t free_mem=0;
    size_t total_mem=0;
#ifdef USE_CAIF_CUDA
    cudaMemGetInfo(&free_mem,&total_mem);
#else
    (void)total_mem;
#endif
    const bool use_naive_backward=(attn_matrix_bytes*2<=free_mem);

    if(use_naive_backward==true)
    {
      // cuBLAS attention backward: recompute attention matrix, then use
      // optimized batched matmul for gradients. Faster than flash
      // backward for short sequences (the cutoff is roughly seq_len ≈ 1024
      // on consumer GPUs; the runtime selector above keys off free VRAM).
      //
      // Memory reuse: a single `scratch` buffer of shape [bh, seq, seq]
      // sequentially holds scores, then grad_attn, then grad_scores (softmax
      // backward writes in-place). Peak concurrent [bh,seq,seq] allocations:
      // 2 (scratch + attn) instead of 4, matching the `use_naive_backward`
      // free-mem check.

      // Phase A: scratch = scores = Q @ K^T (scaled, masked)
      CAIF_DeviceTensor scratch=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},
                                                                 ctx.Stream(),
                                                                 StorageDtype());
      CAIF_Ops::BatchedMatMulTransposeB(CachedQHeads(),k_expanded,scratch,
                                        static_cast<int>(seq_len),
                                        static_cast<int>(head_dim),
                                        static_cast<int>(seq_len),
                                        static_cast<int>(bh),
                                        ctx);
      CAIF_Ops::Scale(scratch,scale);

      ApplyScoreBias(scratch,batch,seq_len,ctx);

      if(has_prefix==true)
      {
        launch_prefix_mask_fill<StorageT>(scratch.template DevicePtr<StorageT>(),
                                          ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                          static_cast<int>(batch),
                                          static_cast<int>(Config().num_heads),
                                          static_cast<int>(seq_len),
                                          ctx.Stream().Handle());
      }
      else if(Config().causal==true)
      {
        launch_causal_mask_fill<StorageT>(scratch.template DevicePtr<StorageT>(),
                                          static_cast<int>(bh),
                                          static_cast<int>(seq_len),
                                          ctx.Stream().Handle());
      }
      CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized({bh,seq_len,seq_len},
                                                              ctx.Stream(),
                                                              scratch.Dtype());
      launch_attention_softmax<StorageT>(scratch.template DevicePtr<StorageT>(),
                                         attn.template DevicePtr<StorageT>(),
                                         static_cast<int>(bh*seq_len),
                                         static_cast<int>(seq_len),
                                         ctx.Stream().Handle());
      // scratch contents (scores) are dead from here on.

      // Phase B: scratch = grad_attn = grad_context @ V_expanded^T
      CAIF_Ops::BatchedMatMulTransposeB(grad_context,v_expanded,scratch,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(bh),
                                              ctx);

      grad_v_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                    ctx.Stream(),
                                                    StorageDtype());
      CAIF_Ops::BatchedMatMulTransposeA(attn,grad_context,grad_v_heads,
                                              static_cast<int>(seq_len),
                                              static_cast<int>(seq_len),
                                              static_cast<int>(head_dim),
                                              static_cast<int>(bh),
                                              ctx);

      // Phase C: scratch = grad_scores = softmax_backward(grad_attn, attn)
      // In-place: kernel reads dy[col] and writes dx[col] per-thread
      // per-col with no cross-thread sharing of cols, so dx==dy is safe.
      // Templated dispatch on StorageT — kernel has 3 instantiations.
      launch_attention_softmax_backward<StorageT>(scratch.template DevicePtr<StorageT>(),
                                                  attn.template DevicePtr<StorageT>(),
                                                  scratch.template DevicePtr<StorageT>(),
                                                  static_cast<int>(bh*seq_len),
                                                  static_cast<int>(seq_len),
                                                  ctx.Stream().Handle());

      // Causal or prefix-LM mask gradient
      if(has_prefix==true)
      {
        launch_prefix_mask_grad<StorageT>(scratch.template DevicePtr<StorageT>(),
                                          ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                          static_cast<int>(batch),
                                          static_cast<int>(Config().num_heads),
                                          static_cast<int>(seq_len),
                                          ctx.Stream().Handle());
      }
      else if(Config().causal==true)
      {
        launch_causal_mask_grad<StorageT>(scratch.template DevicePtr<StorageT>(),
                                          static_cast<int>(bh),
                                          static_cast<int>(seq_len),
                                          ctx.Stream().Handle());
      }

      // Subclass hook: accumulate score bias gradient before scale
      BackwardScoreBias(scratch,batch,seq_len,ctx);

      // Scale gradient
      CAIF_Ops::Scale(scratch,scale);

      grad_q_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                    ctx.Stream(),
                                                    StorageDtype());
      CAIF_Ops::BatchedMatMul(scratch,k_expanded,grad_q_heads,
                              static_cast<int>(seq_len),
                              static_cast<int>(seq_len),
                              static_cast<int>(head_dim),
                              static_cast<int>(bh),
                              ctx);

      grad_k_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                    ctx.Stream(),
                                                    StorageDtype());
      CAIF_Ops::BatchedMatMulTransposeA(scratch,CachedQHeads(),
                                        grad_k_heads,
                                        static_cast<int>(seq_len),
                                        static_cast<int>(seq_len),
                                        static_cast<int>(head_dim),
                                        static_cast<int>(bh),
                                        ctx);
    }
    else
    {
      grad_q_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                    ctx.Stream(),
                                                    StorageDtype());
      grad_k_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                    ctx.Stream(),
                                                    StorageDtype());
      grad_v_heads=CAIF_DeviceTensor::Uninitialized({bh,seq_len,head_dim},
                                                    ctx.Stream(),
                                                    StorageDtype());

      if(has_prefix==true)
      {
        launch_flash_attention_backward_prefix<StorageT>(
                                            CachedQHeads().template DevicePtr<StorageT>(),
                                            k_expanded.template DevicePtr<StorageT>(),
                                            v_expanded.template DevicePtr<StorageT>(),
                                            CachedOutput().template DevicePtr<StorageT>(),
                                            grad_context.template DevicePtr<StorageT>(),
                                            CachedLogsumexp().template DevicePtr<float>(),
                                            grad_q_heads.template DevicePtr<StorageT>(),
                                            grad_k_heads.template DevicePtr<StorageT>(),
                                            grad_v_heads.template DevicePtr<StorageT>(),
                                            ctx.PrefixLengths().DevicePtr<uint32_t>(),
                                            static_cast<int>(batch),
                                            static_cast<int>(Config().num_heads),
                                            static_cast<int>(seq_len),
                                            static_cast<int>(head_dim),
                                            scale,
                                            ctx.Stream().Handle());
      }
      else
      {
        int causal_flag=0;
        if(Config().causal==true)
        {
          causal_flag=1;
        }
        launch_flash_attention_backward<StorageT>(
                                            CachedQHeads().template DevicePtr<StorageT>(),
                                            k_expanded.template DevicePtr<StorageT>(),
                                            v_expanded.template DevicePtr<StorageT>(),
                                            CachedOutput().template DevicePtr<StorageT>(),
                                            grad_context.template DevicePtr<StorageT>(),
                                            CachedLogsumexp().template DevicePtr<float>(),
                                            grad_q_heads.template DevicePtr<StorageT>(),
                                            grad_k_heads.template DevicePtr<StorageT>(),
                                            grad_v_heads.template DevicePtr<StorageT>(),
                                            static_cast<int>(bh),
                                            static_cast<int>(seq_len),
                                            static_cast<int>(head_dim),
                                            scale,
                                            causal_flag,
                                            ctx.Stream().Handle());
      }
    }

    // Step 10: RoPE backward (inverse rotation on grad_Q and grad_K).
    // Mirror the forward dispatch: full or partial-rotary variant.
    if(Config().use_rope==true)
    {
      int rope_dim_eff=static_cast<int>(head_dim);
      if(Config().rope_dim!=0)
      {
        rope_dim_eff=Config().rope_dim;
      }
      if(rope_dim_eff==static_cast<int>(head_dim))
      {
        launch_rope_backward<StorageT>(grad_q_heads.template DevicePtr<StorageT>(),
                                       static_cast<int>(bh),
                                       static_cast<int>(seq_len),
                                       static_cast<int>(head_dim),
                                       Config().rope_base,
                                       Config().rope_style,
                                       ctx.Stream().Handle());
        launch_rope_backward<StorageT>(grad_k_heads.template DevicePtr<StorageT>(),
                                       static_cast<int>(bh),
                                       static_cast<int>(seq_len),
                                       static_cast<int>(head_dim),
                                       Config().rope_base,
                                       Config().rope_style,
                                       ctx.Stream().Handle());
      }
      else
      {
        launch_rope_backward_partial<StorageT>(grad_q_heads.template DevicePtr<StorageT>(),
                                               static_cast<int>(bh),
                                               static_cast<int>(seq_len),
                                               static_cast<int>(head_dim),
                                               rope_dim_eff,
                                               Config().rope_base,
                                               Config().rope_style,
                                               ctx.Stream().Handle());
        launch_rope_backward_partial<StorageT>(grad_k_heads.template DevicePtr<StorageT>(),
                                               static_cast<int>(bh),
                                               static_cast<int>(seq_len),
                                               static_cast<int>(head_dim),
                                               rope_dim_eff,
                                               Config().rope_base,
                                               Config().rope_style,
                                               ctx.Stream().Handle());
      }
    }

    // Step 11: GQA reduce (sum expanded grads back to kv_heads)
    CAIF_DeviceTensor grad_k_reduced;
    CAIF_DeviceTensor grad_v_reduced;
    if(num_kv_heads!=num_heads)
    {
      grad_k_reduced=CAIF_DeviceTensor::Uninitialized(
                       {bkv,seq_len,head_dim},ctx.Stream(),StorageDtype());
      grad_v_reduced=CAIF_DeviceTensor::Uninitialized(
                       {bkv,seq_len,head_dim},ctx.Stream(),StorageDtype());
      launch_gqa_reduce_kv<StorageT>(grad_k_heads.template DevicePtr<StorageT>(),
                                     grad_k_reduced.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
      launch_gqa_reduce_kv<StorageT>(grad_v_heads.template DevicePtr<StorageT>(),
                                     grad_v_reduced.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    }
    else
    {
      grad_k_reduced=std::move(grad_k_heads);
      grad_v_reduced=std::move(grad_v_heads);
    }

    CAIF_DeviceTensor grad_q_flat=CAIF_DeviceTensor::Uninitialized({bs*qk_dim},
                                                                   ctx.Stream(),
                                                                   StorageDtype());
    launch_transpose_0213<StorageT>(grad_q_heads.template DevicePtr<StorageT>(),
                                    grad_q_flat.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(num_heads),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());
    grad_q_flat.Reshape({bs,qk_dim});

    CAIF_DeviceTensor grad_k_flat=CAIF_DeviceTensor::Uninitialized({bs*kv_dim},
                                                                   ctx.Stream(),
                                                                   StorageDtype());
    CAIF_DeviceTensor grad_v_flat=CAIF_DeviceTensor::Uninitialized({bs*kv_dim},
                                                                   ctx.Stream(),
                                                                   StorageDtype());
    launch_transpose_0213<StorageT>(grad_k_reduced.template DevicePtr<StorageT>(),
                                    grad_k_flat.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(num_kv_heads),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());
    launch_transpose_0213<StorageT>(grad_v_reduced.template DevicePtr<StorageT>(),
                                    grad_v_flat.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(num_kv_heads),
                                    static_cast<int>(seq_len),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());
    grad_k_flat.Reshape({bs,kv_dim});
    grad_v_flat.Reshape({bs,kv_dim});

    if(HasQNormGamma()==true)
    {
      CAIF_DeviceTensor grad_q_pre_norm=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},
                                                                        ctx.Stream(),
                                                                        StorageDtype());
      CAIF_DeviceTensor grad_q_gamma_unused=CAIF_DeviceTensor::Uninitialized({qk_dim},
                                                                            ctx.Stream());
      launch_rmsnorm_backward<StorageT>(grad_q_flat.template DevicePtr<StorageT>(),
                                        CachedQPreNorm().template DevicePtr<StorageT>(),
                                        QNormGamma().template DevicePtr<float>(),
                                        CachedQRms().template DevicePtr<float>(),
                                        grad_q_pre_norm.template DevicePtr<StorageT>(),
                                        grad_q_gamma_unused.template DevicePtr<float>(),
                                        Config().qk_norm_eps,
                                        static_cast<int>(bs),
                                        static_cast<int>(qk_dim),
                                        ctx.Stream().Handle());
      grad_q_flat=std::move(grad_q_pre_norm);
    }
    if(HasKNormGamma()==true)
    {
      CAIF_DeviceTensor grad_k_pre_norm=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},
                                                                        ctx.Stream(),
                                                                        StorageDtype());
      CAIF_DeviceTensor grad_k_gamma_unused=CAIF_DeviceTensor::Uninitialized({kv_dim},
                                                                            ctx.Stream());
      launch_rmsnorm_backward<StorageT>(grad_k_flat.template DevicePtr<StorageT>(),
                                        CachedKPreNorm().template DevicePtr<StorageT>(),
                                        KNormGamma().template DevicePtr<float>(),
                                        CachedKRms().template DevicePtr<float>(),
                                        grad_k_pre_norm.template DevicePtr<StorageT>(),
                                        grad_k_gamma_unused.template DevicePtr<float>(),
                                        Config().qk_norm_eps,
                                        static_cast<int>(bs),
                                        static_cast<int>(kv_dim),
                                        ctx.Stream().Handle());
      grad_k_flat=std::move(grad_k_pre_norm);
    }

    CAIF_DeviceTensor grad_input;
    if(HasProjections()==true)
    {
      CAIF_DeviceTensor gi_q=QProj().Backward(grad_q_flat,ctx);
      CAIF_DeviceTensor gi_k=KProj().Backward(grad_k_flat,ctx);
      CAIF_DeviceTensor gi_v=VProj().Backward(grad_v_flat,ctx);
      grad_input=CAIF_DeviceTensor::Uninitialized({bs,dim},ctx.Stream(),StorageDtype());
      CAIF_Ops::Add(gi_q,gi_k,grad_input);
      CAIF_Ops::Add(grad_input,gi_v,grad_input);
    }
    else
    {
      CAIF_DeviceTensor grad_wq_delta=CAIF_DeviceTensor::Uninitialized({dim,qk_dim},
                                                                      ctx.Stream(),
                                                                      StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedInput(),grad_q_flat,grad_wq_delta,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWQ(),grad_wq_delta,GradWQMut());
      CAIF_DeviceTensor grad_wk_delta=CAIF_DeviceTensor::Uninitialized({dim,kv_dim},
                                                                      ctx.Stream(),
                                                                      StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedInput(),grad_k_flat,grad_wk_delta,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWK(),grad_wk_delta,GradWKMut());
      CAIF_DeviceTensor grad_wv_delta=CAIF_DeviceTensor::Uninitialized({dim,kv_dim},
                                                                      ctx.Stream(),
                                                                      StorageDtype());
      CAIF_Ops::MatMulTransposeA(CachedInput(),grad_v_flat,grad_wv_delta,ctx,ComputeDtype());
      CAIF_Ops::Add(GradWV(),grad_wv_delta,GradWVMut());
      CAIF_DeviceTensor gi_q=CAIF_DeviceTensor::Uninitialized({bs,dim},
                                                              ctx.Stream(),
                                                              StorageDtype());
      CAIF_DeviceTensor gi_k=CAIF_DeviceTensor::Uninitialized({bs,dim},
                                                              ctx.Stream(),
                                                              StorageDtype());
      CAIF_DeviceTensor gi_v=CAIF_DeviceTensor::Uninitialized({bs,dim},
                                                              ctx.Stream(),
                                                              StorageDtype());
      CAIF_Ops::MatMulTransposeB(grad_q_flat,WQ(),gi_q,ctx,ComputeDtype());
      CAIF_Ops::MatMulTransposeB(grad_k_flat,WK(),gi_k,ctx,ComputeDtype());
      CAIF_Ops::MatMulTransposeB(grad_v_flat,WV(),gi_v,ctx,ComputeDtype());
      grad_input=CAIF_DeviceTensor::Uninitialized({bs,dim},ctx.Stream(),StorageDtype());
      CAIF_Ops::Add(gi_q,gi_k,grad_input);
      CAIF_Ops::Add(grad_input,gi_v,grad_input);
    }

    grad_input.Reshape({batch,seq_len,dim});

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    if(HasProjections()==true)
    {
      QProj().ZeroGradients();
      KProj().ZeroGradients();
      VProj().ZeroGradients();
      OProj().ZeroGradients();
    }
    else
    {
      GradWQMut().FillZero();
      GradWKMut().FillZero();
      GradWVMut().FillZero();
      GradWOMut().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ParameterTensorCount()const
{
  try
  {
    if(HasProjections()==true)
    {
      return QProj().ParameterTensorCount()+
             KProj().ParameterTensorCount()+
             VProj().ParameterTensorCount()+
             OProj().ParameterTensorCount();
    }
    return g_caif_attention_weight_count;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(HasProjections()==true)
    {
      size_t offset=0;
      CAIF_DeviceLayer *projs[]={&QProj(),
                                &KProj(),
                                &VProj(),
                                &OProj()};
      for(size_t p=0;p<g_caif_attention_weight_count;++p)
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
    switch(static_cast<ParamSlot_e>(index))
    {
      case ParamSlot_e::WQ_e:
        SetWQkvDirty(true);
        return WQMut();
      case ParamSlot_e::WK_e:
        SetWQkvDirty(true);
        return WKMut();
      case ParamSlot_e::WV_e:
        SetWQkvDirty(true);
        return WVMut();
      case ParamSlot_e::WO_e:
        return WOMut();
    }
    THROW_CAIFE("DeviceMultiHeadAttention::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ParameterTensor(
                          size_t index)const
{
  try
  {
    if(HasProjections()==true)
    {
      size_t offset=0;
      const CAIF_DeviceLayer *projs[]={&QProj(),
                                      &KProj(),
                                      &VProj(),
                                      &OProj()};
      for(size_t p=0;p<g_caif_attention_weight_count;++p)
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
    switch(static_cast<ParamSlot_e>(index))
    {
      case ParamSlot_e::WQ_e:
        return WQ();
      case ParamSlot_e::WK_e:
        return WK();
      case ParamSlot_e::WV_e:
        return WV();
      case ParamSlot_e::WO_e:
        return WO();
    }
    THROW_CAIFE("DeviceMultiHeadAttention::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(HasProjections()==true)
    {
      size_t offset=0;
      CAIF_DeviceLayer *projs[]={&QProj(),
                                &KProj(),
                                &VProj(),
                                &OProj()};
      for(size_t p=0;p<g_caif_attention_weight_count;++p)
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
    switch(static_cast<ParamSlot_e>(index))
    {
      case ParamSlot_e::WQ_e:
        return GradWQMut();
      case ParamSlot_e::WK_e:
        return GradWKMut();
      case ParamSlot_e::WV_e:
        return GradWVMut();
      case ParamSlot_e::WO_e:
        return GradWOMut();
    }
    THROW_CAIFE("DeviceMultiHeadAttention::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::GradientTensor(
                          size_t index)const
{
  try
  {
    if(HasProjections()==true)
    {
      size_t offset=0;
      const CAIF_DeviceLayer *projs[]={&QProj(),
                                      &KProj(),
                                      &VProj(),
                                      &OProj()};
      for(size_t p=0;p<g_caif_attention_weight_count;++p)
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
    switch(static_cast<ParamSlot_e>(index))
    {
      case ParamSlot_e::WQ_e:
        return GradWQ();
      case ParamSlot_e::WK_e:
        return GradWK();
      case ParamSlot_e::WV_e:
        return GradWV();
      case ParamSlot_e::WO_e:
        return GradWO();
    }
    THROW_CAIFE("DeviceMultiHeadAttention::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::TotalParameterCount()const
{
  try
  {
    if(HasProjections()==true)
    {
      return QProj().TotalParameterCount()+
             KProj().TotalParameterCount()+
             VProj().TotalParameterCount()+
             OProj().TotalParameterCount();
    }
    return WQ().TotalElements()+
           WK().TotalElements()+
           WV().TotalElements()+
           WO().TotalElements();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string causal_str;
    if(Config().causal==true)
    {
      causal_str="true";
    }
    else
    {
      causal_str="false";
    }
    std::string desc="MultiHeadAttention(dim="+std::to_string(Config().dim)+
                     ",heads="+std::to_string(Config().num_heads)+
                     ",head_dim="+std::to_string(Config().head_dim)+
                     ",causal="+causal_str;
    if(Config().num_kv_heads!=Config().num_heads)
    {
      desc+=",kv_heads="+std::to_string(Config().num_kv_heads);
    }
    if(Config().use_rope==true)
    {
      desc+=",rope=true";
    }
    if(HasProjections()==true)
    {
      desc+=",projections";
    }
    desc+=")";
    return desc;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ParameterNames(
                           const std::string &prefix)const
{
  try
  {
    if(HasProjections()==true)
    {
      std::vector<std::string> names;
      std::vector<std::string> sub;
      sub=QProj().ParameterNames(prefix+g_caif_name_q_proj);
      names.insert(names.end(),sub.begin(),sub.end());
      sub=KProj().ParameterNames(prefix+g_caif_name_k_proj);
      names.insert(names.end(),sub.begin(),sub.end());
      sub=VProj().ParameterNames(prefix+g_caif_name_v_proj);
      names.insert(names.end(),sub.begin(),sub.end());
      sub=OProj().ParameterNames(prefix+g_caif_name_o_proj);
      names.insert(names.end(),sub.begin(),sub.end());
      return names;
    }
    std::vector<std::string> names;
    names.push_back(prefix+g_caif_name_q_proj+g_caif_name_weight);
    names.push_back(prefix+g_caif_name_k_proj+g_caif_name_weight);
    names.push_back(prefix+g_caif_name_v_proj+g_caif_name_weight);
    names.push_back(prefix+g_caif_name_o_proj+g_caif_name_weight);
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::FrozenTensorCount()const
{
  try
  {
    if(HasProjections()==false)
    {
      return 0;
    }
    return QProj().FrozenTensorCount()+
           KProj().FrozenTensorCount()+
           VProj().FrozenTensorCount()+
           OProj().FrozenTensorCount();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::FrozenTensorFP32(size_t index)const
{
  try
  {
    if(HasProjections()==false)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::FrozenTensorFP32: no frozen tensors");
    }
    size_t offset=0;
    const size_t qc=QProj().FrozenTensorCount();
    if(index<offset+qc)
    {
      return QProj().FrozenTensorFP32(index-offset);
    }
    offset+=qc;
    const size_t kc=KProj().FrozenTensorCount();
    if(index<offset+kc)
    {
      return KProj().FrozenTensorFP32(index-offset);
    }
    offset+=kc;
    const size_t vc=VProj().FrozenTensorCount();
    if(index<offset+vc)
    {
      return VProj().FrozenTensorFP32(index-offset);
    }
    offset+=vc;
    const size_t oc=OProj().FrozenTensorCount();
    if(index<offset+oc)
    {
      return OProj().FrozenTensorFP32(index-offset);
    }
    THROW_CAIFE("DeviceMultiHeadAttention::FrozenTensorFP32: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::FrozenTensorNames(
                           const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    if(HasProjections()==false)
    {
      return names;
    }
    std::vector<std::string> sub;
    sub=QProj().FrozenTensorNames(prefix+"q_proj.");
    names.insert(names.end(),sub.begin(),sub.end());
    sub=KProj().FrozenTensorNames(prefix+"k_proj.");
    names.insert(names.end(),sub.begin(),sub.end());
    sub=VProj().FrozenTensorNames(prefix+"v_proj.");
    names.insert(names.end(),sub.begin(),sub.end());
    sub=OProj().FrozenTensorNames(prefix+"o_proj.");
    names.insert(names.end(),sub.begin(),sub.end());
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    if(HasProjections()==true)
    {
      return;
    }

    std::mt19937 rng(seed);

    const uint32_t qk_dim=Config().num_heads*Config().head_dim;
    const uint32_t kv_dim=Config().num_kv_heads*Config().head_dim;
    const uint32_t dim=Config().dim;

    constexpr float g_xavier_six=6.0f;

    const float limit_q=std::sqrt(g_xavier_six/static_cast<float>(dim+qk_dim));
    std::uniform_real_distribution<float> dist_q(-limit_q,limit_q);
    std::vector<float> wq_data(dim*qk_dim);
    for(size_t i=0;i<wq_data.size();++i)
    {
      wq_data[i]=dist_q(rng);
    }
    WQMut().CopyFromHostFp32(wq_data.data(),wq_data.size());

    const float limit_kv=std::sqrt(g_xavier_six/static_cast<float>(dim+kv_dim));
    std::uniform_real_distribution<float> dist_kv(-limit_kv,limit_kv);
    std::vector<float> wk_data(dim*kv_dim);
    for(size_t i=0;i<wk_data.size();++i)
    {
      wk_data[i]=dist_kv(rng);
    }
    WKMut().CopyFromHostFp32(wk_data.data(),wk_data.size());

    std::vector<float> wv_data(dim*kv_dim);
    for(size_t i=0;i<wv_data.size();++i)
    {
      wv_data[i]=dist_kv(rng);
    }
    WVMut().CopyFromHostFp32(wv_data.data(),wv_data.size());

    const float limit_o=std::sqrt(g_xavier_six/static_cast<float>(qk_dim+dim));
    std::uniform_real_distribution<float> dist_o(-limit_o,limit_o);
    std::vector<float> wo_data(qk_dim*dim);
    for(size_t i=0;i<wo_data.size();++i)
    {
      wo_data[i]=dist_o(rng);
    }
    WOMut().CopyFromHostFp32(wo_data.data(),wo_data.size());

    SetWQkvDirty(true);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::BuildFusedWqkv()
{
  try
  {
    if(HasProjections()==true)
    {
      return;
    }
    if(WQ().IsAllocated()==false||
       WK().IsAllocated()==false||
       WV().IsAllocated()==false)
    {
      return;
    }
    if(WQ().Dtype()!=WK().Dtype()||WQ().Dtype()!=WV().Dtype())
    {
      THROW_CAIFE("BuildFusedWqkv: W_q/W_k/W_v dtype mismatch");
    }

    const uint32_t dim=Config().dim;
    const uint32_t qk_dim=Config().num_heads*Config().head_dim;
    const uint32_t kv_dim=Config().num_kv_heads*Config().head_dim;
    const uint32_t total_dim=qk_dim+2u*kv_dim;

    SetWQkv(CAIF_DeviceTensor::Uninitialized({dim,total_dim},
                                             WQMut().Stream(),
                                             WQ().Dtype()));

    const size_t elem_bytes=WQ().DtypeInfo().ElementSizeBytes();
    const size_t dst_pitch=static_cast<size_t>(total_dim)*elem_bytes;
    const size_t qk_bytes=static_cast<size_t>(qk_dim)*elem_bytes;
    const size_t kv_bytes=static_cast<size_t>(kv_dim)*elem_bytes;
    cudaStream_t stream=WQMut().Stream().Handle();
    uint8_t *dst=static_cast<uint8_t*>(WQkvMut().DeviceDataRaw());

    cudaError_t rc=cudaMemcpy2DAsync(dst,
                                     dst_pitch,
                                     WQMut().DeviceDataRaw(),
                                     qk_bytes,
                                     qk_bytes,
                                     dim,
                                     cudaMemcpyDeviceToDevice,
                                     stream);
    if(rc!=cudaSuccess)
    {
      THROW_CAIFE("BuildFusedWqkv: cudaMemcpy2DAsync(Q) failed");
    }
    rc=cudaMemcpy2DAsync(dst+qk_bytes,
                         dst_pitch,
                         WKMut().DeviceDataRaw(),
                         kv_bytes,
                         kv_bytes,
                         dim,
                         cudaMemcpyDeviceToDevice,
                         stream);
    if(rc!=cudaSuccess)
    {
      THROW_CAIFE("BuildFusedWqkv: cudaMemcpy2DAsync(K) failed");
    }
    rc=cudaMemcpy2DAsync(dst+qk_bytes+kv_bytes,
                         dst_pitch,
                         WVMut().DeviceDataRaw(),
                         kv_bytes,
                         kv_bytes,
                         dim,
                         cudaMemcpyDeviceToDevice,
                         stream);
    if(rc!=cudaSuccess)
    {
      THROW_CAIFE("BuildFusedWqkv: cudaMemcpy2DAsync(V) failed");
    }

    SetWQkvDirty(false);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::EnableKVCache(uint32_t batch_size,
                                                  uint32_t max_seq_len)
{
  try
  {
    const uint32_t num_kv_heads=Config().num_kv_heads;
    const uint32_t head_dim=Config().head_dim;
    const uint32_t bkv=batch_size*num_kv_heads;

    // Allocate cache tensors: [batch*num_kv_heads, max_seq_len, head_dim]
    // This matches the attention layout for efficient extraction
    SetKVCacheK(CAIF_DeviceTensor::Zeros({bkv,max_seq_len,head_dim},
                                         Stream(),
                                         StorageDtype()));
    SetKVCacheV(CAIF_DeviceTensor::Zeros({bkv,max_seq_len,head_dim},
                                         Stream(),
                                         StorageDtype()));
    SetKVCacheLen(0);
    SetKVCacheMaxLen(max_seq_len);
    SetKVCacheBatch(batch_size);
    SetKVCacheEnabled(true);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::DisableKVCache()
{
  try
  {
    SetKVCacheK(CAIF_DeviceTensor());
    SetKVCacheV(CAIF_DeviceTensor());
    SetKVCacheLen(0);
    SetKVCacheMaxLen(0);
    SetKVCacheBatch(0);
    SetKVCacheEnabled(false);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ResetKVCache()
{
  try
  {
    if(IsKVCacheEnabled()==false)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ResetKVCache: cache not enabled");
    }
    SetKVCacheLen(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::ForwardCached(
                   const CAIF_DeviceTensor &input,
                   CAIF_RunContext &ctx)
{
  try
  {
    if(IsKVCacheEnabled()==false)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: cache not enabled");
    }

    // Validate input shape [batch, seq_len, dim]
    const auto &shape=input.Shape();
    if(shape.size()!=3)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: input must be 3D");
    }
    if(shape[2]!=Config().dim)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: last dim must match config dim");
    }
    if(shape[0]!=KVCacheBatch())
    {
      THROW_CAIFE("DeviceMultiHeadAttention::ForwardCached: batch size must match cache");
    }

    const uint32_t batch=shape[0];
    const uint32_t new_len=shape[1];
    const uint32_t dim=Config().dim;
    const uint32_t num_heads=Config().num_heads;
    const uint32_t num_kv_heads=Config().num_kv_heads;
    const uint32_t head_dim=Config().head_dim;
    const uint32_t bs=batch*new_len;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;

    if(KVCacheLength()+new_len>KVCacheMaxLen())
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

    if(HasProjections()==true)
    {
      q_proj=QProj().Forward(flat_input,ctx);
      k_proj=KProj().Forward(flat_input,ctx);
      v_proj=VProj().Forward(flat_input,ctx);
      if(QBias().IsEmpty()==false)
      {
        CAIF_Ops::BiasAdd(q_proj,QBias(),q_proj);
      }
      if(KBias().IsEmpty()==false)
      {
        CAIF_Ops::BiasAdd(k_proj,KBias(),k_proj);
      }
      if(VBias().IsEmpty()==false)
      {
        CAIF_Ops::BiasAdd(v_proj,VBias(),v_proj);
      }
    }
    else
    {
      q_proj=CAIF_DeviceTensor::Uninitialized({bs,qk_dim},ctx.Stream(),StorageDtype());
      k_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},ctx.Stream(),StorageDtype());
      v_proj=CAIF_DeviceTensor::Uninitialized({bs,kv_dim},ctx.Stream(),StorageDtype());
      CAIF_Ops::MatMul(flat_input,WQ(),q_proj,ctx,ComputeDtype());
      CAIF_Ops::MatMul(flat_input,WK(),k_proj,ctx,ComputeDtype());
      CAIF_Ops::MatMul(flat_input,WV(),v_proj,ctx,ComputeDtype());
    }

    const uint32_t bkv=batch*num_kv_heads;
    CAIF_DeviceTensor k_transposed=CAIF_DeviceTensor::Uninitialized({bkv,new_len,head_dim},
                                                                   ctx.Stream(),
                                                                   StorageDtype());
    CAIF_DeviceTensor v_transposed=CAIF_DeviceTensor::Uninitialized({bkv,new_len,head_dim},
                                                                   ctx.Stream(),
                                                                   StorageDtype());

    launch_transpose_0213<StorageT>(k_proj.template DevicePtr<StorageT>(),
                                    k_transposed.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(new_len),
                                    static_cast<int>(num_kv_heads),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());
    launch_transpose_0213<StorageT>(v_proj.template DevicePtr<StorageT>(),
                                    v_transposed.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(new_len),
                                    static_cast<int>(num_kv_heads),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());

    const uint32_t old_cache_len=KVCacheLength();

    if(Config().use_rope==true)
    {
      launch_rope_forward_offset<StorageT>(k_transposed.template DevicePtr<StorageT>(),
                                           static_cast<int>(bkv),
                                           static_cast<int>(new_len),
                                           static_cast<int>(head_dim),
                                           Config().rope_base,
                                           static_cast<int>(old_cache_len),
                                           Config().rope_style,
                                           ctx.Stream().Handle());
    }

    launch_kv_cache_append_transposed<StorageT>(k_transposed.template DevicePtr<StorageT>(),
                                                KVCacheKMut().template DevicePtr<StorageT>(),
                                                static_cast<int>(bkv),
                                                static_cast<int>(new_len),
                                                static_cast<int>(KVCacheLength()),
                                                static_cast<int>(KVCacheMaxLen()),
                                                static_cast<int>(head_dim),
                                                ctx.Stream().Handle());
    launch_kv_cache_append_transposed<StorageT>(v_transposed.template DevicePtr<StorageT>(),
                                                KVCacheVMut().template DevicePtr<StorageT>(),
                                                static_cast<int>(bkv),
                                                static_cast<int>(new_len),
                                                static_cast<int>(KVCacheLength()),
                                                static_cast<int>(KVCacheMaxLen()),
                                                static_cast<int>(head_dim),
                                                ctx.Stream().Handle());

    const uint32_t total_len=KVCacheLength()+new_len;
    SetKVCacheLen(total_len);

    const uint32_t bh=batch*num_heads;
    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized({bh,new_len,head_dim},
                                                                   ctx.Stream(),
                                                                   StorageDtype());
    launch_transpose_0213<StorageT>(q_proj.template DevicePtr<StorageT>(),
                                    q_transposed.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(new_len),
                                    static_cast<int>(num_heads),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());

    if(Config().use_rope==true)
    {
      launch_rope_forward_offset<StorageT>(q_transposed.template DevicePtr<StorageT>(),
                                           static_cast<int>(bh),
                                           static_cast<int>(new_len),
                                           static_cast<int>(head_dim),
                                           Config().rope_base,
                                           static_cast<int>(old_cache_len),
                                           Config().rope_style,
                                           ctx.Stream().Handle());
    }

    // Step 8: Extract K/V from cache
    // Cache is [bkv, max_seq_len, head_dim], we need [bkv, total_len, head_dim]
    // Copy valid portion from each row
    CAIF_DeviceTensor k_full=CAIF_DeviceTensor::Uninitialized(
                              {bkv,total_len,head_dim},ctx.Stream(),StorageDtype());
    CAIF_DeviceTensor v_full=CAIF_DeviceTensor::Uninitialized(
                              {bkv,total_len,head_dim},ctx.Stream(),StorageDtype());

    // Copy valid portion from cache to k_full/v_full
    // For each bkv row, copy [0:total_len] from cache[bkv, :, :].
    // Cache + destination both carry StorageT (allocated above with
    // StorageDtype()) so the bulk memcpy uses sizeof(StorageT).
    StorageT *k_full_ptr=k_full.template DevicePtr<StorageT>();
    StorageT *v_full_ptr=v_full.template DevicePtr<StorageT>();
    const StorageT *cache_k_ptr=KVCacheK().template DevicePtr<StorageT>();
    const StorageT *cache_v_ptr=KVCacheV().template DevicePtr<StorageT>();
    for(uint32_t row=0;row<bkv;++row)
    {
      const size_t src_offset=row*KVCacheMaxLen()*head_dim;
      const size_t dst_offset=row*total_len*head_dim;
      const size_t copy_size=total_len*head_dim*sizeof(StorageT);
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(k_full_ptr+dst_offset,
                      cache_k_ptr+src_offset,
                      copy_size,cudaMemcpyDeviceToDevice,ctx.Stream().Handle());
      cudaMemcpyAsync(v_full_ptr+dst_offset,
                      cache_v_ptr+src_offset,
                      copy_size,cudaMemcpyDeviceToDevice,ctx.Stream().Handle());
#else
      std::memcpy(k_full_ptr+dst_offset,
                  cache_k_ptr+src_offset,
                  copy_size);
      std::memcpy(v_full_ptr+dst_offset,
                  cache_v_ptr+src_offset,
                  copy_size);
#endif
    }

    // Step 9: GQA expand K/V if needed
    CAIF_DeviceTensor k_expanded;
    CAIF_DeviceTensor v_expanded;
    if(num_kv_heads!=num_heads)
    {
      const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);
      k_expanded=CAIF_DeviceTensor::Uninitialized({bh,total_len,head_dim},
                                                  ctx.Stream(),
                                                  StorageDtype());
      v_expanded=CAIF_DeviceTensor::Uninitialized({bh,total_len,head_dim},
                                                  ctx.Stream(),
                                                  StorageDtype());
      launch_gqa_repeat_kv<StorageT>(k_full.template DevicePtr<StorageT>(),
                                     k_expanded.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(total_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
      launch_gqa_repeat_kv<StorageT>(v_full.template DevicePtr<StorageT>(),
                                     v_expanded.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     repeat_factor,
                                     static_cast<int>(total_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    }
    else
    {
      k_expanded=std::move(k_full);
      v_expanded=std::move(v_full);
    }

    CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},
                                                              ctx.Stream(),
                                                              StorageDtype());
    CAIF_Ops::BatchedMatMulTransposeB(q_transposed,k_expanded,scores,
                                      static_cast<int>(new_len),
                                      static_cast<int>(head_dim),
                                      static_cast<int>(total_len),
                                      static_cast<int>(bh),
                                      ctx);

    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_Ops::Scale(scores,scale);

    if(Config().causal==true)
    {
      const uint32_t offset=total_len-new_len;
      launch_causal_mask_fill_offset<StorageT>(scores.template DevicePtr<StorageT>(),
                                               static_cast<int>(bh),
                                               static_cast<int>(new_len),
                                               static_cast<int>(total_len),
                                               static_cast<int>(offset),
                                               ctx.Stream().Handle());
    }

    CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized({bh,new_len,total_len},
                                                            ctx.Stream(),
                                                            scores.Dtype());
    launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                       attn.template DevicePtr<StorageT>(),
                                       static_cast<int>(bh*new_len),
                                       static_cast<int>(total_len),
                                       ctx.Stream().Handle());

    CAIF_DeviceTensor context=CAIF_DeviceTensor::Uninitialized({bh,new_len,head_dim},
                                                               ctx.Stream(),
                                                               StorageDtype());
    CAIF_Ops::BatchedMatMul(attn,v_expanded,context,
                            static_cast<int>(new_len),
                            static_cast<int>(total_len),
                            static_cast<int>(head_dim),
                            static_cast<int>(bh),
                            ctx);

    CAIF_DeviceTensor merged=CAIF_DeviceTensor::Uninitialized({bs*qk_dim},
                                                              ctx.Stream(),
                                                              StorageDtype());
    launch_transpose_0213<StorageT>(context.template DevicePtr<StorageT>(),
                                    merged.template DevicePtr<StorageT>(),
                                    static_cast<int>(batch),
                                    static_cast<int>(num_heads),
                                    static_cast<int>(new_len),
                                    static_cast<int>(head_dim),
                                    ctx.Stream().Handle());
    merged.Reshape({bs,qk_dim});

    CAIF_DeviceTensor output_flat;
    if(HasProjections()==true)
    {
      output_flat=OProj().Forward(merged,ctx);
    }
    else
    {
      output_flat=CAIF_DeviceTensor::Uninitialized({bs,dim},ctx.Stream(),StorageDtype());
      CAIF_Ops::MatMul(merged,WO(),output_flat,ctx,ComputeDtype());
    }

    output_flat.Reshape({batch,new_len,dim});

    return output_flat;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadWQ(CAIF_DeviceTensor &&w_q)
{
  try
  {
    if(HasProjections()==true)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWQ: not valid when using sub-projections");
    }
    const uint32_t expected_out=Config().num_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=w_q.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().dim ||
       shape[1]!=expected_out)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWQ: shape mismatch, expected [dim, num_heads*head_dim]");
    }
    _w_q=std::move(w_q);
    BuildFusedWqkv();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadWK(CAIF_DeviceTensor &&w_k)
{
  try
  {
    if(HasProjections()==true)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWK: not valid when using sub-projections");
    }
    const uint32_t expected_out=Config().num_kv_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=w_k.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().dim ||
       shape[1]!=expected_out)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWK: shape mismatch, expected [dim, num_kv_heads*head_dim]");
    }
    _w_k=std::move(w_k);
    BuildFusedWqkv();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadWV(CAIF_DeviceTensor &&w_v)
{
  try
  {
    if(HasProjections()==true)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWV: not valid when using sub-projections");
    }
    const uint32_t expected_out=Config().num_kv_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=w_v.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().dim ||
       shape[1]!=expected_out)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWV: shape mismatch, expected [dim, num_kv_heads*head_dim]");
    }
    _w_v=std::move(w_v);
    BuildFusedWqkv();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadWO(CAIF_DeviceTensor &&w_o)
{
  try
  {
    if(HasProjections()==true)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWO: not valid when using sub-projections");
    }
    const uint32_t expected_in=Config().num_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=w_o.Shape();
    if(shape.size()!=2 ||
       shape[0]!=expected_in ||
       shape[1]!=Config().dim)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadWO: shape mismatch, expected [num_heads*head_dim, dim]");
    }
    _w_o=std::move(w_o);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadQBias(CAIF_DeviceTensor &&q_bias)
{
  try
  {
    const uint32_t expected=Config().num_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=q_bias.Shape();
    if(shape.size()!=1 || shape[0]!=expected)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadQBias: shape mismatch, expected [num_heads*head_dim]");
    }
    _projections.q_bias=std::move(q_bias);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadKBias(CAIF_DeviceTensor &&k_bias)
{
  try
  {
    const uint32_t expected=Config().num_kv_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=k_bias.Shape();
    if(shape.size()!=1 || shape[0]!=expected)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadKBias: shape mismatch, expected [num_kv_heads*head_dim]");
    }
    _projections.k_bias=std::move(k_bias);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadVBias(CAIF_DeviceTensor &&v_bias)
{
  try
  {
    const uint32_t expected=Config().num_kv_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=v_bias.Shape();
    if(shape.size()!=1 || shape[0]!=expected)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadVBias: shape mismatch, expected [num_kv_heads*head_dim]");
    }
    _projections.v_bias=std::move(v_bias);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadQNormGamma(CAIF_DeviceTensor &&q_norm_gamma)
{
  try
  {
    const uint32_t expected=Config().num_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=q_norm_gamma.Shape();
    if(shape.size()!=1 || shape[0]!=expected)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadQNormGamma: shape mismatch, expected [num_heads*head_dim]");
    }
    if(q_norm_gamma.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadQNormGamma: gamma must"
                  " be fp32 (rmsnorm kernel reads it via float*)");
    }
    _projections.q_norm_gamma=std::move(q_norm_gamma);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::LoadKNormGamma(CAIF_DeviceTensor &&k_norm_gamma)
{
  try
  {
    const uint32_t expected=Config().num_kv_heads*Config().head_dim;
    const std::vector<uint32_t> &shape=k_norm_gamma.Shape();
    if(shape.size()!=1 || shape[0]!=expected)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadKNormGamma: shape mismatch, expected [num_kv_heads*head_dim]");
    }
    if(k_norm_gamma.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("DeviceMultiHeadAttention::LoadKNormGamma: gamma must"
                  " be fp32 (rmsnorm kernel reads it via float*)");
    }
    _projections.k_norm_gamma=std::move(k_norm_gamma);
  }
  CAIF_CATCH_BLOCK()
}
// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceMultiHeadAttention<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceMultiHeadAttention<float,__half>;
template class CAIF_DeviceMultiHeadAttention<float,__nv_bfloat16>;
template class CAIF_DeviceMultiHeadAttention<__half,float>;
template class CAIF_DeviceMultiHeadAttention<__half,__half>;
template class CAIF_DeviceMultiHeadAttention<__half,__nv_bfloat16>;
template class CAIF_DeviceMultiHeadAttention<__nv_bfloat16,float>;
template class CAIF_DeviceMultiHeadAttention<__nv_bfloat16,__half>;
template class CAIF_DeviceMultiHeadAttention<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
