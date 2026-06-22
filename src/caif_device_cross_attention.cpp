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

#include "caif_device_cross_attention.h"
#include "caif_ops.h"
#include "caif_cuda_kernels_attention_support.cuh"
#include "caif_cuda_kernels_flash_cross.cuh"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <cmath>
#include <random>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceCrossAttention<ComputeT,StorageT>::CAIF_DeviceCrossAttention(
                           const CAIF_DeviceCrossAttentionConfig &config,
                           CAIF_CudaStream &stream):
                           CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                           _config(config),
                           _w_q(),
                           _w_k(),
                           _w_v(),
                           _w_o(),
                           _grad_w_q(),
                           _grad_w_k(),
                           _grad_w_v(),
                           _grad_w_o(),
                           _cached_decoder_input(),
                           _cached_encoder_input(),
                           _cached_q_heads(),
                           _cached_k_heads(),
                           _cached_v_heads(),
                           _cached_concat(),
                           _cached_logsumexp(),
                           _cached_output(),
                           _cached_batch(0),
                           _cached_dec_seq_len(0),
                           _cached_enc_seq_len(0),
                           _use_flash_attention(true),
                           _cached_use_flash(false)
{
  try
  {
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention: dim must be > 0");
    }
    if(config.KvInputDim()==0)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention: kv_input_dim must be > 0");
    }
    if(config.NumHeads()==0)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention: num_heads must be > 0");
    }
    if(config.NumKvHeads()==0)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention: num_kv_heads must be > 0");
    }
    if(config.Dim()%config.NumHeads()!=0)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention: dim must be divisible by num_heads");
    }
    if(config.NumHeads()%config.NumKvHeads()!=0)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention: num_heads must be divisible by num_kv_heads");
    }
    const uint32_t qk_dim=Config().NumHeads()*Config().HeadDim();
    const uint32_t kv_dim=Config().NumKvHeads()*Config().HeadDim();
    const uint32_t kv_input_dim=Config().KvInputDim();
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    // W_Q / W_O are sized by the decoder-stream width (Config().Dim());
    // W_K / W_V are sized by the encoder-output width (kv_input_dim),
    // which differs from dim when a frozen pretrained encoder of one
    // hidden size feeds a decoder of another.
    SetWQ(CAIF_DeviceTensor::Uninitialized({Config().Dim(),qk_dim},stream,sdt));
    SetWK(CAIF_DeviceTensor::Uninitialized({kv_input_dim,kv_dim},stream,sdt));
    SetWV(CAIF_DeviceTensor::Uninitialized({kv_input_dim,kv_dim},stream,sdt));
    SetWO(CAIF_DeviceTensor::Uninitialized({qk_dim,Config().Dim()},stream,sdt));

    SetGradWQ(CAIF_DeviceTensor::Zeros({Config().Dim(),qk_dim},stream,sdt));
    SetGradWK(CAIF_DeviceTensor::Zeros({kv_input_dim,kv_dim},stream,sdt));
    SetGradWV(CAIF_DeviceTensor::Zeros({kv_input_dim,kv_dim},stream,sdt));
    SetGradWO(CAIF_DeviceTensor::Zeros({qk_dim,Config().Dim()},stream,sdt));

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceCrossAttention<ComputeT,StorageT>::CAIF_DeviceCrossAttention(
                           CAIF_DeviceCrossAttention &&other):
                           CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                           _config(other._config),
                           _w_q(std::move(other._w_q)),
                           _w_k(std::move(other._w_k)),
                           _w_v(std::move(other._w_v)),
                           _w_o(std::move(other._w_o)),
                           _grad_w_q(std::move(other._grad_w_q)),
                           _grad_w_k(std::move(other._grad_w_k)),
                           _grad_w_v(std::move(other._grad_w_v)),
                           _grad_w_o(std::move(other._grad_w_o)),
                           _cached_decoder_input(std::move(other._cached_decoder_input)),
                           _cached_encoder_input(std::move(other._cached_encoder_input)),
                           _cached_q_heads(std::move(other._cached_q_heads)),
                           _cached_k_heads(std::move(other._cached_k_heads)),
                           _cached_v_heads(std::move(other._cached_v_heads)),
                           _cached_concat(std::move(other._cached_concat)),
                           _cached_logsumexp(std::move(other._cached_logsumexp)),
                           _cached_output(std::move(other._cached_output)),
                           _cached_batch(other._cached_batch),
                           _cached_dec_seq_len(other._cached_dec_seq_len),
                           _cached_enc_seq_len(other._cached_enc_seq_len),
                           _use_flash_attention(other._use_flash_attention),
                           _cached_use_flash(other._cached_use_flash)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceCrossAttention<ComputeT,StorageT> &
CAIF_DeviceCrossAttention<ComputeT,StorageT>::operator=(CAIF_DeviceCrossAttention &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      SetConfig(other.Config());
      SetWQ(std::move(other.WQ()));
      SetWK(std::move(other.WK()));
      SetWV(std::move(other.WV()));
      SetWO(std::move(other.WO()));
      SetGradWQ(std::move(other.GradWQ()));
      SetGradWK(std::move(other.GradWK()));
      SetGradWV(std::move(other.GradWV()));
      SetGradWO(std::move(other.GradWO()));
      SetCachedDecoderInput(std::move(other.CachedDecoderInput()));
      SetCachedEncoderInput(std::move(other.CachedEncoderInput()));
      SetCachedQHeads(std::move(other.CachedQHeads()));
      SetCachedKHeads(std::move(other.CachedKHeads()));
      SetCachedVHeads(std::move(other.CachedVHeads()));
      SetCachedConcat(std::move(other.CachedConcat()));
      SetCachedLogsumexp(std::move(other.CachedLogsumexp()));
      SetCachedOutput(std::move(other.CachedOutput()));
      SetCachedBatch(other.CachedBatch());
      SetCachedDecSeqLen(other.CachedDecSeqLen());
      SetCachedEncSeqLen(other.CachedEncSeqLen());
      SetUseFlashAttention(other.UseFlashAttention());
      SetCachedUseFlash(other.CachedUseFlash());
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceCrossAttention<ComputeT,StorageT>::ForwardCross(
                           const CAIF_DeviceTensor &decoder_input,
                           const CAIF_DeviceTensor &encoder_output,
                           CAIF_RunContext &ctx)
{
  try
  {
    const std::vector<uint32_t> &dec_shape=decoder_input.Shape();
    if(dec_shape.size()!=3)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::ForwardCross: "
                  "decoder_input must be 3D [batch,seq_len,dim]");
    }
    if(dec_shape[2]!=Config().Dim())
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::ForwardCross: "
                  "decoder_input last dim must match config dim");
    }
    const std::vector<uint32_t> &enc_shape=encoder_output.Shape();
    if(enc_shape.size()!=3)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::ForwardCross: "
                  "encoder_output must be 3D [batch,seq_len,kv_input_dim]");
    }
    if(enc_shape[0]!=dec_shape[0])
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::ForwardCross: batch mismatch");
    }
    if(enc_shape[2]!=Config().KvInputDim())
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::ForwardCross: "
                  "encoder last dim must match config kv_input_dim");
    }
    AssertInputDtype(decoder_input);
    AssertInputDtype(encoder_output);

    const uint32_t batch=dec_shape[0];
    const uint32_t dec_seq_len=dec_shape[1];
    const uint32_t enc_seq_len=enc_shape[1];
    const uint32_t dim=Config().Dim();
    const uint32_t kv_input_dim=Config().KvInputDim();
    const uint32_t num_heads=Config().NumHeads();
    const uint32_t num_kv_heads=Config().NumKvHeads();
    const uint32_t head_dim=Config().HeadDim();
    const uint32_t bs_dec=batch*dec_seq_len;
    const uint32_t bs_enc=batch*enc_seq_len;
    const uint32_t bh=batch*num_heads;
    const uint32_t bkv=batch*num_kv_heads;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    CAIF_DeviceTensor flat_dec=decoder_input.Clone();
    flat_dec.Reshape({bs_dec,dim});
    CAIF_DeviceTensor flat_enc=encoder_output.Clone();
    flat_enc.Reshape({bs_enc,kv_input_dim});

    CAIF_DeviceTensor q_proj=CAIF_DeviceTensor::Uninitialized({bs_dec,qk_dim},ctx.Stream(),sdt);
    CAIF_DeviceTensor k_proj=CAIF_DeviceTensor::Uninitialized({bs_enc,kv_dim},ctx.Stream(),sdt);
    CAIF_DeviceTensor v_proj=CAIF_DeviceTensor::Uninitialized({bs_enc,kv_dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMul(flat_dec,WQ(),q_proj,ctx,cdt);
    CAIF_Ops::MatMul(flat_enc,WK(),k_proj,ctx,cdt);
    CAIF_Ops::MatMul(flat_enc,WV(),v_proj,ctx,cdt);

    CAIF_DeviceTensor q_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bh*dec_seq_len*head_dim},ctx.Stream(),sdt);
    launch_transpose_0213<StorageT>(q_proj.template DevicePtr<StorageT>(),
                                     q_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(dec_seq_len),
                                     static_cast<int>(num_heads),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    q_transposed.Reshape({bh,dec_seq_len,head_dim});

    CAIF_DeviceTensor k_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv*enc_seq_len*head_dim},ctx.Stream(),sdt);
    CAIF_DeviceTensor v_transposed=CAIF_DeviceTensor::Uninitialized(
                                    {bkv*enc_seq_len*head_dim},ctx.Stream(),sdt);
    launch_transpose_0213<StorageT>(k_proj.template DevicePtr<StorageT>(),
                                     k_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(enc_seq_len),
                                     static_cast<int>(num_kv_heads),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    launch_transpose_0213<StorageT>(v_proj.template DevicePtr<StorageT>(),
                                     v_transposed.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(enc_seq_len),
                                     static_cast<int>(num_kv_heads),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    k_transposed.Reshape({bkv,enc_seq_len,head_dim});
    v_transposed.Reshape({bkv,enc_seq_len,head_dim});

    CAIF_DeviceTensor k_expanded;
    CAIF_DeviceTensor v_expanded;
    if(num_kv_heads!=num_heads)
    {
      const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);
      k_expanded=CAIF_DeviceTensor::Uninitialized({bh,enc_seq_len,head_dim},ctx.Stream(),sdt);
      v_expanded=CAIF_DeviceTensor::Uninitialized({bh,enc_seq_len,head_dim},ctx.Stream(),sdt);
      launch_gqa_repeat_kv<StorageT>(k_transposed.template DevicePtr<StorageT>(),
                                      k_expanded.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(num_kv_heads),
                                      repeat_factor,
                                      static_cast<int>(enc_seq_len),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
      launch_gqa_repeat_kv<StorageT>(v_transposed.template DevicePtr<StorageT>(),
                                      v_expanded.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(num_kv_heads),
                                      repeat_factor,
                                      static_cast<int>(enc_seq_len),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
    }
    else
    {
      k_expanded=std::move(k_transposed);
      v_expanded=std::move(v_transposed);
    }

    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_DeviceTensor context=CAIF_DeviceTensor::Uninitialized(
                               {bh,dec_seq_len,head_dim},ctx.Stream(),sdt);
    CAIF_DeviceTensor logsumexp;

    const size_t attn_matrix_bytes=static_cast<size_t>(bh)*dec_seq_len*
                                   enc_seq_len*sizeof(float);
    size_t free_mem=0;
    size_t total_mem=0;
#ifdef USE_CAIF_CUDA
    cudaMemGetInfo(&free_mem,&total_mem);
#else
    (void)total_mem;
#endif
    const bool use_flash=(attn_matrix_bytes*2>free_mem);

    if(use_flash==true)
    {
      logsumexp=CAIF_DeviceTensor::Uninitialized({bh,dec_seq_len},ctx.Stream());

      launch_flash_attention_forward_cross<StorageT>(
                  q_transposed.template DevicePtr<StorageT>(),
                  k_expanded.template DevicePtr<StorageT>(),
                  v_expanded.template DevicePtr<StorageT>(),
                  context.template DevicePtr<StorageT>(),
                  logsumexp.DevicePtr<float>(),
                  static_cast<int>(bh),
                  static_cast<int>(dec_seq_len),
                  static_cast<int>(enc_seq_len),
                  static_cast<int>(head_dim),
                  scale,
                  ctx.Stream().Handle());
    }
    else
    {
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized(
                                {bh,dec_seq_len,enc_seq_len},ctx.Stream(),sdt);
      CAIF_Ops::BatchedMatMulTransposeB(q_transposed,k_expanded,scores,
                                        static_cast<int>(dec_seq_len),
                                        static_cast<int>(head_dim),
                                        static_cast<int>(enc_seq_len),
                                        static_cast<int>(bh),
                                        ctx);

      CAIF_Ops::Scale(scores,scale);

      CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized(
                              {bh,dec_seq_len,enc_seq_len},ctx.Stream(),sdt);
      launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                          attn.template DevicePtr<StorageT>(),
                                          static_cast<int>(bh*dec_seq_len),
                                          static_cast<int>(enc_seq_len),
                                          ctx.Stream().Handle());

      CAIF_Ops::BatchedMatMul(attn,v_expanded,context,
                              static_cast<int>(dec_seq_len),
                              static_cast<int>(enc_seq_len),
                              static_cast<int>(head_dim),
                              static_cast<int>(bh),
                              ctx);
    }

    CAIF_DeviceTensor merged=CAIF_DeviceTensor::Uninitialized(
                              {bs_dec*qk_dim},ctx.Stream(),sdt);
    launch_transpose_0213<StorageT>(context.template DevicePtr<StorageT>(),
                                     merged.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_heads),
                                     static_cast<int>(dec_seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    merged.Reshape({bs_dec,qk_dim});

    CAIF_DeviceTensor output_flat=CAIF_DeviceTensor::Uninitialized(
                                   {bs_dec,dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMul(merged,WO(),output_flat,ctx,cdt);
    output_flat.Reshape({batch,dec_seq_len,dim});

    if(ctx.Training()==true)
    {
      SetCachedDecoderInput(std::move(flat_dec));
      SetCachedEncoderInput(std::move(flat_enc));
      SetCachedQHeads(std::move(q_transposed));
      if(num_kv_heads!=num_heads)
      {
        SetCachedKHeads(std::move(k_transposed));
        SetCachedVHeads(std::move(v_transposed));
      }
      else
      {
        SetCachedKHeads(std::move(k_expanded));
        SetCachedVHeads(std::move(v_expanded));
      }
      SetCachedUseFlash(use_flash);
      if(use_flash==true)
      {
        SetCachedLogsumexp(std::move(logsumexp));
        SetCachedOutput(std::move(context));
      }
      SetCachedConcat(std::move(merged));
      SetCachedBatch(batch);
      SetCachedDecSeqLen(dec_seq_len);
      SetCachedEncSeqLen(enc_seq_len);
    }

    return output_flat;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceCrossAttention<ComputeT,StorageT>::ForwardImpl(
                           const CAIF_DeviceTensor &input,
                           CAIF_RunContext &ctx)
{
  try
  {
    if(ctx.HasEncoderContext()==false)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::Forward: encoder context not set");
    }
    return ForwardCross(input,ctx.EncoderContext(),ctx);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceCrossAttention<ComputeT,StorageT>::BackwardCross(
                           const CAIF_DeviceTensor &grad_output,
                           CAIF_DeviceTensor &grad_encoder_output,
                           CAIF_RunContext &ctx)
{
  try
  {
    if(CachedDecoderInput().IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::BackwardCross: must call ForwardCross with training=true first");
    }

    const uint32_t batch=CachedBatch();
    const uint32_t dec_seq_len=CachedDecSeqLen();
    const uint32_t enc_seq_len=CachedEncSeqLen();
    const uint32_t dim=Config().Dim();
    const uint32_t kv_input_dim=Config().KvInputDim();
    const uint32_t num_heads=Config().NumHeads();
    const uint32_t num_kv_heads=Config().NumKvHeads();
    const uint32_t head_dim=Config().HeadDim();
    const uint32_t bs_dec=batch*dec_seq_len;
    const uint32_t bs_enc=batch*enc_seq_len;
    const uint32_t bh=batch*num_heads;
    const uint32_t bkv=batch*num_kv_heads;
    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;
    const int repeat_factor=static_cast<int>(num_heads/num_kv_heads);
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    CAIF_DeviceTensor grad_out_flat=CAIF_DeviceTensor::WrapView(
                                     const_cast<void *>(grad_output.DeviceDataRaw()),
                                     {bs_dec,dim},
                                     ctx.Stream(),
                                     grad_output.Dtype());

    CAIF_DeviceTensor grad_concat=CAIF_DeviceTensor::Uninitialized(
                                   {bs_dec,qk_dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeB(grad_out_flat,WO(),grad_concat,ctx,cdt);
    CAIF_DeviceTensor grad_wo_delta=CAIF_DeviceTensor::Uninitialized(
                                     {qk_dim,dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeA(CachedConcat(),grad_out_flat,grad_wo_delta,ctx,cdt);
    CAIF_Ops::Add(GradWO(),grad_wo_delta,GradWO());

    CAIF_DeviceTensor grad_context=CAIF_DeviceTensor::Uninitialized(
                                    {bh*dec_seq_len*head_dim},ctx.Stream(),sdt);
    launch_transpose_0213<StorageT>(grad_concat.template DevicePtr<StorageT>(),
                                     grad_context.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(dec_seq_len),
                                     static_cast<int>(num_heads),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    grad_context.Reshape({bh,dec_seq_len,head_dim});

    CAIF_DeviceTensor k_expanded_storage;
    CAIF_DeviceTensor v_expanded_storage;
    if(num_kv_heads!=num_heads)
    {
      k_expanded_storage=CAIF_DeviceTensor::Uninitialized(
                           {bh,enc_seq_len,head_dim},ctx.Stream(),sdt);
      v_expanded_storage=CAIF_DeviceTensor::Uninitialized(
                           {bh,enc_seq_len,head_dim},ctx.Stream(),sdt);
      launch_gqa_repeat_kv<StorageT>(CachedKHeads().template DevicePtr<StorageT>(),
                                      k_expanded_storage.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(num_kv_heads),
                                      repeat_factor,
                                      static_cast<int>(enc_seq_len),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
      launch_gqa_repeat_kv<StorageT>(CachedVHeads().template DevicePtr<StorageT>(),
                                      v_expanded_storage.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(num_kv_heads),
                                      repeat_factor,
                                      static_cast<int>(enc_seq_len),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
    }
    const CAIF_DeviceTensor &k_expanded=(num_kv_heads!=num_heads)?
                                        k_expanded_storage:CachedKHeads();
    const CAIF_DeviceTensor &v_expanded=(num_kv_heads!=num_heads)?
                                        v_expanded_storage:CachedVHeads();

    const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
    CAIF_DeviceTensor grad_q_heads;
    CAIF_DeviceTensor grad_k_heads;
    CAIF_DeviceTensor grad_v_heads;

    const size_t attn_matrix_bytes=static_cast<size_t>(bh)*dec_seq_len*
                                   enc_seq_len*sizeof(float);
    size_t free_mem=0;
    size_t total_mem=0;
#ifdef USE_CAIF_CUDA
    cudaMemGetInfo(&free_mem,&total_mem);
#else
    (void)total_mem;
#endif
    const bool use_naive_backward=(attn_matrix_bytes*2<=free_mem)||
                                  (CachedUseFlash()==false);

    if(use_naive_backward==true)
    {
      CAIF_DeviceTensor scores=CAIF_DeviceTensor::Uninitialized(
                                {bh,dec_seq_len,enc_seq_len},ctx.Stream(),sdt);
      CAIF_Ops::BatchedMatMulTransposeB(CachedQHeads(),k_expanded,scores,
                                        static_cast<int>(dec_seq_len),
                                        static_cast<int>(head_dim),
                                        static_cast<int>(enc_seq_len),
                                        static_cast<int>(bh),
                                        ctx);
      CAIF_Ops::Scale(scores,scale);
      CAIF_DeviceTensor attn=CAIF_DeviceTensor::Uninitialized(
                              {bh,dec_seq_len,enc_seq_len},ctx.Stream(),sdt);
      launch_attention_softmax<StorageT>(scores.template DevicePtr<StorageT>(),
                                          attn.template DevicePtr<StorageT>(),
                                          static_cast<int>(bh*dec_seq_len),
                                          static_cast<int>(enc_seq_len),
                                          ctx.Stream().Handle());

      CAIF_DeviceTensor grad_attn=CAIF_DeviceTensor::Uninitialized(
                                   {bh,dec_seq_len,enc_seq_len},ctx.Stream(),sdt);
      CAIF_Ops::BatchedMatMulTransposeB(grad_context,v_expanded,grad_attn,
                                        static_cast<int>(dec_seq_len),
                                        static_cast<int>(head_dim),
                                        static_cast<int>(enc_seq_len),
                                        static_cast<int>(bh),
                                        ctx);

      grad_v_heads=CAIF_DeviceTensor::Uninitialized(
                     {bh,enc_seq_len,head_dim},ctx.Stream(),sdt);
      CAIF_Ops::BatchedMatMulTransposeA(attn,grad_context,grad_v_heads,
                                        static_cast<int>(dec_seq_len),
                                        static_cast<int>(enc_seq_len),
                                        static_cast<int>(head_dim),
                                        static_cast<int>(bh),
                                        ctx);

      CAIF_DeviceTensor grad_scores=CAIF_DeviceTensor::Uninitialized(
                                     {bh,dec_seq_len,enc_seq_len},ctx.Stream(),sdt);
      launch_attention_softmax_backward<StorageT>(
                                   grad_attn.template DevicePtr<StorageT>(),
                                   attn.template DevicePtr<StorageT>(),
                                   grad_scores.template DevicePtr<StorageT>(),
                                   static_cast<int>(bh*dec_seq_len),
                                   static_cast<int>(enc_seq_len),
                                   ctx.Stream().Handle());

      CAIF_Ops::Scale(grad_scores,scale);

      grad_q_heads=CAIF_DeviceTensor::Uninitialized(
                     {bh,dec_seq_len,head_dim},ctx.Stream(),sdt);
      CAIF_Ops::BatchedMatMul(grad_scores,k_expanded,grad_q_heads,
                              static_cast<int>(dec_seq_len),
                              static_cast<int>(enc_seq_len),
                              static_cast<int>(head_dim),
                              static_cast<int>(bh),
                              ctx);

      grad_k_heads=CAIF_DeviceTensor::Uninitialized(
                     {bh,enc_seq_len,head_dim},ctx.Stream(),sdt);
      CAIF_Ops::BatchedMatMulTransposeA(grad_scores,CachedQHeads(),
                                        grad_k_heads,
                                        static_cast<int>(dec_seq_len),
                                        static_cast<int>(enc_seq_len),
                                        static_cast<int>(head_dim),
                                        static_cast<int>(bh),
                                        ctx);
    }
    else
    {
      grad_q_heads=CAIF_DeviceTensor::Uninitialized(
                     {bh,dec_seq_len,head_dim},ctx.Stream(),sdt);
      grad_k_heads=CAIF_DeviceTensor::Uninitialized(
                     {bh,enc_seq_len,head_dim},ctx.Stream(),sdt);
      grad_v_heads=CAIF_DeviceTensor::Uninitialized(
                     {bh,enc_seq_len,head_dim},ctx.Stream(),sdt);

      launch_flash_attention_backward_cross<StorageT>(
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
                                  static_cast<int>(dec_seq_len),
                                  static_cast<int>(enc_seq_len),
                                  static_cast<int>(head_dim),
                                  scale,
                                  ctx.Stream().Handle());
    }

    CAIF_DeviceTensor grad_k_reduced;
    CAIF_DeviceTensor grad_v_reduced;
    if(num_kv_heads!=num_heads)
    {
      grad_k_reduced=CAIF_DeviceTensor::Uninitialized(
                       {bkv,enc_seq_len,head_dim},ctx.Stream(),sdt);
      grad_v_reduced=CAIF_DeviceTensor::Uninitialized(
                       {bkv,enc_seq_len,head_dim},ctx.Stream(),sdt);
      launch_gqa_reduce_kv<StorageT>(grad_k_heads.template DevicePtr<StorageT>(),
                                      grad_k_reduced.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(num_kv_heads),
                                      repeat_factor,
                                      static_cast<int>(enc_seq_len),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
      launch_gqa_reduce_kv<StorageT>(grad_v_heads.template DevicePtr<StorageT>(),
                                      grad_v_reduced.template DevicePtr<StorageT>(),
                                      static_cast<int>(batch),
                                      static_cast<int>(num_kv_heads),
                                      repeat_factor,
                                      static_cast<int>(enc_seq_len),
                                      static_cast<int>(head_dim),
                                      ctx.Stream().Handle());
    }
    else
    {
      grad_k_reduced=std::move(grad_k_heads);
      grad_v_reduced=std::move(grad_v_heads);
    }

    CAIF_DeviceTensor grad_q_flat=CAIF_DeviceTensor::Uninitialized(
                                   {bs_dec*qk_dim},ctx.Stream(),sdt);
    launch_transpose_0213<StorageT>(grad_q_heads.template DevicePtr<StorageT>(),
                                     grad_q_flat.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_heads),
                                     static_cast<int>(dec_seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    grad_q_flat.Reshape({bs_dec,qk_dim});

    CAIF_DeviceTensor grad_k_flat=CAIF_DeviceTensor::Uninitialized(
                                   {bs_enc*kv_dim},ctx.Stream(),sdt);
    CAIF_DeviceTensor grad_v_flat=CAIF_DeviceTensor::Uninitialized(
                                   {bs_enc*kv_dim},ctx.Stream(),sdt);
    launch_transpose_0213<StorageT>(grad_k_reduced.template DevicePtr<StorageT>(),
                                     grad_k_flat.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     static_cast<int>(enc_seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    launch_transpose_0213<StorageT>(grad_v_reduced.template DevicePtr<StorageT>(),
                                     grad_v_flat.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(num_kv_heads),
                                     static_cast<int>(enc_seq_len),
                                     static_cast<int>(head_dim),
                                     ctx.Stream().Handle());
    grad_k_flat.Reshape({bs_enc,kv_dim});
    grad_v_flat.Reshape({bs_enc,kv_dim});

    CAIF_DeviceTensor grad_wq_delta=CAIF_DeviceTensor::Uninitialized(
                                     {dim,qk_dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeA(CachedDecoderInput(),grad_q_flat,grad_wq_delta,ctx,cdt);
    CAIF_Ops::Add(GradWQ(),grad_wq_delta,GradWQ());

    // W_K / W_V gradients project against the encoder-output width, so
    // their deltas and the gradient flowing back into the encoder output
    // (gi_k / gi_v / grad_enc) are kv_input_dim-wide, not dim-wide.
    CAIF_DeviceTensor grad_wk_delta=CAIF_DeviceTensor::Uninitialized(
                                     {kv_input_dim,kv_dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeA(CachedEncoderInput(),grad_k_flat,grad_wk_delta,ctx,cdt);
    CAIF_Ops::Add(GradWK(),grad_wk_delta,GradWK());

    CAIF_DeviceTensor grad_wv_delta=CAIF_DeviceTensor::Uninitialized(
                                     {kv_input_dim,kv_dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeA(CachedEncoderInput(),grad_v_flat,grad_wv_delta,ctx,cdt);
    CAIF_Ops::Add(GradWV(),grad_wv_delta,GradWV());

    CAIF_DeviceTensor grad_dec_input=CAIF_DeviceTensor::Uninitialized(
                                      {bs_dec,dim},ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeB(grad_q_flat,WQ(),grad_dec_input,ctx,cdt);
    grad_dec_input.Reshape({batch,dec_seq_len,dim});

    CAIF_DeviceTensor gi_k=CAIF_DeviceTensor::Uninitialized({bs_enc,kv_input_dim},
                                                            ctx.Stream(),sdt);
    CAIF_DeviceTensor gi_v=CAIF_DeviceTensor::Uninitialized({bs_enc,kv_input_dim},
                                                            ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeB(grad_k_flat,WK(),gi_k,ctx,cdt);
    CAIF_Ops::MatMulTransposeB(grad_v_flat,WV(),gi_v,ctx,cdt);
    CAIF_DeviceTensor grad_enc=CAIF_DeviceTensor::Uninitialized({bs_enc,kv_input_dim},
                                                               ctx.Stream(),sdt);
    CAIF_Ops::Add(gi_k,gi_v,grad_enc);
    grad_enc.Reshape({batch,enc_seq_len,kv_input_dim});

    if(grad_encoder_output.IsEmpty()==true)
    {
      grad_encoder_output=std::move(grad_enc);
    }
    else
    {
      CAIF_Ops::Add(grad_encoder_output,grad_enc,grad_encoder_output);
    }

    return grad_dec_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceCrossAttention<ComputeT,StorageT>::BackwardImpl(
                           const CAIF_DeviceTensor &grad_output,
                           CAIF_RunContext &ctx)
{
  try
  {
    if(ctx.HasGradEncoderContext()==false)
    {
      THROW_CAIFE("CAIF_DeviceCrossAttention::Backward: grad encoder context not set");
    }
    return BackwardCross(grad_output,ctx.GradEncoderContext(),ctx);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceCrossAttention<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    GradWQ().FillZero();
    GradWK().FillZero();
    GradWV().FillZero();
    GradWO().FillZero();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceCrossAttention<ComputeT,StorageT>::XavierInit(
                                  CAIF_DeviceTensor &tensor,
                                  std::mt19937 &gen,
                                  uint32_t fan_in,
                                  uint32_t fan_out)
{
  const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                               static_cast<float>(fan_in+fan_out));
  std::uniform_real_distribution<float> dist(-limit,limit);
  const size_t n=tensor.TotalElements();
  std::vector<float> data(n);
  for(size_t i=0;i<n;++i)
  {
    data[i]=dist(gen);
  }
  tensor.CopyFromHostFp32(data.data(),n);
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceCrossAttention<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    const uint32_t dim=Config().Dim();
    const uint32_t kv_input_dim=Config().KvInputDim();
    const uint32_t qk_dim=Config().NumHeads()*Config().HeadDim();
    const uint32_t kv_dim=Config().NumKvHeads()*Config().HeadDim();

    std::mt19937 gen(seed);
    XavierInit(WQ(),gen,dim,qk_dim);
    XavierInit(WK(),gen,kv_input_dim,kv_dim);
    XavierInit(WV(),gen,kv_input_dim,kv_dim);
    XavierInit(WO(),gen,qk_dim,dim);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceCrossAttention<ComputeT,StorageT>::ParameterTensorCount()const
{
  return g_caif_attention_weight_count;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceCrossAttention<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
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
    THROW_CAIFE("CAIF_DeviceCrossAttention::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceCrossAttention<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
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
    THROW_CAIFE("CAIF_DeviceCrossAttention::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceCrossAttention<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
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
    THROW_CAIFE("CAIF_DeviceCrossAttention::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceCrossAttention<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
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
    THROW_CAIFE("CAIF_DeviceCrossAttention::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceCrossAttention<ComputeT,StorageT>::TotalParameterCount()const
{
  return _w_q.TotalElements()+
         WK().TotalElements()+
         WV().TotalElements()+
         WO().TotalElements();
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceCrossAttention<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc=std::string(g_serial_tag_cross_attention)+
                     g_serial_open_paren+
                     g_serial_kv_dim+
                     std::to_string(Config().Dim())+
                     g_serial_comma+
                     g_serial_kv_heads+
                     std::to_string(Config().NumHeads())+
                     g_serial_comma+
                     g_serial_kv_head_dim+
                     std::to_string(Config().HeadDim());
    if(Config().NumKvHeads()!=Config().NumHeads())
    {
      desc+=g_serial_comma;
      desc+=g_serial_kv_kv_heads;
      desc+=std::to_string(Config().NumKvHeads());
    }
    desc+=g_serial_close_paren;
    return desc;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceCrossAttention<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
  return {prefix+reg.Name(CAIF_ParamRole::Role_e::AttnWQ_e),
          prefix+reg.Name(CAIF_ParamRole::Role_e::AttnWK_e),
          prefix+reg.Name(CAIF_ParamRole::Role_e::AttnWV_e),
          prefix+reg.Name(CAIF_ParamRole::Role_e::AttnWO_e)};
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceCrossAttention<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceCrossAttention<float,__half>;
template class CAIF_DeviceCrossAttention<float,__nv_bfloat16>;
template class CAIF_DeviceCrossAttention<__half,float>;
template class CAIF_DeviceCrossAttention<__half,__half>;
template class CAIF_DeviceCrossAttention<__half,__nv_bfloat16>;
template class CAIF_DeviceCrossAttention<__nv_bfloat16,float>;
template class CAIF_DeviceCrossAttention<__nv_bfloat16,__half>;
template class CAIF_DeviceCrossAttention<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
