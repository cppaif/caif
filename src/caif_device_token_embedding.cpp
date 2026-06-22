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

// Token-embedding has two input encodings:
//   1. Native integer (input.Dtype()==Int32/UInt32) — reinterpreted via
//      `DeviceDataRaw()` to `unsigned int *`. No DevicePtr() call.
//   2. fp32-compat (input.Dtype()==Float32, float-encoded token IDs) —
//      uses `DevicePtr<float>()`.
// Per-site comments at each call site below name the fp32 contract.

#include "caif_device_token_embedding.h"
#include "caif_device_token_embedding_factory.h"
#include "caif_cuda_kernels_embeddings.cuh"
#include "caif_constants.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <vector>
#include <cmath>
#include <cstring>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#endif

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::CAIF_DeviceTokenEmbedding(
                                          const CAIF_DeviceTokenEmbeddingConfig &config,
                                          CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _embedding_table(new CAIF_DeviceTensor()),
                                          _embedding_table_grad(new CAIF_DeviceTensor()),
                                          _token_ids_device(nullptr),
                                          _token_ids_capacity(0),
                                          _cached_num_tokens(0)
{
  try
  {
    if(config.VocabSize()==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: vocab_size must be > 0");
    }
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: dim must be > 0");
    }

    // Xavier uniform init: limit = sqrt(scale / (fan_in + fan_out)).
    const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                 static_cast<float>(config.VocabSize()+config.Dim()));
    const size_t table_size=static_cast<size_t>(config.VocabSize())*config.Dim();
    std::vector<float> init_data(table_size);
    for(size_t i=0;i<table_size;++i)
    {
      // Deterministic pseudo-random via fractional part of i*phi.
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      init_data[i]=(t-std::floor(t))*2.0f*limit-limit;
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    SetEmbeddingTable(CAIF_DeviceTensor::Uninitialized({config.VocabSize(),config.Dim()},stream,sdt));
    EmbeddingTable().CopyFromHostFp32(init_data.data(),table_size);

    // Gradient is accumulated in fp32 regardless of storage dtype so repeated
    // tokens (common in LM training) do not lose precision in bf16/fp16.
    const CAIF_DataType::CAIF_DataType_e grad_dtype=CAIF_DataType::CAIF_DataType_e::Float32;
    SetEmbeddingTableGrad(CAIF_DeviceTensor::Zeros({config.VocabSize(),config.Dim()},stream,grad_dtype));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::CAIF_DeviceTokenEmbedding(
                                          const CAIF_DeviceTokenEmbeddingConfig &config,
                                          CAIF_DeviceTensor &shared_table,
                                          CAIF_DeviceTensor &shared_grad,
                                          CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _embedding_table(&shared_table),
                                          _embedding_table_grad(&shared_grad),
                                          _token_ids_device(nullptr),
                                          _token_ids_capacity(0),
                                          _cached_num_tokens(0)
{
  try
  {
    if(config.VocabSize()==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: vocab_size must be > 0");
    }
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: dim must be > 0");
    }
    const std::vector<uint32_t> &shape=shared_table.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: shared table must be 2D");
    }
    if(shape[0]!=config.VocabSize()||shape[1]!=config.Dim())
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: shared table shape must be [vocab_size, dim]");
    }
    if(shared_grad.Shape()!=shape)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: shared grad shape must match shared table shape");
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::~CAIF_DeviceTokenEmbedding()
{
  // `delete nullptr` is a no-op in C++, so a CAIF_DeviceSharedTokenEmbedding
  // that has nulled these pointers in its own destructor before the base
  // destructor runs leaves the donor's storage untouched.
  delete EmbeddingTablePtr();
  delete EmbeddingTableGradPtr();
  FreeTokenIdBuffer();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::CAIF_DeviceTokenEmbedding(
                                  CAIF_DeviceTokenEmbedding &&other):
                                CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                                _config(other.Config()),
                                _embedding_table(other.EmbeddingTablePtr()),
                                _embedding_table_grad(other.EmbeddingTableGradPtr()),
                                _token_ids_device(other.TokenIdsDevice()),
                                _token_ids_capacity(other.TokenIdsCapacity()),
                                _cached_num_tokens(other.CachedNumTokens())
{
  // Pointer ownership transfer: source must release the pointers so its
  // destructor's `delete` does not free storage we now own (or storage
  // a borrower's donor still owns, in the shared-embedding case).
  other.SetEmbeddingTablePtr(nullptr);
  other.SetEmbeddingTableGradPtr(nullptr);
  other.SetTokenIdsDevice(nullptr);
  other.SetTokenIdsCapacity(0);
  other.SetCachedNumTokens(0);
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT> &
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::operator=(CAIF_DeviceTokenEmbedding &&other)
{
  if(this!=&other)
  {
    delete EmbeddingTablePtr();
    delete EmbeddingTableGradPtr();
    FreeTokenIdBuffer();
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    SetConfig(other.Config());
    SetEmbeddingTablePtr(other.EmbeddingTablePtr());
    SetEmbeddingTableGradPtr(other.EmbeddingTableGradPtr());
    SetTokenIdsDevice(other.TokenIdsDevice());
    SetTokenIdsCapacity(other.TokenIdsCapacity());
    SetCachedNumTokens(other.CachedNumTokens());
    other.SetEmbeddingTablePtr(nullptr);
    other.SetEmbeddingTableGradPtr(nullptr);
    other.SetTokenIdsDevice(nullptr);
    other.SetTokenIdsCapacity(0);
    other.SetCachedNumTokens(0);
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::EnsureTokenIdCapacity(size_t num_tokens)
{
#ifdef USE_CAIF_CUDA
  if(num_tokens>TokenIdsCapacity())
  {
    FreeTokenIdBuffer();
    uint32_t *new_buffer=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&new_buffer),num_tokens*sizeof(uint32_t));
    SetTokenIdsDevice(new_buffer);
    SetTokenIdsCapacity(num_tokens);
  }
#else
  (void)num_tokens;
#endif
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::FreeTokenIdBuffer()
{
#ifdef USE_CAIF_CUDA
  if(TokenIdsDevice()!=nullptr)
  {
    cudaFree(TokenIdsDevice());
    SetTokenIdsDevice(nullptr);
    SetTokenIdsCapacity(0);
  }
#endif
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::ForwardFromIds(const uint32_t *host_token_ids,
                                                              uint32_t batch,
                                                              uint32_t seq_len,
                                                              bool training)
{
  try
  {
    const uint32_t num_tokens=batch*seq_len;

    EnsureTokenIdCapacity(num_tokens);
#ifdef USE_CAIF_CUDA
    cudaMemcpyAsync(TokenIdsDevice(),
                    host_token_ids,
                    num_tokens*sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    Stream().Handle());
#endif

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({batch,seq_len,Config().Dim()},
                                                              Stream(),StorageDtype());

    launch_embedding_lookup<StorageT>(StoragePtr(EmbeddingTable()),
                                      TokenIdsDevice(),
                                      StoragePtr(output),
                                      static_cast<int>(num_tokens),
                                      static_cast<int>(Config().Dim()),
                                      Stream().Handle());

    if(training==true)
    {
      SetCachedNumTokens(num_tokens);
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                          CAIF_RunContext &ctx)
{
  try
  {
    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding::Forward: input must be 2D [batch, seq_len]");
    }

    const uint32_t batch=shape[0];
    const uint32_t seq_len=shape[1];
    const uint32_t num_tokens=batch*seq_len;

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({batch,seq_len,Config().Dim()},
                                                              ctx.Stream(),StorageDtype());

    const CAIF_DataType::CAIF_DataType_e dtype=input.Dtype();
    const bool is_uint_ids=(dtype==CAIF_DataType::CAIF_DataType_e::UInt32||
                            dtype==CAIF_DataType::CAIF_DataType_e::Int32);

    if(is_uint_ids==true)
    {
      // Native integer path: no float->uint cast in the kernel. Input
      // tensor is Int32/UInt32 (4-byte elements); reinterpret via the
      // raw data pointer so the fp32-only DevicePtr() overload isn't
      // touched.
      const unsigned int *ids=reinterpret_cast<const unsigned int *>(input.DeviceDataRaw());
      launch_embedding_lookup<StorageT>(StoragePtr(EmbeddingTable()),
                                        ids,
                                        StoragePtr(output),
                                        static_cast<int>(num_tokens),
                                        static_cast<int>(Config().Dim()),
                                        ctx.Stream().Handle());

      if(ctx.Training()==true)
      {
        EnsureTokenIdCapacity(num_tokens);
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(TokenIdsDevice(),
                        ids,
                        num_tokens*sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice,
                        ctx.Stream().Handle());
#endif
        SetCachedNumTokens(num_tokens);
      }
    }
    else
    {
      // Compat path: float-encoded token IDs (in-kernel cast). input is
      // Float32 by definition here (we're in the else branch of the
      // Int32/UInt32 check), so DevicePtr<float>() is the correct typed
      // accessor.
      launch_embedding_lookup_float<StorageT>(StoragePtr(EmbeddingTable()),
                                              input.template DevicePtr<float>(),
                                              StoragePtr(output),
                                              static_cast<int>(num_tokens),
                                              static_cast<int>(Config().Dim()),
                                              ctx.Stream().Handle());

      if(ctx.Training()==true)
      {
        EnsureTokenIdCapacity(num_tokens);
        // fp32: input is Float32 in this else-branch (see comment above).
        launch_float_to_uint(input.template DevicePtr<float>(),
                             TokenIdsDevice(),
                             static_cast<int>(num_tokens),
                             ctx.Stream().Handle());
        SetCachedNumTokens(num_tokens);
      }
    }

    if(OutputScale()!=1.0f)
    {
      CAIF_Ops::Scale(output,OutputScale());
    }
    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                           CAIF_RunContext &ctx)
{
  try
  {
    if(CachedNumTokens()==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding::Backward: must call Forward with training=true first");
    }
    AssertInputDtype(grad_output);

    // F6: the forward multiplied the lookup by output_scale, so the table
    // gradient is grad_output * output_scale (chain rule). Accumulate from a
    // scaled clone when the factor is non-unit; from the original otherwise.
    //
    // No FillZero here — gradients are accumulated into the table via atomicAdd
    // and zeroed once per training step by ZeroGradients (matches the
    // CAIF_DeviceLinearHead tied-weight contract). Zeroing in Backward would
    // wipe the contribution of a co-tied CAIF_DeviceSharedTokenEmbedding whose
    // Backward already ran earlier in the same step.
    if(OutputScale()!=1.0f)
    {
      CAIF_DeviceTensor scaled_grad=grad_output.Clone();
      CAIF_Ops::Scale(scaled_grad,OutputScale());
      launch_embedding_backward<StorageT>(StoragePtr(scaled_grad),
                                          TokenIdsDevice(),
                                          EmbeddingTableGrad().template DevicePtr<float>(),
                                          static_cast<int>(CachedNumTokens()),
                                          static_cast<int>(Config().Dim()),
                                          ctx.Stream().Handle());
    }
    else
    {
      launch_embedding_backward<StorageT>(StoragePtr(grad_output),
                                          TokenIdsDevice(),
                                          EmbeddingTableGrad().template DevicePtr<float>(),
                                          static_cast<int>(CachedNumTokens()),
                                          static_cast<int>(Config().Dim()),
                                          ctx.Stream().Handle());
    }

    // Input is non-differentiable discrete token IDs.
    return CAIF_DeviceTensor();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    EmbeddingTableGrad().FillZero();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::ParameterTensorCount()const
{
  return g_caif_embedding_parameter_count;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return EmbeddingTable();
    }
    THROW_CAIFE("CAIF_DeviceTokenEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return EmbeddingTable();
    }
    THROW_CAIFE("CAIF_DeviceTokenEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return EmbeddingTableGrad();
    }
    THROW_CAIFE("CAIF_DeviceTokenEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return EmbeddingTableGrad();
    }
    THROW_CAIFE("CAIF_DeviceTokenEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::TotalParameterCount()const
{
  return static_cast<size_t>(Config().VocabSize())*Config().Dim();
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::Description()const
{
  try
  {
    return g_serial_tag_token_embedding+
           g_serial_open_paren+
           g_serial_kv_vocab+
           std::to_string(Config().VocabSize())+
           g_serial_comma+
           g_serial_kv_dim+
           std::to_string(Config().Dim())+
           g_serial_close_paren;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::TokenEmbeddingTable_e));
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::LoadEmbeddingTable(CAIF_DeviceTensor &&table)
{
  try
  {
    const std::vector<uint32_t> &shape=table.Shape();
    if(shape.size()!=2||shape[0]!=Config().VocabSize()||shape[1]!=Config().Dim())
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding::LoadEmbeddingTable: shape mismatch");
    }
    SetEmbeddingTable(std::move(table));
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceTokenEmbedding<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceTokenEmbedding<float,__half>;
template class CAIF_DeviceTokenEmbedding<float,__nv_bfloat16>;
template class CAIF_DeviceTokenEmbedding<__half,float>;
template class CAIF_DeviceTokenEmbedding<__half,__half>;
template class CAIF_DeviceTokenEmbedding<__half,__nv_bfloat16>;
template class CAIF_DeviceTokenEmbedding<__nv_bfloat16,float>;
template class CAIF_DeviceTokenEmbedding<__nv_bfloat16,__half>;
template class CAIF_DeviceTokenEmbedding<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
