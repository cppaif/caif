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
//      uses `DevicePtr<float>()` per the type-dispatch full plan.
// Per-site comments at each call site below name the fp32 contract.

#include "caif_device_token_embedding.h"
#include "caif_device_token_embedding_factory.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
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
                                          const Config_t &config,
                                          CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _embedding_table(),
                                          _embedding_table_grad(),
                                          _token_ids_device(nullptr),
                                          _token_ids_capacity(0),
                                          _cached_num_tokens(0)
{
  try
  {
    if(config.vocab_size==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: vocab_size must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding: dim must be > 0");
    }

    // Xavier uniform init: limit = sqrt(scale / (fan_in + fan_out)).
    const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                 static_cast<float>(config.vocab_size+config.dim));
    const size_t table_size=static_cast<size_t>(config.vocab_size)*config.dim;
    std::vector<float> init_data(table_size);
    for(size_t i=0;i<table_size;++i)
    {
      // Deterministic pseudo-random via fractional part of i*phi.
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      init_data[i]=(t-std::floor(t))*2.0f*limit-limit;
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    SetEmbeddingTable(CAIF_DeviceTensor::Uninitialized({config.vocab_size,config.dim},stream,sdt));
    EmbeddingTableMut().CopyFromHostFp32(init_data.data(),table_size);

    _embedding_table_grad=CAIF_DeviceTensor::Zeros({config.vocab_size,config.dim},stream,sdt);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::~CAIF_DeviceTokenEmbedding()
{
  FreeTokenIdBuffer();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::CAIF_DeviceTokenEmbedding(
                                  CAIF_DeviceTokenEmbedding &&other):
                                CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                                _config(other._config),
                                _embedding_table(std::move(other._embedding_table)),
                                _embedding_table_grad(std::move(other._embedding_table_grad)),
                                _token_ids_device(other._token_ids_device),
                                _token_ids_capacity(other._token_ids_capacity),
                                _cached_num_tokens(other._cached_num_tokens)
{
  other._token_ids_device=nullptr;
  other._token_ids_capacity=0;
  other._cached_num_tokens=0;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTokenEmbedding<ComputeT,StorageT> &
CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::operator=(CAIF_DeviceTokenEmbedding &&other)
{
  if(this!=&other)
  {
    FreeTokenIdBuffer();
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    _config=other._config;
    _embedding_table=std::move(other._embedding_table);
    _embedding_table_grad=std::move(other._embedding_table_grad);
    _token_ids_device=other._token_ids_device;
    _token_ids_capacity=other._token_ids_capacity;
    _cached_num_tokens=other._cached_num_tokens;
    other._token_ids_device=nullptr;
    other._token_ids_capacity=0;
    other._cached_num_tokens=0;
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::EnsureTokenIdCapacity(size_t num_tokens)
{
#ifdef USE_CAIF_CUDA
  if(num_tokens>_token_ids_capacity)
  {
    FreeTokenIdBuffer();
    cudaMalloc(reinterpret_cast<void**>(&_token_ids_device),num_tokens*sizeof(uint32_t));
    _token_ids_capacity=num_tokens;
  }
#else
  (void)num_tokens;
#endif
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::FreeTokenIdBuffer()
{
#ifdef USE_CAIF_CUDA
  if(_token_ids_device!=nullptr)
  {
    cudaFree(_token_ids_device);
    _token_ids_device=nullptr;
    _token_ids_capacity=0;
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
    cudaMemcpyAsync(_token_ids_device,
                    host_token_ids,
                    num_tokens*sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    Stream().Handle());
#endif

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(
      {batch,seq_len,_config.dim},Stream(),StorageDtype());

    launch_embedding_lookup<StorageT>(StoragePtr(_embedding_table),
                                      _token_ids_device,
                                      StoragePtr(output),
                                      static_cast<int>(num_tokens),
                                      static_cast<int>(_config.dim),
                                      Stream().Handle());

    if(training==true)
    {
      _cached_num_tokens=num_tokens;
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

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(
      {batch,seq_len,_config.dim},ctx.Stream(),StorageDtype());

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
      launch_embedding_lookup<StorageT>(StoragePtr(_embedding_table),
                                        ids,
                                        StoragePtr(output),
                                        static_cast<int>(num_tokens),
                                        static_cast<int>(_config.dim),
                                        ctx.Stream().Handle());

      if(ctx.Training()==true)
      {
        EnsureTokenIdCapacity(num_tokens);
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(_token_ids_device,
                        ids,
                        num_tokens*sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice,
                        ctx.Stream().Handle());
#endif
        _cached_num_tokens=num_tokens;
      }
    }
    else
    {
      // Compat path: float-encoded token IDs (in-kernel cast). input is
      // Float32 by definition here (we're in the else branch of the
      // Int32/UInt32 check), so DevicePtr<float>() is the correct typed
      // accessor.
      launch_embedding_lookup_float<StorageT>(StoragePtr(_embedding_table),
                                              input.DevicePtr<float>(),
                                              StoragePtr(output),
                                              static_cast<int>(num_tokens),
                                              static_cast<int>(_config.dim),
                                              ctx.Stream().Handle());

      if(ctx.Training()==true)
      {
        EnsureTokenIdCapacity(num_tokens);
        // fp32: input is Float32 in this else-branch (see comment above).
        launch_float_to_uint(input.DevicePtr<float>(),
                             _token_ids_device,
                             static_cast<int>(num_tokens),
                             ctx.Stream().Handle());
        _cached_num_tokens=num_tokens;
      }
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
    if(_cached_num_tokens==0)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding::Backward: must call Forward with training=true first");
    }
    AssertInputDtype(grad_output);

    _embedding_table_grad.FillZero();

    launch_embedding_backward<StorageT>(StoragePtr(grad_output),
                                        _token_ids_device,
                                        StoragePtr(_embedding_table_grad),
                                        static_cast<int>(_cached_num_tokens),
                                        static_cast<int>(_config.dim),
                                        ctx.Stream().Handle());

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
    _embedding_table_grad.FillZero();
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
      return _embedding_table;
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
      return _embedding_table;
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
      return _embedding_table_grad;
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
      return _embedding_table_grad;
    }
    THROW_CAIFE("CAIF_DeviceTokenEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::TotalParameterCount()const
{
  return static_cast<size_t>(_config.vocab_size)*_config.dim;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::Description()const
{
  try
  {
    return "TokenEmbedding(vocab="+std::to_string(_config.vocab_size)+
           ",dim="+std::to_string(_config.dim)+")";
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
    names.push_back(prefix+"weight");
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
    if(shape.size()!=2||shape[0]!=_config.vocab_size||shape[1]!=_config.dim)
    {
      THROW_CAIFE("CAIF_DeviceTokenEmbedding::LoadEmbeddingTable: shape mismatch");
    }
    _embedding_table=std::move(table);
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
