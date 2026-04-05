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

#include "caif_device_token_embedding.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>
#include <cmath>
#include <cstring>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

namespace instance
{

CAIF_DeviceTokenEmbedding::CAIF_DeviceTokenEmbedding(const CAIF_DeviceTokenEmbedding::Config_t &config,
                                                   CAIF_CudaStream &stream):
                                                   CAIF_DeviceLayer(stream),
                                                   _config(config),
                                                   _embedding_table(),
                                                   _embedding_table_grad(),
                                                   _output_buffer(),
                                                   _token_ids_device(nullptr),
                                                   _token_ids_capacity(0),
                                                   _cached_num_tokens(0),
                                                   _output_batch(0),
                                                   _output_seq_len(0)
{
  try
  {
    if(config.vocab_size==0)
    {
      THROW_CAIFE("DeviceTokenEmbedding: vocab_size must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceTokenEmbedding: dim must be > 0");
    }

    // Xavier uniform init: limit = sqrt(6 / (fan_in + fan_out))
    const float limit=std::sqrt(6.0f/static_cast<float>(config.vocab_size+config.dim));
    const size_t table_size=static_cast<size_t>(config.vocab_size)*config.dim;
    std::vector<float> init_data(table_size);
    for(size_t i=0;i<table_size;++i)
    {
      // Simple deterministic pseudo-random using golden ratio
      const float t=static_cast<float>(i)*0.6180339887f;
      init_data[i]=(t-std::floor(t))*2.0f*limit-limit;
    }

    _embedding_table=CAIF_DeviceTensor::Uninitialized({config.vocab_size,config.dim},stream);
    _embedding_table.CopyFromHost(init_data.data(),table_size);

    _embedding_table_grad=CAIF_DeviceTensor::Zeros({config.vocab_size,config.dim},stream);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTokenEmbedding::~CAIF_DeviceTokenEmbedding()
{
  FreeTokenIdBuffer();
}

CAIF_DeviceTokenEmbedding::CAIF_DeviceTokenEmbedding(CAIF_DeviceTokenEmbedding &&other)noexcept:
                                                   CAIF_DeviceLayer(std::move(other)),
                                                   _config(other._config),
                                                   _embedding_table(std::move(other._embedding_table)),
                                                   _embedding_table_grad(std::move(other._embedding_table_grad)),
                                                   _output_buffer(std::move(other._output_buffer)),
                                                   _token_ids_device(other._token_ids_device),
                                                   _token_ids_capacity(other._token_ids_capacity),
                                                   _cached_num_tokens(other._cached_num_tokens),
                                                   _output_batch(other._output_batch),
                                                   _output_seq_len(other._output_seq_len)
{
  other._token_ids_device=nullptr;
  other._token_ids_capacity=0;
  other._cached_num_tokens=0;
  other._output_batch=0;
  other._output_seq_len=0;
}

CAIF_DeviceTokenEmbedding &CAIF_DeviceTokenEmbedding::operator=(CAIF_DeviceTokenEmbedding &&other)noexcept
{
  if(this!=&other)
  {
    FreeTokenIdBuffer();
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _embedding_table=std::move(other._embedding_table);
    _embedding_table_grad=std::move(other._embedding_table_grad);
    _output_buffer=std::move(other._output_buffer);
    _token_ids_device=other._token_ids_device;
    _token_ids_capacity=other._token_ids_capacity;
    _cached_num_tokens=other._cached_num_tokens;
    _output_batch=other._output_batch;
    _output_seq_len=other._output_seq_len;
    other._token_ids_device=nullptr;
    other._token_ids_capacity=0;
    other._cached_num_tokens=0;
    other._output_batch=0;
    other._output_seq_len=0;
  }
  return *this;
}

void CAIF_DeviceTokenEmbedding::EnsureTokenIdCapacity(size_t num_tokens)
{
#ifdef USE_CAIF_CUDA
  if(num_tokens>_token_ids_capacity)
  {
    FreeTokenIdBuffer();
    cudaMalloc(reinterpret_cast<void**>(&_token_ids_device),num_tokens*sizeof(uint32_t));
    _token_ids_capacity=num_tokens;
  }
#endif
}

void CAIF_DeviceTokenEmbedding::FreeTokenIdBuffer()
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

void CAIF_DeviceTokenEmbedding::EnsureOutputBuffer(uint32_t batch,uint32_t seq_len)
{
  if(batch!=_output_batch||seq_len!=_output_seq_len)
  {
    _output_buffer=CAIF_DeviceTensor::Uninitialized({batch,seq_len,_config.dim},*_stream);
    _output_batch=batch;
    _output_seq_len=seq_len;
  }
}

CAIF_DeviceTensor CAIF_DeviceTokenEmbedding::ForwardFromIds(const uint32_t *host_token_ids,
                                                          uint32_t batch,
                                                          uint32_t seq_len,
                                                          bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceTokenEmbedding: layer has been moved from");
    }

    const uint32_t num_tokens=batch*seq_len;

    // Copy token IDs to device
    EnsureTokenIdCapacity(num_tokens);
#ifdef USE_CAIF_CUDA
    cudaMemcpyAsync(_token_ids_device,
                    host_token_ids,
                    num_tokens*sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    _stream->Handle());
#endif

    // Reuse output buffer if shape matches
    EnsureOutputBuffer(batch,seq_len);

    launch_embedding_lookup(_embedding_table.DevicePtr(),
                            _token_ids_device,
                            _output_buffer.DevicePtr(),
                            static_cast<int>(num_tokens),
                            static_cast<int>(_config.dim),
                            _stream->Handle());

    if(training==true)
    {
      _cached_num_tokens=num_tokens;
    }

    return _output_buffer.Clone();
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceTokenEmbedding::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceTokenEmbedding: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("DeviceTokenEmbedding::Forward: input must be 2D [batch, seq_len]");
    }

    const uint32_t batch=shape[0];
    const uint32_t seq_len=shape[1];
    const uint32_t num_tokens=batch*seq_len;

    // Reuse output buffer if shape matches
    EnsureOutputBuffer(batch,seq_len);

    launch_embedding_lookup_float(_embedding_table.DevicePtr(),
                                  input.DevicePtr(),
                                  _output_buffer.DevicePtr(),
                                  static_cast<int>(num_tokens),
                                  static_cast<int>(_config.dim),
                                  _stream->Handle());

    if(training==true)
    {
      // Convert float IDs to uint32 on GPU (no host roundtrip)
      EnsureTokenIdCapacity(num_tokens);
      launch_float_to_uint(input.DevicePtr(),
                            _token_ids_device,
                            static_cast<int>(num_tokens),
                            _stream->Handle());
      _cached_num_tokens=num_tokens;
    }

    return _output_buffer.Clone();
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceTokenEmbedding::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceTokenEmbedding: layer has been moved from");
    }
    if(_cached_num_tokens==0)
    {
      THROW_CAIFE(
        "DeviceTokenEmbedding::Backward: must call Forward with training=true first");
    }

    // Zero grad table before scatter-add
    _embedding_table_grad.Fill(0.0f);

    launch_embedding_backward(grad_output.DevicePtr(),
                              _token_ids_device,
                              _embedding_table_grad.DevicePtr(),
                              static_cast<int>(_cached_num_tokens),
                              static_cast<int>(_config.dim),
                              _stream->Handle());

    // Return empty tensor (input is non-differentiable discrete token IDs)
    return CAIF_DeviceTensor();
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceTokenEmbedding::ZeroGradients()
{
  try
  {
    _embedding_table_grad.Fill(0.0f);
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceTokenEmbedding::ParameterTensorCount()const
{
  try
  {
    return g_caif_embedding_parameter_count;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceTokenEmbedding::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _embedding_table;
    }
    THROW_CAIFE("DeviceTokenEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceTokenEmbedding::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _embedding_table;
    }
    THROW_CAIFE("DeviceTokenEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceTokenEmbedding::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _embedding_table_grad;
    }
    THROW_CAIFE("DeviceTokenEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceTokenEmbedding::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _embedding_table_grad;
    }
    THROW_CAIFE("DeviceTokenEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceTokenEmbedding::TotalParameterCount()const
{
  try
  {
    return static_cast<size_t>(_config.vocab_size)*_config.dim;
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceTokenEmbedding::Description()const
{
  try
  {
    return "TokenEmbedding(vocab="+std::to_string(_config.vocab_size)+
           ",dim="+std::to_string(_config.dim)+")";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceTokenEmbedding::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"weight");
    return names;
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
