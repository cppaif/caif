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
// Device-resident tabular embedding implementation
//------------------------------------------------------------------------------
#include "caif_device_tabular_embedding.h"
#include "caif_device_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <cmath>
#include <random>

namespace instance
{

CAIF_DeviceTabularEmbedding::CAIF_DeviceTabularEmbedding(const Config_t &config,
                                                       CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                       _config(config),
                                                       _cached_batch(0),
                                                       _cached_seq_len(0)
{
  try
  {
    if(config.num_features==0)
    {
      THROW_CAIFE("DeviceTabularEmbedding: num_features must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceTabularEmbedding: dim must be > 0");
    }

    // Allocate parameters
    _w_proj=CAIF_DeviceTensor::Zeros({config.num_features,config.dim},stream);
    _b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream);

    // Allocate gradients
    _grad_w_proj=CAIF_DeviceTensor::Zeros({config.num_features,config.dim},stream);
    _grad_b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream);

    // Initialize weights with Xavier uniform
    InitializeWeights(0);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTabularEmbedding::CAIF_DeviceTabularEmbedding(
  CAIF_DeviceTabularEmbedding &&other):CAIF_DeviceLayer(std::move(other)),
                                      _config(other._config),
                                      _w_proj(std::move(other._w_proj)),
                                      _b_proj(std::move(other._b_proj)),
                                      _grad_w_proj(std::move(other._grad_w_proj)),
                                      _grad_b_proj(std::move(other._grad_b_proj)),
                                      _cached_input(std::move(other._cached_input)),
                                      _cached_batch(other._cached_batch),
                                      _cached_seq_len(other._cached_seq_len)
{
}

CAIF_DeviceTabularEmbedding &CAIF_DeviceTabularEmbedding::operator=(CAIF_DeviceTabularEmbedding &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _w_proj=std::move(other._w_proj);
    _b_proj=std::move(other._b_proj);
    _grad_w_proj=std::move(other._grad_w_proj);
    _grad_b_proj=std::move(other._grad_b_proj);
    _cached_input=std::move(other._cached_input);
    _cached_batch=other._cached_batch;
    _cached_seq_len=other._cached_seq_len;
  }
  return *this;
}

CAIF_DeviceTensor CAIF_DeviceTabularEmbedding::Forward(const CAIF_DeviceTensor &input,
                                                     bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceTabularEmbedding: layer has been moved from");
    }

    const auto &shape=input.Shape();
    uint32_t batch=0;
    uint32_t seq_len=1;
    uint32_t num_features=0;

    // Handle 2D [batch, num_features] or 3D [batch, seq_len, num_features]
    if(shape.size()==2)
    {
      batch=shape[0];
      num_features=shape[1];
      seq_len=1;
    }
    else if(shape.size()==3)
    {
      batch=shape[0];
      seq_len=shape[1];
      num_features=shape[2];
    }
    else
    {
      THROW_CAIFE("DeviceTabularEmbedding: input must be 2D or 3D");
    }

    if(num_features!=_config.num_features)
    {
      THROW_CAIFE("DeviceTabularEmbedding: input feature dim mismatch");
    }

    const uint32_t total_rows=batch*seq_len;
    const uint32_t dim=_config.dim;

    // Cache for backward
    if(training==true)
    {
      _cached_input=input.Clone();
      _cached_batch=batch;
      _cached_seq_len=seq_len;
    }

    // Flatten to [total_rows, num_features]
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({total_rows,num_features});

    // Project: output = input @ W_proj
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({total_rows,dim},*_stream);
    CAIF_DeviceOps::MatMul(flat_input,_w_proj,output);

    // Add bias
    launch_bias_add_2d(output.DevicePtr(),_b_proj.DevicePtr(),
                       output.DevicePtr(),
                       static_cast<int>(total_rows),
                       static_cast<int>(dim),
                       _stream->Handle());

    // Reshape to [batch, seq_len, dim]
    output.Reshape({batch,seq_len,dim});

    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceTabularEmbedding::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceTabularEmbedding: layer has been moved from");
    }

    const uint32_t batch=_cached_batch;
    const uint32_t seq_len=_cached_seq_len;
    const uint32_t num_features=_config.num_features;
    const uint32_t dim=_config.dim;
    const uint32_t total_rows=batch*seq_len;

    // Flatten grad_output to [total_rows, dim]
    CAIF_DeviceTensor flat_grad=grad_output.Clone();
    flat_grad.Reshape({total_rows,dim});

    // grad_b_proj = sum(grad_output, axis=0)
    launch_bias_grad_2d(flat_grad.DevicePtr(),_grad_b_proj.DevicePtr(),
                        static_cast<int>(total_rows),
                        static_cast<int>(dim),
                        _stream->Handle());

    // grad_w_proj = input^T @ grad_output
    // input: [total_rows, num_features], grad_output: [total_rows, dim]
    // grad_w_proj: [num_features, dim]
    CAIF_DeviceTensor flat_input=_cached_input.Clone();
    flat_input.Reshape({total_rows,num_features});
    CAIF_DeviceOps::MatMulTransposeA(flat_input,flat_grad,_grad_w_proj);

    // grad_input = grad_output @ W_proj^T
    // grad_output: [total_rows, dim], W_proj: [num_features, dim]
    // grad_input: [total_rows, num_features]
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({total_rows,num_features},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(flat_grad,_w_proj,grad_input);

    // Reshape to original input shape
    if(seq_len==1)
    {
      grad_input.Reshape({batch,num_features});
    }
    else
    {
      grad_input.Reshape({batch,seq_len,num_features});
    }

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceTabularEmbedding::ZeroGradients()
{
  try
  {
    _grad_w_proj.Fill(0.0f);
    _grad_b_proj.Fill(0.0f);
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceTabularEmbedding::ParameterTensorCount()const
{
  return 2;  // w_proj, b_proj
}

CAIF_DeviceTensor &CAIF_DeviceTabularEmbedding::ParameterTensor(size_t index)
{
  if(index==0)
  {
    return _w_proj;
  }
  if(index==1)
  {
    return _b_proj;
  }
  THROW_CAIFE("DeviceTabularEmbedding: parameter index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceTabularEmbedding::ParameterTensor(size_t index)const
{
  if(index==0)
  {
    return _w_proj;
  }
  if(index==1)
  {
    return _b_proj;
  }
  THROW_CAIFE("DeviceTabularEmbedding: parameter index out of range");
}

CAIF_DeviceTensor &CAIF_DeviceTabularEmbedding::GradientTensor(size_t index)
{
  if(index==0)
  {
    return _grad_w_proj;
  }
  if(index==1)
  {
    return _grad_b_proj;
  }
  THROW_CAIFE("DeviceTabularEmbedding: gradient index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceTabularEmbedding::GradientTensor(size_t index)const
{
  if(index==0)
  {
    return _grad_w_proj;
  }
  if(index==1)
  {
    return _grad_b_proj;
  }
  THROW_CAIFE("DeviceTabularEmbedding: gradient index out of range");
}

size_t CAIF_DeviceTabularEmbedding::TotalParameterCount()const
{
  return _config.num_features*_config.dim+_config.dim;
}

std::string CAIF_DeviceTabularEmbedding::Description()const
{
  try
  {
    return "TabularEmbedding(features="+std::to_string(_config.num_features)+
           ",dim="+std::to_string(_config.dim)+")";
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceTabularEmbedding::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"proj.weight");
    names.push_back(prefix+"proj.bias");
    return names;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceTabularEmbedding::InitializeWeights(uint32_t seed)
{
  try
  {
    std::mt19937 rng(seed);

    // Xavier uniform for W_proj
    const float limit=std::sqrt(6.0f/static_cast<float>(_config.num_features+_config.dim));
    std::uniform_real_distribution<float> dist(-limit,limit);

    std::vector<float> w_data(_config.num_features*_config.dim);
    for(size_t i=0;i<w_data.size();++i)
    {
      w_data[i]=dist(rng);
    }
    _w_proj.CopyFromHost(w_data.data(),w_data.size());

    // Bias initialized to zero (already done)
  }
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
