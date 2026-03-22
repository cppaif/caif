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

#include "caif_device_layernorm.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>

namespace instance
{

CAIF_DeviceLayerNorm::CAIF_DeviceLayerNorm(uint32_t dim,
                                          CAIF_CudaStream &stream,
                                          float epsilon):CAIF_DeviceLayer(stream),
                                                          _dim(dim),
                                                          _epsilon(epsilon),
                                                          _gamma(),
                                                          _beta(),
                                                          _gamma_grad(),
                                                          _beta_grad(),
                                                          _cached_rows(0),
                                                          _last_input(),
                                                          _mean_cache(),
                                                          _rstd_cache()
{
  try
  {
    if(dim==0)
    {
      THROW_CAIFE("DeviceLayerNorm: dim must be > 0");
    }

    // Allocate gamma initialized to 1.0
    _gamma=CAIF_DeviceTensor::Uninitialized({dim},stream);
    std::vector<float> ones(dim,1.0f);
    _gamma.CopyFromHost(ones.data(),dim);

    // Allocate beta initialized to 0.0
    _beta=CAIF_DeviceTensor::Zeros({dim},stream);

    _gamma_grad=CAIF_DeviceTensor::Zeros({dim},stream);
    _beta_grad=CAIF_DeviceTensor::Zeros({dim},stream);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceLayerNorm::CAIF_DeviceLayerNorm(CAIF_DeviceLayerNorm &&other):CAIF_DeviceLayer(std::move(other)),
                                                                       _dim(other._dim),
                                                                       _epsilon(other._epsilon),
                                                                       _gamma(std::move(other._gamma)),
                                                                       _beta(std::move(other._beta)),
                                                                       _gamma_grad(std::move(other._gamma_grad)),
                                                                       _beta_grad(std::move(other._beta_grad)),
                                                                       _cached_rows(other._cached_rows),
                                                                       _last_input(std::move(other._last_input)),
                                                                       _mean_cache(std::move(other._mean_cache)),
                                                                       _rstd_cache(std::move(other._rstd_cache))
{
  other._cached_rows=0;
}

CAIF_DeviceLayerNorm &CAIF_DeviceLayerNorm::operator=(CAIF_DeviceLayerNorm &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _dim=other._dim;
      _epsilon=other._epsilon;
      _gamma=std::move(other._gamma);
      _beta=std::move(other._beta);
      _gamma_grad=std::move(other._gamma_grad);
      _beta_grad=std::move(other._beta_grad);
      _cached_rows=other._cached_rows;
      other._cached_rows=0;
      _last_input=std::move(other._last_input);
      _mean_cache=std::move(other._mean_cache);
      _rstd_cache=std::move(other._rstd_cache);
    }
    return *this;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceLayerNorm::Forward(const CAIF_DeviceTensor &input,
                                               bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceLayerNorm: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.empty()==true)
    {
      THROW_CAIFE("DeviceLayerNorm::Forward: input tensor is empty");
    }
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("DeviceLayerNorm::Forward: last dimension must match dim");
    }

    // Flatten leading dimensions to rows
    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    // Pre-allocate mean/rstd caches, reuse when row count matches
    if(rows!=_cached_rows)
    {
      _mean_cache=CAIF_DeviceTensor::Uninitialized(
                    {static_cast<uint32_t>(rows)},*_stream);
      _rstd_cache=CAIF_DeviceTensor::Uninitialized(
                    {static_cast<uint32_t>(rows)},*_stream);
      _cached_rows=rows;
    }

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(shape,*_stream);

    launch_layernorm_forward(input.DevicePtr(),
                             _gamma.DevicePtr(),
                             _beta.DevicePtr(),
                             output.DevicePtr(),
                             _mean_cache.DevicePtr(),
                             _rstd_cache.DevicePtr(),
                             _epsilon,
                             rows,
                             static_cast<int>(_dim),
                             _stream->Handle());

    if(training==true)
    {
      _last_input=input.Clone();
    }

    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceLayerNorm::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceLayerNorm: layer has been moved from");
    }
    if(_last_input.IsEmpty()==true)
    {
      THROW_CAIFE("DeviceLayerNorm::Backward: must call Forward with training=true first");
    }

    const auto &shape=grad_output.Shape();
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("DeviceLayerNorm::Backward: last dimension must match dim");
    }

    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized(shape,*_stream);

    // Zero grad_gamma and grad_beta before atomicAdd accumulation
    _gamma_grad.Fill(0.0f);
    _beta_grad.Fill(0.0f);

    launch_layernorm_backward(grad_output.DevicePtr(),
                              _last_input.DevicePtr(),
                              _gamma.DevicePtr(),
                              _mean_cache.DevicePtr(),
                              _rstd_cache.DevicePtr(),
                              grad_input.DevicePtr(),
                              _gamma_grad.DevicePtr(),
                              _beta_grad.DevicePtr(),
                              rows,
                              static_cast<int>(_dim),
                              _stream->Handle());

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceLayerNorm::ZeroGradients()
{
  try
  {
    _gamma_grad.Fill(0.0f);
    _beta_grad.Fill(0.0f);
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceLayerNorm::ParameterTensorCount()const
{
  try
  {
    return 2;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceLayerNorm::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    if(index==1)
    {
      return _beta;
    }
    THROW_CAIFE("DeviceLayerNorm::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceLayerNorm::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    if(index==1)
    {
      return _beta;
    }
    THROW_CAIFE("DeviceLayerNorm::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceLayerNorm::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    if(index==1)
    {
      return _beta_grad;
    }
    THROW_CAIFE("DeviceLayerNorm::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceLayerNorm::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    if(index==1)
    {
      return _beta_grad;
    }
    THROW_CAIFE("DeviceLayerNorm::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceLayerNorm::TotalParameterCount()const
{
  try
  {
    return static_cast<size_t>(_dim)*2;
  }
  CCAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceLayerNorm::Description()const
{
  try
  {
    return "LayerNorm("+std::to_string(_dim)+")";
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceLayerNorm::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"weight");
    names.push_back(prefix+"bias");
    return names;
  }
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
