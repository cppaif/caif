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

#include "caif_device_rmsnorm.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>

namespace instance
{

CAIF_DeviceRMSNorm::CAIF_DeviceRMSNorm(uint32_t dim,
                                      CAIF_CudaStream &stream,
                                      float epsilon):CAIF_DeviceLayer(stream),
                                                      _dim(dim),
                                                      _epsilon(epsilon),
                                                      _gamma(),
                                                      _gamma_grad(),
                                                      _last_input(),
                                                      _rms_cache()
{
  try
  {
    if(dim==0)
    {
      THROW_CAIFE("DeviceRMSNorm: dim must be > 0");
    }

    // Allocate gamma initialized to 1.0
    _gamma=CAIF_DeviceTensor::Uninitialized({dim},stream);
    std::vector<float> ones(dim,1.0f);
    _gamma.CopyFromHost(ones.data(),dim);

    _gamma_grad=CAIF_DeviceTensor::Zeros({dim},stream);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceRMSNorm::CAIF_DeviceRMSNorm(CAIF_DeviceRMSNorm &&other):CAIF_DeviceLayer(std::move(other)),
                                                                  _dim(other._dim),
                                                                  _epsilon(other._epsilon),
                                                                  _gamma(std::move(other._gamma)),
                                                                  _gamma_grad(std::move(other._gamma_grad)),
                                                                  _last_input(std::move(other._last_input)),
                                                                  _rms_cache(std::move(other._rms_cache))
{
}

CAIF_DeviceRMSNorm &CAIF_DeviceRMSNorm::operator=(CAIF_DeviceRMSNorm &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _dim=other._dim;
      _epsilon=other._epsilon;
      _gamma=std::move(other._gamma);
      _gamma_grad=std::move(other._gamma_grad);
      _last_input=std::move(other._last_input);
      _rms_cache=std::move(other._rms_cache);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceRMSNorm::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceRMSNorm: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.empty()==true)
    {
      THROW_CAIFE("DeviceRMSNorm::Forward: input tensor is empty");
    }
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("DeviceRMSNorm::Forward: last dimension must match dim");
    }

    // Flatten leading dimensions to rows
    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(shape,*_stream);
    CAIF_DeviceTensor rms_cache=CAIF_DeviceTensor::Uninitialized(
                                 {static_cast<uint32_t>(rows)},*_stream);

    launch_rmsnorm_forward(input.DevicePtr(),
                           _gamma.DevicePtr(),
                           output.DevicePtr(),
                           rms_cache.DevicePtr(),
                           _epsilon,
                           rows,
                           static_cast<int>(_dim),
                           _stream->Handle());

    if(training==true)
    {
      _last_input=input.Clone();
      _rms_cache=std::move(rms_cache);
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceRMSNorm::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceRMSNorm: layer has been moved from");
    }
    if(_last_input.IsEmpty()==true)
    {
      THROW_CAIFE("DeviceRMSNorm::Backward: must call Forward with training=true first");
    }

    const auto &shape=grad_output.Shape();
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("DeviceRMSNorm::Backward: last dimension must match dim");
    }

    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized(shape,*_stream);

    // Zero grad_gamma before atomicAdd accumulation
    _gamma_grad.Fill(0.0f);

    launch_rmsnorm_backward(grad_output.DevicePtr(),
                            _last_input.DevicePtr(),
                            _gamma.DevicePtr(),
                            _rms_cache.DevicePtr(),
                            grad_input.DevicePtr(),
                            _gamma_grad.DevicePtr(),
                            _epsilon,
                            rows,
                            static_cast<int>(_dim),
                            _stream->Handle());

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceRMSNorm::ZeroGradients()
{
  try
  {
    _gamma_grad.Fill(0.0f);
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceRMSNorm::ParameterTensorCount()const
{
  try
  {
    return 1;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceRMSNorm::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    THROW_CAIFE("DeviceRMSNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceRMSNorm::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    THROW_CAIFE("DeviceRMSNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceRMSNorm::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    THROW_CAIFE("DeviceRMSNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceRMSNorm::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    THROW_CAIFE("DeviceRMSNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceRMSNorm::TotalParameterCount()const
{
  try
  {
    return static_cast<size_t>(_dim);
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceRMSNorm::Description()const
{
  try
  {
    return "RMSNorm("+std::to_string(_dim)+")";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceRMSNorm::ParameterNames(const std::string &prefix)const
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
