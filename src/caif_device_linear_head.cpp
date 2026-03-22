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

#include "caif_device_linear_head.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <cmath>
#include <vector>

namespace instance
{

CAIF_DeviceLinearHead::CAIF_DeviceLinearHead(const CAIF_DeviceLinearHead::Config_t &config,
                                           CAIF_CudaStream &stream):
                                           CAIF_DeviceLayer(stream),
                                           _config(config),
                                           _weight_tied(false),
                                           _frozen(false),
                                           _weight(),
                                           _weight_grad(),
                                           _tied_weight(nullptr),
                                           _tied_weight_grad(nullptr),
                                           _bias(),
                                           _bias_grad(),
                                           _cached_input(),
                                           _cached_shape()
{
  try
  {
    if(config.input_dim==0)
    {
      THROW_CAIFE("DeviceLinearHead: input_dim must be > 0");
    }
    if(config.output_dim==0)
    {
      THROW_CAIFE("DeviceLinearHead: output_dim must be > 0");
    }

    // Xavier uniform init for weight
    const float limit=std::sqrt(6.0f/static_cast<float>(config.input_dim+config.output_dim));
    const size_t weight_size=static_cast<size_t>(config.input_dim)*config.output_dim;
    std::vector<float> w_init(weight_size);
    for(size_t i=0;i<weight_size;++i)
    {
      const float t=static_cast<float>(i)*0.6180339887f;
      w_init[i]=(t-std::floor(t))*2.0f*limit-limit;
    }

    _weight=CAIF_DeviceTensor::Uninitialized({config.input_dim,config.output_dim},stream);
    _weight.CopyFromHost(w_init.data(),weight_size);
    _weight_grad=CAIF_DeviceTensor::Zeros({config.input_dim,config.output_dim},stream);

    if(config.use_bias==true)
    {
      _bias=CAIF_DeviceTensor::Zeros({config.output_dim},stream);
      _bias_grad=CAIF_DeviceTensor::Zeros({config.output_dim},stream);
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceLinearHead::CAIF_DeviceLinearHead(const CAIF_DeviceLinearHead::Config_t &config,
                                           CAIF_DeviceTensor &tied_weight,
                                           CAIF_DeviceTensor &tied_weight_grad,
                                           CAIF_CudaStream &stream):
                                           CAIF_DeviceLayer(stream),
                                           _config(config),
                                           _weight_tied(true),
                                           _frozen(false),
                                           _weight(),
                                           _weight_grad(),
                                           _tied_weight(&tied_weight),
                                           _tied_weight_grad(&tied_weight_grad),
                                           _bias(),
                                           _bias_grad(),
                                           _cached_input(),
                                           _cached_shape()
{
  try
  {
    if(config.input_dim==0)
    {
      THROW_CAIFE("DeviceLinearHead: input_dim must be > 0");
    }
    if(config.output_dim==0)
    {
      THROW_CAIFE("DeviceLinearHead: output_dim must be > 0");
    }

    // Verify tied weight shape: should be [output_dim, input_dim] (embedding table)
    const auto &tied_shape=tied_weight.Shape();
    if(tied_shape.size()!=2)
    {
      THROW_CAIFE("DeviceLinearHead: tied weight must be 2D");
    }
    if(tied_shape[0]!=config.output_dim||tied_shape[1]!=config.input_dim)
    {
      THROW_CAIFE("DeviceLinearHead: tied weight shape must be [output_dim, input_dim]");
    }

    if(config.use_bias==true)
    {
      _bias=CAIF_DeviceTensor::Zeros({config.output_dim},stream);
      _bias_grad=CAIF_DeviceTensor::Zeros({config.output_dim},stream);
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceLinearHead::CAIF_DeviceLinearHead(CAIF_DeviceLinearHead &&other)noexcept:
                                           CAIF_DeviceLayer(std::move(other)),
                                           _config(other._config),
                                           _weight_tied(other._weight_tied),
                                           _frozen(other._frozen),
                                           _weight(std::move(other._weight)),
                                           _weight_grad(std::move(other._weight_grad)),
                                           _tied_weight(other._tied_weight),
                                           _tied_weight_grad(other._tied_weight_grad),
                                           _bias(std::move(other._bias)),
                                           _bias_grad(std::move(other._bias_grad)),
                                           _cached_input(std::move(other._cached_input)),
                                           _cached_shape(std::move(other._cached_shape))
{
  other._tied_weight=nullptr;
  other._tied_weight_grad=nullptr;
}

CAIF_DeviceLinearHead &CAIF_DeviceLinearHead::operator=(CAIF_DeviceLinearHead &&other)noexcept
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _weight_tied=other._weight_tied;
    _frozen=other._frozen;
    _weight=std::move(other._weight);
    _weight_grad=std::move(other._weight_grad);
    _tied_weight=other._tied_weight;
    _tied_weight_grad=other._tied_weight_grad;
    _bias=std::move(other._bias);
    _bias_grad=std::move(other._bias_grad);
    _cached_input=std::move(other._cached_input);
    _cached_shape=std::move(other._cached_shape);
    other._tied_weight=nullptr;
    other._tied_weight_grad=nullptr;
  }
  return *this;
}

CAIF_DeviceTensor CAIF_DeviceLinearHead::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceLinearHead: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.empty()==true)
    {
      THROW_CAIFE("DeviceLinearHead::Forward: input shape is empty");
    }
    if(shape.back()!=_config.input_dim)
    {
      THROW_CAIFE("DeviceLinearHead::Forward: last dim must match input_dim");
    }

    // Flatten leading dims: [d0, d1, ..., input_dim] -> [N, input_dim]
    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }

    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({n,_config.input_dim});

    // MatMul: [N, input_dim] @ [input_dim, output_dim] = [N, output_dim]
    // Or for tied weights: [N, input_dim] @ [output_dim, input_dim]^T = [N, output_dim]
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({n,_config.output_dim},*_stream);

    if(_weight_tied==true)
    {
      // Tied: output = input @ tied_weight^T
      CAIF_DeviceOps::MatMulTransposeB(flat_input,*_tied_weight,output);
    }
    else
    {
      // Untied: output = input @ weight
      CAIF_DeviceOps::MatMul(flat_input,_weight,output);
    }

    // Add bias if enabled
    if(_config.use_bias==true)
    {
      CAIF_DeviceOps::BiasAdd(output,_bias,output);
    }

    // Reshape to output shape
    std::vector<uint32_t> out_shape;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      out_shape.push_back(shape[i]);
    }
    out_shape.push_back(_config.output_dim);
    output.Reshape(out_shape);

    // Cache for backward
    if(training==true)
    {
      _cached_input=input.Clone();
      _cached_shape=std::vector<uint32_t>(shape.begin(),shape.end());
    }

    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceLinearHead::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceLinearHead: layer has been moved from");
    }
    if(_cached_shape.empty()==true)
    {
      THROW_CAIFE("DeviceLinearHead::Backward: must call Forward with training=true first");
    }

    // Flatten grad_output: [..., output_dim] -> [N, output_dim]
    const auto &grad_shape=grad_output.Shape();
    uint32_t n=1;
    for(size_t i=0;i<grad_shape.size()-1;++i)
    {
      n*=grad_shape[i];
    }

    CAIF_DeviceTensor flat_grad=grad_output.Clone();
    flat_grad.Reshape({n,_config.output_dim});

    // Flatten cached input
    CAIF_DeviceTensor flat_input=_cached_input.Clone();
    flat_input.Reshape({n,_config.input_dim});

    // Weight gradient and input gradient
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({n,_config.input_dim},*_stream);

    if(_frozen==true)
    {
      // Frozen: skip weight/bias gradient, only compute grad_input
      if(_weight_tied==true)
      {
        CAIF_DeviceOps::MatMul(flat_grad,*_tied_weight,grad_input);
      }
      else
      {
        CAIF_DeviceOps::MatMulTransposeB(flat_grad,_weight,grad_input);
      }
    }
    else
    {
      // Bias gradient: sum over batch dimension
      if(_config.use_bias==true)
      {
        CAIF_DeviceOps::BiasGradient(flat_grad,_bias_grad);
      }

      if(_weight_tied==true)
      {
        // Tied: forward was input @ tied_weight^T
        CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized(
          {_config.output_dim,_config.input_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(flat_grad,flat_input,grad_w_delta);
        CAIF_DeviceOps::Add(*_tied_weight_grad,grad_w_delta,*_tied_weight_grad);

        // grad_input = grad_output @ tied_weight
        CAIF_DeviceOps::MatMul(flat_grad,*_tied_weight,grad_input);
      }
      else
      {
        // Untied: forward was input @ weight
        CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized(
          {_config.input_dim,_config.output_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(flat_input,flat_grad,grad_w_delta);
        CAIF_DeviceOps::Add(_weight_grad,grad_w_delta,_weight_grad);

        // grad_input = grad_output @ weight^T
        CAIF_DeviceOps::MatMulTransposeB(flat_grad,_weight,grad_input);
      }
    }

    // Reshape grad_input to original input shape
    grad_input.Reshape(_cached_shape);

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceLinearHead::ZeroGradients()
{
  try
  {
    if(_weight_tied==false)
    {
      _weight_grad.Fill(0.0f);
    }
    // Note: tied weight grad is zeroed by embedding layer, not here

    if(_config.use_bias==true)
    {
      _bias_grad.Fill(0.0f);
    }
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceLinearHead::ParameterTensorCount()const
{
  try
  {
    size_t count=0;
    if(_weight_tied==false)
    {
      count+=1;  // weight
    }
    if(_config.use_bias==true)
    {
      count+=1;  // bias
    }
    return count;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceLinearHead::ParameterTensor(size_t index)
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    THROW_CAIFE("DeviceLinearHead::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceLinearHead::ParameterTensor(size_t index)const
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    THROW_CAIFE("DeviceLinearHead::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceLinearHead::GradientTensor(size_t index)
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight_grad;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    THROW_CAIFE("DeviceLinearHead::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceLinearHead::GradientTensor(size_t index)const
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight_grad;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    THROW_CAIFE("DeviceLinearHead::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceLinearHead::TotalParameterCount()const
{
  try
  {
    size_t total=0;
    if(_weight_tied==false)
    {
      total+=static_cast<size_t>(_config.input_dim)*_config.output_dim;
    }
    if(_config.use_bias==true)
    {
      total+=_config.output_dim;
    }
    return total;
  }
  CCAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceLinearHead::Description()const
{
  try
  {
    std::string desc="LinearHead(in="+std::to_string(_config.input_dim)+
                     ",out="+std::to_string(_config.output_dim)+")";
    if(_weight_tied==true)
    {
      desc="LinearHead(in="+std::to_string(_config.input_dim)+
           ",out="+std::to_string(_config.output_dim)+
           ",tied=true)";
    }
    return desc;
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceLinearHead::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    if(_weight_tied==false)
    {
      names.push_back(prefix+"weight");
    }
    if(_config.use_bias==true)
    {
      names.push_back(prefix+"bias");
    }
    return names;
  }
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
