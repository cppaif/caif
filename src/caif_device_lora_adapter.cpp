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

#include "caif_device_lora_adapter.h"
#include "caif_device_ops.h"
#include "caif_host_tensor.h"
#include "caif_exception.h"
#include <random>
#include <cmath>
#include <ctime>

using namespace instance;

CAIF_DeviceLoRAAdapter::CAIF_DeviceLoRAAdapter(const LoRAConfig_t &config,
                                             std::unique_ptr<CAIF_DeviceLayer> base_layer,
                                             CAIF_CudaStream &stream,
                                             uint32_t seed):CAIF_DeviceLayer(stream),
                                                            _config(config),
                                                            _base_layer(std::move(base_layer)),
                                                            _lora_a(),
                                                            _lora_b(),
                                                            _grad_lora_a(),
                                                            _grad_lora_b(),
                                                            _cached_input(),
                                                            _cached_lora_hidden()
{
  try
  {
    if(config.rank==0||config.input_dim==0||config.output_dim==0)
    {
      THROW_CAIFE("LoRAAdapter: rank, input_dim, output_dim must be > 0");
    }
    if(_base_layer==nullptr)
    {
      THROW_CAIFE("LoRAAdapter: base_layer must not be null");
    }

    // Allocate LoRA tensors: A=[rank, input_dim], B=[output_dim, rank]
    _lora_a=CAIF_DeviceTensor::Uninitialized({config.rank,config.input_dim},stream);
    _lora_b=CAIF_DeviceTensor::Zeros({config.output_dim,config.rank},stream);

    _grad_lora_a=CAIF_DeviceTensor::Zeros({config.rank,config.input_dim},stream);
    _grad_lora_b=CAIF_DeviceTensor::Zeros({config.output_dim,config.rank},stream);

    // Initialize A with Kaiming uniform: U(-bound, bound), bound=sqrt(1/input_dim)
    if(seed==0)
    {
      seed=static_cast<uint32_t>(std::time(nullptr));
    }
    const float bound=std::sqrt(1.0f/static_cast<float>(config.input_dim));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-bound,bound);

    const size_t a_count=static_cast<size_t>(config.rank)*config.input_dim;
    std::vector<float> host_a(a_count);
    for(size_t i=0;i<a_count;++i)
    {
      host_a[i]=dist(gen);
    }
    _lora_a.CopyFromHost(host_a.data(),a_count);

    // B is already zeros (LoRA starts as identity)
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceLoRAAdapter::CAIF_DeviceLoRAAdapter(CAIF_DeviceLoRAAdapter &&other):CAIF_DeviceLayer(std::move(other)),
                                                      _config(other._config),
                                                      _base_layer(std::move(other._base_layer)),
                                                      _lora_a(std::move(other._lora_a)),
                                                      _lora_b(std::move(other._lora_b)),
                                                      _grad_lora_a(std::move(other._grad_lora_a)),
                                                      _grad_lora_b(std::move(other._grad_lora_b)),
                                                      _cached_input(std::move(other._cached_input)),
                                                      _cached_lora_hidden(std::move(other._cached_lora_hidden))
{
}

CAIF_DeviceLoRAAdapter &CAIF_DeviceLoRAAdapter::operator=(CAIF_DeviceLoRAAdapter &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _config=other._config;
      _base_layer=std::move(other._base_layer);
      _lora_a=std::move(other._lora_a);
      _lora_b=std::move(other._lora_b);
      _grad_lora_a=std::move(other._grad_lora_a);
      _grad_lora_b=std::move(other._grad_lora_b);
      _cached_input=std::move(other._cached_input);
      _cached_lora_hidden=std::move(other._cached_lora_hidden);
    }
    return *this;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceLoRAAdapter::Forward(const CAIF_DeviceTensor &input,
                                                 bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("LoRAAdapter: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("LoRAAdapter::Forward: input must be at least 2D");
    }
    if(shape.back()!=_config.input_dim)
    {
      THROW_CAIFE("LoRAAdapter::Forward: last dim must match input_dim");
    }

    // Base layer forward
    CAIF_DeviceTensor base_out=_base_layer->Forward(input,training);

    // Reshape to 2D for LoRA matmuls
    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }
    CAIF_DeviceTensor input_2d=input.Clone();
    input_2d.Reshape({n,_config.input_dim});

    // Cache for backward
    if(training==true)
    {
      _cached_input=input_2d.Clone();
    }

    // lora_hidden = input_2d @ A^T: [N, input_dim] @ [input_dim, rank] = [N, rank]
    CAIF_DeviceTensor lora_hidden=CAIF_DeviceTensor::Uninitialized({n,_config.rank},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(input_2d,_lora_a,lora_hidden);

    // Cache for backward
    if(training==true)
    {
      _cached_lora_hidden=lora_hidden.Clone();
    }

    // lora_out = lora_hidden @ B^T: [N, rank] @ [rank, output_dim] = [N, output_dim]
    CAIF_DeviceTensor lora_out=CAIF_DeviceTensor::Uninitialized({n,_config.output_dim},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(lora_hidden,_lora_b,lora_out);

    // Scale by alpha/rank
    const float scale=_config.alpha/static_cast<float>(_config.rank);
    CAIF_DeviceOps::Scale(lora_out,scale);

    // Reshape lora_out to match base_out shape
    if(shape.size()>2)
    {
      std::vector<uint32_t> out_shape(shape.begin(),shape.end()-1);
      out_shape.push_back(_config.output_dim);
      lora_out.Reshape(out_shape);
    }

    // output = base_out + lora_out
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(base_out.Shape(),*_stream);
    CAIF_DeviceOps::Add(base_out,lora_out,output);

    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceLoRAAdapter::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("LoRAAdapter: layer has been moved from");
    }
    if(_cached_input.IsEmpty()==true)
    {
      THROW_CAIFE("LoRAAdapter::Backward: must call Forward with training=true first");
    }

    const auto &shape=grad_output.Shape();

    // Reshape to 2D
    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }
    CAIF_DeviceTensor grad_2d=grad_output.Clone();
    grad_2d.Reshape({n,_config.output_dim});

    const float scale=_config.alpha/static_cast<float>(_config.rank);

    // d_lora_scaled = grad_output * scale
    CAIF_DeviceTensor d_lora_scaled=grad_2d.Clone();
    CAIF_DeviceOps::Scale(d_lora_scaled,scale);

    // _grad_lora_b += d_lora_scaled^T @ _cached_lora_hidden
    // [output_dim, N] @ [N, rank] = [output_dim, rank]
    CAIF_DeviceOps::MatMulTransposeA(d_lora_scaled,_cached_lora_hidden,_grad_lora_b);

    // d_lora_hidden = d_lora_scaled @ B: [N, output_dim] @ [output_dim, rank] = [N, rank]
    CAIF_DeviceTensor d_lora_hidden=CAIF_DeviceTensor::Uninitialized({n,_config.rank},*_stream);
    CAIF_DeviceOps::MatMul(d_lora_scaled,_lora_b,d_lora_hidden);

    // _grad_lora_a += d_lora_hidden^T @ _cached_input
    // [rank, N] @ [N, input_dim] = [rank, input_dim]
    CAIF_DeviceOps::MatMulTransposeA(d_lora_hidden,_cached_input,_grad_lora_a);

    // d_input_lora = d_lora_hidden @ A: [N, rank] @ [rank, input_dim] = [N, input_dim]
    CAIF_DeviceTensor d_input_lora=CAIF_DeviceTensor::Uninitialized({n,_config.input_dim},*_stream);
    CAIF_DeviceOps::MatMul(d_lora_hidden,_lora_a,d_input_lora);

    // Base layer backward
    CAIF_DeviceTensor grad_base=_base_layer->Backward(grad_output);

    // Reshape d_input_lora to match grad_base shape
    if(shape.size()>2)
    {
      std::vector<uint32_t> in_shape(shape.begin(),shape.end()-1);
      in_shape.push_back(_config.input_dim);
      d_input_lora.Reshape(in_shape);
    }

    // grad_input = grad_base + d_input_lora
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized(
        grad_base.Shape(),*_stream);
    CAIF_DeviceOps::Add(grad_base,d_input_lora,grad_input);

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceLoRAAdapter::ZeroGradients()
{
  try
  {
    _grad_lora_a.Fill(0.0f);
    _grad_lora_b.Fill(0.0f);
    _base_layer->ZeroGradients();
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceLoRAAdapter::ParameterTensorCount()const
{
  return 2;
}

CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _lora_a;
    }
    if(index==1)
    {
      return _lora_b;
    }
    THROW_CAIFE("LoRAAdapter::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _lora_a;
    }
    if(index==1)
    {
      return _lora_b;
    }
    THROW_CAIFE("LoRAAdapter::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _grad_lora_a;
    }
    if(index==1)
    {
      return _grad_lora_b;
    }
    THROW_CAIFE("LoRAAdapter::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _grad_lora_a;
    }
    if(index==1)
    {
      return _grad_lora_b;
    }
    THROW_CAIFE("LoRAAdapter::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceLoRAAdapter::TotalParameterCount()const
{
  try
  {
    return static_cast<size_t>(_config.rank)*_config.input_dim+
           static_cast<size_t>(_config.output_dim)*_config.rank;
  }
  CCAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceLoRAAdapter::Description()const
{
  try
  {
    std::string desc="LoRA(rank="+std::to_string(_config.rank)+
                     ",alpha="+std::to_string(static_cast<int>(_config.alpha))+
                     ","+std::to_string(_config.input_dim)+
                     ","+std::to_string(_config.output_dim)+")";
    if(_base_layer!=nullptr)
    {
      desc+="+"+_base_layer->Description();
    }
    return desc;
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceLoRAAdapter::ParameterNames(
    const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"lora_a.weight");
    names.push_back(prefix+"lora_b.weight");
    return names;
  }
  CCAIF_CATCH_BLOCK()
}
