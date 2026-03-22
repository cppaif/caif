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
// Device-resident MoE Expert implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_expert.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <random>
#include <cmath>

using namespace instance;

CAIF_DeviceMoEExpert::CAIF_DeviceMoEExpert(const Config_t &config,CAIF_CudaStream &stream)
  :CAIF_DeviceLayer(stream)
  ,_config(config)
  ,_use_projections(false)
{
  try
  {
    // Validate config
    if(_config.input_dim==0)
    {
      THROW_CAIFE("MoEExpert: input_dim must be > 0");
    }
    if(_config.hidden_dim==0)
    {
      THROW_CAIFE("MoEExpert: hidden_dim must be > 0");
    }

    // Allocate weights
    if(_config.use_gated==true)
    {
      _w_gate=CAIF_DeviceTensor::Uninitialized({_config.input_dim,_config.hidden_dim},stream);
      _grad_w_gate=CAIF_DeviceTensor::Zeros({_config.input_dim,_config.hidden_dim},stream);

      if(_config.use_bias==true)
      {
        _b_gate=CAIF_DeviceTensor::Zeros({_config.hidden_dim},stream);
        _grad_b_gate=CAIF_DeviceTensor::Zeros({_config.hidden_dim},stream);
      }
    }

    _w_up=CAIF_DeviceTensor::Uninitialized({_config.input_dim,_config.hidden_dim},stream);
    _w_down=CAIF_DeviceTensor::Uninitialized({_config.hidden_dim,_config.input_dim},stream);
    _grad_w_up=CAIF_DeviceTensor::Zeros({_config.input_dim,_config.hidden_dim},stream);
    _grad_w_down=CAIF_DeviceTensor::Zeros({_config.hidden_dim,_config.input_dim},stream);

    if(_config.use_bias==true)
    {
      _b_up=CAIF_DeviceTensor::Zeros({_config.hidden_dim},stream);
      _b_down=CAIF_DeviceTensor::Zeros({_config.input_dim},stream);
      _grad_b_up=CAIF_DeviceTensor::Zeros({_config.hidden_dim},stream);
      _grad_b_down=CAIF_DeviceTensor::Zeros({_config.input_dim},stream);
    }

    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    float scale_up=std::sqrt(2.0f/static_cast<float>(_config.input_dim+_config.hidden_dim));
    float scale_down=std::sqrt(2.0f/static_cast<float>(_config.hidden_dim+_config.input_dim));

    std::normal_distribution<float> dist_up(0.0f,scale_up);
    std::normal_distribution<float> dist_down(0.0f,scale_down);

    // Initialize w_up
    {
      std::vector<float> data(_config.input_dim*_config.hidden_dim);
      for(size_t i=0;i<data.size();++i)
      {
        data[i]=dist_up(gen);
      }
      _w_up.CopyFromHost(data.data(),data.size());
    }

    // Initialize w_down
    {
      std::vector<float> data(_config.hidden_dim*_config.input_dim);
      for(size_t i=0;i<data.size();++i)
      {
        data[i]=dist_down(gen);
      }
      _w_down.CopyFromHost(data.data(),data.size());
    }

    // Initialize w_gate if gated
    if(_config.use_gated==true)
    {
      std::vector<float> data(_config.input_dim*_config.hidden_dim);
      for(size_t i=0;i<data.size();++i)
      {
        data[i]=dist_up(gen);
      }
      _w_gate.CopyFromHost(data.data(),data.size());
    }

    _stream->Synchronize();
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceMoEExpert::CAIF_DeviceMoEExpert(const Config_t &config,
                                         MoEExpertProjections_t projections,
                                         CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                                  _config(config),
                                                                  _projections(std::move(projections)),
                                                                  _use_projections(true)
{
  try
  {
    if(_config.input_dim==0)
    {
      THROW_CAIFE("MoEExpert: input_dim must be > 0");
    }
    if(_config.hidden_dim==0)
    {
      THROW_CAIFE("MoEExpert: hidden_dim must be > 0");
    }
    if(_projections.up==nullptr)
    {
      THROW_CAIFE("MoEExpert: up projection must not be null");
    }
    if(_projections.down==nullptr)
    {
      THROW_CAIFE("MoEExpert: down projection must not be null");
    }
    if(_config.use_gated==true&&_projections.gate==nullptr)
    {
      THROW_CAIFE("MoEExpert: gate projection required for gated mode");
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceMoEExpert::CAIF_DeviceMoEExpert(CAIF_DeviceMoEExpert &&other)
  :CAIF_DeviceLayer(std::move(other))
  ,_config(other._config)
  ,_projections(std::move(other._projections))
  ,_use_projections(other._use_projections)
  ,_w_gate(std::move(other._w_gate))
  ,_w_up(std::move(other._w_up))
  ,_w_down(std::move(other._w_down))
  ,_b_gate(std::move(other._b_gate))
  ,_b_up(std::move(other._b_up))
  ,_b_down(std::move(other._b_down))
  ,_grad_w_gate(std::move(other._grad_w_gate))
  ,_grad_w_up(std::move(other._grad_w_up))
  ,_grad_w_down(std::move(other._grad_w_down))
  ,_grad_b_gate(std::move(other._grad_b_gate))
  ,_grad_b_up(std::move(other._grad_b_up))
  ,_grad_b_down(std::move(other._grad_b_down))
  ,_cached_input(std::move(other._cached_input))
  ,_cached_gate_out(std::move(other._cached_gate_out))
  ,_cached_up_out(std::move(other._cached_up_out))
  ,_cached_hidden(std::move(other._cached_hidden))
{
}

CAIF_DeviceMoEExpert &CAIF_DeviceMoEExpert::operator=(CAIF_DeviceMoEExpert &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _projections=std::move(other._projections);
    _use_projections=other._use_projections;
    _w_gate=std::move(other._w_gate);
    _w_up=std::move(other._w_up);
    _w_down=std::move(other._w_down);
    _b_gate=std::move(other._b_gate);
    _b_up=std::move(other._b_up);
    _b_down=std::move(other._b_down);
    _grad_w_gate=std::move(other._grad_w_gate);
    _grad_w_up=std::move(other._grad_w_up);
    _grad_w_down=std::move(other._grad_w_down);
    _grad_b_gate=std::move(other._grad_b_gate);
    _grad_b_up=std::move(other._grad_b_up);
    _grad_b_down=std::move(other._grad_b_down);
    _cached_input=std::move(other._cached_input);
    _cached_gate_out=std::move(other._cached_gate_out);
    _cached_up_out=std::move(other._cached_up_out);
    _cached_hidden=std::move(other._cached_hidden);
  }
  return *this;
}

CAIF_DeviceTensor CAIF_DeviceMoEExpert::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    // Input: [num_tokens, input_dim]
    const auto &shape=input.Shape();
    if(shape.size()!=2||shape[1]!=_config.input_dim)
    {
      THROW_CAIFE("MoEExpert::Forward: expected input shape [N, input_dim]");
    }

    const uint32_t num_tokens=shape[0];

    // Cache input for backward (only when not using projections)
    if(training==true&&_use_projections==false)
    {
      _cached_input=input.Clone();
    }

    if(_config.use_gated==true)
    {
      // Gated FFN: output = (gate * SiLU(up)) @ down
      CAIF_DeviceTensor gate_out;
      if(_use_projections==true)
      {
        gate_out=_projections.gate->Forward(input,training);
      }
      else
      {
        gate_out=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
        CAIF_DeviceOps::MatMul(input,_w_gate,gate_out);
        if(_config.use_bias==true)
        {
          CAIF_DeviceOps::AddBias(gate_out,_b_gate,gate_out);
        }
      }

      CAIF_DeviceTensor up_out;
      if(_use_projections==true)
      {
        up_out=_projections.up->Forward(input,training);
      }
      else
      {
        up_out=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
        CAIF_DeviceOps::MatMul(input,_w_up,up_out);
        if(_config.use_bias==true)
        {
          CAIF_DeviceOps::AddBias(up_out,_b_up,up_out);
        }
      }

      // SiLU activation on up, then multiply by gate
      CAIF_DeviceTensor up_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::SiLU(up_out,up_activated);

      CAIF_DeviceTensor hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::Multiply(gate_out,up_activated,hidden);

      if(training==true)
      {
        _cached_gate_out=std::move(gate_out);
        _cached_up_out=std::move(up_out);
        _cached_hidden=hidden.Clone();
      }

      // Down projection
      CAIF_DeviceTensor output;
      if(_use_projections==true)
      {
        output=_projections.down->Forward(hidden,training);
      }
      else
      {
        output=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
        CAIF_DeviceOps::MatMul(hidden,_w_down,output);
        if(_config.use_bias==true)
        {
          CAIF_DeviceOps::AddBias(output,_b_down,output);
        }
      }

      return output;
    }
    else
    {
      // Standard FFN: output = SiLU(input @ w_up + b_up) @ w_down + b_down
      CAIF_DeviceTensor up_out;
      if(_use_projections==true)
      {
        up_out=_projections.up->Forward(input,training);
      }
      else
      {
        up_out=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
        CAIF_DeviceOps::MatMul(input,_w_up,up_out);
        if(_config.use_bias==true)
        {
          CAIF_DeviceOps::AddBias(up_out,_b_up,up_out);
        }
      }

      CAIF_DeviceTensor hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::SiLU(up_out,hidden);

      if(training==true)
      {
        _cached_up_out=std::move(up_out);
        _cached_hidden=hidden.Clone();
      }

      CAIF_DeviceTensor output;
      if(_use_projections==true)
      {
        output=_projections.down->Forward(hidden,training);
      }
      else
      {
        output=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
        CAIF_DeviceOps::MatMul(hidden,_w_down,output);
        if(_config.use_bias==true)
        {
          CAIF_DeviceOps::AddBias(output,_b_down,output);
        }
      }

      return output;
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoEExpert::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    // grad_output: [num_tokens, input_dim]
    const auto &shape=grad_output.Shape();
    const uint32_t num_tokens=shape[0];

    // Backward through down projection
    CAIF_DeviceTensor grad_hidden;
    if(_use_projections==true)
    {
      grad_hidden=_projections.down->Backward(grad_output);
    }
    else
    {
      grad_hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::MatMulTransposeB(grad_output,_w_down,grad_hidden);

      CAIF_DeviceTensor grad_w_down_batch=CAIF_DeviceTensor::Uninitialized({_config.hidden_dim,_config.input_dim},
                                                                         *_stream);
      CAIF_DeviceOps::MatMulTransposeA(_cached_hidden,grad_output,grad_w_down_batch);
      CAIF_DeviceOps::Add(_grad_w_down,grad_w_down_batch,_grad_w_down);

      if(_config.use_bias==true)
      {
        CAIF_DeviceTensor grad_b_down_batch=CAIF_DeviceTensor::Uninitialized({_config.input_dim},*_stream);
        CAIF_DeviceOps::SumAxis(grad_output,0,grad_b_down_batch);
        CAIF_DeviceOps::Add(_grad_b_down,grad_b_down_batch,_grad_b_down);
      }
    }

    CAIF_DeviceTensor grad_input;

    if(_config.use_gated==true)
    {
      // Backward through gated activation
      CAIF_DeviceTensor up_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::SiLU(_cached_up_out,up_activated);

      CAIF_DeviceTensor grad_gate=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::Multiply(grad_hidden,up_activated,grad_gate);

      CAIF_DeviceTensor grad_up_activated=
        CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::Multiply(grad_hidden,_cached_gate_out,grad_up_activated);

      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::SiLUBackward(_cached_up_out,grad_up_activated,grad_up);

      // Backward through gate and up projections
      if(_use_projections==true)
      {
        CAIF_DeviceTensor gi_up=_projections.up->Backward(grad_up);
        CAIF_DeviceTensor gi_gate=_projections.gate->Backward(grad_gate);
        grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
        CAIF_DeviceOps::Add(gi_up,gi_gate,grad_input);
      }
      else
      {
        CAIF_DeviceTensor grad_w_up_batch=CAIF_DeviceTensor::Uninitialized({_config.input_dim,
                                                                        _config.hidden_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_up,grad_w_up_batch);
        CAIF_DeviceOps::Add(_grad_w_up,grad_w_up_batch,_grad_w_up);

        CAIF_DeviceTensor grad_w_gate_batch=CAIF_DeviceTensor::Uninitialized({_config.input_dim,_config.hidden_dim},
                                                                           *_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_gate,grad_w_gate_batch);
        CAIF_DeviceOps::Add(_grad_w_gate,grad_w_gate_batch,_grad_w_gate);

        if(_config.use_bias==true)
        {
          CAIF_DeviceTensor grad_b_up_batch=CAIF_DeviceTensor::Uninitialized({_config.hidden_dim},*_stream);
          CAIF_DeviceOps::SumAxis(grad_up,0,grad_b_up_batch);
          CAIF_DeviceOps::Add(_grad_b_up,grad_b_up_batch,_grad_b_up);

          CAIF_DeviceTensor grad_b_gate_batch=CAIF_DeviceTensor::Uninitialized({_config.hidden_dim},*_stream);
          CAIF_DeviceOps::SumAxis(grad_gate,0,grad_b_gate_batch);
          CAIF_DeviceOps::Add(_grad_b_gate,grad_b_gate_batch,_grad_b_gate);
        }

        CAIF_DeviceTensor grad_input_up=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeB(grad_up,_w_up,grad_input_up);

        CAIF_DeviceTensor grad_input_gate=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeB(grad_gate,_w_gate,grad_input_gate);

        grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
        CAIF_DeviceOps::Add(grad_input_up,grad_input_gate,grad_input);
      }
    }
    else
    {
      // Backward through SiLU activation
      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.hidden_dim},*_stream);
      CAIF_DeviceOps::SiLUBackward(_cached_up_out,grad_hidden,grad_up);

      if(_use_projections==true)
      {
        grad_input=_projections.up->Backward(grad_up);
      }
      else
      {
        CAIF_DeviceTensor grad_w_up_batch=CAIF_DeviceTensor::Uninitialized({_config.input_dim,_config.hidden_dim},
                                                                         *_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_up,grad_w_up_batch);
        CAIF_DeviceOps::Add(_grad_w_up,grad_w_up_batch,_grad_w_up);

        if(_config.use_bias==true)
        {
          CAIF_DeviceTensor grad_b_up_batch=CAIF_DeviceTensor::Uninitialized({_config.hidden_dim},*_stream);
          CAIF_DeviceOps::SumAxis(grad_up,0,grad_b_up_batch);
          CAIF_DeviceOps::Add(_grad_b_up,grad_b_up_batch,_grad_b_up);
        }

        grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeB(grad_up,_w_up,grad_input);
      }
    }

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceMoEExpert::ZeroGradients()
{
  try
  {
    if(_use_projections==true)
    {
      if(_config.use_gated==true&&_projections.gate!=nullptr)
      {
        _projections.gate->ZeroGradients();
      }
      _projections.up->ZeroGradients();
      _projections.down->ZeroGradients();
    }
    else
    {
      if(_config.use_gated==true)
      {
        _grad_w_gate.Fill(0.0f);
        if(_config.use_bias==true)
        {
          _grad_b_gate.Fill(0.0f);
        }
      }

      _grad_w_up.Fill(0.0f);
      _grad_w_down.Fill(0.0f);

      if(_config.use_bias==true)
      {
        _grad_b_up.Fill(0.0f);
        _grad_b_down.Fill(0.0f);
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMoEExpert::ParameterTensorCount()const
{
  if(_use_projections==true)
  {
    size_t count=_projections.up->ParameterTensorCount()+
                 _projections.down->ParameterTensorCount();
    if(_config.use_gated==true&&_projections.gate!=nullptr)
    {
      count+=_projections.gate->ParameterTensorCount();
    }
    return count;
  }
  size_t count=2;  // w_up, w_down
  if(_config.use_gated==true)
  {
    count+=1;  // w_gate
  }
  if(_config.use_bias==true)
  {
    count+=2;  // b_up, b_down
    if(_config.use_gated==true)
    {
      count+=1;  // b_gate
    }
  }
  return count;
}

CAIF_DeviceTensor &CAIF_DeviceMoEExpert::ParameterTensor(size_t index)
{
  if(_use_projections==true)
  {
    size_t offset=0;
    if(_config.use_gated==true&&_projections.gate!=nullptr)
    {
      const size_t count=_projections.gate->ParameterTensorCount();
      if(index<offset+count)
      {
        return _projections.gate->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=_projections.up->ParameterTensorCount();
      if(index<offset+count)
      {
        return _projections.up->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=_projections.down->ParameterTensorCount();
      if(index<offset+count)
      {
        return _projections.down->ParameterTensor(index-offset);
      }
    }
    THROW_CAIFE("MoEExpert::ParameterTensor: index out of range");
  }
  size_t idx=0;
  if(_config.use_gated==true)
  {
    if(index==idx)
    {
      return _w_gate;
    }
    ++idx;
  }
  if(index==idx)
  {
    return _w_up;
  }
  ++idx;
  if(index==idx)
  {
    return _w_down;
  }
  ++idx;
  if(_config.use_bias==true)
  {
    if(_config.use_gated==true)
    {
      if(index==idx)
      {
        return _b_gate;
      }
      ++idx;
    }
    if(index==idx)
    {
      return _b_up;
    }
    ++idx;
    if(index==idx)
    {
      return _b_down;
    }
  }
  THROW_CAIFE("MoEExpert::ParameterTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMoEExpert::ParameterTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoEExpert*>(this)->ParameterTensor(index);
}

CAIF_DeviceTensor &CAIF_DeviceMoEExpert::GradientTensor(size_t index)
{
  if(_use_projections==true)
  {
    size_t offset=0;
    if(_config.use_gated==true&&_projections.gate!=nullptr)
    {
      const size_t count=_projections.gate->ParameterTensorCount();
      if(index<offset+count)
      {
        return _projections.gate->GradientTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=_projections.up->ParameterTensorCount();
      if(index<offset+count)
      {
        return _projections.up->GradientTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=_projections.down->ParameterTensorCount();
      if(index<offset+count)
      {
        return _projections.down->GradientTensor(index-offset);
      }
    }
    THROW_CAIFE("MoEExpert::GradientTensor: index out of range");
  }
  size_t idx=0;
  if(_config.use_gated==true)
  {
    if(index==idx)
    {
      return _grad_w_gate;
    }
    ++idx;
  }
  if(index==idx)
  {
    return _grad_w_up;
  }
  ++idx;
  if(index==idx)
  {
    return _grad_w_down;
  }
  ++idx;
  if(_config.use_bias==true)
  {
    if(_config.use_gated==true)
    {
      if(index==idx)
      {
        return _grad_b_gate;
      }
      ++idx;
    }
    if(index==idx)
    {
      return _grad_b_up;
    }
    ++idx;
    if(index==idx)
    {
      return _grad_b_down;
    }
  }
  THROW_CAIFE("MoEExpert::GradientTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMoEExpert::GradientTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoEExpert*>(this)->GradientTensor(index);
}

size_t CAIF_DeviceMoEExpert::TotalParameterCount()const
{
  if(_use_projections==true)
  {
    size_t total=_projections.up->TotalParameterCount()+
                 _projections.down->TotalParameterCount();
    if(_config.use_gated==true&&_projections.gate!=nullptr)
    {
      total+=_projections.gate->TotalParameterCount();
    }
    return total;
  }
  size_t count=0;
  count+=_config.input_dim*_config.hidden_dim;  // w_up
  count+=_config.hidden_dim*_config.input_dim;  // w_down
  if(_config.use_gated==true)
  {
    count+=_config.input_dim*_config.hidden_dim;  // w_gate
  }
  if(_config.use_bias==true)
  {
    count+=_config.hidden_dim;  // b_up
    count+=_config.input_dim;   // b_down
    if(_config.use_gated==true)
    {
      count+=_config.hidden_dim;  // b_gate
    }
  }
  return count;
}

std::string CAIF_DeviceMoEExpert::Description()const
{
  std::string desc="MoEExpert[";
  desc+=std::to_string(_config.input_dim)+"->"+std::to_string(_config.hidden_dim);
  desc+="->"+std::to_string(_config.input_dim);
  if(_config.use_gated==true)
  {
    desc+=",gated";
  }
  if(_config.use_bias==true)
  {
    desc+=",bias";
  }
  desc+="]";
  return desc;
}

std::vector<std::string> CAIF_DeviceMoEExpert::ParameterNames(const std::string &prefix)const
{
  if(_use_projections==true)
  {
    std::vector<std::string> names;
    std::vector<std::string> sub;
    if(_config.use_gated==true&&_projections.gate!=nullptr)
    {
      sub=_projections.gate->ParameterNames(prefix+"gate_proj.");
      names.insert(names.end(),sub.begin(),sub.end());
    }
    sub=_projections.up->ParameterNames(prefix+"up_proj.");
    names.insert(names.end(),sub.begin(),sub.end());
    sub=_projections.down->ParameterNames(prefix+"down_proj.");
    names.insert(names.end(),sub.begin(),sub.end());
    return names;
  }
  std::vector<std::string> names;
  if(_config.use_gated==true)
  {
    names.push_back(prefix+"w_gate");
  }
  names.push_back(prefix+"w_up");
  names.push_back(prefix+"w_down");
  if(_config.use_bias==true)
  {
    if(_config.use_gated==true)
    {
      names.push_back(prefix+"b_gate");
    }
    names.push_back(prefix+"b_up");
    names.push_back(prefix+"b_down");
  }
  return names;
}
