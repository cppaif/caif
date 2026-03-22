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

#include "caif_device_ffn.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <cmath>
#include <random>
#include <vector>

namespace instance
{

CAIF_DeviceFFN::CAIF_DeviceFFN(const FFNConfig_t &config,
                             std::unique_ptr<CAIF_DeviceActivation> activation,
                             CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                     _config(config),
                                                     _activation(),
                                                     _is_gated(false),
                                                     _use_projections(false),
                                                     _w1(),
                                                     _w2(),
                                                     _grad_w1(),
                                                     _grad_w2(),
                                                     _w_gate(),
                                                     _w_up(),
                                                     _w_down(),
                                                     _grad_w_gate(),
                                                     _grad_w_up(),
                                                     _grad_w_down(),
                                                     _cached_input(),
                                                     _cached_pre_activation(),
                                                     _cached_post_activation(),
                                                     _cached_gate_input(),
                                                     _cached_up_input(),
                                                     _cached_act_output(),
                                                     _cached_input_shape()
{
  try
  {
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceFFN: dim must be > 0");
    }
    if(config.ffn_dim==0)
    {
      THROW_CAIFE("DeviceFFN: ffn_dim must be > 0");
    }
    if(activation==nullptr)
    {
      THROW_CAIFE("DeviceFFN: activation must not be null");
    }

    _activation=activation->Clone();
    _is_gated=_activation->IsGated();

    if(_is_gated==false)
    {
      _w1=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream);
      _w2=CAIF_DeviceTensor::Zeros({_config.ffn_dim,_config.dim},stream);
      _grad_w1=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream);
      _grad_w2=CAIF_DeviceTensor::Zeros({_config.ffn_dim,_config.dim},stream);
    }
    else
    {
      _w_gate=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream);
      _w_up=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream);
      _w_down=CAIF_DeviceTensor::Zeros({_config.ffn_dim,_config.dim},stream);
      _grad_w_gate=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream);
      _grad_w_up=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream);
      _grad_w_down=CAIF_DeviceTensor::Zeros({_config.ffn_dim,_config.dim},stream);
    }

    InitializeWeights(0);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceFFN::CAIF_DeviceFFN(const FFNConfig_t &config,
                             FFNProjections_t projections,
                             std::unique_ptr<CAIF_DeviceActivation> activation,
                             CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                     _config(config),
                                                     _activation(),
                                                     _is_gated(false),
                                                     _projections(std::move(projections)),
                                                     _use_projections(true),
                                                     _cached_input(),
                                                     _cached_pre_activation(),
                                                     _cached_post_activation(),
                                                     _cached_gate_input(),
                                                     _cached_up_input(),
                                                     _cached_act_output(),
                                                     _cached_input_shape()
{
  try
  {
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceFFN: dim must be > 0");
    }
    if(config.ffn_dim==0)
    {
      THROW_CAIFE("DeviceFFN: ffn_dim must be > 0");
    }
    if(activation==nullptr)
    {
      THROW_CAIFE("DeviceFFN: activation must not be null");
    }
    if(_projections.up==nullptr)
    {
      THROW_CAIFE("DeviceFFN: up projection must not be null");
    }
    if(_projections.down==nullptr)
    {
      THROW_CAIFE("DeviceFFN: down projection must not be null");
    }

    _activation=activation->Clone();
    _is_gated=_activation->IsGated();

    if(_is_gated==true&&_projections.gate==nullptr)
    {
      THROW_CAIFE("DeviceFFN: gate projection required for gated activation");
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceFFN::CAIF_DeviceFFN(
  CAIF_DeviceFFN &&other):CAIF_DeviceLayer(std::move(other)),
                          _config(other._config),
                          _activation(std::move(other._activation)),
                          _is_gated(other._is_gated),
                          _projections(std::move(other._projections)),
                          _use_projections(other._use_projections),
                          _w1(std::move(other._w1)),
                          _w2(std::move(other._w2)),
                          _grad_w1(std::move(other._grad_w1)),
                          _grad_w2(std::move(other._grad_w2)),
                          _w_gate(std::move(other._w_gate)),
                          _w_up(std::move(other._w_up)),
                          _w_down(std::move(other._w_down)),
                          _grad_w_gate(std::move(other._grad_w_gate)),
                          _grad_w_up(std::move(other._grad_w_up)),
                          _grad_w_down(std::move(other._grad_w_down)),
                          _cached_input(std::move(other._cached_input)),
                          _cached_pre_activation(std::move(other._cached_pre_activation)),
                          _cached_post_activation(std::move(other._cached_post_activation)),
                          _cached_gate_input(std::move(other._cached_gate_input)),
                          _cached_up_input(std::move(other._cached_up_input)),
                          _cached_act_output(std::move(other._cached_act_output)),
                          _cached_input_shape(std::move(other._cached_input_shape))
{
}

CAIF_DeviceFFN &CAIF_DeviceFFN::operator=(CAIF_DeviceFFN &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _config=other._config;
      _activation=std::move(other._activation);
      _is_gated=other._is_gated;
      _projections=std::move(other._projections);
      _use_projections=other._use_projections;
      _w1=std::move(other._w1);
      _w2=std::move(other._w2);
      _grad_w1=std::move(other._grad_w1);
      _grad_w2=std::move(other._grad_w2);
      _w_gate=std::move(other._w_gate);
      _w_up=std::move(other._w_up);
      _w_down=std::move(other._w_down);
      _grad_w_gate=std::move(other._grad_w_gate);
      _grad_w_up=std::move(other._grad_w_up);
      _grad_w_down=std::move(other._grad_w_down);
      _cached_input=std::move(other._cached_input);
      _cached_pre_activation=std::move(other._cached_pre_activation);
      _cached_post_activation=std::move(other._cached_post_activation);
      _cached_gate_input=std::move(other._cached_gate_input);
      _cached_up_input=std::move(other._cached_up_input);
      _cached_act_output=std::move(other._cached_act_output);
      _cached_input_shape=std::move(other._cached_input_shape);
    }
    return *this;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceFFN::Forward(const CAIF_DeviceTensor &input,
                                        bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceFFN: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("DeviceFFN::Forward: input must be at least 2D");
    }
    if(shape[shape.size()-1]!=_config.dim)
    {
      THROW_CAIFE("DeviceFFN::Forward: last dim must match config dim");
    }

    // Compute N = product of all dims except last
    uint32_t n_rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n_rows*=shape[i];
    }

    // Flatten to [N, dim] — avoid clone when input is already 2D
    const bool needs_reshape=(shape.size()!=2);
    CAIF_DeviceTensor flat_input_storage;
    if(needs_reshape==true)
    {
      flat_input_storage=input.Clone();
      flat_input_storage.Reshape({n_rows,_config.dim});
    }
    const CAIF_DeviceTensor &flat_input=(needs_reshape==true)?
                                        flat_input_storage:input;

    CAIF_DeviceTensor output_flat;

    if(_is_gated==false)
    {
      // Pointwise path
      auto *pw_act=static_cast<const CAIF_DevicePointwiseActivation *>(
                     _activation.get());

      // hidden = flat_input @ W1 -> [N, ffn_dim]
      CAIF_DeviceTensor hidden;
      if(_use_projections==true)
      {
        hidden=_projections.up->Forward(flat_input,training);
      }
      else
      {
        hidden=CAIF_DeviceTensor::Uninitialized({n_rows,_config.ffn_dim},*_stream);
        CAIF_DeviceOps::MatMul(flat_input,_w1,hidden);
      }

      // act = activation(hidden) -> [N, ffn_dim]
      CAIF_DeviceTensor act=CAIF_DeviceTensor::Uninitialized(
                             {n_rows,_config.ffn_dim},*_stream);
      pw_act->Forward(hidden,act);

      // output = act @ W2 -> [N, dim]
      if(_use_projections==true)
      {
        output_flat=_projections.down->Forward(act,training);
      }
      else
      {
        output_flat=CAIF_DeviceTensor::Uninitialized({n_rows,_config.dim},*_stream);
        CAIF_DeviceOps::MatMul(act,_w2,output_flat);
      }

      if(training==true)
      {
        if(_use_projections==false)
        {
          if(needs_reshape==true)
          {
            _cached_input=std::move(flat_input_storage);
          }
          else
          {
            _cached_input=input.Clone();
            _cached_input.Reshape({n_rows,_config.dim});
          }
        }
        _cached_pre_activation=std::move(hidden);
        if(pw_act->NeedsPostActivation()==true)
        {
          _cached_post_activation=act.Clone();
        }
        _cached_act_output=std::move(act);
        _cached_input_shape=std::vector<uint32_t>(shape.begin(),shape.end());
      }
    }
    else
    {
      // Gated path
      auto *gated_act=static_cast<const CAIF_DeviceGatedActivation *>(
                        _activation.get());

      // gate = flat_input @ W_gate -> [N, ffn_dim]
      CAIF_DeviceTensor gate;
      if(_use_projections==true)
      {
        gate=_projections.gate->Forward(flat_input,training);
      }
      else
      {
        gate=CAIF_DeviceTensor::Uninitialized({n_rows,_config.ffn_dim},*_stream);
        CAIF_DeviceOps::MatMul(flat_input,_w_gate,gate);
      }

      // up = flat_input @ W_up -> [N, ffn_dim]
      CAIF_DeviceTensor up;
      if(_use_projections==true)
      {
        up=_projections.up->Forward(flat_input,training);
      }
      else
      {
        up=CAIF_DeviceTensor::Uninitialized({n_rows,_config.ffn_dim},*_stream);
        CAIF_DeviceOps::MatMul(flat_input,_w_up,up);
      }

      // act = activation(gate, up) -> [N, ffn_dim]
      CAIF_DeviceTensor act=CAIF_DeviceTensor::Uninitialized(
                             {n_rows,_config.ffn_dim},*_stream);
      gated_act->Forward(gate,up,act);

      // output = act @ W_down -> [N, dim]
      if(_use_projections==true)
      {
        output_flat=_projections.down->Forward(act,training);
      }
      else
      {
        output_flat=CAIF_DeviceTensor::Uninitialized({n_rows,_config.dim},*_stream);
        CAIF_DeviceOps::MatMul(act,_w_down,output_flat);
      }

      if(training==true)
      {
        if(_use_projections==false)
        {
          if(needs_reshape==true)
          {
            _cached_input=std::move(flat_input_storage);
          }
          else
          {
            _cached_input=input.Clone();
            _cached_input.Reshape({n_rows,_config.dim});
          }
        }
        _cached_gate_input=std::move(gate);
        _cached_up_input=std::move(up);
        _cached_act_output=std::move(act);
        _cached_input_shape=std::vector<uint32_t>(shape.begin(),shape.end());
      }
    }

    // Reshape output to original leading dims + dim
    if(needs_reshape==true)
    {
      output_flat.Reshape(std::vector<uint32_t>(shape.begin(),shape.end()));
    }

    return output_flat;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceFFN::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceFFN: layer has been moved from");
    }
    if(_use_projections==false&&_cached_input.IsEmpty()==true)
    {
      THROW_CAIFE("DeviceFFN::Backward: must call Forward with training=true first");
    }

    const uint32_t dim=_config.dim;
    const uint32_t ffn_dim=_config.ffn_dim;

    // Compute N from grad_output shape
    const auto &grad_shape=grad_output.Shape();
    uint32_t n_rows=1;
    for(size_t i=0;i<grad_shape.size()-1;++i)
    {
      n_rows*=grad_shape[i];
    }

    // Flatten grad_output to [N, dim] — avoid clone when already 2D
    const bool grad_needs_reshape=(grad_shape.size()!=2);
    CAIF_DeviceTensor grad_out_flat_storage;
    if(grad_needs_reshape==true)
    {
      grad_out_flat_storage=grad_output.Clone();
      grad_out_flat_storage.Reshape({n_rows,dim});
    }
    const CAIF_DeviceTensor &grad_out_flat=(grad_needs_reshape==true)?
                                           grad_out_flat_storage:grad_output;

    CAIF_DeviceTensor grad_input;

    if(_is_gated==false)
    {
      // Pointwise backward
      auto *pw_act=static_cast<const CAIF_DevicePointwiseActivation *>(_activation.get());

      // grad_act = backward through down projection -> [N, ffn_dim]
      CAIF_DeviceTensor grad_act;
      if(_use_projections==true)
      {
        grad_act=_projections.down->Backward(grad_out_flat);
      }
      else
      {
        grad_act=CAIF_DeviceTensor::Uninitialized({n_rows,ffn_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeB(grad_out_flat,_w2,grad_act);

        CAIF_DeviceTensor grad_w2_delta=CAIF_DeviceTensor::Uninitialized({ffn_dim,dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_act_output,grad_out_flat,grad_w2_delta);
        CAIF_DeviceOps::Add(_grad_w2,grad_w2_delta,_grad_w2);
      }

      // grad_hidden = activation backward -> [N, ffn_dim]
      CAIF_DeviceTensor grad_hidden=CAIF_DeviceTensor::Uninitialized(
                                     {n_rows,ffn_dim},*_stream);
      pw_act->Backward(grad_act,_cached_pre_activation,
                        _cached_post_activation,grad_hidden);

      // backward through up projection -> grad_input [N, dim]
      if(_use_projections==true)
      {
        grad_input=_projections.up->Backward(grad_hidden);
      }
      else
      {
        CAIF_DeviceTensor grad_w1_delta=CAIF_DeviceTensor::Uninitialized({dim,ffn_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_hidden,grad_w1_delta);
        CAIF_DeviceOps::Add(_grad_w1,grad_w1_delta,_grad_w1);

        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeB(grad_hidden,_w1,grad_input);
      }
    }
    else
    {
      // Gated backward
      auto *gated_act=static_cast<const CAIF_DeviceGatedActivation *>(
                        _activation.get());

      // grad_act = backward through down projection -> [N, ffn_dim]
      CAIF_DeviceTensor grad_act;
      if(_use_projections==true)
      {
        grad_act=_projections.down->Backward(grad_out_flat);
      }
      else
      {
        grad_act=CAIF_DeviceTensor::Uninitialized({n_rows,ffn_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeB(grad_out_flat,_w_down,grad_act);

        CAIF_DeviceTensor grad_wdown_delta=CAIF_DeviceTensor::Uninitialized({ffn_dim,dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_act_output,grad_out_flat,grad_wdown_delta);
        CAIF_DeviceOps::Add(_grad_w_down,grad_wdown_delta,_grad_w_down);
      }

      // activation backward -> grad_gate, grad_up [N, ffn_dim]
      CAIF_DeviceTensor grad_gate=CAIF_DeviceTensor::Uninitialized( {n_rows,ffn_dim},*_stream);
      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized( {n_rows,ffn_dim},*_stream);
      gated_act->Backward(grad_act,_cached_gate_input, _cached_up_input,grad_gate,grad_up);

      // backward through gate and up projections -> grad_input
      if(_use_projections==true)
      {
        CAIF_DeviceTensor gi_gate=_projections.gate->Backward(grad_gate);
        CAIF_DeviceTensor gi_up=_projections.up->Backward(grad_up);
        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},*_stream);
        CAIF_DeviceOps::Add(gi_gate,gi_up,grad_input);
      }
      else
      {
        CAIF_DeviceTensor grad_wgate_delta=CAIF_DeviceTensor::Uninitialized( {dim,ffn_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_gate,grad_wgate_delta);
        CAIF_DeviceOps::Add(_grad_w_gate,grad_wgate_delta,_grad_w_gate);

        CAIF_DeviceTensor grad_wup_delta=CAIF_DeviceTensor::Uninitialized( {dim,ffn_dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_up,grad_wup_delta);
        CAIF_DeviceOps::Add(_grad_w_up,grad_wup_delta,_grad_w_up);

        CAIF_DeviceTensor gi_gate=CAIF_DeviceTensor::Uninitialized( {n_rows,dim},*_stream);
        CAIF_DeviceTensor gi_up=CAIF_DeviceTensor::Uninitialized( {n_rows,dim},*_stream);
        CAIF_DeviceOps::MatMulTransposeB(grad_gate,_w_gate,gi_gate);
        CAIF_DeviceOps::MatMulTransposeB(grad_up,_w_up,gi_up);

        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},*_stream);
        CAIF_DeviceOps::Add(gi_gate,gi_up,grad_input);
      }
    }

    // Reshape to original input shape
    grad_input.Reshape(_cached_input_shape);

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceFFN::ZeroGradients()
{
  try
  {
    if(_use_projections==true)
    {
      if(_is_gated==true&&_projections.gate!=nullptr)
      {
        _projections.gate->ZeroGradients();
      }
      _projections.up->ZeroGradients();
      _projections.down->ZeroGradients();
    }
    else
    {
      if(_is_gated==false)
      {
        _grad_w1.Fill(0.0f);
        _grad_w2.Fill(0.0f);
      }
      else
      {
        _grad_w_gate.Fill(0.0f);
        _grad_w_up.Fill(0.0f);
        _grad_w_down.Fill(0.0f);
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceFFN::ParameterTensorCount()const
{
  try
  {
    if(_use_projections==true)
    {
      size_t count=_projections.up->ParameterTensorCount()+
                   _projections.down->ParameterTensorCount();
      if(_is_gated==true&&_projections.gate!=nullptr)
      {
        count+=_projections.gate->ParameterTensorCount();
      }
      return count;
    }
    if(_is_gated==false)
    {
      return 2;
    }
    return 3;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceFFN::ParameterTensor(size_t index)
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      if(_is_gated==true&&_projections.gate!=nullptr)
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
      THROW_CAIFE("DeviceFFN::ParameterTensor: index out of range");
    }
    if(_is_gated==false)
    {
      if(index==0)
      {
        return _w1;
      }
      if(index==1)
      {
        return _w2;
      }
      THROW_CAIFE("DeviceFFN::ParameterTensor: index out of range");
    }
    else
    {
      if(index==0)
      {
        return _w_gate;
      }
      if(index==1)
      {
        return _w_up;
      }
      if(index==2)
      {
        return _w_down;
      }
      THROW_CAIFE("DeviceFFN::ParameterTensor: index out of range");
    }
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceFFN::ParameterTensor(size_t index)const
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      if(_is_gated==true&&_projections.gate!=nullptr)
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
      THROW_CAIFE("DeviceFFN::ParameterTensor: index out of range");
    }
    if(_is_gated==false)
    {
      if(index==0)
      {
        return _w1;
      }
      if(index==1)
      {
        return _w2;
      }
      THROW_CAIFE("DeviceFFN::ParameterTensor: index out of range");
    }
    else
    {
      if(index==0)
      {
        return _w_gate;
      }
      if(index==1)
      {
        return _w_up;
      }
      if(index==2)
      {
        return _w_down;
      }
      THROW_CAIFE("DeviceFFN::ParameterTensor: index out of range");
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceFFN::GradientTensor(size_t index)
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      if(_is_gated==true&&_projections.gate!=nullptr)
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
      THROW_CAIFE("DeviceFFN::GradientTensor: index out of range");
    }
    if(_is_gated==false)
    {
      if(index==0)
      {
        return _grad_w1;
      }
      if(index==1)
      {
        return _grad_w2;
      }
      THROW_CAIFE("DeviceFFN::GradientTensor: index out of range");
    }
    else
    {
      if(index==0)
      {
        return _grad_w_gate;
      }
      if(index==1)
      {
        return _grad_w_up;
      }
      if(index==2)
      {
        return _grad_w_down;
      }
      THROW_CAIFE("DeviceFFN::GradientTensor: index out of range");
    }
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceFFN::GradientTensor(size_t index)const
{
  try
  {
    if(_use_projections==true)
    {
      size_t offset=0;
      if(_is_gated==true&&_projections.gate!=nullptr)
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
      THROW_CAIFE("DeviceFFN::GradientTensor: index out of range");
    }
    if(_is_gated==false)
    {
      if(index==0)
      {
        return _grad_w1;
      }
      if(index==1)
      {
        return _grad_w2;
      }
      THROW_CAIFE("DeviceFFN::GradientTensor: index out of range");
    }
    else
    {
      if(index==0)
      {
        return _grad_w_gate;
      }
      if(index==1)
      {
        return _grad_w_up;
      }
      if(index==2)
      {
        return _grad_w_down;
      }
      THROW_CAIFE("DeviceFFN::GradientTensor: index out of range");
    }
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceFFN::TotalParameterCount()const
{
  try
  {
    if(_use_projections==true)
    {
      size_t total=_projections.up->TotalParameterCount()+
                   _projections.down->TotalParameterCount();
      if(_is_gated==true&&_projections.gate!=nullptr)
      {
        total+=_projections.gate->TotalParameterCount();
      }
      return total;
    }
    if(_is_gated==false)
    {
      return _w1.TotalElements()+_w2.TotalElements();
    }
    return _w_gate.TotalElements()+
           _w_up.TotalElements()+
           _w_down.TotalElements();
  }
  CCAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceFFN::Description()const
{
  try
  {
    std::string desc="FFN(dim="+std::to_string(_config.dim)+
                     ",ffn_dim="+std::to_string(_config.ffn_dim)+
                     ",activation="+_activation->Description();
    if(_use_projections==true)
    {
      desc+=",projections";
    }
    desc+=")";
    return desc;
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceFFN::ParameterNames(const std::string &prefix)const
{
  try
  {
    if(_use_projections==true)
    {
      std::vector<std::string> names;
      std::vector<std::string> sub;
      if(_is_gated==true&&_projections.gate!=nullptr)
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
    if(_is_gated==false)
    {
      names.push_back(prefix+"fc1.weight");
      names.push_back(prefix+"fc2.weight");
    }
    else
    {
      names.push_back(prefix+"gate_proj.weight");
      names.push_back(prefix+"up_proj.weight");
      names.push_back(prefix+"down_proj.weight");
    }
    return names;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceFFN::InitializeWeights(uint32_t seed)
{
  try
  {
    std::mt19937 rng(seed);

    if(_is_gated==false)
    {
      // W1: [dim, ffn_dim]
      const float limit1=std::sqrt(6.0f/static_cast<float>(_config.dim+_config.ffn_dim));
      std::uniform_real_distribution<float> dist1(-limit1,limit1);
      std::vector<float> w1_data(_config.dim*_config.ffn_dim);
      for(size_t i=0;i<w1_data.size();++i)
      {
        w1_data[i]=dist1(rng);
      }
      _w1.CopyFromHost(w1_data.data(),w1_data.size());

      // W2: [ffn_dim, dim]
      const float limit2=std::sqrt(6.0f/static_cast<float>(_config.ffn_dim+_config.dim));
      std::uniform_real_distribution<float> dist2(-limit2,limit2);
      std::vector<float> w2_data(_config.ffn_dim*_config.dim);
      for(size_t i=0;i<w2_data.size();++i)
      {
        w2_data[i]=dist2(rng);
      }
      _w2.CopyFromHost(w2_data.data(),w2_data.size());
    }
    else
    {
      const float limit_in=std::sqrt(6.0f/static_cast<float>(_config.dim+_config.ffn_dim));
      std::uniform_real_distribution<float> dist_in(-limit_in,limit_in);

      // W_gate: [dim, ffn_dim]
      std::vector<float> wg_data(_config.dim*_config.ffn_dim);
      for(size_t i=0;i<wg_data.size();++i)
      {
        wg_data[i]=dist_in(rng);
      }
      _w_gate.CopyFromHost(wg_data.data(),wg_data.size());

      // W_up: [dim, ffn_dim]
      std::vector<float> wu_data(_config.dim*_config.ffn_dim);
      for(size_t i=0;i<wu_data.size();++i)
      {
        wu_data[i]=dist_in(rng);
      }
      _w_up.CopyFromHost(wu_data.data(),wu_data.size());

      // W_down: [ffn_dim, dim]
      const float limit_out=std::sqrt( 6.0f/static_cast<float>(_config.ffn_dim+_config.dim));
      std::uniform_real_distribution<float> dist_out(-limit_out,limit_out);
      std::vector<float> wd_data(_config.ffn_dim*_config.dim);
      for(size_t i=0;i<wd_data.size();++i)
      {
        wd_data[i]=dist_out(rng);
      }
      _w_down.CopyFromHost(wd_data.data(),wd_data.size());
    }
  }
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
