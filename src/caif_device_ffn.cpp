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
#include "caif_constants.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include <cmath>
#include <random>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceFFN<ComputeT,StorageT>::CAIF_DeviceFFN(const FFNConfig_t &config,
                                                  std::unique_ptr<CAIF_DeviceActivation> activation,
                                                  CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
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
      THROW_CAIFE("CAIF_DeviceFFN: dim must be > 0");
    }
    if(config.ffn_dim==0)
    {
      THROW_CAIFE("CAIF_DeviceFFN: ffn_dim must be > 0");
    }
    if(activation==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: activation must not be null");
    }

    _activation=activation->Clone();
    _is_gated=_activation->IsGated();

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    if(_is_gated==false)
    {
      _w1=CAIF_DeviceTensor::Uninitialized({_config.dim,_config.ffn_dim},stream,sdt);
      _w2=CAIF_DeviceTensor::Uninitialized({_config.ffn_dim,_config.dim},stream,sdt);
      _grad_w1=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream,sdt);
      _grad_w2=CAIF_DeviceTensor::Zeros({_config.ffn_dim,_config.dim},stream,sdt);
    }
    else
    {
      _w_gate=CAIF_DeviceTensor::Uninitialized({_config.dim,_config.ffn_dim},stream,sdt);
      _w_up=CAIF_DeviceTensor::Uninitialized({_config.dim,_config.ffn_dim},stream,sdt);
      _w_down=CAIF_DeviceTensor::Uninitialized({_config.ffn_dim,_config.dim},stream,sdt);
      _grad_w_gate=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream,sdt);
      _grad_w_up=CAIF_DeviceTensor::Zeros({_config.dim,_config.ffn_dim},stream,sdt);
      _grad_w_down=CAIF_DeviceTensor::Zeros({_config.ffn_dim,_config.dim},stream,sdt);
    }

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceFFN<ComputeT,StorageT>::CAIF_DeviceFFN(const FFNConfig_t &config,
                                                   FFNProjections_t projections,
                                                   std::unique_ptr<CAIF_DeviceActivation> activation,
                                                   CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
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
      THROW_CAIFE("CAIF_DeviceFFN: dim must be > 0");
    }
    if(config.ffn_dim==0)
    {
      THROW_CAIFE("CAIF_DeviceFFN: ffn_dim must be > 0");
    }
    if(activation==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: activation must not be null");
    }
    if(_projections.up==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: up projection must not be null");
    }
    if(_projections.down==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: down projection must not be null");
    }

    _activation=activation->Clone();
    _is_gated=_activation->IsGated();

    if(_is_gated==true&&_projections.gate==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: gate projection required for gated activation");
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceFFN<ComputeT,StorageT>::CAIF_DeviceFFN(CAIF_DeviceFFN &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceFFN<ComputeT,StorageT> &
CAIF_DeviceFFN<ComputeT,StorageT>::operator=(CAIF_DeviceFFN &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
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
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::UploadAtStorage(CAIF_DeviceTensor &dst,
                                                        const std::vector<float> &host_data)
{
  dst.CopyFromHostFp32(host_data.data(),host_data.size());
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceFFN<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                CAIF_RunContext &ctx)
{
  try
  {
    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("CAIF_DeviceFFN::Forward: input must be at least 2D");
    }
    if(shape[shape.size()-1]!=_config.dim)
    {
      THROW_CAIFE("CAIF_DeviceFFN::Forward: last dim must match config dim");
    }
    if(_use_projections==false)
    {
      AssertInputDtype(input);
    }

    uint32_t n_rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n_rows*=shape[i];
    }

    const bool needs_reshape=(shape.size()!=2);
    CAIF_DeviceTensor flat_input_storage;
    if(needs_reshape==true)
    {
      flat_input_storage=input.Clone();
      flat_input_storage.Reshape({n_rows,_config.dim});
    }
    const CAIF_DeviceTensor &flat_input=(needs_reshape==true)?
                                        flat_input_storage:input;

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    CAIF_DeviceTensor output_flat;

    if(_is_gated==false)
    {
      const CAIF_DevicePointwiseActivation *pw_act=
          static_cast<const CAIF_DevicePointwiseActivation *>(_activation.get());

      CAIF_DeviceTensor hidden;
      if(_use_projections==true)
      {
        hidden=_projections.up->Forward(flat_input,ctx);
      }
      else
      {
        hidden=CAIF_DeviceTensor::Uninitialized({n_rows,_config.ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(flat_input,_w1,hidden,ctx,cdt);
      }

      CAIF_DeviceTensor act=CAIF_DeviceTensor::Uninitialized(
                             {n_rows,_config.ffn_dim},ctx.Stream(),sdt);
      pw_act->Forward(hidden,act);

      if(_use_projections==true)
      {
        output_flat=_projections.down->Forward(act,ctx);
      }
      else
      {
        output_flat=CAIF_DeviceTensor::Uninitialized({n_rows,_config.dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(act,_w2,output_flat,ctx,cdt);
      }

      if(ctx.Training()==true)
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
      const CAIF_DeviceGatedActivation *gated_act=
          static_cast<const CAIF_DeviceGatedActivation *>(_activation.get());

      CAIF_DeviceTensor gate;
      if(_use_projections==true)
      {
        gate=_projections.gate->Forward(flat_input,ctx);
      }
      else
      {
        gate=CAIF_DeviceTensor::Uninitialized({n_rows,_config.ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(flat_input,_w_gate,gate,ctx,cdt);
      }

      CAIF_DeviceTensor up;
      if(_use_projections==true)
      {
        up=_projections.up->Forward(flat_input,ctx);
      }
      else
      {
        up=CAIF_DeviceTensor::Uninitialized({n_rows,_config.ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(flat_input,_w_up,up,ctx,cdt);
      }

      CAIF_DeviceTensor act=CAIF_DeviceTensor::Uninitialized(
                             {n_rows,_config.ffn_dim},ctx.Stream(),sdt);
      gated_act->Forward(gate,up,act);

      if(_use_projections==true)
      {
        output_flat=_projections.down->Forward(act,ctx);
      }
      else
      {
        output_flat=CAIF_DeviceTensor::Uninitialized({n_rows,_config.dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(act,_w_down,output_flat,ctx,cdt);
      }

      if(ctx.Training()==true)
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

    if(needs_reshape==true)
    {
      output_flat.Reshape(std::vector<uint32_t>(shape.begin(),shape.end()));
    }

    return output_flat;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceFFN<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                 CAIF_RunContext &ctx)
{
  try
  {
    if(_use_projections==false&&_cached_input.IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::Backward: must call Forward with training=true first");
    }

    const uint32_t dim=_config.dim;
    const uint32_t ffn_dim=_config.ffn_dim;
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    const std::vector<uint32_t> &grad_shape=grad_output.Shape();
    uint32_t n_rows=1;
    for(size_t i=0;i<grad_shape.size()-1;++i)
    {
      n_rows*=grad_shape[i];
    }

    const bool grad_needs_reshape=(grad_shape.size()!=2);
    CAIF_DeviceTensor grad_out_flat_storage;
    if(grad_needs_reshape==true)
    {
      grad_out_flat_storage=CAIF_DeviceTensor::WrapView(
                             const_cast<void *>(grad_output.DeviceDataRaw()),
                             {n_rows,dim},
                             ctx.Stream(),
                             grad_output.Dtype());
    }
    const CAIF_DeviceTensor &grad_out_flat=(grad_needs_reshape==true)?
                                           grad_out_flat_storage:grad_output;

    CAIF_DeviceTensor grad_input;

    if(_is_gated==false)
    {
      const CAIF_DevicePointwiseActivation *pw_act=
          static_cast<const CAIF_DevicePointwiseActivation *>(_activation.get());

      CAIF_DeviceTensor grad_act;
      if(_use_projections==true)
      {
        grad_act=_projections.down->Backward(grad_out_flat,ctx);
      }
      else
      {
        grad_act=CAIF_DeviceTensor::Uninitialized({n_rows,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_out_flat,_w2,grad_act,ctx,cdt);

        CAIF_DeviceTensor grad_w2_delta=CAIF_DeviceTensor::Uninitialized(
                                          {ffn_dim,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(_cached_act_output,grad_out_flat,grad_w2_delta,ctx,cdt);
        CAIF_Ops::Add(_grad_w2,grad_w2_delta,_grad_w2);
      }

      CAIF_DeviceTensor grad_hidden=CAIF_DeviceTensor::Uninitialized(
                                     {n_rows,ffn_dim},ctx.Stream(),sdt);
      pw_act->Backward(grad_act,_cached_pre_activation,
                        _cached_post_activation,grad_hidden);

      if(_use_projections==true)
      {
        grad_input=_projections.up->Backward(grad_hidden,ctx);
      }
      else
      {
        CAIF_DeviceTensor grad_w1_delta=CAIF_DeviceTensor::Uninitialized(
                                          {dim,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(_cached_input,grad_hidden,grad_w1_delta,ctx,cdt);
        CAIF_Ops::Add(_grad_w1,grad_w1_delta,_grad_w1);

        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_hidden,_w1,grad_input,ctx,cdt);
      }
    }
    else
    {
      const CAIF_DeviceGatedActivation *gated_act=
          static_cast<const CAIF_DeviceGatedActivation *>(_activation.get());

      CAIF_DeviceTensor grad_act;
      if(_use_projections==true)
      {
        grad_act=_projections.down->Backward(grad_out_flat,ctx);
      }
      else
      {
        grad_act=CAIF_DeviceTensor::Uninitialized({n_rows,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_out_flat,_w_down,grad_act,ctx,cdt);

        CAIF_DeviceTensor grad_wdown_delta=CAIF_DeviceTensor::Uninitialized(
                                              {ffn_dim,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(_cached_act_output,grad_out_flat,grad_wdown_delta,ctx,cdt);
        CAIF_Ops::Add(_grad_w_down,grad_wdown_delta,_grad_w_down);
      }

      CAIF_DeviceTensor grad_gate=CAIF_DeviceTensor::Uninitialized(
                                    {n_rows,ffn_dim},ctx.Stream(),sdt);
      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized(
                                  {n_rows,ffn_dim},ctx.Stream(),sdt);
      gated_act->Backward(grad_act,_cached_gate_input,_cached_up_input,grad_gate,grad_up);

      if(_use_projections==true)
      {
        CAIF_DeviceTensor gi_gate=_projections.gate->Backward(grad_gate,ctx);
        CAIF_DeviceTensor gi_up=_projections.up->Backward(grad_up,ctx);
        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::Add(gi_gate,gi_up,grad_input);
      }
      else
      {
        CAIF_DeviceTensor grad_wgate_delta=CAIF_DeviceTensor::Uninitialized(
                                              {dim,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(_cached_input,grad_gate,grad_wgate_delta,ctx,cdt);
        CAIF_Ops::Add(_grad_w_gate,grad_wgate_delta,_grad_w_gate);

        CAIF_DeviceTensor grad_wup_delta=CAIF_DeviceTensor::Uninitialized(
                                            {dim,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(_cached_input,grad_up,grad_wup_delta,ctx,cdt);
        CAIF_Ops::Add(_grad_w_up,grad_wup_delta,_grad_w_up);

        CAIF_DeviceTensor gi_gate=CAIF_DeviceTensor::Uninitialized(
                                    {n_rows,dim},ctx.Stream(),sdt);
        CAIF_DeviceTensor gi_up=CAIF_DeviceTensor::Uninitialized(
                                  {n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_gate,_w_gate,gi_gate,ctx,cdt);
        CAIF_Ops::MatMulTransposeB(grad_up,_w_up,gi_up,ctx,cdt);

        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::Add(gi_gate,gi_up,grad_input);
      }
    }

    grad_input.Reshape(_cached_input_shape);
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::ZeroGradients()
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
        _grad_w1.FillZero();
        _grad_w2.FillZero();
      }
      else
      {
        _grad_w_gate.FillZero();
        _grad_w_up.FillZero();
        _grad_w_down.FillZero();
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceFFN<ComputeT,StorageT>::ParameterTensorCount()const
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
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceFFN<ComputeT,StorageT>::ParameterTensor(size_t index)
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
      THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
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
      THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
    }
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
    THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceFFN<ComputeT,StorageT>::ParameterTensor(size_t index)const
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
      THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
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
      THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
    }
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
    THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceFFN<ComputeT,StorageT>::GradientTensor(size_t index)
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
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
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
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
    }
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
    THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceFFN<ComputeT,StorageT>::GradientTensor(size_t index)const
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
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
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
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
    }
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
    THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceFFN<ComputeT,StorageT>::TotalParameterCount()const
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
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceFFN<ComputeT,StorageT>::Description()const
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
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceFFN<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
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
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    std::mt19937 rng(seed);

    if(_is_gated==false)
    {
      const float limit1=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(_config.dim+_config.ffn_dim));
      std::uniform_real_distribution<float> dist1(-limit1,limit1);
      std::vector<float> w1_data(static_cast<size_t>(_config.dim)*_config.ffn_dim);
      for(size_t i=0;i<w1_data.size();++i)
      {
        w1_data[i]=dist1(rng);
      }
      UploadAtStorage(_w1,w1_data);

      const float limit2=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(_config.ffn_dim+_config.dim));
      std::uniform_real_distribution<float> dist2(-limit2,limit2);
      std::vector<float> w2_data(static_cast<size_t>(_config.ffn_dim)*_config.dim);
      for(size_t i=0;i<w2_data.size();++i)
      {
        w2_data[i]=dist2(rng);
      }
      UploadAtStorage(_w2,w2_data);
    }
    else
    {
      const float limit_in=std::sqrt(g_caif_xavier_uniform_scale/
                                      static_cast<float>(_config.dim+_config.ffn_dim));
      std::uniform_real_distribution<float> dist_in(-limit_in,limit_in);

      std::vector<float> wg_data(static_cast<size_t>(_config.dim)*_config.ffn_dim);
      for(size_t i=0;i<wg_data.size();++i)
      {
        wg_data[i]=dist_in(rng);
      }
      UploadAtStorage(_w_gate,wg_data);

      std::vector<float> wu_data(static_cast<size_t>(_config.dim)*_config.ffn_dim);
      for(size_t i=0;i<wu_data.size();++i)
      {
        wu_data[i]=dist_in(rng);
      }
      UploadAtStorage(_w_up,wu_data);

      const float limit_out=std::sqrt(g_caif_xavier_uniform_scale/
                                       static_cast<float>(_config.ffn_dim+_config.dim));
      std::uniform_real_distribution<float> dist_out(-limit_out,limit_out);
      std::vector<float> wd_data(static_cast<size_t>(_config.ffn_dim)*_config.dim);
      for(size_t i=0;i<wd_data.size();++i)
      {
        wd_data[i]=dist_out(rng);
      }
      UploadAtStorage(_w_down,wd_data);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadW1(CAIF_DeviceTensor &&w1)
{
  try
  {
    if(_use_projections==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW1: not valid when using sub-projections");
    }
    if(_is_gated==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW1: not valid in gated mode");
    }
    const std::vector<uint32_t> &shape=w1.Shape();
    if(shape.size()!=2 ||
       shape[0]!=_config.dim ||
       shape[1]!=_config.ffn_dim)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW1: shape mismatch, expected [dim, ffn_dim]");
    }
    _w1=std::move(w1);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadW2(CAIF_DeviceTensor &&w2)
{
  try
  {
    if(_use_projections==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW2: not valid when using sub-projections");
    }
    if(_is_gated==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW2: not valid in gated mode");
    }
    const std::vector<uint32_t> &shape=w2.Shape();
    if(shape.size()!=2 ||
       shape[0]!=_config.ffn_dim ||
       shape[1]!=_config.dim)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW2: shape mismatch, expected [ffn_dim, dim]");
    }
    _w2=std::move(w2);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadWGate(CAIF_DeviceTensor &&w_gate)
{
  try
  {
    if(_use_projections==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWGate: not valid when using sub-projections");
    }
    if(_is_gated==false)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWGate: not valid in pointwise mode");
    }
    const std::vector<uint32_t> &shape=w_gate.Shape();
    if(shape.size()!=2 ||
       shape[0]!=_config.dim ||
       shape[1]!=_config.ffn_dim)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWGate: shape mismatch, expected [dim, ffn_dim]");
    }
    _w_gate=std::move(w_gate);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadWUp(CAIF_DeviceTensor &&w_up)
{
  try
  {
    if(_use_projections==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWUp: not valid when using sub-projections");
    }
    if(_is_gated==false)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWUp: not valid in pointwise mode");
    }
    const std::vector<uint32_t> &shape=w_up.Shape();
    if(shape.size()!=2 ||
       shape[0]!=_config.dim ||
       shape[1]!=_config.ffn_dim)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWUp: shape mismatch, expected [dim, ffn_dim]");
    }
    _w_up=std::move(w_up);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadWDown(CAIF_DeviceTensor &&w_down)
{
  try
  {
    if(_use_projections==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWDown: not valid when using sub-projections");
    }
    if(_is_gated==false)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWDown: not valid in pointwise mode");
    }
    const std::vector<uint32_t> &shape=w_down.Shape();
    if(shape.size()!=2 ||
       shape[0]!=_config.ffn_dim ||
       shape[1]!=_config.dim)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWDown: shape mismatch, expected [ffn_dim, dim]");
    }
    _w_down=std::move(w_down);
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceFFN<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceFFN<float,__half>;
template class CAIF_DeviceFFN<float,__nv_bfloat16>;
template class CAIF_DeviceFFN<__half,float>;
template class CAIF_DeviceFFN<__half,__half>;
template class CAIF_DeviceFFN<__half,__nv_bfloat16>;
template class CAIF_DeviceFFN<__nv_bfloat16,float>;
template class CAIF_DeviceFFN<__nv_bfloat16,__half>;
template class CAIF_DeviceFFN<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
