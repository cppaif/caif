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
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <cmath>
#include <random>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceFFN<ComputeT,StorageT>::CAIF_DeviceFFN(const CAIF_DeviceFFNConfig &config,
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
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DeviceFFN: dim must be > 0");
    }
    if(config.FfnDim()==0)
    {
      THROW_CAIFE("CAIF_DeviceFFN: ffn_dim must be > 0");
    }
    if(activation==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: activation must not be null");
    }

    _activation=activation->Clone();
    SetIsGated(Activation().IsGated());

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    if(IsGated()==false)
    {
      SetW1(CAIF_DeviceTensor::Uninitialized({Config().Dim(),Config().FfnDim()},stream,sdt));
      SetW2(CAIF_DeviceTensor::Uninitialized({Config().FfnDim(),Config().Dim()},stream,sdt));
      SetGradW1(CAIF_DeviceTensor::Zeros({Config().Dim(),Config().FfnDim()},stream,sdt));
      SetGradW2(CAIF_DeviceTensor::Zeros({Config().FfnDim(),Config().Dim()},stream,sdt));
    }
    else
    {
      SetWGate(CAIF_DeviceTensor::Uninitialized({Config().Dim(),Config().FfnDim()},stream,sdt));
      SetWUp(CAIF_DeviceTensor::Uninitialized({Config().Dim(),Config().FfnDim()},stream,sdt));
      SetWDown(CAIF_DeviceTensor::Uninitialized({Config().FfnDim(),Config().Dim()},stream,sdt));
      SetGradWGate(CAIF_DeviceTensor::Zeros({Config().Dim(),Config().FfnDim()},stream,sdt));
      SetGradWUp(CAIF_DeviceTensor::Zeros({Config().Dim(),Config().FfnDim()},stream,sdt));
      SetGradWDown(CAIF_DeviceTensor::Zeros({Config().FfnDim(),Config().Dim()},stream,sdt));
    }

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceFFN<ComputeT,StorageT>::CAIF_DeviceFFN(const CAIF_DeviceFFNConfig &config,
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
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DeviceFFN: dim must be > 0");
    }
    if(config.FfnDim()==0)
    {
      THROW_CAIFE("CAIF_DeviceFFN: ffn_dim must be > 0");
    }
    if(activation==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: activation must not be null");
    }
    if(Projections().up==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: up projection must not be null");
    }
    if(Projections().down==nullptr)
    {
      THROW_CAIFE("CAIF_DeviceFFN: down projection must not be null");
    }

    _activation=activation->Clone();
    SetIsGated(Activation().IsGated());

    if(IsGated()==true&&Projections().gate==nullptr)
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
      SetConfig(other.Config());
      _activation=std::move(other._activation);
      SetIsGated(other.IsGated());
      SetProjections(std::move(other.ProjectionsMut()));
      SetUseProjections(other.UseProjections());
      SetW1(std::move(other.W1Mut()));
      SetW2(std::move(other.W2Mut()));
      SetGradW1(std::move(other.GradW1()));
      SetGradW2(std::move(other.GradW2()));
      SetWGate(std::move(other.WGateMut()));
      SetWUp(std::move(other.WUpMut()));
      SetWDown(std::move(other.WDownMut()));
      SetGradWGate(std::move(other.GradWGate()));
      SetGradWUp(std::move(other.GradWUp()));
      SetGradWDown(std::move(other.GradWDown()));
      SetCachedInput(std::move(other.CachedInput()));
      SetCachedPreActivation(std::move(other.CachedPreActivation()));
      SetCachedPostActivation(std::move(other.CachedPostActivation()));
      SetCachedGateInput(std::move(other.CachedGateInput()));
      SetCachedUpInput(std::move(other.CachedUpInput()));
      SetCachedActOutput(std::move(other.CachedActOutput()));
      SetCachedInputShape(std::move(other.CachedInputShape()));
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
    if(shape[shape.size()-1]!=Config().Dim())
    {
      THROW_CAIFE("CAIF_DeviceFFN::Forward: last dim must match config dim");
    }
    if(UseProjections()==false)
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
      flat_input_storage.Reshape({n_rows,Config().Dim()});
    }
    const CAIF_DeviceTensor &flat_input=(needs_reshape==true)?
                                        flat_input_storage:input;

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    CAIF_DeviceTensor output_flat;

    if(IsGated()==false)
    {
      const CAIF_DevicePointwiseActivation *pw_act=
          static_cast<const CAIF_DevicePointwiseActivation *>((&Activation()));

      CAIF_DeviceTensor hidden;
      if(UseProjections()==true)
      {
        hidden=Projections().up->Forward(flat_input,ctx);
      }
      else
      {
        hidden=CAIF_DeviceTensor::Uninitialized({n_rows,Config().FfnDim()},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(flat_input,W1(),hidden,ctx,cdt);
      }

      CAIF_DeviceTensor act=CAIF_DeviceTensor::Uninitialized(
                             {n_rows,Config().FfnDim()},ctx.Stream(),sdt);
      pw_act->Forward(hidden,act);

      if(UseProjections()==true)
      {
        output_flat=Projections().down->Forward(act,ctx);
      }
      else
      {
        output_flat=CAIF_DeviceTensor::Uninitialized({n_rows,Config().Dim()},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(act,W2(),output_flat,ctx,cdt);
      }

      if(ctx.Training()==true)
      {
        if(UseProjections()==false)
        {
          if(needs_reshape==true)
          {
            CachedInput()=std::move(flat_input_storage);
          }
          else
          {
            CachedInput()=input.Clone();
            CachedInput().Reshape({n_rows,Config().Dim()});
          }
        }
        CachedPreActivation()=std::move(hidden);
        if(pw_act->NeedsPostActivation()==true)
        {
          CachedPostActivation()=act.Clone();
        }
        CachedActOutput()=std::move(act);
        CachedInputShape()=std::vector<uint32_t>(shape.begin(),shape.end());
      }
    }
    else
    {
      const CAIF_DeviceGatedActivation *gated_act=
          static_cast<const CAIF_DeviceGatedActivation *>((&Activation()));

      CAIF_DeviceTensor gate;
      if(UseProjections()==true)
      {
        gate=Projections().gate->Forward(flat_input,ctx);
      }
      else
      {
        gate=CAIF_DeviceTensor::Uninitialized({n_rows,Config().FfnDim()},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(flat_input,WGateMut(),gate,ctx,cdt);
      }

      CAIF_DeviceTensor up;
      if(UseProjections()==true)
      {
        up=Projections().up->Forward(flat_input,ctx);
      }
      else
      {
        up=CAIF_DeviceTensor::Uninitialized({n_rows,Config().FfnDim()},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(flat_input,WUpMut(),up,ctx,cdt);
      }

      CAIF_DeviceTensor act=CAIF_DeviceTensor::Uninitialized(
                             {n_rows,Config().FfnDim()},ctx.Stream(),sdt);
      gated_act->Forward(gate,up,act);

      if(UseProjections()==true)
      {
        output_flat=Projections().down->Forward(act,ctx);
      }
      else
      {
        output_flat=CAIF_DeviceTensor::Uninitialized({n_rows,Config().Dim()},ctx.Stream(),sdt);
        CAIF_Ops::MatMul(act,WDownMut(),output_flat,ctx,cdt);
      }

      if(ctx.Training()==true)
      {
        if(UseProjections()==false)
        {
          if(needs_reshape==true)
          {
            CachedInput()=std::move(flat_input_storage);
          }
          else
          {
            CachedInput()=input.Clone();
            CachedInput().Reshape({n_rows,Config().Dim()});
          }
        }
        CachedGateInput()=std::move(gate);
        CachedUpInput()=std::move(up);
        CachedActOutput()=std::move(act);
        CachedInputShape()=std::vector<uint32_t>(shape.begin(),shape.end());
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
    if(UseProjections()==false&&CachedInput().IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::Backward: must call Forward with training=true first");
    }

    const uint32_t dim=Config().Dim();
    const uint32_t ffn_dim=Config().FfnDim();
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

    if(IsGated()==false)
    {
      const CAIF_DevicePointwiseActivation *pw_act=
          static_cast<const CAIF_DevicePointwiseActivation *>((&Activation()));

      CAIF_DeviceTensor grad_act;
      if(UseProjections()==true)
      {
        grad_act=Projections().down->Backward(grad_out_flat,ctx);
      }
      else
      {
        grad_act=CAIF_DeviceTensor::Uninitialized({n_rows,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_out_flat,W2(),grad_act,ctx,cdt);

        CAIF_DeviceTensor grad_w2_delta=CAIF_DeviceTensor::Uninitialized(
                                          {ffn_dim,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(CachedActOutput(),grad_out_flat,grad_w2_delta,ctx,cdt);
        CAIF_Ops::Add(GradW2(),grad_w2_delta,GradW2());
      }

      CAIF_DeviceTensor grad_hidden=CAIF_DeviceTensor::Uninitialized(
                                     {n_rows,ffn_dim},ctx.Stream(),sdt);
      pw_act->Backward(grad_act,CachedPreActivation(),
                        CachedPostActivation(),grad_hidden);

      if(UseProjections()==true)
      {
        grad_input=Projections().up->Backward(grad_hidden,ctx);
      }
      else
      {
        CAIF_DeviceTensor grad_w1_delta=CAIF_DeviceTensor::Uninitialized(
                                          {dim,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_hidden,grad_w1_delta,ctx,cdt);
        CAIF_Ops::Add(GradW1(),grad_w1_delta,GradW1());

        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_hidden,W1(),grad_input,ctx,cdt);
      }
    }
    else
    {
      const CAIF_DeviceGatedActivation *gated_act=
          static_cast<const CAIF_DeviceGatedActivation *>((&Activation()));

      CAIF_DeviceTensor grad_act;
      if(UseProjections()==true)
      {
        grad_act=Projections().down->Backward(grad_out_flat,ctx);
      }
      else
      {
        grad_act=CAIF_DeviceTensor::Uninitialized({n_rows,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_out_flat,WDownMut(),grad_act,ctx,cdt);

        CAIF_DeviceTensor grad_wdown_delta=CAIF_DeviceTensor::Uninitialized(
                                              {ffn_dim,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(CachedActOutput(),grad_out_flat,grad_wdown_delta,ctx,cdt);
        CAIF_Ops::Add(GradWDown(),grad_wdown_delta,GradWDown());
      }

      CAIF_DeviceTensor grad_gate=CAIF_DeviceTensor::Uninitialized(
                                    {n_rows,ffn_dim},ctx.Stream(),sdt);
      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized(
                                  {n_rows,ffn_dim},ctx.Stream(),sdt);
      gated_act->Backward(grad_act,CachedGateInput(),CachedUpInput(),grad_gate,grad_up);

      if(UseProjections()==true)
      {
        CAIF_DeviceTensor gi_gate=Projections().gate->Backward(grad_gate,ctx);
        CAIF_DeviceTensor gi_up=Projections().up->Backward(grad_up,ctx);
        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::Add(gi_gate,gi_up,grad_input);
      }
      else
      {
        CAIF_DeviceTensor grad_wgate_delta=CAIF_DeviceTensor::Uninitialized(
                                              {dim,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_gate,grad_wgate_delta,ctx,cdt);
        CAIF_Ops::Add(GradWGate(),grad_wgate_delta,GradWGate());

        CAIF_DeviceTensor grad_wup_delta=CAIF_DeviceTensor::Uninitialized(
                                            {dim,ffn_dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_up,grad_wup_delta,ctx,cdt);
        CAIF_Ops::Add(GradWUp(),grad_wup_delta,GradWUp());

        CAIF_DeviceTensor gi_gate=CAIF_DeviceTensor::Uninitialized(
                                    {n_rows,dim},ctx.Stream(),sdt);
        CAIF_DeviceTensor gi_up=CAIF_DeviceTensor::Uninitialized(
                                  {n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::MatMulTransposeB(grad_gate,WGateMut(),gi_gate,ctx,cdt);
        CAIF_Ops::MatMulTransposeB(grad_up,WUpMut(),gi_up,ctx,cdt);

        grad_input=CAIF_DeviceTensor::Uninitialized({n_rows,dim},ctx.Stream(),sdt);
        CAIF_Ops::Add(gi_gate,gi_up,grad_input);
      }
    }

    grad_input.Reshape(CachedInputShape());
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    if(UseProjections()==true)
    {
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        Projections().gate->ZeroGradients();
      }
      Projections().up->ZeroGradients();
      Projections().down->ZeroGradients();
    }
    else
    {
      if(IsGated()==false)
      {
        GradW1().FillZero();
        GradW2().FillZero();
      }
      else
      {
        GradWGate().FillZero();
        GradWUp().FillZero();
        GradWDown().FillZero();
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
    if(UseProjections()==true)
    {
      size_t count=Projections().up->ParameterTensorCount()+
                   Projections().down->ParameterTensorCount();
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        count+=Projections().gate->ParameterTensorCount();
      }
      return count;
    }
    if(IsGated()==false)
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
    if(UseProjections()==true)
    {
      size_t offset=0;
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        const size_t count=Projections().gate->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().gate->ParameterTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().up->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().up->ParameterTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().down->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().down->ParameterTensor(index-offset);
        }
      }
      THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
    }
    if(IsGated()==false)
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
      return WGateMut();
    }
    if(index==1)
    {
      return WUpMut();
    }
    if(index==2)
    {
      return WDownMut();
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
    if(UseProjections()==true)
    {
      size_t offset=0;
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        const size_t count=Projections().gate->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().gate->ParameterTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().up->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().up->ParameterTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().down->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().down->ParameterTensor(index-offset);
        }
      }
      THROW_CAIFE("CAIF_DeviceFFN::ParameterTensor: index out of range");
    }
    if(IsGated()==false)
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
      return WGate();
    }
    if(index==1)
    {
      return WUp();
    }
    if(index==2)
    {
      return WDown();
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
    if(UseProjections()==true)
    {
      size_t offset=0;
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        const size_t count=Projections().gate->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().gate->GradientTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().up->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().up->GradientTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().down->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().down->GradientTensor(index-offset);
        }
      }
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
    }
    if(IsGated()==false)
    {
      if(index==0)
      {
        return GradW1();
      }
      if(index==1)
      {
        return GradW2();
      }
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
    }
    if(index==0)
    {
      return GradWGate();
    }
    if(index==1)
    {
      return GradWUp();
    }
    if(index==2)
    {
      return GradWDown();
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
    if(UseProjections()==true)
    {
      size_t offset=0;
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        const size_t count=Projections().gate->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().gate->GradientTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().up->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().up->GradientTensor(index-offset);
        }
        offset+=count;
      }
      {
        const size_t count=Projections().down->ParameterTensorCount();
        if(index<offset+count)
        {
          return Projections().down->GradientTensor(index-offset);
        }
      }
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
    }
    if(IsGated()==false)
    {
      if(index==0)
      {
        return GradW1();
      }
      if(index==1)
      {
        return GradW2();
      }
      THROW_CAIFE("CAIF_DeviceFFN::GradientTensor: index out of range");
    }
    if(index==0)
    {
      return GradWGate();
    }
    if(index==1)
    {
      return GradWUp();
    }
    if(index==2)
    {
      return GradWDown();
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
    if(UseProjections()==true)
    {
      size_t total=Projections().up->TotalParameterCount()+
                   Projections().down->TotalParameterCount();
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        total+=Projections().gate->TotalParameterCount();
      }
      return total;
    }
    if(IsGated()==false)
    {
      return W1().TotalElements()+W2().TotalElements();
    }
    return WGate().TotalElements()+
           WUp().TotalElements()+
           WDown().TotalElements();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceFFN<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc=std::string(g_serial_tag_ffn)+
                     g_serial_open_paren+
                     g_serial_kv_dim+
                     std::to_string(Config().Dim())+
                     g_serial_comma+
                     g_serial_kv_ffn_dim+
                     std::to_string(Config().FfnDim())+
                     g_serial_comma+
                     g_serial_kv_activation+
                     Activation().Description();
    if(UseProjections()==true)
    {
      desc+=g_serial_flag_projections;
    }
    desc+=g_serial_close_paren;
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
    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
    if(UseProjections()==true)
    {
      std::vector<std::string> names;
      std::vector<std::string> sub;
      if(IsGated()==true&&Projections().gate!=nullptr)
      {
        sub=Projections().gate->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWGate_e)+".");
        names.insert(names.end(),sub.begin(),sub.end());
      }
      sub=Projections().up->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWUp_e)+".");
      names.insert(names.end(),sub.begin(),sub.end());
      sub=Projections().down->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWDown_e)+".");
      names.insert(names.end(),sub.begin(),sub.end());
      return names;
    }
    std::vector<std::string> names;
    if(IsGated()==false)
    {
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWUp_e));
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWDown_e));
    }
    else
    {
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWGate_e));
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWUp_e));
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::FFNWDown_e));
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

    if(IsGated()==false)
    {
      const float limit1=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(Config().Dim()+Config().FfnDim()));
      std::uniform_real_distribution<float> dist1(-limit1,limit1);
      std::vector<float> w1_data(static_cast<size_t>(Config().Dim())*Config().FfnDim());
      for(size_t i=0;i<w1_data.size();++i)
      {
        w1_data[i]=dist1(rng);
      }
      UploadAtStorage(W1Mut(),w1_data);

      const float limit2=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(Config().FfnDim()+Config().Dim()));
      std::uniform_real_distribution<float> dist2(-limit2,limit2);
      std::vector<float> w2_data(static_cast<size_t>(Config().FfnDim())*Config().Dim());
      for(size_t i=0;i<w2_data.size();++i)
      {
        w2_data[i]=dist2(rng);
      }
      UploadAtStorage(W2Mut(),w2_data);
    }
    else
    {
      const float limit_in=std::sqrt(g_caif_xavier_uniform_scale/
                                      static_cast<float>(Config().Dim()+Config().FfnDim()));
      std::uniform_real_distribution<float> dist_in(-limit_in,limit_in);

      std::vector<float> wg_data(static_cast<size_t>(Config().Dim())*Config().FfnDim());
      for(size_t i=0;i<wg_data.size();++i)
      {
        wg_data[i]=dist_in(rng);
      }
      UploadAtStorage(WGateMut(),wg_data);

      std::vector<float> wu_data(static_cast<size_t>(Config().Dim())*Config().FfnDim());
      for(size_t i=0;i<wu_data.size();++i)
      {
        wu_data[i]=dist_in(rng);
      }
      UploadAtStorage(WUpMut(),wu_data);

      const float limit_out=std::sqrt(g_caif_xavier_uniform_scale/
                                       static_cast<float>(Config().FfnDim()+Config().Dim()));
      std::uniform_real_distribution<float> dist_out(-limit_out,limit_out);
      std::vector<float> wd_data(static_cast<size_t>(Config().FfnDim())*Config().Dim());
      for(size_t i=0;i<wd_data.size();++i)
      {
        wd_data[i]=dist_out(rng);
      }
      UploadAtStorage(WDownMut(),wd_data);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadW1(CAIF_DeviceTensor &&w1)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW1: not valid when using sub-projections");
    }
    if(IsGated()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW1: not valid in gated mode");
    }
    const std::vector<uint32_t> &shape=w1.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().Dim() ||
       shape[1]!=Config().FfnDim())
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW1: shape mismatch, expected [dim, ffn_dim]");
    }
    SetW1(std::move(w1));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadW2(CAIF_DeviceTensor &&w2)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW2: not valid when using sub-projections");
    }
    if(IsGated()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW2: not valid in gated mode");
    }
    const std::vector<uint32_t> &shape=w2.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().FfnDim() ||
       shape[1]!=Config().Dim())
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadW2: shape mismatch, expected [ffn_dim, dim]");
    }
    SetW2(std::move(w2));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadWGate(CAIF_DeviceTensor &&w_gate)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWGate: not valid when using sub-projections");
    }
    if(IsGated()==false)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWGate: not valid in pointwise mode");
    }
    const std::vector<uint32_t> &shape=w_gate.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().Dim() ||
       shape[1]!=Config().FfnDim())
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWGate: shape mismatch, expected [dim, ffn_dim]");
    }
    SetWGate(std::move(w_gate));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadWUp(CAIF_DeviceTensor &&w_up)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWUp: not valid when using sub-projections");
    }
    if(IsGated()==false)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWUp: not valid in pointwise mode");
    }
    const std::vector<uint32_t> &shape=w_up.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().Dim() ||
       shape[1]!=Config().FfnDim())
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWUp: shape mismatch, expected [dim, ffn_dim]");
    }
    SetWUp(std::move(w_up));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFFN<ComputeT,StorageT>::LoadWDown(CAIF_DeviceTensor &&w_down)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWDown: not valid when using sub-projections");
    }
    if(IsGated()==false)
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWDown: not valid in pointwise mode");
    }
    const std::vector<uint32_t> &shape=w_down.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().FfnDim() ||
       shape[1]!=Config().Dim())
    {
      THROW_CAIFE("CAIF_DeviceFFN::LoadWDown: shape mismatch, expected [ffn_dim, dim]");
    }
    SetWDown(std::move(w_down));
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
