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
#include "caif_constants.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <cmath>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT>::CAIF_DeviceLinearHead(const CAIF_DeviceLinearHeadConfig &config,
                                                                CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
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
    if(config.InputDim()==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: input_dim must be > 0");
    }
    if(config.OutputDim()==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: output_dim must be > 0");
    }

    const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                 static_cast<float>(config.InputDim()+config.OutputDim()));
    const size_t weight_size=static_cast<size_t>(config.InputDim())*config.OutputDim();
    std::vector<float> w_init(weight_size);
    for(size_t i=0;i<weight_size;++i)
    {
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      w_init[i]=(t-std::floor(t))*2.0f*limit-limit;
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    SetWeight(CAIF_DeviceTensor::Uninitialized({config.InputDim(),config.OutputDim()},stream,sdt));
    WeightMut().CopyFromHostFp32(w_init.data(),weight_size);
    SetWeightGrad(CAIF_DeviceTensor::Zeros({config.InputDim(),config.OutputDim()},stream,sdt));

    if(config.UseBias()==true)
    {
      SetBias(CAIF_DeviceTensor::Zeros({config.OutputDim()},stream,sdt));
      SetBiasGrad(CAIF_DeviceTensor::Zeros({config.OutputDim()},stream,sdt));
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT>::CAIF_DeviceLinearHead(const CAIF_DeviceLinearHeadConfig &config,
                                                                CAIF_DeviceTensor &tied_weight,
                                                                CAIF_DeviceTensor &tied_weight_grad,
                                                                CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
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
    if(config.InputDim()==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: input_dim must be > 0");
    }
    if(config.OutputDim()==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: output_dim must be > 0");
    }

    const std::vector<uint32_t> &tied_shape=tied_weight.Shape();
    if(tied_shape.size()!=2)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: tied weight must be 2D");
    }
    if(tied_shape[0]!=config.OutputDim()||tied_shape[1]!=config.InputDim())
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: tied weight shape must be [output_dim, input_dim]");
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    if(config.UseBias()==true)
    {
      SetBias(CAIF_DeviceTensor::Zeros({config.OutputDim()},stream,sdt));
      SetBiasGrad(CAIF_DeviceTensor::Zeros({config.OutputDim()},stream,sdt));
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT>::CAIF_DeviceLinearHead(CAIF_DeviceLinearHead &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT> &
CAIF_DeviceLinearHead<ComputeT,StorageT>::operator=(CAIF_DeviceLinearHead &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    SetConfig(other.Config());
    SetWeightTied(other.IsWeightTied());
    SetFrozen(other.Frozen());
    SetWeight(std::move(other.WeightMut()));
    SetWeightGrad(std::move(other.WeightGrad()));
    SetTiedWeight(other.TiedWeight());
    SetTiedWeightGrad(other.TiedWeightGrad());
    SetBias(std::move(other.Bias()));
    SetBiasGrad(std::move(other.BiasGrad()));
    SetCachedInput(std::move(other.CachedInput()));
    SetCachedShape(std::move(other.CachedShape()));
    other._tied_weight=nullptr;
    other._tied_weight_grad=nullptr;
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceLinearHead<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                       CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);
    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::Forward: input shape is empty");
    }
    if(shape.back()!=Config().InputDim())
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::Forward: last dim must match input_dim");
    }

    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }

    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({n,Config().InputDim()});

    CAIF_DeviceTensor output=AllocateOutput({n,Config().OutputDim()},ctx);

    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();
    if(IsWeightTied()==true)
    {
      CAIF_Ops::MatMulTransposeB(flat_input,*TiedWeight(),output,ctx,cdt);
    }
    else
    {
      CAIF_Ops::MatMul(flat_input,WeightMut(),output,ctx,cdt);
    }

    if(Config().UseBias()==true)
    {
      CAIF_Ops::BiasAdd(output,Bias(),output);
    }

    if(OutputScale()!=1.0f)
    {
      CAIF_Ops::Scale(output,OutputScale());
    }

    std::vector<uint32_t> out_shape;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      out_shape.push_back(shape[i]);
    }
    out_shape.push_back(Config().OutputDim());
    output.Reshape(out_shape);

    if(ctx.Training()==true)
    {
      SetCachedInput(input.Clone());
      SetCachedShape(std::vector<uint32_t>(shape.begin(),shape.end()));
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceLinearHead<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                        CAIF_RunContext &ctx)
{
  try
  {
    if(CachedShape().empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::Backward: must call Forward with training=true first");
    }

    const std::vector<uint32_t> &grad_shape=grad_output.Shape();
    uint32_t n=1;
    for(size_t i=0;i<grad_shape.size()-1;++i)
    {
      n*=grad_shape[i];
    }

    CAIF_DeviceTensor flat_grad=grad_output.Clone();
    flat_grad.Reshape({n,Config().OutputDim()});

    // F6: the forward scaled the logits by output_scale, so every downstream
    // gradient (input and weights) carries that factor (chain rule). flat_grad
    // is already a clone, so scale it in place.
    if(OutputScale()!=1.0f)
    {
      CAIF_Ops::Scale(flat_grad,OutputScale());
    }

    CAIF_DeviceTensor flat_input=CachedInput().Clone();
    flat_input.Reshape({n,Config().InputDim()});

    CAIF_DeviceTensor grad_input=AllocateOutput({n,Config().InputDim()},ctx);

    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();
    if(Frozen()==true)
    {
      if(IsWeightTied()==true)
      {
        CAIF_Ops::MatMul(flat_grad,*TiedWeight(),grad_input,ctx,cdt);
      }
      else
      {
        CAIF_Ops::MatMulTransposeB(flat_grad,WeightMut(),grad_input,ctx,cdt);
      }
    }
    else
    {
      if(Config().UseBias()==true)
      {
        CAIF_Ops::BiasGradient(flat_grad,BiasGrad());
      }

      if(IsWeightTied()==true)
      {
        const CAIF_DataType::CAIF_DataType_e idt=flat_grad.Dtype();
        const CAIF_DataType::CAIF_DataType_e wdt=TiedWeightGrad()->Dtype();
        CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized(
                                          {Config().OutputDim(),Config().InputDim()},
                                          ctx.Stream(),idt);
        CAIF_Ops::MatMulTransposeA(flat_grad,flat_input,grad_w_delta,ctx,cdt);
        if(wdt==idt)
        {
          CAIF_Ops::Add(*TiedWeightGrad(),grad_w_delta,*TiedWeightGrad());
        }
        else
        {
          // The shared embedding gradient is kept in fp32 while the head runs
          // in a lower storage dtype; upcast the delta so the tied-weight
          // accumulation stays in fp32 (matches CAIF_DeviceTokenEmbedding).
          CAIF_DeviceTensor grad_w_delta_acc=CAIF_DeviceTensor::Uninitialized(
                                               {Config().OutputDim(),Config().InputDim()},
                                               ctx.Stream(),wdt);
          CAIF_Ops::Cast(grad_w_delta,grad_w_delta_acc,ctx);
          CAIF_Ops::Add(*TiedWeightGrad(),grad_w_delta_acc,*TiedWeightGrad());
        }

        CAIF_Ops::MatMul(flat_grad,*TiedWeight(),grad_input,ctx,cdt);
      }
      else
      {
        const CAIF_DataType::CAIF_DataType_e wdt=WeightGrad().Dtype();
        CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized(
                                          {Config().InputDim(),Config().OutputDim()},
                                          ctx.Stream(),wdt);
        CAIF_Ops::MatMulTransposeA(flat_input,flat_grad,grad_w_delta,ctx,cdt);
        CAIF_Ops::Add(WeightGrad(),grad_w_delta,WeightGrad());

        CAIF_Ops::MatMulTransposeB(flat_grad,WeightMut(),grad_input,ctx,cdt);
      }
    }

    grad_input.Reshape(CachedShape());
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLinearHead<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    if(IsWeightTied()==false)
    {
      WeightGrad().FillZero();
    }
    if(Config().UseBias()==true)
    {
      BiasGrad().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterTensorCount()const
{
  try
  {
    size_t count=0;
    if(IsWeightTied()==false)
    {
      count+=1;
    }
    if(Config().UseBias()==true)
    {
      count+=1;
    }
    return count;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(IsWeightTied()==false)
    {
      if(index==0)
      {
        return _weight;
      }
      if(index==1&&Config().UseBias()==true)
      {
        return _bias;
      }
    }
    else
    {
      if(index==0&&Config().UseBias()==true)
      {
        return _bias;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(IsWeightTied()==false)
    {
      if(index==0)
      {
        return _weight;
      }
      if(index==1&&Config().UseBias()==true)
      {
        return _bias;
      }
    }
    else
    {
      if(index==0&&Config().UseBias()==true)
      {
        return _bias;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(IsWeightTied()==false)
    {
      if(index==0)
      {
        return _weight_grad;
      }
      if(index==1&&Config().UseBias()==true)
      {
        return _bias_grad;
      }
    }
    else
    {
      if(index==0&&Config().UseBias()==true)
      {
        return _bias_grad;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(IsWeightTied()==false)
    {
      if(index==0)
      {
        return _weight_grad;
      }
      if(index==1&&Config().UseBias()==true)
      {
        return _bias_grad;
      }
    }
    else
    {
      if(index==0&&Config().UseBias()==true)
      {
        return _bias_grad;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLinearHead<ComputeT,StorageT>::TotalParameterCount()const
{
  try
  {
    size_t total=0;
    if(IsWeightTied()==false)
    {
      total+=static_cast<size_t>(Config().InputDim())*Config().OutputDim();
    }
    if(Config().UseBias()==true)
    {
      total+=Config().OutputDim();
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceLinearHead<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc=std::string(g_serial_tag_linear_head)+
                     g_serial_open_paren+
                     g_serial_kv_in+
                     std::to_string(Config().InputDim())+
                     g_serial_comma+
                     g_serial_kv_out+
                     std::to_string(Config().OutputDim());
    if(IsWeightTied()==true)
    {
      desc+=g_serial_flag_tied_true;
    }
    desc+=g_serial_close_paren;
    return desc;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
    std::vector<std::string> names;
    if(IsWeightTied()==false)
    {
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::LinearHeadWeight_e));
    }
    if(Config().UseBias()==true)
    {
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::LinearHeadBias_e));
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLinearHead<ComputeT,StorageT>::LoadWeight(CAIF_DeviceTensor &&weight)
{
  try
  {
    if(IsWeightTied()==true)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadWeight: cannot load weight into a weight-tied head");
    }
    const std::vector<uint32_t> &shape=weight.Shape();
    if(shape.size()!=2 ||
       shape[0]!=Config().InputDim() ||
       shape[1]!=Config().OutputDim())
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadWeight: shape mismatch, expected [input_dim, output_dim]");
    }
    SetWeight(std::move(weight));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLinearHead<ComputeT,StorageT>::LoadBias(CAIF_DeviceTensor &&bias)
{
  try
  {
    if(Config().UseBias()==false)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadBias: use_bias is false");
    }
    const std::vector<uint32_t> &shape=bias.Shape();
    if(shape.size()!=1 || shape[0]!=Config().OutputDim())
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadBias: shape mismatch, expected [output_dim]");
    }
    SetBias(std::move(bias));
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceLinearHead<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceLinearHead<float,__half>;
template class CAIF_DeviceLinearHead<float,__nv_bfloat16>;
template class CAIF_DeviceLinearHead<__half,float>;
template class CAIF_DeviceLinearHead<__half,__half>;
template class CAIF_DeviceLinearHead<__half,__nv_bfloat16>;
template class CAIF_DeviceLinearHead<__nv_bfloat16,float>;
template class CAIF_DeviceLinearHead<__nv_bfloat16,__half>;
template class CAIF_DeviceLinearHead<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
