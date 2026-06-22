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
#include "caif_ops.h"
#include "caif_host_tensor.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <random>
#include <cmath>
#include <ctime>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::CAIF_DeviceLoRAAdapter(const CAIF_DeviceLoRAAdapterConfig &config,
                                             std::unique_ptr<CAIF_DeviceLayer> base_layer,
                                             CAIF_CudaStream &stream,
                                             uint32_t seed):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
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
    if(config.Rank()==0||config.InputDim()==0||config.OutputDim()==0)
    {
      THROW_CAIFE("LoRAAdapter: rank, input_dim, output_dim must be > 0");
    }
    if(HasBaseLayer()==false)
    {
      THROW_CAIFE("LoRAAdapter: base_layer must not be null");
    }

    // Allocate LoRA tensors at the templated storage dtype.
    // A=[rank, input_dim], B=[output_dim, rank]
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    SetLoRAA(CAIF_DeviceTensor::Uninitialized({config.Rank(),config.InputDim()},stream,sd));
    SetLoRAB(CAIF_DeviceTensor::Zeros({config.OutputDim(),config.Rank()},stream,sd));

    SetGradLoRAA(CAIF_DeviceTensor::Zeros({config.Rank(),config.InputDim()},stream,sd));
    SetGradLoRAB(CAIF_DeviceTensor::Zeros({config.OutputDim(),config.Rank()},stream,sd));

    // Initialize A with Kaiming uniform: U(-bound, bound), bound=sqrt(1/input_dim)
    if(seed==0)
    {
      seed=static_cast<uint32_t>(std::time(nullptr));
    }
    const float bound=std::sqrt(1.0f/static_cast<float>(config.InputDim()));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-bound,bound);

    const size_t a_count=static_cast<size_t>(config.Rank())*config.InputDim();
    std::vector<float> host_a(a_count);
    for(size_t i=0;i<a_count;++i)
    {
      host_a[i]=dist(gen);
    }
    LoRAAMut().CopyFromHostFp32(host_a.data(),a_count);

    // B is already zeros (LoRA starts as identity)
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::CAIF_DeviceLoRAAdapter(CAIF_DeviceLoRAAdapter<ComputeT,StorageT> &&other):CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceLoRAAdapter<ComputeT,StorageT> &CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::operator=(CAIF_DeviceLoRAAdapter<ComputeT,StorageT> &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      SetConfig(other.Config());
      _base_layer=std::move(other._base_layer);
      SetLoRAA(std::move(other.LoRAAMut()));
      SetLoRAB(std::move(other.LoRABMut()));
      SetGradLoRAA(std::move(other.GradLoRAA()));
      SetGradLoRAB(std::move(other.GradLoRAB()));
      SetCachedInput(std::move(other.CachedInput()));
      SetCachedLoRAHidden(std::move(other.CachedLoRAHidden()));
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                     CAIF_RunContext &ctx)
{
  try
  {
    if(CAIF_DeviceLayer::HasStream()==false)
    {
      THROW_CAIFE("LoRAAdapter: stream is null");
    }

    const auto &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("LoRAAdapter::Forward: input must be at least 2D");
    }
    if(shape.back()!=Config().InputDim())
    {
      THROW_CAIFE("LoRAAdapter::Forward: last dim must match input_dim");
    }

    // Base layer forward
    CAIF_DeviceTensor base_out=BaseLayer().Forward(input,ctx);

    // Reshape to 2D for LoRA matmuls
    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }
    CAIF_DeviceTensor input_2d=input.Clone();
    input_2d.Reshape({n,Config().InputDim()});

    // Cache for backward
    if(ctx.Training()==true)
    {
      SetCachedInput(input_2d.Clone());
    }

    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    // lora_hidden = input_2d @ A^T: [N, input_dim] @ [input_dim, rank] = [N, rank]
    CAIF_DeviceTensor lora_hidden=CAIF_DeviceTensor::Uninitialized({n,Config().Rank()},Stream(),sd);
    CAIF_Ops::MatMulTransposeB(input_2d,LoRAAMut(),lora_hidden,ctx,cdt);

    // Cache for backward
    if(ctx.Training()==true)
    {
      SetCachedLoRAHidden(lora_hidden.Clone());
    }

    // lora_out = lora_hidden @ B^T: [N, rank] @ [rank, output_dim] = [N, output_dim]
    CAIF_DeviceTensor lora_out=CAIF_DeviceTensor::Uninitialized({n,Config().OutputDim()},Stream(),sd);
    CAIF_Ops::MatMulTransposeB(lora_hidden,LoRABMut(),lora_out,ctx,cdt);

    // Scale by alpha/rank
    const float scale=Config().Alpha()/static_cast<float>(Config().Rank());
    CAIF_Ops::Scale(lora_out,scale);

    // Reshape lora_out to match base_out shape
    if(shape.size()>2)
    {
      std::vector<uint32_t> out_shape(shape.begin(),shape.end()-1);
      out_shape.push_back(Config().OutputDim());
      lora_out.Reshape(out_shape);
    }

    // output = base_out + lora_out — base may emit fp32 (e.g. FrozenLinear's
    // fp32-accumulated output); cast to StorageT so the Add matches lora_out.
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(base_out.Shape(),Stream(),sd);
    if(base_out.Dtype()==sd)
    {
      CAIF_Ops::Add(base_out,lora_out,output);
    }
    else
    {
      CAIF_DeviceTensor base_out_cast=base_out.To(sd);
      CAIF_Ops::Add(base_out_cast,lora_out,output);
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)
{
  try
  {
    if(CAIF_DeviceLayer::HasStream()==false)
    {
      THROW_CAIFE("LoRAAdapter: stream is null");
    }
    if(CachedInput().IsEmpty()==true)
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
    grad_2d.Reshape({n,Config().OutputDim()});

    const float scale=Config().Alpha()/static_cast<float>(Config().Rank());
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    // d_lora_scaled = grad_output * scale
    CAIF_DeviceTensor d_lora_scaled=grad_2d.Clone();
    CAIF_Ops::Scale(d_lora_scaled,scale);

    // _grad_lora_b += d_lora_scaled^T @ _cached_lora_hidden
    // [output_dim, N] @ [N, rank] = [output_dim, rank]
    CAIF_Ops::MatMulTransposeA(d_lora_scaled,CachedLoRAHidden(),GradLoRAB(),ctx,cdt);

    // d_lora_hidden = d_lora_scaled @ B: [N, output_dim] @ [output_dim, rank] = [N, rank]
    CAIF_DeviceTensor d_lora_hidden=CAIF_DeviceTensor::Uninitialized({n,Config().Rank()},Stream(),sd);
    CAIF_Ops::MatMul(d_lora_scaled,LoRABMut(),d_lora_hidden,ctx,cdt);

    // _grad_lora_a += d_lora_hidden^T @ _cached_input
    // [rank, N] @ [N, input_dim] = [rank, input_dim]
    CAIF_Ops::MatMulTransposeA(d_lora_hidden,CachedInput(),GradLoRAA(),ctx,cdt);

    // d_input_lora = d_lora_hidden @ A: [N, rank] @ [rank, input_dim] = [N, input_dim]
    CAIF_DeviceTensor d_input_lora=CAIF_DeviceTensor::Uninitialized({n,Config().InputDim()},Stream(),sd);
    CAIF_Ops::MatMul(d_lora_hidden,LoRAAMut(),d_input_lora,ctx,cdt);

    // Base layer backward
    CAIF_DeviceTensor grad_base=BaseLayer().Backward(grad_output,ctx);

    // Reshape d_input_lora to match grad_base shape
    if(shape.size()>2)
    {
      std::vector<uint32_t> in_shape(shape.begin(),shape.end()-1);
      in_shape.push_back(Config().InputDim());
      d_input_lora.Reshape(in_shape);
    }

    // grad_input = grad_base + d_input_lora — base layer (e.g. FrozenLinear)
    // may emit fp32 grads; cast to StorageT to match d_input_lora.
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized(grad_base.Shape(),Stream(),sd);
    if(grad_base.Dtype()==sd)
    {
      CAIF_Ops::Add(grad_base,d_input_lora,grad_input);
    }
    else
    {
      CAIF_DeviceTensor grad_base_cast=grad_base.To(sd);
      CAIF_Ops::Add(grad_base_cast,d_input_lora,grad_input);
    }

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    GradLoRAA().FillZero();
    GradLoRAB().FillZero();
    BaseLayer().ZeroGradients();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::ParameterTensorCount()const
{
  try
  {
    return BaseLayer().ParameterTensorCount()+2;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    const size_t base_count=BaseLayer().ParameterTensorCount();
    if(index<base_count)
    {
      return _base_layer->ParameterTensor(index);
    }
    if(index==base_count)
    {
      return _lora_a;
    }
    if(index==base_count+1)
    {
      return _lora_b;
    }
    THROW_CAIFE("LoRAAdapter::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    const size_t base_count=BaseLayer().ParameterTensorCount();
    if(index<base_count)
    {
      return _base_layer->ParameterTensor(index);
    }
    if(index==base_count)
    {
      return _lora_a;
    }
    if(index==base_count+1)
    {
      return _lora_b;
    }
    THROW_CAIFE("LoRAAdapter::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    const size_t base_count=BaseLayer().ParameterTensorCount();
    if(index<base_count)
    {
      return _base_layer->GradientTensor(index);
    }
    if(index==base_count)
    {
      return _grad_lora_a;
    }
    if(index==base_count+1)
    {
      return _grad_lora_b;
    }
    THROW_CAIFE("LoRAAdapter::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    const size_t base_count=BaseLayer().ParameterTensorCount();
    if(index<base_count)
    {
      return _base_layer->GradientTensor(index);
    }
    if(index==base_count)
    {
      return _grad_lora_a;
    }
    if(index==base_count+1)
    {
      return _grad_lora_b;
    }
    THROW_CAIFE("LoRAAdapter::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::TotalParameterCount()const
{
  try
  {
    return _base_layer->TotalParameterCount()+
           static_cast<size_t>(Config().Rank())*Config().InputDim()+
           static_cast<size_t>(Config().OutputDim())*Config().Rank();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc=std::string(g_serial_tag_lora)+
                     g_serial_open_paren+
                     g_serial_kv_rank+
                     std::to_string(Config().Rank())+
                     g_serial_comma+
                     g_serial_kv_alpha+
                     std::to_string(static_cast<int>(Config().Alpha()))+
                     g_serial_comma+
                     std::to_string(Config().InputDim())+
                     g_serial_comma+
                     std::to_string(Config().OutputDim())+
                     g_serial_close_paren;
    if(HasBaseLayer()==true)
    {
      desc+=g_serial_lora_plus;
      desc+=BaseLayer().Description();
    }
    return desc;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::ParameterNames(
    const std::string &prefix)const
{
  try
  {
    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
    std::vector<std::string> names=BaseLayer().ParameterNames(prefix);
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::LoRA_A_e)+"."+reg.Name(CAIF_ParamRole::Role_e::GenericWeight_e));
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::LoRA_B_e)+"."+reg.Name(CAIF_ParamRole::Role_e::GenericWeight_e));
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::FrozenTensorCount()const
{
  try
  {
    return BaseLayer().FrozenTensorCount();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::FrozenTensorFP32(size_t index)const
{
  try
  {
    return BaseLayer().FrozenTensorFP32(index);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::FrozenTensorNames(
    const std::string &prefix)const
{
  try
  {
    return BaseLayer().FrozenTensorNames(prefix);
  }
  CAIF_CATCH_BLOCK()
}
template class CAIF_DeviceLoRAAdapter<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceLoRAAdapter<float,__half>;
template class CAIF_DeviceLoRAAdapter<float,__nv_bfloat16>;
template class CAIF_DeviceLoRAAdapter<__half,float>;
template class CAIF_DeviceLoRAAdapter<__half,__half>;
template class CAIF_DeviceLoRAAdapter<__half,__nv_bfloat16>;
template class CAIF_DeviceLoRAAdapter<__nv_bfloat16,float>;
template class CAIF_DeviceLoRAAdapter<__nv_bfloat16,__half>;
template class CAIF_DeviceLoRAAdapter<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
