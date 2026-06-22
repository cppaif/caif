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

#include "caif_device_transformer_block.h"
#include "caif_ops.h"
#include "caif_role_registry.h"
#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

#include <cstdint>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::CAIF_DeviceTransformerBlock(
                                                  const CAIF_DeviceTransformerBlockConfig &config,
                                                  std::unique_ptr<CAIF_DeviceActivation> activation,
                                                  CAIF_CudaStream &stream):CAIF_DeviceContainer(stream),
                                                                           _config(config),
                                                                           _effective_ffn_dim(0),
                                                                           _norm1(nullptr),
                                                                           _attention(nullptr),
                                                                           _norm2(nullptr),
                                                                           _ffn(nullptr)
{
  try
  {
    if(config.Dim()==0)
    {
      THROW_CAIFE("TransformerBlock: dim must be > 0");
    }
    if(config.NumHeads()==0)
    {
      THROW_CAIFE("TransformerBlock: num_heads must be > 0");
    }
    if(config.NumKvHeads()==0)
    {
      THROW_CAIFE("TransformerBlock: num_kv_heads must be > 0");
    }
    if(config.Dim()%config.NumHeads()!=0)
    {
      THROW_CAIFE("TransformerBlock: dim must be divisible by num_heads");
    }

    if(config.FfnDim()==0)
    {
      SetEffectiveFFNDim(ComputeDefaultFFNDim(config.Dim()));
    }
    else
    {
      SetEffectiveFFNDim(config.FfnDim());
    }

    auto norm1=std::make_unique<CAIF_DeviceRMSNorm<ComputeT,StorageT>>(config.Dim(),stream);
    SetNorm1Ptr(norm1.get());
    AddLayer(std::move(norm1));

    CAIF_DeviceMultiHeadAttentionConfig attn_config(config.Dim(),
                                                    config.NumHeads(),
                                                    config.NumKvHeads(),
                                                    config.Dim()/config.NumHeads(),
                                                    config.Causal(),
                                                    config.UseRope(),
                                                    config.RopeBase(),
                                                    config.DropoutRate());
    attn_config.SetRopeStyle(config.RopeStyle());
    auto attention=std::make_unique<CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>>(attn_config,
                                                                                     stream);
    SetAttentionPtr(attention.get());
    AddLayer(std::move(attention));

    auto norm2=std::make_unique<CAIF_DeviceRMSNorm<ComputeT,StorageT>>(config.Dim(),stream);
    SetNorm2Ptr(norm2.get());
    AddLayer(std::move(norm2));

    CAIF_DeviceFFNConfig ffn_config(config.Dim(),EffectiveFFNDim());
    auto ffn=std::make_unique<CAIF_DeviceFFN<ComputeT,StorageT>>(ffn_config,
                                                                 std::move(activation),
                                                                 stream);
    SetFFNPtr(ffn.get());
    AddLayer(std::move(ffn));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::CAIF_DeviceTransformerBlock(
                                                  const CAIF_DeviceTransformerBlockConfig &config,
                                                  CAIF_CudaStream &stream):
                  CAIF_DeviceTransformerBlock(config,
                                              std::make_unique<CAIF_DeviceSwiGLUActivation<ComputeT,StorageT>>(),
                                              stream)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::CAIF_DeviceTransformerBlock(
                              CAIF_DeviceTransformerBlock &&other):CAIF_DeviceContainer(std::move(other)),
                                                                   _config(other.Config()),
                                                                   _effective_ffn_dim(other.EffectiveFFNDim()),
                                                                   _norm1(other.Norm1Ptr()),
                                                                   _attention(other.AttentionPtr()),
                                                                   _norm2(other.Norm2Ptr()),
                                                                   _ffn(other.FFNPtr())
{
  other.SetNorm1Ptr(nullptr);
  other.SetAttentionPtr(nullptr);
  other.SetNorm2Ptr(nullptr);
  other.SetFFNPtr(nullptr);
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT> &
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::operator=(CAIF_DeviceTransformerBlock &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceContainer::operator=(std::move(other));
      SetConfig(other.Config());
      SetEffectiveFFNDim(other.EffectiveFFNDim());
      SetNorm1Ptr(other.Norm1Ptr());
      SetAttentionPtr(other.AttentionPtr());
      SetNorm2Ptr(other.Norm2Ptr());
      SetFFNPtr(other.FFNPtr());
      other.SetNorm1Ptr(nullptr);
      other.SetAttentionPtr(nullptr);
      other.SetNorm2Ptr(nullptr);
      other.SetFFNPtr(nullptr);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                             CAIF_RunContext &ctx)
{
  try
  {
    if(HasStream()==false)
    {
      THROW_CAIFE("TransformerBlock: stream is null");
    }

    constexpr CAIF_DataType::CAIF_DataType_e sd=StorageDtype();

    // Pre-norm residual 1: h = x + attention(norm1(x))
    CAIF_DeviceTensor norm1_out=Norm1().Forward(input,ctx);
    CAIF_DeviceTensor attn_out=Attention().Forward(norm1_out,ctx);
    CAIF_DeviceTensor h=CAIF_DeviceTensor::Uninitialized(input.Shape(),Stream(),sd);
    CAIF_Ops::Add(input,attn_out,h);

    // Pre-norm residual 2: out = h + ffn(norm2(h))
    CAIF_DeviceTensor norm2_out=Norm2().Forward(h,ctx);
    CAIF_DeviceTensor ffn_out=FFN().Forward(norm2_out,ctx);
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(h.Shape(),Stream(),sd);
    CAIF_Ops::Add(h,ffn_out,output);

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                              CAIF_RunContext &ctx)
{
  try
  {
    if(HasStream()==false)
    {
      THROW_CAIFE("TransformerBlock: stream is null");
    }

    constexpr CAIF_DataType::CAIF_DataType_e sd=StorageDtype();

    // Backward through residual 2.
    CAIF_DeviceTensor d_norm2_out=FFN().Backward(grad_output,ctx);
    CAIF_DeviceTensor d_h_from_ffn=Norm2().Backward(d_norm2_out,ctx);
    CAIF_DeviceTensor d_h=CAIF_DeviceTensor::Uninitialized(grad_output.Shape(),Stream(),sd);
    CAIF_Ops::Add(grad_output,d_h_from_ffn,d_h);

    // Backward through residual 1.
    CAIF_DeviceTensor d_norm1_out=Attention().Backward(d_h,ctx);
    CAIF_DeviceTensor d_x_from_attn=Norm1().Backward(d_norm1_out,ctx);
    CAIF_DeviceTensor d_x=CAIF_DeviceTensor::Uninitialized(d_h.Shape(),Stream(),sd);
    CAIF_Ops::Add(d_h,d_x_from_attn,d_x);

    return d_x;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceTransformerBlock<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string causal_str;
    if(Config().Causal()==true)
    {
      causal_str=g_serial_json_true;
    }
    else
    {
      causal_str=g_serial_json_false;
    }
    return std::string(g_serial_tag_transformer_block)+
           g_serial_open_paren+
           g_serial_kv_dim+
           std::to_string(Config().Dim())+
           g_serial_comma+
           g_serial_kv_heads+
           std::to_string(Config().NumHeads())+
           g_serial_comma+
           g_serial_kv_kv_heads+
           std::to_string(Config().NumKvHeads())+
           g_serial_comma+
           g_serial_kv_ffn_dim+
           std::to_string(EffectiveFFNDim())+
           g_serial_comma+
           g_serial_kv_causal+
           causal_str+
           g_serial_close_paren;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    auto norm1_names=Norm1().ParameterNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathAttnNorm_e));
    names.insert(names.end(),norm1_names.begin(),norm1_names.end());

    auto attn_names=Attention().ParameterNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathAttn_e));
    names.insert(names.end(),attn_names.begin(),attn_names.end());

    auto norm2_names=Norm2().ParameterNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathFFNNorm_e));
    names.insert(names.end(),norm2_names.begin(),norm2_names.end());

    auto ffn_names=FFN().ParameterNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathFFN_e));
    names.insert(names.end(),ffn_names.begin(),ffn_names.end());

    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceTransformerBlock<ComputeT,StorageT>::FrozenTensorCount()const
{
  try
  {
    return Norm1().FrozenTensorCount()+
           Attention().FrozenTensorCount()+
           Norm2().FrozenTensorCount()+
           FFN().FrozenTensorCount();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::FrozenTensorFP32(size_t index)const
{
  try
  {
    size_t offset=0;
    const size_t n1=Norm1().FrozenTensorCount();
    if(index<offset+n1)
    {
      return Norm1().FrozenTensorFP32(index-offset);
    }
    offset+=n1;
    const size_t a=Attention().FrozenTensorCount();
    if(index<offset+a)
    {
      return Attention().FrozenTensorFP32(index-offset);
    }
    offset+=a;
    const size_t n2=Norm2().FrozenTensorCount();
    if(index<offset+n2)
    {
      return Norm2().FrozenTensorFP32(index-offset);
    }
    offset+=n2;
    const size_t f=FFN().FrozenTensorCount();
    if(index<offset+f)
    {
      return FFN().FrozenTensorFP32(index-offset);
    }
    THROW_CAIFE("DeviceTransformerBlock::FrozenTensorFP32: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::FrozenTensorNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    std::vector<std::string> sub;
    sub=Norm1().FrozenTensorNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathAttnNorm_e));
    names.insert(names.end(),sub.begin(),sub.end());
    sub=Attention().FrozenTensorNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathAttn_e));
    names.insert(names.end(),sub.begin(),sub.end());
    sub=Norm2().FrozenTensorNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathFFNNorm_e));
    names.insert(names.end(),sub.begin(),sub.end());
    sub=FFN().FrozenTensorNames(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathFFN_e));
    names.insert(names.end(),sub.begin(),sub.end());
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
uint32_t CAIF_DeviceTransformerBlock<ComputeT,StorageT>::ComputeDefaultFFNDim(uint32_t dim)
{
  // LLaMA-style: round_to(4 * dim * 2/3, 256)
  uint32_t raw=g_caif_ffn_multiplier_numerator*
               dim*
               g_caif_ffn_gated_numerator/
               g_caif_ffn_gated_denominator;
  uint32_t aligned=((raw+g_caif_ffn_alignment-1)/g_caif_ffn_alignment)*
                   g_caif_ffn_alignment;
  return aligned;
}

template class CAIF_DeviceTransformerBlock<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceTransformerBlock<float,__half>;
template class CAIF_DeviceTransformerBlock<float,__nv_bfloat16>;
template class CAIF_DeviceTransformerBlock<__half,float>;
template class CAIF_DeviceTransformerBlock<__half,__half>;
template class CAIF_DeviceTransformerBlock<__half,__nv_bfloat16>;
template class CAIF_DeviceTransformerBlock<__nv_bfloat16,float>;
template class CAIF_DeviceTransformerBlock<__nv_bfloat16,__half>;
template class CAIF_DeviceTransformerBlock<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
