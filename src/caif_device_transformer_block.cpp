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
#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_constants.h"
#include "caif_exception.h"

#include <cstdint>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::CAIF_DeviceTransformerBlock(
                                                  const TransformerBlockConfig_t &config,
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
    if(config.dim==0)
    {
      THROW_CAIFE("TransformerBlock: dim must be > 0");
    }
    if(config.num_heads==0)
    {
      THROW_CAIFE("TransformerBlock: num_heads must be > 0");
    }
    if(config.num_kv_heads==0)
    {
      THROW_CAIFE("TransformerBlock: num_kv_heads must be > 0");
    }
    if(config.dim%config.num_heads!=0)
    {
      THROW_CAIFE("TransformerBlock: dim must be divisible by num_heads");
    }

    if(config.ffn_dim==0)
    {
      _effective_ffn_dim=ComputeDefaultFFNDim(config.dim);
    }
    else
    {
      _effective_ffn_dim=config.ffn_dim;
    }

    auto norm1=std::make_unique<CAIF_DeviceRMSNorm<ComputeT,StorageT>>(config.dim,stream);
    _norm1=norm1.get();
    AddLayer(std::move(norm1));

    typename CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::AttentionConfig_t attn_config;
    attn_config.dim=config.dim;
    attn_config.num_heads=config.num_heads;
    attn_config.num_kv_heads=config.num_kv_heads;
    attn_config.head_dim=config.dim/config.num_heads;
    attn_config.causal=config.causal;
    attn_config.use_rope=config.use_rope;
    attn_config.rope_base=config.rope_base;
    attn_config.rope_style=config.rope_style;
    attn_config.dropout_rate=config.dropout_rate;
    auto attention=std::make_unique<CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>>(attn_config,stream);
    _attention=attention.get();
    AddLayer(std::move(attention));

    auto norm2=std::make_unique<CAIF_DeviceRMSNorm<ComputeT,StorageT>>(config.dim,stream);
    _norm2=norm2.get();
    AddLayer(std::move(norm2));

    typename CAIF_DeviceFFN<ComputeT,StorageT>::FFNConfig_t ffn_config;
    ffn_config.dim=config.dim;
    ffn_config.ffn_dim=_effective_ffn_dim;
    auto ffn=std::make_unique<CAIF_DeviceFFN<ComputeT,StorageT>>(ffn_config,
                                                                 std::move(activation),
                                                                 stream);
    _ffn=ffn.get();
    AddLayer(std::move(ffn));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::CAIF_DeviceTransformerBlock(
                                                  const TransformerBlockConfig_t &config,
                                                  CAIF_CudaStream &stream):
                  CAIF_DeviceTransformerBlock(config,
                                              std::make_unique<CAIF_DeviceSwiGLUActivation<ComputeT,StorageT>>(),
                                              stream)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT>::CAIF_DeviceTransformerBlock(
                              CAIF_DeviceTransformerBlock &&other):CAIF_DeviceContainer(std::move(other)),
                                                                   _config(other._config),
                                                                   _effective_ffn_dim(other._effective_ffn_dim),
                                                                   _norm1(other._norm1),
                                                                   _attention(other._attention),
                                                                   _norm2(other._norm2),
                                                                   _ffn(other._ffn)
{
  other._norm1=nullptr;
  other._attention=nullptr;
  other._norm2=nullptr;
  other._ffn=nullptr;
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
      _config=other._config;
      _effective_ffn_dim=other._effective_ffn_dim;
      _norm1=other._norm1;
      _attention=other._attention;
      _norm2=other._norm2;
      _ffn=other._ffn;
      other._norm1=nullptr;
      other._attention=nullptr;
      other._norm2=nullptr;
      other._ffn=nullptr;
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
    CAIF_DeviceTensor norm1_out=_norm1->Forward(input,ctx);
    CAIF_DeviceTensor attn_out=_attention->Forward(norm1_out,ctx);
    CAIF_DeviceTensor h=CAIF_DeviceTensor::Uninitialized(input.Shape(),Stream(),sd);
    CAIF_Ops::Add(input,attn_out,h);

    // Pre-norm residual 2: out = h + ffn(norm2(h))
    CAIF_DeviceTensor norm2_out=_norm2->Forward(h,ctx);
    CAIF_DeviceTensor ffn_out=_ffn->Forward(norm2_out,ctx);
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
    CAIF_DeviceTensor d_norm2_out=_ffn->Backward(grad_output,ctx);
    CAIF_DeviceTensor d_h_from_ffn=_norm2->Backward(d_norm2_out,ctx);
    CAIF_DeviceTensor d_h=CAIF_DeviceTensor::Uninitialized(grad_output.Shape(),Stream(),sd);
    CAIF_Ops::Add(grad_output,d_h_from_ffn,d_h);

    // Backward through residual 1.
    CAIF_DeviceTensor d_norm1_out=_attention->Backward(d_h,ctx);
    CAIF_DeviceTensor d_x_from_attn=_norm1->Backward(d_norm1_out,ctx);
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
    if(_config.causal==true)
    {
      causal_str="true";
    }
    else
    {
      causal_str="false";
    }
    return std::string(g_caif_description_transformer_block_prefix)+
           "(dim="+std::to_string(_config.dim)+
           ",heads="+std::to_string(_config.num_heads)+
           ",kv_heads="+std::to_string(_config.num_kv_heads)+
           ",ffn_dim="+std::to_string(_effective_ffn_dim)+
           ",causal="+causal_str+")";
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
    auto norm1_names=Norm1().ParameterNames(prefix+g_caif_name_input_layernorm);
    names.insert(names.end(),norm1_names.begin(),norm1_names.end());

    auto attn_names=Attention().ParameterNames(prefix+g_caif_name_self_attn);
    names.insert(names.end(),attn_names.begin(),attn_names.end());

    auto norm2_names=Norm2().ParameterNames(prefix+g_caif_name_post_attention_layernorm);
    names.insert(names.end(),norm2_names.begin(),norm2_names.end());

    auto ffn_names=FFN().ParameterNames(prefix+g_caif_name_mlp);
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
    sub=Norm1().FrozenTensorNames(prefix+g_caif_name_input_layernorm);
    names.insert(names.end(),sub.begin(),sub.end());
    sub=Attention().FrozenTensorNames(prefix+g_caif_name_self_attn);
    names.insert(names.end(),sub.begin(),sub.end());
    sub=Norm2().FrozenTensorNames(prefix+g_caif_name_post_attention_layernorm);
    names.insert(names.end(),sub.begin(),sub.end());
    sub=FFN().FrozenTensorNames(prefix+g_caif_name_mlp);
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
