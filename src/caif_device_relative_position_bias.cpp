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

#include "caif_device_relative_position_bias.h"
#include "caif_device_relative_position_bias_factory.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include <cmath>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::CAIF_DeviceRelativePositionBias(
                                                  const Config_t &config,
                                                  CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _embedding(),
                                          _grad_embedding()
{
  try
  {
    if(config.num_heads==0)
    {
      THROW_CAIFE("CAIF_DeviceRelativePositionBias: num_heads must be > 0");
    }
    if(config.num_buckets==0)
    {
      THROW_CAIFE("CAIF_DeviceRelativePositionBias: num_buckets must be > 0");
    }

    const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                 static_cast<float>(config.num_heads+config.num_buckets));
    const size_t total=static_cast<size_t>(config.num_heads)*config.num_buckets;
    std::vector<float> init_data(total);
    for(size_t i=0;i<total;++i)
    {
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      init_data[i]=(t-std::floor(t))*2.0f*limit-limit;
    }

    _embedding=CAIF_DeviceTensor::Uninitialized({config.num_heads,config.num_buckets},stream);
    _embedding.CopyFromHost(init_data.data(),total);

    _grad_embedding=CAIF_DeviceTensor::Zeros({config.num_heads,config.num_buckets},stream);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::CAIF_DeviceRelativePositionBias(
                                              CAIF_DeviceRelativePositionBias &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _config(other._config),
                              _embedding(std::move(other._embedding)),
                              _grad_embedding(std::move(other._grad_embedding))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceRelativePositionBias<ComputeT,StorageT> &
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::operator=(CAIF_DeviceRelativePositionBias &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      _config=other._config;
      _embedding=std::move(other._embedding);
      _grad_embedding=std::move(other._grad_embedding);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::ComputeBias(uint32_t q_len,uint32_t k_len)
{
  try
  {
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({_config.num_heads,q_len,k_len},
                                                              Stream(),StorageDtype());

    CAIF_Ops::ComputeRelativePositionBias(_embedding,
                                          output,
                                          _config.max_distance,
                                          _config.bidirectional);

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::AccumulateGradient(
    const CAIF_DeviceTensor &grad_bias,
    uint32_t q_len,
    uint32_t k_len)
{
  try
  {
    (void)q_len;
    (void)k_len;

    CAIF_Ops::AccumulateRelativePositionBiasGradient(grad_bias,
                                                     _grad_embedding,
                                                     _config.max_distance,
                                                     _config.bidirectional);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                                  CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("CAIF_DeviceRelativePositionBias::Forward: input must be >= 2D");
    }
    const uint32_t seq_len=shape[1];
    return ComputeBias(seq_len,seq_len);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                                   CAIF_RunContext &ctx)
{
  try
  {
    (void)grad_output;
    (void)ctx;
    return CAIF_DeviceTensor();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    _grad_embedding.FillZero();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::ParameterTensorCount()const
{
  return 1;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index!=0)
    {
      THROW_CAIFE("CAIF_DeviceRelativePositionBias::ParameterTensor: index out of range");
    }
    return _embedding;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index!=0)
    {
      THROW_CAIFE("CAIF_DeviceRelativePositionBias::ParameterTensor: index out of range");
    }
    return _embedding;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index!=0)
    {
      THROW_CAIFE("CAIF_DeviceRelativePositionBias::GradientTensor: index out of range");
    }
    return _grad_embedding;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index!=0)
    {
      THROW_CAIFE("CAIF_DeviceRelativePositionBias::GradientTensor: index out of range");
    }
    return _grad_embedding;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::TotalParameterCount()const
{
  return static_cast<size_t>(_config.num_heads)*_config.num_buckets;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string bidir_str="false";
    if(_config.bidirectional==true)
    {
      bidir_str="true";
    }
    return "RelativePositionBias(heads="+
           std::to_string(_config.num_heads)+
           ",buckets="+
           std::to_string(_config.num_buckets)+
           ",max_dist="+
           std::to_string(_config.max_distance)+
           ",bidir="+
           bidir_str+
           ")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceRelativePositionBias<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"relative_attention_bias.weight");
    return names;
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceRelativePositionBias<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceRelativePositionBias<float,__half>;
template class CAIF_DeviceRelativePositionBias<float,__nv_bfloat16>;
template class CAIF_DeviceRelativePositionBias<__half,float>;
template class CAIF_DeviceRelativePositionBias<__half,__half>;
template class CAIF_DeviceRelativePositionBias<__half,__nv_bfloat16>;
template class CAIF_DeviceRelativePositionBias<__nv_bfloat16,float>;
template class CAIF_DeviceRelativePositionBias<__nv_bfloat16,__half>;
template class CAIF_DeviceRelativePositionBias<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
