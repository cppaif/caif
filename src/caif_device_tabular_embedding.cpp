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

#include "caif_device_tabular_embedding.h"
#include "caif_device_tabular_embedding_factory.h"
#include "caif_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <cmath>
#include <random>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::CAIF_DeviceTabularEmbedding(
                                              const Config_t &config,
                                              CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _w_proj(),
                                          _b_proj(),
                                          _grad_w_proj(),
                                          _grad_b_proj(),
                                          _cached_input(),
                                          _cached_batch(0),
                                          _cached_seq_len(0)
{
  try
  {
    if(config.num_features==0)
    {
      THROW_CAIFE("CAIF_DeviceTabularEmbedding: num_features must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("CAIF_DeviceTabularEmbedding: dim must be > 0");
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    _w_proj=CAIF_DeviceTensor::Zeros({config.num_features,config.dim},stream,sdt);
    _b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream,sdt);
    _grad_w_proj=CAIF_DeviceTensor::Zeros({config.num_features,config.dim},stream,sdt);
    _grad_b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream,sdt);

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::CAIF_DeviceTabularEmbedding(
                                              CAIF_DeviceTabularEmbedding &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _config(other._config),
                              _w_proj(std::move(other._w_proj)),
                              _b_proj(std::move(other._b_proj)),
                              _grad_w_proj(std::move(other._grad_w_proj)),
                              _grad_b_proj(std::move(other._grad_b_proj)),
                              _cached_input(std::move(other._cached_input)),
                              _cached_batch(other._cached_batch),
                              _cached_seq_len(other._cached_seq_len)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTabularEmbedding<ComputeT,StorageT> &
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::operator=(CAIF_DeviceTabularEmbedding &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    _config=other._config;
    _w_proj=std::move(other._w_proj);
    _b_proj=std::move(other._b_proj);
    _grad_w_proj=std::move(other._grad_w_proj);
    _grad_b_proj=std::move(other._grad_b_proj);
    _cached_input=std::move(other._cached_input);
    _cached_batch=other._cached_batch;
    _cached_seq_len=other._cached_seq_len;
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                              CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);

    const std::vector<uint32_t> &shape=input.Shape();
    uint32_t batch=0;
    uint32_t seq_len=1;
    uint32_t num_features=0;

    if(shape.size()==2)
    {
      batch=shape[0];
      num_features=shape[1];
      seq_len=1;
    }
    else if(shape.size()==3)
    {
      batch=shape[0];
      seq_len=shape[1];
      num_features=shape[2];
    }
    else
    {
      THROW_CAIFE("CAIF_DeviceTabularEmbedding: input must be 2D or 3D");
    }

    if(num_features!=_config.num_features)
    {
      THROW_CAIFE("CAIF_DeviceTabularEmbedding: input feature dim mismatch");
    }

    const uint32_t total_rows=batch*seq_len;
    const uint32_t dim=_config.dim;

    if(ctx.Training()==true)
    {
      _cached_input=input.Clone();
      _cached_batch=batch;
      _cached_seq_len=seq_len;
    }

    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({total_rows,num_features});

    CAIF_DeviceTensor output=AllocateOutput({total_rows,dim},ctx);
    CAIF_Ops::MatMul(flat_input,_w_proj,output,ctx);
    CAIF_Ops::BiasAdd(output,_b_proj,output);
    output.Reshape({batch,seq_len,dim});
    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                              CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(grad_output);

    const uint32_t batch=_cached_batch;
    const uint32_t seq_len=_cached_seq_len;
    const uint32_t num_features=_config.num_features;
    const uint32_t dim=_config.dim;
    const uint32_t total_rows=batch*seq_len;

    CAIF_DeviceTensor flat_grad=grad_output.Clone();
    flat_grad.Reshape({total_rows,dim});

    CAIF_Ops::BiasGradient(flat_grad,_grad_b_proj);

    CAIF_DeviceTensor flat_input=_cached_input.Clone();
    flat_input.Reshape({total_rows,num_features});
    CAIF_Ops::MatMulTransposeA(flat_input,flat_grad,_grad_w_proj,ctx);

    CAIF_DeviceTensor grad_input=AllocateOutput({total_rows,num_features},ctx);
    CAIF_Ops::MatMulTransposeB(flat_grad,_w_proj,grad_input,ctx);

    if(seq_len==1)
    {
      grad_input.Reshape({batch,num_features});
    }
    else
    {
      grad_input.Reshape({batch,seq_len,num_features});
    }
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    _grad_w_proj.FillZero();
    _grad_b_proj.FillZero();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::ParameterTensorCount()const
{
  return 2;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  if(index==0)
  {
    return _w_proj;
  }
  if(index==1)
  {
    return _b_proj;
  }
  THROW_CAIFE("CAIF_DeviceTabularEmbedding::ParameterTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  if(index==0)
  {
    return _w_proj;
  }
  if(index==1)
  {
    return _b_proj;
  }
  THROW_CAIFE("CAIF_DeviceTabularEmbedding::ParameterTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)
{
  if(index==0)
  {
    return _grad_w_proj;
  }
  if(index==1)
  {
    return _grad_b_proj;
  }
  THROW_CAIFE("CAIF_DeviceTabularEmbedding::GradientTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  if(index==0)
  {
    return _grad_w_proj;
  }
  if(index==1)
  {
    return _grad_b_proj;
  }
  THROW_CAIFE("CAIF_DeviceTabularEmbedding::GradientTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::TotalParameterCount()const
{
  return _config.num_features*_config.dim+_config.dim;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::Description()const
{
  try
  {
    return "TabularEmbedding(features="+std::to_string(_config.num_features)+
           ",dim="+std::to_string(_config.dim)+")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"proj.weight");
    names.push_back(prefix+"proj.bias");
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceTabularEmbedding<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    std::mt19937 rng(seed);
    const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                 static_cast<float>(_config.num_features+_config.dim));
    std::uniform_real_distribution<float> dist(-limit,limit);

    std::vector<float> w_data(_config.num_features*_config.dim);
    for(size_t i=0;i<w_data.size();++i)
    {
      w_data[i]=dist(rng);
    }
    WProjMut().CopyFromHostFp32(w_data.data(),w_data.size());
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceTabularEmbedding<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceTabularEmbedding<float,__half>;
template class CAIF_DeviceTabularEmbedding<float,__nv_bfloat16>;
template class CAIF_DeviceTabularEmbedding<__half,float>;
template class CAIF_DeviceTabularEmbedding<__half,__half>;
template class CAIF_DeviceTabularEmbedding<__half,__nv_bfloat16>;
template class CAIF_DeviceTabularEmbedding<__nv_bfloat16,float>;
template class CAIF_DeviceTabularEmbedding<__nv_bfloat16,__half>;
template class CAIF_DeviceTabularEmbedding<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
