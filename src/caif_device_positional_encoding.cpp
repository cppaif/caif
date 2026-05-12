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

#include "caif_device_positional_encoding.h"
#include "caif_device_positional_encoding_factory.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include <vector>
#include <cmath>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::CAIF_DevicePositionalEncoding(
                                              const Config_t &config,
                                              CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _pe_table(),
                                          _pe_table_grad(),
                                          _sinusoidal_table(),
                                          _cached_batch(0),
                                          _cached_seq_len(0)
{
  try
  {
    if(config.mode==PositionalEncodingMode_e::None)
    {
      return;
    }

    if(config.max_seq_len==0)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding: max_seq_len must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding: dim must be > 0");
    }

    const size_t table_size=static_cast<size_t>(config.max_seq_len)*config.dim;
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    if(config.mode==PositionalEncodingMode_e::Learned)
    {
      const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(config.max_seq_len+config.dim));
      std::vector<float> init_data(table_size);
      for(size_t i=0;i<table_size;++i)
      {
        const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
        init_data[i]=(t-std::floor(t))*2.0f*limit-limit;
      }
      SetPETable(CAIF_DeviceTensor::Uninitialized({config.max_seq_len,config.dim},stream,sdt));
      PETableMut().CopyFromHostFp32(init_data.data(),table_size);
      _pe_table_grad=CAIF_DeviceTensor::Zeros({config.max_seq_len,config.dim},stream,sdt);
    }
    else
    {
      // Sinusoidal: compute on host and upload.
      std::vector<float> table(table_size);
      for(uint32_t s=0;s<config.max_seq_len;++s)
      {
        for(uint32_t p=0;p<config.dim/2;++p)
        {
          const double freq=1.0/std::pow(g_caif_sinusoidal_base,
                                         2.0*static_cast<double>(p)/
                                           static_cast<double>(config.dim));
          const double angle=static_cast<double>(s)*freq;
          table[s*config.dim+2*p]=static_cast<float>(std::sin(angle));
          table[s*config.dim+2*p+1]=static_cast<float>(std::cos(angle));
        }
      }
      SetSinusoidalTable(CAIF_DeviceTensor::Uninitialized({config.max_seq_len,config.dim},
                                                            stream,sdt));
      SinusoidalTableMut().CopyFromHostFp32(table.data(),table_size);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::CAIF_DevicePositionalEncoding(
                                            CAIF_DevicePositionalEncoding &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _config(other._config),
                              _pe_table(std::move(other._pe_table)),
                              _pe_table_grad(std::move(other._pe_table_grad)),
                              _sinusoidal_table(std::move(other._sinusoidal_table)),
                              _cached_batch(other._cached_batch),
                              _cached_seq_len(other._cached_seq_len)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePositionalEncoding<ComputeT,StorageT> &
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::operator=(CAIF_DevicePositionalEncoding &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      _config=other._config;
      _pe_table=std::move(other._pe_table);
      _pe_table_grad=std::move(other._pe_table_grad);
      _sinusoidal_table=std::move(other._sinusoidal_table);
      _cached_batch=other._cached_batch;
      _cached_seq_len=other._cached_seq_len;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                              CAIF_RunContext &ctx)
{
  try
  {
    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::Forward: input must be at least 2D");
    }
    if(shape.back()!=_config.dim)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::Forward: last dim must match config.dim");
    }

    uint32_t seq_len;
    uint32_t batch;
    if(shape.size()==3)
    {
      batch=shape[0];
      seq_len=shape[1];
    }
    else if(shape.size()==2)
    {
      batch=1;
      seq_len=shape[0];
    }
    else
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::Forward: input must be 2D or 3D");
    }

    if(seq_len>_config.max_seq_len)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::Forward: seq_len exceeds max_seq_len");
    }

    if(_config.mode==PositionalEncodingMode_e::None)
    {
      return input.Clone();
    }

    AssertInputDtype(input);

    const CAIF_DeviceTensor *pe_tensor;
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      pe_tensor=&_pe_table;
    }
    else
    {
      pe_tensor=&_sinusoidal_table;
    }

    CAIF_DeviceTensor output=AllocateOutput(shape,ctx);
    CAIF_Ops::AddPositionalEncoding(input,*pe_tensor,output);

    if(ctx.Training()==true)
    {
      _cached_batch=batch;
      _cached_seq_len=seq_len;
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                                CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    if(_cached_batch==0)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::Backward: must call Forward with training=true first");
    }

    if(_config.mode==PositionalEncodingMode_e::None)
    {
      return grad_output.Clone();
    }

    CAIF_DeviceTensor grad_input=grad_output.Clone();

    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      _pe_table_grad.FillZero();
      CAIF_Ops::PositionalEncodingBackward(grad_output,_pe_table_grad);
    }

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DevicePositionalEncoding<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      _pe_table_grad.FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DevicePositionalEncoding<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(_config.mode==PositionalEncodingMode_e::Learned)
  {
    return 1;
  }
  return 0;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table;
    }
    THROW_CAIFE("CAIF_DevicePositionalEncoding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table;
    }
    THROW_CAIFE("CAIF_DevicePositionalEncoding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table_grad;
    }
    THROW_CAIFE("CAIF_DevicePositionalEncoding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table_grad;
    }
    THROW_CAIFE("CAIF_DevicePositionalEncoding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DevicePositionalEncoding<ComputeT,StorageT>::TotalParameterCount()const
{
  if(_config.mode==PositionalEncodingMode_e::Learned)
  {
    return static_cast<size_t>(_config.max_seq_len)*_config.dim;
  }
  return 0;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DevicePositionalEncoding<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string mode_str;
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      mode_str="learned";
    }
    else if(_config.mode==PositionalEncodingMode_e::Sinusoidal)
    {
      mode_str="sinusoidal";
    }
    else
    {
      mode_str="none";
    }
    return "PositionalEncoding(max_seq="+std::to_string(_config.max_seq_len)+
           ",dim="+std::to_string(_config.dim)+
           ",mode="+mode_str+")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      names.push_back(prefix+"weight");
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DevicePositionalEncoding<ComputeT,StorageT>::LoadPETable(CAIF_DeviceTensor &&table)
{
  try
  {
    if(_config.mode!=PositionalEncodingMode_e::Learned)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::LoadPETable: only valid in Learned mode");
    }
    const std::vector<uint32_t> &shape=table.Shape();
    if(shape.size()!=2||
       shape[0]!=_config.max_seq_len||
       shape[1]!=_config.dim)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::LoadPETable: shape mismatch");
    }
    _pe_table=std::move(table);
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DevicePositionalEncoding<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DevicePositionalEncoding<float,__half>;
template class CAIF_DevicePositionalEncoding<float,__nv_bfloat16>;
template class CAIF_DevicePositionalEncoding<__half,float>;
template class CAIF_DevicePositionalEncoding<__half,__half>;
template class CAIF_DevicePositionalEncoding<__half,__nv_bfloat16>;
template class CAIF_DevicePositionalEncoding<__nv_bfloat16,float>;
template class CAIF_DevicePositionalEncoding<__nv_bfloat16,__half>;
template class CAIF_DevicePositionalEncoding<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
