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
#include "caif_role_registry.h"
#include "caif_positional_encoding_mode.h"
#include "caif_serialization_constants.h"
#include <vector>
#include <cmath>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DevicePositionalEncoding<ComputeT,StorageT>::CAIF_DevicePositionalEncoding(
                                              const CAIF_DevicePositionalEncodingConfig &config,
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
    if(config.Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::None)
    {
      return;
    }

    if(config.MaxSeqLen()==0)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding: max_seq_len must be > 0");
    }
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding: dim must be > 0");
    }

    const size_t table_size=static_cast<size_t>(config.MaxSeqLen())*config.Dim();
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    if(config.Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
    {
      const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(config.MaxSeqLen()+config.Dim()));
      std::vector<float> init_data(table_size);
      for(size_t i=0;i<table_size;++i)
      {
        const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
        init_data[i]=(t-std::floor(t))*2.0f*limit-limit;
      }
      SetPETable(CAIF_DeviceTensor::Uninitialized({config.MaxSeqLen(),config.Dim()},stream,sdt));
      PETableMut().CopyFromHostFp32(init_data.data(),table_size);
      SetPETableGrad(CAIF_DeviceTensor::Zeros({config.MaxSeqLen(),config.Dim()},stream,sdt));
    }
    else
    {
      // Sinusoidal: compute on host and upload.
      std::vector<float> table(table_size);
      for(uint32_t s=0;s<config.MaxSeqLen();++s)
      {
        for(uint32_t p=0;p<config.Dim()/2;++p)
        {
          const double freq=1.0/std::pow(g_caif_sinusoidal_base,
                                         2.0*static_cast<double>(p)/
                                           static_cast<double>(config.Dim()));
          const double angle=static_cast<double>(s)*freq;
          table[s*config.Dim()+2*p]=static_cast<float>(std::sin(angle));
          table[s*config.Dim()+2*p+1]=static_cast<float>(std::cos(angle));
        }
      }
      SetSinusoidalTable(CAIF_DeviceTensor::Uninitialized({config.MaxSeqLen(),config.Dim()},
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
      SetConfig(other.Config());
      SetPETable(std::move(other.PETableMut()));
      SetPETableGrad(std::move(other.PETableGrad()));
      SetSinusoidalTable(std::move(other.SinusoidalTableMut()));
      SetCachedBatch(other.CachedBatch());
      SetCachedSeqLen(other.CachedSeqLen());
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
    if(shape.back()!=Config().Dim())
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

    if(seq_len>Config().MaxSeqLen())
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::Forward: seq_len exceeds max_seq_len");
    }

    if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::None)
    {
      return input.Clone();
    }

    AssertInputDtype(input);

    const CAIF_DeviceTensor *pe_tensor;
    if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
    {
      pe_tensor=&PETableMut();
    }
    else
    {
      pe_tensor=&SinusoidalTableMut();
    }

    CAIF_DeviceTensor output=AllocateOutput(shape,ctx);
    CAIF_Ops::AddPositionalEncoding(input,*pe_tensor,output);

    if(ctx.Training()==true)
    {
      SetCachedBatch(batch);
      SetCachedSeqLen(seq_len);
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
    if(CachedBatch()==0)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::Backward: must call Forward with training=true first");
    }

    if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::None)
    {
      return grad_output.Clone();
    }

    CAIF_DeviceTensor grad_input=grad_output.Clone();

    if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
    {
      PETableGrad().FillZero();
      CAIF_Ops::PositionalEncodingBackward(grad_output,PETableGrad());
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
    if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
    {
      PETableGrad().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DevicePositionalEncoding<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
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
    if(index==0&&Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
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
    if(index==0&&Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
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
    if(index==0&&Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
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
    if(index==0&&Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
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
  if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
  {
    return static_cast<size_t>(Config().MaxSeqLen())*Config().Dim();
  }
  return 0;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DevicePositionalEncoding<ComputeT,StorageT>::Description()const
{
  try
  {
    return std::string(g_serial_tag_positional_encoding)+
           g_serial_open_paren+
           g_serial_kv_max_seq+
           std::to_string(Config().MaxSeqLen())+
           g_serial_comma+
           g_serial_kv_dim+
           std::to_string(Config().Dim())+
           g_serial_comma+
           g_serial_kv_mode+
           CAIF_PositionalEncodingMode::Name(Mode())+
           g_serial_close_paren;
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
    if(Mode()==CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
    {
      names.push_back(prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PositionEmbeddingTable_e));
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
    if(Mode()!=CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned)
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::LoadPETable: only valid in Learned mode");
    }
    const std::vector<uint32_t> &shape=table.Shape();
    if(shape.size()!=2||
       shape[0]!=Config().MaxSeqLen()||
       shape[1]!=Config().Dim())
    {
      THROW_CAIFE("CAIF_DevicePositionalEncoding::LoadPETable: shape mismatch");
    }
    SetPETable(std::move(table));
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
