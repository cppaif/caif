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
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>
#include <cmath>

namespace instance
{

CAIF_DevicePositionalEncoding::CAIF_DevicePositionalEncoding(const CAIF_DevicePositionalEncoding::Config_t &config,
                                                           CAIF_CudaStream &stream):
                                                           CAIF_DeviceLayer(stream),
                                                           _config(config),
                                                           _pe_table(),
                                                           _pe_table_grad(),
                                                           _sinusoidal_table(),
                                                           _cached_batch(0),
                                                           _cached_seq_len(0)
{
  try
  {
    if(config.max_seq_len==0)
    {
      THROW_CAIFE("DevicePositionalEncoding: max_seq_len must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("DevicePositionalEncoding: dim must be > 0");
    }

    const size_t table_size=static_cast<size_t>(config.max_seq_len)*config.dim;

    if(config.mode==PositionalEncodingMode_e::Learned)
    {
      // Xavier uniform init
      const float limit=std::sqrt(6.0f/static_cast<float>(config.max_seq_len+config.dim));
      std::vector<float> init_data(table_size);
      for(size_t i=0;i<table_size;++i)
      {
        const float t=static_cast<float>(i)*0.6180339887f;
        init_data[i]=(t-std::floor(t))*2.0f*limit-limit;
      }
      _pe_table=CAIF_DeviceTensor::Uninitialized({config.max_seq_len,config.dim},stream);
      _pe_table.CopyFromHost(init_data.data(),table_size);

      _pe_table_grad=CAIF_DeviceTensor::Zeros({config.max_seq_len,config.dim},stream);
    }
    else
    {
      // Sinusoidal: compute on host and upload
      std::vector<float> table(table_size);
      for(uint32_t s=0;s<config.max_seq_len;++s)
      {
        for(uint32_t p=0;p<config.dim/2;++p)
        {
          const double freq=1.0/std::pow(g_caif_sinusoidal_base,
                                       2.0*static_cast<double>(p)/static_cast<double>(config.dim));
          const double angle=static_cast<double>(s)*freq;
          table[s*config.dim+2*p]=static_cast<float>(std::sin(angle));
          table[s*config.dim+2*p+1]=static_cast<float>(std::cos(angle));
        }
        // If dim is odd, last element stays 0
      }
      _sinusoidal_table=CAIF_DeviceTensor::Uninitialized({config.max_seq_len,config.dim},stream);
      _sinusoidal_table.CopyFromHost(table.data(),table_size);
    }
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DevicePositionalEncoding::CAIF_DevicePositionalEncoding(CAIF_DevicePositionalEncoding &&other):
                                                           CAIF_DeviceLayer(std::move(other)),
                                                           _config(other._config),
                                                           _pe_table(std::move(other._pe_table)),
                                                           _pe_table_grad(std::move(other._pe_table_grad)),
                                                           _sinusoidal_table(std::move(other._sinusoidal_table)),
                                                           _cached_batch(other._cached_batch),
                                                           _cached_seq_len(other._cached_seq_len)
{
}

CAIF_DevicePositionalEncoding &CAIF_DevicePositionalEncoding::operator=(CAIF_DevicePositionalEncoding &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
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

CAIF_DeviceTensor CAIF_DevicePositionalEncoding::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DevicePositionalEncoding: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("DevicePositionalEncoding::Forward: input must be at least 2D");
    }
    if(shape.back()!=_config.dim)
    {
      THROW_CAIFE("DevicePositionalEncoding::Forward: last dim must match config.dim");
    }

    // Extract batch and seq_len from shape
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
      THROW_CAIFE("DevicePositionalEncoding::Forward: input must be 2D or 3D");
    }

    if(seq_len>_config.max_seq_len)
    {
      THROW_CAIFE("DevicePositionalEncoding::Forward: seq_len exceeds max_seq_len");
    }

    // Get PE table pointer
    const float *pe_ptr;
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      pe_ptr=_pe_table.DevicePtr();
    }
    else
    {
      pe_ptr=_sinusoidal_table.DevicePtr();
    }

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(shape,*_stream);

    launch_add_positional_encoding(input.DevicePtr(),
                                   pe_ptr,
                                   output.DevicePtr(),
                                   static_cast<int>(batch),
                                   static_cast<int>(seq_len),
                                   static_cast<int>(_config.dim),
                                   _stream->Handle());

    if(training==true)
    {
      _cached_batch=batch;
      _cached_seq_len=seq_len;
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DevicePositionalEncoding::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DevicePositionalEncoding: layer has been moved from");
    }
    if(_cached_batch==0)
    {
      THROW_CAIFE(
        "DevicePositionalEncoding::Backward: "
        "must call Forward with training=true first");
    }

    // grad_input = grad_output (identity w.r.t. input)
    CAIF_DeviceTensor grad_input=grad_output.Clone();

    // If Learned: accumulate grad_pe_table
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      _pe_table_grad.Fill(0.0f);

      launch_pe_table_backward(grad_output.DevicePtr(),
                               _pe_table_grad.DevicePtr(),
                               static_cast<int>(_cached_batch),
                               static_cast<int>(_cached_seq_len),
                               static_cast<int>(_config.dim),
                               _stream->Handle());
    }

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DevicePositionalEncoding::ZeroGradients()
{
  try
  {
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      _pe_table_grad.Fill(0.0f);
    }
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DevicePositionalEncoding::ParameterTensorCount()const
{
  try
  {
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DevicePositionalEncoding::ParameterTensor(size_t index)
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table;
    }
    THROW_CAIFE("DevicePositionalEncoding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DevicePositionalEncoding::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table;
    }
    THROW_CAIFE("DevicePositionalEncoding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DevicePositionalEncoding::GradientTensor(size_t index)
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table_grad;
    }
    THROW_CAIFE("DevicePositionalEncoding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DevicePositionalEncoding::GradientTensor(size_t index)const
{
  try
  {
    if(index==0&&_config.mode==PositionalEncodingMode_e::Learned)
    {
      return _pe_table_grad;
    }
    THROW_CAIFE("DevicePositionalEncoding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DevicePositionalEncoding::TotalParameterCount()const
{
  try
  {
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      return static_cast<size_t>(_config.max_seq_len)*_config.dim;
    }
    else
    {
      return 0;
    }
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DevicePositionalEncoding::Description()const
{
  try
  {
    std::string mode_str;
    if(_config.mode==PositionalEncodingMode_e::Learned)
    {
      mode_str="learned";
    }
    else
    {
      mode_str="sinusoidal";
    }
    return "PositionalEncoding(max_seq="+std::to_string(_config.max_seq_len)+
           ",dim="+std::to_string(_config.dim)+
           ",mode="+mode_str+")";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DevicePositionalEncoding::ParameterNames(const std::string &prefix)const
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

}//end instance namespace
