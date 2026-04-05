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

#include "caif_device_frozen_linear.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include "caif_cuda_kernels.h"

using namespace instance;

CAIF_DeviceFrozenLinear::CAIF_DeviceFrozenLinear(uint32_t input_dim,
                                               uint32_t output_dim,
                                               CAIF_DataType::CAIF_DataType_e storage_dtype,
                                               CAIF_CudaStream &stream,
                                               uint32_t group_size,
                                               bool cache_fp32):CAIF_DeviceLayer(stream),
                                                                 _input_dim(input_dim),
                                                                 _output_dim(output_dim),
                                                                 _storage_dtype(storage_dtype),
                                                                 _group_size(group_size),
                                                                 _weight(),
                                                                 _scales(),
                                                                 _cache_fp32(cache_fp32),
                                                                 _cached_fp32_weight(),
                                                                 _cached_input()
{
  try
  {
    if(input_dim==0||output_dim==0)
    {
      THROW_CAIFE("FrozenLinear: input_dim and output_dim must be > 0");
    }
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceFrozenLinear::CAIF_DeviceFrozenLinear(
  CAIF_DeviceFrozenLinear &&other):CAIF_DeviceLayer(std::move(other)),
                                  _input_dim(other._input_dim),
                                  _output_dim(other._output_dim),
                                  _storage_dtype(other._storage_dtype),
                                  _group_size(other._group_size),
                                  _weight(std::move(other._weight)),
                                  _scales(std::move(other._scales)),
                                  _cache_fp32(other._cache_fp32),
                                  _cached_fp32_weight(std::move(other._cached_fp32_weight)),
                                  _cached_input(std::move(other._cached_input))
{
}

CAIF_DeviceFrozenLinear &CAIF_DeviceFrozenLinear::operator=(CAIF_DeviceFrozenLinear &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _input_dim=other._input_dim;
      _output_dim=other._output_dim;
      _storage_dtype=other._storage_dtype;
      _group_size=other._group_size;
      _weight=std::move(other._weight);
      _scales=std::move(other._scales);
      _cache_fp32=other._cache_fp32;
      _cached_fp32_weight=std::move(other._cached_fp32_weight);
      _cached_input=std::move(other._cached_input);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceFrozenLinear::LoadFromTensor(CAIF_DeviceTensor &&weight)
{
  try
  {
    _weight=std::move(weight);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceFrozenLinear::LoadScalesFromHost(const void *data,size_t num_bytes)
{
  try
  {
    const size_t total_elements=static_cast<size_t>(_input_dim)*_output_dim;
    const uint32_t num_groups=static_cast<uint32_t>((total_elements+_group_size-1)/_group_size);
    _scales=CAIF_DeviceTensor::Zeros({num_groups},*_stream,CAIF_DataType::CAIF_DataType_e::Float16);
    _scales.CopyFromHostRaw(data,num_bytes);
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_DeviceFrozenLinear::NeedsScales()const
{
  return _storage_dtype==CAIF_DataType::CAIF_DataType_e::Int4;
}

void CAIF_DeviceFrozenLinear::ClearFP32Cache()
{
  _cached_fp32_weight=CAIF_DeviceTensor();
}

CAIF_DeviceTensor CAIF_DeviceFrozenLinear::ConvertToFP32()const
{
  try
  {
    if(_weight.IsEmpty()==true)
    {
      THROW_CAIFE("FrozenLinear: weight not loaded");
    }

    // All paths return weight in PyTorch layout [output_dim, input_dim].
    // Forward uses MatMulTransposeB to compute input @ W^T.
    if(_storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      CAIF_DeviceTensor result=_weight.Clone();
      result.Reshape({_output_dim,_input_dim});
      return result;
    }
    if(_storage_dtype==CAIF_DataType::CAIF_DataType_e::Float16||
       _storage_dtype==CAIF_DataType::CAIF_DataType_e::BFloat16||
       _storage_dtype==CAIF_DataType::CAIF_DataType_e::Int8)
    {
      CAIF_DeviceTensor result=_weight.To(CAIF_DataType::CAIF_DataType_e::Float32);
      result.Reshape({_output_dim,_input_dim});
      return result;
    }
    if(_storage_dtype==CAIF_DataType::CAIF_DataType_e::Int4)
    {
#ifdef USE_CAIF_CUDA
      const size_t total_elements=static_cast<size_t>(_input_dim)*_output_dim;
      CAIF_DeviceTensor result=CAIF_DeviceTensor::Uninitialized({_output_dim,_input_dim},*_stream);
      launch_dequantize_int4(_weight.DevicePtr(),
                              _scales.DevicePtr(),
                              static_cast<float*>(result.DevicePtr()),
                              static_cast<int>(total_elements),
                              static_cast<int>(_group_size),
                              _stream->Handle());
      return result;
#else
      THROW_CAIFE("INT4 dequantization requires CUDA");
#endif
    }

    THROW_CAIFE("FrozenLinear: unsupported storage dtype for conversion");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceFrozenLinear::Forward(const CAIF_DeviceTensor &input,
                                                  bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("FrozenLinear: layer has been moved from");
    }
    if(_weight.IsEmpty()==true)
    {
      THROW_CAIFE("FrozenLinear: weight not loaded");
    }

    const auto &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("FrozenLinear::Forward: input must be at least 2D");
    }
    if(shape.back()!=_input_dim)
    {
      THROW_CAIFE("FrozenLinear::Forward: last dim must match input_dim");
    }

    // Get FP32 weight -- cache or compute on-the-fly depending on _cache_fp32
    CAIF_DeviceTensor local_fp32;
    if(_cache_fp32==true)
    {
      if(_cached_fp32_weight.IsEmpty()==true)
      {
        _cached_fp32_weight=ConvertToFP32();
      }
    }
    else
    {
      local_fp32=ConvertToFP32();
    }
    const CAIF_DeviceTensor &fp32_weight=(_cache_fp32==true)?_cached_fp32_weight:local_fp32;

    // Reshape to 2D for matmul: [N, input_dim]
    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }
    CAIF_DeviceTensor input_2d=input.Clone();
    input_2d.Reshape({n,_input_dim});

    // Cache input for backward
    if(training==true)
    {
      _cached_input=input_2d.Clone();
    }

    // output = input @ weight^T: [N, input_dim] @ [output_dim, input_dim]^T = [N, output_dim]
    CAIF_DeviceTensor output_2d=CAIF_DeviceTensor::Uninitialized({n,_output_dim},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(input_2d,fp32_weight,output_2d);

    // Restore original batch dims
    if(shape.size()>2)
    {
      std::vector<uint32_t> out_shape(shape.begin(),shape.end()-1);
      out_shape.push_back(_output_dim);
      output_2d.Reshape(out_shape);
    }
    return output_2d;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceFrozenLinear::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("FrozenLinear: layer has been moved from");
    }
    if(_cached_input.IsEmpty()==true)
    {
      THROW_CAIFE("FrozenLinear::Backward: must call Forward with training=true first");
    }
    // Get FP32 weight for backward
    CAIF_DeviceTensor local_fp32;
    if(_cache_fp32==true)
    {
      if(_cached_fp32_weight.IsEmpty()==true)
      {
        THROW_CAIFE("FrozenLinear::Backward: no cached FP32 weight");
      }
    }
    else
    {
      local_fp32=ConvertToFP32();
    }
    const CAIF_DeviceTensor &fp32_weight=(_cache_fp32==true)?_cached_fp32_weight:local_fp32;

    const auto &shape=grad_output.Shape();

    // Reshape to 2D
    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }
    CAIF_DeviceTensor grad_2d=grad_output.Clone();
    grad_2d.Reshape({n,_output_dim});

    // grad_input = grad_output @ weight: [N, output_dim] @ [output_dim, input_dim] = [N, input_dim]
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({n,_input_dim},*_stream);
    CAIF_DeviceOps::MatMul(grad_2d,fp32_weight,grad_input);

    // Restore original batch dims
    if(shape.size()>2)
    {
      std::vector<uint32_t> in_shape(shape.begin(),shape.end()-1);
      in_shape.push_back(_input_dim);
      grad_input.Reshape(in_shape);
    }
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceFrozenLinear::ZeroGradients()
{
  // No trainable parameters, nothing to zero
}

size_t CAIF_DeviceFrozenLinear::ParameterTensorCount()const
{
  return 0;
}

CAIF_DeviceTensor &CAIF_DeviceFrozenLinear::ParameterTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceFrozenLinear::ParameterTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceFrozenLinear::GradientTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceFrozenLinear::GradientTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceFrozenLinear::TotalParameterCount()const
{
  return 0;
}

std::string CAIF_DeviceFrozenLinear::Description()const
{
  try
  {
    CAIF_DataType dt(_storage_dtype);
    return "FrozenLinear("+std::to_string(_input_dim)+
           ","+std::to_string(_output_dim)+
           ","+dt.Name()+")";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceFrozenLinear::ParameterNames(const std::string &prefix)const
{
  try
  {
    static_cast<void>(prefix);
    return {};
  }
  CAIF_CATCH_BLOCK()
}
