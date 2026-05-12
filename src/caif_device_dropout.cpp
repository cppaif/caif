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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// Device-resident dropout layer implementation (templated).
//------------------------------------------------------------------------------

#include "caif_device_dropout.h"
#include "caif_device_dropout_factory.h"
#include "caif_ops.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace instance
{


static uint64_t SplitMix64(uint64_t x)
{
  uint64_t z=(x+0x9E3779B97F4A7C15ULL);
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
  z=(z^(z>>27))*0x94D049BB133111EBULL;
  return z^(z>>31);
}

static float UniformFromBits(const uint64_t bits)
{
  const uint64_t mantissa=(bits>>11)&((1ULL<<53)-1ULL);
  const double unit_double=static_cast<double>(mantissa)/static_cast<double>(1ULL<<53);
  return static_cast<float>(unit_double);
}

constexpr float DROPOUT_FULL_KEEP_RATE=0.0f;
constexpr uint64_t DROPOUT_COUNTER_MIX=0xD1342543DE82EF95ULL;


template<typename ComputeT,typename StorageT>
CAIF_DeviceDropout<ComputeT,StorageT>::CAIF_DeviceDropout(float rate,
                                                          CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _rate(rate),
                                          _cached_mask(),
                                          _cached_mask_active(false)
{
  try
  {
    if(_rate<0.0f || _rate>=1.0f)
    {
      THROW_CAIFE("CAIF_DeviceDropout: rate must be in [0,1)");
    }
  }
  CAIF_CATCH_BLOCK();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceDropout<ComputeT,StorageT>::CAIF_DeviceDropout(CAIF_DeviceDropout &&other):
                                  CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                                  _rate(other._rate),
                                  _cached_mask(std::move(other._cached_mask)),
                                  _cached_mask_active(other._cached_mask_active)
{
  other._cached_mask_active=false;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceDropout<ComputeT,StorageT> &
CAIF_DeviceDropout<ComputeT,StorageT>::operator=(CAIF_DeviceDropout &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      _rate=other._rate;
      _cached_mask=std::move(other._cached_mask);
      _cached_mask_active=other._cached_mask_active;
      other._cached_mask_active=false;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK();
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceDropout<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);

    if(ctx.Training()==false || _rate==DROPOUT_FULL_KEEP_RATE)
    {
      _cached_mask_active=false;
      return input.Clone();
    }

    const float keep_probability=1.0f-_rate;
    const float scale=1.0f/keep_probability;
    const size_t element_count=input.TotalElements();
    std::vector<float> mask_host(element_count,0.0f);

    const uint64_t seed=ctx.RandomSeed();
    const uint64_t call_counter=ctx.NextRandomCounter();
    const uint64_t base=SplitMix64(seed^(DROPOUT_COUNTER_MIX*(call_counter+1ULL)));
    for(size_t i=0;i<element_count;++i)
    {
      const uint64_t bits=SplitMix64(base^static_cast<uint64_t>(i));
      const float u=UniformFromBits(bits);
      if(u<keep_probability)
      {
        mask_host[i]=scale;
      }
      else
      {
        mask_host[i]=0.0f;
      }
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    CAIF_DeviceTensor mask;
    CAIF_DeviceTensor output;
    if(input.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      if(sdt==fp32)
      {
        mask=CAIF_DeviceTensor::ZerosHost(input.Shape());
        std::memcpy(mask.DeviceDataRaw(),
                    mask_host.data(),
                    element_count*sizeof(float));
      }
      else
      {
        CAIF_DeviceTensor staging=CAIF_DeviceTensor::ZerosHost(input.Shape());
        std::memcpy(staging.DeviceDataRaw(),
                    mask_host.data(),
                    element_count*sizeof(float));
        mask=staging.To(sdt);
      }
      output=CAIF_DeviceTensor::ZerosHost(input.Shape(),sdt);
    }
    else
    {
      CAIF_CudaStream &stream=ctx.Stream();
      mask=CAIF_DeviceTensor::Uninitialized(input.Shape(),stream,sdt);
      mask.CopyFromHostFp32(mask_host.data(),element_count);
      output=CAIF_DeviceTensor::Zeros(input.Shape(),stream,sdt);
    }
    CAIF_Ops::Multiply(input,mask,output);

    _cached_mask=std::move(mask);
    _cached_mask_active=true;
    return output;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceDropout<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                     CAIF_RunContext &ctx)
{
  try
  {
    if(_cached_mask_active==false)
    {
      return grad_output.Clone();
    }
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    CAIF_DeviceTensor grad_input;
    if(grad_output.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      grad_input=CAIF_DeviceTensor::ZerosHost(grad_output.Shape(),sdt);
    }
    else
    {
      CAIF_CudaStream &stream=ctx.Stream();
      grad_input=CAIF_DeviceTensor::Zeros(grad_output.Shape(),stream,sdt);
    }
    CAIF_Ops::Multiply(grad_output,_cached_mask,grad_input);
    return grad_input;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceDropout<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DeviceDropout: no parameters");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceDropout<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DeviceDropout: no parameters");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceDropout<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DeviceDropout: no gradients");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceDropout<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DeviceDropout: no gradients");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceDropout<ComputeT,StorageT>::Description()const
{
  return "CAIF_DeviceDropout";
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceDropout<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  (void)prefix;
  return std::vector<std::string>();
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceDropout<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceDropout<float,__half>;
template class CAIF_DeviceDropout<float,__nv_bfloat16>;
template class CAIF_DeviceDropout<__half,float>;
template class CAIF_DeviceDropout<__half,__half>;
template class CAIF_DeviceDropout<__half,__nv_bfloat16>;
template class CAIF_DeviceDropout<__nv_bfloat16,float>;
template class CAIF_DeviceDropout<__nv_bfloat16,__half>;
template class CAIF_DeviceDropout<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
