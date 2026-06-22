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

#include "caif_device_flatten.h"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceFlatten<ComputeT,StorageT>::CAIF_DeviceFlatten(CAIF_CudaStream &stream):
                                                          Base_t(stream),
                                                          _cached_input_shape()
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceFlatten<ComputeT,StorageT>::CAIF_DeviceFlatten(CAIF_DeviceFlatten &&other):
                                          Base_t(std::move(other)),
                                          _cached_input_shape(std::move(other.CachedInputShape()))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceFlatten<ComputeT,StorageT> &
CAIF_DeviceFlatten<ComputeT,StorageT>::operator=(CAIF_DeviceFlatten &&other)
{
  try
  {
    if(this!=&other)
    {
      Base_t::operator=(std::move(other));
      SetCachedInputShape(std::move(other.CachedInputShape()));
    }
    return *this;
  }
  CAIF_CATCH_BLOCK();
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceFlatten<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                   CAIF_RunContext &ctx)
{
  try
  {
    static_cast<void>(ctx);
    SetCachedInputShape(input.Shape());
    if(CachedInputShape().size()<2u)
    {
      THROW_CAIFE("Flatten input must have at least 2 dims");
    }
    const uint32_t batch=CachedInputShape()[0];
    uint32_t feature_count=1u;
    for(size_t i=1;i<CachedInputShape().size();++i)
    {
      feature_count*=CachedInputShape()[i];
    }
    CAIF_DeviceTensor output=input.Clone();
    const std::vector<uint32_t> flat_shape={batch,feature_count};
    output.Reshape(flat_shape);
    return output;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceFlatten<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    static_cast<void>(ctx);
    if(CachedInputShape().empty()==true)
    {
      THROW_CAIFE("Flatten backward called before forward");
    }
    CAIF_DeviceTensor grad_input=grad_output.Clone();
    grad_input.Reshape(CachedInputShape());
    return grad_input;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceFlatten<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Flatten has no parameters");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceFlatten<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Flatten has no parameters");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceFlatten<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Flatten has no gradients");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceFlatten<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Flatten has no gradients");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceFlatten<ComputeT,StorageT>::Description()const
{
  return g_serial_tag_flatten;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceFlatten<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  static_cast<void>(prefix);
  return std::vector<std::string>();
}

template class CAIF_DeviceFlatten<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceFlatten<float,__half>;
template class CAIF_DeviceFlatten<float,__nv_bfloat16>;
template class CAIF_DeviceFlatten<__half,float>;
template class CAIF_DeviceFlatten<__half,__half>;
template class CAIF_DeviceFlatten<__half,__nv_bfloat16>;
template class CAIF_DeviceFlatten<__nv_bfloat16,float>;
template class CAIF_DeviceFlatten<__nv_bfloat16,__half>;
template class CAIF_DeviceFlatten<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
