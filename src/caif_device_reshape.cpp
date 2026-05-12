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

#include "caif_device_reshape.h"
#include "caif_constants.h"
#include "caif_exception.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceReshape<ComputeT,StorageT>::CAIF_DeviceReshape(const std::vector<uint32_t> &target_shape,
                                                          CAIF_CudaStream &stream):
                                                          Base_t(stream),
                                                          _target_shape(target_shape),
                                                          _cached_input_shape()
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceReshape<ComputeT,StorageT>::CAIF_DeviceReshape(CAIF_DeviceReshape &&other):
                                          Base_t(std::move(other)),
                                          _target_shape(std::move(other._target_shape)),
                                          _cached_input_shape(std::move(other._cached_input_shape))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceReshape<ComputeT,StorageT> &
CAIF_DeviceReshape<ComputeT,StorageT>::operator=(CAIF_DeviceReshape &&other)
{
  try
  {
    if(this!=&other)
    {
      Base_t::operator=(std::move(other));
      _target_shape=std::move(other._target_shape);
      _cached_input_shape=std::move(other._cached_input_shape);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK();
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceReshape<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                   CAIF_RunContext &ctx)
{
  try
  {
    static_cast<void>(ctx);
    _cached_input_shape=input.Shape();
    size_t input_elements=1u;
    for(const uint32_t d:_cached_input_shape)
    {
      input_elements*=static_cast<size_t>(d);
    }
    size_t target_elements=1u;
    for(const uint32_t d:_target_shape)
    {
      target_elements*=static_cast<size_t>(d);
    }
    if(input_elements!=target_elements)
    {
      THROW_CAIFE("Reshape target element count does not match input");
    }
    CAIF_DeviceTensor output=input.Clone();
    output.Reshape(_target_shape);
    return output;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceReshape<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    static_cast<void>(ctx);
    if(_cached_input_shape.empty()==true)
    {
      THROW_CAIFE("Reshape backward called before forward");
    }
    CAIF_DeviceTensor grad_input=grad_output.Clone();
    grad_input.Reshape(_cached_input_shape);
    return grad_input;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceReshape<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Reshape has no parameters");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceReshape<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Reshape has no parameters");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceReshape<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Reshape has no gradients");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceReshape<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("Reshape has no gradients");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceReshape<ComputeT,StorageT>::Description()const
{
  return g_caif_description_reshape;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceReshape<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  static_cast<void>(prefix);
  return std::vector<std::string>();
}

template class CAIF_DeviceReshape<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceReshape<float,__half>;
template class CAIF_DeviceReshape<float,__nv_bfloat16>;
template class CAIF_DeviceReshape<__half,float>;
template class CAIF_DeviceReshape<__half,__half>;
template class CAIF_DeviceReshape<__half,__nv_bfloat16>;
template class CAIF_DeviceReshape<__nv_bfloat16,float>;
template class CAIF_DeviceReshape<__nv_bfloat16,__half>;
template class CAIF_DeviceReshape<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
