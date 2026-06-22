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
// CAIF_DevicePooling2D abstract base implementation (templated).
//------------------------------------------------------------------------------

#include "caif_device_pooling2d.h"
#include "caif_exception.h"
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DevicePooling2D<ComputeT,StorageT>::CAIF_DevicePooling2D(const CAIF_DevicePooling2DConfig &config,
                                                              CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _cached_input_shape()
{
  try
  {
    if(config.PoolHeight()==0 || config.PoolWidth()==0)
    {
      THROW_CAIFE("CAIF_DevicePooling2D: pool dims must be > 0");
    }
    if(config.StrideHeight()==0 || config.StrideWidth()==0)
    {
      THROW_CAIFE("CAIF_DevicePooling2D: stride dims must be > 0");
    }
  }
  CAIF_CATCH_BLOCK();
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePooling2D<ComputeT,StorageT>::CAIF_DevicePooling2D(CAIF_DevicePooling2D &&other):
                                  CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                                  _config(other._config),
                                  _cached_input_shape(std::move(other._cached_input_shape))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePooling2D<ComputeT,StorageT> &
CAIF_DevicePooling2D<ComputeT,StorageT>::operator=(CAIF_DevicePooling2D &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      SetConfig(other.Config());
      SetCachedInputShape(std::move(other.CachedInputShape()));
    }
    return *this;
  }
  CAIF_CATCH_BLOCK();
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DevicePooling2D<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DevicePooling2D: no parameters");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DevicePooling2D<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DevicePooling2D: no parameters");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DevicePooling2D<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DevicePooling2D: no gradients");
  }
  CAIF_CATCH_BLOCK();
  static CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DevicePooling2D<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    (void)index;
    THROW_CAIFE("CAIF_DevicePooling2D: no gradients");
  }
  CAIF_CATCH_BLOCK();
  static const CAIF_DeviceTensor null_tensor;
  return null_tensor;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DevicePooling2D<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  (void)prefix;
  return std::vector<std::string>();
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DevicePooling2D<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DevicePooling2D<float,__half>;
template class CAIF_DevicePooling2D<float,__nv_bfloat16>;
template class CAIF_DevicePooling2D<__half,float>;
template class CAIF_DevicePooling2D<__half,__half>;
template class CAIF_DevicePooling2D<__half,__nv_bfloat16>;
template class CAIF_DevicePooling2D<__nv_bfloat16,float>;
template class CAIF_DevicePooling2D<__nv_bfloat16,__half>;
template class CAIF_DevicePooling2D<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
