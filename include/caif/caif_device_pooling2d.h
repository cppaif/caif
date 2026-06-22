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
// Abstract base for 2D pooling layers (templated on <ComputeT, StorageT>).
//
// Stage 5b: host-location, fp32-only path. Non-fp32 cells throw at
// ForwardImpl until a cuDNN device backend lands.
//
// Input layout:  [N, H, W, C]   (channels-last)
// Output layout: [N, H_out, W_out, C]
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_pooling2d_config.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DevicePooling2D:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:

    ~CAIF_DevicePooling2D()override=default;

    const CAIF_DevicePooling2DConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DevicePooling2DConfig &c){_config=c;}

    const std::vector<uint32_t> &CachedInputShape()const{return _cached_input_shape;}
    std::vector<uint32_t> &CachedInputShape(){return _cached_input_shape;}
    void SetCachedInputShape(const std::vector<uint32_t> &shape){_cached_input_shape=shape;}
    void SetCachedInputShape(std::vector<uint32_t> &&shape){_cached_input_shape=std::move(shape);}

    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    CAIF_DeviceTensor &CachedInput(){return _cached_input;}
    void SetCachedInput(CAIF_DeviceTensor &&t){_cached_input=std::move(t);}

    const CAIF_DeviceTensor &CachedOutput()const{return _cached_output;}
    CAIF_DeviceTensor &CachedOutput(){return _cached_output;}
    void SetCachedOutput(CAIF_DeviceTensor &&t){_cached_output=std::move(t);}

    void ZeroGradients()override{}
    size_t ParameterTensorCount()const override{return 0;}
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override{return 0;}
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Pooling2D_e;
    }

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;
    CAIF_DevicePooling2D(const CAIF_DevicePooling2DConfig &config,CAIF_CudaStream &stream);
    CAIF_DevicePooling2D(CAIF_DevicePooling2D &&other);
    CAIF_DevicePooling2D &operator=(CAIF_DevicePooling2D &&other);

    CAIF_DevicePooling2DConfig _config;
    std::vector<uint32_t> _cached_input_shape;
    // Cached input/output tensors for backward. cuDNN's pooling backward
    // signature needs both forward operands; the host path leaves these
    // empty and uses _cached_input_shape only.
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_output;

  private:
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DevicePooling2D<float,float>;
extern template class CAIF_DevicePooling2D<float,__half>;
extern template class CAIF_DevicePooling2D<float,__nv_bfloat16>;
extern template class CAIF_DevicePooling2D<__half,float>;
extern template class CAIF_DevicePooling2D<__half,__half>;
extern template class CAIF_DevicePooling2D<__half,__nv_bfloat16>;
extern template class CAIF_DevicePooling2D<__nv_bfloat16,float>;
extern template class CAIF_DevicePooling2D<__nv_bfloat16,__half>;
extern template class CAIF_DevicePooling2D<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DevicePooling2D<float,float>;
#endif

}//end instance namespace
