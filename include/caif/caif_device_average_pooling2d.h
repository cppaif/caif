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
// 2D Average pooling (templated on <ComputeT, StorageT>).
// Backward spreads grad_output evenly over the window.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_pooling2d.h"

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceAveragePooling2D:public CAIF_DevicePooling2D<ComputeT,StorageT>
{
  public:
    typedef CAIF_DevicePooling2D<ComputeT,StorageT> Base_t;
    typedef CAIF_DevicePooling2DConfig Config_t;

    CAIF_DeviceAveragePooling2D(const Config_t &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceAveragePooling2D()override=default;

    CAIF_DeviceAveragePooling2D(CAIF_DeviceAveragePooling2D &&other);
    CAIF_DeviceAveragePooling2D &operator=(CAIF_DeviceAveragePooling2D &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    std::string Description()const override;

  protected:
    using Base_t::Config;
    using Base_t::CachedInputShape;
    using Base_t::SetCachedInputShape;
    using Base_t::CachedInput;
    using Base_t::SetCachedInput;
    using Base_t::CachedOutput;
    using Base_t::SetCachedOutput;
    using Base_t::StorageDtype;
    using Base_t::Stream;

  private:
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceAveragePooling2D<float,float>;
extern template class CAIF_DeviceAveragePooling2D<float,__half>;
extern template class CAIF_DeviceAveragePooling2D<float,__nv_bfloat16>;
extern template class CAIF_DeviceAveragePooling2D<__half,float>;
extern template class CAIF_DeviceAveragePooling2D<__half,__half>;
extern template class CAIF_DeviceAveragePooling2D<__half,__nv_bfloat16>;
extern template class CAIF_DeviceAveragePooling2D<__nv_bfloat16,float>;
extern template class CAIF_DeviceAveragePooling2D<__nv_bfloat16,__half>;
extern template class CAIF_DeviceAveragePooling2D<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceAveragePooling2D<float,float>;
#endif

}//end instance namespace
