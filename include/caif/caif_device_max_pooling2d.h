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
// 2D Max pooling (templated on <ComputeT, StorageT>).
// Backward gates grad_output through the cached argmax index.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_pooling2d.h"

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceMaxPooling2D:public CAIF_DevicePooling2D<ComputeT,StorageT>
{
  public:
    typedef CAIF_DevicePooling2D<ComputeT,StorageT> Base_t;
    typedef typename Base_t::Config_t Config_t;

    CAIF_DeviceMaxPooling2D(const Config_t &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceMaxPooling2D()override=default;

    CAIF_DeviceMaxPooling2D(CAIF_DeviceMaxPooling2D &&other);
    CAIF_DeviceMaxPooling2D &operator=(CAIF_DeviceMaxPooling2D &&other);

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
    const std::vector<int32_t> &CachedMaxIndices()const{return _cached_max_indices;}
    std::vector<int32_t> &CachedMaxIndices(){return _cached_max_indices;}
    void SetCachedMaxIndices(std::vector<int32_t> &&v){_cached_max_indices=std::move(v);}

    std::vector<int32_t> _cached_max_indices;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceMaxPooling2D<float,float>;
extern template class CAIF_DeviceMaxPooling2D<float,__half>;
extern template class CAIF_DeviceMaxPooling2D<float,__nv_bfloat16>;
extern template class CAIF_DeviceMaxPooling2D<__half,float>;
extern template class CAIF_DeviceMaxPooling2D<__half,__half>;
extern template class CAIF_DeviceMaxPooling2D<__half,__nv_bfloat16>;
extern template class CAIF_DeviceMaxPooling2D<__nv_bfloat16,float>;
extern template class CAIF_DeviceMaxPooling2D<__nv_bfloat16,__half>;
extern template class CAIF_DeviceMaxPooling2D<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceMaxPooling2D<float,float>;
#endif

}//end instance namespace
