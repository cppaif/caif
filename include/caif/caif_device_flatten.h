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
// CAIF_DeviceFlatten<ComputeT, StorageT> — collapses every non-batch dim into
// one. Pure shape op; carries no kernel calls. Templated for type-level
// symmetry with the rest of the layer hierarchy so containers can hold a
// flatten cell at the same <C, S> as the surrounding chain.
//
// Input:  [N, d1, d2, ..., dK]
// Output: [N, d1*d2*...*dK]
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer.h"
#include "caif_device_layer_typed.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceFlatten:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    typedef CAIF_DeviceLayerTyped<ComputeT,StorageT> Base_t;

    using Base_t::Stream;

    explicit CAIF_DeviceFlatten(CAIF_CudaStream &stream);
    ~CAIF_DeviceFlatten()override=default;

    CAIF_DeviceFlatten(CAIF_DeviceFlatten &&other);
    CAIF_DeviceFlatten &operator=(CAIF_DeviceFlatten &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Flatten_e;
    }

    void ZeroGradients()override{}
    size_t ParameterTensorCount()const override{return 0;}
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override{return 0;}
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    std::vector<uint32_t> &CachedInputShape(){return _cached_input_shape;}
    const std::vector<uint32_t> &CachedInputShape()const{return _cached_input_shape;}
    void SetCachedInputShape(const std::vector<uint32_t> &shape){_cached_input_shape=shape;}
    void SetCachedInputShape(std::vector<uint32_t> &&shape){_cached_input_shape=std::move(shape);}

  protected:

  private:
    std::vector<uint32_t> _cached_input_shape;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceFlatten<float,float>;
extern template class CAIF_DeviceFlatten<float,__half>;
extern template class CAIF_DeviceFlatten<float,__nv_bfloat16>;
extern template class CAIF_DeviceFlatten<__half,float>;
extern template class CAIF_DeviceFlatten<__half,__half>;
extern template class CAIF_DeviceFlatten<__half,__nv_bfloat16>;
extern template class CAIF_DeviceFlatten<__nv_bfloat16,float>;
extern template class CAIF_DeviceFlatten<__nv_bfloat16,__half>;
extern template class CAIF_DeviceFlatten<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceFlatten<float,float>;
#endif

}//end instance namespace
