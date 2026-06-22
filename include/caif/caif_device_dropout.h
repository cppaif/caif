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
// Device-resident dropout layer (templated on <ComputeT, StorageT>).
//
// Mask tensor follows StorageT so it can multiply input directly via
// CAIF_Ops::Multiply (which is dtype-aware). Mask values are computed
// on host as fp32 then staged through fp32 → StorageT.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceDropout:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    CAIF_DeviceDropout(float rate,CAIF_CudaStream &stream);
    ~CAIF_DeviceDropout()override=default;

    CAIF_DeviceDropout(CAIF_DeviceDropout &&other);
    CAIF_DeviceDropout &operator=(CAIF_DeviceDropout &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Dropout_e;
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

    float Rate()const{return _rate;}
    void SetRate(float rate){_rate=rate;}

    CAIF_DeviceTensor &CachedMask(){return _cached_mask;}
    const CAIF_DeviceTensor &CachedMask()const{return _cached_mask;}
    void SetCachedMask(CAIF_DeviceTensor &&mask){_cached_mask=std::move(mask);}

    bool CachedMaskActive()const{return _cached_mask_active;}
    void SetCachedMaskActive(bool active){_cached_mask_active=active;}

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

  private:
    float _rate;
    CAIF_DeviceTensor _cached_mask;
    bool _cached_mask_active;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceDropout<float,float>;
extern template class CAIF_DeviceDropout<float,__half>;
extern template class CAIF_DeviceDropout<float,__nv_bfloat16>;
extern template class CAIF_DeviceDropout<__half,float>;
extern template class CAIF_DeviceDropout<__half,__half>;
extern template class CAIF_DeviceDropout<__half,__nv_bfloat16>;
extern template class CAIF_DeviceDropout<__nv_bfloat16,float>;
extern template class CAIF_DeviceDropout<__nv_bfloat16,__half>;
extern template class CAIF_DeviceDropout<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceDropout<float,float>;
#endif

}//end instance namespace
