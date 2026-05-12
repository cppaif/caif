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
// Device-resident RMSNorm layer (templated on <ComputeT, StorageT>).
//
// Uniform two-parameter signature with both defaulting to `float`. Every
// (ComputeT, StorageT) cell from the cuBLAS-Lt-supported grid is a legal
// instantiation. The runtime factory CAIF_DeviceRMSNormFactory::Create
// (in caif_device_rmsnorm_factory.h) is the bridge for callers that have
// the dtypes only as runtime values.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceRMSNorm:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    CAIF_DeviceRMSNorm(uint32_t dim,
                       CAIF_CudaStream &stream,
                       float epsilon=g_caif_epsilon);
    ~CAIF_DeviceRMSNorm()override=default;

    // Move
    CAIF_DeviceRMSNorm(CAIF_DeviceRMSNorm &&other);
    CAIF_DeviceRMSNorm &operator=(CAIF_DeviceRMSNorm &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::RMSNorm_e;
    }
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors (StorageDtype()/ComputeDtype() inherited from
    // CAIF_DeviceLayerTyped — no per-layer copy needed).
    uint32_t Dim()const{return _dim;}
    float Epsilon()const{return _epsilon;}

    /**
     * @brief Replace the gamma weight tensor. Takes ownership by move.
     * The incoming tensor must have shape [dim]; otherwise throws.
     */
    void LoadGamma(CAIF_DeviceTensor &&gamma);

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
    uint32_t _dim;
    float _epsilon;
    CAIF_DeviceTensor _gamma;       // [dim] at StorageDtype()
    CAIF_DeviceTensor _gamma_grad;  // [dim] at StorageDtype()

    // Cached for backward
    CAIF_DeviceTensor _last_input;  // [rows, dim] at StorageDtype()
    CAIF_DeviceTensor _rms_cache;   // [rows] at fp32 (kernel-internal)
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceRMSNorm<float,float>;
extern template class CAIF_DeviceRMSNorm<float,__half>;
extern template class CAIF_DeviceRMSNorm<float,__nv_bfloat16>;
extern template class CAIF_DeviceRMSNorm<__half,float>;
extern template class CAIF_DeviceRMSNorm<__half,__half>;
extern template class CAIF_DeviceRMSNorm<__half,__nv_bfloat16>;
extern template class CAIF_DeviceRMSNorm<__nv_bfloat16,float>;
extern template class CAIF_DeviceRMSNorm<__nv_bfloat16,__half>;
extern template class CAIF_DeviceRMSNorm<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceRMSNorm<float,float>;
#endif

}//end instance namespace
