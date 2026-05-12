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
// T5-style Relative Position Bias layer (templated on
// <ComputeT, StorageT>).
//
// The embedding table and its gradient stay fp32 by design (atomicAdd
// safety). StorageT controls only the dtype of the produced [num_heads,
// q_len, k_len] bias output (which feeds attention scores and so must
// match the attention StorageT).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceRelativePositionBias:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    struct Config_t
    {
      uint32_t num_heads;
      uint32_t num_buckets;
      uint32_t max_distance;
      bool bidirectional;
    };

    CAIF_DeviceRelativePositionBias(const Config_t &config,
                                    CAIF_CudaStream &stream);
    ~CAIF_DeviceRelativePositionBias()override=default;

    CAIF_DeviceRelativePositionBias(CAIF_DeviceRelativePositionBias &&other);
    CAIF_DeviceRelativePositionBias &operator=(CAIF_DeviceRelativePositionBias &&other);

    CAIF_DeviceTensor ComputeBias(uint32_t q_len,uint32_t k_len);
    void AccumulateGradient(const CAIF_DeviceTensor &grad_bias,
                            uint32_t q_len,uint32_t k_len);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::RelativePositionBias_e;
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

    const Config_t &Config()const{return _config;}

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
    Config_t _config;

    // Embedding stays fp32 by design — atomicAdd safety, also low-rank
    // (num_heads * num_buckets) so dtype savings are negligible.
    CAIF_DeviceTensor _embedding;
    CAIF_DeviceTensor _grad_embedding;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceRelativePositionBias<float,float>;
extern template class CAIF_DeviceRelativePositionBias<float,__half>;
extern template class CAIF_DeviceRelativePositionBias<float,__nv_bfloat16>;
extern template class CAIF_DeviceRelativePositionBias<__half,float>;
extern template class CAIF_DeviceRelativePositionBias<__half,__half>;
extern template class CAIF_DeviceRelativePositionBias<__half,__nv_bfloat16>;
extern template class CAIF_DeviceRelativePositionBias<__nv_bfloat16,float>;
extern template class CAIF_DeviceRelativePositionBias<__nv_bfloat16,__half>;
extern template class CAIF_DeviceRelativePositionBias<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceRelativePositionBias<float,float>;
#endif

}//end instance namespace
