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
// LoRA (Low-Rank Adaptation) adapter (templated on <ComputeT, StorageT>).
//
// LoRA A/B matrices are stored at StorageT and computed with ComputeT, so
// the adapter composes cleanly inside a templated MHA at any
// (compute, storage) pair the rest of the framework supports. Defaults
// make `CAIF_DeviceLoRAAdapter<>` valid syntax for the legacy fp32 path.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_run_context.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#ifdef USE_CAIF_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <type_traits>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceLoRAAdapter:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    struct LoRAConfig_t
    {
      uint32_t rank;
      float alpha;
      uint32_t input_dim;
      uint32_t output_dim;
    };

    CAIF_DeviceLoRAAdapter(const LoRAConfig_t &config,
                           std::unique_ptr<CAIF_DeviceLayer> base_layer,
                           CAIF_CudaStream &stream,
                           uint32_t seed=0);

    ~CAIF_DeviceLoRAAdapter()override=default;

    CAIF_DeviceLoRAAdapter(CAIF_DeviceLoRAAdapter &&other);
    CAIF_DeviceLoRAAdapter &operator=(CAIF_DeviceLoRAAdapter &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::LoRAAdapter_e;
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
    size_t FrozenTensorCount()const override;
    CAIF_DeviceTensor FrozenTensorFP32(size_t index)const override;
    std::vector<std::string> FrozenTensorNames(const std::string &prefix="")const override;

    const LoRAConfig_t &Config()const{return _config;}

    CAIF_DeviceLayer &BaseLayer()
    {
      if(_base_layer==nullptr)
      {
        THROW_CAIFE("LoRAAdapter: base_layer is null");
      }
      return *_base_layer;
    }
    const CAIF_DeviceLayer &BaseLayer()const
    {
      if(_base_layer==nullptr)
      {
        THROW_CAIFE("LoRAAdapter: base_layer is null");
      }
      return *_base_layer;
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

  private:
    CAIF_DeviceTensor &LoRAAMut(){return _lora_a;}

    LoRAConfig_t _config;
    std::unique_ptr<CAIF_DeviceLayer> _base_layer;

    CAIF_DeviceTensor _lora_a;
    CAIF_DeviceTensor _lora_b;

    CAIF_DeviceTensor _grad_lora_a;
    CAIF_DeviceTensor _grad_lora_b;

    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_lora_hidden;
};

extern template class CAIF_DeviceLoRAAdapter<float,float>;
#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceLoRAAdapter<float,__half>;
extern template class CAIF_DeviceLoRAAdapter<float,__nv_bfloat16>;
extern template class CAIF_DeviceLoRAAdapter<__half,float>;
extern template class CAIF_DeviceLoRAAdapter<__half,__half>;
extern template class CAIF_DeviceLoRAAdapter<__half,__nv_bfloat16>;
extern template class CAIF_DeviceLoRAAdapter<__nv_bfloat16,float>;
extern template class CAIF_DeviceLoRAAdapter<__nv_bfloat16,__half>;
extern template class CAIF_DeviceLoRAAdapter<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
