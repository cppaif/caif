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
// Device-resident Linear Head — output projection layer (templated on
// <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceLinearHead:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    struct Config_t
    {
      uint32_t input_dim;
      uint32_t output_dim;
      bool use_bias;
    };

    CAIF_DeviceLinearHead(const Config_t &config,CAIF_CudaStream &stream);

    CAIF_DeviceLinearHead(const Config_t &config,
                          CAIF_DeviceTensor &tied_weight,
                          CAIF_DeviceTensor &tied_weight_grad,
                          CAIF_CudaStream &stream);

    ~CAIF_DeviceLinearHead()override=default;

    CAIF_DeviceLinearHead(CAIF_DeviceLinearHead &&other);
    CAIF_DeviceLinearHead &operator=(CAIF_DeviceLinearHead &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::LinearHead_e;
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

    uint32_t InputDim()const{return _config.input_dim;}
    uint32_t OutputDim()const{return _config.output_dim;}
    bool UseBias()const{return _config.use_bias;}
    bool IsWeightTied()const{return _weight_tied;}
    bool Frozen()const{return _frozen;}
    void SetFrozen(bool frozen){_frozen=frozen;}

    void LoadWeight(CAIF_DeviceTensor &&weight);
    void LoadBias(CAIF_DeviceTensor &&bias);

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
    CAIF_DeviceTensor &WeightMut(){return _weight;}
    void SetWeight(CAIF_DeviceTensor &&w){_weight=std::move(w);}

    Config_t _config;
    bool _weight_tied;
    bool _frozen;

    CAIF_DeviceTensor _weight;
    CAIF_DeviceTensor _weight_grad;

    CAIF_DeviceTensor *_tied_weight;
    CAIF_DeviceTensor *_tied_weight_grad;

    CAIF_DeviceTensor _bias;
    CAIF_DeviceTensor _bias_grad;

    CAIF_DeviceTensor _cached_input;
    std::vector<uint32_t> _cached_shape;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceLinearHead<float,float>;
extern template class CAIF_DeviceLinearHead<float,__half>;
extern template class CAIF_DeviceLinearHead<float,__nv_bfloat16>;
extern template class CAIF_DeviceLinearHead<__half,float>;
extern template class CAIF_DeviceLinearHead<__half,__half>;
extern template class CAIF_DeviceLinearHead<__half,__nv_bfloat16>;
extern template class CAIF_DeviceLinearHead<__nv_bfloat16,float>;
extern template class CAIF_DeviceLinearHead<__nv_bfloat16,__half>;
extern template class CAIF_DeviceLinearHead<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceLinearHead<float,float>;
#endif

}//end instance namespace
