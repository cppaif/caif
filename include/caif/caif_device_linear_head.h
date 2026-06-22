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
#include "caif_device_linear_head_config.h"
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
    CAIF_DeviceLinearHead(const CAIF_DeviceLinearHeadConfig &config,CAIF_CudaStream &stream);

    CAIF_DeviceLinearHead(const CAIF_DeviceLinearHeadConfig &config,
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

    uint32_t InputDim()const{return Config().InputDim();}
    uint32_t OutputDim()const{return Config().OutputDim();}
    bool UseBias()const{return Config().UseBias();}
    float OutputScale()const{return Config().OutputScale();}
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
    const CAIF_DeviceLinearHeadConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DeviceLinearHeadConfig &c){_config=c;}
    void SetWeightTied(const bool v){_weight_tied=v;}

    const CAIF_DeviceTensor &Weight()const{return _weight;}
    CAIF_DeviceTensor &WeightMut(){return _weight;}
    void SetWeight(CAIF_DeviceTensor &&w){_weight=std::move(w);}

    const CAIF_DeviceTensor &WeightGrad()const{return _weight_grad;}
    CAIF_DeviceTensor &WeightGrad(){return _weight_grad;}
    void SetWeightGrad(CAIF_DeviceTensor &&t){_weight_grad=std::move(t);}

    CAIF_DeviceTensor *TiedWeight(){return _tied_weight;}
    const CAIF_DeviceTensor *TiedWeight()const{return _tied_weight;}
    void SetTiedWeight(CAIF_DeviceTensor *p){_tied_weight=p;}

    CAIF_DeviceTensor *TiedWeightGrad(){return _tied_weight_grad;}
    const CAIF_DeviceTensor *TiedWeightGrad()const{return _tied_weight_grad;}
    void SetTiedWeightGrad(CAIF_DeviceTensor *p){_tied_weight_grad=p;}

    const CAIF_DeviceTensor &Bias()const{return _bias;}
    CAIF_DeviceTensor &Bias(){return _bias;}
    void SetBias(CAIF_DeviceTensor &&t){_bias=std::move(t);}

    const CAIF_DeviceTensor &BiasGrad()const{return _bias_grad;}
    CAIF_DeviceTensor &BiasGrad(){return _bias_grad;}
    void SetBiasGrad(CAIF_DeviceTensor &&t){_bias_grad=std::move(t);}

    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    CAIF_DeviceTensor &CachedInput(){return _cached_input;}
    void SetCachedInput(CAIF_DeviceTensor &&t){_cached_input=std::move(t);}

    const std::vector<uint32_t> &CachedShape()const{return _cached_shape;}
    std::vector<uint32_t> &CachedShape(){return _cached_shape;}
    void SetCachedShape(std::vector<uint32_t> &&v){_cached_shape=std::move(v);}

    CAIF_DeviceLinearHeadConfig _config;
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
