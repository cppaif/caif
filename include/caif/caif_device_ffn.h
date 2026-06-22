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
// Device-resident generic Feed-Forward Network layer (templated on
// <ComputeT, StorageT>).
//
// Activation strategy is held polymorphically as
// std::unique_ptr<CAIF_DeviceActivation>; the activation's Forward/Backward
// take dtype-erased CAIF_DeviceTensor so the polymorphic call remains
// dtype-correct regardless of which (ComputeT, StorageT) cell FFN is.
//
// Pointwise mode (2 weights):
//   hidden = input @ W1
//   act    = activation(hidden)
//   output = act @ W2
//
// Gated mode (3 weights):
//   gate   = input @ W_gate
//   up     = input @ W_up
//   act    = activation(gate, up)
//   output = act @ W_down
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_ffn_config.h"
#include "caif_device_activation.h"
#include "caif_device_pointwise_activation.h"
#include "caif_device_gated_activation.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceFFN:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    struct FFNProjections_t
    {
      std::unique_ptr<CAIF_DeviceLayer> gate;
      std::unique_ptr<CAIF_DeviceLayer> up;
      std::unique_ptr<CAIF_DeviceLayer> down;
    };

    CAIF_DeviceFFN(const CAIF_DeviceFFNConfig &config,
                   std::unique_ptr<CAIF_DeviceActivation> activation,
                   CAIF_CudaStream &stream);
    CAIF_DeviceFFN(const CAIF_DeviceFFNConfig &config,
                   FFNProjections_t projections,
                   std::unique_ptr<CAIF_DeviceActivation> activation,
                   CAIF_CudaStream &stream);
    ~CAIF_DeviceFFN()override=default;

    CAIF_DeviceFFN(CAIF_DeviceFFN &&other);
    CAIF_DeviceFFN &operator=(CAIF_DeviceFFN &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::FFN_e;
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

    const CAIF_DeviceFFNConfig &Config()const{return _config;}
    bool IsGated()const{return _is_gated;}

    void InitializeWeights(uint32_t seed=0)override;

    void LoadW1(CAIF_DeviceTensor &&w1);
    void LoadW2(CAIF_DeviceTensor &&w2);
    void LoadWGate(CAIF_DeviceTensor &&w_gate);
    void LoadWUp(CAIF_DeviceTensor &&w_up);
    void LoadWDown(CAIF_DeviceTensor &&w_down);

    // Read-side weight accessors for layer-surgery scenarios (e.g.
    // add-MoE: extract a base FFN's weights, hand them to a frozen
    // expert wrapper, then replace this FFN sublayer with a MoE
    // sublayer that carries the frozen expert as Expert 0). The
    // gated path uses W_Gate/W_Up/W_Down; the non-gated path uses
    // W1/W2.
    const CAIF_DeviceTensor &WGate()const{return _w_gate;}
    const CAIF_DeviceTensor &WUp()const{return _w_up;}
    const CAIF_DeviceTensor &WDown()const{return _w_down;}
    const CAIF_DeviceTensor &W1()const{return _w1;}
    const CAIF_DeviceTensor &W2()const{return _w2;}

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

    void UploadAtStorage(CAIF_DeviceTensor &dst,const std::vector<float> &host_data);

  private:
    // Internal accessor / setter pairs for every member. Method bodies
    // route through these per CODING_GUIDELINES.md §Member Access.
    void SetConfig(const CAIF_DeviceFFNConfig &c){_config=c;}
    void SetIsGated(const bool v){_is_gated=v;}
    void SetUseProjections(const bool v){_use_projections=v;}
    bool UseProjections()const{return _use_projections;}

    CAIF_DeviceActivation &Activation()
    {
      if(_activation==nullptr)
      {
        THROW_CAIFE("CAIF_DeviceFFN::Activation: activation is null");
      }
      return *_activation;
    }
    const CAIF_DeviceActivation &Activation()const
    {
      if(_activation==nullptr)
      {
        THROW_CAIFE("CAIF_DeviceFFN::Activation: activation is null");
      }
      return *_activation;
    }

    const FFNProjections_t &Projections()const{return _projections;}
    FFNProjections_t &ProjectionsMut(){return _projections;}
    void SetProjections(FFNProjections_t &&v){_projections=std::move(v);}

    CAIF_DeviceTensor &W1Mut(){return _w1;}
    void SetW1(CAIF_DeviceTensor &&t){_w1=std::move(t);}
    CAIF_DeviceTensor &W2Mut(){return _w2;}
    void SetW2(CAIF_DeviceTensor &&t){_w2=std::move(t);}

    const CAIF_DeviceTensor &GradW1()const{return _grad_w1;}
    CAIF_DeviceTensor &GradW1(){return _grad_w1;}
    void SetGradW1(CAIF_DeviceTensor &&t){_grad_w1=std::move(t);}
    const CAIF_DeviceTensor &GradW2()const{return _grad_w2;}
    CAIF_DeviceTensor &GradW2(){return _grad_w2;}
    void SetGradW2(CAIF_DeviceTensor &&t){_grad_w2=std::move(t);}

    CAIF_DeviceTensor &WGateMut(){return _w_gate;}
    void SetWGate(CAIF_DeviceTensor &&t){_w_gate=std::move(t);}
    CAIF_DeviceTensor &WUpMut(){return _w_up;}
    void SetWUp(CAIF_DeviceTensor &&t){_w_up=std::move(t);}
    CAIF_DeviceTensor &WDownMut(){return _w_down;}
    void SetWDown(CAIF_DeviceTensor &&t){_w_down=std::move(t);}

    const CAIF_DeviceTensor &GradWGate()const{return _grad_w_gate;}
    CAIF_DeviceTensor &GradWGate(){return _grad_w_gate;}
    void SetGradWGate(CAIF_DeviceTensor &&t){_grad_w_gate=std::move(t);}
    const CAIF_DeviceTensor &GradWUp()const{return _grad_w_up;}
    CAIF_DeviceTensor &GradWUp(){return _grad_w_up;}
    void SetGradWUp(CAIF_DeviceTensor &&t){_grad_w_up=std::move(t);}
    const CAIF_DeviceTensor &GradWDown()const{return _grad_w_down;}
    CAIF_DeviceTensor &GradWDown(){return _grad_w_down;}
    void SetGradWDown(CAIF_DeviceTensor &&t){_grad_w_down=std::move(t);}

    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    CAIF_DeviceTensor &CachedInput(){return _cached_input;}
    void SetCachedInput(CAIF_DeviceTensor &&t){_cached_input=std::move(t);}

    const CAIF_DeviceTensor &CachedPreActivation()const{return _cached_pre_activation;}
    CAIF_DeviceTensor &CachedPreActivation(){return _cached_pre_activation;}
    void SetCachedPreActivation(CAIF_DeviceTensor &&t){_cached_pre_activation=std::move(t);}

    const CAIF_DeviceTensor &CachedPostActivation()const{return _cached_post_activation;}
    CAIF_DeviceTensor &CachedPostActivation(){return _cached_post_activation;}
    void SetCachedPostActivation(CAIF_DeviceTensor &&t){_cached_post_activation=std::move(t);}

    const CAIF_DeviceTensor &CachedGateInput()const{return _cached_gate_input;}
    CAIF_DeviceTensor &CachedGateInput(){return _cached_gate_input;}
    void SetCachedGateInput(CAIF_DeviceTensor &&t){_cached_gate_input=std::move(t);}

    const CAIF_DeviceTensor &CachedUpInput()const{return _cached_up_input;}
    CAIF_DeviceTensor &CachedUpInput(){return _cached_up_input;}
    void SetCachedUpInput(CAIF_DeviceTensor &&t){_cached_up_input=std::move(t);}

    const CAIF_DeviceTensor &CachedActOutput()const{return _cached_act_output;}
    CAIF_DeviceTensor &CachedActOutput(){return _cached_act_output;}
    void SetCachedActOutput(CAIF_DeviceTensor &&t){_cached_act_output=std::move(t);}

    const std::vector<uint32_t> &CachedInputShape()const{return _cached_input_shape;}
    std::vector<uint32_t> &CachedInputShape(){return _cached_input_shape;}
    void SetCachedInputShape(std::vector<uint32_t> &&v){_cached_input_shape=std::move(v);}

    CAIF_DeviceFFNConfig _config;
    std::unique_ptr<CAIF_DeviceActivation> _activation;
    bool _is_gated;
    FFNProjections_t _projections;
    bool _use_projections;

    CAIF_DeviceTensor _w1;
    CAIF_DeviceTensor _w2;
    CAIF_DeviceTensor _grad_w1;
    CAIF_DeviceTensor _grad_w2;

    CAIF_DeviceTensor _w_gate;
    CAIF_DeviceTensor _w_up;
    CAIF_DeviceTensor _w_down;
    CAIF_DeviceTensor _grad_w_gate;
    CAIF_DeviceTensor _grad_w_up;
    CAIF_DeviceTensor _grad_w_down;

    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_pre_activation;
    CAIF_DeviceTensor _cached_post_activation;
    CAIF_DeviceTensor _cached_gate_input;
    CAIF_DeviceTensor _cached_up_input;
    CAIF_DeviceTensor _cached_act_output;
    std::vector<uint32_t> _cached_input_shape;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceFFN<float,float>;
extern template class CAIF_DeviceFFN<float,__half>;
extern template class CAIF_DeviceFFN<float,__nv_bfloat16>;
extern template class CAIF_DeviceFFN<__half,float>;
extern template class CAIF_DeviceFFN<__half,__half>;
extern template class CAIF_DeviceFFN<__half,__nv_bfloat16>;
extern template class CAIF_DeviceFFN<__nv_bfloat16,float>;
extern template class CAIF_DeviceFFN<__nv_bfloat16,__half>;
extern template class CAIF_DeviceFFN<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceFFN<float,float>;
#endif

}//end instance namespace
