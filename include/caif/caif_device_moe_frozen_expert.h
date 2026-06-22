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
// Frozen MoE expert: 3 FrozenLinear sub-layers (gate, up, down) wrapping
// frozen FFN weights. Implements the same Forward/Backward/ForwardInto
// surface as CAIF_DeviceMoEExpert via the shared
// CAIF_DeviceMoEExpertBase<C, S> abstract base. Has 0 trainable
// parameters; backward returns the input gradient (so upstream gradient
// flow continues) but emits no parameter gradients.
//
// The wrapper's <ComputeT, StorageT> describe the activation pipeline
// (input dtype, output dtype). Each internal FrozenLinear can carry its
// own weight storage dtype (fp32 / fp16 / bf16 / int8 / int4) — caller
// constructs the FrozenLinear sub-layers and moves them in. The wrapper
// composes them with the gated-FFN dataflow:
//
//   gate (if use_gated): a = gate_layer.Forward(input)
//                        u = up_layer.Forward(input)
//                        h = SiLU(u) * a
//   else:                u = up_layer.Forward(input)
//                        h = SiLU(u)
//   out = down_layer.Forward(h)
//
// The expected use is the add-moe / layer-surgery path — Expert 0 wraps
// the pretrained FFN's gate/up/down at int8 or int4 storage, leaving
// the surrounding MoELayer's StorageT free to be fp16/bf16.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_frozen_linear.h"
#include "caif_device_moe_expert_base.h"
#include "caif_device_moe_frozen_expert_config.h"
#include "caif_device_tensor.h"
#include "caif_run_context.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceMoEFrozenExpert:public CAIF_DeviceMoEExpertBase<ComputeT,StorageT>
{
  public:

    // Sub-layer pack — one FrozenLinear per matmul. Each can be at any
    // storage cell (fp32/fp16/bf16/int8/int4) the FrozenLinear template
    // grid supports; their output is at ComputeT (fp32 if ComputeT=fp32,
    // else the FrozenLinear's templated compute dtype). The expert
    // composes them via gated-FFN dataflow.
    struct FrozenSubLayers_t
    {
      std::unique_ptr<CAIF_DeviceFrozenLinearBase> gate; // input_dim -> hidden_dim, optional
      std::unique_ptr<CAIF_DeviceFrozenLinearBase> up;   // input_dim -> hidden_dim
      std::unique_ptr<CAIF_DeviceFrozenLinearBase> down; // hidden_dim -> input_dim
    };

    CAIF_DeviceMoEFrozenExpert(const CAIF_DeviceMoEFrozenExpertConfig &config,
                               FrozenSubLayers_t sub_layers,
                               CAIF_CudaStream &stream);
    ~CAIF_DeviceMoEFrozenExpert()override=default;

    CAIF_DeviceMoEFrozenExpert(CAIF_DeviceMoEFrozenExpert &&other);
    CAIF_DeviceMoEFrozenExpert &operator=(CAIF_DeviceMoEFrozenExpert &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;
    void ForwardInto(const CAIF_DeviceTensor &input,
                     CAIF_DeviceTensor &output,
                     CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::MoEExpert_e;
    }
    void ZeroGradients()override{}                          // no params
    size_t ParameterTensorCount()const override{return 0;}  // no params
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override{return 0;}   // no params
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    const CAIF_DeviceMoEFrozenExpertConfig &Config()const{return _config;}
    uint32_t InputDim()const override{return Config().InputDim();}
    uint32_t HiddenDim()const override{return Config().HiddenDim();}
    bool UseGated()const{return Config().UseGated();}

    // FrozenLinear-base accessors for the offload walker. Gate is
    // optional — present iff UseGated()==true. Up and Down are always
    // present. References are non-const so a caller's offload-policy
    // walker can call SetOffloadPolicy / register with a block scheduler.
    bool HasGate()const{return SubLayers().gate!=nullptr;}
    CAIF_DeviceFrozenLinearBase &Gate()
    {
      if(SubLayersMut().gate==nullptr)
      {
        THROW_CAIFE("MoEFrozenExpert::Gate(): use_gated=false on this expert");
      }
      return *SubLayersMut().gate;
    }
    CAIF_DeviceFrozenLinearBase &Up()
    {
      if(SubLayersMut().up==nullptr)
      {
        THROW_CAIFE("MoEFrozenExpert::Up(): up sub-layer is null");
      }
      return *SubLayersMut().up;
    }
    CAIF_DeviceFrozenLinearBase &Down()
    {
      if(SubLayersMut().down==nullptr)
      {
        THROW_CAIFE("MoEFrozenExpert::Down(): down sub-layer is null");
      }
      return *SubLayersMut().down;
    }

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  private:
    // Internal accessors — single point of access for every member
    // even from inside this class's own methods. See
    // CODING_GUIDELINES.md §Member Access. *Mut() forms hand out a
    // non-const reference for in-place mutation; Set*() forms do
    // rebind via rvalue-only setter.
    void SetConfig(const CAIF_DeviceMoEFrozenExpertConfig &c){_config=c;}
    const FrozenSubLayers_t &SubLayers()const{return _sub_layers;}
    FrozenSubLayers_t &SubLayersMut(){return _sub_layers;}
    void SetSubLayers(FrozenSubLayers_t &&v){_sub_layers=std::move(v);}
    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    CAIF_DeviceTensor &CachedInput(){return _cached_input;}
    void SetCachedInput(CAIF_DeviceTensor &&v){_cached_input=std::move(v);}
    const CAIF_DeviceTensor &CachedGateOut()const{return _cached_gate_out;}
    CAIF_DeviceTensor &CachedGateOut(){return _cached_gate_out;}
    void SetCachedGateOut(CAIF_DeviceTensor &&v){_cached_gate_out=std::move(v);}
    const CAIF_DeviceTensor &CachedUpOut()const{return _cached_up_out;}
    CAIF_DeviceTensor &CachedUpOut(){return _cached_up_out;}
    void SetCachedUpOut(CAIF_DeviceTensor &&v){_cached_up_out=std::move(v);}
    const CAIF_DeviceTensor &CachedHidden()const{return _cached_hidden;}
    CAIF_DeviceTensor &CachedHidden(){return _cached_hidden;}
    void SetCachedHidden(CAIF_DeviceTensor &&v){_cached_hidden=std::move(v);}

    CAIF_DeviceMoEFrozenExpertConfig _config;
    FrozenSubLayers_t _sub_layers;

    // Forward intermediates cached for backward. Populated only when
    // ctx.Training()==true.
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_gate_out;
    CAIF_DeviceTensor _cached_up_out;
    CAIF_DeviceTensor _cached_hidden;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceMoEFrozenExpert<float,float>;
extern template class CAIF_DeviceMoEFrozenExpert<float,__half>;
extern template class CAIF_DeviceMoEFrozenExpert<float,__nv_bfloat16>;
extern template class CAIF_DeviceMoEFrozenExpert<__half,float>;
extern template class CAIF_DeviceMoEFrozenExpert<__half,__half>;
extern template class CAIF_DeviceMoEFrozenExpert<__half,__nv_bfloat16>;
extern template class CAIF_DeviceMoEFrozenExpert<__nv_bfloat16,float>;
extern template class CAIF_DeviceMoEFrozenExpert<__nv_bfloat16,__half>;
extern template class CAIF_DeviceMoEFrozenExpert<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceMoEFrozenExpert<float,float>;
#endif

}//end instance namespace
