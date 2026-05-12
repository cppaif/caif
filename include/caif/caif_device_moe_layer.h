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
// Device-resident MoE Layer (templated on <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_moe_router.h"
#include "caif_device_moe_expert.h"
#include "caif_device_moe_expert_base.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceMoELayer:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    enum class OverflowStrategy_e:uint8_t
    {
      Drop=0,
      NoOp=1,
      Redistribute=2
    };

    CAIF_DeviceMoELayer(uint32_t input_dim,
                        uint32_t hidden_dim,
                        uint32_t num_experts,
                        uint32_t top_k,
                        bool expert_use_gated,
                        bool expert_use_bias,
                        uint32_t num_shared_experts,
                        uint32_t shared_hidden_dim,
                        bool router_use_bias,
                        float router_noise_std,
                        float capacity_factor,
                        OverflowStrategy_e overflow_strategy,
                        float balance_loss_weight,
                        float z_loss_weight,
                        CAIF_CudaStream &stream,
                        CAIF_DeviceMoELayerFactory::GatingKind_e gating_kind=
                          CAIF_DeviceMoELayerFactory::GatingKind_e::SoftmaxTopK_e,
                        bool norm_topk_prob=true,
                        float routed_scaling_factor=1.0f);

    CAIF_DeviceMoELayer(uint32_t input_dim,
                        uint32_t hidden_dim,
                        uint32_t top_k,
                        bool router_use_bias,
                        float router_noise_std,
                        float capacity_factor,
                        OverflowStrategy_e overflow_strategy,
                        float balance_loss_weight,
                        float z_loss_weight,
                        std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> routed_experts,
                        std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> shared_experts,
                        CAIF_CudaStream &stream,
                        CAIF_DeviceMoELayerFactory::GatingKind_e gating_kind=
                          CAIF_DeviceMoELayerFactory::GatingKind_e::SoftmaxTopK_e,
                        bool norm_topk_prob=true,
                        float routed_scaling_factor=1.0f);

    ~CAIF_DeviceMoELayer()override=default;

    CAIF_DeviceMoELayer(CAIF_DeviceMoELayer &&other);
    CAIF_DeviceMoELayer &operator=(CAIF_DeviceMoELayer &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::MoE_e;
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
    float AuxLoss()const override{return _last_balance_loss+_last_z_loss;}

    uint32_t InputDim()const{return _input_dim;}
    uint32_t HiddenDim()const{return _hidden_dim;}
    uint32_t NumExperts()const{return static_cast<uint32_t>(_experts.size());}
    uint32_t NumSharedExperts()const{return static_cast<uint32_t>(_shared_experts.size());}
    uint32_t TopK()const{return _top_k;}
    const CAIF_DeviceMoERouter<ComputeT,StorageT> &Router()const{return *_router;}
    CAIF_DeviceMoERouter<ComputeT,StorageT> &Router(){return *_router;}
    const CAIF_DeviceMoEExpertBase<ComputeT,StorageT> &Expert(size_t index)const{return *_experts[index];}
    CAIF_DeviceMoEExpertBase<ComputeT,StorageT> &Expert(size_t index){return *_experts[index];}
    const CAIF_DeviceMoEExpertBase<ComputeT,StorageT> &SharedExpert(size_t index)const{return *_shared_experts[index];}
    CAIF_DeviceMoEExpertBase<ComputeT,StorageT> &SharedExpert(size_t index){return *_shared_experts[index];}

    // Move-out accessors used by add-MoE Phase 4 — when an existing
    // CAIF_DeviceMoELayer needs to be replaced with a wider one (e.g.
    // GLM layers 1-46: extend the 64 trainable experts with N new
    // ones), the builder takes ownership of the existing experts /
    // shared experts, wraps the trainables as frozen, and constructs
    // a fresh layer with the captured experts plus N new trainables
    // before ReplaceLayer-ing the slot. Post-Take* the layer is
    // empty and unusable until destroyed; callers must ReplaceLayer
    // (or otherwise discard the layer) immediately after.
    typedef std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> ExpertVec_t;
    ExpertVec_t TakeAllExperts(){return std::move(_experts);}
    ExpertVec_t TakeAllSharedExperts(){return std::move(_shared_experts);}

    float LastBalanceLoss()const{return _last_balance_loss;}
    float LastZLoss()const{return _last_z_loss;}

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
    uint32_t _input_dim;
    uint32_t _hidden_dim;
    uint32_t _top_k;
    float    _capacity_factor;
    OverflowStrategy_e _overflow_strategy;
    float    _balance_loss_weight;
    float    _z_loss_weight;

    std::unique_ptr<CAIF_DeviceMoERouter<ComputeT,StorageT>> _router;
    std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> _experts;
    std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> _shared_experts;

    typename CAIF_DeviceMoERouter<ComputeT,StorageT>::RouterOutput_t _cached_routing;
    // No `_cached_expert_outputs` member: BackwardImpl reads
    // `_ws_expert_output_buffer` directly (the packed per-expert workspace
    // that ForwardImpl wrote into via `ForwardInto`). A separate vector of
    // per-expert tensors would be redundant.
    std::vector<uint32_t> _cached_token_counts;
    CAIF_DeviceTensor _cached_logsumexp;

    float _last_balance_loss;
    float _last_z_loss;

    std::vector<uint32_t> _overflow_tokens;

    CAIF_DeviceTensor _ws_dispatch_map;
    CAIF_DeviceTensor _ws_expert_offsets;
    CAIF_DeviceTensor _ws_expert_input_buffer;
    CAIF_DeviceTensor _ws_expert_output_buffer;
    CAIF_DeviceTensor _ws_grad_expert_output_buffer;
    CAIF_DeviceTensor _ws_grad_expert_input_buffer;

    float ComputeBalanceLoss(const CAIF_DeviceTensor &router_probs);
    float ComputeZLoss(const CAIF_DeviceTensor &router_logits);
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceMoELayer<float,float>;
extern template class CAIF_DeviceMoELayer<float,__half>;
extern template class CAIF_DeviceMoELayer<float,__nv_bfloat16>;
extern template class CAIF_DeviceMoELayer<__half,float>;
extern template class CAIF_DeviceMoELayer<__half,__half>;
extern template class CAIF_DeviceMoELayer<__half,__nv_bfloat16>;
extern template class CAIF_DeviceMoELayer<__nv_bfloat16,float>;
extern template class CAIF_DeviceMoELayer<__nv_bfloat16,__half>;
extern template class CAIF_DeviceMoELayer<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceMoELayer<float,float>;
#endif

}//end instance namespace
