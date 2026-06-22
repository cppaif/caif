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
// Device-resident MoE Router (templated on <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_device_moe_layer_factory.h"
#include "caif_device_moe_router_config.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceMoERouter:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    // RoutingType_e now lives on CAIF_DeviceMoERouterConfig; aliased here so
    // existing CAIF_DeviceMoERouter<...>::RoutingType_e references keep working.
    using RoutingType_e=CAIF_DeviceMoERouterConfig::RoutingType_e;

    struct RouterOutput_t
    {
      CAIF_DeviceTensor expert_indices;
      CAIF_DeviceTensor expert_weights;
      CAIF_DeviceTensor router_logits;
      CAIF_DeviceTensor router_probs;
    };

    CAIF_DeviceMoERouter(const CAIF_DeviceMoERouterConfig &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceMoERouter()override=default;

    CAIF_DeviceMoERouter(CAIF_DeviceMoERouter &&other);
    CAIF_DeviceMoERouter &operator=(CAIF_DeviceMoERouter &&other);

    RouterOutput_t Route(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::MoERouter_e;
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

    CAIF_DeviceTensor BackwardRouting(const CAIF_DeviceTensor &grad_weights,CAIF_RunContext &ctx);
    CAIF_DeviceTensor BackwardRoutingAuxAware(const CAIF_DeviceTensor &grad_weights,
                                              const CAIF_DeviceTensor &balance_bias,
                                              const CAIF_DeviceTensor &z_logsumexp_scaled,
                                              CAIF_RunContext &ctx);

    const CAIF_DeviceMoERouterConfig &Config()const{return _config;}
    uint32_t InputDim()const{return Config().InputDim();}
    uint32_t NumExperts()const{return Config().NumExperts();}
    uint32_t TopK()const{return Config().TopK();}

    // Bias the router so that expert `expert_index` dominates at
    // initialization. Sets the router's bias term to a one-hot pattern
    // whose `expert_index` slot carries `bias_magnitude` and all other
    // slots carry zero (and zeroes the matrix weight so the bias alone
    // determines the logits at init). For typical input distributions
    // a `bias_magnitude` of ~5.0 yields softmax(expert_index) > 0.99.
    //
    // Used by the add-moe / layer-surgery path so the model starts in
    // the "frozen base behavior" state — Expert 0 (the frozen-FFN
    // expert) gets all the routing weight at init, and training learns
    // to route to the trainable experts where useful.
    //
    // Caller contract: `expert_index < NumExperts()`. The router config
    // must have been built with `use_bias=true`; mismatch throws.
    void InitFavorExpert(uint32_t expert_index,float bias_magnitude);

    // Read-side accessors to the router's trained weights. Used by
    // add-MoE Phase 4 — when an existing CAIF_DeviceMoELayer's MoE
    // layer is being widened (e.g. GLM layers 1-46 going from 64
    // experts to 64+N), the builder reads the trained `_w_router`
    // and `_b_router` out of the existing router, allocates a wider
    // tensor in caller space whose top `original_num_experts` rows
    // match the original, fills the bottom rows with init values,
    // and hands it back to the new (wider) router via LoadWRouter /
    // LoadBRouter. Without preservation the trained routing signal
    // is lost at the moment of substitution.
    const CAIF_DeviceTensor &WRouter()const{return _w_router;}
    const CAIF_DeviceTensor &BRouter()const{return _b_router;}

    // Move-in setters for the router weights. Caller must respect
    // shape: `_w_router` is `[input_dim, num_experts]`, `_b_router`
    // is `[num_experts]`. Mismatch throws. Used to preserve trained
    // router weights when widening a MoE layer (see `WRouter()`
    // docstring above).
    void LoadWRouter(CAIF_DeviceTensor &&w);
    void LoadBRouter(CAIF_DeviceTensor &&b);

    // Aux-loss-free load-balancing bias update (DeepSeek-V3). Called by the
    // training loop once per optimizer step: counts the most recent training
    // forward's expert load (from the cached routing indices) and nudges each
    // router bias toward balance (bias[e] += rate*sign(mean_load - load[e])).
    // No-op unless Config().BiasUpdateRate() > 0. Single-forward-per-step
    // counts the whole batch; with gradient accumulation it counts the last
    // micro-batch only.
    void UpdateAuxLossFreeBias();

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
    CAIF_DeviceTensor &WRouterMut(){return _w_router;}
    CAIF_DeviceTensor &BRouterMut(){return _b_router;}
    CAIF_DeviceTensor &GradWRouterMut(){return _grad_w_router;}
    CAIF_DeviceTensor &GradBRouterMut(){return _grad_b_router;}
    CAIF_DeviceTensor &CachedInputMut(){return _cached_input;}
    CAIF_DeviceTensor &CachedLogitsMut(){return _cached_logits;}
    CAIF_DeviceTensor &CachedProbsMut(){return _cached_probs;}
    CAIF_DeviceTensor &CachedIndicesMut(){return _cached_indices;}
    CAIF_DeviceTensor &GradTopkPreviewMut(){return _grad_topk_preview;}

    const CAIF_DeviceTensor &GradWRouter()const{return _grad_w_router;}
    const CAIF_DeviceTensor &GradBRouter()const{return _grad_b_router;}
    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    const CAIF_DeviceTensor &CachedLogits()const{return _cached_logits;}
    const CAIF_DeviceTensor &CachedProbs()const{return _cached_probs;}
    const CAIF_DeviceTensor &CachedIndices()const{return _cached_indices;}
    const CAIF_DeviceTensor &GradTopkPreview()const{return _grad_topk_preview;}

    void SetWRouter(CAIF_DeviceTensor &&w){_w_router=std::move(w);}
    void SetBRouter(CAIF_DeviceTensor &&b){_b_router=std::move(b);}
    void SetGradWRouter(CAIF_DeviceTensor &&t){_grad_w_router=std::move(t);}
    void SetGradBRouter(CAIF_DeviceTensor &&t){_grad_b_router=std::move(t);}
    void SetCachedInput(CAIF_DeviceTensor &&t){_cached_input=std::move(t);}
    void SetCachedLogits(CAIF_DeviceTensor &&t){_cached_logits=std::move(t);}
    void SetCachedProbs(CAIF_DeviceTensor &&t){_cached_probs=std::move(t);}
    void SetCachedIndices(CAIF_DeviceTensor &&t){_cached_indices=std::move(t);}
    void SetGradTopkPreview(CAIF_DeviceTensor &&t){_grad_topk_preview=std::move(t);}

    void SetConfig(const CAIF_DeviceMoERouterConfig &c){_config=c;}

    CAIF_DeviceMoERouterConfig _config;

    CAIF_DeviceTensor _w_router;
    CAIF_DeviceTensor _b_router;

    CAIF_DeviceTensor _grad_w_router;
    CAIF_DeviceTensor _grad_b_router;

    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_logits;
    CAIF_DeviceTensor _cached_probs;
    CAIF_DeviceTensor _cached_indices;
    CAIF_DeviceTensor _grad_topk_preview;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceMoERouter<float,float>;
extern template class CAIF_DeviceMoERouter<float,__half>;
extern template class CAIF_DeviceMoERouter<float,__nv_bfloat16>;
extern template class CAIF_DeviceMoERouter<__half,float>;
extern template class CAIF_DeviceMoERouter<__half,__half>;
extern template class CAIF_DeviceMoERouter<__half,__nv_bfloat16>;
extern template class CAIF_DeviceMoERouter<__nv_bfloat16,float>;
extern template class CAIF_DeviceMoERouter<__nv_bfloat16,__half>;
extern template class CAIF_DeviceMoERouter<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceMoERouter<float,float>;
#endif

}//end instance namespace
