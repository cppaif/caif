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
// Device-resident MoE Expert (templated on <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
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
class CAIF_DeviceMoEExpert:public CAIF_DeviceMoEExpertBase<ComputeT,StorageT>
{
  public:
    struct Config_t
    {
      uint32_t input_dim;
      uint32_t hidden_dim;
      bool use_gated;
      bool use_bias;
    };

    struct MoEExpertProjections_t
    {
      std::unique_ptr<CAIF_DeviceLayer> gate;
      std::unique_ptr<CAIF_DeviceLayer> up;
      std::unique_ptr<CAIF_DeviceLayer> down;
    };

    CAIF_DeviceMoEExpert(const Config_t &config,CAIF_CudaStream &stream);
    CAIF_DeviceMoEExpert(const Config_t &config,
                         MoEExpertProjections_t projections,
                         CAIF_CudaStream &stream);
    ~CAIF_DeviceMoEExpert()override=default;

    CAIF_DeviceMoEExpert(CAIF_DeviceMoEExpert &&other);
    CAIF_DeviceMoEExpert &operator=(CAIF_DeviceMoEExpert &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;

    // Output-into-slice variant of ForwardImpl. Control flow + training
    // caches (`_cached_input`, `_cached_gate_out`, `_cached_up_out`,
    // `_cached_hidden`) are byte-identical to ForwardImpl; the only
    // difference is that the final down-projection writes through the
    // caller-provided `output` reference instead of into a freshly
    // allocated owning tensor.
    //
    // Used by CAIF_DeviceMoELayer's per-expert loop to write directly
    // into a slice of `_ws_expert_output_buffer`, removing one
    // cudaMallocAsync + one D2D cudaMemcpyAsync per expert from the
    // forward path (compare_multirun.py min-vs-min wins on the
    // after_forwardinto run vs post-phase4 peak: MoE Forward prod
    // training fp32 -0.4 ms, MoE Backward prod fp32 -1.3 ms).
    //
    // Caller contract:
    //   - `output` already allocated and sized exactly
    //     [num_tokens, input_dim] = [input.Shape()[0], _config.input_dim].
    //   - `output.Dtype()` == StorageDtype(). Mismatch throws.
    //   - `input` and `output` reside on the same stream as Stream();
    //     cross-stream is not supported.
    //   - `output` may alias workspace memory (the MoE layer's expert-
    //     output workspace slice) but must NOT alias `input`.
    //
    // Caveat — `_use_projections==true`: pluggable CAIF_LinearLayer
    // projections allocate their own output internally; ForwardInto
    // falls back to a single cudaMemcpyAsync into `output` in that
    // branch. The bench/perf-critical path uses the non-projection
    // constructor and avoids the copy entirely (MatMul writes straight
    // into `output`).
    void ForwardInto(const CAIF_DeviceTensor &input,
                     CAIF_DeviceTensor &output,
                     CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::MoEExpert_e;
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
    uint32_t InputDim()const override{return _config.input_dim;}
    uint32_t HiddenDim()const override{return _config.hidden_dim;}
    bool UseGated()const{return _config.use_gated;}

    // Read-side weight accessors for layer-surgery scenarios — Phase 4
    // add-MoE on a model with an existing MoE FFN (e.g. GLM layers
    // 1-46): the builder clones each existing trainable expert's
    // weights into a CAIF_DeviceMoEFrozenExpert so the originals can
    // sit alongside new trainable experts in the new MoE layer.
    // Mirrors CAIF_DeviceFFN's WGate/WUp/WDown surface that Phase 1
    // already uses for the dense-FFN swap. Only valid when
    // `_use_projections==false` (the default — pluggable-projection
    // experts hold their weights inside the projection sub-layers and
    // expose them via ParameterTensor/ParameterNames instead).
    const CAIF_DeviceTensor &WGate()const{return _w_gate;}
    const CAIF_DeviceTensor &WUp()const{return _w_up;}
    const CAIF_DeviceTensor &WDown()const{return _w_down;}

    // Move-in setters used when loading a pretrained MoE checkpoint —
    // ANVL_MoEStrategy's LoadedTrainable_e path constructs a default
    // MoEExpert via the factory, then loads each per-expert weight via
    // safetensors and hands the tensor in via these setters. Mirrors
    // CAIF_DeviceFFN's LoadWGate/LoadWUp/LoadWDown surface. Caller
    // contract: shapes are
    //   _w_gate / _w_up: [input_dim, hidden_dim]
    //   _w_down:         [hidden_dim, input_dim]
    // dtype must equal StorageDtype(); mismatch throws. Not valid when
    // `_use_projections==true`.
    void LoadWGate(CAIF_DeviceTensor &&w);
    void LoadWUp(CAIF_DeviceTensor &&w);
    void LoadWDown(CAIF_DeviceTensor &&w);

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
    // Internal accessors — single point of access for every member
    // even from inside this class's own methods. See
    // CODING_GUIDELINES.md §Member Access.
    CAIF_DeviceTensor &WGateMut(){return _w_gate;}
    CAIF_DeviceTensor &WUpMut(){return _w_up;}
    CAIF_DeviceTensor &WDownMut(){return _w_down;}
    const MoEExpertProjections_t &Projections()const{return _projections;}
    MoEExpertProjections_t &ProjectionsMut(){return _projections;}
    bool UseProjections()const{return _use_projections;}
    const CAIF_DeviceTensor &BGate()const{return _b_gate;}
    CAIF_DeviceTensor &BGateMut(){return _b_gate;}
    const CAIF_DeviceTensor &BUp()const{return _b_up;}
    CAIF_DeviceTensor &BUpMut(){return _b_up;}
    const CAIF_DeviceTensor &BDown()const{return _b_down;}
    CAIF_DeviceTensor &BDownMut(){return _b_down;}
    const CAIF_DeviceTensor &GradWGate()const{return _grad_w_gate;}
    CAIF_DeviceTensor &GradWGateMut(){return _grad_w_gate;}
    const CAIF_DeviceTensor &GradWUp()const{return _grad_w_up;}
    CAIF_DeviceTensor &GradWUpMut(){return _grad_w_up;}
    const CAIF_DeviceTensor &GradWDown()const{return _grad_w_down;}
    CAIF_DeviceTensor &GradWDownMut(){return _grad_w_down;}
    const CAIF_DeviceTensor &GradBGate()const{return _grad_b_gate;}
    CAIF_DeviceTensor &GradBGateMut(){return _grad_b_gate;}
    const CAIF_DeviceTensor &GradBUp()const{return _grad_b_up;}
    CAIF_DeviceTensor &GradBUpMut(){return _grad_b_up;}
    const CAIF_DeviceTensor &GradBDown()const{return _grad_b_down;}
    CAIF_DeviceTensor &GradBDownMut(){return _grad_b_down;}
    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    CAIF_DeviceTensor &CachedInputMut(){return _cached_input;}
    const CAIF_DeviceTensor &CachedGateOut()const{return _cached_gate_out;}
    CAIF_DeviceTensor &CachedGateOutMut(){return _cached_gate_out;}
    const CAIF_DeviceTensor &CachedUpOut()const{return _cached_up_out;}
    CAIF_DeviceTensor &CachedUpOutMut(){return _cached_up_out;}
    const CAIF_DeviceTensor &CachedHidden()const{return _cached_hidden;}
    CAIF_DeviceTensor &CachedHiddenMut(){return _cached_hidden;}
    void SetWGate(CAIF_DeviceTensor &&v){_w_gate=std::move(v);}
    void SetWUp(CAIF_DeviceTensor &&v){_w_up=std::move(v);}
    void SetWDown(CAIF_DeviceTensor &&v){_w_down=std::move(v);}
    void SetBGate(CAIF_DeviceTensor &&v){_b_gate=std::move(v);}
    void SetBUp(CAIF_DeviceTensor &&v){_b_up=std::move(v);}
    void SetBDown(CAIF_DeviceTensor &&v){_b_down=std::move(v);}
    void SetGradWGate(CAIF_DeviceTensor &&v){_grad_w_gate=std::move(v);}
    void SetGradWUp(CAIF_DeviceTensor &&v){_grad_w_up=std::move(v);}
    void SetGradWDown(CAIF_DeviceTensor &&v){_grad_w_down=std::move(v);}
    void SetGradBGate(CAIF_DeviceTensor &&v){_grad_b_gate=std::move(v);}
    void SetGradBUp(CAIF_DeviceTensor &&v){_grad_b_up=std::move(v);}
    void SetGradBDown(CAIF_DeviceTensor &&v){_grad_b_down=std::move(v);}
    void SetCachedInput(CAIF_DeviceTensor &&v){_cached_input=std::move(v);}
    void SetCachedGateOut(CAIF_DeviceTensor &&v){_cached_gate_out=std::move(v);}
    void SetCachedUpOut(CAIF_DeviceTensor &&v){_cached_up_out=std::move(v);}
    void SetCachedHidden(CAIF_DeviceTensor &&v){_cached_hidden=std::move(v);}

    Config_t _config;
    MoEExpertProjections_t _projections;
    bool _use_projections;

    CAIF_DeviceTensor _w_gate;
    CAIF_DeviceTensor _w_up;
    CAIF_DeviceTensor _w_down;

    CAIF_DeviceTensor _b_gate;
    CAIF_DeviceTensor _b_up;
    CAIF_DeviceTensor _b_down;

    CAIF_DeviceTensor _grad_w_gate;
    CAIF_DeviceTensor _grad_w_up;
    CAIF_DeviceTensor _grad_w_down;
    CAIF_DeviceTensor _grad_b_gate;
    CAIF_DeviceTensor _grad_b_up;
    CAIF_DeviceTensor _grad_b_down;

    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_gate_out;
    CAIF_DeviceTensor _cached_up_out;
    CAIF_DeviceTensor _cached_hidden;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceMoEExpert<float,float>;
extern template class CAIF_DeviceMoEExpert<float,__half>;
extern template class CAIF_DeviceMoEExpert<float,__nv_bfloat16>;
extern template class CAIF_DeviceMoEExpert<__half,float>;
extern template class CAIF_DeviceMoEExpert<__half,__half>;
extern template class CAIF_DeviceMoEExpert<__half,__nv_bfloat16>;
extern template class CAIF_DeviceMoEExpert<__nv_bfloat16,float>;
extern template class CAIF_DeviceMoEExpert<__nv_bfloat16,__half>;
extern template class CAIF_DeviceMoEExpert<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceMoEExpert<float,float>;
#endif

}//end instance namespace
