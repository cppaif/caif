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
// CAIF_DevicePreNormBlock<ComputeT, StorageT> — generic pre-norm residual
// block. Composes an arbitrary number of (norm, layer) pairs with residual
// connections. Each stage applies:
//
//   x = x + layer(norm(x))
//
// Every stage requires BOTH norm and layer — no nullptr. Sublayers are
// stored flat in the un-templated CAIF_DeviceContainer base's _sublayers
// vector as interleaved pairs [norm0, layer0, norm1, layer1, ...].
//
// The block templates on <ComputeT, StorageT> for the residual buffer
// allocation; the polymorphic sublayers may carry their own <C', S'> cells.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_container.h"
#include "caif_device_layer_typed.h"
#include "caif_block_offload_scheduler.h"
#include "caif_run_context.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DevicePreNormBlock:public CAIF_DeviceContainer
{
  public:
    typedef CAIF_DeviceLayerTyped<ComputeT,StorageT> Typed_t;

    struct SubLayer_t
    {
      std::string norm_prefix;
      std::string layer_prefix;
      std::unique_ptr<CAIF_DeviceLayer> norm;
      std::unique_ptr<CAIF_DeviceLayer> layer;
    };

    typedef std::vector<SubLayer_t> SubLayerVec_t;

    CAIF_DevicePreNormBlock(SubLayerVec_t sub_layers,CAIF_CudaStream &stream);
    ~CAIF_DevicePreNormBlock()override=default;

    CAIF_DevicePreNormBlock(CAIF_DevicePreNormBlock &&other);
    CAIF_DevicePreNormBlock &operator=(CAIF_DevicePreNormBlock &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::PreNormBlock_e;
    }
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;
    size_t FrozenTensorCount()const override;
    CAIF_DeviceTensor FrozenTensorFP32(size_t index)const override;
    std::vector<std::string> FrozenTensorNames(const std::string &prefix="")const override;

    size_t SubLayerCount()const{return _sublayers.size()/g_caif_prenorm_stage_stride;}

    void SetNormsTrainable(bool trainable);
    bool NormsTrainable()const{return _norms_trainable;}

    // Activation (gradient) checkpointing.
    //
    // When `_checkpointed` is true and `ctx.Training()` is true on entry
    // to ForwardImpl, the block:
    //   - clones the block input into `_saved_input`,
    //   - flips `ctx.Training()` to false for the forward sweep — every
    //     saving sublayer (MHA, MLA, FFN, MoE, norms) ALREADY gates its
    //     internal `_cached_*` writes on `ctx.Training()`, so caching is
    //     skipped for free,
    //   - restores `ctx.Training()` to true on the way out.
    //
    // BackwardImpl, before running the real backward sweep, re-runs the
    // forward loop on `_saved_input` with `ctx.Training()` true so each
    // sublayer's caches are repopulated, then runs Backward as usual and
    // releases `_saved_input`.
    //
    // Memory win: only one block's activations live at a time during
    // backward instead of all N blocks' simultaneously during forward.
    // Compute cost: one extra forward pass per block per training step.
    // Composes with DP / TP / PP / ZeRO without modification — purely
    // per-device.
    void SetCheckpointed(const bool b)override{_checkpointed=b;}
    bool Checkpointed()const override{return _checkpointed;}

    // Saved-input accessors used by the checkpoint forward / backward
    // pair. The block clones the forward input on entry to the
    // checkpoint branch and consumes it on entry to backward; outside
    // those two calls the saved tensor stays empty.
    const CAIF_DeviceTensor &SavedInput()const{return _saved_input;}
    void SetSavedInput(CAIF_DeviceTensor t){_saved_input=std::move(t);}
    bool HasSavedInput()const{return _saved_input.IsAllocated();}
    void ClearSavedInput(){_saved_input=CAIF_DeviceTensor();}

    // CPU offload — block-level scheduler that drives Prefetch / Evict on
    // offload-aware sublayers (CAIF_DeviceFrozenLinearBase). The block
    // creates the scheduler on demand. ForwardLoop / BackwardLoop call
    // OnEnter/Exit hooks per stage. Callers register offloadable sublayers
    // (e.g. frozen-experts inside MoE) via `OffloadSchedulerMut().RegisterAtStage`.
    CAIF_BlockOffloadScheduler &OffloadSchedulerMut();
    bool HasOffloadScheduler()const{return _offload_scheduler!=nullptr;}

    static constexpr CAIF_DataType::CAIF_DataType_e ComputeDtype()
    {
      return CAIF_StorageDtype_t<ComputeT>::Value;
    }
    static constexpr CAIF_DataType::CAIF_DataType_e StorageDtype()
    {
      return CAIF_StorageDtype_t<StorageT>::Value;
    }

  protected:

  private:
    // Forward / backward bodies factored out of ForwardImpl /
    // BackwardImpl so the checkpoint-recompute path can re-run the
    // forward loop without re-entering the checkpoint-mode branch.
    CAIF_DeviceTensor ForwardLoop(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx);
    CAIF_DeviceTensor BackwardLoop(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx);

    // Internal accessors — keep method bodies free of direct _member access.
    const std::vector<std::string> &NormPrefixes()const{return _norm_prefixes;}
    std::vector<std::string> &NormPrefixes(){return _norm_prefixes;}
    void SetNormPrefixes(std::vector<std::string> &&v){_norm_prefixes=std::move(v);}
    const std::vector<std::string> &LayerPrefixes()const{return _layer_prefixes;}
    std::vector<std::string> &LayerPrefixes(){return _layer_prefixes;}
    void SetLayerPrefixes(std::vector<std::string> &&v){_layer_prefixes=std::move(v);}
    CAIF_DeviceTensor &SavedInput(){return _saved_input;}

    std::vector<std::string> _norm_prefixes;
    std::vector<std::string> _layer_prefixes;
    bool _norms_trainable;
    bool _checkpointed;
    CAIF_DeviceTensor _saved_input;
    std::unique_ptr<CAIF_BlockOffloadScheduler> _offload_scheduler;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DevicePreNormBlock<float,float>;
extern template class CAIF_DevicePreNormBlock<float,__half>;
extern template class CAIF_DevicePreNormBlock<float,__nv_bfloat16>;
extern template class CAIF_DevicePreNormBlock<__half,float>;
extern template class CAIF_DevicePreNormBlock<__half,__half>;
extern template class CAIF_DevicePreNormBlock<__half,__nv_bfloat16>;
extern template class CAIF_DevicePreNormBlock<__nv_bfloat16,float>;
extern template class CAIF_DevicePreNormBlock<__nv_bfloat16,__half>;
extern template class CAIF_DevicePreNormBlock<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DevicePreNormBlock<float,float>;
#endif

}//end instance namespace
