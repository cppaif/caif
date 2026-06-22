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
// Abstract base class for MoE experts.
//
// Adds the `ForwardInto(input, output, ctx)` write-into-slice entry point
// to the layer interface so CAIF_DeviceMoELayer can hold a heterogeneous
// vector of trainable + frozen experts polymorphically:
//
//   std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<C, S>>>
//
// Two concrete derivations:
//
//   CAIF_DeviceMoEExpert<C, S>           — trainable expert; weights live
//                                          at StorageT, compute at ComputeT.
//                                          Its 9-cell <C, S> grid covers
//                                          {fp32, fp16, bf16}^2.
//
//   CAIF_DeviceMoEFrozenExpert<C, S>     — frozen expert built from 3
//                                          FrozenLinear sub-layers (gate,
//                                          up, down). FrozenLinear's own
//                                          15-cell grid (5 storage x 3
//                                          compute) lets the frozen expert
//                                          carry int4/int8 weights even
//                                          when the surrounding layer's
//                                          StorageT is fp16/bf16. The
//                                          wrapper handles the
//                                          dequantize-and-cast at the
//                                          FrozenLinear output before the
//                                          gated activation.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_tensor.h"
#include "caif_run_context.h"

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceMoEExpertBase:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    explicit CAIF_DeviceMoEExpertBase(CAIF_CudaStream &stream)
      :CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream){}
    ~CAIF_DeviceMoEExpertBase()override=default;

    CAIF_DeviceMoEExpertBase(const CAIF_DeviceMoEExpertBase &)=delete;
    CAIF_DeviceMoEExpertBase &operator=(const CAIF_DeviceMoEExpertBase &)=delete;
    CAIF_DeviceMoEExpertBase(CAIF_DeviceMoEExpertBase &&other)
      :CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)){}
    CAIF_DeviceMoEExpertBase &operator=(CAIF_DeviceMoEExpertBase &&other)
    {
      if(this!=&other)
      {
        CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      }
      return *this;
    }

    // Output-into-slice variant of ForwardImpl. CAIF_DeviceMoELayer's
    // per-expert loop calls this with `output` = a non-owning view into
    // a packed expert-output workspace slice. Caller contract:
    //   - `output` already allocated and sized exactly
    //     [num_tokens, input_dim] = [input.Shape()[0], InputDim()].
    //   - `output.Dtype()` == StorageDtype(). Mismatch throws.
    //   - `input` and `output` reside on the same stream as Stream();
    //     cross-stream is not supported.
    //   - `output` may alias workspace memory but must NOT alias `input`.
    virtual void ForwardInto(const CAIF_DeviceTensor &input,
                             CAIF_DeviceTensor &output,
                             CAIF_RunContext &ctx)=0;

    // Width along which the layer routes — the dispatch-buffer's
    // hidden-dim. `ForwardInto`'s `output` shape is
    // [num_tokens, InputDim()].
    virtual uint32_t InputDim()const=0;

    // Per-expert FFN intermediate width. The trainable expert reads
    // its config; the frozen expert reads it from its gate/up
    // FrozenLinear. The MoE layer uses this when sizing per-expert
    // intermediate VRAM in Tier I.1 (8.5.D — mixed-size experts).
    virtual uint32_t HiddenDim()const=0;
};

}//end instance namespace
