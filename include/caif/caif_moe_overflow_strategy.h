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
// CAIF_MoEOverflowStrategy — closed-value enumeration for the policy applied to
// token-to-expert assignments that exceed a finite per-expert capacity on the
// MoE GPU dispatch path. Lives outside the templated CAIF_DeviceMoELayer so
// every <ComputeT, StorageT> instantiation, and the configs / composers that
// carry the value, share one enum type instead of a per-instantiation one.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_MoEOverflowStrategy:public CAIF_Base
{
  public:
    // Capacity, when finite, is
    //   capacity = ceil(num_tokens / num_experts * capacity_factor * top_k)
    // (the GShard/Switch formula); see CAIF_DeviceMoELayer::ForwardImpl for
    // where it is applied. ForwardImpl accepts only Drop and NoDrop (NoOp /
    // Redistribute throw); a non-positive capacity_factor is treated as no-drop
    // regardless of the strategy.
    //
    //   Drop          GShard/Switch-style capacity dropping. Assignments past an
    //                 expert's capacity are discarded; the combine omits the
    //                 dropped term and does NOT renormalize the surviving gate
    //                 weights (the GShard/Switch convention), so a token that
    //                 loses one of its top_k experts yields a smaller-magnitude
    //                 MoE branch. Intended for training-throughput-bounded runs
    //                 that deliberately cap per-expert load. This is NOT HF
    //                 parity: HF DeepSeek-V2 / Mixtral / GLM-MoE never drop.
    //   NoDrop        No capacity limit. Every token reaches all of its top_k
    //                 experts regardless of routing imbalance (dispatch runs
    //                 with capacity 0, the kernel's "unlimited" sentinel), so
    //                 nothing is dropped and no renormalization is needed. This
    //                 is the HF-parity path and the correct choice for inference
    //                 and for add-MoE cold-start, where a freshly grafted router
    //                 is maximally imbalanced and Drop would silently discard
    //                 tokens. capacity_factor is ignored under NoDrop.
    //   NoOp          Reserved; not implemented on the GPU dispatch path.
    //   Redistribute  Reserved; not implemented on the GPU dispatch path.
    enum class CAIF_MoEOverflowStrategy_e:uint8_t
    {
      Drop=0,
      NoOp=1,
      Redistribute=2,
      NoDrop=3
    };
};

}//end instance namespace
