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
// Configuration for CAIF_DeviceLoRAAdapter. All four fields (rank, alpha,
// input_dim, output_dim) are required by the constructor so a LoRA adapter can
// never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceLoRAAdapterConfig:public CAIF_Base
{
  public:
    // All four fields are required: the low-rank dimension, the scaling alpha,
    // and the input/output dimensions of the wrapped projection.
    CAIF_DeviceLoRAAdapterConfig(const uint32_t rank,
                                 const float alpha,
                                 const uint32_t input_dim,
                                 const uint32_t output_dim);

    // Low-rank decomposition rank.
    uint32_t Rank()const{return _rank;}
    void SetRank(const uint32_t rank){_rank=rank;}

    // LoRA scaling factor (the effective scale is alpha / rank).
    float Alpha()const{return _alpha;}
    void SetAlpha(const float alpha){_alpha=alpha;}

    // Input dimension of the wrapped projection.
    uint32_t InputDim()const{return _input_dim;}
    void SetInputDim(const uint32_t input_dim){_input_dim=input_dim;}

    // Output dimension of the wrapped projection.
    uint32_t OutputDim()const{return _output_dim;}
    void SetOutputDim(const uint32_t output_dim){_output_dim=output_dim;}

  protected:

  private:
    uint32_t _rank;
    float _alpha;
    uint32_t _input_dim;
    uint32_t _output_dim;
};

}//end instance namespace
