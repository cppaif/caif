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
// Configuration for CAIF_DeviceMoEFrozenExpert. All three fields (input_dim,
// hidden_dim, use_gated) are required by the constructor so a frozen expert can
// never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceMoEFrozenExpertConfig:public CAIF_Base
{
  public:
    // All three fields are required: the input/hidden dimensions and whether
    // the expert uses a gated (SwiGLU-style) FFN.
    CAIF_DeviceMoEFrozenExpertConfig(const uint32_t input_dim,
                                     const uint32_t hidden_dim,
                                     const bool use_gated);

    // Input/output dimension of the expert FFN.
    uint32_t InputDim()const{return _input_dim;}
    void SetInputDim(const uint32_t input_dim){_input_dim=input_dim;}

    // Hidden (intermediate) dimension of the expert FFN.
    uint32_t HiddenDim()const{return _hidden_dim;}
    void SetHiddenDim(const uint32_t hidden_dim){_hidden_dim=hidden_dim;}

    // Use a gated (SwiGLU-style) FFN.
    bool UseGated()const{return _use_gated;}
    void SetUseGated(const bool use_gated){_use_gated=use_gated;}

  protected:

  private:
    uint32_t _input_dim;
    uint32_t _hidden_dim;
    bool _use_gated;
};

}//end instance namespace
