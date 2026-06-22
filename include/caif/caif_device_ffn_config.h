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
// Configuration for CAIF_DeviceFFN. Both fields (dim, ffn_dim) are required by
// the constructor so an FFN can never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceFFNConfig:public CAIF_Base
{
  public:
    // Both fields are required: the model dimension and the FFN hidden dim.
    CAIF_DeviceFFNConfig(const uint32_t dim,const uint32_t ffn_dim);

    // Model (input/output) dimension.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // FFN hidden (intermediate) dimension.
    uint32_t FfnDim()const{return _ffn_dim;}
    void SetFfnDim(const uint32_t ffn_dim){_ffn_dim=ffn_dim;}

  protected:

  private:
    uint32_t _dim;
    uint32_t _ffn_dim;
};

}//end instance namespace
