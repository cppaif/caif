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
// Configuration for CAIF_DeviceLinearHead. input_dim, output_dim and use_bias
// are required by the constructor so a head can never be built half-configured.
// output_scale carries a documented no-op default (1.0, set in the
// constructor's initializer list, not as an in-class initializer) and is
// adjusted through its setter.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceLinearHeadConfig:public CAIF_Base
{
  public:
    // input_dim, output_dim and use_bias are the required, projection-defining
    // fields. output_scale defaults to 1.0 (no-op) and is set via
    // SetOutputScale().
    CAIF_DeviceLinearHeadConfig(const uint32_t input_dim,const uint32_t output_dim,const bool use_bias);

    // Input (projection source) dimension.
    uint32_t InputDim()const{return _input_dim;}
    void SetInputDim(const uint32_t input_dim){_input_dim=input_dim;}

    // Output (projection target / logits) dimension.
    uint32_t OutputDim()const{return _output_dim;}
    void SetOutputDim(const uint32_t output_dim){_output_dim=output_dim;}

    // Add a learnable bias term to the projection.
    bool UseBias()const{return _use_bias;}
    void SetUseBias(const bool use_bias){_use_bias=use_bias;}

    // Multiplies the head output / logits (and scales the gradient back on the
    // backward path). 1.0 = no-op; used for logit-scaled archs (F6).
    float OutputScale()const{return _output_scale;}
    void SetOutputScale(const float output_scale){_output_scale=output_scale;}

  protected:

  private:
    uint32_t _input_dim;
    uint32_t _output_dim;
    bool _use_bias;
    float _output_scale;
};

}//end instance namespace
