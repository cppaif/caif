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
// CAIF_DeviceGatedActivation — abstract middle base for gated activations
// (SwiGLU, GeGLU, ReGLU, GLU, Bilinear). IsGated() returns true. FFN uses 3
// weight matrices with these. Each leaf class lives in its own header
// (caif_device_<name>_activation.h).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_activation.h"
#include "caif_device_tensor.h"

namespace instance
{

class CAIF_DeviceGatedActivation:public CAIF_DeviceActivation
{
  public:
    bool IsGated()const override{return true;}

    // Forward: output = gate_activation(gate_input) * up_input
    virtual void Forward(const CAIF_DeviceTensor &gate_input,
                         const CAIF_DeviceTensor &up_input,
                         CAIF_DeviceTensor &output)const=0;

    // Backward: compute gradients w.r.t. gate_input and up_input.
    virtual void Backward(const CAIF_DeviceTensor &grad_output,
                          const CAIF_DeviceTensor &cached_gate_input,
                          const CAIF_DeviceTensor &cached_up_input,
                          CAIF_DeviceTensor &grad_gate,
                          CAIF_DeviceTensor &grad_up)const=0;

  protected:
    CAIF_DeviceGatedActivation()=default;

  private:
};

}//end instance namespace
