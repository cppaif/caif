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
// CAIF_DevicePointwiseActivation — abstract middle base for element-wise
// activations (ReLU, GELU, Sigmoid, Tanh, Swish, LeakyReLU, ELU, Linear).
// IsGated() returns false. FFN uses 2 weight matrices with these.
// Each leaf class lives in its own header (caif_device_<name>_activation.h).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_activation.h"
#include "caif_device_tensor.h"

namespace instance
{

class CAIF_DevicePointwiseActivation:public CAIF_DeviceActivation
{
  public:
    bool IsGated()const override{return false;}

    // Apply activation element-wise: output = f(input)
    virtual void Forward(const CAIF_DeviceTensor &input,
                         CAIF_DeviceTensor &output)const=0;

    // Compute activation gradient.
    // pre_activation = input to the activation (before activation was applied).
    // post_activation = output of the activation (after activation was applied).
    // Not all activations need both; subclasses use what they need.
    // grad_input = grad_output * f'(pre_activation)
    virtual void Backward(const CAIF_DeviceTensor &grad_output,
                          const CAIF_DeviceTensor &pre_activation,
                          const CAIF_DeviceTensor &post_activation,
                          CAIF_DeviceTensor &grad_input)const=0;

    // Whether backward needs the post_activation tensor cached.
    // Default false. Overridden by Sigmoid, Tanh, ELU, Swish.
    virtual bool NeedsPostActivation()const{return false;}

  protected:
    CAIF_DevicePointwiseActivation()=default;

  private:
};

}//end instance namespace
