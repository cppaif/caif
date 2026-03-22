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
// AIF - AI Framework
// Device activation strategy hierarchy: base + pointwise + gated abstractions
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_ACTIVATION_H
#define CAIF_DEVICE_ACTIVATION_H

#include "caif_device_tensor.h"
#include <memory>
#include <string>

namespace instance
{

//------------------------------------------------------------------------------
// CAIF_DeviceActivation -- abstract base for all activation strategies
//------------------------------------------------------------------------------
class CAIF_DeviceActivation
{
  public:
    virtual ~CAIF_DeviceActivation()=default;

    virtual bool IsGated()const=0;
    virtual std::string Description()const=0;

    // Clone for polymorphic copy (FFN needs to own its activation)
    virtual std::unique_ptr<CAIF_DeviceActivation> Clone()const=0;

  protected:
    CAIF_DeviceActivation()=default;

    // Non-copyable, non-movable (use Clone for polymorphic copy)
    CAIF_DeviceActivation(const CAIF_DeviceActivation &)=default;
    CAIF_DeviceActivation &operator=(const CAIF_DeviceActivation &)=default;

  private:
};

//------------------------------------------------------------------------------
// CAIF_DevicePointwiseActivation -- abstract middle class for element-wise
// activations (ReLU, GELU, Sigmoid, Tanh, Swish, LeakyReLU, ELU, Linear).
// IsGated() returns false. FFN uses 2 weight matrices with these.
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// CAIF_DeviceGatedActivation -- abstract middle class for gated activations
// (SwiGLU, GeGLU, ReGLU, GLU, Bilinear).
// IsGated() returns true. FFN uses 3 weight matrices with these.
//------------------------------------------------------------------------------
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

#endif  // CAIF_DEVICE_ACTIVATION_H
