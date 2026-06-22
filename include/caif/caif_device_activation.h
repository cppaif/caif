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
// CAIF_DeviceActivation — abstract base for all activation strategies.
// Pointwise and gated middle bases live in their own headers
// (caif_device_pointwise_activation.h, caif_device_gated_activation.h).
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"

#include <memory>
#include <string>

namespace instance
{

class CAIF_DeviceActivation:public CAIF_Base
{
  public:
    // Closed enumeration of the element-wise activation functions a dense layer
    // selects by value. (The polymorphic strategy hierarchy in this same file —
    // pointwise / gated middles — is the runtime object form; this enum is the
    // selector a CAIF_DeviceDenseLayer / its factory keys on.)
    enum class CAIF_DeviceActivation_e
    {
      None,
      ReLU,
      Sigmoid,
      Tanh,
      Softmax,
      LeakyReLU,
      ELU,
      GELU,
      Swish,
      GELUExact
    };

    virtual ~CAIF_DeviceActivation()=default;

    virtual bool IsGated()const=0;
    virtual std::string Description()const=0;

    // Clone for polymorphic copy (FFN needs to own its activation)
    virtual std::unique_ptr<CAIF_DeviceActivation> Clone()const=0;

  protected:
    CAIF_DeviceActivation()=default;

    // Non-copyable / non-movable (use Clone for polymorphic copy)
    CAIF_DeviceActivation(const CAIF_DeviceActivation &)=default;
    CAIF_DeviceActivation &operator=(const CAIF_DeviceActivation &)=default;

  private:
};

}//end instance namespace
