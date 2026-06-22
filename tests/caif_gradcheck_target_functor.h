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
// Forward-wrapper interface for CAIF_GradCheck. A test supplies a concrete
// functor whose ForwardOnly(...) runs the layer's forward and returns the
// perturbed output; CAIF_GradCheck drives it for the finite-difference
// directional-derivative check.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_tensor.h"
#include "caif_run_context.h"

namespace instance
{

class CAIF_GradCheckTargetFunctor
{
  public:
    virtual ~CAIF_GradCheckTargetFunctor()=default;
    virtual CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                          CAIF_RunContext &ctx)=0;
  protected:
  private:
};

}//end instance namespace
