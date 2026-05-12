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

#pragma once

#include "caif_optimizer.h"

namespace instance
{

class CAIF_SgdOptimizer:public CAIF_Optimizer
{
  public:
    CAIF_SgdOptimizer(const float lr,
                      const float weight_decay,
                      CAIF_CudaStream &stream);
    ~CAIF_SgdOptimizer()override=default;

    CAIF_OptimizerType_e Type()const override
    {
      return CAIF_OptimizerType_e::SGD;
    }

  protected:
    // Plain SGD has no per-parameter state; AllocateState is a no-op.
    void AllocateState(const CAIF_DeviceTensor &param)override;
    void UpdateOne(CAIF_DeviceTensor &target,
                   const CAIF_DeviceTensor &grad,
                   const size_t idx)override;
};

}//end instance namespace
