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

#include <vector>

namespace instance
{

class CAIF_RmspropOptimizer:public CAIF_Optimizer
{
  public:
    CAIF_RmspropOptimizer(const float lr,
                          const float alpha,
                          const float epsilon,
                          const float weight_decay,
                          CAIF_CudaStream &stream);
    ~CAIF_RmspropOptimizer()override=default;

    CAIF_OptimizerType_e Type()const override
    {
      return CAIF_OptimizerType_e::RMSprop;
    }

  protected:
    void AllocateState(const CAIF_DeviceTensor &param)override;
    void UpdateOne(CAIF_DeviceTensor &target,
                   const CAIF_DeviceTensor &grad,
                   const size_t idx)override;

  private:
    float Alpha()const{return _alpha;}
    float Epsilon()const{return _epsilon;}
    const std::vector<CAIF_DeviceTensor> &AvgSqs()const{return _avg_sq;}
    std::vector<CAIF_DeviceTensor> &AvgSqsMut(){return _avg_sq;}

    float _alpha;
    float _epsilon;
    std::vector<CAIF_DeviceTensor> _avg_sq;
};

}//end instance namespace
