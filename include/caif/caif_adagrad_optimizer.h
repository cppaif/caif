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
#include "caif_optimizer_type.h"

#include <vector>

namespace instance
{

class CAIF_AdaGradOptimizer:public CAIF_Optimizer
{
  public:
    CAIF_AdaGradOptimizer(const float lr,
                          const float epsilon,
                          const float weight_decay,
                          CAIF_CudaStream &stream);
    ~CAIF_AdaGradOptimizer()override=default;

    CAIF_OptimizerType::CAIF_OptimizerType_e Type()const override
    {
      return CAIF_OptimizerType::CAIF_OptimizerType_e::AdaGrad;
    }

  protected:
    void AllocateState(const CAIF_DeviceTensor &param)override;
    void UpdateOne(CAIF_DeviceTensor &target,
                   const CAIF_DeviceTensor &grad,
                   const size_t idx)override;
    bool BatchedStep(CAIF_DeviceNetwork &network)override;

  private:
    float Epsilon()const{return _epsilon;}
    const std::vector<CAIF_DeviceTensor> &Accums()const{return _accum;}
    std::vector<CAIF_DeviceTensor> &AccumsMut(){return _accum;}
    std::vector<float *> &HostAccumPtrsMut(){return _h_accum_ptrs;}
    CAIF_DeviceTensor &DeviceAccumPtrsMut(){return _d_accum_ptrs;}

    float _epsilon;
    std::vector<CAIF_DeviceTensor> _accum;

    // Batched-step scratch: host staging + device buffer of per-tensor accum
    // pointers, in trainable order. Filled by BatchedStep.
    std::vector<float *> _h_accum_ptrs;
    CAIF_DeviceTensor _d_accum_ptrs;
};

}//end instance namespace
