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

class CAIF_AdamOptimizer:public CAIF_Optimizer
{
  public:
    CAIF_AdamOptimizer(const float lr,
                       const float beta1,
                       const float beta2,
                       const float epsilon,
                       const float weight_decay,
                       CAIF_CudaStream &stream);
    ~CAIF_AdamOptimizer()override=default;

    CAIF_OptimizerType_e Type()const override
    {
      return CAIF_OptimizerType_e::Adam;
    }

  protected:
    void AllocateState(const CAIF_DeviceTensor &param)override;
    void UpdateOne(CAIF_DeviceTensor &target,
                   const CAIF_DeviceTensor &grad,
                   const size_t idx)override;

  private:
    float Beta1()const{return _beta1;}
    float Beta2()const{return _beta2;}
    float Epsilon()const{return _epsilon;}
    const std::vector<CAIF_DeviceTensor> &MStates()const{return _m;}
    std::vector<CAIF_DeviceTensor> &MStatesMut(){return _m;}
    const std::vector<CAIF_DeviceTensor> &VStates()const{return _v;}
    std::vector<CAIF_DeviceTensor> &VStatesMut(){return _v;}

    float _beta1;
    float _beta2;
    float _epsilon;

    // First and second moment estimates per trainable parameter.  Always
    // fp32 regardless of the param's native dtype (matches the AMP
    // master-weight rule in CAIF_Optimizer).
    std::vector<CAIF_DeviceTensor> _m;
    std::vector<CAIF_DeviceTensor> _v;
};

}//end instance namespace
