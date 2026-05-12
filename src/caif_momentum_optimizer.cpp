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

#include "caif_momentum_optimizer.h"
#include "caif_ops.h"
#include "caif_exception.h"

namespace instance
{

CAIF_MomentumOptimizer::CAIF_MomentumOptimizer(const float lr,
                                               const float momentum,
                                               const float weight_decay,
                                               CAIF_CudaStream &stream):
                                              CAIF_Optimizer(lr,weight_decay,stream),
                                              _momentum(momentum),
                                              _velocity()
{
}

void CAIF_MomentumOptimizer::AllocateState(const CAIF_DeviceTensor &param)
{
  try
  {
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    const std::vector<uint32_t> &shape=param.Shape();
    VelocitiesMut().push_back(CAIF_DeviceTensor::Zeros(shape,Stream(),fp32));
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_MomentumOptimizer::UpdateOne(CAIF_DeviceTensor &target,
                                       const CAIF_DeviceTensor &grad,
                                       const size_t idx)
{
  try
  {
    CAIF_Ops::MomentumUpdate(target,
                             grad,
                             VelocitiesMut()[idx],
                             LearningRate(),
                             Momentum(),
                             WeightDecay());
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
