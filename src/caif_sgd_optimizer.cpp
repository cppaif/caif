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

#include "caif_sgd_optimizer.h"
#include "caif_ops.h"
#include "caif_cuda_kernels_optimizers.cuh"
#include "caif_exception.h"

namespace instance
{

CAIF_SgdOptimizer::CAIF_SgdOptimizer(const float lr,
                                     const float weight_decay,
                                     CAIF_CudaStream &stream):
                                              CAIF_Optimizer(lr,weight_decay,stream)
{
}

void CAIF_SgdOptimizer::AllocateState(const CAIF_DeviceTensor &param)
{
  (void)param;// no per-parameter state
}

void CAIF_SgdOptimizer::UpdateOne(CAIF_DeviceTensor &target,
                                  const CAIF_DeviceTensor &grad,
                                  const size_t idx)
{
  try
  {
    (void)idx;
    CAIF_Ops::SgdUpdate(target,grad,LearningRate(),WeightDecay());
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_SgdOptimizer::BatchedStep(CAIF_DeviceNetwork &network)
{
#ifdef USE_CAIF_CUDA
  try
  {
    if(BuildSharedBatch(network)==false)
    {
      return false;
    }
    launch_multi_tensor_sgd<float>(BatchTargetsDevice(),
                                   BatchGradsDevice(),
                                   BatchOffsetsDevice(),
                                   BatchNumTensors(),
                                   BatchTotalElements(),
                                   LearningRate(),
                                   WeightDecay(),
                                   Stream().Handle());
    return true;
  }
  CAIF_CATCH_BLOCK()
#else
  (void)network;
  return false;
#endif
}

}//end instance namespace
