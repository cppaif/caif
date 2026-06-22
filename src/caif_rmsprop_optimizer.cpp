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

#include "caif_rmsprop_optimizer.h"
#include "caif_ops.h"
#include "caif_cuda_kernels_optimizers.cuh"
#include "caif_exception.h"

namespace instance
{

CAIF_RmspropOptimizer::CAIF_RmspropOptimizer(const float lr,
                                             const float alpha,
                                             const float epsilon,
                                             const float weight_decay,
                                             CAIF_CudaStream &stream):
                                              CAIF_Optimizer(lr,weight_decay,stream),
                                              _alpha(alpha),
                                              _epsilon(epsilon),
                                              _avg_sq()
{
}

void CAIF_RmspropOptimizer::AllocateState(const CAIF_DeviceTensor &param)
{
  try
  {
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    const std::vector<uint32_t> &shape=param.Shape();
    AvgSqsMut().push_back(CAIF_DeviceTensor::Zeros(shape,Stream(),fp32));
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_RmspropOptimizer::UpdateOne(CAIF_DeviceTensor &target,
                                      const CAIF_DeviceTensor &grad,
                                      const size_t idx)
{
  try
  {
    CAIF_Ops::RmspropUpdate(target,
                            grad,
                            AvgSqsMut()[idx],
                            LearningRate(),
                            Alpha(),
                            Epsilon(),
                            WeightDecay());
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_RmspropOptimizer::BatchedStep(CAIF_DeviceNetwork &network)
{
#ifdef USE_CAIF_CUDA
  try
  {
    if(BuildSharedBatch(network)==false)
    {
      return false;
    }
    const int n=BatchNumTensors();
    HostAvgSqPtrsMut().resize(static_cast<size_t>(n));
    for(int idx=0;idx<n;++idx)
    {
      HostAvgSqPtrsMut()[idx]=AvgSqsMut()[static_cast<size_t>(idx)].DevicePtr<float>();
    }
    UploadScratch(DeviceAvgSqPtrsMut(),
                  HostAvgSqPtrsMut().data(),
                  n*sizeof(float *),
                  Stream());
    launch_multi_tensor_rmsprop<float>(BatchTargetsDevice(),
                                       BatchGradsDevice(),
                                       reinterpret_cast<float *const *>(DeviceAvgSqPtrsMut().DeviceDataRaw()),
                                       BatchOffsetsDevice(),
                                       n,
                                       BatchTotalElements(),
                                       LearningRate(),
                                       Alpha(),
                                       Epsilon(),
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
