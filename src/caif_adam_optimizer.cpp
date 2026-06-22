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

#include "caif_adam_optimizer.h"
#include "caif_ops.h"
#include "caif_cuda_kernels_optimizers.cuh"
#include "caif_exception.h"

#include <cmath>

namespace instance
{

CAIF_AdamOptimizer::CAIF_AdamOptimizer(const float lr,
                                       const float beta1,
                                       const float beta2,
                                       const float epsilon,
                                       const float weight_decay,
                                       CAIF_CudaStream &stream):
                                                CAIF_Optimizer(lr,weight_decay,stream),
                                                _beta1(beta1),
                                                _beta2(beta2),
                                                _epsilon(epsilon),
                                                _m(),
                                                _v()
{
}

void CAIF_AdamOptimizer::AllocateState(const CAIF_DeviceTensor &param)
{
  try
  {
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    const std::vector<uint32_t> &shape=param.Shape();
    MStatesMut().push_back(CAIF_DeviceTensor::Zeros(shape,Stream(),fp32));
    VStatesMut().push_back(CAIF_DeviceTensor::Zeros(shape,Stream(),fp32));
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_AdamOptimizer::UpdateOne(CAIF_DeviceTensor &target,
                                   const CAIF_DeviceTensor &grad,
                                   const size_t idx)
{
  try
  {
    CAIF_Ops::AdamUpdate(target,
                         grad,
                         MStatesMut()[idx],
                         VStatesMut()[idx],
                         LearningRate(),
                         Beta1(),
                         Beta2(),
                         Epsilon(),
                         WeightDecay(),
                         StepCount());
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_AdamOptimizer::BatchedStep(CAIF_DeviceNetwork &network)
{
#ifdef USE_CAIF_CUDA
  try
  {
    if(BuildSharedBatch(network)==false)
    {
      return false;
    }
    const int n=BatchNumTensors();
    HostMPtrsMut().resize(static_cast<size_t>(n));
    HostVPtrsMut().resize(static_cast<size_t>(n));
    for(int idx=0;idx<n;++idx)
    {
      HostMPtrsMut()[idx]=MStatesMut()[static_cast<size_t>(idx)].DevicePtr<float>();
      HostVPtrsMut()[idx]=VStatesMut()[static_cast<size_t>(idx)].DevicePtr<float>();
    }
    UploadScratch(DeviceMPtrsMut(),HostMPtrsMut().data(),n*sizeof(float *),Stream());
    UploadScratch(DeviceVPtrsMut(),HostVPtrsMut().data(),n*sizeof(float *),Stream());

    const float t=static_cast<float>(StepCount());
    const float bias_correction1=1.0f-std::pow(Beta1(),t);
    const float bias_correction2=1.0f-std::pow(Beta2(),t);

    launch_multi_tensor_adam<float>(BatchTargetsDevice(),
                                    BatchGradsDevice(),
                                    reinterpret_cast<float *const *>(DeviceMPtrsMut().DeviceDataRaw()),
                                    reinterpret_cast<float *const *>(DeviceVPtrsMut().DeviceDataRaw()),
                                    BatchOffsetsDevice(),
                                    n,
                                    BatchTotalElements(),
                                    LearningRate(),
                                    Beta1(),
                                    Beta2(),
                                    Epsilon(),
                                    WeightDecay(),
                                    bias_correction1,
                                    bias_correction2,
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
