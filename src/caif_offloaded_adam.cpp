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

#include "caif_offloaded_adam.h"
#include "caif_ops.h"
#include "caif_exception.h"

#include <cstring>

namespace instance
{

CAIF_OffloadedAdam::CAIF_OffloadedAdam(const float lr,
                                       const float beta1,
                                       const float beta2,
                                       const float epsilon,
                                       const float weight_decay,
                                       CAIF_CudaStream &stream):
                                                CAIF_Optimizer(lr,weight_decay,stream),
                                                _beta1(beta1),
                                                _beta2(beta2),
                                                _epsilon(epsilon),
                                                _host_m(),
                                                _host_v()
{
}

void CAIF_OffloadedAdam::AllocateState(const CAIF_DeviceTensor &param)
{
  try
  {
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    const std::vector<uint32_t> &shape=param.Shape();

    std::unique_ptr<CAIF_HostPinnedTensor> host_m(new CAIF_HostPinnedTensor(shape,fp32));
    std::unique_ptr<CAIF_HostPinnedTensor> host_v(new CAIF_HostPinnedTensor(shape,fp32));

    std::memset(host_m->HostPtr(),0,host_m->Bytes());
    std::memset(host_v->HostPtr(),0,host_v->Bytes());

    HostMMut().push_back(std::move(host_m));
    HostVMut().push_back(std::move(host_v));
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_OffloadedAdam::UpdateOne(CAIF_DeviceTensor &target,
                                   const CAIF_DeviceTensor &grad,
                                   const size_t idx)
{
  try
  {
    if(idx>=HostM().size())
    {
      THROW_CAIFE("CAIF_OffloadedAdam::UpdateOne: idx out of range");
    }

    CAIF_DeviceTensor m_dev=HostM()[idx]->PrefetchToDevice(Stream());
    CAIF_DeviceTensor v_dev=HostV()[idx]->PrefetchToDevice(Stream());

    CAIF_Ops::AdamUpdate(target,
                         grad,
                         m_dev,
                         v_dev,
                         LearningRate(),
                         Beta1(),
                         Beta2(),
                         Epsilon(),
                         WeightDecay(),
                         StepCount());

    HostMMut()[idx]->CopyFromDevice(m_dev);
    HostVMut()[idx]->CopyFromDevice(v_dev);
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace