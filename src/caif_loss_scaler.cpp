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

#include "caif_loss_scaler.h"
#include "caif_device_network.h"
#include "caif_device_tensor.h"
#include "caif_ops.h"
#include "caif_data_type.h"
#include "caif_exception.h"

namespace instance
{

CAIF_LossScaler::CAIF_LossScaler(CAIF_CudaStream &stream,
                                 float init_scale,
                                 float growth_factor,
                                 float backoff_factor,
                                 uint32_t growth_interval):_scale(init_scale),
                                                           _growth_factor(growth_factor),
                                                           _backoff_factor(backoff_factor),
                                                           _growth_interval(growth_interval),
                                                           _growth_tracker(0),
                                                           _stream(&stream)
{
}

void CAIF_LossScaler::ScaleLossGrad(CAIF_DeviceTensor &loss_grad)
{
  try
  {
    CAIF_Ops::Scale(loss_grad,Scale());
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_LossScaler::Step(CAIF_DeviceNetwork &network)
{
  try
  {
    CAIF_DeviceTensor found_inf=CAIF_DeviceTensor::Zeros({1},Stream(),CAIF_DataType::CAIF_DataType_e::Float32);
    const float inv_scale=1.0f/Scale();
    network.UnscaleGradsCheckInf(inv_scale,found_inf);

    float flag=0.0f;
    found_inf.CopyToHost(&flag);
    const bool overflow=(flag!=0.0f);

    if(overflow==false)
    {
      network.OptimizerStep();
    }
    UpdateScale(overflow);
    return overflow==false;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_LossScaler::UpdateScale(const bool overflow)
{
  if(overflow==true)
  {
    SetScale(Scale()*BackoffFactor());
    SetGrowthTracker(0);
    return;
  }
  SetGrowthTracker(GrowthTracker()+1);
  if(GrowthTracker()>=GrowthInterval())
  {
    SetScale(Scale()*GrowthFactor());
    SetGrowthTracker(0);
  }
}

}//end instance namespace
