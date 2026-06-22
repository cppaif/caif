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

#include "caif_base.h"
#include "caif_constants.h"
#include "caif_cuda_stream.h"

#include <cstdint>

namespace instance
{

class CAIF_DeviceNetwork;
class CAIF_DeviceTensor;

// Dynamic mixed-precision loss scaler — the CAIF equivalent of
// torch.cuda.amp.GradScaler. fp16 training underflows tiny gradients to zero;
// scaling the loss up by a large factor before Backward() keeps them
// representable, and the optimizer step unscales them back. The scale is
// adaptive: it grows while steps stay healthy and backs off the moment any
// gradient overflows to inf/nan (the scale was too large), and that
// overflowing step is skipped so a poisoned update never lands.
//
// Per training step:
//   scaler.ScaleLossGrad(loss_grad);   // before network.Backward(loss_grad)
//   network.Backward(loss_grad);
//   scaler.Step(network);              // unscale + overflow-check + (conditional
//                                      // OptimizerStep) + adaptive scale update
//
// The scaler and the network must share a CUDA stream (the overflow flag and
// the grads are read/written on it).
class CAIF_LossScaler:public CAIF_Base
{
  public:
    CAIF_LossScaler(CAIF_CudaStream &stream,
                    float init_scale=g_caif_loss_scaler_init_scale,
                    float growth_factor=g_caif_loss_scaler_growth_factor,
                    float backoff_factor=g_caif_loss_scaler_backoff_factor,
                    uint32_t growth_interval=g_caif_loss_scaler_growth_interval);
    ~CAIF_LossScaler()override=default;

    // Current scale factor.
    float Scale()const{return _scale;}

    // Scale the loss-gradient seed up by Scale() in place, before Backward().
    // Equivalent to scaling the loss itself: every gradient in the graph then
    // comes out multiplied by Scale().
    void ScaleLossGrad(CAIF_DeviceTensor &loss_grad);

    // Unscale every trainable gradient, check for overflow, run the optimizer
    // step ONLY if nothing overflowed, then adapt the scale (grow after
    // growth_interval clean steps, halve on overflow). Returns true if the step
    // was applied, false if it was skipped because of overflow.
    bool Step(CAIF_DeviceNetwork &network);

    // Number of consecutive overflow-free steps required before the scale grows.
    uint32_t GrowthInterval()const{return _growth_interval;}

  protected:

  private:
    float GrowthFactor()const{return _growth_factor;}
    float BackoffFactor()const{return _backoff_factor;}
    uint32_t GrowthTracker()const{return _growth_tracker;}
    void SetScale(const float scale){_scale=scale;}
    void SetGrowthTracker(const uint32_t tracker){_growth_tracker=tracker;}
    CAIF_CudaStream &Stream()const{return *_stream;}

    // Apply the adaptive-scale rule after one step's overflow verdict.
    void UpdateScale(const bool overflow);

    float _scale;
    float _growth_factor;
    float _backoff_factor;
    uint32_t _growth_interval;
    uint32_t _growth_tracker;
    CAIF_CudaStream *_stream;
};

}//end instance namespace
