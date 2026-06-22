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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// Gradcheck forward-wrapper for the SigmoidNoauxTc MoE router. ForwardOnly()
// runs Route(...) on the perturbed input and returns the expert-weight tensor
// the finite-difference check differentiates. Used by
// test_moe_router_sigmoid_noauxtc_backward.cpp.
//------------------------------------------------------------------------------
#pragma once

#include "caif_gradcheck_target_functor.h"
#include "caif_device_moe_router.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"

#include <utility>

namespace instance
{

class CAIF_SigmoidNoauxTcRouterFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    CAIF_SigmoidNoauxTcRouterFunctor(const CAIF_DeviceMoERouterConfig &cfg,
                                     CAIF_CudaStream &stream):_router(cfg,stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      CAIF_DeviceMoERouter<float,float>::RouterOutput_t out=_router.Route(perturbed,ctx);
      return std::move(out.expert_weights);
    }

    CAIF_DeviceMoERouter<float,float> &Router(){return _router;}

  protected:

  private:
    CAIF_DeviceMoERouter<float,float> _router;
};

}//end instance namespace
