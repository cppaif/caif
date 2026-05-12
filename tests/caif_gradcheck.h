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
// Shared finite-difference gradcheck. Replaces 8+ inline FD loops in the
// tests directory.
//
// The directional-derivative form L(x) = sum_i g_i * y_i(x) is used: caller
// provides a functor whose Evaluate(...) runs the forward and returns the
// scalar loss, plus an analytical-gradient tensor produced by the layer's
// own backward. The check compares <analytical_grad, dx> to
// (L(x + eps*dx) - L(x - eps*dx)) / (2*eps).
//
// Precision matching: auto-sets perturbation-forward compute-type to match
// the backward pass via a RunContextPassScope(Backward_e) around the two
// evaluations. That single mechanism replaces every SetInBackwardPass /
// SetPreciseGradients bracket that used to live in test code.
//------------------------------------------------------------------------------
#ifndef CAIF_GRADCHECK_H
#define CAIF_GRADCHECK_H

#include "caif_device_tensor.h"
#include "caif_run_context.h"
#include <cstdint>
#include <vector>

namespace instance
{

class CAIF_GradCheckTargetFunctor
{
  public:
    virtual ~CAIF_GradCheckTargetFunctor()=default;
    virtual CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                          CAIF_RunContext &ctx)=0;
  protected:
  private:
};

class CAIF_GradCheck
{
  public:
    /**
     * @brief Check analytical gradient against finite-difference estimate.
     *
     * @param functor              User-supplied forward wrapper.
     * @param input_host           Baseline input values, flattened.
     * @param input_shape          Shape of the baseline input.
     * @param upstream_grad_host   Upstream gradient g (shape matches output).
     * @param analytical_grad_host Analytical dL/dx produced by the layer's
     *                             backward(g). Must be flattened in the same
     *                             order as `input_host`.
     * @param ctx                  Run context seeded by caller (stream set,
     *                             training flag set as the forward requires).
     * @param rel_tolerance        Acceptance threshold for
     *                             |analytical - numerical| / max(|analytical|, eps).
     * @return true iff the relative error is within tolerance.
     */
    static bool Check(CAIF_GradCheckTargetFunctor &functor,
                      const std::vector<float> &input_host,
                      const std::vector<uint32_t> &input_shape,
                      const std::vector<float> &upstream_grad_host,
                      const std::vector<float> &analytical_grad_host,
                      CAIF_RunContext &ctx,
                      const float rel_tolerance,
                      const CAIF_DeviceTensor::Location_e location=
                        CAIF_DeviceTensor::Location_e::Device_e);

  protected:
  private:
    static float DirectionalLoss(CAIF_GradCheckTargetFunctor &functor,
                                 const std::vector<float> &perturbed_host,
                                 const std::vector<uint32_t> &input_shape,
                                 const std::vector<float> &upstream_grad_host,
                                 CAIF_RunContext &ctx,
                                 const CAIF_DeviceTensor::Location_e location);
    static float Dot(const std::vector<float> &a,const std::vector<float> &b);
};

}//end instance namespace

#endif  // CAIF_GRADCHECK_H
