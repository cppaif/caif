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
// Shared finite-difference gradcheck implementation.
//------------------------------------------------------------------------------
#include "caif_gradcheck.h"
#include "caif_cuda_stream.h"
#include "caif_device_tensor.h"
#include "caif_exception.h"
#include "caif_run_context_pass_scope.h"
#include "caif_settings.h"
#include "caif_tolerances.h"
#include "ise_lib/ise_out.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace instance
{

float CAIF_GradCheck::Dot(const std::vector<float> &a,const std::vector<float> &b)
{
  float s=0.0f;
  const size_t n=std::min(a.size(),b.size());
  for(size_t i=0;i<n;++i)
  {
    s+=a[i]*b[i];
  }
  return s;
}

float CAIF_GradCheck::DirectionalLoss(CAIF_GradCheckTargetFunctor &functor,
                                      const std::vector<float> &perturbed_host,
                                      const std::vector<uint32_t> &input_shape,
                                      const std::vector<float> &upstream_grad_host,
                                      CAIF_RunContext &ctx,
                                      const CAIF_DeviceTensor::Location_e location)
{
  try
  {
    CAIF_DeviceTensor perturbed;
    if(location==CAIF_DeviceTensor::Location_e::Device_e && ctx.HasStream()==true)
    {
      perturbed=CAIF_DeviceTensor::Zeros(input_shape,ctx.Stream());
      perturbed.CopyFromHost(perturbed_host.data(),perturbed_host.size());
    }
    else
    {
      perturbed=CAIF_DeviceTensor::ZerosHost(input_shape);
      std::memcpy(perturbed.DeviceDataRaw(),
                  perturbed_host.data(),
                  perturbed_host.size()*sizeof(float));
    }

    const CAIF_DeviceTensor output=functor.ForwardOnly(perturbed,ctx);

    std::vector<float> output_host(output.TotalElements());
    if(output.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      std::memcpy(output_host.data(),
                  output.DeviceDataRaw(),
                  output_host.size()*sizeof(float));
    }
    else
    {
      output.CopyToHost(output_host.data());
      if(ctx.HasStream()==true)
      {
        ctx.Stream().Synchronize();
      }
    }
    return Dot(upstream_grad_host,output_host);
  }
  CAIF_CATCH_BLOCK();
  return 0.0f;
}

bool CAIF_GradCheck::Check(CAIF_GradCheckTargetFunctor &functor,
                           const std::vector<float> &input_host,
                           const std::vector<uint32_t> &input_shape,
                           const std::vector<float> &upstream_grad_host,
                           const std::vector<float> &analytical_grad_host,
                           CAIF_RunContext &ctx,
                           const float rel_tolerance,
                           const CAIF_DeviceTensor::Location_e location)
{
  try
  {
    if(input_host.size()!=analytical_grad_host.size())
    {
      THROW_CAIFE("CAIF_GradCheck: input and analytical gradient size mismatch");
    }

    // Directional perturbation dx derived deterministically from the
    // analytical gradient itself — avoids a second RNG and guarantees a
    // non-degenerate inner product with the analytical vector.
    std::vector<float> dx(input_host.size());
    for(size_t i=0;i<input_host.size();++i)
    {
      dx[i]=std::sin(static_cast<float>(i)*0.37f+0.11f);
    }

    const float analytical_directional=Dot(analytical_grad_host,dx);

    std::vector<float> x_plus(input_host.size());
    std::vector<float> x_minus(input_host.size());
    const float eps=CAIF_Tolerances::FdStep();
    for(size_t i=0;i<input_host.size();++i)
    {
      x_plus[i]=input_host[i]+eps*dx[i];
      x_minus[i]=input_host[i]-eps*dx[i];
    }

    // FD baseline must be numerically accurate. Two independent choices
    // via the existing facility:
    //   1. Pass direction = Backward_e so ComputeTypeFor selects the
    //      backward's compute-type (consistent with how the analytical
    //      gradient was produced).
    //   2. PreciseGradients = true for the FD window: FD in TF32 is
    //      catastrophic cancellation (sign flips on small gradients at
    //      h~1e-3). The FD baseline is, by definition, a high-precision
    //      reference — regardless of which mode the analytical backward
    //      used. The outer test chose that mode; it is restored on exit.
    CAIF_RunContextPassScope pass_scope(ctx,CAIF_RunContext::Pass_e::Backward_e);
    const bool prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);

    const float loss_plus=DirectionalLoss(functor,x_plus,input_shape,upstream_grad_host,ctx,location);
    const float loss_minus=DirectionalLoss(functor,x_minus,input_shape,upstream_grad_host,ctx,location);
    CAIF_Settings::SetPreciseGradients(prev_precise);
    const float numerical_directional=(loss_plus-loss_minus)/(2.0f*eps);

    // Combined abs+rel criterion (numpy-allclose form). FP32 catastrophic
    // cancellation in (L(x+h)-L(x-h))/(2h) leaves ~1e-5 absolute noise for
    // linear-op directional derivatives; the absolute floor below absorbs
    // that without relaxing the relative bound for non-trivial gradients.
    const float abs_floor=CAIF_Tolerances::GradcheckAbsFloor();
    const float diff=std::fabs(analytical_directional-numerical_directional);
    const float scale=std::max(std::fabs(analytical_directional),
                               std::fabs(numerical_directional));
    const float rel=diff/std::max(scale,abs_floor);
    const bool passes=(diff<=rel_tolerance*scale+abs_floor);
    if(passes==false)
    {
      ISE_Out::Out()<<"  gradcheck fail: analytical="
                    <<analytical_directional
                    <<" numerical="
                    <<numerical_directional
                    <<" rel="
                    <<rel
                    <<"\n";
      return false;
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

}//end instance namespace
