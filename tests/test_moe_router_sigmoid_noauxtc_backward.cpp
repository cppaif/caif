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
// Test: CAIF_DeviceMoERouter SigmoidNoauxTc backward gradient check.
//
// Validates that BackwardRouting(grad_weights) computed by the router for
// gating_kind = SigmoidNoauxTc_e matches the finite-difference reference
// w.r.t. the input tensor. Exercises the chain:
//   probs = sigmoid(input @ w_router)
//   selection = probs + b_router (selection-only)
//   indices = TopK(selection)
//   gathered = probs[indices]
//   if norm_topk_prob: weights = gathered / sum(gathered)
//   weights *= routed_scaling_factor
// and the corresponding backward through Scale, NormalizeRowsBackwardTopKGather,
// ScatterAdd, SigmoidBackward, and the linear-projection transposes.
//
// b_router is intentionally non-zero (so the gather of original sigmoid scores
// at chosen indices actually differs from the bias-corrected scores —
// otherwise Phase 1b is a no-op).
//
// Input values are chosen so the per-token top-k selection is comfortably
// separated; perturbations small enough not to flip rankings.
//------------------------------------------------------------------------------
#include "caif_device_moe_router.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "caif_gradcheck.h"
#include "ise_lib/ise_out.h"

#include <cstdint>
#include <random>
#include <vector>

using namespace instance;

namespace
{

class SigmoidNoauxTcRouterFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    SigmoidNoauxTcRouterFunctor(const CAIF_DeviceMoERouter<float,float>::Config_t &cfg,
                                CAIF_CudaStream &stream):_router(cfg,stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      CAIF_DeviceMoERouter<float,float>::RouterOutput_t out=_router.Route(perturbed,ctx);
      return std::move(out.expert_weights);
    }

    CAIF_DeviceMoERouter<float,float> &Router(){return _router;}

  private:
    CAIF_DeviceMoERouter<float,float> _router;
};

bool RunOne(const bool norm_topk_prob,
            const float routed_scaling_factor,
            const char *test_name)
{
  try
  {
    // Small + extreme-separation setup so the TopK selection boundary is
    // far away from any ε-scale perturbation the gradcheck harness applies
    // (FdStep = 1e-3).  num_experts = top_k + 2 keeps the selection set
    // stable: chosen experts have logits ≥ 5, unchosen ≤ -5; bias is
    // small (< 0.05) so it can't reorder rankings.  W_router is identity
    // -shaped on the leading num_experts dims so per-token logit i depends
    // only on input column i — easy to reason about per-element.
    constexpr uint32_t num_tokens=2;
    constexpr uint32_t input_dim=6;
    constexpr uint32_t num_experts=6;
    constexpr uint32_t top_k=4;

    CAIF_CudaStream stream;
    CAIF_DeviceMoERouter<float,float>::Config_t cfg;
    cfg.input_dim=input_dim;
    cfg.num_experts=num_experts;
    cfg.top_k=top_k;
    cfg.routing_type=CAIF_DeviceMoERouter<float,float>::RoutingType_e::TopK;
    cfg.use_bias=true;
    cfg.noise_std=0.0f;
    cfg.gating_kind=CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e;
    cfg.norm_topk_prob=norm_topk_prob;
    cfg.routed_scaling_factor=routed_scaling_factor;

    SigmoidNoauxTcRouterFunctor functor(cfg,stream);

    // Identity-shaped w_router: logits[t,e] depends only on input[t,e].
    std::vector<float> w_host(input_dim*num_experts,0.0f);
    for(uint32_t i=0;i<input_dim;++i)
    {
      w_host[i*num_experts+i]=1.0f;
    }
    // Tiny bias: well under any ε*dx logit shift the FD perturbation can
    // induce, so it can't flip top-k rankings.
    std::vector<float> b_host(num_experts);
    for(size_t i=0;i<b_host.size();++i)
    {
      b_host[i]=0.01f*static_cast<float>(i+1);
    }

    CAIF_DeviceTensor w_t=CAIF_DeviceTensor::FromHostData(w_host.data(),{input_dim,num_experts},stream);
    CAIF_DeviceTensor b_t=CAIF_DeviceTensor::FromHostData(b_host.data(),{num_experts},stream);
    functor.Router().LoadWRouter(std::move(w_t));
    functor.Router().LoadBRouter(std::move(b_t));

    // Logits span [-2, +3] for the chosen experts (4 values) and [-8, -7]
    // for the unchosen (2 values).  4-unit gap between the lowest chosen
    // and highest unchosen → ε=1e-3 input perturbations can't flip
    // rankings.  Logits in [-2, 3] keep sigmoid' in [0.045, 0.105] so
    // gradients are big enough to clear FP32 FD-cancellation noise.
    std::vector<float> x_host={3.0f,2.0f,1.0f,-2.0f,-7.0f,-8.0f,-2.0f,1.0f,2.0f,3.0f,-8.0f,-7.0f};
    CAIF_DeviceTensor x=CAIF_DeviceTensor::FromHostData(x_host.data(),{num_tokens,input_dim},stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Forward populates the cached input/probs/indices used by backward.
    CAIF_DeviceMoERouter<float,float>::RouterOutput_t out=functor.Router().Route(x,ctx);

    std::mt19937 gen(0xBEEFu);
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    std::vector<float> g_host(num_tokens*top_k);
    for(size_t i=0;i<g_host.size();++i)
    {
      g_host[i]=dist(gen);
    }
    CAIF_DeviceTensor grad_w=CAIF_DeviceTensor::FromHostData(g_host.data(),{num_tokens,top_k},stream);

    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=functor.Router().BackwardRouting(grad_w,ctx);
    stream.Synchronize();

    std::vector<float> analytical(num_tokens*input_dim,0.0f);
    grad_input.CopyToHostRaw(analytical.data());

    return CAIF_GradCheck::Check(functor,x_host,{num_tokens,input_dim},g_host,analytical,ctx,5e-3f);
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception in "<<test_name<<": "<<e<<std::endl;
    return false;
  }
  catch(...)
  {
    ISE_Out::ErrLog()<<"Unknown exception in "<<test_name<<std::endl;
    return false;
  }
}

bool TestNormTrueScale1()
{
  return RunOne(true,1.0f,"NormTrueScale1");
}

bool TestNormFalseScale1()
{
  return RunOne(false,1.0f,"NormFalseScale1");
}

bool TestNormTrueScale2()
{
  return RunOne(true,2.0f,"NormTrueScale2");
}

}//end anonymous namespace

int main()
{
  ISE_Out::Out()<<"SigmoidNoauxTc Router Backward Gradcheck\n";
  ISE_Out::Out()<<"=========================================\n";

  CAIF_TestHarness::Report("norm_topk_prob=true scale=1.0",TestNormTrueScale1());
  CAIF_TestHarness::Report("norm_topk_prob=false scale=1.0",TestNormFalseScale1());
  CAIF_TestHarness::Report("norm_topk_prob=true scale=2.0",TestNormTrueScale2());

  ISE_Out::Out()<<"\n";
  ISE_Out::Out()<<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"  Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
  return CAIF_TestHarness::FinalExitCode();
}