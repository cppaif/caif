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
#include "caif_sigmoid_noaux_tc_router_functor.h"
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

namespace instance
{

constexpr uint32_t g_caif_moe_sigmoid_test_num_tokens=2;
constexpr uint32_t g_caif_moe_sigmoid_test_input_dim=6;
constexpr uint32_t g_caif_moe_sigmoid_test_num_experts=6;
constexpr uint32_t g_caif_moe_sigmoid_test_top_k=4;
constexpr float g_caif_moe_sigmoid_test_w_identity_val=1.0f;
constexpr float g_caif_moe_sigmoid_test_bias_scale=0.01f;
constexpr float g_caif_moe_sigmoid_test_grad_tol=5e-3f;
constexpr uint32_t g_caif_moe_sigmoid_test_rng_seed=0xBEEFu;
constexpr float g_caif_moe_sigmoid_test_grad_min=-0.5f;
constexpr float g_caif_moe_sigmoid_test_grad_max=0.5f;

class CAIF_MoERouterSigmoidNoauxTcTests
{
  public:
    static void RunAll();

  protected:

  private:
    static bool RunOne(bool norm_topk_prob,
                       float routed_scaling_factor,
                       const std::string &test_name);
};

bool CAIF_MoERouterSigmoidNoauxTcTests::RunOne(const bool norm_topk_prob,
                                               const float routed_scaling_factor,
                                               const std::string &test_name)
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
    const uint32_t num_tokens=g_caif_moe_sigmoid_test_num_tokens;
    const uint32_t input_dim=g_caif_moe_sigmoid_test_input_dim;
    const uint32_t num_experts=g_caif_moe_sigmoid_test_num_experts;
    const uint32_t top_k=g_caif_moe_sigmoid_test_top_k;

    CAIF_CudaStream stream;
    CAIF_DeviceMoERouterConfig cfg(input_dim,
                                   num_experts,
                                   top_k,
                                   CAIF_DeviceMoERouter<float,
                                   float>::RoutingType_e::TopK,
                                   true,
                                   0.0f);
    cfg.SetGatingKind(CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e);
    cfg.SetNormTopkProb(norm_topk_prob);
    cfg.SetRoutedScalingFactor(routed_scaling_factor);

    CAIF_SigmoidNoauxTcRouterFunctor functor(cfg,stream);

    // Identity-shaped w_router: logits[t,e] depends only on input[t,e].
    std::vector<float> w_host(static_cast<size_t>(input_dim*num_experts),0.0f);
    for(uint32_t i=0;i<input_dim;++i)
    {
      w_host[static_cast<size_t>(i*num_experts+i)]=g_caif_moe_sigmoid_test_w_identity_val;
    }
    // Tiny bias: well under any ε*dx logit shift the FD perturbation can
    // induce, so it can't flip top-k rankings.
    std::vector<float> b_host(static_cast<size_t>(num_experts));
    for(size_t i=0;i<b_host.size();++i)
    {
      b_host[i]=g_caif_moe_sigmoid_test_bias_scale*static_cast<float>(i+1);
    }

    CAIF_DeviceTensor w_t=CAIF_DeviceTensor::FromHostData(w_host.data(),
                                                           {input_dim,num_experts},
                                                           stream);
    CAIF_DeviceTensor b_t=CAIF_DeviceTensor::FromHostData(b_host.data(),
                                                           {num_experts},
                                                           stream);
    functor.Router().LoadWRouter(std::move(w_t));
    functor.Router().LoadBRouter(std::move(b_t));

    // Logits span [-2, +3] for the chosen experts (4 values) and [-8, -7]
    // for the unchosen (2 values).  4-unit gap between the lowest chosen
    // and highest unchosen → ε=1e-3 input perturbations can't flip
    // rankings.  Logits in [-2, 3] keep sigmoid' in [0.045, 0.105] so
    // gradients are big enough to clear FP32 FD-cancellation noise.
    std::vector<float> x_host={3.0f,2.0f,1.0f,-2.0f,-7.0f,-8.0f,
                                -2.0f,1.0f,2.0f,3.0f,-8.0f,-7.0f};
    CAIF_DeviceTensor x=CAIF_DeviceTensor::FromHostData(x_host.data(),
                                                         {num_tokens,input_dim},
                                                         stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Forward populates the cached input/probs/indices used by backward.
    CAIF_DeviceMoERouter<float,float>::RouterOutput_t out=functor.Router().Route(x,ctx);

    std::mt19937 gen(g_caif_moe_sigmoid_test_rng_seed);
    std::uniform_real_distribution<float> dist(g_caif_moe_sigmoid_test_grad_min,
                                               g_caif_moe_sigmoid_test_grad_max);
    std::vector<float> g_host(static_cast<size_t>(num_tokens*top_k));
    for(size_t i=0;i<g_host.size();++i)
    {
      g_host[i]=dist(gen);
    }
    CAIF_DeviceTensor grad_w=CAIF_DeviceTensor::FromHostData(g_host.data(),
                                                              {num_tokens,top_k},
                                                              stream);

    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=functor.Router().BackwardRouting(grad_w,ctx);
    stream.Synchronize();

    std::vector<float> analytical(static_cast<size_t>(num_tokens*input_dim),0.0f);
    grad_input.CopyToHostRaw(analytical.data());

    return CAIF_GradCheck::Check(functor,
                                 x_host,
                                 {num_tokens,input_dim},
                                 g_host,
                                 analytical,
                                 ctx,
                                 g_caif_moe_sigmoid_test_grad_tol);
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception in "
                     <<test_name
                     <<": "
                     <<e
                     <<"\n";
    return false;
  }
  catch(...)
  {
    ISE_Out::ErrLog()<<"Unknown exception in "
                     <<test_name
                     <<"\n";
    return false;
  }
}

void CAIF_MoERouterSigmoidNoauxTcTests::RunAll()
{
  ISE_Out::Out()<<"SigmoidNoauxTc Router Backward Gradcheck\n"
                <<"=========================================\n";

  CAIF_TestHarness::Report("norm_topk_prob=true scale=1.0",
                           RunOne(true,1.0f,"NormTrueScale1"));
  CAIF_TestHarness::Report("norm_topk_prob=false scale=1.0",
                           RunOne(false,1.0f,"NormFalseScale1"));
  CAIF_TestHarness::Report("norm_topk_prob=true scale=2.0",
                           RunOne(true,2.0f,"NormTrueScale2"));

  ISE_Out::Out()<<"\n";
  ISE_Out::Out()<<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"  Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

}//end instance namespace

int main()
{
  instance::CAIF_MoERouterSigmoidNoauxTcTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
