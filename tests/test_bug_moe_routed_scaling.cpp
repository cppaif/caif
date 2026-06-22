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
// routed_scaling_factor was applied (and reversed in
// backward) only when gating_kind == SigmoidNoauxTc. HF DeepseekV2MoEGate
// applies it unconditionally (topk_weight *= routed_scaling_factor) regardless
// of scoring_func, so a softmax-gated build with a non-unit factor (e.g.
// DeepSeek-V2-236B's 16.0) silently ignored it.
//
// With norm_topk_prob == true the normalized top-k weights sum to 1 per token,
// so after the factor they must sum to routed_scaling_factor. This test builds
// a SoftmaxTopK router with routed_scaling_factor = 4.0 and asserts each token's
// combine weights sum to 4.0. Before the fix the softmax path leaves the sum at
// 1.0 and it FAILS; after the fix it PASSES.
//------------------------------------------------------------------------------
#include "caif_device_moe_router.h"
#include "caif_device_moe_layer_factory.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_c7_input_dim=8;
constexpr uint32_t g_c7_num_experts=6;
constexpr uint32_t g_c7_top_k=2;
constexpr uint32_t g_c7_num_tokens=4;
constexpr float g_c7_routed_scaling_factor=4.0f;
constexpr float g_c7_tolerance=1.0e-3f;

class CAIF_MoERoutedScalingBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestSoftmaxRoutedScalingApplied();
};

void CAIF_MoERoutedScalingBugTest::TestSoftmaxRoutedScalingApplied()
{
  try
  {
    CAIF_CudaStream stream;

    CAIF_DeviceMoERouterConfig cfg(g_c7_input_dim,
                                   g_c7_num_experts,
                                   g_c7_top_k,
                                   CAIF_DeviceMoERouter<float,
                                   float>::RoutingType_e::TopK,
                                   false,
                                   0.0f);
    cfg.SetGatingKind(CAIF_DeviceMoELayerFactory::GatingKind_e::SoftmaxTopK_e);
    cfg.SetNormTopkProb(true);
    cfg.SetRoutedScalingFactor(g_c7_routed_scaling_factor);

    CAIF_DeviceMoERouter<float,float> router(cfg,stream);

    const size_t total=static_cast<size_t>(g_c7_num_tokens)*g_c7_input_dim;
    std::vector<float> host_input(total);
    for(size_t i=0;i<total;++i)
    {
      host_input[i]=static_cast<float>((i%5)+1)*0.21f-0.5f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {g_c7_num_tokens,g_c7_input_dim},
                                                            stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceMoERouter<float,float>::RouterOutput_t out=router.Route(input,ctx);

    std::vector<float> weights(static_cast<size_t>(g_c7_num_tokens)*g_c7_top_k);
    out.expert_weights.CopyToHost(weights.data());

    bool ok=true;
    float worst_sum=g_c7_routed_scaling_factor;
    for(uint32_t t=0;t<g_c7_num_tokens;++t)
    {
      float row_sum=0.0f;
      for(uint32_t k=0;k<g_c7_top_k;++k)
      {
        row_sum+=weights[static_cast<size_t>(t)*g_c7_top_k+k];
      }
      if(std::fabs(row_sum-g_c7_routed_scaling_factor)>=g_c7_tolerance)
      {
        ok=false;
      }
      if(std::fabs(row_sum-g_c7_routed_scaling_factor)>std::fabs(worst_sum-g_c7_routed_scaling_factor))
      {
        worst_sum=row_sum;
      }
    }

    if(ok==false)
    {
      ISE_Out::Out()<<"  softmax routed_scaling_factor not applied: a token's"
                    <<" combine weights summed to "
                    <<worst_sum
                    <<" (expected "
                    <<g_c7_routed_scaling_factor
                    <<")\n";
    }
    CAIF_TestHarness::Report("BugC7::MoERouter::SoftmaxRoutedScalingApplied",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugC7::MoERouter::SoftmaxRoutedScalingApplied")
}

void CAIF_MoERoutedScalingBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug C7: routed_scaling_factor on softmax gating ==="
                <<"\n\n";
  TestSoftmaxRoutedScalingApplied();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_MoERoutedScalingBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
