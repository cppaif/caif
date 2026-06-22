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
// DeepSeek-V3 routing: group-limited routing + aux-loss-free
// bias update on CAIF_DeviceMoERouter.
//
//  - GroupRouting: 8 experts in 4 groups of 2, topk_group=1. The scores are set
//    so the single highest-sigmoid expert (#2) sits in a group whose top-2 sum
//    loses to group 0; with grouping that expert is masked and the top-k comes
//    entirely from group 0. Without grouping #2 would be selected — so this
//    proves the group mask is applied (and changes the result).
//  - BiasUpdate: all tokens forced onto expert 0, then UpdateAuxLossFreeBias
//    nudges the overloaded expert's bias down by `rate` and every other
//    expert's up by `rate` (sign of mean_load - load).
//------------------------------------------------------------------------------
#include "caif_device_moe_router.h"
#include "caif_device_moe_router_config.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cstdint>
#include <vector>
#include <cmath>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr float g_caif_f3_bias_rate=0.1f;
constexpr float g_caif_f3_bias_tol=1.0e-4f;

class CAIF_MoEGroupRoutingTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void LoadIdentityRouter(CAIF_DeviceMoERouter<float,float> &router,
                                   const uint32_t input_dim,
                                   const uint32_t num_experts,
                                   CAIF_CudaStream &stream);
    static void TestGroupRouting();
    static void TestBiasUpdate();
};

void CAIF_MoEGroupRoutingTest::LoadIdentityRouter(CAIF_DeviceMoERouter<float,float> &router,
                                                  const uint32_t input_dim,
                                                  const uint32_t num_experts,
                                                  CAIF_CudaStream &stream)
{
  // Identity-shaped W so logits[t,e] == input[t,e]; zero bias to start.
  std::vector<float> w_host(static_cast<size_t>(input_dim)*num_experts,0.0f);
  for(uint32_t i=0;i<input_dim&&i<num_experts;++i)
  {
    w_host[static_cast<size_t>(i)*num_experts+i]=1.0f;
  }
  std::vector<float> b_host(num_experts,0.0f);
  router.LoadWRouter(CAIF_DeviceTensor::FromHostData(w_host.data(),{input_dim,num_experts},stream));
  router.LoadBRouter(CAIF_DeviceTensor::FromHostData(b_host.data(),{num_experts},stream));
}

void CAIF_MoEGroupRoutingTest::TestGroupRouting()
{
  const std::string test_name="MoEGroupRouting::GroupMaskRestrictsToTopGroups";
  try
  {
    const uint32_t input_dim=8;
    const uint32_t num_experts=8;
    const uint32_t top_k=2;
    CAIF_CudaStream stream;
    CAIF_DeviceMoERouterConfig cfg(input_dim,
                                   num_experts,
                                   top_k,
                                   CAIF_DeviceMoERouterConfig::RoutingType_e::TopK,
                                   true,
                                   0.0f);
    cfg.SetGatingKind(CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e);
    cfg.SetNGroup(4);
    cfg.SetTopkGroup(1);

    CAIF_DeviceMoERouter<float,float> router(cfg,stream);
    LoadIdentityRouter(router,input_dim,num_experts,stream);

    // group0={0,1}=[5,5] top-2 sum ~1.986; group1={2,3}=[6,-10] sum ~0.998
    // (expert 2 is the single highest sigmoid); groups 2,3 ~0. topk_group=1
    // keeps only group0, so the top-2 must be {0,1}; expert 2 is masked out.
    std::vector<float> x_host={5.0f,5.0f,6.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f};
    CAIF_DeviceTensor x=CAIF_DeviceTensor::FromHostData(x_host.data(),{1,input_dim},stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceMoERouter<float,float>::RouterOutput_t out=router.Route(x,ctx);
    stream.Synchronize();

    std::vector<int32_t> idx(top_k);
    out.expert_indices.CopyToHostRaw(idx.data());

    bool passed=true;
    for(uint32_t k=0;k<top_k;++k)
    {
      if(idx[k]!=0&&idx[k]!=1)
      {
        passed=false;
      }
    }
    if(passed==false)
    {
      ISE_Out::Out()<<"  selected experts (expected both in group 0 {0,1}): "
                    <<idx[0]<<","<<idx[1]<<"\n";
    }
    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name.c_str())
}

void CAIF_MoEGroupRoutingTest::TestBiasUpdate()
{
  const std::string test_name="MoEGroupRouting::AuxLossFreeBiasUpdate";
  try
  {
    const uint32_t input_dim=4;
    const uint32_t num_experts=4;
    const uint32_t top_k=1;
    const uint32_t num_tokens=4;
    CAIF_CudaStream stream;
    CAIF_DeviceMoERouterConfig cfg(input_dim,
                                   num_experts,
                                   top_k,
                                   CAIF_DeviceMoERouterConfig::RoutingType_e::TopK,
                                   true,
                                   0.0f);
    cfg.SetGatingKind(CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e);
    cfg.SetBiasUpdateRate(g_caif_f3_bias_rate);

    CAIF_DeviceMoERouter<float,float> router(cfg,stream);
    LoadIdentityRouter(router,input_dim,num_experts,stream);

    // Every token forced onto expert 0 (logit 10 vs -10) -> load [4,0,0,0],
    // mean 1. Overloaded expert 0 bias goes down by rate; the rest up by rate.
    std::vector<float> x_host(static_cast<size_t>(num_tokens)*input_dim,-10.0f);
    for(uint32_t t=0;t<num_tokens;++t)
    {
      x_host[static_cast<size_t>(t)*input_dim]=10.0f;
    }
    CAIF_DeviceTensor x=CAIF_DeviceTensor::FromHostData(x_host.data(),{num_tokens,input_dim},stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    router.Route(x,ctx);
    stream.Synchronize();

    router.UpdateAuxLossFreeBias();
    stream.Synchronize();

    std::vector<float> bias(num_experts);
    router.BRouter().CopyToHostRaw(bias.data());

    const bool down_ok=(std::fabs(bias[0]-(-g_caif_f3_bias_rate))<g_caif_f3_bias_tol);
    bool up_ok=true;
    for(uint32_t e=1;e<num_experts;++e)
    {
      if(std::fabs(bias[e]-g_caif_f3_bias_rate)>g_caif_f3_bias_tol)
      {
        up_ok=false;
      }
    }
    const bool passed=(down_ok==true&&up_ok==true);
    if(passed==false)
    {
      ISE_Out::Out()<<"  bias after update (expected [-0.1,0.1,0.1,0.1]): "
                    <<bias[0]<<","<<bias[1]<<","<<bias[2]<<","<<bias[3]<<"\n";
    }
    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name.c_str())
}

void CAIF_MoEGroupRoutingTest::RunAll()
{
  ISE_Out::Out()<<"=== DeepSeek-V3 routing: group routing + aux-loss-free bias ==="
                <<"\n\n";
  TestGroupRouting();
  TestBiasUpdate();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_MoEGroupRoutingTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
