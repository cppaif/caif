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

//--------------------------------------------------------------------------
// Example: CAIF_DeviceMoERouter with `norm_topk_prob` toggle
//
// `norm_topk_prob` is a config flag on the MoE router's
// SoftmaxTopK gating regime. After the router selects the top-k
// experts for each token (by softmax score):
//
//   - norm_topk_prob = true   — re-normalise the k selected weights
//                               so they sum to 1.0. This is the
//                               HuggingFace DeepSeek-V2 / GLM-4-MoE
//                               default.
//
//   - norm_topk_prob = false  — keep the raw softmax-over-all-experts
//                               values for the top-k slots. Their sum
//                               is < 1.0 in general. This is the
//                               OLMoE / Olmo2 default.
//
// This example builds two routers with identical weights and config
// EXCEPT for `norm_topk_prob`, routes the same input through each,
// and prints the per-token sum of the top-k expert weights. The
// `norm` router rows sum to 1.0; the raw router rows sum to less.
//--------------------------------------------------------------------------

#include "caif_device_moe_router.h"
#include "caif_device_moe_layer_factory.h"
#include "caif_device_context.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_run_context_pass_scope.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <memory>

using namespace instance;

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF MoE router norm_topk_prob toggle example ==="<<std::endl;

    typedef CAIF_DeviceMoERouter<float,float> Router_t;

    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    const uint32_t input_dim=64;
    const uint32_t num_experts=8;
    const uint32_t top_k=2;
    const uint32_t num_tokens=4;
    const uint32_t init_seed=7;

    CAIF_DeviceMoERouterConfig base_cfg(input_dim,
                                        num_experts,
                                        top_k,
                                        Router_t::RoutingType_e::TopK,
                                        false,
                                        0.0f);
    base_cfg.SetGatingKind(CAIF_DeviceMoELayerFactory::GatingKind_e::SoftmaxTopK_e);
    base_cfg.SetRoutedScalingFactor(1.0f);

    CAIF_DeviceMoERouterConfig cfg_norm=base_cfg;
    cfg_norm.SetNormTopkProb(true);

    CAIF_DeviceMoERouterConfig cfg_raw=base_cfg;
    cfg_raw.SetNormTopkProb(false);

    Router_t router_norm(cfg_norm,stream);
    Router_t router_raw(cfg_raw,stream);

    // Initialise both routers with the same seed so the underlying
    // router weights are identical — the only difference between the
    // two routes will be the norm_topk_prob toggle.
    router_norm.InitializeWeights(init_seed);
    router_raw.InitializeWeights(init_seed);

    // Synthetic input — small enough to print every row.
    const uint32_t total_elements=num_tokens*input_dim;
    std::vector<float> input_host(total_elements);
    for(uint32_t i=0;i<total_elements;++i)
    {
      input_host[i]=static_cast<float>(i%17)/17.0f-0.5f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_host.data(),
                                                            {num_tokens,input_dim},
                                                            stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    CAIF_RunContextPassScope forward_scope(ctx,CAIF_RunContext::Pass_e::Forward_e);

    Router_t::RouterOutput_t out_norm=router_norm.Route(input,ctx);
    Router_t::RouterOutput_t out_raw=router_raw.Route(input,ctx);
    stream.Synchronize();

    // expert_weights shape: [num_tokens, top_k]. Download and sum per
    // row to surface the normalised-vs-raw difference.
    const uint32_t weight_count=num_tokens*top_k;
    std::vector<float> w_norm_host(weight_count);
    std::vector<float> w_raw_host(weight_count);
    out_norm.expert_weights.CopyToHost(w_norm_host.data());
    out_raw.expert_weights.CopyToHost(w_raw_host.data());
    stream.Synchronize();

    ISE_Out::Out()<<"Per-token sum of top-"<<top_k<<" expert weights"
                  <<" (norm vs raw):"<<std::endl;
    for(uint32_t t=0;t<num_tokens;++t)
    {
      float sum_norm=0.0f;
      float sum_raw=0.0f;
      for(uint32_t k=0;k<top_k;++k)
      {
        sum_norm+=w_norm_host[t*top_k+k];
        sum_raw+=w_raw_host[t*top_k+k];
      }
      ISE_Out::Out()<<"  token "<<t
                    <<": norm_sum="<<sum_norm
                    <<"  raw_sum="<<sum_raw
                    <<std::endl;
    }

    ISE_Out::Out()<<"=== Done ==="<<std::endl;
    return 0;
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception: "<<e<<std::endl;
    return 1;
  }
}
