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
// MoE (Mixture of Experts) Tests
//------------------------------------------------------------------------------
#include "caif_device_moe_layer.h"
#include "caif_test_harness.h"
#include "caif_test_constants.h"
#include "caif_device_moe_router.h"
#include "caif_device_moe_expert.h"
#include "caif_ops.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace instance;

static void ReportResult(const char *test_name,bool passed)
{
  CAIF_TestHarness::Report(test_name,passed);
}

static bool FloatEqual(float a,float b,float tolerance=1e-4f)
{
  return CAIF_TestHarness::FloatEqual(a,b,tolerance);
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Test 1: MoE Expert forward shape
//------------------------------------------------------------------------------
static void TestExpertForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=8;
    const uint32_t hidden_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpert<float,float>::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.use_gated=false;
    config.use_bias=true;

    CAIF_DeviceMoEExpert<float,float> expert(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch*seq_len,dim},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=expert.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=2||shape[0]!=batch*seq_len||shape[1]!=dim)
    {
      std::cout<<"  Shape mismatch: expected [8,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
      passed=false;
    }

    ReportResult("MoEExpert::ForwardShape",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 2: MoE Expert gated forward shape
//------------------------------------------------------------------------------
static void TestExpertGatedForwardShape()
{
  try
  {
    const uint32_t num_tokens=8;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpert<float,float>::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.use_gated=true;
    config.use_bias=true;

    CAIF_DeviceMoEExpert<float,float> expert(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=expert.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=dim)
    {
      std::cout<<"  Shape mismatch\n";
      passed=false;
    }

    ReportResult("MoEExpert::GatedForwardShape",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 3: MoE Router routing output shape
//------------------------------------------------------------------------------
static void TestRouterOutputShape()
{
  try
  {
    const uint32_t num_tokens=16;
    const uint32_t dim=8;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;
    CAIF_DeviceMoERouter<float,float>::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter<float,float>::RoutingType_e::TopK;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter<float,float> router(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter<float,float>::RouterOutput_t output=router.Route(input,ctx);

    bool passed=true;

    // Check indices shape [num_tokens, top_k]
    const auto &idx_shape=output.expert_indices.Shape();
    if(idx_shape.size()!=2||idx_shape[0]!=num_tokens||idx_shape[1]!=top_k)
    {
      std::cout<<"  Indices shape mismatch\n";
      passed=false;
    }

    // Check weights shape [num_tokens, top_k]
    const auto &wt_shape=output.expert_weights.Shape();
    if(wt_shape.size()!=2||wt_shape[0]!=num_tokens||wt_shape[1]!=top_k)
    {
      std::cout<<"  Weights shape mismatch\n";
      passed=false;
    }

    // Check probs shape [num_tokens, num_experts]
    const auto &prob_shape=output.router_probs.Shape();
    if(prob_shape.size()!=2||prob_shape[0]!=num_tokens||prob_shape[1]!=num_experts)
    {
      std::cout<<"  Probs shape mismatch\n";
      passed=false;
    }

    ReportResult("MoERouter::OutputShape",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 3b: MoE Router top-k indices match argmax of probs (correctness)
//
// Regression for the bug where CAIF_Ops::TopK wrote int32 bytes into
// a float32-allocated indices tensor.  Consumers (DispatchTokens,
// CombineOutputs, gather kernels) read the same memory as float, which
// reinterprets the int32 bit pattern of small values (0..7) as near-zero
// denormals and rounds to 0 — routing every token to expert 0.
//
// This test drives the router with a logits matrix whose argmax differs
// row-by-row and asserts expert_indices[t, 0] equals the expected argmax.
//------------------------------------------------------------------------------
static void TestRouterTopKIndicesMatchArgmax()
{
  try
  {
    const uint32_t num_tokens=8;
    const uint32_t dim=4;
    const uint32_t num_experts=8;
    const uint32_t top_k=1;

    CAIF_CudaStream stream;
    CAIF_DeviceMoERouter<float,float>::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter<float,float>::RoutingType_e::TopK;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter<float,float> router(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Build an input x router weight matrix whose argmax is deterministic
    // per row: craft an input where row t has a simple pattern, and inject
    // a known router weight override via FromHostData on the router's
    // internal _w_router tensor isn't available publicly — instead we
    // drive the router with random weights (stable within one run) and
    // then verify indices against a reference computed from the returned
    // router_probs.  That isolates the TopK <-> probs contract.
    std::vector<float> host_input(num_tokens*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>((i*37+11)%97)*0.01f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter<float,float>::RouterOutput_t output=router.Route(input,ctx);

    // Pull probs + indices to host and verify indices match argmax(probs).
    CAIF_HostTensor probs_h=output.router_probs.ToHost();
    std::vector<int32_t> idx_host(num_tokens*top_k);
    output.expert_indices.CopyToHostRaw(idx_host.data());
    const float *probs=probs_h.Data();

    bool passed=true;
    for(uint32_t t=0;t<num_tokens;++t)
    {
      uint32_t expected=0;
      float best=probs[t*num_experts+0];
      for(uint32_t e=1;e<num_experts;++e)
      {
        if(probs[t*num_experts+e]>best)
        {
          best=probs[t*num_experts+e];
          expected=e;
        }
      }
      const int32_t actual=idx_host[t*top_k+0];
      if(static_cast<uint32_t>(actual)!=expected)
      {
        ISE_Out::Out()<<"  Token "
                      <<t
                      <<" expected argmax="
                      <<expected
                      <<" got indices[]="
                      <<actual
                      <<std::endl;
        passed=false;
      }
    }

    // Also verify token_counts would not collapse to a single expert when
    // probs are diverse: the number of distinct expert indices used must
    // be >= 2 (probability of all eight tokens sharing one expert by
    // accident with random input is negligible).
    uint32_t distinct=0;
    {
      bool seen[16]={false,false,false,false,false,false,false,false,
                     false,false,false,false,false,false,false,false};
      for(uint32_t t=0;t<num_tokens;++t)
      {
        const int32_t idx=idx_host[t*top_k+0];
        if(idx>=0&&idx<static_cast<int32_t>(num_experts)&&seen[idx]==false)
        {
          seen[idx]=true;
          ++distinct;
        }
      }
    }
    if(distinct<2)
    {
      ISE_Out::Out()<<"  Only "
                    <<distinct
                    <<" distinct expert(s) selected across "
                    <<num_tokens
                    <<" tokens — routing likely collapsed to a single expert"
                    <<std::endl;
      passed=false;
    }

    ReportResult("MoERouter::TopKIndicesMatchArgmax",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 3c: MoE Router::InitFavorExpert biases init toward a chosen expert.
//
// After InitFavorExpert(target, 5.0f) the softmax over experts should
// concentrate >0.97 on the target expert and <0.02 on every other expert,
// regardless of input — because the bias dominates and the matrix weight
// is zeroed at init. This is the hook used by the add-moe / layer-surgery
// path so the model starts in the "frozen base behavior" state.
//------------------------------------------------------------------------------
static void TestRouterInitFavorExpert()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceMoERouter<float,float>::Config_t config;
    config.input_dim=g_caif_moe_init_favor_test_dim;
    config.num_experts=g_caif_moe_init_favor_test_experts;
    config.top_k=g_caif_moe_init_favor_test_topk;
    config.routing_type=CAIF_DeviceMoERouter<float,float>::RoutingType_e::TopK;
    config.use_bias=true;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter<float,float> router(config,stream);
    router.InitFavorExpert(g_caif_moe_init_favor_target_expert,
                            g_caif_moe_init_favor_bias_magnitude);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(g_caif_moe_init_favor_test_tokens
                                    *g_caif_moe_init_favor_test_dim,
                                    g_caif_moe_init_favor_input_value);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {g_caif_moe_init_favor_test_tokens,
                                                                g_caif_moe_init_favor_test_dim},
                                                              stream);

    CAIF_DeviceMoERouter<float,float>::RouterOutput_t output=router.Route(input,ctx);
    CAIF_HostTensor h_probs=output.router_probs.ToHost();

    bool passed=true;
    const uint32_t target=g_caif_moe_init_favor_target_expert;
    for(uint32_t t=0;t<g_caif_moe_init_favor_test_tokens;++t)
    {
      const float fav_prob=h_probs.Data()[t*g_caif_moe_init_favor_test_experts+target];
      if(fav_prob<g_caif_moe_init_favor_min_prob)
      {
        std::cout<<"  token "<<t<<" favored prob "<<fav_prob
                 <<" < expected min "<<g_caif_moe_init_favor_min_prob<<"\n";
        passed=false;
      }
      for(uint32_t e=0;e<g_caif_moe_init_favor_test_experts;++e)
      {
        if(e==target)
        {
          continue;
        }
        const float other_prob=h_probs.Data()[t*g_caif_moe_init_favor_test_experts+e];
        if(other_prob>g_caif_moe_init_favor_max_other_prob)
        {
          std::cout<<"  token "<<t<<" other expert "<<e<<" prob "<<other_prob
                   <<" > max "<<g_caif_moe_init_favor_max_other_prob<<"\n";
          passed=false;
        }
      }
    }
    ReportResult("MoERouter::InitFavorExpert",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 3d: MoE Layer with mixed-size experts (Phase 8.5.D).
//
// Each expert can carry its own hidden_dim; the dispatch/combine
// buffers stay uniform at [total_assigned, input_dim] (those are not
// hidden-dim-dependent), and each expert's per-token intermediate
// activation is sized to its own hidden_dim. This test builds 3
// experts at hidden dims {32, 16, 8} and asserts forward+backward
// produces finite, correctly-shaped output.
//
// The "no max-pad slack" contract from Tier I.1 is satisfied by
// construction: the per-expert intermediate VRAM lives inside each
// CAIF_DeviceMoEExpert<C,S>'s ForwardImpl frame, sized exactly per
// `_config.hidden_dim`, freed at end-of-frame. There is no shared
// max-padded buffer to slack on.
//------------------------------------------------------------------------------
static void TestMoELayerMixedSizeExperts()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<float,float>>> routed;
    std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<float,float>>> shared;

    const std::vector<uint32_t> dims={
      g_caif_moe_mixed_test_hidden_a,
      g_caif_moe_mixed_test_hidden_b,
      g_caif_moe_mixed_test_hidden_c
    };
    for(uint32_t hdim:dims)
    {
      CAIF_DeviceMoEExpert<float,float>::Config_t cfg;
      cfg.input_dim=g_caif_moe_mixed_test_input_dim;
      cfg.hidden_dim=hdim;
      cfg.use_gated=true;
      cfg.use_bias=false;
      routed.push_back(std::make_unique<CAIF_DeviceMoEExpert<float,float>>(cfg,stream));
    }

    CAIF_DeviceMoELayer<float,float> layer(g_caif_moe_mixed_test_input_dim,
                                              g_caif_moe_mixed_test_hidden_a,
                                              g_caif_moe_mixed_test_topk,
                                              false,
                                              0.0f,
                                              g_caif_moe_mixed_test_capacity_factor,
                                              CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                                              0.0f,
                                              0.0f,
                                              std::move(routed),
                                              std::move(shared),
                                              stream);

    bool passed=true;
    if(layer.NumExperts()!=dims.size())
    {
      std::cout<<"  expected "<<dims.size()<<" experts, got "<<layer.NumExperts()<<"\n";
      passed=false;
    }
    for(size_t i=0;i<dims.size();++i)
    {
      if(layer.Expert(i).HiddenDim()!=dims[i])
      {
        std::cout<<"  expert "<<i<<" hidden_dim "<<layer.Expert(i).HiddenDim()
                 <<" expected "<<dims[i]<<"\n";
        passed=false;
      }
    }

    std::vector<float> host_input(g_caif_moe_mixed_test_num_tokens
                                    *g_caif_moe_mixed_test_input_dim,
                                    g_caif_moe_mixed_test_input_value);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {g_caif_moe_mixed_test_num_tokens,
                                                                g_caif_moe_mixed_test_input_dim},
                                                              stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceTensor out=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=out.ToHost();
    if(h_out.Shape().size()!=2
       ||h_out.Shape()[0]!=g_caif_moe_mixed_test_num_tokens
       ||h_out.Shape()[1]!=g_caif_moe_mixed_test_input_dim)
    {
      std::cout<<"  forward shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_out.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        std::cout<<"  non-finite forward at "<<i<<"\n";
        passed=false;
      }
    }

    if(passed==true)
    {
      std::vector<float> grad_data(g_caif_moe_mixed_test_num_tokens
                                     *g_caif_moe_mixed_test_input_dim,
                                     g_caif_moe_mixed_test_grad_value);
      CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                                   {g_caif_moe_mixed_test_num_tokens,
                                                                     g_caif_moe_mixed_test_input_dim},
                                                                   stream);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      CAIF_DeviceTensor grad_in=layer.Backward(grad_out,ctx);
      CAIF_HostTensor h_gi=grad_in.ToHost();
      for(size_t i=0;i<h_gi.TotalElements();++i)
      {
        if(std::isfinite(h_gi.Data()[i])==false)
        {
          std::cout<<"  non-finite backward at "<<i<<"\n";
          passed=false;
          break;
        }
      }
    }

    ReportResult("MoELayer::MixedSizeExperts",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 4: MoE Router weights sum to 1
//------------------------------------------------------------------------------
static void TestRouterWeightsNormalized()
{
  try
  {
    const uint32_t num_tokens=8;
    const uint32_t dim=8;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;
    CAIF_DeviceMoERouter<float,float>::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter<float,float>::RoutingType_e::TopK;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter<float,float> router(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.01f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter<float,float>::RouterOutput_t output=router.Route(input,ctx);
    CAIF_HostTensor weights=output.expert_weights.ToHost();

    bool passed=true;
    for(uint32_t t=0;t<num_tokens;++t)
    {
      float sum=0.0f;
      for(uint32_t k=0;k<top_k;++k)
      {
        sum+=weights.Data()[t*top_k+k];
      }
      if(FloatEqual(sum,1.0f,1e-3f)==false)
      {
        std::cout<<"  Token "<<t<<" weights sum to "<<sum<<" (expected 1.0)\n";
        passed=false;
        break;
      }
    }

    ReportResult("MoERouter::WeightsNormalized",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 5: MoE Layer forward shape
//------------------------------------------------------------------------------
static void TestMoELayerForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        0,
                        0,
                        false,
                        0.0f,
                        1.5f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.0f,
                        0.0f,
                        stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=moe.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      std::cout<<"  Shape mismatch: expected [2,4,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
      passed=false;
    }

    ReportResult("MoELayer::ForwardShape",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 6: MoE Layer auxiliary losses
//------------------------------------------------------------------------------
static void TestMoELayerAuxLosses()
{
  try
  {
    const uint32_t num_tokens=16;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        0,
                        0,
                        false,
                        0.0f,
                        2.0f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.01f,
                        0.001f,
                        stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.01f-0.5f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    moe.Forward(input,ctx);
    const float balance_loss=moe.LastBalanceLoss();
    const float z_loss=moe.LastZLoss();

    bool passed=true;

    // Balance loss should be non-negative
    if(balance_loss<0.0f)
    {
      std::cout<<"  Balance loss is negative: "<<balance_loss<<"\n";
      passed=false;
    }

    // Z-loss should be non-negative
    if(z_loss<0.0f)
    {
      std::cout<<"  Z-loss is negative: "<<z_loss<<"\n";
      passed=false;
    }

    std::cout<<"  Balance loss: "<<balance_loss<<"\n";
    std::cout<<"  Z-loss: "<<z_loss<<"\n";

    ReportResult("MoELayer::AuxLosses",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 7: MoE Layer backward
//------------------------------------------------------------------------------
static void TestMoELayerBackward()
{
  try
  {
    const uint32_t num_tokens=8;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        0,
                        0,
                        false,
                        0.0f,
                        2.0f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.0f,
                        0.0f,
                        stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    // Forward with training=true
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    moe.Forward(input,ctx);

    // Backward
    std::vector<float> grad_ones(num_tokens*dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_ones.data(),{num_tokens,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=moe.Backward(grad_output,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    // Check shape
    const auto &shape=host_grad.Shape();
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=dim)
    {
      std::cout<<"  Gradient shape mismatch\n";
      passed=false;
    }

    // Check gradients are not all zero
    bool any_nonzero=false;
    for(size_t i=0;i<host_grad.TotalElements();++i)
    {
      if(host_grad.Data()[i]!=0.0f)
      {
        any_nonzero=true;
        break;
      }
    }
    if(any_nonzero==false)
    {
      std::cout<<"  All gradients are zero\n";
      passed=false;
    }

    ReportResult("MoELayer::Backward",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 8: MoE Layer parameter count
//------------------------------------------------------------------------------
static void TestMoELayerParameterCount()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        0,
                        0,
                        false,
                        0.0f,
                        1.5f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.0f,
                        0.0f,
                        stream);

    bool passed=true;

    // Router: 1 tensor (w_router [dim x num_experts])
    // Each expert (non-gated): 4 tensors (w1, b1, w2, b2)
    // Total: 1 + 4*num_experts = 17 tensors
    const size_t expected_count=1+4*num_experts;
    if(moe.ParameterTensorCount()!=expected_count)
    {
      std::cout<<"  ParameterTensorCount expected "<<expected_count
               <<", got "<<moe.ParameterTensorCount()<<"\n";
      passed=false;
    }

    // Check total parameter count
    // Router: dim * num_experts = 8 * 4 = 32
    // Each expert: dim*hidden + hidden + hidden*dim + dim = 8*16+16+16*8+8 = 296
    // Total: 32 + 4*296 = 1216
    const size_t expected_total=dim*num_experts+num_experts*(dim*hidden_dim+hidden_dim+hidden_dim*dim+dim);
    if(moe.TotalParameterCount()!=expected_total)
    {
      std::cout<<"  TotalParameterCount expected "<<expected_total
               <<", got "<<moe.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("MoELayer::ParameterCount",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 9: MoE Layer capacity enforcement
//------------------------------------------------------------------------------
static void TestMoELayerCapacity()
{
  try
  {
    const uint32_t num_tokens=32;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=1;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        0,
                        0,
                        false,
                        0.0f,
                        0.5f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.0f,
                        0.0f,
                        stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceTensor output=moe.Forward(input,ctx);
    const auto &shape=output.Shape();

    bool passed=true;

    // Capacity_factor=0.5 forces drops; Forward must still return a
    // correctly-shaped output (dropped-token contributions are zero).
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=dim)
    {
      std::cout<<"  Output shape mismatch under capacity limit\n";
      passed=false;
    }

    ReportResult("MoELayer::CapacityEnforcement",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 10: Expert Choice routing
//------------------------------------------------------------------------------
static void TestExpertChoiceRouting()
{
  try
  {
    const uint32_t num_tokens=16;
    const uint32_t dim=8;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;
    CAIF_DeviceMoERouter<float,float>::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter<float,float>::RoutingType_e::ExpertChoice;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter<float,float> router(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.01f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter<float,float>::RouterOutput_t output=router.Route(input,ctx);

    bool passed=true;

    // Expert Choice: indices are [num_experts, capacity]
    // capacity = (num_tokens * top_k) / num_experts = (16 * 2) / 4 = 8
    const uint32_t expected_capacity=(num_tokens*top_k)/num_experts;
    const auto &idx_shape=output.expert_indices.Shape();

    if(idx_shape.size()!=2||idx_shape[0]!=num_experts||idx_shape[1]!=expected_capacity)
    {
      std::cout<<"  Expert Choice indices shape mismatch: expected ["
               <<num_experts<<","<<expected_capacity<<"], got ["
               <<idx_shape[0]<<","<<idx_shape[1]<<"]\n";
      passed=false;
    }

    // Check weights shape matches indices
    const auto &wt_shape=output.expert_weights.Shape();
    if(wt_shape!=idx_shape)
    {
      std::cout<<"  Expert Choice weights shape doesn't match indices\n";
      passed=false;
    }

    ReportResult("MoERouter::ExpertChoiceRouting",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 11: Soft MoE routing
//------------------------------------------------------------------------------
static void TestSoftMoERouting()
{
  try
  {
    const uint32_t num_tokens=8;
    const uint32_t dim=8;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;  // Ignored for Soft routing

    CAIF_CudaStream stream;
    CAIF_DeviceMoERouter<float,float>::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter<float,float>::RoutingType_e::Soft;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter<float,float> router(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter<float,float>::RouterOutput_t output=router.Route(input,ctx);

    bool passed=true;

    // Soft MoE: indices are [num_tokens, num_experts] (all experts)
    const auto &idx_shape=output.expert_indices.Shape();
    if(idx_shape.size()!=2||idx_shape[0]!=num_tokens||idx_shape[1]!=num_experts)
    {
      std::cout<<"  Soft MoE indices shape mismatch: expected ["
               <<num_tokens<<","<<num_experts<<"]\n";
      passed=false;
    }

    // Weights should be full softmax [num_tokens, num_experts]
    const auto &wt_shape=output.expert_weights.Shape();
    if(wt_shape.size()!=2||wt_shape[0]!=num_tokens||wt_shape[1]!=num_experts)
    {
      std::cout<<"  Soft MoE weights shape mismatch\n";
      passed=false;
    }

    // Check weights sum to 1 per token
    CAIF_HostTensor weights=output.expert_weights.ToHost();
    for(uint32_t t=0;t<num_tokens;++t)
    {
      float sum=0.0f;
      for(uint32_t e=0;e<num_experts;++e)
      {
        sum+=weights.Data()[t*num_experts+e];
      }
      if(FloatEqual(sum,1.0f,1e-3f)==false)
      {
        std::cout<<"  Token "<<t<<" weights sum to "<<sum<<" (expected 1.0)\n";
        passed=false;
        break;
      }
    }

    ReportResult("MoERouter::SoftMoERouting",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 12: Overflow strategy - NoOp
//------------------------------------------------------------------------------
static void TestOverflowNoOp()
{
  try
  {
    const uint32_t num_tokens=32;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=1;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        0,
                        0,
                        false,
                        0.0f,
                        0.5f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::NoOp,
                        0.0f,
                        0.0f,
                        stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    // Only Drop overflow is supported on the GPU dispatch path; NoOp
    // must throw CAIFE loudly. See caif_device_moe_layer.cpp comment
    // at the THROW_CAIFE site.
    bool threw=false;
    try
    {
      CAIF_DeviceTensor output=moe.Forward(input,ctx);
    }
    catch(const CAIF_Exception &)
    {
      threw=true;
    }

    ReportResult("MoELayer::OverflowNoOp",threw);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 13: Shared Experts (DeepSeekMoE style)
//------------------------------------------------------------------------------
static void TestSharedExperts()
{
  try
  {
    const uint32_t num_tokens=16;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t num_shared=2;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        num_shared,
                        0,
                        false,
                        0.0f,
                        2.0f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.0f,
                        0.0f,
                        stream);

    bool passed=true;

    // Check shared expert count
    if(moe.NumSharedExperts()!=num_shared)
    {
      std::cout<<"  NumSharedExperts expected "<<num_shared<<", got "<<moe.NumSharedExperts()<<"\n";
      passed=false;
    }

    // Parameter count should include shared experts
    // Router: 1 tensor (dim * num_experts = 32)
    // Each routed expert (non-gated with bias): 4 tensors
    // Each shared expert (non-gated with bias): 4 tensors
    // Total tensors: 1 + 4*num_experts + 4*num_shared = 1 + 16 + 8 = 25
    const size_t expected_tensors=1+4*num_experts+4*num_shared;
    if(moe.ParameterTensorCount()!=expected_tensors)
    {
      std::cout<<"  ParameterTensorCount expected "<<expected_tensors
               <<", got "<<moe.ParameterTensorCount()<<"\n";
      passed=false;
    }

    // Test forward pass
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceTensor output=moe.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Check output shape
    const auto &shape=host_output.Shape();
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=dim)
    {
      std::cout<<"  Output shape mismatch\n";
      passed=false;
    }

    // Output should be non-zero (routed + shared experts)
    bool any_nonzero=false;
    for(size_t i=0;i<host_output.TotalElements();++i)
    {
      if(host_output.Data()[i]!=0.0f)
      {
        any_nonzero=true;
        break;
      }
    }
    if(any_nonzero==false)
    {
      std::cout<<"  Output is all zeros\n";
      passed=false;
    }

    ReportResult("MoELayer::SharedExperts",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 14: Shared Experts with custom hidden dim
//------------------------------------------------------------------------------
static void TestSharedExpertsCustomHiddenDim()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t shared_hidden=32;  // Different from routed experts
    const uint32_t num_experts=4;
    const uint32_t num_shared=1;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        num_shared,
                        shared_hidden,
                        false,
                        0.0f,
                        2.0f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.0f,
                        0.0f,
                        stream);

    bool passed=true;

    // Total params:
    // Router: dim * num_experts = 8 * 4 = 32
    // Routed experts: 4 * (dim*hidden + hidden + hidden*dim + dim) = 4 * (128+16+128+8) = 4*280 = 1120
    // Shared expert: 1 * (dim*shared_hidden + shared_hidden + shared_hidden*dim + dim)
    //              = 1 * (8*32 + 32 + 32*8 + 8) = 1 * (256+32+256+8) = 552
    // Total = 32 + 1120 + 552 = 1704
    const size_t expected_total=32+4*(dim*hidden_dim+hidden_dim+hidden_dim*dim+dim)+
                                  num_shared*(dim*shared_hidden+shared_hidden+shared_hidden*dim+dim);

    if(moe.TotalParameterCount()!=expected_total)
    {
      std::cout<<"  TotalParameterCount expected "<<expected_total
               <<", got "<<moe.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("MoELayer::SharedExpertsCustomHiddenDim",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 16: Shared experts backward pass
//------------------------------------------------------------------------------
static void TestSharedExpertsBackward()
{
  try
  {
    const uint32_t num_tokens=8;
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t num_shared=2;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;CAIF_DeviceMoELayer<float,float> moe(dim,
                        hidden_dim,
                        num_experts,
                        top_k,
                        false,
                        true,
                        num_shared,
                        0,
                        false,
                        0.0f,
                        2.0f,
                        CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop,
                        0.0f,
                        0.0f,
                        stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    // Forward with training=true
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    moe.Forward(input,ctx);

    // Backward
    std::vector<float> grad_ones(num_tokens*dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_ones.data(),{num_tokens,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=moe.Backward(grad_output,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    // Check shape
    const auto &shape=host_grad.Shape();
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=dim)
    {
      std::cout<<"  Gradient shape mismatch\n";
      passed=false;
    }

    // Check gradients are not all zero (shared experts contribute)
    bool any_nonzero=false;
    for(size_t i=0;i<host_grad.TotalElements();++i)
    {
      if(host_grad.Data()[i]!=0.0f)
      {
        any_nonzero=true;
        break;
      }
    }
    if(any_nonzero==false)
    {
      std::cout<<"  All gradients are zero\n";
      passed=false;
    }

    ReportResult("MoELayer::SharedExpertsBackward",passed);
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 21: GPU-optimized MoE operations (Phase 6)
//------------------------------------------------------------------------------
static void TestGPUOptimizedMoEOps()
{
  try
  {
    const uint32_t num_tokens=8;
    const uint32_t dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;
    bool passed=true;

    // Create random router logits [num_tokens, num_experts]
    std::vector<float> logits_host(num_tokens*num_experts);
    for(size_t i=0;i<logits_host.size();++i)
    {
      logits_host[i]=static_cast<float>(i%10)/10.0f-0.5f;
    }
    CAIF_DeviceTensor router_logits=CAIF_DeviceTensor::FromHostData(
                                      logits_host.data(),{num_tokens,num_experts},stream);

    // Allocate outputs for MoETopKGating
    CAIF_DeviceTensor expert_indices=CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},stream,
                                       CAIF_DataType::CAIF_DataType_e::Int32);
    CAIF_DeviceTensor expert_weights=CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},stream);
    CAIF_DeviceTensor router_probs=CAIF_DeviceTensor::Uninitialized({num_tokens,num_experts},stream);

    // Test MoETopKGating
    CAIF_Ops::MoETopKGating(router_logits,num_experts,top_k,
                                  expert_indices,expert_weights,router_probs);

    // Verify outputs
    std::vector<int32_t> indices_out(num_tokens*top_k);
    std::vector<float> weights_out(num_tokens*top_k);
    expert_indices.CopyToHostRaw(indices_out.data());
    expert_weights.CopyToHost(weights_out.data());

    // Check indices are valid expert IDs
    for(size_t i=0;i<indices_out.size();++i)
    {
      const int32_t idx=indices_out[i];
      if(idx<0||idx>=static_cast<int32_t>(num_experts))
      {
        std::cout<<"  Invalid expert index: "<<idx<<"\n";
        passed=false;
        break;
      }
    }

    // Check weights are positive and sum to ~1 per token
    for(uint32_t t=0;t<num_tokens;++t)
    {
      float sum=0.0f;
      for(uint32_t k=0;k<top_k;++k)
      {
        if(weights_out[t*top_k+k]<0.0f)
        {
          std::cout<<"  Negative weight at token "<<t<<"\n";
          passed=false;
        }
        sum+=weights_out[t*top_k+k];
      }
      if(std::abs(sum-1.0f)>0.01f)
      {
        std::cout<<"  Weights don't sum to 1 for token "<<t<<": "<<sum<<"\n";
        passed=false;
      }
    }

    // Test MoEBuildDispatchMap
    CAIF_DeviceTensor dispatch_map=CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},stream);
    CAIF_DeviceTensor expert_offsets=CAIF_DeviceTensor::Uninitialized({num_experts+1},stream);

    uint32_t total_assigned=CAIF_Ops::MoEBuildDispatchMap(
                               expert_indices,num_experts,top_k,0,
                               dispatch_map,expert_offsets);

    std::cout<<"  Total assigned tokens: "<<total_assigned<<"\n";

    if(total_assigned==0||total_assigned>num_tokens*top_k)
    {
      std::cout<<"  Invalid total_assigned count\n";
      passed=false;
    }

    // Create input tokens
    std::vector<float> input_host(num_tokens*dim);
    for(size_t i=0;i<input_host.size();++i)
    {
      input_host[i]=static_cast<float>(i)/100.0f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              input_host.data(),{num_tokens,dim},stream);

    // Allocate expert buffer
    CAIF_DeviceTensor expert_buffer=CAIF_DeviceTensor::Zeros({total_assigned,dim},stream);

    // Test MoEDispatchGPU
    CAIF_Ops::MoEDispatchGPU(input,expert_indices,dispatch_map,
                                   expert_offsets,top_k,expert_buffer);

    // Verify buffer has data
    std::vector<float> buffer_host(total_assigned*dim);
    expert_buffer.CopyToHost(buffer_host.data());

    bool has_nonzero=false;
    for(size_t i=0;i<buffer_host.size();++i)
    {
      if(buffer_host[i]!=0.0f)
      {
        has_nonzero=true;
        break;
      }
    }
    if(has_nonzero==false)
    {
      std::cout<<"  Expert buffer is all zeros after dispatch\n";
      passed=false;
    }

    // Test MoECombineGPU
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Zeros({num_tokens,dim},stream);
    CAIF_Ops::MoECombineGPU(expert_buffer,expert_indices,expert_weights,
                                  dispatch_map,expert_offsets,top_k,output);

    // Verify output has data
    std::vector<float> output_host(num_tokens*dim);
    output.CopyToHost(output_host.data());

    has_nonzero=false;
    for(size_t i=0;i<output_host.size();++i)
    {
      if(output_host[i]!=0.0f)
      {
        has_nonzero=true;
        break;
      }
    }
    if(has_nonzero==false)
    {
      std::cout<<"  Output is all zeros after combine\n";
      passed=false;
    }

    ReportResult("MoE::GPUOptimizedOps",passed);
  }
  CAIF_CATCH_BLOCK()
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    std::cout<<"=== CAIF MoE Tests ===\n\n";

#ifdef USE_CAIF_CUDA
    TestExpertForwardShape();
    TestExpertGatedForwardShape();
    TestRouterOutputShape();
    TestRouterTopKIndicesMatchArgmax();
    TestRouterInitFavorExpert();
    TestMoELayerMixedSizeExperts();
    TestRouterWeightsNormalized();
    TestMoELayerForwardShape();
    TestMoELayerAuxLosses();
    TestMoELayerBackward();
    TestMoELayerParameterCount();
    TestMoELayerCapacity();
    TestExpertChoiceRouting();
    TestSoftMoERouting();
    TestOverflowNoOp();
    TestSharedExperts();
    TestSharedExpertsCustomHiddenDim();
    TestSharedExpertsBackward();
    TestGPUOptimizedMoEOps();
#else
    std::cout<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

    return CAIF_TestHarness::FinalExitCode();
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<e<<std::endl;
    return 1;
  }
  catch(const std::exception &e)
  {
    ISE_Out::ErrLog()<<"std::exception: "<<e.what()<<std::endl;
    return 1;
  }
}
