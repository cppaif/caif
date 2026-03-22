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
// AIF - AI Framework
// MoE (Mixture of Experts) Tests
//------------------------------------------------------------------------------
#include "caif_device_moe_layer.h"
#include "caif_device_moe_router.h"
#include "caif_device_moe_expert.h"
#include "caif_device_moe_block.h"
#include "caif_device_moe_transformer_model.h"
#include "caif_device_ops.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    std::cout<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    std::cout<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
}

static bool FloatEqual(float a,float b,float tolerance=1e-3f)
{
  return std::fabs(a-b)<tolerance;
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
    CAIF_DeviceMoEExpert::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.use_gated=false;
    config.use_bias=true;

    CAIF_DeviceMoEExpert expert(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch*seq_len,dim},stream);

    CAIF_DeviceTensor output=expert.Forward(input,false);
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
  CCAIF_CATCH_BLOCK()
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
    CAIF_DeviceMoEExpert::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.use_gated=true;
    config.use_bias=true;

    CAIF_DeviceMoEExpert expert(config,stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceTensor output=expert.Forward(input,false);
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
  CCAIF_CATCH_BLOCK()
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
    CAIF_DeviceMoERouter::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter::RoutingType_e::TopK;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter router(config,stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter::RouterOutput_t output=router.Route(input,false);

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
  CCAIF_CATCH_BLOCK()
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
    CAIF_DeviceMoERouter::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter::RoutingType_e::TopK;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter router(config,stream);

    std::vector<float> host_input(num_tokens*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.01f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter::RouterOutput_t output=router.Route(input,false);
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
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=0;
    config.shared_hidden_dim=0;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=1.5f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=moe.Forward(input,false);
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
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=0;
    config.shared_hidden_dim=0;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.01f;
    config.z_loss_weight=0.001f;

    CAIF_DeviceMoELayer moe(config,stream);

    std::vector<float> host_input(num_tokens*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.01f-0.5f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoELayer::MoEOutput_t output=moe.ForwardMoE(input,true);

    bool passed=true;

    // Balance loss should be non-negative
    if(output.balance_loss<0.0f)
    {
      std::cout<<"  Balance loss is negative: "<<output.balance_loss<<"\n";
      passed=false;
    }

    // Z-loss should be non-negative
    if(output.z_loss<0.0f)
    {
      std::cout<<"  Z-loss is negative: "<<output.z_loss<<"\n";
      passed=false;
    }

    // Expert counts should sum to approximately num_tokens * top_k
    CAIF_HostTensor counts=output.expert_counts.ToHost();
    float total_count=0.0f;
    for(uint32_t i=0;i<num_experts;++i)
    {
      total_count+=counts.Data()[i];
    }

    // With capacity limits, might be less than num_tokens * top_k
    if(total_count<=0.0f)
    {
      std::cout<<"  No tokens routed to any expert\n";
      passed=false;
    }

    std::cout<<"  Balance loss: "<<output.balance_loss<<"\n";
    std::cout<<"  Z-loss: "<<output.z_loss<<"\n";
    std::cout<<"  Total routed: "<<total_count<<" (expected ~"<<(num_tokens*top_k)<<")\n";

    ReportResult("MoELayer::AuxLosses",passed);
  }
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=0;
    config.shared_hidden_dim=0;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    // Forward with training=true
    moe.Forward(input,true);

    // Backward
    std::vector<float> grad_ones(num_tokens*dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_ones.data(),{num_tokens,dim},stream);
    CAIF_DeviceTensor grad_input=moe.Backward(grad_output);
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
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=0;
    config.shared_hidden_dim=0;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=1.5f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

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
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=0;
    config.shared_hidden_dim=0;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=0.5f;  // Intentionally low to trigger overflow
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoELayer::MoEOutput_t output=moe.ForwardMoE(input,false);
    CAIF_HostTensor counts=output.expert_counts.ToHost();

    bool passed=true;

    // With capacity_factor=0.5, capacity = (32/4)*0.5*1 = 4 per expert
    // Total should be less than num_tokens*top_k=32 due to drops
    float total_count=0.0f;
    for(uint32_t i=0;i<num_experts;++i)
    {
      total_count+=counts.Data()[i];
      std::cout<<"  Expert "<<i<<" count: "<<counts.Data()[i]<<"\n";
    }

    // Max possible is num_experts * capacity = 4 * 4 = 16
    if(total_count>num_experts*4+1)  // +1 for rounding
    {
      std::cout<<"  Total count "<<total_count<<" exceeds expected capacity\n";
      passed=false;
    }

    std::cout<<"  Total routed: "<<total_count<<" (capacity limit ~16)\n";

    ReportResult("MoELayer::CapacityEnforcement",passed);
  }
  CCAIF_CATCH_BLOCK()
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
    CAIF_DeviceMoERouter::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter::RoutingType_e::ExpertChoice;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter router(config,stream);

    std::vector<float> host_input(num_tokens*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.01f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter::RouterOutput_t output=router.Route(input,false);

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
  CCAIF_CATCH_BLOCK()
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
    CAIF_DeviceMoERouter::Config_t config;
    config.input_dim=dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.routing_type=CAIF_DeviceMoERouter::RoutingType_e::Soft;
    config.use_bias=false;
    config.noise_std=0.0f;

    CAIF_DeviceMoERouter router(config,stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceMoERouter::RouterOutput_t output=router.Route(input,false);

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
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=0;
    config.shared_hidden_dim=0;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=0.5f;  // Low capacity to trigger overflow
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::NoOp;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

    std::vector<float> host_input(num_tokens*dim,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    // Should not crash with NoOp overflow
    CAIF_DeviceTensor output=moe.Forward(input,false);

    bool passed=true;

    // Check output shape
    const auto &shape=output.Shape();
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=dim)
    {
      std::cout<<"  Output shape mismatch\n";
      passed=false;
    }

    ReportResult("MoELayer::OverflowNoOp",passed);
  }
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=num_shared;
    config.shared_hidden_dim=0;  // Use default hidden_dim
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

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
    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceTensor output=moe.Forward(input,false);
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
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=num_shared;
    config.shared_hidden_dim=shared_hidden;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

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
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 15: Fine-grained experts
//------------------------------------------------------------------------------
static void TestFineGrainedExperts()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t hidden_dim=16;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;
    const uint32_t fine_factor=2;  // 2x experts at 1/2 hidden dim

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=0;
    config.shared_hidden_dim=0;
    config.fine_grained=true;
    config.fine_grained_factor=fine_factor;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

    bool passed=true;

    // Fine-grained: actual_num_experts = num_experts * factor = 4 * 2 = 8
    // actual_hidden_dim = hidden_dim / factor = 16 / 2 = 8
    const uint32_t actual_num_experts=num_experts*fine_factor;
    const uint32_t actual_hidden_dim=hidden_dim/fine_factor;

    // Check IsFineGrained accessor
    if(moe.IsFineGrained()!=true)
    {
      std::cout<<"  IsFineGrained() should return true\n";
      passed=false;
    }

    // Router routes to actual_num_experts
    // Router params: dim * actual_num_experts = 8 * 8 = 64
    // Each expert: dim*actual_hidden + actual_hidden + actual_hidden*dim + dim
    //            = 8*8 + 8 + 8*8 + 8 = 64 + 8 + 64 + 8 = 144
    // Total: 64 + 8*144 = 64 + 1152 = 1216
    const size_t expected_total=dim*actual_num_experts+
                                actual_num_experts*(dim*actual_hidden_dim+actual_hidden_dim+
                                                    actual_hidden_dim*dim+dim);

    if(moe.TotalParameterCount()!=expected_total)
    {
      std::cout<<"  TotalParameterCount expected "<<expected_total
               <<", got "<<moe.TotalParameterCount()<<"\n";
      passed=false;
    }

    // Test forward pass
    const uint32_t num_tokens=8;
    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    CAIF_DeviceTensor output=moe.Forward(input,false);

    // Check output shape
    const auto &shape=output.Shape();
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=dim)
    {
      std::cout<<"  Output shape mismatch\n";
      passed=false;
    }

    ReportResult("MoELayer::FineGrainedExperts",passed);
  }
  CCAIF_CATCH_BLOCK()
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

    CAIF_CudaStream stream;
    CAIF_DeviceMoELayer::Config_t config;
    config.input_dim=dim;
    config.hidden_dim=hidden_dim;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_use_gated=false;
    config.expert_use_bias=true;
    config.num_shared_experts=num_shared;
    config.shared_hidden_dim=0;
    config.fine_grained=false;
    config.fine_grained_factor=1;
    config.router_use_bias=false;
    config.router_noise_std=0.0f;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;

    CAIF_DeviceMoELayer moe(config,stream);

    std::vector<float> host_input(num_tokens*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{num_tokens,dim},stream);

    // Forward with training=true
    moe.Forward(input,true);

    // Backward
    std::vector<float> grad_ones(num_tokens*dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_ones.data(),{num_tokens,dim},stream);
    CAIF_DeviceTensor grad_input=moe.Backward(grad_output);
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
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 17: MoE Block forward
//------------------------------------------------------------------------------
static void TestMoEBlockForward()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=32;
    const uint32_t num_heads=4;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;
    const uint32_t expert_ffn_dim=64;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEBlock::Config_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.dropout_rate=0.0f;
    config.causal=true;
    config.use_rope=true;
    config.rope_base=10000.0f;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_ffn_dim=expert_ffn_dim;
    config.expert_use_gated=true;
    config.num_shared_experts=0;
    config.shared_ffn_dim=0;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.0f;
    config.z_loss_weight=0.0f;
    config.router_noise_std=0.0f;

    CAIF_DeviceMoEBlock block(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      std::cout<<"  Shape mismatch: expected ["<<batch<<","<<seq_len<<","<<dim<<"]\n";
      passed=false;
    }

    ReportResult("MoEBlock::Forward",passed);
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 18: MoE Block aux losses
//------------------------------------------------------------------------------
static void TestMoEBlockAuxLosses()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=32;
    const uint32_t num_heads=4;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;
    const uint32_t expert_ffn_dim=64;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEBlock::Config_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.dropout_rate=0.0f;
    config.causal=true;
    config.use_rope=true;
    config.rope_base=10000.0f;
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_ffn_dim=expert_ffn_dim;
    config.expert_use_gated=true;
    config.num_shared_experts=0;
    config.shared_ffn_dim=0;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.01f;
    config.z_loss_weight=0.001f;
    config.router_noise_std=0.0f;

    CAIF_DeviceMoEBlock block(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.01f-0.5f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    block.Forward(input,true);
    const auto &aux=block.LastAuxLosses();

    bool passed=true;

    // Balance loss should be non-negative
    if(aux.balance_loss<0.0f)
    {
      std::cout<<"  Balance loss is negative: "<<aux.balance_loss<<"\n";
      passed=false;
    }

    // Z-loss should be non-negative
    if(aux.z_loss<0.0f)
    {
      std::cout<<"  Z-loss is negative: "<<aux.z_loss<<"\n";
      passed=false;
    }

    std::cout<<"  Balance loss: "<<aux.balance_loss<<"\n";
    std::cout<<"  Z-loss: "<<aux.z_loss<<"\n";

    ReportResult("MoEBlock::AuxLosses",passed);
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 19: MoE Transformer Model - all MoE layers
//------------------------------------------------------------------------------
static void TestMoETransformerModelAllMoE()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=8;
    const uint32_t vocab_size=100;
    const uint32_t dim=32;
    const uint32_t num_heads=4;
    const uint32_t num_layers=2;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;
    CAIF_DeviceMoETransformerModel::Config_t config;
    config.vocab_size=vocab_size;
    config.max_seq_len=seq_len;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.num_layers=num_layers;
    config.ffn_dim=64;
    config.causal=true;
    config.use_rope=true;
    config.rope_base=10000.0f;
    config.pe_mode=PositionalEncodingMode_e::Learned;
    config.output_dim=vocab_size;
    config.tie_weights=false;
    config.moe_layer_interval=1;  // All layers are MoE
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_ffn_dim=64;
    config.expert_use_gated=true;
    config.num_shared_experts=0;
    config.shared_ffn_dim=0;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.01f;
    config.z_loss_weight=0.001f;
    config.router_noise_std=0.0f;

    CAIF_DeviceMoETransformerModel model(config,stream);

    bool passed=true;

    // Check layer counts
    if(model.NumMoELayers()!=num_layers)
    {
      std::cout<<"  NumMoELayers expected "<<num_layers<<", got "<<model.NumMoELayers()<<"\n";
      passed=false;
    }
    if(model.NumDenseLayers()!=0)
    {
      std::cout<<"  NumDenseLayers expected 0, got "<<model.NumDenseLayers()<<"\n";
      passed=false;
    }

    // Create input token IDs
    std::vector<float> host_input(batch*seq_len);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%vocab_size);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len},stream);

    // Forward pass
    CAIF_DeviceTensor output=model.Forward(input,true);
    CAIF_HostTensor host_output=output.ToHost();

    // Check output shape [batch, seq_len, vocab_size]
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=vocab_size)
    {
      std::cout<<"  Output shape mismatch\n";
      passed=false;
    }

    // Check aux losses accumulated
    const auto &aux=model.TotalAuxLosses();
    std::cout<<"  Total balance loss: "<<aux.balance_loss<<"\n";
    std::cout<<"  Total z-loss: "<<aux.z_loss<<"\n";

    if(aux.balance_loss<0.0f||aux.z_loss<0.0f)
    {
      std::cout<<"  Negative aux losses\n";
      passed=false;
    }

    ReportResult("MoETransformerModel::AllMoE",passed);
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Test 20: MoE Transformer Model - interleaved layers
//------------------------------------------------------------------------------
static void TestMoETransformerModelInterleaved()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=8;
    const uint32_t vocab_size=100;
    const uint32_t dim=32;
    const uint32_t num_heads=4;
    const uint32_t num_layers=4;
    const uint32_t num_experts=4;
    const uint32_t top_k=2;

    CAIF_CudaStream stream;
    CAIF_DeviceMoETransformerModel::Config_t config;
    config.vocab_size=vocab_size;
    config.max_seq_len=seq_len;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.num_layers=num_layers;
    config.ffn_dim=64;
    config.causal=true;
    config.use_rope=true;
    config.rope_base=10000.0f;
    config.pe_mode=PositionalEncodingMode_e::Learned;
    config.output_dim=vocab_size;
    config.tie_weights=false;
    config.moe_layer_interval=2;  // Every 2nd layer is MoE (layers 1, 3)
    config.num_experts=num_experts;
    config.top_k=top_k;
    config.expert_ffn_dim=64;
    config.expert_use_gated=true;
    config.num_shared_experts=0;
    config.shared_ffn_dim=0;
    config.capacity_factor=2.0f;
    config.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
    config.balance_loss_weight=0.01f;
    config.z_loss_weight=0.001f;
    config.router_noise_std=0.0f;

    CAIF_DeviceMoETransformerModel model(config,stream);

    bool passed=true;

    // With interval=2, layers 1 and 3 should be MoE (0-indexed)
    // Layer 0: (0+1)%2 = 1 != 0, not MoE
    // Layer 1: (1+1)%2 = 0, MoE
    // Layer 2: (2+1)%2 = 1 != 0, not MoE
    // Layer 3: (3+1)%2 = 0, MoE
    if(model.NumMoELayers()!=2)
    {
      std::cout<<"  NumMoELayers expected 2, got "<<model.NumMoELayers()<<"\n";
      passed=false;
    }
    if(model.NumDenseLayers()!=2)
    {
      std::cout<<"  NumDenseLayers expected 2, got "<<model.NumDenseLayers()<<"\n";
      passed=false;
    }

    // Check IsMoELayer
    if(model.IsMoELayer(0)==true)
    {
      std::cout<<"  Layer 0 should not be MoE\n";
      passed=false;
    }
    if(model.IsMoELayer(1)==false)
    {
      std::cout<<"  Layer 1 should be MoE\n";
      passed=false;
    }
    if(model.IsMoELayer(2)==true)
    {
      std::cout<<"  Layer 2 should not be MoE\n";
      passed=false;
    }
    if(model.IsMoELayer(3)==false)
    {
      std::cout<<"  Layer 3 should be MoE\n";
      passed=false;
    }

    // Forward pass
    std::vector<float> host_input(batch*seq_len);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%vocab_size);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len},stream);

    CAIF_DeviceTensor output=model.Forward(input,false);

    // Check output shape
    const auto &shape=output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=vocab_size)
    {
      std::cout<<"  Output shape mismatch\n";
      passed=false;
    }

    std::cout<<"  Interleaved: "<<model.NumDenseLayers()<<" dense + "
             <<model.NumMoELayers()<<" MoE layers\n";

    ReportResult("MoETransformerModel::Interleaved",passed);
  }
  CCAIF_CATCH_BLOCK()
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
    CAIF_DeviceTensor expert_indices=CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},stream);
    CAIF_DeviceTensor expert_weights=CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},stream);
    CAIF_DeviceTensor router_probs=CAIF_DeviceTensor::Uninitialized({num_tokens,num_experts},stream);

    // Test MoETopKGating
    CAIF_DeviceOps::MoETopKGating(router_logits,num_experts,top_k,
                                  expert_indices,expert_weights,router_probs);

    // Verify outputs
    std::vector<float> indices_out(num_tokens*top_k);
    std::vector<float> weights_out(num_tokens*top_k);
    expert_indices.CopyToHost(indices_out.data());
    expert_weights.CopyToHost(weights_out.data());

    // Check indices are valid expert IDs
    for(size_t i=0;i<indices_out.size();++i)
    {
      const int32_t idx=static_cast<int32_t>(indices_out[i]);
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

    uint32_t total_assigned=CAIF_DeviceOps::MoEBuildDispatchMap(
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
    CAIF_DeviceOps::MoEDispatchGPU(input,expert_indices,dispatch_map,
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
    CAIF_DeviceOps::MoECombineGPU(expert_buffer,expert_indices,expert_weights,
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
  CCAIF_CATCH_BLOCK()
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    std::cout<<"=== AIF MoE Tests ===\n\n";

#ifdef USE_CAIF_CUDA
    TestExpertForwardShape();
    TestExpertGatedForwardShape();
    TestRouterOutputShape();
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
    TestFineGrainedExperts();
    TestSharedExpertsBackward();
    TestMoEBlockForward();
    TestMoEBlockAuxLosses();
    TestMoETransformerModelAllMoE();
    TestMoETransformerModelInterleaved();
    TestGPUOptimizedMoEOps();
#else
    std::cout<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

    std::cout<<"\n=== Summary ===\n";
    std::cout<<"Passed: "<<g_tests_passed<<"\n";
    std::cout<<"Failed: "<<g_tests_failed<<"\n";

    if(g_tests_failed>0)
    {
      return 1;
    }
    return 0;
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
