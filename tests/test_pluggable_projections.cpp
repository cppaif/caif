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
// Test: Pluggable projections for MLA, FFN, and MoEExpert
//------------------------------------------------------------------------------
#include "caif_device_ml_attention.h"
#include "caif_test_harness.h"
#include "caif_device_ffn.h"
#include "caif_device_moe_expert.h"
#include "caif_device_frozen_linear.h"
#include "caif_device_lora_adapter.h"
#include "caif_device_gelu_activation.h"
#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <random>

namespace instance
{

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Pluggable projection correctness tests for MLA, FFN, and MoEExpert.
//------------------------------------------------------------------------------
class CAIF_PluggableProjectionsTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::unique_ptr<CAIF_DeviceFrozenLinear<float,float>> MakeFrozenLinear(
                                                                   uint32_t in_dim,
                                                                   uint32_t out_dim,
                                                                   CAIF_CudaStream &stream,
                                                                   uint32_t seed=42);
    static std::unique_ptr<CAIF_DeviceLoRAAdapter<float,float>> MakeLoRAFrozen(
                                                                  uint32_t in_dim,
                                                                  uint32_t out_dim,
                                                                  uint32_t rank,
                                                                  float alpha,
                                                                  CAIF_CudaStream &stream,
                                                                  uint32_t seed=42);
    static bool AllFinite(const CAIF_HostTensor &h);
    static bool AnyNonZero(const CAIF_HostTensor &h);
    static CAIF_DeviceMLAttentionConfig MakeMLAConfig();

    // MLA tests
    static void TestMLAProjectionsForwardShape();
    static void TestMLALoRAProjectionsForward();
    static void TestMLAProjectionsParameterCount();
    static void TestMLAProjectionsParameterNames();
    static void TestMLALoRAProjectionsBackward();

    // FFN tests
    static void TestFFNProjectionsForwardShape();
    static void TestFFNLoRAProjectionsForward();
    static void TestFFNProjectionsParameterCount();
    static void TestFFNLoRAProjectionsBackward();
    static void TestFFNPointwiseProjectionsForward();

    // MoEExpert tests
    static void TestMoEExpertProjectionsForwardShape();
    static void TestMoEExpertProjectionsParameterCount();
    static void TestMoEExpertLoRAProjectionsBackward();
};

std::unique_ptr<CAIF_DeviceFrozenLinear<float,float>>
CAIF_PluggableProjectionsTests::MakeFrozenLinear(const uint32_t in_dim,
                                                  const uint32_t out_dim,
                                                  CAIF_CudaStream &stream,
                                                  const uint32_t seed)
{
  auto layer=std::make_unique<CAIF_DeviceFrozenLinear<float,float>>(in_dim,
                                                                     out_dim,
                                                                     stream,
                                                                     g_caif_quant_default_group_size,
                                                                     false);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.3f,0.3f);
  const size_t count=static_cast<size_t>(in_dim)*out_dim;
  std::vector<float> w(count);
  for(size_t i=0;i<count;++i)
  {
    w[i]=dist(gen);
  }
  CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),{in_dim,out_dim},stream);
  layer->LoadFromTensor(std::move(weight));
  return layer;
}

std::unique_ptr<CAIF_DeviceLoRAAdapter<float,float>>
CAIF_PluggableProjectionsTests::MakeLoRAFrozen(const uint32_t in_dim,
                                                const uint32_t out_dim,
                                                const uint32_t rank,
                                                const float alpha,
                                                CAIF_CudaStream &stream,
                                                const uint32_t seed)
{
  auto frozen=MakeFrozenLinear(in_dim,out_dim,stream,seed);
  CAIF_DeviceLoRAAdapterConfig lora_cfg(rank,alpha,in_dim,out_dim);
  return std::make_unique<CAIF_DeviceLoRAAdapter<float,float>>(lora_cfg,std::move(frozen),stream,seed);
}

bool CAIF_PluggableProjectionsTests::AllFinite(const CAIF_HostTensor &h)
{
  for(size_t i=0;i<h.TotalElements();++i)
  {
    if(std::isfinite(h.Data()[i])==false)
    {
      ISE_Out::Out()<<"  Non-finite at "
                    <<i
                    <<": "
                    <<h.Data()[i]
                    <<"\n";
      return false;
    }
  }
  return true;
}

bool CAIF_PluggableProjectionsTests::AnyNonZero(const CAIF_HostTensor &h)
{
  for(size_t i=0;i<h.TotalElements();++i)
  {
    if(h.Data()[i]!=0.0f)
    {
      return true;
    }
  }
  return false;
}

CAIF_DeviceMLAttentionConfig CAIF_PluggableProjectionsTests::MakeMLAConfig()
{
  CAIF_DeviceMLAttentionConfig config(32,2,16,12,8,24,32,false,10000.0f,1e-5f);
  return config;
}

//==============================================================================
// MLA Projection Tests
//==============================================================================

//------------------------------------------------------------------------------
// MLA Test 1: FP32 FrozenLinear projections -- forward shape
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMLAProjectionsForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    auto config=MakeMLAConfig();

    // Derived dimensions
    const uint32_t qk_head_dim=config.QkNopeHeadDim()+config.QkRopeHeadDim();
    const uint32_t q_proj_dim=config.NumHeads()*qk_head_dim;
    const uint32_t kv_compress_dim=config.KvLoraRank()+config.QkRopeHeadDim();
    const uint32_t kv_decomp_dim=config.NumHeads()*(config.QkNopeHeadDim()+config.VHeadDim());
    const uint32_t o_input_dim=config.NumHeads()*config.VHeadDim();

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention<float,float>::MLAProjections_t proj;
    proj.q_compress=MakeFrozenLinear(config.Dim(),config.QLoraRank(),stream,10);
    proj.q_decompress=MakeFrozenLinear(config.QLoraRank(),q_proj_dim,stream,20);
    proj.kv_compress=MakeFrozenLinear(config.Dim(),kv_compress_dim,stream,30);
    proj.kv_decompress=MakeFrozenLinear(config.KvLoraRank(),kv_decomp_dim,stream,40);
    proj.o_proj=MakeFrozenLinear(o_input_dim,config.Dim(),stream,50);

    CAIF_DeviceMLAttention<float,float> mla(config,std::move(proj),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*config.Dim(),0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,config.Dim()},
                                                            stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mla.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3 || shape[0]!=batch || shape[1]!=seq_len || shape[2]!=config.Dim())
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    CAIF_TestHarness::Report("MLA::Projections::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::Projections::ForwardShape")
}

//------------------------------------------------------------------------------
// MLA Test 2: LoRA(FrozenLinear) projections -- forward shape + finite
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMLALoRAProjectionsForward()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.QkNopeHeadDim()+config.QkRopeHeadDim();
    const uint32_t q_proj_dim=config.NumHeads()*qk_head_dim;
    const uint32_t kv_compress_dim=config.KvLoraRank()+config.QkRopeHeadDim();
    const uint32_t kv_decomp_dim=config.NumHeads()*(config.QkNopeHeadDim()+config.VHeadDim());
    const uint32_t o_input_dim=config.NumHeads()*config.VHeadDim();

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention<float,float>::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.Dim(),config.QLoraRank(),lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.QLoraRank(),q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.Dim(),kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.KvLoraRank(),kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.Dim(),lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention<float,float> mla(config,std::move(proj),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*config.Dim());
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,config.Dim()},
                                                            stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mla.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3 || shape[0]!=batch || shape[1]!=seq_len || shape[2]!=config.Dim())
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    CAIF_TestHarness::Report("MLA::LoRAProjections::Forward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::LoRAProjections::Forward")
}

//------------------------------------------------------------------------------
// MLA Test 3: Projections -- ParameterTensorCount == sum of sub-layers + 2 norms
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMLAProjectionsParameterCount()
{
  try
  {
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.QkNopeHeadDim()+config.QkRopeHeadDim();
    const uint32_t q_proj_dim=config.NumHeads()*qk_head_dim;
    const uint32_t kv_compress_dim=config.KvLoraRank()+config.QkRopeHeadDim();
    const uint32_t kv_decomp_dim=config.NumHeads()*(config.QkNopeHeadDim()+config.VHeadDim());
    const uint32_t o_input_dim=config.NumHeads()*config.VHeadDim();

    CAIF_CudaStream stream;

    // LoRA has 2 trainable tensors (A and B), FrozenLinear has 0
    // So each LoRA-wrapped projection has 2 params
    CAIF_DeviceMLAttention<float,float>::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.Dim(),config.QLoraRank(),lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.QLoraRank(),q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.Dim(),kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.KvLoraRank(),kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.Dim(),lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention<float,float> mla(config,std::move(proj),stream);

    // 5 projections * 2 LoRA params each + 2 norm gammas = 12
    const size_t expected_count=12;
    bool passed=true;
    if(mla.ParameterTensorCount()!=expected_count)
    {
      ISE_Out::Out()<<"  Expected "
                    <<expected_count
                    <<", got "
                    <<mla.ParameterTensorCount()
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("MLA::Projections::ParameterTensorCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::Projections::ParameterTensorCount")
}

//------------------------------------------------------------------------------
// MLA Test 4: Projections -- ParameterNames include prefixes
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMLAProjectionsParameterNames()
{
  try
  {
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.QkNopeHeadDim()+config.QkRopeHeadDim();
    const uint32_t q_proj_dim=config.NumHeads()*qk_head_dim;
    const uint32_t kv_compress_dim=config.KvLoraRank()+config.QkRopeHeadDim();
    const uint32_t kv_decomp_dim=config.NumHeads()*(config.QkNopeHeadDim()+config.VHeadDim());
    const uint32_t o_input_dim=config.NumHeads()*config.VHeadDim();

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention<float,float>::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.Dim(),config.QLoraRank(),lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.QLoraRank(),q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.Dim(),kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.KvLoraRank(),kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.Dim(),lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention<float,float> mla(config,std::move(proj),stream);
    auto names=mla.ParameterNames("attn.");

    bool passed=true;

    // Check that names contain expected projection prefixes
    bool found_q_a=false;
    bool found_q_b=false;
    bool found_kv_a=false;
    bool found_kv_b=false;
    bool found_o=false;
    bool found_q_norm=false;
    bool found_kv_norm=false;

    // The MLA layer emits CAIF-neutral role names from the role registry
    // (HuggingFace/safetensors names are the weight-mapper's responsibility),
    // so the prefixes checked here are the role tags, not the HF DSv2 names.
    for(const auto &name:names)
    {
      if(name.find("attn.w_q_compress.")!=std::string::npos)
      {
        found_q_a=true;
      }
      if(name.find("attn.w_q_decompress.")!=std::string::npos)
      {
        found_q_b=true;
      }
      if(name.find("attn.w_kv_compress.")!=std::string::npos)
      {
        found_kv_a=true;
      }
      if(name.find("attn.w_kv_decompress.")!=std::string::npos)
      {
        found_kv_b=true;
      }
      if(name.find("attn.w_o.")!=std::string::npos)
      {
        found_o=true;
      }
      if(name.find("q_norm_gamma")!=std::string::npos)
      {
        found_q_norm=true;
      }
      if(name.find("kv_norm_gamma")!=std::string::npos)
      {
        found_kv_norm=true;
      }
    }

    if(found_q_a==false)
    {
      ISE_Out::Out()<<"  Missing w_q_compress prefix\n";
      passed=false;
    }
    if(found_q_b==false)
    {
      ISE_Out::Out()<<"  Missing w_q_decompress prefix\n";
      passed=false;
    }
    if(found_kv_a==false)
    {
      ISE_Out::Out()<<"  Missing w_kv_compress prefix\n";
      passed=false;
    }
    if(found_kv_b==false)
    {
      ISE_Out::Out()<<"  Missing w_kv_decompress prefix\n";
      passed=false;
    }
    if(found_o==false)
    {
      ISE_Out::Out()<<"  Missing w_o prefix\n";
      passed=false;
    }
    if(found_q_norm==false)
    {
      ISE_Out::Out()<<"  Missing q_norm_gamma\n";
      passed=false;
    }
    if(found_kv_norm==false)
    {
      ISE_Out::Out()<<"  Missing kv_norm_gamma\n";
      passed=false;
    }

    CAIF_TestHarness::Report("MLA::Projections::ParameterNames",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::Projections::ParameterNames")
}

//------------------------------------------------------------------------------
// MLA Test 5: LoRA projections backward -- non-zero LoRA gradients
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMLALoRAProjectionsBackward()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.QkNopeHeadDim()+config.QkRopeHeadDim();
    const uint32_t q_proj_dim=config.NumHeads()*qk_head_dim;
    const uint32_t kv_compress_dim=config.KvLoraRank()+config.QkRopeHeadDim();
    const uint32_t kv_decomp_dim=config.NumHeads()*(config.QkNopeHeadDim()+config.VHeadDim());
    const uint32_t o_input_dim=config.NumHeads()*config.VHeadDim();

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention<float,float>::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.Dim(),config.QLoraRank(),lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.QLoraRank(),q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.Dim(),kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.KvLoraRank(),kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.Dim(),lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention<float,float> mla(config,std::move(proj),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*config.Dim());
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,config.Dim()},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mla.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*config.Dim(),1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,config.Dim()},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mla.Backward(grad_out,ctx);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;

    // Check input gradient is finite
    if(AllFinite(h_grad)==false)
    {
      passed=false;
    }

    // Check at least some LoRA gradients are non-zero
    bool found_nonzero_grad=false;
    for(size_t p=0;p<mla.ParameterTensorCount() && passed==true;++p)
    {
      CAIF_HostTensor wg=mla.GradientTensor(p).ToHost();
      if(AllFinite(wg)==false)
      {
        ISE_Out::Out()<<"  Non-finite weight grad at param "
                      <<p
                      <<"\n";
        passed=false;
        break;
      }
      if(AnyNonZero(wg)==true)
      {
        found_nonzero_grad=true;
      }
    }
    if(found_nonzero_grad==false && passed==true)
    {
      ISE_Out::Out()<<"  All LoRA gradients are zero\n";
      passed=false;
    }

    CAIF_TestHarness::Report("MLA::LoRAProjections::Backward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::LoRAProjections::Backward")
}

//==============================================================================
// FFN Projection Tests
//==============================================================================

//------------------------------------------------------------------------------
// FFN Test 1: Gated FFN with FrozenLinear projections -- forward shape + finite
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestFFNProjectionsForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    CAIF_DeviceFFN<float,float>::FFNProjections_t proj;
    proj.gate=MakeFrozenLinear(dim,ffn_dim,stream,10);
    proj.up=MakeFrozenLinear(dim,ffn_dim,stream,20);
    proj.down=MakeFrozenLinear(ffn_dim,dim,stream,30);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(proj),std::move(activation),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3 || shape[0]!=batch || shape[1]!=seq_len || shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    CAIF_TestHarness::Report("FFN::Projections::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::Projections::ForwardShape")
}

//------------------------------------------------------------------------------
// FFN Test 2: LoRA(FrozenLinear) projections -- forward shape + finite
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestFFNLoRAProjectionsForward()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    CAIF_DeviceFFN<float,float>::FFNProjections_t proj;
    proj.gate=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,10);
    proj.up=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,20);
    proj.down=MakeLoRAFrozen(ffn_dim,dim,lora_rank,lora_alpha,stream,30);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(proj),std::move(activation),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3 || shape[0]!=batch || shape[1]!=seq_len || shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    CAIF_TestHarness::Report("FFN::LoRAProjections::Forward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::LoRAProjections::Forward")
}

//------------------------------------------------------------------------------
// FFN Test 3: Projections -- ParameterTensorCount correct
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestFFNProjectionsParameterCount()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    // Frozen only (no LoRA) -- 0 params per projection
    {
      CAIF_DeviceFFN<float,float>::FFNProjections_t proj;
      proj.gate=MakeFrozenLinear(dim,ffn_dim,stream,10);
      proj.up=MakeFrozenLinear(dim,ffn_dim,stream,20);
      proj.down=MakeFrozenLinear(ffn_dim,dim,stream,30);

      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn(config,std::move(proj),std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=0)
      {
        ISE_Out::Out()<<"  Frozen-only expected 0, got "
                      <<ffn.ParameterTensorCount()
                      <<"\n";
        CAIF_TestHarness::Report("FFN::Projections::ParameterTensorCount",false);
        return;
      }
    }

    // LoRA-wrapped -- 2 params per projection, 3 projections = 6
    {
      CAIF_DeviceFFN<float,float>::FFNProjections_t proj;
      proj.gate=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,10);
      proj.up=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,20);
      proj.down=MakeLoRAFrozen(ffn_dim,dim,lora_rank,lora_alpha,stream,30);

      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn(config,std::move(proj),std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=6)
      {
        ISE_Out::Out()<<"  LoRA expected 6, got "
                      <<ffn.ParameterTensorCount()
                      <<"\n";
        CAIF_TestHarness::Report("FFN::Projections::ParameterTensorCount",false);
        return;
      }
    }

    CAIF_TestHarness::Report("FFN::Projections::ParameterTensorCount",true);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::Projections::ParameterTensorCount")
}

//------------------------------------------------------------------------------
// FFN Test 4: LoRA projections backward -- non-zero gradients
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestFFNLoRAProjectionsBackward()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    CAIF_DeviceFFN<float,float>::FFNProjections_t proj;
    proj.gate=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,10);
    proj.up=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,20);
    proj.down=MakeLoRAFrozen(ffn_dim,dim,lora_rank,lora_alpha,stream,30);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(proj),std::move(activation),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    ffn.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=ffn.Backward(grad_out,ctx);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    if(AllFinite(h_grad)==false)
    {
      passed=false;
    }

    bool found_nonzero=false;
    for(size_t p=0;p<ffn.ParameterTensorCount() && passed==true;++p)
    {
      CAIF_HostTensor wg=ffn.GradientTensor(p).ToHost();
      if(AllFinite(wg)==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at param "
                      <<p
                      <<"\n";
        passed=false;
        break;
      }
      if(AnyNonZero(wg)==true)
      {
        found_nonzero=true;
      }
    }
    if(found_nonzero==false && passed==true)
    {
      ISE_Out::Out()<<"  All LoRA gradients are zero\n";
      passed=false;
    }

    CAIF_TestHarness::Report("FFN::LoRAProjections::Backward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::LoRAProjections::Backward")
}

//------------------------------------------------------------------------------
// FFN Test 5: Pointwise FFN with FrozenLinear projections (no gate)
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestFFNPointwiseProjectionsForward()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t ffn_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    CAIF_DeviceFFN<float,float>::FFNProjections_t proj;
    proj.up=MakeFrozenLinear(dim,ffn_dim,stream,20);
    proj.down=MakeFrozenLinear(ffn_dim,dim,stream,30);
    // gate is nullptr for pointwise

    auto activation=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(proj),std::move(activation),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3 || shape[0]!=batch || shape[1]!=seq_len || shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    CAIF_TestHarness::Report("FFN::PointwiseProjections::Forward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::PointwiseProjections::Forward")
}

//==============================================================================
// MoEExpert Projection Tests
//==============================================================================

//------------------------------------------------------------------------------
// MoEExpert Test 1: FrozenLinear projections -- forward shape + finite
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMoEExpertProjectionsForwardShape()
{
  try
  {
    const uint32_t num_tokens=4;
    const uint32_t input_dim=8;
    const uint32_t hidden_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpertConfig config(input_dim,hidden_dim,true,false);

    CAIF_DeviceMoEExpert<float,float>::MoEExpertProjections_t proj;
    proj.gate=MakeFrozenLinear(input_dim,hidden_dim,stream,10);
    proj.up=MakeFrozenLinear(input_dim,hidden_dim,stream,20);
    proj.down=MakeFrozenLinear(hidden_dim,input_dim,stream,30);

    CAIF_DeviceMoEExpert<float,float> expert(config,std::move(proj),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(num_tokens*input_dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {num_tokens,input_dim},
                                                            stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=expert.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2 || shape[0]!=num_tokens || shape[1]!=input_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    CAIF_TestHarness::Report("MoEExpert::Projections::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MoEExpert::Projections::ForwardShape")
}

//------------------------------------------------------------------------------
// MoEExpert Test 2: Projections -- ParameterTensorCount correct
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMoEExpertProjectionsParameterCount()
{
  try
  {
    const uint32_t input_dim=8;
    const uint32_t hidden_dim=32;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpertConfig config(input_dim,hidden_dim,true,false);

    bool passed=true;

    // Frozen only -- 0 params
    {
      CAIF_DeviceMoEExpert<float,float>::MoEExpertProjections_t proj;
      proj.gate=MakeFrozenLinear(input_dim,hidden_dim,stream,10);
      proj.up=MakeFrozenLinear(input_dim,hidden_dim,stream,20);
      proj.down=MakeFrozenLinear(hidden_dim,input_dim,stream,30);

      CAIF_DeviceMoEExpert<float,float> expert(config,std::move(proj),stream);
      if(expert.ParameterTensorCount()!=0)
      {
        ISE_Out::Out()<<"  Frozen-only expected 0, got "
                      <<expert.ParameterTensorCount()
                      <<"\n";
        passed=false;
      }
    }

    // LoRA-wrapped -- 2 per projection * 3 = 6
    {
      CAIF_DeviceMoEExpert<float,float>::MoEExpertProjections_t proj;
      proj.gate=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,10);
      proj.up=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,20);
      proj.down=MakeLoRAFrozen(hidden_dim,input_dim,lora_rank,lora_alpha,stream,30);

      CAIF_DeviceMoEExpert<float,float> expert(config,std::move(proj),stream);
      if(expert.ParameterTensorCount()!=6)
      {
        ISE_Out::Out()<<"  LoRA expected 6, got "
                      <<expert.ParameterTensorCount()
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("MoEExpert::Projections::ParameterTensorCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MoEExpert::Projections::ParameterTensorCount")
}

//------------------------------------------------------------------------------
// MoEExpert Test 3: LoRA projections backward -- finite gradients
//------------------------------------------------------------------------------
void CAIF_PluggableProjectionsTests::TestMoEExpertLoRAProjectionsBackward()
{
  try
  {
    const uint32_t num_tokens=4;
    const uint32_t input_dim=8;
    const uint32_t hidden_dim=32;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpertConfig config(input_dim,hidden_dim,true,false);

    CAIF_DeviceMoEExpert<float,float>::MoEExpertProjections_t proj;
    proj.gate=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,10);
    proj.up=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,20);
    proj.down=MakeLoRAFrozen(hidden_dim,input_dim,lora_rank,lora_alpha,stream,30);

    CAIF_DeviceMoEExpert<float,float> expert(config,std::move(proj),stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(num_tokens*input_dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {num_tokens,input_dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    expert.Forward(input,ctx);

    std::vector<float> grad_ones(num_tokens*input_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {num_tokens,input_dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=expert.Backward(grad_out,ctx);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    if(AllFinite(h_grad)==false)
    {
      passed=false;
    }

    bool found_nonzero=false;
    for(size_t p=0;p<expert.ParameterTensorCount() && passed==true;++p)
    {
      CAIF_HostTensor wg=expert.GradientTensor(p).ToHost();
      if(AllFinite(wg)==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at param "
                      <<p
                      <<"\n";
        passed=false;
        break;
      }
      if(AnyNonZero(wg)==true)
      {
        found_nonzero=true;
      }
    }
    if(found_nonzero==false && passed==true)
    {
      ISE_Out::Out()<<"  All LoRA gradients are zero\n";
      passed=false;
    }

    CAIF_TestHarness::Report("MoEExpert::LoRAProjections::Backward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MoEExpert::LoRAProjections::Backward")
}

void CAIF_PluggableProjectionsTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF Pluggable Projections Tests ===\n\n";
  // MLA tests
  TestMLAProjectionsForwardShape();
  TestMLALoRAProjectionsForward();
  TestMLAProjectionsParameterCount();
  TestMLAProjectionsParameterNames();
  TestMLALoRAProjectionsBackward();

  // FFN tests
  TestFFNProjectionsForwardShape();
  TestFFNLoRAProjectionsForward();
  TestFFNProjectionsParameterCount();
  TestFFNLoRAProjectionsBackward();
  TestFFNPointwiseProjectionsForward();

  // MoEExpert tests
  TestMoEExpertProjectionsForwardShape();
  TestMoEExpertProjectionsParameterCount();
  TestMoEExpertLoRAProjectionsBackward();

  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_PluggableProjectionsTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
