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
// Test: Pluggable projections for MLA, FFN, and MoEExpert
//------------------------------------------------------------------------------
#include "caif_device_ml_attention.h"
#include "caif_device_ffn.h"
#include "caif_device_moe_expert.h"
#include "caif_device_frozen_linear.h"
#include "caif_device_lora_adapter.h"
#include "caif_device_pointwise_activations.h"
#include "caif_device_gated_activations.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <random>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    ISE_Out::Out()<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    ISE_Out::Out()<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Helper: create FrozenLinear with random FP32 weights
//------------------------------------------------------------------------------
static std::unique_ptr<CAIF_DeviceFrozenLinear> MakeFrozenLinear(uint32_t in_dim,
                                                                uint32_t out_dim,
                                                                CAIF_CudaStream &stream,
                                                                uint32_t seed=42)
{
  auto layer=std::make_unique<CAIF_DeviceFrozenLinear>(in_dim,out_dim,
                                                      CAIF_DataType::CAIF_DataType_e::Float32,
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

//------------------------------------------------------------------------------
// Helper: wrap a FrozenLinear with LoRA
//------------------------------------------------------------------------------
static std::unique_ptr<CAIF_DeviceLoRAAdapter> MakeLoRAFrozen(uint32_t in_dim,
                                                              uint32_t out_dim,
                                                              uint32_t rank,
                                                              float alpha,
                                                              CAIF_CudaStream &stream,
                                                              uint32_t seed=42)
{
  auto frozen=MakeFrozenLinear(in_dim,out_dim,stream,seed);
  CAIF_DeviceLoRAAdapter::LoRAConfig_t lora_cfg;
  lora_cfg.rank=rank;
  lora_cfg.alpha=alpha;
  lora_cfg.input_dim=in_dim;
  lora_cfg.output_dim=out_dim;
  return std::make_unique<CAIF_DeviceLoRAAdapter>(lora_cfg,std::move(frozen),stream,seed);
}

//------------------------------------------------------------------------------
// Helper: check all values finite
//------------------------------------------------------------------------------
static bool AllFinite(const CAIF_HostTensor &h)
{
  for(size_t i=0;i<h.TotalElements();++i)
  {
    if(std::isfinite(h.Data()[i])==false)
    {
      ISE_Out::Out()<<"  Non-finite at "<<i<<": "<<h.Data()[i]<<"\n";
      return false;
    }
  }
  return true;
}

//------------------------------------------------------------------------------
// Helper: check any non-zero
//------------------------------------------------------------------------------
static bool AnyNonZero(const CAIF_HostTensor &h)
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

//------------------------------------------------------------------------------
// Helper: MLA test config (same as test_device_ml_attention.cpp)
//------------------------------------------------------------------------------
static CAIF_DeviceMLAttention::MLAConfig_t MakeMLAConfig()
{
  CAIF_DeviceMLAttention::MLAConfig_t config;
  config.dim=32;
  config.num_heads=2;
  config.q_lora_rank=16;
  config.kv_lora_rank=12;
  config.qk_nope_head_dim=24;
  config.qk_rope_head_dim=8;
  config.v_head_dim=32;
  config.causal=false;
  config.rope_base=10000.0f;
  config.rms_norm_eps=1e-5f;
  return config;
}

//==============================================================================
// MLA Projection Tests
//==============================================================================

//------------------------------------------------------------------------------
// MLA Test 1: FP32 FrozenLinear projections -- forward shape
//------------------------------------------------------------------------------
static void TestMLAProjectionsForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    auto config=MakeMLAConfig();

    // Derived dimensions
    const uint32_t qk_head_dim=config.qk_nope_head_dim+config.qk_rope_head_dim;
    const uint32_t q_proj_dim=config.num_heads*qk_head_dim;
    const uint32_t kv_compress_dim=config.kv_lora_rank+config.qk_rope_head_dim;
    const uint32_t kv_decomp_dim=config.num_heads*(config.qk_nope_head_dim+config.v_head_dim);
    const uint32_t o_input_dim=config.num_heads*config.v_head_dim;

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention::MLAProjections_t proj;
    proj.q_compress=MakeFrozenLinear(config.dim,config.q_lora_rank,stream,10);
    proj.q_decompress=MakeFrozenLinear(config.q_lora_rank,q_proj_dim,stream,20);
    proj.kv_compress=MakeFrozenLinear(config.dim,kv_compress_dim,stream,30);
    proj.kv_decompress=MakeFrozenLinear(config.kv_lora_rank,kv_decomp_dim,stream,40);
    proj.o_proj=MakeFrozenLinear(o_input_dim,config.dim,stream,50);

    CAIF_DeviceMLAttention mla(config,std::move(proj),stream);

    std::vector<float> host_input(batch*seq_len*config.dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,config.dim},stream);
    CAIF_DeviceTensor output=mla.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=config.dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    ReportResult("MLA::Projections::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::Projections::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// MLA Test 2: LoRA(FrozenLinear) projections -- forward shape + finite
//------------------------------------------------------------------------------
static void TestMLALoRAProjectionsForward()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.qk_nope_head_dim+config.qk_rope_head_dim;
    const uint32_t q_proj_dim=config.num_heads*qk_head_dim;
    const uint32_t kv_compress_dim=config.kv_lora_rank+config.qk_rope_head_dim;
    const uint32_t kv_decomp_dim=config.num_heads*(config.qk_nope_head_dim+config.v_head_dim);
    const uint32_t o_input_dim=config.num_heads*config.v_head_dim;

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.dim,config.q_lora_rank,lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.q_lora_rank,q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.dim,kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.kv_lora_rank,kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.dim,lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention mla(config,std::move(proj),stream);

    std::vector<float> host_input(batch*seq_len*config.dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,config.dim},stream);
    CAIF_DeviceTensor output=mla.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=config.dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    ReportResult("MLA::LoRAProjections::Forward",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::LoRAProjections::Forward",false);
  }
}

//------------------------------------------------------------------------------
// MLA Test 3: Projections -- ParameterTensorCount == sum of sub-layers + 2 norms
//------------------------------------------------------------------------------
static void TestMLAProjectionsParameterCount()
{
  try
  {
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.qk_nope_head_dim+config.qk_rope_head_dim;
    const uint32_t q_proj_dim=config.num_heads*qk_head_dim;
    const uint32_t kv_compress_dim=config.kv_lora_rank+config.qk_rope_head_dim;
    const uint32_t kv_decomp_dim=config.num_heads*(config.qk_nope_head_dim+config.v_head_dim);
    const uint32_t o_input_dim=config.num_heads*config.v_head_dim;

    CAIF_CudaStream stream;

    // LoRA has 2 trainable tensors (A and B), FrozenLinear has 0
    // So each LoRA-wrapped projection has 2 params
    CAIF_DeviceMLAttention::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.dim,config.q_lora_rank,lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.q_lora_rank,q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.dim,kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.kv_lora_rank,kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.dim,lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention mla(config,std::move(proj),stream);

    // 5 projections * 2 LoRA params each + 2 norm gammas = 12
    const size_t expected_count=12;
    bool passed=true;
    if(mla.ParameterTensorCount()!=expected_count)
    {
      ISE_Out::Out()<<"  Expected "<<expected_count<<", got "<<mla.ParameterTensorCount()<<"\n";
      passed=false;
    }

    ReportResult("MLA::Projections::ParameterTensorCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::Projections::ParameterTensorCount",false);
  }
}

//------------------------------------------------------------------------------
// MLA Test 4: Projections -- ParameterNames include prefixes
//------------------------------------------------------------------------------
static void TestMLAProjectionsParameterNames()
{
  try
  {
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.qk_nope_head_dim+config.qk_rope_head_dim;
    const uint32_t q_proj_dim=config.num_heads*qk_head_dim;
    const uint32_t kv_compress_dim=config.kv_lora_rank+config.qk_rope_head_dim;
    const uint32_t kv_decomp_dim=config.num_heads*(config.qk_nope_head_dim+config.v_head_dim);
    const uint32_t o_input_dim=config.num_heads*config.v_head_dim;

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.dim,config.q_lora_rank,lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.q_lora_rank,q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.dim,kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.kv_lora_rank,kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.dim,lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention mla(config,std::move(proj),stream);
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

    for(const auto &name:names)
    {
      if(name.find("attn.q_a_proj.")!=std::string::npos)
      {
        found_q_a=true;
      }
      if(name.find("attn.q_b_proj.")!=std::string::npos)
      {
        found_q_b=true;
      }
      if(name.find("attn.kv_a_proj_with_mqa.")!=std::string::npos)
      {
        found_kv_a=true;
      }
      if(name.find("attn.kv_b_proj.")!=std::string::npos)
      {
        found_kv_b=true;
      }
      if(name.find("attn.o_proj.")!=std::string::npos)
      {
        found_o=true;
      }
      if(name.find("q_a_layernorm")!=std::string::npos)
      {
        found_q_norm=true;
      }
      if(name.find("kv_a_layernorm")!=std::string::npos)
      {
        found_kv_norm=true;
      }
    }

    if(found_q_a==false)
    {
      ISE_Out::Out()<<"  Missing q_a_proj prefix\n";
      passed=false;
    }
    if(found_q_b==false)
    {
      ISE_Out::Out()<<"  Missing q_b_proj prefix\n";
      passed=false;
    }
    if(found_kv_a==false)
    {
      ISE_Out::Out()<<"  Missing kv_a_proj_with_mqa prefix\n";
      passed=false;
    }
    if(found_kv_b==false)
    {
      ISE_Out::Out()<<"  Missing kv_b_proj prefix\n";
      passed=false;
    }
    if(found_o==false)
    {
      ISE_Out::Out()<<"  Missing o_proj prefix\n";
      passed=false;
    }
    if(found_q_norm==false)
    {
      ISE_Out::Out()<<"  Missing q_a_layernorm\n";
      passed=false;
    }
    if(found_kv_norm==false)
    {
      ISE_Out::Out()<<"  Missing kv_a_layernorm\n";
      passed=false;
    }

    ReportResult("MLA::Projections::ParameterNames",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::Projections::ParameterNames",false);
  }
}

//------------------------------------------------------------------------------
// MLA Test 5: LoRA projections backward -- non-zero LoRA gradients
//------------------------------------------------------------------------------
static void TestMLALoRAProjectionsBackward()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;
    auto config=MakeMLAConfig();

    const uint32_t qk_head_dim=config.qk_nope_head_dim+config.qk_rope_head_dim;
    const uint32_t q_proj_dim=config.num_heads*qk_head_dim;
    const uint32_t kv_compress_dim=config.kv_lora_rank+config.qk_rope_head_dim;
    const uint32_t kv_decomp_dim=config.num_heads*(config.qk_nope_head_dim+config.v_head_dim);
    const uint32_t o_input_dim=config.num_heads*config.v_head_dim;

    CAIF_CudaStream stream;

    CAIF_DeviceMLAttention::MLAProjections_t proj;
    proj.q_compress=MakeLoRAFrozen(config.dim,config.q_lora_rank,lora_rank,lora_alpha,stream,10);
    proj.q_decompress=MakeLoRAFrozen(config.q_lora_rank,q_proj_dim,lora_rank,lora_alpha,stream,20);
    proj.kv_compress=MakeLoRAFrozen(config.dim,kv_compress_dim,lora_rank,lora_alpha,stream,30);
    proj.kv_decompress=MakeLoRAFrozen(config.kv_lora_rank,kv_decomp_dim,lora_rank,lora_alpha,stream,40);
    proj.o_proj=MakeLoRAFrozen(o_input_dim,config.dim,lora_rank,lora_alpha,stream,50);

    CAIF_DeviceMLAttention mla(config,std::move(proj),stream);

    std::vector<float> host_input(batch*seq_len*config.dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,config.dim},stream);
    mla.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*config.dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                              {batch,seq_len,config.dim},stream);
    CAIF_DeviceTensor grad_input=mla.Backward(grad_out);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;

    // Check input gradient is finite
    if(AllFinite(h_grad)==false)
    {
      passed=false;
    }

    // Check at least some LoRA gradients are non-zero
    bool found_nonzero_grad=false;
    for(size_t p=0;p<mla.ParameterTensorCount()&&passed==true;++p)
    {
      CAIF_HostTensor wg=mla.GradientTensor(p).ToHost();
      if(AllFinite(wg)==false)
      {
        ISE_Out::Out()<<"  Non-finite weight grad at param "<<p<<"\n";
        passed=false;
        break;
      }
      if(AnyNonZero(wg)==true)
      {
        found_nonzero_grad=true;
      }
    }
    if(found_nonzero_grad==false&&passed==true)
    {
      ISE_Out::Out()<<"  All LoRA gradients are zero\n";
      passed=false;
    }

    ReportResult("MLA::LoRAProjections::Backward",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::LoRAProjections::Backward",false);
  }
}

//==============================================================================
// FFN Projection Tests
//==============================================================================

//------------------------------------------------------------------------------
// FFN Test 1: Gated FFN with FrozenLinear projections -- forward shape + finite
//------------------------------------------------------------------------------
static void TestFFNProjectionsForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    CAIF_DeviceFFN::FFNProjections_t proj;
    proj.gate=MakeFrozenLinear(dim,ffn_dim,stream,10);
    proj.up=MakeFrozenLinear(dim,ffn_dim,stream,20);
    proj.down=MakeFrozenLinear(ffn_dim,dim,stream,30);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(proj),std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=ffn.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    ReportResult("FFN::Projections::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::Projections::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// FFN Test 2: LoRA(FrozenLinear) projections -- forward shape + finite
//------------------------------------------------------------------------------
static void TestFFNLoRAProjectionsForward()
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
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    CAIF_DeviceFFN::FFNProjections_t proj;
    proj.gate=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,10);
    proj.up=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,20);
    proj.down=MakeLoRAFrozen(ffn_dim,dim,lora_rank,lora_alpha,stream,30);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(proj),std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=ffn.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    ReportResult("FFN::LoRAProjections::Forward",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::LoRAProjections::Forward",false);
  }
}

//------------------------------------------------------------------------------
// FFN Test 3: Projections -- ParameterTensorCount correct
//------------------------------------------------------------------------------
static void TestFFNProjectionsParameterCount()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    // Frozen only (no LoRA) -- 0 params per projection
    {
      CAIF_DeviceFFN::FFNProjections_t proj;
      proj.gate=MakeFrozenLinear(dim,ffn_dim,stream,10);
      proj.up=MakeFrozenLinear(dim,ffn_dim,stream,20);
      proj.down=MakeFrozenLinear(ffn_dim,dim,stream,30);

      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn(config,std::move(proj),std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=0)
      {
        ISE_Out::Out()<<"  Frozen-only expected 0, got "<<ffn.ParameterTensorCount()<<"\n";
        ReportResult("FFN::Projections::ParameterTensorCount",false);
        return;
      }
    }

    // LoRA-wrapped -- 2 params per projection, 3 projections = 6
    {
      CAIF_DeviceFFN::FFNProjections_t proj;
      proj.gate=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,10);
      proj.up=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,20);
      proj.down=MakeLoRAFrozen(ffn_dim,dim,lora_rank,lora_alpha,stream,30);

      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn(config,std::move(proj),std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=6)
      {
        ISE_Out::Out()<<"  LoRA expected 6, got "<<ffn.ParameterTensorCount()<<"\n";
        ReportResult("FFN::Projections::ParameterTensorCount",false);
        return;
      }
    }

    ReportResult("FFN::Projections::ParameterTensorCount",true);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::Projections::ParameterTensorCount",false);
  }
}

//------------------------------------------------------------------------------
// FFN Test 4: LoRA projections backward -- non-zero gradients
//------------------------------------------------------------------------------
static void TestFFNLoRAProjectionsBackward()
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
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    CAIF_DeviceFFN::FFNProjections_t proj;
    proj.gate=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,10);
    proj.up=MakeLoRAFrozen(dim,ffn_dim,lora_rank,lora_alpha,stream,20);
    proj.down=MakeLoRAFrozen(ffn_dim,dim,lora_rank,lora_alpha,stream,30);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(proj),std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,dim},stream);
    ffn.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                              {batch,seq_len,dim},stream);
    CAIF_DeviceTensor grad_input=ffn.Backward(grad_out);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    if(AllFinite(h_grad)==false)
    {
      passed=false;
    }

    bool found_nonzero=false;
    for(size_t p=0;p<ffn.ParameterTensorCount()&&passed==true;++p)
    {
      CAIF_HostTensor wg=ffn.GradientTensor(p).ToHost();
      if(AllFinite(wg)==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at param "<<p<<"\n";
        passed=false;
        break;
      }
      if(AnyNonZero(wg)==true)
      {
        found_nonzero=true;
      }
    }
    if(found_nonzero==false&&passed==true)
    {
      ISE_Out::Out()<<"  All LoRA gradients are zero\n";
      passed=false;
    }

    ReportResult("FFN::LoRAProjections::Backward",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::LoRAProjections::Backward",false);
  }
}

//==============================================================================
// MoEExpert Projection Tests
//==============================================================================

//------------------------------------------------------------------------------
// MoEExpert Test 1: FrozenLinear projections -- forward shape + finite
//------------------------------------------------------------------------------
static void TestMoEExpertProjectionsForwardShape()
{
  try
  {
    const uint32_t num_tokens=4;
    const uint32_t input_dim=8;
    const uint32_t hidden_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpert::Config_t config;
    config.input_dim=input_dim;
    config.hidden_dim=hidden_dim;
    config.use_gated=true;
    config.use_bias=false;

    CAIF_DeviceMoEExpert::MoEExpertProjections_t proj;
    proj.gate=MakeFrozenLinear(input_dim,hidden_dim,stream,10);
    proj.up=MakeFrozenLinear(input_dim,hidden_dim,stream,20);
    proj.down=MakeFrozenLinear(hidden_dim,input_dim,stream,30);

    CAIF_DeviceMoEExpert expert(config,std::move(proj),stream);

    std::vector<float> host_input(num_tokens*input_dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {num_tokens,input_dim},stream);
    CAIF_DeviceTensor output=expert.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2||shape[0]!=num_tokens||shape[1]!=input_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    ReportResult("MoEExpert::Projections::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MoEExpert::Projections::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// MoEExpert Test 2: Projections -- ParameterTensorCount correct
//------------------------------------------------------------------------------
static void TestMoEExpertProjectionsParameterCount()
{
  try
  {
    const uint32_t input_dim=8;
    const uint32_t hidden_dim=32;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpert::Config_t config;
    config.input_dim=input_dim;
    config.hidden_dim=hidden_dim;
    config.use_gated=true;
    config.use_bias=false;

    bool passed=true;

    // Frozen only -- 0 params
    {
      CAIF_DeviceMoEExpert::MoEExpertProjections_t proj;
      proj.gate=MakeFrozenLinear(input_dim,hidden_dim,stream,10);
      proj.up=MakeFrozenLinear(input_dim,hidden_dim,stream,20);
      proj.down=MakeFrozenLinear(hidden_dim,input_dim,stream,30);

      CAIF_DeviceMoEExpert expert(config,std::move(proj),stream);
      if(expert.ParameterTensorCount()!=0)
      {
        ISE_Out::Out()<<"  Frozen-only expected 0, got "<<expert.ParameterTensorCount()<<"\n";
        passed=false;
      }
    }

    // LoRA-wrapped -- 2 per projection * 3 = 6
    {
      CAIF_DeviceMoEExpert::MoEExpertProjections_t proj;
      proj.gate=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,10);
      proj.up=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,20);
      proj.down=MakeLoRAFrozen(hidden_dim,input_dim,lora_rank,lora_alpha,stream,30);

      CAIF_DeviceMoEExpert expert(config,std::move(proj),stream);
      if(expert.ParameterTensorCount()!=6)
      {
        ISE_Out::Out()<<"  LoRA expected 6, got "<<expert.ParameterTensorCount()<<"\n";
        passed=false;
      }
    }

    ReportResult("MoEExpert::Projections::ParameterTensorCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MoEExpert::Projections::ParameterTensorCount",false);
  }
}

//------------------------------------------------------------------------------
// MoEExpert Test 3: LoRA projections backward -- finite gradients
//------------------------------------------------------------------------------
static void TestMoEExpertLoRAProjectionsBackward()
{
  try
  {
    const uint32_t num_tokens=4;
    const uint32_t input_dim=8;
    const uint32_t hidden_dim=32;
    const uint32_t lora_rank=4;
    const float lora_alpha=8.0f;

    CAIF_CudaStream stream;
    CAIF_DeviceMoEExpert::Config_t config;
    config.input_dim=input_dim;
    config.hidden_dim=hidden_dim;
    config.use_gated=true;
    config.use_bias=false;

    CAIF_DeviceMoEExpert::MoEExpertProjections_t proj;
    proj.gate=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,10);
    proj.up=MakeLoRAFrozen(input_dim,hidden_dim,lora_rank,lora_alpha,stream,20);
    proj.down=MakeLoRAFrozen(hidden_dim,input_dim,lora_rank,lora_alpha,stream,30);

    CAIF_DeviceMoEExpert expert(config,std::move(proj),stream);

    std::vector<float> host_input(num_tokens*input_dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {num_tokens,input_dim},stream);
    expert.Forward(input,true);

    std::vector<float> grad_ones(num_tokens*input_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                              {num_tokens,input_dim},stream);
    CAIF_DeviceTensor grad_input=expert.Backward(grad_out);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    if(AllFinite(h_grad)==false)
    {
      passed=false;
    }

    bool found_nonzero=false;
    for(size_t p=0;p<expert.ParameterTensorCount()&&passed==true;++p)
    {
      CAIF_HostTensor wg=expert.GradientTensor(p).ToHost();
      if(AllFinite(wg)==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at param "<<p<<"\n";
        passed=false;
        break;
      }
      if(AnyNonZero(wg)==true)
      {
        found_nonzero=true;
      }
    }
    if(found_nonzero==false&&passed==true)
    {
      ISE_Out::Out()<<"  All LoRA gradients are zero\n";
      passed=false;
    }

    ReportResult("MoEExpert::LoRAProjections::Backward",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MoEExpert::LoRAProjections::Backward",false);
  }
}

//------------------------------------------------------------------------------
// FFN Test 5: Pointwise FFN with FrozenLinear projections (no gate)
//------------------------------------------------------------------------------
static void TestFFNPointwiseProjectionsForward()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t ffn_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    CAIF_DeviceFFN::FFNProjections_t proj;
    proj.up=MakeFrozenLinear(dim,ffn_dim,stream,20);
    proj.down=MakeFrozenLinear(ffn_dim,dim,stream,30);
    // gate is nullptr for pointwise

    auto activation=std::make_unique<CAIF_DeviceGELUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(proj),std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=ffn.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    if(passed==true)
    {
      passed=AllFinite(h_out);
    }

    ReportResult("FFN::PointwiseProjections::Forward",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::PointwiseProjections::Forward",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    ISE_Out::Out()<<"=== AIF Pluggable Projections Tests ===\n\n";

#ifdef USE_CAIF_CUDA
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
#else
    ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

    ISE_Out::Out()<<"\n=== Summary ===\n";
    ISE_Out::Out()<<"Passed: "<<g_tests_passed<<"\n";
    ISE_Out::Out()<<"Failed: "<<g_tests_failed<<"\n";

    if(g_tests_failed>0)
    {
      return 1;
    }
    return 0;
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"AIF Exception: "<<e<<std::endl;
    return 1;
  }
  catch(const std::exception &e)
  {
    ISE_Out::ErrLog()<<"std::exception: "<<e.what()<<std::endl;
    return 1;
  }
}
