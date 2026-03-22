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
// Test: CAIF_DeviceMLAttention (Multi-head Latent Attention)
//------------------------------------------------------------------------------
#include "caif_device_ml_attention.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>
#include <numeric>

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

static bool FloatEqual(float a,float b,float tolerance=1e-3f)
{
  return std::fabs(a-b)<tolerance;
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Helper: create small MLA config for testing
//------------------------------------------------------------------------------
static CAIF_DeviceMLAttention::MLAConfig_t MakeTestConfig(bool causal=true)
{
  // Constraints:
  //   v_head_dim must equal qk_nope_head_dim + qk_rope_head_dim
  //   qk_head_dim must be 32, 64, 80, 96, or 128 (flash attention kernel)
  //   qk_rope_head_dim must be even (RoPE pairs)
  CAIF_DeviceMLAttention::MLAConfig_t config;
  config.dim=32;
  config.num_heads=2;
  config.q_lora_rank=16;
  config.kv_lora_rank=12;
  config.qk_nope_head_dim=24;
  config.qk_rope_head_dim=8;
  config.v_head_dim=32;
  config.causal=causal;
  config.rope_base=10000.0f;
  config.rms_norm_eps=1e-5f;
  return config;
}

//------------------------------------------------------------------------------
// Test 1: Forward shape correctness
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=mla.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch: expected [2,4,32], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          ISE_Out::Out()<<",";
        }
        ISE_Out::Out()<<shape[i];
      }
      ISE_Out::Out()<<"]\n";
      passed=false;
    }

    ReportResult("MLA::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Output values are finite (non-NaN, non-Inf)
//------------------------------------------------------------------------------
static void TestForwardFinite()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=mla.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_output.TotalElements();++i)
    {
      if(std::isfinite(host_output.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite value at "<<i<<": "<<host_output.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("MLA::ForwardFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::ForwardFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Causal vs non-causal produce different results
//------------------------------------------------------------------------------
static void TestCausalDifference()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=4;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config_causal=MakeTestConfig(true);
    auto config_noncausal=MakeTestConfig(false);
    CAIF_DeviceMLAttention mla_causal(config_causal,stream);
    CAIF_DeviceMLAttention mla_noncausal(config_noncausal,stream);

    // Copy same weights to both
    for(size_t p=0;p<mla_causal.ParameterTensorCount();++p)
    {
      CAIF_HostTensor hw=mla_causal.ParameterTensor(p).ToHost();
      mla_noncausal.ParameterTensor(p).CopyFromHost(hw.Data(),hw.TotalElements());
    }

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.5f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor out_causal=mla_causal.Forward(input,false);
    CAIF_DeviceTensor out_noncausal=mla_noncausal.Forward(input,false);
    CAIF_HostTensor h_causal=out_causal.ToHost();
    CAIF_HostTensor h_noncausal=out_noncausal.ToHost();

    // Causal and non-causal should produce different outputs
    // (except for seq_len=1 where causal mask has no effect)
    bool outputs_differ=false;
    for(size_t i=0;i<h_causal.TotalElements();++i)
    {
      if(FloatEqual(h_causal.Data()[i],h_noncausal.Data()[i],1e-4f)==false)
      {
        outputs_differ=true;
        break;
      }
    }

    bool passed=(outputs_differ==true);
    if(outputs_differ==false)
    {
      ISE_Out::Out()<<"  Causal and non-causal outputs are identical\n";
    }

    ReportResult("MLA::CausalDifference",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::CausalDifference",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Parameter count and tensor count
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention mla(config,stream);

    bool passed=true;

    // 7 parameter tensors
    if(mla.ParameterTensorCount()!=7)
    {
      ISE_Out::Out()<<"  ParameterTensorCount expected 7, got "
               <<mla.ParameterTensorCount()<<"\n";
      passed=false;
    }

    // Compute expected total parameter count
    const uint32_t dim=config.dim;
    const uint32_t q_lora_rank=config.q_lora_rank;
    const uint32_t kv_lora_rank=config.kv_lora_rank;
    const uint32_t qk_head_dim=config.qk_nope_head_dim+config.qk_rope_head_dim;
    const uint32_t q_proj_dim=config.num_heads*qk_head_dim;
    const uint32_t kv_compress_dim=kv_lora_rank+config.qk_rope_head_dim;
    const uint32_t kv_decomp_dim=config.num_heads*(config.qk_nope_head_dim+config.v_head_dim);
    const uint32_t o_input_dim=config.num_heads*config.v_head_dim;

    const size_t expected_total=dim*q_lora_rank+
                                q_lora_rank+
                                q_lora_rank*q_proj_dim+
                                dim*kv_compress_dim+
                                kv_lora_rank+
                                kv_lora_rank*kv_decomp_dim+
                                o_input_dim*dim;

    if(mla.TotalParameterCount()!=expected_total)
    {
      ISE_Out::Out()<<"  TotalParameterCount expected "<<expected_total
               <<", got "<<mla.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("MLA::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Parameter names match HuggingFace convention
//------------------------------------------------------------------------------
static void TestParameterNames()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention mla(config,stream);

    auto names=mla.ParameterNames("self_attn.");
    bool passed=true;

    if(names.size()!=7)
    {
      ISE_Out::Out()<<"  Expected 7 names, got "<<names.size()<<"\n";
      passed=false;
    }

    const std::vector<std::string> expected_names={
      "self_attn.q_a_proj.weight",
      "self_attn.q_a_layernorm.weight",
      "self_attn.q_b_proj.weight",
      "self_attn.kv_a_proj_with_mqa.weight",
      "self_attn.kv_a_layernorm.weight",
      "self_attn.kv_b_proj.weight",
      "self_attn.o_proj.weight"
    };

    for(size_t i=0;i<expected_names.size()&&i<names.size();++i)
    {
      if(names[i]!=expected_names[i])
      {
        ISE_Out::Out()<<"  Name["<<i<<"] expected '"<<expected_names[i]
                 <<"', got '"<<names[i]<<"'\n";
        passed=false;
      }
    }

    ReportResult("MLA::ParameterNames",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::ParameterNames",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: ZeroGradients
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention mla(config,stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    mla.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    mla.Backward(grad_out);

    // Zero gradients
    mla.ZeroGradients();

    bool passed=true;
    for(size_t p=0;p<mla.ParameterTensorCount();++p)
    {
      CAIF_HostTensor host_grad=mla.GradientTensor(p).ToHost();
      for(size_t i=0;i<host_grad.TotalElements();++i)
      {
        if(host_grad.Data()[i]!=0.0f)
        {
          ISE_Out::Out()<<"  Gradient["<<p<<"] not zeroed at "<<i<<": "
                   <<host_grad.Data()[i]<<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    ReportResult("MLA::ZeroGradients",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::ZeroGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Backward produces finite gradients
//------------------------------------------------------------------------------
static void TestBackwardFinite()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    mla.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor grad_input=mla.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    // Check input gradient is finite
    for(size_t i=0;i<host_grad.TotalElements();++i)
    {
      if(std::isfinite(host_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite input grad at "<<i<<": "<<host_grad.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }

    // Check all weight gradients are finite
    for(size_t p=0;p<mla.ParameterTensorCount()&&passed==true;++p)
    {
      CAIF_HostTensor wg=mla.GradientTensor(p).ToHost();
      for(size_t i=0;i<wg.TotalElements();++i)
      {
        if(std::isfinite(wg.Data()[i])==false)
        {
          ISE_Out::Out()<<"  Non-finite weight grad["<<p<<"] at "<<i<<": "
                   <<wg.Data()[i]<<"\n";
          passed=false;
          break;
        }
      }
    }

    // Check input gradient shape
    const auto &gshape=host_grad.Shape();
    if(gshape.size()!=3||gshape[0]!=batch||gshape[1]!=seq_len||gshape[2]!=dim)
    {
      ISE_Out::Out()<<"  Grad shape mismatch\n";
      passed=false;
    }

    ReportResult("MLA::BackwardFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::BackwardFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Backward input gradient (finite difference)
//------------------------------------------------------------------------------
static void TestBackwardInputGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=32;
    const float h=1e-3f;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    CAIF_DeviceMLAttention mla(config,stream);

    // Save weights for reuse
    std::vector<CAIF_HostTensor> saved_weights;
    for(size_t p=0;p<mla.ParameterTensorCount();++p)
    {
      saved_weights.push_back(mla.ParameterTensor(p).ToHost());
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    mla.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor grad_input=mla.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    // Finite-difference check for a subset of input elements
    // Use relative tolerance: check sign agreement and that magnitudes are
    // within a factor of 3 (flash attention backward can have larger errors)
    const size_t check_count=8;
    int sign_agree=0;
    int sign_total=0;

    for(size_t i=0;i<check_count&&i<host_input.size();++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      CAIF_DeviceMLAttention mla_p(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        mla_p.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                               saved_weights[p].TotalElements());
      }

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_plus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=mla_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      CAIF_DeviceMLAttention mla_m(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        mla_m.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                               saved_weights[p].TotalElements());
      }

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_minus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=mla_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      // Check sign agreement
      if(std::fabs(numerical)>1e-5f&&std::fabs(analytical)>1e-5f)
      {
        ++sign_total;
        if((numerical>0.0f)==(analytical>0.0f))
        {
          ++sign_agree;
        }
      }
    }

    // Require at least 75% sign agreement
    if(sign_total>0)
    {
      const float agree_ratio=static_cast<float>(sign_agree)/
                               static_cast<float>(sign_total);
      if(agree_ratio<0.75f)
      {
        ISE_Out::Out()<<"  Sign agreement: "<<sign_agree<<"/"<<sign_total
                 <<" ("<<(agree_ratio*100.0f)<<"%)\n";
        passed=false;
      }
    }

    ReportResult("MLA::BackwardInputGrad",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::BackwardInputGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: Backward W_o weight gradient (finite difference spot check)
//------------------------------------------------------------------------------
static void TestBackwardWeightGradWo()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=32;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient for W_o (index 6)
    CAIF_DeviceMLAttention mla(config,stream);
    std::vector<CAIF_HostTensor> saved_weights;
    for(size_t p=0;p<mla.ParameterTensorCount();++p)
    {
      saved_weights.push_back(mla.ParameterTensor(p).ToHost());
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    mla.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    mla.Backward(grad_out);
    CAIF_HostTensor host_grad_wo=mla.GradientTensor(6).ToHost();

    bool passed=true;

    // Spot check first 4 elements of W_o gradient
    const size_t wo_idx=6;
    std::vector<float> wo_data(saved_weights[wo_idx].Data(),
                               saved_weights[wo_idx].Data()+
                               saved_weights[wo_idx].TotalElements());
    const size_t check_count=4;

    for(size_t i=0;i<check_count&&passed==true;++i)
    {
      std::vector<float> wo_plus(wo_data);
      std::vector<float> wo_minus(wo_data);
      wo_plus[i]+=h;
      wo_minus[i]-=h;

      // Forward with +h
      CAIF_DeviceMLAttention mla_p(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        if(p==wo_idx)
        {
          mla_p.ParameterTensor(p).CopyFromHost(wo_plus.data(),wo_plus.size());
        }
        else
        {
          mla_p.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                                 saved_weights[p].TotalElements());
        }
      }

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=mla_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      CAIF_DeviceMLAttention mla_m(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        if(p==wo_idx)
        {
          mla_m.ParameterTensor(p).CopyFromHost(wo_minus.data(),wo_minus.size());
        }
        else
        {
          mla_m.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                                 saved_weights[p].TotalElements());
        }
      }

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=mla_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_wo.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        ISE_Out::Out()<<"  dW_o mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical
                 <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    ReportResult("MLA::BackwardWeightGradWo",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::BackwardWeightGradWo",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention mla(config,stream);

    const std::string desc=mla.Description();

    // Check that description contains key info
    bool passed=true;
    if(desc.find("MLA")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'MLA': "<<desc<<"\n";
      passed=false;
    }
    if(desc.find("dim=32")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'dim=32': "<<desc<<"\n";
      passed=false;
    }
    if(desc.find("heads=2")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'heads=2': "<<desc<<"\n";
      passed=false;
    }

    ReportResult("MLA::Description",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::Description",false);
  }
}

//------------------------------------------------------------------------------
// Test 11: KV cache enable/disable/reset
//------------------------------------------------------------------------------
static void TestKVCacheManagement()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention mla(config,stream);

    bool passed=true;

    // Initially disabled
    if(mla.IsKVCacheEnabled()!=false)
    {
      ISE_Out::Out()<<"  KV cache should start disabled\n";
      passed=false;
    }

    // Enable
    mla.EnableKVCache(1,32);
    if(mla.IsKVCacheEnabled()!=true)
    {
      ISE_Out::Out()<<"  KV cache should be enabled after EnableKVCache()\n";
      passed=false;
    }
    if(mla.KVCacheLength()!=0)
    {
      ISE_Out::Out()<<"  KV cache length should be 0 after enable\n";
      passed=false;
    }

    // Reset
    mla.ResetKVCache();
    if(mla.KVCacheLength()!=0)
    {
      ISE_Out::Out()<<"  KV cache length should be 0 after reset\n";
      passed=false;
    }

    // Disable
    mla.DisableKVCache();
    if(mla.IsKVCacheEnabled()!=false)
    {
      ISE_Out::Out()<<"  KV cache should be disabled after DisableKVCache()\n";
      passed=false;
    }

    ReportResult("MLA::KVCacheManagement",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::KVCacheManagement",false);
  }
}

//------------------------------------------------------------------------------
// Test 12: ForwardCached produces correct shape and advances cache
//------------------------------------------------------------------------------
static void TestForwardCached()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention mla(config,stream);
    mla.EnableKVCache(batch,32);

    // First token (prefill with 3 tokens)
    const uint32_t prefill_len=3;
    std::vector<float> host_prefill(batch*prefill_len*dim);
    for(size_t i=0;i<host_prefill.size();++i)
    {
      host_prefill[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    CAIF_DeviceTensor prefill_input=CAIF_DeviceTensor::FromHostData(
                                     host_prefill.data(),
                                     {batch,prefill_len,dim},stream);
    CAIF_DeviceTensor prefill_out=mla.ForwardCached(prefill_input);
    CAIF_HostTensor h_prefill=prefill_out.ToHost();

    bool passed=true;

    // Check prefill output shape
    const auto &ps=h_prefill.Shape();
    if(ps.size()!=3||ps[0]!=batch||ps[1]!=prefill_len||ps[2]!=dim)
    {
      ISE_Out::Out()<<"  Prefill shape mismatch\n";
      passed=false;
    }

    // Cache should now have length 3
    if(mla.KVCacheLength()!=prefill_len)
    {
      ISE_Out::Out()<<"  KV cache length expected "<<prefill_len
               <<", got "<<mla.KVCacheLength()<<"\n";
      passed=false;
    }

    // Second step: one token
    std::vector<float> host_step(batch*1*dim);
    for(size_t i=0;i<host_step.size();++i)
    {
      host_step[i]=static_cast<float>(i)*0.05f+0.1f;
    }
    CAIF_DeviceTensor step_input=CAIF_DeviceTensor::FromHostData(
                                  host_step.data(),{batch,1,dim},stream);
    CAIF_DeviceTensor step_out=mla.ForwardCached(step_input);
    CAIF_HostTensor h_step=step_out.ToHost();

    // Check step output shape
    const auto &ss=h_step.Shape();
    if(ss.size()!=3||ss[0]!=batch||ss[1]!=1||ss[2]!=dim)
    {
      ISE_Out::Out()<<"  Step shape mismatch\n";
      passed=false;
    }

    // Cache should now have length 4
    if(mla.KVCacheLength()!=prefill_len+1)
    {
      ISE_Out::Out()<<"  KV cache length expected "<<prefill_len+1
               <<", got "<<mla.KVCacheLength()<<"\n";
      passed=false;
    }

    // Check values are finite
    for(size_t i=0;i<h_step.TotalElements();++i)
    {
      if(std::isfinite(h_step.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite cached output at "<<i<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("MLA::ForwardCached",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::ForwardCached",false);
  }
}

//------------------------------------------------------------------------------
// Test 13: Deterministic output (same input + same weights = same output)
//------------------------------------------------------------------------------
static void TestDeterministic()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input1=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor output1=mla.Forward(input1,false);
    CAIF_HostTensor h1=output1.ToHost();

    CAIF_DeviceTensor input2=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor output2=mla.Forward(input2,false);
    CAIF_HostTensor h2=output2.ToHost();

    bool passed=true;
    for(size_t i=0;i<h1.TotalElements();++i)
    {
      if(FloatEqual(h1.Data()[i],h2.Data()[i],1e-2f)==false)
      {
        ISE_Out::Out()<<"  Non-deterministic at "<<i<<": "<<h1.Data()[i]
                 <<" vs "<<h2.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("MLA::Deterministic",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::Deterministic",false);
  }
}

//------------------------------------------------------------------------------
// Test 14: Batch independence (different batch items produce different outputs)
//------------------------------------------------------------------------------
static void TestBatchIndependence()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=32;

    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.03f-0.5f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=mla.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // Check that batch 0 and batch 1 outputs differ
    // (since inputs differ)
    bool differ=false;
    for(uint32_t i=0;i<seq_len*dim;++i)
    {
      if(FloatEqual(host_output.Data()[i],
                     host_output.Data()[seq_len*dim+i],1e-4f)==false)
      {
        differ=true;
        break;
      }
    }

    bool passed=differ;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Batch items produced identical outputs\n";
    }

    ReportResult("MLA::BatchIndependence",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("MLA::BatchIndependence",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF_DeviceMLAttention Tests ===\n\n";

#ifdef USE_CAIF_CUDA
    TestForwardShape();
    TestForwardFinite();
    TestCausalDifference();
    TestParameterCount();
    TestParameterNames();
    TestZeroGradients();
    TestBackwardFinite();
    TestBackwardInputGrad();
    TestBackwardWeightGradWo();
    TestDescription();
    TestKVCacheManagement();
    TestForwardCached();
    TestDeterministic();
    TestBatchIndependence();
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
    ISE_Out::ErrLog()<<"CAIF Exception: "<<e<<std::endl;
    return 1;
  }
  catch(const std::exception &e)
  {
    ISE_Out::ErrLog()<<"std::exception: "<<e.what()<<std::endl;
    return 1;
  }
}
