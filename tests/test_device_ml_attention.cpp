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
// Test: CAIF_DeviceMLAttention<float,float> (Multi-head Latent Attention)
//------------------------------------------------------------------------------
#include "caif_device_ml_attention.h"
#include "caif_test_harness.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_mla_test_dim=32;
constexpr uint32_t g_caif_mla_test_num_heads=2;
constexpr uint32_t g_caif_mla_test_q_lora_rank=16;
constexpr uint32_t g_caif_mla_test_kv_lora_rank=12;
constexpr uint32_t g_caif_mla_test_qk_nope_head_dim=24;
constexpr uint32_t g_caif_mla_test_qk_rope_head_dim=8;
constexpr uint32_t g_caif_mla_test_v_head_dim=32;
constexpr float g_caif_mla_test_rope_base=10000.0f;
constexpr float g_caif_mla_test_rms_norm_eps=1e-5f;
constexpr float g_caif_mla_test_input_scale=0.05f;
constexpr float g_caif_mla_test_input_offset_a=-0.3f;
constexpr float g_caif_mla_test_input_offset_b=-0.5f;
constexpr float g_caif_mla_test_input_offset_c=-0.2f;
constexpr float g_caif_mla_test_input_scale_b=0.1f;
constexpr float g_caif_mla_test_input_scale_c=0.03f;
constexpr float g_caif_mla_test_fd_step=1e-3f;
constexpr float g_caif_mla_test_fd_tol=5e-2f;
constexpr float g_caif_mla_test_det_tol=1e-2f;
constexpr float g_caif_mla_test_nonzero_min=1e-5f;
constexpr float g_caif_mla_test_sign_agree_ratio=0.75f;
constexpr size_t g_caif_mla_test_fd_check_count=8;
constexpr size_t g_caif_mla_test_wo_check_count=4;
constexpr size_t g_caif_mla_test_wo_idx=6;
constexpr uint32_t g_caif_mla_test_param_count=7;
constexpr uint32_t g_caif_mla_test_cache_max_seq=32;
constexpr uint32_t g_caif_mla_test_prefill_len=3;

//------------------------------------------------------------------------------
// MLA (Multi-head Latent Attention) correctness tests.
//------------------------------------------------------------------------------
class CAIF_MLAttentionTests
{
  public:
    static void RunAll();

  protected:

  private:
    // Constraints on config:
    //   v_head_dim must equal qk_nope_head_dim + qk_rope_head_dim
    //   qk_head_dim must be 32, 64, 80, 96, or 128 (flash attention kernel)
    //   qk_rope_head_dim must be even (RoPE pairs)
    static CAIF_DeviceMLAttentionConfig MakeTestConfig(bool causal=true);

    static void TestForwardShape();
    static void TestForwardFinite();
    static void TestCausalDifference();
    static void TestParameterCount();
    static void TestParameterNames();
    static void TestZeroGradients();
    static void TestBackwardFinite();
    static void TestBackwardInputGrad();
    static void TestBackwardWeightGradWo();
    static void TestDescription();
    static void TestKVCacheManagement();
    static void TestForwardCached();
    static void TestDeterministic();
    static void TestBatchIndependence();
    static void TestCachedDecodeMatchesForward();
};

CAIF_DeviceMLAttentionConfig
CAIF_MLAttentionTests::MakeTestConfig(const bool causal)
{
  CAIF_DeviceMLAttentionConfig config(g_caif_mla_test_dim,
                                      g_caif_mla_test_num_heads,
                                      g_caif_mla_test_q_lora_rank,
                                      g_caif_mla_test_kv_lora_rank,
                                      g_caif_mla_test_qk_rope_head_dim,
                                      g_caif_mla_test_qk_nope_head_dim,
                                      g_caif_mla_test_v_head_dim,
                                      causal,
                                      g_caif_mla_test_rope_base,
                                      g_caif_mla_test_rms_norm_eps);
  return config;
}

//------------------------------------------------------------------------------
// Test 1: Forward shape correctness
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mla.Forward(input,ctx);
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

    CAIF_TestHarness::Report("MLA::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Output values are finite (non-NaN, non-Inf)
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestForwardFinite()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_mla_test_input_scale+g_caif_mla_test_input_offset_a;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mla.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_output.TotalElements();++i)
    {
      if(std::isfinite(host_output.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite value at "
                      <<i
                      <<": "
                      <<host_output.Data()[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("MLA::ForwardFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::ForwardFinite")
}

//------------------------------------------------------------------------------
// Test 3: Causal vs non-causal produce different results
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestCausalDifference()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=4;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config_causal=MakeTestConfig(true);
    auto config_noncausal=MakeTestConfig(false);
    CAIF_DeviceMLAttention<float,float> mla_causal(config_causal,stream);
    CAIF_DeviceMLAttention<float,float> mla_noncausal(config_noncausal,stream);

    // Copy same weights to both
    for(size_t p=0;p<mla_causal.ParameterTensorCount();++p)
    {
      CAIF_HostTensor hw=mla_causal.ParameterTensor(p).ToHost();
      mla_noncausal.ParameterTensor(p).CopyFromHost(hw.Data(),hw.TotalElements());
    }

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_mla_test_input_scale+g_caif_mla_test_input_offset_b;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor out_causal=mla_causal.Forward(input,ctx);
    CAIF_DeviceTensor out_noncausal=mla_noncausal.Forward(input,ctx);
    CAIF_HostTensor h_causal=out_causal.ToHost();
    CAIF_HostTensor h_noncausal=out_noncausal.ToHost();

    // Causal and non-causal should produce different outputs
    // (except for seq_len=1 where causal mask has no effect)
    bool outputs_differ=false;
    for(size_t i=0;i<h_causal.TotalElements();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_causal.Data()[i],h_noncausal.Data()[i],1e-4f)==false)
      {
        outputs_differ=true;
        break;
      }
    }

    const bool passed=(outputs_differ==true);
    if(outputs_differ==false)
    {
      ISE_Out::Out()<<"  Causal and non-causal outputs are identical\n";
    }

    CAIF_TestHarness::Report("MLA::CausalDifference",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::CausalDifference")
}

//------------------------------------------------------------------------------
// Test 4: Parameter count and tensor count
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    bool passed=true;

    // 7 parameter tensors
    if(mla.ParameterTensorCount()!=g_caif_mla_test_param_count)
    {
      ISE_Out::Out()<<"  ParameterTensorCount expected "
                    <<g_caif_mla_test_param_count
                    <<", got "
                    <<mla.ParameterTensorCount()
                    <<"\n";
      passed=false;
    }

    // Compute expected total parameter count
    const uint32_t dim=config.Dim();
    const uint32_t q_lora_rank=config.QLoraRank();
    const uint32_t kv_lora_rank=config.KvLoraRank();
    const uint32_t qk_head_dim=config.QkNopeHeadDim()+config.QkRopeHeadDim();
    const uint32_t q_proj_dim=config.NumHeads()*qk_head_dim;
    const uint32_t kv_compress_dim=kv_lora_rank+config.QkRopeHeadDim();
    const uint32_t kv_decomp_dim=config.NumHeads()*(config.QkNopeHeadDim()+config.VHeadDim());
    const uint32_t o_input_dim=config.NumHeads()*config.VHeadDim();

    const size_t expected_total=dim*q_lora_rank+
                                q_lora_rank+
                                q_lora_rank*q_proj_dim+
                                dim*kv_compress_dim+
                                kv_lora_rank+
                                kv_lora_rank*kv_decomp_dim+
                                o_input_dim*dim;

    if(mla.TotalParameterCount()!=expected_total)
    {
      ISE_Out::Out()<<"  TotalParameterCount expected "
                    <<expected_total
                    <<", got "
                    <<mla.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("MLA::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 5: Parameter names match convention
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestParameterNames()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    auto names=mla.ParameterNames("self_attn.");
    bool passed=true;

    if(names.size()!=g_caif_mla_test_param_count)
    {
      ISE_Out::Out()<<"  Expected "
                    <<g_caif_mla_test_param_count
                    <<" names, got "
                    <<names.size()
                    <<"\n";
      passed=false;
    }

    const std::vector<std::string> expected_names={
      "self_attn.w_q_compress",
      "self_attn.q_norm_gamma",
      "self_attn.w_q_decompress",
      "self_attn.w_kv_compress",
      "self_attn.kv_norm_gamma",
      "self_attn.w_kv_decompress",
      "self_attn.w_o"
    };

    for(size_t i=0;i<expected_names.size()&&i<names.size();++i)
    {
      if(names[i]!=expected_names[i])
      {
        ISE_Out::Out()<<"  Name["
                      <<i
                      <<"] expected '"
                      <<expected_names[i]
                      <<"', got '"
                      <<names[i]
                      <<"'\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("MLA::ParameterNames",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::ParameterNames")
}

//------------------------------------------------------------------------------
// Test 6: ZeroGradients
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestZeroGradients()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mla.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                                {batch,seq_len,dim},
                                                                stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    mla.Backward(grad_out,ctx);

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
          ISE_Out::Out()<<"  Gradient["
                        <<p
                        <<"] not zeroed at "
                        <<i
                        <<": "
                        <<host_grad.Data()[i]
                        <<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    CAIF_TestHarness::Report("MLA::ZeroGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::ZeroGradients")
}

//------------------------------------------------------------------------------
// Test 7: Backward produces finite gradients
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestBackwardFinite()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(false);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_mla_test_input_scale_b+g_caif_mla_test_input_offset_c;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mla.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                                {batch,seq_len,dim},
                                                                stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mla.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    // Check input gradient is finite
    for(size_t i=0;i<host_grad.TotalElements();++i)
    {
      if(std::isfinite(host_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite input grad at "
                      <<i
                      <<": "
                      <<host_grad.Data()[i]
                      <<"\n";
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
          ISE_Out::Out()<<"  Non-finite weight grad["
                        <<p
                        <<"] at "
                        <<i
                        <<": "
                        <<wg.Data()[i]
                        <<"\n";
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

    CAIF_TestHarness::Report("MLA::BackwardFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::BackwardFinite")
}

//------------------------------------------------------------------------------
// Test 8: Backward input gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestBackwardInputGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(false);

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_mla_test_input_scale_b+g_caif_mla_test_input_offset_c;
    }

    // Get analytical gradient
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    // Save weights for reuse
    std::vector<CAIF_HostTensor> saved_weights;
    for(size_t p=0;p<mla.ParameterTensorCount();++p)
    {
      saved_weights.push_back(mla.ParameterTensor(p).ToHost());
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mla.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                                {batch,seq_len,dim},
                                                                stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mla.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    // Finite-difference check for a subset of input elements.
    // Use sign agreement check (flash attention backward can have larger errors).
    int sign_agree=0;
    int sign_total=0;

    const bool fd_prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    for(size_t i=0;i<g_caif_mla_test_fd_check_count&&i<host_input.size();++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=g_caif_mla_test_fd_step;
      input_minus[i]-=g_caif_mla_test_fd_step;

      // Forward with +h
      CAIF_DeviceMLAttention<float,float> mla_p(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        mla_p.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                               saved_weights[p].TotalElements());
      }

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(input_plus.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
      CAIF_DeviceTensor out_p=mla_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      CAIF_DeviceMLAttention<float,float> mla_m(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        mla_m.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                               saved_weights[p].TotalElements());
      }

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(input_minus.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
      CAIF_DeviceTensor out_m=mla_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_mla_test_fd_step);
      const float analytical=host_grad.Data()[i];

      // Check sign agreement
      if(std::fabs(numerical)>g_caif_mla_test_nonzero_min&&
         std::fabs(analytical)>g_caif_mla_test_nonzero_min)
      {
        ++sign_total;
        if((numerical>0.0f)==(analytical>0.0f))
        {
          ++sign_agree;
        }
      }
    }

    CAIF_Settings::SetPreciseGradients(fd_prev_precise);

    // Require at least 75% sign agreement
    if(sign_total>0)
    {
      const float agree_ratio=static_cast<float>(sign_agree)/
                               static_cast<float>(sign_total);
      if(agree_ratio<g_caif_mla_test_sign_agree_ratio)
      {
        ISE_Out::Out()<<"  Sign agreement: "
                      <<sign_agree
                      <<"/"
                      <<sign_total
                      <<" ("
                      <<(agree_ratio*100.0f)
                      <<"%)\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("MLA::BackwardInputGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::BackwardInputGrad")
}

//------------------------------------------------------------------------------
// Test 9: Backward W_o weight gradient (finite difference spot check)
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestBackwardWeightGradWo()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(false);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_mla_test_input_scale_b+g_caif_mla_test_input_offset_c;
    }

    // Get analytical gradient for W_o (index 6)
    CAIF_DeviceMLAttention<float,float> mla(config,stream);
    std::vector<CAIF_HostTensor> saved_weights;
    for(size_t p=0;p<mla.ParameterTensorCount();++p)
    {
      saved_weights.push_back(mla.ParameterTensor(p).ToHost());
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mla.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                                {batch,seq_len,dim},
                                                                stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    mla.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_wo=mla.GradientTensor(g_caif_mla_test_wo_idx).ToHost();

    bool passed=true;

    // Spot check first few elements of W_o gradient
    std::vector<float> wo_data(saved_weights[g_caif_mla_test_wo_idx].Data(),
                               saved_weights[g_caif_mla_test_wo_idx].Data()+
                               saved_weights[g_caif_mla_test_wo_idx].TotalElements());

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    for(size_t i=0;i<g_caif_mla_test_wo_check_count&&passed==true;++i)
    {
      std::vector<float> wo_plus(wo_data);
      std::vector<float> wo_minus(wo_data);
      wo_plus[i]+=g_caif_mla_test_fd_step;
      wo_minus[i]-=g_caif_mla_test_fd_step;

      // Forward with +h
      CAIF_DeviceMLAttention<float,float> mla_p(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        if(p==g_caif_mla_test_wo_idx)
        {
          mla_p.ParameterTensor(p).CopyFromHost(wo_plus.data(),wo_plus.size());
        }
        else
        {
          mla_p.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                                 saved_weights[p].TotalElements());
        }
      }

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
      CAIF_DeviceTensor out_p=mla_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      CAIF_DeviceMLAttention<float,float> mla_m(config,stream);
      for(size_t p=0;p<saved_weights.size();++p)
      {
        if(p==g_caif_mla_test_wo_idx)
        {
          mla_m.ParameterTensor(p).CopyFromHost(wo_minus.data(),wo_minus.size());
        }
        else
        {
          mla_m.ParameterTensor(p).CopyFromHost(saved_weights[p].Data(),
                                                 saved_weights[p].TotalElements());
        }
      }

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
      CAIF_DeviceTensor out_m=mla_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_mla_test_fd_step);
      const float analytical=host_grad_wo.Data()[i];

      if(std::fabs(numerical-analytical)>g_caif_mla_test_fd_tol)
      {
        ISE_Out::Out()<<"  dW_o mismatch at "
                      <<i
                      <<": analytical="
                      <<analytical
                      <<" numerical="
                      <<numerical
                      <<" diff="
                      <<std::fabs(numerical-analytical)
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("MLA::BackwardWeightGradWo",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::BackwardWeightGradWo")
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    const std::string desc=mla.Description();

    // Check that description contains key info
    bool passed=true;
    if(desc.find("MLA")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'MLA': "
                    <<desc
                    <<"\n";
      passed=false;
    }
    if(desc.find("dim=32")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'dim=32': "
                    <<desc
                    <<"\n";
      passed=false;
    }
    if(desc.find("heads=2")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'heads=2': "
                    <<desc
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("MLA::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::Description")
}

//------------------------------------------------------------------------------
// Test 11: KV cache enable/disable/reset
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestKVCacheManagement()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    bool passed=true;

    // Initially disabled
    if(mla.IsKVCacheEnabled()!=false)
    {
      ISE_Out::Out()<<"  KV cache should start disabled\n";
      passed=false;
    }

    // Enable
    mla.EnableKVCache(1,g_caif_mla_test_cache_max_seq);
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

    CAIF_TestHarness::Report("MLA::KVCacheManagement",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::KVCacheManagement")
}

//------------------------------------------------------------------------------
// Test 12: ForwardCached produces correct shape and advances cache
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestForwardCached()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);
    mla.EnableKVCache(batch,g_caif_mla_test_cache_max_seq);

    // Prefill with 3 tokens
    const uint32_t prefill_len=g_caif_mla_test_prefill_len;
    std::vector<float> host_prefill(batch*prefill_len*dim);
    for(size_t i=0;i<host_prefill.size();++i)
    {
      host_prefill[i]=static_cast<float>(i)*g_caif_mla_test_input_scale+g_caif_mla_test_input_offset_a;
    }
    CAIF_DeviceTensor prefill_input=CAIF_DeviceTensor::FromHostData(host_prefill.data(),
                                                                     {batch,prefill_len,dim},
                                                                     stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor prefill_out=mla.ForwardCached(prefill_input,ctx);
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
      ISE_Out::Out()<<"  KV cache length expected "
                    <<prefill_len
                    <<", got "
                    <<mla.KVCacheLength()
                    <<"\n";
      passed=false;
    }

    // Second step: one token
    std::vector<float> host_step(batch*1*dim);
    for(size_t i=0;i<host_step.size();++i)
    {
      host_step[i]=static_cast<float>(i)*g_caif_mla_test_input_scale+0.1f;
    }
    CAIF_DeviceTensor step_input=CAIF_DeviceTensor::FromHostData(host_step.data(),
                                                                   {batch,1,dim},
                                                                   stream);
    CAIF_DeviceTensor step_out=mla.ForwardCached(step_input,ctx);
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
      ISE_Out::Out()<<"  KV cache length expected "
                    <<prefill_len+1
                    <<", got "
                    <<mla.KVCacheLength()
                    <<"\n";
      passed=false;
    }

    // Check values are finite
    for(size_t i=0;i<h_step.TotalElements();++i)
    {
      if(std::isfinite(h_step.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite cached output at "
                      <<i
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("MLA::ForwardCached",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::ForwardCached")
}

//------------------------------------------------------------------------------
// Test 13: Deterministic output (same input + same weights = same output)
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestDeterministic()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_mla_test_input_scale+g_caif_mla_test_input_offset_a;
    }

    CAIF_DeviceTensor input1=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output1=mla.Forward(input1,ctx);
    CAIF_HostTensor h1=output1.ToHost();

    CAIF_DeviceTensor input2=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
    CAIF_DeviceTensor output2=mla.Forward(input2,ctx);
    CAIF_HostTensor h2=output2.ToHost();

    bool passed=true;
    for(size_t i=0;i<h1.TotalElements();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h1.Data()[i],h2.Data()[i],g_caif_mla_test_det_tol)==false)
      {
        ISE_Out::Out()<<"  Non-deterministic at "
                      <<i
                      <<": "
                      <<h1.Data()[i]
                      <<" vs "
                      <<h2.Data()[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("MLA::Deterministic",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::Deterministic")
}

//------------------------------------------------------------------------------
// Test 14: Batch independence (different batch items produce different outputs)
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestBatchIndependence()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=g_caif_mla_test_dim;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeTestConfig(true);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_mla_test_input_scale_c+g_caif_mla_test_input_offset_b;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                             {batch,seq_len,dim},
                                                             stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mla.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Check that batch 0 and batch 1 outputs differ (since inputs differ)
    bool differ=false;
    for(uint32_t i=0;i<seq_len*dim;++i)
    {
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],
                                       host_output.Data()[seq_len*dim+i],1e-4f)==false)
      {
        differ=true;
        break;
      }
    }

    const bool passed=differ;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Batch items produced identical outputs\n";
    }

    CAIF_TestHarness::Report("MLA::BatchIndependence",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::BatchIndependence")
}

//------------------------------------------------------------------------------
// Matrix-absorption decode correctness: incremental cached decode (one token at
// a time) must match the full non-cached forward over the same sequence. Step 0
// hits the standard prefill path; steps 1+ (cache_len>0, new_len==1, batch==1,
// no q-LoRA) hit the absorbed compressed-space decode. Run in Accuracy_e for a
// tight fp32 comparison — a math error blows far past the small reassociation
// gap the weight folds introduce.
//------------------------------------------------------------------------------
void CAIF_MLAttentionTests::TestCachedDecodeMatchesForward()
{
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    constexpr uint32_t batch=1;
    constexpr uint32_t seq=6;
    constexpr uint32_t dim=g_caif_mla_test_dim;
    constexpr float tol=2.0e-2f;

    CAIF_Settings::SetPreciseGradients(true);

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // q_lora_rank==0 plus a threshold of 1 (set below) exercise the absorbed
    // decode path on this short sequence; the default would keep it standard.
    CAIF_DeviceMLAttentionConfig cfg(dim,
                                     g_caif_mla_test_num_heads,
                                     0,
                                     g_caif_mla_test_kv_lora_rank,
                                     g_caif_mla_test_qk_rope_head_dim,
                                     g_caif_mla_test_qk_nope_head_dim,
                                     g_caif_mla_test_v_head_dim,
                                     true,
                                     g_caif_mla_test_rope_base,
                                     g_caif_mla_test_rms_norm_eps);
    cfg.SetDecodeAbsorbThreshold(1);
    CAIF_DeviceMLAttention<float,float> mla(cfg,stream);

    std::vector<float> host_in(static_cast<size_t>(batch)*seq*dim);
    for(size_t i=0;i<host_in.size();++i)
    {
      host_in[i]=static_cast<float>(i)*g_caif_mla_test_input_scale+g_caif_mla_test_input_offset_a;
    }
    CAIF_DeviceTensor full_in=CAIF_DeviceTensor::FromHostData(host_in.data(),{batch,seq,dim},stream);

    CAIF_DeviceTensor ref=mla.Forward(full_in,ctx);
    std::vector<float> ref_host(static_cast<size_t>(batch)*seq*dim);
    ref.CopyToHost(ref_host.data());

    mla.EnableKVCache(batch,seq);
    mla.ResetKVCache();
    bool ok=true;
    float worst=0.0f;
    for(uint32_t t=0;t<seq;++t)
    {
      std::vector<float> tok(dim);
      for(uint32_t d=0;d<dim;++d)
      {
        tok[d]=host_in[static_cast<size_t>(t)*dim+d];
      }
      CAIF_DeviceTensor tok_in=CAIF_DeviceTensor::FromHostData(tok.data(),{batch,1,dim},stream);
      CAIF_DeviceTensor step=mla.ForwardCached(tok_in,ctx);
      std::vector<float> step_host(dim);
      step.CopyToHost(step_host.data());
      for(uint32_t d=0;d<dim;++d)
      {
        const float refv=ref_host[static_cast<size_t>(t)*dim+d];
        const float stepv=step_host[d];
        const float diff=std::fabs(stepv-refv);
        if(diff>worst)
        {
          worst=diff;
        }
        float bound=tol;
        if(std::fabs(refv)>1.0f)
        {
          bound=tol*std::fabs(refv);
        }
        if(std::isfinite(stepv)==false||diff>=bound)
        {
          ok=false;
        }
      }
    }

    if(ok==false)
    {
      ISE_Out::Out()<<"  incremental decode diverged from full forward (worst "
                    <<worst
                    <<", tol "
                    <<tol
                    <<")\n";
    }
    CAIF_TestHarness::Report("MLA::CachedDecodeMatchesForward",ok);
  }
  CAIF_TEST_CATCH_BLOCK("MLA::CachedDecodeMatchesForward")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

void CAIF_MLAttentionTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DeviceMLAttention<float,float> Tests ==="
                <<"\n\n";
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
  TestCachedDecodeMatchesForward();
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
  try
  {
    instance::ISE_Out::Out()<<"=== CAIF_DeviceMLAttention<float,float> Tests ==="
                             <<"\n\n";
#ifdef USE_CAIF_CUDA
    instance::CAIF_MLAttentionTests::RunAll();
#else
    instance::ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif
    return instance::CAIF_TestHarness::FinalExitCode();
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::ISE_Out::ErrLog()<<"CAIF Exception: "
                                <<e
                                <<"\n";
    return 1;
  }
  catch(const std::exception &e)
  {
    instance::ISE_Out::ErrLog()<<"std::exception: "
                                <<e.what()
                                <<"\n";
    return 1;
  }
}
