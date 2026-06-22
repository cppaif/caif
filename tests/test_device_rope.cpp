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
// Tests for RoPE encoding in CAIF_DeviceMultiHeadAttention<float,float>.
//------------------------------------------------------------------------------
#include "caif_device_multi_head_attention.h"
#include "caif_test_harness.h"
#include "caif_device_network.h"
#include "caif_device_tensor.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cuda_kernels_attention_support.cuh"
#include "caif_constants.h"
#include "caif_cpu_reference/caif_cpu_softmax.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
#include "caif_cpu_reference/caif_cpu_rope.h"
#include "caif_grad_mode.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_rope_test_batch=1;
constexpr uint32_t g_caif_rope_test_seq=3;
constexpr uint32_t g_caif_rope_test_dim=8;
constexpr uint32_t g_caif_rope_test_heads=2;
constexpr float g_caif_rope_test_fwd_tol=1e-3f;
constexpr float g_caif_rope_test_fd_h=1e-3f;
constexpr int g_caif_rope_test_kernel_bh=2;
constexpr int g_caif_rope_test_kernel_seq=3;
constexpr int g_caif_rope_test_kernel_hd=4;
constexpr int g_caif_rope_test_kernel_style=0;
constexpr float g_caif_rope_test_diff_eps=1e-5f;
constexpr size_t g_caif_rope_test_wgrad_spot=4;

//------------------------------------------------------------------------------
// RoPE encoding tests for MHA with RoPE enabled.
//------------------------------------------------------------------------------
class CAIF_RoPETests
{
  public:
    static void RunAll();

  protected:

  private:
    typedef CAIF_DeviceMultiHeadAttentionConfig AttentionConfig_t;

    //--------------------------------------------------------------------------
    // CPU reference MHA with RoPE.
    // input: [batch, seq_len, dim]
    // weights: w_q, w_k, w_v [dim, num_heads*head_dim], w_o [num_heads*head_dim, dim]
    // output: [batch, seq_len, dim]
    //--------------------------------------------------------------------------
    static void CpuMHAWithRoPE(const float *input,
                                const float *w_q,
                                const float *w_k,
                                const float *w_v,
                                const float *w_o,
                                float *output,
                                int batch,
                                int seq_len,
                                int dim,
                                int num_heads,
                                int head_dim,
                                bool causal,
                                float rope_base);

    // Helper: create RoPE MHA config.
    static AttentionConfig_t MakeRoPEConfig(uint32_t dim,
                                            uint32_t num_heads,
                                            bool causal,
                                            bool use_rope);

    static void TestRoPEForwardShape();
    static void TestRoPEForwardDifference();
    static void TestRoPEForwardVsCPU();
    static void TestRoPEForwardCausal();
    static void TestRoPEBackwardInputGrad(const GradMode_t &mode);
    static void TestRoPEBackwardWeightGrad(const GradMode_t &mode);
    static void TestRoPEPositionEquivariance();
    static void TestRoPEDescriptionString();

    //--------------------------------------------------------------------------
    // Step-2 diagnostic: same shape as RoPE::BackwardInputGrad but use_rope=false.
    // If this PASSES and the RoPE version FAILS, bug is rope-chain-specific in
    // MHA Backward. If both fail, layout/FD/shape issue independent of RoPE.
    //--------------------------------------------------------------------------
    static void TestMHANoRoPEBackwardInputGradTiny(const GradMode_t &mode,
                                                   uint32_t seq_len,
                                                   uint32_t dim,
                                                   uint32_t num_heads,
                                                   const char *shape_label);

    //--------------------------------------------------------------------------
    // Isolated RoPE kernel gradcheck: verifies launch_rope_forward/backward
    // without any MHA surrounding it. If this passes and RoPE::BackwardInputGrad
    // still fails, the bug is in the MHA chain, not the rope kernel.
    //--------------------------------------------------------------------------
    static void TestRoPEKernelIsolated(const GradMode_t &mode);
};

void CAIF_RoPETests::CpuMHAWithRoPE(const float *input,
                                      const float *w_q,
                                      const float *w_k,
                                      const float *w_v,
                                      const float *w_o,
                                      float *output,
                                      const int batch,
                                      const int seq_len,
                                      const int dim,
                                      const int num_heads,
                                      const int head_dim,
                                      const bool causal,
                                      const float rope_base)
{
  const int bs=batch*seq_len;
  const int qk_dim=num_heads*head_dim;

  // Project Q, K, V
  std::vector<float> q_proj(bs*qk_dim);
  std::vector<float> k_proj(bs*qk_dim);
  std::vector<float> v_proj(bs*qk_dim);
  CAIF_CpuMatMul::Apply(input,w_q,q_proj.data(),bs,dim,qk_dim);
  CAIF_CpuMatMul::Apply(input,w_k,k_proj.data(),bs,dim,qk_dim);
  CAIF_CpuMatMul::Apply(input,w_v,v_proj.data(),bs,dim,qk_dim);

  // Split heads and compute attention per head
  std::vector<float> concat(bs*qk_dim,0.0f);

  for(int b=0;b<batch;++b)
  {
    for(int h=0;h<num_heads;++h)
    {
      // Extract Q, K, V for this head: [seq_len, head_dim]
      std::vector<float> q_head(seq_len*head_dim);
      std::vector<float> k_head(seq_len*head_dim);
      std::vector<float> v_head(seq_len*head_dim);

      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          const int flat_idx=(b*seq_len+s)*qk_dim+h*head_dim+d;
          q_head[s*head_dim+d]=q_proj[flat_idx];
          k_head[s*head_dim+d]=k_proj[flat_idx];
          v_head[s*head_dim+d]=v_proj[flat_idx];
        }
      }

      // Apply RoPE to Q and K heads
      CAIF_CpuRoPE::Apply(q_head.data(),1,seq_len,head_dim,rope_base);
      CAIF_CpuRoPE::Apply(k_head.data(),1,seq_len,head_dim,rope_base);

      // scores = Q @ K^T -> [seq_len, seq_len]
      std::vector<float> scores(seq_len*seq_len);
      CAIF_CpuMatMul::TransposeB(q_head.data(),k_head.data(),scores.data(),
                                  seq_len,head_dim,seq_len);

      // Scale
      const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
      for(int i=0;i<seq_len*seq_len;++i)
      {
        scores[i]*=scale;
      }

      // Causal mask
      if(causal==true)
      {
        for(int i=0;i<seq_len;++i)
        {
          for(int j=i+1;j<seq_len;++j)
          {
            scores[i*seq_len+j]=-1e9f;
          }
        }
      }

      // Softmax
      CAIF_CpuSoftmax::Apply(scores.data(),seq_len,seq_len);

      // context = attn @ V -> [seq_len, head_dim]
      std::vector<float> ctx(seq_len*head_dim);
      CAIF_CpuMatMul::Apply(scores.data(),v_head.data(),ctx.data(),
                             seq_len,seq_len,head_dim);

      // Write to concat
      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          concat[(b*seq_len+s)*qk_dim+h*head_dim+d]=ctx[s*head_dim+d];
        }
      }
    }
  }

  // Output projection
  CAIF_CpuMatMul::Apply(concat.data(),w_o,output,bs,qk_dim,dim);
}

CAIF_RoPETests::AttentionConfig_t CAIF_RoPETests::MakeRoPEConfig(
  const uint32_t dim,const uint32_t num_heads,const bool causal,const bool use_rope)
{
  AttentionConfig_t config(dim,num_heads,num_heads,dim/num_heads,causal,use_rope,g_caif_rope_default_base,0.0f);
  return config;
}

//------------------------------------------------------------------------------
// Test 1: RoPE forward preserves output shape
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEForwardShape()
{
  try
  {
    const uint32_t batch=g_caif_rope_test_batch;
    const uint32_t seq_len=g_caif_rope_test_seq;
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    AttentionConfig_t config=MakeRoPEConfig(dim,num_heads,false,true);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }

    CAIF_TestHarness::Report("RoPE::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RoPE::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: RoPE output differs from non-RoPE (same weights)
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEForwardDifference()
{
  try
  {
    const uint32_t batch=g_caif_rope_test_batch;
    const uint32_t seq_len=g_caif_rope_test_seq;
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Create two MHA layers with same weights but different RoPE setting
    AttentionConfig_t config_rope=MakeRoPEConfig(dim,num_heads,false,true);
    AttentionConfig_t config_norope=MakeRoPEConfig(dim,num_heads,false,false);
    CAIF_DeviceMultiHeadAttention<float,float> mha_rope(config_rope,stream);
    CAIF_DeviceMultiHeadAttention<float,float> mha_norope(config_norope,stream);

    // Copy weights from rope to norope
    for(size_t p=0;p<mha_rope.ParameterTensorCount();++p)
    {
      CAIF_HostTensor hw=mha_rope.ParameterTensor(p).ToHost();
      mha_norope.ParameterTensor(p).CopyFromHost(hw.Data(),hw.TotalElements());
    }

    // Same input
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
    CAIF_DeviceTensor out_rope=mha_rope.Forward(input,ctx);
    CAIF_DeviceTensor out_norope=mha_norope.Forward(input,ctx);

    CAIF_HostTensor h_rope=out_rope.ToHost();
    CAIF_HostTensor h_norope=out_norope.ToHost();

    // They should differ
    bool any_diff=false;
    for(size_t i=0;i<h_rope.TotalElements();++i)
    {
      if(std::fabs(h_rope.Data()[i]-h_norope.Data()[i])>g_caif_rope_test_diff_eps)
      {
        any_diff=true;
        break;
      }
    }

    CAIF_TestHarness::Report("RoPE::ForwardDifference",any_diff);
  }
  CAIF_TEST_CATCH_BLOCK("RoPE::ForwardDifference")
}

//------------------------------------------------------------------------------
// Test 3: RoPE forward matches CPU reference
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEForwardVsCPU()
{
  try
  {
    const uint32_t batch=g_caif_rope_test_batch;
    const uint32_t seq_len=g_caif_rope_test_seq;
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;
    const uint32_t head_dim=dim/num_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    AttentionConfig_t config=MakeRoPEConfig(dim,num_heads,false,true);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

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
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference
    std::vector<float> expected(batch*seq_len*dim);
    CpuMHAWithRoPE(host_input.data(),
                   h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                   expected.data(),
                   static_cast<int>(batch),static_cast<int>(seq_len),
                   static_cast<int>(dim),static_cast<int>(num_heads),
                   static_cast<int>(head_dim),false,
                   g_caif_rope_default_base);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],expected[i],
                                      g_caif_rope_test_fwd_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": got "
                      <<host_output.Data()[i]
                      <<" expected "
                      <<expected[i]
                      <<" diff="
                      <<std::fabs(host_output.Data()[i]-expected[i])
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("RoPE::ForwardVsCPU",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RoPE::ForwardVsCPU")
}

//------------------------------------------------------------------------------
// Test 4: RoPE forward with causal masking matches CPU
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEForwardCausal()
{
  try
  {
    const uint32_t batch=g_caif_rope_test_batch;
    const uint32_t seq_len=g_caif_rope_test_seq;
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;
    const uint32_t head_dim=dim/num_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    AttentionConfig_t config=MakeRoPEConfig(dim,num_heads,true,true);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

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
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    std::vector<float> expected(batch*seq_len*dim);
    CpuMHAWithRoPE(host_input.data(),
                   h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                   expected.data(),
                   static_cast<int>(batch),static_cast<int>(seq_len),
                   static_cast<int>(dim),static_cast<int>(num_heads),
                   static_cast<int>(head_dim),true,
                   g_caif_rope_default_base);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],expected[i],
                                      g_caif_rope_test_fwd_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": got "
                      <<host_output.Data()[i]
                      <<" expected "
                      <<expected[i]
                      <<" diff="
                      <<std::fabs(host_output.Data()[i]-expected[i])
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("RoPE::ForwardCausal",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RoPE::ForwardCausal")
}

//------------------------------------------------------------------------------
// Test 5: RoPE backward input gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEBackwardInputGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("RoPE::BackwardInputGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const uint32_t batch=g_caif_rope_test_batch;
    const uint32_t seq_len=g_caif_rope_test_seq;
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;
    const float h=g_caif_rope_test_fd_h;
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    AttentionConfig_t config=MakeRoPEConfig(dim,num_heads,false,true);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    // FD reference must be numerically accurate regardless of the outer mode
    // (TF32 FD is catastrophic cancellation). Force high-precision with
    // pass=Backward_e so ComputeTypeFor selects CUBLAS_COMPUTE_32F for the
    // perturbation forwards.
    const bool fd_prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);

    bool passed=true;

    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      CAIF_DeviceMultiHeadAttention<float,float> mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_p.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(input_plus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      ctx.SetTraining(false);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      CAIF_DeviceMultiHeadAttention<float,float> mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_m.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(input_minus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dx mismatch at "
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
    CAIF_Settings::SetPreciseGradients(fd_prev_precise);

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 6: RoPE backward weight gradient (finite difference on W_q)
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEBackwardWeightGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("RoPE::BackwardWeightGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const uint32_t batch=g_caif_rope_test_batch;
    const uint32_t seq_len=g_caif_rope_test_seq;
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;
    const float h=g_caif_rope_test_fd_h;
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    AttentionConfig_t config=MakeRoPEConfig(dim,num_heads,false,true);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    mha.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_wq=mha.GradientTensor(0).ToHost();

    // High-precision FD reference (see TestRoPEBackwardInputGrad comment).
    const bool fd_prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);

    bool passed=true;
    std::vector<float> wq_data(h_wq.Data(),h_wq.Data()+h_wq.TotalElements());

    for(size_t i=0;i<g_caif_rope_test_wgrad_spot&&passed==true;++i)
    {
      std::vector<float> wq_plus(wq_data);
      std::vector<float> wq_minus(wq_data);
      wq_plus[i]+=h;
      wq_minus[i]-=h;

      CAIF_DeviceMultiHeadAttention<float,float> mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(wq_plus.data(),wq_plus.size());
      mha_p.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      ctx.SetTraining(false);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DeviceMultiHeadAttention<float,float> mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(wq_minus.data(),wq_minus.size());
      mha_m.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_wq.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dW_q mismatch at "
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
    CAIF_Settings::SetPreciseGradients(fd_prev_precise);

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 7: RoPE position-dependent encoding.
// RoPE should affect each position differently. We verify that the per-position
// difference between RoPE and non-RoPE outputs varies by position (is not a
// uniform shift). This proves RoPE encodes position-specific information.
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEPositionEquivariance()
{
  try
  {
    const uint32_t batch=g_caif_rope_test_batch;
    const uint32_t seq_len=g_caif_rope_test_seq;
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    AttentionConfig_t config_rope=MakeRoPEConfig(dim,num_heads,false,true);
    AttentionConfig_t config_norope=MakeRoPEConfig(dim,num_heads,false,false);

    CAIF_DeviceMultiHeadAttention<float,float> mha_rope(config_rope,stream);
    CAIF_DeviceMultiHeadAttention<float,float> mha_norope(config_norope,stream);

    // Copy same weights to both
    for(size_t p=0;p<mha_rope.ParameterTensorCount();++p)
    {
      CAIF_HostTensor hw=mha_rope.ParameterTensor(p).ToHost();
      mha_norope.ParameterTensor(p).CopyFromHost(hw.Data(),hw.TotalElements());
    }

    // Input with different content per position so V varies
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
    CAIF_DeviceTensor out_rope=mha_rope.Forward(input,ctx);
    CAIF_HostTensor h_rope=out_rope.ToHost();

    CAIF_DeviceTensor out_norope=mha_norope.Forward(input,ctx);
    CAIF_HostTensor h_norope=out_norope.ToHost();

    // Compute per-position difference vectors between RoPE and non-RoPE.
    // diff[s] = rope_output[s] - norope_output[s].
    // If RoPE is position-dependent, diff[0] should differ from diff[1].
    std::vector<float> diff0(dim);
    std::vector<float> diff1(dim);
    for(uint32_t d=0;d<dim;++d)
    {
      diff0[d]=h_rope.Data()[0*dim+d]-h_norope.Data()[0*dim+d];
      diff1[d]=h_rope.Data()[1*dim+d]-h_norope.Data()[1*dim+d];
    }

    // Check that diff0 != diff1 (RoPE affects different positions differently)
    bool diffs_vary=false;
    for(uint32_t d=0;d<dim;++d)
    {
      if(std::fabs(diff0[d]-diff1[d])>g_caif_rope_test_diff_eps)
      {
        diffs_vary=true;
        break;
      }
    }

    CAIF_TestHarness::Report("RoPE::PositionEquivariance",diffs_vary);
  }
  CAIF_TEST_CATCH_BLOCK("RoPE::PositionEquivariance")
}

//------------------------------------------------------------------------------
// Test 8: Description string includes rope=true
//------------------------------------------------------------------------------
void CAIF_RoPETests::TestRoPEDescriptionString()
{
  try
  {
    const uint32_t dim=g_caif_rope_test_dim;
    const uint32_t num_heads=g_caif_rope_test_heads;

    CAIF_CudaStream stream;
    AttentionConfig_t config=MakeRoPEConfig(dim,num_heads,true,true);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    const std::string desc=mha.Description();
    const std::string expected=
      "MultiHeadAttention(dim=8,heads=2,head_dim=4,causal=true,rope=true)";
    bool passed=(desc==expected);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected '"
                    <<expected
                    <<"', got '"
                    <<desc
                    <<"'\n";
    }

    CAIF_TestHarness::Report("RoPE::DescriptionString",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RoPE::DescriptionString")
}

void CAIF_RoPETests::TestMHANoRoPEBackwardInputGradTiny(const GradMode_t &mode,
                                                         const uint32_t seq_len,
                                                         const uint32_t dim,
                                                         const uint32_t num_heads,
                                                         const char *shape_label)
{
  const std::string test_name=std::string("Diag::MHANoRoPE::")+shape_label+"::"+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const uint32_t batch=g_caif_rope_test_batch;
    const float h=g_caif_rope_test_fd_h;
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    AttentionConfig_t config=MakeRoPEConfig(dim,num_heads,false,false);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      CAIF_DeviceMultiHeadAttention<float,float> mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_p.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(input_plus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      ctx.SetTraining(false);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      double sum_plus=0.0;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=static_cast<double>(hout_p.Data()[j]);
      }

      CAIF_DeviceMultiHeadAttention<float,float> mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_m.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(input_minus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      double sum_minus=0.0;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=static_cast<double>(hout_m.Data()[j]);
      }

      const float numerical=static_cast<float>((sum_plus-sum_minus)/(2.0*h));
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dx mismatch at "
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

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

void CAIF_RoPETests::TestRoPEKernelIsolated(const GradMode_t &mode)
{
  const std::string test_name=std::string("RoPE::KernelIsolated::")+mode.Label();
  try
  {
    const int batch_heads=g_caif_rope_test_kernel_bh;
    const int seq_len=g_caif_rope_test_kernel_seq;
    const int head_dim=g_caif_rope_test_kernel_hd;
    const int total=batch_heads*seq_len*head_dim;
    const float base=g_caif_rope_default_base;
    const int style=g_caif_rope_test_kernel_style;
    const float h=g_caif_rope_test_fd_h;
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;

    std::vector<float> host_x(total);
    for(int i=0;i<total;++i)
    {
      host_x[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    std::vector<float> host_ones(total,1.0f);
    const uint32_t total_u=static_cast<uint32_t>(total);
    CAIF_DeviceTensor d_grad=CAIF_DeviceTensor::FromHostData(host_ones.data(),
                                                             {total_u},
                                                             stream);
    launch_rope_backward<float>(d_grad.DevicePtr<float>(),
                                batch_heads,
                                seq_len,
                                head_dim,
                                base,
                                style,
                                stream.Handle());
    CAIF_HostTensor h_grad=d_grad.ToHost();

    bool passed=true;

    for(int i=0;i<total&&passed==true;++i)
    {
      std::vector<float> x_plus(host_x);
      std::vector<float> x_minus(host_x);
      x_plus[i]+=h;
      x_minus[i]-=h;

      CAIF_DeviceTensor d_plus=CAIF_DeviceTensor::FromHostData(x_plus.data(),
                                                               {total_u},
                                                               stream);
      launch_rope_forward<float>(d_plus.DevicePtr<float>(),
                                 batch_heads,
                                 seq_len,
                                 head_dim,
                                 base,
                                 style,
                                 stream.Handle());
      CAIF_HostTensor hy_plus=d_plus.ToHost();
      double sum_plus=0.0;
      for(int j=0;j<total;++j)
      {
        sum_plus+=static_cast<double>(hy_plus.Data()[j]);
      }

      CAIF_DeviceTensor d_minus=CAIF_DeviceTensor::FromHostData(x_minus.data(),
                                                                {total_u},
                                                                stream);
      launch_rope_forward<float>(d_minus.DevicePtr<float>(),
                                 batch_heads,
                                 seq_len,
                                 head_dim,
                                 base,
                                 style,
                                 stream.Handle());
      CAIF_HostTensor hy_minus=d_minus.ToHost();
      double sum_minus=0.0;
      for(int j=0;j<total;++j)
      {
        sum_minus+=static_cast<double>(hy_minus.Data()[j]);
      }

      const float numerical=static_cast<float>((sum_plus-sum_minus)/(2.0*h));
      const float analytical=h_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  rope-kernel mismatch at "
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

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
}

void CAIF_RoPETests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF RoPE Tests ===\n\n";
  TestRoPEKernelIsolated(g_caif_grad_mode_precise);
  TestRoPEKernelIsolated(g_caif_grad_mode_tf32);
  TestMHANoRoPEBackwardInputGradTiny(g_caif_grad_mode_precise,2,4,2,"s2_hd2_h2");
  TestMHANoRoPEBackwardInputGradTiny(g_caif_grad_mode_precise,3,4,2,"s3_hd2_h2");
  TestMHANoRoPEBackwardInputGradTiny(g_caif_grad_mode_precise,2,8,2,"s2_hd4_h2");
  TestMHANoRoPEBackwardInputGradTiny(g_caif_grad_mode_precise,3,8,2,"s3_hd4_h2");
  TestMHANoRoPEBackwardInputGradTiny(g_caif_grad_mode_precise,2,4,1,"s2_hd4_h1");
  TestMHANoRoPEBackwardInputGradTiny(g_caif_grad_mode_precise,2,2,1,"s2_hd2_h1");
  TestRoPEForwardShape();
  TestRoPEForwardDifference();
  TestRoPEForwardVsCPU();
  TestRoPEForwardCausal();
  TestRoPEBackwardInputGrad(g_caif_grad_mode_precise);
  TestRoPEBackwardInputGrad(g_caif_grad_mode_tf32);
  TestRoPEBackwardWeightGrad(g_caif_grad_mode_precise);
  TestRoPEBackwardWeightGrad(g_caif_grad_mode_tf32);
  TestRoPEPositionEquivariance();
  TestRoPEDescriptionString();
}

#endif  // USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_RoPETests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
