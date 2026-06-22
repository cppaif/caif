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
// Test: Grouped-Query Attention (GQA) forward/backward correctness.
//
// Tests cover forward shape, CPU reference parity, MQA (kv_heads=1), causal
// masking, finite-difference input and weight gradients, parameter count,
// MHA equivalence when kv_heads==heads, and description string.
//------------------------------------------------------------------------------
#include "caif_device_multi_head_attention.h"
#include "caif_test_harness.h"
#include "caif_device_network.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include "caif_cpu_reference/caif_cpu_softmax.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
#include "caif_grad_mode.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_gqa_test_batch=1;
constexpr uint32_t g_caif_gqa_test_seq_len=3;
constexpr uint32_t g_caif_gqa_test_dim=8;
constexpr uint32_t g_caif_gqa_test_num_heads=4;
constexpr uint32_t g_caif_gqa_test_num_kv_heads=2;
constexpr uint32_t g_caif_gqa_test_head_dim=2;
constexpr uint32_t g_caif_gqa_test_num_kv_heads_mqa=1;
constexpr float g_caif_gqa_test_input_scale=0.05f;
constexpr float g_caif_gqa_test_input_offset=-0.3f;
constexpr float g_caif_gqa_test_backward_scale=0.1f;
constexpr float g_caif_gqa_test_backward_offset=-0.2f;
constexpr float g_caif_gqa_test_fd_h=1e-3f;
constexpr size_t g_caif_gqa_test_check_count=4;
constexpr float g_caif_gqa_test_fwd_tol=1e-3f;
constexpr float g_caif_gqa_test_equiv_tol=1e-5f;
constexpr float g_caif_gqa_test_neg_inf=-1e9f;

//------------------------------------------------------------------------------
// CPU reference MHA with GQA (num_kv_heads may differ from num_heads).
// K/V are projected with kv_dim, then KV heads are repeated for attention.
//------------------------------------------------------------------------------
static void CpuMHAWithGQA(const float *input,
                           const float *w_q,
                           const float *w_k,
                           const float *w_v,
                           const float *w_o,
                           float *output,
                           int batch,
                           int seq_len,
                           int dim,
                           int num_heads,
                           int num_kv_heads,
                           int head_dim,
                           bool causal)
{
  const int bs=batch*seq_len;
  const int qk_dim=num_heads*head_dim;
  const int kv_dim=num_kv_heads*head_dim;
  const int repeat_factor=num_heads/num_kv_heads;

  // Project Q: [bs, dim] @ [dim, qk_dim] -> [bs, qk_dim]
  std::vector<float> q_proj(bs*qk_dim);
  CAIF_CpuMatMul::Apply(input,w_q,q_proj.data(),bs,dim,qk_dim);

  // Project K, V: [bs, dim] @ [dim, kv_dim] -> [bs, kv_dim]
  std::vector<float> k_proj(bs*kv_dim);
  std::vector<float> v_proj(bs*kv_dim);
  CAIF_CpuMatMul::Apply(input,w_k,k_proj.data(),bs,dim,kv_dim);
  CAIF_CpuMatMul::Apply(input,w_v,v_proj.data(),bs,dim,kv_dim);

  // Split heads and compute attention
  std::vector<float> concat(bs*qk_dim,0.0f);

  for(int b=0;b<batch;++b)
  {
    for(int h=0;h<num_heads;++h)
    {
      const int kv_h=h/repeat_factor;

      // Extract Q for this head: [seq_len, head_dim]
      std::vector<float> q_head(seq_len*head_dim);
      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          q_head[s*head_dim+d]=q_proj[(b*seq_len+s)*qk_dim+h*head_dim+d];
        }
      }

      // Extract K, V for the corresponding kv_head: [seq_len, head_dim]
      std::vector<float> k_head(seq_len*head_dim);
      std::vector<float> v_head(seq_len*head_dim);
      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          k_head[s*head_dim+d]=k_proj[(b*seq_len+s)*kv_dim+kv_h*head_dim+d];
          v_head[s*head_dim+d]=v_proj[(b*seq_len+s)*kv_dim+kv_h*head_dim+d];
        }
      }

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
            scores[i*seq_len+j]=g_caif_gqa_test_neg_inf;
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

  // Output projection: [bs, qk_dim] @ [qk_dim, dim] -> [bs, dim]
  CAIF_CpuMatMul::Apply(concat.data(),w_o,output,bs,qk_dim,dim);
}

//------------------------------------------------------------------------------
// GQA tests class.
//------------------------------------------------------------------------------
class CAIF_GQATests
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceMultiHeadAttentionConfig MakeGQAConfig(
      uint32_t dim,
      uint32_t num_heads,
      uint32_t num_kv_heads,
      uint32_t head_dim,
      bool causal);

    static void TestGQAForwardShape();
    static void TestGQAForwardVsCPU();
    static void TestGQAForwardMQA();
    static void TestGQAForwardCausal();
    static void TestGQABackwardInputGrad(const GradMode_t &mode);
    static void TestGQABackwardWeightGrad(const GradMode_t &mode);
    static void TestGQABackwardWeightGradK(const GradMode_t &mode);
    static void TestGQAParameterCount();
    static void TestGQAMHAEquivalence();
    static void TestGQADescriptionString();
};

CAIF_DeviceMultiHeadAttentionConfig CAIF_GQATests::MakeGQAConfig(
  const uint32_t dim,
  const uint32_t num_heads,
  const uint32_t num_kv_heads,
  const uint32_t head_dim,
  const bool causal)
{
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             num_heads,
                                             num_kv_heads,
                                             head_dim,
                                             causal,
                                             false,
                                             g_caif_rope_default_base,
                                             0.0f);
  return config;
}

//------------------------------------------------------------------------------
// Test 1: GQA forward output shape correct
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQAForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    false);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    std::vector<float> host_input(g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3 ||
       shape[0]!=g_caif_gqa_test_batch ||
       shape[1]!=g_caif_gqa_test_seq_len ||
       shape[2]!=g_caif_gqa_test_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }

    CAIF_TestHarness::Report("GQA::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: GQA forward vs CPU reference (heads=4, kv_heads=2)
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQAForwardVsCPU()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    false);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    const size_t n_input=g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim;
    std::vector<float> host_input(n_input);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_gqa_test_input_scale+g_caif_gqa_test_input_offset;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    std::vector<float> expected(n_input);
    CpuMHAWithGQA(host_input.data(),
                  h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                  expected.data(),
                  static_cast<int>(g_caif_gqa_test_batch),
                  static_cast<int>(g_caif_gqa_test_seq_len),
                  static_cast<int>(g_caif_gqa_test_dim),
                  static_cast<int>(g_caif_gqa_test_num_heads),
                  static_cast<int>(g_caif_gqa_test_num_kv_heads),
                  static_cast<int>(g_caif_gqa_test_head_dim),
                  false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      const float diff=std::fabs(host_output.Data()[i]-expected[i]);
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],expected[i],g_caif_gqa_test_fwd_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": got "
                     <<host_output.Data()[i]
                     <<" expected "
                     <<expected[i]
                     <<" diff="
                     <<diff
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("GQA::ForwardVsCPU",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::ForwardVsCPU")
}

//------------------------------------------------------------------------------
// Test 3: GQA Multi-Query Attention (kv_heads=1) matches CPU reference
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQAForwardMQA()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads_mqa,
                                    g_caif_gqa_test_head_dim,
                                    false);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    const size_t n_input=g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim;
    std::vector<float> host_input(n_input);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_gqa_test_input_scale+g_caif_gqa_test_input_offset;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    std::vector<float> expected(n_input);
    CpuMHAWithGQA(host_input.data(),
                  h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                  expected.data(),
                  static_cast<int>(g_caif_gqa_test_batch),
                  static_cast<int>(g_caif_gqa_test_seq_len),
                  static_cast<int>(g_caif_gqa_test_dim),
                  static_cast<int>(g_caif_gqa_test_num_heads),
                  static_cast<int>(g_caif_gqa_test_num_kv_heads_mqa),
                  static_cast<int>(g_caif_gqa_test_head_dim),
                  false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      const float diff=std::fabs(host_output.Data()[i]-expected[i]);
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],expected[i],g_caif_gqa_test_fwd_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": got "
                     <<host_output.Data()[i]
                     <<" expected "
                     <<expected[i]
                     <<" diff="
                     <<diff
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("GQA::ForwardMQA",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::ForwardMQA")
}

//------------------------------------------------------------------------------
// Test 4: GQA + causal matches CPU reference
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQAForwardCausal()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    true);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    const size_t n_input=g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim;
    std::vector<float> host_input(n_input);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_gqa_test_input_scale+g_caif_gqa_test_input_offset;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    std::vector<float> expected(n_input);
    CpuMHAWithGQA(host_input.data(),
                  h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                  expected.data(),
                  static_cast<int>(g_caif_gqa_test_batch),
                  static_cast<int>(g_caif_gqa_test_seq_len),
                  static_cast<int>(g_caif_gqa_test_dim),
                  static_cast<int>(g_caif_gqa_test_num_heads),
                  static_cast<int>(g_caif_gqa_test_num_kv_heads),
                  static_cast<int>(g_caif_gqa_test_head_dim),
                  true);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      const float diff=std::fabs(host_output.Data()[i]-expected[i]);
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],expected[i],g_caif_gqa_test_fwd_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": got "
                     <<host_output.Data()[i]
                     <<" expected "
                     <<expected[i]
                     <<" diff="
                     <<diff
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("GQA::ForwardCausal",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::ForwardCausal")
}

//------------------------------------------------------------------------------
// Test 5: GQA backward input gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQABackwardInputGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("GQA::BackwardInputGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    false);

    const size_t n_input=g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim;
    std::vector<float> host_input(n_input);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_gqa_test_backward_scale+g_caif_gqa_test_backward_offset;
    }

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(n_input,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    // FD reference must be high-precision regardless of outer mode.
    const bool fd_prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);

    bool passed=true;

    ctx.SetTraining(false);
    for(size_t i=0;i<host_input.size() && passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=g_caif_gqa_test_fd_h;
      input_minus[i]-=g_caif_gqa_test_fd_h;

      CAIF_DeviceMultiHeadAttention<float,float> mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_p.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
        input_plus.data(),
        {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
        stream);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<n_input;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DeviceMultiHeadAttention<float,float> mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_m.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
        input_minus.data(),
        {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
        stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<n_input;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_gqa_test_fd_h);
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
  CAIF_TEST_CATCH_BLOCK("GQA::BackwardInputGrad")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 6: GQA backward weight gradient (finite difference on W_q)
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQABackwardWeightGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("GQA::BackwardWeightGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    false);

    const size_t n_input=g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim;
    std::vector<float> host_input(n_input);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_gqa_test_backward_scale+g_caif_gqa_test_backward_offset;
    }

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(n_input,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    mha.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_wq=mha.GradientTensor(0).ToHost();

    bool passed=true;
    std::vector<float> wq_data(h_wq.Data(),h_wq.Data()+h_wq.TotalElements());

    ctx.SetTraining(false);
    for(size_t i=0;i<g_caif_gqa_test_check_count && passed==true;++i)
    {
      std::vector<float> wq_plus(wq_data);
      std::vector<float> wq_minus(wq_data);
      wq_plus[i]+=g_caif_gqa_test_fd_h;
      wq_minus[i]-=g_caif_gqa_test_fd_h;

      CAIF_DeviceMultiHeadAttention<float,float> mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(wq_plus.data(),wq_plus.size());
      mha_p.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
        host_input.data(),
        {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
        stream);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<n_input;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DeviceMultiHeadAttention<float,float> mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(wq_minus.data(),wq_minus.size());
      mha_m.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
        host_input.data(),
        {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
        stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<n_input;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_gqa_test_fd_h);
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

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::BackwardWeightGrad")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 7: GQA backward W_k gradient (critical: must sum over repeated groups)
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQABackwardWeightGradK(const GradMode_t &mode)
{
  const std::string test_name=std::string("GQA::BackwardWeightGrad_K::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    false);

    const size_t n_input=g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim;
    std::vector<float> host_input(n_input);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_gqa_test_backward_scale+g_caif_gqa_test_backward_offset;
    }

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(n_input,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    mha.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_wk=mha.GradientTensor(1).ToHost();

    const bool fd_prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);

    bool passed=true;
    std::vector<float> wk_data(h_wk.Data(),h_wk.Data()+h_wk.TotalElements());

    ctx.SetTraining(false);
    for(size_t i=0;i<g_caif_gqa_test_check_count && passed==true;++i)
    {
      std::vector<float> wk_plus(wk_data);
      std::vector<float> wk_minus(wk_data);
      wk_plus[i]+=g_caif_gqa_test_fd_h;
      wk_minus[i]-=g_caif_gqa_test_fd_h;

      CAIF_DeviceMultiHeadAttention<float,float> mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_p.ParameterTensor(1).CopyFromHost(wk_plus.data(),wk_plus.size());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
        host_input.data(),
        {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
        stream);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<n_input;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DeviceMultiHeadAttention<float,float> mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_m.ParameterTensor(1).CopyFromHost(wk_minus.data(),wk_minus.size());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
        host_input.data(),
        {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
        stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<n_input;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_gqa_test_fd_h);
      const float analytical=host_grad_wk.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dW_k mismatch at "
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
  CAIF_TEST_CATCH_BLOCK("GQA::BackwardWeightGrad_K")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 8: GQA parameter count
// total params = dim*qk_dim + 2*dim*kv_dim + qk_dim*dim
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQAParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    false);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    const uint32_t qk_dim=g_caif_gqa_test_num_heads*g_caif_gqa_test_head_dim;
    const uint32_t kv_dim=g_caif_gqa_test_num_kv_heads*g_caif_gqa_test_head_dim;
    const size_t expected_total=g_caif_gqa_test_dim*qk_dim+
                                2*g_caif_gqa_test_dim*kv_dim+
                                qk_dim*g_caif_gqa_test_dim;

    bool passed=true;
    if(mha.TotalParameterCount()!=expected_total)
    {
      ISE_Out::Out()<<"  TotalParameterCount expected "
                   <<expected_total
                   <<", got "
                   <<mha.TotalParameterCount()
                   <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("GQA::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 9: GQA with kv_heads==heads produces same output as standard MHA
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQAMHAEquivalence()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Standard MHA (kv_heads == heads)
    const auto config_mha=MakeGQAConfig(g_caif_gqa_test_dim,
                                        g_caif_gqa_test_num_heads,
                                        g_caif_gqa_test_num_heads,
                                        g_caif_gqa_test_head_dim,
                                        false);
    CAIF_DeviceMultiHeadAttention<float,float> mha_std(config_mha,stream);

    // GQA with kv_heads==heads (should behave identically)
    const auto config_gqa=MakeGQAConfig(g_caif_gqa_test_dim,
                                        g_caif_gqa_test_num_heads,
                                        g_caif_gqa_test_num_heads,
                                        g_caif_gqa_test_head_dim,
                                        false);
    CAIF_DeviceMultiHeadAttention<float,float> mha_gqa(config_gqa,stream);

    // Copy weights
    for(size_t p=0;p<mha_std.ParameterTensorCount();++p)
    {
      CAIF_HostTensor hw=mha_std.ParameterTensor(p).ToHost();
      mha_gqa.ParameterTensor(p).CopyFromHost(hw.Data(),hw.TotalElements());
    }

    const size_t n_input=g_caif_gqa_test_batch*g_caif_gqa_test_seq_len*g_caif_gqa_test_dim;
    std::vector<float> host_input(n_input);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_gqa_test_input_scale+g_caif_gqa_test_input_offset;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_gqa_test_batch,g_caif_gqa_test_seq_len,g_caif_gqa_test_dim},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor out_std=mha_std.Forward(input,ctx);
    CAIF_DeviceTensor out_gqa=mha_gqa.Forward(input,ctx);

    CAIF_HostTensor h_std=out_std.ToHost();
    CAIF_HostTensor h_gqa=out_gqa.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_std.TotalElements();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_std.Data()[i],h_gqa.Data()[i],g_caif_gqa_test_equiv_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": std="
                     <<h_std.Data()[i]
                     <<" gqa="
                     <<h_gqa.Data()[i]
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("GQA::MHAEquivalence",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::MHAEquivalence")
}

//------------------------------------------------------------------------------
// Test 10: Description includes kv_heads=N when GQA active
//------------------------------------------------------------------------------
void CAIF_GQATests::TestGQADescriptionString()
{
  try
  {
    CAIF_CudaStream stream;
    const auto config=MakeGQAConfig(g_caif_gqa_test_dim,
                                    g_caif_gqa_test_num_heads,
                                    g_caif_gqa_test_num_kv_heads,
                                    g_caif_gqa_test_head_dim,
                                    true);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    const std::string desc=mha.Description();
    const std::string expected=
      "MultiHeadAttention(dim=8,heads=4,head_dim=2,causal=true,kv_heads=2)";
    bool passed=(desc==expected);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected '"
                   <<expected
                   <<"', got '"
                   <<desc
                   <<"'\n";
    }

    CAIF_TestHarness::Report("GQA::DescriptionString",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GQA::DescriptionString")
}

void CAIF_GQATests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF GQA Tests ==="
               <<"\n\n";
  TestGQAForwardShape();
  TestGQAForwardVsCPU();
  TestGQAForwardMQA();
  TestGQAForwardCausal();
  TestGQABackwardInputGrad(g_caif_grad_mode_precise);
  TestGQABackwardInputGrad(g_caif_grad_mode_tf32);
  TestGQABackwardWeightGrad(g_caif_grad_mode_precise);
  TestGQABackwardWeightGrad(g_caif_grad_mode_tf32);
  TestGQABackwardWeightGradK(g_caif_grad_mode_precise);
  TestGQABackwardWeightGradK(g_caif_grad_mode_tf32);
  TestGQAParameterCount();
  TestGQAMHAEquivalence();
  TestGQADescriptionString();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_GQATests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
               <<"\n";
  return 0;
#endif
}
