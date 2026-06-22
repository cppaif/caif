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
// Test: CAIF_DeviceT5Attention + CAIF_DeviceRelativePositionBias correctness
//       and benchmark vs base MHA.
//------------------------------------------------------------------------------
#include "caif_device_t5_attention.h"
#include "caif_test_harness.h"
#include "caif_device_multi_head_attention.h"
#include "caif_device_relative_position_bias.h"
#include "caif_run_context.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include "caif_cpu_reference/caif_cpu_softmax.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
#include "caif_cpu_reference/caif_cpu_relative_position.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr float g_caif_t5_test_forward_tol=5e-3f;
constexpr float g_caif_t5_test_rpb_tol=1e-4f;
constexpr int g_caif_t5_bench_warmup_iters=5;
constexpr int g_caif_t5_bench_iters=50;
constexpr int g_caif_t5_rpb_bench_warmup=10;
constexpr int g_caif_t5_rpb_bench_iters=100;

//------------------------------------------------------------------------------
// T5 Attention + Relative Position Bias correctness + benchmark tests.
//------------------------------------------------------------------------------
class CAIF_T5AttentionTests
{
  public:
    static void RunAll();

  protected:

  private:
    // CPU reference: full MHA forward with position bias
    static void CpuT5MHA(const float *input,
                         const float *w_q,
                         const float *w_k,
                         const float *w_v,
                         const float *w_o,
                         const float *pos_bias,
                         float *output,
                         int batch,
                         int seq_len,
                         int dim,
                         int num_heads,
                         int head_dim,
                         bool causal);

    static CAIF_DeviceMultiHeadAttentionConfig MakeConfig(
                                                                           uint32_t dim,
                                                                           uint32_t num_heads,
                                                                           bool causal);

    static void TestRPBForwardBidirectional();
    static void TestRPBForwardUnidirectional();
    static void TestRPBShape();
    static void TestRPBParameters();
    static void TestT5AttentionForward();
    static void TestT5AttentionCausal();
    static void TestT5AttentionBackward();
    static void TestBaseMHARegression();
    static void BenchmarkRPBForward();
    static void BenchmarkT5vsBaseMHA();
    static void BenchmarkT5ForwardBackward();
};

//------------------------------------------------------------------------------
// CPU reference MHA forward with optional position bias
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::CpuT5MHA(const float *input,
                                      const float *w_q,
                                      const float *w_k,
                                      const float *w_v,
                                      const float *w_o,
                                      const float *pos_bias,
                                      float *output,
                                      const int batch,
                                      const int seq_len,
                                      const int dim,
                                      const int num_heads,
                                      const int head_dim,
                                      const bool causal)
{
  const int bs=batch*seq_len;
  const int qk_dim=num_heads*head_dim;

  std::vector<float> q_proj(bs*qk_dim);
  std::vector<float> k_proj(bs*qk_dim);
  std::vector<float> v_proj(bs*qk_dim);
  CAIF_CpuMatMul::Apply(input,w_q,q_proj.data(),bs,dim,qk_dim);
  CAIF_CpuMatMul::Apply(input,w_k,k_proj.data(),bs,dim,qk_dim);
  CAIF_CpuMatMul::Apply(input,w_v,v_proj.data(),bs,dim,qk_dim);

  std::vector<float> concat(bs*qk_dim,0.0f);

  for(int b=0;b<batch;++b)
  {
    for(int h=0;h<num_heads;++h)
    {
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

      // Add position bias [num_heads, seq_len, seq_len]
      if(pos_bias!=nullptr)
      {
        const int bias_offset=h*seq_len*seq_len;
        for(int i=0;i<seq_len*seq_len;++i)
        {
          scores[i]+=pos_bias[bias_offset+i];
        }
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

      CAIF_CpuSoftmax::Apply(scores.data(),seq_len,seq_len);

      std::vector<float> ctx(seq_len*head_dim);
      CAIF_CpuMatMul::Apply(scores.data(),v_head.data(),ctx.data(),seq_len,seq_len,head_dim);

      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          concat[(b*seq_len+s)*qk_dim+h*head_dim+d]=ctx[s*head_dim+d];
        }
      }
    }
  }

  CAIF_CpuMatMul::Apply(concat.data(),w_o,output,bs,qk_dim,dim);
}

CAIF_DeviceMultiHeadAttentionConfig
CAIF_T5AttentionTests::MakeConfig(const uint32_t dim,
                                   const uint32_t num_heads,
                                   const bool causal)
{
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             num_heads,
                                             num_heads,
                                             dim/num_heads,
                                             causal,
                                             false,
                                             g_caif_rope_default_base,
                                             0.0f);
  return config;
}

//------------------------------------------------------------------------------
// Test 1: RPB forward matches CPU reference (bidirectional)
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestRPBForwardBidirectional()
{
  try
  {
    const uint32_t num_heads=4;
    const uint32_t num_buckets=32;
    const uint32_t max_distance=128;
    const uint32_t q_len=8;
    const uint32_t k_len=8;

    CAIF_CudaStream stream;
    CAIF_DeviceRelativePositionBiasConfig config(num_heads,num_buckets,max_distance,true);

    CAIF_DeviceRelativePositionBias<float,float> rpb(config,stream);

    // Get embedding for CPU reference
    CAIF_HostTensor h_emb=rpb.ParameterTensor(0).ToHost();

    CAIF_DeviceTensor bias=rpb.ComputeBias(q_len,k_len);
    CAIF_HostTensor h_bias=bias.ToHost();

    // CPU reference
    std::vector<float> expected(num_heads*q_len*k_len);
    CAIF_CpuRelativePosition::BiasForward(h_emb.Data(),
                                          expected.data(),
                                          static_cast<int>(num_heads),
                                          static_cast<int>(q_len),
                                          static_cast<int>(k_len),
                                          static_cast<int>(num_buckets),
                                          static_cast<int>(max_distance),
                                          true);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_bias.Data()[i],expected[i],g_caif_t5_test_rpb_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": gpu="
                      <<h_bias.Data()[i]
                      <<" cpu="
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("RPB::ForwardBidirectional",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RPB::ForwardBidirectional")
}

//------------------------------------------------------------------------------
// Test 2: RPB forward matches CPU reference (unidirectional/causal)
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestRPBForwardUnidirectional()
{
  try
  {
    const uint32_t num_heads=4;
    const uint32_t num_buckets=32;
    const uint32_t max_distance=128;
    const uint32_t q_len=8;
    const uint32_t k_len=8;

    CAIF_CudaStream stream;
    CAIF_DeviceRelativePositionBiasConfig config(num_heads,num_buckets,max_distance,false);

    CAIF_DeviceRelativePositionBias<float,float> rpb(config,stream);
    CAIF_HostTensor h_emb=rpb.ParameterTensor(0).ToHost();

    CAIF_DeviceTensor bias=rpb.ComputeBias(q_len,k_len);
    CAIF_HostTensor h_bias=bias.ToHost();

    std::vector<float> expected(num_heads*q_len*k_len);
    CAIF_CpuRelativePosition::BiasForward(h_emb.Data(),
                                          expected.data(),
                                          static_cast<int>(num_heads),
                                          static_cast<int>(q_len),
                                          static_cast<int>(k_len),
                                          static_cast<int>(num_buckets),
                                          static_cast<int>(max_distance),
                                          false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_bias.Data()[i],expected[i],g_caif_t5_test_rpb_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": gpu="
                      <<h_bias.Data()[i]
                      <<" cpu="
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("RPB::ForwardUnidirectional",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RPB::ForwardUnidirectional")
}

//------------------------------------------------------------------------------
// Test 3: RPB output shape correctness
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestRPBShape()
{
  try
  {
    const uint32_t num_heads=8;
    const uint32_t num_buckets=32;
    const uint32_t q_len=16;
    const uint32_t k_len=16;

    CAIF_CudaStream stream;
    CAIF_DeviceRelativePositionBiasConfig config(num_heads,num_buckets,128,true);

    CAIF_DeviceRelativePositionBias<float,float> rpb(config,stream);
    CAIF_DeviceTensor bias=rpb.ComputeBias(q_len,k_len);

    const auto &shape=bias.Shape();
    bool passed=(shape.size()==3 &&
                 shape[0]==num_heads &&
                 shape[1]==q_len &&
                 shape[2]==k_len);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected shape [8,16,16], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          ISE_Out::Out()<<",";
        }
        ISE_Out::Out()<<shape[i];
      }
      ISE_Out::Out()<<"]\n";
    }

    CAIF_TestHarness::Report("RPB::Shape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RPB::Shape")
}

//------------------------------------------------------------------------------
// Test 4: RPB parameter count and names
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestRPBParameters()
{
  try
  {
    const uint32_t num_heads=4;
    const uint32_t num_buckets=32;

    CAIF_CudaStream stream;
    CAIF_DeviceRelativePositionBiasConfig config(num_heads,num_buckets,128,true);

    CAIF_DeviceRelativePositionBias<float,float> rpb(config,stream);

    bool passed=true;
    if(rpb.ParameterTensorCount()!=1)
    {
      ISE_Out::Out()<<"  Expected 1 parameter tensor, got "
                    <<rpb.ParameterTensorCount()
                    <<"\n";
      passed=false;
    }
    if(rpb.TotalParameterCount()!=num_heads*num_buckets)
    {
      ISE_Out::Out()<<"  Expected "
                    <<num_heads*num_buckets
                    <<" params, got "
                    <<rpb.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    auto names=rpb.ParameterNames("block.0.");
    if(names.size()!=1 || names[0]!="block.0.rpb")
    {
      ISE_Out::Out()<<"  Unexpected param name: "
                    <<names[0]
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("RPB::Parameters",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RPB::Parameters")
}

//------------------------------------------------------------------------------
// Test 5: T5 attention forward matches CPU reference with bias
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestT5AttentionForward()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=4;
    const uint32_t dim=16;
    const uint32_t num_heads=4;
    const uint32_t head_dim=dim/num_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeConfig(dim,num_heads,false);
    CAIF_DeviceT5Attention<float,float> t5attn(config,stream);

    // Compute position bias
    CAIF_DeviceRelativePositionBiasConfig rpb_config(num_heads,
                                                     g_caif_rpb_default_num_buckets,
                                                     g_caif_rpb_default_max_distance,
                                                     true);
    CAIF_DeviceRelativePositionBias<float,float> rpb(rpb_config,stream);

    CAIF_DeviceTensor bias=rpb.ComputeBias(seq_len,seq_len);
    ctx.SetPositionBias(bias);

    // Get weights for CPU reference
    CAIF_HostTensor h_wq=t5attn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=t5attn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=t5attn.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=t5attn.ParameterTensor(3).ToHost();
    CAIF_HostTensor h_bias=bias.ToHost();

    // Deterministic input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,dim},stream);

    // GPU forward
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=t5attn.Forward(input,ctx);
    CAIF_HostTensor h_output=output.ToHost();

    // CPU reference
    std::vector<float> expected(batch*seq_len*dim);
    CpuT5MHA(host_input.data(),
             h_wq.Data(),
             h_wk.Data(),
             h_wv.Data(),
             h_wo.Data(),
             h_bias.Data(),
             expected.data(),
             static_cast<int>(batch),
             static_cast<int>(seq_len),
             static_cast<int>(dim),
             static_cast<int>(num_heads),
             static_cast<int>(head_dim),
             false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_output.Data()[i],expected[i],g_caif_t5_test_forward_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": gpu="
                      <<h_output.Data()[i]
                      <<" cpu="
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("T5Attention::Forward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("T5Attention::Forward")
}

//------------------------------------------------------------------------------
// Test 6: T5 attention forward with causal mask
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestT5AttentionCausal()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=4;
    const uint32_t dim=16;
    const uint32_t num_heads=4;
    const uint32_t head_dim=dim/num_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeConfig(dim,num_heads,true);
    CAIF_DeviceT5Attention<float,float> t5attn(config,stream);

    CAIF_DeviceRelativePositionBiasConfig rpb_config(num_heads,
                                                     g_caif_rpb_default_num_buckets,
                                                     g_caif_rpb_default_max_distance,
                                                     false);
    CAIF_DeviceRelativePositionBias<float,float> rpb(rpb_config,stream);

    CAIF_DeviceTensor bias=rpb.ComputeBias(seq_len,seq_len);
    ctx.SetPositionBias(bias);

    CAIF_HostTensor h_wq=t5attn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=t5attn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=t5attn.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=t5attn.ParameterTensor(3).ToHost();
    CAIF_HostTensor h_bias=bias.ToHost();

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.03f-0.2f;
    }
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,dim},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=t5attn.Forward(input,ctx);
    CAIF_HostTensor h_output=output.ToHost();

    std::vector<float> expected(batch*seq_len*dim);
    CpuT5MHA(host_input.data(),
             h_wq.Data(),
             h_wk.Data(),
             h_wv.Data(),
             h_wo.Data(),
             h_bias.Data(),
             expected.data(),
             static_cast<int>(batch),
             static_cast<int>(seq_len),
             static_cast<int>(dim),
             static_cast<int>(num_heads),
             static_cast<int>(head_dim),
             true);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_output.Data()[i],expected[i],g_caif_t5_test_forward_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": gpu="
                      <<h_output.Data()[i]
                      <<" cpu="
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("T5Attention::ForwardCausal",passed);
  }
  CAIF_TEST_CATCH_BLOCK("T5Attention::ForwardCausal")
}

//------------------------------------------------------------------------------
// Test 7: T5 attention backward produces non-zero gradients
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestT5AttentionBackward()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=16;
    const uint32_t num_heads=4;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeConfig(dim,num_heads,false);
    CAIF_DeviceT5Attention<float,float> t5attn(config,stream);

    CAIF_DeviceRelativePositionBiasConfig rpb_config(num_heads,
                                                     g_caif_rpb_default_num_buckets,
                                                     g_caif_rpb_default_max_distance,
                                                     true);
    CAIF_DeviceRelativePositionBias<float,float> rpb(rpb_config,stream);

    CAIF_DeviceTensor bias=rpb.ComputeBias(seq_len,seq_len);
    auto grad_bias=CAIF_DeviceTensor::Zeros({num_heads,seq_len,seq_len},stream);
    ctx.SetPositionBias(bias);
    ctx.SetGradPositionBias(grad_bias);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.02f-0.5f;
    }
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,dim},stream);

    // Forward with training
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=t5attn.Forward(input,ctx);

    // Backward with ones gradient
    std::vector<float> grad_data(batch*seq_len*dim,1.0f);
    auto grad_output=CAIF_DeviceTensor::FromHostData(grad_data.data(),{batch,seq_len,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=t5attn.Backward(grad_output,ctx);

    // Check grad_input shape and non-zero
    CAIF_HostTensor h_grad_input=grad_input.ToHost();
    bool passed=true;
    const auto &shape=h_grad_input.Shape();
    if(shape.size()!=3 || shape[0]!=batch || shape[1]!=seq_len || shape[2]!=dim)
    {
      ISE_Out::Out()<<"  grad_input shape mismatch\n";
      passed=false;
    }

    bool any_nonzero=false;
    for(size_t i=0;i<h_grad_input.TotalElements();++i)
    {
      if(h_grad_input.Data()[i]!=0.0f)
      {
        any_nonzero=true;
        break;
      }
    }
    if(any_nonzero==false)
    {
      ISE_Out::Out()<<"  grad_input is all zeros\n";
      passed=false;
    }

    // Check position bias gradient is non-zero
    CAIF_HostTensor h_grad_bias=grad_bias.ToHost();
    bool bias_grad_nonzero=false;
    for(size_t i=0;i<h_grad_bias.TotalElements();++i)
    {
      if(h_grad_bias.Data()[i]!=0.0f)
      {
        bias_grad_nonzero=true;
        break;
      }
    }
    if(bias_grad_nonzero==false)
    {
      ISE_Out::Out()<<"  position bias gradient is all zeros\n";
      passed=false;
    }

    // Check weight gradients are non-zero
    t5attn.ZeroGradients();
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    output=t5attn.Forward(input,ctx);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    grad_input=t5attn.Backward(grad_output,ctx);

    bool weight_grad_nonzero=false;
    for(size_t p=0;p<t5attn.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_grad=t5attn.GradientTensor(p).ToHost();
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(h_grad.Data()[i]!=0.0f)
        {
          weight_grad_nonzero=true;
          break;
        }
      }
      if(weight_grad_nonzero==true)
      {
        break;
      }
    }
    if(weight_grad_nonzero==false)
    {
      ISE_Out::Out()<<"  weight gradients are all zeros\n";
      passed=false;
    }

    CAIF_TestHarness::Report("T5Attention::Backward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("T5Attention::Backward")
}

//------------------------------------------------------------------------------
// Test 8: Base MHA regression — ensure no performance/behavior change
// Runs standard MHA (no T5, no position bias) and verifies it still
// matches the same CPU reference as test_device_attention.cpp
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::TestBaseMHARegression()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=4;
    const uint32_t dim=128;
    const uint32_t num_heads=2;
    const uint32_t head_dim=dim/num_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeConfig(dim,num_heads,false);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.001f-0.3f;
    }
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,dim},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mha.Forward(input,ctx);
    CAIF_HostTensor h_output=output.ToHost();

    // CPU reference (no bias)
    std::vector<float> expected(batch*seq_len*dim);
    CpuT5MHA(host_input.data(),
             h_wq.Data(),
             h_wk.Data(),
             h_wv.Data(),
             h_wo.Data(),
             nullptr,
             expected.data(),
             static_cast<int>(batch),
             static_cast<int>(seq_len),
             static_cast<int>(dim),
             static_cast<int>(num_heads),
             static_cast<int>(head_dim),
             false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_output.Data()[i],expected[i],g_caif_t5_test_forward_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": gpu="
                      <<h_output.Data()[i]
                      <<" cpu="
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("BaseMHA::Regression",passed);
  }
  CAIF_TEST_CATCH_BLOCK("BaseMHA::Regression")
}

//------------------------------------------------------------------------------
// Benchmark: RPB forward timing
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::BenchmarkRPBForward()
{
  try
  {
    const uint32_t num_heads=8;
    const uint32_t num_buckets=32;
    const uint32_t max_distance=128;
    const uint32_t seq_len=512;

    CAIF_CudaStream stream;
    CAIF_DeviceRelativePositionBiasConfig config(num_heads,num_buckets,max_distance,true);

    CAIF_DeviceRelativePositionBias<float,float> rpb(config,stream);

    // Warmup
    for(int i=0;i<g_caif_t5_rpb_bench_warmup;++i)
    {
      CAIF_DeviceTensor bias=rpb.ComputeBias(seq_len,seq_len);
    }
    cudaStreamSynchronize(stream.Handle());

    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_t5_rpb_bench_iters;++i)
    {
      CAIF_DeviceTensor bias=rpb.ComputeBias(seq_len,seq_len);
    }
    cudaStreamSynchronize(stream.Handle());
    auto end=std::chrono::high_resolution_clock::now();

    const double elapsed_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double per_iter_us=elapsed_ms*1000.0/static_cast<double>(g_caif_t5_rpb_bench_iters);

    ISE_Out::Out()<<"[BENCH] RPB forward (heads="
                  <<num_heads
                  <<",seq="
                  <<seq_len
                  <<",buckets="
                  <<num_buckets
                  <<"): "
                  <<per_iter_us
                  <<" us/iter"
                  <<" ("
                  <<g_caif_t5_rpb_bench_iters
                  <<" iters)\n";
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::Out()<<"[BENCH] RPB forward: FAILED ("
                  <<e
                  <<")\n";
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"[BENCH] RPB forward: FAILED ("
                  <<e.what()
                  <<")\n";
  }
}

//------------------------------------------------------------------------------
// Benchmark: T5 attention forward vs base MHA forward
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::BenchmarkT5vsBaseMHA()
{
  try
  {
    const uint32_t batch=4;
    const uint32_t seq_len=128;
    const uint32_t dim=256;
    const uint32_t num_heads=8;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    auto config=MakeConfig(dim,num_heads,false);

    // Base MHA
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

    // T5 attention
    CAIF_DeviceT5Attention<float,float> t5attn(config,stream);
    CAIF_DeviceRelativePositionBiasConfig rpb_config(num_heads,
                                                     g_caif_rpb_default_num_buckets,
                                                     g_caif_rpb_default_max_distance,
                                                     true);
    CAIF_DeviceRelativePositionBias<float,float> rpb(rpb_config,stream);
    CAIF_DeviceTensor bias=rpb.ComputeBias(seq_len,seq_len);
    ctx.SetPositionBias(bias);

    // Input
    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,dim},stream);

    // Warmup base MHA
    for(int i=0;i<g_caif_t5_bench_warmup_iters;++i)
    {
      mha.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    // Bench base MHA
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_t5_bench_iters;++i)
    {
      mha.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    auto end=std::chrono::high_resolution_clock::now();
    const double base_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double base_per_iter=base_ms/static_cast<double>(g_caif_t5_bench_iters);

    // Warmup T5
    for(int i=0;i<g_caif_t5_bench_warmup_iters;++i)
    {
      t5attn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    // Bench T5
    start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_t5_bench_iters;++i)
    {
      t5attn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    end=std::chrono::high_resolution_clock::now();
    const double t5_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double t5_per_iter=t5_ms/static_cast<double>(g_caif_t5_bench_iters);

    const double overhead_pct=((t5_per_iter-base_per_iter)/base_per_iter)*100.0;

    ISE_Out::Out()<<"[BENCH] Base MHA forward (batch="
                  <<batch
                  <<",seq="
                  <<seq_len
                  <<",dim="
                  <<dim
                  <<",heads="
                  <<num_heads
                  <<"): "
                  <<base_per_iter
                  <<" ms/iter\n";
    ISE_Out::Out()<<"[BENCH] T5 attention forward (same config + RPB): "
                  <<t5_per_iter
                  <<" ms/iter\n";
    ISE_Out::Out()<<"[BENCH] T5 overhead vs base MHA: "
                  <<overhead_pct
                  <<"%%\n";
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::Out()<<"[BENCH] T5 vs Base MHA: FAILED ("
                  <<e
                  <<")\n";
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"[BENCH] T5 vs Base MHA: FAILED ("
                  <<e.what()
                  <<")\n";
  }
}

//------------------------------------------------------------------------------
// Benchmark: T5 attention forward+backward timing
//------------------------------------------------------------------------------
void CAIF_T5AttentionTests::BenchmarkT5ForwardBackward()
{
  try
  {
    const uint32_t batch=4;
    const uint32_t seq_len=128;
    const uint32_t dim=256;
    const uint32_t num_heads=8;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto config=MakeConfig(dim,num_heads,false);

    CAIF_DeviceT5Attention<float,float> t5attn(config,stream);
    CAIF_DeviceRelativePositionBiasConfig rpb_config(num_heads,
                                                     g_caif_rpb_default_num_buckets,
                                                     g_caif_rpb_default_max_distance,
                                                     true);
    CAIF_DeviceRelativePositionBias<float,float> rpb(rpb_config,stream);
    CAIF_DeviceTensor bias=rpb.ComputeBias(seq_len,seq_len);
    auto grad_bias=CAIF_DeviceTensor::Zeros({num_heads,seq_len,seq_len},stream);
    ctx.SetPositionBias(bias);
    ctx.SetGradPositionBias(grad_bias);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,dim},stream);
    std::vector<float> grad_data(batch*seq_len*dim,1.0f);
    auto grad_output=CAIF_DeviceTensor::FromHostData(grad_data.data(),{batch,seq_len,dim},stream);

    // Warmup
    for(int i=0;i<g_caif_t5_bench_warmup_iters;++i)
    {
      t5attn.ZeroGradients();
      ctx.SetTraining(true);
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=t5attn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      t5attn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_t5_bench_iters;++i)
    {
      t5attn.ZeroGradients();
      ctx.SetTraining(true);
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=t5attn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      t5attn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    auto end=std::chrono::high_resolution_clock::now();

    const double elapsed_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double per_iter=elapsed_ms/static_cast<double>(g_caif_t5_bench_iters);

    ISE_Out::Out()<<"[BENCH] T5 attention fwd+bwd (batch="
                  <<batch
                  <<",seq="
                  <<seq_len
                  <<",dim="
                  <<dim
                  <<",heads="
                  <<num_heads
                  <<"): "
                  <<per_iter
                  <<" ms/iter\n";
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::Out()<<"[BENCH] T5 fwd+bwd: FAILED ("
                  <<e
                  <<")\n";
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"[BENCH] T5 fwd+bwd: FAILED ("
                  <<e.what()
                  <<")\n";
  }
}

void CAIF_T5AttentionTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF T5 Attention & RPB Tests ===\n\n";

  // Correctness tests
  ISE_Out::Out()<<"--- Correctness ---\n";
  TestRPBForwardBidirectional();
  TestRPBForwardUnidirectional();
  TestRPBShape();
  TestRPBParameters();
  TestT5AttentionForward();
  TestT5AttentionCausal();
  TestT5AttentionBackward();
  TestBaseMHARegression();

  // Benchmarks (sequential — shared GPU)
  ISE_Out::Out()<<"\n--- Benchmarks ---\n";
  BenchmarkRPBForward();
  BenchmarkT5vsBaseMHA();
  BenchmarkT5ForwardBackward();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_T5AttentionTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
