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
// Test: CAIF_DeviceTransformerModel<float,float> end-to-end correctness.
//
// Tests cover forward shape, non-zero output, causal/non-causal difference,
// RoPE vs no-RoPE difference, backward input gradient (empty for discrete
// tokens), backward weight gradient via finite difference, parameter tensor
// count, total parameter count, gradient zeroing, and description string.
//------------------------------------------------------------------------------
#include "caif_device_transformer_model.h"
#include "caif_test_harness.h"
#include "caif_device_network.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_grad_mode.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_tmodel_test_vocab=32;
constexpr uint32_t g_caif_tmodel_test_max_seq=16;
constexpr uint32_t g_caif_tmodel_test_dim=16;
constexpr uint32_t g_caif_tmodel_test_heads=2;
constexpr uint32_t g_caif_tmodel_test_kv_heads=2;
constexpr uint32_t g_caif_tmodel_test_layers=2;
constexpr uint32_t g_caif_tmodel_test_batch=2;
constexpr uint32_t g_caif_tmodel_test_seq=8;
constexpr uint32_t g_caif_tmodel_test_small_batch=1;
constexpr uint32_t g_caif_tmodel_test_small_seq=4;
constexpr float g_caif_tmodel_test_nonzero_tol=0.01f;
constexpr float g_caif_tmodel_test_diff_tol=0.001f;
constexpr float g_caif_tmodel_test_fd_h=1e-3f;
constexpr int g_caif_tmodel_test_fd_count=4;
constexpr size_t g_caif_tmodel_test_param_idx=0;
// Expected parameter tensor count for the standard 2-layer test config:
// embedding:1 + pe(learned):1 + 2*(norm1:1+attn:4+norm2:1+ffn:3) + finalnorm:1 + head:1 = 22
constexpr size_t g_caif_tmodel_test_expected_param_count=22;
// Expected total param count for the small 1-layer RoPE config (see test body for derivation)
constexpr size_t g_caif_tmodel_test_expected_total_params=1304;

//------------------------------------------------------------------------------
// Transformer model tests.
//------------------------------------------------------------------------------
class CAIF_TransformerModelTests
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceTransformerModelConfig CreateTestConfig(const bool causal,const bool use_rope);

    static void TestForwardShape();
    static void TestForwardNonZero();
    static void TestCausalDifference();
    static void TestRoPEDifference();
    static void TestBackwardInputGrad();
    static void TestBackwardWeightGrad(const GradMode_t &mode);
    static void TestParameterTensorCount();
    static void TestTotalParameterCount();
    static void TestZeroGradients();
    static void TestDescription();
};

CAIF_DeviceTransformerModelConfig CAIF_TransformerModelTests::CreateTestConfig(
  const bool causal,
  const bool use_rope)
{
  // ffn_dim defaults to 0 (auto-compute) and output_dim to 0 (== vocab_size);
  // num_kv_heads is the only optional architecture field this test overrides.
  CAIF_DeviceTransformerModelConfig config(g_caif_tmodel_test_vocab,
                                           g_caif_tmodel_test_max_seq,
                                           g_caif_tmodel_test_dim,
                                           g_caif_tmodel_test_heads,
                                           g_caif_tmodel_test_layers,
                                           CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned,
                                           causal,
                                           use_rope,
                                           false);
  config.SetNumKvHeads(g_caif_tmodel_test_kv_heads);
  return config;
}

//------------------------------------------------------------------------------
// Test 1: Forward output shape
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel<float,float> model(config,stream);

    const size_t n_tokens=g_caif_tmodel_test_batch*g_caif_tmodel_test_seq;
    std::vector<float> token_ids(n_tokens);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i%g_caif_tmodel_test_vocab);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      token_ids.data(),
      {g_caif_tmodel_test_batch,g_caif_tmodel_test_seq},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=model.Forward(input,ctx);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==3 &&
                 shape[0]==g_caif_tmodel_test_batch &&
                 shape[1]==g_caif_tmodel_test_seq &&
                 shape[2]==g_caif_tmodel_test_vocab);

    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected shape [2,8,32], got [";
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

    CAIF_TestHarness::Report("TransformerModel::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Forward output is non-zero
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestForwardNonZero()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel<float,float> model(config,stream);

    const size_t n_tokens=g_caif_tmodel_test_small_batch*g_caif_tmodel_test_small_seq;
    std::vector<float> token_ids(n_tokens);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      token_ids.data(),
      {g_caif_tmodel_test_small_batch,g_caif_tmodel_test_small_seq},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=model.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Check that output has non-zero values
    float sum_abs=0.0f;
    for(size_t i=0;i<host_output.TotalElements();++i)
    {
      sum_abs+=std::fabs(host_output.Data()[i]);
    }

    bool passed=(sum_abs>g_caif_tmodel_test_nonzero_tol);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Output sum_abs too small: "
                   <<sum_abs
                   <<"\n";
    }

    CAIF_TestHarness::Report("TransformerModel::ForwardNonZero",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::ForwardNonZero")
}

//------------------------------------------------------------------------------
// Test 3: Causal vs non-causal produces different output
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestCausalDifference()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Create two models with same weights but different causal setting
    const auto config_causal=CreateTestConfig(true,false);
    const auto config_noncausal=CreateTestConfig(false,false);

    CAIF_DeviceTransformerModel<float,float> model_causal(config_causal,stream);
    CAIF_DeviceTransformerModel<float,float> model_noncausal(config_noncausal,stream);

    // Copy weights from causal to non-causal to ensure same initialization
    for(size_t i=0;i<model_causal.ParameterTensorCount();++i)
    {
      std::vector<float> weights(model_causal.ParameterTensor(i).TotalElements());
      model_causal.ParameterTensor(i).CopyToHost(weights.data());
      model_noncausal.ParameterTensor(i).CopyFromHost(weights.data(),weights.size());
    }

    const size_t n_tokens=g_caif_tmodel_test_small_batch*g_caif_tmodel_test_small_seq;
    std::vector<float> token_ids(n_tokens);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      token_ids.data(),
      {g_caif_tmodel_test_small_batch,g_caif_tmodel_test_small_seq},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor out_causal=model_causal.Forward(input,ctx);
    CAIF_HostTensor host_causal=out_causal.ToHost();

    CAIF_DeviceTensor out_noncausal=model_noncausal.Forward(input,ctx);
    CAIF_HostTensor host_noncausal=out_noncausal.ToHost();

    // Check that outputs differ
    float diff_sum=0.0f;
    for(size_t i=0;i<host_causal.TotalElements();++i)
    {
      diff_sum+=std::fabs(host_causal.Data()[i]-host_noncausal.Data()[i]);
    }

    bool passed=(diff_sum>g_caif_tmodel_test_diff_tol);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Causal and non-causal outputs are identical (diff_sum="
                   <<diff_sum
                   <<")\n";
    }

    CAIF_TestHarness::Report("TransformerModel::CausalDifference",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::CausalDifference")
}

//------------------------------------------------------------------------------
// Test 4: RoPE vs no RoPE produces different output
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestRoPEDifference()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const auto config_rope=CreateTestConfig(true,true);
    const auto config_no_rope=CreateTestConfig(true,false);

    CAIF_DeviceTransformerModel<float,float> model_rope(config_rope,stream);
    CAIF_DeviceTransformerModel<float,float> model_no_rope(config_no_rope,stream);

    // Note: Parameter counts differ (RoPE has no positional encoding params)
    // so we can't copy weights directly. Just verify outputs differ.

    const size_t n_tokens=g_caif_tmodel_test_small_batch*g_caif_tmodel_test_small_seq;
    std::vector<float> token_ids(n_tokens);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      token_ids.data(),
      {g_caif_tmodel_test_small_batch,g_caif_tmodel_test_small_seq},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor out_rope=model_rope.Forward(input,ctx);
    CAIF_HostTensor host_rope=out_rope.ToHost();

    CAIF_DeviceTensor out_no_rope=model_no_rope.Forward(input,ctx);
    CAIF_HostTensor host_no_rope=out_no_rope.ToHost();

    // Check that outputs differ (different architectures)
    float diff_sum=0.0f;
    for(size_t i=0;i<host_rope.TotalElements();++i)
    {
      diff_sum+=std::fabs(host_rope.Data()[i]-host_no_rope.Data()[i]);
    }

    bool passed=(diff_sum>g_caif_tmodel_test_diff_tol);
    if(passed==false)
    {
      ISE_Out::Out()<<"  RoPE and non-RoPE outputs are identical (diff_sum="
                   <<diff_sum
                   <<")\n";
    }

    CAIF_TestHarness::Report("TransformerModel::RoPEDifference",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::RoPEDifference")
}

//------------------------------------------------------------------------------
// Test 5: Backward produces correct gradient shape (empty for token embedding)
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestBackwardInputGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel<float,float> model(config,stream);

    const size_t n_tokens=g_caif_tmodel_test_batch*g_caif_tmodel_test_small_seq;
    std::vector<float> token_ids(n_tokens);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i%g_caif_tmodel_test_vocab);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      token_ids.data(),
      {g_caif_tmodel_test_batch,g_caif_tmodel_test_small_seq},
      stream);

    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=model.Forward(input,ctx);

    // Create grad_output
    const size_t n_grad=g_caif_tmodel_test_batch*g_caif_tmodel_test_small_seq*g_caif_tmodel_test_vocab;
    std::vector<float> grad_data(n_grad,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
      grad_data.data(),
      {g_caif_tmodel_test_batch,g_caif_tmodel_test_small_seq,g_caif_tmodel_test_vocab},
      stream);

    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=model.Backward(grad_output,ctx);

    // For token embedding, input gradient should be empty (discrete tokens)
    bool passed=(grad_input.TotalElements()==0);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected empty grad_input, got "
                   <<grad_input.TotalElements()
                   <<" elements\n";
    }

    CAIF_TestHarness::Report("TransformerModel::BackwardInputGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::BackwardInputGrad")
}

//------------------------------------------------------------------------------
// Test 6: Backward weight gradient (finite-difference on one weight)
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestBackwardWeightGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("TransformerModel::BackwardWeightGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const float tolerance=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=CreateTestConfig(true,false);

    const size_t n_tokens=g_caif_tmodel_test_small_batch*g_caif_tmodel_test_small_seq;
    std::vector<float> token_ids(n_tokens);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      token_ids.data(),
      {g_caif_tmodel_test_small_batch,g_caif_tmodel_test_small_seq},
      stream);

    const size_t n_grad=g_caif_tmodel_test_small_batch*g_caif_tmodel_test_small_seq*g_caif_tmodel_test_vocab;
    std::vector<float> grad_ones(n_grad,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {g_caif_tmodel_test_small_batch,g_caif_tmodel_test_small_seq,g_caif_tmodel_test_vocab},
      stream);

    // Get analytical gradient
    CAIF_DeviceTransformerModel<float,float> model(config,stream);
    model.ZeroGradients();
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    model.Forward(input,ctx);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    model.Backward(grad_output,ctx);

    // Pick a parameter tensor to check (use embedding table = index 0)
    CAIF_HostTensor host_grad=model.GradientTensor(g_caif_tmodel_test_param_idx).ToHost();

    // Get current weights
    std::vector<float> weights(model.ParameterTensor(g_caif_tmodel_test_param_idx).TotalElements());
    model.ParameterTensor(g_caif_tmodel_test_param_idx).CopyToHost(weights.data());

    // FD reference must be high-precision regardless of outer mode
    // (TF32 FD = catastrophic cancellation).
    const bool fd_prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);

    // Finite-difference check on a few elements
    bool passed=true;
    for(int idx=0;idx<g_caif_tmodel_test_fd_count && passed==true;++idx)
    {
      // f(w+h)
      std::vector<float> w_plus(weights);
      w_plus[idx]+=g_caif_tmodel_test_fd_h;
      CAIF_DeviceTransformerModel<float,float> model_p(config,stream);
      model_p.ParameterTensor(g_caif_tmodel_test_param_idx).CopyFromHost(w_plus.data(),w_plus.size());
      ctx.SetTraining(false);
      CAIF_DeviceTensor out_p=model_p.Forward(input,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<hout_p.TotalElements();++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // f(w-h)
      std::vector<float> w_minus(weights);
      w_minus[idx]-=g_caif_tmodel_test_fd_h;
      CAIF_DeviceTransformerModel<float,float> model_m(config,stream);
      model_m.ParameterTensor(g_caif_tmodel_test_param_idx).CopyFromHost(w_minus.data(),w_minus.size());
      CAIF_DeviceTensor out_m=model_m.Forward(input,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<hout_m.TotalElements();++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_tmodel_test_fd_h);
      const float analytical=host_grad.Data()[idx];
      const float diff=std::fabs(numerical-analytical);

      if(diff>tolerance*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  Weight grad mismatch at "
                     <<idx
                     <<": analytical="
                     <<analytical
                     <<", numerical="
                     <<numerical
                     <<", diff="
                     <<diff
                     <<"\n";
        passed=false;
      }
    }
    CAIF_Settings::SetPreciseGradients(fd_prev_precise);

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::BackwardWeightGrad")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 7: Parameter tensor count
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestParameterTensorCount()
{
  try
  {
    CAIF_CudaStream stream;
    const auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel<float,float> model(config,stream);

    const size_t actual=model.ParameterTensorCount();

    bool passed=(actual==g_caif_tmodel_test_expected_param_count);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected "
                   <<g_caif_tmodel_test_expected_param_count
                   <<" param tensors, got "
                   <<actual
                   <<"\n";
    }

    CAIF_TestHarness::Report("TransformerModel::ParameterTensorCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::ParameterTensorCount")
}

//------------------------------------------------------------------------------
// Test 8: Total parameter count
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestTotalParameterCount()
{
  try
  {
    CAIF_CudaStream stream;

    // Use smaller config for easier calculation:
    // Embedding: 16*8=128, Pos enc: 0 (RoPE)
    // Block (1 layer): norm1:8 + attn:W_q(64)+W_k(64)+W_v(64)+W_o(64)=256 + norm2:8 + ffn:768
    // Final norm: 8, Head: 8*16=128  => total=1304
    CAIF_DeviceTransformerModelConfig config(16,
                                             8,
                                             8,
                                             2,
                                             1,
                                             CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned,
                                             true,
                                             true,
                                             false);
    config.SetNumKvHeads(2);
    // Explicit FFN dim
    config.SetFfnDim(32);

    CAIF_DeviceTransformerModel<float,float> model(config,stream);

    const size_t actual=model.TotalParameterCount();

    bool passed=(actual==g_caif_tmodel_test_expected_total_params);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected "
                   <<g_caif_tmodel_test_expected_total_params
                   <<" params, got "
                   <<actual
                   <<"\n";
    }

    CAIF_TestHarness::Report("TransformerModel::TotalParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::TotalParameterCount")
}

//------------------------------------------------------------------------------
// Test 9: Zero gradients
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestZeroGradients()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel<float,float> model(config,stream);

    // Forward + backward to create non-zero gradients
    const size_t n_tokens=g_caif_tmodel_test_small_batch*g_caif_tmodel_test_small_seq;
    std::vector<float> token_ids(n_tokens);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      token_ids.data(),
      {g_caif_tmodel_test_small_batch,g_caif_tmodel_test_small_seq},
      stream);

    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    model.Forward(input,ctx);

    const size_t n_grad=g_caif_tmodel_test_small_batch*g_caif_tmodel_test_small_seq*g_caif_tmodel_test_vocab;
    std::vector<float> grad_ones(n_grad,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {g_caif_tmodel_test_small_batch,g_caif_tmodel_test_small_seq,g_caif_tmodel_test_vocab},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    model.Backward(grad_output,ctx);

    // Zero all gradients
    model.ZeroGradients();

    // Check that all gradients are zero
    bool passed=true;
    for(size_t i=0;i<model.ParameterTensorCount() && passed==true;++i)
    {
      CAIF_HostTensor grad=model.GradientTensor(i).ToHost();
      for(size_t j=0;j<grad.TotalElements();++j)
      {
        if(grad.Data()[j]!=0.0f)
        {
          ISE_Out::Out()<<"  Gradient not zeroed at param "
                       <<i
                       <<" element "
                       <<j
                       <<": "
                       <<grad.Data()[j]
                       <<"\n";
          passed=false;
          break;
        }
      }
    }

    CAIF_TestHarness::Report("TransformerModel::ZeroGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::ZeroGradients")
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
void CAIF_TransformerModelTests::TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    const auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel<float,float> model(config,stream);

    const std::string desc=model.Description();
    const std::string expected="TransformerModel(dim=16,heads=2,layers=2,vocab=32)";

    bool passed=(desc==expected);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected '"
                   <<expected
                   <<"', got '"
                   <<desc
                   <<"'\n";
    }

    CAIF_TestHarness::Report("TransformerModel::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerModel::Description")
}

void CAIF_TransformerModelTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF DeviceTransformerModel Tests ==="
               <<"\n\n";
  TestForwardShape();
  TestForwardNonZero();
  TestCausalDifference();
  TestRoPEDifference();
  TestBackwardInputGrad();
  TestBackwardWeightGrad(g_caif_grad_mode_precise);
  TestBackwardWeightGrad(g_caif_grad_mode_tf32);
  TestParameterTensorCount();
  TestTotalParameterCount();
  TestZeroGradients();
  TestDescription();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_TransformerModelTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
               <<"\n";
  return 0;
#endif
}
