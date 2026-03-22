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

#include "caif_device_transformer_model.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
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

#ifdef USE_CAIF_CUDA

// Helper to create a basic transformer model config
static CAIF_DeviceTransformerModel::Config_t CreateTestConfig(bool causal,bool use_rope)
{
  CAIF_DeviceTransformerModel::Config_t config;
  config.vocab_size=32;
  config.max_seq_len=16;
  config.dim=16;
  config.num_heads=2;
  config.num_kv_heads=2;
  config.num_layers=2;
  config.ffn_dim=0;  // Auto-compute
  config.causal=causal;
  config.use_rope=use_rope;
  config.pe_mode=PositionalEncodingMode_e::Learned;
  config.output_dim=0;  // Use vocab_size
  config.tie_weights=false;
  return config;
}

//------------------------------------------------------------------------------
// Test 1: Forward output shape
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=8;
    constexpr uint32_t vocab_size=32;

    CAIF_CudaStream stream;
    auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel model(config,stream);

    // Create token IDs as float [batch, seq_len]
    std::vector<float> token_ids(batch*seq_len);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i%vocab_size);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             token_ids.data(),{batch,seq_len},stream);

    CAIF_DeviceTensor output=model.Forward(input,false);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==3&&
                 shape[0]==batch&&
                 shape[1]==seq_len&&
                 shape[2]==vocab_size);

    if(passed==false)
    {
      std::cout<<"  Expected shape [2,8,32], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
    }

    ReportResult("TransformerModel::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward output is non-zero
//------------------------------------------------------------------------------
static void TestForwardNonZero()
{
  try
  {
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel model(config,stream);

    std::vector<float> token_ids(batch*seq_len);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             token_ids.data(),{batch,seq_len},stream);

    CAIF_DeviceTensor output=model.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // Check that output has non-zero values
    float sum_abs=0.0f;
    for(size_t i=0;i<host_output.TotalElements();++i)
    {
      sum_abs+=std::fabs(host_output.Data()[i]);
    }

    bool passed=(sum_abs>0.01f);
    if(passed==false)
    {
      std::cout<<"  Output sum_abs too small: "<<sum_abs<<"\n";
    }

    ReportResult("TransformerModel::ForwardNonZero",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::ForwardNonZero",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Causal vs non-causal produces different output
//------------------------------------------------------------------------------
static void TestCausalDifference()
{
  try
  {
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;

    // Create two models with same weights but different causal setting
    auto config_causal=CreateTestConfig(true,false);
    auto config_noncausal=CreateTestConfig(false,false);

    CAIF_DeviceTransformerModel model_causal(config_causal,stream);
    CAIF_DeviceTransformerModel model_noncausal(config_noncausal,stream);

    // Copy weights from causal to non-causal to ensure same initialization
    for(size_t i=0;i<model_causal.ParameterTensorCount();++i)
    {
      std::vector<float> weights(model_causal.ParameterTensor(i).TotalElements());
      model_causal.ParameterTensor(i).CopyToHost(weights.data());
      model_noncausal.ParameterTensor(i).CopyFromHost(weights.data(),weights.size());
    }

    std::vector<float> token_ids(batch*seq_len);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             token_ids.data(),{batch,seq_len},stream);

    CAIF_DeviceTensor out_causal=model_causal.Forward(input,false);
    CAIF_HostTensor host_causal=out_causal.ToHost();

    CAIF_DeviceTensor out_noncausal=model_noncausal.Forward(input,false);
    CAIF_HostTensor host_noncausal=out_noncausal.ToHost();

    // Check that outputs differ
    float diff_sum=0.0f;
    for(size_t i=0;i<host_causal.TotalElements();++i)
    {
      diff_sum+=std::fabs(host_causal.Data()[i]-host_noncausal.Data()[i]);
    }

    bool passed=(diff_sum>0.001f);
    if(passed==false)
    {
      std::cout<<"  Causal and non-causal outputs are identical (diff_sum="<<diff_sum<<")\n";
    }

    ReportResult("TransformerModel::CausalDifference",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::CausalDifference",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: RoPE vs no RoPE produces different output
//------------------------------------------------------------------------------
static void TestRoPEDifference()
{
  try
  {
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;

    auto config_rope=CreateTestConfig(true,true);
    auto config_no_rope=CreateTestConfig(true,false);

    CAIF_DeviceTransformerModel model_rope(config_rope,stream);
    CAIF_DeviceTransformerModel model_no_rope(config_no_rope,stream);

    // Note: Parameter counts differ (RoPE has no positional encoding params)
    // so we can't copy weights directly. Just verify outputs differ.

    std::vector<float> token_ids(batch*seq_len);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             token_ids.data(),{batch,seq_len},stream);

    CAIF_DeviceTensor out_rope=model_rope.Forward(input,false);
    CAIF_HostTensor host_rope=out_rope.ToHost();

    CAIF_DeviceTensor out_no_rope=model_no_rope.Forward(input,false);
    CAIF_HostTensor host_no_rope=out_no_rope.ToHost();

    // Check that outputs differ (different architectures)
    float diff_sum=0.0f;
    for(size_t i=0;i<host_rope.TotalElements();++i)
    {
      diff_sum+=std::fabs(host_rope.Data()[i]-host_no_rope.Data()[i]);
    }

    bool passed=(diff_sum>0.001f);
    if(passed==false)
    {
      std::cout<<"  RoPE and non-RoPE outputs are identical (diff_sum="<<diff_sum<<")\n";
    }

    ReportResult("TransformerModel::RoPEDifference",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::RoPEDifference",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Backward produces correct gradient shape (empty for token embedding)
//------------------------------------------------------------------------------
static void TestBackwardInputGrad()
{
  try
  {
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=4;
    constexpr uint32_t vocab_size=32;

    CAIF_CudaStream stream;
    auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel model(config,stream);

    std::vector<float> token_ids(batch*seq_len);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i%vocab_size);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             token_ids.data(),{batch,seq_len},stream);

    CAIF_DeviceTensor output=model.Forward(input,true);

    // Create grad_output
    std::vector<float> grad_data(batch*seq_len*vocab_size,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_data.data(),{batch,seq_len,vocab_size},stream);

    CAIF_DeviceTensor grad_input=model.Backward(grad_output);

    // For token embedding, input gradient should be empty (discrete tokens)
    bool passed=(grad_input.TotalElements()==0);
    if(passed==false)
    {
      std::cout<<"  Expected empty grad_input, got "<<grad_input.TotalElements()<<" elements\n";
    }

    ReportResult("TransformerModel::BackwardInputGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::BackwardInputGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Backward weight gradient (finite-difference on one weight)
//------------------------------------------------------------------------------
static void TestBackwardWeightGrad()
{
  try
  {
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;
    constexpr uint32_t vocab_size=32;
    constexpr float h=1e-3f;
    constexpr float tolerance=5e-2f;

    CAIF_CudaStream stream;
    auto config=CreateTestConfig(true,false);

    std::vector<float> token_ids(batch*seq_len);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             token_ids.data(),{batch,seq_len},stream);

    std::vector<float> grad_ones(batch*seq_len*vocab_size,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_ones.data(),{batch,seq_len,vocab_size},stream);

    // Get analytical gradient
    CAIF_DeviceTransformerModel model(config,stream);
    model.ZeroGradients();
    model.Forward(input,true);
    model.Backward(grad_output);

    // Pick a parameter tensor to check (use embedding table = index 0)
    constexpr size_t param_idx=0;
    CAIF_HostTensor host_grad=model.GradientTensor(param_idx).ToHost();

    // Get current weights
    std::vector<float> weights(model.ParameterTensor(param_idx).TotalElements());
    model.ParameterTensor(param_idx).CopyToHost(weights.data());

    // Finite-difference check on a few elements
    bool passed=true;
    constexpr int num_checks=4;
    for(int idx=0;idx<num_checks&&passed==true;++idx)
    {
      // f(w+h)
      std::vector<float> w_plus(weights);
      w_plus[idx]+=h;
      CAIF_DeviceTransformerModel model_p(config,stream);
      model_p.ParameterTensor(param_idx).CopyFromHost(w_plus.data(),w_plus.size());
      CAIF_DeviceTensor out_p=model_p.Forward(input,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<hout_p.TotalElements();++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // f(w-h)
      std::vector<float> w_minus(weights);
      w_minus[idx]-=h;
      CAIF_DeviceTransformerModel model_m(config,stream);
      model_m.ParameterTensor(param_idx).CopyFromHost(w_minus.data(),w_minus.size());
      CAIF_DeviceTensor out_m=model_m.Forward(input,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<hout_m.TotalElements();++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      float numerical=(sum_plus-sum_minus)/(2.0f*h);
      float analytical=host_grad.Data()[idx];
      float diff=std::fabs(numerical-analytical);

      if(diff>tolerance)
      {
        std::cout<<"  Weight grad mismatch at "<<idx<<": analytical="<<analytical
                 <<", numerical="<<numerical<<", diff="<<diff<<"\n";
        passed=false;
      }
    }

    ReportResult("TransformerModel::BackwardWeightGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::BackwardWeightGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Parameter tensor count
//------------------------------------------------------------------------------
static void TestParameterTensorCount()
{
  try
  {
    CAIF_CudaStream stream;
    auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel model(config,stream);

    // Expected counts:
    // - Embedding: 1 (table)
    // - Positional encoding (learned): 1 (pe_table)
    // - Per block: 9 (norm1:1 + attn:4 + norm2:1 + ffn:3 for SwiGLU)
    // - Final norm: 1 (gamma)
    // - Head (untied, no bias): 1 (weight)
    // Total = 1 + 1 + 2*9 + 1 + 1 = 22

    constexpr size_t expected=22;
    size_t actual=model.ParameterTensorCount();

    bool passed=(actual==expected);
    if(passed==false)
    {
      std::cout<<"  Expected "<<expected<<" param tensors, got "<<actual<<"\n";
    }

    ReportResult("TransformerModel::ParameterTensorCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::ParameterTensorCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Total parameter count
//------------------------------------------------------------------------------
static void TestTotalParameterCount()
{
  try
  {
    CAIF_CudaStream stream;

    // Use smaller config for easier calculation
    CAIF_DeviceTransformerModel::Config_t config;
    config.vocab_size=16;
    config.max_seq_len=8;
    config.dim=8;
    config.num_heads=2;
    config.num_kv_heads=2;
    config.num_layers=1;
    config.ffn_dim=32;  // Explicit
    config.causal=true;
    config.use_rope=true;  // No positional encoding params
    config.pe_mode=PositionalEncodingMode_e::Learned;
    config.output_dim=0;
    config.tie_weights=false;

    CAIF_DeviceTransformerModel model(config,stream);

    // Calculate expected:
    // Embedding: vocab_size * dim = 16 * 8 = 128
    // Pos enc: 0 (RoPE)
    // Block (1 layer):
    //   norm1: dim = 8
    //   attn: W_q(8*8) + W_k(8*8) + W_v(8*8) + W_o(8*8) = 256
    //   norm2: dim = 8
    //   ffn (SwiGLU): W_gate(8*32) + W_up(8*32) + W_down(32*8) = 768
    // Final norm: dim = 8
    // Head: dim * vocab_size = 8 * 16 = 128
    // Total = 128 + 0 + 8 + 256 + 8 + 768 + 8 + 128 = 1304

    constexpr size_t expected=1304;
    size_t actual=model.TotalParameterCount();

    bool passed=(actual==expected);
    if(passed==false)
    {
      std::cout<<"  Expected "<<expected<<" params, got "<<actual<<"\n";
    }

    ReportResult("TransformerModel::TotalParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::TotalParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: Zero gradients
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;
    constexpr uint32_t vocab_size=32;

    CAIF_CudaStream stream;
    auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel model(config,stream);

    // Forward + backward to create non-zero gradients
    std::vector<float> token_ids(batch*seq_len);
    for(size_t i=0;i<token_ids.size();++i)
    {
      token_ids[i]=static_cast<float>(i);
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             token_ids.data(),{batch,seq_len},stream);

    model.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*vocab_size,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_ones.data(),{batch,seq_len,vocab_size},stream);
    model.Backward(grad_output);

    // Zero all gradients
    model.ZeroGradients();

    // Check that all gradients are zero
    bool passed=true;
    for(size_t i=0;i<model.ParameterTensorCount()&&passed==true;++i)
    {
      CAIF_HostTensor grad=model.GradientTensor(i).ToHost();
      for(size_t j=0;j<grad.TotalElements();++j)
      {
        if(grad.Data()[j]!=0.0f)
        {
          std::cout<<"  Gradient not zeroed at param "<<i<<" element "<<j
                   <<": "<<grad.Data()[j]<<"\n";
          passed=false;
          break;
        }
      }
    }

    ReportResult("TransformerModel::ZeroGradients",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::ZeroGradients",false);
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
    auto config=CreateTestConfig(true,false);
    CAIF_DeviceTransformerModel model(config,stream);

    const std::string desc=model.Description();
    const std::string expected="TransformerModel(dim=16,heads=2,layers=2,vocab=32)";

    bool passed=(desc==expected);
    if(passed==false)
    {
      std::cout<<"  Expected '"<<expected<<"', got '"<<desc<<"'\n";
    }

    ReportResult("TransformerModel::Description",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerModel::Description",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF DeviceTransformerModel Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestForwardShape();
  TestForwardNonZero();
  TestCausalDifference();
  TestRoPEDifference();
  TestBackwardInputGrad();
  TestBackwardWeightGrad();
  TestParameterTensorCount();
  TestTotalParameterCount();
  TestZeroGradients();
  TestDescription();
#else
  std::cout<<"CUDA not enabled, skipping GPU tests\n";
#endif

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  return (g_tests_failed==0)?0:1;
}
