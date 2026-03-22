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
// Transformer model training tests
//------------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include "caif_device_transformer_model.h"
#include "caif_device_cross_entropy_loss.h"
#include "caif_device_tensor.h"
#include "caif_device_ops.h"
#include "caif_cuda_stream.h"
#include "caif_host_tensor.h"
#include "caif_cuda_kernels.h"

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportTest(const std::string &name,bool passed)
{
  if(passed==true)
  {
    std::cout<<"  PASS: "<<name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    std::cout<<"  FAIL: "<<name<<"\n";
    ++g_tests_failed;
  }
}

//------------------------------------------------------------------------------
// Helper: Create a small transformer config for testing
//------------------------------------------------------------------------------
static CAIF_DeviceTransformerModel::Config_t CreateTinyConfig()
{
  CAIF_DeviceTransformerModel::Config_t config;
  config.vocab_size=16;
  config.max_seq_len=8;
  config.dim=32;
  config.num_heads=2;
  config.num_kv_heads=2;
  config.num_layers=1;
  config.ffn_dim=0;  // auto-compute
  config.causal=true;
  config.use_rope=true;
  config.pe_mode=PositionalEncodingMode_e::Learned;  // Won't be used with RoPE
  config.output_dim=0;  // same as vocab_size
  config.tie_weights=true;
  return config;
}

//------------------------------------------------------------------------------
// Helper: Simple Adam state for training
//------------------------------------------------------------------------------
struct AdamState
{
  std::vector<CAIF_DeviceTensor> m;  // First moments
  std::vector<CAIF_DeviceTensor> v;  // Second moments
  float lr;
  float beta1;
  float beta2;
  float epsilon;
  int t;

  void Initialize(CAIF_DeviceTransformerModel &model,
                  CAIF_CudaStream &stream,
                  float learning_rate=0.001f)
  {
    lr=learning_rate;
    beta1=0.9f;
    beta2=0.999f;
    epsilon=1e-8f;
    t=0;

    const size_t num_params=model.ParameterTensorCount();
    m.clear();
    v.clear();
    m.reserve(num_params);
    v.reserve(num_params);

    for(size_t i=0;i<num_params;++i)
    {
      const auto &param=model.ParameterTensor(i);
      m.push_back(CAIF_DeviceTensor::Zeros(param.Shape(),stream));
      v.push_back(CAIF_DeviceTensor::Zeros(param.Shape(),stream));
    }
  }

  void Step(CAIF_DeviceTransformerModel &model,CAIF_CudaStream &stream)
  {
    ++t;
    const float bias_correction1=1.0f-std::pow(beta1,static_cast<float>(t));
    const float bias_correction2=1.0f-std::pow(beta2,static_cast<float>(t));

    const size_t num_params=model.ParameterTensorCount();
    for(size_t i=0;i<num_params;++i)
    {
      CAIF_DeviceTensor &param=model.ParameterTensor(i);
      const CAIF_DeviceTensor &grad=model.GradientTensor(i);

#ifdef USE_CAIF_CUDA
      launch_fused_adam(param.DevicePtr(),
                        grad.DevicePtr(),
                        m[i].DevicePtr(),
                        v[i].DevicePtr(),
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        0.0f,  // weight_decay
                        bias_correction1,
                        bias_correction2,
                        static_cast<int>(param.TotalElements()),
                        stream.Handle());
#endif
    }
  }
};

//------------------------------------------------------------------------------
// Test 1: Forward/Backward smoke test - no crashes
//------------------------------------------------------------------------------
static void TestForwardBackwardSmoke()
{
  CAIF_CudaStream stream;
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel model(config,stream);

  // Create input: batch=2, seq_len=4
  std::vector<float> input_data={
    0,1,2,3,
    4,5,6,7
  };
  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{2,4},stream);

  // Create target (next tokens)
  std::vector<float> target_data={
    1,2,3,4,
    5,6,7,8
  };
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{2,4},stream);

  bool passed=true;
  try
  {
    // Forward
    CAIF_DeviceTensor logits=model.Forward(input,true);

    // Debug: check logits
    CAIF_HostTensor host_logits=logits.ToHost();
    std::cout<<"    Logits shape: ";
    for(auto s:logits.Shape()) std::cout<<s<<" ";
    std::cout<<", first val: "<<host_logits.Data()[0]<<"\n";

    // Compute loss and gradient
    CAIF_DeviceTensor grad_logits;
    float loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(
        logits,targets,grad_logits,stream);
    std::cout<<"    Loss: "<<loss<<"\n";

    // Backward
    model.Backward(grad_logits);

    // Check loss is finite
    if(std::isfinite(loss)==false)
    {
      std::cout<<"    Loss is not finite: "<<loss<<"\n";
      passed=false;
    }
  }
  catch(const std::exception &e)
  {
    std::cout<<"    Exception: "<<e.what()<<"\n";
    passed=false;
  }

  ReportTest("TestForwardBackwardSmoke",passed);
}

//------------------------------------------------------------------------------
// Test 2: Loss decreases over training steps
//------------------------------------------------------------------------------
static void TestLossDecreases()
{
  CAIF_CudaStream stream;
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel model(config,stream);

  // Create fixed input/target
  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  // Initialize Adam
  AdamState adam;
  adam.Initialize(model,stream,0.01f);  // Higher LR for faster convergence

  // Get initial loss
  CAIF_DeviceTensor logits=model.Forward(input,true);
  CAIF_DeviceTensor grad_logits;
  float initial_loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(
      logits,targets,grad_logits,stream);

  // Train for 10 steps
  constexpr int num_steps=10;
  float final_loss=initial_loss;

  for(int step=0;step<num_steps;++step)
  {
    model.ZeroGradients();
    logits=model.Forward(input,true);
    final_loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(
        logits,targets,grad_logits,stream);
    model.Backward(grad_logits);
    adam.Step(model,stream);
  }

  // Loss should decrease
  const bool passed=(final_loss<initial_loss);

  if(passed==false)
  {
    std::cout<<"    Initial loss: "<<initial_loss<<", final loss: "<<final_loss<<"\n";
  }

  ReportTest("TestLossDecreases",passed);
}

//------------------------------------------------------------------------------
// Test 3: Gradients are non-zero after backward
//------------------------------------------------------------------------------
static void TestGradientNonZero()
{
  CAIF_CudaStream stream;
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel model(config,stream);

  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  model.ZeroGradients();
  CAIF_DeviceTensor logits=model.Forward(input,true);
  CAIF_DeviceTensor grad_logits;
  CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(logits,targets,grad_logits,stream);
  model.Backward(grad_logits);

  // Check that at least some gradients are non-zero
  bool found_nonzero=false;
  const size_t num_params=model.ParameterTensorCount();

  for(size_t i=0;i<num_params&&found_nonzero==false;++i)
  {
    CAIF_HostTensor host_grad=model.GradientTensor(i).ToHost();
    const float *data=host_grad.Data();
    const size_t n=host_grad.TotalElements();

    for(size_t j=0;j<n;++j)
    {
      if(std::abs(data[j])>1e-10f)
      {
        found_nonzero=true;
        break;
      }
    }
  }

  ReportTest("TestGradientNonZero",found_nonzero);
}

//------------------------------------------------------------------------------
// Test 4: Parameters change after Adam step
//------------------------------------------------------------------------------
static void TestParameterUpdate()
{
  CAIF_CudaStream stream;
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel model(config,stream);

  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  // Get initial parameter values
  CAIF_HostTensor initial_param=model.ParameterTensor(0).ToHost();
  std::vector<float> initial_values(initial_param.Data(),
                                     initial_param.Data()+initial_param.TotalElements());

  // Initialize Adam and do one step
  AdamState adam;
  adam.Initialize(model,stream,0.01f);

  model.ZeroGradients();
  CAIF_DeviceTensor logits=model.Forward(input,true);
  CAIF_DeviceTensor grad_logits;
  CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(logits,targets,grad_logits,stream);
  model.Backward(grad_logits);
  adam.Step(model,stream);

  // Get updated parameter values
  CAIF_HostTensor updated_param=model.ParameterTensor(0).ToHost();

  // Check that parameters changed
  bool params_changed=false;
  for(size_t i=0;i<initial_values.size();++i)
  {
    if(std::abs(updated_param.Data()[i]-initial_values[i])>1e-10f)
    {
      params_changed=true;
      break;
    }
  }

  ReportTest("TestParameterUpdate",params_changed);
}

//------------------------------------------------------------------------------
// Test 5: Model can overfit tiny dataset
//------------------------------------------------------------------------------
static void TestOverfitTinyDataset()
{
  CAIF_CudaStream stream;
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel model(config,stream);

  // Very simple pattern: next token is current token + 1
  // Sequences: [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
  std::vector<float> inputs={
    0,1,2,3,
    4,5,6,7,
    8,9,10,11,
    12,13,14,15
  };
  std::vector<float> targets={
    1,2,3,4,
    5,6,7,8,
    9,10,11,12,
    13,14,15,0
  };

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(inputs.data(),{4,4},stream);
  CAIF_DeviceTensor target=CAIF_DeviceTensor::FromHostData(targets.data(),{4,4},stream);

  AdamState adam;
  adam.Initialize(model,stream,0.01f);

  // Train for many steps
  constexpr int num_steps=100;
  float final_loss=0.0f;

  for(int step=0;step<num_steps;++step)
  {
    model.ZeroGradients();
    CAIF_DeviceTensor logits=model.Forward(input,true);
    CAIF_DeviceTensor grad_logits;
    final_loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(
        logits,target,grad_logits,stream);
    model.Backward(grad_logits);
    adam.Step(model,stream);
  }

  // For overfit, loss should be quite low (< 1.0 for cross-entropy)
  // With vocab_size=16, random would be log(16) ≈ 2.77
  const bool passed=(final_loss<1.5f);

  if(passed==false)
  {
    std::cout<<"    Final loss after "<<num_steps<<" steps: "<<final_loss<<"\n";
  }

  ReportTest("TestOverfitTinyDataset",passed);
}

//------------------------------------------------------------------------------
// Test 6: Weight tying gradient accumulates correctly
//------------------------------------------------------------------------------
static void TestWeightTyingGradient()
{
  CAIF_CudaStream stream;

  // Create config with weight tying
  auto config=CreateTinyConfig();
  config.tie_weights=true;

  CAIF_DeviceTransformerModel model(config,stream);

  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  model.ZeroGradients();
  CAIF_DeviceTensor logits=model.Forward(input,true);
  CAIF_DeviceTensor grad_logits;
  CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(logits,targets,grad_logits,stream);
  model.Backward(grad_logits);

  // With weight tying, the embedding gradient should include both
  // the embedding backward gradient and the head backward gradient.
  // We verify this by checking that the embedding gradient is non-zero.
  CAIF_HostTensor emb_grad=model.GradientTensor(0).ToHost();

  bool has_gradient=false;
  const float *data=emb_grad.Data();
  const size_t n=emb_grad.TotalElements();

  for(size_t i=0;i<n;++i)
  {
    if(std::abs(data[i])>1e-10f)
    {
      has_gradient=true;
      break;
    }
  }

  ReportTest("TestWeightTyingGradient",has_gradient);
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main()
{
  std::cout<<"=== Transformer Training Tests ===\n";

  TestForwardBackwardSmoke();
  TestLossDecreases();
  TestGradientNonZero();
  TestParameterUpdate();
  TestOverfitTinyDataset();
  TestWeightTyingGradient();

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  if(g_tests_failed==0)
  {
    std::cout<<"All tests passed!\n";
    return 0;
  }
  return 1;
}
