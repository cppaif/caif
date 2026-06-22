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
// Transformer model training tests
//------------------------------------------------------------------------------
#include "caif_device_transformer_model.h"
#include "caif_device_cross_entropy_loss.h"
#include "caif_device_tensor.h"
#include "caif_ops.h"
#include "caif_cuda_stream.h"
#include "caif_host_tensor.h"
#include "caif_cuda_kernels_optimizers.cuh"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>

namespace instance
{

constexpr uint32_t g_caif_transformer_test_vocab_size=16;
constexpr uint32_t g_caif_transformer_test_max_seq_len=8;
constexpr uint32_t g_caif_transformer_test_dim=32;
constexpr uint32_t g_caif_transformer_test_num_heads=2;
constexpr uint32_t g_caif_transformer_test_num_kv_heads=2;
constexpr uint32_t g_caif_transformer_test_num_layers=1;
constexpr float g_caif_transformer_test_adam_lr_default=0.001f;
constexpr float g_caif_transformer_test_adam_lr_fast=0.01f;
constexpr float g_caif_transformer_test_adam_beta1=0.9f;
constexpr float g_caif_transformer_test_adam_beta2=0.999f;
constexpr float g_caif_transformer_test_adam_eps=1e-8f;
constexpr int g_caif_transformer_test_train_steps=10;
constexpr int g_caif_transformer_test_overfit_steps=100;
constexpr float g_caif_transformer_test_overfit_loss_max=1.5f;
constexpr float g_caif_transformer_test_nonzero_tol=1e-10f;

//------------------------------------------------------------------------------
// Simple Adam optimizer state for training tests.
//------------------------------------------------------------------------------
class CAIF_AdamOptState
{
  public:
    CAIF_AdamOptState():_lr(g_caif_transformer_test_adam_lr_default),
                        _beta1(g_caif_transformer_test_adam_beta1),
                        _beta2(g_caif_transformer_test_adam_beta2),
                        _epsilon(g_caif_transformer_test_adam_eps),
                        _t(0)
    {
    }

    void Initialize(CAIF_DeviceTransformerModel<float,float> &model,
                    CAIF_CudaStream &stream,
                    const float learning_rate=g_caif_transformer_test_adam_lr_default);
    void Step(CAIF_DeviceTransformerModel<float,float> &model,CAIF_CudaStream &stream);

  protected:

  private:
    float Lr()const{return _lr;}
    float Beta1()const{return _beta1;}
    float Beta2()const{return _beta2;}
    float Epsilon()const{return _epsilon;}
    int T()const{return _t;}

    float _lr;
    float _beta1;
    float _beta2;
    float _epsilon;
    int _t;
    std::vector<CAIF_DeviceTensor> _m;
    std::vector<CAIF_DeviceTensor> _v;
};

void CAIF_AdamOptState::Initialize(CAIF_DeviceTransformerModel<float,float> &model,
                                   CAIF_CudaStream &stream,
                                   const float learning_rate)
{
  _lr=learning_rate;
  _t=0;
  const size_t num_params=model.ParameterTensorCount();
  _m.clear();
  _v.clear();
  _m.reserve(num_params);
  _v.reserve(num_params);
  for(size_t i=0;i<num_params;++i)
  {
    const auto &param=model.ParameterTensor(i);
    _m.push_back(CAIF_DeviceTensor::Zeros(param.Shape(),stream));
    _v.push_back(CAIF_DeviceTensor::Zeros(param.Shape(),stream));
  }
}

void CAIF_AdamOptState::Step(CAIF_DeviceTransformerModel<float,float> &model,
                             CAIF_CudaStream &stream)
{
  ++_t;
  const float bias_correction1=1.0f-std::pow(Beta1(),static_cast<float>(T()));
  const float bias_correction2=1.0f-std::pow(Beta2(),static_cast<float>(T()));
  const size_t num_params=model.ParameterTensorCount();
  for(size_t i=0;i<num_params;++i)
  {
    CAIF_DeviceTensor &param=model.ParameterTensor(i);
    const CAIF_DeviceTensor &grad=model.GradientTensor(i);
#ifdef USE_CAIF_CUDA
    launch_fused_adam(param.DevicePtr<float>(),
                      grad.DevicePtr<float>(),
                      _m[i].DevicePtr<float>(),
                      _v[i].DevicePtr<float>(),
                      Lr(),
                      Beta1(),
                      Beta2(),
                      Epsilon(),
                      0.0f,
                      bias_correction1,
                      bias_correction2,
                      static_cast<int>(param.TotalElements()),
                      stream.Handle());
#endif
  }
}

//------------------------------------------------------------------------------
// Transformer training correctness tests.
//------------------------------------------------------------------------------
class CAIF_TransformerTrainingTests
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceTransformerModelConfig CreateTinyConfig();

    static void TestForwardBackwardSmoke();
    static void TestLossDecreases();
    static void TestGradientNonZero();
    static void TestParameterUpdate();
    static void TestOverfitTinyDataset();
    static void TestWeightTyingGradient();
};

CAIF_DeviceTransformerModelConfig
CAIF_TransformerTrainingTests::CreateTinyConfig()
{
  // ffn_dim defaults to 0 (auto-compute) and output_dim to 0 (== vocab_size);
  // num_kv_heads is the only optional architecture field overridden here.
  CAIF_DeviceTransformerModelConfig config(g_caif_transformer_test_vocab_size,
                                           g_caif_transformer_test_max_seq_len,
                                           g_caif_transformer_test_dim,
                                           g_caif_transformer_test_num_heads,
                                           g_caif_transformer_test_num_layers,
                                           CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned,
                                           true,
                                           true,
                                           true);
  config.SetNumKvHeads(g_caif_transformer_test_num_kv_heads);
  return config;
}

//------------------------------------------------------------------------------
// Test 1: Forward/Backward smoke test - no crashes
//------------------------------------------------------------------------------
void CAIF_TransformerTrainingTests::TestForwardBackwardSmoke()
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel<float,float> model(config,stream);

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
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor logits=model.Forward(input,ctx);

    CAIF_HostTensor host_logits=logits.ToHost();
    ISE_Out::Out()<<"    Logits shape: ";
    for(auto s:logits.Shape())
    {
      ISE_Out::Out()<<s
                    <<" ";
    }
    ISE_Out::Out()<<", first val: "
                  <<host_logits.Data()[0]
                  <<"\n";

    // Compute loss and gradient
    CAIF_DeviceTensor grad_logits;
    const float loss=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
      logits,targets,grad_logits,stream);
    ISE_Out::Out()<<"    Loss: "
                  <<loss
                  <<"\n";

    // Backward
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    model.Backward(grad_logits,ctx);

    // Check loss is finite
    if(std::isfinite(loss)==false)
    {
      ISE_Out::Out()<<"    Loss is not finite: "
                    <<loss
                    <<"\n";
      passed=false;
    }
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::Out()<<"    Exception: "
                  <<e
                  <<"\n";
    passed=false;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"    Exception: "
                  <<e.what()
                  <<"\n";
    passed=false;
  }

  CAIF_TestHarness::Report("TransformerTraining::ForwardBackwardSmoke",passed);
}

//------------------------------------------------------------------------------
// Test 2: Loss decreases over training steps
//------------------------------------------------------------------------------
void CAIF_TransformerTrainingTests::TestLossDecreases()
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel<float,float> model(config,stream);

  // Create fixed input/target
  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  // Initialize Adam with higher LR for faster convergence
  CAIF_AdamOptState adam;
  adam.Initialize(model,stream,g_caif_transformer_test_adam_lr_fast);

  // Get initial loss
  ctx.SetTraining(true);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor logits=model.Forward(input,ctx);
  CAIF_DeviceTensor grad_logits;
  const float initial_loss=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
    logits,targets,grad_logits,stream);

  // Train for steps
  float final_loss=initial_loss;
  for(int step=0;step<g_caif_transformer_test_train_steps;++step)
  {
    model.ZeroGradients();
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    logits=model.Forward(input,ctx);
    final_loss=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
      logits,targets,grad_logits,stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    model.Backward(grad_logits,ctx);
    adam.Step(model,stream);
  }

  // Loss should decrease
  const bool passed=(final_loss<initial_loss);
  if(passed==false)
  {
    ISE_Out::Out()<<"    Initial loss: "
                  <<initial_loss
                  <<", final loss: "
                  <<final_loss
                  <<"\n";
  }

  CAIF_TestHarness::Report("TransformerTraining::LossDecreases",passed);
}

//------------------------------------------------------------------------------
// Test 3: Gradients are non-zero after backward
//------------------------------------------------------------------------------
void CAIF_TransformerTrainingTests::TestGradientNonZero()
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel<float,float> model(config,stream);

  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  model.ZeroGradients();
  ctx.SetTraining(true);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor logits=model.Forward(input,ctx);
  CAIF_DeviceTensor grad_logits;
  CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
    logits,targets,grad_logits,stream);
  ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
  model.Backward(grad_logits,ctx);

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
      if(std::abs(data[j])>g_caif_transformer_test_nonzero_tol)
      {
        found_nonzero=true;
        break;
      }
    }
  }

  CAIF_TestHarness::Report("TransformerTraining::GradientNonZero",found_nonzero);
}

//------------------------------------------------------------------------------
// Test 4: Parameters change after Adam step
//------------------------------------------------------------------------------
void CAIF_TransformerTrainingTests::TestParameterUpdate()
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel<float,float> model(config,stream);

  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  // Get initial parameter values
  CAIF_HostTensor initial_param=model.ParameterTensor(0).ToHost();
  std::vector<float> initial_values(initial_param.Data(),
                                     initial_param.Data()+initial_param.TotalElements());

  // Initialize Adam and do one step
  CAIF_AdamOptState adam;
  adam.Initialize(model,stream,g_caif_transformer_test_adam_lr_fast);

  model.ZeroGradients();
  ctx.SetTraining(true);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor logits=model.Forward(input,ctx);
  CAIF_DeviceTensor grad_logits;
  CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
    logits,targets,grad_logits,stream);
  ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
  model.Backward(grad_logits,ctx);
  adam.Step(model,stream);

  // Get updated parameter values
  CAIF_HostTensor updated_param=model.ParameterTensor(0).ToHost();

  // Check that parameters changed
  bool params_changed=false;
  for(size_t i=0;i<initial_values.size();++i)
  {
    if(std::abs(updated_param.Data()[i]-initial_values[i])>g_caif_transformer_test_nonzero_tol)
    {
      params_changed=true;
      break;
    }
  }

  CAIF_TestHarness::Report("TransformerTraining::ParameterUpdate",params_changed);
}

//------------------------------------------------------------------------------
// Test 5: Model can overfit tiny dataset
//------------------------------------------------------------------------------
void CAIF_TransformerTrainingTests::TestOverfitTinyDataset()
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  auto config=CreateTinyConfig();
  CAIF_DeviceTransformerModel<float,float> model(config,stream);

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

  CAIF_AdamOptState adam;
  adam.Initialize(model,stream,g_caif_transformer_test_adam_lr_fast);

  // Train for many steps
  float final_loss=0.0f;
  for(int step=0;step<g_caif_transformer_test_overfit_steps;++step)
  {
    model.ZeroGradients();
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor logits=model.Forward(input,ctx);
    CAIF_DeviceTensor grad_logits;
    final_loss=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
      logits,target,grad_logits,stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    model.Backward(grad_logits,ctx);
    adam.Step(model,stream);
  }

  // For overfit, loss should be quite low (< 1.5 for cross-entropy).
  // With vocab_size=16, random would be log(16) ≈ 2.77.
  const bool passed=(final_loss<g_caif_transformer_test_overfit_loss_max);
  if(passed==false)
  {
    ISE_Out::Out()<<"    Final loss after "
                  <<g_caif_transformer_test_overfit_steps
                  <<" steps: "
                  <<final_loss
                  <<"\n";
  }

  CAIF_TestHarness::Report("TransformerTraining::OverfitTinyDataset",passed);
}

//------------------------------------------------------------------------------
// Test 6: Weight tying gradient accumulates correctly
//------------------------------------------------------------------------------
void CAIF_TransformerTrainingTests::TestWeightTyingGradient()
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);

  // Create config with weight tying
  auto config=CreateTinyConfig();
  config.SetTieWeights(true);

  CAIF_DeviceTransformerModel<float,float> model(config,stream);

  std::vector<float> input_data={0,1,2,3};
  std::vector<float> target_data={1,2,3,4};

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{1,4},stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(target_data.data(),{1,4},stream);

  model.ZeroGradients();
  ctx.SetTraining(true);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor logits=model.Forward(input,ctx);
  CAIF_DeviceTensor grad_logits;
  CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
    logits,targets,grad_logits,stream);
  ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
  model.Backward(grad_logits,ctx);

  // With weight tying, the embedding gradient should include both
  // the embedding backward gradient and the head backward gradient.
  // We verify this by checking that the embedding gradient is non-zero.
  CAIF_HostTensor emb_grad=model.GradientTensor(0).ToHost();
  bool has_gradient=false;
  const float *data=emb_grad.Data();
  const size_t n=emb_grad.TotalElements();
  for(size_t i=0;i<n;++i)
  {
    if(std::abs(data[i])>g_caif_transformer_test_nonzero_tol)
    {
      has_gradient=true;
      break;
    }
  }

  CAIF_TestHarness::Report("TransformerTraining::WeightTyingGradient",has_gradient);
}

void CAIF_TransformerTrainingTests::RunAll()
{
  ISE_Out::Out()<<"=== Transformer Training Tests ==="
                <<"\n\n";
  TestForwardBackwardSmoke();
  TestLossDecreases();
  TestGradientNonZero();
  TestParameterUpdate();
  TestOverfitTinyDataset();
  TestWeightTyingGradient();
  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

}//end instance namespace

int main()
{
  instance::CAIF_TransformerTrainingTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
