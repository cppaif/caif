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
// Test: SGD / Momentum / RMSprop / AdaGrad single-step correctness.
//
// CAIF_OptimizerTestLayer holds a single fp32 param + fp32 grad tensor; each
// case fills both with known values, calls the matching
// CAIF_DeviceNetwork::Initialize* shim, runs OptimizerStep() once (or twice
// for the multi-step cases), and asserts the resulting param matches the
// analytic formula for that optimizer. Multi-step cases catch state-vector
// accumulation bugs that a single step would miss.
//------------------------------------------------------------------------------
#include "caif_device_network.h"
#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_ops.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <memory>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_opt_test_rows=4;
constexpr uint32_t g_caif_opt_test_cols=8;
constexpr float g_caif_opt_test_p0=1.0f;
constexpr float g_caif_opt_test_grad=0.1f;
constexpr float g_caif_opt_test_lr=0.01f;
constexpr float g_caif_opt_test_momentum=0.9f;
constexpr float g_caif_opt_test_sgd_wd=0.05f;
constexpr float g_caif_opt_test_rmsprop_alpha=0.99f;
constexpr float g_caif_opt_test_rmsprop_eps=1.0e-8f;
constexpr float g_caif_opt_test_adagrad_eps=1.0e-10f;
constexpr float g_caif_opt_test_tol=1.0e-5f;
constexpr float g_caif_opt_test_tol_loose=1.0e-4f;

//------------------------------------------------------------------------------
// Test-only layer: a single fp32 parameter + fp32 gradient tensor. Forward and
// Backward are unused (the network's optimizer plumbing is what is probed), so
// they throw if called.
//------------------------------------------------------------------------------
class CAIF_OptimizerTestLayer:public CAIF_DeviceLayer
{
  public:
    explicit CAIF_OptimizerTestLayer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                              _param(MakeZeroFp32(stream)),
                                                              _grad(MakeZeroFp32(stream))
    {
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_OptimizerTestLayer::ForwardImpl not used in this test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_OptimizerTestLayer::BackwardImpl not used in this test");
    }
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Dense_e;
    }
    void ZeroGradients()override
    {
      GradientTensor(0).Fill(0.0f);
    }
    size_t ParameterTensorCount()const override
    {
      return 1;
    }
    CAIF_DeviceTensor &ParameterTensor(size_t)override
    {
      return _param;
    }
    const CAIF_DeviceTensor &ParameterTensor(size_t)const override
    {
      return _param;
    }
    CAIF_DeviceTensor &GradientTensor(size_t)override
    {
      return _grad;
    }
    const CAIF_DeviceTensor &GradientTensor(size_t)const override
    {
      return _grad;
    }
    size_t TotalParameterCount()const override
    {
      return ParameterTensor(0).TotalElements();
    }
    CAIF_DataType::CAIF_DataType_e RuntimeStorageDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }
    CAIF_DataType::CAIF_DataType_e RuntimeComputeDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }
    std::string Description()const override
    {
      return "CAIF_OptimizerTestLayer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"weight"};
    }

  protected:

  private:
    static CAIF_DeviceTensor MakeZeroFp32(CAIF_CudaStream &stream)
    {
      return CAIF_DeviceTensor::Zeros({g_caif_opt_test_rows,g_caif_opt_test_cols},
                                      stream,
                                      CAIF_DataType::CAIF_DataType_e::Float32);
    }

    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

//------------------------------------------------------------------------------
// Optimizer single-/multi-step analytic checks.
//------------------------------------------------------------------------------
class CAIF_OptimizerTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void FillTensor(CAIF_DeviceTensor &t,const float value);
    static std::vector<float> ReadTensor(const CAIF_DeviceTensor &t);
    static bool AllClose(const std::vector<float> &got,const float expected,const float tol);
    static CAIF_DeviceNetwork BuildOneLayerNet(CAIF_CudaStream &stream,
                                               const float param_value,
                                               const float grad_value);

    static void TestSgdSingleStep();
    static void TestSgdWeightDecay();
    static void TestMomentumSingleStep();
    static void TestMomentumTwoSteps();
    static void TestRmspropSingleStep();
    static void TestAdaGradSingleStep();
    static void TestAdaGradAccumGrows();
    static void TestStepBeforeInitThrows();
};

void CAIF_OptimizerTests::FillTensor(CAIF_DeviceTensor &t,const float value)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n,value);
  t.CopyFromHost(host.data(),n);
}

std::vector<float> CAIF_OptimizerTests::ReadTensor(const CAIF_DeviceTensor &t)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n);
  t.CopyToHost(host.data());
  return host;
}

bool CAIF_OptimizerTests::AllClose(const std::vector<float> &got,
                                   const float expected,
                                   const float tol)
{
  for(size_t i=0;i<got.size();++i)
  {
    const float diff=std::fabs(got[i]-expected);
    if(diff>tol)
    {
      return false;
    }
  }
  return true;
}

CAIF_DeviceNetwork CAIF_OptimizerTests::BuildOneLayerNet(CAIF_CudaStream &stream,
                                                         const float param_value,
                                                         const float grad_value)
{
  CAIF_DeviceNetwork net(stream);
  std::unique_ptr<CAIF_OptimizerTestLayer> layer=
    std::make_unique<CAIF_OptimizerTestLayer>(stream);
  FillTensor(layer->ParameterTensor(0),param_value);
  FillTensor(layer->GradientTensor(0),grad_value);
  net.AddLayer(std::move(layer));
  return net;
}

//------------------------------------------------------------------------------
// SGD single-step:  param -= lr * (grad + wd*param)
//------------------------------------------------------------------------------
void CAIF_OptimizerTests::TestSgdSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.InitializeSgd(g_caif_opt_test_lr,0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float expected=g_caif_opt_test_p0-g_caif_opt_test_lr*g_caif_opt_test_grad;
    CAIF_TestHarness::Report("Optimizer::Sgd::SingleStep",
                             AllClose(got,expected,g_caif_opt_test_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Sgd::SingleStep")
}

void CAIF_OptimizerTests::TestSgdWeightDecay()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.InitializeSgd(g_caif_opt_test_lr,g_caif_opt_test_sgd_wd);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float decayed_grad=g_caif_opt_test_grad+g_caif_opt_test_sgd_wd*g_caif_opt_test_p0;
    const float expected=g_caif_opt_test_p0-g_caif_opt_test_lr*decayed_grad;
    CAIF_TestHarness::Report("Optimizer::Sgd::WeightDecay",
                             AllClose(got,expected,g_caif_opt_test_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Sgd::WeightDecay")
}

//------------------------------------------------------------------------------
// Momentum single-step:  v = momentum*v + (grad + wd*param);  param -= lr*v
// At t=1 with v0=0:      v1 = grad,  param1 = p0 - lr*grad
//------------------------------------------------------------------------------
void CAIF_OptimizerTests::TestMomentumSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.InitializeMomentum(g_caif_opt_test_lr,g_caif_opt_test_momentum,0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float expected=g_caif_opt_test_p0-g_caif_opt_test_lr*g_caif_opt_test_grad;
    CAIF_TestHarness::Report("Optimizer::Momentum::SingleStep",
                             AllClose(got,expected,g_caif_opt_test_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Momentum::SingleStep")
}

//------------------------------------------------------------------------------
// Two-step momentum:  v2 = momentum*v1 + grad;  param2 = param1 - lr*v2
// With constant grad: v1=g, v2=momentum*g + g = g*(momentum+1)
// param2 = p0 - lr*g - lr*g*(momentum+1) = p0 - lr*g*(2 + momentum)
//------------------------------------------------------------------------------
void CAIF_OptimizerTests::TestMomentumTwoSteps()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.InitializeMomentum(g_caif_opt_test_lr,g_caif_opt_test_momentum,0.0f);
    net.OptimizerStep();
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float steps=2.0f+g_caif_opt_test_momentum;
    const float expected=g_caif_opt_test_p0-g_caif_opt_test_lr*g_caif_opt_test_grad*steps;
    CAIF_TestHarness::Report("Optimizer::Momentum::TwoSteps",
                             AllClose(got,expected,g_caif_opt_test_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Momentum::TwoSteps")
}

//------------------------------------------------------------------------------
// RMSprop single-step: avg_sq = alpha*0 + (1-alpha)*grad^2;
//                      param  = p0 - lr*grad / (sqrt(avg_sq) + epsilon)
//------------------------------------------------------------------------------
void CAIF_OptimizerTests::TestRmspropSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.InitializeRmsprop(g_caif_opt_test_lr,
                          g_caif_opt_test_rmsprop_alpha,
                          g_caif_opt_test_rmsprop_eps,
                          0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float avg_sq=(1.0f-g_caif_opt_test_rmsprop_alpha)*g_caif_opt_test_grad*g_caif_opt_test_grad;
    const float denom=std::sqrt(avg_sq)+g_caif_opt_test_rmsprop_eps;
    const float expected=g_caif_opt_test_p0-g_caif_opt_test_lr*g_caif_opt_test_grad/denom;
    CAIF_TestHarness::Report("Optimizer::Rmsprop::SingleStep",
                             AllClose(got,expected,g_caif_opt_test_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Rmsprop::SingleStep")
}

//------------------------------------------------------------------------------
// AdaGrad single-step: accum = grad^2;
//                      param = p0 - lr*grad / (sqrt(accum) + epsilon)
//------------------------------------------------------------------------------
void CAIF_OptimizerTests::TestAdaGradSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.InitializeAdaGrad(g_caif_opt_test_lr,g_caif_opt_test_adagrad_eps,0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float accum=g_caif_opt_test_grad*g_caif_opt_test_grad;
    const float denom=std::sqrt(accum)+g_caif_opt_test_adagrad_eps;
    const float expected=g_caif_opt_test_p0-g_caif_opt_test_lr*g_caif_opt_test_grad/denom;
    CAIF_TestHarness::Report("Optimizer::AdaGrad::SingleStep",
                             AllClose(got,expected,g_caif_opt_test_tol_loose));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::AdaGrad::SingleStep")
}

//------------------------------------------------------------------------------
// Two-step AdaGrad: accum keeps growing -> per-step LR shrinks.
// step1: accum = g^2;    p1 = p0 - lr*g/(sqrt(g^2)+eps)
// step2: accum = 2*g^2;  p2 = p1 - lr*g/(sqrt(2*g^2)+eps)
//------------------------------------------------------------------------------
void CAIF_OptimizerTests::TestAdaGradAccumGrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.InitializeAdaGrad(g_caif_opt_test_lr,g_caif_opt_test_adagrad_eps,0.0f);
    net.OptimizerStep();
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float g_sq=g_caif_opt_test_grad*g_caif_opt_test_grad;
    const float denom1=std::sqrt(g_sq)+g_caif_opt_test_adagrad_eps;
    const float p1=g_caif_opt_test_p0-g_caif_opt_test_lr*g_caif_opt_test_grad/denom1;
    const float denom2=std::sqrt(2.0f*g_sq)+g_caif_opt_test_adagrad_eps;
    const float expected=p1-g_caif_opt_test_lr*g_caif_opt_test_grad/denom2;
    CAIF_TestHarness::Report("Optimizer::AdaGrad::AccumGrows",
                             AllClose(got,expected,g_caif_opt_test_tol_loose));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::AdaGrad::AccumGrows")
}

//------------------------------------------------------------------------------
// Negative case: calling OptimizerStep before any Initialize* must throw.
//------------------------------------------------------------------------------
void CAIF_OptimizerTests::TestStepBeforeInitThrows()
{
  bool threw=false;
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,g_caif_opt_test_p0,g_caif_opt_test_grad);
    net.OptimizerStep();
  }
  catch(const ISE_Exception &)
  {
    threw=true;
  }
  catch(...)
  {
    threw=false;
  }
  CAIF_TestHarness::Report("Optimizer::StepBeforeInitThrows",threw);
}

void CAIF_OptimizerTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF Optimizer Tests ==="
                <<"\n\n";
  TestSgdSingleStep();
  TestSgdWeightDecay();
  TestMomentumSingleStep();
  TestMomentumTwoSteps();
  TestRmspropSingleStep();
  TestAdaGradSingleStep();
  TestAdaGradAccumGrows();
  TestStepBeforeInitThrows();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_OptimizerTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
