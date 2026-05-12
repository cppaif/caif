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
// Pattern mirrors test_adam_master_weights.cpp: a TestParamLayer holds a
// single fp32 param + fp32 grad tensor, the test fills both with known
// values, calls the corresponding CAIF_DeviceNetwork::Initialize* shim,
// runs OptimizerStep() once, and asserts the resulting param matches
// the analytic formula for that optimizer.
//
// Multi-step monotonic-loss tests would catch state-update bugs that
// single-step doesn't (state vector accumulates incorrectly across
// steps).  Added below as a second case per optimizer.
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

#include <cmath>
#include <memory>
#include <vector>

using namespace instance;

namespace
{

void ReportResult(const char *name,bool ok)
{
  CAIF_TestHarness::Report(name,ok);
}

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_test_rows=4;
constexpr uint32_t g_test_cols=8;

class TestParamLayer:public CAIF_DeviceLayer
{
  public:
    explicit TestParamLayer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream)
    {
      const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
      const std::vector<uint32_t> shape={g_test_rows,g_test_cols};
      _param=CAIF_DeviceTensor::Zeros(shape,stream,fp32);
      _grad=CAIF_DeviceTensor::Zeros(shape,stream,fp32);
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,
                                  CAIF_RunContext &)override
    {
      THROW_CAIFE("TestParamLayer::ForwardImpl not used in this test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,
                                   CAIF_RunContext &)override
    {
      THROW_CAIFE("TestParamLayer::BackwardImpl not used in this test");
    }
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Dense_e;
    }
    void ZeroGradients()override
    {
      _grad.Fill(0.0f);
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
      return _param.TotalElements();
    }
    std::string Description()const override
    {
      return "TestParamLayer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"weight"};
    }

  private:
    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

void FillTensor(CAIF_DeviceTensor &t,const float value)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n,value);
  t.CopyFromHost(host.data(),n);
}

std::vector<float> ReadTensor(const CAIF_DeviceTensor &t)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n);
  t.CopyToHost(host.data());
  return host;
}

bool AllClose(const std::vector<float> &got,
              const float expected,
              const float tol=1e-5f)
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

CAIF_DeviceNetwork BuildOneLayerNet(CAIF_CudaStream &stream,
                                    const float param_value,
                                    const float grad_value)
{
  CAIF_DeviceNetwork net(stream);
  std::unique_ptr<TestParamLayer> layer=std::make_unique<TestParamLayer>(stream);
  FillTensor(layer->ParameterTensor(0),param_value);
  FillTensor(layer->GradientTensor(0),grad_value);
  net.AddLayer(std::move(layer));
  return net;
}

//------------------------------------------------------------------------------
// SGD single-step:  param -= lr * (grad + wd*param)
//------------------------------------------------------------------------------
void TestSgdSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    const float p0=1.0f;
    const float g=0.1f;
    const float lr=0.01f;
    const float wd=0.0f;

    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,p0,g);
    net.InitializeSgd(lr,wd);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float expected=p0-lr*g;
    ReportResult("Optimizer::Sgd::SingleStep",AllClose(got,expected));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Sgd::SingleStep")
}

void TestSgdWeightDecay()
{
  try
  {
    CAIF_CudaStream stream;
    const float p0=1.0f;
    const float g=0.1f;
    const float lr=0.01f;
    const float wd=0.05f;

    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,p0,g);
    net.InitializeSgd(lr,wd);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float expected=p0-lr*(g+wd*p0);
    ReportResult("Optimizer::Sgd::WeightDecay",AllClose(got,expected));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Sgd::WeightDecay")
}

//------------------------------------------------------------------------------
// Momentum single-step:  v = momentum*v + (grad + wd*param);  param -= lr*v
// At t=1 with v0=0:      v1 = grad,  param1 = p0 - lr*grad
//------------------------------------------------------------------------------
void TestMomentumSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    const float p0=1.0f;
    const float g=0.1f;
    const float lr=0.01f;
    const float momentum=0.9f;

    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,p0,g);
    net.InitializeMomentum(lr,momentum,0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float expected=p0-lr*g;
    ReportResult("Optimizer::Momentum::SingleStep",AllClose(got,expected));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Momentum::SingleStep")
}

// Two-step momentum:  v2 = momentum*v1 + grad;  param2 = param1 - lr*v2
// With constant grad: v1=g, v2=momentum*g + g = g*(momentum+1)
// param2 = p0 - lr*g - lr*g*(momentum+1) = p0 - lr*g*(2 + momentum)
void TestMomentumTwoSteps()
{
  try
  {
    CAIF_CudaStream stream;
    const float p0=1.0f;
    const float g=0.1f;
    const float lr=0.01f;
    const float momentum=0.9f;

    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,p0,g);
    net.InitializeMomentum(lr,momentum,0.0f);
    net.OptimizerStep();
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float expected=p0-lr*g*(2.0f+momentum);
    ReportResult("Optimizer::Momentum::TwoSteps",AllClose(got,expected,1e-5f));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Momentum::TwoSteps")
}

//------------------------------------------------------------------------------
// RMSprop single-step: avg_sq = alpha*0 + (1-alpha)*grad^2;
//                      param  = p0 - lr*grad / (sqrt(avg_sq) + epsilon)
//------------------------------------------------------------------------------
void TestRmspropSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    const float p0=1.0f;
    const float g=0.1f;
    const float lr=0.01f;
    const float alpha=0.99f;
    const float eps=1e-8f;

    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,p0,g);
    net.InitializeRmsprop(lr,alpha,eps,0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float avg_sq=(1.0f-alpha)*g*g;
    const float expected=p0-lr*g/(std::sqrt(avg_sq)+eps);
    ReportResult("Optimizer::Rmsprop::SingleStep",AllClose(got,expected));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::Rmsprop::SingleStep")
}

//------------------------------------------------------------------------------
// AdaGrad single-step: accum = grad^2;
//                      param = p0 - lr*grad / (sqrt(accum) + epsilon)
//------------------------------------------------------------------------------
void TestAdaGradSingleStep()
{
  try
  {
    CAIF_CudaStream stream;
    const float p0=1.0f;
    const float g=0.1f;
    const float lr=0.01f;
    const float eps=1e-10f;

    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,p0,g);
    net.InitializeAdaGrad(lr,eps,0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float accum=g*g;
    const float expected=p0-lr*g/(std::sqrt(accum)+eps);
    ReportResult("Optimizer::AdaGrad::SingleStep",AllClose(got,expected,1e-4f));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::AdaGrad::SingleStep")
}

// Two-step AdaGrad: accum keeps growing -> per-step LR shrinks.
// step1: accum = g^2;       p1 = p0 - lr*g/(sqrt(g^2)+eps) ~= p0 - lr (when eps tiny)
// step2: accum = 2*g^2;     p2 = p1 - lr*g/(sqrt(2)*|g| + eps) ~= p1 - lr/sqrt(2)
void TestAdaGradAccumGrows()
{
  try
  {
    CAIF_CudaStream stream;
    const float p0=1.0f;
    const float g=0.1f;
    const float lr=0.01f;
    const float eps=1e-10f;

    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,p0,g);
    net.InitializeAdaGrad(lr,eps,0.0f);
    net.OptimizerStep();
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));
    const float p1=p0-lr*g/(std::sqrt(g*g)+eps);
    const float expected=p1-lr*g/(std::sqrt(2.0f*g*g)+eps);
    ReportResult("Optimizer::AdaGrad::AccumGrows",AllClose(got,expected,1e-4f));
  }
  CAIF_TEST_CATCH_BLOCK("Optimizer::AdaGrad::AccumGrows")
}

//------------------------------------------------------------------------------
// Negative case: calling OptimizerStep before any Initialize* must throw.
//------------------------------------------------------------------------------
void TestStepBeforeInitThrows()
{
  bool threw=false;
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net=BuildOneLayerNet(stream,1.0f,0.1f);
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
  ReportResult("Optimizer::StepBeforeInitThrows",threw);
}

#endif// USE_CAIF_CUDA

}// anon namespace

int main()
{
#ifdef USE_CAIF_CUDA
  std::cout<<"=== CAIF Optimizer Tests ==="<<std::endl;
  std::cout<<""<<std::endl;
  TestSgdSingleStep();
  TestSgdWeightDecay();
  TestMomentumSingleStep();
  TestMomentumTwoSteps();
  TestRmspropSingleStep();
  TestAdaGradSingleStep();
  TestAdaGradAccumGrows();
  TestStepBeforeInitThrows();
  return CAIF_TestHarness::FinalExitCode();
#else
  std::cout<<"Skipped (USE_CAIF_CUDA not defined)"<<std::endl;
  return 0;
#endif
}
