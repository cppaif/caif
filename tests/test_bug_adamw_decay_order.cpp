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
// AdamW weight decay is applied to the POST-step weight.
//
// PyTorch AdamW decouples weight decay by shrinking the weight BEFORE the
// adaptive step:  p <- p*(1 - lr*wd);  p <- p - lr*m_hat/(sqrt(v_hat)+eps).
// The current fused_adam_kernel instead does the adaptive step first and then
// p <- p - lr*wd*p on the already-updated weight, which differs by the
// second-order term lr^2*wd*g/(sqrt(g^2)+eps).
//
// At step 1 with zero initial moments, bias correction makes m_hat=g and
// v_hat=g^2 exactly, so the correct single-step result is analytic:
//   expected = p0*(1 - lr*wd) - lr*g/(sqrt(g^2)+eps)
// This test asserts that correct value. It FAILS against the current code and
// PASSES once the decay is moved ahead of the adaptive step.
//------------------------------------------------------------------------------
#include "caif_device_network.h"
#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <memory>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_adamw_bug_rows=1;
constexpr uint32_t g_caif_adamw_bug_cols=1;
constexpr float g_caif_adamw_bug_p0=1.0f;
constexpr float g_caif_adamw_bug_grad=1.0f;
constexpr float g_caif_adamw_bug_lr=0.1f;
constexpr float g_caif_adamw_bug_wd=0.5f;
constexpr float g_caif_adamw_bug_tol=1.0e-4f;

//------------------------------------------------------------------------------
// Test-only layer: a single fp32 parameter + fp32 gradient. Forward/Backward
// are unused (the optimizer step is what is probed), so they throw if called.
//------------------------------------------------------------------------------
class CAIF_AdamWDecayBugLayer:public CAIF_DeviceLayer
{
  public:
    explicit CAIF_AdamWDecayBugLayer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                              _param(MakeZeroFp32(stream)),
                                                              _grad(MakeZeroFp32(stream))
    {
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_AdamWDecayBugLayer::ForwardImpl not used in this test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_AdamWDecayBugLayer::BackwardImpl not used in this test");
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
      return "CAIF_AdamWDecayBugLayer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"weight"};
    }

  protected:

  private:
    static CAIF_DeviceTensor MakeZeroFp32(CAIF_CudaStream &stream)
    {
      return CAIF_DeviceTensor::Zeros({g_caif_adamw_bug_rows,g_caif_adamw_bug_cols},
                                      stream,
                                      CAIF_DataType::CAIF_DataType_e::Float32);
    }

    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

class CAIF_AdamWDecayOrderBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void FillTensor(CAIF_DeviceTensor &t,const float value);
    static std::vector<float> ReadTensor(const CAIF_DeviceTensor &t);
    static void TestDecoupledDecayOrder();
};

void CAIF_AdamWDecayOrderBugTest::FillTensor(CAIF_DeviceTensor &t,const float value)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n,value);
  t.CopyFromHost(host.data(),n);
}

std::vector<float> CAIF_AdamWDecayOrderBugTest::ReadTensor(const CAIF_DeviceTensor &t)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n);
  t.CopyToHost(host.data());
  return host;
}

void CAIF_AdamWDecayOrderBugTest::TestDecoupledDecayOrder()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net(stream);
    std::unique_ptr<CAIF_AdamWDecayBugLayer> layer=
      std::make_unique<CAIF_AdamWDecayBugLayer>(stream);
    FillTensor(layer->ParameterTensor(0),g_caif_adamw_bug_p0);
    FillTensor(layer->GradientTensor(0),g_caif_adamw_bug_grad);
    net.AddLayer(std::move(layer));

    net.InitializeAdam(g_caif_adamw_bug_lr,
                       g_caif_default_beta1,
                       g_caif_default_beta2,
                       g_caif_adam_epsilon,
                       g_caif_adamw_bug_wd);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));

    const float g=g_caif_adamw_bug_grad;
    const float denom=std::sqrt(g*g)+g_caif_adam_epsilon;
    const float decayed=g_caif_adamw_bug_p0*(1.0f-g_caif_adamw_bug_lr*g_caif_adamw_bug_wd);
    const float expected=decayed-g_caif_adamw_bug_lr*g/denom;

    bool ok=true;
    for(size_t i=0;i<got.size();++i)
    {
      if(std::fabs(got[i]-expected)>g_caif_adamw_bug_tol)
      {
        ok=false;
      }
    }
    if(ok==false)
    {
      ISE_Out::Out()<<"  got="
                    <<got[0]
                    <<" expected(decoupled)="
                    <<expected
                    <<"\n";
    }
    CAIF_TestHarness::Report("BugC1::AdamW::DecoupledDecayOrder",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugC1::AdamW::DecoupledDecayOrder")
}

void CAIF_AdamWDecayOrderBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug C1: AdamW decoupled weight-decay order ==="
                <<"\n\n";
  TestDecoupledDecayOrder();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_AdamWDecayOrderBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
