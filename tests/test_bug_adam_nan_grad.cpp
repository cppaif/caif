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
// fused_adam_kernel silently zeros NaN/Inf gradients.
//
// PyTorch Adam never sanitizes gradients: a NaN gradient propagates into the
// moments and the parameter so divergence is visible and a mixed-precision
// loss-scaler can detect overflow and skip the step. The current kernel does
// `if(isnan(g)||isinf(g)) g=0;` every step, so a NaN gradient leaves the
// parameter finite and unchanged — masking divergence.
//
// This test injects a NaN gradient and asserts the parameter becomes
// non-finite (the NaN propagated). It FAILS against the current code (param
// stays at its initial finite value) and PASSES once the sanitization is
// removed.
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
#include <limits>
#include <memory>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_adamnan_bug_rows=4;
constexpr uint32_t g_caif_adamnan_bug_cols=4;
constexpr float g_caif_adamnan_bug_p0=1.0f;
constexpr float g_caif_adamnan_bug_lr=0.01f;

//------------------------------------------------------------------------------
// Test-only layer: a single fp32 parameter + fp32 gradient. Forward/Backward
// are unused (the optimizer step is what is probed), so they throw if called.
//------------------------------------------------------------------------------
class CAIF_AdamNanBugLayer:public CAIF_DeviceLayer
{
  public:
    explicit CAIF_AdamNanBugLayer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                           _param(MakeZeroFp32(stream)),
                                                           _grad(MakeZeroFp32(stream))
    {
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_AdamNanBugLayer::ForwardImpl not used in this test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_AdamNanBugLayer::BackwardImpl not used in this test");
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
      return "CAIF_AdamNanBugLayer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"weight"};
    }

  protected:

  private:
    static CAIF_DeviceTensor MakeZeroFp32(CAIF_CudaStream &stream)
    {
      return CAIF_DeviceTensor::Zeros({g_caif_adamnan_bug_rows,g_caif_adamnan_bug_cols},
                                      stream,
                                      CAIF_DataType::CAIF_DataType_e::Float32);
    }

    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

class CAIF_AdamNanGradBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void FillTensor(CAIF_DeviceTensor &t,const float value);
    static std::vector<float> ReadTensor(const CAIF_DeviceTensor &t);
    static void TestNanGradientPropagates();
};

void CAIF_AdamNanGradBugTest::FillTensor(CAIF_DeviceTensor &t,const float value)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n,value);
  t.CopyFromHost(host.data(),n);
}

std::vector<float> CAIF_AdamNanGradBugTest::ReadTensor(const CAIF_DeviceTensor &t)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n);
  t.CopyToHost(host.data());
  return host;
}

void CAIF_AdamNanGradBugTest::TestNanGradientPropagates()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net(stream);
    std::unique_ptr<CAIF_AdamNanBugLayer> layer=
      std::make_unique<CAIF_AdamNanBugLayer>(stream);
    FillTensor(layer->ParameterTensor(0),g_caif_adamnan_bug_p0);
    FillTensor(layer->GradientTensor(0),std::numeric_limits<float>::quiet_NaN());
    net.AddLayer(std::move(layer));

    net.InitializeAdam(g_caif_adamnan_bug_lr,
                       g_caif_default_beta1,
                       g_caif_default_beta2,
                       g_caif_adam_epsilon,
                       0.0f);
    net.OptimizerStep();

    const std::vector<float> got=ReadTensor(net.Layer(0).ParameterTensor(0));

    bool propagated=true;
    for(size_t i=0;i<got.size();++i)
    {
      if(std::isfinite(got[i])==true)
      {
        propagated=false;
      }
    }
    if(propagated==false)
    {
      ISE_Out::Out()<<"  param stayed finite (NaN grad was silently zeroed): got[0]="
                    <<got[0]
                    <<"\n";
    }
    CAIF_TestHarness::Report("BugC2::Adam::NanGradientPropagates",propagated);
  }
  CAIF_TEST_CATCH_BLOCK("BugC2::Adam::NanGradientPropagates")
}

void CAIF_AdamNanGradBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug C2: Adam NaN/Inf gradient sanitization ==="
                <<"\n\n";
  TestNanGradientPropagates();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_AdamNanGradBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
