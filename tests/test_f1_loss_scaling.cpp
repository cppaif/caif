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
// Mixed-precision loss scaling (CAIF_LossScaler).
//
// Covers the four pieces of the GradScaler protocol:
//   1. ScaleLossGrad scales the loss seed up by Scale().
//   2. UnscaleCheckInf divides grads by the scale and leaves found_inf clear
//      on finite input.
//   3. UnscaleCheckInf flags found_inf on a non-finite gradient.
//   4. Step() applies the optimizer step on clean grads (and grows the scale
//      after growth_interval clean steps) but SKIPS it on overflow, leaving the
//      parameter untouched and halving the scale.
//
// fp32 params/grads are used so the unscale arithmetic is exact for assertions;
// the kernel path is identical for fp16/bf16 (the launcher is instantiated for
// all three and the math is fp32 internally).
//------------------------------------------------------------------------------
#include "caif_loss_scaler.h"
#include "caif_device_network.h"
#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_ops.h"
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

constexpr uint32_t g_caif_f1_rows=4;
constexpr uint32_t g_caif_f1_cols=4;
constexpr float g_caif_f1_scale=65536.0f;
constexpr float g_caif_f1_p0=1.0f;
constexpr float g_caif_f1_lr=0.01f;
constexpr float g_caif_f1_tol=1e-3f;

//------------------------------------------------------------------------------
// Test-only layer: a single fp32 parameter + fp32 gradient. Forward/Backward
// are unused (the optimizer step is what is probed), so they throw if called.
//------------------------------------------------------------------------------
class CAIF_F1Layer:public CAIF_DeviceLayer
{
  public:
    explicit CAIF_F1Layer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                   _param(MakeZeroFp32(stream)),
                                                   _grad(MakeZeroFp32(stream))
    {
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_F1Layer::ForwardImpl not used in this test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_F1Layer::BackwardImpl not used in this test");
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
      return "CAIF_F1Layer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"weight"};
    }

  protected:

  private:
    static CAIF_DeviceTensor MakeZeroFp32(CAIF_CudaStream &stream)
    {
      return CAIF_DeviceTensor::Zeros({g_caif_f1_rows,g_caif_f1_cols},
                                      stream,
                                      CAIF_DataType::CAIF_DataType_e::Float32);
    }

    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

class CAIF_F1LossScalingTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void FillTensor(CAIF_DeviceTensor &t,const float value);
    static std::vector<float> ReadTensor(const CAIF_DeviceTensor &t);
    static void TestScaleLossGrad();
    static void TestUnscaleFinite();
    static void TestUnscaleFlagsOverflow();
    static void TestStepSkipsOnOverflow();
    static void TestStepAppliesAndGrows();
};

void CAIF_F1LossScalingTest::FillTensor(CAIF_DeviceTensor &t,const float value)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n,value);
  t.CopyFromHost(host.data(),n);
}

std::vector<float> CAIF_F1LossScalingTest::ReadTensor(const CAIF_DeviceTensor &t)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n);
  t.CopyToHost(host.data());
  return host;
}

void CAIF_F1LossScalingTest::TestScaleLossGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_LossScaler scaler(stream,g_caif_f1_scale);
    CAIF_DeviceTensor seed=CAIF_DeviceTensor::Zeros({g_caif_f1_rows},
                                                    stream,
                                                    CAIF_DataType::CAIF_DataType_e::Float32);
    FillTensor(seed,2.0f);
    scaler.ScaleLossGrad(seed);
    const std::vector<float> got=ReadTensor(seed);

    bool ok=true;
    for(size_t i=0;i<got.size();++i)
    {
      if(std::fabs(got[i]-2.0f*g_caif_f1_scale)>g_caif_f1_tol)
      {
        ok=false;
      }
    }
    CAIF_TestHarness::Report("F1::ScaleLossGrad::ScalesByScale",ok);
  }
  CAIF_TEST_CATCH_BLOCK("F1::ScaleLossGrad::ScalesByScale")
}

void CAIF_F1LossScalingTest::TestUnscaleFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceTensor grad=CAIF_DeviceTensor::Zeros({g_caif_f1_rows},
                                                    stream,
                                                    CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor found_inf=CAIF_DeviceTensor::Zeros({1},
                                                         stream,
                                                         CAIF_DataType::CAIF_DataType_e::Float32);
    // grad = 2 * scale  ->  unscaled should be exactly 2 (powers of two).
    FillTensor(grad,2.0f*g_caif_f1_scale);
    CAIF_Ops::UnscaleCheckInf(grad,1.0f/g_caif_f1_scale,found_inf);

    const std::vector<float> g=ReadTensor(grad);
    const std::vector<float> flag=ReadTensor(found_inf);

    bool ok=(std::fabs(flag[0]-0.0f)<g_caif_f1_tol);
    for(size_t i=0;i<g.size();++i)
    {
      if(std::fabs(g[i]-2.0f)>g_caif_f1_tol)
      {
        ok=false;
      }
    }
    CAIF_TestHarness::Report("F1::Unscale::FiniteUnscaledNoFlag",ok);
  }
  CAIF_TEST_CATCH_BLOCK("F1::Unscale::FiniteUnscaledNoFlag")
}

void CAIF_F1LossScalingTest::TestUnscaleFlagsOverflow()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceTensor grad=CAIF_DeviceTensor::Zeros({g_caif_f1_rows},
                                                    stream,
                                                    CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor found_inf=CAIF_DeviceTensor::Zeros({1},
                                                         stream,
                                                         CAIF_DataType::CAIF_DataType_e::Float32);
    std::vector<float> host(grad.TotalElements(),1.0f);
    host[0]=std::numeric_limits<float>::infinity();
    grad.CopyFromHost(host.data(),host.size());
    CAIF_Ops::UnscaleCheckInf(grad,1.0f,found_inf);

    const std::vector<float> flag=ReadTensor(found_inf);
    const bool ok=(std::fabs(flag[0]-1.0f)<g_caif_f1_tol);
    CAIF_TestHarness::Report("F1::Unscale::OverflowSetsFlag",ok);
  }
  CAIF_TEST_CATCH_BLOCK("F1::Unscale::OverflowSetsFlag")
}

void CAIF_F1LossScalingTest::TestStepSkipsOnOverflow()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net(stream);
    std::unique_ptr<CAIF_F1Layer> layer=std::make_unique<CAIF_F1Layer>(stream);
    FillTensor(layer->ParameterTensor(0),g_caif_f1_p0);
    FillTensor(layer->GradientTensor(0),std::numeric_limits<float>::infinity());
    net.AddLayer(std::move(layer));
    net.InitializeAdam(g_caif_f1_lr,
                       g_caif_default_beta1,
                       g_caif_default_beta2,
                       g_caif_adam_epsilon,
                       0.0f);

    CAIF_LossScaler scaler(stream,g_caif_f1_scale);
    const bool applied=scaler.Step(net);

    const std::vector<float> param=ReadTensor(net.Layer(0).ParameterTensor(0));
    bool unchanged=true;
    for(size_t i=0;i<param.size();++i)
    {
      if(std::fabs(param[i]-g_caif_f1_p0)>g_caif_f1_tol)
      {
        unchanged=false;
      }
    }
    const bool scale_backed_off=
      (std::fabs(scaler.Scale()-g_caif_f1_scale*g_caif_loss_scaler_backoff_factor)<g_caif_f1_tol);

    const bool ok=(applied==false)&&(unchanged==true)&&(scale_backed_off==true);
    if(ok==false)
    {
      ISE_Out::Out()<<"  applied="<<applied
                    <<" param0="<<param[0]
                    <<" scale="<<scaler.Scale()
                    <<"\n";
    }
    CAIF_TestHarness::Report("F1::Step::SkipsOnOverflowAndBacksOff",ok);
  }
  CAIF_TEST_CATCH_BLOCK("F1::Step::SkipsOnOverflowAndBacksOff")
}

void CAIF_F1LossScalingTest::TestStepAppliesAndGrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceNetwork net(stream);
    std::unique_ptr<CAIF_F1Layer> layer=std::make_unique<CAIF_F1Layer>(stream);
    FillTensor(layer->ParameterTensor(0),g_caif_f1_p0);
    // A finite, already-scaled gradient: unscales to 1.0, well within range.
    FillTensor(layer->GradientTensor(0),1.0f*g_caif_f1_scale);
    net.AddLayer(std::move(layer));
    net.InitializeAdam(g_caif_f1_lr,
                       g_caif_default_beta1,
                       g_caif_default_beta2,
                       g_caif_adam_epsilon,
                       0.0f);

    // growth_interval=2: the scale should double after two clean steps.
    const uint32_t growth_interval=2;
    CAIF_LossScaler scaler(stream,
                           g_caif_f1_scale,
                           g_caif_loss_scaler_growth_factor,
                           g_caif_loss_scaler_backoff_factor,
                           growth_interval);

    const bool applied1=scaler.Step(net);
    const float scale_after1=scaler.Scale();
    const bool applied2=scaler.Step(net);
    const float scale_after2=scaler.Scale();

    const std::vector<float> param=ReadTensor(net.Layer(0).ParameterTensor(0));
    bool moved=false;
    for(size_t i=0;i<param.size();++i)
    {
      if(std::fabs(param[i]-g_caif_f1_p0)>g_caif_f1_tol)
      {
        moved=true;
      }
    }

    const bool grew=
      (std::fabs(scale_after1-g_caif_f1_scale)<g_caif_f1_tol)&&
      (std::fabs(scale_after2-g_caif_f1_scale*g_caif_loss_scaler_growth_factor)<g_caif_f1_tol);

    const bool ok=(applied1==true)&&(applied2==true)&&(moved==true)&&(grew==true);
    if(ok==false)
    {
      ISE_Out::Out()<<"  applied1="<<applied1
                    <<" applied2="<<applied2
                    <<" moved="<<moved
                    <<" scale1="<<scale_after1
                    <<" scale2="<<scale_after2
                    <<"\n";
    }
    CAIF_TestHarness::Report("F1::Step::AppliesCleanStepAndGrowsScale",ok);
  }
  CAIF_TEST_CATCH_BLOCK("F1::Step::AppliesCleanStepAndGrowsScale")
}

void CAIF_F1LossScalingTest::RunAll()
{
  ISE_Out::Out()<<"=== Feature F1: mixed-precision loss scaling ==="
                <<"\n\n";
  TestScaleLossGrad();
  TestUnscaleFinite();
  TestUnscaleFlagsOverflow();
  TestStepSkipsOnOverflow();
  TestStepAppliesAndGrows();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_F1LossScalingTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
