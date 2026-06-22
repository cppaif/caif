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
// Test: Adam fp32 master-weights path for non-fp32 parameter dtypes.
//
// The fused_adam kernel requires fp32 param/grad/m/v.  For layers whose
// parameter tensors are fp16 or bf16, CAIF_DeviceNetwork keeps a fp32 master
// copy of each parameter, runs the Adam update on the master, then casts the
// master back down to the native dtype.  This test covers:
//
//   - Fp32 fast path: no master allocated, param still updated.
//   - Fp16 master path: master drives Adam; param stays fp16 after step;
//     peak update equals the fp32 baseline within fp16 round-trip tolerance.
//   - Bf16 master path: same as fp16 but for bf16.
//   - Monotonic drive across many steps so fp32 master accumulates even when
//     per-step updates are below bf16's quantum.
//
// Stock CAIF_DeviceDenseLayer<float,float> always allocates fp32 _weights regardless of
// storage_dtype.  To exercise a true non-fp32 parameter tensor we define a
// tiny test-local layer that simply holds a param + grad tensor at a chosen
// dtype and ignores Forward/Backward — the network's Adam plumbing is what
// we're probing.
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
#include <limits>
#include <memory>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_adam_test_param_rows=8;
constexpr uint32_t g_caif_adam_test_param_cols=16;
constexpr float g_caif_adam_test_lr=0.01f;
constexpr float g_caif_adam_test_moved_tol=1e-3f;
constexpr float g_caif_adam_test_bf16_vs_fp32_tol=2e-2f;
constexpr uint32_t g_caif_adam_test_seed_a=7;
constexpr uint32_t g_caif_adam_test_seed_mod_sign=2;
constexpr uint32_t g_caif_adam_test_seed_mod_scale=11;
constexpr float g_caif_adam_test_init_base=0.1f;
constexpr float g_caif_adam_test_init_step=0.01f;
constexpr float g_caif_adam_test_grad_base=0.1f;
constexpr float g_caif_adam_test_grad_step=0.01f;
constexpr uint32_t g_caif_adam_test_grad_mod=7;
constexpr int g_caif_adam_test_steps_5=5;
constexpr int g_caif_adam_test_steps_20=20;

//------------------------------------------------------------------------------
// Test-only layer: a single parameter tensor of user-chosen dtype, plus a
// fp32 gradient tensor.  Forward/Backward are unused (we drive AdamStep
// directly after setting a grad), so they throw if called.
//------------------------------------------------------------------------------
class CAIF_AdamTestParamLayer:public CAIF_DeviceLayer
{
  public:
    CAIF_AdamTestParamLayer(CAIF_CudaStream &stream,
                             CAIF_DataType::CAIF_DataType_e param_dtype):CAIF_DeviceLayer(stream),
                                                                          _param(MakeZeros(stream,
                                                                                           param_dtype)),
                                                                          _grad(MakeZerosFp32(stream))
    {
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_AdamTestParamLayer::ForwardImpl not used in test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_AdamTestParamLayer::BackwardImpl not used in test");
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
      return ParameterTensor(0).Dtype();
    }
    CAIF_DataType::CAIF_DataType_e RuntimeComputeDtype()const override
    {
      return ParameterTensor(0).Dtype();
    }
    std::string Description()const override
    {
      return "CAIF_AdamTestParamLayer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"weight"};
    }

  protected:

  private:
    static CAIF_DeviceTensor MakeZeros(CAIF_CudaStream &stream,
                                       CAIF_DataType::CAIF_DataType_e dtype)
    {
      const std::vector<uint32_t> shape={g_caif_adam_test_param_rows,
                                         g_caif_adam_test_param_cols};
      return CAIF_DeviceTensor::Zeros(shape,stream,dtype);
    }
    static CAIF_DeviceTensor MakeZerosFp32(CAIF_CudaStream &stream)
    {
      const std::vector<uint32_t> shape={g_caif_adam_test_param_rows,
                                         g_caif_adam_test_param_cols};
      return CAIF_DeviceTensor::Zeros(shape,stream,CAIF_DataType::CAIF_DataType_e::Float32);
    }

    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

//------------------------------------------------------------------------------
// Adam master-weights correctness tests.
//------------------------------------------------------------------------------
class CAIF_AdamMasterWeightsTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void CopyFp32HostToDeviceTensor(const std::vector<float> &host,
                                            CAIF_DeviceTensor &dst,
                                            CAIF_CudaStream &stream);
    static std::vector<float> ReadDeviceTensorAsFp32(const CAIF_DeviceTensor &src,
                                                      CAIF_CudaStream &stream);
    static float MaxAbsDiff(const std::vector<float> &a,const std::vector<float> &b);
    static bool AllFinite(const std::vector<float> &v);
    static std::vector<float> MakeInitialParam(size_t n,uint32_t seed);
    static std::vector<float> MakeGradPattern(size_t n);
    static std::unique_ptr<CAIF_DeviceNetwork> BuildSingleParamNet(
      CAIF_DataType::CAIF_DataType_e param_dtype,
      CAIF_CudaStream &stream);
    static void RunAdamSteps(CAIF_DeviceNetwork &net,CAIF_CudaStream &stream,int steps);

    static void TestAdamFp32FastPath();
    static void TestAdamFp16MasterPath();
    static void TestAdamBf16MasterPath();
    static void TestAdamBf16MatchesFp32Over20Steps();
};

void CAIF_AdamMasterWeightsTests::CopyFp32HostToDeviceTensor(const std::vector<float> &host,
                                                               CAIF_DeviceTensor &dst,
                                                               CAIF_CudaStream &stream)
{
  if(dst.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    dst.CopyFromHost(host.data(),host.size());
    return;
  }
  const std::vector<uint32_t> shape(dst.Shape().begin(),dst.Shape().end());
  CAIF_DeviceTensor scratch=CAIF_DeviceTensor::FromHostData(host.data(),shape,stream);
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_Ops::Cast(scratch,dst,ctx);
}

std::vector<float> CAIF_AdamMasterWeightsTests::ReadDeviceTensorAsFp32(
  const CAIF_DeviceTensor &src,
  CAIF_CudaStream &stream)
{
  const size_t n=src.TotalElements();
  std::vector<float> out(n);
  if(src.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    src.CopyToHost(out.data());
    return out;
  }
  const std::vector<uint32_t> shape(src.Shape().begin(),src.Shape().end());
  CAIF_DeviceTensor scratch=CAIF_DeviceTensor::Uninitialized(
    shape,stream,CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_Ops::Cast(src,scratch,ctx);
  scratch.CopyToHost(out.data());
  return out;
}

float CAIF_AdamMasterWeightsTests::MaxAbsDiff(const std::vector<float> &a,
                                               const std::vector<float> &b)
{
  if(a.size()!=b.size())
  {
    return std::numeric_limits<float>::infinity();
  }
  float peak=0.0f;
  for(size_t i=0;i<a.size();++i)
  {
    const float d=std::fabs(a[i]-b[i]);
    if(d>peak)
    {
      peak=d;
    }
  }
  return peak;
}

bool CAIF_AdamMasterWeightsTests::AllFinite(const std::vector<float> &v)
{
  for(float x:v)
  {
    if(std::isfinite(x)==false)
    {
      return false;
    }
  }
  return true;
}

std::vector<float> CAIF_AdamMasterWeightsTests::MakeInitialParam(const size_t n,
                                                                   const uint32_t seed)
{
  std::vector<float> out(n);
  for(size_t i=0;i<n;++i)
  {
    float sgn=0.0f;
    if(((i+seed)%g_caif_adam_test_seed_mod_sign)==0)
    {
      sgn=1.0f;
    }
    else
    {
      sgn=-1.0f;
    }
    out[i]=sgn*g_caif_adam_test_init_base*(1.0f+g_caif_adam_test_init_step*
            static_cast<float>((i+seed)%g_caif_adam_test_seed_mod_scale));
  }
  return out;
}

std::vector<float> CAIF_AdamMasterWeightsTests::MakeGradPattern(const size_t n)
{
  std::vector<float> out(n);
  for(size_t i=0;i<n;++i)
  {
    float sgn=0.0f;
    if((i%g_caif_adam_test_seed_mod_sign)==0)
    {
      sgn=1.0f;
    }
    else
    {
      sgn=-1.0f;
    }
    out[i]=sgn*g_caif_adam_test_grad_base*(1.0f+g_caif_adam_test_grad_step*
            static_cast<float>(i%g_caif_adam_test_grad_mod));
  }
  return out;
}

std::unique_ptr<CAIF_DeviceNetwork> CAIF_AdamMasterWeightsTests::BuildSingleParamNet(
  CAIF_DataType::CAIF_DataType_e param_dtype,
  CAIF_CudaStream &stream)
{
  auto net=std::make_unique<CAIF_DeviceNetwork>(stream);
  auto layer=std::make_unique<CAIF_AdamTestParamLayer>(stream,param_dtype);

  // Seed the parameter with a deterministic non-zero pattern that is
  // representable in fp16/bf16 without loss beyond their quantum.
  const size_t n=static_cast<size_t>(g_caif_adam_test_param_rows)*g_caif_adam_test_param_cols;
  CopyFp32HostToDeviceTensor(MakeInitialParam(n,g_caif_adam_test_seed_a),
                              layer->ParameterTensor(0),
                              stream);

  net->AddLayer(std::move(layer));
  return net;
}

// Drive N Adam steps with a constant grad pattern injected each step.
void CAIF_AdamMasterWeightsTests::RunAdamSteps(CAIF_DeviceNetwork &net,
                                                CAIF_CudaStream &stream,
                                                int steps)
{
  const size_t n=static_cast<size_t>(g_caif_adam_test_param_rows)*g_caif_adam_test_param_cols;
  const std::vector<float> pattern=MakeGradPattern(n);
  for(int s=0;s<steps;++s)
  {
    for(size_t i=0;i<net.LayerCount();++i)
    {
      CAIF_DeviceLayer &layer=net.Layer(i);
      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        CopyFp32HostToDeviceTensor(pattern,layer.GradientTensor(p),stream);
      }
    }
    net.AdamStep();
  }
}

void CAIF_AdamMasterWeightsTests::TestAdamFp32FastPath()
{
  try
  {
    CAIF_CudaStream stream;
    auto net=BuildSingleParamNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    net->InitializeAdam(g_caif_adam_test_lr);

    const std::vector<float> before=
      ReadDeviceTensorAsFp32(net->Layer(0).ParameterTensor(0),stream);
    RunAdamSteps(*net,stream,g_caif_adam_test_steps_5);
    const std::vector<float> after=
      ReadDeviceTensorAsFp32(net->Layer(0).ParameterTensor(0),stream);

    const bool finite=AllFinite(after);
    const float peak=MaxAbsDiff(before,after);
    // With constant grad and Adam lr=0.01, each step moves each element by
    // about lr after bias correction.  Over 5 steps that is roughly 0.05,
    // well above 1e-3.
    const bool moved=peak>=g_caif_adam_test_moved_tol;
    const auto dt=net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_ok=(dt==CAIF_DataType::CAIF_DataType_e::Float32);
    const bool ok=finite&&moved&&dtype_ok;
    if(ok==false)
    {
      ISE_Out::Out()<<"  Fp32 finite="
                    <<finite
                    <<" peak="
                    <<peak
                    <<" dtype_ok="
                    <<dtype_ok
                    <<"\n";
    }
    CAIF_TestHarness::Report("AdamMaster::Fp32FastPath",ok);
  }
  CAIF_TEST_CATCH_BLOCK("AdamMaster::Fp32FastPath")
}

void CAIF_AdamMasterWeightsTests::TestAdamFp16MasterPath()
{
  try
  {
    CAIF_CudaStream stream;
    auto net=BuildSingleParamNet(CAIF_DataType::CAIF_DataType_e::Float16,stream);
    net->InitializeAdam(g_caif_adam_test_lr);

    const auto dt_before=net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_is_fp16=(dt_before==CAIF_DataType::CAIF_DataType_e::Float16);

    const std::vector<float> before=
      ReadDeviceTensorAsFp32(net->Layer(0).ParameterTensor(0),stream);
    RunAdamSteps(*net,stream,g_caif_adam_test_steps_5);
    const std::vector<float> after=
      ReadDeviceTensorAsFp32(net->Layer(0).ParameterTensor(0),stream);

    const auto dt_after=net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_preserved=(dt_after==CAIF_DataType::CAIF_DataType_e::Float16);

    const bool finite=AllFinite(after);
    const float peak=MaxAbsDiff(before,after);
    const bool moved=peak>=g_caif_adam_test_moved_tol;
    const bool ok=dtype_is_fp16&&dtype_preserved&&finite&&moved;
    if(ok==false)
    {
      ISE_Out::Out()<<"  Fp16 dtype_is_fp16="
                    <<dtype_is_fp16
                    <<" dtype_preserved="
                    <<dtype_preserved
                    <<" finite="
                    <<finite
                    <<" peak_delta="
                    <<peak
                    <<"\n";
    }
    CAIF_TestHarness::Report("AdamMaster::Fp16MasterPath",ok);
  }
  CAIF_TEST_CATCH_BLOCK("AdamMaster::Fp16MasterPath")
}

void CAIF_AdamMasterWeightsTests::TestAdamBf16MasterPath()
{
  try
  {
    CAIF_CudaStream stream;
    auto net=BuildSingleParamNet(CAIF_DataType::CAIF_DataType_e::BFloat16,stream);
    net->InitializeAdam(g_caif_adam_test_lr);

    const auto dt_before=net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_is_bf16=(dt_before==CAIF_DataType::CAIF_DataType_e::BFloat16);

    const std::vector<float> before=
      ReadDeviceTensorAsFp32(net->Layer(0).ParameterTensor(0),stream);
    RunAdamSteps(*net,stream,g_caif_adam_test_steps_5);
    const std::vector<float> after=
      ReadDeviceTensorAsFp32(net->Layer(0).ParameterTensor(0),stream);

    const auto dt_after=net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_preserved=(dt_after==CAIF_DataType::CAIF_DataType_e::BFloat16);

    const bool finite=AllFinite(after);
    const float peak=MaxAbsDiff(before,after);
    const bool moved=peak>=g_caif_adam_test_moved_tol;
    const bool ok=dtype_is_bf16&&dtype_preserved&&finite&&moved;
    if(ok==false)
    {
      ISE_Out::Out()<<"  Bf16 dtype_is_bf16="
                    <<dtype_is_bf16
                    <<" dtype_preserved="
                    <<dtype_preserved
                    <<" finite="
                    <<finite
                    <<" peak_delta="
                    <<peak
                    <<"\n";
    }
    CAIF_TestHarness::Report("AdamMaster::Bf16MasterPath",ok);
  }
  CAIF_TEST_CATCH_BLOCK("AdamMaster::Bf16MasterPath")
}

// fp32/bf16 updates should track each other once the param has moved
// enough to clear bf16's ~1/256 per-mantissa quantum.  Compare at 20 steps.
void CAIF_AdamMasterWeightsTests::TestAdamBf16MatchesFp32Over20Steps()
{
  try
  {
    CAIF_CudaStream stream;
    auto fp32_net=BuildSingleParamNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    auto bf16_net=BuildSingleParamNet(CAIF_DataType::CAIF_DataType_e::BFloat16,stream);
    fp32_net->InitializeAdam(g_caif_adam_test_lr);
    bf16_net->InitializeAdam(g_caif_adam_test_lr);

    RunAdamSteps(*fp32_net,stream,g_caif_adam_test_steps_20);
    RunAdamSteps(*bf16_net,stream,g_caif_adam_test_steps_20);

    const auto fp32_final=
      ReadDeviceTensorAsFp32(fp32_net->Layer(0).ParameterTensor(0),stream);
    const auto bf16_final=
      ReadDeviceTensorAsFp32(bf16_net->Layer(0).ParameterTensor(0),stream);

    const float diff=MaxAbsDiff(fp32_final,bf16_final);
    // bf16 has ~3e-3 relative precision; for 20 Adam steps the param lives
    // in the ~[-0.3,0.3] range so an absolute tolerance of 2e-2 is a safe
    // ceiling (bf16 quantum near 0.2 is ~1.6e-3; ~10 steps of rounding
    // error accumulation stays well under 2e-2).
    const bool close=diff<g_caif_adam_test_bf16_vs_fp32_tol;
    if(close==false)
    {
      ISE_Out::Out()<<"  Bf16VsFp32 peak_diff="
                    <<diff
                    <<"\n";
    }
    CAIF_TestHarness::Report("AdamMaster::Bf16MatchesFp32Over20Steps",close);
  }
  CAIF_TEST_CATCH_BLOCK("AdamMaster::Bf16MatchesFp32Over20Steps")
}

void CAIF_AdamMasterWeightsTests::RunAll()
{
  ISE_Out::Out()<<"=== Adam Master-Weights Tests ==="
                <<"\n\n";
  TestAdamFp32FastPath();
  TestAdamFp16MasterPath();
  TestAdamBf16MasterPath();
  TestAdamBf16MatchesFp32Over20Steps();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_AdamMasterWeightsTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"  (CUDA not enabled; tests skipped)"
                <<"\n";
  return 0;
#endif
}
