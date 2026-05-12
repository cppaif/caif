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
// Test: CAIF_DeviceNetwork::Load on-load dtype cast.
//
// The old path used CopyToHost(float*) + CopyFromHost(float*) which only
// worked when both the on-disk tensor and the in-memory parameter were
// fp32.  The updated path:
//   - same-dtype → raw device-to-device byte copy (works for any dtype,
//     including packed int4/int8).
//   - float-to-float cross dtype (fp32↔fp16↔bf16) → CAIF_Ops::Cast.
//   - any other cross-dtype → throws with a clear error.
//
// This test exercises each path with a minimal test-local layer that owns
// a single parameter tensor of a chosen dtype, so we don't depend on any
// particular layer's storage_dtype behaviour.
//------------------------------------------------------------------------------
#include "caif_device_network.h"
#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_ops.h"
#include "caif_safetensors_format.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace instance;

namespace
{

void ReportResult(const char *name,bool ok)
{
  CAIF_TestHarness::Report(name,ok);
}

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_rows=8;
constexpr uint32_t g_cols=16;

class TestParamLayer:public CAIF_DeviceLayer
{
  public:
    TestParamLayer(CAIF_CudaStream &stream,
                   CAIF_DataType::CAIF_DataType_e param_dtype):CAIF_DeviceLayer(stream)
    {
      const std::vector<uint32_t> shape={g_rows,g_cols};
      _param=CAIF_DeviceTensor::Zeros(shape,stream,param_dtype);
      _grad=CAIF_DeviceTensor::Zeros(shape,stream,CAIF_DataType::CAIF_DataType_e::Float32);
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("TestParamLayer::ForwardImpl not used in test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("TestParamLayer::BackwardImpl not used in test");
    }
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Dense_e;
    }
    void ZeroGradients()override{_grad.Fill(0.0f);}
    size_t ParameterTensorCount()const override{return 1;}
    CAIF_DeviceTensor &ParameterTensor(size_t)override{return _param;}
    const CAIF_DeviceTensor &ParameterTensor(size_t)const override{return _param;}
    CAIF_DeviceTensor &GradientTensor(size_t)override{return _grad;}
    const CAIF_DeviceTensor &GradientTensor(size_t)const override{return _grad;}
    size_t TotalParameterCount()const override{return _param.TotalElements();}
    std::string Description()const override{return "TestParamLayer";}
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"weight"};
    }

  private:
    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

void CopyFp32HostToDeviceTensor(const std::vector<float> &host,
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

std::vector<float> ReadDeviceTensorAsFp32(const CAIF_DeviceTensor &src,
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

std::vector<float> MakeSeedPattern(const size_t n,const uint32_t seed)
{
  std::vector<float> out(n);
  for(size_t i=0;i<n;++i)
  {
    const float sgn=(((i+seed)%2)==0)?1.0f:-1.0f;
    out[i]=sgn*0.25f*(1.0f+0.01f*static_cast<float>((i+seed)%13));
  }
  return out;
}

float MaxAbsDiff(const std::vector<float> &a,const std::vector<float> &b)
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

std::unique_ptr<CAIF_DeviceNetwork> BuildNet(CAIF_DataType::CAIF_DataType_e param_dtype,
                                             CAIF_CudaStream &stream)
{
  auto net=std::make_unique<CAIF_DeviceNetwork>(stream);
  net->AddLayer(std::make_unique<TestParamLayer>(stream,param_dtype));
  return net;
}

void FillNetParam(CAIF_DeviceNetwork &net,
                  const std::vector<float> &data,
                  CAIF_CudaStream &stream)
{
  CopyFp32HostToDeviceTensor(data,net.Layer(0).ParameterTensor(0),stream);
}

struct CleanupFile
{
  std::string path;
  ~CleanupFile()
  {
    if(path.empty()==false)
    {
      std::remove(path.c_str());
    }
  }
};

// Round-trip tolerance: fp16 rounding is ~2e-3; bf16 is ~8e-3.
constexpr float g_fp16_tol=3e-3f;
constexpr float g_bf16_tol=1e-2f;

void TestSameDtypeFp32RoundTrip()
{
  try
  {
    const std::string path="test_safetensors_cast_fp32.safetensors";
    CleanupFile cleanup{path};

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_rows)*g_cols;
    const auto pattern=MakeSeedPattern(n,1);

    auto save_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    FillNetParam(*save_net,pattern,stream);
    save_net->SaveSafeTensors(path);

    auto load_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    load_net->LoadSafeTensors(path);

    const auto loaded=
      ReadDeviceTensorAsFp32(load_net->Layer(0).ParameterTensor(0),stream);
    const float diff=MaxAbsDiff(pattern,loaded);
    ReportResult("OnLoadCast::SameDtypeFp32RoundTrip",diff<=1e-6f);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::SameDtypeFp32RoundTrip")
}

void TestSameDtypeBf16RoundTrip()
{
  try
  {
    const std::string path="test_safetensors_cast_bf16.safetensors";
    CleanupFile cleanup{path};

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_rows)*g_cols;
    const auto pattern=MakeSeedPattern(n,2);

    auto save_net=BuildNet(CAIF_DataType::CAIF_DataType_e::BFloat16,stream);
    FillNetParam(*save_net,pattern,stream);
    save_net->SaveSafeTensors(path);

    auto load_net=BuildNet(CAIF_DataType::CAIF_DataType_e::BFloat16,stream);
    load_net->LoadSafeTensors(path);

    // Both sides are bf16 — raw byte copy should reproduce the saved
    // tensor exactly (no second rounding).
    const auto saved_as_fp32=
      ReadDeviceTensorAsFp32(save_net->Layer(0).ParameterTensor(0),stream);
    const auto loaded_as_fp32=
      ReadDeviceTensorAsFp32(load_net->Layer(0).ParameterTensor(0),stream);
    const float diff=MaxAbsDiff(saved_as_fp32,loaded_as_fp32);

    const auto dt=load_net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_ok=(dt==CAIF_DataType::CAIF_DataType_e::BFloat16);

    ReportResult("OnLoadCast::SameDtypeBf16RoundTrip",diff<=1e-6f&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::SameDtypeBf16RoundTrip")
}

void TestCastFp32ToFp16()
{
  try
  {
    const std::string path="test_safetensors_cast_fp32_to_fp16.safetensors";
    CleanupFile cleanup{path};

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_rows)*g_cols;
    const auto pattern=MakeSeedPattern(n,3);

    auto save_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    FillNetParam(*save_net,pattern,stream);
    save_net->SaveSafeTensors(path);

    auto load_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float16,stream);
    load_net->LoadSafeTensors(path);

    const auto loaded=
      ReadDeviceTensorAsFp32(load_net->Layer(0).ParameterTensor(0),stream);
    const auto dt=load_net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_ok=(dt==CAIF_DataType::CAIF_DataType_e::Float16);
    const float diff=MaxAbsDiff(pattern,loaded);
    ReportResult("OnLoadCast::Fp32FileToFp16Param",diff<=g_fp16_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Fp32FileToFp16Param")
}

void TestCastFp32ToBf16()
{
  try
  {
    const std::string path="test_safetensors_cast_fp32_to_bf16.safetensors";
    CleanupFile cleanup{path};

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_rows)*g_cols;
    const auto pattern=MakeSeedPattern(n,4);

    auto save_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    FillNetParam(*save_net,pattern,stream);
    save_net->SaveSafeTensors(path);

    auto load_net=BuildNet(CAIF_DataType::CAIF_DataType_e::BFloat16,stream);
    load_net->LoadSafeTensors(path);

    const auto loaded=
      ReadDeviceTensorAsFp32(load_net->Layer(0).ParameterTensor(0),stream);
    const auto dt=load_net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_ok=(dt==CAIF_DataType::CAIF_DataType_e::BFloat16);
    const float diff=MaxAbsDiff(pattern,loaded);
    ReportResult("OnLoadCast::Fp32FileToBf16Param",diff<=g_bf16_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Fp32FileToBf16Param")
}

void TestCastBf16ToFp32()
{
  try
  {
    const std::string path="test_safetensors_cast_bf16_to_fp32.safetensors";
    CleanupFile cleanup{path};

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_rows)*g_cols;
    const auto pattern=MakeSeedPattern(n,5);

    auto save_net=BuildNet(CAIF_DataType::CAIF_DataType_e::BFloat16,stream);
    FillNetParam(*save_net,pattern,stream);
    save_net->SaveSafeTensors(path);

    // Reference: what the bf16 param actually looked like after rounding,
    // so we can tell apart cast error vs. save-side rounding.
    const auto saved_as_fp32=
      ReadDeviceTensorAsFp32(save_net->Layer(0).ParameterTensor(0),stream);

    auto load_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    load_net->LoadSafeTensors(path);

    const auto loaded=
      ReadDeviceTensorAsFp32(load_net->Layer(0).ParameterTensor(0),stream);
    const auto dt=load_net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_ok=(dt==CAIF_DataType::CAIF_DataType_e::Float32);
    // Cast from bf16→fp32 is exact, so this should match the already-rounded
    // bf16 values to ~0.
    const float diff=MaxAbsDiff(saved_as_fp32,loaded);
    ReportResult("OnLoadCast::Bf16FileToFp32Param",diff<=1e-6f&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Bf16FileToFp32Param")
}

void TestCastFp16ToBf16()
{
  try
  {
    const std::string path="test_safetensors_cast_fp16_to_bf16.safetensors";
    CleanupFile cleanup{path};

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_rows)*g_cols;
    const auto pattern=MakeSeedPattern(n,6);

    auto save_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float16,stream);
    FillNetParam(*save_net,pattern,stream);
    save_net->SaveSafeTensors(path);

    auto load_net=BuildNet(CAIF_DataType::CAIF_DataType_e::BFloat16,stream);
    load_net->LoadSafeTensors(path);

    const auto loaded=
      ReadDeviceTensorAsFp32(load_net->Layer(0).ParameterTensor(0),stream);
    const auto dt=load_net->Layer(0).ParameterTensor(0).Dtype();
    const bool dtype_ok=(dt==CAIF_DataType::CAIF_DataType_e::BFloat16);
    // Two rounding stages (fp16 then bf16) compared against the original
    // fp32 pattern — the looser tolerance (g_bf16_tol) covers both.
    const float diff=MaxAbsDiff(pattern,loaded);
    ReportResult("OnLoadCast::Fp16FileToBf16Param",diff<=g_bf16_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Fp16FileToBf16Param")
}

#endif// USE_CAIF_CUDA

}// anon

int main()
{
  ISE_Out::Out()<<"=== Safetensors On-Load Cast Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestSameDtypeFp32RoundTrip();
  TestSameDtypeBf16RoundTrip();
  TestCastFp32ToFp16();
  TestCastFp32ToBf16();
  TestCastBf16ToFp32();
  TestCastFp16ToBf16();
#else
  ISE_Out::Out()<<"  (CUDA not enabled; tests skipped)\n";
#endif

  return CAIF_TestHarness::FinalExitCode();
}
