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

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_stcast_test_rows=8;
constexpr uint32_t g_caif_stcast_test_cols=16;
constexpr float g_caif_stcast_test_fp16_tol=3e-3f;
constexpr float g_caif_stcast_test_bf16_tol=1e-2f;
constexpr float g_caif_stcast_test_exact_tol=1e-6f;
constexpr uint32_t g_caif_stcast_test_seed_fp32=1;
constexpr uint32_t g_caif_stcast_test_seed_bf16=2;
constexpr uint32_t g_caif_stcast_test_seed_fp32_to_fp16=3;
constexpr uint32_t g_caif_stcast_test_seed_fp32_to_bf16=4;
constexpr uint32_t g_caif_stcast_test_seed_bf16_to_fp32=5;
constexpr uint32_t g_caif_stcast_test_seed_fp16_to_bf16=6;
constexpr uint32_t g_caif_stcast_test_seed_mod_sign=2;
constexpr uint32_t g_caif_stcast_test_seed_mod_scale=13;
constexpr float g_caif_stcast_test_pattern_base=0.25f;
constexpr float g_caif_stcast_test_pattern_step=0.01f;

//------------------------------------------------------------------------------
// RAII file-cleanup guard.
//------------------------------------------------------------------------------
class CAIF_SafetensorsCastTestCleanupFile
{
  public:
    explicit CAIF_SafetensorsCastTestCleanupFile(const std::string &path):_path(path){}
    ~CAIF_SafetensorsCastTestCleanupFile()
    {
      if(Path().empty()==false)
      {
        std::remove(Path().c_str());
      }
    }

    const std::string &Path()const{return _path;}

  protected:

  private:
    std::string _path;
};

//------------------------------------------------------------------------------
// Test-only layer: a single parameter tensor of user-chosen dtype, plus a
// fp32 gradient tensor.  Forward/Backward are unused (we drive the network's
// load plumbing directly), so they throw if called.
//------------------------------------------------------------------------------
class CAIF_SafetensorsCastTestParamLayer:public CAIF_DeviceLayer
{
  public:
    CAIF_SafetensorsCastTestParamLayer(CAIF_CudaStream &stream,
                                        CAIF_DataType::CAIF_DataType_e param_dtype)
      :CAIF_DeviceLayer(stream),
       _param(MakeZeros(stream,param_dtype)),
       _grad(MakeZerosFp32(stream))
    {
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_SafetensorsCastTestParamLayer::ForwardImpl not used in test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_SafetensorsCastTestParamLayer::BackwardImpl not used in test");
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
      return "CAIF_SafetensorsCastTestParamLayer";
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
      const std::vector<uint32_t> shape={g_caif_stcast_test_rows,g_caif_stcast_test_cols};
      return CAIF_DeviceTensor::Zeros(shape,stream,dtype);
    }
    static CAIF_DeviceTensor MakeZerosFp32(CAIF_CudaStream &stream)
    {
      const std::vector<uint32_t> shape={g_caif_stcast_test_rows,g_caif_stcast_test_cols};
      return CAIF_DeviceTensor::Zeros(shape,stream,CAIF_DataType::CAIF_DataType_e::Float32);
    }

    CAIF_DeviceTensor _param;
    CAIF_DeviceTensor _grad;
};

//------------------------------------------------------------------------------
// On-load dtype cast correctness tests.
//------------------------------------------------------------------------------
class CAIF_SafetensorsCastTests
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
    static std::vector<float> MakeSeedPattern(size_t n,uint32_t seed);
    static float MaxAbsDiff(const std::vector<float> &a,const std::vector<float> &b);
    static std::unique_ptr<CAIF_DeviceNetwork> BuildNet(CAIF_DataType::CAIF_DataType_e param_dtype,
                                                         CAIF_CudaStream &stream);
    static void FillNetParam(CAIF_DeviceNetwork &net,
                              const std::vector<float> &data,
                              CAIF_CudaStream &stream);

    static void TestSameDtypeFp32RoundTrip();
    static void TestSameDtypeBf16RoundTrip();
    static void TestCastFp32ToFp16();
    static void TestCastFp32ToBf16();
    static void TestCastBf16ToFp32();
    static void TestCastFp16ToBf16();
};

void CAIF_SafetensorsCastTests::CopyFp32HostToDeviceTensor(const std::vector<float> &host,
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

std::vector<float> CAIF_SafetensorsCastTests::ReadDeviceTensorAsFp32(
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

std::vector<float> CAIF_SafetensorsCastTests::MakeSeedPattern(const size_t n,
                                                                const uint32_t seed)
{
  std::vector<float> out(n);
  for(size_t i=0;i<n;++i)
  {
    float sgn=0.0f;
    if(((i+seed)%g_caif_stcast_test_seed_mod_sign)==0)
    {
      sgn=1.0f;
    }
    else
    {
      sgn=-1.0f;
    }
    out[i]=sgn*g_caif_stcast_test_pattern_base*(1.0f+g_caif_stcast_test_pattern_step*
            static_cast<float>((i+seed)%g_caif_stcast_test_seed_mod_scale));
  }
  return out;
}

float CAIF_SafetensorsCastTests::MaxAbsDiff(const std::vector<float> &a,
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

std::unique_ptr<CAIF_DeviceNetwork> CAIF_SafetensorsCastTests::BuildNet(
  CAIF_DataType::CAIF_DataType_e param_dtype,
  CAIF_CudaStream &stream)
{
  auto net=std::make_unique<CAIF_DeviceNetwork>(stream);
  net->AddLayer(std::make_unique<CAIF_SafetensorsCastTestParamLayer>(stream,param_dtype));
  return net;
}

void CAIF_SafetensorsCastTests::FillNetParam(CAIF_DeviceNetwork &net,
                                               const std::vector<float> &data,
                                               CAIF_CudaStream &stream)
{
  CopyFp32HostToDeviceTensor(data,net.Layer(0).ParameterTensor(0),stream);
}

void CAIF_SafetensorsCastTests::TestSameDtypeFp32RoundTrip()
{
  try
  {
    const std::string path="test_safetensors_cast_fp32.safetensors";
    CAIF_SafetensorsCastTestCleanupFile cleanup(path);

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_caif_stcast_test_rows)*g_caif_stcast_test_cols;
    const auto pattern=MakeSeedPattern(n,g_caif_stcast_test_seed_fp32);

    auto save_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    FillNetParam(*save_net,pattern,stream);
    save_net->SaveSafeTensors(path);

    auto load_net=BuildNet(CAIF_DataType::CAIF_DataType_e::Float32,stream);
    load_net->LoadSafeTensors(path);

    const auto loaded=
      ReadDeviceTensorAsFp32(load_net->Layer(0).ParameterTensor(0),stream);
    const float diff=MaxAbsDiff(pattern,loaded);
    CAIF_TestHarness::Report("OnLoadCast::SameDtypeFp32RoundTrip",
                              diff<=g_caif_stcast_test_exact_tol);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::SameDtypeFp32RoundTrip")
}

void CAIF_SafetensorsCastTests::TestSameDtypeBf16RoundTrip()
{
  try
  {
    const std::string path="test_safetensors_cast_bf16.safetensors";
    CAIF_SafetensorsCastTestCleanupFile cleanup(path);

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_caif_stcast_test_rows)*g_caif_stcast_test_cols;
    const auto pattern=MakeSeedPattern(n,g_caif_stcast_test_seed_bf16);

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

    CAIF_TestHarness::Report("OnLoadCast::SameDtypeBf16RoundTrip",
                              diff<=g_caif_stcast_test_exact_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::SameDtypeBf16RoundTrip")
}

void CAIF_SafetensorsCastTests::TestCastFp32ToFp16()
{
  try
  {
    const std::string path="test_safetensors_cast_fp32_to_fp16.safetensors";
    CAIF_SafetensorsCastTestCleanupFile cleanup(path);

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_caif_stcast_test_rows)*g_caif_stcast_test_cols;
    const auto pattern=MakeSeedPattern(n,g_caif_stcast_test_seed_fp32_to_fp16);

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
    CAIF_TestHarness::Report("OnLoadCast::Fp32FileToFp16Param",
                              diff<=g_caif_stcast_test_fp16_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Fp32FileToFp16Param")
}

void CAIF_SafetensorsCastTests::TestCastFp32ToBf16()
{
  try
  {
    const std::string path="test_safetensors_cast_fp32_to_bf16.safetensors";
    CAIF_SafetensorsCastTestCleanupFile cleanup(path);

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_caif_stcast_test_rows)*g_caif_stcast_test_cols;
    const auto pattern=MakeSeedPattern(n,g_caif_stcast_test_seed_fp32_to_bf16);

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
    CAIF_TestHarness::Report("OnLoadCast::Fp32FileToBf16Param",
                              diff<=g_caif_stcast_test_bf16_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Fp32FileToBf16Param")
}

void CAIF_SafetensorsCastTests::TestCastBf16ToFp32()
{
  try
  {
    const std::string path="test_safetensors_cast_bf16_to_fp32.safetensors";
    CAIF_SafetensorsCastTestCleanupFile cleanup(path);

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_caif_stcast_test_rows)*g_caif_stcast_test_cols;
    const auto pattern=MakeSeedPattern(n,g_caif_stcast_test_seed_bf16_to_fp32);

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
    CAIF_TestHarness::Report("OnLoadCast::Bf16FileToFp32Param",
                              diff<=g_caif_stcast_test_exact_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Bf16FileToFp32Param")
}

void CAIF_SafetensorsCastTests::TestCastFp16ToBf16()
{
  try
  {
    const std::string path="test_safetensors_cast_fp16_to_bf16.safetensors";
    CAIF_SafetensorsCastTestCleanupFile cleanup(path);

    CAIF_CudaStream stream;
    const size_t n=static_cast<size_t>(g_caif_stcast_test_rows)*g_caif_stcast_test_cols;
    const auto pattern=MakeSeedPattern(n,g_caif_stcast_test_seed_fp16_to_bf16);

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
    // fp32 pattern — the looser tolerance (g_caif_stcast_test_bf16_tol) covers both.
    const float diff=MaxAbsDiff(pattern,loaded);
    CAIF_TestHarness::Report("OnLoadCast::Fp16FileToBf16Param",
                              diff<=g_caif_stcast_test_bf16_tol&&dtype_ok);
  }
  CAIF_TEST_CATCH_BLOCK("OnLoadCast::Fp16FileToBf16Param")
}

void CAIF_SafetensorsCastTests::RunAll()
{
  ISE_Out::Out()<<"=== Safetensors On-Load Cast Tests ==="
                <<"\n\n";
  TestSameDtypeFp32RoundTrip();
  TestSameDtypeBf16RoundTrip();
  TestCastFp32ToFp16();
  TestCastFp32ToBf16();
  TestCastBf16ToFp32();
  TestCastFp16ToBf16();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_SafetensorsCastTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"  (CUDA not enabled; tests skipped)"
                <<"\n";
  return 0;
#endif
}
