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
// Test: CAIF_Ops::Cast pairwise dtype conversion parity (fp32/fp16/bf16).
//
// Verifies round-trip (src -> dst -> src) reproduces the original tensor
// within the expected rounding tolerance of the intermediate dtype, and that
// identity casts (src == dst) are a pure copy.
//------------------------------------------------------------------------------
#include "caif_device_tensor.h"
#include "caif_ops.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <random>
#include <vector>

using namespace instance;

namespace
{

void ReportResult(const char *name,bool ok)
{
  CAIF_TestHarness::Report(name,ok);
}

#ifdef USE_CAIF_CUDA

constexpr float g_fp16_round_tol=2e-3f;
constexpr float g_bf16_round_tol=8e-3f;
constexpr float g_identity_tol=0.0f;

std::vector<float> MakeRandom(const size_t n,const uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    v[i]=dist(gen);
  }
  return v;
}

bool MaxAbsClose(const std::vector<float> &ref,
                 const std::vector<float> &got,
                 const float tol)
{
  if(ref.size()!=got.size())
  {
    return false;
  }
  for(size_t i=0;i<ref.size();++i)
  {
    const float diff=std::fabs(ref[i]-got[i]);
    if(diff>tol)
    {
      ISE_Out::Out()<<"  mismatch i="
                    <<i
                    <<" ref="
                    <<ref[i]
                    <<" got="
                    <<got[i]
                    <<" diff="
                    <<diff
                    <<"\n";
      return false;
    }
  }
  return true;
}

std::vector<float> CastRoundTripViaOp(const std::vector<float> &src_host,
                                      const CAIF_DataType::CAIF_DataType_e intermediate,
                                      CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  const std::vector<uint32_t> shape={static_cast<uint32_t>(src_host.size())};
  CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
  CAIF_DeviceTensor mid=CAIF_DeviceTensor::Uninitialized(shape,stream,intermediate);
  CAIF_DeviceTensor back=CAIF_DeviceTensor::Uninitialized(shape,
                                                          stream,
                                                          CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_Ops::Cast(src,mid,ctx);
  CAIF_Ops::Cast(mid,back,ctx);
  std::vector<float> out(src_host.size());
  back.CopyToHost(out.data());
  return out;
}

void TestCastFP32Identity()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<float> src=MakeRandom(256,1);
    std::vector<float> got=CastRoundTripViaOp(src,
                                              CAIF_DataType::CAIF_DataType_e::Float32,
                                              stream);
    ReportResult("Cast::FP32Identity",MaxAbsClose(src,got,g_identity_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP32Identity")
}

void TestCastFP32ToFP16RoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<float> src=MakeRandom(512,2);
    std::vector<float> got=CastRoundTripViaOp(src,
                                              CAIF_DataType::CAIF_DataType_e::Float16,
                                              stream);
    ReportResult("Cast::FP32ToFP16RoundTrip",MaxAbsClose(src,got,g_fp16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP32ToFP16RoundTrip")
}

void TestCastFP32ToBF16RoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<float> src=MakeRandom(512,3);
    std::vector<float> got=CastRoundTripViaOp(src,
                                              CAIF_DataType::CAIF_DataType_e::BFloat16,
                                              stream);
    ReportResult("Cast::FP32ToBF16RoundTrip",MaxAbsClose(src,got,g_bf16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP32ToBF16RoundTrip")
}

void TestCastFP16ToBF16Chain()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host=MakeRandom(512,4);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(src_host.size())};
    CAIF_DeviceTensor src_fp32=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor as_fp16=CAIF_DeviceTensor::Uninitialized(shape,
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float16);
    CAIF_DeviceTensor as_bf16=CAIF_DeviceTensor::Uninitialized(shape,
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::BFloat16);
    CAIF_DeviceTensor back_fp32=CAIF_DeviceTensor::Uninitialized(shape,
                                                                 stream,
                                                                 CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::Cast(src_fp32,as_fp16,ctx);
    CAIF_Ops::Cast(as_fp16,as_bf16,ctx);
    CAIF_Ops::Cast(as_bf16,back_fp32,ctx);
    std::vector<float> got(src_host.size());
    back_fp32.CopyToHost(got.data());
    ReportResult("Cast::FP16ToBF16Chain",MaxAbsClose(src_host,got,g_bf16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP16ToBF16Chain")
}

void TestCastBF16ToFP16Chain()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host=MakeRandom(512,5);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(src_host.size())};
    CAIF_DeviceTensor src_fp32=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor as_bf16=CAIF_DeviceTensor::Uninitialized(shape,
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::BFloat16);
    CAIF_DeviceTensor as_fp16=CAIF_DeviceTensor::Uninitialized(shape,
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float16);
    CAIF_DeviceTensor back_fp32=CAIF_DeviceTensor::Uninitialized(shape,
                                                                 stream,
                                                                 CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::Cast(src_fp32,as_bf16,ctx);
    CAIF_Ops::Cast(as_bf16,as_fp16,ctx);
    CAIF_Ops::Cast(as_fp16,back_fp32,ctx);
    std::vector<float> got(src_host.size());
    back_fp32.CopyToHost(got.data());
    ReportResult("Cast::BF16ToFP16Chain",MaxAbsClose(src_host,got,g_bf16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::BF16ToFP16Chain")
}

void TestCastShapeMismatchThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host(16,1.0f);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),{16},stream);
    CAIF_DeviceTensor dst=CAIF_DeviceTensor::Uninitialized({8},
                                                           stream,
                                                           CAIF_DataType::CAIF_DataType_e::Float16);
    bool threw=false;
    try
    {
      CAIF_Ops::Cast(src,dst,ctx);
    }
    catch(...)
    {
      threw=true;
    }
    ReportResult("Cast::ShapeMismatchThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Cast::ShapeMismatchThrows")
}

void TestCastIntegerRejected()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host(16,0.5f);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),{16},stream);
    CAIF_DeviceTensor dst=CAIF_DeviceTensor::Uninitialized({16},
                                                           stream,
                                                           CAIF_DataType::CAIF_DataType_e::Int8);
    bool threw=false;
    try
    {
      CAIF_Ops::Cast(src,dst,ctx);
    }
    catch(...)
    {
      threw=true;
    }
    ReportResult("Cast::IntegerRejected",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Cast::IntegerRejected")
}

#endif// USE_CAIF_CUDA

}// anon

int main()
{
  ISE_Out::Out()<<"=== Cast Op Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestCastFP32Identity();
  TestCastFP32ToFP16RoundTrip();
  TestCastFP32ToBF16RoundTrip();
  TestCastFP16ToBF16Chain();
  TestCastBF16ToFP16Chain();
  TestCastShapeMismatchThrows();
  TestCastIntegerRejected();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
  return CAIF_TestHarness::FinalExitCode();
}
