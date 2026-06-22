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

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr float g_caif_cast_test_fp16_round_tol=2e-3f;
constexpr float g_caif_cast_test_bf16_round_tol=8e-3f;
constexpr float g_caif_cast_test_identity_tol=0.0f;
constexpr size_t g_caif_cast_test_n_small=256;
constexpr size_t g_caif_cast_test_n_large=512;
constexpr size_t g_caif_cast_test_n_mismatch_src=16;
constexpr size_t g_caif_cast_test_n_mismatch_dst=8;
constexpr uint32_t g_caif_cast_test_seed_identity=1;
constexpr uint32_t g_caif_cast_test_seed_fp16=2;
constexpr uint32_t g_caif_cast_test_seed_bf16=3;
constexpr uint32_t g_caif_cast_test_seed_fp16_to_bf16=4;
constexpr uint32_t g_caif_cast_test_seed_bf16_to_fp16=5;
constexpr float g_caif_cast_test_int8_fill=0.5f;

//------------------------------------------------------------------------------
// Cast pairwise dtype conversion tests.
//------------------------------------------------------------------------------
class CAIF_CastOpsTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeRandom(size_t n,uint32_t seed);
    static bool MaxAbsClose(const std::vector<float> &ref,
                             const std::vector<float> &got,
                             float tol);
    static std::vector<float> CastRoundTripViaOp(const std::vector<float> &src_host,
                                                  CAIF_DataType::CAIF_DataType_e intermediate,
                                                  CAIF_CudaStream &stream);

    static void TestCastFP32Identity();
    static void TestCastFP32ToFP16RoundTrip();
    static void TestCastFP32ToBF16RoundTrip();
    static void TestCastFP16ToBF16Chain();
    static void TestCastBF16ToFP16Chain();
    static void TestCastShapeMismatchThrows();
    static void TestCastIntegerRejected();
};

std::vector<float> CAIF_CastOpsTests::MakeRandom(const size_t n,const uint32_t seed)
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

bool CAIF_CastOpsTests::MaxAbsClose(const std::vector<float> &ref,
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

std::vector<float> CAIF_CastOpsTests::CastRoundTripViaOp(
  const std::vector<float> &src_host,
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

void CAIF_CastOpsTests::TestCastFP32Identity()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<float> src=MakeRandom(g_caif_cast_test_n_small,g_caif_cast_test_seed_identity);
    std::vector<float> got=CastRoundTripViaOp(src,
                                               CAIF_DataType::CAIF_DataType_e::Float32,
                                               stream);
    CAIF_TestHarness::Report("Cast::FP32Identity",MaxAbsClose(src,got,g_caif_cast_test_identity_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP32Identity")
}

void CAIF_CastOpsTests::TestCastFP32ToFP16RoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<float> src=MakeRandom(g_caif_cast_test_n_large,g_caif_cast_test_seed_fp16);
    std::vector<float> got=CastRoundTripViaOp(src,
                                               CAIF_DataType::CAIF_DataType_e::Float16,
                                               stream);
    CAIF_TestHarness::Report("Cast::FP32ToFP16RoundTrip",
                              MaxAbsClose(src,got,g_caif_cast_test_fp16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP32ToFP16RoundTrip")
}

void CAIF_CastOpsTests::TestCastFP32ToBF16RoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<float> src=MakeRandom(g_caif_cast_test_n_large,g_caif_cast_test_seed_bf16);
    std::vector<float> got=CastRoundTripViaOp(src,
                                               CAIF_DataType::CAIF_DataType_e::BFloat16,
                                               stream);
    CAIF_TestHarness::Report("Cast::FP32ToBF16RoundTrip",
                              MaxAbsClose(src,got,g_caif_cast_test_bf16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP32ToBF16RoundTrip")
}

void CAIF_CastOpsTests::TestCastFP16ToBF16Chain()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host=MakeRandom(g_caif_cast_test_n_large,
                                            g_caif_cast_test_seed_fp16_to_bf16);
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
    CAIF_TestHarness::Report("Cast::FP16ToBF16Chain",
                              MaxAbsClose(src_host,got,g_caif_cast_test_bf16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::FP16ToBF16Chain")
}

void CAIF_CastOpsTests::TestCastBF16ToFP16Chain()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host=MakeRandom(g_caif_cast_test_n_large,
                                            g_caif_cast_test_seed_bf16_to_fp16);
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
    CAIF_TestHarness::Report("Cast::BF16ToFP16Chain",
                              MaxAbsClose(src_host,got,g_caif_cast_test_bf16_round_tol));
  }
  CAIF_TEST_CATCH_BLOCK("Cast::BF16ToFP16Chain")
}

void CAIF_CastOpsTests::TestCastShapeMismatchThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host(g_caif_cast_test_n_mismatch_src,1.0f);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(
      src_host.data(),
      {static_cast<uint32_t>(g_caif_cast_test_n_mismatch_src)},
      stream);
    CAIF_DeviceTensor dst=CAIF_DeviceTensor::Uninitialized(
      {static_cast<uint32_t>(g_caif_cast_test_n_mismatch_dst)},
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
    CAIF_TestHarness::Report("Cast::ShapeMismatchThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Cast::ShapeMismatchThrows")
}

void CAIF_CastOpsTests::TestCastIntegerRejected()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    std::vector<float> src_host(g_caif_cast_test_n_mismatch_src,g_caif_cast_test_int8_fill);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(
      src_host.data(),
      {static_cast<uint32_t>(g_caif_cast_test_n_mismatch_src)},
      stream);
    CAIF_DeviceTensor dst=CAIF_DeviceTensor::Uninitialized(
      {static_cast<uint32_t>(g_caif_cast_test_n_mismatch_src)},
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
    CAIF_TestHarness::Report("Cast::IntegerRejected",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Cast::IntegerRejected")
}

void CAIF_CastOpsTests::RunAll()
{
  ISE_Out::Out()<<"=== Cast Op Tests ==="
                <<"\n\n";
  TestCastFP32Identity();
  TestCastFP32ToFP16RoundTrip();
  TestCastFP32ToBF16RoundTrip();
  TestCastFP16ToBF16Chain();
  TestCastBF16ToFP16Chain();
  TestCastShapeMismatchThrows();
  TestCastIntegerRejected();
  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_CastOpsTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
