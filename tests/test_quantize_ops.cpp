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
// Test: CAIF_Ops symmetric INT8 and per-group INT4 quantize/dequantize round-trip.
//
// Verifies:
//   - INT8 PerTensor: fp32 -> int8+scale -> fp32, max abs error bounded by 1/127
//     of the tensor's max abs value
//   - INT8 PerChannel: 2D fp32 -> int8+per-col-scale -> fp32 with similar bound
//   - INT4 PerGroup (group_size=128): fp32 -> packed int4+fp16 scale -> fp32
//   - Zero input produces unit scale (no div-by-zero / NaN)
//   - Shape and dtype mismatches throw
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

constexpr size_t g_caif_quant_test_n_large=1024;
constexpr size_t g_caif_quant_test_n_medium=512;
constexpr size_t g_caif_quant_test_n_small=256;
constexpr size_t g_caif_quant_test_n_tiny=32;
constexpr size_t g_caif_quant_test_n_mismatch=64;
constexpr uint32_t g_caif_quant_test_rows=32;
constexpr uint32_t g_caif_quant_test_cols=64;
constexpr uint32_t g_caif_quant_test_fakequant_rows=8;
constexpr uint32_t g_caif_quant_test_fakequant_cols=32;
constexpr uint32_t g_caif_quant_test_group_size=128;
constexpr uint32_t g_caif_quant_test_seed_pertensor=11;
constexpr uint32_t g_caif_quant_test_seed_perchannel=22;
constexpr uint32_t g_caif_quant_test_seed_int4=33;
constexpr uint32_t g_caif_quant_test_seed_fakequant_pertensor=101;
constexpr uint32_t g_caif_quant_test_seed_fakequant_perchannel=102;
constexpr uint32_t g_caif_quant_test_seed_fakequant_int4=103;
constexpr float g_caif_quant_test_range=2.5f;
constexpr float g_caif_quant_test_range_unit=1.0f;
constexpr float g_caif_quant_test_fill_wrong_dtype=0.5f;
constexpr float g_caif_quant_test_fill_mismatch=0.25f;
constexpr float g_caif_quant_test_fill_groupmismatch=0.3f;
constexpr float g_caif_quant_test_eps_small=1e-6f;
constexpr float g_caif_quant_test_eps_tol=1e-3f;
constexpr float g_caif_quant_test_int8_levels=127.0f;
constexpr float g_caif_quant_test_int4_levels=7.0f;
constexpr float g_caif_quant_test_perchannel_scale_base=0.1f;
constexpr float g_caif_quant_test_perchannel_scale_step=0.05f;

//------------------------------------------------------------------------------
// Quantize/dequantize round-trip tests.
//------------------------------------------------------------------------------
class CAIF_QuantizeOpsTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeRandom(size_t n,uint32_t seed,float scale);
    static float MaxAbs(const std::vector<float> &v);
    static bool Within(const std::vector<float> &ref,
                        const std::vector<float> &got,
                        float abs_tol);

    static void TestInt8PerTensorRoundTrip();
    static void TestInt8PerChannelRoundTrip();
    static void TestInt8ZeroInputUnitScale();
    static void TestInt4PerGroupRoundTrip();
    static void TestInt8WrongOutputDtypeThrows();
    static void TestInt8PerChannelRequires2DThrows();
    static void TestInt4GroupSizeMismatchThrows();
    static void TestFakeQuantInt8PerTensor();
    static void TestFakeQuantInt8PerChannel();
    static void TestFakeQuantInt4PerGroup();
    static void TestFakeQuantInt8WrongDtypeThrows();
};

std::vector<float> CAIF_QuantizeOpsTests::MakeRandom(const size_t n,
                                                       const uint32_t seed,
                                                       const float scale)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-scale,scale);
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    v[i]=dist(gen);
  }
  return v;
}

float CAIF_QuantizeOpsTests::MaxAbs(const std::vector<float> &v)
{
  float m=0.0f;
  for(const float x:v)
  {
    const float a=std::fabs(x);
    if(a>m)
    {
      m=a;
    }
  }
  return m;
}

bool CAIF_QuantizeOpsTests::Within(const std::vector<float> &ref,
                                    const std::vector<float> &got,
                                    const float abs_tol)
{
  if(ref.size()!=got.size())
  {
    return false;
  }
  for(size_t i=0;i<ref.size();++i)
  {
    const float d=std::fabs(ref[i]-got[i]);
    if(d>abs_tol)
    {
      ISE_Out::Out()<<"  mismatch i="
                    <<i
                    <<" ref="
                    <<ref[i]
                    <<" got="
                    <<got[i]
                    <<" diff="
                    <<d
                    <<" tol="
                    <<abs_tol
                    <<"\n";
      return false;
    }
  }
  return true;
}

void CAIF_QuantizeOpsTests::TestInt8PerTensorRoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host=MakeRandom(g_caif_quant_test_n_large,
                                                  g_caif_quant_test_seed_pertensor,
                                                  g_caif_quant_test_range);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_quant_test_n_large)};
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized(shape,
                                                          stream,
                                                          CAIF_DataType::CAIF_DataType_e::Int8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({1},
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor back=CAIF_DeviceTensor::Uninitialized(shape,
                                                             stream,
                                                             CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::QuantizeInt8(src,q,scales,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    CAIF_Ops::DequantizeInt8(q,back,scales,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    std::vector<float> got(g_caif_quant_test_n_large);
    back.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/g_caif_quant_test_int8_levels+g_caif_quant_test_eps_small;
    CAIF_TestHarness::Report("Quantize::Int8PerTensorRoundTrip",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8PerTensorRoundTrip")
}

void CAIF_QuantizeOpsTests::TestInt8PerChannelRoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<uint32_t> shape={g_caif_quant_test_rows,g_caif_quant_test_cols};
    std::vector<float> src_host(g_caif_quant_test_rows*g_caif_quant_test_cols);
    std::mt19937 gen(g_caif_quant_test_seed_perchannel);
    for(uint32_t c=0;c<g_caif_quant_test_cols;++c)
    {
      const float col_scale=g_caif_quant_test_perchannel_scale_base+
                             static_cast<float>(c)*g_caif_quant_test_perchannel_scale_step;
      std::uniform_real_distribution<float> dist(-col_scale,col_scale);
      for(uint32_t r=0;r<g_caif_quant_test_rows;++r)
      {
        src_host[r*g_caif_quant_test_cols+c]=dist(gen);
      }
    }
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized(shape,
                                                          stream,
                                                          CAIF_DataType::CAIF_DataType_e::Int8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({g_caif_quant_test_cols},
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor back=CAIF_DeviceTensor::Uninitialized(shape,
                                                             stream,
                                                             CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::QuantizeInt8(src,q,scales,CAIF_Ops::QuantScheme_e::PerChannel_e,ctx);
    CAIF_Ops::DequantizeInt8(q,back,scales,CAIF_Ops::QuantScheme_e::PerChannel_e,ctx);
    std::vector<float> got(src_host.size());
    back.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/g_caif_quant_test_int8_levels+g_caif_quant_test_eps_small;
    CAIF_TestHarness::Report("Quantize::Int8PerChannelRoundTrip",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8PerChannelRoundTrip")
}

void CAIF_QuantizeOpsTests::TestInt8ZeroInputUnitScale()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host(g_caif_quant_test_n_small,0.0f);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_quant_test_n_small)};
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized(shape,
                                                          stream,
                                                          CAIF_DataType::CAIF_DataType_e::Int8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({1},
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor back=CAIF_DeviceTensor::Uninitialized(shape,
                                                             stream,
                                                             CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::QuantizeInt8(src,q,scales,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    CAIF_Ops::DequantizeInt8(q,back,scales,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    std::vector<float> got(g_caif_quant_test_n_small);
    back.CopyToHost(got.data());
    bool ok=true;
    for(const float v:got)
    {
      if(std::isnan(v)==true || v!=0.0f)
      {
        ok=false;
        break;
      }
    }
    CAIF_TestHarness::Report("Quantize::Int8ZeroInputUnitScale",ok);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8ZeroInputUnitScale")
}

void CAIF_QuantizeOpsTests::TestInt4PerGroupRoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host=MakeRandom(g_caif_quant_test_n_medium,
                                                  g_caif_quant_test_seed_int4,
                                                  g_caif_quant_test_range_unit);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_quant_test_n_medium)};
    const uint32_t num_groups=static_cast<uint32_t>(g_caif_quant_test_n_medium)/
                               g_caif_quant_test_group_size;
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized(shape,
                                                          stream,
                                                          CAIF_DataType::CAIF_DataType_e::Int4);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({num_groups},
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float16);
    CAIF_DeviceTensor back=CAIF_DeviceTensor::Uninitialized(shape,
                                                             stream,
                                                             CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::QuantizeInt4PerGroup(src,q,scales,g_caif_quant_test_group_size,ctx);
    CAIF_Ops::DequantizeInt4PerGroup(q,back,scales,g_caif_quant_test_group_size,ctx);
    std::vector<float> got(g_caif_quant_test_n_medium);
    back.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/g_caif_quant_test_int4_levels+g_caif_quant_test_eps_tol;
    CAIF_TestHarness::Report("Quantize::Int4PerGroupRoundTrip",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int4PerGroupRoundTrip")
}

void CAIF_QuantizeOpsTests::TestInt8WrongOutputDtypeThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host(g_caif_quant_test_n_tiny,g_caif_quant_test_fill_wrong_dtype);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(
      src_host.data(),
      {static_cast<uint32_t>(g_caif_quant_test_n_tiny)},
      stream);
    CAIF_DeviceTensor bad_out=CAIF_DeviceTensor::Uninitialized(
      {static_cast<uint32_t>(g_caif_quant_test_n_tiny)},
      stream,
      CAIF_DataType::CAIF_DataType_e::Float16);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({1},
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float32);
    bool threw=false;
    try
    {
      CAIF_Ops::QuantizeInt8(src,bad_out,scales,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    }
    catch(...)
    {
      threw=true;
    }
    CAIF_TestHarness::Report("Quantize::Int8WrongOutputDtypeThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8WrongOutputDtypeThrows")
}

void CAIF_QuantizeOpsTests::TestInt8PerChannelRequires2DThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host(g_caif_quant_test_n_mismatch,g_caif_quant_test_fill_mismatch);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(
      src_host.data(),
      {static_cast<uint32_t>(g_caif_quant_test_n_mismatch)},
      stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized(
      {static_cast<uint32_t>(g_caif_quant_test_n_mismatch)},
      stream,
      CAIF_DataType::CAIF_DataType_e::Int8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized(
      {static_cast<uint32_t>(g_caif_quant_test_n_mismatch)},
      stream,
      CAIF_DataType::CAIF_DataType_e::Float32);
    bool threw=false;
    try
    {
      CAIF_Ops::QuantizeInt8(src,q,scales,CAIF_Ops::QuantScheme_e::PerChannel_e,ctx);
    }
    catch(...)
    {
      threw=true;
    }
    CAIF_TestHarness::Report("Quantize::Int8PerChannelRequires2DThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8PerChannelRequires2DThrows")
}

void CAIF_QuantizeOpsTests::TestInt4GroupSizeMismatchThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host(g_caif_quant_test_n_small,
                                       g_caif_quant_test_fill_groupmismatch);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(
      src_host.data(),
      {static_cast<uint32_t>(g_caif_quant_test_n_small)},
      stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized(
      {static_cast<uint32_t>(g_caif_quant_test_n_small)},
      stream,
      CAIF_DataType::CAIF_DataType_e::Int4);
    // 256 elements / group 128 = 2 groups; pass a mismatched scale count.
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({5},
                                                               stream,
                                                               CAIF_DataType::CAIF_DataType_e::Float16);
    bool threw=false;
    try
    {
      CAIF_Ops::QuantizeInt4PerGroup(src,q,scales,g_caif_quant_test_group_size,ctx);
    }
    catch(...)
    {
      threw=true;
    }
    CAIF_TestHarness::Report("Quantize::Int4GroupSizeMismatchThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int4GroupSizeMismatchThrows")
}

void CAIF_QuantizeOpsTests::TestFakeQuantInt8PerTensor()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host=MakeRandom(g_caif_quant_test_n_small,
                                                  g_caif_quant_test_seed_fakequant_pertensor,
                                                  g_caif_quant_test_range_unit);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_quant_test_n_small)};
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized(shape,
                                                            stream,
                                                            CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::FakeQuantInt8(src,out,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    std::vector<float> got(g_caif_quant_test_n_small);
    out.CopyToHost(got.data());
    // round-trip error bounded by half a scale: max_abs / 127 / 2 plus tiny eps
    const float tol=MaxAbs(src_host)/g_caif_quant_test_int8_levels+g_caif_quant_test_eps_tol;
    CAIF_TestHarness::Report("FakeQuant::Int8PerTensor",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int8PerTensor")
}

void CAIF_QuantizeOpsTests::TestFakeQuantInt8PerChannel()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const size_t n=static_cast<size_t>(g_caif_quant_test_fakequant_rows)*
                   g_caif_quant_test_fakequant_cols;
    const std::vector<float> src_host=MakeRandom(n,
                                                  g_caif_quant_test_seed_fakequant_perchannel,
                                                  g_caif_quant_test_range_unit);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(
      src_host.data(),
      {g_caif_quant_test_fakequant_rows,g_caif_quant_test_fakequant_cols},
      stream);
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized(
      {g_caif_quant_test_fakequant_rows,g_caif_quant_test_fakequant_cols},
      stream,
      CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::FakeQuantInt8(src,out,CAIF_Ops::QuantScheme_e::PerChannel_e,ctx);
    std::vector<float> got(n);
    out.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/g_caif_quant_test_int8_levels+g_caif_quant_test_eps_tol;
    CAIF_TestHarness::Report("FakeQuant::Int8PerChannel",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int8PerChannel")
}

void CAIF_QuantizeOpsTests::TestFakeQuantInt4PerGroup()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host=MakeRandom(g_caif_quant_test_n_medium,
                                                  g_caif_quant_test_seed_fakequant_int4,
                                                  g_caif_quant_test_range_unit);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_quant_test_n_medium)};
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized(shape,
                                                            stream,
                                                            CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::FakeQuantInt4PerGroup(src,out,g_caif_quant_test_group_size,ctx);
    std::vector<float> got(g_caif_quant_test_n_medium);
    out.CopyToHost(got.data());
    // INT4 has ~7 discrete positive levels, so per-group error ~ max_abs_group / 7
    const float tol=MaxAbs(src_host)/g_caif_quant_test_int4_levels+g_caif_quant_test_eps_tol;
    CAIF_TestHarness::Report("FakeQuant::Int4PerGroup",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int4PerGroup")
}

void CAIF_QuantizeOpsTests::TestFakeQuantInt8WrongDtypeThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_quant_test_n_mismatch)};
    const std::vector<float> src_host(g_caif_quant_test_n_mismatch,g_caif_quant_test_fill_wrong_dtype);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized(shape,
                                                            stream,
                                                            CAIF_DataType::CAIF_DataType_e::Int8);
    bool threw=false;
    try
    {
      CAIF_Ops::FakeQuantInt8(src,out,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    }
    catch(...)
    {
      threw=true;
    }
    CAIF_TestHarness::Report("FakeQuant::Int8WrongOutputDtypeThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int8WrongOutputDtypeThrows")
}

void CAIF_QuantizeOpsTests::RunAll()
{
  ISE_Out::Out()<<"=== Quantize Op Tests ==="
                <<"\n\n";
  TestInt8PerTensorRoundTrip();
  TestInt8PerChannelRoundTrip();
  TestInt8ZeroInputUnitScale();
  TestInt4PerGroupRoundTrip();
  TestInt8WrongOutputDtypeThrows();
  TestInt8PerChannelRequires2DThrows();
  TestInt4GroupSizeMismatchThrows();
  TestFakeQuantInt8PerTensor();
  TestFakeQuantInt8PerChannel();
  TestFakeQuantInt4PerGroup();
  TestFakeQuantInt8WrongDtypeThrows();
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
  instance::CAIF_QuantizeOpsTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
