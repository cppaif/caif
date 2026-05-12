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

using namespace instance;

namespace
{

void ReportResult(const char *name,bool ok)
{
  CAIF_TestHarness::Report(name,ok);
}

#ifdef USE_CAIF_CUDA

std::vector<float> MakeRandom(const size_t n,const uint32_t seed,const float scale)
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

float MaxAbs(const std::vector<float> &v)
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

bool Within(const std::vector<float> &ref,
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

void TestInt8PerTensorRoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const size_t n=1024;
    const std::vector<float> src_host=MakeRandom(n,11,2.5f);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(n)};
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
    std::vector<float> got(n);
    back.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/127.0f+1e-6f;
    ReportResult("Quantize::Int8PerTensorRoundTrip",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8PerTensorRoundTrip")
}

void TestInt8PerChannelRoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const uint32_t rows=32;
    const uint32_t cols=64;
    const std::vector<uint32_t> shape={rows,cols};
    std::vector<float> src_host(rows*cols);
    std::mt19937 gen(22);
    for(uint32_t c=0;c<cols;++c)
    {
      const float col_scale=0.1f+static_cast<float>(c)*0.05f;
      std::uniform_real_distribution<float> dist(-col_scale,col_scale);
      for(uint32_t r=0;r<rows;++r)
      {
        src_host[r*cols+c]=dist(gen);
      }
    }
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized(shape,
                                                         stream,
                                                         CAIF_DataType::CAIF_DataType_e::Int8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({cols},
                                                              stream,
                                                              CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor back=CAIF_DeviceTensor::Uninitialized(shape,
                                                            stream,
                                                            CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::QuantizeInt8(src,q,scales,CAIF_Ops::QuantScheme_e::PerChannel_e,ctx);
    CAIF_Ops::DequantizeInt8(q,back,scales,CAIF_Ops::QuantScheme_e::PerChannel_e,ctx);
    std::vector<float> got(src_host.size());
    back.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/127.0f+1e-6f;
    ReportResult("Quantize::Int8PerChannelRoundTrip",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8PerChannelRoundTrip")
}

void TestInt8ZeroInputUnitScale()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const size_t n=256;
    const std::vector<float> src_host(n,0.0f);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(n)};
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
    std::vector<float> got(n);
    back.CopyToHost(got.data());
    bool ok=true;
    for(const float v:got)
    {
      if(std::isnan(v)||v!=0.0f)
      {
        ok=false;
        break;
      }
    }
    ReportResult("Quantize::Int8ZeroInputUnitScale",ok);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8ZeroInputUnitScale")
}

void TestInt4PerGroupRoundTrip()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const size_t n=512;
    const uint32_t group_size=128;
    const std::vector<float> src_host=MakeRandom(n,33,1.0f);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(n)};
    const uint32_t num_groups=static_cast<uint32_t>(n)/group_size;
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
    CAIF_Ops::QuantizeInt4PerGroup(src,q,scales,group_size,ctx);
    CAIF_Ops::DequantizeInt4PerGroup(q,back,scales,group_size,ctx);
    std::vector<float> got(n);
    back.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/7.0f+1e-3f;
    ReportResult("Quantize::Int4PerGroupRoundTrip",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int4PerGroupRoundTrip")
}

void TestInt8WrongOutputDtypeThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host(32,0.5f);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),{32},stream);
    CAIF_DeviceTensor bad_out=CAIF_DeviceTensor::Uninitialized({32},
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
    ReportResult("Quantize::Int8WrongOutputDtypeThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8WrongOutputDtypeThrows")
}

void TestInt8PerChannelRequires2DThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host(64,0.25f);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),{64},stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized({64},
                                                         stream,
                                                         CAIF_DataType::CAIF_DataType_e::Int8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({64},
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
    ReportResult("Quantize::Int8PerChannelRequires2DThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int8PerChannelRequires2DThrows")
}

void TestInt4GroupSizeMismatchThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<float> src_host(256,0.3f);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),{256},stream);
    CAIF_DeviceTensor q=CAIF_DeviceTensor::Uninitialized({256},
                                                         stream,
                                                         CAIF_DataType::CAIF_DataType_e::Int4);
    // 256 elements / group 128 = 2 groups; pass a mismatched scale count.
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({5},
                                                              stream,
                                                              CAIF_DataType::CAIF_DataType_e::Float16);
    bool threw=false;
    try
    {
      CAIF_Ops::QuantizeInt4PerGroup(src,q,scales,128,ctx);
    }
    catch(...)
    {
      threw=true;
    }
    ReportResult("Quantize::Int4GroupSizeMismatchThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("Quantize::Int4GroupSizeMismatchThrows")
}

void TestFakeQuantInt8PerTensor()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const size_t n=256;
    const std::vector<float> src_host=MakeRandom(n,101,1.0f);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(n)};
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized(shape,
                                                           stream,
                                                           CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::FakeQuantInt8(src,out,CAIF_Ops::QuantScheme_e::PerTensor_e,ctx);
    std::vector<float> got(n);
    out.CopyToHost(got.data());
    // round-trip error bounded by half a scale: max_abs / 127 / 2 plus tiny eps
    const float tol=MaxAbs(src_host)/127.0f+1e-3f;
    ReportResult("FakeQuant::Int8PerTensor",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int8PerTensor")
}

void TestFakeQuantInt8PerChannel()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const uint32_t rows=8;
    const uint32_t cols=32;
    const size_t n=static_cast<size_t>(rows)*cols;
    const std::vector<float> src_host=MakeRandom(n,102,1.0f);
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),
                                                          {rows,cols},
                                                          stream);
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized({rows,cols},
                                                           stream,
                                                           CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::FakeQuantInt8(src,out,CAIF_Ops::QuantScheme_e::PerChannel_e,ctx);
    std::vector<float> got(n);
    out.CopyToHost(got.data());
    const float tol=MaxAbs(src_host)/127.0f+1e-3f;
    ReportResult("FakeQuant::Int8PerChannel",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int8PerChannel")
}

void TestFakeQuantInt4PerGroup()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const size_t n=512;
    const uint32_t group_size=128;
    const std::vector<float> src_host=MakeRandom(n,103,1.0f);
    const std::vector<uint32_t> shape={static_cast<uint32_t>(n)};
    CAIF_DeviceTensor src=CAIF_DeviceTensor::FromHostData(src_host.data(),shape,stream);
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized(shape,
                                                           stream,
                                                           CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::FakeQuantInt4PerGroup(src,out,group_size,ctx);
    std::vector<float> got(n);
    out.CopyToHost(got.data());
    // INT4 has ~7 discrete positive levels, so per-group error ~ max_abs_group / 7
    const float tol=MaxAbs(src_host)/7.0f+1e-3f;
    ReportResult("FakeQuant::Int4PerGroup",Within(src_host,got,tol));
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int4PerGroup")
}

void TestFakeQuantInt8WrongDtypeThrows()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::vector<uint32_t> shape={64};
    const std::vector<float> src_host(64,0.5f);
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
    ReportResult("FakeQuant::Int8WrongOutputDtypeThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("FakeQuant::Int8WrongOutputDtypeThrows")
}

#endif// USE_CAIF_CUDA

}// anon

int main()
{
  ISE_Out::Out()<<"=== Quantize Op Tests ===\n\n";

#ifdef USE_CAIF_CUDA
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
