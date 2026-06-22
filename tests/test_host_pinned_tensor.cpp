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
// Test: CAIF_HostPinnedTensor — construction, dtype, byte count, round-trip
// to/from device, writeback from device, and move semantics.
//------------------------------------------------------------------------------
#include "caif_host_pinned_tensor.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_test_harness.h"
#include "caif_data_type.h"
#include "ise_lib/ise_out.h"

#include <cstring>
#include <vector>
#include <cstdint>

namespace instance
{

constexpr uint32_t g_caif_pinned_test_rows=4;
constexpr uint32_t g_caif_pinned_test_cols=8;
constexpr size_t g_caif_pinned_test_n=32u;
constexpr size_t g_caif_pinned_test_bytes=128u;
constexpr size_t g_caif_pinned_test_bytes_shape=4u*8u*4u;
constexpr size_t g_caif_pinned_test_uint8_n=64u;
constexpr float g_caif_pinned_test_float_scale=0.5f;
constexpr float g_caif_pinned_test_float_bias=-1.0f;
constexpr float g_caif_pinned_test_tol=1e-6f;

class CAIF_HostPinnedTensorTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestConstructAndFree();
    static void TestRoundTripFloat32();
    static void TestRoundTripBytes();
    static void TestCopyFromDeviceWriteback();
    static void TestMoveSemantics();
};

void CAIF_HostPinnedTensorTests::TestConstructAndFree()
{
  try
  {
    const std::vector<uint32_t> shape={g_caif_pinned_test_rows,g_caif_pinned_test_cols};
    CAIF_HostPinnedTensor t(shape,CAIF_DataType::CAIF_DataType_e::Float32);
    bool passed=true;
    if(t.Shape()!=shape)
    {
      ISE_Out::ErrLog()<<"  shape mismatch"
                       <<"\n";
      passed=false;
    }
    if(t.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      ISE_Out::ErrLog()<<"  dtype mismatch"
                       <<"\n";
      passed=false;
    }
    if(t.Bytes()!=g_caif_pinned_test_bytes_shape)
    {
      ISE_Out::ErrLog()<<"  bytes mismatch (expected 128, got "
                       <<t.Bytes()
                       <<")"
                       <<"\n";
      passed=false;
    }
    if(t.IsAllocated()==false)
    {
      ISE_Out::ErrLog()<<"  IsAllocated() is false after ctor"
                       <<"\n";
      passed=false;
    }
    if(t.HostPtr()==nullptr)
    {
      ISE_Out::ErrLog()<<"  HostPtr() is null after ctor"
                       <<"\n";
      passed=false;
    }
    CAIF_TestHarness::Report("HostPinnedTensor::ConstructAndFree",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostPinnedTensor::ConstructAndFree")
}

void CAIF_HostPinnedTensorTests::TestRoundTripFloat32()
{
  try
  {
    CAIF_CudaStream stream;
    const std::vector<uint32_t> shape={g_caif_pinned_test_rows,g_caif_pinned_test_cols};
    CAIF_HostPinnedTensor pinned(shape,CAIF_DataType::CAIF_DataType_e::Float32);

    float *src=static_cast<float*>(pinned.HostPtr());
    for(size_t i=0;i<g_caif_pinned_test_n;++i)
    {
      src[i]=static_cast<float>(i)*g_caif_pinned_test_float_scale
             +g_caif_pinned_test_float_bias;
    }

    CAIF_DeviceTensor dev=pinned.PrefetchToDevice(stream);
    stream.Synchronize();

    std::vector<float> back(g_caif_pinned_test_n,0.0f);
    dev.CopyToHostRaw(back.data());

    bool passed=true;
    for(size_t i=0;i<g_caif_pinned_test_n;++i)
    {
      const float expected=static_cast<float>(i)*g_caif_pinned_test_float_scale
                           +g_caif_pinned_test_float_bias;
      if(CAIF_TestHarness::FloatEqual(back[i],expected,g_caif_pinned_test_tol)==false)
      {
        ISE_Out::ErrLog()<<"  mismatch at i="
                         <<i
                         <<": expected "
                         <<expected
                         <<", got "
                         <<back[i]
                         <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("HostPinnedTensor::RoundTripFloat32",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostPinnedTensor::RoundTripFloat32")
}

// uint8 path exercises raw byte transfer
void CAIF_HostPinnedTensorTests::TestRoundTripBytes()
{
  try
  {
    CAIF_CudaStream stream;
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_pinned_test_uint8_n)};
    CAIF_HostPinnedTensor pinned(shape,CAIF_DataType::CAIF_DataType_e::UInt8);

    uint8_t *src=static_cast<uint8_t*>(pinned.HostPtr());
    for(size_t i=0;i<g_caif_pinned_test_uint8_n;++i)
    {
      src[i]=static_cast<uint8_t>((i*37u+11u)&0xffu);
    }

    CAIF_DeviceTensor dev=pinned.PrefetchToDevice(stream);
    stream.Synchronize();

    std::vector<uint8_t> back(g_caif_pinned_test_uint8_n,0u);
    dev.CopyToHostRaw(back.data());

    bool passed=true;
    for(size_t i=0;i<g_caif_pinned_test_uint8_n;++i)
    {
      const uint8_t expected=static_cast<uint8_t>((i*37u+11u)&0xffu);
      if(back[i]!=expected)
      {
        ISE_Out::ErrLog()<<"  mismatch at i="
                         <<i
                         <<": expected 0x"
                         <<std::hex
                         <<static_cast<unsigned int>(expected)
                         <<", got 0x"
                         <<static_cast<unsigned int>(back[i])
                         <<std::dec
                         <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report(
      "HostPinnedTensor::RoundTripBytes (uint8 path exercises raw byte transfer)",
      passed);
  }
  CAIF_TEST_CATCH_BLOCK(
    "HostPinnedTensor::RoundTripBytes (uint8 path exercises raw byte transfer)")
}

void CAIF_HostPinnedTensorTests::TestCopyFromDeviceWriteback()
{
  try
  {
    CAIF_CudaStream stream;
    const std::vector<uint32_t> shape={g_caif_pinned_test_rows,g_caif_pinned_test_cols};
    CAIF_HostPinnedTensor pinned(shape,CAIF_DataType::CAIF_DataType_e::Float32);

    float *host=static_cast<float*>(pinned.HostPtr());
    for(size_t i=0;i<g_caif_pinned_test_n;++i)
    {
      host[i]=0.0f;
    }

    std::vector<float> seed(g_caif_pinned_test_n,0.0f);
    for(size_t i=0;i<g_caif_pinned_test_n;++i)
    {
      seed[i]=static_cast<float>(i+1);
    }
    CAIF_DeviceTensor dev=CAIF_DeviceTensor::Uninitialized(shape,
                                                            stream,
                                                            CAIF_DataType::CAIF_DataType_e::Float32);
    dev.CopyFromHostRaw(seed.data(),g_caif_pinned_test_n*sizeof(float));

    pinned.CopyFromDevice(dev);

    bool passed=true;
    for(size_t i=0;i<g_caif_pinned_test_n;++i)
    {
      const float expected=static_cast<float>(i+1);
      if(CAIF_TestHarness::FloatEqual(host[i],expected,g_caif_pinned_test_tol)==false)
      {
        ISE_Out::ErrLog()<<"  writeback mismatch at i="
                         <<i
                         <<": expected "
                         <<expected
                         <<", got "
                         <<host[i]
                         <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("HostPinnedTensor::CopyFromDeviceWriteback",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostPinnedTensor::CopyFromDeviceWriteback")
}

void CAIF_HostPinnedTensorTests::TestMoveSemantics()
{
  try
  {
    const std::vector<uint32_t> shape={g_caif_pinned_test_rows,g_caif_pinned_test_cols};
    CAIF_HostPinnedTensor a(shape,CAIF_DataType::CAIF_DataType_e::Float32);
    void *original_host_ptr=a.HostPtr();

    CAIF_HostPinnedTensor b(std::move(a));

    bool passed=true;
    if(b.HostPtr()!=original_host_ptr)
    {
      ISE_Out::ErrLog()<<"  moved-to does not own original host buffer"
                       <<"\n";
      passed=false;
    }
    if(a.IsAllocated()==true)
    {
      ISE_Out::ErrLog()<<"  moved-from still IsAllocated()==true"
                       <<"\n";
      passed=false;
    }
    if(b.Shape()!=shape)
    {
      ISE_Out::ErrLog()<<"  moved-to shape mismatch"
                       <<"\n";
      passed=false;
    }
    CAIF_TestHarness::Report("HostPinnedTensor::MoveSemantics",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostPinnedTensor::MoveSemantics")
}

void CAIF_HostPinnedTensorTests::RunAll()
{
  CAIF_TestHarness::Reset();
  TestConstructAndFree();
  TestRoundTripFloat32();
  TestRoundTripBytes();
  TestCopyFromDeviceWriteback();
  TestMoveSemantics();
}

}//end instance namespace

int main(int /*argc*/,char ** /*argv*/)
{
  instance::CAIF_HostPinnedTensorTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
