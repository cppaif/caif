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

#include "caif_host_pinned_tensor.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_test_harness.h"
#include "caif_data_type.h"

#include "ise_lib/ise_out.h"

#include <cstring>
#include <vector>
#include <cstdint>

using namespace instance;

static const std::vector<uint32_t> g_test_shape={4u,8u};

static void TestConstructAndFree()
{
  const char *test_name="HostPinnedTensor::ConstructAndFree";
  try
  {
    CAIF_HostPinnedTensor t(g_test_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    bool passed=true;
    if(t.Shape()!=g_test_shape)
    {
      ISE_Out::ErrLog()<<"  shape mismatch"<<std::endl;
      passed=false;
    }
    if(t.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      ISE_Out::ErrLog()<<"  dtype mismatch"<<std::endl;
      passed=false;
    }
    if(t.Bytes()!=4u*8u*4u)
    {
      ISE_Out::ErrLog()<<"  bytes mismatch (expected 128, got "
                       <<t.Bytes()
                       <<")"
                       <<std::endl;
      passed=false;
    }
    if(t.IsAllocated()==false)
    {
      ISE_Out::ErrLog()<<"  IsAllocated() is false after ctor"<<std::endl;
      passed=false;
    }
    if(t.HostPtr()==nullptr)
    {
      ISE_Out::ErrLog()<<"  HostPtr() is null after ctor"<<std::endl;
      passed=false;
    }
    CAIF_TestHarness::Report(test_name,passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name)
}

static void TestRoundTripFloat32()
{
  const char *test_name="HostPinnedTensor::RoundTripFloat32";
  try
  {
    CAIF_CudaStream stream;
    CAIF_HostPinnedTensor pinned(g_test_shape,CAIF_DataType::CAIF_DataType_e::Float32);

    const size_t n=4u*8u;
    float *src=static_cast<float*>(pinned.HostPtr());
    for(size_t i=0;i<n;++i)
    {
      src[i]=static_cast<float>(i)*0.5f-1.0f;
    }

    CAIF_DeviceTensor dev=pinned.PrefetchToDevice(stream);
    stream.Synchronize();

    std::vector<float> back(n,0.0f);
    dev.CopyToHostRaw(back.data());

    bool passed=true;
    for(size_t i=0;i<n;++i)
    {
      const float expected=static_cast<float>(i)*0.5f-1.0f;
      if(CAIF_TestHarness::FloatEqual(back[i],expected,1e-6f)==false)
      {
        ISE_Out::ErrLog()<<"  mismatch at i="
                         <<i
                         <<": expected "
                         <<expected
                         <<", got "
                         <<back[i]
                         <<std::endl;
        passed=false;
      }
    }
    CAIF_TestHarness::Report(test_name,passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name)
}

static void TestRoundTripBytes()
{
  const char *test_name="HostPinnedTensor::RoundTripBytes (uint8 path exercises raw byte transfer)";
  try
  {
    CAIF_CudaStream stream;
    const std::vector<uint32_t> shape={64u};
    CAIF_HostPinnedTensor pinned(shape,CAIF_DataType::CAIF_DataType_e::UInt8);

    uint8_t *src=static_cast<uint8_t*>(pinned.HostPtr());
    for(size_t i=0;i<64u;++i)
    {
      src[i]=static_cast<uint8_t>((i*37u+11u)&0xffu);
    }

    CAIF_DeviceTensor dev=pinned.PrefetchToDevice(stream);
    stream.Synchronize();

    std::vector<uint8_t> back(64u,0u);
    dev.CopyToHostRaw(back.data());

    bool passed=true;
    for(size_t i=0;i<64u;++i)
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
                         <<std::endl;
        passed=false;
      }
    }
    CAIF_TestHarness::Report(test_name,passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name)
}

static void TestCopyFromDeviceWriteback()
{
  const char *test_name="HostPinnedTensor::CopyFromDeviceWriteback";
  try
  {
    CAIF_CudaStream stream;
    CAIF_HostPinnedTensor pinned(g_test_shape,CAIF_DataType::CAIF_DataType_e::Float32);

    const size_t n=4u*8u;
    float *host=static_cast<float*>(pinned.HostPtr());
    for(size_t i=0;i<n;++i)
    {
      host[i]=0.0f;
    }

    std::vector<float> seed(n,0.0f);
    for(size_t i=0;i<n;++i)
    {
      seed[i]=static_cast<float>(i+1);
    }
    CAIF_DeviceTensor dev=CAIF_DeviceTensor::Uninitialized(g_test_shape,
                                                            stream,
                                                            CAIF_DataType::CAIF_DataType_e::Float32);
    dev.CopyFromHostRaw(seed.data(),n*sizeof(float));

    pinned.CopyFromDevice(dev);

    bool passed=true;
    for(size_t i=0;i<n;++i)
    {
      const float expected=static_cast<float>(i+1);
      if(CAIF_TestHarness::FloatEqual(host[i],expected,1e-6f)==false)
      {
        ISE_Out::ErrLog()<<"  writeback mismatch at i="
                         <<i
                         <<": expected "
                         <<expected
                         <<", got "
                         <<host[i]
                         <<std::endl;
        passed=false;
      }
    }
    CAIF_TestHarness::Report(test_name,passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name)
}

static void TestMoveSemantics()
{
  const char *test_name="HostPinnedTensor::MoveSemantics";
  try
  {
    CAIF_HostPinnedTensor a(g_test_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    void *original_host_ptr=a.HostPtr();

    CAIF_HostPinnedTensor b(std::move(a));

    bool passed=true;
    if(b.HostPtr()!=original_host_ptr)
    {
      ISE_Out::ErrLog()<<"  moved-to does not own original host buffer"<<std::endl;
      passed=false;
    }
    if(a.IsAllocated()==true)
    {
      ISE_Out::ErrLog()<<"  moved-from still IsAllocated()==true"<<std::endl;
      passed=false;
    }
    if(b.Shape()!=g_test_shape)
    {
      ISE_Out::ErrLog()<<"  moved-to shape mismatch"<<std::endl;
      passed=false;
    }
    CAIF_TestHarness::Report(test_name,passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name)
}

int main(int /*argc*/,char ** /*argv*/)
{
  CAIF_TestHarness::Reset();
  TestConstructAndFree();
  TestRoundTripFloat32();
  TestRoundTripBytes();
  TestCopyFromDeviceWriteback();
  TestMoveSemantics();
  return CAIF_TestHarness::FinalExitCode();
}