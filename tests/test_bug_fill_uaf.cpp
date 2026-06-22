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
// CAIF_DeviceTensor::Fill(value!=0) used to async-copy
// from a stack-local pageable std::vector that was destroyed before the copy
// completed (use-after-free), plus a host allocation + H2D transfer on every
// fill. It now launches a device fill kernel directly.
//
// This test seeds a device fp32 tensor with one value, fills it with a
// different non-zero value, and asserts every element equals the fill value.
//------------------------------------------------------------------------------
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

// 512 x 512 fp32 elements — large enough that the old pageable async copy
// could race its freed host buffer.
constexpr uint32_t g_caif_fill_bug_n=262144;
constexpr float g_caif_fill_bug_initial=1.0f;
constexpr float g_caif_fill_bug_value=3.14159f;

class CAIF_FillUAFBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestDeviceFillNonZero();
};

void CAIF_FillUAFBugTest::TestDeviceFillNonZero()
{
  try
  {
    CAIF_CudaStream stream;
    std::vector<float> initial(g_caif_fill_bug_n,g_caif_fill_bug_initial);
    CAIF_DeviceTensor tensor=CAIF_DeviceTensor::FromHostData(initial.data(),{g_caif_fill_bug_n},stream);
    tensor.Fill(g_caif_fill_bug_value);

    std::vector<float> host(g_caif_fill_bug_n);
    tensor.CopyToHost(host.data());

    bool ok=true;
    for(size_t i=0;i<g_caif_fill_bug_n;++i)
    {
      if(host[i]!=g_caif_fill_bug_value)
      {
        ok=false;
        break;
      }
    }
    if(ok==false)
    {
      ISE_Out::Out()<<"  device Fill did not set every element to the fill value\n";
    }
    CAIF_TestHarness::Report("BugR2::DeviceFill::NonZero",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugR2::DeviceFill::NonZero")
}

void CAIF_FillUAFBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug R2: device Fill(non-zero) ==="
                <<"\n\n";
  TestDeviceFillNonZero();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FillUAFBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
