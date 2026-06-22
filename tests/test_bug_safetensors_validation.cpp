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
// CAIF_SafeTensorsFormat::LoadSingleTensor validated only
// that (data_offset_end - data_offset_start) matched the shape-derived byte
// count; it never checked the offsets against the file size or the read result,
// so a truncated/hostile checkpoint loaded zero-padded garbage with no error.
//
// This test forges a file with a VALID header declaring one F32 (4 data bytes)
// but writes no tensor data, so the claimed region runs past EOF. Load must
// throw. It FAILS against the old loader (silent short read) and PASSES once
// the bounds/read checks are in.
//------------------------------------------------------------------------------
#include "caif_safetensors_format.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

class CAIF_SafeTensorsValidationBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestTruncatedDataThrows();
};

void CAIF_SafeTensorsValidationBugTest::TestTruncatedDataThrows()
{
  const std::string path="r3_malformed.safetensors";
  try
  {
    // Valid SafeTensors header declaring one F32 (4 data bytes), but the file
    // is written with no tensor data — the claimed region runs past EOF.
    const std::string json="{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    const uint64_t header_len=json.size();

    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len),sizeof(header_len));
    out.write(json.data(),static_cast<std::streamsize>(json.size()));
    out.close();

    CAIF_CudaStream stream;
    CAIF_SafeTensorsFormat fmt;
    bool threw=false;
    try
    {
      fmt.Load(path,stream);
    }
    catch(const CAIF_Exception &)
    {
      threw=true;
    }
    std::remove(path.c_str());

    if(threw==false)
    {
      ISE_Out::Out()<<"  Load accepted a truncated file instead of throwing\n";
    }
    CAIF_TestHarness::Report("BugR3::SafeTensors::TruncatedDataThrows",threw);
  }
  CAIF_TEST_CATCH_BLOCK("BugR3::SafeTensors::TruncatedDataThrows")
}

void CAIF_SafeTensorsValidationBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug R3: SafeTensors data-bounds validation ==="
                <<"\n\n";
  TestTruncatedDataThrows();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_SafeTensorsValidationBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
