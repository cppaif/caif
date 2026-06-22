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
// The host MatMul family lacks the device path's dtype
// guard. CAIF_Ops::MatMulDevice rejects mismatched input/output dtypes via
// RequireMatchingDtype (cuBLAS cannot mix input dtypes in one GEMM), but the
// host backend (CAIF_HostGemm::MatMul2DFloat) silently upcasts mismatched
// operands to fp32 and computes, so host and device disagree on which inputs
// are legal.
//
// This test drives the host MatMul path with an fp32 A and an fp16 B. Both are
// host-resident, so IsHost(A)==true routes CAIF_Ops::MatMul to MatMulHost. The
// guard is a dtype-metadata check that fires before any data is read, so a
// zero-fill operand is sufficient. Before the F11 fix the host upcasts and does
// NOT throw -> FAILS; after the guard is added it throws like the device path
// -> PASSES.
//------------------------------------------------------------------------------
#include "caif_ops.h"
#include "caif_device_tensor.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cstdint>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_f11_matmul_m=2;
constexpr uint32_t g_f11_matmul_k=2;
constexpr uint32_t g_f11_matmul_n=2;

class CAIF_HostMatMulDtypeGuardBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestHostMatMulRejectsMismatchedDtype();
};

void CAIF_HostMatMulDtypeGuardBugTest::TestHostMatMulRejectsMismatchedDtype()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // A is fp32, B is fp16 -> mismatched input dtypes. Host-resident so the
    // dispatcher routes to MatMulHost.
    CAIF_DeviceTensor a_host=CAIF_DeviceTensor::ZerosHost({g_f11_matmul_m,g_f11_matmul_k});
    CAIF_DeviceTensor b_host=CAIF_DeviceTensor::ZerosHost({g_f11_matmul_k,g_f11_matmul_n},
                                                          CAIF_DataType::CAIF_DataType_e::Float16);
    CAIF_DeviceTensor c_host=CAIF_DeviceTensor::ZerosHost({g_f11_matmul_m,g_f11_matmul_n});

    bool threw=false;
    try
    {
      CAIF_Ops::MatMul(a_host,b_host,c_host,ctx);
    }
    catch(const CAIF_Exception &)
    {
      threw=true;
    }

    if(threw==false)
    {
      ISE_Out::Out()<<"  host MatMul accepted mismatched input dtypes"
                    <<" (fp32 A x fp16 B) without throwing; the device path"
                    <<" rejects this via RequireMatchingDtype\n";
    }
    CAIF_TestHarness::Report("BugF11::MatMulHost::RejectsMismatchedDtype",threw);
  }
  CAIF_TEST_CATCH_BLOCK("BugF11::MatMulHost::RejectsMismatchedDtype")
}

void CAIF_HostMatMulDtypeGuardBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug F11: host MatMul dtype guard ===\n\n";
  TestHostMatMulRejectsMismatchedDtype();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_HostMatMulDtypeGuardBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
