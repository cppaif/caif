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
// 64-bit element-count / index path. Proves a tensor with
// more than INT_MAX (2,147,483,647) elements is processed correctly end to end.
//
// The scalar fill + scalar tanh kernels index EVERY element one-per-thread, so
// an element whose linear index is past the 32-bit ceiling is owned by a thread
// whose `blockIdx.x*blockDim.x` product exceeds INT_MAX. With the old `int`
// indexing that product overflowed to negative, so the element was never
// written (or written out of bounds) and reading it back yields the wrong
// value. With int64 indexing it is exact. This test fills a 2.2B-element fp32
// tensor with 0.5, applies tanh in place, and asserts the element at index
// INT_MAX+1000 equals tanh(0.5).
//
// fp32 (8.8 GB, in-place) is deliberate: vectorized kernels divide the count by
// the lane width, so a tensor large enough to overflow their vector count does
// not fit in 32 GB. Only a SCALAR kernel (index up to n) overflows in range,
// which is exactly what this exercises.
//------------------------------------------------------------------------------
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "caif_cuda_kernels_activations.cuh"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <climits>
#include <cstdint>

namespace instance
{

#ifdef USE_CAIF_CUDA

// 2.2e9 > INT_MAX (2,147,483,647); fits uint32_t (< 4.29e9); ~8.8 GB as fp32.
constexpr uint32_t g_caif_big_n=2200000000u;
constexpr float g_caif_big_fill=0.5f;
constexpr float g_caif_big_tol=1e-5f;

class CAIF_LargeTensorInt64Test
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestScalarKernelPastIntMax();
};

void CAIF_LargeTensorInt64Test::TestScalarKernelPastIntMax()
{
  try
  {
    CAIF_CudaStream stream;
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    CAIF_DeviceTensor tensor=CAIF_DeviceTensor::Uninitialized({g_caif_big_n},stream,fp32);

    // Scalar fill across all 2.2B elements, then in-place scalar tanh — both
    // kernels touch every element, so the elements past INT_MAX exercise the
    // 64-bit global index.
    tensor.Fill(g_caif_big_fill);
    float *data=tensor.DevicePtr<float>();
    launch_tanh_forward<float>(data,data,static_cast<int64_t>(g_caif_big_n),stream.Handle());

    const cudaError_t sync_status=cudaStreamSynchronize(stream.Handle());
    bool ok=(sync_status==cudaSuccess);
    if(ok==false)
    {
      ISE_Out::Out()<<"  stream sync failed: "<<cudaGetErrorString(sync_status)<<"\n";
    }

    // Read element 0 (sanity) and one whose linear index is just past the
    // 32-bit ceiling. Both must equal tanh(0.5).
    const int64_t past_int_max=static_cast<int64_t>(INT_MAX)+1000;
    const float expected=std::tanh(g_caif_big_fill);
    float got_first=0.0f;
    float got_past=0.0f;
    cudaMemcpy(&got_first,data,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&got_past,data+past_int_max,sizeof(float),cudaMemcpyDeviceToHost);

    const bool first_ok=(std::fabs(got_first-expected)<g_caif_big_tol);
    const bool past_ok=(std::fabs(got_past-expected)<g_caif_big_tol);
    if(past_ok==false)
    {
      ISE_Out::Out()<<"  element at index INT_MAX+1000 = "<<got_past
                    <<", expected tanh(0.5) = "<<expected
                    <<" (32-bit index overflow not fixed)\n";
    }
    ok=ok&&first_ok&&past_ok;

    CAIF_TestHarness::Report("LargeTensor::ScalarKernelPastIntMax",ok);
  }
  CAIF_TEST_CATCH_BLOCK("LargeTensor::ScalarKernelPastIntMax")
}

void CAIF_LargeTensorInt64Test::RunAll()
{
  ISE_Out::Out()<<"=== F8: 64-bit index path (>INT_MAX-element tensor) ==="
                <<"\n\n";
  TestScalarKernelPastIntMax();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_LargeTensorInt64Test::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
