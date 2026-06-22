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
// CAIF_DeviceDenseLayer OVERWRITES its weight/bias
// gradients on each backward, while CAIF_DeviceFFN and CAIF_DeviceLinearHead
// ACCUMULATE (delta + Add). PyTorch-style gradient accumulation runs several
// backward passes before a single optimizer step / zero, summing the grads.
//
// This test runs two forward+backward cycles with identical input and
// grad_output and no interleaved ZeroGradients. With accumulation the weight
// gradient after the second cycle is exactly 2x the first; with the current
// overwrite it equals the first. The test asserts the 2x (accumulation)
// result: it FAILS against the current dense layer and PASSES once the dense
// layer accumulates like FFN / LinearHead.
//------------------------------------------------------------------------------
#include "caif_device_dense_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_dense_bug_batch=2;
constexpr uint32_t g_caif_dense_bug_in=4;
constexpr uint32_t g_caif_dense_bug_out=4;
constexpr uint32_t g_caif_dense_bug_seed=1u;
constexpr float g_caif_dense_bug_input_value=1.0f;
constexpr float g_caif_dense_bug_grad_value=1.0f;
constexpr float g_caif_dense_bug_tol=1.0e-4f;

class CAIF_DenseGradAccumBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void FillTensor(CAIF_DeviceTensor &t,const float value);
    static std::vector<float> ReadTensor(const CAIF_DeviceTensor &t);
    static void TestGradientAccumulation();
};

void CAIF_DenseGradAccumBugTest::FillTensor(CAIF_DeviceTensor &t,const float value)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n,value);
  t.CopyFromHost(host.data(),n);
}

std::vector<float> CAIF_DenseGradAccumBugTest::ReadTensor(const CAIF_DeviceTensor &t)
{
  const size_t n=t.TotalElements();
  std::vector<float> host(n);
  t.CopyToHost(host.data());
  return host;
}

void CAIF_DenseGradAccumBugTest::TestGradientAccumulation()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    CAIF_DeviceDenseLayer<float,float> dense(g_caif_dense_bug_in,
                                             g_caif_dense_bug_out,
                                             CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,
                                             stream,
                                             true);
    dense.InitializeWeights(g_caif_dense_bug_seed);

    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    CAIF_DeviceTensor input=CAIF_DeviceTensor::Zeros({g_caif_dense_bug_batch,g_caif_dense_bug_in},
                                                     stream,
                                                     fp32);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::Zeros(
                                    {g_caif_dense_bug_batch,g_caif_dense_bug_out},
                                    stream,
                                    fp32);
    FillTensor(input,g_caif_dense_bug_input_value);
    FillTensor(grad_output,g_caif_dense_bug_grad_value);

    dense.Forward(input,ctx);
    dense.Backward(grad_output,ctx);
    const std::vector<float> grad_after_one=ReadTensor(dense.WeightGradients());

    dense.Forward(input,ctx);
    dense.Backward(grad_output,ctx);
    const std::vector<float> grad_after_two=ReadTensor(dense.WeightGradients());

    bool nonzero=false;
    for(size_t i=0;i<grad_after_one.size();++i)
    {
      if(std::fabs(grad_after_one[i])>g_caif_dense_bug_tol)
      {
        nonzero=true;
      }
    }

    bool accumulated=true;
    for(size_t i=0;i<grad_after_two.size();++i)
    {
      const float expected=2.0f*grad_after_one[i];
      if(std::fabs(grad_after_two[i]-expected)>g_caif_dense_bug_tol)
      {
        accumulated=false;
      }
    }

    const bool ok=(nonzero==true && accumulated==true);
    if(ok==false)
    {
      ISE_Out::Out()<<"  nonzero="
                    <<nonzero
                    <<" grad1[0]="
                    <<grad_after_one[0]
                    <<" grad2[0]="
                    <<grad_after_two[0]
                    <<" (expected grad2==2*grad1)\n";
    }
    CAIF_TestHarness::Report("BugC3::DenseLayer::GradientAccumulation",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugC3::DenseLayer::GradientAccumulation")
}

void CAIF_DenseGradAccumBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug C3: DenseLayer gradient accumulation ==="
                <<"\n\n";
  TestGradientAccumulation();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_DenseGradAccumBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
