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
// layernorm_forward_kernel computed the per-row variance
// in one pass as E[x^2] - E[x]^2. When a row has a large DC offset relative to
// its spread (residual-stream activations at depth), E[x^2] and mean^2 are both
// large and nearly equal, so the fp32 subtraction loses the variance to
// catastrophic cancellation — it can even go negative, leaving rsqrt to ride
// the +epsilon floor. PyTorch/HF compute variance the stable two-pass way
// (mean first, then mean((x-mean)^2)).
//
// This test feeds a row with a large offset (1e4) and a small deterministic
// spread, gamma=1 / beta=0 (the layer's init), and compares the device output
// against an independent fp64 two-pass reference. The one-pass kernel produces
// large error / non-finite output and FAILS; the two-pass fix matches and
// PASSES.
//------------------------------------------------------------------------------
#include "caif_device_layernorm.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_ln_var_dim=256;
constexpr uint32_t g_ln_var_rows=2;
// Large DC offset relative to the spread is what drives the one-pass formula
// into catastrophic cancellation; 1e4 squared (1e8) dwarfs the true variance.
constexpr float g_ln_var_offset=10000.0f;
constexpr float g_ln_var_epsilon=1.0e-5f;
// One-pass error is orders of magnitude above this; the two-pass result sits
// well below it.
constexpr float g_ln_var_tolerance=1.0e-2f;

class CAIF_LayerNormVarianceBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static float SpreadValue(const uint32_t col);
    static void TestLargeOffsetVarianceStable();
};

// Deterministic small per-column spread around the offset (range ~[-0.5,0.5]),
// so the true variance is O(0.1) while the offset is 1e4.
float CAIF_LayerNormVarianceBugTest::SpreadValue(const uint32_t col)
{
  return static_cast<float>(static_cast<int32_t>(col%11)-5)*0.1f;
}

void CAIF_LayerNormVarianceBugTest::TestLargeOffsetVarianceStable()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLayerNorm<float,float> norm(g_ln_var_dim,stream,g_ln_var_epsilon);

    const size_t total=static_cast<size_t>(g_ln_var_rows)*g_ln_var_dim;
    std::vector<float> host_input(total);
    for(uint32_t r=0;r<g_ln_var_rows;++r)
    {
      for(uint32_t c=0;c<g_ln_var_dim;++c)
      {
        host_input[static_cast<size_t>(r)*g_ln_var_dim+c]=g_ln_var_offset+SpreadValue(c);
      }
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {g_ln_var_rows,g_ln_var_dim},
                                                            stream);
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceTensor output=norm.Forward(input,ctx);

    std::vector<float> host_output(total);
    output.CopyToHost(host_output.data());

    // Independent fp64 two-pass reference: mean, then mean((x-mean)^2), then
    // y=(x-mean)/sqrt(var+eps) with gamma=1, beta=0.
    bool ok=true;
    float worst=0.0f;
    for(uint32_t r=0;r<g_ln_var_rows;++r)
    {
      double mean=0.0;
      for(uint32_t c=0;c<g_ln_var_dim;++c)
      {
        mean+=static_cast<double>(host_input[static_cast<size_t>(r)*g_ln_var_dim+c]);
      }
      mean/=static_cast<double>(g_ln_var_dim);

      double var=0.0;
      for(uint32_t c=0;c<g_ln_var_dim;++c)
      {
        const double d=static_cast<double>(host_input[static_cast<size_t>(r)*g_ln_var_dim+c])-mean;
        var+=d*d;
      }
      var/=static_cast<double>(g_ln_var_dim);
      const double rstd=1.0/std::sqrt(var+static_cast<double>(g_ln_var_epsilon));

      for(uint32_t c=0;c<g_ln_var_dim;++c)
      {
        const double x=static_cast<double>(host_input[static_cast<size_t>(r)*g_ln_var_dim+c]);
        const double y_ref=(x-mean)*rstd;
        const float y=host_output[static_cast<size_t>(r)*g_ln_var_dim+c];
        const float diff=std::fabs(y-static_cast<float>(y_ref));
        const bool finite_ok=(std::isfinite(y)==true);
        if(finite_ok==false||diff>=g_ln_var_tolerance)
        {
          ok=false;
        }
        if(finite_ok==true&&diff>worst)
        {
          worst=diff;
        }
      }
    }

    if(ok==false)
    {
      ISE_Out::Out()<<"  LayerNorm large-offset output diverged from fp64 two-pass"
                    <<" reference (worst finite abs diff "
                    <<worst
                    <<", tol "
                    <<g_ln_var_tolerance
                    <<")\n";
    }
    CAIF_TestHarness::Report("BugC5::LayerNorm::LargeOffsetVarianceStable",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugC5::LayerNorm::LargeOffsetVarianceStable")
}

void CAIF_LayerNormVarianceBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug C5: LayerNorm two-pass variance stability ==="
                <<"\n\n";
  TestLargeOffsetVarianceStable();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_LayerNormVarianceBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
