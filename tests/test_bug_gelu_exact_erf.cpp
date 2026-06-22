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
// GELU was hard-wired to the tanh approximation
//   f(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// with no exact (erf) form, so models whose activation config asks for the
// PyTorch nn.GELU() / BERT-original GELU
//   f(x) = 0.5*x*(1 + erf(x/sqrt(2)))
// silently ran the approximation (a ~2e-4 parity gap around |x|~2.3).
//
// These tests drive the new exact path through CAIF_Ops::GELU/GELUBackward
// (host + device) and through CAIF_DeviceGELUActivation(Exact), and check it
// against an independent fp64 erf reference. A silent fall-back to tanh FAILS
// (the device output sits closer to the tanh reference than the erf one at the
// point of maximum divergence); the exact kernels PASS. The Tanh default is
// also re-checked so the feature does not move existing numerics.
//------------------------------------------------------------------------------
#include "caif_ops.h"
#include "caif_gelu_approximation.h"
#include "caif_device_gelu_activation.h"
#include "caif_serialization_constants.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr int32_t g_gelu_erf_n=256;
// fp32 erff vs an fp64 erf reference: a few ulps on values up to ~6.
constexpr float g_gelu_erf_tol=5.0e-5f;
constexpr float g_gelu_erf_grad_tol=1.0e-4f;
// The exact and tanh formulas must diverge by more than this somewhere in the
// sampled range, so "matches erf but not tanh" is a meaningful claim.
constexpr float g_gelu_erf_distinct=1.0e-4f;
// Exact-GELU math constants (fp64 reference side).
constexpr double g_gelu_erf_inv_sqrt2=0.70710678118654752;
constexpr double g_gelu_erf_inv_sqrt2pi=0.39894228040143268;
constexpr double g_gelu_tanh_k=0.79788456080286536;
constexpr double g_gelu_tanh_c=0.044715;

class CAIF_GeluExactErfTest
{
  public:
    static void RunAll();

  protected:

  private:
    static double Input(const int32_t i);
    static double GeluExact(const double x);
    static double GeluTanh(const double x);
    static double GeluExactGrad(const double x);
    static double GeluTanhGrad(const double x);
    static std::vector<float> HostInputs();

    static void TestDeviceExactForward();
    static void TestDeviceExactBackward();
    static void TestHostExactForward();
    static void TestActivationObjectAndClone();
};

// Deterministic sweep over [-6.4, 6.35], crossing the |x|~2.3 region where the
// exact and tanh formulas diverge the most.
double CAIF_GeluExactErfTest::Input(const int32_t i)
{
  return static_cast<double>(i-128)*0.05;
}

double CAIF_GeluExactErfTest::GeluExact(const double x)
{
  return 0.5*x*(1.0+std::erf(x*g_gelu_erf_inv_sqrt2));
}

double CAIF_GeluExactErfTest::GeluTanh(const double x)
{
  const double inner=g_gelu_tanh_k*(x+g_gelu_tanh_c*x*x*x);
  return 0.5*x*(1.0+std::tanh(inner));
}

double CAIF_GeluExactErfTest::GeluExactGrad(const double x)
{
  const double cdf=0.5*(1.0+std::erf(x*g_gelu_erf_inv_sqrt2));
  const double pdf=g_gelu_erf_inv_sqrt2pi*std::exp(-0.5*x*x);
  return cdf+x*pdf;
}

double CAIF_GeluExactErfTest::GeluTanhGrad(const double x)
{
  const double inner=g_gelu_tanh_k*(x+g_gelu_tanh_c*x*x*x);
  const double th=std::tanh(inner);
  const double di=g_gelu_tanh_k*(1.0+3.0*g_gelu_tanh_c*x*x);
  return 0.5*(1.0+th)+0.5*x*(1.0-th*th)*di;
}

std::vector<float> CAIF_GeluExactErfTest::HostInputs()
{
  std::vector<float> in(static_cast<size_t>(g_gelu_erf_n));
  for(int32_t i=0;i<g_gelu_erf_n;++i)
  {
    in[static_cast<size_t>(i)]=static_cast<float>(Input(i));
  }
  return in;
}

void CAIF_GeluExactErfTest::TestDeviceExactForward()
{
  try
  {
    const std::vector<float> in=HostInputs();
    CAIF_CudaStream stream;
    CAIF_DeviceTensor in_dev=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    in_dev.CopyFromHost(in.data(),in.size());
    CAIF_DeviceTensor out_dev=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    CAIF_Ops::GELU(in_dev,out_dev,CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);
    stream.Synchronize();

    std::vector<float> got(static_cast<size_t>(g_gelu_erf_n));
    out_dev.CopyToHost(got.data());

    bool matches_erf=true;
    double worst_erf=0.0;
    double max_divergence=0.0;
    int32_t i_div=0;
    for(int32_t i=0;i<g_gelu_erf_n;++i)
    {
      const double x=Input(i);
      const double err=std::fabs(static_cast<double>(got[static_cast<size_t>(i)])-GeluExact(x));
      if(err>=static_cast<double>(g_gelu_erf_tol))
      {
        matches_erf=false;
      }
      if(err>worst_erf)
      {
        worst_erf=err;
      }
      const double divergence=std::fabs(GeluExact(x)-GeluTanh(x));
      if(divergence>max_divergence)
      {
        max_divergence=divergence;
        i_div=i;
      }
    }

    // At the point of maximum exact-vs-tanh divergence the device output must
    // sit closer to the erf reference than the tanh one — proof it ran erf.
    const double x_div=Input(i_div);
    const double got_div=static_cast<double>(got[static_cast<size_t>(i_div)]);
    const bool closer_to_erf=(std::fabs(got_div-GeluExact(x_div))<
                              std::fabs(got_div-GeluTanh(x_div)));
    const bool meaningful=(max_divergence>static_cast<double>(g_gelu_erf_distinct));
    const bool ok=(matches_erf==true&&closer_to_erf==true&&meaningful==true);

    if(ok==false)
    {
      ISE_Out::Out()<<"  device exact GELU: worst |out-erf| "
                    <<worst_erf
                    <<" (tol "
                    <<g_gelu_erf_tol
                    <<"), max exact-vs-tanh divergence "
                    <<max_divergence
                    <<", closer_to_erf "
                    <<closer_to_erf
                    <<"\n";
    }
    CAIF_TestHarness::Report("BugF5::GELU::DeviceExactForward",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugF5::GELU::DeviceExactForward")
}

void CAIF_GeluExactErfTest::TestDeviceExactBackward()
{
  try
  {
    const std::vector<float> in=HostInputs();
    const std::vector<float> ones(static_cast<size_t>(g_gelu_erf_n),1.0f);
    CAIF_CudaStream stream;
    CAIF_DeviceTensor in_dev=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    in_dev.CopyFromHost(in.data(),in.size());
    CAIF_DeviceTensor go_dev=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    go_dev.CopyFromHost(ones.data(),ones.size());
    CAIF_DeviceTensor gi_dev=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    CAIF_Ops::GELUBackward(go_dev,
                           in_dev,
                           gi_dev,
                           CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);
    stream.Synchronize();

    std::vector<float> got(static_cast<size_t>(g_gelu_erf_n));
    gi_dev.CopyToHost(got.data());

    bool matches_erf=true;
    double worst_erf=0.0;
    double max_divergence=0.0;
    int32_t i_div=0;
    for(int32_t i=0;i<g_gelu_erf_n;++i)
    {
      const double x=Input(i);
      const double err=std::fabs(static_cast<double>(got[static_cast<size_t>(i)])-
                                 GeluExactGrad(x));
      if(err>=static_cast<double>(g_gelu_erf_grad_tol))
      {
        matches_erf=false;
      }
      if(err>worst_erf)
      {
        worst_erf=err;
      }
      const double divergence=std::fabs(GeluExactGrad(x)-GeluTanhGrad(x));
      if(divergence>max_divergence)
      {
        max_divergence=divergence;
        i_div=i;
      }
    }

    const double x_div=Input(i_div);
    const double got_div=static_cast<double>(got[static_cast<size_t>(i_div)]);
    const bool closer_to_erf=(std::fabs(got_div-GeluExactGrad(x_div))<
                              std::fabs(got_div-GeluTanhGrad(x_div)));
    const bool meaningful=(max_divergence>static_cast<double>(g_gelu_erf_distinct));
    const bool ok=(matches_erf==true&&closer_to_erf==true&&meaningful==true);

    if(ok==false)
    {
      ISE_Out::Out()<<"  device exact GELU backward: worst |grad-analytic| "
                    <<worst_erf
                    <<" (tol "
                    <<g_gelu_erf_grad_tol
                    <<"), max exact-vs-tanh divergence "
                    <<max_divergence
                    <<", closer_to_erf "
                    <<closer_to_erf
                    <<"\n";
    }
    CAIF_TestHarness::Report("BugF5::GELU::DeviceExactBackward",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugF5::GELU::DeviceExactBackward")
}

void CAIF_GeluExactErfTest::TestHostExactForward()
{
  try
  {
    const std::vector<float> in=HostInputs();
    CAIF_DeviceTensor in_host=CAIF_DeviceTensor::ZerosHost({1,g_gelu_erf_n});
    std::memcpy(in_host.DeviceDataRaw(),
                in.data(),
                in.size()*sizeof(float));
    CAIF_DeviceTensor out_host=CAIF_DeviceTensor::ZerosHost({1,g_gelu_erf_n});
    CAIF_Ops::GELU(in_host,
                   out_host,
                   CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);

    std::vector<float> got(static_cast<size_t>(g_gelu_erf_n));
    std::memcpy(got.data(),
                out_host.DeviceDataRaw(),
                got.size()*sizeof(float));

    bool ok=true;
    double worst=0.0;
    for(int32_t i=0;i<g_gelu_erf_n;++i)
    {
      const double x=Input(i);
      const double err=std::fabs(static_cast<double>(got[static_cast<size_t>(i)])-GeluExact(x));
      if(err>=static_cast<double>(g_gelu_erf_tol))
      {
        ok=false;
      }
      if(err>worst)
      {
        worst=err;
      }
    }

    if(ok==false)
    {
      ISE_Out::Out()<<"  host exact GELU: worst |out-erf| "
                    <<worst
                    <<" (tol "
                    <<g_gelu_erf_tol
                    <<")\n";
    }
    CAIF_TestHarness::Report("BugF5::GELU::HostExactForward",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugF5::GELU::HostExactForward")
}

void CAIF_GeluExactErfTest::TestActivationObjectAndClone()
{
  try
  {
    const std::vector<float> in=HostInputs();
    CAIF_CudaStream stream;
    CAIF_DeviceTensor in_dev=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    in_dev.CopyFromHost(in.data(),in.size());

    CAIF_DeviceGELUActivation<float,float> act_exact(
      CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);
    CAIF_DeviceGELUActivation<float,float> act_default;

    CAIF_DeviceTensor out_exact=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    CAIF_DeviceTensor out_clone=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);
    CAIF_DeviceTensor out_default=CAIF_DeviceTensor::Zeros({1,g_gelu_erf_n},stream);

    act_exact.Forward(in_dev,out_exact);
    std::unique_ptr<CAIF_DeviceActivation> clone=act_exact.Clone();
    static_cast<const CAIF_DevicePointwiseActivation *>(clone.get())->Forward(in_dev,out_clone);
    act_default.Forward(in_dev,out_default);
    stream.Synchronize();

    std::vector<float> exact(static_cast<size_t>(g_gelu_erf_n));
    std::vector<float> cloned(static_cast<size_t>(g_gelu_erf_n));
    std::vector<float> deflt(static_cast<size_t>(g_gelu_erf_n));
    out_exact.CopyToHost(exact.data());
    out_clone.CopyToHost(cloned.data());
    out_default.CopyToHost(deflt.data());

    bool exact_ok=true;
    bool clone_ok=true;
    bool default_ok=true;
    for(int32_t i=0;i<g_gelu_erf_n;++i)
    {
      const double x=Input(i);
      const size_t k=static_cast<size_t>(i);
      if(std::fabs(static_cast<double>(exact[k])-GeluExact(x))>=
         static_cast<double>(g_gelu_erf_tol))
      {
        exact_ok=false;
      }
      if(std::fabs(static_cast<double>(cloned[k])-GeluExact(x))>=
         static_cast<double>(g_gelu_erf_tol))
      {
        clone_ok=false;
      }
      if(std::fabs(static_cast<double>(deflt[k])-GeluTanh(x))>=
         static_cast<double>(g_gelu_erf_tol))
      {
        default_ok=false;
      }
    }

    const bool desc_ok=(act_exact.Description()==g_serial_tag_gelu_exact&&
                        act_default.Description()==g_serial_tag_gelu);
    const bool ok=(exact_ok==true&&clone_ok==true&&default_ok==true&&desc_ok==true);

    if(ok==false)
    {
      ISE_Out::Out()<<"  activation object: exact_ok "
                    <<exact_ok
                    <<" clone_ok "
                    <<clone_ok
                    <<" default_ok "
                    <<default_ok
                    <<" desc_ok "
                    <<desc_ok
                    <<"\n";
    }
    CAIF_TestHarness::Report("BugF5::GELU::ActivationObjectAndClone",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugF5::GELU::ActivationObjectAndClone")
}

void CAIF_GeluExactErfTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug F5: exact (erf) GELU ==="
                <<"\n\n";
  TestDeviceExactForward();
  TestDeviceExactBackward();
  TestHostExactForward();
  TestActivationObjectAndClone();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_GeluExactErfTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
