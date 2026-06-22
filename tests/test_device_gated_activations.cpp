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
// Direct tests for the 5 gated activation device layers (SwiGLU, GeGLU,
// ReGLU, GLU, Bilinear). GeGLU has its own dedicated FFN-level test in
// test_device_geglu.cpp; this file exercises the bare gated-activation
// classes directly with a {gate, up} -> output forward + a finite-diff
// backward parity check. SwiGLU is also exercised across {fp32, fp16,
// bf16} to verify the templated cells run their declared storage in DRAM.
//------------------------------------------------------------------------------

#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cpu_reference/caif_cpu_activations.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

#ifdef USE_CAIF_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{

constexpr int32_t g_caif_gated_act_test_n=64;
constexpr float g_caif_gated_act_test_fp32_tol=2e-4f;
constexpr float g_caif_gated_act_test_fp16_tol=8e-3f;
constexpr float g_caif_gated_act_test_bf16_tol=3e-2f;
constexpr float g_caif_gated_act_test_finite_diff_h=1e-3f;
constexpr float g_caif_gated_act_test_grad_tol=8e-3f;

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Gated activation forward + backward parity tests.
//------------------------------------------------------------------------------
class CAIF_GatedActivationsTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeInput(const int32_t n,const int32_t seed);
    static float MaxAbsDiff(const float *a,const float *b,const int32_t n);
    static float CpuSigmoid(const float x);
    static float CpuRelu(const float x);

    static CAIF_DeviceTensor MakeFp32Device(const std::vector<float> &data,
                                            CAIF_CudaStream &stream);
    static CAIF_DeviceTensor MakeFp32Like(const CAIF_DeviceTensor &x,
                                          CAIF_CudaStream &stream);
    static std::vector<float> ReadFp32(const CAIF_DeviceTensor &x,
                                       CAIF_CudaStream &stream);

    // CPU references — gated activation forward functions matching the kernel
    // formulations.
    static void CpuSwiGLU(const float *gate,const float *up,float *out,const int32_t n);
    static void CpuGeGLU(const float *gate,const float *up,float *out,const int32_t n);
    static void CpuReGLU(const float *gate,const float *up,float *out,const int32_t n);
    static void CpuGLU(const float *gate,const float *up,float *out,const int32_t n);
    static void CpuBilinear(const float *gate,const float *up,float *out,const int32_t n);

    typedef void (*CpuGatedFn_t)(const float *,const float *,float *,int32_t);

    template<typename ActT>
    static bool RunGatedForwardParity(CpuGatedFn_t cpu_fn,
                                      const int32_t seed_g,
                                      const int32_t seed_u);

    template<typename ActT>
    static bool RunGatedBackwardFD(CpuGatedFn_t cpu_fn,
                                   const int32_t seed_g,
                                   const int32_t seed_u);

    template<typename StorageT>
    static bool RunSwiGLUDtype(const CAIF_DataType::CAIF_DataType_e storage_dt,
                               const float tol);
};

std::vector<float> CAIF_GatedActivationsTests::MakeInput(const int32_t n,const int32_t seed)
{
  std::vector<float> v(static_cast<size_t>(n));
  for(int32_t i=0;i<n;++i)
  {
    const float t=static_cast<float>((i*7+seed)%19)*0.27f;
    v[static_cast<size_t>(i)]=(t-2.5f)*0.55f;
  }
  return v;
}

float CAIF_GatedActivationsTests::MaxAbsDiff(const float *a,const float *b,const int32_t n)
{
  float m=0.0f;
  for(int32_t i=0;i<n;++i)
  {
    const float d=std::fabs(a[i]-b[i]);
    if(d>m)
    {
      m=d;
    }
  }
  return m;
}

float CAIF_GatedActivationsTests::CpuSigmoid(const float x)
{
  return 1.0f/(1.0f+std::exp(-x));
}

float CAIF_GatedActivationsTests::CpuRelu(const float x)
{
  if(x>0.0f)
  {
    return x;
  }
  return 0.0f;
}

CAIF_DeviceTensor CAIF_GatedActivationsTests::MakeFp32Device(const std::vector<float> &data,
                                                              CAIF_CudaStream &stream)
{
  const std::vector<uint32_t> shape={static_cast<uint32_t>(data.size())};
  return CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
}

CAIF_DeviceTensor CAIF_GatedActivationsTests::MakeFp32Like(const CAIF_DeviceTensor &x,
                                                            CAIF_CudaStream &stream)
{
  return CAIF_DeviceTensor::Uninitialized(x.Shape(),
                                          stream,
                                          CAIF_DataType::CAIF_DataType_e::Float32);
}

std::vector<float> CAIF_GatedActivationsTests::ReadFp32(const CAIF_DeviceTensor &x,
                                                         CAIF_CudaStream &stream)
{
  std::vector<float> out(x.TotalElements());
  x.CopyToHost(out.data());
  stream.Synchronize();
  return out;
}

void CAIF_GatedActivationsTests::CpuSwiGLU(const float *gate,
                                            const float *up,
                                            float *out,
                                            const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CAIF_CpuActivations::Swish(gate[i])*up[i];
  }
}

void CAIF_GatedActivationsTests::CpuGeGLU(const float *gate,
                                           const float *up,
                                           float *out,
                                           const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CAIF_CpuActivations::GELU(gate[i])*up[i];
  }
}

void CAIF_GatedActivationsTests::CpuReGLU(const float *gate,
                                           const float *up,
                                           float *out,
                                           const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CpuRelu(gate[i])*up[i];
  }
}

void CAIF_GatedActivationsTests::CpuGLU(const float *gate,
                                         const float *up,
                                         float *out,
                                         const int32_t n)
{
  // GLU = sigmoid(gate) * up.
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CpuSigmoid(gate[i])*up[i];
  }
}

void CAIF_GatedActivationsTests::CpuBilinear(const float *gate,
                                              const float *up,
                                              float *out,
                                              const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=gate[i]*up[i];
  }
}

template<typename ActT>
bool CAIF_GatedActivationsTests::RunGatedForwardParity(CpuGatedFn_t cpu_fn,
                                                        const int32_t seed_g,
                                                        const int32_t seed_u)
{
  CAIF_CudaStream stream;
  std::vector<float> hg=MakeInput(g_caif_gated_act_test_n,seed_g);
  std::vector<float> hu=MakeInput(g_caif_gated_act_test_n,seed_u);
  std::vector<float> ref(static_cast<size_t>(g_caif_gated_act_test_n));
  cpu_fn(hg.data(),hu.data(),ref.data(),g_caif_gated_act_test_n);

  CAIF_DeviceTensor gate=MakeFp32Device(hg,stream);
  CAIF_DeviceTensor up=MakeFp32Device(hu,stream);
  CAIF_DeviceTensor out=MakeFp32Like(gate,stream);

  ActT act;
  act.Forward(gate,up,out);
  const std::vector<float> got=ReadFp32(out,stream);
  return MaxAbsDiff(got.data(),ref.data(),g_caif_gated_act_test_n)<=g_caif_gated_act_test_fp32_tol;
}

template<typename ActT>
bool CAIF_GatedActivationsTests::RunGatedBackwardFD(CpuGatedFn_t cpu_fn,
                                                     const int32_t seed_g,
                                                     const int32_t seed_u)
{
  CAIF_CudaStream stream;
  const std::vector<float> hg=MakeInput(g_caif_gated_act_test_n,seed_g);
  const std::vector<float> hu=MakeInput(g_caif_gated_act_test_n,seed_u);
  std::vector<float> ones(static_cast<size_t>(g_caif_gated_act_test_n),1.0f);

  CAIF_DeviceTensor gate=MakeFp32Device(hg,stream);
  CAIF_DeviceTensor up=MakeFp32Device(hu,stream);
  CAIF_DeviceTensor go=MakeFp32Device(ones,stream);
  CAIF_DeviceTensor gg=MakeFp32Like(gate,stream);
  CAIF_DeviceTensor gu=MakeFp32Like(up,stream);

  ActT act;
  act.Backward(go,gate,up,gg,gu);
  const std::vector<float> got_gg=ReadFp32(gg,stream);
  const std::vector<float> got_gu=ReadFp32(gu,stream);

  bool ok=true;
  for(int32_t i=0;i<g_caif_gated_act_test_n;++i)
  {
    // Finite-difference reference: with grad_output==1 across all
    // elements, partial w.r.t. gate[i] is element-wise (cpu_fn changes
    // only at index i when only gate[i] is perturbed).
    std::vector<float> hg_p=hg;
    std::vector<float> hg_m=hg;
    hg_p[static_cast<size_t>(i)]+=g_caif_gated_act_test_finite_diff_h;
    hg_m[static_cast<size_t>(i)]-=g_caif_gated_act_test_finite_diff_h;
    std::vector<float> out_p(static_cast<size_t>(g_caif_gated_act_test_n));
    std::vector<float> out_m(static_cast<size_t>(g_caif_gated_act_test_n));
    cpu_fn(hg_p.data(),hu.data(),out_p.data(),g_caif_gated_act_test_n);
    cpu_fn(hg_m.data(),hu.data(),out_m.data(),g_caif_gated_act_test_n);
    const float fd_g=(out_p[static_cast<size_t>(i)]-out_m[static_cast<size_t>(i)])/
                     (2.0f*g_caif_gated_act_test_finite_diff_h);
    if(std::fabs(got_gg[static_cast<size_t>(i)]-fd_g)>g_caif_gated_act_test_grad_tol)
    {
      ok=false;
    }

    std::vector<float> hu_p=hu;
    std::vector<float> hu_m=hu;
    hu_p[static_cast<size_t>(i)]+=g_caif_gated_act_test_finite_diff_h;
    hu_m[static_cast<size_t>(i)]-=g_caif_gated_act_test_finite_diff_h;
    cpu_fn(hg.data(),hu_p.data(),out_p.data(),g_caif_gated_act_test_n);
    cpu_fn(hg.data(),hu_m.data(),out_m.data(),g_caif_gated_act_test_n);
    const float fd_u=(out_p[static_cast<size_t>(i)]-out_m[static_cast<size_t>(i)])/
                     (2.0f*g_caif_gated_act_test_finite_diff_h);
    if(std::fabs(got_gu[static_cast<size_t>(i)]-fd_u)>g_caif_gated_act_test_grad_tol)
    {
      ok=false;
    }
  }
  return ok;
}

template<typename StorageT>
bool CAIF_GatedActivationsTests::RunSwiGLUDtype(const CAIF_DataType::CAIF_DataType_e storage_dt,
                                                 const float tol)
{
  CAIF_CudaStream stream;
  constexpr int32_t seed_g=21;
  constexpr int32_t seed_u=22;
  const std::vector<float> hg=MakeInput(g_caif_gated_act_test_n,seed_g);
  const std::vector<float> hu=MakeInput(g_caif_gated_act_test_n,seed_u);
  std::vector<float> ref(static_cast<size_t>(g_caif_gated_act_test_n));
  CpuSwiGLU(hg.data(),hu.data(),ref.data(),g_caif_gated_act_test_n);

  CAIF_DeviceTensor g_fp32=MakeFp32Device(hg,stream);
  CAIF_DeviceTensor u_fp32=MakeFp32Device(hu,stream);
  CAIF_DeviceTensor g_dev=g_fp32.To(storage_dt);
  CAIF_DeviceTensor u_dev=u_fp32.To(storage_dt);
  CAIF_DeviceTensor o_dev=CAIF_DeviceTensor::Uninitialized(g_dev.Shape(),
                                                            stream,
                                                            storage_dt);
  CAIF_DeviceSwiGLUActivation<float,StorageT> act;
  act.Forward(g_dev,u_dev,o_dev);
  CAIF_DeviceTensor o_fp32=o_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
  const std::vector<float> got=ReadFp32(o_fp32,stream);
  return MaxAbsDiff(got.data(),ref.data(),g_caif_gated_act_test_n)<=tol;
}

void CAIF_GatedActivationsTests::RunAll()
{
  CAIF_TestHarness::Reset();
  ISE_Out::Out()<<"Gated Activations Tests\n"
                <<"=======================\n";

  typedef CAIF_DeviceSwiGLUActivation<float,float> SwiT;
  typedef CAIF_DeviceGeGLUActivation<float,float> GegT;
  typedef CAIF_DeviceReGLUActivation<float,float> RegT;
  typedef CAIF_DeviceGLUActivation<float,float> GluT;
  typedef CAIF_DeviceBilinearActivation<float,float> BilT;

  constexpr int32_t seed_swi_g=31;
  constexpr int32_t seed_swi_u=32;
  constexpr int32_t seed_geg_g=33;
  constexpr int32_t seed_geg_u=34;
  constexpr int32_t seed_reg_g=35;
  constexpr int32_t seed_reg_u=36;
  constexpr int32_t seed_glu_g=37;
  constexpr int32_t seed_glu_u=38;
  constexpr int32_t seed_bil_g=39;
  constexpr int32_t seed_bil_u=40;

  CAIF_TestHarness::Report("GatedAct::SwiGLU::Forward fp32",
                           RunGatedForwardParity<SwiT>(CpuSwiGLU,seed_swi_g,seed_swi_u));
  CAIF_TestHarness::Report("GatedAct::SwiGLU::Backward fp32",
                           RunGatedBackwardFD<SwiT>(CpuSwiGLU,seed_swi_g,seed_swi_u));

  CAIF_TestHarness::Report("GatedAct::GeGLU::Forward fp32",
                           RunGatedForwardParity<GegT>(CpuGeGLU,seed_geg_g,seed_geg_u));
  CAIF_TestHarness::Report("GatedAct::GeGLU::Backward fp32",
                           RunGatedBackwardFD<GegT>(CpuGeGLU,seed_geg_g,seed_geg_u));

  CAIF_TestHarness::Report("GatedAct::ReGLU::Forward fp32",
                           RunGatedForwardParity<RegT>(CpuReGLU,seed_reg_g,seed_reg_u));
  CAIF_TestHarness::Report("GatedAct::ReGLU::Backward fp32",
                           RunGatedBackwardFD<RegT>(CpuReGLU,seed_reg_g,seed_reg_u));

  CAIF_TestHarness::Report("GatedAct::GLU::Forward fp32",
                           RunGatedForwardParity<GluT>(CpuGLU,seed_glu_g,seed_glu_u));
  CAIF_TestHarness::Report("GatedAct::GLU::Backward fp32",
                           RunGatedBackwardFD<GluT>(CpuGLU,seed_glu_g,seed_glu_u));

  CAIF_TestHarness::Report("GatedAct::Bilinear::Forward fp32",
                           RunGatedForwardParity<BilT>(CpuBilinear,seed_bil_g,seed_bil_u));
  CAIF_TestHarness::Report("GatedAct::Bilinear::Backward fp32",
                           RunGatedBackwardFD<BilT>(CpuBilinear,seed_bil_g,seed_bil_u));

  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  CAIF_TestHarness::Report("GatedAct::SwiGLU device fp16",
                           RunSwiGLUDtype<__half>(Dtype_e::Float16,
                                                  g_caif_gated_act_test_fp16_tol));
  CAIF_TestHarness::Report("GatedAct::SwiGLU device bf16",
                           RunSwiGLUDtype<__nv_bfloat16>(Dtype_e::BFloat16,
                                                         g_caif_gated_act_test_bf16_tol));
}

#endif // USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_GatedActivationsTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"USE_CAIF_CUDA off — gated activation tests skipped.\n";
  return 0;
#endif
}
