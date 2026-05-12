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
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

#ifdef USE_CAIF_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

using namespace instance;

namespace
{

constexpr int32_t g_n=64;
constexpr float g_fp32_tol=2e-4f;
constexpr float g_fp16_tol=8e-3f;
constexpr float g_bf16_tol=3e-2f;
constexpr float g_finite_diff_h=1e-3f;
constexpr float g_grad_tol=8e-3f;

void ReportResult(const char *test_name,const bool passed)
{
  CAIF_TestHarness::Report(test_name,passed);
}

std::vector<float> MakeInput(const int32_t n,const int32_t seed)
{
  std::vector<float> v(n);
  for(int32_t i=0;i<n;++i)
  {
    const float t=static_cast<float>((i*7+seed)%19)*0.27f;
    v[i]=(t-2.5f)*0.55f;
  }
  return v;
}

float MaxAbsDiff(const float *a,const float *b,const int32_t n)
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

float CpuSigmoid(const float x)
{
  return 1.0f/(1.0f+std::exp(-x));
}

float CpuRelu(const float x)
{
  if(x>0.0f)
  {
    return x;
  }
  return 0.0f;
}

#ifdef USE_CAIF_CUDA

CAIF_DeviceTensor MakeFp32Device(const std::vector<float> &data,
                                 CAIF_CudaStream &stream)
{
  const std::vector<uint32_t> shape={static_cast<uint32_t>(data.size())};
  return CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
}

CAIF_DeviceTensor MakeFp32Like(const CAIF_DeviceTensor &x,CAIF_CudaStream &stream)
{
  return CAIF_DeviceTensor::Uninitialized(x.Shape(),
                                          stream,
                                          CAIF_DataType::CAIF_DataType_e::Float32);
}

std::vector<float> ReadFp32(const CAIF_DeviceTensor &x,CAIF_CudaStream &stream)
{
  std::vector<float> out(x.TotalElements());
  x.CopyToHost(out.data());
  stream.Synchronize();
  return out;
}

// CPU references — gated activation forward functions matching the kernel
// formulations.
void CpuSwiGLU(const float *gate,const float *up,float *out,const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CAIF_CpuActivations::Swish(gate[i])*up[i];
  }
}

void CpuGeGLU(const float *gate,const float *up,float *out,const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CAIF_CpuActivations::GELU(gate[i])*up[i];
  }
}

void CpuReGLU(const float *gate,const float *up,float *out,const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CpuRelu(gate[i])*up[i];
  }
}

void CpuGLU(const float *gate,const float *up,float *out,const int32_t n)
{
  // GLU = sigmoid(gate) * up.
  for(int32_t i=0;i<n;++i)
  {
    out[i]=CpuSigmoid(gate[i])*up[i];
  }
}

void CpuBilinear(const float *gate,const float *up,float *out,const int32_t n)
{
  for(int32_t i=0;i<n;++i)
  {
    out[i]=gate[i]*up[i];
  }
}

template<typename ActT,typename CpuFn>
bool RunGatedForwardParity(CpuFn cpu_fn,const int32_t seed_g,const int32_t seed_u)
{
  CAIF_CudaStream stream;
  std::vector<float> hg=MakeInput(g_n,seed_g);
  std::vector<float> hu=MakeInput(g_n,seed_u);
  std::vector<float> ref(g_n);
  cpu_fn(hg.data(),hu.data(),ref.data(),g_n);

  CAIF_DeviceTensor gate=MakeFp32Device(hg,stream);
  CAIF_DeviceTensor up=MakeFp32Device(hu,stream);
  CAIF_DeviceTensor out=MakeFp32Like(gate,stream);

  ActT act;
  act.Forward(gate,up,out);
  const std::vector<float> got=ReadFp32(out,stream);
  return MaxAbsDiff(got.data(),ref.data(),g_n)<=g_fp32_tol;
}

template<typename ActT,typename CpuFn>
bool RunGatedBackwardFD(CpuFn cpu_fn,const int32_t seed_g,const int32_t seed_u)
{
  CAIF_CudaStream stream;
  const std::vector<float> hg=MakeInput(g_n,seed_g);
  const std::vector<float> hu=MakeInput(g_n,seed_u);
  std::vector<float> ones(g_n,1.0f);

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
  for(int32_t i=0;i<g_n;++i)
  {
    // Finite-difference reference: with grad_output==1 across all
    // elements, partial w.r.t. gate[i] is element-wise (cpu_fn changes
    // only at index i when only gate[i] is perturbed).
    std::vector<float> hg_p=hg;
    std::vector<float> hg_m=hg;
    hg_p[i]+=g_finite_diff_h;
    hg_m[i]-=g_finite_diff_h;
    std::vector<float> out_p(g_n);
    std::vector<float> out_m(g_n);
    cpu_fn(hg_p.data(),hu.data(),out_p.data(),g_n);
    cpu_fn(hg_m.data(),hu.data(),out_m.data(),g_n);
    const float fd_g=(out_p[i]-out_m[i])/(2.0f*g_finite_diff_h);
    if(std::fabs(got_gg[i]-fd_g)>g_grad_tol)
    {
      ok=false;
    }

    std::vector<float> hu_p=hu;
    std::vector<float> hu_m=hu;
    hu_p[i]+=g_finite_diff_h;
    hu_m[i]-=g_finite_diff_h;
    cpu_fn(hg.data(),hu_p.data(),out_p.data(),g_n);
    cpu_fn(hg.data(),hu_m.data(),out_m.data(),g_n);
    const float fd_u=(out_p[i]-out_m[i])/(2.0f*g_finite_diff_h);
    if(std::fabs(got_gu[i]-fd_u)>g_grad_tol)
    {
      ok=false;
    }
  }
  return ok;
}

template<typename StorageT>
bool RunSwiGLUDtype(const CAIF_DataType::CAIF_DataType_e storage_dt,const float tol)
{
  CAIF_CudaStream stream;
  const std::vector<float> hg=MakeInput(g_n,21);
  const std::vector<float> hu=MakeInput(g_n,22);
  std::vector<float> ref(g_n);
  CpuSwiGLU(hg.data(),hu.data(),ref.data(),g_n);

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
  return MaxAbsDiff(got.data(),ref.data(),g_n)<=tol;
}

#endif // USE_CAIF_CUDA

}//end anonymous namespace

int main()
{
  CAIF_TestHarness::Reset();
#ifdef USE_CAIF_CUDA
  std::cout<<"Gated Activations Tests\n";
  std::cout<<"=======================\n";

  using SwiT=CAIF_DeviceSwiGLUActivation<float,float>;
  using GegT=CAIF_DeviceGeGLUActivation<float,float>;
  using RegT=CAIF_DeviceReGLUActivation<float,float>;
  using GluT=CAIF_DeviceGLUActivation<float,float>;
  using BilT=CAIF_DeviceBilinearActivation<float,float>;

  ReportResult("GatedAct::SwiGLU::Forward fp32",
               RunGatedForwardParity<SwiT>(CpuSwiGLU,31,32));
  ReportResult("GatedAct::SwiGLU::Backward fp32",
               RunGatedBackwardFD<SwiT>(CpuSwiGLU,31,32));

  ReportResult("GatedAct::GeGLU::Forward fp32",
               RunGatedForwardParity<GegT>(CpuGeGLU,33,34));
  ReportResult("GatedAct::GeGLU::Backward fp32",
               RunGatedBackwardFD<GegT>(CpuGeGLU,33,34));

  ReportResult("GatedAct::ReGLU::Forward fp32",
               RunGatedForwardParity<RegT>(CpuReGLU,35,36));
  ReportResult("GatedAct::ReGLU::Backward fp32",
               RunGatedBackwardFD<RegT>(CpuReGLU,35,36));

  ReportResult("GatedAct::GLU::Forward fp32",
               RunGatedForwardParity<GluT>(CpuGLU,37,38));
  ReportResult("GatedAct::GLU::Backward fp32",
               RunGatedBackwardFD<GluT>(CpuGLU,37,38));

  ReportResult("GatedAct::Bilinear::Forward fp32",
               RunGatedForwardParity<BilT>(CpuBilinear,39,40));
  ReportResult("GatedAct::Bilinear::Backward fp32",
               RunGatedBackwardFD<BilT>(CpuBilinear,39,40));

  using Dtype_e=CAIF_DataType::CAIF_DataType_e;
  ReportResult("GatedAct::SwiGLU device fp16",
               (RunSwiGLUDtype<__half>(Dtype_e::Float16,g_fp16_tol)));
  ReportResult("GatedAct::SwiGLU device bf16",
               (RunSwiGLUDtype<__nv_bfloat16>(Dtype_e::BFloat16,g_bf16_tol)));
#else
  std::cout<<"USE_CAIF_CUDA off — gated activation tests skipped.\n";
#endif
  return CAIF_TestHarness::FinalExitCode();
}
