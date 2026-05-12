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
// Direct tests for the 8 pointwise activation device layers (ReLU, GELU,
// Sigmoid, Tanh, Swish, LeakyReLU, ELU, Linear). Each test exercises the
// templated forward + backward against an inline CPU reference. ReLU /
// GELU are also exercised across the full {fp32, fp16, bf16} storage
// dtype sweep to verify the templated kernels operate on the declared
// storage in DRAM.
//------------------------------------------------------------------------------

#include "caif_device_relu_activation.h"
#include "caif_device_gelu_activation.h"
#include "caif_device_sigmoid_activation.h"
#include "caif_device_tanh_activation.h"
#include "caif_device_swish_activation.h"
#include "caif_device_leaky_relu_activation.h"
#include "caif_device_elu_activation.h"
#include "caif_device_linear_activation.h"
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
constexpr float g_finite_diff_h=1e-3f;
constexpr float g_fp32_tol=1e-4f;
constexpr float g_fp16_tol=5e-3f;
constexpr float g_bf16_tol=2e-2f;
constexpr float g_leaky_alpha=0.01f;
constexpr float g_elu_alpha=1.0f;

void ReportResult(const char *test_name,const bool passed)
{
  CAIF_TestHarness::Report(test_name,passed);
}

bool FloatClose(const float a,const float b,const float tol)
{
  return std::fabs(a-b)<=tol;
}

std::vector<float> MakeInput(const int32_t n,const int32_t seed)
{
  std::vector<float> v(n);
  for(int32_t i=0;i<n;++i)
  {
    const float t=static_cast<float>((i+seed)%17)*0.31f;
    v[i]=(t-2.5f)*0.6f;
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

// CPU references for the 8 pointwise activations, plus per-element derivatives
// used by the backward parity tests.
float CpuRelu(const float x)
{
  if(x>0.0f)
  {
    return x;
  }
  return 0.0f;
}

float CpuReluGrad(const float x)
{
  if(x>0.0f)
  {
    return 1.0f;
  }
  return 0.0f;
}

float CpuSigmoid(const float x)
{
  return 1.0f/(1.0f+std::exp(-x));
}

float CpuSigmoidGrad(const float post)
{
  return post*(1.0f-post);
}

float CpuTanh(const float x)
{
  return std::tanh(x);
}

float CpuTanhGrad(const float post)
{
  return 1.0f-post*post;
}

float CpuSwishGrad(const float pre)
{
  const float sig=CpuSigmoid(pre);
  return sig+pre*sig*(1.0f-sig);
}

float CpuLeaky(const float x,const float alpha)
{
  if(x>0.0f)
  {
    return x;
  }
  return alpha*x;
}

float CpuLeakyGrad(const float x,const float alpha)
{
  if(x>0.0f)
  {
    return 1.0f;
  }
  return alpha;
}

float CpuElu(const float x,const float alpha)
{
  if(x>0.0f)
  {
    return x;
  }
  return alpha*(std::exp(x)-1.0f);
}

float CpuEluGrad(const float x,const float post,const float alpha)
{
  if(x>0.0f)
  {
    return 1.0f;
  }
  return post+alpha;
}

#ifdef USE_CAIF_CUDA

// Helpers: build a fp32 device tensor on stream from host data.
CAIF_DeviceTensor MakeFp32Device(const std::vector<float> &data,
                                 CAIF_CudaStream &stream)
{
  const std::vector<uint32_t> shape={static_cast<uint32_t>(data.size())};
  return CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
}

// Convenience: alloc a fp32 device tensor with the same shape on the
// caller's stream.
CAIF_DeviceTensor MakeFp32Like(const CAIF_DeviceTensor &x,CAIF_CudaStream &stream)
{
  return CAIF_DeviceTensor::Uninitialized(x.Shape(),
                                          stream,
                                          CAIF_DataType::CAIF_DataType_e::Float32);
}

// Read a fp32 device tensor back to host as a flat vector. Takes a stream
// because const tensor only exposes const Stream().
std::vector<float> ReadFp32(const CAIF_DeviceTensor &x,CAIF_CudaStream &stream)
{
  std::vector<float> out(x.TotalElements());
  x.CopyToHost(out.data());
  stream.Synchronize();
  return out;
}

//------------------------------------------------------------------------------
// FP32 forward + backward parity per activation
//------------------------------------------------------------------------------
template<typename ActT>
bool RunForwardParity(const std::vector<float> &expected,const int32_t seed)
{
  CAIF_CudaStream stream;
  std::vector<float> host_in=MakeInput(g_n,seed);
  CAIF_DeviceTensor x=MakeFp32Device(host_in,stream);
  CAIF_DeviceTensor y=MakeFp32Like(x,stream);

  ActT act;
  act.Forward(x,y);
  const std::vector<float> got=ReadFp32(y,stream);
  return MaxAbsDiff(got.data(),expected.data(),g_n)<=g_fp32_tol;
}

template<typename ActT>
bool RunBackwardParity(const std::vector<float> &grad_expected,
                       const std::vector<float> &input_host,
                       const std::vector<float> &post_host)
{
  CAIF_CudaStream stream;
  std::vector<float> ones(g_n,1.0f);
  CAIF_DeviceTensor pre=MakeFp32Device(input_host,stream);
  CAIF_DeviceTensor post=MakeFp32Device(post_host,stream);
  CAIF_DeviceTensor go=MakeFp32Device(ones,stream);
  CAIF_DeviceTensor gi=MakeFp32Like(pre,stream);

  ActT act;
  act.Backward(go,pre,post,gi);
  const std::vector<float> got=ReadFp32(gi,stream);
  return MaxAbsDiff(got.data(),grad_expected.data(),g_n)<=g_fp32_tol;
}

void TestReluFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,1);
    std::vector<float> ref(g_n);
    std::vector<float> grad_ref(g_n);
    for(int32_t i=0;i<g_n;++i)
    {
      ref[i]=CpuRelu(in[i]);
      grad_ref[i]=CpuReluGrad(in[i]);
    }
    const bool fwd=RunForwardParity<CAIF_DeviceReLUActivation<float,float>>(ref,1);
    const bool bwd=RunBackwardParity<CAIF_DeviceReLUActivation<float,float>>(grad_ref,
                                                                              in,
                                                                              ref);
    ReportResult("PointwiseAct::ReLU::Forward fp32",fwd);
    ReportResult("PointwiseAct::ReLU::Backward fp32",bwd);
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::ReLU::fp32")
}

void TestGeluFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,2);
    std::vector<float> ref(g_n);
    for(int32_t i=0;i<g_n;++i)
    {
      ref[i]=CAIF_CpuActivations::GELU(in[i]);
    }
    const bool fwd=RunForwardParity<CAIF_DeviceGELUActivation<float,float>>(ref,2);
    ReportResult("PointwiseAct::GELU::Forward fp32",fwd);

    // Backward parity via finite difference (no closed form on the CPU side
    // here matches the kernel's tanh-based GELU approximation).
    CAIF_CudaStream stream;
    std::vector<float> ones(g_n,1.0f);
    CAIF_DeviceTensor pre=MakeFp32Device(in,stream);
    CAIF_DeviceTensor post=MakeFp32Device(ref,stream);
    CAIF_DeviceTensor go=MakeFp32Device(ones,stream);
    CAIF_DeviceTensor gi=MakeFp32Like(pre,stream);
    CAIF_DeviceGELUActivation<float,float> act;
    act.Backward(go,pre,post,gi);
    const std::vector<float> got=ReadFp32(gi,stream);

    bool bwd=true;
    for(int32_t i=0;i<g_n;++i)
    {
      const float xp=in[i]+g_finite_diff_h;
      const float xm=in[i]-g_finite_diff_h;
      const float fd=(CAIF_CpuActivations::GELU(xp)-CAIF_CpuActivations::GELU(xm))/
                     (2.0f*g_finite_diff_h);
      if(FloatClose(got[i],fd,5e-3f)==false)
      {
        bwd=false;
      }
    }
    ReportResult("PointwiseAct::GELU::Backward fp32",bwd);
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::GELU::fp32")
}

void TestSigmoidFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,3);
    std::vector<float> ref(g_n);
    std::vector<float> grad_ref(g_n);
    for(int32_t i=0;i<g_n;++i)
    {
      ref[i]=CpuSigmoid(in[i]);
      grad_ref[i]=CpuSigmoidGrad(ref[i]);
    }
    using SigT=CAIF_DeviceSigmoidActivation<float,float>;
    ReportResult("PointwiseAct::Sigmoid::Forward fp32",
                 RunForwardParity<SigT>(ref,3));
    ReportResult("PointwiseAct::Sigmoid::Backward fp32",
                 RunBackwardParity<SigT>(grad_ref,in,ref));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::Sigmoid::fp32")
}

void TestTanhFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,4);
    std::vector<float> ref(g_n);
    std::vector<float> grad_ref(g_n);
    for(int32_t i=0;i<g_n;++i)
    {
      ref[i]=CpuTanh(in[i]);
      grad_ref[i]=CpuTanhGrad(ref[i]);
    }
    using TanhT=CAIF_DeviceTanhActivation<float,float>;
    ReportResult("PointwiseAct::Tanh::Forward fp32",RunForwardParity<TanhT>(ref,4));
    ReportResult("PointwiseAct::Tanh::Backward fp32",
                 RunBackwardParity<TanhT>(grad_ref,in,ref));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::Tanh::fp32")
}

void TestSwishFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,5);
    std::vector<float> ref(g_n);
    std::vector<float> grad_ref(g_n);
    for(int32_t i=0;i<g_n;++i)
    {
      ref[i]=CAIF_CpuActivations::Swish(in[i]);
      grad_ref[i]=CpuSwishGrad(in[i]);
    }
    using SwT=CAIF_DeviceSwishActivation<float,float>;
    ReportResult("PointwiseAct::Swish::Forward fp32",RunForwardParity<SwT>(ref,5));
    ReportResult("PointwiseAct::Swish::Backward fp32",
                 RunBackwardParity<SwT>(grad_ref,in,ref));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::Swish::fp32")
}

void TestLeakyReluFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,6);
    std::vector<float> ref(g_n);
    std::vector<float> grad_ref(g_n);
    for(int32_t i=0;i<g_n;++i)
    {
      ref[i]=CpuLeaky(in[i],g_leaky_alpha);
      grad_ref[i]=CpuLeakyGrad(in[i],g_leaky_alpha);
    }
    using LRT=CAIF_DeviceLeakyReLUActivation<float,float>;
    ReportResult("PointwiseAct::LeakyReLU::Forward fp32",RunForwardParity<LRT>(ref,6));
    ReportResult("PointwiseAct::LeakyReLU::Backward fp32",
                 RunBackwardParity<LRT>(grad_ref,in,ref));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::LeakyReLU::fp32")
}

void TestEluFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,7);
    std::vector<float> ref(g_n);
    std::vector<float> grad_ref(g_n);
    for(int32_t i=0;i<g_n;++i)
    {
      ref[i]=CpuElu(in[i],g_elu_alpha);
      grad_ref[i]=CpuEluGrad(in[i],ref[i],g_elu_alpha);
    }
    using ET=CAIF_DeviceELUActivation<float,float>;
    ReportResult("PointwiseAct::ELU::Forward fp32",RunForwardParity<ET>(ref,7));
    ReportResult("PointwiseAct::ELU::Backward fp32",
                 RunBackwardParity<ET>(grad_ref,in,ref));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::ELU::fp32")
}

void TestLinearFp32()
{
  try
  {
    const std::vector<float> in=MakeInput(g_n,8);
    std::vector<float> ones(g_n,1.0f);
    using LinT=CAIF_DeviceLinearActivation<float,float>;
    // Linear: f(x)=x, f'(x)=1; forward output should equal input.
    ReportResult("PointwiseAct::Linear::Forward fp32",
                 RunForwardParity<LinT>(in,8));
    ReportResult("PointwiseAct::Linear::Backward fp32",
                 RunBackwardParity<LinT>(ones,in,in));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::Linear::fp32")
}

//------------------------------------------------------------------------------
// dtype sweep — verify the templated cells on fp16 / bf16 storage produce
// numerically-close output to the fp32 reference (within dtype-appropriate
// tolerance). This is the contract that "every dtype cell runs its declared
// storage in DRAM" hinges on.
//------------------------------------------------------------------------------
template<typename StorageT>
bool RunReluDtypeSweep(const CAIF_DataType::CAIF_DataType_e storage_dt,const float tol)
{
  CAIF_CudaStream stream;
  std::vector<float> host_in=MakeInput(g_n,11);

  // fp32 reference.
  std::vector<float> ref(g_n);
  for(int32_t i=0;i<g_n;++i)
  {
    ref[i]=CpuRelu(host_in[i]);
  }

  // Cast input to StorageT, run templated cell, cast output back to fp32.
  CAIF_DeviceTensor x_fp32=MakeFp32Device(host_in,stream);
  CAIF_DeviceTensor x_dev=x_fp32.To(storage_dt);
  CAIF_DeviceTensor y_dev=CAIF_DeviceTensor::Uninitialized(x_dev.Shape(),
                                                            stream,
                                                            storage_dt);
  CAIF_DeviceReLUActivation<float,StorageT> act;
  act.Forward(x_dev,y_dev);
  CAIF_DeviceTensor y_fp32=y_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
  const std::vector<float> got=ReadFp32(y_fp32,stream);
  return MaxAbsDiff(got.data(),ref.data(),g_n)<=tol;
}

template<typename StorageT>
bool RunGeluDtypeSweep(const CAIF_DataType::CAIF_DataType_e storage_dt,const float tol)
{
  CAIF_CudaStream stream;
  std::vector<float> host_in=MakeInput(g_n,12);

  std::vector<float> ref(g_n);
  for(int32_t i=0;i<g_n;++i)
  {
    ref[i]=CAIF_CpuActivations::GELU(host_in[i]);
  }

  CAIF_DeviceTensor x_fp32=MakeFp32Device(host_in,stream);
  CAIF_DeviceTensor x_dev=x_fp32.To(storage_dt);
  CAIF_DeviceTensor y_dev=CAIF_DeviceTensor::Uninitialized(x_dev.Shape(),
                                                            stream,
                                                            storage_dt);
  CAIF_DeviceGELUActivation<float,StorageT> act;
  act.Forward(x_dev,y_dev);
  CAIF_DeviceTensor y_fp32=y_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
  const std::vector<float> got=ReadFp32(y_fp32,stream);
  return MaxAbsDiff(got.data(),ref.data(),g_n)<=tol;
}

void TestReluDtypeSweep()
{
  try
  {
    using Dtype_e=CAIF_DataType::CAIF_DataType_e;
    ReportResult("PointwiseAct::ReLU device fp16",
                 (RunReluDtypeSweep<__half>(Dtype_e::Float16,g_fp16_tol)));
    ReportResult("PointwiseAct::ReLU device bf16",
                 (RunReluDtypeSweep<__nv_bfloat16>(Dtype_e::BFloat16,g_bf16_tol)));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::ReLU::dtype-sweep")
}

void TestGeluDtypeSweep()
{
  try
  {
    using Dtype_e=CAIF_DataType::CAIF_DataType_e;
    ReportResult("PointwiseAct::GELU device fp16",
                 (RunGeluDtypeSweep<__half>(Dtype_e::Float16,g_fp16_tol)));
    ReportResult("PointwiseAct::GELU device bf16",
                 (RunGeluDtypeSweep<__nv_bfloat16>(Dtype_e::BFloat16,g_bf16_tol)));
  }
  CAIF_TEST_CATCH_BLOCK("PointwiseAct::GELU::dtype-sweep")
}

#endif // USE_CAIF_CUDA

}//end anonymous namespace

int main()
{
  CAIF_TestHarness::Reset();
#ifdef USE_CAIF_CUDA
  std::cout<<"Pointwise Activations Tests\n";
  std::cout<<"===========================\n";

  TestReluFp32();
  TestGeluFp32();
  TestSigmoidFp32();
  TestTanhFp32();
  TestSwishFp32();
  TestLeakyReluFp32();
  TestEluFp32();
  TestLinearFp32();

  TestReluDtypeSweep();
  TestGeluDtypeSweep();
#else
  std::cout<<"USE_CAIF_CUDA off — pointwise activation tests skipped.\n";
#endif
  return CAIF_TestHarness::FinalExitCode();
}
