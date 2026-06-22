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
// Test: CAIF_OffloadedAdam parity vs. GPU-resident Adam kernel.
//
// Parity test: run AdamUpdate kernel directly with GPU-resident m/v
// (matching CAIF_AdamOptimizer's path) and compare against an
// equivalent step performed via CAIF_OffloadedAdam's prefetch / kernel
// / writeback cycle. Same hyperparameters, same input grad, same
// initial m/v/param. Result must be bit-equivalent within float
// noise (host pinned memory plus host->device->host round-trip is
// just a memcpy chain — no precision loss expected).
//
// Single-parameter test using CAIF_Ops::AdamUpdate directly so we
// don't have to construct a full CAIF_DeviceNetwork just to run one
// optimizer step.
//------------------------------------------------------------------------------
#include "caif_offloaded_adam.h"
#include "caif_adam_optimizer.h"
#include "caif_device_tensor.h"
#include "caif_device_network.h"
#include "caif_device_layer.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_test_harness.h"
#include "caif_ops.h"
#include "caif_data_type.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

constexpr size_t g_caif_offadam_test_n=32u;
constexpr float g_caif_offadam_test_param_scale=0.01f;
constexpr float g_caif_offadam_test_param_bias=-0.16f;
constexpr float g_caif_offadam_test_grad_scale=0.001f;
constexpr float g_caif_offadam_test_grad_bias=-0.016f;
constexpr float g_caif_offadam_test_lr=1e-3f;
constexpr float g_caif_offadam_test_beta1=0.9f;
constexpr float g_caif_offadam_test_beta2=0.999f;
constexpr float g_caif_offadam_test_eps=1e-8f;
constexpr float g_caif_offadam_test_wd=0.01f;
constexpr int g_caif_offadam_test_step=1;
constexpr float g_caif_offadam_test_tol=1e-7f;
constexpr int g_caif_offadam_multistep_steps=5;
constexpr float g_caif_offadam_multistep_tol=1e-5f;

//------------------------------------------------------------------------------
// Test-only layer: two fixed fp32 parameter tensors + grads, so the offloaded
// optimizer's per-parameter prefetch / kernel / async-writeback loop runs over
// more than one tensor across steps.
//------------------------------------------------------------------------------
class CAIF_OffAdamLayer:public CAIF_DeviceLayer
{
  public:
    explicit CAIF_OffAdamLayer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                        _p0(MakeT(stream,{6},0.5f,0.01f)),
                                                        _p1(MakeT(stream,{4,3},0.3f,0.02f)),
                                                        _g0(MakeT(stream,{6},0.1f,-0.001f)),
                                                        _g1(MakeT(stream,{4,3},0.05f,0.002f))
    {
    }
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_OffAdamLayer::ForwardImpl not used in this test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_OffAdamLayer::BackwardImpl not used in this test");
    }
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Dense_e;
    }
    void ZeroGradients()override
    {
      GradientTensor(0).Fill(0.0f);
      GradientTensor(1).Fill(0.0f);
    }
    size_t ParameterTensorCount()const override
    {
      return 2;
    }
    CAIF_DeviceTensor &ParameterTensor(size_t i)override
    {
      if(i==0)
      {
        return _p0;
      }
      return _p1;
    }
    const CAIF_DeviceTensor &ParameterTensor(size_t i)const override
    {
      if(i==0)
      {
        return _p0;
      }
      return _p1;
    }
    CAIF_DeviceTensor &GradientTensor(size_t i)override
    {
      if(i==0)
      {
        return _g0;
      }
      return _g1;
    }
    const CAIF_DeviceTensor &GradientTensor(size_t i)const override
    {
      if(i==0)
      {
        return _g0;
      }
      return _g1;
    }
    size_t TotalParameterCount()const override
    {
      return ParameterTensor(0).TotalElements()+ParameterTensor(1).TotalElements();
    }
    CAIF_DataType::CAIF_DataType_e RuntimeStorageDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }
    CAIF_DataType::CAIF_DataType_e RuntimeComputeDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }
    std::string Description()const override
    {
      return "CAIF_OffAdamLayer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"p0",prefix+"p1"};
    }

  private:
    static CAIF_DeviceTensor MakeT(CAIF_CudaStream &stream,
                                   const std::vector<uint32_t> &shape,
                                   const float base,
                                   const float step)
    {
      CAIF_DeviceTensor t=CAIF_DeviceTensor::Zeros(shape,stream,CAIF_DataType::CAIF_DataType_e::Float32);
      const size_t n=t.TotalElements();
      std::vector<float> host(n);
      for(size_t i=0;i<n;++i)
      {
        host[i]=base+step*static_cast<float>(i);
      }
      t.CopyFromHost(host.data(),n);
      return t;
    }

    CAIF_DeviceTensor _p0;
    CAIF_DeviceTensor _p1;
    CAIF_DeviceTensor _g0;
    CAIF_DeviceTensor _g1;
};

class CAIF_OffloadedAdamTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestParityOneStep();
    static std::vector<float> ReadParams(CAIF_DeviceNetwork &net);
    static std::vector<float> RunSteps(const bool offloaded);
    static void TestParityMultiStep();
};

void CAIF_OffloadedAdamTests::TestParityOneStep()
{
  try
  {
    CAIF_CudaStream stream;
    const std::vector<uint32_t> shape={static_cast<uint32_t>(g_caif_offadam_test_n)};

    std::vector<float> param_init(g_caif_offadam_test_n,0.0f);
    std::vector<float> grad_seed(g_caif_offadam_test_n,0.0f);
    for(size_t i=0;i<g_caif_offadam_test_n;++i)
    {
      param_init[i]=static_cast<float>(i)*g_caif_offadam_test_param_scale
                    +g_caif_offadam_test_param_bias;
      grad_seed[i]=static_cast<float>(i)*g_caif_offadam_test_grad_scale
                   +g_caif_offadam_test_grad_bias;
    }

    CAIF_DeviceTensor param_a=CAIF_DeviceTensor::Uninitialized(
                                shape,stream,
                                CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor param_b=CAIF_DeviceTensor::Uninitialized(
                                shape,stream,
                                CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor grad_a=CAIF_DeviceTensor::Uninitialized(
                                shape,stream,
                                CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor grad_b=CAIF_DeviceTensor::Uninitialized(
                                shape,stream,
                                CAIF_DataType::CAIF_DataType_e::Float32);
    param_a.CopyFromHostRaw(param_init.data(),
                            g_caif_offadam_test_n*sizeof(float));
    param_b.CopyFromHostRaw(param_init.data(),
                            g_caif_offadam_test_n*sizeof(float));
    grad_a.CopyFromHostRaw(grad_seed.data(),
                           g_caif_offadam_test_n*sizeof(float));
    grad_b.CopyFromHostRaw(grad_seed.data(),
                           g_caif_offadam_test_n*sizeof(float));

    // Path A — GPU-resident m/v, run AdamUpdate directly.
    CAIF_DeviceTensor m_a=CAIF_DeviceTensor::Zeros(shape,stream,
                            CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor v_a=CAIF_DeviceTensor::Zeros(shape,stream,
                            CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::AdamUpdate(param_a,grad_a,m_a,v_a,
                         g_caif_offadam_test_lr,
                         g_caif_offadam_test_beta1,
                         g_caif_offadam_test_beta2,
                         g_caif_offadam_test_eps,
                         g_caif_offadam_test_wd,
                         g_caif_offadam_test_step);

    // Path B — host pinned m/v, prefetch + AdamUpdate + writeback.
    CAIF_HostPinnedTensor host_m(shape,CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_HostPinnedTensor host_v(shape,CAIF_DataType::CAIF_DataType_e::Float32);
    float *pm=static_cast<float*>(host_m.HostPtr());
    float *pv=static_cast<float*>(host_v.HostPtr());
    for(size_t i=0;i<g_caif_offadam_test_n;++i)
    {
      pm[i]=0.0f;
      pv[i]=0.0f;
    }
    CAIF_DeviceTensor m_b=host_m.PrefetchToDevice(stream);
    CAIF_DeviceTensor v_b=host_v.PrefetchToDevice(stream);
    CAIF_Ops::AdamUpdate(param_b,grad_b,m_b,v_b,
                         g_caif_offadam_test_lr,
                         g_caif_offadam_test_beta1,
                         g_caif_offadam_test_beta2,
                         g_caif_offadam_test_eps,
                         g_caif_offadam_test_wd,
                         g_caif_offadam_test_step);
    host_m.CopyFromDevice(m_b);
    host_v.CopyFromDevice(v_b);

    std::vector<float> param_a_host(g_caif_offadam_test_n,0.0f);
    std::vector<float> param_b_host(g_caif_offadam_test_n,0.0f);
    param_a.CopyToHost(param_a_host.data());
    param_b.CopyToHost(param_b_host.data());

    bool passed=true;
    for(size_t i=0;i<g_caif_offadam_test_n;++i)
    {
      if(CAIF_TestHarness::FloatEqual(param_a_host[i],
                                      param_b_host[i],
                                      g_caif_offadam_test_tol)==false)
      {
        ISE_Out::ErrLog()<<"  param mismatch at i="
                         <<i
                         <<": gpu="
                         <<param_a_host[i]
                         <<" offloaded="
                         <<param_b_host[i]
                         <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("OffloadedAdam::ParityOneStep_vs_GpuResidentKernel",passed);
  }
  CAIF_TEST_CATCH_BLOCK("OffloadedAdam::ParityOneStep_vs_GpuResidentKernel")
}

std::vector<float> CAIF_OffloadedAdamTests::ReadParams(CAIF_DeviceNetwork &net)
{
  std::vector<float> out;
  CAIF_DeviceLayer &layer=net.Layer(0);
  for(size_t i=0;i<layer.ParameterTensorCount();++i)
  {
    const CAIF_DeviceTensor &p=layer.ParameterTensor(i);
    const size_t n=p.TotalElements();
    std::vector<float> host(n);
    p.CopyToHost(host.data());
    for(size_t j=0;j<n;++j)
    {
      out.push_back(host[j]);
    }
  }
  return out;
}

std::vector<float> CAIF_OffloadedAdamTests::RunSteps(const bool offloaded)
{
  CAIF_CudaStream stream;
  CAIF_DeviceNetwork net(stream);
  net.AddLayer(std::make_unique<CAIF_OffAdamLayer>(stream));

  std::unique_ptr<CAIF_Optimizer> opt;
  if(offloaded==true)
  {
    opt=std::make_unique<CAIF_OffloadedAdam>(g_caif_offadam_test_lr,
                                             g_caif_offadam_test_beta1,
                                             g_caif_offadam_test_beta2,
                                             g_caif_offadam_test_eps,
                                             g_caif_offadam_test_wd,
                                             stream);
  }
  else
  {
    opt=std::make_unique<CAIF_AdamOptimizer>(g_caif_offadam_test_lr,
                                             g_caif_offadam_test_beta1,
                                             g_caif_offadam_test_beta2,
                                             g_caif_offadam_test_eps,
                                             g_caif_offadam_test_wd,
                                             stream);
  }
  opt->Initialize(net);
  for(int k=0;k<g_caif_offadam_multistep_steps;++k)
  {
    opt->Step(net);
  }
  return ReadParams(net);
}

void CAIF_OffloadedAdamTests::TestParityMultiStep()
{
  try
  {
    const std::vector<float> resident=RunSteps(false);
    const std::vector<float> offloaded=RunSteps(true);

    bool passed=(resident.size()==offloaded.size());
    for(size_t i=0;passed==true&&i<resident.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(resident[i],
                                      offloaded[i],
                                      g_caif_offadam_multistep_tol)==false)
      {
        ISE_Out::ErrLog()<<"  multistep param mismatch at i="
                         <<i
                         <<": resident="
                         <<resident[i]
                         <<" offloaded="
                         <<offloaded[i]
                         <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("OffloadedAdam::ParityMultiStep_vs_GpuResident",passed);
  }
  CAIF_TEST_CATCH_BLOCK("OffloadedAdam::ParityMultiStep_vs_GpuResident")
}

void CAIF_OffloadedAdamTests::RunAll()
{
  CAIF_TestHarness::Reset();
  TestParityOneStep();
  TestParityMultiStep();
}

}//end instance namespace

int main(int /*argc*/,char ** /*argv*/)
{
  instance::CAIF_OffloadedAdamTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
