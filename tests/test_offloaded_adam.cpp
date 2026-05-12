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

#include "caif_offloaded_adam.h"
#include "caif_adam_optimizer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_test_harness.h"
#include "caif_ops.h"

#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace instance;

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

static void TestParityOneStep()
{
  const char *test_name="OffloadedAdam::ParityOneStep_vs_GpuResidentKernel";
  try
  {
    CAIF_CudaStream stream;
    const std::vector<uint32_t> shape={32u};
    const size_t n=32u;

    std::vector<float> param_init(n,0.0f);
    std::vector<float> grad_seed(n,0.0f);
    for(size_t i=0;i<n;++i)
    {
      param_init[i]=static_cast<float>(i)*0.01f-0.16f;
      grad_seed[i]=static_cast<float>(i)*0.001f-0.016f;
    }

    const float lr=1e-3f;
    const float beta1=0.9f;
    const float beta2=0.999f;
    const float eps=1e-8f;
    const float wd=0.01f;
    const int step=1;

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
    param_a.CopyFromHostRaw(param_init.data(),n*sizeof(float));
    param_b.CopyFromHostRaw(param_init.data(),n*sizeof(float));
    grad_a.CopyFromHostRaw(grad_seed.data(),n*sizeof(float));
    grad_b.CopyFromHostRaw(grad_seed.data(),n*sizeof(float));

    // Path A — GPU-resident m/v, run AdamUpdate directly.
    CAIF_DeviceTensor m_a=CAIF_DeviceTensor::Zeros(shape,stream,
                            CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor v_a=CAIF_DeviceTensor::Zeros(shape,stream,
                            CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::AdamUpdate(param_a,grad_a,m_a,v_a,lr,beta1,beta2,eps,wd,step);

    // Path B — host pinned m/v, prefetch + AdamUpdate + writeback.
    CAIF_HostPinnedTensor host_m(shape,CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_HostPinnedTensor host_v(shape,CAIF_DataType::CAIF_DataType_e::Float32);
    {
      float *p=static_cast<float*>(host_m.HostPtr());
      for(size_t i=0;i<n;++i){p[i]=0.0f;}
    }
    {
      float *p=static_cast<float*>(host_v.HostPtr());
      for(size_t i=0;i<n;++i){p[i]=0.0f;}
    }
    CAIF_DeviceTensor m_b=host_m.PrefetchToDevice(stream);
    CAIF_DeviceTensor v_b=host_v.PrefetchToDevice(stream);
    CAIF_Ops::AdamUpdate(param_b,grad_b,m_b,v_b,lr,beta1,beta2,eps,wd,step);
    host_m.CopyFromDevice(m_b);
    host_v.CopyFromDevice(v_b);

    std::vector<float> param_a_host(n,0.0f);
    std::vector<float> param_b_host(n,0.0f);
    param_a.CopyToHost(param_a_host.data());
    param_b.CopyToHost(param_b_host.data());

    bool passed=true;
    for(size_t i=0;i<n;++i)
    {
      if(CAIF_TestHarness::FloatEqual(param_a_host[i],param_b_host[i],1e-7f)==false)
      {
        ISE_Out::ErrLog()<<"  param mismatch at i="
                         <<i
                         <<": gpu="
                         <<param_a_host[i]
                         <<" offloaded="
                         <<param_b_host[i]
                         <<std::endl;
        passed=false;
      }
    }
    CAIF_TestHarness::Report(test_name,passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name)
}

int main(int /*argc*/,char ** /*argv*/)
{
  CAIF_TestHarness::Reset();
  TestParityOneStep();
  return CAIF_TestHarness::FinalExitCode();
}