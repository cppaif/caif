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
// Direct tests for CAIF_DeviceDenseLayer<ComputeT, StorageT> — the primary
// fully-connected layer. Exercises forward shape + activation
// correctness against a CPU MatMul reference and verifies the templated
// fp16 / bf16 storage cells produce numerically-close output to fp32.
//------------------------------------------------------------------------------

#include "caif_device_dense_layer.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
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

constexpr uint32_t g_batch=4;
constexpr uint32_t g_input=8;
constexpr uint32_t g_output=6;
constexpr float g_fp32_tol=2e-3f;
constexpr float g_fp16_tol=8e-2f;
constexpr float g_bf16_tol=2e-1f;
constexpr int32_t g_seed=4242;

void ReportResult(const char *test_name,const bool passed)
{
  CAIF_TestHarness::Report(test_name,passed);
}

std::vector<float> MakeData(const size_t n,const int32_t seed)
{
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    const float t=static_cast<float>((i+seed)%23)*0.13f;
    v[i]=(t-1.5f)*0.4f;
  }
  return v;
}

float MaxAbsDiff(const float *a,const float *b,const size_t n)
{
  float m=0.0f;
  for(size_t i=0;i<n;++i)
  {
    const float d=std::fabs(a[i]-b[i]);
    if(d>m)
    {
      m=d;
    }
  }
  return m;
}

#ifdef USE_CAIF_CUDA

// CPU reference: y = activation(x @ W + b). Matches the device layer's
// (input @ weights + bias) -> activation flow.
void CpuDenseForward(const float *x,
                     const float *w,
                     const float *b,
                     const CAIF_DeviceActivation_e act,
                     float *out,
                     const uint32_t batch,
                     const uint32_t in_dim,
                     const uint32_t out_dim,
                     const bool use_bias)
{
  CAIF_CpuMatMul::Apply(x,w,out,
                        static_cast<int>(batch),
                        static_cast<int>(in_dim),
                        static_cast<int>(out_dim));
  for(uint32_t r=0;r<batch;++r)
  {
    for(uint32_t c=0;c<out_dim;++c)
    {
      const size_t idx=static_cast<size_t>(r)*out_dim+c;
      if(use_bias==true)
      {
        out[idx]+=b[c];
      }
      const float v=out[idx];
      if(act==CAIF_DeviceActivation_e::ReLU)
      {
        if(v>0.0f)
        {
          out[idx]=v;
        }
        else
        {
          out[idx]=0.0f;
        }
      }
      else if(act==CAIF_DeviceActivation_e::GELU)
      {
        out[idx]=CAIF_CpuActivations::GELU(v);
      }
      // None / Linear: out[idx] unchanged.
    }
  }
}

// Build a fp32 device tensor from a host vector with the given shape.
CAIF_DeviceTensor MakeFp32Device(const std::vector<float> &data,
                                 const std::vector<uint32_t> &shape,
                                 CAIF_CudaStream &stream)
{
  return CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
}

void OverwriteWeights(CAIF_DeviceDenseLayer<float,float> &layer,
                      const std::vector<float> &w_host,
                      const std::vector<float> &b_host,
                      CAIF_CudaStream &stream)
{
  layer.Weights().CopyFromHost(w_host.data(),w_host.size());
  layer.Bias().CopyFromHost(b_host.data(),b_host.size());
  stream.Synchronize();
}

void TestDenseLinearFp32()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);

    CAIF_DeviceDenseLayer<float,float> layer(g_input,g_output,
                                              CAIF_DeviceActivation_e::None,
                                              stream);
    const std::vector<float> x_host=MakeData(g_batch*g_input,g_seed+1);
    const std::vector<float> w_host=MakeData(g_input*g_output,g_seed+2);
    const std::vector<float> b_host=MakeData(g_output,g_seed+3);
    OverwriteWeights(layer,w_host,b_host,stream);

    CAIF_DeviceTensor x=MakeFp32Device(x_host,{g_batch,g_input},stream);
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    CAIF_HostTensor host_y=y.ToHost();

    std::vector<float> ref(g_batch*g_output);
    CpuDenseForward(x_host.data(),w_host.data(),b_host.data(),
                    CAIF_DeviceActivation_e::None,ref.data(),
                    g_batch,g_input,g_output,true);
    const bool ok=MaxAbsDiff(host_y.Data(),ref.data(),g_batch*g_output)<=g_fp32_tol;
    ReportResult("DenseLayer::Linear fp32",ok);
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::Linear fp32")
}

void TestDenseReluFp32()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);

    CAIF_DeviceDenseLayer<float,float> layer(g_input,g_output,
                                              CAIF_DeviceActivation_e::ReLU,
                                              stream);
    const std::vector<float> x_host=MakeData(g_batch*g_input,g_seed+11);
    const std::vector<float> w_host=MakeData(g_input*g_output,g_seed+12);
    const std::vector<float> b_host=MakeData(g_output,g_seed+13);
    OverwriteWeights(layer,w_host,b_host,stream);

    CAIF_DeviceTensor x=MakeFp32Device(x_host,{g_batch,g_input},stream);
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    CAIF_HostTensor host_y=y.ToHost();

    std::vector<float> ref(g_batch*g_output);
    CpuDenseForward(x_host.data(),w_host.data(),b_host.data(),
                    CAIF_DeviceActivation_e::ReLU,ref.data(),
                    g_batch,g_input,g_output,true);
    const bool ok=MaxAbsDiff(host_y.Data(),ref.data(),g_batch*g_output)<=g_fp32_tol;
    ReportResult("DenseLayer::ReLU fp32",ok);
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::ReLU fp32")
}

void TestDenseBackwardFp32()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    CAIF_DeviceDenseLayer<float,float> layer(g_input,g_output,
                                              CAIF_DeviceActivation_e::None,
                                              stream);
    const std::vector<float> x_host=MakeData(g_batch*g_input,g_seed+21);
    const std::vector<float> w_host=MakeData(g_input*g_output,g_seed+22);
    const std::vector<float> b_host=MakeData(g_output,g_seed+23);
    OverwriteWeights(layer,w_host,b_host,stream);

    CAIF_DeviceTensor x=MakeFp32Device(x_host,{g_batch,g_input},stream);
    layer.ZeroGradients();
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    std::vector<float> grad_host(g_batch*g_output,1.0f);
    CAIF_DeviceTensor grad_y=MakeFp32Device(grad_host,{g_batch,g_output},stream);
    CAIF_DeviceTensor grad_x=layer.Backward(grad_y,ctx);

    // grad_x shape must match input.
    bool ok=grad_x.Shape().size()==2 &&
            grad_x.Shape()[0]==g_batch &&
            grad_x.Shape()[1]==g_input;
    // bias_grads should equal column-sum of grad_y when use_bias==true and
    // activation is None: each column gets g_batch contributions of 1.0.
    CAIF_HostTensor bias_g=layer.BiasGradients().ToHost();
    for(uint32_t c=0;c<g_output;++c)
    {
      const float expected=static_cast<float>(g_batch);
      if(std::fabs(bias_g.Data()[c]-expected)>g_fp32_tol)
      {
        ok=false;
      }
    }
    ReportResult("DenseLayer::Backward fp32 (bias-grad column sum)",ok);
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::Backward fp32")
}

template<typename StorageT>
bool RunDenseDtype(const CAIF_DataType::CAIF_DataType_e storage_dt,const float tol)
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  ctx.SetTraining(false);

  // Build the fp32 reference on a fresh fp32 layer first.
  CAIF_DeviceDenseLayer<float,float> ref_layer(g_input,g_output,
                                                CAIF_DeviceActivation_e::None,
                                                stream);
  const std::vector<float> x_host=MakeData(g_batch*g_input,g_seed+31);
  const std::vector<float> w_host=MakeData(g_input*g_output,g_seed+32);
  const std::vector<float> b_host=MakeData(g_output,g_seed+33);
  OverwriteWeights(ref_layer,w_host,b_host,stream);
  CAIF_DeviceTensor x_fp32=MakeFp32Device(x_host,{g_batch,g_input},stream);
  CAIF_DeviceTensor y_ref=ref_layer.Forward(x_fp32,ctx);
  CAIF_HostTensor host_ref=y_ref.ToHost();

  // Now templated cell on StorageT.
  CAIF_DeviceDenseLayer<float,StorageT> dev_layer(g_input,g_output,
                                                   CAIF_DeviceActivation_e::None,
                                                   stream);
  // Cast weights/bias to StorageT and copy the underlying bytes into the
  // dev_layer's internal tensors. We go via tensor.To() to convert.
  CAIF_DeviceTensor w_fp32=MakeFp32Device(w_host,{g_input,g_output},stream);
  CAIF_DeviceTensor b_fp32=MakeFp32Device(b_host,{g_output},stream);
  CAIF_DeviceTensor w_dev=w_fp32.To(storage_dt);
  CAIF_DeviceTensor b_dev=b_fp32.To(storage_dt);
  dev_layer.Weights().CopyFromHostRaw(w_dev.DeviceDataRaw(),
                                      w_dev.TotalElements()*sizeof(StorageT));
  // Reading device->device requires going through host; do a host trip.
  std::vector<uint8_t> w_bytes(w_dev.TotalElements()*sizeof(StorageT));
  std::vector<uint8_t> b_bytes(b_dev.TotalElements()*sizeof(StorageT));
  w_dev.CopyToHostRaw(w_bytes.data());
  b_dev.CopyToHostRaw(b_bytes.data());
  stream.Synchronize();
  dev_layer.Weights().CopyFromHostRaw(w_bytes.data(),w_bytes.size());
  dev_layer.Bias().CopyFromHostRaw(b_bytes.data(),b_bytes.size());
  stream.Synchronize();

  CAIF_DeviceTensor x_dev=x_fp32.To(storage_dt);
  CAIF_DeviceTensor y_dev=dev_layer.Forward(x_dev,ctx);
  CAIF_DeviceTensor y_dev_fp32=y_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_HostTensor host_dev=y_dev_fp32.ToHost();

  return MaxAbsDiff(host_dev.Data(),host_ref.Data(),g_batch*g_output)<=tol;
}

void TestDenseDtypeSweep()
{
  try
  {
    using Dtype_e=CAIF_DataType::CAIF_DataType_e;
    ReportResult("DenseLayer::fp16 storage",
                 (RunDenseDtype<__half>(Dtype_e::Float16,g_fp16_tol)));
    ReportResult("DenseLayer::bf16 storage",
                 (RunDenseDtype<__nv_bfloat16>(Dtype_e::BFloat16,g_bf16_tol)));
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::dtype-sweep")
}

#endif // USE_CAIF_CUDA

}//end anonymous namespace

int main()
{
  CAIF_TestHarness::Reset();
#ifdef USE_CAIF_CUDA
  std::cout<<"Dense Layer Tests\n";
  std::cout<<"=================\n";
  TestDenseLinearFp32();
  TestDenseReluFp32();
  TestDenseBackwardFp32();
  TestDenseDtypeSweep();
#else
  std::cout<<"USE_CAIF_CUDA off — dense layer tests skipped.\n";
#endif
  return CAIF_TestHarness::FinalExitCode();
}
