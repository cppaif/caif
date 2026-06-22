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

constexpr uint32_t g_caif_dense_test_batch=4;
constexpr uint32_t g_caif_dense_test_input=8;
constexpr uint32_t g_caif_dense_test_output=6;
constexpr float g_caif_dense_test_fp32_tol=2e-3f;
constexpr float g_caif_dense_test_fp16_tol=8e-2f;
constexpr float g_caif_dense_test_bf16_tol=2e-1f;
constexpr int32_t g_caif_dense_test_seed=4242;

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Dense layer forward + backward correctness tests.
//------------------------------------------------------------------------------
class CAIF_DenseLayerTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeData(const size_t n,const int32_t seed);
    static float MaxAbsDiff(const float *a,const float *b,const size_t n);

    // CPU reference: y = activation(x @ W + b). Matches the device layer's
    // (input @ weights + bias) -> activation flow.
    static void CpuDenseForward(const float *x,
                                const float *w,
                                const float *b,
                                const CAIF_DeviceActivation::CAIF_DeviceActivation_e act,
                                float *out,
                                const uint32_t batch,
                                const uint32_t in_dim,
                                const uint32_t out_dim,
                                const bool use_bias);

    static CAIF_DeviceTensor MakeFp32Device(const std::vector<float> &data,
                                            const std::vector<uint32_t> &shape,
                                            CAIF_CudaStream &stream);

    static void OverwriteWeights(CAIF_DeviceDenseLayer<float,float> &layer,
                                 const std::vector<float> &w_host,
                                 const std::vector<float> &b_host,
                                 CAIF_CudaStream &stream);

    template<typename StorageT>
    static bool RunDenseDtype(const CAIF_DataType::CAIF_DataType_e storage_dt,
                              const float tol);

    static void TestDenseLinearFp32();
    static void TestDenseReluFp32();
    static void TestDenseBackwardFp32();
    static void TestDenseDtypeSweep();
};

std::vector<float> CAIF_DenseLayerTests::MakeData(const size_t n,const int32_t seed)
{
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    const float t=static_cast<float>((i+static_cast<size_t>(seed))%23)*0.13f;
    v[i]=(t-1.5f)*0.4f;
  }
  return v;
}

float CAIF_DenseLayerTests::MaxAbsDiff(const float *a,const float *b,const size_t n)
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

void CAIF_DenseLayerTests::CpuDenseForward(const float *x,
                                           const float *w,
                                           const float *b,
                                           const CAIF_DeviceActivation::CAIF_DeviceActivation_e act,
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
      if(act==CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU)
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
      else if(act==CAIF_DeviceActivation::CAIF_DeviceActivation_e::GELU)
      {
        out[idx]=CAIF_CpuActivations::GELU(v);
      }
      // None / Linear: out[idx] unchanged.
    }
  }
}

CAIF_DeviceTensor CAIF_DenseLayerTests::MakeFp32Device(const std::vector<float> &data,
                                                       const std::vector<uint32_t> &shape,
                                                       CAIF_CudaStream &stream)
{
  return CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
}

void CAIF_DenseLayerTests::OverwriteWeights(CAIF_DeviceDenseLayer<float,float> &layer,
                                            const std::vector<float> &w_host,
                                            const std::vector<float> &b_host,
                                            CAIF_CudaStream &stream)
{
  layer.Weights().CopyFromHost(w_host.data(),w_host.size());
  layer.Bias().CopyFromHost(b_host.data(),b_host.size());
  stream.Synchronize();
}

void CAIF_DenseLayerTests::TestDenseLinearFp32()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);

    CAIF_DeviceDenseLayer<float,float> layer(g_caif_dense_test_input,
                                             g_caif_dense_test_output,
                                             CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,
                                             stream);
    const std::vector<float> x_host=MakeData(g_caif_dense_test_batch*g_caif_dense_test_input,
                                             g_caif_dense_test_seed+1);
    const std::vector<float> w_host=MakeData(g_caif_dense_test_input*g_caif_dense_test_output,
                                             g_caif_dense_test_seed+2);
    const std::vector<float> b_host=MakeData(g_caif_dense_test_output,g_caif_dense_test_seed+3);
    OverwriteWeights(layer,w_host,b_host,stream);

    CAIF_DeviceTensor x=MakeFp32Device(x_host,
                                       {g_caif_dense_test_batch,g_caif_dense_test_input},
                                       stream);
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    CAIF_HostTensor host_y=y.ToHost();

    std::vector<float> ref(g_caif_dense_test_batch*g_caif_dense_test_output);
    CpuDenseForward(x_host.data(),w_host.data(),b_host.data(),
                    CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,ref.data(),
                    g_caif_dense_test_batch,g_caif_dense_test_input,
                    g_caif_dense_test_output,true);
    const bool ok=MaxAbsDiff(host_y.Data(),ref.data(),
                             g_caif_dense_test_batch*g_caif_dense_test_output)
                  <=g_caif_dense_test_fp32_tol;
    CAIF_TestHarness::Report("DenseLayer::Linear fp32",ok);
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::Linear fp32")
}

void CAIF_DenseLayerTests::TestDenseReluFp32()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);

    CAIF_DeviceDenseLayer<float,float> layer(g_caif_dense_test_input,
                                             g_caif_dense_test_output,
                                             CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU,
                                             stream);
    const std::vector<float> x_host=MakeData(g_caif_dense_test_batch*g_caif_dense_test_input,
                                             g_caif_dense_test_seed+11);
    const std::vector<float> w_host=MakeData(g_caif_dense_test_input*g_caif_dense_test_output,
                                             g_caif_dense_test_seed+12);
    const std::vector<float> b_host=MakeData(g_caif_dense_test_output,g_caif_dense_test_seed+13);
    OverwriteWeights(layer,w_host,b_host,stream);

    CAIF_DeviceTensor x=MakeFp32Device(x_host,
                                       {g_caif_dense_test_batch,g_caif_dense_test_input},
                                       stream);
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    CAIF_HostTensor host_y=y.ToHost();

    std::vector<float> ref(g_caif_dense_test_batch*g_caif_dense_test_output);
    CpuDenseForward(x_host.data(),w_host.data(),b_host.data(),
                    CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU,ref.data(),
                    g_caif_dense_test_batch,g_caif_dense_test_input,
                    g_caif_dense_test_output,true);
    const bool ok=MaxAbsDiff(host_y.Data(),ref.data(),
                             g_caif_dense_test_batch*g_caif_dense_test_output)
                  <=g_caif_dense_test_fp32_tol;
    CAIF_TestHarness::Report("DenseLayer::ReLU fp32",ok);
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::ReLU fp32")
}

void CAIF_DenseLayerTests::TestDenseBackwardFp32()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    CAIF_DeviceDenseLayer<float,float> layer(g_caif_dense_test_input,
                                             g_caif_dense_test_output,
                                             CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,
                                             stream);
    const std::vector<float> x_host=MakeData(g_caif_dense_test_batch*g_caif_dense_test_input,
                                             g_caif_dense_test_seed+21);
    const std::vector<float> w_host=MakeData(g_caif_dense_test_input*g_caif_dense_test_output,
                                             g_caif_dense_test_seed+22);
    const std::vector<float> b_host=MakeData(g_caif_dense_test_output,g_caif_dense_test_seed+23);
    OverwriteWeights(layer,w_host,b_host,stream);

    CAIF_DeviceTensor x=MakeFp32Device(x_host,
                                       {g_caif_dense_test_batch,g_caif_dense_test_input},
                                       stream);
    layer.ZeroGradients();
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    std::vector<float> grad_host(g_caif_dense_test_batch*g_caif_dense_test_output,1.0f);
    CAIF_DeviceTensor grad_y=MakeFp32Device(grad_host,
                                            {g_caif_dense_test_batch,g_caif_dense_test_output},
                                            stream);
    CAIF_DeviceTensor grad_x=layer.Backward(grad_y,ctx);

    // grad_x shape must match input.
    bool ok=grad_x.Shape().size()==2 &&
            grad_x.Shape()[0]==g_caif_dense_test_batch &&
            grad_x.Shape()[1]==g_caif_dense_test_input;
    // bias_grads should equal column-sum of grad_y when use_bias==true and
    // activation is None: each column gets g_caif_dense_test_batch contributions of 1.0.
    CAIF_HostTensor bias_g=layer.BiasGradients().ToHost();
    for(uint32_t c=0;c<g_caif_dense_test_output;++c)
    {
      const float expected=static_cast<float>(g_caif_dense_test_batch);
      if(std::fabs(bias_g.Data()[c]-expected)>g_caif_dense_test_fp32_tol)
      {
        ok=false;
      }
    }
    CAIF_TestHarness::Report("DenseLayer::Backward fp32 (bias-grad column sum)",ok);
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::Backward fp32")
}

template<typename StorageT>
bool CAIF_DenseLayerTests::RunDenseDtype(const CAIF_DataType::CAIF_DataType_e storage_dt,
                                         const float tol)
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  ctx.SetTraining(false);

  // Build the fp32 reference on a fresh fp32 layer first.
  CAIF_DeviceDenseLayer<float,float> ref_layer(g_caif_dense_test_input,
                                               g_caif_dense_test_output,
                                               CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,
                                               stream);
  const std::vector<float> x_host=MakeData(g_caif_dense_test_batch*g_caif_dense_test_input,
                                           g_caif_dense_test_seed+31);
  const std::vector<float> w_host=MakeData(g_caif_dense_test_input*g_caif_dense_test_output,
                                           g_caif_dense_test_seed+32);
  const std::vector<float> b_host=MakeData(g_caif_dense_test_output,g_caif_dense_test_seed+33);
  OverwriteWeights(ref_layer,w_host,b_host,stream);
  CAIF_DeviceTensor x_fp32=MakeFp32Device(x_host,
                                          {g_caif_dense_test_batch,g_caif_dense_test_input},
                                          stream);
  CAIF_DeviceTensor y_ref=ref_layer.Forward(x_fp32,ctx);
  CAIF_HostTensor host_ref=y_ref.ToHost();

  // Now templated cell on StorageT.
  CAIF_DeviceDenseLayer<float,StorageT> dev_layer(g_caif_dense_test_input,
                                                  g_caif_dense_test_output,
                                                  CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,
                                                  stream);
  // Cast weights/bias to StorageT and copy the underlying bytes into the
  // dev_layer's internal tensors. We go via tensor.To() to convert.
  CAIF_DeviceTensor w_fp32=MakeFp32Device(w_host,
                                          {g_caif_dense_test_input,g_caif_dense_test_output},
                                          stream);
  CAIF_DeviceTensor b_fp32=MakeFp32Device(b_host,{g_caif_dense_test_output},stream);
  CAIF_DeviceTensor w_dev=w_fp32.To(storage_dt);
  CAIF_DeviceTensor b_dev=b_fp32.To(storage_dt);
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

  return MaxAbsDiff(host_dev.Data(),host_ref.Data(),
                    g_caif_dense_test_batch*g_caif_dense_test_output)<=tol;
}

void CAIF_DenseLayerTests::TestDenseDtypeSweep()
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    CAIF_TestHarness::Report("DenseLayer::fp16 storage",
                             RunDenseDtype<__half>(Dtype_e::Float16,
                                                   g_caif_dense_test_fp16_tol));
    CAIF_TestHarness::Report("DenseLayer::bf16 storage",
                             RunDenseDtype<__nv_bfloat16>(Dtype_e::BFloat16,
                                                          g_caif_dense_test_bf16_tol));
  }
  CAIF_TEST_CATCH_BLOCK("DenseLayer::dtype-sweep")
}

void CAIF_DenseLayerTests::RunAll()
{
  CAIF_TestHarness::Reset();
  ISE_Out::Out()<<"Dense Layer Tests\n"
                <<"=================\n";
  TestDenseLinearFp32();
  TestDenseReluFp32();
  TestDenseBackwardFp32();
  TestDenseDtypeSweep();
}

#endif // USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_DenseLayerTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"USE_CAIF_CUDA off — dense layer tests skipped.\n";
  return 0;
#endif
}
