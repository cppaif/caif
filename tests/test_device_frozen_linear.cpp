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
// Test: CAIF_DeviceFrozenLinear (Dtype-agnostic frozen linear layer)
//------------------------------------------------------------------------------
#include "caif_device_frozen_linear.h"
#include "caif_device_frozen_linear_factory.h"
#include "caif_device_frozen_linear_base.h"
#include "caif_test_harness.h"
#include "caif_device_dense_layer.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cuda_kernels_quant.cuh"
#include "caif_constants.h"
#include "caif_exception.h"
#include "caif_ops.h"
#include "caif_run_context.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <random>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_frozen_linear_test_input_dim=16;
constexpr uint32_t g_caif_frozen_linear_test_output_dim=8;
constexpr uint32_t g_caif_frozen_linear_test_batch=2;
constexpr float g_caif_frozen_linear_test_match_tol=1e-4f;

//------------------------------------------------------------------------------
// FrozenLinear dtype-agnostic correctness and accuracy tests.
//------------------------------------------------------------------------------
class CAIF_FrozenLinearTests
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceFrozenLinear<float,float> MakeFP32Frozen(CAIF_CudaStream &stream,
                                                               uint32_t seed=42);
    static std::vector<float> MakeRandomInput(uint32_t n,uint32_t dim,uint32_t seed=123);
    static bool CompareAgainstFP32(const CAIF_HostTensor &ref,
                                   const CAIF_HostTensor &got,
                                   const float tol,
                                   const char *label);
    static CAIF_HostTensor RunFP32Reference(const std::vector<float> &weights,
                                            const std::vector<float> &input_host,
                                            CAIF_CudaStream &stream);
    static bool RunInt8Case(const std::vector<float> &weights,
                            const std::vector<float> &host_input,
                            const CAIF_HostTensor &ref,
                            const CAIF_Ops::QuantScheme_e scheme,
                            const CAIF_DataType::CAIF_DataType_e compute,
                            const char *label,
                            const float tol,
                            CAIF_CudaStream &stream);
    static bool RunInt4Case(const std::vector<float> &weights,
                            const std::vector<float> &host_input,
                            const CAIF_HostTensor &ref,
                            const CAIF_DataType::CAIF_DataType_e compute,
                            const char *label,
                            const float tol,
                            CAIF_CudaStream &stream);

    static void TestFP32ForwardShape();
    static void TestFP32ForwardFinite();
    static void TestFP32BackwardShapeFinite();
    static void TestFP16ForwardShapeFinite();
    static void TestBF16ForwardShapeFinite();
    static void TestINT8ForwardShapeFinite();
    static void TestINT4ForwardShapeFinite();
    static void TestINT8PerTensorAccuracy();
    static void TestINT8PerChannelAccuracy();
    static void TestINT4PerGroupAccuracy();
    static void TestINT4DequantizeMatmulParity();
    static void TestParameterTensorCount();
    static void TestTotalParameterCount();
    static void TestDescription();
    static void TestLoadFromTensor();
    static void Test3DInputForwardShape();
    static void TestBackwardNonZero();
    static void TestMatchesDenseLayer();
};

CAIF_DeviceFrozenLinear<float,float> CAIF_FrozenLinearTests::MakeFP32Frozen(CAIF_CudaStream &stream,
                                                                            const uint32_t seed)
{
  CAIF_DeviceFrozenLinear<float,float> layer(g_caif_frozen_linear_test_input_dim,
                                             g_caif_frozen_linear_test_output_dim,
                                             stream);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);
  const size_t count=static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                     g_caif_frozen_linear_test_output_dim;
  std::vector<float> w(count);
  for(size_t i=0;i<count;++i)
  {
    w[i]=dist(gen);
  }
  CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(
                             w.data(),
                             {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                             stream);
  layer.LoadFromTensor(std::move(weight));
  return layer;
}

std::vector<float> CAIF_FrozenLinearTests::MakeRandomInput(const uint32_t n,
                                                           const uint32_t dim,
                                                           const uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  std::vector<float> data(static_cast<size_t>(n)*dim);
  for(size_t i=0;i<data.size();++i)
  {
    data[i]=dist(gen);
  }
  return data;
}

bool CAIF_FrozenLinearTests::CompareAgainstFP32(const CAIF_HostTensor &ref,
                                                const CAIF_HostTensor &got,
                                                const float tol,
                                                const char *label)
{
  if(ref.TotalElements()!=got.TotalElements())
  {
    ISE_Out::Out()<<"  "
                  <<label
                  <<" size mismatch\n";
    return false;
  }
  float max_abs=0.0f;
  for(size_t i=0;i<ref.TotalElements();++i)
  {
    const float a=std::fabs(ref.Data()[i]);
    if(a>max_abs)
    {
      max_abs=a;
    }
  }
  const float denom=std::max(max_abs,1e-6f);
  for(size_t i=0;i<ref.TotalElements();++i)
  {
    if(std::isfinite(got.Data()[i])==false)
    {
      ISE_Out::Out()<<"  "
                    <<label
                    <<" non-finite at "
                    <<i
                    <<"\n";
      return false;
    }
    const float rel=std::fabs(ref.Data()[i]-got.Data()[i])/denom;
    if(rel>tol)
    {
      ISE_Out::Out()<<"  "
                    <<label
                    <<" mismatch i="
                    <<i
                    <<" ref="
                    <<ref.Data()[i]
                    <<" got="
                    <<got.Data()[i]
                    <<" rel="
                    <<rel
                    <<"\n";
      return false;
    }
  }
  return true;
}

CAIF_HostTensor CAIF_FrozenLinearTests::RunFP32Reference(const std::vector<float> &weights,
                                                         const std::vector<float> &input_host,
                                                         CAIF_CudaStream &stream)
{
  CAIF_DeviceFrozenLinear<float,float> ref(g_caif_frozen_linear_test_input_dim,
                                           g_caif_frozen_linear_test_output_dim,
                                           stream);
  CAIF_DeviceTensor w=CAIF_DeviceTensor::FromHostData(
                        weights.data(),
                        {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                        stream);
  ref.LoadFromTensor(std::move(w));
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  ctx.SetTraining(false);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                            input_host.data(),
                            {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                            stream);
  CAIF_DeviceTensor out=ref.Forward(input,ctx);
  return out.ToHost();
}

bool CAIF_FrozenLinearTests::RunInt8Case(const std::vector<float> &weights,
                                         const std::vector<float> &host_input,
                                         const CAIF_HostTensor &ref,
                                         const CAIF_Ops::QuantScheme_e scheme,
                                         const CAIF_DataType::CAIF_DataType_e compute,
                                         const char *label,
                                         const float tol,
                                         CAIF_CudaStream &stream)
{
  std::unique_ptr<CAIF_DeviceLayer> layer=
    CAIF_DeviceFrozenLinearFactory::Create(g_caif_frozen_linear_test_input_dim,
                                           g_caif_frozen_linear_test_output_dim,
                                           CAIF_DataType::CAIF_DataType_e::Int8,
                                           stream,
                                           g_caif_quant_default_group_size,
                                           true,
                                           compute,
                                           scheme);
  CAIF_DeviceFrozenLinearBase *layer_fl=
    dynamic_cast<CAIF_DeviceFrozenLinearBase*>(layer.get());
  CAIF_DeviceTensor w_fp32=CAIF_DeviceTensor::FromHostData(
                             weights.data(),
                             {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                             stream);
  CAIF_DeviceTensor w_int8=CAIF_DeviceTensor::Zeros(
                              {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                              stream,
                              CAIF_DataType::CAIF_DataType_e::Int8);
  uint32_t scale_count=1u;
  if(scheme==CAIF_Ops::QuantScheme_e::PerChannel_e)
  {
    scale_count=g_caif_frozen_linear_test_output_dim;
  }
  CAIF_DeviceTensor scales=CAIF_DeviceTensor::Zeros({scale_count},
                                                    stream,
                                                    CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_Ops::QuantizeInt8(w_fp32,w_int8,scales,scheme,ctx);
  stream.Synchronize();

  layer_fl->LoadFromTensor(std::move(w_int8));
  std::vector<float> host_scales(scale_count);
  scales.CopyToHost(host_scales.data());
  layer_fl->LoadScalesFromHost(host_scales.data(),host_scales.size()*sizeof(float));

  ctx.SetTraining(false);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                            host_input.data(),
                            {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                            stream);
  CAIF_DeviceTensor out=layer->Forward(input,ctx);
  CAIF_HostTensor got=out.ToHost();
  return CompareAgainstFP32(ref,got,tol,label);
}

bool CAIF_FrozenLinearTests::RunInt4Case(const std::vector<float> &weights,
                                         const std::vector<float> &host_input,
                                         const CAIF_HostTensor &ref,
                                         const CAIF_DataType::CAIF_DataType_e compute,
                                         const char *label,
                                         const float tol,
                                         CAIF_CudaStream &stream)
{
  const uint32_t group_size=g_caif_quant_default_group_size;
  const size_t count=weights.size();
  std::unique_ptr<CAIF_DeviceLayer> layer=
    CAIF_DeviceFrozenLinearFactory::Create(g_caif_frozen_linear_test_input_dim,
                                           g_caif_frozen_linear_test_output_dim,
                                           CAIF_DataType::CAIF_DataType_e::Int4,
                                           stream,
                                           group_size,
                                           true,
                                           compute);
  CAIF_DeviceFrozenLinearBase *layer_fl=
    dynamic_cast<CAIF_DeviceFrozenLinearBase*>(layer.get());
  CAIF_DeviceTensor w_fp32=CAIF_DeviceTensor::FromHostData(
                             weights.data(),
                             {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                             stream);
  const size_t packed_bytes=(count+1)/2;
  const uint32_t num_groups=static_cast<uint32_t>((count+group_size-1)/group_size);
  CAIF_DeviceTensor packed=CAIF_DeviceTensor::Zeros({static_cast<uint32_t>(packed_bytes)},
                                                    stream,
                                                    CAIF_DataType::CAIF_DataType_e::UInt8);
  CAIF_DeviceTensor scales=CAIF_DeviceTensor::Zeros({num_groups},
                                                    stream,
                                                    CAIF_DataType::CAIF_DataType_e::Float16);
  // scales is fp16 storage; route through DeviceDataRaw to bypass the
  // fp16/bf16 DevicePtr() tripwire (caif_device_tensor.h).
  launch_quantize_to_int4(w_fp32.DevicePtr<float>(),
                          packed.DeviceDataRaw(),
                          scales.DeviceDataRaw(),
                          static_cast<int>(count),
                          static_cast<int>(group_size),
                          stream.Handle());
  stream.Synchronize();

  layer_fl->LoadFromTensor(std::move(packed));
  std::vector<uint8_t> host_scales(num_groups*sizeof(uint16_t));
  scales.CopyToHostRaw(host_scales.data());
  layer_fl->LoadScalesFromHost(host_scales.data(),host_scales.size());

  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  ctx.SetTraining(false);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                            host_input.data(),
                            {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                            stream);
  CAIF_DeviceTensor out=layer->Forward(input,ctx);
  CAIF_HostTensor got=out.ToHost();
  return CompareAgainstFP32(ref,got,tol,label);
}

//------------------------------------------------------------------------------
// Test 1: FP32 forward shape
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestFP32ForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2 ||
       shape[0]!=g_caif_frozen_linear_test_batch ||
       shape[1]!=g_caif_frozen_linear_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::FP32ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::FP32ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: FP32 forward finite values
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestFP32ForwardFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_out.TotalElements();++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "
                      <<i
                      <<"\n";
        passed=false;
        break;
      }
    }
    CAIF_TestHarness::Report("FrozenLinear::FP32ForwardFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::FP32ForwardFinite")
}

//------------------------------------------------------------------------------
// Test 3: FP32 backward shape + finite
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestFP32BackwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    layer.Forward(input,ctx);

    std::vector<float> grad_data(static_cast<size_t>(g_caif_frozen_linear_test_batch)*
                                 g_caif_frozen_linear_test_output_dim,
                                 1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                 grad_data.data(),
                                 {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_output_dim},
                                 stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=layer.Backward(grad_out,ctx);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    const auto &shape=h_grad.Shape();
    if(shape.size()!=2 ||
       shape[0]!=g_caif_frozen_linear_test_batch ||
       shape[1]!=g_caif_frozen_linear_test_input_dim)
    {
      ISE_Out::Out()<<"  Grad shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_grad.TotalElements() && passed==true;++i)
    {
      if(std::isfinite(h_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at "
                      <<i
                      <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("FrozenLinear::FP32BackwardShapeFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::FP32BackwardShapeFinite")
}

//------------------------------------------------------------------------------
// Test 4: FP16 forward shape + finite
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestFP16ForwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFrozenLinear<float,__half> layer(g_caif_frozen_linear_test_input_dim,
                                                g_caif_frozen_linear_test_output_dim,
                                                stream);

    // Create FP32 weights, convert to FP16, load
    auto w=MakeRandomInput(g_caif_frozen_linear_test_input_dim,
                           g_caif_frozen_linear_test_output_dim,
                           77);
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(
                               w.data(),
                               {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                               stream);
    CAIF_DeviceTensor fp16_w=fp32_w.To(CAIF_DataType::CAIF_DataType_e::Float16);
    layer.LoadFromTensor(std::move(fp16_w));

    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2 ||
       shape[0]!=g_caif_frozen_linear_test_batch ||
       shape[1]!=g_caif_frozen_linear_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_out.TotalElements() && passed==true;++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "
                      <<i
                      <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("FrozenLinear::FP16ForwardShapeFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::FP16ForwardShapeFinite")
}

//------------------------------------------------------------------------------
// Test 5: BF16 forward shape + finite
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestBF16ForwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFrozenLinear<float,__nv_bfloat16> layer(g_caif_frozen_linear_test_input_dim,
                                                       g_caif_frozen_linear_test_output_dim,
                                                       stream);

    auto w=MakeRandomInput(g_caif_frozen_linear_test_input_dim,
                           g_caif_frozen_linear_test_output_dim,
                           88);
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(
                               w.data(),
                               {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                               stream);
    CAIF_DeviceTensor bf16_w=fp32_w.To(CAIF_DataType::CAIF_DataType_e::BFloat16);
    layer.LoadFromTensor(std::move(bf16_w));

    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2 ||
       shape[0]!=g_caif_frozen_linear_test_batch ||
       shape[1]!=g_caif_frozen_linear_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_out.TotalElements() && passed==true;++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "
                      <<i
                      <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("FrozenLinear::BF16ForwardShapeFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::BF16ForwardShapeFinite")
}

//------------------------------------------------------------------------------
// Test 6: INT8 forward shape + finite
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestINT8ForwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFrozenLinear<float,int8_t> layer(g_caif_frozen_linear_test_input_dim,
                                                g_caif_frozen_linear_test_output_dim,
                                                stream);

    // Create small FP32 weights (within INT8 range), convert to INT8
    const size_t count=static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                       g_caif_frozen_linear_test_output_dim;
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-5.0f,5.0f);
    std::vector<float> w(count);
    for(size_t i=0;i<count;++i)
    {
      w[i]=dist(gen);
    }
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(
                               w.data(),
                               {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                               stream);
    CAIF_DeviceTensor int8_w=fp32_w.To(CAIF_DataType::CAIF_DataType_e::Int8);
    layer.LoadFromTensor(std::move(int8_w));

    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2 ||
       shape[0]!=g_caif_frozen_linear_test_batch ||
       shape[1]!=g_caif_frozen_linear_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_out.TotalElements() && passed==true;++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "
                      <<i
                      <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("FrozenLinear::INT8ForwardShapeFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::INT8ForwardShapeFinite")
}

//------------------------------------------------------------------------------
// Test 7: INT4 forward shape + finite (with scales)
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestINT4ForwardShapeFinite()
{
  try
  {
    const uint32_t group_size=g_caif_quant_default_group_size;
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear<float,caif_int4_packed_t> layer(g_caif_frozen_linear_test_input_dim,
                                                            g_caif_frozen_linear_test_output_dim,
                                                            stream,
                                                            group_size);

    // Create FP32 weights, quantize to INT4 on device, then load
    const size_t count=static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                       g_caif_frozen_linear_test_output_dim;
    std::vector<float> w(count);
    std::mt19937 gen(55);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for(size_t i=0;i<count;++i)
    {
      w[i]=dist(gen);
    }

    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(
                               w.data(),
                               {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                               stream);

    // Allocate packed INT4 output and scales
    const size_t packed_bytes=(count+1)/2;
    const uint32_t num_groups=static_cast<uint32_t>((count+group_size-1)/group_size);
    CAIF_DeviceTensor packed=CAIF_DeviceTensor::Zeros({static_cast<uint32_t>(packed_bytes)},
                                                      stream,
                                                      CAIF_DataType::CAIF_DataType_e::UInt8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Zeros({num_groups},
                                                      stream,
                                                      CAIF_DataType::CAIF_DataType_e::Float16);

    // Quantize on device. scales is fp16 storage; route through DeviceDataRaw
    // so the fp16/bf16 DevicePtr() tripwire (caif_device_tensor.h) doesn't
    // fire on a legitimate type-erased call.
    launch_quantize_to_int4(fp32_w.DevicePtr<float>(),
                            packed.DeviceDataRaw(),
                            scales.DeviceDataRaw(),
                            static_cast<int>(count),
                            static_cast<int>(group_size),
                            stream.Handle());
    stream.Synchronize();

    // Load packed weight into the frozen layer
    layer.LoadFromTensor(std::move(packed));

    // Download scales to host, then upload via LoadScalesFromHost
    const size_t scales_bytes=num_groups*sizeof(uint16_t);
    std::vector<uint8_t> host_scales(scales_bytes);
    scales.CopyToHostRaw(host_scales.data());
    layer.LoadScalesFromHost(host_scales.data(),scales_bytes);

    // Forward pass
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2 ||
       shape[0]!=g_caif_frozen_linear_test_batch ||
       shape[1]!=g_caif_frozen_linear_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_out.TotalElements() && passed==true;++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "
                      <<i
                      <<"\n";
        passed=false;
      }
    }
    CAIF_TestHarness::Report("FrozenLinear::INT4ForwardShapeFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::INT4ForwardShapeFinite")
}

//------------------------------------------------------------------------------
// Test 8: ParameterTensorCount == 0
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestParameterTensorCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);

    bool passed=layer.ParameterTensorCount()==0;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 0, got "
                    <<layer.ParameterTensorCount()
                    <<"\n";
    }
    CAIF_TestHarness::Report("FrozenLinear::ParameterTensorCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::ParameterTensorCount")
}

//------------------------------------------------------------------------------
// Test 9: TotalParameterCount == 0
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestTotalParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);

    bool passed=layer.TotalParameterCount()==0;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 0, got "
                    <<layer.TotalParameterCount()
                    <<"\n";
    }
    CAIF_TestHarness::Report("FrozenLinear::TotalParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::TotalParameterCount")
}

//------------------------------------------------------------------------------
// Test 10: Description contains dtype + dims
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);
    const std::string desc=layer.Description();

    bool passed=true;
    if(desc.find("FrozenLinear")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'FrozenLinear': "
                    <<desc
                    <<"\n";
      passed=false;
    }
    if(desc.find("fp32")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'fp32': "
                    <<desc
                    <<"\n";
      passed=false;
    }
    if(desc.find(std::to_string(g_caif_frozen_linear_test_input_dim))==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing input_dim: "
                    <<desc
                    <<"\n";
      passed=false;
    }
    if(desc.find(std::to_string(g_caif_frozen_linear_test_output_dim))==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing output_dim: "
                    <<desc
                    <<"\n";
      passed=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::Description")
}

//------------------------------------------------------------------------------
// Test 11: LoadFromTensor works
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestLoadFromTensor()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear<float,float> layer(g_caif_frozen_linear_test_input_dim,
                                               g_caif_frozen_linear_test_output_dim,
                                               stream);

    std::vector<float> w(static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                         g_caif_frozen_linear_test_output_dim,
                         1.0f);
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(
                               w.data(),
                               {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                               stream);
    layer.LoadFromTensor(std::move(weight));

    // Forward should work after load
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=h_out.TotalElements()==static_cast<size_t>(g_caif_frozen_linear_test_batch)*
                                       g_caif_frozen_linear_test_output_dim;
    CAIF_TestHarness::Report("FrozenLinear::LoadFromTensor",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::LoadFromTensor")
}

//------------------------------------------------------------------------------
// Test 12: 3D input [batch, seq_len, dim] forward shape
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::Test3DInputForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(batch*seq_len,g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {batch,seq_len,g_caif_frozen_linear_test_input_dim},
                              stream);
    CAIF_DeviceTensor output=layer.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3 ||
       shape[0]!=batch ||
       shape[1]!=seq_len ||
       shape[2]!=g_caif_frozen_linear_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch: got [";
      for(size_t i=0;i<shape.size();++i)
      {
        ISE_Out::Out()<<shape[i];
        if(i+1<shape.size())
        {
          ISE_Out::Out()<<",";
        }
      }
      ISE_Out::Out()<<"]\n";
      passed=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::3DInputForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::3DInputForwardShape")
}

//------------------------------------------------------------------------------
// Test 13: Backward passes gradient through (non-zero, finite)
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestBackwardNonZero()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFrozenLinear<float,float> layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    layer.Forward(input,ctx);

    std::vector<float> grad_data(static_cast<size_t>(g_caif_frozen_linear_test_batch)*
                                 g_caif_frozen_linear_test_output_dim,
                                 1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                 grad_data.data(),
                                 {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_output_dim},
                                 stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=layer.Backward(grad_out,ctx);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    bool any_nonzero=false;
    for(size_t i=0;i<h_grad.TotalElements();++i)
    {
      if(std::isfinite(h_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at "
                      <<i
                      <<"\n";
        passed=false;
        break;
      }
      if(h_grad.Data()[i]!=0.0f)
      {
        any_nonzero=true;
      }
    }
    if(any_nonzero==false)
    {
      ISE_Out::Out()<<"  Gradient is all zeros\n";
      passed=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::BackwardNonZero",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::BackwardNonZero")
}

//------------------------------------------------------------------------------
// Test 14: FP32 FrozenLinear matches DenseLayer output (identical weights)
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestMatchesDenseLayer()
{
  try
  {
    CAIF_CudaStream stream;

    // Create dense layer (no bias, no activation)
    CAIF_DeviceDenseLayer<float,float> dense(g_caif_frozen_linear_test_input_dim,
                                             g_caif_frozen_linear_test_output_dim,
                                             CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,
                                             stream,
                                             false);

    // Copy dense weights to frozen layer.
    // Dense stores weights as [input_dim, output_dim]; FrozenLinear
    // expects [output_dim, input_dim] (the HuggingFace safetensors
    // layout). Transpose host-side.
    CAIF_HostTensor dense_weights=dense.Weights().ToHost();
    std::vector<float> transposed(static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                                  g_caif_frozen_linear_test_output_dim);
    for(uint32_t i=0;i<g_caif_frozen_linear_test_input_dim;++i)
    {
      for(uint32_t o=0;o<g_caif_frozen_linear_test_output_dim;++o)
      {
        transposed[o*g_caif_frozen_linear_test_input_dim+i]=
          dense_weights.Data()[i*g_caif_frozen_linear_test_output_dim+o];
      }
    }

    CAIF_DeviceFrozenLinear<float,float> frozen(g_caif_frozen_linear_test_input_dim,
                                                g_caif_frozen_linear_test_output_dim,
                                                stream);
    CAIF_DeviceTensor frozen_w=CAIF_DeviceTensor::FromHostData(
                                 transposed.data(),
                                 {g_caif_frozen_linear_test_output_dim,g_caif_frozen_linear_test_input_dim},
                                 stream);
    frozen.LoadFromTensor(std::move(frozen_w));

    // Same input
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    auto host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                    g_caif_frozen_linear_test_input_dim,
                                    999);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                              host_input.data(),
                              {g_caif_frozen_linear_test_batch,g_caif_frozen_linear_test_input_dim},
                              stream);
    CAIF_DeviceTensor dense_out=dense.Forward(input,ctx);
    CAIF_DeviceTensor frozen_out=frozen.Forward(input,ctx);

    CAIF_HostTensor h_dense=dense_out.ToHost();
    CAIF_HostTensor h_frozen=frozen_out.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_dense.TotalElements();++i)
    {
      const float diff=std::fabs(h_dense.Data()[i]-h_frozen.Data()[i]);
      if(diff>g_caif_frozen_linear_test_match_tol)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": dense="
                      <<h_dense.Data()[i]
                      <<" frozen="
                      <<h_frozen.Data()[i]
                      <<"\n";
        passed=false;
        break;
      }
    }
    CAIF_TestHarness::Report("FrozenLinear::MatchesDenseLayer",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::MatchesDenseLayer")
}

//------------------------------------------------------------------------------
// Quantized storage dtype accuracy tests.
//
// For each storage+compute combo, quantize FP32 weights and compare the
// FrozenLinear forward output against an FP32-storage FP32-compute reference.
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestINT8PerTensorAccuracy()
{
  try
  {
    CAIF_CudaStream stream;
    const size_t count=static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                       g_caif_frozen_linear_test_output_dim;
    std::vector<float> weights(count);
    std::mt19937 gen(71);
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    for(size_t i=0;i<count;++i)
    {
      weights[i]=dist(gen);
    }
    std::vector<float> host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                                  g_caif_frozen_linear_test_input_dim,
                                                  201);
    CAIF_HostTensor ref=RunFP32Reference(weights,host_input,stream);

    bool all_ok=true;
    if(RunInt8Case(weights,host_input,ref,
                   CAIF_Ops::QuantScheme_e::PerTensor_e,
                   CAIF_DataType::CAIF_DataType_e::Float32,
                   "INT8pt_fp32c",2e-2f,stream)==false)
    {
      all_ok=false;
    }
    if(RunInt8Case(weights,host_input,ref,
                   CAIF_Ops::QuantScheme_e::PerTensor_e,
                   CAIF_DataType::CAIF_DataType_e::Float16,
                   "INT8pt_fp16c",3e-2f,stream)==false)
    {
      all_ok=false;
    }
    if(RunInt8Case(weights,host_input,ref,
                   CAIF_Ops::QuantScheme_e::PerTensor_e,
                   CAIF_DataType::CAIF_DataType_e::BFloat16,
                   "INT8pt_bf16c",4e-2f,stream)==false)
    {
      all_ok=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::INT8PerTensorAccuracy",all_ok);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::INT8PerTensorAccuracy")
}

void CAIF_FrozenLinearTests::TestINT8PerChannelAccuracy()
{
  try
  {
    CAIF_CudaStream stream;
    const size_t count=static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                       g_caif_frozen_linear_test_output_dim;
    std::vector<float> weights(count);
    std::mt19937 gen(72);
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    for(size_t i=0;i<count;++i)
    {
      weights[i]=dist(gen);
    }
    std::vector<float> host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                                  g_caif_frozen_linear_test_input_dim,
                                                  202);
    CAIF_HostTensor ref=RunFP32Reference(weights,host_input,stream);

    bool all_ok=true;
    if(RunInt8Case(weights,host_input,ref,
                   CAIF_Ops::QuantScheme_e::PerChannel_e,
                   CAIF_DataType::CAIF_DataType_e::Float32,
                   "INT8pc_fp32c",2e-2f,stream)==false)
    {
      all_ok=false;
    }
    if(RunInt8Case(weights,host_input,ref,
                   CAIF_Ops::QuantScheme_e::PerChannel_e,
                   CAIF_DataType::CAIF_DataType_e::Float16,
                   "INT8pc_fp16c",3e-2f,stream)==false)
    {
      all_ok=false;
    }
    if(RunInt8Case(weights,host_input,ref,
                   CAIF_Ops::QuantScheme_e::PerChannel_e,
                   CAIF_DataType::CAIF_DataType_e::BFloat16,
                   "INT8pc_bf16c",4e-2f,stream)==false)
    {
      all_ok=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::INT8PerChannelAccuracy",all_ok);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::INT8PerChannelAccuracy")
}

void CAIF_FrozenLinearTests::TestINT4PerGroupAccuracy()
{
  try
  {
    CAIF_CudaStream stream;
    const size_t count=static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                       g_caif_frozen_linear_test_output_dim;
    std::vector<float> weights(count);
    std::mt19937 gen(73);
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    for(size_t i=0;i<count;++i)
    {
      weights[i]=dist(gen);
    }
    std::vector<float> host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                                  g_caif_frozen_linear_test_input_dim,
                                                  203);
    CAIF_HostTensor ref=RunFP32Reference(weights,host_input,stream);

    bool all_ok=true;
    if(RunInt4Case(weights,host_input,ref,
                   CAIF_DataType::CAIF_DataType_e::Float32,
                   "INT4pg_fp32c",1.5e-1f,stream)==false)
    {
      all_ok=false;
    }
    if(RunInt4Case(weights,host_input,ref,
                   CAIF_DataType::CAIF_DataType_e::Float16,
                   "INT4pg_fp16c",1.5e-1f,stream)==false)
    {
      all_ok=false;
    }
    if(RunInt4Case(weights,host_input,ref,
                   CAIF_DataType::CAIF_DataType_e::BFloat16,
                   "INT4pg_bf16c",1.5e-1f,stream)==false)
    {
      all_ok=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::INT4PerGroupAccuracy",all_ok);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::INT4PerGroupAccuracy")
}

//------------------------------------------------------------------------------
// Phase 8.5.B: explicit INT4 numerical-parity test against a
// dequantize-then-fp32-matmul reference.
//
// Existing TestINT4PerGroupAccuracy compares INT4-on-device output against
// `RunFP32Reference` which uses the *original* full-precision weights —
// that mixes the int4 quantization round-trip noise into the tolerance,
// so a real matmul kernel bug at int4 storage would be hidden under the
// quant-noise band.
//
// This test isolates the matmul: quantize fp32 weights to int4 packed,
// then dequantize them back to fp32 (so the reference sees the same
// rounded values that landed in storage), and matmul those dequantized
// weights as the reference. The device int4 forward path should match
// to a tight tolerance because both sides operate on identical numerical
// inputs — the only difference is the device path matmuls in int4 with
// per-group scales, the reference matmuls in fp32 on the dequantized
// weights.
//------------------------------------------------------------------------------
void CAIF_FrozenLinearTests::TestINT4DequantizeMatmulParity()
{
  try
  {
    const uint32_t group_size=g_caif_quant_default_group_size;
    CAIF_CudaStream stream;

    const size_t count=static_cast<size_t>(g_caif_frozen_linear_test_input_dim)*
                       g_caif_frozen_linear_test_output_dim;
    std::vector<float> weights(count);
    std::mt19937 gen(91);
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    for(size_t i=0;i<count;++i)
    {
      weights[i]=dist(gen);
    }
    std::vector<float> host_input=MakeRandomInput(g_caif_frozen_linear_test_batch,
                                                  g_caif_frozen_linear_test_input_dim,
                                                  303);

    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(
                               weights.data(),
                               {g_caif_frozen_linear_test_input_dim,g_caif_frozen_linear_test_output_dim},
                               stream);
    const size_t packed_bytes=(count+1)/2;
    const uint32_t num_groups=static_cast<uint32_t>((count+group_size-1)/group_size);
    CAIF_DeviceTensor packed=CAIF_DeviceTensor::Zeros({static_cast<uint32_t>(packed_bytes)},
                                                      stream,
                                                      CAIF_DataType::CAIF_DataType_e::UInt8);
    CAIF_DeviceTensor scales_fp16=CAIF_DeviceTensor::Zeros({num_groups},
                                                           stream,
                                                           CAIF_DataType::CAIF_DataType_e::Float16);
    launch_quantize_to_int4(fp32_w.DevicePtr<float>(),
                            packed.DeviceDataRaw(),
                            scales_fp16.DeviceDataRaw(),
                            static_cast<int>(count),
                            static_cast<int>(group_size),
                            stream.Handle());
    stream.Synchronize();

    // Dequantize the int4 packed weights back to fp32 — this gives us
    // the same rounded numerical state that the device int4 path will
    // see, isolating the matmul from the quant noise. The packing
    // preserved the original [in_dim, out_dim] flat order, so the
    // dequant output is the same layout (no transpose).
    CAIF_DeviceTensor dequant_fp32=CAIF_DeviceTensor::Uninitialized(
                                     {g_caif_frozen_linear_test_input_dim,
                                      g_caif_frozen_linear_test_output_dim},
                                     stream,
                                     CAIF_DataType::CAIF_DataType_e::Float32);
    launch_dequantize_int4(packed.DeviceDataRaw(),
                           scales_fp16.DeviceDataRaw(),
                           dequant_fp32.DevicePtr<float>(),
                           static_cast<int>(count),
                           static_cast<int>(group_size),
                           stream.Handle());
    stream.Synchronize();

    // Reference matmul: fp32 layer wrapping the dequantized weights in
    // their native [in,out] layout (LoadFromTensor's expected shape).
    CAIF_HostTensor h_dequant=dequant_fp32.ToHost();
    std::vector<float> dequant_weights(h_dequant.Data(),h_dequant.Data()+count);
    CAIF_HostTensor ref=RunFP32Reference(dequant_weights,host_input,stream);

    // Device path: int4 FrozenLinear with fp32 compute (the cleanest
    // cell to isolate matmul correctness).
    bool all_ok=true;
    if(RunInt4Case(weights,host_input,ref,
                   CAIF_DataType::CAIF_DataType_e::Float32,
                   "INT4pg_fp32c_dequant_ref",2e-2f,stream)==false)
    {
      all_ok=false;
    }
    CAIF_TestHarness::Report("FrozenLinear::INT4DequantizeMatmulParity",all_ok);
  }
  CAIF_TEST_CATCH_BLOCK("FrozenLinear::INT4DequantizeMatmulParity")
}

void CAIF_FrozenLinearTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DeviceFrozenLinear Tests ===\n\n";
  TestFP32ForwardShape();
  TestFP32ForwardFinite();
  TestFP32BackwardShapeFinite();
  TestFP16ForwardShapeFinite();
  TestBF16ForwardShapeFinite();
  TestINT8ForwardShapeFinite();
  TestINT4ForwardShapeFinite();
  TestINT8PerTensorAccuracy();
  TestINT8PerChannelAccuracy();
  TestINT4PerGroupAccuracy();
  TestINT4DequantizeMatmulParity();
  TestParameterTensorCount();
  TestTotalParameterCount();
  TestDescription();
  TestLoadFromTensor();
  Test3DInputForwardShape();
  TestBackwardNonZero();
  TestMatchesDenseLayer();
  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FrozenLinearTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
