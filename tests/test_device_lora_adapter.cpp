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
// Test: CAIF_DeviceLoRAAdapter<float,float> (LoRA low-rank adapter)
//------------------------------------------------------------------------------
#include "caif_device_lora_adapter.h"
#include "caif_test_harness.h"
#include "caif_device_frozen_linear.h"
#include "caif_int4_packed_t.h"
#include "caif_device_dense_layer.h"
#include "caif_device_pre_norm_block.h"
#include "caif_device_rmsnorm.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cuda_kernels.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "caif_run_context.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <random>

using namespace instance;

static void ReportResult(const char *test_name,bool passed)
{
  CAIF_TestHarness::Report(test_name,passed);
}

#ifdef USE_CAIF_CUDA

static const uint32_t g_test_input_dim=16;
static const uint32_t g_test_output_dim=8;
static const uint32_t g_test_batch=2;
static const uint32_t g_test_rank=4;
static const float g_test_alpha=8.0f;

//------------------------------------------------------------------------------
// Helper: create a LoRA adapter wrapping a FP32 FrozenLinear
//------------------------------------------------------------------------------
static CAIF_DeviceLoRAAdapter<float,float> MakeLoRA(CAIF_CudaStream &stream,uint32_t seed=42)
{
  auto frozen=std::make_unique<CAIF_DeviceFrozenLinear<float,float>>(g_test_input_dim,g_test_output_dim,stream);
  // Load random weights
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);
  const size_t count=static_cast<size_t>(g_test_input_dim)*g_test_output_dim;
  std::vector<float> w(count);
  for(size_t i=0;i<count;++i)
  {
    w[i]=dist(gen);
  }
  CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),
                                                          {g_test_input_dim,g_test_output_dim},stream);
  frozen->LoadFromTensor(std::move(weight));

  CAIF_DeviceLoRAAdapter<float,float>::LoRAConfig_t config;
  config.rank=g_test_rank;
  config.alpha=g_test_alpha;
  config.input_dim=g_test_input_dim;
  config.output_dim=g_test_output_dim;

  return CAIF_DeviceLoRAAdapter<float,float>(config,std::move(frozen),stream,seed+1);
}

//------------------------------------------------------------------------------
// Helper: create random input
//------------------------------------------------------------------------------
static std::vector<float> MakeRandomInput(uint32_t n,uint32_t dim,uint32_t seed=123)
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

//------------------------------------------------------------------------------
// Test 1: Forward shape preserved
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=lora.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2||shape[0]!=g_test_batch||shape[1]!=g_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    ReportResult("LoRAAdapter::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Forward finite values
//------------------------------------------------------------------------------
static void TestForwardFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=lora.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_out.TotalElements();++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "<<i<<"\n";
        passed=false;
        break;
      }
    }
    ReportResult("LoRAAdapter::ForwardFinite",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::ForwardFinite")
}

//------------------------------------------------------------------------------
// Test 3: Zero-init B => output == base output
//------------------------------------------------------------------------------
static void TestZeroInitBMatchesBase()
{
  try
  {
    CAIF_CudaStream stream;

    // Create frozen linear
    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear<float,float>>(g_test_input_dim,g_test_output_dim,stream);
    std::vector<float> w(static_cast<size_t>(g_test_input_dim)*g_test_output_dim,0.1f);
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);

    // Get base output before wrapping
    CAIF_DeviceFrozenLinear<float,float> frozen_copy(g_test_input_dim,g_test_output_dim,stream);
    CAIF_DeviceTensor w_copy=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);
    frozen_copy.LoadFromTensor(std::move(w_copy));

    frozen->LoadFromTensor(std::move(weight));

    CAIF_DeviceLoRAAdapter<float,float>::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=g_test_input_dim;
    config.output_dim=g_test_output_dim;

    CAIF_DeviceLoRAAdapter<float,float> lora(config,std::move(frozen),stream,42);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim,777);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);

    CAIF_DeviceTensor base_out=frozen_copy.Forward(input,ctx);
    CAIF_DeviceTensor lora_out=lora.Forward(input,ctx);

    CAIF_HostTensor h_base=base_out.ToHost();
    CAIF_HostTensor h_lora=lora_out.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_base.TotalElements();++i)
    {
      const float diff=std::fabs(h_base.Data()[i]-h_lora.Data()[i]);
      if(diff>1e-4f)
      {
        ISE_Out::Out()<<"  Mismatch at "<<i<<": base="<<h_base.Data()[i]
                      <<" lora="<<h_lora.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }
    ReportResult("LoRAAdapter::ZeroInitBMatchesBase",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::ZeroInitBMatchesBase")
}

//------------------------------------------------------------------------------
// Test 4: ParameterTensorCount == 2
//------------------------------------------------------------------------------
static void TestParameterTensorCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    bool passed=lora.ParameterTensorCount()==2;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 2, got "<<lora.ParameterTensorCount()<<"\n";
    }
    ReportResult("LoRAAdapter::ParameterTensorCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::ParameterTensorCount")
}

//------------------------------------------------------------------------------
// Test 5: TotalParameterCount == rank*input + output*rank
//------------------------------------------------------------------------------
static void TestTotalParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    const size_t expected=static_cast<size_t>(g_test_rank)*g_test_input_dim+
                          static_cast<size_t>(g_test_output_dim)*g_test_rank;
    bool passed=lora.TotalParameterCount()==expected;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected "<<expected<<", got "<<lora.TotalParameterCount()<<"\n";
    }
    ReportResult("LoRAAdapter::TotalParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::TotalParameterCount")
}

//------------------------------------------------------------------------------
// Test 6: ParameterNames returns lora_a.weight, lora_b.weight
//------------------------------------------------------------------------------
static void TestParameterNames()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    auto names=lora.ParameterNames("layer.");
    bool passed=true;
    if(names.size()!=2)
    {
      ISE_Out::Out()<<"  Expected 2 names, got "<<names.size()<<"\n";
      passed=false;
    }
    else
    {
      if(names[0]!="layer.lora_a.weight")
      {
        ISE_Out::Out()<<"  names[0]="<<names[0]<<" expected layer.lora_a.weight\n";
        passed=false;
      }
      if(names[1]!="layer.lora_b.weight")
      {
        ISE_Out::Out()<<"  names[1]="<<names[1]<<" expected layer.lora_b.weight\n";
        passed=false;
      }
    }
    ReportResult("LoRAAdapter::ParameterNames",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::ParameterNames")
}

//------------------------------------------------------------------------------
// Test 7: Backward finite input gradients
//------------------------------------------------------------------------------
static void TestBackwardFiniteInputGradients()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    lora.Forward(input,ctx);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=lora.Backward(grad_out,ctx);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    const auto &shape=h_grad.Shape();
    if(shape.size()!=2||shape[0]!=g_test_batch||shape[1]!=g_test_input_dim)
    {
      ISE_Out::Out()<<"  Grad shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_grad.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(h_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at "<<i<<"\n";
        passed=false;
      }
    }
    ReportResult("LoRAAdapter::BackwardFiniteInputGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::BackwardFiniteInputGradients")
}

//------------------------------------------------------------------------------
// Test 8: Backward non-zero LoRA gradients
//------------------------------------------------------------------------------
static void TestBackwardNonZeroLoRAGradients()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    // Set lora_b to non-zero so grad_lora_a is also non-zero
    // (when B=0, d_lora_hidden=grad@B=0, so grad_a=0 is mathematically correct)
    const size_t b_count=static_cast<size_t>(g_test_output_dim)*g_test_rank;
    std::vector<float> b_data(b_count,0.1f);
    lora.ParameterTensor(1).CopyFromHost(b_data.data(),b_count);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    lora.Forward(input,ctx);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    lora.Backward(grad_out,ctx);

    bool passed=true;
    // Check both LoRA gradient tensors are non-zero
    for(size_t p=0;p<lora.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_grad=lora.GradientTensor(p).ToHost();
      bool any_nonzero=false;
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(std::isfinite(h_grad.Data()[i])==false)
        {
          ISE_Out::Out()<<"  Non-finite LoRA grad["<<p<<"] at "<<i<<"\n";
          passed=false;
          break;
        }
        if(h_grad.Data()[i]!=0.0f)
        {
          any_nonzero=true;
        }
      }
      if(any_nonzero==false&&passed==true)
      {
        ISE_Out::Out()<<"  LoRA grad["<<p<<"] is all zeros\n";
        passed=false;
      }
    }
    ReportResult("LoRAAdapter::BackwardNonZeroLoRAGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::BackwardNonZeroLoRAGradients")
}

//------------------------------------------------------------------------------
// Test 9: ZeroGradients zeroes LoRA grads
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    lora.Forward(input,ctx);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    lora.Backward(grad_out,ctx);
    lora.ZeroGradients();

    bool passed=true;
    for(size_t p=0;p<lora.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_grad=lora.GradientTensor(p).ToHost();
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(h_grad.Data()[i]!=0.0f)
        {
          ISE_Out::Out()<<"  Grad["<<p<<"]["<<i<<"] not zero: "<<h_grad.Data()[i]<<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }
    ReportResult("LoRAAdapter::ZeroGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::ZeroGradients")
}

//------------------------------------------------------------------------------
// Test 10: Description contains "LoRA" + rank
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);
    const std::string desc=lora.Description();

    bool passed=true;
    if(desc.find("LoRA")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'LoRA': "<<desc<<"\n";
      passed=false;
    }
    if(desc.find(std::to_string(g_test_rank))==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing rank: "<<desc<<"\n";
      passed=false;
    }
    ReportResult("LoRAAdapter::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::Description")
}

//------------------------------------------------------------------------------
// Test 11: 3D input forward shape
//------------------------------------------------------------------------------
static void Test3DInputForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLoRAAdapter<float,float> lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(batch*seq_len,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,g_test_input_dim},stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=lora.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=g_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    ReportResult("LoRAAdapter::3DInputForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::3DInputForwardShape")
}

//------------------------------------------------------------------------------
// Test 12: LoRA wrapping FrozenLinear(FP16) end-to-end
//------------------------------------------------------------------------------
static void TestLoRAWithFP16Frozen()
{
  try
  {
    CAIF_CudaStream stream;

    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear<float,__half>>(g_test_input_dim,
                                                                        g_test_output_dim,
                                                                        stream);
    std::vector<float> w(static_cast<size_t>(g_test_input_dim)*g_test_output_dim);
    std::mt19937 gen(11);
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    for(size_t i=0;i<w.size();++i)
    {
      w[i]=dist(gen);
    }
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);
    CAIF_DeviceTensor fp16_w=fp32_w.To(CAIF_DataType::CAIF_DataType_e::Float16);
    frozen->LoadFromTensor(std::move(fp16_w));

    CAIF_DeviceLoRAAdapter<float,float>::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=g_test_input_dim;
    config.output_dim=g_test_output_dim;

    CAIF_DeviceLoRAAdapter<float,float> lora(config,std::move(frozen),stream,22);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=lora.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_out.TotalElements();++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "<<i<<"\n";
        passed=false;
        break;
      }
    }

    // Also verify backward works
    if(passed==true)
    {
      std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
      CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                                {g_test_batch,g_test_output_dim},stream);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      CAIF_DeviceTensor grad_input=lora.Backward(grad_out,ctx);
      CAIF_HostTensor h_grad=grad_input.ToHost();
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(std::isfinite(h_grad.Data()[i])==false)
        {
          ISE_Out::Out()<<"  Non-finite backward at "<<i<<"\n";
          passed=false;
          break;
        }
      }
    }
    ReportResult("LoRAAdapter::LoRAWithFP16Frozen",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::LoRAWithFP16Frozen")
}

//------------------------------------------------------------------------------
// Test 13: LoRA wrapping FrozenLinear(INT4) end-to-end
//------------------------------------------------------------------------------
static void TestLoRAWithINT4Frozen()
{
  try
  {
    const uint32_t group_size=g_caif_quant_default_group_size;
    CAIF_CudaStream stream;

    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear<float,caif_int4_packed_t>>(
                                                            g_test_input_dim,
                                                            g_test_output_dim,
                                                            stream,
                                                            group_size);

    const size_t count=static_cast<size_t>(g_test_input_dim)*g_test_output_dim;
    std::vector<float> w(count);
    std::mt19937 gen(33);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for(size_t i=0;i<count;++i)
    {
      w[i]=dist(gen);
    }
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);

    // Quantize to INT4
    const size_t packed_bytes=(count+1)/2;
    const uint32_t num_groups=static_cast<uint32_t>((count+group_size-1)/group_size);
    CAIF_DeviceTensor packed=CAIF_DeviceTensor::Zeros({static_cast<uint32_t>(packed_bytes)},stream,
                                                     CAIF_DataType::CAIF_DataType_e::UInt8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Zeros({num_groups},stream,
                                                     CAIF_DataType::CAIF_DataType_e::Float16);
    // scales is fp16 storage; route through DeviceDataRaw to bypass the
    // fp16/bf16 DevicePtr() tripwire (caif_device_tensor.h).
    launch_quantize_to_int4(fp32_w.DevicePtr<float>(),
                             packed.DeviceDataRaw(),
                             scales.DeviceDataRaw(),
                             static_cast<int>(count),
                             static_cast<int>(group_size),
                             stream.Handle());
    stream.Synchronize();

    frozen->LoadFromTensor(std::move(packed));
    const size_t scales_bytes=num_groups*sizeof(uint16_t);
    std::vector<uint8_t> host_scales(scales_bytes);
    scales.CopyToHostRaw(host_scales.data());
    frozen->LoadScalesFromHost(host_scales.data(),scales_bytes);

    CAIF_DeviceLoRAAdapter<float,float>::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=g_test_input_dim;
    config.output_dim=g_test_output_dim;

    CAIF_DeviceLoRAAdapter<float,float> lora(config,std::move(frozen),stream,44);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=lora.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_out.TotalElements();++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "<<i<<"\n";
        passed=false;
        break;
      }
    }

    // Backward
    if(passed==true)
    {
      std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
      CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                                {g_test_batch,g_test_output_dim},stream);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      CAIF_DeviceTensor grad_input=lora.Backward(grad_out,ctx);
      CAIF_HostTensor h_grad=grad_input.ToHost();
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(std::isfinite(h_grad.Data()[i])==false)
        {
          ISE_Out::Out()<<"  Non-finite backward at "<<i<<"\n";
          passed=false;
          break;
        }
      }
    }
    ReportResult("LoRAAdapter::LoRAWithINT4Frozen",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::LoRAWithINT4Frozen")
}

//------------------------------------------------------------------------------
// Test 14: LoRA+FrozenLinear inside PreNormBlock composition
//------------------------------------------------------------------------------
static void TestLoRAInPreNormBlock()
{
  try
  {
    const uint32_t dim=g_test_input_dim;
    CAIF_CudaStream stream;

    // Build a FrozenLinear(dim->dim) + LoRA as a sub-layer inside PreNormBlock
    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear<float,float>>(dim,dim,stream);
    std::vector<float> w(static_cast<size_t>(dim)*dim,0.01f);
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),{dim,dim},stream);
    frozen->LoadFromTensor(std::move(weight));

    CAIF_DeviceLoRAAdapter<float,float>::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=dim;
    config.output_dim=dim;

    auto lora=std::make_unique<CAIF_DeviceLoRAAdapter<float,float>>(config,std::move(frozen),stream,77);

    // Build PreNormBlock with RMSNorm + LoRA
    CAIF_DevicePreNormBlock<float,float>::SubLayerVec_t sub_layers;
    CAIF_DevicePreNormBlock<float,float>::SubLayer_t stage;
    stage.norm_prefix="norm.";
    stage.layer_prefix="linear.";
    stage.norm=std::make_unique<CAIF_DeviceRMSNorm<float,float>>(dim,stream);
    stage.layer=std::move(lora);
    sub_layers.push_back(std::move(stage));

    CAIF_DevicePreNormBlock<float,float> block(std::move(sub_layers),stream);

    // Forward
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    auto host_input=MakeRandomInput(batch*seq_len,dim,555);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=block.Forward(input,ctx);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_out.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite at "<<i<<"\n";
        passed=false;
      }
    }

    // Backward
    if(passed==true)
    {
      std::vector<float> grad_data(static_cast<size_t>(batch)*seq_len*dim,1.0f);
      CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                                {batch,seq_len,dim},stream);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      CAIF_DeviceTensor grad_input=block.Backward(grad_out,ctx);
      CAIF_HostTensor h_grad=grad_input.ToHost();
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(std::isfinite(h_grad.Data()[i])==false)
        {
          ISE_Out::Out()<<"  Non-finite backward at "<<i<<"\n";
          passed=false;
          break;
        }
      }
    }

    // Check ParameterTensorCount: RMSNorm(1) + LoRA(2) = 3
    if(block.ParameterTensorCount()!=3)
    {
      ISE_Out::Out()<<"  Expected 3 params, got "<<block.ParameterTensorCount()<<"\n";
      passed=false;
    }

    ReportResult("LoRAAdapter::LoRAInPreNormBlock",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LoRAAdapter::LoRAInPreNormBlock")
}

//------------------------------------------------------------------------------
// Phase 7 dtype-sweep: exercise LoRA at every <ComputeT,StorageT> cell that
// `CAIF_DeviceLoRAAdapter` instantiates against same-cell FrozenLinear bases.
// Forward+backward must be finite end-to-end; this verifies the templated
// LoRA cells aren't paper-only ("looks fine on paper, untested in practice"
// is what bit the dispatch sweep — Tier G of TYPE_DISPATCH_FULL_PLAN.md).
//------------------------------------------------------------------------------
template<typename T> static CAIF_DataType::CAIF_DataType_e DtypeFromCpp();
template<> CAIF_DataType::CAIF_DataType_e DtypeFromCpp<float>(){return CAIF_DataType::CAIF_DataType_e::Float32;}
template<> CAIF_DataType::CAIF_DataType_e DtypeFromCpp<__half>(){return CAIF_DataType::CAIF_DataType_e::Float16;}
template<> CAIF_DataType::CAIF_DataType_e DtypeFromCpp<__nv_bfloat16>(){return CAIF_DataType::CAIF_DataType_e::BFloat16;}

template<typename ComputeT,typename StorageT>
static void TestLoRACellForwardBackward(const char *cell_name,uint32_t seed)
{
  try
  {
    CAIF_CudaStream stream;
    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear<ComputeT,StorageT>>(
                  g_test_input_dim,
                  g_test_output_dim,
                  stream);
    std::vector<float> w(static_cast<size_t>(g_test_input_dim)*g_test_output_dim);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.5f,0.5f);
    for(size_t i=0;i<w.size();++i)
    {
      w[i]=dist(gen);
    }
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),
                                                              {g_test_input_dim,g_test_output_dim},
                                                              stream);
    const CAIF_DataType::CAIF_DataType_e storage_dtype=DtypeFromCpp<StorageT>();
    CAIF_DeviceTensor storage_w;
    if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      storage_w=std::move(fp32_w);
    }
    else
    {
      storage_w=fp32_w.To(storage_dtype);
    }
    frozen->LoadFromTensor(std::move(storage_w));

    typename CAIF_DeviceLoRAAdapter<ComputeT,StorageT>::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=g_test_input_dim;
    config.output_dim=g_test_output_dim;
    CAIF_DeviceLoRAAdapter<ComputeT,StorageT> lora(config,std::move(frozen),stream,seed+1u);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim,seed*7u+1u);
    CAIF_DeviceTensor input_fp32=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                                  {g_test_batch,g_test_input_dim},
                                                                  stream);
    CAIF_DeviceTensor input;
    if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      input=std::move(input_fp32);
    }
    else
    {
      input=input_fp32.To(storage_dtype);
    }

    CAIF_DeviceTensor output=lora.Forward(input,ctx);
    CAIF_DeviceTensor output_fp32;
    if(output.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      output_fp32=std::move(output);
    }
    else
    {
      output_fp32=output.To(CAIF_DataType::CAIF_DataType_e::Float32);
    }
    CAIF_HostTensor h_out=output_fp32.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_out.TotalElements();++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  ["<<cell_name<<"] non-finite forward at "<<i<<"\n";
        passed=false;
        break;
      }
    }

    if(passed==true)
    {
      std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
      CAIF_DeviceTensor grad_fp32=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                                   {g_test_batch,g_test_output_dim},
                                                                   stream);
      CAIF_DeviceTensor grad_out;
      if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        grad_out=std::move(grad_fp32);
      }
      else
      {
        grad_out=grad_fp32.To(storage_dtype);
      }
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      CAIF_DeviceTensor grad_input=lora.Backward(grad_out,ctx);
      CAIF_DeviceTensor grad_input_fp32;
      if(grad_input.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        grad_input_fp32=std::move(grad_input);
      }
      else
      {
        grad_input_fp32=grad_input.To(CAIF_DataType::CAIF_DataType_e::Float32);
      }
      CAIF_HostTensor h_grad=grad_input_fp32.ToHost();
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(std::isfinite(h_grad.Data()[i])==false)
        {
          ISE_Out::Out()<<"  ["<<cell_name<<"] non-finite backward at "<<i<<"\n";
          passed=false;
          break;
        }
      }
    }

    std::string label=std::string("LoRAAdapter::Cell::")+cell_name;
    ReportResult(label.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK((std::string("LoRAAdapter::Cell::")+cell_name).c_str())
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF_DeviceLoRAAdapter<float,float> Tests ===\n\n";

#ifdef USE_CAIF_CUDA
    TestForwardShape();
    TestForwardFinite();
    TestZeroInitBMatchesBase();
    TestParameterTensorCount();
    TestTotalParameterCount();
    TestParameterNames();
    TestBackwardFiniteInputGradients();
    TestBackwardNonZeroLoRAGradients();
    TestZeroGradients();
    TestDescription();
    Test3DInputForwardShape();
    TestLoRAWithFP16Frozen();
    TestLoRAWithINT4Frozen();
    TestLoRAInPreNormBlock();

    // Phase 7 dtype-sweep: 4 representative cells covering both mixed
    // (compute=float, storage=fp16/bf16) and same-cell (fp16/bf16 both
    // sides). The <float,float> cell is already exercised by every
    // test above.
    TestLoRACellForwardBackward<float,__half>("float_half",101u);
    TestLoRACellForwardBackward<float,__nv_bfloat16>("float_bfloat16",202u);
    TestLoRACellForwardBackward<__half,__half>("half_half",303u);
    TestLoRACellForwardBackward<__nv_bfloat16,__nv_bfloat16>("bfloat16_bfloat16",404u);
#else
    ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

    ISE_Out::Out()<<"\n=== Summary ===\n";
    ISE_Out::Out()<<"Passed: "<<CAIF_TestHarness::PassedCount()<<"\n";
    ISE_Out::Out()<<"Failed: "<<CAIF_TestHarness::FailedCount()<<"\n";

    if(CAIF_TestHarness::FailedCount()>0)
    {
      return 1;
    }
    return 0;
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception: "<<e<<std::endl;
    return 1;
  }
  catch(const std::exception &e)
  {
    ISE_Out::ErrLog()<<"std::exception: "<<e.what()<<std::endl;
    return 1;
  }
}
