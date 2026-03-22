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
// AIF - AI Framework
// Test: CAIF_DeviceLoRAAdapter (LoRA low-rank adapter)
//------------------------------------------------------------------------------
#include "caif_device_lora_adapter.h"
#include "caif_device_frozen_linear.h"
#include "caif_device_dense_layer.h"
#include "caif_device_pre_norm_block.h"
#include "caif_device_rmsnorm.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cuda_kernels.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>
#include <memory>
#include <random>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    ISE_Out::Out()<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    ISE_Out::Out()<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
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
static CAIF_DeviceLoRAAdapter MakeLoRA(CAIF_CudaStream &stream,uint32_t seed=42)
{
  auto frozen=std::make_unique<CAIF_DeviceFrozenLinear>(g_test_input_dim,g_test_output_dim,
                                                       CAIF_DataType::CAIF_DataType_e::Float32,stream);
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

  CAIF_DeviceLoRAAdapter::LoRAConfig_t config;
  config.rank=g_test_rank;
  config.alpha=g_test_alpha;
  config.input_dim=g_test_input_dim;
  config.output_dim=g_test_output_dim;

  return CAIF_DeviceLoRAAdapter(config,std::move(frozen),stream,seed+1);
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
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=lora.Forward(input,false);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward finite values
//------------------------------------------------------------------------------
static void TestForwardFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=lora.Forward(input,false);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::ForwardFinite",false);
  }
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
    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear>(g_test_input_dim,g_test_output_dim,
                                                         CAIF_DataType::CAIF_DataType_e::Float32,stream);
    std::vector<float> w(static_cast<size_t>(g_test_input_dim)*g_test_output_dim,0.1f);
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);

    // Get base output before wrapping
    CAIF_DeviceFrozenLinear frozen_copy(g_test_input_dim,g_test_output_dim,
                                       CAIF_DataType::CAIF_DataType_e::Float32,stream);
    CAIF_DeviceTensor w_copy=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);
    frozen_copy.LoadFromTensor(std::move(w_copy));

    frozen->LoadFromTensor(std::move(weight));

    CAIF_DeviceLoRAAdapter::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=g_test_input_dim;
    config.output_dim=g_test_output_dim;

    CAIF_DeviceLoRAAdapter lora(config,std::move(frozen),stream,42);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim,777);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);

    CAIF_DeviceTensor base_out=frozen_copy.Forward(input,false);
    CAIF_DeviceTensor lora_out=lora.Forward(input,false);

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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::ZeroInitBMatchesBase",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: ParameterTensorCount == 2
//------------------------------------------------------------------------------
static void TestParameterTensorCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    bool passed=lora.ParameterTensorCount()==2;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 2, got "<<lora.ParameterTensorCount()<<"\n";
    }
    ReportResult("LoRAAdapter::ParameterTensorCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::ParameterTensorCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: TotalParameterCount == rank*input + output*rank
//------------------------------------------------------------------------------
static void TestTotalParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    const size_t expected=static_cast<size_t>(g_test_rank)*g_test_input_dim+
                          static_cast<size_t>(g_test_output_dim)*g_test_rank;
    bool passed=lora.TotalParameterCount()==expected;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected "<<expected<<", got "<<lora.TotalParameterCount()<<"\n";
    }
    ReportResult("LoRAAdapter::TotalParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::TotalParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: ParameterNames returns lora_a.weight, lora_b.weight
//------------------------------------------------------------------------------
static void TestParameterNames()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::ParameterNames",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Backward finite input gradients
//------------------------------------------------------------------------------
static void TestBackwardFiniteInputGradients()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    lora.Forward(input,true);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    CAIF_DeviceTensor grad_input=lora.Backward(grad_out);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::BackwardFiniteInputGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Backward non-zero LoRA gradients
//------------------------------------------------------------------------------
static void TestBackwardNonZeroLoRAGradients()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    // Set lora_b to non-zero so grad_lora_a is also non-zero
    // (when B=0, d_lora_hidden=grad@B=0, so grad_a=0 is mathematically correct)
    const size_t b_count=static_cast<size_t>(g_test_output_dim)*g_test_rank;
    std::vector<float> b_data(b_count,0.1f);
    lora.ParameterTensor(1).CopyFromHost(b_data.data(),b_count);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    lora.Forward(input,true);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    lora.Backward(grad_out);

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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::BackwardNonZeroLoRAGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: ZeroGradients zeroes LoRA grads
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    lora.Forward(input,true);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    lora.Backward(grad_out);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::ZeroGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Description contains "LoRA" + rank
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::Description",false);
  }
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
    CAIF_DeviceLoRAAdapter lora=MakeLoRA(stream);

    auto host_input=MakeRandomInput(batch*seq_len,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,g_test_input_dim},stream);
    CAIF_DeviceTensor output=lora.Forward(input,false);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::3DInputForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 12: LoRA wrapping FrozenLinear(FP16) end-to-end
//------------------------------------------------------------------------------
static void TestLoRAWithFP16Frozen()
{
  try
  {
    CAIF_CudaStream stream;

    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear>(g_test_input_dim,g_test_output_dim,
                                                         CAIF_DataType::CAIF_DataType_e::Float16,stream);
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

    CAIF_DeviceLoRAAdapter::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=g_test_input_dim;
    config.output_dim=g_test_output_dim;

    CAIF_DeviceLoRAAdapter lora(config,std::move(frozen),stream,22);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=lora.Forward(input,true);
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
      CAIF_DeviceTensor grad_input=lora.Backward(grad_out);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::LoRAWithFP16Frozen",false);
  }
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

    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear>(g_test_input_dim,g_test_output_dim,
                                                         CAIF_DataType::CAIF_DataType_e::Int4,
                                                         stream,group_size);

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
    launch_quantize_to_int4(static_cast<const float*>(fp32_w.DevicePtr()),
                             packed.DevicePtr(),
                             scales.DevicePtr(),
                             static_cast<int>(count),
                             static_cast<int>(group_size),
                             stream.Handle());
    stream.Synchronize();

    frozen->LoadFromTensor(std::move(packed));
    const size_t scales_bytes=num_groups*sizeof(uint16_t);
    std::vector<uint8_t> host_scales(scales_bytes);
    scales.CopyToHostRaw(host_scales.data());
    frozen->LoadScalesFromHost(host_scales.data(),scales_bytes);

    CAIF_DeviceLoRAAdapter::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=g_test_input_dim;
    config.output_dim=g_test_output_dim;

    CAIF_DeviceLoRAAdapter lora(config,std::move(frozen),stream,44);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=lora.Forward(input,true);
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
      CAIF_DeviceTensor grad_input=lora.Backward(grad_out);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::LoRAWithINT4Frozen",false);
  }
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
    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear>(dim,dim,
                                                         CAIF_DataType::CAIF_DataType_e::Float32,stream);
    std::vector<float> w(static_cast<size_t>(dim)*dim,0.01f);
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),{dim,dim},stream);
    frozen->LoadFromTensor(std::move(weight));

    CAIF_DeviceLoRAAdapter::LoRAConfig_t config;
    config.rank=g_test_rank;
    config.alpha=g_test_alpha;
    config.input_dim=dim;
    config.output_dim=dim;

    auto lora=std::make_unique<CAIF_DeviceLoRAAdapter>(config,std::move(frozen),stream,77);

    // Build PreNormBlock with RMSNorm + LoRA
    CAIF_DevicePreNormBlock::SubLayerVec_t sub_layers;
    CAIF_DevicePreNormBlock::SubLayer_t stage;
    stage.norm_prefix="norm.";
    stage.layer_prefix="linear.";
    stage.norm=std::make_unique<CAIF_DeviceRMSNorm>(dim,stream);
    stage.layer=std::move(lora);
    sub_layers.push_back(std::move(stage));

    CAIF_DevicePreNormBlock block(std::move(sub_layers),stream);

    // Forward
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    auto host_input=MakeRandomInput(batch*seq_len,dim,555);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=block.Forward(input,true);
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
      CAIF_DeviceTensor grad_input=block.Backward(grad_out);
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
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("LoRAAdapter::LoRAInPreNormBlock",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF_DeviceLoRAAdapter Tests ===\n\n";

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
#else
    ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

    ISE_Out::Out()<<"\n=== Summary ===\n";
    ISE_Out::Out()<<"Passed: "<<g_tests_passed<<"\n";
    ISE_Out::Out()<<"Failed: "<<g_tests_failed<<"\n";

    if(g_tests_failed>0)
    {
      return 1;
    }
    return 0;
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"AIF Exception: "<<e<<std::endl;
    return 1;
  }
  catch(const std::exception &e)
  {
    ISE_Out::ErrLog()<<"std::exception: "<<e.what()<<std::endl;
    return 1;
  }
}
