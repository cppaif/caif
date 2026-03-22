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
// Test: CAIF_DeviceFrozenLinear (Dtype-agnostic frozen linear layer)
//------------------------------------------------------------------------------
#include "caif_device_frozen_linear.h"
#include "caif_device_dense_layer.h"
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

//------------------------------------------------------------------------------
// Helper: create a FrozenLinear with FP32 random weights loaded
//------------------------------------------------------------------------------
static CAIF_DeviceFrozenLinear MakeFP32Frozen(CAIF_CudaStream &stream,uint32_t seed=42)
{
  CAIF_DeviceFrozenLinear layer(g_test_input_dim,g_test_output_dim,
                               CAIF_DataType::CAIF_DataType_e::Float32,stream);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);
  const size_t count=static_cast<size_t>(g_test_input_dim)*g_test_output_dim;
  std::vector<float> w(count);
  for(size_t i=0;i<count;++i)
  {
    w[i]=dist(gen);
  }
  CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),{g_test_input_dim,g_test_output_dim},stream);
  layer.LoadFromTensor(std::move(weight));
  return layer;
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
// Test 1: FP32 forward shape
//------------------------------------------------------------------------------
static void TestFP32ForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2||shape[0]!=g_test_batch||shape[1]!=g_test_output_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }
    ReportResult("FrozenLinear::FP32ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::FP32ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: FP32 forward finite values
//------------------------------------------------------------------------------
static void TestFP32ForwardFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
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
    ReportResult("FrozenLinear::FP32ForwardFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::FP32ForwardFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: FP32 backward shape + finite
//------------------------------------------------------------------------------
static void TestFP32BackwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    layer.Forward(input,true);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    CAIF_DeviceTensor grad_input=layer.Backward(grad_out);
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
    ReportResult("FrozenLinear::FP32BackwardShapeFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::FP32BackwardShapeFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: FP16 forward shape + finite
//------------------------------------------------------------------------------
static void TestFP16ForwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer(g_test_input_dim,g_test_output_dim,
                                 CAIF_DataType::CAIF_DataType_e::Float16,stream);

    // Create FP32 weights, convert to FP16, load
    auto w=MakeRandomInput(g_test_input_dim,g_test_output_dim,77);
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);
    CAIF_DeviceTensor fp16_w=fp32_w.To(CAIF_DataType::CAIF_DataType_e::Float16);
    layer.LoadFromTensor(std::move(fp16_w));

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2||shape[0]!=g_test_batch||shape[1]!=g_test_output_dim)
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
    ReportResult("FrozenLinear::FP16ForwardShapeFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::FP16ForwardShapeFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: BF16 forward shape + finite
//------------------------------------------------------------------------------
static void TestBF16ForwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer(g_test_input_dim,g_test_output_dim,
                                 CAIF_DataType::CAIF_DataType_e::BFloat16,stream);

    auto w=MakeRandomInput(g_test_input_dim,g_test_output_dim,88);
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);
    CAIF_DeviceTensor bf16_w=fp32_w.To(CAIF_DataType::CAIF_DataType_e::BFloat16);
    layer.LoadFromTensor(std::move(bf16_w));

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2||shape[0]!=g_test_batch||shape[1]!=g_test_output_dim)
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
    ReportResult("FrozenLinear::BF16ForwardShapeFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::BF16ForwardShapeFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: INT8 forward shape + finite
//------------------------------------------------------------------------------
static void TestINT8ForwardShapeFinite()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer(g_test_input_dim,g_test_output_dim,
                                 CAIF_DataType::CAIF_DataType_e::Int8,stream);

    // Create small FP32 weights (within INT8 range), convert to INT8
    const size_t count=static_cast<size_t>(g_test_input_dim)*g_test_output_dim;
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-5.0f,5.0f);
    std::vector<float> w(count);
    for(size_t i=0;i<count;++i)
    {
      w[i]=dist(gen);
    }
    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);
    CAIF_DeviceTensor int8_w=fp32_w.To(CAIF_DataType::CAIF_DataType_e::Int8);
    layer.LoadFromTensor(std::move(int8_w));

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2||shape[0]!=g_test_batch||shape[1]!=g_test_output_dim)
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
    ReportResult("FrozenLinear::INT8ForwardShapeFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::INT8ForwardShapeFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: INT4 forward shape + finite (with scales)
//------------------------------------------------------------------------------
static void TestINT4ForwardShapeFinite()
{
  try
  {
    const uint32_t group_size=g_caif_quant_default_group_size;
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer(g_test_input_dim,g_test_output_dim,
                                 CAIF_DataType::CAIF_DataType_e::Int4,stream,group_size);

    // Create FP32 weights, quantize to INT4 on device, then load
    const size_t count=static_cast<size_t>(g_test_input_dim)*g_test_output_dim;
    std::vector<float> w(count);
    std::mt19937 gen(55);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for(size_t i=0;i<count;++i)
    {
      w[i]=dist(gen);
    }

    CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);

    // Allocate packed INT4 output and scales
    const size_t packed_bytes=(count+1)/2;
    const uint32_t num_groups=static_cast<uint32_t>((count+group_size-1)/group_size);
    CAIF_DeviceTensor packed=CAIF_DeviceTensor::Zeros({static_cast<uint32_t>(packed_bytes)},stream,
                                                     CAIF_DataType::CAIF_DataType_e::UInt8);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Zeros({num_groups},stream,
                                                     CAIF_DataType::CAIF_DataType_e::Float16);

    // Quantize on device
    launch_quantize_to_int4(static_cast<const float*>(fp32_w.DevicePtr()),
                             packed.DevicePtr(),
                             scales.DevicePtr(),
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
    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=2||shape[0]!=g_test_batch||shape[1]!=g_test_output_dim)
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
    ReportResult("FrozenLinear::INT4ForwardShapeFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::INT4ForwardShapeFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: ParameterTensorCount == 0
//------------------------------------------------------------------------------
static void TestParameterTensorCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);

    bool passed=layer.ParameterTensorCount()==0;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 0, got "<<layer.ParameterTensorCount()<<"\n";
    }
    ReportResult("FrozenLinear::ParameterTensorCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::ParameterTensorCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: TotalParameterCount == 0
//------------------------------------------------------------------------------
static void TestTotalParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);

    bool passed=layer.TotalParameterCount()==0;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 0, got "<<layer.TotalParameterCount()<<"\n";
    }
    ReportResult("FrozenLinear::TotalParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::TotalParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Description contains dtype + dims
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);
    const std::string desc=layer.Description();

    bool passed=true;
    if(desc.find("FrozenLinear")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'FrozenLinear': "<<desc<<"\n";
      passed=false;
    }
    if(desc.find("fp32")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'fp32': "<<desc<<"\n";
      passed=false;
    }
    if(desc.find(std::to_string(g_test_input_dim))==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing input_dim: "<<desc<<"\n";
      passed=false;
    }
    if(desc.find(std::to_string(g_test_output_dim))==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing output_dim: "<<desc<<"\n";
      passed=false;
    }
    ReportResult("FrozenLinear::Description",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::Description",false);
  }
}

//------------------------------------------------------------------------------
// Test 11: LoadFromTensor works
//------------------------------------------------------------------------------
static void TestLoadFromTensor()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer(g_test_input_dim,g_test_output_dim,
                                 CAIF_DataType::CAIF_DataType_e::Float32,stream);

    std::vector<float> w(static_cast<size_t>(g_test_input_dim)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(w.data(),
                                                            {g_test_input_dim,g_test_output_dim},stream);
    layer.LoadFromTensor(std::move(weight));

    // Forward should work after load
    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=h_out.TotalElements()==static_cast<size_t>(g_test_batch)*g_test_output_dim;
    ReportResult("FrozenLinear::LoadFromTensor",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::LoadFromTensor",false);
  }
}

//------------------------------------------------------------------------------
// Test 12: 3D input [batch, seq_len, dim] forward shape
//------------------------------------------------------------------------------
static void Test3DInputForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(batch*seq_len,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,g_test_input_dim},stream);
    CAIF_DeviceTensor output=layer.Forward(input,false);
    CAIF_HostTensor h_out=output.ToHost();

    bool passed=true;
    const auto &shape=h_out.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=g_test_output_dim)
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
    ReportResult("FrozenLinear::3DInputForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::3DInputForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 13: Backward passes gradient through (non-zero, finite)
//------------------------------------------------------------------------------
static void TestBackwardNonZero()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceFrozenLinear layer=MakeFP32Frozen(stream);

    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    layer.Forward(input,true);

    std::vector<float> grad_data(static_cast<size_t>(g_test_batch)*g_test_output_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                              {g_test_batch,g_test_output_dim},stream);
    CAIF_DeviceTensor grad_input=layer.Backward(grad_out);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    bool any_nonzero=false;
    for(size_t i=0;i<h_grad.TotalElements();++i)
    {
      if(std::isfinite(h_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite grad at "<<i<<"\n";
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
    ReportResult("FrozenLinear::BackwardNonZero",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::BackwardNonZero",false);
  }
}

//------------------------------------------------------------------------------
// Test 14: FP32 FrozenLinear matches DenseLayer output (identical weights)
//------------------------------------------------------------------------------
static void TestMatchesDenseLayer()
{
  try
  {
    CAIF_CudaStream stream;

    // Create dense layer (no bias, no activation)
    CAIF_DeviceDenseLayer dense(g_test_input_dim,g_test_output_dim,
                               CAIF_DeviceActivation_e::None,stream,false);

    // Copy dense weights to frozen layer
    CAIF_HostTensor dense_weights=dense.Weights().ToHost();

    CAIF_DeviceFrozenLinear frozen(g_test_input_dim,g_test_output_dim,
                                  CAIF_DataType::CAIF_DataType_e::Float32,stream);
    CAIF_DeviceTensor frozen_w=CAIF_DeviceTensor::FromHostData(dense_weights.Data(),
                                                              {g_test_input_dim,g_test_output_dim},stream);
    frozen.LoadFromTensor(std::move(frozen_w));

    // Same input
    auto host_input=MakeRandomInput(g_test_batch,g_test_input_dim,999);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_test_batch,g_test_input_dim},stream);
    CAIF_DeviceTensor dense_out=dense.Forward(input,false);
    CAIF_DeviceTensor frozen_out=frozen.Forward(input,false);

    CAIF_HostTensor h_dense=dense_out.ToHost();
    CAIF_HostTensor h_frozen=frozen_out.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_dense.TotalElements();++i)
    {
      const float diff=std::fabs(h_dense.Data()[i]-h_frozen.Data()[i]);
      if(diff>1e-4f)
      {
        ISE_Out::Out()<<"  Mismatch at "<<i<<": dense="<<h_dense.Data()[i]
                      <<" frozen="<<h_frozen.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }
    ReportResult("FrozenLinear::MatchesDenseLayer",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("FrozenLinear::MatchesDenseLayer",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF_DeviceFrozenLinear Tests ===\n\n";

#ifdef USE_CAIF_CUDA
    TestFP32ForwardShape();
    TestFP32ForwardFinite();
    TestFP32BackwardShapeFinite();
    TestFP16ForwardShapeFinite();
    TestBF16ForwardShapeFinite();
    TestINT8ForwardShapeFinite();
    TestINT4ForwardShapeFinite();
    TestParameterTensorCount();
    TestTotalParameterCount();
    TestDescription();
    TestLoadFromTensor();
    Test3DInputForwardShape();
    TestBackwardNonZero();
    TestMatchesDenseLayer();
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
