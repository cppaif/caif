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

#include "caif_device_linear_head.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    std::cout<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    std::cout<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
}

static bool FloatEqual(float a,float b,float tolerance=1e-3f)
{
  return std::fabs(a-b)<tolerance;
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// CPU reference: linear projection
//------------------------------------------------------------------------------
static void CpuLinear(const float *input,
                      const float *weight,
                      const float *bias,
                      float *output,
                      int n,
                      int input_dim,
                      int output_dim,
                      bool use_bias)
{
  for(int i=0;i<n;++i)
  {
    for(int o=0;o<output_dim;++o)
    {
      float sum=0.0f;
      for(int j=0;j<input_dim;++j)
      {
        sum+=input[i*input_dim+j]*weight[j*output_dim+o];
      }
      if(use_bias==true)
      {
        sum+=bias[o];
      }
      output[i*output_dim+o]=sum;
    }
  }
}

// CPU reference: linear with transposed weight (for tied weights)
static void CpuLinearTranspose(const float *input,
                               const float *weight_t,
                               const float *bias,
                               float *output,
                               int n,
                               int input_dim,
                               int output_dim,
                               bool use_bias)
{
  // weight_t is [output_dim, input_dim]
  // output = input @ weight_t^T = input[n, input_dim] @ weight_t[output_dim, input_dim]^T
  for(int i=0;i<n;++i)
  {
    for(int o=0;o<output_dim;++o)
    {
      float sum=0.0f;
      for(int j=0;j<input_dim;++j)
      {
        sum+=input[i*input_dim+j]*weight_t[o*input_dim+j];
      }
      if(use_bias==true)
      {
        sum+=bias[o];
      }
      output[i*output_dim+o]=sum;
    }
  }
}

//------------------------------------------------------------------------------
// Test 1: Forward output shape
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=4;
    constexpr uint32_t input_dim=64;
    constexpr uint32_t output_dim=128;

    CAIF_CudaStream stream;
    CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,true};
    CAIF_DeviceLinearHead head(config,stream);

    std::vector<float> input_data(batch*seq_len*input_dim,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,input_dim},stream);

    CAIF_DeviceTensor output=head.Forward(input,false);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==3&&
                 shape[0]==batch&&
                 shape[1]==seq_len&&
                 shape[2]==output_dim);

    ReportResult("LinearHead::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward shape no bias
//------------------------------------------------------------------------------
static void TestForwardShapeNoBias()
{
  try
  {
    constexpr uint32_t batch=2;
    constexpr uint32_t input_dim=32;
    constexpr uint32_t output_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,false};
    CAIF_DeviceLinearHead head(config,stream);

    std::vector<float> input_data(batch*input_dim,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,input_dim},stream);

    CAIF_DeviceTensor output=head.Forward(input,false);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==2&&
                 shape[0]==batch&&
                 shape[1]==output_dim);

    ReportResult("LinearHead::ForwardShapeNoBias",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::ForwardShapeNoBias",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Forward values vs CPU reference
//------------------------------------------------------------------------------
static void TestForwardValues()
{
  try
  {
    constexpr uint32_t batch=2;
    constexpr uint32_t input_dim=4;
    constexpr uint32_t output_dim=3;

    CAIF_CudaStream stream;
    CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,true};
    CAIF_DeviceLinearHead head(config,stream);

    // Get weight and bias
    std::vector<float> weight_data(input_dim*output_dim);
    std::vector<float> bias_data(output_dim);
    head.ParameterTensor(0).CopyToHost(weight_data.data());

    // Get bias from second parameter
    if(head.ParameterTensorCount()>1)
    {
      head.ParameterTensor(1).CopyToHost(bias_data.data());
    }

    // Create input
    std::vector<float> input_data(batch*input_dim);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.1f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,input_dim},stream);

    // GPU forward
    CAIF_DeviceTensor output=head.Forward(input,false);
    std::vector<float> gpu_output(batch*output_dim);
    output.CopyToHost(gpu_output.data());

    // CPU reference
    std::vector<float> cpu_output(batch*output_dim);
    CpuLinear(input_data.data(),weight_data.data(),bias_data.data(),
              cpu_output.data(),batch,input_dim,output_dim,true);

    bool passed=true;
    for(size_t i=0;i<cpu_output.size();++i)
    {
      if(FloatEqual(gpu_output[i],cpu_output[i],1e-3f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": GPU="<<gpu_output[i]<<", CPU="<<cpu_output[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("LinearHead::ForwardValues",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::ForwardValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Weight-tied forward
//------------------------------------------------------------------------------
static void TestWeightTiedForward()
{
  try
  {
    constexpr uint32_t input_dim=4;
    constexpr uint32_t output_dim=6;
    constexpr uint32_t batch=2;

    CAIF_CudaStream stream;

    // Create "embedding table" - shape [output_dim, input_dim]
    std::vector<float> emb_data(output_dim*input_dim);
    for(size_t i=0;i<emb_data.size();++i)
    {
      emb_data[i]=static_cast<float>(i)*0.05f;
    }
    CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
                                 emb_data.data(),{output_dim,input_dim},stream);
    CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros({output_dim,input_dim},stream);

    // Create tied linear head
    CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,true};
    CAIF_DeviceLinearHead head(config,embedding,embedding_grad,stream);

    // Create input
    std::vector<float> input_data(batch*input_dim);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.1f+0.1f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,input_dim},stream);

    // Get bias
    std::vector<float> bias_data(output_dim,0.0f);
    if(head.ParameterTensorCount()>0)
    {
      head.ParameterTensor(0).CopyToHost(bias_data.data());
    }

    // GPU forward
    CAIF_DeviceTensor output=head.Forward(input,false);
    std::vector<float> gpu_output(batch*output_dim);
    output.CopyToHost(gpu_output.data());

    // CPU reference with transposed weight
    std::vector<float> cpu_output(batch*output_dim);
    CpuLinearTranspose(input_data.data(),emb_data.data(),bias_data.data(),
                       cpu_output.data(),batch,input_dim,output_dim,true);

    bool passed=true;
    for(size_t i=0;i<cpu_output.size();++i)
    {
      if(FloatEqual(gpu_output[i],cpu_output[i],1e-3f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": GPU="<<gpu_output[i]<<", CPU="<<cpu_output[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("LinearHead::WeightTiedForward",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::WeightTiedForward",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Backward input gradient (finite difference)
//------------------------------------------------------------------------------
static void TestBackwardInputGrad()
{
  try
  {
    constexpr uint32_t batch=2;
    constexpr uint32_t input_dim=4;
    constexpr uint32_t output_dim=3;
    constexpr float h=1e-3f;
    constexpr float tolerance=5e-2f;

    CAIF_CudaStream stream;
    CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,true};
    CAIF_DeviceLinearHead head(config,stream);

    // Create input
    std::vector<float> input_data(batch*input_dim);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.1f+0.1f;
    }

    // Forward and backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,input_dim},stream);
    CAIF_DeviceTensor output=head.Forward(input,true);

    // Create grad_output (all ones for sum reduction)
    std::vector<float> grad_out_data(batch*output_dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_out_data.data(),{batch,output_dim},stream);

    CAIF_DeviceTensor grad_input=head.Backward(grad_output);
    std::vector<float> analytical_grad(batch*input_dim);
    grad_input.CopyToHost(analytical_grad.data());

    // Finite difference for a few elements
    bool passed=true;
    for(int idx=0;idx<4;++idx)
    {
      // f(x+h)
      std::vector<float> input_plus=input_data;
      input_plus[idx]+=h;
      CAIF_DeviceTensor inp_plus=CAIF_DeviceTensor::FromHostData(
                                  input_plus.data(),{batch,input_dim},stream);
      CAIF_DeviceTensor out_plus=head.Forward(inp_plus,false);
      std::vector<float> out_plus_data(batch*output_dim);
      out_plus.CopyToHost(out_plus_data.data());
      float sum_plus=0.0f;
      for(auto v:out_plus_data)
      {
        sum_plus+=v;
      }

      // f(x-h)
      std::vector<float> input_minus=input_data;
      input_minus[idx]-=h;
      CAIF_DeviceTensor inp_minus=CAIF_DeviceTensor::FromHostData(
                                   input_minus.data(),{batch,input_dim},stream);
      CAIF_DeviceTensor out_minus=head.Forward(inp_minus,false);
      std::vector<float> out_minus_data(batch*output_dim);
      out_minus.CopyToHost(out_minus_data.data());
      float sum_minus=0.0f;
      for(auto v:out_minus_data)
      {
        sum_minus+=v;
      }

      float numerical_grad=(sum_plus-sum_minus)/(2.0f*h);
      float diff=std::fabs(analytical_grad[idx]-numerical_grad);

      if(diff>tolerance)
      {
        std::cout<<"  idx="<<idx<<": analytical="<<analytical_grad[idx]
                 <<", numerical="<<numerical_grad<<", diff="<<diff<<"\n";
        passed=false;
      }
    }

    ReportResult("LinearHead::BackwardInputGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::BackwardInputGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Backward weight gradient (finite difference)
//------------------------------------------------------------------------------
static void TestBackwardWeightGrad()
{
  try
  {
    constexpr uint32_t batch=2;
    constexpr uint32_t input_dim=4;
    constexpr uint32_t output_dim=3;
    constexpr float h=1e-3f;
    constexpr float tolerance=5e-2f;

    CAIF_CudaStream stream;
    CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,true};

    // Create input
    std::vector<float> input_data(batch*input_dim);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.1f+0.1f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,input_dim},stream);

    // Create grad_output
    std::vector<float> grad_out_data(batch*output_dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_out_data.data(),{batch,output_dim},stream);

    // Get analytical gradient
    CAIF_DeviceLinearHead head(config,stream);
    head.ZeroGradients();
    head.Forward(input,true);
    head.Backward(grad_output);

    std::vector<float> analytical_grad(input_dim*output_dim);
    head.GradientTensor(0).CopyToHost(analytical_grad.data());

    // Finite difference for a few weight elements
    bool passed=true;
    for(int idx=0;idx<4;++idx)
    {
      // Get current weight
      std::vector<float> weight_data(input_dim*output_dim);
      head.ParameterTensor(0).CopyToHost(weight_data.data());

      // f(w+h)
      weight_data[idx]+=h;
      head.ParameterTensor(0).CopyFromHost(weight_data.data(),weight_data.size());
      CAIF_DeviceTensor out_plus=head.Forward(input,false);
      std::vector<float> out_plus_data(batch*output_dim);
      out_plus.CopyToHost(out_plus_data.data());
      float sum_plus=0.0f;
      for(auto v:out_plus_data)
      {
        sum_plus+=v;
      }

      // f(w-h)
      weight_data[idx]-=2.0f*h;
      head.ParameterTensor(0).CopyFromHost(weight_data.data(),weight_data.size());
      CAIF_DeviceTensor out_minus=head.Forward(input,false);
      std::vector<float> out_minus_data(batch*output_dim);
      out_minus.CopyToHost(out_minus_data.data());
      float sum_minus=0.0f;
      for(auto v:out_minus_data)
      {
        sum_minus+=v;
      }

      // Restore weight
      weight_data[idx]+=h;
      head.ParameterTensor(0).CopyFromHost(weight_data.data(),weight_data.size());

      float numerical_grad=(sum_plus-sum_minus)/(2.0f*h);
      float diff=std::fabs(analytical_grad[idx]-numerical_grad);

      if(diff>tolerance)
      {
        std::cout<<"  idx="<<idx<<": analytical="<<analytical_grad[idx]
                 <<", numerical="<<numerical_grad<<", diff="<<diff<<"\n";
        passed=false;
      }
    }

    ReportResult("LinearHead::BackwardWeightGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::BackwardWeightGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Weight-tied backward gradient
//------------------------------------------------------------------------------
static void TestWeightTiedBackwardGrad()
{
  try
  {
    constexpr uint32_t input_dim=4;
    constexpr uint32_t output_dim=6;
    constexpr uint32_t batch=2;

    CAIF_CudaStream stream;

    // Create "embedding table"
    std::vector<float> emb_data(output_dim*input_dim);
    for(size_t i=0;i<emb_data.size();++i)
    {
      emb_data[i]=static_cast<float>(i)*0.05f;
    }
    CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
                                 emb_data.data(),{output_dim,input_dim},stream);
    CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros({output_dim,input_dim},stream);

    // Create tied linear head
    CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,false};
    CAIF_DeviceLinearHead head(config,embedding,embedding_grad,stream);

    // Create input
    std::vector<float> input_data(batch*input_dim);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.1f+0.1f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,input_dim},stream);

    // Create grad_output
    std::vector<float> grad_out_data(batch*output_dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
                                   grad_out_data.data(),{batch,output_dim},stream);

    // Forward and backward
    embedding_grad.Fill(0.0f);
    head.Forward(input,true);
    head.Backward(grad_output);

    // Check that embedding_grad is non-zero
    std::vector<float> emb_grad_data(output_dim*input_dim);
    embedding_grad.CopyToHost(emb_grad_data.data());

    float sum=0.0f;
    for(auto v:emb_grad_data)
    {
      sum+=std::fabs(v);
    }

    bool passed=(sum>0.01f);
    if(passed==false)
    {
      std::cout<<"  Tied weight gradient sum is too small: "<<sum<<"\n";
    }

    ReportResult("LinearHead::WeightTiedBackwardGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::WeightTiedBackwardGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Parameter count
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    constexpr uint32_t input_dim=8;
    constexpr uint32_t output_dim=16;

    CAIF_CudaStream stream;

    // Untied with bias: 2 params (W + b)
    {
      CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,true};
      CAIF_DeviceLinearHead head(config,stream);
      if(head.ParameterTensorCount()!=2)
      {
        std::cout<<"  Untied+bias: expected 2, got "<<head.ParameterTensorCount()<<"\n";
        ReportResult("LinearHead::ParameterCount",false);
        return;
      }
    }

    // Untied no bias: 1 param (W)
    {
      CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,false};
      CAIF_DeviceLinearHead head(config,stream);
      if(head.ParameterTensorCount()!=1)
      {
        std::cout<<"  Untied no bias: expected 1, got "<<head.ParameterTensorCount()<<"\n";
        ReportResult("LinearHead::ParameterCount",false);
        return;
      }
    }

    // Tied with bias: 1 param (b only)
    {
      std::vector<float> emb_data(output_dim*input_dim,0.1f);
      CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
                                   emb_data.data(),{output_dim,input_dim},stream);
      CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros({output_dim,input_dim},stream);

      CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,true};
      CAIF_DeviceLinearHead head(config,embedding,embedding_grad,stream);
      if(head.ParameterTensorCount()!=1)
      {
        std::cout<<"  Tied+bias: expected 1, got "<<head.ParameterTensorCount()<<"\n";
        ReportResult("LinearHead::ParameterCount",false);
        return;
      }
    }

    // Tied no bias: 0 params
    {
      std::vector<float> emb_data(output_dim*input_dim,0.1f);
      CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
                                   emb_data.data(),{output_dim,input_dim},stream);
      CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros({output_dim,input_dim},stream);

      CAIF_DeviceLinearHead::Config_t config{input_dim,output_dim,false};
      CAIF_DeviceLinearHead head(config,embedding,embedding_grad,stream);
      if(head.ParameterTensorCount()!=0)
      {
        std::cout<<"  Tied no bias: expected 0, got "<<head.ParameterTensorCount()<<"\n";
        ReportResult("LinearHead::ParameterCount",false);
        return;
      }
    }

    ReportResult("LinearHead::ParameterCount",true);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("LinearHead::ParameterCount",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF DeviceLinearHead Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestForwardShape();
  TestForwardShapeNoBias();
  TestForwardValues();
  TestWeightTiedForward();
  TestBackwardInputGrad();
  TestBackwardWeightGrad();
  TestWeightTiedBackwardGrad();
  TestParameterCount();
#else
  std::cout<<"CUDA not enabled, skipping GPU tests\n";
#endif

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  return (g_tests_failed==0)?0:1;
}
