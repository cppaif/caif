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

#include "caif_device_patch_embedding.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_cpu_reference/caif_cpu_patch_extract.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
#include "caif_cpu_reference/caif_cpu_linear.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace instance;

static void ReportResult(const char *test_name,bool passed)
{
  CAIF_TestHarness::Report(test_name,passed);
}

static bool FloatEqual(float a,float b,float tolerance=1e-4f)
{
  return CAIF_TestHarness::FloatEqual(a,b,tolerance);
}

#ifdef USE_CAIF_CUDA

// Test config used across most tests
static CAIF_DevicePatchEmbedding<float,float>::Config_t TestConfig(bool cls)
{
  CAIF_DevicePatchEmbedding<float,float>::Config_t config;
  config.image_height=8;
  config.image_width=8;
  config.channels=3;
  config.patch_size=4;
  config.dim=6;
  config.use_cls_token=cls;
  return config;
}

//------------------------------------------------------------------------------
// Test 1: Forward shape without CLS
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePatchEmbedding<float,float>::Config_t config=TestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    // Input: [1,8,8,3]
    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    std::vector<float> input_data(batch*h*w*c,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    const auto &shape=output.Shape();

    // num_patches = (8/4)*(8/4) = 4
    bool passed=(shape.size()==3&&shape[0]==1&&shape[1]==4&&shape[2]==6);
    if(passed==false)
    {
      std::cout<<"  Expected [1,4,6], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
    }

    ReportResult("PatchEmbedding::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Forward shape with CLS
//------------------------------------------------------------------------------
static void TestForwardShapeCLS()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePatchEmbedding<float,float>::Config_t config=TestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    std::vector<float> input_data(batch*h*w*c,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    const auto &shape=output.Shape();

    // num_patches+1 = 5
    bool passed=(shape.size()==3&&shape[0]==1&&shape[1]==5&&shape[2]==6);
    if(passed==false)
    {
      std::cout<<"  Expected [1,5,6], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
    }

    ReportResult("PatchEmbedding::ForwardShapeCLS",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardShapeCLS")
}

//------------------------------------------------------------------------------
// Test 3: Forward values match CPU reference (no CLS)
//------------------------------------------------------------------------------
static void TestForwardValues()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePatchEmbedding<float,float>::Config_t config=TestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    constexpr uint32_t patch_size=4;
    constexpr uint32_t dim=6;
    constexpr uint32_t num_patches_h=2;
    constexpr uint32_t num_patches_w=2;
    constexpr uint32_t num_patches=4;
    constexpr uint32_t patch_flat=patch_size*patch_size*c;  // 48

    // Create input with known values
    std::vector<float> input_data(batch*h*w*c);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.01f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Read W_proj and b_proj
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();

    // CPU reference
    std::vector<float> patches(batch*num_patches*patch_flat);
    CAIF_CpuPatchExtract::Apply(input_data.data(),patches.data(),
                       batch,h,w,c,patch_size,num_patches_h,num_patches_w);

    std::vector<float> projected(num_patches*dim);
    CAIF_CpuMatMul::Apply(patches.data(),host_w.Data(),projected.data(),
               num_patches,patch_flat,dim);
    CAIF_CpuLinear::BiasAdd(projected.data(),host_b.Data(),num_patches,dim);

    bool passed=true;
    for(size_t i=0;i<num_patches*dim;++i)
    {
      if(FloatEqual(host_output.Data()[i],projected[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<projected[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("PatchEmbedding::ForwardValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardValues")
}

//------------------------------------------------------------------------------
// Test 4: CLS token at position 0, patches at 1..N
//------------------------------------------------------------------------------
static void TestForwardCLSValues()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePatchEmbedding<float,float>::Config_t config=TestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    constexpr uint32_t patch_size=4;
    constexpr uint32_t dim=6;
    constexpr uint32_t num_patches_h=2;
    constexpr uint32_t num_patches_w=2;
    constexpr uint32_t num_patches=4;
    constexpr uint32_t patch_flat=patch_size*patch_size*c;

    std::vector<float> input_data(batch*h*w*c);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.01f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Read CLS token
    CAIF_HostTensor host_cls=emb.ParameterTensor(2).ToHost();
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();

    // Position 0 should be CLS token
    bool passed=true;
    for(uint32_t d=0;d<dim;++d)
    {
      if(FloatEqual(host_output.Data()[d],host_cls.Data()[d])==false)
      {
        std::cout<<"  CLS mismatch at d="<<d<<": got "<<host_output.Data()[d]
                 <<" expected "<<host_cls.Data()[d]<<"\n";
        passed=false;
        break;
      }
    }

    // Positions 1..4 should be projected patches
    if(passed==true)
    {
      std::vector<float> patches(num_patches*patch_flat);
      CAIF_CpuPatchExtract::Apply(input_data.data(),patches.data(),
                         batch,h,w,c,patch_size,num_patches_h,num_patches_w);
      std::vector<float> projected(num_patches*dim);
      CAIF_CpuMatMul::Apply(patches.data(),host_w.Data(),projected.data(),
                 num_patches,patch_flat,dim);
      CAIF_CpuLinear::BiasAdd(projected.data(),host_b.Data(),num_patches,dim);

      for(uint32_t p=0;p<num_patches&&passed==true;++p)
      {
        for(uint32_t d=0;d<dim&&passed==true;++d)
        {
          // Output position p+1
          const float actual=host_output.Data()[(p+1)*dim+d];
          const float expected=projected[p*dim+d];
          if(FloatEqual(actual,expected)==false)
          {
            std::cout<<"  Patch "<<p<<" dim "<<d<<": got "<<actual
                     <<" expected "<<expected<<"\n";
            passed=false;
          }
        }
      }
    }

    ReportResult("PatchEmbedding::ForwardCLSValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardCLSValues")
}

//------------------------------------------------------------------------------
// Test 5: Backward input gradient (finite-difference)
//------------------------------------------------------------------------------
static void TestBackwardInputGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePatchEmbedding<float,float>::Config_t config=TestConfig(false);

    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    constexpr uint32_t dim=6;
    constexpr uint32_t num_patches=4;
    constexpr float fd_h=1e-3f;
    constexpr float grad_tol=5e-2f;

    // Create input
    const size_t input_size=batch*h*w*c;
    std::vector<float> input_data(input_size);
    for(size_t i=0;i<input_size;++i)
    {
      input_data[i]=static_cast<float>(i)*0.01f-0.5f;
    }

    // Read weights for consistent perturbed runs
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    const size_t w_size=emb.ParameterTensor(0).TotalElements();
    const size_t b_size=emb.ParameterTensor(1).TotalElements();

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);

    std::vector<float> grad_ones(batch*num_patches*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),
                                {batch,num_patches,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=emb.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Finite-difference check on a subset of input elements
    bool passed=true;
    constexpr int check_stride=16;  // Check every 16th element
    for(size_t i=0;i<input_size&&passed==true;i+=check_stride)
    {
      std::vector<float> inp_plus(input_data);
      std::vector<float> inp_minus(input_data);
      inp_plus[i]+=fd_h;
      inp_minus[i]-=fd_h;

      // f(x+h)
      CAIF_DevicePatchEmbedding<float,float> emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               inp_plus.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*num_patches*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // f(x-h)
      CAIF_DevicePatchEmbedding<float,float> emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               inp_minus.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*num_patches*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*fd_h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dx mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical<<"\n";
        passed=false;
      }
    }

    ReportResult("PatchEmbedding::BackwardInputGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::BackwardInputGrad")
}

//------------------------------------------------------------------------------
// Test 6: Backward weight gradient (finite-difference on W_proj)
//------------------------------------------------------------------------------
static void TestBackwardWeightGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePatchEmbedding<float,float>::Config_t config=TestConfig(false);

    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    constexpr uint32_t dim=6;
    constexpr uint32_t num_patches=4;
    constexpr float fd_h=1e-3f;
    constexpr float grad_tol=5e-2f;

    std::vector<float> input_data(batch*h*w*c);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.01f-0.5f;
    }

    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    const size_t w_size=host_w.TotalElements();
    const size_t b_size=host_b.TotalElements();
    std::vector<float> w_data(host_w.Data(),host_w.Data()+w_size);

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    std::vector<float> grad_ones(batch*num_patches*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),
                                {batch,num_patches,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    emb.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_w=emb.GradientTensor(0).ToHost();

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Finite-difference on subset of W_proj elements
    bool passed=true;
    constexpr int check_stride=12;
    for(size_t i=0;i<w_size&&passed==true;i+=check_stride)
    {
      std::vector<float> w_plus(w_data);
      std::vector<float> w_minus(w_data);
      w_plus[i]+=fd_h;
      w_minus[i]-=fd_h;

      CAIF_DevicePatchEmbedding<float,float> emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(w_plus.data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*num_patches*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DevicePatchEmbedding<float,float> emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(w_minus.data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*num_patches*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*fd_h);
      const float analytical=host_grad_w.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dW mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical<<"\n";
        passed=false;
      }
    }

    ReportResult("PatchEmbedding::BackwardWeightGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::BackwardWeightGrad")
}

//------------------------------------------------------------------------------
// Test 7: Backward CLS gradient (finite-difference on cls_token)
//------------------------------------------------------------------------------
static void TestBackwardCLSGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePatchEmbedding<float,float>::Config_t config=TestConfig(true);

    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    constexpr uint32_t dim=6;
    constexpr uint32_t num_patches=4;
    constexpr uint32_t out_seq=num_patches+1;
    constexpr float fd_h=1e-3f;
    constexpr float grad_tol=5e-2f;

    std::vector<float> input_data(batch*h*w*c);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.01f-0.5f;
    }

    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    CAIF_HostTensor host_cls=emb.ParameterTensor(2).ToHost();
    const size_t w_size=host_w.TotalElements();
    const size_t b_size=host_b.TotalElements();
    std::vector<float> cls_data(host_cls.Data(),host_cls.Data()+dim);

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    std::vector<float> grad_ones(batch*out_seq*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),
                                {batch,out_seq,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    emb.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_cls=emb.GradientTensor(2).ToHost();

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    bool passed=true;
    for(uint32_t d=0;d<dim&&passed==true;++d)
    {
      std::vector<float> cls_plus(cls_data);
      std::vector<float> cls_minus(cls_data);
      cls_plus[d]+=fd_h;
      cls_minus[d]-=fd_h;

      CAIF_DevicePatchEmbedding<float,float> emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      emb_p.ParameterTensor(2).CopyFromHost(cls_plus.data(),dim);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*out_seq*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DevicePatchEmbedding<float,float> emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      emb_m.ParameterTensor(2).CopyFromHost(cls_minus.data(),dim);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*out_seq*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*fd_h);
      const float analytical=host_grad_cls.Data()[d];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dCLS mismatch at d="<<d<<": analytical="<<analytical
                 <<" numerical="<<numerical<<"\n";
        passed=false;
      }
    }

    ReportResult("PatchEmbedding::BackwardCLSGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::BackwardCLSGrad")
}

//------------------------------------------------------------------------------
// Test 8: Parameter count
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    CAIF_CudaStream stream;

    // Without CLS
    CAIF_DevicePatchEmbedding<float,float>::Config_t config_no=TestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb_no(config_no,stream);
    bool passed=true;
    if(emb_no.ParameterTensorCount()!=2)
    {
      std::cout<<"  No CLS: ParameterTensorCount expected 2, got "
               <<emb_no.ParameterTensorCount()<<"\n";
      passed=false;
    }

    // With CLS
    CAIF_DevicePatchEmbedding<float,float>::Config_t config_cls=TestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb_cls(config_cls,stream);
    if(emb_cls.ParameterTensorCount()!=3)
    {
      std::cout<<"  CLS: ParameterTensorCount expected 3, got "
               <<emb_cls.ParameterTensorCount()<<"\n";
      passed=false;
    }

    ReportResult("PatchEmbedding::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 9: Patch size validation (rejects H % patch_size != 0)
//------------------------------------------------------------------------------
static void TestPatchSizeValidation()
{
  try
  {
    CAIF_CudaStream stream;

    CAIF_DevicePatchEmbedding<float,float>::Config_t config;
    config.image_height=7;  // Not divisible by 4
    config.image_width=8;
    config.channels=3;
    config.patch_size=4;
    config.dim=6;
    config.use_cls_token=false;

    bool threw=false;
    try
    {
      CAIF_DevicePatchEmbedding<float,float> emb(config,stream);
    }
    catch(...)
    {
      threw=true;
    }

    bool passed=(threw==true);
    if(passed==false)
    {
      std::cout<<"  Expected exception for H=7, patch_size=4\n";
    }

    ReportResult("PatchEmbedding::PatchSizeValidation",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::PatchSizeValidation")
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    CAIF_CudaStream stream;

    CAIF_DevicePatchEmbedding<float,float>::Config_t config_no=TestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb_no(config_no,stream);
    const std::string desc_no=emb_no.Description();
    const std::string expected_no="PatchEmbedding(patch=4,ch=3,dim=6)";

    CAIF_DevicePatchEmbedding<float,float>::Config_t config_cls=TestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb_cls(config_cls,stream);
    const std::string desc_cls=emb_cls.Description();
    const std::string expected_cls="PatchEmbedding(patch=4,ch=3,dim=6,cls=true)";

    bool passed=(desc_no==expected_no&&desc_cls==expected_cls);
    if(desc_no!=expected_no)
    {
      std::cout<<"  No CLS: expected '"<<expected_no<<"', got '"<<desc_no<<"'\n";
    }
    if(desc_cls!=expected_cls)
    {
      std::cout<<"  CLS: expected '"<<expected_cls<<"', got '"<<desc_cls<<"'\n";
    }

    ReportResult("PatchEmbedding::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::Description")
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF_DevicePatchEmbedding<float,float> Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestForwardShape();
  TestForwardShapeCLS();
  TestForwardValues();
  TestForwardCLSValues();
  TestBackwardInputGrad();
  TestBackwardWeightGrad();
  TestBackwardCLSGrad();
  TestParameterCount();
  TestPatchSizeValidation();
  TestDescription();
#else
  std::cout<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

  return CAIF_TestHarness::FinalExitCode();
}
