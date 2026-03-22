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
// CPU reference: extract patches, project, bias add
//------------------------------------------------------------------------------
static void CpuExtractPatches(const float *input,
                               float *output,
                               int batch,
                               int height,
                               int width,
                               int channels,
                               int patch_size,
                               int num_patches_h,
                               int num_patches_w)
{
  const int patch_flat=patch_size*patch_size*channels;
  const int num_patches=num_patches_h*num_patches_w;
  for(int b=0;b<batch;++b)
  {
    for(int ph=0;ph<num_patches_h;++ph)
    {
      for(int pw=0;pw<num_patches_w;++pw)
      {
        const int patch_idx=ph*num_patches_w+pw;
        const int out_row=b*num_patches+patch_idx;
        for(int lh=0;lh<patch_size;++lh)
        {
          for(int lw=0;lw<patch_size;++lw)
          {
            const int gh=ph*patch_size+lh;
            const int gw=pw*patch_size+lw;
            for(int c=0;c<channels;++c)
            {
              const int flat_pos=(lh*patch_size+lw)*channels+c;
              const int in_idx=((b*height+gh)*width+gw)*channels+c;
              output[out_row*patch_flat+flat_pos]=input[in_idx];
            }
          }
        }
      }
    }
  }
}

static void CpuMatMul(const float *a,
                       const float *b,
                       float *c,
                       int m,
                       int k,
                       int n)
{
  for(int i=0;i<m;++i)
  {
    for(int j=0;j<n;++j)
    {
      float sum=0.0f;
      for(int p=0;p<k;++p)
      {
        sum+=a[i*k+p]*b[p*n+j];
      }
      c[i*n+j]=sum;
    }
  }
}

static void CpuBiasAdd(float *data,const float *bias,int rows,int cols)
{
  for(int r=0;r<rows;++r)
  {
    for(int c=0;c<cols;++c)
    {
      data[r*cols+c]+=bias[c];
    }
  }
}

// Test config used across most tests
static CAIF_DevicePatchEmbedding::Config_t TestConfig(bool cls)
{
  CAIF_DevicePatchEmbedding::Config_t config;
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
    CAIF_DevicePatchEmbedding::Config_t config=TestConfig(false);
    CAIF_DevicePatchEmbedding emb(config,stream);

    // Input: [1,8,8,3]
    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    std::vector<float> input_data(batch*h*w*c,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);

    CAIF_DeviceTensor output=emb.Forward(input,false);
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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward shape with CLS
//------------------------------------------------------------------------------
static void TestForwardShapeCLS()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePatchEmbedding::Config_t config=TestConfig(true);
    CAIF_DevicePatchEmbedding emb(config,stream);

    constexpr uint32_t batch=1;
    constexpr uint32_t h=8;
    constexpr uint32_t w=8;
    constexpr uint32_t c=3;
    std::vector<float> input_data(batch*h*w*c,0.5f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);

    CAIF_DeviceTensor output=emb.Forward(input,false);
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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::ForwardShapeCLS",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Forward values match CPU reference (no CLS)
//------------------------------------------------------------------------------
static void TestForwardValues()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePatchEmbedding::Config_t config=TestConfig(false);
    CAIF_DevicePatchEmbedding emb(config,stream);

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

    CAIF_DeviceTensor output=emb.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // Read W_proj and b_proj
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();

    // CPU reference
    std::vector<float> patches(batch*num_patches*patch_flat);
    CpuExtractPatches(input_data.data(),patches.data(),
                       batch,h,w,c,patch_size,num_patches_h,num_patches_w);

    std::vector<float> projected(num_patches*dim);
    CpuMatMul(patches.data(),host_w.Data(),projected.data(),
               num_patches,patch_flat,dim);
    CpuBiasAdd(projected.data(),host_b.Data(),num_patches,dim);

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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::ForwardValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: CLS token at position 0, patches at 1..N
//------------------------------------------------------------------------------
static void TestForwardCLSValues()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePatchEmbedding::Config_t config=TestConfig(true);
    CAIF_DevicePatchEmbedding emb(config,stream);

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

    CAIF_DeviceTensor output=emb.Forward(input,false);
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
      CpuExtractPatches(input_data.data(),patches.data(),
                         batch,h,w,c,patch_size,num_patches_h,num_patches_w);
      std::vector<float> projected(num_patches*dim);
      CpuMatMul(patches.data(),host_w.Data(),projected.data(),
                 num_patches,patch_flat,dim);
      CpuBiasAdd(projected.data(),host_b.Data(),num_patches,dim);

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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::ForwardCLSValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Backward input gradient (finite-difference)
//------------------------------------------------------------------------------
static void TestBackwardInputGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePatchEmbedding::Config_t config=TestConfig(false);

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
    CAIF_DevicePatchEmbedding emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    const size_t w_size=emb.ParameterTensor(0).TotalElements();
    const size_t b_size=emb.ParameterTensor(1).TotalElements();

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);
    CAIF_DeviceTensor output=emb.Forward(input,true);

    std::vector<float> grad_ones(batch*num_patches*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),
                                {batch,num_patches,dim},stream);
    CAIF_DeviceTensor grad_input=emb.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

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
      CAIF_DevicePatchEmbedding emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               inp_plus.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*num_patches*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // f(x-h)
      CAIF_DevicePatchEmbedding emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               inp_minus.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,false);
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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::BackwardInputGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Backward weight gradient (finite-difference on W_proj)
//------------------------------------------------------------------------------
static void TestBackwardWeightGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePatchEmbedding::Config_t config=TestConfig(false);

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

    CAIF_DevicePatchEmbedding emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    const size_t w_size=host_w.TotalElements();
    const size_t b_size=host_b.TotalElements();
    std::vector<float> w_data(host_w.Data(),host_w.Data()+w_size);

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);
    CAIF_DeviceTensor output=emb.Forward(input,true);
    std::vector<float> grad_ones(batch*num_patches*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),
                                {batch,num_patches,dim},stream);
    emb.Backward(grad_out);
    CAIF_HostTensor host_grad_w=emb.GradientTensor(0).ToHost();

    // Finite-difference on subset of W_proj elements
    bool passed=true;
    constexpr int check_stride=12;
    for(size_t i=0;i<w_size&&passed==true;i+=check_stride)
    {
      std::vector<float> w_plus(w_data);
      std::vector<float> w_minus(w_data);
      w_plus[i]+=fd_h;
      w_minus[i]-=fd_h;

      CAIF_DevicePatchEmbedding emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(w_plus.data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*num_patches*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DevicePatchEmbedding emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(w_minus.data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,false);
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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::BackwardWeightGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Backward CLS gradient (finite-difference on cls_token)
//------------------------------------------------------------------------------
static void TestBackwardCLSGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePatchEmbedding::Config_t config=TestConfig(true);

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

    CAIF_DevicePatchEmbedding emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    CAIF_HostTensor host_cls=emb.ParameterTensor(2).ToHost();
    const size_t w_size=host_w.TotalElements();
    const size_t b_size=host_b.TotalElements();
    std::vector<float> cls_data(host_cls.Data(),host_cls.Data()+dim);

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,h,w,c},stream);
    CAIF_DeviceTensor output=emb.Forward(input,true);
    std::vector<float> grad_ones(batch*out_seq*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),
                                {batch,out_seq,dim},stream);
    emb.Backward(grad_out);
    CAIF_HostTensor host_grad_cls=emb.GradientTensor(2).ToHost();

    bool passed=true;
    for(uint32_t d=0;d<dim&&passed==true;++d)
    {
      std::vector<float> cls_plus(cls_data);
      std::vector<float> cls_minus(cls_data);
      cls_plus[d]+=fd_h;
      cls_minus[d]-=fd_h;

      CAIF_DevicePatchEmbedding emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      emb_p.ParameterTensor(2).CopyFromHost(cls_plus.data(),dim);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*out_seq*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DevicePatchEmbedding emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      emb_m.ParameterTensor(2).CopyFromHost(cls_minus.data(),dim);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_data.data(),{batch,h,w,c},stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,false);
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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::BackwardCLSGrad",false);
  }
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
    CAIF_DevicePatchEmbedding::Config_t config_no=TestConfig(false);
    CAIF_DevicePatchEmbedding emb_no(config_no,stream);
    bool passed=true;
    if(emb_no.ParameterTensorCount()!=2)
    {
      std::cout<<"  No CLS: ParameterTensorCount expected 2, got "
               <<emb_no.ParameterTensorCount()<<"\n";
      passed=false;
    }

    // With CLS
    CAIF_DevicePatchEmbedding::Config_t config_cls=TestConfig(true);
    CAIF_DevicePatchEmbedding emb_cls(config_cls,stream);
    if(emb_cls.ParameterTensorCount()!=3)
    {
      std::cout<<"  CLS: ParameterTensorCount expected 3, got "
               <<emb_cls.ParameterTensorCount()<<"\n";
      passed=false;
    }

    ReportResult("PatchEmbedding::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: Patch size validation (rejects H % patch_size != 0)
//------------------------------------------------------------------------------
static void TestPatchSizeValidation()
{
  try
  {
    CAIF_CudaStream stream;

    CAIF_DevicePatchEmbedding::Config_t config;
    config.image_height=7;  // Not divisible by 4
    config.image_width=8;
    config.channels=3;
    config.patch_size=4;
    config.dim=6;
    config.use_cls_token=false;

    bool threw=false;
    try
    {
      CAIF_DevicePatchEmbedding emb(config,stream);
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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::PatchSizeValidation",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    CAIF_CudaStream stream;

    CAIF_DevicePatchEmbedding::Config_t config_no=TestConfig(false);
    CAIF_DevicePatchEmbedding emb_no(config_no,stream);
    const std::string desc_no=emb_no.Description();
    const std::string expected_no="PatchEmbedding(patch=4,ch=3,dim=6)";

    CAIF_DevicePatchEmbedding::Config_t config_cls=TestConfig(true);
    CAIF_DevicePatchEmbedding emb_cls(config_cls,stream);
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
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PatchEmbedding::Description",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF_DevicePatchEmbedding Tests ===\n\n";

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

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  if(g_tests_failed>0)
  {
    return 1;
  }
  return 0;
}
