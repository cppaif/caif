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
// Test: CAIF_DevicePatchEmbedding<float,float> forward/backward correctness.
//
// Tests cover forward shape (with/without CLS), CPU reference parity,
// CLS token position, finite-difference input/weight/CLS gradients,
// parameter count, patch-size validation, and description string.
//------------------------------------------------------------------------------
#include "caif_device_patch_embedding.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_cpu_reference/caif_cpu_patch_extract.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
#include "caif_cpu_reference/caif_cpu_linear.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_patchmb_test_img_h=8;
constexpr uint32_t g_caif_patchmb_test_img_w=8;
constexpr uint32_t g_caif_patchmb_test_channels=3;
constexpr uint32_t g_caif_patchmb_test_patch_size=4;
constexpr uint32_t g_caif_patchmb_test_dim=6;
constexpr uint32_t g_caif_patchmb_test_num_patches_h=2;
constexpr uint32_t g_caif_patchmb_test_num_patches_w=2;
constexpr uint32_t g_caif_patchmb_test_num_patches=4;
constexpr uint32_t g_caif_patchmb_test_patch_flat=g_caif_patchmb_test_patch_size*
                                                    g_caif_patchmb_test_patch_size*
                                                    g_caif_patchmb_test_channels;
constexpr float g_caif_patchmb_test_input_fill=0.5f;
constexpr float g_caif_patchmb_test_input_scale=0.01f;
constexpr float g_caif_patchmb_test_input_offset=-0.5f;
constexpr float g_caif_patchmb_test_fd_h=1e-3f;
constexpr float g_caif_patchmb_test_grad_tol=5e-2f;
constexpr float g_caif_patchmb_test_float_tol=1e-4f;
// Subset strides to keep finite-difference checks tractable
constexpr int g_caif_patchmb_test_input_check_stride=16;
constexpr int g_caif_patchmb_test_weight_check_stride=12;

//------------------------------------------------------------------------------
// Patch embedding tests.
//------------------------------------------------------------------------------
class CAIF_PatchEmbeddingTests
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DevicePatchEmbeddingConfig MakeTestConfig(bool cls);

    static void TestForwardShape();
    static void TestForwardShapeCLS();
    static void TestForwardValues();
    static void TestForwardCLSValues();
    static void TestBackwardInputGrad();
    static void TestBackwardWeightGrad();
    static void TestBackwardCLSGrad();
    static void TestParameterCount();
    static void TestPatchSizeValidation();
    static void TestDescription();
};

CAIF_DevicePatchEmbeddingConfig CAIF_PatchEmbeddingTests::MakeTestConfig(
  const bool cls)
{
  CAIF_DevicePatchEmbeddingConfig config(g_caif_patchmb_test_img_h,
                                         g_caif_patchmb_test_img_w,
                                         g_caif_patchmb_test_channels,
                                         g_caif_patchmb_test_patch_size,
                                         g_caif_patchmb_test_dim,
                                         cls);
  return config;
}

//------------------------------------------------------------------------------
// Test 1: Forward shape without CLS
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestForwardShape()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePatchEmbeddingConfig config=MakeTestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    const size_t n_input=batch*g_caif_patchmb_test_img_h*g_caif_patchmb_test_img_w*
                         g_caif_patchmb_test_channels;
    std::vector<float> input_data(n_input,g_caif_patchmb_test_input_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    const auto &shape=output.Shape();

    // num_patches = (8/4)*(8/4) = 4
    bool passed=(shape.size()==3 &&
                 shape[0]==1 &&
                 shape[1]==g_caif_patchmb_test_num_patches &&
                 shape[2]==g_caif_patchmb_test_dim);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected [1,4,6], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          ISE_Out::Out()<<",";
        }
        ISE_Out::Out()<<shape[i];
      }
      ISE_Out::Out()<<"]\n";
    }

    CAIF_TestHarness::Report("PatchEmbedding::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Forward shape with CLS
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestForwardShapeCLS()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePatchEmbeddingConfig config=MakeTestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    const size_t n_input=batch*g_caif_patchmb_test_img_h*g_caif_patchmb_test_img_w*
                         g_caif_patchmb_test_channels;
    std::vector<float> input_data(n_input,g_caif_patchmb_test_input_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    const auto &shape=output.Shape();

    // num_patches+1 = 5
    bool passed=(shape.size()==3 &&
                 shape[0]==1 &&
                 shape[1]==g_caif_patchmb_test_num_patches+1 &&
                 shape[2]==g_caif_patchmb_test_dim);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected [1,5,6], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          ISE_Out::Out()<<",";
        }
        ISE_Out::Out()<<shape[i];
      }
      ISE_Out::Out()<<"]\n";
    }

    CAIF_TestHarness::Report("PatchEmbedding::ForwardShapeCLS",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardShapeCLS")
}

//------------------------------------------------------------------------------
// Test 3: Forward values match CPU reference (no CLS)
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestForwardValues()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePatchEmbeddingConfig config=MakeTestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    const size_t n_input=batch*g_caif_patchmb_test_img_h*g_caif_patchmb_test_img_w*
                         g_caif_patchmb_test_channels;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_patchmb_test_input_scale;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Read W_proj and b_proj
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();

    // CPU reference
    std::vector<float> patches(batch*g_caif_patchmb_test_num_patches*g_caif_patchmb_test_patch_flat);
    CAIF_CpuPatchExtract::Apply(input_data.data(),patches.data(),
                                batch,
                                g_caif_patchmb_test_img_h,
                                g_caif_patchmb_test_img_w,
                                g_caif_patchmb_test_channels,
                                g_caif_patchmb_test_patch_size,
                                g_caif_patchmb_test_num_patches_h,
                                g_caif_patchmb_test_num_patches_w);

    std::vector<float> projected(g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim);
    CAIF_CpuMatMul::Apply(patches.data(),host_w.Data(),projected.data(),
                          g_caif_patchmb_test_num_patches,
                          g_caif_patchmb_test_patch_flat,
                          g_caif_patchmb_test_dim);
    CAIF_CpuLinear::BiasAdd(projected.data(),host_b.Data(),
                            g_caif_patchmb_test_num_patches,
                            g_caif_patchmb_test_dim);

    bool passed=true;
    for(size_t i=0;i<g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim;++i)
    {
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],projected[i],g_caif_patchmb_test_float_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": got "
                     <<host_output.Data()[i]
                     <<" expected "
                     <<projected[i]
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("PatchEmbedding::ForwardValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardValues")
}

//------------------------------------------------------------------------------
// Test 4: CLS token at position 0, patches at 1..N
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestForwardCLSValues()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePatchEmbeddingConfig config=MakeTestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);

    const size_t n_input=batch*g_caif_patchmb_test_img_h*g_caif_patchmb_test_img_w*
                         g_caif_patchmb_test_channels;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_patchmb_test_input_scale;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
      stream);

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
    for(uint32_t d=0;d<g_caif_patchmb_test_dim;++d)
    {
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[d],host_cls.Data()[d],g_caif_patchmb_test_float_tol)==false)
      {
        ISE_Out::Out()<<"  CLS mismatch at d="
                     <<d
                     <<": got "
                     <<host_output.Data()[d]
                     <<" expected "
                     <<host_cls.Data()[d]
                     <<"\n";
        passed=false;
        break;
      }
    }

    // Positions 1..4 should be projected patches
    if(passed==true)
    {
      std::vector<float> patches(g_caif_patchmb_test_num_patches*g_caif_patchmb_test_patch_flat);
      CAIF_CpuPatchExtract::Apply(input_data.data(),patches.data(),
                                  batch,
                                  g_caif_patchmb_test_img_h,
                                  g_caif_patchmb_test_img_w,
                                  g_caif_patchmb_test_channels,
                                  g_caif_patchmb_test_patch_size,
                                  g_caif_patchmb_test_num_patches_h,
                                  g_caif_patchmb_test_num_patches_w);
      std::vector<float> projected(g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim);
      CAIF_CpuMatMul::Apply(patches.data(),host_w.Data(),projected.data(),
                            g_caif_patchmb_test_num_patches,
                            g_caif_patchmb_test_patch_flat,
                            g_caif_patchmb_test_dim);
      CAIF_CpuLinear::BiasAdd(projected.data(),host_b.Data(),
                              g_caif_patchmb_test_num_patches,
                              g_caif_patchmb_test_dim);

      for(uint32_t p=0;p<g_caif_patchmb_test_num_patches && passed==true;++p)
      {
        for(uint32_t d=0;d<g_caif_patchmb_test_dim && passed==true;++d)
        {
          // Output position p+1
          const float actual=host_output.Data()[(p+1)*g_caif_patchmb_test_dim+d];
          const float expected=projected[p*g_caif_patchmb_test_dim+d];
          if(CAIF_TestHarness::FloatEqual(actual,expected,g_caif_patchmb_test_float_tol)==false)
          {
            ISE_Out::Out()<<"  Patch "
                         <<p
                         <<" dim "
                         <<d
                         <<": got "
                         <<actual
                         <<" expected "
                         <<expected
                         <<"\n";
            passed=false;
          }
        }
      }
    }

    CAIF_TestHarness::Report("PatchEmbedding::ForwardCLSValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ForwardCLSValues")
}

//------------------------------------------------------------------------------
// Test 5: Backward input gradient (finite-difference)
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestBackwardInputGrad()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePatchEmbeddingConfig config=MakeTestConfig(false);

    const size_t input_size=batch*g_caif_patchmb_test_img_h*g_caif_patchmb_test_img_w*
                            g_caif_patchmb_test_channels;
    std::vector<float> input_data(input_size);
    for(size_t i=0;i<input_size;++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_patchmb_test_input_scale+
                    g_caif_patchmb_test_input_offset;
    }

    // Read weights for consistent perturbed runs
    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    const size_t w_size=emb.ParameterTensor(0).TotalElements();
    const size_t b_size=emb.ParameterTensor(1).TotalElements();

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);

    std::vector<float> grad_ones(batch*g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {batch,g_caif_patchmb_test_num_patches,g_caif_patchmb_test_dim},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=emb.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Finite-difference check on a subset of input elements
    bool passed=true;
    for(size_t i=0;i<input_size && passed==true;i+=g_caif_patchmb_test_input_check_stride)
    {
      std::vector<float> inp_plus(input_data);
      std::vector<float> inp_minus(input_data);
      inp_plus[i]+=g_caif_patchmb_test_fd_h;
      inp_minus[i]-=g_caif_patchmb_test_fd_h;

      // f(x+h)
      CAIF_DevicePatchEmbedding<float,float> emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
        inp_plus.data(),
        {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
        stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // f(x-h)
      CAIF_DevicePatchEmbedding<float,float> emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
        inp_minus.data(),
        {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
        stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_patchmb_test_fd_h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>g_caif_patchmb_test_grad_tol)
      {
        ISE_Out::Out()<<"  dx mismatch at "
                     <<i
                     <<": analytical="
                     <<analytical
                     <<" numerical="
                     <<numerical
                     <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("PatchEmbedding::BackwardInputGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::BackwardInputGrad")
}

//------------------------------------------------------------------------------
// Test 6: Backward weight gradient (finite-difference on W_proj)
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestBackwardWeightGrad()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePatchEmbeddingConfig config=MakeTestConfig(false);

    const size_t n_input=batch*g_caif_patchmb_test_img_h*g_caif_patchmb_test_img_w*
                         g_caif_patchmb_test_channels;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_patchmb_test_input_scale+
                    g_caif_patchmb_test_input_offset;
    }

    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    const size_t w_size=host_w.TotalElements();
    const size_t b_size=host_b.TotalElements();
    std::vector<float> w_data(host_w.Data(),host_w.Data()+w_size);

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    std::vector<float> grad_ones(batch*g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {batch,g_caif_patchmb_test_num_patches,g_caif_patchmb_test_dim},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    emb.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_w=emb.GradientTensor(0).ToHost();

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Finite-difference on subset of W_proj elements
    bool passed=true;
    for(size_t i=0;i<w_size && passed==true;i+=g_caif_patchmb_test_weight_check_stride)
    {
      std::vector<float> w_plus(w_data);
      std::vector<float> w_minus(w_data);
      w_plus[i]+=g_caif_patchmb_test_fd_h;
      w_minus[i]-=g_caif_patchmb_test_fd_h;

      CAIF_DevicePatchEmbedding<float,float> emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(w_plus.data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
        input_data.data(),
        {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
        stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DevicePatchEmbedding<float,float> emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(w_minus.data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
        input_data.data(),
        {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
        stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*g_caif_patchmb_test_num_patches*g_caif_patchmb_test_dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_patchmb_test_fd_h);
      const float analytical=host_grad_w.Data()[i];

      if(std::fabs(numerical-analytical)>g_caif_patchmb_test_grad_tol)
      {
        ISE_Out::Out()<<"  dW mismatch at "
                     <<i
                     <<": analytical="
                     <<analytical
                     <<" numerical="
                     <<numerical
                     <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("PatchEmbedding::BackwardWeightGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::BackwardWeightGrad")
}

//------------------------------------------------------------------------------
// Test 7: Backward CLS gradient (finite-difference on cls_token)
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestBackwardCLSGrad()
{
  try
  {
    constexpr uint32_t batch=1;
    const uint32_t out_seq=g_caif_patchmb_test_num_patches+1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePatchEmbeddingConfig config=MakeTestConfig(true);

    const size_t n_input=batch*g_caif_patchmb_test_img_h*g_caif_patchmb_test_img_w*
                         g_caif_patchmb_test_channels;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_patchmb_test_input_scale+
                    g_caif_patchmb_test_input_offset;
    }

    CAIF_DevicePatchEmbedding<float,float> emb(config,stream);
    CAIF_HostTensor host_w=emb.ParameterTensor(0).ToHost();
    CAIF_HostTensor host_b=emb.ParameterTensor(1).ToHost();
    CAIF_HostTensor host_cls=emb.ParameterTensor(2).ToHost();
    const size_t w_size=host_w.TotalElements();
    const size_t b_size=host_b.TotalElements();
    std::vector<float> cls_data(host_cls.Data(),host_cls.Data()+g_caif_patchmb_test_dim);

    // Analytical backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=emb.Forward(input,ctx);
    std::vector<float> grad_ones(batch*out_seq*g_caif_patchmb_test_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_ones.data(),
      {batch,out_seq,g_caif_patchmb_test_dim},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    emb.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_cls=emb.GradientTensor(2).ToHost();

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    bool passed=true;
    for(uint32_t d=0;d<g_caif_patchmb_test_dim && passed==true;++d)
    {
      std::vector<float> cls_plus(cls_data);
      std::vector<float> cls_minus(cls_data);
      cls_plus[d]+=g_caif_patchmb_test_fd_h;
      cls_minus[d]-=g_caif_patchmb_test_fd_h;

      CAIF_DevicePatchEmbedding<float,float> emb_p(config,stream);
      emb_p.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_p.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      emb_p.ParameterTensor(2).CopyFromHost(cls_plus.data(),g_caif_patchmb_test_dim);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
        input_data.data(),
        {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
        stream);
      CAIF_DeviceTensor out_p=emb_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*out_seq*g_caif_patchmb_test_dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DevicePatchEmbedding<float,float> emb_m(config,stream);
      emb_m.ParameterTensor(0).CopyFromHost(host_w.Data(),w_size);
      emb_m.ParameterTensor(1).CopyFromHost(host_b.Data(),b_size);
      emb_m.ParameterTensor(2).CopyFromHost(cls_minus.data(),g_caif_patchmb_test_dim);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
        input_data.data(),
        {batch,g_caif_patchmb_test_img_h,g_caif_patchmb_test_img_w,g_caif_patchmb_test_channels},
        stream);
      CAIF_DeviceTensor out_m=emb_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*out_seq*g_caif_patchmb_test_dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_patchmb_test_fd_h);
      const float analytical=host_grad_cls.Data()[d];

      if(std::fabs(numerical-analytical)>g_caif_patchmb_test_grad_tol)
      {
        ISE_Out::Out()<<"  dCLS mismatch at d="
                     <<d
                     <<": analytical="
                     <<analytical
                     <<" numerical="
                     <<numerical
                     <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("PatchEmbedding::BackwardCLSGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::BackwardCLSGrad")
}

//------------------------------------------------------------------------------
// Test 8: Parameter count
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestParameterCount()
{
  try
  {
    CAIF_CudaStream stream;

    // Without CLS
    const CAIF_DevicePatchEmbeddingConfig config_no=MakeTestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb_no(config_no,stream);
    bool passed=true;
    if(emb_no.ParameterTensorCount()!=2)
    {
      ISE_Out::Out()<<"  No CLS: ParameterTensorCount expected 2, got "
                   <<emb_no.ParameterTensorCount()
                   <<"\n";
      passed=false;
    }

    // With CLS
    const CAIF_DevicePatchEmbeddingConfig config_cls=MakeTestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb_cls(config_cls,stream);
    if(emb_cls.ParameterTensorCount()!=3)
    {
      ISE_Out::Out()<<"  CLS: ParameterTensorCount expected 3, got "
                   <<emb_cls.ParameterTensorCount()
                   <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("PatchEmbedding::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 9: Patch size validation (rejects H % patch_size != 0)
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestPatchSizeValidation()
{
  try
  {
    CAIF_CudaStream stream;

    // image_height=7 is not divisible by patch_size=4 (validation must throw)
    CAIF_DevicePatchEmbeddingConfig config(7,8,3,4,6,false);

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
      ISE_Out::Out()<<"  Expected exception for H=7, patch_size=4\n";
    }

    CAIF_TestHarness::Report("PatchEmbedding::PatchSizeValidation",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::PatchSizeValidation")
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
void CAIF_PatchEmbeddingTests::TestDescription()
{
  try
  {
    CAIF_CudaStream stream;

    const CAIF_DevicePatchEmbeddingConfig config_no=MakeTestConfig(false);
    CAIF_DevicePatchEmbedding<float,float> emb_no(config_no,stream);
    const std::string desc_no=emb_no.Description();
    const std::string expected_no="PatchEmbedding(patch=4,ch=3,dim=6)";

    const CAIF_DevicePatchEmbeddingConfig config_cls=MakeTestConfig(true);
    CAIF_DevicePatchEmbedding<float,float> emb_cls(config_cls,stream);
    const std::string desc_cls=emb_cls.Description();
    const std::string expected_cls="PatchEmbedding(patch=4,ch=3,dim=6,cls=true)";

    bool passed=(desc_no==expected_no && desc_cls==expected_cls);
    if(desc_no!=expected_no)
    {
      ISE_Out::Out()<<"  No CLS: expected '"
                   <<expected_no
                   <<"', got '"
                   <<desc_no
                   <<"'\n";
    }
    if(desc_cls!=expected_cls)
    {
      ISE_Out::Out()<<"  CLS: expected '"
                   <<expected_cls
                   <<"', got '"
                   <<desc_cls
                   <<"'\n";
    }

    CAIF_TestHarness::Report("PatchEmbedding::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PatchEmbedding::Description")
}

void CAIF_PatchEmbeddingTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DevicePatchEmbedding<float,float> Tests ==="
               <<"\n\n";
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
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_PatchEmbeddingTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
               <<"\n";
  return 0;
#endif
}
