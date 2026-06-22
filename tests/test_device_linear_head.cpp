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
// Test: CAIF_DeviceLinearHead<float,float> forward/backward correctness.
//
// Tests cover forward shape (with/without bias), CPU reference parity,
// weight-tied forward, finite-difference input/weight gradients,
// weight-tied backward, and parameter count.
//------------------------------------------------------------------------------
#include "caif_device_linear_head.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_cpu_reference/caif_cpu_linear.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_linhead_test_batch_seq=2;
constexpr uint32_t g_caif_linhead_test_seq_len=4;
constexpr uint32_t g_caif_linhead_test_input_dim_shape=64;
constexpr uint32_t g_caif_linhead_test_output_dim_shape=128;
constexpr uint32_t g_caif_linhead_test_small_input=4;
constexpr uint32_t g_caif_linhead_test_small_output=3;
constexpr uint32_t g_caif_linhead_test_tied_input=4;
constexpr uint32_t g_caif_linhead_test_tied_output=6;
constexpr uint32_t g_caif_linhead_test_nobias_input=32;
constexpr uint32_t g_caif_linhead_test_nobias_output=16;
constexpr uint32_t g_caif_linhead_test_param_input=8;
constexpr uint32_t g_caif_linhead_test_param_output=16;
constexpr float g_caif_linhead_test_fill=0.5f;
constexpr float g_caif_linhead_test_input_scale=0.1f;
constexpr float g_caif_linhead_test_tied_scale=0.05f;
constexpr float g_caif_linhead_test_tied_offset=0.1f;
constexpr float g_caif_linhead_test_fd_h=1e-3f;
constexpr float g_caif_linhead_test_fd_tol=5e-2f;
constexpr float g_caif_linhead_test_fwd_tol=1e-3f;
constexpr float g_caif_linhead_test_tied_fill=0.1f;
constexpr float g_caif_linhead_test_sum_min=0.01f;
constexpr int g_caif_linhead_test_fd_count=4;

//------------------------------------------------------------------------------
// Linear head tests.
//------------------------------------------------------------------------------
class CAIF_LinearHeadTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestForwardShape();
    static void TestForwardShapeNoBias();
    static void TestForwardValues();
    static void TestWeightTiedForward();
    static void TestBackwardInputGrad();
    static void TestBackwardWeightGrad();
    static void TestWeightTiedBackwardGrad();
    static void TestParameterCount();
};

//------------------------------------------------------------------------------
// Test 1: Forward output shape
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_input_dim_shape,
                                             g_caif_linhead_test_output_dim_shape,
                                             true};
    CAIF_DeviceLinearHead<float,float> head(config,stream);

    const size_t n_input=g_caif_linhead_test_batch_seq*
                         g_caif_linhead_test_seq_len*
                         g_caif_linhead_test_input_dim_shape;
    std::vector<float> input_data(n_input,g_caif_linhead_test_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_seq_len,g_caif_linhead_test_input_dim_shape},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=head.Forward(input,ctx);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==3 &&
                 shape[0]==g_caif_linhead_test_batch_seq &&
                 shape[1]==g_caif_linhead_test_seq_len &&
                 shape[2]==g_caif_linhead_test_output_dim_shape);

    CAIF_TestHarness::Report("LinearHead::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Forward shape no bias
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestForwardShapeNoBias()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_nobias_input,
                                             g_caif_linhead_test_nobias_output,
                                             false};
    CAIF_DeviceLinearHead<float,float> head(config,stream);

    const size_t n_input=g_caif_linhead_test_batch_seq*g_caif_linhead_test_nobias_input;
    std::vector<float> input_data(n_input,g_caif_linhead_test_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_nobias_input},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=head.Forward(input,ctx);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==2 &&
                 shape[0]==g_caif_linhead_test_batch_seq &&
                 shape[1]==g_caif_linhead_test_nobias_output);

    CAIF_TestHarness::Report("LinearHead::ForwardShapeNoBias",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::ForwardShapeNoBias")
}

//------------------------------------------------------------------------------
// Test 3: Forward values vs CPU reference
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestForwardValues()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_small_input,
                                             g_caif_linhead_test_small_output,
                                             true};
    CAIF_DeviceLinearHead<float,float> head(config,stream);

    // Get weight and bias
    std::vector<float> weight_data(g_caif_linhead_test_small_input*g_caif_linhead_test_small_output);
    std::vector<float> bias_data(g_caif_linhead_test_small_output);
    head.ParameterTensor(0).CopyToHost(weight_data.data());

    // Get bias from second parameter
    if(head.ParameterTensorCount()>1)
    {
      head.ParameterTensor(1).CopyToHost(bias_data.data());
    }

    // Create input
    const size_t n_input=g_caif_linhead_test_batch_seq*g_caif_linhead_test_small_input;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_linhead_test_input_scale;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_small_input},
      stream);

    // GPU forward
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=head.Forward(input,ctx);
    std::vector<float> gpu_output(g_caif_linhead_test_batch_seq*g_caif_linhead_test_small_output);
    output.CopyToHost(gpu_output.data());

    // CPU reference
    std::vector<float> cpu_output(g_caif_linhead_test_batch_seq*g_caif_linhead_test_small_output);
    CAIF_CpuLinear::Apply(input_data.data(),weight_data.data(),bias_data.data(),
                          cpu_output.data(),
                          g_caif_linhead_test_batch_seq,
                          g_caif_linhead_test_small_input,
                          g_caif_linhead_test_small_output,
                          true);

    bool passed=true;
    for(size_t i=0;i<cpu_output.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(gpu_output[i],cpu_output[i],g_caif_linhead_test_fwd_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": GPU="
                     <<gpu_output[i]
                     <<", CPU="
                     <<cpu_output[i]
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("LinearHead::ForwardValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::ForwardValues")
}

//------------------------------------------------------------------------------
// Test 4: Weight-tied forward
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestWeightTiedForward()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Create "embedding table" - shape [output_dim, input_dim]
    const size_t n_emb=g_caif_linhead_test_tied_output*g_caif_linhead_test_tied_input;
    std::vector<float> emb_data(n_emb);
    for(size_t i=0;i<emb_data.size();++i)
    {
      emb_data[i]=static_cast<float>(i)*g_caif_linhead_test_tied_scale;
    }
    CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
      emb_data.data(),
      {g_caif_linhead_test_tied_output,g_caif_linhead_test_tied_input},
      stream);
    CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros(
      {g_caif_linhead_test_tied_output,g_caif_linhead_test_tied_input},
      stream);

    // Create tied linear head
    const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_tied_input,
                                             g_caif_linhead_test_tied_output,
                                             true};
    CAIF_DeviceLinearHead<float,float> head(config,embedding,embedding_grad,stream);

    // Create input
    const size_t n_input=g_caif_linhead_test_batch_seq*g_caif_linhead_test_tied_input;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_linhead_test_input_scale+g_caif_linhead_test_tied_offset;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_tied_input},
      stream);

    // Get bias
    std::vector<float> bias_data(g_caif_linhead_test_tied_output,0.0f);
    if(head.ParameterTensorCount()>0)
    {
      head.ParameterTensor(0).CopyToHost(bias_data.data());
    }

    // GPU forward
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=head.Forward(input,ctx);
    std::vector<float> gpu_output(g_caif_linhead_test_batch_seq*g_caif_linhead_test_tied_output);
    output.CopyToHost(gpu_output.data());

    // CPU reference with transposed weight
    std::vector<float> cpu_output(g_caif_linhead_test_batch_seq*g_caif_linhead_test_tied_output);
    CAIF_CpuLinear::ApplyTranspose(input_data.data(),emb_data.data(),bias_data.data(),
                                   cpu_output.data(),
                                   g_caif_linhead_test_batch_seq,
                                   g_caif_linhead_test_tied_input,
                                   g_caif_linhead_test_tied_output,
                                   true);

    bool passed=true;
    for(size_t i=0;i<cpu_output.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(gpu_output[i],cpu_output[i],g_caif_linhead_test_fwd_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": GPU="
                     <<gpu_output[i]
                     <<", CPU="
                     <<cpu_output[i]
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("LinearHead::WeightTiedForward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::WeightTiedForward")
}

//------------------------------------------------------------------------------
// Test 5: Backward input gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestBackwardInputGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_small_input,
                                             g_caif_linhead_test_small_output,
                                             true};
    CAIF_DeviceLinearHead<float,float> head(config,stream);

    // Create input
    const size_t n_input=g_caif_linhead_test_batch_seq*g_caif_linhead_test_small_input;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_linhead_test_input_scale+g_caif_linhead_test_tied_offset;
    }

    // Forward and backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_small_input},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    head.Forward(input,ctx);

    // Create grad_output (all ones for sum reduction)
    const size_t n_out=g_caif_linhead_test_batch_seq*g_caif_linhead_test_small_output;
    std::vector<float> grad_out_data(n_out,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
      grad_out_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_small_output},
      stream);

    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=head.Backward(grad_output,ctx);
    std::vector<float> analytical_grad(n_input);
    grad_input.CopyToHost(analytical_grad.data());

    // Finite difference for a few elements
    bool passed=true;
    for(int idx=0;idx<g_caif_linhead_test_fd_count;++idx)
    {
      // f(x+h)
      std::vector<float> input_plus=input_data;
      input_plus[idx]+=g_caif_linhead_test_fd_h;
      CAIF_DeviceTensor inp_plus=CAIF_DeviceTensor::FromHostData(
        input_plus.data(),
        {g_caif_linhead_test_batch_seq,g_caif_linhead_test_small_input},
        stream);
      ctx.SetTraining(false);
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out_plus=head.Forward(inp_plus,ctx);
      std::vector<float> out_plus_data(n_out);
      out_plus.CopyToHost(out_plus_data.data());
      float sum_plus=0.0f;
      for(size_t k=0;k<out_plus_data.size();++k)
      {
        sum_plus+=out_plus_data[k];
      }

      // f(x-h)
      std::vector<float> input_minus=input_data;
      input_minus[idx]-=g_caif_linhead_test_fd_h;
      CAIF_DeviceTensor inp_minus=CAIF_DeviceTensor::FromHostData(
        input_minus.data(),
        {g_caif_linhead_test_batch_seq,g_caif_linhead_test_small_input},
        stream);
      CAIF_DeviceTensor out_minus=head.Forward(inp_minus,ctx);
      std::vector<float> out_minus_data(n_out);
      out_minus.CopyToHost(out_minus_data.data());
      float sum_minus=0.0f;
      for(size_t k=0;k<out_minus_data.size();++k)
      {
        sum_minus+=out_minus_data[k];
      }

      const float numerical_grad=(sum_plus-sum_minus)/(2.0f*g_caif_linhead_test_fd_h);
      const float diff=std::fabs(analytical_grad[idx]-numerical_grad);

      if(diff>g_caif_linhead_test_fd_tol)
      {
        ISE_Out::Out()<<"  idx="
                     <<idx
                     <<": analytical="
                     <<analytical_grad[idx]
                     <<", numerical="
                     <<numerical_grad
                     <<", diff="
                     <<diff
                     <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("LinearHead::BackwardInputGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::BackwardInputGrad")
}

//------------------------------------------------------------------------------
// Test 6: Backward weight gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestBackwardWeightGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_small_input,
                                             g_caif_linhead_test_small_output,
                                             true};

    // Create input
    const size_t n_input=g_caif_linhead_test_batch_seq*g_caif_linhead_test_small_input;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_linhead_test_input_scale+g_caif_linhead_test_tied_offset;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_small_input},
      stream);

    // Create grad_output
    const size_t n_out=g_caif_linhead_test_batch_seq*g_caif_linhead_test_small_output;
    std::vector<float> grad_out_data(n_out,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
      grad_out_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_small_output},
      stream);

    // Get analytical gradient
    CAIF_DeviceLinearHead<float,float> head(config,stream);
    head.ZeroGradients();
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    head.Forward(input,ctx);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    head.Backward(grad_output,ctx);

    const size_t n_weight=g_caif_linhead_test_small_input*g_caif_linhead_test_small_output;
    std::vector<float> analytical_grad(n_weight);
    head.GradientTensor(0).CopyToHost(analytical_grad.data());

    // Finite difference for a few weight elements
    bool passed=true;
    for(int idx=0;idx<g_caif_linhead_test_fd_count;++idx)
    {
      // Get current weight
      std::vector<float> weight_data(n_weight);
      head.ParameterTensor(0).CopyToHost(weight_data.data());

      // f(w+h)
      weight_data[idx]+=g_caif_linhead_test_fd_h;
      head.ParameterTensor(0).CopyFromHost(weight_data.data(),weight_data.size());
      ctx.SetTraining(false);
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out_plus=head.Forward(input,ctx);
      std::vector<float> out_plus_data(n_out);
      out_plus.CopyToHost(out_plus_data.data());
      float sum_plus=0.0f;
      for(size_t k=0;k<out_plus_data.size();++k)
      {
        sum_plus+=out_plus_data[k];
      }

      // f(w-h)
      weight_data[idx]-=2.0f*g_caif_linhead_test_fd_h;
      head.ParameterTensor(0).CopyFromHost(weight_data.data(),weight_data.size());
      CAIF_DeviceTensor out_minus=head.Forward(input,ctx);
      std::vector<float> out_minus_data(n_out);
      out_minus.CopyToHost(out_minus_data.data());
      float sum_minus=0.0f;
      for(size_t k=0;k<out_minus_data.size();++k)
      {
        sum_minus+=out_minus_data[k];
      }

      // Restore weight
      weight_data[idx]+=g_caif_linhead_test_fd_h;
      head.ParameterTensor(0).CopyFromHost(weight_data.data(),weight_data.size());

      const float numerical_grad=(sum_plus-sum_minus)/(2.0f*g_caif_linhead_test_fd_h);
      const float diff=std::fabs(analytical_grad[idx]-numerical_grad);

      if(diff>g_caif_linhead_test_fd_tol)
      {
        ISE_Out::Out()<<"  idx="
                     <<idx
                     <<": analytical="
                     <<analytical_grad[idx]
                     <<", numerical="
                     <<numerical_grad
                     <<", diff="
                     <<diff
                     <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("LinearHead::BackwardWeightGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::BackwardWeightGrad")
}

//------------------------------------------------------------------------------
// Test 7: Weight-tied backward gradient
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestWeightTiedBackwardGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Create "embedding table"
    const size_t n_emb=g_caif_linhead_test_tied_output*g_caif_linhead_test_tied_input;
    std::vector<float> emb_data(n_emb);
    for(size_t i=0;i<emb_data.size();++i)
    {
      emb_data[i]=static_cast<float>(i)*g_caif_linhead_test_tied_scale;
    }
    CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
      emb_data.data(),
      {g_caif_linhead_test_tied_output,g_caif_linhead_test_tied_input},
      stream);
    CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros(
      {g_caif_linhead_test_tied_output,g_caif_linhead_test_tied_input},
      stream);

    // Create tied linear head
    const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_tied_input,
                                             g_caif_linhead_test_tied_output,
                                             false};
    CAIF_DeviceLinearHead<float,float> head(config,embedding,embedding_grad,stream);

    // Create input
    const size_t n_input=g_caif_linhead_test_batch_seq*g_caif_linhead_test_tied_input;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_linhead_test_input_scale+g_caif_linhead_test_tied_offset;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_tied_input},
      stream);

    // Create grad_output
    const size_t n_out=g_caif_linhead_test_batch_seq*g_caif_linhead_test_tied_output;
    std::vector<float> grad_out_data(n_out,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(
      grad_out_data.data(),
      {g_caif_linhead_test_batch_seq,g_caif_linhead_test_tied_output},
      stream);

    // Forward and backward
    embedding_grad.Fill(0.0f);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    head.Forward(input,ctx);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    head.Backward(grad_output,ctx);

    // Check that embedding_grad is non-zero
    std::vector<float> emb_grad_data(n_emb);
    embedding_grad.CopyToHost(emb_grad_data.data());

    float sum=0.0f;
    for(size_t k=0;k<emb_grad_data.size();++k)
    {
      sum+=std::fabs(emb_grad_data[k]);
    }

    bool passed=(sum>g_caif_linhead_test_sum_min);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Tied weight gradient sum is too small: "
                   <<sum
                   <<"\n";
    }

    CAIF_TestHarness::Report("LinearHead::WeightTiedBackwardGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::WeightTiedBackwardGrad")
}

//------------------------------------------------------------------------------
// Test 8: Parameter count
//------------------------------------------------------------------------------
void CAIF_LinearHeadTests::TestParameterCount()
{
  try
  {
    CAIF_CudaStream stream;

    // Untied with bias: 2 params (W + b)
    {
      const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_param_input,
                                               g_caif_linhead_test_param_output,
                                               true};
      CAIF_DeviceLinearHead<float,float> head(config,stream);
      if(head.ParameterTensorCount()!=2)
      {
        ISE_Out::Out()<<"  Untied+bias: expected 2, got "
                     <<head.ParameterTensorCount()
                     <<"\n";
        CAIF_TestHarness::Report("LinearHead::ParameterCount",false);
        return;
      }
    }

    // Untied no bias: 1 param (W)
    {
      const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_param_input,
                                               g_caif_linhead_test_param_output,
                                               false};
      CAIF_DeviceLinearHead<float,float> head(config,stream);
      if(head.ParameterTensorCount()!=1)
      {
        ISE_Out::Out()<<"  Untied no bias: expected 1, got "
                     <<head.ParameterTensorCount()
                     <<"\n";
        CAIF_TestHarness::Report("LinearHead::ParameterCount",false);
        return;
      }
    }

    // Tied with bias: 1 param (b only)
    {
      const size_t n_emb=g_caif_linhead_test_param_output*g_caif_linhead_test_param_input;
      std::vector<float> emb_data(n_emb,g_caif_linhead_test_tied_fill);
      CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
        emb_data.data(),
        {g_caif_linhead_test_param_output,g_caif_linhead_test_param_input},
        stream);
      CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros(
        {g_caif_linhead_test_param_output,g_caif_linhead_test_param_input},
        stream);

      const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_param_input,
                                               g_caif_linhead_test_param_output,
                                               true};
      CAIF_DeviceLinearHead<float,float> head(config,embedding,embedding_grad,stream);
      if(head.ParameterTensorCount()!=1)
      {
        ISE_Out::Out()<<"  Tied+bias: expected 1, got "
                     <<head.ParameterTensorCount()
                     <<"\n";
        CAIF_TestHarness::Report("LinearHead::ParameterCount",false);
        return;
      }
    }

    // Tied no bias: 0 params
    {
      const size_t n_emb=g_caif_linhead_test_param_output*g_caif_linhead_test_param_input;
      std::vector<float> emb_data(n_emb,g_caif_linhead_test_tied_fill);
      CAIF_DeviceTensor embedding=CAIF_DeviceTensor::FromHostData(
        emb_data.data(),
        {g_caif_linhead_test_param_output,g_caif_linhead_test_param_input},
        stream);
      CAIF_DeviceTensor embedding_grad=CAIF_DeviceTensor::Zeros(
        {g_caif_linhead_test_param_output,g_caif_linhead_test_param_input},
        stream);

      const CAIF_DeviceLinearHeadConfig config{g_caif_linhead_test_param_input,
                                               g_caif_linhead_test_param_output,
                                               false};
      CAIF_DeviceLinearHead<float,float> head(config,embedding,embedding_grad,stream);
      if(head.ParameterTensorCount()!=0)
      {
        ISE_Out::Out()<<"  Tied no bias: expected 0, got "
                     <<head.ParameterTensorCount()
                     <<"\n";
        CAIF_TestHarness::Report("LinearHead::ParameterCount",false);
        return;
      }
    }

    CAIF_TestHarness::Report("LinearHead::ParameterCount",true);
  }
  CAIF_TEST_CATCH_BLOCK("LinearHead::ParameterCount")
}

void CAIF_LinearHeadTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF DeviceLinearHead Tests ==="
               <<"\n\n";
  TestForwardShape();
  TestForwardShapeNoBias();
  TestForwardValues();
  TestWeightTiedForward();
  TestBackwardInputGrad();
  TestBackwardWeightGrad();
  TestWeightTiedBackwardGrad();
  TestParameterCount();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_LinearHeadTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
               <<"\n";
  return 0;
#endif
}
