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
// Test: CAIF_DevicePositionalEncoding<float,float> learned and sinusoidal modes.
//
// Tests cover learned forward shape, learned forward values (input + pe_table),
// sinusoidal sin/cos formula parity, sinusoidal orthogonality, learned backward
// input gradient (identity), learned backward weight gradient (batch sum),
// parameter count, and description string.
//------------------------------------------------------------------------------
#include "caif_device_positional_encoding.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_posenc_test_max_seq=8;
constexpr uint32_t g_caif_posenc_test_dim=8;
constexpr uint32_t g_caif_posenc_test_batch=2;
constexpr uint32_t g_caif_posenc_test_seq=4;
constexpr float g_caif_posenc_test_input_fill=1.0f;
constexpr float g_caif_posenc_test_input_scale=0.1f;
constexpr float g_caif_posenc_test_grad_scale=0.1f;
constexpr float g_caif_posenc_test_grad_scale2=0.01f;
constexpr float g_caif_posenc_test_ortho_tol=1e-6f;
constexpr float g_caif_posenc_test_float_tol=1e-4f;

//------------------------------------------------------------------------------
// Positional encoding tests.
//------------------------------------------------------------------------------
class CAIF_PositionalEncodingTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestLearnedForwardShape();
    static void TestLearnedForwardValues();
    static void TestSinusoidalForwardValues();
    static void TestSinusoidalOrthogonality();
    static void TestLearnedBackwardInputGrad();
    static void TestLearnedBackwardWeightGrad();
    static void TestParameterCount();
    static void TestDescription();
};

//------------------------------------------------------------------------------
// Test 1: Learned forward shape
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestLearnedForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePositionalEncodingConfig config{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding<float,float> pe(config,stream);

    const size_t n_input=g_caif_posenc_test_batch*g_caif_posenc_test_seq*g_caif_posenc_test_dim;
    std::vector<float> input_data(n_input,g_caif_posenc_test_input_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_posenc_test_batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=pe.Forward(input,ctx);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==3 &&
                 shape[0]==g_caif_posenc_test_batch &&
                 shape[1]==g_caif_posenc_test_seq &&
                 shape[2]==g_caif_posenc_test_dim);
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected [1,4,8], got [";
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

    CAIF_TestHarness::Report("PositionalEncoding::LearnedForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::LearnedForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Learned forward values = input + pe_table
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestLearnedForwardValues()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePositionalEncodingConfig config{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding<float,float> pe(config,stream);

    // Read PE table
    CAIF_HostTensor host_pe=pe.ParameterTensor(0).ToHost();

    // Create known input (batch=1 for simplicity)
    constexpr uint32_t batch=1;
    const size_t n_input=batch*g_caif_posenc_test_seq*g_caif_posenc_test_dim;
    std::vector<float> input_data(n_input);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*g_caif_posenc_test_input_scale;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=pe.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    for(uint32_t s=0;s<g_caif_posenc_test_seq;++s)
    {
      for(uint32_t d=0;d<g_caif_posenc_test_dim;++d)
      {
        const size_t idx=s*g_caif_posenc_test_dim+d;
        const float expected=input_data[idx]+host_pe.Data()[s*g_caif_posenc_test_dim+d];
        if(CAIF_TestHarness::FloatEqual(host_output.Data()[idx],expected,g_caif_posenc_test_float_tol)==false)
        {
          ISE_Out::Out()<<"  Mismatch at ["
                       <<s
                       <<","
                       <<d
                       <<"]: got "
                       <<host_output.Data()[idx]
                       <<" expected "
                       <<expected
                       <<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    CAIF_TestHarness::Report("PositionalEncoding::LearnedForwardValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::LearnedForwardValues")
}

//------------------------------------------------------------------------------
// Test 3: Sinusoidal forward values match sin/cos formula
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestSinusoidalForwardValues()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePositionalEncodingConfig config{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding<float,float> pe(config,stream);

    // Zero input -> output should be pure PE
    const size_t n_input=batch*g_caif_posenc_test_seq*g_caif_posenc_test_dim;
    std::vector<float> input_data(n_input,0.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=pe.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Compute expected sin/cos values
    bool passed=true;
    for(uint32_t s=0;s<g_caif_posenc_test_seq && passed==true;++s)
    {
      for(uint32_t p=0;p<g_caif_posenc_test_dim/2 && passed==true;++p)
      {
        const double freq=1.0/std::pow(g_caif_sinusoidal_base,
                                       2.0*static_cast<double>(p)/
                                       static_cast<double>(g_caif_posenc_test_dim));
        const double angle=static_cast<double>(s)*freq;
        const float expected_sin=static_cast<float>(std::sin(angle));
        const float expected_cos=static_cast<float>(std::cos(angle));

        const float actual_sin=host_output.Data()[s*g_caif_posenc_test_dim+2*p];
        const float actual_cos=host_output.Data()[s*g_caif_posenc_test_dim+2*p+1];

        if(CAIF_TestHarness::FloatEqual(actual_sin,expected_sin,g_caif_posenc_test_float_tol)==false)
        {
          ISE_Out::Out()<<"  Sin mismatch at s="
                       <<s
                       <<" p="
                       <<p
                       <<": got "
                       <<actual_sin
                       <<" expected "
                       <<expected_sin
                       <<"\n";
          passed=false;
        }
        if(CAIF_TestHarness::FloatEqual(actual_cos,expected_cos,g_caif_posenc_test_float_tol)==false)
        {
          ISE_Out::Out()<<"  Cos mismatch at s="
                       <<s
                       <<" p="
                       <<p
                       <<": got "
                       <<actual_cos
                       <<" expected "
                       <<expected_cos
                       <<"\n";
          passed=false;
        }
      }
    }

    CAIF_TestHarness::Report("PositionalEncoding::SinusoidalForwardValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::SinusoidalForwardValues")
}

//------------------------------------------------------------------------------
// Test 4: Sinusoidal orthogonality (different positions have different PE)
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestSinusoidalOrthogonality()
{
  try
  {
    constexpr uint32_t batch=1;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePositionalEncodingConfig config{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding<float,float> pe(config,stream);

    const size_t n_input=batch*g_caif_posenc_test_seq*g_caif_posenc_test_dim;
    std::vector<float> input_data(n_input,0.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=pe.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Check that PE vectors at different positions are different
    bool passed=true;
    for(uint32_t s1=0;s1<g_caif_posenc_test_seq && passed==true;++s1)
    {
      for(uint32_t s2=s1+1;s2<g_caif_posenc_test_seq && passed==true;++s2)
      {
        bool all_same=true;
        for(uint32_t d=0;d<g_caif_posenc_test_dim;++d)
        {
          if(CAIF_TestHarness::FloatEqual(host_output.Data()[s1*g_caif_posenc_test_dim+d],
                                          host_output.Data()[s2*g_caif_posenc_test_dim+d],
                                          g_caif_posenc_test_ortho_tol)==false)
          {
            all_same=false;
            break;
          }
        }
        if(all_same==true)
        {
          ISE_Out::Out()<<"  Positions "
                       <<s1
                       <<" and "
                       <<s2
                       <<" have identical PE vectors\n";
          passed=false;
        }
      }
    }

    CAIF_TestHarness::Report("PositionalEncoding::SinusoidalOrthogonality",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::SinusoidalOrthogonality")
}

//------------------------------------------------------------------------------
// Test 5: Learned backward input gradient = grad_output (identity)
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestLearnedBackwardInputGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePositionalEncodingConfig config{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding<float,float> pe(config,stream);

    const size_t n_input=g_caif_posenc_test_batch*g_caif_posenc_test_seq*g_caif_posenc_test_dim;
    std::vector<float> input_data(n_input,g_caif_posenc_test_input_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_posenc_test_batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    pe.Forward(input,ctx);

    // Create specific grad_output
    std::vector<float> grad_data(n_input);
    for(size_t i=0;i<grad_data.size();++i)
    {
      grad_data[i]=static_cast<float>(i)*g_caif_posenc_test_grad_scale;
    }
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_data.data(),
      {g_caif_posenc_test_batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);

    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=pe.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    // grad_input should equal grad_output (identity)
    bool passed=true;
    for(size_t i=0;i<grad_data.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(host_grad.Data()[i],grad_data[i],g_caif_posenc_test_float_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                     <<i
                     <<": got "
                     <<host_grad.Data()[i]
                     <<" expected "
                     <<grad_data[i]
                     <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("PositionalEncoding::LearnedBackwardInputGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::LearnedBackwardInputGrad")
}

//------------------------------------------------------------------------------
// Test 6: Learned backward weight gradient (accumulated over batch)
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestLearnedBackwardWeightGrad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const CAIF_DevicePositionalEncodingConfig config{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding<float,float> pe(config,stream);

    const size_t n_input=g_caif_posenc_test_batch*g_caif_posenc_test_seq*g_caif_posenc_test_dim;
    std::vector<float> input_data(n_input,g_caif_posenc_test_input_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
      input_data.data(),
      {g_caif_posenc_test_batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    pe.Forward(input,ctx);

    // grad_output with known values
    std::vector<float> grad_data(n_input);
    for(size_t i=0;i<grad_data.size();++i)
    {
      grad_data[i]=static_cast<float>(i)*g_caif_posenc_test_grad_scale2;
    }
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
      grad_data.data(),
      {g_caif_posenc_test_batch,g_caif_posenc_test_seq,g_caif_posenc_test_dim},
      stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    pe.Backward(grad_out,ctx);

    CAIF_HostTensor host_grad_pe=pe.GradientTensor(0).ToHost();

    // Expected: grad_table[s,d] = sum over batch of grad_output[b,s,d]
    bool passed=true;
    for(uint32_t s=0;s<g_caif_posenc_test_seq && passed==true;++s)
    {
      for(uint32_t d=0;d<g_caif_posenc_test_dim && passed==true;++d)
      {
        float expected=0.0f;
        for(uint32_t b=0;b<g_caif_posenc_test_batch;++b)
        {
          expected+=grad_data[(b*g_caif_posenc_test_seq+s)*g_caif_posenc_test_dim+d];
        }
        const float actual=host_grad_pe.Data()[s*g_caif_posenc_test_dim+d];
        if(CAIF_TestHarness::FloatEqual(actual,expected,g_caif_posenc_test_float_tol)==false)
        {
          ISE_Out::Out()<<"  Mismatch at ["
                       <<s
                       <<","
                       <<d
                       <<"]: got "
                       <<actual
                       <<" expected "
                       <<expected
                       <<"\n";
          passed=false;
        }
      }
    }

    CAIF_TestHarness::Report("PositionalEncoding::LearnedBackwardWeightGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::LearnedBackwardWeightGrad")
}

//------------------------------------------------------------------------------
// Test 7: Parameter count
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestParameterCount()
{
  try
  {
    CAIF_CudaStream stream;

    // Learned: 1 tensor
    const CAIF_DevicePositionalEncodingConfig config_l{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding<float,float> pe_l(config_l,stream);

    // Sinusoidal: 0 tensors
    const CAIF_DevicePositionalEncodingConfig config_s{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding<float,float> pe_s(config_s,stream);

    bool passed=true;
    if(pe_l.ParameterTensorCount()!=1)
    {
      ISE_Out::Out()<<"  Learned: expected 1, got "
                   <<pe_l.ParameterTensorCount()
                   <<"\n";
      passed=false;
    }
    if(pe_l.TotalParameterCount()!=g_caif_posenc_test_max_seq*g_caif_posenc_test_dim)
    {
      ISE_Out::Out()<<"  Learned total: expected "
                   <<g_caif_posenc_test_max_seq*g_caif_posenc_test_dim
                   <<", got "
                   <<pe_l.TotalParameterCount()
                   <<"\n";
      passed=false;
    }
    if(pe_s.ParameterTensorCount()!=0)
    {
      ISE_Out::Out()<<"  Sinusoidal: expected 0, got "
                   <<pe_s.ParameterTensorCount()
                   <<"\n";
      passed=false;
    }
    if(pe_s.TotalParameterCount()!=0)
    {
      ISE_Out::Out()<<"  Sinusoidal total: expected 0, got "
                   <<pe_s.TotalParameterCount()
                   <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("PositionalEncoding::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 8: Description string
//------------------------------------------------------------------------------
void CAIF_PositionalEncodingTests::TestDescription()
{
  try
  {
    CAIF_CudaStream stream;

    const CAIF_DevicePositionalEncodingConfig config_l{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding<float,float> pe_l(config_l,stream);

    const CAIF_DevicePositionalEncodingConfig config_s{
      g_caif_posenc_test_max_seq,
      g_caif_posenc_test_dim,
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding<float,float> pe_s(config_s,stream);

    const std::string desc_l=pe_l.Description();
    const std::string expected_l="PositionalEncoding(max_seq=8,dim=8,mode=learned)";

    const std::string desc_s=pe_s.Description();
    const std::string expected_s="PositionalEncoding(max_seq=8,dim=8,mode=sinusoidal)";

    bool passed=(desc_l==expected_l && desc_s==expected_s);
    if(desc_l!=expected_l)
    {
      ISE_Out::Out()<<"  Learned: expected '"
                   <<expected_l
                   <<"', got '"
                   <<desc_l
                   <<"'\n";
    }
    if(desc_s!=expected_s)
    {
      ISE_Out::Out()<<"  Sinusoidal: expected '"
                   <<expected_s
                   <<"', got '"
                   <<desc_s
                   <<"'\n";
    }

    CAIF_TestHarness::Report("PositionalEncoding::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("PositionalEncoding::Description")
}

void CAIF_PositionalEncodingTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DevicePositionalEncoding<float,float> Tests ==="
               <<"\n\n";
  TestLearnedForwardShape();
  TestLearnedForwardValues();
  TestSinusoidalForwardValues();
  TestSinusoidalOrthogonality();
  TestLearnedBackwardInputGrad();
  TestLearnedBackwardWeightGrad();
  TestParameterCount();
  TestDescription();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_PositionalEncodingTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
               <<"\n";
  return 0;
#endif
}
