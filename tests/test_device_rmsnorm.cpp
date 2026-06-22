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
// Tests for CAIF_DeviceRMSNorm<float,float>.
//------------------------------------------------------------------------------
#include "caif_device_rmsnorm.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_test_harness.h"
#include "caif_cpu_reference/caif_cpu_rmsnorm.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_rmsnorm_test_rows_2d=4;
constexpr uint32_t g_caif_rmsnorm_test_dim=8;
constexpr uint32_t g_caif_rmsnorm_test_batch_3d=2;
constexpr uint32_t g_caif_rmsnorm_test_seq_3d=3;
constexpr uint32_t g_caif_rmsnorm_test_rows_gamma=2;
constexpr uint32_t g_caif_rmsnorm_test_dim_param=64;
constexpr uint32_t g_caif_rmsnorm_test_dim_desc=128;
constexpr uint32_t g_caif_rmsnorm_test_rows_bwd=2;
constexpr float g_caif_rmsnorm_test_tol=1e-4f;
constexpr float g_caif_rmsnorm_test_grad_tol=1e-2f;
constexpr float g_caif_rmsnorm_test_fd_h=1e-3f;

//------------------------------------------------------------------------------
// RMSNorm forward and backward correctness tests.
//------------------------------------------------------------------------------
class CAIF_RMSNormTests
{
  public:
    static void RunAll();

  protected:

  private:
    static bool FloatEqual(float a,float b,float tolerance=g_caif_rmsnorm_test_tol);

    static void TestForward2D();
    static void TestForward3D();
    static void TestGammaScaling();
    static void TestBackwardGradients();
    static void TestParameterCount();
    static void TestZeroGradients();
    static void TestDescription();
};

bool CAIF_RMSNormTests::FloatEqual(const float a,const float b,const float tolerance)
{
  return CAIF_TestHarness::FloatEqual(a,b,tolerance);
}

//------------------------------------------------------------------------------
// Test 1: Forward correctness (2D input)
//------------------------------------------------------------------------------
void CAIF_RMSNormTests::TestForward2D()
{
  try
  {
    const uint32_t rows=g_caif_rmsnorm_test_rows_2d;
    const uint32_t dim=g_caif_rmsnorm_test_dim;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceRMSNorm<float,float> norm(dim,stream,epsilon);

    // Create known input
    std::vector<float> host_input(rows*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.5f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{rows,dim},stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=norm.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference (gamma=1.0)
    std::vector<float> gamma_vals(dim,1.0f);
    std::vector<float> expected(rows*dim);
    CAIF_CpuRMSNorm::Apply(host_input.data(),gamma_vals.data(),expected.data(),
                           rows,dim,epsilon);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i])==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": got "
                      <<host_output.Data()[i]
                      <<" expected "
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("RMSNorm::Forward2D",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RMSNorm::Forward2D")
}

//------------------------------------------------------------------------------
// Test 2: Forward correctness (3D input treated as [rows, dim])
//------------------------------------------------------------------------------
void CAIF_RMSNormTests::TestForward3D()
{
  try
  {
    const uint32_t batch=g_caif_rmsnorm_test_batch_3d;
    const uint32_t seq_len=g_caif_rmsnorm_test_seq_3d;
    const uint32_t dim=g_caif_rmsnorm_test_dim;
    const uint32_t rows=batch*seq_len;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceRMSNorm<float,float> norm(dim,stream,epsilon);

    std::vector<float> host_input(rows*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-1.0f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=norm.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // Verify shape is preserved
    bool passed=true;
    if(host_output.Shape().size()!=3||
       host_output.Shape()[0]!=batch||
       host_output.Shape()[1]!=seq_len||
       host_output.Shape()[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }

    // CPU reference
    std::vector<float> gamma_vals(dim,1.0f);
    std::vector<float> expected(rows*dim);
    CAIF_CpuRMSNorm::Apply(host_input.data(),gamma_vals.data(),expected.data(),
                           rows,dim,epsilon);

    for(size_t i=0;i<expected.size()&&passed==true;++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i])==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": got "
                      <<host_output.Data()[i]
                      <<" expected "
                      <<expected[i]
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("RMSNorm::Forward3D",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RMSNorm::Forward3D")
}

//------------------------------------------------------------------------------
// Test 3: Non-unit gamma values affect output
//------------------------------------------------------------------------------
void CAIF_RMSNormTests::TestGammaScaling()
{
  try
  {
    const uint32_t rows=g_caif_rmsnorm_test_rows_gamma;
    const uint32_t dim=g_caif_rmsnorm_test_dim;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceRMSNorm<float,float> norm(dim,stream,epsilon);

    // Set gamma to non-unit values
    std::vector<float> gamma_vals(dim);
    for(uint32_t i=0;i<dim;++i)
    {
      gamma_vals[i]=0.5f+static_cast<float>(i)*0.25f;
    }
    norm.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);

    std::vector<float> host_input(rows*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.2f-0.8f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{rows,dim},stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=norm.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference with custom gamma
    std::vector<float> expected(rows*dim);
    CAIF_CpuRMSNorm::Apply(host_input.data(),gamma_vals.data(),expected.data(),
                           rows,dim,epsilon);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i])==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": got "
                      <<host_output.Data()[i]
                      <<" expected "
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("RMSNorm::GammaScaling",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RMSNorm::GammaScaling")
}

//------------------------------------------------------------------------------
// Test 4: Backward correctness (finite-difference gradient check)
//------------------------------------------------------------------------------
void CAIF_RMSNormTests::TestBackwardGradients()
{
  try
  {
    const uint32_t rows=g_caif_rmsnorm_test_rows_bwd;
    const uint32_t dim=g_caif_rmsnorm_test_dim;
    const float epsilon=g_caif_epsilon;
    const float h=g_caif_rmsnorm_test_fd_h;
    const float grad_tol=g_caif_rmsnorm_test_grad_tol;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Set non-trivial gamma for a meaningful gradient check
    std::vector<float> gamma_vals(dim);
    for(uint32_t i=0;i<dim;++i)
    {
      gamma_vals[i]=0.8f+static_cast<float>(i)*0.1f;
    }

    std::vector<float> host_input(rows*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.15f-0.6f;
    }

    // Analytical backward pass
    CAIF_DeviceRMSNorm<float,float> norm(dim,stream,epsilon);
    norm.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{rows,dim},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    norm.Forward(input,ctx);

    // grad_output = ones (d(sum)/d(output) = 1 for each element)
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::Uninitialized({rows,dim},stream);
    std::vector<float> ones(rows*dim,1.0f);
    grad_out.CopyFromHost(ones.data(),rows*dim);

    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=norm.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_input=grad_input.ToHost();
    CAIF_HostTensor host_grad_gamma=norm.GradientTensor(0).ToHost();

    bool passed=true;

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Finite-difference check for grad_input
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with perturbed input (+h)
      CAIF_DeviceRMSNorm<float,float> norm_p(dim,stream,epsilon);
      norm_p.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(input_plus.data(),
                                                              {rows,dim},
                                                              stream);
      CAIF_DeviceTensor out_p=norm_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with perturbed input (-h)
      CAIF_DeviceRMSNorm<float,float> norm_m(dim,stream,epsilon);
      norm_m.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(input_minus.data(),
                                                              {rows,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=norm_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_input.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
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

    // Finite-difference check for grad_gamma
    for(uint32_t c=0;c<dim&&passed==true;++c)
    {
      std::vector<float> gamma_plus(gamma_vals);
      std::vector<float> gamma_minus(gamma_vals);
      gamma_plus[c]+=h;
      gamma_minus[c]-=h;

      CAIF_DeviceRMSNorm<float,float> norm_gp(dim,stream,epsilon);
      norm_gp.ParameterTensor(0).CopyFromHost(gamma_plus.data(),dim);
      CAIF_DeviceTensor inp_gp=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                               {rows,dim},
                                                               stream);
      CAIF_DeviceTensor out_gp=norm_gp.Forward(inp_gp,ctx);
      CAIF_HostTensor hout_gp=out_gp.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_plus+=hout_gp.Data()[j];
      }

      CAIF_DeviceRMSNorm<float,float> norm_gm(dim,stream,epsilon);
      norm_gm.ParameterTensor(0).CopyFromHost(gamma_minus.data(),dim);
      CAIF_DeviceTensor inp_gm=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                               {rows,dim},
                                                               stream);
      CAIF_DeviceTensor out_gm=norm_gm.Forward(inp_gm,ctx);
      CAIF_HostTensor hout_gm=out_gm.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_minus+=hout_gm.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_gamma.Data()[c];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        ISE_Out::Out()<<"  dgamma mismatch at "
                      <<c
                      <<": analytical="
                      <<analytical
                      <<" numerical="
                      <<numerical
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("RMSNorm::BackwardGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RMSNorm::BackwardGradients")
}

//------------------------------------------------------------------------------
// Test 5: Parameter counts
//------------------------------------------------------------------------------
void CAIF_RMSNormTests::TestParameterCount()
{
  try
  {
    const uint32_t dim=g_caif_rmsnorm_test_dim_param;
    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm<float,float> norm(dim,stream);

    bool passed=true;
    if(norm.ParameterTensorCount()!=1)
    {
      ISE_Out::Out()<<"  ParameterTensorCount expected 1, got "
                    <<norm.ParameterTensorCount()
                    <<"\n";
      passed=false;
    }
    if(norm.TotalParameterCount()!=dim)
    {
      ISE_Out::Out()<<"  TotalParameterCount expected "
                    <<dim
                    <<", got "
                    <<norm.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("RMSNorm::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RMSNorm::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 6: ZeroGradients
//------------------------------------------------------------------------------
void CAIF_RMSNormTests::TestZeroGradients()
{
  try
  {
    const uint32_t rows=g_caif_rmsnorm_test_rows_gamma;
    const uint32_t dim=g_caif_rmsnorm_test_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceRMSNorm<float,float> norm(dim,stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(rows*dim,1.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{rows,dim},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    norm.Forward(input,ctx);

    std::vector<float> grad_ones(rows*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),{rows,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    norm.Backward(grad_out,ctx);

    // Zero gradients
    norm.ZeroGradients();

    CAIF_HostTensor host_grad=norm.GradientTensor(0).ToHost();
    bool passed=true;
    for(size_t i=0;i<dim;++i)
    {
      if(host_grad.Data()[i]!=0.0f)
      {
        ISE_Out::Out()<<"  Gradient not zeroed at "
                      <<i
                      <<": "
                      <<host_grad.Data()[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("RMSNorm::ZeroGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RMSNorm::ZeroGradients")
}

//------------------------------------------------------------------------------
// Test 7: Description string
//------------------------------------------------------------------------------
void CAIF_RMSNormTests::TestDescription()
{
  try
  {
    const uint32_t dim=g_caif_rmsnorm_test_dim_desc;
    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm<float,float> norm(dim,stream);

    const std::string desc=norm.Description();
    bool passed=(desc=="RMSNorm(128)");
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 'RMSNorm(128)', got '"
                    <<desc
                    <<"'\n";
    }

    CAIF_TestHarness::Report("RMSNorm::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("RMSNorm::Description")
}

void CAIF_RMSNormTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DeviceRMSNorm<float,float> Tests ===\n\n";
  TestForward2D();
  TestForward3D();
  TestGammaScaling();
  TestBackwardGradients();
  TestParameterCount();
  TestZeroGradients();
  TestDescription();
  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

#endif  // USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_RMSNormTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
