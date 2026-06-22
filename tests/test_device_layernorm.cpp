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
// Tests for CAIF_DeviceLayerNorm<float,float>.
//------------------------------------------------------------------------------
#include "caif_device_layernorm.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_cpu_reference/caif_cpu_layernorm.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_layernorm_test_rows_2d=4;
constexpr uint32_t g_caif_layernorm_test_dim=8;
constexpr uint32_t g_caif_layernorm_test_batch_3d=2;
constexpr uint32_t g_caif_layernorm_test_seq_3d=3;
constexpr uint32_t g_caif_layernorm_test_rows_gamma=2;
constexpr uint32_t g_caif_layernorm_test_rows_bwd=2;
constexpr uint32_t g_caif_layernorm_test_dim_param=64;
constexpr uint32_t g_caif_layernorm_test_dim_desc=256;
constexpr float g_caif_layernorm_test_tol=1e-4f;
constexpr float g_caif_layernorm_test_grad_tol=1e-2f;
constexpr float g_caif_layernorm_test_fd_h=1e-3f;

//------------------------------------------------------------------------------
// LayerNorm forward and backward correctness tests.
//------------------------------------------------------------------------------
class CAIF_LayerNormTests
{
  public:
    static void RunAll();

  protected:

  private:
    static bool FloatEqual(float a,float b,float tolerance=g_caif_layernorm_test_tol);

    static void TestForward2D();
    static void TestForward3D();
    static void TestGammaBeta();
    static void TestBackwardGradients();
    static void TestParameterCount();
    static void TestZeroGradients();
    static void TestDescription();
};

bool CAIF_LayerNormTests::FloatEqual(const float a,const float b,const float tolerance)
{
  return CAIF_TestHarness::FloatEqual(a,b,tolerance);
}

//------------------------------------------------------------------------------
// Test 1: Forward correctness (2D input)
//------------------------------------------------------------------------------
void CAIF_LayerNormTests::TestForward2D()
{
  try
  {
    const uint32_t rows=g_caif_layernorm_test_rows_2d;
    const uint32_t dim=g_caif_layernorm_test_dim;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLayerNorm<float,float> norm(dim,stream,epsilon);

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

    // CPU reference (gamma=1.0, beta=0.0)
    std::vector<float> gamma_vals(dim,1.0f);
    std::vector<float> beta_vals(dim,0.0f);
    std::vector<float> expected(rows*dim);
    CAIF_CpuLayerNorm::Apply(host_input.data(),gamma_vals.data(),beta_vals.data(),
                             expected.data(),rows,dim,epsilon);

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

    CAIF_TestHarness::Report("LayerNorm::Forward2D",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LayerNorm::Forward2D")
}

//------------------------------------------------------------------------------
// Test 2: Forward correctness (3D input treated as [rows, dim])
//------------------------------------------------------------------------------
void CAIF_LayerNormTests::TestForward3D()
{
  try
  {
    const uint32_t batch=g_caif_layernorm_test_batch_3d;
    const uint32_t seq_len=g_caif_layernorm_test_seq_3d;
    const uint32_t dim=g_caif_layernorm_test_dim;
    const uint32_t rows=batch*seq_len;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLayerNorm<float,float> norm(dim,stream,epsilon);

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

    // Verify shape preserved
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
    std::vector<float> beta_vals(dim,0.0f);
    std::vector<float> expected(rows*dim);
    CAIF_CpuLayerNorm::Apply(host_input.data(),gamma_vals.data(),beta_vals.data(),
                             expected.data(),rows,dim,epsilon);

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

    CAIF_TestHarness::Report("LayerNorm::Forward3D",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LayerNorm::Forward3D")
}

//------------------------------------------------------------------------------
// Test 3: Non-trivial gamma and beta values
//------------------------------------------------------------------------------
void CAIF_LayerNormTests::TestGammaBeta()
{
  try
  {
    const uint32_t rows=g_caif_layernorm_test_rows_gamma;
    const uint32_t dim=g_caif_layernorm_test_dim;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLayerNorm<float,float> norm(dim,stream,epsilon);

    // Set custom gamma and beta
    std::vector<float> gamma_vals(dim);
    std::vector<float> beta_vals(dim);
    for(uint32_t i=0;i<dim;++i)
    {
      gamma_vals[i]=0.5f+static_cast<float>(i)*0.25f;
      beta_vals[i]=-0.3f+static_cast<float>(i)*0.1f;
    }
    norm.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
    norm.ParameterTensor(1).CopyFromHost(beta_vals.data(),dim);

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

    // CPU reference with custom gamma/beta
    std::vector<float> expected(rows*dim);
    CAIF_CpuLayerNorm::Apply(host_input.data(),gamma_vals.data(),beta_vals.data(),
                             expected.data(),rows,dim,epsilon);

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

    CAIF_TestHarness::Report("LayerNorm::GammaBeta",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LayerNorm::GammaBeta")
}

//------------------------------------------------------------------------------
// Test 4: Backward correctness (finite-difference gradient check)
//------------------------------------------------------------------------------
void CAIF_LayerNormTests::TestBackwardGradients()
{
  try
  {
    const uint32_t rows=g_caif_layernorm_test_rows_bwd;
    const uint32_t dim=g_caif_layernorm_test_dim;
    const float epsilon=g_caif_epsilon;
    const float h=g_caif_layernorm_test_fd_h;
    const float grad_tol=g_caif_layernorm_test_grad_tol;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> gamma_vals(dim);
    std::vector<float> beta_vals(dim);
    for(uint32_t i=0;i<dim;++i)
    {
      gamma_vals[i]=0.8f+static_cast<float>(i)*0.1f;
      beta_vals[i]=-0.2f+static_cast<float>(i)*0.05f;
    }

    std::vector<float> host_input(rows*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.15f-0.6f;
    }

    // Analytical backward pass
    CAIF_DeviceLayerNorm<float,float> norm(dim,stream,epsilon);
    norm.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
    norm.ParameterTensor(1).CopyFromHost(beta_vals.data(),dim);

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{rows,dim},stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    norm.Forward(input,ctx);

    std::vector<float> ones(rows*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(ones.data(),{rows,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=norm.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_input=grad_input.ToHost();
    CAIF_HostTensor host_grad_gamma=norm.GradientTensor(0).ToHost();
    CAIF_HostTensor host_grad_beta=norm.GradientTensor(1).ToHost();

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

      CAIF_DeviceLayerNorm<float,float> norm_p(dim,stream,epsilon);
      norm_p.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      norm_p.ParameterTensor(1).CopyFromHost(beta_vals.data(),dim);
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

      CAIF_DeviceLayerNorm<float,float> norm_m(dim,stream,epsilon);
      norm_m.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      norm_m.ParameterTensor(1).CopyFromHost(beta_vals.data(),dim);
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

      CAIF_DeviceLayerNorm<float,float> norm_gp(dim,stream,epsilon);
      norm_gp.ParameterTensor(0).CopyFromHost(gamma_plus.data(),dim);
      norm_gp.ParameterTensor(1).CopyFromHost(beta_vals.data(),dim);
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

      CAIF_DeviceLayerNorm<float,float> norm_gm(dim,stream,epsilon);
      norm_gm.ParameterTensor(0).CopyFromHost(gamma_minus.data(),dim);
      norm_gm.ParameterTensor(1).CopyFromHost(beta_vals.data(),dim);
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

    // Finite-difference check for grad_beta
    for(uint32_t c=0;c<dim&&passed==true;++c)
    {
      std::vector<float> beta_plus(beta_vals);
      std::vector<float> beta_minus(beta_vals);
      beta_plus[c]+=h;
      beta_minus[c]-=h;

      CAIF_DeviceLayerNorm<float,float> norm_bp(dim,stream,epsilon);
      norm_bp.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      norm_bp.ParameterTensor(1).CopyFromHost(beta_plus.data(),dim);
      CAIF_DeviceTensor inp_bp=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                               {rows,dim},
                                                               stream);
      CAIF_DeviceTensor out_bp=norm_bp.Forward(inp_bp,ctx);
      CAIF_HostTensor hout_bp=out_bp.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_plus+=hout_bp.Data()[j];
      }

      CAIF_DeviceLayerNorm<float,float> norm_bm(dim,stream,epsilon);
      norm_bm.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      norm_bm.ParameterTensor(1).CopyFromHost(beta_minus.data(),dim);
      CAIF_DeviceTensor inp_bm=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                               {rows,dim},
                                                               stream);
      CAIF_DeviceTensor out_bm=norm_bm.Forward(inp_bm,ctx);
      CAIF_HostTensor hout_bm=out_bm.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_minus+=hout_bm.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_beta.Data()[c];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        ISE_Out::Out()<<"  dbeta mismatch at "
                      <<c
                      <<": analytical="
                      <<analytical
                      <<" numerical="
                      <<numerical
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("LayerNorm::BackwardGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LayerNorm::BackwardGradients")
}

//------------------------------------------------------------------------------
// Test 5: Parameter counts
//------------------------------------------------------------------------------
void CAIF_LayerNormTests::TestParameterCount()
{
  try
  {
    const uint32_t dim=g_caif_layernorm_test_dim_param;
    CAIF_CudaStream stream;
    CAIF_DeviceLayerNorm<float,float> norm(dim,stream);

    bool passed=true;
    if(norm.ParameterTensorCount()!=2)
    {
      ISE_Out::Out()<<"  ParameterTensorCount expected 2, got "
                    <<norm.ParameterTensorCount()
                    <<"\n";
      passed=false;
    }
    if(norm.TotalParameterCount()!=dim*2)
    {
      ISE_Out::Out()<<"  TotalParameterCount expected "
                    <<dim*2
                    <<", got "
                    <<norm.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("LayerNorm::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LayerNorm::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 6: ZeroGradients
//------------------------------------------------------------------------------
void CAIF_LayerNormTests::TestZeroGradients()
{
  try
  {
    const uint32_t rows=g_caif_layernorm_test_rows_gamma;
    const uint32_t dim=g_caif_layernorm_test_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceLayerNorm<float,float> norm(dim,stream);

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

    CAIF_HostTensor host_grad_gamma=norm.GradientTensor(0).ToHost();
    CAIF_HostTensor host_grad_beta=norm.GradientTensor(1).ToHost();
    bool passed=true;
    for(size_t i=0;i<dim;++i)
    {
      if(host_grad_gamma.Data()[i]!=0.0f)
      {
        ISE_Out::Out()<<"  Gamma gradient not zeroed at "
                      <<i
                      <<"\n";
        passed=false;
        break;
      }
      if(host_grad_beta.Data()[i]!=0.0f)
      {
        ISE_Out::Out()<<"  Beta gradient not zeroed at "
                      <<i
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("LayerNorm::ZeroGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LayerNorm::ZeroGradients")
}

//------------------------------------------------------------------------------
// Test 7: Description string
//------------------------------------------------------------------------------
void CAIF_LayerNormTests::TestDescription()
{
  try
  {
    const uint32_t dim=g_caif_layernorm_test_dim_desc;
    CAIF_CudaStream stream;
    CAIF_DeviceLayerNorm<float,float> norm(dim,stream);

    const std::string desc=norm.Description();
    bool passed=(desc=="LayerNorm(256)");
    if(passed==false)
    {
      ISE_Out::Out()<<"  Expected 'LayerNorm(256)', got '"
                    <<desc
                    <<"'\n";
    }

    CAIF_TestHarness::Report("LayerNorm::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("LayerNorm::Description")
}

void CAIF_LayerNormTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DeviceLayerNorm<float,float> Tests ===\n\n";
  TestForward2D();
  TestForward3D();
  TestGammaBeta();
  TestBackwardGradients();
  TestParameterCount();
  TestZeroGradients();
  TestDescription();
}

#endif  // USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_LayerNormTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
