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

#include "caif_device_rmsnorm.h"
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

static bool FloatEqual(float a,float b,float tolerance=1e-4f)
{
  return std::fabs(a-b)<tolerance;
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// CPU reference RMSNorm for verification
//------------------------------------------------------------------------------
static void CpuRMSNorm(const float *input,
                        const float *gamma,
                        float *output,
                        int rows,
                        int dim,
                        float epsilon)
{
  for(int r=0;r<rows;++r)
  {
    float sum_sq=0.0f;
    for(int c=0;c<dim;++c)
    {
      const float val=input[r*dim+c];
      sum_sq+=val*val;
    }
    const float rms=std::sqrt(sum_sq/static_cast<float>(dim)+epsilon);
    for(int c=0;c<dim;++c)
    {
      output[r*dim+c]=input[r*dim+c]/rms*gamma[c];
    }
  }
}

//------------------------------------------------------------------------------
// Test 1: Forward correctness (2D input)
//------------------------------------------------------------------------------
static void TestForward2D()
{
  try
  {
    const uint32_t rows=4;
    const uint32_t dim=8;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm norm(dim,stream,epsilon);

    // Create known input
    std::vector<float> host_input(rows*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.5f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{rows,dim},stream);

    // GPU forward
    CAIF_DeviceTensor output=norm.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference (gamma=1.0)
    std::vector<float> gamma_vals(dim,1.0f);
    std::vector<float> expected(rows*dim);
    CpuRMSNorm(host_input.data(),gamma_vals.data(),expected.data(),
                rows,dim,epsilon);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("RMSNorm::Forward2D",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("RMSNorm::Forward2D",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward correctness (3D input treated as [rows, dim])
//------------------------------------------------------------------------------
static void TestForward3D()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t rows=batch*seq_len;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm norm(dim,stream,epsilon);

    std::vector<float> host_input(rows*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-1.0f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=norm.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // Verify shape is preserved
    bool passed=true;
    if(host_output.Shape().size()!=3||
       host_output.Shape()[0]!=batch||
       host_output.Shape()[1]!=seq_len||
       host_output.Shape()[2]!=dim)
    {
      std::cout<<"  Shape mismatch\n";
      passed=false;
    }

    // CPU reference
    std::vector<float> gamma_vals(dim,1.0f);
    std::vector<float> expected(rows*dim);
    CpuRMSNorm(host_input.data(),gamma_vals.data(),expected.data(),
                rows,dim,epsilon);

    for(size_t i=0;i<expected.size()&&passed==true;++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]<<"\n";
        passed=false;
      }
    }

    ReportResult("RMSNorm::Forward3D",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("RMSNorm::Forward3D",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Non-unit gamma values affect output
//------------------------------------------------------------------------------
static void TestGammaScaling()
{
  try
  {
    const uint32_t rows=2;
    const uint32_t dim=8;
    const float epsilon=g_caif_epsilon;

    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm norm(dim,stream,epsilon);

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

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{rows,dim},stream);
    CAIF_DeviceTensor output=norm.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference with custom gamma
    std::vector<float> expected(rows*dim);
    CpuRMSNorm(host_input.data(),gamma_vals.data(),expected.data(),
                rows,dim,epsilon);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("RMSNorm::GammaScaling",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("RMSNorm::GammaScaling",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Backward correctness (finite-difference gradient check)
//------------------------------------------------------------------------------
static void TestBackwardGradients()
{
  try
  {
    const uint32_t rows=2;
    const uint32_t dim=8;
    const float epsilon=g_caif_epsilon;
    const float h=1e-3f;
    const float grad_tol=1e-2f;

    CAIF_CudaStream stream;

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
    CAIF_DeviceRMSNorm norm(dim,stream,epsilon);
    norm.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{rows,dim},stream);
    CAIF_DeviceTensor output=norm.Forward(input,true);

    // grad_output = ones (d(sum)/d(output) = 1 for each element)
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::Uninitialized({rows,dim},stream);
    std::vector<float> ones(rows*dim,1.0f);
    grad_out.CopyFromHost(ones.data(),rows*dim);

    CAIF_DeviceTensor grad_input=norm.Backward(grad_out);
    CAIF_HostTensor host_grad_input=grad_input.ToHost();
    CAIF_HostTensor host_grad_gamma=norm.GradientTensor(0).ToHost();

    bool passed=true;

    // Finite-difference check for grad_input
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with perturbed input (+h)
      CAIF_DeviceRMSNorm norm_p(dim,stream,epsilon);
      norm_p.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_plus.data(),{rows,dim},stream);
      CAIF_DeviceTensor out_p=norm_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with perturbed input (-h)
      CAIF_DeviceRMSNorm norm_m(dim,stream,epsilon);
      norm_m.ParameterTensor(0).CopyFromHost(gamma_vals.data(),dim);
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_minus.data(),{rows,dim},stream);
      CAIF_DeviceTensor out_m=norm_m.Forward(inp_m,false);
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
        std::cout<<"  dx mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical<<"\n";
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

      CAIF_DeviceRMSNorm norm_gp(dim,stream,epsilon);
      norm_gp.ParameterTensor(0).CopyFromHost(gamma_plus.data(),dim);
      CAIF_DeviceTensor inp_gp=CAIF_DeviceTensor::FromHostData(
                                host_input.data(),{rows,dim},stream);
      CAIF_DeviceTensor out_gp=norm_gp.Forward(inp_gp,false);
      CAIF_HostTensor hout_gp=out_gp.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<rows*dim;++j)
      {
        sum_plus+=hout_gp.Data()[j];
      }

      CAIF_DeviceRMSNorm norm_gm(dim,stream,epsilon);
      norm_gm.ParameterTensor(0).CopyFromHost(gamma_minus.data(),dim);
      CAIF_DeviceTensor inp_gm=CAIF_DeviceTensor::FromHostData(
                                host_input.data(),{rows,dim},stream);
      CAIF_DeviceTensor out_gm=norm_gm.Forward(inp_gm,false);
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
        std::cout<<"  dgamma mismatch at "<<c<<": analytical="<<analytical
                 <<" numerical="<<numerical<<"\n";
        passed=false;
      }
    }

    ReportResult("RMSNorm::BackwardGradients",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("RMSNorm::BackwardGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Parameter counts
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    const uint32_t dim=64;
    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm norm(dim,stream);

    bool passed=true;
    if(norm.ParameterTensorCount()!=1)
    {
      std::cout<<"  ParameterTensorCount expected 1, got "
               <<norm.ParameterTensorCount()<<"\n";
      passed=false;
    }
    if(norm.TotalParameterCount()!=dim)
    {
      std::cout<<"  TotalParameterCount expected "<<dim<<", got "
               <<norm.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("RMSNorm::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("RMSNorm::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: ZeroGradients
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    const uint32_t rows=2;
    const uint32_t dim=8;
    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm norm(dim,stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(rows*dim,1.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{rows,dim},stream);
    norm.Forward(input,true);

    std::vector<float> grad_ones(rows*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{rows,dim},stream);
    norm.Backward(grad_out);

    // Zero gradients
    norm.ZeroGradients();

    CAIF_HostTensor host_grad=norm.GradientTensor(0).ToHost();
    bool passed=true;
    for(size_t i=0;i<dim;++i)
    {
      if(host_grad.Data()[i]!=0.0f)
      {
        std::cout<<"  Gradient not zeroed at "<<i<<": "
                 <<host_grad.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("RMSNorm::ZeroGradients",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("RMSNorm::ZeroGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    const uint32_t dim=128;
    CAIF_CudaStream stream;
    CAIF_DeviceRMSNorm norm(dim,stream);

    const std::string desc=norm.Description();
    bool passed=(desc=="RMSNorm(128)");
    if(passed==false)
    {
      std::cout<<"  Expected 'RMSNorm(128)', got '"<<desc<<"'\n";
    }

    ReportResult("RMSNorm::Description",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("RMSNorm::Description",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF_DeviceRMSNorm Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestForward2D();
  TestForward3D();
  TestGammaScaling();
  TestBackwardGradients();
  TestParameterCount();
  TestZeroGradients();
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
