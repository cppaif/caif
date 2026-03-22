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

#include "caif_device_positional_encoding.h"
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
// Test 1: Learned forward shape
//------------------------------------------------------------------------------
static void TestLearnedForwardShape()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePositionalEncoding::Config_t config{max_seq,dim,PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding pe(config,stream);

    std::vector<float> input_data(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=pe.Forward(input,false);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==3&&shape[0]==batch&&
                 shape[1]==seq_len&&shape[2]==dim);
    if(passed==false)
    {
      std::cout<<"  Expected [1,4,8], got [";
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

    ReportResult("PositionalEncoding::LearnedForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::LearnedForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Learned forward values = input + pe_table
//------------------------------------------------------------------------------
static void TestLearnedForwardValues()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePositionalEncoding::Config_t config{max_seq,dim,PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding pe(config,stream);

    // Read PE table
    CAIF_HostTensor host_pe=pe.ParameterTensor(0).ToHost();

    // Create known input
    std::vector<float> input_data(batch*seq_len*dim);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i)*0.1f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=pe.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    for(uint32_t s=0;s<seq_len;++s)
    {
      for(uint32_t d=0;d<dim;++d)
      {
        const size_t idx=s*dim+d;
        const float expected=input_data[idx]+host_pe.Data()[s*dim+d];
        if(FloatEqual(host_output.Data()[idx],expected)==false)
        {
          std::cout<<"  Mismatch at ["<<s<<","<<d<<"]: got "
                   <<host_output.Data()[idx]<<" expected "<<expected<<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    ReportResult("PositionalEncoding::LearnedForwardValues",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::LearnedForwardValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Sinusoidal forward values match sin/cos formula
//------------------------------------------------------------------------------
static void TestSinusoidalForwardValues()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePositionalEncoding::Config_t config{max_seq,dim,
                                      PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding pe(config,stream);

    // Zero input -> output should be pure PE
    std::vector<float> input_data(batch*seq_len*dim,0.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=pe.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // Compute expected sin/cos values
    bool passed=true;
    for(uint32_t s=0;s<seq_len&&passed==true;++s)
    {
      for(uint32_t p=0;p<dim/2&&passed==true;++p)
      {
        const double freq=1.0/std::pow(g_caif_sinusoidal_base,
                                        2.0*static_cast<double>(p)/
                                        static_cast<double>(dim));
        const double angle=static_cast<double>(s)*freq;
        const float expected_sin=static_cast<float>(std::sin(angle));
        const float expected_cos=static_cast<float>(std::cos(angle));

        const float actual_sin=host_output.Data()[s*dim+2*p];
        const float actual_cos=host_output.Data()[s*dim+2*p+1];

        if(FloatEqual(actual_sin,expected_sin)==false)
        {
          std::cout<<"  Sin mismatch at s="<<s<<" p="<<p
                   <<": got "<<actual_sin<<" expected "<<expected_sin<<"\n";
          passed=false;
        }
        if(FloatEqual(actual_cos,expected_cos)==false)
        {
          std::cout<<"  Cos mismatch at s="<<s<<" p="<<p
                   <<": got "<<actual_cos<<" expected "<<expected_cos<<"\n";
          passed=false;
        }
      }
    }

    ReportResult("PositionalEncoding::SinusoidalForwardValues",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::SinusoidalForwardValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Sinusoidal orthogonality (different positions have different PE)
//------------------------------------------------------------------------------
static void TestSinusoidalOrthogonality()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePositionalEncoding::Config_t config{max_seq,dim,
                                      PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding pe(config,stream);

    std::vector<float> input_data(batch*seq_len*dim,0.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=pe.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // Check that PE vectors at different positions are different
    bool passed=true;
    for(uint32_t s1=0;s1<seq_len&&passed==true;++s1)
    {
      for(uint32_t s2=s1+1;s2<seq_len&&passed==true;++s2)
      {
        bool all_same=true;
        for(uint32_t d=0;d<dim;++d)
        {
          if(FloatEqual(host_output.Data()[s1*dim+d],
                        host_output.Data()[s2*dim+d],1e-6f)==false)
          {
            all_same=false;
            break;
          }
        }
        if(all_same==true)
        {
          std::cout<<"  Positions "<<s1<<" and "<<s2
                   <<" have identical PE vectors\n";
          passed=false;
        }
      }
    }

    ReportResult("PositionalEncoding::SinusoidalOrthogonality",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::SinusoidalOrthogonality",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Learned backward input gradient = grad_output (identity)
//------------------------------------------------------------------------------
static void TestLearnedBackwardInputGrad()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePositionalEncoding::Config_t config{max_seq,dim,PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding pe(config,stream);

    std::vector<float> input_data(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,dim},stream);
    pe.Forward(input,true);

    // Create specific grad_output
    std::vector<float> grad_data(batch*seq_len*dim);
    for(size_t i=0;i<grad_data.size();++i)
    {
      grad_data[i]=static_cast<float>(i)*0.1f;
    }
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_data.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor grad_input=pe.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    // grad_input should equal grad_output (identity)
    bool passed=true;
    for(size_t i=0;i<grad_data.size();++i)
    {
      if(FloatEqual(host_grad.Data()[i],grad_data[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_grad.Data()[i]
                 <<" expected "<<grad_data[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("PositionalEncoding::LearnedBackwardInputGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::LearnedBackwardInputGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Learned backward weight gradient (accumulated over batch)
//------------------------------------------------------------------------------
static void TestLearnedBackwardWeightGrad()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePositionalEncoding::Config_t config{max_seq,dim,PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding pe(config,stream);

    std::vector<float> input_data(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,dim},stream);
    pe.Forward(input,true);

    // grad_output with known values
    std::vector<float> grad_data(batch*seq_len*dim);
    for(size_t i=0;i<grad_data.size();++i)
    {
      grad_data[i]=static_cast<float>(i)*0.01f;
    }
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_data.data(),{batch,seq_len,dim},stream);
    pe.Backward(grad_out);

    CAIF_HostTensor host_grad_pe=pe.GradientTensor(0).ToHost();

    // Expected: grad_table[s,d] = sum over batch of grad_output[b,s,d]
    bool passed=true;
    for(uint32_t s=0;s<seq_len&&passed==true;++s)
    {
      for(uint32_t d=0;d<dim&&passed==true;++d)
      {
        float expected=0.0f;
        for(uint32_t b=0;b<batch;++b)
        {
          expected+=grad_data[(b*seq_len+s)*dim+d];
        }
        const float actual=host_grad_pe.Data()[s*dim+d];
        if(FloatEqual(actual,expected)==false)
        {
          std::cout<<"  Mismatch at ["<<s<<","<<d<<"]: got "<<actual
                   <<" expected "<<expected<<"\n";
          passed=false;
        }
      }
    }

    ReportResult("PositionalEncoding::LearnedBackwardWeightGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::LearnedBackwardWeightGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Parameter count
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;

    CAIF_CudaStream stream;

    // Learned: 1 tensor
    CAIF_DevicePositionalEncoding::Config_t config_l{max_seq,dim,
                                        PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding pe_l(config_l,stream);

    // Sinusoidal: 0 tensors
    CAIF_DevicePositionalEncoding::Config_t config_s{max_seq,dim,
                                        PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding pe_s(config_s,stream);

    bool passed=true;
    if(pe_l.ParameterTensorCount()!=1)
    {
      std::cout<<"  Learned: expected 1, got "<<pe_l.ParameterTensorCount()<<"\n";
      passed=false;
    }
    if(pe_l.TotalParameterCount()!=max_seq*dim)
    {
      std::cout<<"  Learned total: expected "<<max_seq*dim
               <<", got "<<pe_l.TotalParameterCount()<<"\n";
      passed=false;
    }
    if(pe_s.ParameterTensorCount()!=0)
    {
      std::cout<<"  Sinusoidal: expected 0, got "<<pe_s.ParameterTensorCount()<<"\n";
      passed=false;
    }
    if(pe_s.TotalParameterCount()!=0)
    {
      std::cout<<"  Sinusoidal total: expected 0, got "
               <<pe_s.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("PositionalEncoding::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    constexpr uint32_t max_seq=8;
    constexpr uint32_t dim=8;

    CAIF_CudaStream stream;

    CAIF_DevicePositionalEncoding::Config_t config_l{max_seq,dim,
                                        PositionalEncodingMode_e::Learned};
    CAIF_DevicePositionalEncoding pe_l(config_l,stream);

    CAIF_DevicePositionalEncoding::Config_t config_s{max_seq,dim,
                                        PositionalEncodingMode_e::Sinusoidal};
    CAIF_DevicePositionalEncoding pe_s(config_s,stream);

    const std::string desc_l=pe_l.Description();
    const std::string expected_l="PositionalEncoding(max_seq=8,dim=8,mode=learned)";

    const std::string desc_s=pe_s.Description();
    const std::string expected_s="PositionalEncoding(max_seq=8,dim=8,mode=sinusoidal)";

    bool passed=(desc_l==expected_l&&desc_s==expected_s);
    if(desc_l!=expected_l)
    {
      std::cout<<"  Learned: expected '"<<expected_l<<"', got '"<<desc_l<<"'\n";
    }
    if(desc_s!=expected_s)
    {
      std::cout<<"  Sinusoidal: expected '"<<expected_s<<"', got '"<<desc_s<<"'\n";
    }

    ReportResult("PositionalEncoding::Description",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PositionalEncoding::Description",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF_DevicePositionalEncoding Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestLearnedForwardShape();
  TestLearnedForwardValues();
  TestSinusoidalForwardValues();
  TestSinusoidalOrthogonality();
  TestLearnedBackwardInputGrad();
  TestLearnedBackwardWeightGrad();
  TestParameterCount();
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
