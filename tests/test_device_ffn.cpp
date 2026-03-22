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

#include "caif_device_ffn.h"
#include "caif_device_pointwise_activations.h"
#include "caif_device_gated_activations.h"
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
// CPU reference functions
//------------------------------------------------------------------------------

static void CpuMatMul(const float *a,const float *b,float *c,
                       int m,int k,int n)
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

static void CpuGELU(float *data,int n)
{
  const float sqrt_2_over_pi=0.7978845608f;
  const float coeff=0.044715f;
  for(int i=0;i<n;++i)
  {
    const float x=data[i];
    const float inner=sqrt_2_over_pi*(x+coeff*x*x*x);
    data[i]=0.5f*x*(1.0f+std::tanh(inner));
  }
}

static float CpuSwish(float x)
{
  return x/(1.0f+std::exp(-x));
}

static void CpuSwiGLU(const float *gate,const float *up,float *out,int n)
{
  for(int i=0;i<n;++i)
  {
    out[i]=CpuSwish(gate[i])*up[i];
  }
}

// CPU pointwise FFN: output = GELU(input @ W1) @ W2
static void CpuFFNPointwise(const float *input,
                             const float *w1,
                             const float *w2,
                             float *output,
                             int n_rows,
                             int dim,
                             int ffn_dim)
{
  // hidden = input @ W1 -> [n_rows, ffn_dim]
  std::vector<float> hidden(n_rows*ffn_dim);
  CpuMatMul(input,w1,hidden.data(),n_rows,dim,ffn_dim);

  // activation
  CpuGELU(hidden.data(),n_rows*ffn_dim);

  // output = hidden @ W2 -> [n_rows, dim]
  CpuMatMul(hidden.data(),w2,output,n_rows,ffn_dim,dim);
}

// CPU gated FFN: output = SwiGLU(input @ W_gate, input @ W_up) @ W_down
static void CpuFFNGated(const float *input,
                         const float *w_gate,
                         const float *w_up,
                         const float *w_down,
                         float *output,
                         int n_rows,
                         int dim,
                         int ffn_dim)
{
  // gate = input @ W_gate -> [n_rows, ffn_dim]
  std::vector<float> gate(n_rows*ffn_dim);
  CpuMatMul(input,w_gate,gate.data(),n_rows,dim,ffn_dim);

  // up = input @ W_up -> [n_rows, ffn_dim]
  std::vector<float> up(n_rows*ffn_dim);
  CpuMatMul(input,w_up,up.data(),n_rows,dim,ffn_dim);

  // act = SwiGLU(gate, up) -> [n_rows, ffn_dim]
  std::vector<float> act(n_rows*ffn_dim);
  CpuSwiGLU(gate.data(),up.data(),act.data(),n_rows*ffn_dim);

  // output = act @ W_down -> [n_rows, dim]
  CpuMatMul(act.data(),w_down,output,n_rows,ffn_dim,dim);
}

//------------------------------------------------------------------------------
// Test 1: Pointwise forward shape
//------------------------------------------------------------------------------
static void TestPointwiseForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=8;
    const uint32_t ffn_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    auto activation=std::make_unique<CAIF_DeviceGELUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=ffn.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      std::cout<<"  Shape mismatch: expected [2,4,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
      passed=false;
    }

    ReportResult("FFN::PointwiseForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::PointwiseForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Gated forward shape
//------------------------------------------------------------------------------
static void TestGatedForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=ffn.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      std::cout<<"  Shape mismatch: expected [2,4,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
      passed=false;
    }

    ReportResult("FFN::GatedForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::GatedForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Pointwise forward values vs CPU reference
//------------------------------------------------------------------------------
static void TestPointwiseForwardValues()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t ffn_dim=32;
    const int n_rows=static_cast<int>(batch*seq_len);

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    auto activation=std::make_unique<CAIF_DeviceGELUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    // Get weights for CPU reference
    CAIF_HostTensor h_w1=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_w2=ffn.ParameterTensor(1).ToHost();

    // Create deterministic input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    // GPU forward
    CAIF_DeviceTensor output=ffn.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference
    std::vector<float> expected(batch*seq_len*dim);
    CpuFFNPointwise(host_input.data(),
                     h_w1.Data(),h_w2.Data(),
                     expected.data(),
                     n_rows,
                     static_cast<int>(dim),
                     static_cast<int>(ffn_dim));

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i],1e-3f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]
                 <<" diff="<<std::fabs(host_output.Data()[i]-expected[i])<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("FFN::PointwiseForwardValues",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::PointwiseForwardValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Gated forward values vs CPU reference
//------------------------------------------------------------------------------
static void TestGatedForwardValues()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t ffn_dim=16;
    const int n_rows=static_cast<int>(batch*seq_len);

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    // Get weights for CPU reference
    CAIF_HostTensor h_wg=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wu=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wd=ffn.ParameterTensor(2).ToHost();

    // Create deterministic input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    // GPU forward
    CAIF_DeviceTensor output=ffn.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference
    std::vector<float> expected(batch*seq_len*dim);
    CpuFFNGated(host_input.data(),
                 h_wg.Data(),h_wu.Data(),h_wd.Data(),
                 expected.data(),
                 n_rows,
                 static_cast<int>(dim),
                 static_cast<int>(ffn_dim));

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i],1e-3f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]
                 <<" diff="<<std::fabs(host_output.Data()[i]-expected[i])<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("FFN::GatedForwardValues",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::GatedForwardValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Pointwise backward input gradient (finite difference)
//------------------------------------------------------------------------------
static void TestPointwiseBackwardGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=4;
    const uint32_t ffn_dim=8;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    auto activation=std::make_unique<CAIF_DeviceGELUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    // Save weights
    CAIF_HostTensor h_w1=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_w2=ffn.ParameterTensor(1).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    ffn.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor grad_input=ffn.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      auto act_p=std::make_unique<CAIF_DeviceGELUActivation>();
      CAIF_DeviceFFN ffn_p(config,std::move(act_p),stream);
      ffn_p.ParameterTensor(0).CopyFromHost(h_w1.Data(),h_w1.TotalElements());
      ffn_p.ParameterTensor(1).CopyFromHost(h_w2.Data(),h_w2.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_plus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=ffn_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      auto act_m=std::make_unique<CAIF_DeviceGELUActivation>();
      CAIF_DeviceFFN ffn_m(config,std::move(act_m),stream);
      ffn_m.ParameterTensor(0).CopyFromHost(h_w1.Data(),h_w1.TotalElements());
      ffn_m.ParameterTensor(1).CopyFromHost(h_w2.Data(),h_w2.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_minus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=ffn_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dx mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical
                 <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    ReportResult("FFN::PointwiseBackwardGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::PointwiseBackwardGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Gated backward input gradient (finite difference)
//------------------------------------------------------------------------------
static void TestGatedBackwardGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=4;
    const uint32_t ffn_dim=8;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    // Save weights
    CAIF_HostTensor h_wg=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wu=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wd=ffn.ParameterTensor(2).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    ffn.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor grad_input=ffn.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      auto act_p=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn_p(config,std::move(act_p),stream);
      ffn_p.ParameterTensor(0).CopyFromHost(h_wg.Data(),h_wg.TotalElements());
      ffn_p.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_p.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_plus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=ffn_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      auto act_m=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn_m(config,std::move(act_m),stream);
      ffn_m.ParameterTensor(0).CopyFromHost(h_wg.Data(),h_wg.TotalElements());
      ffn_m.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_m.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_minus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=ffn_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dx mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical
                 <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    ReportResult("FFN::GatedBackwardGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::GatedBackwardGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: Gated backward weight gradient (finite difference, W_gate spot check)
//------------------------------------------------------------------------------
static void TestGatedBackwardWeightGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=4;
    const uint32_t ffn_dim=8;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    CAIF_HostTensor h_wg=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wu=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wd=ffn.ParameterTensor(2).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    ffn.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    ffn.Backward(grad_out);
    CAIF_HostTensor host_grad_wg=ffn.GradientTensor(0).ToHost();

    bool passed=true;

    // Spot check first 4 elements of W_gate gradient
    const size_t check_count=4;
    std::vector<float> wg_data(h_wg.Data(),
                                h_wg.Data()+h_wg.TotalElements());

    for(size_t i=0;i<check_count&&passed==true;++i)
    {
      std::vector<float> wg_plus(wg_data);
      std::vector<float> wg_minus(wg_data);
      wg_plus[i]+=h;
      wg_minus[i]-=h;

      // Forward with +h
      auto act_p=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn_p(config,std::move(act_p),stream);
      ffn_p.ParameterTensor(0).CopyFromHost(wg_plus.data(),wg_plus.size());
      ffn_p.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_p.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=ffn_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      auto act_m=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn_m(config,std::move(act_m),stream);
      ffn_m.ParameterTensor(0).CopyFromHost(wg_minus.data(),wg_minus.size());
      ffn_m.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_m.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=ffn_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_wg.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dW_gate mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical
                 <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    ReportResult("FFN::GatedBackwardWeightGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::GatedBackwardWeightGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Parameter count
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t ffn_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    bool passed=true;

    // Pointwise: 2 parameter tensors
    {
      auto activation=std::make_unique<CAIF_DeviceGELUActivation>();
      CAIF_DeviceFFN ffn(config,std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=2)
      {
        std::cout<<"  Pointwise ParameterTensorCount expected 2, got "
                 <<ffn.ParameterTensorCount()<<"\n";
        passed=false;
      }

      const size_t expected_total=dim*ffn_dim+ffn_dim*dim;
      if(ffn.TotalParameterCount()!=expected_total)
      {
        std::cout<<"  Pointwise TotalParameterCount expected "<<expected_total
                 <<", got "<<ffn.TotalParameterCount()<<"\n";
        passed=false;
      }
    }

    // Gated: 3 parameter tensors
    {
      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn(config,std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=3)
      {
        std::cout<<"  Gated ParameterTensorCount expected 3, got "
                 <<ffn.ParameterTensorCount()<<"\n";
        passed=false;
      }

      const size_t expected_total=dim*ffn_dim+dim*ffn_dim+ffn_dim*dim;
      if(ffn.TotalParameterCount()!=expected_total)
      {
        std::cout<<"  Gated TotalParameterCount expected "<<expected_total
                 <<", got "<<ffn.TotalParameterCount()<<"\n";
        passed=false;
      }
    }

    ReportResult("FFN::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: ZeroGradients
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=4;
    const uint32_t ffn_dim=8;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
    CAIF_DeviceFFN ffn(config,std::move(activation),stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    ffn.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    ffn.Backward(grad_out);

    // Zero gradients
    ffn.ZeroGradients();

    bool passed=true;
    for(size_t p=0;p<ffn.ParameterTensorCount();++p)
    {
      CAIF_HostTensor host_grad=ffn.GradientTensor(p).ToHost();
      for(size_t i=0;i<host_grad.TotalElements();++i)
      {
        if(host_grad.Data()[i]!=0.0f)
        {
          std::cout<<"  Gradient["<<p<<"] not zeroed at "<<i<<": "
                   <<host_grad.Data()[i]<<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    ReportResult("FFN::ZeroGradients",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::ZeroGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t ffn_dim=32;

    CAIF_CudaStream stream;
    CAIF_DeviceFFN::FFNConfig_t config;
    config.dim=dim;
    config.ffn_dim=ffn_dim;

    bool passed=true;

    // Pointwise
    {
      auto activation=std::make_unique<CAIF_DeviceGELUActivation>();
      CAIF_DeviceFFN ffn(config,std::move(activation),stream);
      const std::string desc=ffn.Description();
      const std::string expected="FFN(dim=8,ffn_dim=32,activation=GELU)";
      if(desc!=expected)
      {
        std::cout<<"  Pointwise expected '"<<expected<<"', got '"<<desc<<"'\n";
        passed=false;
      }
    }

    // Gated
    {
      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
      CAIF_DeviceFFN ffn(config,std::move(activation),stream);
      const std::string desc=ffn.Description();
      const std::string expected="FFN(dim=8,ffn_dim=32,activation=SwiGLU)";
      if(desc!=expected)
      {
        std::cout<<"  Gated expected '"<<expected<<"', got '"<<desc<<"'\n";
        passed=false;
      }
    }

    ReportResult("FFN::Description",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FFN::Description",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF_DeviceFFN Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestPointwiseForwardShape();
  TestGatedForwardShape();
  TestPointwiseForwardValues();
  TestGatedForwardValues();
  TestPointwiseBackwardGrad();
  TestGatedBackwardGrad();
  TestGatedBackwardWeightGrad();
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
