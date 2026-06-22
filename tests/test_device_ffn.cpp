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
// Tests for CAIF_DeviceFFN<float,float>.
//------------------------------------------------------------------------------
#include "caif_device_ffn.h"
#include "caif_test_harness.h"
#include "caif_device_gelu_activation.h"
#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_device_network.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
#include "caif_cpu_reference/caif_cpu_activations.h"
#include "caif_grad_mode.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_ffn_test_batch_shape=2;
constexpr uint32_t g_caif_ffn_test_seq_shape=4;
constexpr uint32_t g_caif_ffn_test_dim_shape=8;
constexpr uint32_t g_caif_ffn_test_ffn_dim_pw=32;
constexpr uint32_t g_caif_ffn_test_ffn_dim_gated=16;
constexpr uint32_t g_caif_ffn_test_batch_val=2;
constexpr uint32_t g_caif_ffn_test_seq_val=3;
constexpr uint32_t g_caif_ffn_test_dim_val=8;
constexpr uint32_t g_caif_ffn_test_batch_bwd=1;
constexpr uint32_t g_caif_ffn_test_seq_bwd=2;
constexpr uint32_t g_caif_ffn_test_dim_bwd=4;
constexpr uint32_t g_caif_ffn_test_ffn_dim_bwd=8;
constexpr uint32_t g_caif_ffn_test_dim_param=8;
constexpr uint32_t g_caif_ffn_test_ffn_dim_param=32;
constexpr size_t g_caif_ffn_test_wgrad_spot=4;
constexpr float g_caif_ffn_test_val_tol=1e-3f;
constexpr float g_caif_ffn_test_fd_h=1e-3f;

//------------------------------------------------------------------------------
// FFN forward and backward correctness tests.
//------------------------------------------------------------------------------
class CAIF_FFNTests
{
  public:
    static void RunAll();

  protected:

  private:

    //--------------------------------------------------------------------------
    // CPU reference composites (primitives come from caif_cpu_reference/*.h).
    //--------------------------------------------------------------------------

    // CPU pointwise FFN: output = GELU(input @ W1) @ W2
    static void CpuFFNPointwise(const float *input,
                                 const float *w1,
                                 const float *w2,
                                 float *output,
                                 int n_rows,
                                 int dim,
                                 int ffn_dim);

    // CPU gated FFN: output = SwiGLU(input @ W_gate, input @ W_up) @ W_down
    static void CpuFFNGated(const float *input,
                             const float *w_gate,
                             const float *w_up,
                             const float *w_down,
                             float *output,
                             int n_rows,
                             int dim,
                             int ffn_dim);

    static void TestPointwiseForwardShape();
    static void TestGatedForwardShape();
    static void TestPointwiseForwardValues();
    static void TestGatedForwardValues();
    static void TestPointwiseBackwardGrad(const GradMode_t &mode);
    static void TestGatedBackwardGrad(const GradMode_t &mode);
    static void TestGatedBackwardWeightGrad(const GradMode_t &mode);
    static void TestParameterCount();
    static void TestZeroGradients();
    static void TestDescription();
};

void CAIF_FFNTests::CpuFFNPointwise(const float *input,
                                     const float *w1,
                                     const float *w2,
                                     float *output,
                                     const int n_rows,
                                     const int dim,
                                     const int ffn_dim)
{
  // hidden = input @ W1 -> [n_rows, ffn_dim]
  std::vector<float> hidden(n_rows*ffn_dim);
  CAIF_CpuMatMul::Apply(input,w1,hidden.data(),n_rows,dim,ffn_dim);

  // activation
  CAIF_CpuActivations::GELUArray(hidden.data(),n_rows*ffn_dim);

  // output = hidden @ W2 -> [n_rows, dim]
  CAIF_CpuMatMul::Apply(hidden.data(),w2,output,n_rows,ffn_dim,dim);
}

void CAIF_FFNTests::CpuFFNGated(const float *input,
                                  const float *w_gate,
                                  const float *w_up,
                                  const float *w_down,
                                  float *output,
                                  const int n_rows,
                                  const int dim,
                                  const int ffn_dim)
{
  // gate = input @ W_gate -> [n_rows, ffn_dim]
  std::vector<float> gate(n_rows*ffn_dim);
  CAIF_CpuMatMul::Apply(input,w_gate,gate.data(),n_rows,dim,ffn_dim);

  // up = input @ W_up -> [n_rows, ffn_dim]
  std::vector<float> up(n_rows*ffn_dim);
  CAIF_CpuMatMul::Apply(input,w_up,up.data(),n_rows,dim,ffn_dim);

  // act = SwiGLU(gate, up) -> [n_rows, ffn_dim]
  std::vector<float> act(n_rows*ffn_dim);
  CAIF_CpuActivations::SwiGLU(gate.data(),up.data(),act.data(),n_rows*ffn_dim);

  // output = act @ W_down -> [n_rows, dim]
  CAIF_CpuMatMul::Apply(act.data(),w_down,output,n_rows,ffn_dim,dim);
}

//------------------------------------------------------------------------------
// Test 1: Pointwise forward shape
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestPointwiseForwardShape()
{
  try
  {
    const uint32_t batch=g_caif_ffn_test_batch_shape;
    const uint32_t seq_len=g_caif_ffn_test_seq_shape;
    const uint32_t dim=g_caif_ffn_test_dim_shape;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_pw;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    auto activation=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch: expected [2,4,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          ISE_Out::Out()<<",";
        }
        ISE_Out::Out()<<shape[i];
      }
      ISE_Out::Out()<<"]\n";
      passed=false;
    }

    CAIF_TestHarness::Report("FFN::PointwiseForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::PointwiseForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Gated forward shape
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestGatedForwardShape()
{
  try
  {
    const uint32_t batch=g_caif_ffn_test_batch_shape;
    const uint32_t seq_len=g_caif_ffn_test_seq_shape;
    const uint32_t dim=g_caif_ffn_test_dim_shape;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_gated;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch: expected [2,4,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          ISE_Out::Out()<<",";
        }
        ISE_Out::Out()<<shape[i];
      }
      ISE_Out::Out()<<"]\n";
      passed=false;
    }

    CAIF_TestHarness::Report("FFN::GatedForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::GatedForwardShape")
}

//------------------------------------------------------------------------------
// Test 3: Pointwise forward values vs CPU reference
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestPointwiseForwardValues()
{
  try
  {
    const uint32_t batch=g_caif_ffn_test_batch_val;
    const uint32_t seq_len=g_caif_ffn_test_seq_val;
    const uint32_t dim=g_caif_ffn_test_dim_val;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_pw;
    const int n_rows=static_cast<int>(batch*seq_len);

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    auto activation=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

    // Get weights for CPU reference
    CAIF_HostTensor h_w1=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_w2=ffn.ParameterTensor(1).ToHost();

    // Create deterministic input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    // GPU forward
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
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
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],expected[i],g_caif_ffn_test_val_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": got "
                      <<host_output.Data()[i]
                      <<" expected "
                      <<expected[i]
                      <<" diff="
                      <<std::fabs(host_output.Data()[i]-expected[i])
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("FFN::PointwiseForwardValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::PointwiseForwardValues")
}

//------------------------------------------------------------------------------
// Test 4: Gated forward values vs CPU reference
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestGatedForwardValues()
{
  try
  {
    const uint32_t batch=g_caif_ffn_test_batch_val;
    const uint32_t seq_len=g_caif_ffn_test_seq_val;
    const uint32_t dim=g_caif_ffn_test_dim_val;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_gated;
    const int n_rows=static_cast<int>(batch*seq_len);

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

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

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    // GPU forward — compare GPU algorithm to CPU reference in full FP32
    // (three matmuls: Wg, Wu, Wd, so TF32 drift exceeds 1e-3 tolerance).
    const bool prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();
    CAIF_Settings::SetPreciseGradients(prev_precise);

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
      if(CAIF_TestHarness::FloatEqual(host_output.Data()[i],expected[i],g_caif_ffn_test_val_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": got "
                      <<host_output.Data()[i]
                      <<" expected "
                      <<expected[i]
                      <<" diff="
                      <<std::fabs(host_output.Data()[i]-expected[i])
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("FFN::GatedForwardValues",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::GatedForwardValues")
}

//------------------------------------------------------------------------------
// Test 5: Pointwise backward input gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestPointwiseBackwardGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("FFN::PointwiseBackwardGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const uint32_t batch=g_caif_ffn_test_batch_bwd;
    const uint32_t seq_len=g_caif_ffn_test_seq_bwd;
    const uint32_t dim=g_caif_ffn_test_dim_bwd;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_bwd;
    const float h=g_caif_ffn_test_fd_h;
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    auto activation=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

    // Save weights
    CAIF_HostTensor h_w1=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_w2=ffn.ParameterTensor(1).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    ffn.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=ffn.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    ctx.SetTraining(false);
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      auto act_p=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn_p(config,std::move(act_p),stream);
      ffn_p.ParameterTensor(0).CopyFromHost(h_w1.Data(),h_w1.TotalElements());
      ffn_p.ParameterTensor(1).CopyFromHost(h_w2.Data(),h_w2.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(input_plus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_p=ffn_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      auto act_m=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn_m(config,std::move(act_m),stream);
      ffn_m.ParameterTensor(0).CopyFromHost(h_w1.Data(),h_w1.TotalElements());
      ffn_m.ParameterTensor(1).CopyFromHost(h_w2.Data(),h_w2.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(input_minus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=ffn_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dx mismatch at "
                      <<i
                      <<": analytical="
                      <<analytical
                      <<" numerical="
                      <<numerical
                      <<" diff="
                      <<std::fabs(numerical-analytical)
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 6: Gated backward input gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestGatedBackwardGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("FFN::GatedBackwardGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const uint32_t batch=g_caif_ffn_test_batch_bwd;
    const uint32_t seq_len=g_caif_ffn_test_seq_bwd;
    const uint32_t dim=g_caif_ffn_test_dim_bwd;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_bwd;
    const float h=g_caif_ffn_test_fd_h;
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

    // Save weights
    CAIF_HostTensor h_wg=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wu=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wd=ffn.ParameterTensor(2).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    ffn.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=ffn.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    ctx.SetTraining(false);
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      auto act_p=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn_p(config,std::move(act_p),stream);
      ffn_p.ParameterTensor(0).CopyFromHost(h_wg.Data(),h_wg.TotalElements());
      ffn_p.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_p.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(input_plus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_p=ffn_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      auto act_m=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn_m(config,std::move(act_m),stream);
      ffn_m.ParameterTensor(0).CopyFromHost(h_wg.Data(),h_wg.TotalElements());
      ffn_m.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_m.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(input_minus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=ffn_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dx mismatch at "
                      <<i
                      <<": analytical="
                      <<analytical
                      <<" numerical="
                      <<numerical
                      <<" diff="
                      <<std::fabs(numerical-analytical)
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 7: Gated backward weight gradient (finite difference, W_gate spot check)
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestGatedBackwardWeightGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("FFN::GatedBackwardWeightGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const uint32_t batch=g_caif_ffn_test_batch_bwd;
    const uint32_t seq_len=g_caif_ffn_test_seq_bwd;
    const uint32_t dim=g_caif_ffn_test_dim_bwd;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_bwd;
    const float h=g_caif_ffn_test_fd_h;
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

    CAIF_HostTensor h_wg=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wu=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wd=ffn.ParameterTensor(2).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    ffn.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    ffn.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad_wg=ffn.GradientTensor(0).ToHost();

    bool passed=true;

    // Spot check first g_caif_ffn_test_wgrad_spot elements of W_gate gradient
    std::vector<float> wg_data(h_wg.Data(),h_wg.Data()+h_wg.TotalElements());

    ctx.SetTraining(false);
    for(size_t i=0;i<g_caif_ffn_test_wgrad_spot&&passed==true;++i)
    {
      std::vector<float> wg_plus(wg_data);
      std::vector<float> wg_minus(wg_data);
      wg_plus[i]+=h;
      wg_minus[i]-=h;

      // Forward with +h
      auto act_p=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn_p(config,std::move(act_p),stream);
      ffn_p.ParameterTensor(0).CopyFromHost(wg_plus.data(),wg_plus.size());
      ffn_p.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_p.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_p=ffn_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      auto act_m=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn_m(config,std::move(act_m),stream);
      ffn_m.ParameterTensor(0).CopyFromHost(wg_minus.data(),wg_minus.size());
      ffn_m.ParameterTensor(1).CopyFromHost(h_wu.Data(),h_wu.TotalElements());
      ffn_m.ParameterTensor(2).CopyFromHost(h_wd.Data(),h_wd.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=ffn_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_wg.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dW_gate mismatch at "
                      <<i
                      <<": analytical="
                      <<analytical
                      <<" numerical="
                      <<numerical
                      <<" diff="
                      <<std::fabs(numerical-analytical)
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 8: Parameter count
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestParameterCount()
{
  try
  {
    const uint32_t dim=g_caif_ffn_test_dim_param;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_param;

    CAIF_CudaStream stream;
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    bool passed=true;

    // Pointwise: 2 parameter tensors
    {
      auto activation=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=2)
      {
        ISE_Out::Out()<<"  Pointwise ParameterTensorCount expected 2, got "
                      <<ffn.ParameterTensorCount()
                      <<"\n";
        passed=false;
      }

      const size_t expected_total=dim*ffn_dim+ffn_dim*dim;
      if(ffn.TotalParameterCount()!=expected_total)
      {
        ISE_Out::Out()<<"  Pointwise TotalParameterCount expected "
                      <<expected_total
                      <<", got "
                      <<ffn.TotalParameterCount()
                      <<"\n";
        passed=false;
      }
    }

    // Gated: 3 parameter tensors
    {
      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

      if(ffn.ParameterTensorCount()!=3)
      {
        ISE_Out::Out()<<"  Gated ParameterTensorCount expected 3, got "
                      <<ffn.ParameterTensorCount()
                      <<"\n";
        passed=false;
      }

      const size_t expected_total=dim*ffn_dim+dim*ffn_dim+ffn_dim*dim;
      if(ffn.TotalParameterCount()!=expected_total)
      {
        ISE_Out::Out()<<"  Gated TotalParameterCount expected "
                      <<expected_total
                      <<", got "
                      <<ffn.TotalParameterCount()
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("FFN::ParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::ParameterCount")
}

//------------------------------------------------------------------------------
// Test 9: ZeroGradients
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestZeroGradients()
{
  try
  {
    const uint32_t batch=g_caif_ffn_test_batch_bwd;
    const uint32_t seq_len=g_caif_ffn_test_seq_bwd;
    const uint32_t dim=g_caif_ffn_test_dim_bwd;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_bwd;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    ffn.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    ffn.Backward(grad_out,ctx);

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
          ISE_Out::Out()<<"  Gradient["
                        <<p
                        <<"] not zeroed at "
                        <<i
                        <<": "
                        <<host_grad.Data()[i]
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

    CAIF_TestHarness::Report("FFN::ZeroGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::ZeroGradients")
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
void CAIF_FFNTests::TestDescription()
{
  try
  {
    const uint32_t dim=g_caif_ffn_test_dim_param;
    const uint32_t ffn_dim=g_caif_ffn_test_ffn_dim_param;

    CAIF_CudaStream stream;
    CAIF_DeviceFFNConfig config(dim,ffn_dim);

    bool passed=true;

    // Pointwise
    {
      auto activation=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);
      const std::string desc=ffn.Description();
      const std::string expected="FFN(dim=8,ffn_dim=32,activation=GELU)";
      if(desc!=expected)
      {
        ISE_Out::Out()<<"  Pointwise expected '"
                      <<expected
                      <<"', got '"
                      <<desc
                      <<"'\n";
        passed=false;
      }
    }

    // Gated
    {
      auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
      CAIF_DeviceFFN<float,float> ffn(config,std::move(activation),stream);
      const std::string desc=ffn.Description();
      const std::string expected="FFN(dim=8,ffn_dim=32,activation=SwiGLU)";
      if(desc!=expected)
      {
        ISE_Out::Out()<<"  Gated expected '"
                      <<expected
                      <<"', got '"
                      <<desc
                      <<"'\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("FFN::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("FFN::Description")
}

void CAIF_FFNTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DeviceFFN<float,float> Tests ===\n\n";
  TestPointwiseForwardShape();
  TestGatedForwardShape();
  TestPointwiseForwardValues();
  TestGatedForwardValues();
  TestPointwiseBackwardGrad(g_caif_grad_mode_precise);
  TestPointwiseBackwardGrad(g_caif_grad_mode_tf32);
  TestGatedBackwardGrad(g_caif_grad_mode_precise);
  TestGatedBackwardGrad(g_caif_grad_mode_tf32);
  TestGatedBackwardWeightGrad(g_caif_grad_mode_precise);
  TestGatedBackwardWeightGrad(g_caif_grad_mode_tf32);
  TestParameterCount();
  TestZeroGradients();
  TestDescription();
}

#endif  // USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FFNTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
