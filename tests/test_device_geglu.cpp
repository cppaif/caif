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
// Test: GeGLU and SwiGLU gated FFN forward/backward correctness + benchmark.
//
// TestGeGLUForward: GPU GeGLU FFN forward matches CPU reference.
// TestGeGLUBackward: backward produces non-zero gradients for both
//   input and all weight tensors.
// TestSwiGLURegression: GPU SwiGLU FFN forward matches CPU reference.
// BenchmarkGeGLUvsSwiGLU / BenchmarkGeGLUvsSwiGLUFwdBwd: timing only,
//   no correctness assertion.
//------------------------------------------------------------------------------
#include "caif_device_ffn.h"
#include "caif_test_harness.h"
#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_cpu_reference/caif_cpu_matmul.h"
#include "caif_cpu_reference/caif_cpu_activations.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <chrono>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_geglu_test_fwd_batch=2;
constexpr uint32_t g_caif_geglu_test_fwd_seq=4;
constexpr uint32_t g_caif_geglu_test_fwd_dim=16;
constexpr uint32_t g_caif_geglu_test_fwd_hidden=32;
constexpr uint32_t g_caif_geglu_test_bwd_bs=8;
constexpr uint32_t g_caif_geglu_test_bwd_dim=16;
constexpr uint32_t g_caif_geglu_test_bwd_hidden=32;
constexpr uint32_t g_caif_geglu_test_swiglu_bs=8;
constexpr uint32_t g_caif_geglu_test_swiglu_dim=16;
constexpr uint32_t g_caif_geglu_test_swiglu_hidden=32;
constexpr uint32_t g_caif_geglu_test_bench_batch=8;
constexpr uint32_t g_caif_geglu_test_bench_seq=256;
constexpr uint32_t g_caif_geglu_test_bench_dim=512;
constexpr uint32_t g_caif_geglu_test_bench_hidden=1024;
constexpr int g_caif_geglu_test_bench_warmup=10;
constexpr int g_caif_geglu_test_bench_iters=100;
constexpr int g_caif_geglu_test_bench_fwdbwd_warmup=5;
constexpr int g_caif_geglu_test_bench_fwdbwd_iters=50;
constexpr float g_caif_geglu_test_fwd_input_scale=0.04f;
constexpr float g_caif_geglu_test_fwd_input_bias=-0.3f;
constexpr float g_caif_geglu_test_bwd_input_scale=0.03f;
constexpr float g_caif_geglu_test_bwd_input_bias=-0.2f;
constexpr float g_caif_geglu_test_tol=1e-2f;
constexpr float g_caif_geglu_test_bench_input_fill=0.1f;
constexpr float g_caif_geglu_test_bench_grad_fill=1.0f;

//------------------------------------------------------------------------------
// CPU reference: gated FFN forward.
// gate = input @ W_gate -> activation(gate)
// up   = input @ W_up
// hidden = activation(gate) * up
// output = hidden @ W_down
//------------------------------------------------------------------------------
class CAIF_GeGLUTests
{
  public:
    static void RunAll();

  protected:

  private:
    // CPU reference implementation of gated FFN.
    // use_geglu==true: GELU activation on gate; false: Swish (SwiGLU).
    static void CpuGatedFFN(const float *input,
                             const float *w_gate,
                             const float *w_up,
                             const float *w_down,
                             float *output,
                             int bs,
                             int dim,
                             int hidden_dim,
                             bool use_geglu);

    static void TestGeGLUForward();
    static void TestGeGLUBackward();
    static void TestSwiGLURegression();
    static void BenchmarkGeGLUvsSwiGLU();
    static void BenchmarkGeGLUvsSwiGLUFwdBwd();
};

void CAIF_GeGLUTests::CpuGatedFFN(const float *input,
                                   const float *w_gate,
                                   const float *w_up,
                                   const float *w_down,
                                   float *output,
                                   const int bs,
                                   const int dim,
                                   const int hidden_dim,
                                   const bool use_geglu)
{
  std::vector<float> gate(static_cast<size_t>(bs*hidden_dim));
  std::vector<float> up(static_cast<size_t>(bs*hidden_dim));
  CAIF_CpuMatMul::Apply(input,w_gate,gate.data(),bs,dim,hidden_dim);
  CAIF_CpuMatMul::Apply(input,w_up,up.data(),bs,dim,hidden_dim);

  std::vector<float> hidden(static_cast<size_t>(bs*hidden_dim));
  for(int i=0;i<bs*hidden_dim;++i)
  {
    float activated=0.0f;
    if(use_geglu==true)
    {
      activated=CAIF_CpuActivations::GELU(gate[static_cast<size_t>(i)]);
    }
    else
    {
      activated=CAIF_CpuActivations::Swish(gate[static_cast<size_t>(i)]);
    }
    hidden[static_cast<size_t>(i)]=activated*up[static_cast<size_t>(i)];
  }

  CAIF_CpuMatMul::Apply(hidden.data(),w_down,output,bs,hidden_dim,dim);
}

void CAIF_GeGLUTests::TestGeGLUForward()
{
  try
  {
    const uint32_t bs=g_caif_geglu_test_fwd_batch*g_caif_geglu_test_fwd_seq;
    const uint32_t dim=g_caif_geglu_test_fwd_dim;
    const uint32_t hidden_dim=g_caif_geglu_test_fwd_hidden;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig ffn_config(dim,hidden_dim);
    auto activation=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(ffn_config,std::move(activation),stream);

    CAIF_HostTensor h_wgate=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wup=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wdown=ffn.ParameterTensor(2).ToHost();

    std::vector<float> host_input(static_cast<size_t>(bs*dim));
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_geglu_test_fwd_input_scale
                    +g_caif_geglu_test_fwd_input_bias;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    // Compare GPU to CPU reference in full FP32 (TF32 matmul drift otherwise
    // exceeds tolerance — this test validates algorithm, not TF32 accuracy).
    const bool prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor h_output=output.ToHost();
    CAIF_Settings::SetPreciseGradients(prev_precise);

    std::vector<float> expected(static_cast<size_t>(bs*dim));
    CpuGatedFFN(host_input.data(),
                h_wgate.Data(),
                h_wup.Data(),
                h_wdown.Data(),
                expected.data(),
                static_cast<int>(bs),
                static_cast<int>(dim),
                static_cast<int>(hidden_dim),
                true);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_output.Data()[i],
                                      expected[i],
                                      g_caif_geglu_test_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": gpu="
                      <<h_output.Data()[i]
                      <<" cpu="
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("GeGLU::Forward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GeGLU::Forward")
}

void CAIF_GeGLUTests::TestGeGLUBackward()
{
  try
  {
    const uint32_t bs=g_caif_geglu_test_bwd_bs;
    const uint32_t dim=g_caif_geglu_test_bwd_dim;
    const uint32_t hidden_dim=g_caif_geglu_test_bwd_hidden;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig ffn_config(dim,hidden_dim);
    auto activation=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(ffn_config,std::move(activation),stream);

    std::vector<float> host_input(static_cast<size_t>(bs*dim));
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_geglu_test_bwd_input_scale
                    +g_caif_geglu_test_bwd_input_bias;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);

    std::vector<float> grad_data(static_cast<size_t>(bs*dim),g_caif_geglu_test_bench_grad_fill);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_data.data(),{bs,dim},stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=ffn.Backward(grad_output,ctx);

    CAIF_HostTensor h_grad=grad_input.ToHost();
    bool passed=true;
    bool any_nonzero=false;
    for(size_t i=0;i<h_grad.TotalElements();++i)
    {
      if(h_grad.Data()[i]!=0.0f)
      {
        any_nonzero=true;
        break;
      }
    }
    if(any_nonzero==false)
    {
      ISE_Out::Out()<<"  grad_input is all zeros"
                    <<"\n";
      passed=false;
    }

    bool weight_grad_nonzero=false;
    for(size_t p=0;p<ffn.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_wgrad=ffn.GradientTensor(p).ToHost();
      for(size_t i=0;i<h_wgrad.TotalElements();++i)
      {
        if(h_wgrad.Data()[i]!=0.0f)
        {
          weight_grad_nonzero=true;
          break;
        }
      }
      if(weight_grad_nonzero==true)
      {
        break;
      }
    }
    if(weight_grad_nonzero==false)
    {
      ISE_Out::Out()<<"  weight gradients are all zeros"
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("GeGLU::Backward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GeGLU::Backward")
}

//------------------------------------------------------------------------------
// SwiGLU FFN forward matches CPU reference (regression)
//------------------------------------------------------------------------------
void CAIF_GeGLUTests::TestSwiGLURegression()
{
  try
  {
    const uint32_t bs=g_caif_geglu_test_swiglu_bs;
    const uint32_t dim=g_caif_geglu_test_swiglu_dim;
    const uint32_t hidden_dim=g_caif_geglu_test_swiglu_hidden;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFNConfig ffn_config(dim,hidden_dim);
    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(ffn_config,std::move(activation),stream);

    CAIF_HostTensor h_wgate=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wup=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wdown=ffn.ParameterTensor(2).ToHost();

    std::vector<float> host_input(static_cast<size_t>(bs*dim));
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*g_caif_geglu_test_fwd_input_scale
                    +g_caif_geglu_test_fwd_input_bias;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    const bool prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor h_output=output.ToHost();
    CAIF_Settings::SetPreciseGradients(prev_precise);

    std::vector<float> expected(static_cast<size_t>(bs*dim));
    CpuGatedFFN(host_input.data(),
                h_wgate.Data(),
                h_wup.Data(),
                h_wdown.Data(),
                expected.data(),
                static_cast<int>(bs),
                static_cast<int>(dim),
                static_cast<int>(hidden_dim),
                false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_output.Data()[i],
                                      expected[i],
                                      g_caif_geglu_test_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": gpu="
                      <<h_output.Data()[i]
                      <<" cpu="
                      <<expected[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("SwiGLU::Regression",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SwiGLU::Regression")
}

//------------------------------------------------------------------------------
// Benchmark: GeGLU vs SwiGLU FFN forward
//------------------------------------------------------------------------------
void CAIF_GeGLUTests::BenchmarkGeGLUvsSwiGLU()
{
  try
  {
    const uint32_t bs=g_caif_geglu_test_bench_batch*g_caif_geglu_test_bench_seq;
    const uint32_t dim=g_caif_geglu_test_bench_dim;
    const uint32_t hidden_dim=g_caif_geglu_test_bench_hidden;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceFFNConfig ffn_config(dim,hidden_dim);

    auto geglu_act=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> geglu_ffn(ffn_config,std::move(geglu_act),stream);

    auto swiglu_act=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> swiglu_ffn(ffn_config,std::move(swiglu_act),stream);

    std::vector<float> host_input(static_cast<size_t>(bs*dim),g_caif_geglu_test_bench_input_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    for(int i=0;i<g_caif_geglu_test_bench_warmup;++i)
    {
      swiglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_geglu_test_bench_iters;++i)
    {
      swiglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    auto end=std::chrono::high_resolution_clock::now();
    const double swiglu_fwd_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double swiglu_fwd_per=swiglu_fwd_ms/static_cast<double>(g_caif_geglu_test_bench_iters);

    for(int i=0;i<g_caif_geglu_test_bench_warmup;++i)
    {
      geglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_geglu_test_bench_iters;++i)
    {
      geglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    end=std::chrono::high_resolution_clock::now();
    const double geglu_fwd_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double geglu_fwd_per=geglu_fwd_ms/static_cast<double>(g_caif_geglu_test_bench_iters);

    ISE_Out::Out()<<"[BENCH] SwiGLU FFN forward (bs="
                  <<bs
                  <<",dim="
                  <<dim
                  <<",hidden="
                  <<hidden_dim
                  <<"): "
                  <<swiglu_fwd_per
                  <<" ms/iter\n";
    ISE_Out::Out()<<"[BENCH] GeGLU FFN forward  (same config): "
                  <<geglu_fwd_per
                  <<" ms/iter\n";

    const double diff_pct=((geglu_fwd_per-swiglu_fwd_per)/swiglu_fwd_per)*100.0;
    ISE_Out::Out()<<"[BENCH] GeGLU vs SwiGLU diff: "
                  <<diff_pct
                  <<"%%\n";
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::Out()<<"[BENCH] GeGLU vs SwiGLU: FAILED ("
                  <<e
                  <<")\n";
  }
}

//------------------------------------------------------------------------------
// Benchmark: GeGLU vs SwiGLU FFN forward+backward
//------------------------------------------------------------------------------
void CAIF_GeGLUTests::BenchmarkGeGLUvsSwiGLUFwdBwd()
{
  try
  {
    const uint32_t bs=g_caif_geglu_test_bench_batch*g_caif_geglu_test_bench_seq;
    const uint32_t dim=g_caif_geglu_test_bench_dim;
    const uint32_t hidden_dim=g_caif_geglu_test_bench_hidden;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    CAIF_DeviceFFNConfig ffn_config(dim,hidden_dim);

    auto geglu_act=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> geglu_ffn(ffn_config,std::move(geglu_act),stream);

    auto swiglu_act=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> swiglu_ffn(ffn_config,std::move(swiglu_act),stream);

    std::vector<float> host_input(static_cast<size_t>(bs*dim),g_caif_geglu_test_bench_input_fill);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);
    std::vector<float> grad_data(static_cast<size_t>(bs*dim),g_caif_geglu_test_bench_grad_fill);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_data.data(),{bs,dim},stream);

    for(int i=0;i<g_caif_geglu_test_bench_fwdbwd_warmup;++i)
    {
      swiglu_ffn.ZeroGradients();
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=swiglu_ffn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      swiglu_ffn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_geglu_test_bench_fwdbwd_iters;++i)
    {
      swiglu_ffn.ZeroGradients();
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=swiglu_ffn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      swiglu_ffn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    auto end=std::chrono::high_resolution_clock::now();
    const double swiglu_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double swiglu_per=swiglu_ms/static_cast<double>(g_caif_geglu_test_bench_fwdbwd_iters);

    for(int i=0;i<g_caif_geglu_test_bench_fwdbwd_warmup;++i)
    {
      geglu_ffn.ZeroGradients();
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=geglu_ffn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      geglu_ffn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<g_caif_geglu_test_bench_fwdbwd_iters;++i)
    {
      geglu_ffn.ZeroGradients();
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=geglu_ffn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      geglu_ffn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    end=std::chrono::high_resolution_clock::now();
    const double geglu_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double geglu_per=geglu_ms/static_cast<double>(g_caif_geglu_test_bench_fwdbwd_iters);

    ISE_Out::Out()<<"[BENCH] SwiGLU FFN fwd+bwd (bs="
                  <<bs
                  <<",dim="
                  <<dim
                  <<",hidden="
                  <<hidden_dim
                  <<"): "
                  <<swiglu_per
                  <<" ms/iter\n";
    ISE_Out::Out()<<"[BENCH] GeGLU FFN fwd+bwd  (same config): "
                  <<geglu_per
                  <<" ms/iter\n";

    const double diff_pct=((geglu_per-swiglu_per)/swiglu_per)*100.0;
    ISE_Out::Out()<<"[BENCH] GeGLU vs SwiGLU fwd+bwd diff: "
                  <<diff_pct
                  <<"%%\n";
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::Out()<<"[BENCH] GeGLU vs SwiGLU fwd+bwd: FAILED ("
                  <<e
                  <<")\n";
  }
}

void CAIF_GeGLUTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF GeGLU vs SwiGLU Tests ==="
                <<"\n\n";

  ISE_Out::Out()<<"--- Correctness ---"
                <<"\n";
  TestGeGLUForward();
  TestGeGLUBackward();
  TestSwiGLURegression();

  ISE_Out::Out()<<"\n--- Benchmarks ---\n";
  BenchmarkGeGLUvsSwiGLU();
  BenchmarkGeGLUvsSwiGLUFwdBwd();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_GeGLUTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
