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
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

using namespace instance;

static void ReportResult(const char *test_name,bool passed)
{
  CAIF_TestHarness::Report(test_name,passed);
}

static bool FloatEqual(float a,float b,float tolerance=1e-4f)
{
  return CAIF_TestHarness::FloatEqual(a,b,tolerance);
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// CPU reference functions (primitives in caif_cpu_reference/)
//------------------------------------------------------------------------------

// CPU reference: gated FFN forward
// gate = input @ W_gate -> activation(gate)
// up = input @ W_up
// hidden = activation(gate) * up
// output = hidden @ W_down
static void CpuGatedFFN(const float *input,
                         const float *w_gate,
                         const float *w_up,
                         const float *w_down,
                         float *output,
                         int bs,
                         int dim,
                         int hidden_dim,
                         bool use_geglu)
{
  std::vector<float> gate(bs*hidden_dim);
  std::vector<float> up(bs*hidden_dim);
  CAIF_CpuMatMul::Apply(input,w_gate,gate.data(),bs,dim,hidden_dim);
  CAIF_CpuMatMul::Apply(input,w_up,up.data(),bs,dim,hidden_dim);

  // Apply activation to gate and multiply with up
  std::vector<float> hidden(bs*hidden_dim);
  for(int i=0;i<bs*hidden_dim;++i)
  {
    float activated=0.0f;
    if(use_geglu==true)
    {
      activated=CAIF_CpuActivations::GELU(gate[i]);
    }
    else
    {
      activated=CAIF_CpuActivations::Swish(gate[i]);
    }
    hidden[i]=activated*up[i];
  }

  CAIF_CpuMatMul::Apply(hidden.data(),w_down,output,bs,hidden_dim,dim);
}

//------------------------------------------------------------------------------
// Test 1: GeGLU FFN forward matches CPU reference
//------------------------------------------------------------------------------
static void TestGeGLUForward()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=16;
    const uint32_t hidden_dim=32;
    const uint32_t bs=batch*seq_len;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFN<float,float>::FFNConfig_t ffn_config;
    ffn_config.dim=dim;
    ffn_config.ffn_dim=hidden_dim;
    auto activation=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(ffn_config,std::move(activation),stream);

    // Get weights
    CAIF_HostTensor h_wgate=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wup=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wdown=ffn.ParameterTensor(2).ToHost();

    // Deterministic input
    std::vector<float> host_input(bs*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.04f-0.3f;
    }
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    // Compare GPU algorithm to CPU reference in full FP32 (TF32 matmul drift
     // otherwise exceeds tolerance — this test validates algorithm, not
     // TF32 accuracy).
    const bool prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor h_output=output.ToHost();
    CAIF_Settings::SetPreciseGradients(prev_precise);

    // CPU reference
    std::vector<float> expected(bs*dim);
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
      if(FloatEqual(h_output.Data()[i],expected[i],1e-2f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": gpu="<<h_output.Data()[i]
                 <<" cpu="<<expected[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("GeGLU::Forward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GeGLU::Forward")
}

//------------------------------------------------------------------------------
// Test 2: GeGLU backward produces non-zero gradients
//------------------------------------------------------------------------------
static void TestGeGLUBackward()
{
  try
  {
    const uint32_t bs=8;
    const uint32_t dim=16;
    const uint32_t hidden_dim=32;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFN<float,float>::FFNConfig_t ffn_config;
    ffn_config.dim=dim;
    ffn_config.ffn_dim=hidden_dim;
    auto activation=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(ffn_config,std::move(activation),stream);

    std::vector<float> host_input(bs*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.03f-0.2f;
    }
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);

    std::vector<float> grad_data(bs*dim,1.0f);
    auto grad_output=CAIF_DeviceTensor::FromHostData(grad_data.data(),{bs,dim},stream);
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
      std::cout<<"  grad_input is all zeros\n";
      passed=false;
    }

    // Check weight gradients
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
      std::cout<<"  weight gradients are all zeros\n";
      passed=false;
    }

    ReportResult("GeGLU::Backward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("GeGLU::Backward")
}

//------------------------------------------------------------------------------
// Test 3: SwiGLU FFN forward matches CPU reference (regression)
//------------------------------------------------------------------------------
static void TestSwiGLURegression()
{
  try
  {
    const uint32_t bs=8;
    const uint32_t dim=16;
    const uint32_t hidden_dim=32;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceFFN<float,float>::FFNConfig_t ffn_config;
    ffn_config.dim=dim;
    ffn_config.ffn_dim=hidden_dim;
    auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> ffn(ffn_config,std::move(activation),stream);

    CAIF_HostTensor h_wgate=ffn.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wup=ffn.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wdown=ffn.ParameterTensor(2).ToHost();

    std::vector<float> host_input(bs*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.04f-0.3f;
    }
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    const bool prev_precise=CAIF_Settings::PreciseGradients();
    CAIF_Settings::SetPreciseGradients(true);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor output=ffn.Forward(input,ctx);
    CAIF_HostTensor h_output=output.ToHost();
    CAIF_Settings::SetPreciseGradients(prev_precise);

    std::vector<float> expected(bs*dim);
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
      if(FloatEqual(h_output.Data()[i],expected[i],1e-2f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": gpu="<<h_output.Data()[i]
                 <<" cpu="<<expected[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("SwiGLU::Regression",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SwiGLU::Regression")
}

//------------------------------------------------------------------------------
// Benchmark: GeGLU vs SwiGLU FFN forward
//------------------------------------------------------------------------------
static void BenchmarkGeGLUvsSwiGLU()
{
  try
  {
    const uint32_t batch=8;
    const uint32_t seq_len=256;
    const uint32_t dim=512;
    const uint32_t hidden_dim=1024;
    const uint32_t bs=batch*seq_len;
    const int warmup_iters=10;
    const int bench_iters=100;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // GeGLU FFN
    CAIF_DeviceFFN<float,float>::FFNConfig_t ffn_config;
    ffn_config.dim=dim;
    ffn_config.ffn_dim=hidden_dim;

    // GeGLU FFN
    auto geglu_act=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> geglu_ffn(ffn_config,std::move(geglu_act),stream);

    // SwiGLU FFN
    auto swiglu_act=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> swiglu_ffn(ffn_config,std::move(swiglu_act),stream);

    std::vector<float> host_input(bs*dim,0.1f);
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);

    // Warmup SwiGLU
    for(int i=0;i<warmup_iters;++i)
    {
      swiglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    // Bench SwiGLU forward
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<bench_iters;++i)
    {
      swiglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    auto end=std::chrono::high_resolution_clock::now();
    const double swiglu_fwd_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double swiglu_fwd_per=swiglu_fwd_ms/static_cast<double>(bench_iters);

    // Warmup GeGLU
    for(int i=0;i<warmup_iters;++i)
    {
      geglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    // Bench GeGLU forward
    start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<bench_iters;++i)
    {
      geglu_ffn.Forward(input,ctx);
    }
    cudaStreamSynchronize(stream.Handle());
    end=std::chrono::high_resolution_clock::now();
    const double geglu_fwd_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const double geglu_fwd_per=geglu_fwd_ms/static_cast<double>(bench_iters);

    std::cout<<"[BENCH] SwiGLU FFN forward (bs="<<bs
             <<",dim="<<dim
             <<",hidden="<<hidden_dim
             <<"): "<<swiglu_fwd_per<<" ms/iter\n";
    std::cout<<"[BENCH] GeGLU FFN forward  (same config): "
             <<geglu_fwd_per<<" ms/iter\n";

    const double diff_pct=((geglu_fwd_per-swiglu_fwd_per)/swiglu_fwd_per)*100.0;
    std::cout<<"[BENCH] GeGLU vs SwiGLU diff: "<<diff_pct<<"%%\n";
  }
  catch(const CAIF_Exception &e)
  {
    std::cout<<"[BENCH] GeGLU vs SwiGLU: FAILED ("<<e<<")\n";
  }
  catch(const std::exception &e)
  {
    std::cout<<"[BENCH] GeGLU vs SwiGLU: FAILED ("<<e.what()<<")\n";
  }
}

//------------------------------------------------------------------------------
// Benchmark: GeGLU vs SwiGLU FFN forward+backward
//------------------------------------------------------------------------------
static void BenchmarkGeGLUvsSwiGLUFwdBwd()
{
  try
  {
    const uint32_t batch=8;
    const uint32_t seq_len=256;
    const uint32_t dim=512;
    const uint32_t hidden_dim=1024;
    const uint32_t bs=batch*seq_len;
    const int warmup_iters=5;
    const int bench_iters=50;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    CAIF_DeviceFFN<float,float>::FFNConfig_t ffn_config;
    ffn_config.dim=dim;
    ffn_config.ffn_dim=hidden_dim;

    auto geglu_act=std::make_unique<CAIF_DeviceGeGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> geglu_ffn(ffn_config,std::move(geglu_act),stream);

    auto swiglu_act=std::make_unique<CAIF_DeviceSwiGLUActivation<float,float>>();
    CAIF_DeviceFFN<float,float> swiglu_ffn(ffn_config,std::move(swiglu_act),stream);

    std::vector<float> host_input(bs*dim,0.1f);
    auto input=CAIF_DeviceTensor::FromHostData(host_input.data(),{bs,dim},stream);
    std::vector<float> grad_data(bs*dim,1.0f);
    auto grad_output=CAIF_DeviceTensor::FromHostData(grad_data.data(),{bs,dim},stream);

    // Warmup SwiGLU
    for(int i=0;i<warmup_iters;++i)
    {
      swiglu_ffn.ZeroGradients();
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=swiglu_ffn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      swiglu_ffn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    // Bench SwiGLU fwd+bwd
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<bench_iters;++i)
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
    const double swiglu_per=swiglu_ms/static_cast<double>(bench_iters);

    // Warmup GeGLU
    for(int i=0;i<warmup_iters;++i)
    {
      geglu_ffn.ZeroGradients();
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor out=geglu_ffn.Forward(input,ctx);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      geglu_ffn.Backward(grad_output,ctx);
    }
    cudaStreamSynchronize(stream.Handle());

    // Bench GeGLU fwd+bwd
    start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<bench_iters;++i)
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
    const double geglu_per=geglu_ms/static_cast<double>(bench_iters);

    std::cout<<"[BENCH] SwiGLU FFN fwd+bwd (bs="<<bs
             <<",dim="<<dim
             <<",hidden="<<hidden_dim
             <<"): "<<swiglu_per<<" ms/iter\n";
    std::cout<<"[BENCH] GeGLU FFN fwd+bwd  (same config): "
             <<geglu_per<<" ms/iter\n";

    const double diff_pct=((geglu_per-swiglu_per)/swiglu_per)*100.0;
    std::cout<<"[BENCH] GeGLU vs SwiGLU fwd+bwd diff: "<<diff_pct<<"%%\n";
  }
  catch(const CAIF_Exception &e)
  {
    std::cout<<"[BENCH] GeGLU vs SwiGLU fwd+bwd: FAILED ("<<e<<")\n";
  }
  catch(const std::exception &e)
  {
    std::cout<<"[BENCH] GeGLU vs SwiGLU fwd+bwd: FAILED ("<<e.what()<<")\n";
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF GeGLU vs SwiGLU Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  // Correctness tests
  std::cout<<"--- Correctness ---\n";
  TestGeGLUForward();
  TestGeGLUBackward();
  TestSwiGLURegression();

  // Benchmarks (sequential — shared GPU)
  std::cout<<"\n--- Benchmarks ---\n";
  BenchmarkGeGLUvsSwiGLU();
  BenchmarkGeGLUvsSwiGLUFwdBwd();
#else
  std::cout<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

  return CAIF_TestHarness::FinalExitCode();
}
