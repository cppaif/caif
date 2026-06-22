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
// Attention dropout (training only): drops softmax weights with the configured
// rate and scales the survivors by 1/keep (inverted dropout), with the mask
// drawn from the deterministic ctx RNG and cached so the backward gates the
// value gradient and the attention gradient by the identical mask. Applied on
// the explicit path; a training pass with dropout>0 routes away from flash.
//
// Tests (explicit path, head_dim=4):
//  - Forward: a training pass with dropout differs from the eval pass (no
//    dropout), and two training passes with the same seed match (deterministic
//    mask). This proves dropout is applied and reproducible.
//  - GradCheck (FP32): analytical backward vs finite differences, with the seed
//    reset before every forward so all passes share one mask. The backward uses
//    attn_dropped for the value gradient and gates the attention gradient by the
//    same mask; if either is missing the check fails.
//------------------------------------------------------------------------------
#include "caif_device_multi_head_attention.h"
#include "caif_device_multi_head_attention_config.h"
#include "caif_device_tensor.h"
#include "caif_host_tensor.h"
#include "caif_run_context.h"
#include "caif_cuda_stream.h"
#include "caif_settings.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_drop_batch=1;
constexpr uint32_t g_caif_drop_seq=6;
constexpr uint32_t g_caif_drop_heads=2;
constexpr uint32_t g_caif_drop_head_dim=4;
constexpr float g_caif_drop_rate=0.3f;
constexpr uint64_t g_caif_drop_seed=0x5151ABCDULL;
constexpr float g_caif_drop_fd_h=1.0e-3f;
constexpr float g_caif_drop_active_tol=5.0e-3f;
constexpr float g_caif_drop_grad_tol=3.0e-2f;

class CAIF_AttnDropoutTest
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceMultiHeadAttentionConfig MakeConfig();
    static float ForwardSum(const bool training,
                            const std::vector<float> &host_input,
                            const CAIF_HostTensor &wq,
                            const CAIF_HostTensor &wk,
                            const CAIF_HostTensor &wv,
                            const CAIF_HostTensor &wo,
                            CAIF_CudaStream &stream,
                            CAIF_RunContext &ctx);
    static void TestForwardDropout();
    static void TestGradCheck();
};

CAIF_DeviceMultiHeadAttentionConfig CAIF_AttnDropoutTest::MakeConfig()
{
  const uint32_t dim=g_caif_drop_heads*g_caif_drop_head_dim;
  // dropout_rate is the eighth required ctor field; use_rope is false here.
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             g_caif_drop_heads,
                                             g_caif_drop_heads,
                                             g_caif_drop_head_dim,
                                             true,
                                             false,
                                             10000.0f,
                                             g_caif_drop_rate);
  return config;
}

float CAIF_AttnDropoutTest::ForwardSum(const bool training,
                                       const std::vector<float> &host_input,
                                       const CAIF_HostTensor &wq,
                                       const CAIF_HostTensor &wk,
                                       const CAIF_HostTensor &wv,
                                       const CAIF_HostTensor &wo,
                                       CAIF_CudaStream &stream,
                                       CAIF_RunContext &ctx)
{
  const uint32_t dim=g_caif_drop_heads*g_caif_drop_head_dim;
  CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig();
  CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
  mha.ParameterTensor(0).CopyFromHost(wq.Data(),wq.TotalElements());
  mha.ParameterTensor(1).CopyFromHost(wk.Data(),wk.TotalElements());
  mha.ParameterTensor(2).CopyFromHost(wv.Data(),wv.TotalElements());
  mha.ParameterTensor(3).CopyFromHost(wo.Data(),wo.TotalElements());

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_caif_drop_batch,g_caif_drop_seq,dim},
                                                          stream);
  // Reset the seed so the dropout mask is identical across every forward (the
  // counter resets to 0), which makes the layer deterministic and gradcheckable.
  ctx.SetRandomSeed(g_caif_drop_seed);
  ctx.SetTraining(training);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor output=mha.Forward(input,ctx);
  CAIF_HostTensor host_output=output.ToHost();

  float sum=0.0f;
  for(size_t i=0;i<host_output.TotalElements();++i)
  {
    sum+=host_output.Data()[i];
  }
  return sum;
}

void CAIF_AttnDropoutTest::TestForwardDropout()
{
  const std::string test_name="AttnDropout::ForwardDropout::Explicit";
  try
  {
    const uint32_t dim=g_caif_drop_heads*g_caif_drop_head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(g_caif_drop_batch*g_caif_drop_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%17)*0.15f-0.7f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig();
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    const float sum_eval=ForwardSum(false,host_input,wq,wk,wv,wo,stream,ctx);
    const float sum_train_a=ForwardSum(true,host_input,wq,wk,wv,wo,stream,ctx);
    const float sum_train_b=ForwardSum(true,host_input,wq,wk,wv,wo,stream,ctx);

    // Training dropout changes the output relative to eval (some weights are
    // zeroed), and is deterministic for a fixed seed (the two train passes
    // match).
    const bool active_ok=(std::fabs(sum_eval-sum_train_a)>g_caif_drop_active_tol);
    const bool deterministic_ok=(std::fabs(sum_train_a-sum_train_b)<g_caif_drop_active_tol);
    const bool passed=(active_ok==true&&deterministic_ok==true);
    if(passed==false)
    {
      ISE_Out::Out()<<"  eval="<<sum_eval
                    <<" train_a="<<sum_train_a
                    <<" train_b="<<sum_train_b
                    <<" (active_ok="<<active_ok<<" deterministic_ok="<<deterministic_ok<<")\n";
    }
    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name.c_str())
}

void CAIF_AttnDropoutTest::TestGradCheck()
{
  const std::string test_name="AttnDropout::GradCheck::Explicit";
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(true);

    const uint32_t dim=g_caif_drop_heads*g_caif_drop_head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(g_caif_drop_batch*g_caif_drop_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%13)*0.2f-0.6f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig();
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {g_caif_drop_batch,g_caif_drop_seq,dim},
                                                            stream);
    // Same fixed seed as the FD forwards below, so the analytical backward sees
    // the identical dropout mask.
    ctx.SetRandomSeed(g_caif_drop_seed);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(host_input.size(),1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {g_caif_drop_batch,g_caif_drop_seq,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);
    const CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=g_caif_drop_fd_h;
      input_minus[i]-=g_caif_drop_fd_h;

      const float sum_plus=ForwardSum(true,input_plus,wq,wk,wv,wo,stream,ctx);
      const float sum_minus=ForwardSum(true,input_minus,wq,wk,wv,wo,stream,ctx);

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_drop_fd_h);
      const float analytical=host_grad.Data()[i];
      if(std::fabs(numerical-analytical)>g_caif_drop_grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dx mismatch at "<<i
                      <<": analytical="<<analytical
                      <<" numerical="<<numerical
                      <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    CAIF_Settings::SetPreciseGradients(prev_precise);
    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  catch(...)
  {
    CAIF_Settings::SetPreciseGradients(prev_precise);
    throw;
  }
}

void CAIF_AttnDropoutTest::RunAll()
{
  ISE_Out::Out()<<"=== Attention dropout (training-only) ==="
                <<"\n\n";
  TestForwardDropout();
  TestGradCheck();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_AttnDropoutTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
