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
// Attention logit soft-cap (Gemma-2/3): scores = cap*tanh(scores/cap), applied
// after the 1/sqrt(head_dim) scale and before the mask/softmax.
//
// The attention path is chosen by head_dim: a flash-supported head_dim (64)
// exercises the in-kernel cap on the flash path; an unsupported one (4) exercises
// the explicit (materialized-scores) path with the separate cap op. Both paths
// are checked.
//
// Tests per path:
//  - NoOpAtLargeCap: with a huge cap, cap*tanh(s/cap) -> s, so the output must
//    match no-cap. This validates the cap formula independently (a wrong outer
//    factor would not collapse to the identity).
//  - GradCheck: analytical backward vs finite differences with the cap ACTIVE
//    (small cap, scores exceed it). The backward multiplies the score gradient
//    by 1 - tanh^2(s/cap); if that factor is wrong the check fails.
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

constexpr uint32_t g_caif_sc_batch=1;
constexpr uint32_t g_caif_sc_seq=4;
constexpr uint32_t g_caif_sc_heads=2;
constexpr uint32_t g_caif_sc_flash_head_dim=64;   // flash-supported -> flash path
constexpr uint32_t g_caif_sc_explicit_head_dim=4; // unsupported     -> explicit path
constexpr float g_caif_sc_small_cap=2.0f;         // active: scores exceed it
constexpr float g_caif_sc_large_cap=1.0e3f;       // ~no-op: cap*tanh(s/cap)->s
constexpr float g_caif_sc_fd_h=1.0e-3f;
constexpr float g_caif_sc_noop_tol=5.0e-3f;
constexpr float g_caif_sc_grad_tol=3.0e-2f;

class CAIF_AttnSoftcapTest
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceMultiHeadAttentionConfig MakeConfig(const uint32_t head_dim,const float softcap);
    static float ForwardSum(const uint32_t head_dim,
                            const float softcap,
                            const std::vector<float> &host_input,
                            const CAIF_HostTensor &wq,
                            const CAIF_HostTensor &wk,
                            const CAIF_HostTensor &wv,
                            const CAIF_HostTensor &wo,
                            CAIF_CudaStream &stream,
                            CAIF_RunContext &ctx);
    static void TestForwardCap(const uint32_t head_dim,const std::string &label);
    static void TestGradCheck(const uint32_t head_dim,const float softcap,const std::string &label);
};

CAIF_DeviceMultiHeadAttentionConfig CAIF_AttnSoftcapTest::MakeConfig(const uint32_t head_dim,const float softcap)
{
  const uint32_t dim=g_caif_sc_heads*head_dim;
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             g_caif_sc_heads,
                                             g_caif_sc_heads,
                                             head_dim,
                                             true,
                                             false,
                                             10000.0f,
                                             0.0f);
  config.SetAttnLogitSoftcap(softcap);
  return config;
}

float CAIF_AttnSoftcapTest::ForwardSum(const uint32_t head_dim,
                                       const float softcap,
                                       const std::vector<float> &host_input,
                                       const CAIF_HostTensor &wq,
                                       const CAIF_HostTensor &wk,
                                       const CAIF_HostTensor &wv,
                                       const CAIF_HostTensor &wo,
                                       CAIF_CudaStream &stream,
                                       CAIF_RunContext &ctx)
{
  const uint32_t dim=g_caif_sc_heads*head_dim;
  CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,softcap);
  CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
  mha.ParameterTensor(0).CopyFromHost(wq.Data(),wq.TotalElements());
  mha.ParameterTensor(1).CopyFromHost(wk.Data(),wk.TotalElements());
  mha.ParameterTensor(2).CopyFromHost(wv.Data(),wv.TotalElements());
  mha.ParameterTensor(3).CopyFromHost(wo.Data(),wo.TotalElements());

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_caif_sc_batch,g_caif_sc_seq,dim},
                                                          stream);
  ctx.SetTraining(false);
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

void CAIF_AttnSoftcapTest::TestForwardCap(const uint32_t head_dim,const std::string &label)
{
  const std::string test_name=std::string("AttnSoftcap::ForwardCap::")+label;
  try
  {
    const uint32_t dim=g_caif_sc_heads*head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Large enough that the scaled scores exceed the small cap.
    std::vector<float> host_input(g_caif_sc_batch*g_caif_sc_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%17)*0.2f-1.2f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,0.0f);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    const float sum_nocap=ForwardSum(head_dim,0.0f,host_input,wq,wk,wv,wo,stream,ctx);
    const float sum_largecap=ForwardSum(head_dim,g_caif_sc_large_cap,host_input,wq,wk,wv,wo,stream,ctx);
    const float sum_smallcap=ForwardSum(head_dim,g_caif_sc_small_cap,host_input,wq,wk,wv,wo,stream,ctx);

    // Formula: cap*tanh(s/cap) -> s as cap grows, so a huge cap is a no-op.
    const bool noop_ok=(std::fabs(sum_nocap-sum_largecap)<g_caif_sc_noop_tol);
    // Applied: a small active cap must change the output.
    const bool active_ok=(std::fabs(sum_nocap-sum_smallcap)>g_caif_sc_noop_tol);
    const bool passed=(noop_ok==true&&active_ok==true);
    if(passed==false)
    {
      ISE_Out::Out()<<"  no-cap="<<sum_nocap
                    <<" large-cap="<<sum_largecap
                    <<" small-cap="<<sum_smallcap
                    <<" (noop_ok="<<noop_ok<<" active_ok="<<active_ok<<")\n";
    }
    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name.c_str())
}

void CAIF_AttnSoftcapTest::TestGradCheck(const uint32_t head_dim,const float softcap,const std::string &label)
{
  const std::string test_name=std::string("AttnSoftcap::GradCheck::")+label;
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    // FP32 finite differences must run in Accuracy_e (TF32 FD does not converge).
    CAIF_Settings::SetPreciseGradients(true);

    const uint32_t dim=g_caif_sc_heads*head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(g_caif_sc_batch*g_caif_sc_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%13)*0.2f-0.6f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,softcap);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {g_caif_sc_batch,g_caif_sc_seq,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(host_input.size(),1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                              {g_caif_sc_batch,g_caif_sc_seq,dim},
                                                              stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);
    const CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=g_caif_sc_fd_h;
      input_minus[i]-=g_caif_sc_fd_h;

      const float sum_plus=ForwardSum(head_dim,softcap,input_plus,wq,wk,wv,wo,stream,ctx);
      const float sum_minus=ForwardSum(head_dim,softcap,input_minus,wq,wk,wv,wo,stream,ctx);

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_sc_fd_h);
      const float analytical=host_grad.Data()[i];
      if(std::fabs(numerical-analytical)>g_caif_sc_grad_tol*std::max(1.0f,std::fabs(analytical)))
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

void CAIF_AttnSoftcapTest::RunAll()
{
  ISE_Out::Out()<<"=== Attention logit soft-cap (Gemma-2/3) ==="
                <<"\n\n";
  // Forward (both paths): cap formula collapses to identity at a huge cap, and a
  // small cap actually changes the output. Backward derivative is validated by
  // the FP32 explicit gradcheck below. A flash-path finite-difference gradcheck
  // is NOT viable — the TF32 tensor-core flash kernel makes FD non-convergent
  // (verified: it fails identically with the cap disabled), and the suite has no
  // flash FD-gradcheck precedent. The flash backward applies the identical
  // 1 - tanh^2 derivative the explicit gradcheck validates.
  TestForwardCap(g_caif_sc_flash_head_dim,"Flash");
  TestForwardCap(g_caif_sc_explicit_head_dim,"Explicit");
  TestGradCheck(g_caif_sc_explicit_head_dim,g_caif_sc_small_cap,"Explicit");
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_AttnSoftcapTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
