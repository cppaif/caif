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
// Sliding-window attention (Mistral / Gemma-2): query q attends only to keys in
// [q-window+1, q]. Implemented as an extra mask term (q - k >= window -> -inf)
// alongside the causal mask, in the flash kernels and the explicit path.
//
// The path is chosen by head_dim: 64 -> flash, 4 -> explicit. Tests per path:
//  - ForwardWindow: a window >= seq_len masks nothing, so the output matches the
//    no-window (full causal) run; a small window < seq_len changes the output.
//    This proves the mask is correct (no-op when wide) and applied (when narrow).
//  - GradCheck (explicit, FP32): analytical backward vs finite differences with a
//    narrow window. The backward zeroes the gradient at the window-masked
//    positions; if that grad-mask is missing the check fails. The flash path is
//    not FD-gradcheckable (TF32 -> non-convergent, per the soft-cap test); its
//    backward applies the same masked-position skip the explicit path validates.
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

constexpr uint32_t g_caif_sw_batch=1;
constexpr uint32_t g_caif_sw_seq=6;
constexpr uint32_t g_caif_sw_heads=2;
constexpr uint32_t g_caif_sw_flash_head_dim=64;
constexpr uint32_t g_caif_sw_explicit_head_dim=4;
constexpr int g_caif_sw_small_window=2;   // < seq: masks far keys
constexpr int g_caif_sw_wide_window=6;    // == seq: masks nothing (no-op)
constexpr float g_caif_sw_fd_h=1.0e-3f;
constexpr float g_caif_sw_noop_tol=5.0e-3f;
constexpr float g_caif_sw_grad_tol=3.0e-2f;

class CAIF_AttnSlidingWindowTest
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceMultiHeadAttentionConfig MakeConfig(const uint32_t head_dim,const int window);
    static float ForwardSum(const uint32_t head_dim,
                            const int window,
                            const std::vector<float> &host_input,
                            const CAIF_HostTensor &wq,
                            const CAIF_HostTensor &wk,
                            const CAIF_HostTensor &wv,
                            const CAIF_HostTensor &wo,
                            CAIF_CudaStream &stream,
                            CAIF_RunContext &ctx);
    static void TestForwardWindow(const uint32_t head_dim,const std::string &label);
    static void TestGradCheck(const uint32_t head_dim,const int window,const std::string &label);
};

CAIF_DeviceMultiHeadAttentionConfig CAIF_AttnSlidingWindowTest::MakeConfig(const uint32_t head_dim,const int window)
{
  const uint32_t dim=g_caif_sw_heads*head_dim;
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             g_caif_sw_heads,
                                             g_caif_sw_heads,
                                             head_dim,
                                             true,
                                             false,
                                             10000.0f,
                                             0.0f);
  config.SetSlidingWindow(window);
  return config;
}

float CAIF_AttnSlidingWindowTest::ForwardSum(const uint32_t head_dim,
                                             const int window,
                                             const std::vector<float> &host_input,
                                             const CAIF_HostTensor &wq,
                                             const CAIF_HostTensor &wk,
                                             const CAIF_HostTensor &wv,
                                             const CAIF_HostTensor &wo,
                                             CAIF_CudaStream &stream,
                                             CAIF_RunContext &ctx)
{
  const uint32_t dim=g_caif_sw_heads*head_dim;
  CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,window);
  CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
  mha.ParameterTensor(0).CopyFromHost(wq.Data(),wq.TotalElements());
  mha.ParameterTensor(1).CopyFromHost(wk.Data(),wk.TotalElements());
  mha.ParameterTensor(2).CopyFromHost(wv.Data(),wv.TotalElements());
  mha.ParameterTensor(3).CopyFromHost(wo.Data(),wo.TotalElements());

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_caif_sw_batch,g_caif_sw_seq,dim},
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

void CAIF_AttnSlidingWindowTest::TestForwardWindow(const uint32_t head_dim,const std::string &label)
{
  const std::string test_name=std::string("AttnSlidingWindow::ForwardWindow::")+label;
  try
  {
    const uint32_t dim=g_caif_sw_heads*head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(g_caif_sw_batch*g_caif_sw_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%17)*0.15f-0.7f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,0);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    const float sum_nowin=ForwardSum(head_dim,0,host_input,wq,wk,wv,wo,stream,ctx);
    const float sum_wide=ForwardSum(head_dim,g_caif_sw_wide_window,host_input,wq,wk,wv,wo,stream,ctx);
    const float sum_small=ForwardSum(head_dim,g_caif_sw_small_window,host_input,wq,wk,wv,wo,stream,ctx);

    // window >= seq masks nothing -> matches no-window (full causal).
    const bool noop_ok=(std::fabs(sum_nowin-sum_wide)<g_caif_sw_noop_tol);
    // a narrow window masks far keys -> output changes.
    const bool active_ok=(std::fabs(sum_nowin-sum_small)>g_caif_sw_noop_tol);
    const bool passed=(noop_ok==true&&active_ok==true);
    if(passed==false)
    {
      ISE_Out::Out()<<"  no-window="<<sum_nowin
                    <<" wide="<<sum_wide
                    <<" small="<<sum_small
                    <<" (noop_ok="<<noop_ok<<" active_ok="<<active_ok<<")\n";
    }
    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name.c_str())
}

void CAIF_AttnSlidingWindowTest::TestGradCheck(const uint32_t head_dim,const int window,const std::string &label)
{
  const std::string test_name=std::string("AttnSlidingWindow::GradCheck::")+label;
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(true);

    const uint32_t dim=g_caif_sw_heads*head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(g_caif_sw_batch*g_caif_sw_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%13)*0.2f-0.6f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,window);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {g_caif_sw_batch,g_caif_sw_seq,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(host_input.size(),1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                              {g_caif_sw_batch,g_caif_sw_seq,dim},
                                                              stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);
    const CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=g_caif_sw_fd_h;
      input_minus[i]-=g_caif_sw_fd_h;

      const float sum_plus=ForwardSum(head_dim,window,input_plus,wq,wk,wv,wo,stream,ctx);
      const float sum_minus=ForwardSum(head_dim,window,input_minus,wq,wk,wv,wo,stream,ctx);

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_sw_fd_h);
      const float analytical=host_grad.Data()[i];
      if(std::fabs(numerical-analytical)>g_caif_sw_grad_tol*std::max(1.0f,std::fabs(analytical)))
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

void CAIF_AttnSlidingWindowTest::RunAll()
{
  ISE_Out::Out()<<"=== Sliding-window attention (Mistral / Gemma-2) ==="
                <<"\n\n";
  TestForwardWindow(g_caif_sw_flash_head_dim,"Flash");
  TestForwardWindow(g_caif_sw_explicit_head_dim,"Explicit");
  TestGradCheck(g_caif_sw_explicit_head_dim,g_caif_sw_small_window,"Explicit");
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_AttnSlidingWindowTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
