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
// ALiBi linear position bias (MPT / BLOOM): adds slope_h*(k-q) to each score
// before the softmax, with a per-head slope that is a geometric sequence in the
// head index. Implemented in the flash kernels (tc + scalar forward, both
// backward) and the explicit path (a bias add before the mask, re-applied in
// the backward's score recompute). Replaces rotary/learned position encoding.
//
// The path is chosen by head_dim: 64 -> flash, 4 -> explicit. Tests per path:
//  - ForwardAlibi: ALiBi on vs off changes the output (the bias shifts the
//    softmax toward recent keys). This proves the bias is wired and applied on
//    each path.
//  - GradCheck (explicit, FP32): analytical backward vs finite differences with
//    ALiBi on. The backward re-applies the same forward bias when it recomputes
//    scores; if that recompute bias is missing the check fails. The flash path
//    is not FD-gradcheckable (TF32 -> non-convergent, per the soft-cap test);
//    its backward applies the same bias the explicit path validates here.
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

constexpr uint32_t g_caif_alibi_batch=1;
constexpr uint32_t g_caif_alibi_seq=6;
constexpr uint32_t g_caif_alibi_heads=2;
constexpr uint32_t g_caif_alibi_flash_head_dim=64;
constexpr uint32_t g_caif_alibi_explicit_head_dim=4;
constexpr float g_caif_alibi_fd_h=1.0e-3f;
constexpr float g_caif_alibi_active_tol=5.0e-3f;
constexpr float g_caif_alibi_grad_tol=3.0e-2f;

class CAIF_AttnAlibiTest
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceMultiHeadAttentionConfig MakeConfig(const uint32_t head_dim,const bool use_alibi);
    static float ForwardSum(const uint32_t head_dim,
                            const bool use_alibi,
                            const std::vector<float> &host_input,
                            const CAIF_HostTensor &wq,
                            const CAIF_HostTensor &wk,
                            const CAIF_HostTensor &wv,
                            const CAIF_HostTensor &wo,
                            CAIF_CudaStream &stream,
                            CAIF_RunContext &ctx);
    static void TestForwardAlibi(const uint32_t head_dim,const std::string &label);
    static void TestGradCheck(const uint32_t head_dim,const std::string &label);
};

CAIF_DeviceMultiHeadAttentionConfig CAIF_AttnAlibiTest::MakeConfig(const uint32_t head_dim,const bool use_alibi)
{
  const uint32_t dim=g_caif_alibi_heads*head_dim;
  // ALiBi replaces rotary position encoding, so use_rope is false.
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             g_caif_alibi_heads,
                                             g_caif_alibi_heads,
                                             head_dim,
                                             true,
                                             false,
                                             10000.0f,
                                             0.0f);
  config.SetUseAlibi(use_alibi);
  return config;
}

float CAIF_AttnAlibiTest::ForwardSum(const uint32_t head_dim,
                                     const bool use_alibi,
                                     const std::vector<float> &host_input,
                                     const CAIF_HostTensor &wq,
                                     const CAIF_HostTensor &wk,
                                     const CAIF_HostTensor &wv,
                                     const CAIF_HostTensor &wo,
                                     CAIF_CudaStream &stream,
                                     CAIF_RunContext &ctx)
{
  const uint32_t dim=g_caif_alibi_heads*head_dim;
  CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,use_alibi);
  CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
  mha.ParameterTensor(0).CopyFromHost(wq.Data(),wq.TotalElements());
  mha.ParameterTensor(1).CopyFromHost(wk.Data(),wk.TotalElements());
  mha.ParameterTensor(2).CopyFromHost(wv.Data(),wv.TotalElements());
  mha.ParameterTensor(3).CopyFromHost(wo.Data(),wo.TotalElements());

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_caif_alibi_batch,g_caif_alibi_seq,dim},
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

void CAIF_AttnAlibiTest::TestForwardAlibi(const uint32_t head_dim,const std::string &label)
{
  const std::string test_name=std::string("AttnAlibi::ForwardAlibi::")+label;
  try
  {
    const uint32_t dim=g_caif_alibi_heads*head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(g_caif_alibi_batch*g_caif_alibi_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%17)*0.15f-0.7f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,false);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    const float sum_off=ForwardSum(head_dim,false,host_input,wq,wk,wv,wo,stream,ctx);
    const float sum_on=ForwardSum(head_dim,true,host_input,wq,wk,wv,wo,stream,ctx);

    // ALiBi shifts the softmax toward recent keys, so the output changes.
    const bool active_ok=(std::fabs(sum_off-sum_on)>g_caif_alibi_active_tol);
    if(active_ok==false)
    {
      ISE_Out::Out()<<"  alibi-off="<<sum_off
                    <<" alibi-on="<<sum_on
                    <<" (diff="<<std::fabs(sum_off-sum_on)<<")\n";
    }
    CAIF_TestHarness::Report(test_name.c_str(),active_ok);
  }
  CAIF_TEST_CATCH_BLOCK(test_name.c_str())
}

void CAIF_AttnAlibiTest::TestGradCheck(const uint32_t head_dim,const std::string &label)
{
  const std::string test_name=std::string("AttnAlibi::GradCheck::")+label;
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(true);

    const uint32_t dim=g_caif_alibi_heads*head_dim;
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(g_caif_alibi_batch*g_caif_alibi_seq*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%13)*0.2f-0.6f;
    }

    CAIF_DeviceMultiHeadAttentionConfig config=MakeConfig(head_dim,true);
    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    const CAIF_HostTensor wq=mha.ParameterTensor(0).ToHost();
    const CAIF_HostTensor wk=mha.ParameterTensor(1).ToHost();
    const CAIF_HostTensor wv=mha.ParameterTensor(2).ToHost();
    const CAIF_HostTensor wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {g_caif_alibi_batch,g_caif_alibi_seq,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    mha.Forward(input,ctx);

    std::vector<float> grad_ones(host_input.size(),1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {g_caif_alibi_batch,g_caif_alibi_seq,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);
    const CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=g_caif_alibi_fd_h;
      input_minus[i]-=g_caif_alibi_fd_h;

      const float sum_plus=ForwardSum(head_dim,true,input_plus,wq,wk,wv,wo,stream,ctx);
      const float sum_minus=ForwardSum(head_dim,true,input_minus,wq,wk,wv,wo,stream,ctx);

      const float numerical=(sum_plus-sum_minus)/(2.0f*g_caif_alibi_fd_h);
      const float analytical=host_grad.Data()[i];
      if(std::fabs(numerical-analytical)>g_caif_alibi_grad_tol*std::max(1.0f,std::fabs(analytical)))
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

void CAIF_AttnAlibiTest::RunAll()
{
  ISE_Out::Out()<<"=== ALiBi linear position bias (MPT / BLOOM) ==="
                <<"\n\n";
  TestForwardAlibi(g_caif_alibi_flash_head_dim,"Flash");
  TestForwardAlibi(g_caif_alibi_explicit_head_dim,"Explicit");
  TestGradCheck(g_caif_alibi_explicit_head_dim,"Explicit");
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_AttnAlibiTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
