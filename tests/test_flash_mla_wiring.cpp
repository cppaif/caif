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
// Confirm the fused flash-prefill kernel is
// wired correctly into CAIF_DeviceMLAttention::ForwardImpl. A DSv2-Lite-shaped
// (qk=192, v=128) layer is run twice on the same input/weights — once with the
// fused kernel enabled (CAIF_Settings::FlashMlaPrefill) and once forced onto the
// explicit O(seq^2) path — and the outputs must match at a TF32-class tolerance.
// This validates the whole wiring (scale, q/k/v assembly, causal flag, shapes):
// any wiring bug attends/writes the wrong elements and diverges by O(magnitude),
// far above the tolerance. The kernel math itself is covered by the kernel-level
// parity in test_flash_mla_parity.
//------------------------------------------------------------------------------
#include "caif_device_ml_attention.h"
#include "caif_device_ml_attention_config.h"
#include "caif_settings.h"
#include "caif_device_tensor.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

// DSv2-Lite-shaped config: qk_nope + qk_rope = 128 + 64 = 192, v = 128 — the
// decoupled config the fused MLA flash-prefill kernel supports.
constexpr uint32_t g_wire_dim=256;
constexpr uint32_t g_wire_heads=2;
constexpr uint32_t g_wire_q_lora=256;
constexpr uint32_t g_wire_kv_lora=256;
constexpr uint32_t g_wire_qk_rope=64;
constexpr uint32_t g_wire_qk_nope=128;
constexpr uint32_t g_wire_v=128;
constexpr float g_wire_rope_base=10000.0f;
constexpr float g_wire_eps=1e-5f;
constexpr uint32_t g_wire_batch=1;
constexpr uint32_t g_wire_seq=80;     // multi-block (> BC=64)
constexpr uint32_t g_wire_cache_max=256;
constexpr float g_wire_input_scale=0.02f;
constexpr float g_wire_input_offset=-0.4f;
// TF32 fused attention vs fp32 explicit attention, propagated through the layer.
constexpr float g_wire_tol=3.0e-2f;

class CAIF_FlashMlaWiringTest
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<uint32_t> InputShape();
    static CAIF_DeviceMLAttentionConfig MakeConfig(const bool causal);
    static std::vector<float> HostInput();
    static bool Compare(const CAIF_HostTensor &fused,const CAIF_HostTensor &expl,float &worst);
    static void RunCase(const char *name,const bool causal);
    static void RunCachedCase(const char *name,const bool causal);
};

CAIF_DeviceMLAttentionConfig CAIF_FlashMlaWiringTest::MakeConfig(const bool causal)
{
  return CAIF_DeviceMLAttentionConfig(g_wire_dim,
                                      g_wire_heads,
                                      g_wire_q_lora,
                                      g_wire_kv_lora,
                                      g_wire_qk_rope,
                                      g_wire_qk_nope,
                                      g_wire_v,
                                      causal,
                                      g_wire_rope_base,
                                      g_wire_eps);
}

std::vector<float> CAIF_FlashMlaWiringTest::HostInput()
{
  std::vector<float> host(g_wire_batch*g_wire_seq*g_wire_dim);
  for(size_t i=0;i<host.size();++i)
  {
    host[i]=static_cast<float>(i)*g_wire_input_scale+g_wire_input_offset;
  }
  return host;
}

bool CAIF_FlashMlaWiringTest::Compare(const CAIF_HostTensor &fused,
                                      const CAIF_HostTensor &expl,
                                      float &worst)
{
  worst=0.0f;
  bool ok=(fused.TotalElements()==expl.TotalElements());
  for(size_t i=0;i<fused.TotalElements() && ok==true;++i)
  {
    const float a=fused.Data()[i];
    const float b=expl.Data()[i];
    if(std::isfinite(a)==false)
    {
      ok=false;
    }
    const float diff=std::fabs(a-b);
    if(diff>g_wire_tol)
    {
      ok=false;
    }
    if(diff>worst)
    {
      worst=diff;
    }
  }
  return ok;
}

std::vector<uint32_t> CAIF_FlashMlaWiringTest::InputShape()
{
  return std::vector<uint32_t>{g_wire_batch,g_wire_seq,g_wire_dim};
}

void CAIF_FlashMlaWiringTest::RunCase(const char *name,const bool causal)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceMLAttentionConfig config=MakeConfig(causal);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);
    const std::vector<float> host=HostInput();
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host.data(),InputShape(),stream);

    CAIF_Settings::SetFlashMlaPrefill(true);
    CAIF_HostTensor host_fused=mla.Forward(input,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(false);
    CAIF_HostTensor host_explicit=mla.Forward(input,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(true);

    float worst=0.0f;
    const bool ok=Compare(host_fused,host_explicit,worst);
    if(ok==false)
    {
      ISE_Out::Out()<<"  "
                    <<name
                    <<": worst |fused-explicit|="
                    <<worst
                    <<" (tol "
                    <<g_wire_tol
                    <<")\n";
    }
    CAIF_TestHarness::Report(name,ok);
  }
  CAIF_TEST_CATCH_BLOCK(name)
}

void CAIF_FlashMlaWiringTest::RunCachedCase(const char *name,const bool causal)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceMLAttentionConfig config=MakeConfig(causal);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);
    mla.EnableKVCache(g_wire_batch,g_wire_cache_max);
    const std::vector<float> host=HostInput();
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host.data(),InputShape(),stream);

    // Whole-prompt prefill into an empty cache down both paths; ResetKVCache
    // between runs so each sees cache_len=0, new_len=g_wire_seq.
    CAIF_Settings::SetFlashMlaPrefill(true);
    mla.ResetKVCache();
    CAIF_HostTensor host_fused=mla.ForwardCached(input,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(false);
    mla.ResetKVCache();
    CAIF_HostTensor host_explicit=mla.ForwardCached(input,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(true);

    float worst=0.0f;
    const bool ok=Compare(host_fused,host_explicit,worst);
    if(ok==false)
    {
      ISE_Out::Out()<<"  "
                    <<name
                    <<": worst |fused-explicit|="
                    <<worst
                    <<" (tol "
                    <<g_wire_tol
                    <<")\n";
    }
    CAIF_TestHarness::Report(name,ok);
  }
  CAIF_TEST_CATCH_BLOCK(name)
}

void CAIF_FlashMlaWiringTest::RunAll()
{
  ISE_Out::Out()<<"=== Flash-MLA wiring: fused vs explicit (192,128) ==="
                <<"\n\n";
  RunCase("FlashMLA::Wiring::ForwardImpl::Causal",true);
  RunCase("FlashMLA::Wiring::ForwardImpl::NonCausal",false);
  RunCachedCase("FlashMLA::Wiring::ForwardCached::Causal",true);
  RunCachedCase("FlashMLA::Wiring::ForwardCached::NonCausal",false);
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FlashMlaWiringTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
