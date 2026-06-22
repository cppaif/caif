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
// Gate regression: the fused flash-prefill kernel must
// fire ONLY for true prefill (new_len>1). Single-token decode — batched
// (batch>1, new_len==1), single (batch==1, new_len==1), and matrix-absorbed —
// must keep its existing explicit path bit-for-bit.
//
// Probe: CAIF_Settings::FlashMlaPrefill only gates the prefill branch, so the
// fused (TF32) and explicit (fp32) prefills DIFFER. Run each path twice, once
// with the toggle on and once off:
//   - prefill (positive control) must DIFFER across the toggle — proves the
//     toggle flips a real path, so the decode checks below are not vacuous;
//   - every decode shape must be BIT-IDENTICAL across the toggle — if decode
//     were wrongly routed to the fused kernel, the on-run would diverge from
//     the off-run. The cache is warmed explicitly (toggle off) the same way in
//     both runs so it is byte-identical and only the decode step varies.
//
// All shapes are fused-ELIGIBLE (QKHeadDim=192, v=128), so the only thing
// keeping decode off the fused path is the gate logic this test pins down.
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
constexpr uint32_t g_gate_dim=256;
constexpr uint32_t g_gate_heads=2;
constexpr uint32_t g_gate_q_lora=256;
constexpr uint32_t g_gate_kv_lora=256;
constexpr uint32_t g_gate_qk_rope=64;
constexpr uint32_t g_gate_qk_nope=128;
constexpr uint32_t g_gate_v=128;
constexpr float g_gate_rope_base=10000.0f;
constexpr float g_gate_eps=1e-5f;
constexpr uint32_t g_gate_cache_max=256;
// Positive-control prefill length: > BC=64 so the fused path is multi-block.
constexpr uint32_t g_gate_prefill_len=80;
// Tokens used to warm the cache before a decode step (new_len>1 prefill).
constexpr uint32_t g_gate_warm_len=4;
// Absorbed decode needs cache_len>=threshold; 1 triggers it after the warm.
constexpr uint32_t g_gate_absorb_threshold=1;
// Deterministic input fills (distinct streams for the prefill and the decode
// token so the decode does not trivially self-attend).
constexpr float g_gate_prefill_scale=0.02f;
constexpr float g_gate_prefill_offset=-0.4f;
constexpr float g_gate_decode_scale=0.013f;
constexpr float g_gate_decode_offset=0.21f;
// Fused TF32 vs explicit fp32: the prefill positive control must differ by more
// than this (path really flipped) and stay under the TF32-class tolerance.
constexpr float g_gate_toggle_min_diff=1.0e-5f;
constexpr float g_gate_tf32_tol=3.0e-2f;
// Decode held off the fused path still varies run-to-run by ~1e-6 (cuBLAS
// reduction-order non-determinism in the batched scored path), so a bit-exact
// match is unattainable. Instead bound the toggle's effect by the inherent
// off-vs-off noise: flipping the fused toggle must move decode no more than GPU
// noise does — orders below the ~1e-3 gap the fused TF32 kernel would inject.
constexpr float g_gate_noise_factor=3.0f;
constexpr float g_gate_noise_slack=2.0e-6f;

class CAIF_FlashMlaGateTest
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_DeviceMLAttentionConfig MakeConfig(const uint32_t q_lora_rank);
    static std::vector<float> HostData(const uint32_t batch,
                                       const uint32_t seq,
                                       const float scale,
                                       const float offset);
    static float WorstDiff(const CAIF_HostTensor &a,
                           const CAIF_HostTensor &b,
                           bool &all_finite);
    static void RunForwardPrefillControl(const char *name,
                                         const CAIF_DeviceMLAttentionConfig &config);
    static void RunCachedPrefillControl(const char *name,
                                        const CAIF_DeviceMLAttentionConfig &config);
    static void RunDecodeGate(const char *name,
                              const CAIF_DeviceMLAttentionConfig &config,
                              const uint32_t batch);
    static CAIF_HostTensor WarmAndDecode(CAIF_DeviceMLAttention<float,float> &mla,
                                         CAIF_RunContext &ctx,
                                         CAIF_CudaStream &stream,
                                         const std::vector<float> &warm,
                                         const std::vector<uint32_t> &warm_shape,
                                         const std::vector<float> &tok,
                                         const std::vector<uint32_t> &tok_shape,
                                         const bool fused_toggle);
};

CAIF_DeviceMLAttentionConfig CAIF_FlashMlaGateTest::MakeConfig(const uint32_t q_lora_rank)
{
  return CAIF_DeviceMLAttentionConfig(g_gate_dim,
                                      g_gate_heads,
                                      q_lora_rank,
                                      g_gate_kv_lora,
                                      g_gate_qk_rope,
                                      g_gate_qk_nope,
                                      g_gate_v,
                                      true,
                                      g_gate_rope_base,
                                      g_gate_eps);
}

std::vector<float> CAIF_FlashMlaGateTest::HostData(const uint32_t batch,
                                                   const uint32_t seq,
                                                   const float scale,
                                                   const float offset)
{
  std::vector<float> host(static_cast<size_t>(batch)*seq*g_gate_dim);
  for(size_t i=0;i<host.size();++i)
  {
    host[i]=static_cast<float>(i)*scale+offset;
  }
  return host;
}

float CAIF_FlashMlaGateTest::WorstDiff(const CAIF_HostTensor &a,
                                       const CAIF_HostTensor &b,
                                       bool &all_finite)
{
  all_finite=true;
  if(a.TotalElements()!=b.TotalElements())
  {
    all_finite=false;
    return 0.0f;
  }
  float worst=0.0f;
  for(size_t i=0;i<a.TotalElements();++i)
  {
    const float x=a.Data()[i];
    const float y=b.Data()[i];
    if(std::isfinite(x)==false || std::isfinite(y)==false)
    {
      all_finite=false;
    }
    const float diff=std::fabs(x-y);
    if(diff>worst)
    {
      worst=diff;
    }
  }
  return worst;
}

void CAIF_FlashMlaGateTest::RunForwardPrefillControl(const char *name,
                                                     const CAIF_DeviceMLAttentionConfig &config)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceMLAttention<float,float> mla(config,stream);
    const std::vector<float> host=HostData(1u,g_gate_prefill_len,g_gate_prefill_scale,g_gate_prefill_offset);
    const std::vector<uint32_t> shape={1u,g_gate_prefill_len,g_gate_dim};

    CAIF_Settings::SetFlashMlaPrefill(true);
    CAIF_DeviceTensor in_fused=CAIF_DeviceTensor::FromHostData(host.data(),shape,stream);
    CAIF_HostTensor host_fused=mla.Forward(in_fused,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(false);
    CAIF_DeviceTensor in_explicit=CAIF_DeviceTensor::FromHostData(host.data(),shape,stream);
    CAIF_HostTensor host_explicit=mla.Forward(in_explicit,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(true);

    bool all_finite=true;
    const float worst=WorstDiff(host_fused,host_explicit,all_finite);
    const bool ok=(all_finite==true && worst>g_gate_toggle_min_diff && worst<g_gate_tf32_tol);
    if(ok==false)
    {
      ISE_Out::Out()<<"  "
                    <<name
                    <<": expected fused!=explicit within tol, got worst |fused-explicit|="
                    <<worst
                    <<" (min "
                    <<g_gate_toggle_min_diff
                    <<", tol "
                    <<g_gate_tf32_tol
                    <<")\n";
    }
    CAIF_TestHarness::Report(name,ok);
  }
  CAIF_TEST_CATCH_BLOCK(name)
}

void CAIF_FlashMlaGateTest::RunCachedPrefillControl(const char *name,
                                                    const CAIF_DeviceMLAttentionConfig &config)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceMLAttention<float,float> mla(config,stream);
    mla.EnableKVCache(1u,g_gate_cache_max);
    const std::vector<float> host=HostData(1u,g_gate_prefill_len,g_gate_prefill_scale,g_gate_prefill_offset);
    const std::vector<uint32_t> shape={1u,g_gate_prefill_len,g_gate_dim};

    // Whole-prompt prefill into an empty cache down both paths; ResetKVCache
    // between runs so each sees cache_len=0, new_len=g_gate_prefill_len.
    CAIF_Settings::SetFlashMlaPrefill(true);
    mla.ResetKVCache();
    CAIF_DeviceTensor in_fused=CAIF_DeviceTensor::FromHostData(host.data(),shape,stream);
    CAIF_HostTensor host_fused=mla.ForwardCached(in_fused,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(false);
    mla.ResetKVCache();
    CAIF_DeviceTensor in_explicit=CAIF_DeviceTensor::FromHostData(host.data(),shape,stream);
    CAIF_HostTensor host_explicit=mla.ForwardCached(in_explicit,ctx).ToHost();
    CAIF_Settings::SetFlashMlaPrefill(true);

    bool all_finite=true;
    const float worst=WorstDiff(host_fused,host_explicit,all_finite);
    const bool ok=(all_finite==true && worst>g_gate_toggle_min_diff && worst<g_gate_tf32_tol);
    if(ok==false)
    {
      ISE_Out::Out()<<"  "
                    <<name
                    <<": expected fused!=explicit within tol, got worst |fused-explicit|="
                    <<worst
                    <<" (min "
                    <<g_gate_toggle_min_diff
                    <<", tol "
                    <<g_gate_tf32_tol
                    <<")\n";
    }
    CAIF_TestHarness::Report(name,ok);
  }
  CAIF_TEST_CATCH_BLOCK(name)
}

CAIF_HostTensor CAIF_FlashMlaGateTest::WarmAndDecode(CAIF_DeviceMLAttention<float,float> &mla,
                                                     CAIF_RunContext &ctx,
                                                     CAIF_CudaStream &stream,
                                                     const std::vector<float> &warm,
                                                     const std::vector<uint32_t> &warm_shape,
                                                     const std::vector<float> &tok,
                                                     const std::vector<uint32_t> &tok_shape,
                                                     const bool fused_toggle)
{
  // Warm the cache explicitly (toggle off) so it is identical regardless of the
  // decode toggle, then decode the single token under the requested toggle.
  CAIF_Settings::SetFlashMlaPrefill(false);
  mla.ResetKVCache();
  CAIF_DeviceTensor warm_in=CAIF_DeviceTensor::FromHostData(warm.data(),warm_shape,stream);
  mla.ForwardCached(warm_in,ctx);
  CAIF_Settings::SetFlashMlaPrefill(fused_toggle);
  CAIF_DeviceTensor tok_in=CAIF_DeviceTensor::FromHostData(tok.data(),tok_shape,stream);
  return mla.ForwardCached(tok_in,ctx).ToHost();
}

void CAIF_FlashMlaGateTest::RunDecodeGate(const char *name,
                                          const CAIF_DeviceMLAttentionConfig &config,
                                          const uint32_t batch)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceMLAttention<float,float> mla(config,stream);
    mla.EnableKVCache(batch,g_gate_cache_max);

    const std::vector<float> warm=HostData(batch,g_gate_warm_len,g_gate_prefill_scale,g_gate_prefill_offset);
    const std::vector<float> tok=HostData(batch,1u,g_gate_decode_scale,g_gate_decode_offset);
    const std::vector<uint32_t> warm_shape={batch,g_gate_warm_len,g_gate_dim};
    const std::vector<uint32_t> tok_shape={batch,1u,g_gate_dim};

    // Two decodes with the toggle held off establish the run-to-run noise
    // floor; a third with the toggle on measures its effect. The decode path
    // (new_len==1) never reads the toggle, so the on-vs-off move must not exceed
    // the off-vs-off noise. A regression that routes decode through the fused
    // TF32 kernel would move it ~1e-3 — far past the noise.
    const CAIF_HostTensor off_a=WarmAndDecode(mla,ctx,stream,warm,warm_shape,tok,tok_shape,false);
    const CAIF_HostTensor off_b=WarmAndDecode(mla,ctx,stream,warm,warm_shape,tok,tok_shape,false);
    const CAIF_HostTensor on=WarmAndDecode(mla,ctx,stream,warm,warm_shape,tok,tok_shape,true);
    CAIF_Settings::SetFlashMlaPrefill(true);

    bool noise_finite=true;
    bool toggle_finite=true;
    const float baseline=WorstDiff(off_a,off_b,noise_finite);
    const float toggled=WorstDiff(off_a,on,toggle_finite);
    const float bound=baseline*g_gate_noise_factor+g_gate_noise_slack;
    const bool ok=(noise_finite==true && toggle_finite==true && toggled<=bound);
    if(ok==false)
    {
      ISE_Out::Out()<<"  "
                    <<name
                    <<": toggle moved decode by "
                    <<toggled
                    <<" > noise bound "
                    <<bound
                    <<" (off-vs-off "
                    <<baseline
                    <<") — decode may be wrongly routed to the fused kernel\n";
    }
    CAIF_TestHarness::Report(name,ok);
  }
  CAIF_TEST_CATCH_BLOCK(name)
}

void CAIF_FlashMlaGateTest::RunAll()
{
  ISE_Out::Out()<<"=== Flash-MLA decode gate: only prefill is fused ==="
                <<"\n\n";

  CAIF_DeviceMLAttentionConfig cfg_qlora=MakeConfig(g_gate_q_lora);

  // Positive controls — the toggle must visibly flip the prefill path.
  RunForwardPrefillControl("FlashMLA::Gate::ForwardPrefillTogglesPath",cfg_qlora);
  RunCachedPrefillControl("FlashMLA::Gate::CachedPrefillTogglesPath",cfg_qlora);

  // Decode must be untouched by the toggle (never routed to the fused kernel).
  RunDecodeGate("FlashMLA::Gate::SingleDecodeNotFused",cfg_qlora,1u);
  RunDecodeGate("FlashMLA::Gate::BatchedDecodeNotFused",cfg_qlora,2u);

  // q_lora_rank==0 + a threshold of 1 routes the single-token decode through
  // the matrix-absorbed path; it too must ignore the fused toggle.
  CAIF_DeviceMLAttentionConfig cfg_absorb=MakeConfig(0u);
  cfg_absorb.SetDecodeAbsorbThreshold(g_gate_absorb_threshold);
  RunDecodeGate("FlashMLA::Gate::AbsorbedDecodeNotFused",cfg_absorb,1u);
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FlashMlaGateTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
