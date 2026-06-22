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
// The fused MLA flash prefill is
// O(seq) memory. A 32-head DSv2-Lite-shaped layer encodes a 16384-token prompt
// with the fused kernel using a few GB, while the explicit O(seq^2) path's
// [bh, seq, seq] score buffer would need ~34 GB and OOMs the 32 GB dev card.
// Self-guarding: skips if the device lacks the fused working set, and only
// asserts the explicit OOM when the score buffer actually exceeds free memory,
// so the test is portable across cards.
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

#ifdef USE_CAIF_CUDA
#include <cuda_runtime.h>
#endif

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_16k_dim=4096;
constexpr uint32_t g_16k_heads=32;
constexpr uint32_t g_16k_q_lora=512;
constexpr uint32_t g_16k_kv_lora=512;
constexpr uint32_t g_16k_qk_rope=64;
constexpr uint32_t g_16k_qk_nope=128;
constexpr uint32_t g_16k_v=128;
constexpr float g_16k_rope_base=10000.0f;
constexpr float g_16k_eps=1e-5f;
constexpr uint32_t g_16k_batch=1;
constexpr uint32_t g_16k_seq=16384;
constexpr float g_16k_input_scale=0.001f;
constexpr float g_16k_input_offset=-0.05f;
constexpr uint32_t g_16k_input_mod=1000;
constexpr size_t g_16k_bytes_per_gb=1073741824ull;
// Minimum free memory to attempt the fused 16K working set.
constexpr size_t g_16k_min_free_gb=10;

class CAIF_FlashMla16KTest
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestNoOom();
};

void CAIF_FlashMla16KTest::TestNoOom()
{
  try
  {
    size_t free_bytes=0;
    size_t total_bytes=0;
    cudaMemGetInfo(&free_bytes,&total_bytes);

    const size_t bh=static_cast<size_t>(g_16k_batch)*g_16k_heads;
    const size_t scores_bytes=bh*g_16k_seq*g_16k_seq*sizeof(float);
    const size_t out_elems=static_cast<size_t>(g_16k_batch)*g_16k_seq*g_16k_dim;

    ISE_Out::Out()<<"  device free "
                  <<(free_bytes/g_16k_bytes_per_gb)
                  <<" GB; explicit score buffer needs "
                  <<(scores_bytes/g_16k_bytes_per_gb)
                  <<" GB (bh="
                  <<bh
                  <<", seq="
                  <<g_16k_seq
                  <<")\n";

    if(free_bytes<g_16k_min_free_gb*g_16k_bytes_per_gb)
    {
      ISE_Out::Out()<<"  skipped: < "
                    <<g_16k_min_free_gb
                    <<" GB free\n";
      CAIF_TestHarness::Report("FlashMLA::16K::NoOOM",true);
      return;
    }

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceMLAttentionConfig config(g_16k_dim,
                                        g_16k_heads,
                                        g_16k_q_lora,
                                        g_16k_kv_lora,
                                        g_16k_qk_rope,
                                        g_16k_qk_nope,
                                        g_16k_v,
                                        true,
                                        g_16k_rope_base,
                                        g_16k_eps);
    CAIF_DeviceMLAttention<float,float> mla(config,stream);

    std::vector<float> host(out_elems);
    for(size_t i=0;i<host.size();++i)
    {
      host[i]=static_cast<float>(i%g_16k_input_mod)*g_16k_input_scale+g_16k_input_offset;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host.data(),
                                                            {g_16k_batch,g_16k_seq,g_16k_dim},
                                                            stream);

    // FUSED: must encode the 16K prompt without OOM (O(seq) memory).
    CAIF_Settings::SetFlashMlaPrefill(true);
    CAIF_HostTensor out=mla.Forward(input,ctx).ToHost();
    bool fused_ok=(out.TotalElements()==out_elems);
    for(size_t i=0;i<out.TotalElements() && fused_ok==true;++i)
    {
      if(std::isfinite(out.Data()[i])==false)
      {
        fused_ok=false;
      }
    }

    size_t free_after=0;
    cudaMemGetInfo(&free_after,&total_bytes);
    ISE_Out::Out()<<"  FUSED 16K prefill OK; used ~"
                  <<((free_bytes-free_after)/g_16k_bytes_per_gb)
                  <<" GB\n";

    // EXPLICIT: only exercised where the score buffer exceeds free memory.
    bool explicit_oomed=false;
    bool oom_expected=(scores_bytes>free_bytes);
    if(oom_expected==true)
    {
      CAIF_Settings::SetFlashMlaPrefill(false);
      try
      {
        CAIF_HostTensor ex=mla.Forward(input,ctx).ToHost();
        if(ex.TotalElements()!=out_elems)
        {
          explicit_oomed=false;
        }
      }
      catch(CAIF_Exception &oom)
      {
        explicit_oomed=true;
        ISE_Out::Out()<<"  EXPLICIT 16K OOMed as expected (score buffer > free)\n";
      }
      CAIF_Settings::SetFlashMlaPrefill(true);
    }
    else
    {
      ISE_Out::Out()<<"  (card holds the explicit score buffer; OOM not exercised)\n";
    }

    bool ok=fused_ok;
    if(oom_expected==true && explicit_oomed==false)
    {
      ok=false;
    }
    CAIF_TestHarness::Report("FlashMLA::16K::NoOOM",ok);
  }
  CAIF_TEST_CATCH_BLOCK("FlashMLA::16K::NoOOM")
}

void CAIF_FlashMla16KTest::RunAll()
{
  ISE_Out::Out()<<"=== Flash-MLA 16K no-OOM ==="
                <<"\n\n";
  TestNoOom();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FlashMla16KTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
