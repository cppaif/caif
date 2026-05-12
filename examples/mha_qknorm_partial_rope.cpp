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

//--------------------------------------------------------------------------
// Example: CAIF_DeviceMultiHeadAttention with QK-norm + partial-rotary RoPE
//
// Demonstrates two attention features added in the re-arch release:
//
//   1. QK-norm. When q_norm_gamma / k_norm_gamma are loaded via
//      LoadQNormGamma / LoadKNormGamma, MHA applies RMSNorm to Q and K
//      after the projection (+ bias) and before RoPE / GQA-expand.
//      Used by OLMoE, Olmo2, Qwen3. Empty gammas = legacy no-norm path.
//
//   2. Partial-rotary RoPE. When AttentionConfig_t::rope_dim is set to
//      a value less than head_dim, only the first rope_dim dims of
//      each head are rotated; the trailing (head_dim - rope_dim) dims
//      pass through untouched. This matches HF's
//      `partial_rotary_factor < 1.0` (Glm4Moe-style models). Setting
//      rope_dim = 0 (the default) means "rotate the full head_dim"
//      (the legacy behavior).
//
// This example builds a standalone MHA layer, configures both
// features, runs one forward pass with synthetic input, and prints
// the output's shape and total parameter count. No training loop —
// the goal is to show the public surface, not to optimize anything.
//--------------------------------------------------------------------------

#include "caif_device_multi_head_attention.h"
#include "caif_device_context.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_run_context_scope.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <memory>

using namespace instance;

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF MHA QK-norm + partial-rotary RoPE example ==="<<std::endl;

    // Pick attention dtype. Same template grid as everywhere else:
    // 3 compute dtypes (float / __half / __nv_bfloat16) crossed with
    // 3 storage dtypes (same set). <float, float> keeps the example
    // simple; bf16 storage + fp32 compute is the production default
    // for modern HF MoE checkpoints.
    typedef CAIF_DeviceMultiHeadAttention<float,float> MHA_t;

    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    // Attention shape — small enough to run quickly.
    const uint32_t batch_size=2;
    const uint32_t seq_len=16;
    const uint32_t dim=128;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=4;
    const uint32_t head_dim=dim/num_heads;
    const float rope_base=10000.0f;
    const float dropout_rate=0.0f;

    // Partial-rotary RoPE. Glm4Moe-style: rotate only the first half
    // of each head's dim. Setting rope_dim==head_dim has the same
    // effect as rope_dim==0 (full rotation, the default).
    const int rope_dim_partial=static_cast<int>(head_dim)/2;

    MHA_t::AttentionConfig_t cfg;
    cfg.dim=dim;
    cfg.num_heads=num_heads;
    cfg.num_kv_heads=num_kv_heads;
    cfg.head_dim=head_dim;
    cfg.causal=true;
    cfg.use_rope=true;
    cfg.rope_base=rope_base;
    cfg.rope_style=0;
    cfg.rope_dim=rope_dim_partial;
    cfg.dropout_rate=dropout_rate;
    cfg.qk_norm_eps=1.0e-5f;

    MHA_t mha(cfg,stream);
    mha.InitializeWeights();

    ISE_Out::Out()<<"MHA: dim="<<dim
                  <<" heads="<<num_heads
                  <<" head_dim="<<head_dim
                  <<" rope_dim="<<rope_dim_partial
                  <<" (full="<<head_dim<<")"
                  <<std::endl;
    ISE_Out::Out()<<"Trainable params: "<<mha.TotalParameterCount()<<std::endl;

    // Build the QK-norm gammas. Each is a 1-D fp32 tensor sized to the
    // full projection output width: num_heads * head_dim for q,
    // num_kv_heads * head_dim for k. Initialised to 1.0 (the
    // identity — RMSNorm with all-ones gamma is just the
    // unit-variance rescale). In a real loader these would come from
    // the checkpoint's `q_norm.weight` / `k_norm.weight` tensors.
    const uint32_t q_norm_size=num_heads*head_dim;
    const uint32_t k_norm_size=num_kv_heads*head_dim;
    std::vector<float> q_gamma(q_norm_size,1.0f);
    std::vector<float> k_gamma(k_norm_size,1.0f);

    CAIF_DeviceTensor q_norm_gamma=CAIF_DeviceTensor::FromHostData(q_gamma.data(),
                                                                   {q_norm_size},
                                                                   stream);
    CAIF_DeviceTensor k_norm_gamma=CAIF_DeviceTensor::FromHostData(k_gamma.data(),
                                                                   {k_norm_size},
                                                                   stream);
    mha.LoadQNormGamma(std::move(q_norm_gamma));
    mha.LoadKNormGamma(std::move(k_norm_gamma));

    ISE_Out::Out()<<"QK-norm enabled: HasQNormGamma="<<mha.HasQNormGamma()
                  <<" HasKNormGamma="<<mha.HasKNormGamma()
                  <<std::endl;

    // Synthetic forward input — random-ish activations [B, S, dim].
    const uint32_t total_elements=batch_size*seq_len*dim;
    std::vector<float> input_host(total_elements);
    for(uint32_t i=0;i<total_elements;++i)
    {
      input_host[i]=static_cast<float>(i%32)/32.0f-0.5f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_host.data(),
                                                            {batch_size,seq_len,dim},
                                                            stream);

    // RunContext carries per-pass state — stream, pass direction,
    // training flag. Constructed default, configured via setters.
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    CAIF_RunContextPassScope forward_scope(ctx,CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceTensor output=mha.ForwardImpl(input,ctx);
    stream.Synchronize();

    const std::vector<uint32_t> &out_shape=output.Shape();
    ISE_Out::Out()<<"Forward output shape: [";
    for(size_t i=0;i<out_shape.size();++i)
    {
      if(i>0)
      {
        ISE_Out::Out()<<", ";
      }
      ISE_Out::Out()<<out_shape[i];
    }
    ISE_Out::Out()<<"]"<<std::endl;

    ISE_Out::Out()<<"=== Done ==="<<std::endl;
    return 0;
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception: "<<e<<std::endl;
    return 1;
  }
}
