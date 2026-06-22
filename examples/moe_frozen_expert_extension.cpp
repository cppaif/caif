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
// Example: MoE layer-surgery via CAIF_DeviceMoEFrozenExpert
//
// The "add-MoE" pattern: start from a pretrained dense FFN, wrap its
// gate / up / down projection weights as frozen experts, then attach
// new trainable experts alongside in a fresh MoE layer. The router
// learns to mix the frozen pretrained behaviour with the new
// trainable capacity.
//
// This example builds the frozen-expert half of that pattern. Three
// CAIF_DeviceFrozenLinear layers (gate, up, down) hold the
// pretrained-style weights at fp32 storage. They are wrapped in a
// CAIF_DeviceMoEFrozenExpert that composes them via the standard
// gated-FFN dataflow:
//
//     hidden = silu(gate(x)) * up(x)
//     out    = down(hidden)
//
// The wrapper exposes zero trainable parameters — gradients do not
// flow through it. To grow MoE capacity in a real fine-tune, append
// `CAIF_DeviceMoEExpert` (trainable) instances to the layer's expert
// list and widen the router accordingly.
//
// For brevity this example uses fp32 storage on every FrozenLinear.
// In production the pretrained base is usually stored at int8 or
// 4-bit packed int4 (using the templated CAIF_DeviceFrozenLinear
// instantiations), and only the new trainable experts run in
// fp16 / bf16. That mixed-dtype shape is exactly what the
// CAIF_DeviceMoEExpertBase polymorphic base supports.
//--------------------------------------------------------------------------

#include "caif_device_moe_frozen_expert.h"
#include "caif_device_frozen_linear.h"
#include "caif_device_context.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_run_context_pass_scope.h"
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
    ISE_Out::Out()<<"=== CAIF MoEFrozenExpert (MoE-extension) example ==="<<std::endl;

    typedef CAIF_DeviceFrozenLinear<float,float> FrozenLinear_t;
    typedef CAIF_DeviceMoEFrozenExpert<float,float> FrozenExpert_t;

    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    // Expert shape. input_dim = the model's hidden width; hidden_dim
    // = the FFN's inner expansion (typically 2-4x input_dim).
    const uint32_t input_dim=64;
    const uint32_t hidden_dim=256;
    const uint32_t num_tokens=8;

    // Helper: build one FrozenLinear and load a fp32 weight tensor of
    // shape [in, out]. In a real pipeline the floats come from the
    // pretrained checkpoint via safetensors; here we synthesise a
    // simple sinusoidal pattern so the kernel has something to chew.
    const uint32_t group_size=128;

    std::unique_ptr<FrozenLinear_t> gate_layer=std::make_unique<FrozenLinear_t>(input_dim,
                                                                                hidden_dim,
                                                                                stream,
                                                                                group_size,
                                                                                true);
    std::unique_ptr<FrozenLinear_t> up_layer=std::make_unique<FrozenLinear_t>(input_dim,
                                                                              hidden_dim,
                                                                              stream,
                                                                              group_size,
                                                                              true);
    std::unique_ptr<FrozenLinear_t> down_layer=std::make_unique<FrozenLinear_t>(hidden_dim,
                                                                                input_dim,
                                                                                stream,
                                                                                group_size,
                                                                                true);

    // Synthetic pretrained weights.
    const uint32_t gate_elems=input_dim*hidden_dim;
    const uint32_t up_elems=input_dim*hidden_dim;
    const uint32_t down_elems=hidden_dim*input_dim;

    std::vector<float> gate_host(gate_elems);
    std::vector<float> up_host(up_elems);
    std::vector<float> down_host(down_elems);

    const float weight_scale=0.02f;
    for(uint32_t i=0;i<gate_elems;++i)
    {
      gate_host[i]=weight_scale*static_cast<float>((i*3+1)%32-16);
    }
    for(uint32_t i=0;i<up_elems;++i)
    {
      up_host[i]=weight_scale*static_cast<float>((i*5+2)%32-16);
    }
    for(uint32_t i=0;i<down_elems;++i)
    {
      down_host[i]=weight_scale*static_cast<float>((i*7+3)%32-16);
    }

    CAIF_DeviceTensor gate_w=CAIF_DeviceTensor::FromHostData(gate_host.data(),
                                                             {input_dim,hidden_dim},
                                                             stream);
    CAIF_DeviceTensor up_w=CAIF_DeviceTensor::FromHostData(up_host.data(),
                                                           {input_dim,hidden_dim},
                                                           stream);
    CAIF_DeviceTensor down_w=CAIF_DeviceTensor::FromHostData(down_host.data(),
                                                             {hidden_dim,input_dim},
                                                             stream);
    gate_layer->LoadFromTensor(std::move(gate_w));
    up_layer->LoadFromTensor(std::move(up_w));
    down_layer->LoadFromTensor(std::move(down_w));

    // Wrap the three frozen projections into a single MoE-style
    // expert. FrozenSubLayers_t takes ownership.
    FrozenExpert_t::FrozenSubLayers_t sub_layers;
    sub_layers.gate=std::move(gate_layer);
    sub_layers.up=std::move(up_layer);
    sub_layers.down=std::move(down_layer);

    CAIF_DeviceMoEFrozenExpertConfig expert_cfg(input_dim,hidden_dim,true);

    FrozenExpert_t expert(expert_cfg,std::move(sub_layers),stream);

    ISE_Out::Out()<<"FrozenExpert: input_dim="<<expert.InputDim()
                  <<" hidden_dim="<<expert.HiddenDim()
                  <<" use_gated="<<expert.UseGated()
                  <<std::endl;
    ISE_Out::Out()<<"Trainable params (must be 0 — expert is frozen): "
                  <<expert.TotalParameterCount()
                  <<std::endl;

    // Synthetic forward input — [N, input_dim].
    const uint32_t total_elements=num_tokens*input_dim;
    std::vector<float> input_host(total_elements);
    for(uint32_t i=0;i<total_elements;++i)
    {
      input_host[i]=static_cast<float>(i%19)/19.0f-0.5f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_host.data(),
                                                            {num_tokens,input_dim},
                                                            stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    CAIF_RunContextPassScope forward_scope(ctx,CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceTensor output=expert.ForwardImpl(input,ctx);
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
