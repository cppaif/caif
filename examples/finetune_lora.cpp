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
// Example: Fine-tune with LoRA adapters
//
// Demonstrates:
//   - Wrapping a base dense layer with CAIF_DeviceLoRAAdapter
//   - Freezing surrounding layers — only the LoRA A/B matrices train
//   - Forward / cross-entropy loss / backward / Adam step
//   - Saving the model with LoRA weights via SafeTensors
//
// All layers are instantiated at <float, float> here for simplicity.
// In production you'd typically pin the frozen base to bf16 / fp16 /
// int8 / int4 storage via CAIF_DeviceFrozenLinear, then layer LoRA
// in <float, float> on top.
//--------------------------------------------------------------------------

#include "caif_device_network.h"
#include "caif_device_dense_layer.h"
#include "caif_device_lora_adapter.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_linear_head.h"
#include "caif_device_cross_entropy_loss.h"
#include "caif_device_context.h"
#include "caif_cuda_stream.h"
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
    ISE_Out::Out()<<"=== CAIF LoRA Fine-Tuning Example ==="<<std::endl;

    // Pick the layer dtype here. To change it, edit BOTH template args
    // on each typedef below. See CHANGES.md / DESIGN.md for the full
    // 3 compute x 3 storage matrix supported on trainable layers.
    typedef CAIF_DeviceDenseLayer<float,float> DenseLayer_t;
    typedef CAIF_DeviceLoRAAdapter<float,float> LoRAAdapter_t;
    typedef CAIF_DeviceRMSNorm<float,float> RMSNorm_t;
    typedef CAIF_DeviceLinearHead<float,float> LinearHead_t;
    typedef CAIF_DeviceCrossEntropyLoss<float,float> CrossEntropy_t;

    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    // Small classification network for demonstration.
    const uint32_t num_classes=64;
    const uint32_t dim=128;
    const uint32_t lora_rank=8;
    const float lora_alpha=16.0f;
    const uint32_t lora_seed=42;

    CAIF_DeviceNetwork network(stream);

    // Layer 1: base dense layer wrapped with a LoRA adapter.
    // The dense layer's weights stay frozen; only the LoRA A and B
    // matrices (rank * 2 * dim parameters) train.
    LoRAAdapter_t::LoRAConfig_t lora_cfg;
    lora_cfg.rank=lora_rank;
    lora_cfg.alpha=lora_alpha;
    lora_cfg.input_dim=dim;
    lora_cfg.output_dim=dim;

    std::unique_ptr<DenseLayer_t> base_dense=std::make_unique<DenseLayer_t>(
                                                              dim,
                                                              dim,
                                                              CAIF_DeviceActivation_e::None,
                                                              stream,
                                                              true);
    std::unique_ptr<LoRAAdapter_t> lora=std::make_unique<LoRAAdapter_t>(
                                                              lora_cfg,
                                                              std::move(base_dense),
                                                              stream,
                                                              lora_seed);
    network.AddLayer(std::move(lora));

    network.AddLayer(std::make_unique<RMSNorm_t>(dim,stream));

    LinearHead_t::Config_t head_cfg;
    head_cfg.input_dim=dim;
    head_cfg.output_dim=num_classes;
    head_cfg.use_bias=false;
    network.AddLayer(std::make_unique<LinearHead_t>(head_cfg,stream));

    // Freeze the norm and the head — only the LoRA inside the first
    // layer remains trainable.
    const size_t layer_count=network.LayerCount();
    network.SetLayerTrainable(layer_count-1,false);
    network.SetLayerTrainable(layer_count-2,false);

    ISE_Out::Out()<<"Model: "<<network.TotalParameterCount()
             <<" parameters ("<<layer_count<<" layers)"<<std::endl;
    ISE_Out::Out()<<"LoRA rank="<<lora_rank<<" alpha="<<lora_alpha<<std::endl;

    // Synthetic training data — random features, class labels.
    const uint32_t num_samples=128;

    std::vector<float> input_data(num_samples*dim);
    std::vector<float> target_data(num_samples);
    for(uint32_t i=0;i<num_samples*dim;++i)
    {
      input_data[i]=static_cast<float>((i*7+13)%256)/256.0f;
    }
    for(uint32_t i=0;i<num_samples;++i)
    {
      target_data[i]=static_cast<float>(i%num_classes);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                            {num_samples,dim},
                                                            stream);
    CAIF_DeviceTensor target=CAIF_DeviceTensor::FromHostData(target_data.data(),
                                                             {num_samples},
                                                             stream);

    const uint32_t num_steps=30;
    const float learning_rate=1e-3f;
    network.InitializeAdam(learning_rate);

    ISE_Out::Out()<<"Fine-tuning for "<<num_steps<<" steps..."<<std::endl;

    for(uint32_t step=0;step<num_steps;++step)
    {
      network.ZeroGradients();

      CAIF_DeviceTensor output=network.Forward(input,true);

      CAIF_DeviceTensor grad_output;
      const float loss=CrossEntropy_t::ComputeLossAndGradient(output,
                                                              target,
                                                              grad_output,
                                                              stream);

      network.Backward(grad_output);
      network.OptimizerStep();

      if((step+1)%10==0)
      {
        ISE_Out::Out()<<"  step "<<(step+1)<<"/"<<num_steps<<" loss="<<loss<<std::endl;
      }
    }

    const std::string save_path="finetuned_lora.safetensors";
    network.SaveSafeTensors(save_path);
    ISE_Out::Out()<<"Model saved to "<<save_path<<std::endl;

    ISE_Out::Out()<<"=== Done ==="<<std::endl;
    return 0;
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception: "<<e<<std::endl;
    return 1;
  }
}
