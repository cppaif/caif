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
//   - Wrapping a dense layer with a LoRA adapter
//   - Freezing base model weights (only LoRA A/B matrices train)
//   - Training loop with cross-entropy loss
//   - Saving the model with merged LoRA weights
//
// This uses a small network with synthetic data for demonstration.
// In production, load a pretrained model from SafeTensors and wrap
// its attention/FFN projections with LoRA adapters.
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

    // Initialize CUDA
    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    // Model parameters — small for demonstration
    const uint32_t num_classes=64;
    const uint32_t dim=128;
    const uint32_t lora_rank=8;
    const float lora_alpha=16.0f;

    // Build a simple network: DenseLayer (frozen) -> LoRA -> RMSNorm -> LinearHead
    // The base dense layer is frozen; only the LoRA A/B matrices train.
    CAIF_DeviceNetwork network(stream);

    // Base dense layer wrapped with LoRA
    CAIF_DeviceLoRAAdapter::LoRAConfig_t lora_cfg;
    lora_cfg.rank=lora_rank;
    lora_cfg.alpha=lora_alpha;
    lora_cfg.input_dim=dim;
    lora_cfg.output_dim=dim;

    auto dense=std::make_unique<CAIF_DeviceDenseLayer>(dim,dim,CAIF_DeviceActivation_e::None,stream,true);
    auto lora=std::make_unique<CAIF_DeviceLoRAAdapter>(lora_cfg,std::move(dense),stream,42);
    network.AddLayer(std::move(lora));

    // RMSNorm + classification head (both frozen)
    network.AddLayer(std::make_unique<CAIF_DeviceRMSNorm>(dim,stream));

    CAIF_DeviceLinearHead::Config_t head_cfg;
    head_cfg.input_dim=dim;
    head_cfg.output_dim=num_classes;
    head_cfg.use_bias=false;
    network.AddLayer(std::make_unique<CAIF_DeviceLinearHead>(head_cfg,stream));

    // Freeze norm and head — only LoRA adapters train
    const size_t layer_count=network.LayerCount();
    network.SetLayerTrainable(layer_count-1,false);
    network.SetLayerTrainable(layer_count-2,false);

    ISE_Out::Out()<<"Model: "<<network.TotalParameterCount()<<" parameters ("<<layer_count<<" layers)"<<std::endl;
    ISE_Out::Out()<<"LoRA rank="<<lora_rank<<" alpha="<<lora_alpha<<std::endl;

    // Synthetic training data — random features with class labels
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

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{num_samples,dim},stream);
    CAIF_DeviceTensor target=CAIF_DeviceTensor::FromHostData(target_data.data(),{num_samples},stream);

    // Training loop — only LoRA weights update
    const uint32_t num_steps=30;
    const float learning_rate=1e-3f;
    network.InitializeAdam(learning_rate,0.9f,0.999f,1e-8f);

    ISE_Out::Out()<<"Fine-tuning for "<<num_steps<<" steps..."<<std::endl;

    for(uint32_t step=0;step<num_steps;++step)
    {
      network.ZeroGradients();

      CAIF_DeviceTensor output=network.Forward(input,true);

      CAIF_DeviceTensor grad_output;
      const float loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(output,target,grad_output,stream);

      network.Backward(grad_output);
      network.AdamStep();

      if((step+1)%10==0)
      {
        ISE_Out::Out()<<"  step "<<(step+1)<<"/"<<num_steps<<" loss="<<loss<<std::endl;
      }
    }

    // Save model (LoRA weights merged with base)
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
  catch(std::exception &e)
  {
    ISE_Out::ErrLog()<<"Exception: "<<e.what()<<std::endl;
    return 1;
  }
}
