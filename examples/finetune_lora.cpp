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
// Example: Fine-tune a transformer with LoRA adapters
//
// Demonstrates:
//   - Building a transformer and wrapping dense layers with LoRA
//   - Freezing base model weights (only LoRA A/B matrices train)
//   - Training loop with cross-entropy loss
//   - Saving LoRA adapter weights separately
//
// This uses a small model with synthetic data for demonstration.
// In production, load a pretrained model from SafeTensors and wrap
// its attention/FFN projections with LoRA adapters.
//--------------------------------------------------------------------------

#include "caif_device_network.h"
#include "caif_device_token_embedding.h"
#include "caif_device_positional_encoding.h"
#include "caif_device_pre_norm_block.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_dense_layer.h"
#include "caif_device_lora_adapter.h"
#include "caif_device_linear_head.h"
#include "caif_device_cross_entropy_loss.h"
#include "caif_device_context.h"
#include "caif_cuda_stream.h"
#include "caif_device_ffn.h"
#include "caif_device_multi_head_attention.h"
#include "caif_exception.h"
#include "caif_safetensors_format.h"
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
    const uint32_t vocab_size=256;
    const uint32_t max_seq_len=32;
    const uint32_t dim=64;
    const uint32_t num_heads=4;
    const uint32_t ffn_dim=dim*4;
    const uint32_t lora_rank=8;
    const float lora_alpha=16.0f;

    // Build model with LoRA adapters on each transformer block's
    // attention Q/K/V projections.  The base dense layers are created
    // first, then wrapped with CAIF_DeviceLoRAAdapter.
    CAIF_DeviceNetwork network(stream);

    // Token embedding (frozen during fine-tuning)
    CAIF_DeviceTokenEmbedding::Config_t emb_cfg;
    emb_cfg.vocab_size=vocab_size;
    emb_cfg.dim=dim;
    network.AddLayer(std::make_unique<CAIF_DeviceTokenEmbedding>(emb_cfg,stream));
    network.SetLayerTrainable(0,false);

    // One transformer block with LoRA on the attention Q projection
    // In production you would wrap Q, K, V, and FFN projections
    CAIF_DeviceLoRAAdapter::LoRAConfig_t lora_cfg;
    lora_cfg.rank=lora_rank;
    lora_cfg.alpha=lora_alpha;
    lora_cfg.input_dim=dim;
    lora_cfg.output_dim=dim;

    auto q_proj=std::make_unique<CAIF_DeviceDenseLayer>(
      dim,
      dim,
      CAIF_DeviceActivation_e::None,
      stream,
      true);

    auto lora_q=std::make_unique<CAIF_DeviceLoRAAdapter>(
      lora_cfg,
      std::move(q_proj),
      stream,
      42);
    network.AddLayer(std::move(lora_q));

    // RMSNorm + linear head (frozen)
    network.AddLayer(std::make_unique<CAIF_DeviceRMSNorm>(dim,stream));

    CAIF_DeviceLinearHead::Config_t head_cfg;
    head_cfg.input_dim=dim;
    head_cfg.output_dim=vocab_size;
    head_cfg.use_bias=false;
    network.AddLayer(
      std::make_unique<CAIF_DeviceLinearHead>(head_cfg,stream));

    // Freeze the norm and head — only LoRA adapters train
    const size_t layer_count=network.LayerCount();
    network.SetLayerTrainable(layer_count-1,false);
    network.SetLayerTrainable(layer_count-2,false);

    ISE_Out::Out()<<"Model: "<<network.TotalParameterCount()
                  <<" parameters ("<<layer_count<<" layers)"<<std::endl;
    ISE_Out::Out()<<"LoRA rank="<<lora_rank
                  <<" alpha="<<lora_alpha<<std::endl;

    // Synthetic training data
    const uint32_t batch_size=2;
    const uint32_t seq_len=max_seq_len;
    const uint32_t num_tokens=batch_size*seq_len;

    std::vector<float> input_data(num_tokens);
    std::vector<float> target_data(num_tokens);
    for(uint32_t i=0;i<num_tokens;++i)
    {
      input_data[i]=static_cast<float>(i%vocab_size);
      target_data[i]=static_cast<float>((i+1)%vocab_size);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostRaw(
      input_data.data(),
      {batch_size,seq_len},
      CAIF_DataType_e::Float32,
      stream);

    CAIF_DeviceTensor target=CAIF_DeviceTensor::FromHostRaw(
      target_data.data(),
      {batch_size,seq_len},
      CAIF_DataType_e::Float32,
      stream);

    // Training loop — only LoRA weights update
    const uint32_t num_steps=30;
    const float learning_rate=1e-4f;
    network.InitializeAdam(learning_rate,0.9f,0.999f,1e-8f);

    ISE_Out::Out()<<"Fine-tuning for "<<num_steps<<" steps..."
                  <<std::endl;

    for(uint32_t step=0;step<num_steps;++step)
    {
      network.ZeroGradients();

      CAIF_DeviceTensor output=network.Forward(input,true);

      CAIF_DeviceTensor grad_output;
      const float loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(
        output,
        target,
        grad_output,
        stream);

      network.Backward(grad_output);
      network.AdamStep();

      if((step+1)%10==0)
      {
        ISE_Out::Out()<<"  step "<<(step+1)<<"/"<<num_steps
                      <<" loss="<<loss<<std::endl;
      }
    }

    // Save the full model (includes LoRA weights merged with base)
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
