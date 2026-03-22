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
// Example: Train a small transformer language model from scratch
//
// Demonstrates:
//   - Building a transformer model with CAIF_DeviceNetwork
//   - Forward/backward pass with cross-entropy loss
//   - Adam optimizer training loop
//   - Saving the trained model to SafeTensors format
//
// This trains on synthetic data (repeated token patterns) to verify the
// pipeline works end-to-end.  Replace the synthetic data with real
// tokenized text for actual language model training.
//--------------------------------------------------------------------------

#include "caif_device_network.h"
#include "caif_device_transformer_model.h"
#include "caif_device_cross_entropy_loss.h"
#include "caif_device_context.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cstdint>
#include <iostream>

using namespace instance;

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF Train Transformer Example ==="<<std::endl;

    // Initialize CUDA
    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    // Model configuration — a tiny transformer for demonstration
    const uint32_t vocab_size=256;
    const uint32_t max_seq_len=32;
    const uint32_t dim=64;
    const uint32_t num_heads=2;
    const uint32_t num_layers=2;
    const uint32_t ffn_dim=dim*4;

    CAIF_DeviceTransformerModel::Config_t model_cfg;
    model_cfg.vocab_size=vocab_size;
    model_cfg.max_seq_len=max_seq_len;
    model_cfg.dim=dim;
    model_cfg.num_heads=num_heads;
    model_cfg.num_kv_heads=num_heads;
    model_cfg.num_layers=num_layers;
    model_cfg.ffn_dim=ffn_dim;
    model_cfg.causal=true;
    model_cfg.use_rope=true;
    model_cfg.pe_mode=PositionalEncodingMode_e::Sinusoidal;
    model_cfg.output_dim=vocab_size;
    model_cfg.tie_weights=true;

    // Build model inside a DeviceNetwork
    CAIF_DeviceNetwork network(stream);
    auto model=std::make_unique<CAIF_DeviceTransformerModel>(model_cfg,stream);
    network.AddLayer(std::move(model));

    ISE_Out::Out()<<"Model: "<<network.TotalParameterCount()<<" parameters"<<std::endl;

    // Create synthetic training data — a repeating pattern of tokens
    // In production, load real tokenized text here
    const uint32_t batch_size=4;
    const uint32_t seq_len=max_seq_len;
    const uint32_t num_tokens=batch_size*seq_len;

    std::vector<float> input_data(num_tokens);
    std::vector<float> target_data(num_tokens);
    for(uint32_t i=0;i<num_tokens;++i)
    {
      input_data[i]=static_cast<float>(i%vocab_size);
      target_data[i]=static_cast<float>((i+1)%vocab_size);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),{batch_size,seq_len},stream);
    CAIF_DeviceTensor target=CAIF_DeviceTensor::FromHostData(target_data.data(),{batch_size,seq_len},stream);

    // Training loop
    const uint32_t num_steps=50;
    const float learning_rate=1e-3f;
    network.InitializeAdam(learning_rate,0.9f,0.999f,1e-8f);

    ISE_Out::Out()<<"Training for "<<num_steps<<" steps..."<<std::endl;

    for(uint32_t step=0;step<num_steps;++step)
    {
      network.ZeroGradients();

      // Forward
      CAIF_DeviceTensor output=network.Forward(input,true);

      // Loss + gradient
      CAIF_DeviceTensor grad_output;
      const float loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(output,target,grad_output,stream);

      // Backward
      network.Backward(grad_output);

      // Optimizer step
      network.AdamStep();

      if((step+1)%10==0)
      {
        ISE_Out::Out()<<"  step "<<(step+1)<<"/"<<num_steps<<" loss="<<loss<<std::endl;
      }
    }

    // Save trained model
    const std::string save_path="trained_model.safetensors";
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
