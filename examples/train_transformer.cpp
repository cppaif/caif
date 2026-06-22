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
//   - Picking a `<ComputeT, StorageT>` instantiation for the model
//     (here: <float, float>; swap to <float, __nv_bfloat16> for
//     bf16 storage + fp32 compute, etc.)
//   - Building CAIF_DeviceTransformerModel and registering it in a
//     CAIF_DeviceNetwork
//   - Forward / cross-entropy loss / backward / Adam step
//   - Saving the trained model to SafeTensors
//
// Trains on synthetic data (repeated token patterns) — just enough to
// verify the pipeline runs end-to-end. For real training, swap the
// synthetic input/target arrays for tokenized text.
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
#include <memory>

using namespace instance;

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF Train Transformer Example ==="<<std::endl;

    // Pick the model dtype here. To change it, edit BOTH template args
    // below. Supported combinations:
    //   <float, float>           — fp32 storage + fp32 compute (default)
    //   <float, __nv_bfloat16>   — bf16 storage + fp32 compute
    //   <float, __half>          — fp16 storage + fp32 compute
    //   <__half, __half>         — fp16 storage + fp16 compute
    //   ... 9 combinations total (3 compute dtypes x 3 storage dtypes)
    typedef CAIF_DeviceTransformerModel<float,float> Model_t;
    typedef CAIF_DeviceCrossEntropyLoss<float,float> CrossEntropy_t;

    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    // Tiny transformer for demonstration.
    const uint32_t vocab_size=256;
    const uint32_t max_seq_len=32;
    const uint32_t dim=64;
    const uint32_t num_heads=2;
    const uint32_t num_layers=2;
    const uint32_t ffn_dim=dim*4;

    CAIF_DeviceTransformerModelConfig model_cfg(
                                           vocab_size,
                                           max_seq_len,
                                           dim,
                                           num_heads,
                                           num_layers,
                                           CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Sinusoidal,
                                           true,
                                           true,
                                           true);
    model_cfg.SetNumKvHeads(num_heads);
    model_cfg.SetFfnDim(ffn_dim);
    model_cfg.SetOutputDim(vocab_size);

    CAIF_DeviceNetwork network(stream);
    std::unique_ptr<Model_t> model=std::make_unique<Model_t>(model_cfg,stream);
    network.AddLayer(std::move(model));

    ISE_Out::Out()<<"Model: "<<network.TotalParameterCount()<<" parameters"<<std::endl;

    // Synthetic training data — repeating token pattern. Replace with
    // tokenized text for real training.
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

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                            {batch_size,seq_len},
                                                            stream);
    CAIF_DeviceTensor target=CAIF_DeviceTensor::FromHostData(target_data.data(),
                                                             {batch_size,seq_len},
                                                             stream);

    const uint32_t num_steps=50;
    const float learning_rate=1e-3f;
    network.InitializeAdam(learning_rate);

    ISE_Out::Out()<<"Training for "<<num_steps<<" steps..."<<std::endl;

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
}
