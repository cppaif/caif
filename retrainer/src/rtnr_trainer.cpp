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

#include "retrainer/rtnr_trainer.h"

#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

#include "ise_lib/ise_out.h"
#include "caif/caif_device_transformer_block.h"
#include "caif/caif_device_token_embedding.h"
#include "caif/caif_device_positional_encoding.h"
#include "caif/caif_device_linear_head.h"
#include "caif/caif_device_rmsnorm.h"
#include "caif/caif_device_cross_entropy_loss.h"

using namespace instance;

namespace instance
{

RTNR_Trainer::RTNR_Trainer():_has_val_data(false),
                              _current_step(0),
                              _current_epoch(0),
                              _total_steps(0),
                              _initialized(false)
{
}

RTNR_Trainer::~RTNR_Trainer()
{
}

void RTNR_Trainer::Initialize(const RTNR_TrainConfig &config)
{
  try
  {
    _config=config;

    // Create CUDA stream
    _stream=std::make_unique<CAIF_CudaStream>();

    // Create network
    _network=std::make_unique<CAIF_DeviceNetwork>(*_stream);

    _current_step=0;
    _current_epoch=0;
    _initialized=true;

    ISE_Out::Out()<<"[RTNR] Trainer initialized"<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::Initialize")
}

void RTNR_Trainer::BuildModel()
{
  try
  {
    if(_initialized==false)
    {
      THROW_RTNRE("Trainer not initialized");
    }

    if(_config.vocab_size==0||_config.dim==0||_config.num_heads==0||_config.num_layers==0)
    {
      THROW_RTNRE("Invalid model architecture config");
    }

    ISE_Out::Out()<<"[RTNR] Building transformer model:"<<std::endl;
    ISE_Out::Out()<<"  vocab_size="<<_config.vocab_size<<std::endl;
    ISE_Out::Out()<<"  dim="<<_config.dim<<std::endl;
    ISE_Out::Out()<<"  num_heads="<<_config.num_heads<<std::endl;
    ISE_Out::Out()<<"  num_kv_heads="<<_config.num_kv_heads<<std::endl;
    ISE_Out::Out()<<"  num_layers="<<_config.num_layers<<std::endl;
    ISE_Out::Out()<<"  ffn_dim="<<_config.ffn_dim<<std::endl;
    ISE_Out::Out()<<"  max_seq_len="<<_config.max_seq_len<<std::endl;

    // Token embedding
    CAIF_DeviceTokenEmbedding::Config_t token_cfg;
    token_cfg.vocab_size=_config.vocab_size;
    token_cfg.dim=_config.dim;

    auto token_emb=std::make_unique<CAIF_DeviceTokenEmbedding>(token_cfg,*_stream);
    _network->AddLayer(std::move(token_emb));

    // Positional encoding (learned)
    CAIF_DevicePositionalEncoding::Config_t pos_cfg;
    pos_cfg.max_seq_len=_config.max_seq_len;
    pos_cfg.dim=_config.dim;
    pos_cfg.mode=PositionalEncodingMode_e::Learned;

    auto pos_enc=std::make_unique<CAIF_DevicePositionalEncoding>(pos_cfg,*_stream);
    _network->AddLayer(std::move(pos_enc));

    // Transformer blocks
    uint32_t num_kv_heads=_config.num_kv_heads;
    if(num_kv_heads==0)
    {
      num_kv_heads=_config.num_heads;
    }

    uint32_t ffn_dim=_config.ffn_dim;
    if(ffn_dim==0)
    {
      // Default SwiGLU ffn_dim: 4 * dim * 2/3 rounded to 256
      ffn_dim=((_config.dim*4*2/3+255)/256)*256;
    }

    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t block_cfg;
    block_cfg.dim=_config.dim;
    block_cfg.num_heads=_config.num_heads;
    block_cfg.num_kv_heads=num_kv_heads;
    block_cfg.ffn_dim=ffn_dim;
    block_cfg.dropout_rate=0.0f;
    block_cfg.causal=true;
    block_cfg.use_rope=true;
    block_cfg.rope_base=10000.0f;

    for(uint32_t i=0;i<_config.num_layers;++i)
    {
      auto block=std::make_unique<CAIF_DeviceTransformerBlock>(block_cfg,*_stream);
      _network->AddLayer(std::move(block));
    }

    // Final RMSNorm
    auto final_norm=std::make_unique<CAIF_DeviceRMSNorm>(_config.dim,*_stream);
    _network->AddLayer(std::move(final_norm));

    // LM head (output projection to vocab)
    CAIF_DeviceLinearHead::Config_t head_cfg;
    head_cfg.input_dim=_config.dim;
    head_cfg.output_dim=_config.vocab_size;
    head_cfg.use_bias=false;

    auto lm_head=std::make_unique<CAIF_DeviceLinearHead>(head_cfg,*_stream);
    _network->AddLayer(std::move(lm_head));

    ISE_Out::Out()<<"[RTNR] Model built: "<<_network->LayerCount()<<" layers, "
                  <<_network->TotalParameterCount()<<" parameters"<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::BuildModel")
}

void RTNR_Trainer::LoadModel(const std::string &path)
{
  try
  {
    if(_initialized==false)
    {
      THROW_RTNRE("Trainer not initialized");
    }

    ISE_Out::Out()<<"[RTNR] Loading model from: "<<path<<std::endl;

    _network->LoadSafeTensors(path);

    ISE_Out::Out()<<"[RTNR] Model loaded: "<<_network->LayerCount()<<" layers, "
                  <<_network->TotalParameterCount()<<" parameters"<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::LoadModel")
}

void RTNR_Trainer::LoadData(const std::string &train_path,const std::string &val_path)
{
  try
  {
    ISE_Out::Out()<<"[RTNR] Loading training data: "<<train_path<<std::endl;

    // Check file extension
    if(train_path.size()>4&&train_path.substr(train_path.size()-4)==".tok")
    {
      _train_data.LoadBinary(train_path);
    }
    else
    {
      _train_data.Load(train_path);
    }

    ISE_Out::Out()<<"[RTNR] Training data: "<<_train_data.NumSequences()<<" sequences, "
                  <<_train_data.TotalTokens()<<" tokens"<<std::endl;

    if(val_path.empty()==false)
    {
      ISE_Out::Out()<<"[RTNR] Loading validation data: "<<val_path<<std::endl;

      if(val_path.size()>4&&val_path.substr(val_path.size()-4)==".tok")
      {
        _val_data.LoadBinary(val_path);
      }
      else
      {
        _val_data.Load(val_path);
      }

      _has_val_data=true;

      ISE_Out::Out()<<"[RTNR] Validation data: "<<_val_data.NumSequences()<<" sequences, "
                    <<_val_data.TotalTokens()<<" tokens"<<std::endl;
    }

    // Update vocab size if not set
    if(_config.vocab_size==0)
    {
      _config.vocab_size=_train_data.VocabSize();
      ISE_Out::Out()<<"[RTNR] Detected vocab_size="<<_config.vocab_size<<std::endl;
    }
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::LoadData")
}

float RTNR_Trainer::Train()
{
  try
  {
    if(_initialized==false)
    {
      THROW_RTNRE("Trainer not initialized");
    }

    if(_train_data.IsLoaded()==false)
    {
      THROW_RTNRE("Training data not loaded");
    }

    if(_network->LayerCount()==0)
    {
      THROW_RTNRE("Model not built or loaded");
    }

    const size_t num_sequences=_train_data.NumSequences();
    const uint32_t steps_per_epoch=static_cast<uint32_t>((num_sequences+_config.batch_size-1)/_config.batch_size);
    _total_steps=steps_per_epoch*_config.epochs;

    // Compute resume position
    const uint32_t resume_epoch=_config.resume_step/steps_per_epoch;
    const uint32_t resume_batch=_config.resume_step%steps_per_epoch;

    ISE_Out::Out()<<"[RTNR] Starting training:"<<std::endl;
    ISE_Out::Out()<<"  epochs="<<_config.epochs<<std::endl;
    ISE_Out::Out()<<"  batch_size="<<_config.batch_size<<std::endl;
    ISE_Out::Out()<<"  grad_accum_steps="<<_config.grad_accum_steps<<std::endl;
    ISE_Out::Out()<<"  learning_rate="<<_config.learning_rate<<std::endl;
    ISE_Out::Out()<<"  steps_per_epoch="<<steps_per_epoch<<std::endl;
    ISE_Out::Out()<<"  total_steps="<<_total_steps<<std::endl;
    if(_config.min_loss>0.0f)
    {
      ISE_Out::Out()<<"  min_loss="<<_config.min_loss<<std::endl;
      ISE_Out::Out()<<"  min_epochs="<<_config.min_epochs<<std::endl;
    }
    if(_config.resume_step>0)
    {
      ISE_Out::Out()<<"  resume_step="<<_config.resume_step
                    <<" (epoch "<<(resume_epoch+1)
                    <<", batch "<<resume_batch<<")"<<std::endl;
    }

    // Initialize optimizer
    _network->InitializeAdam(_config.learning_rate,0.9f,0.999f,1e-8f);

    float final_loss=0.0f;
    auto train_start=std::chrono::high_resolution_clock::now();

    for(_current_epoch=resume_epoch;_current_epoch<_config.epochs;++_current_epoch)
    {
      ISE_Out::Out()<<"[RTNR] Epoch "<<(_current_epoch+1)<<"/"<<_config.epochs<<std::endl;

      // Shuffle indices (deterministic per epoch for reproducibility)
      std::vector<size_t> indices=ShuffleIndices(num_sequences,_current_epoch);

      float epoch_loss=0.0f;
      uint32_t epoch_steps=0;

      // Determine starting batch within this epoch
      const uint32_t start_batch=
        (_current_epoch==resume_epoch)?resume_batch*_config.batch_size:0;
      _current_step=_current_epoch*steps_per_epoch+
        start_batch/_config.batch_size;

      if(start_batch>0)
      {
        ISE_Out::Out()<<"[RTNR] Skipping to batch "
                      <<(start_batch/_config.batch_size)
                      <<" in epoch "<<(_current_epoch+1)<<std::endl;
      }

      for(uint32_t batch_start=start_batch;
          batch_start<num_sequences;
          batch_start+=_config.batch_size)
      {
        // Collect batch indices
        std::vector<size_t> batch_indices;
        for(uint32_t i=0;
            i<_config.batch_size && batch_start+i<num_sequences;
            ++i)
        {
          batch_indices.push_back(indices[batch_start+i]);
        }

        // Train step
        const float loss=TrainStep(batch_indices);
        epoch_loss+=loss;
        epoch_steps+=1;
        _current_step+=1;

        // Logging
        if(_current_step%_config.log_interval==0)
        {
          const float avg_loss=epoch_loss/static_cast<float>(epoch_steps);
          const float lr=ComputeLearningRate();

          auto now=std::chrono::high_resolution_clock::now();
          const double elapsed=std::chrono::duration<double>(now-train_start).count();
          const double tokens_per_sec=
            (_current_step*_config.batch_size*_config.max_seq_len)/elapsed;

          ISE_Out::Out()<<"  step "<<_current_step<<"/"<<_total_steps
                        <<" loss="<<avg_loss
                        <<" lr="<<lr
                        <<" tok/s="<<static_cast<uint64_t>(tokens_per_sec)
                        <<std::endl;
        }

        // Periodic checkpoint
        if(_config.checkpoint_interval>0 &&
           _current_step%_config.checkpoint_interval==0 &&
           _config.checkpoint_path.empty()==false)
        {
          ISE_Out::Out()<<"[RTNR] Saving checkpoint at step "
                        <<_current_step<<std::endl;
          SaveLoRAWeights(_config.checkpoint_path);
        }
      }

      final_loss=epoch_loss/static_cast<float>(epoch_steps);
      ISE_Out::Out()<<"[RTNR] Epoch "<<(_current_epoch+1)<<" complete, avg_loss="<<final_loss<<std::endl;

      // Validation
      if(_has_val_data==true)
      {
        const float val_loss=Evaluate();
        ISE_Out::Out()<<"[RTNR] Validation loss="<<val_loss<<std::endl;
        final_loss=val_loss;
      }

      // Early stopping: check loss threshold after min_epochs completed
      if(_config.min_loss>0.0f &&
         final_loss<_config.min_loss &&
         (_current_epoch+1)>=_config.min_epochs)
      {
        ISE_Out::Out()<<"[RTNR] Early stop: loss "<<final_loss
                      <<" < min_loss "<<_config.min_loss
                      <<" (after epoch "<<(_current_epoch+1)<<")"
                      <<std::endl;
        break;
      }
    }

    auto train_end=std::chrono::high_resolution_clock::now();
    const double total_time=std::chrono::duration<double>(train_end-train_start).count();
    ISE_Out::Out()<<"[RTNR] Training complete in "<<total_time<<" seconds"<<std::endl;

    return final_loss;
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::Train")
}

float RTNR_Trainer::TrainStep(const std::vector<size_t> &batch_indices)
{
  try
  {
    // Get batch
    auto [input,target]=_train_data.Batch(batch_indices,_config.max_seq_len,*_stream);

    // Zero gradients at start of accumulation cycle
    if(_current_step%_config.grad_accum_steps==0)
    {
      _network->ZeroGradients();
    }

// #ifdef USE_CAIF_CUDA
//     {
//       _stream->Synchronize();
//       size_t free_mem=0;
//       size_t total_mem=0;
//       cudaMemGetInfo(&free_mem,&total_mem);
//       ISE_Out::Out()<<"[RTNR] GPU memory before forward: free="
//                     <<(free_mem/(1024*1024))<<"MB total="
//                     <<(total_mem/(1024*1024))<<"MB used="
//                     <<((total_mem-free_mem)/(1024*1024))<<"MB"<<std::endl;
//     }
// #endif

    // Forward pass
    CAIF_DeviceTensor output=_network->Forward(input,true);

    // Compute loss and gradient in one pass
    CAIF_DeviceTensor grad_output;
    const float loss=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(output,
                                                                         target,
                                                                         grad_output,
                                                                         *_stream);

    // Backward pass
    _network->Backward(grad_output);

    // Optimizer step at end of accumulation cycle
    if((_current_step+1)%_config.grad_accum_steps==0)
    {
      _network->AdamStep();
    }

    return loss;
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::TrainStep")
}

float RTNR_Trainer::Evaluate()
{
  try
  {
    if(_has_val_data==false)
    {
      return 0.0f;
    }

    const size_t num_sequences=_val_data.NumSequences();
    float total_loss=0.0f;
    uint32_t num_batches=0;

    for(uint32_t batch_start=0;batch_start<num_sequences;batch_start+=_config.batch_size)
    {
      std::vector<size_t> batch_indices;
      for(uint32_t i=0;i<_config.batch_size&&batch_start+i<num_sequences;++i)
      {
        batch_indices.push_back(batch_start+i);
      }

      auto [input,target]=_val_data.Batch(batch_indices,_config.max_seq_len,*_stream);

      // Forward only (no training)
      CAIF_DeviceTensor output=_network->Forward(input,false);

      const float loss=CAIF_DeviceCrossEntropyLoss::ComputeLoss(output,target,*_stream);

      total_loss+=loss;
      num_batches+=1;
    }

    return total_loss/static_cast<float>(num_batches);
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::Evaluate")
}

void RTNR_Trainer::SaveModel(const std::string &path)
{
  try
  {
    ISE_Out::Out()<<"[RTNR] Saving model to: "<<path<<std::endl;
    _network->SaveSafeTensors(path);
    ISE_Out::Out()<<"[RTNR] Model saved"<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::SaveModel")
}

void RTNR_Trainer::BuildGLMModel()
{
  try
  {
    if(_initialized==false)
    {
      THROW_RTNRE("Trainer not initialized");
    }

    if(_config.model_dir.empty()==true)
    {
      THROW_RTNRE("model_dir is required for GLM model");
    }

    ISE_Out::Out()<<"[RTNR] Building GLM model from: "<<_config.model_dir<<std::endl;

    // Parse HuggingFace config
    std::string config_path=_config.model_dir+"/config.json";
    RTNR_GLMConfig_t glm_config=RTNR_GLMModelBuilder::ParseConfig(config_path);

    // Determine storage dtype
    CAIF_DataType::CAIF_DataType_e storage_dtype=CAIF_DataType::CAIF_DataType_e::BFloat16;
    if(_config.storage_dtype=="fp32")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Float32;
    }
    else if(_config.storage_dtype=="fp16")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Float16;
    }
    else if(_config.storage_dtype=="bf16")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::BFloat16;
    }
    else if(_config.storage_dtype=="int8")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Int8;
    }
    else if(_config.storage_dtype=="int4")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Int4;
    }

    // Set default LoRA targets if empty
    std::vector<std::string> lora_targets=_config.lora_targets;
    if(lora_targets.empty()==true&&_config.use_lora==true)
    {
      lora_targets={"q","kv","o"};
    }

    // Build model architecture with LoRA wrapping
    _glm_builder.BuildModel(*_network,
                            *_stream,
                            glm_config,
                            storage_dtype,
                            _config.use_lora?_config.lora_r:0,
                            static_cast<float>(_config.lora_alpha),
                            lora_targets);

    // Load base model weights
    _glm_builder.LoadWeights(*_network,
                             *_stream,
                             _config.model_dir,
                             glm_config,
                             storage_dtype);

    // Resume LoRA weights if specified
    if(_config.resume_lora_path.empty()==false)
    {
      ISE_Out::Out()<<"[RTNR] Resuming LoRA weights from: "<<_config.resume_lora_path<<std::endl;
      RTNR_GLMModelBuilder::LoadLoRAWeights(*_network,_config.resume_lora_path,*_stream);
    }

    // Mark embedding (layer 0), final norm, and lm_head as non-trainable
    // so Adam does not allocate moment tensors for their large weight matrices
    const size_t layer_count=_network->LayerCount();
    _network->SetLayerTrainable(0,false);
    _network->SetLayerTrainable(layer_count-1,false);
    if(layer_count>=2)
    {
      _network->SetLayerTrainable(layer_count-2,false);
    }

    // Freeze lm_head so Backward skips the 1.2GB weight gradient allocation
    auto *lm_head=dynamic_cast<CAIF_DeviceLinearHead *>(&_network->Layer(layer_count-1));
    if(lm_head!=nullptr)
    {
      lm_head->SetFrozen(true);
    }

    // Update config from GLM config
    _config.vocab_size=glm_config.vocab_size;
    _config.dim=glm_config.dim;
    _config.num_layers=glm_config.num_layers;
    _config.num_heads=glm_config.num_heads;

    // Sync to flush all async memory operations from weight loading
    _stream->Synchronize();

    ISE_Out::Out()<<"[RTNR] GLM model built: "<<_network->LayerCount()<<" layers, "
                  <<_network->TotalParameterCount()<<" total parameters"<<std::endl;

// #ifdef USE_CAIF_CUDA
//     {
//       size_t free_mem=0;
//       size_t total_mem=0;
//       cudaMemGetInfo(&free_mem,&total_mem);
//       ISE_Out::Out()<<"[RTNR] GPU memory after model load: free="
//                     <<(free_mem/(1024*1024))<<"MB total="
//                     <<(total_mem/(1024*1024))<<"MB used="
//                     <<((total_mem-free_mem)/(1024*1024))<<"MB"<<std::endl;
//     }
// #endif
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::BuildGLMModel")
}

void RTNR_Trainer::BuildQwenModel()
{
  try
  {
    if(_initialized==false)
    {
      THROW_RTNRE("Trainer not initialized");
    }

    if(_config.model_dir.empty()==true)
    {
      THROW_RTNRE("model_dir is required for Qwen model");
    }

    ISE_Out::Out()<<"[RTNR] Building Qwen model from: "
                  <<_config.model_dir
                  <<std::endl;

    // Parse HuggingFace config
    std::string config_path=_config.model_dir+"/config.json";
    RTNR_QwenConfig_t qwen_config=
        RTNR_QwenModelBuilder::ParseConfig(config_path);

    // Determine storage dtype
    CAIF_DataType::CAIF_DataType_e storage_dtype=
        CAIF_DataType::CAIF_DataType_e::BFloat16;
    if(_config.storage_dtype=="fp32")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Float32;
    }
    else if(_config.storage_dtype=="fp16")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Float16;
    }
    else if(_config.storage_dtype=="bf16")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::BFloat16;
    }
    else if(_config.storage_dtype=="int8")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Int8;
    }
    else if(_config.storage_dtype=="int4")
    {
      storage_dtype=CAIF_DataType::CAIF_DataType_e::Int4;
    }

    // Set default LoRA targets if empty
    std::vector<std::string> lora_targets=_config.lora_targets;
    if(lora_targets.empty()==true && _config.use_lora==true)
    {
      lora_targets={"q","k","v","o"};
    }

    // Build model architecture with LoRA wrapping
    _qwen_builder.BuildModel(*_network,
                              *_stream,
                              qwen_config,
                              storage_dtype,
                              _config.use_lora?_config.lora_r:0,
                              static_cast<float>(_config.lora_alpha),
                              lora_targets);

    // Load base model weights
    _qwen_builder.LoadWeights(*_network,
                               *_stream,
                               _config.model_dir,
                               qwen_config,
                               storage_dtype);

    // Resume LoRA weights if specified
    if(_config.resume_lora_path.empty()==false)
    {
      ISE_Out::Out()<<"[RTNR] Resuming LoRA weights from: "
                    <<_config.resume_lora_path
                    <<std::endl;
      RTNR_QwenModelBuilder::LoadLoRAWeights(*_network,
                                              _config.resume_lora_path,
                                              *_stream);
    }

    // Mark embedding (layer 0), final norm, and lm_head as non-trainable
    const size_t layer_count=_network->LayerCount();
    _network->SetLayerTrainable(0,false);
    _network->SetLayerTrainable(layer_count-1,false);
    if(layer_count>=2)
    {
      _network->SetLayerTrainable(layer_count-2,false);
    }

    // Freeze lm_head so Backward skips the weight gradient allocation
    auto *lm_head=dynamic_cast<CAIF_DeviceLinearHead *>(
                      &_network->Layer(layer_count-1));
    if(lm_head!=nullptr)
    {
      lm_head->SetFrozen(true);
    }

    // Update config from Qwen config
    _config.vocab_size=qwen_config.vocab_size;
    _config.dim=qwen_config.dim;
    _config.num_layers=qwen_config.num_layers;
    _config.num_heads=qwen_config.num_heads;
    _config.num_kv_heads=qwen_config.num_kv_heads;

    // Sync to flush all async memory operations from weight loading
    _stream->Synchronize();

    ISE_Out::Out()<<"[RTNR] Qwen model built: "
                  <<_network->LayerCount()
                  <<" layers, "
                  <<_network->TotalParameterCount()
                  <<" total parameters"
                  <<std::endl;

// #ifdef USE_CAIF_CUDA
//     {
//       size_t free_mem=0;
//       size_t total_mem=0;
//       cudaMemGetInfo(&free_mem,&total_mem);
//       ISE_Out::Out()<<"[RTNR] GPU memory after model load: free="
//                     <<(free_mem/(1024*1024))
//                     <<"MB total="
//                     <<(total_mem/(1024*1024))
//                     <<"MB used="
//                     <<((total_mem-free_mem)/(1024*1024))
//                     <<"MB"
//                     <<std::endl;
//     }
// #endif
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::BuildQwenModel")
}

void RTNR_Trainer::SaveLoRAWeights(const std::string &path)
{
  try
  {
    ISE_Out::Out()<<"[RTNR] Saving LoRA weights to: "<<path<<std::endl;
    RTNR_GLMModelBuilder::SaveLoRAWeights(*_network,path);
    ISE_Out::Out()<<"[RTNR] LoRA weights saved"<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_Trainer::SaveLoRAWeights")
}

float RTNR_Trainer::ComputeLearningRate()const
{
  // Linear warmup + constant
  const uint32_t warmup_steps=static_cast<uint32_t>(_total_steps*_config.warmup_ratio);

  if(_current_step<warmup_steps)
  {
    // Linear warmup
    return _config.learning_rate*static_cast<float>(_current_step)/static_cast<float>(warmup_steps);
  }
  else
  {
    // Constant LR (could add cosine decay here)
    return _config.learning_rate;
  }
}

std::vector<size_t> RTNR_Trainer::ShuffleIndices(size_t count,uint32_t seed)const
{
  std::vector<size_t> indices(count);
  for(size_t i=0;i<count;++i)
  {
    indices[i]=i;
  }

  std::mt19937 rng(seed);
  std::shuffle(indices.begin(),indices.end(),rng);

  return indices;
}

}
