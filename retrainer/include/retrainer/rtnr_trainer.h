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

/**
 * @file rtnr_trainer.h
 * @brief LLM training loop for retraining open-weight models
 */

#pragma once

#include "rtnr_exception.h"
#include "rtnr_constants.h"
#include "rtnr_token_loader.h"
#include "rtnr_glm_model_builder.h"
#include "rtnr_qwen_model_builder.h"

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include "caif/caif_device_network.h"
#include "caif/caif_device_tensor.h"
#include "caif/caif_cuda_stream.h"
#include "caif/caif_safetensors_format.h"

namespace instance
{

/**
 * @brief Configuration for training
 */
class RTNR_TrainConfig
{
  public:
    // Data
    std::string train_data_path;
    std::string val_data_path;

    // Model
    std::string base_model_path;
    std::string output_path;

    // Architecture (for building model if not loading)
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t num_layers;
    uint32_t ffn_dim;
    uint32_t max_seq_len;

    // Training
    uint32_t epochs;
    uint32_t batch_size;
    uint32_t grad_accum_steps;
    float learning_rate;
    float warmup_ratio;
    float weight_decay;

    // LoRA (if enabled)
    bool use_lora;
    uint32_t lora_r;
    uint32_t lora_alpha;
    float lora_dropout;

    // GLM-specific
    std::string model_type;                   // "standard" or "glm"
    std::string model_dir;                    // HuggingFace model directory
    std::string storage_dtype;                // "fp32", "fp16", "bf16", "int8", "int4"
    std::vector<std::string> lora_targets;    // {"q","kv","o","gate","up","down"}
    bool save_lora_only;                      // Save only LoRA weights
    std::string resume_lora_path;             // Path to resume LoRA weights from

    // Checkpointing
    uint32_t checkpoint_interval;             // Save checkpoint every N steps (0=disabled)
    std::string checkpoint_path;              // Checkpoint output path

    // Resume
    uint32_t resume_step;                     // Resume training from this step (0=start)

    // Early stopping
    float min_loss;                           // Stop when epoch avg loss < this (0=disabled)
    uint32_t min_epochs;                      // Minimum epochs before min_loss can trigger

    // Logging
    uint32_t log_interval;
    bool verbose;

    RTNR_TrainConfig():vocab_size(0),
                          dim(0),
                          num_heads(0),
                          num_kv_heads(0),
                          num_layers(0),
                          ffn_dim(0),
                          max_seq_len(g_rtnr_default_max_seq_len),
                          epochs(g_rtnr_default_epochs),
                          batch_size(g_rtnr_default_batch_size),
                          grad_accum_steps(g_rtnr_default_grad_accum_steps),
                          learning_rate(g_rtnr_default_learning_rate),
                          warmup_ratio(g_rtnr_default_warmup_ratio),
                          weight_decay(g_rtnr_default_weight_decay),
                          use_lora(false),
                          lora_r(g_rtnr_default_lora_r),
                          lora_alpha(g_rtnr_default_lora_alpha),
                          lora_dropout(g_rtnr_default_lora_dropout),
                          model_type("standard"),
                          storage_dtype("bf16"),
                          save_lora_only(false),
                          checkpoint_interval(g_rtnr_default_checkpoint_interval),
                          resume_step(0),
                          min_loss(0.0f),
                          min_epochs(0),
                          log_interval(g_rtnr_default_log_interval),
                          verbose(false)
    {
    }

  protected:

  private:
};

/**
 * @brief Training statistics for a single step
 */
struct RTNR_StepStats_t
{
  uint32_t step;
  uint32_t epoch;
  float loss;
  float learning_rate;
  double tokens_per_second;
};

/**
 * @brief LLM Trainer for retraining transformer models
 *
 * Handles the training loop:
 * 1. Load pre-trained model from SafeTensors
 * 2. Load tokenized training data
 * 3. Run training loop with Adam optimizer
 * 4. Save fine-tuned model
 */
class RTNR_Trainer
{
  public:
    RTNR_Trainer();
    ~RTNR_Trainer();

    /**
     * @brief Initialize training with configuration
     * @param config Training configuration
     */
    void Initialize(const RTNR_TrainConfig &config);

    /**
     * @brief Build a transformer model from scratch
     *
     * Creates a decoder-only transformer with the architecture specified
     * in the config (dim, heads, layers, etc.)
     */
    void BuildModel();

    /**
     * @brief Build a GLM model from HuggingFace directory
     *
     * Parses config.json, builds model with FrozenLinear projections,
     * loads weights, and wraps specified projections with LoRA.
     */
    void BuildGLMModel();

    /**
     * @brief Build a Qwen model from HuggingFace directory
     *
     * Parses config.json, builds model with FrozenLinear projections,
     * loads weights (including Q/K/V bias), and wraps specified
     * projections with LoRA.
     */
    void BuildQwenModel();

    /**
     * @brief Load base model from SafeTensors file
     * @param path Path to .safetensors file
     */
    void LoadModel(const std::string &path);

    /**
     * @brief Load training data
     * @param train_path Path to training tokens file
     * @param val_path Path to validation tokens file (optional)
     */
    void LoadData(const std::string &train_path,const std::string &val_path="");

    /**
     * @brief Run the training loop
     * @return Final validation loss (or training loss if no val data)
     */
    float Train();

    /**
     * @brief Run one training step
     * @param batch_indices Indices of sequences to use
     * @return Loss for this step
     */
    float TrainStep(const std::vector<size_t> &batch_indices);

    /**
     * @brief Evaluate on validation data
     * @return Validation loss
     */
    float Evaluate();

    /**
     * @brief Save the trained model
     * @param path Output path (.safetensors)
     */
    void SaveModel(const std::string &path);

    /**
     * @brief Save only LoRA weights
     * @param path Output path (.safetensors)
     */
    void SaveLoRAWeights(const std::string &path);

    /**
     * @brief Get current training step
     */
    uint32_t CurrentStep()const{return _current_step;}

    /**
     * @brief Get current epoch
     */
    uint32_t CurrentEpoch()const{return _current_epoch;}

    /**
     * @brief Check if training is initialized
     */
    bool IsInitialized()const{return _initialized;}

  protected:

  private:
    RTNR_TrainConfig _config;
    std::unique_ptr<CAIF_CudaStream> _stream;
    std::unique_ptr<CAIF_DeviceNetwork> _network;
    RTNR_GLMModelBuilder _glm_builder;
    RTNR_QwenModelBuilder _qwen_builder;

    RTNR_TokenLoader _train_data;
    RTNR_TokenLoader _val_data;
    bool _has_val_data;

    uint32_t _current_step;
    uint32_t _current_epoch;
    uint32_t _total_steps;
    bool _initialized;

    /**
     * @brief Compute learning rate with warmup schedule
     */
    float ComputeLearningRate()const;

    /**
     * @brief Shuffle sequence indices for epoch
     */
    std::vector<size_t> ShuffleIndices(size_t count,uint32_t seed)const;
};

}
