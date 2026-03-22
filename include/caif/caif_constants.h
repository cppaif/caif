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

#pragma once

#include <cstdint>
#include <cstddef>
#include "caif_data_type.h"

namespace instance
{
  // Data types moved to class CAIF_DataType (see aif_data_type.h)

  // Device types
  enum class CAIF_DeviceType_e:uint8_t
  {
    CPU,
    GPU,
    TPU
  };

  // Activation functions
  enum class CAIF_ActivationType_e:uint8_t
  {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
    GELU,
    Swish
  };

  // Loss functions
  enum class CAIF_LossType_e:uint8_t
  {
    MeanSquaredError,
    CrossEntropy,
    BinaryCrossEntropy,
    BinaryCrossEntropyWithLogits,
    CategoricalCrossEntropy,
    Huber,
    MeanAbsoluteError
  };

  // Optimizer types
  enum class CAIF_OptimizerType_e:uint8_t
  {
    SGD,
    Adam,
    AdaGrad,
    RMSprop,
    Momentum
  };

  // Layer types
  enum class CAIF_LayerType_e:uint8_t
  {
    Embedding,
    Dense,
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Reshape,
    MultiHeadAttention,
    TransformerEncoder
  };

  // Global constants
  constexpr uint32_t g_caif_max_tensor_dimensions=8;
  constexpr uint32_t g_caif_default_batch_size=32;
  constexpr float g_caif_default_learning_rate=0.001f;
  // Match PyTorch BatchNorm default eps
  constexpr float g_caif_epsilon=1e-5f;
  // Match PyTorch Adam default eps
  constexpr float g_caif_adam_epsilon=1e-8f;
  constexpr uint32_t g_caif_max_threads=16;

  // Mathematical constants
  constexpr double g_caif_pi=3.14159265358979323846;

  // Neural network constants
  constexpr uint32_t g_caif_max_layers=100;
  constexpr uint32_t g_caif_max_layer_size=65536;
  constexpr float g_caif_default_dropout_rate=0.5f;
  // Match PyTorch BatchNorm default momentum
  constexpr float g_caif_default_momentum=0.1f;
  constexpr float g_caif_default_beta1=0.9f;
  constexpr float g_caif_default_beta2=0.999f;
  constexpr float g_caif_weight_init_scale=0.1f;
  constexpr uint32_t g_caif_default_epochs=100;
  // Optimizer behavior constants
  // Disabled by default to match PyTorch behavior (no clipping)
  constexpr float g_caif_grad_clip_threshold=1000000000.0f; // 1e9
  // Activation-aware initialization gains
  constexpr float g_caif_aa_gain_linear=1.0f;
  constexpr float g_caif_aa_gain_sigmoid=1.0f;
  constexpr float g_caif_aa_gain_tanh=1.6666667f;
  constexpr float g_caif_aa_gain_softmax=1.0f;
  constexpr float g_caif_aa_gain_relu=1.41421356237f;
  constexpr float g_caif_aa_gain_leakyrelu=1.41421356237f;
  constexpr float g_caif_aa_gain_elu=1.41421356237f;
  constexpr float g_caif_aa_gain_gelu=1.41421356237f;
  constexpr float g_caif_aa_gain_swish=1.41421356237f;

  // Convolution layer constants
  constexpr uint32_t g_caif_max_kernel_size=15;
  constexpr uint32_t g_caif_max_filters=1024;
  constexpr uint32_t g_caif_max_channels=1024;
  constexpr uint32_t g_caif_max_tensor_dimension=8192;
  constexpr uint32_t g_caif_conv_dimensions=4;  // [batch, height, width, channels]
  constexpr uint32_t g_caif_conv_batch_idx=0;   // Index of batch dimension in conv shape
  constexpr uint32_t g_caif_conv_height_idx=1;  // Index of height dimension in conv shape
  constexpr uint32_t g_caif_conv_width_idx=2;   // Index of width dimension in conv shape
  constexpr uint32_t g_caif_conv_channel_idx=3; // Index of channel dimension in conv shape
  constexpr uint32_t g_caif_2d_matrix_dimensions=2; // Number of dimensions for 2D matrices (rows, columns)

  // Memory alignment
  constexpr size_t g_caif_memory_alignment=32;

  // cublasLt workspace size (4 MB)
  constexpr size_t g_caif_cublaslt_workspace_size=4*1024*1024;
  constexpr size_t g_caif_cache_line_size=64;

  // Multi-head attention layer constants
  constexpr uint32_t g_caif_attention_weight_count=4;  // Query, Key, Value, Output weights
  constexpr uint32_t g_caif_attention_bias_count=4;    // Query, Key, Value, Output biases
  // 8 total parameters
  constexpr uint32_t g_caif_attention_total_params_with_bias=
    g_caif_attention_weight_count+g_caif_attention_bias_count;

  // RoPE (Rotary Position Embeddings) constants
  constexpr float g_caif_rope_default_base=10000.0f;

  // FFN auto-compute constants (LLaMA-style: ffn_dim = round_to(4*dim*2/3, 256))
  constexpr uint32_t g_caif_ffn_multiplier_numerator=4;
  constexpr uint32_t g_caif_ffn_gated_numerator=2;
  constexpr uint32_t g_caif_ffn_gated_denominator=3;
  constexpr uint32_t g_caif_ffn_alignment=256;

  // Embedding layer constants
  constexpr uint32_t g_caif_embedding_parameter_count=1;  // Token embedding: 1 table

  // Sinusoidal positional encoding base frequency
  constexpr double g_caif_sinusoidal_base=10000.0;

  // LoRA (Low-Rank Adaptation) defaults
  constexpr uint32_t g_caif_lora_default_rank=16;
  constexpr float g_caif_lora_default_alpha=32.0f;

  // Quantization defaults
  constexpr uint32_t g_caif_quant_default_group_size=128;

  // INT8 quantization range
  constexpr float g_caif_int8_max=127.0f;

  // INT4 quantization range
  constexpr float g_caif_int4_max=7.0f;
  constexpr uint32_t g_caif_int4_sign_bit=3;
  constexpr uint32_t g_caif_int4_sign_extend=0xFFFFFFF0;
  constexpr uint32_t g_caif_int4_mask=0x0F;
  constexpr uint32_t g_caif_int4_elements_per_byte=2;
}//end instance namespace
