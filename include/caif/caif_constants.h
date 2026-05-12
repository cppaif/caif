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
  // Standard BatchNorm default eps
  constexpr float g_caif_epsilon=1e-5f;
  // Standard Adam default eps
  constexpr float g_caif_adam_epsilon=1e-8f;
  constexpr uint32_t g_caif_max_threads=16;

  // Mathematical constants
  constexpr double g_caif_pi=3.14159265358979323846;

  // Neural network constants
  constexpr uint32_t g_caif_max_layers=100;
  constexpr uint32_t g_caif_max_layer_size=65536;
  constexpr float g_caif_default_dropout_rate=0.5f;
  // Standard BatchNorm default momentum
  constexpr float g_caif_default_momentum=0.1f;
  constexpr float g_caif_default_beta1=0.9f;
  constexpr float g_caif_default_beta2=0.999f;
  // Standard SGD-with-momentum default
  constexpr float g_caif_sgd_default_momentum=0.9f;
  // Standard RMSprop default alpha (smoothing constant)
  constexpr float g_caif_rmsprop_default_alpha=0.99f;
  // Standard RMSprop default eps
  constexpr float g_caif_rmsprop_default_epsilon=1e-8f;
  // Standard AdaGrad default eps
  constexpr float g_caif_adagrad_default_epsilon=1e-10f;
  constexpr float g_caif_weight_init_scale=0.1f;
  constexpr uint32_t g_caif_default_epochs=100;
  // Optimizer behavior constants
  // Disabled by default (no clipping)
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

  // cuBLAS-Lt workspace sizing tiers (bytes) for CAIF_DeviceContext. Chosen
  // at runtime from the active GPU's compute capability. The three tiers
  // correspond to the scratch needed for the fastest cuBLAS-Lt algo set on
  // each generation: pre-Ampere gains nothing from more than a few MB,
  // sm_86 consumer cards need ~16 MB to unlock split-K paths, and sm_80 /
  // sm_89 / sm_90+ datacenter and consumer cards need 32 MB for full
  // coverage of split-K / stream-K variants. Override via
  // CAIF_Settings::SetCublasLtWorkspaceBytes before first handle use.
  constexpr size_t g_caif_cublaslt_workspace_bytes_small=
                                                        4ULL*1024ULL*1024ULL;
  constexpr size_t g_caif_cublaslt_workspace_bytes_medium=
                                                        16ULL*1024ULL*1024ULL;
  constexpr size_t g_caif_cublaslt_workspace_bytes_large=
                                                        32ULL*1024ULL*1024ULL;
  // Free-VRAM safety bound: an override may not consume more than 1/Nth of
  // the device's free VRAM at Initialize() time.
  constexpr int g_caif_cublaslt_workspace_free_vram_divisor=4;

  // Bytes-per-megabyte conversion factor (double) for human-readable
  // workspace / allocation logs.
  constexpr double g_caif_bytes_per_megabyte_d=1024.0*1024.0;

  // Defaults for CAIF_MatMulAlgoCache probe tuning (see caif_ops_device.cpp).
  // Candidates: top-K algos pulled from the cuBLAS-Lt heuristic on a cache
  // miss, timed on real data; fastest is cached. Iters: per-candidate timing
  // repeats to amortize launch jitter. The class holds the live values —
  // these are just the numeric seeds so the cpp carries no bare literals.
  constexpr int g_caif_matmul_probe_candidates_default=30;
  constexpr int g_caif_matmul_probe_iters_default=5;

  // MatMul diagnostic instrumentation toggles. When the kernel-time tracer
  // is on, every cublasLtMatmul call reports its event-timed ms + TFLOPS to
  // stderr together with shape, compute_type, workspace, and the cuBLAS-Lt
  // algo bytes picked on cache miss. Flip these to true only for a perf
  // audit — they force a per-iter cudaEventSynchronize that adds overhead.
  constexpr bool g_caif_matmul_trace_enabled=false;
  constexpr bool g_caif_matmul_skip_probe=false;
  // Unit-conversion constants for the trace TFLOPS calculation.
  constexpr double g_caif_matmul_tflops_per_flop=1.0e-12;
  constexpr double g_caif_matmul_seconds_per_ms=1.0e-3;
  // Modulus used to print pointer-alignment residue in the matmul trace
  // (cuBLAS-Lt heuristics consider pointer alignment when ranking algos).
  constexpr unsigned int g_caif_matmul_trace_alignment_modulus=256u;
  // Optional cuBLAS-Lt workspace-size override for the MatMul probe-and-pick
  // audit. 0 = use the device-context workspace as normal. Non-zero (in MiB)
  // allocates a separate workspace of that size and routes the perf-test
  // matmuls through it so we can measure whether bigger/smaller workspace
  // changes the heuristic's algo pick. Audit-only — set back to 0 after.
  constexpr size_t g_caif_matmul_workspace_override_mib=0;
  constexpr size_t g_caif_bytes_per_mebibyte=1024ULL*1024ULL;

  // MoE Forward stage-timing instrumentation. When enabled, each top-level
  // stage of CAIF_DeviceMoELayer::ForwardImpl reports event-timed ms to
  // stderr — router, dispatch-map, dispatch-gpu, per-expert, combine.
  // Audit-only; flip back to false after a perf audit because the per-stage
  // cudaEventSynchronize forces stream serialization and inflates the bench.
  constexpr bool g_caif_moe_forward_trace_enabled=false;


  // Multi-head attention layer constants.
  // Query, Key, Value, Output weights.
  constexpr uint32_t g_caif_attention_weight_count=4;
  // Query, Key, Value, Output biases.
  constexpr uint32_t g_caif_attention_bias_count=4;
  // Total parameter count when both weights and biases are present.
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

  // Xavier uniform-init scale: limit = sqrt(scale / (fan_in + fan_out)).
  constexpr float g_caif_xavier_uniform_scale=6.0f;

  // Golden-ratio fractional part — used as a deterministic
  // pseudo-random multiplier when initializing embedding tables
  // (i*phi mod 1 produces a low-discrepancy sequence on [0,1)).
  constexpr float g_caif_golden_ratio_frac=0.6180339887f;

  // LoRA (Low-Rank Adaptation) defaults
  constexpr uint32_t g_caif_lora_default_rank=16;
  constexpr float g_caif_lora_default_alpha=32.0f;

  // Quantization defaults
  constexpr uint32_t g_caif_quant_default_group_size=128;

  // INT8 quantization range
  constexpr float g_caif_int8_max=127.0f;

  // CUDA kernel launch block sizes
  constexpr int g_caif_cuda_block_size=256;

  //--------------------------------------------------------------------------
  // Softmax row-reduce block size — per-architecture.
  //
  // Historical note (sm_120 / Blackwell):
  //   Earlier bring-up hit a deterministic miscompute on inter-warp
  //   shared-memory tree reductions with block_size > 32. That pinned
  //   the softmax kernels at block_size = 32 (single warp). The current
  //   kernels (attention_softmax_kernel + _backward) use only
  //   __shfl_xor_sync intra-warp and stage (num_warps) floats for the
  //   cross-warp combine — not the hazardous reduction pattern — so
  //   block_size > 32 is safe again after a numerical validation pass.
  //
  // Values below are the starting point; each should be refined with an
  // empirical sweep on its target arch. Do not change without
  // re-running numerical validation + perf on that arch.
  //--------------------------------------------------------------------------
  constexpr int g_caif_cuda_softmax_block_size_sm75=128;   // Turing
  constexpr int g_caif_cuda_softmax_block_size_sm80=128;   // Ampere (A100)
  constexpr int g_caif_cuda_softmax_block_size_sm86=128;   // Ampere (GA10x)
  constexpr int g_caif_cuda_softmax_block_size_sm89=128;   // Ada Lovelace
  constexpr int g_caif_cuda_softmax_block_size_sm90=128;   // Hopper (H100)
  constexpr int g_caif_cuda_softmax_block_size_sm120=128;  // Blackwell (RTX 50)
  constexpr int g_caif_cuda_softmax_block_size_default=128;

  /**
   * @brief Select softmax block size for a given compute capability.
   *
   * Keeps the arch→block_size mapping in one place. Falls back to the
   * default for unknown / future archs.
   */
  constexpr int SelectSoftmaxBlockSize(const int cc_major,
                                       const int cc_minor)
  {
    if(cc_major==7&&cc_minor==5)
    {
      return g_caif_cuda_softmax_block_size_sm75;
    }
    if(cc_major==8&&cc_minor==0)
    {
      return g_caif_cuda_softmax_block_size_sm80;
    }
    if(cc_major==8&&cc_minor==6)
    {
      return g_caif_cuda_softmax_block_size_sm86;
    }
    if(cc_major==8&&cc_minor==9)
    {
      return g_caif_cuda_softmax_block_size_sm89;
    }
    if(cc_major==9)
    {
      return g_caif_cuda_softmax_block_size_sm90;
    }
    if(cc_major==12)
    {
      return g_caif_cuda_softmax_block_size_sm120;
    }
    return g_caif_cuda_softmax_block_size_default;
  }

  // CUDA warp size
  constexpr int g_caif_cuda_warp_size=32;

  // Full-warp ballot / shuffle participation mask (all 32 lanes).
  constexpr unsigned g_caif_cuda_warp_full_mask=0xffffffffu;

  // Default shared memory per block (bytes) — fallback when device query fails
  constexpr int g_caif_cuda_default_shared_memory=49152;

  // Default max threads per block — fallback when device query fails
  constexpr int g_caif_cuda_max_threads_fallback=1024;

  // Flash attention tile sizes (scalar warp-per-row kernel)
  constexpr int g_caif_fa_fwd_bc=64;
  constexpr int g_caif_fa_bwd_br=64;
  constexpr int g_caif_fa_bwd_bc=128;
  constexpr int g_caif_fa_bwd_dq_br=128;
  constexpr int g_caif_fa_bwd_dq_bc=64;

  // GELU activation constants
  constexpr float g_caif_gelu_sqrt_2_over_pi=0.7978845608f;
  constexpr float g_caif_gelu_coeff=0.044715f;

  // BCE loss epsilon
  constexpr float g_caif_bce_epsilon=1e-6f;

  // ViT LayerNorm epsilon
  constexpr float g_caif_vit_layernorm_eps=1e-6f;

  // Regression accuracy threshold
  constexpr float g_caif_regression_accuracy_threshold=0.1f;

  // Relative Position Bias (T5-style) defaults
  constexpr uint32_t g_caif_rpb_default_num_buckets=32;
  constexpr uint32_t g_caif_rpb_default_max_distance=128;

  // INT4 quantization range
  constexpr float g_caif_int4_max=7.0f;
  constexpr uint32_t g_caif_int4_sign_bit=3;
  constexpr uint32_t g_caif_int4_sign_extend=0xFFFFFFF0;
  constexpr uint32_t g_caif_int4_mask=0x0F;
  constexpr uint32_t g_caif_int4_elements_per_byte=2;

  // Pre-norm block sub-layer layout inside the container's flat sublayer vector.
  // Each stage occupies two slots: [norm, layer].
  constexpr size_t g_caif_prenorm_stage_stride=2;
  constexpr size_t g_caif_prenorm_norm_offset=0;
  constexpr size_t g_caif_prenorm_layer_offset=1;

  // Parameter / frozen-tensor name fragments. These are HuggingFace/Llama-style
  // path components used by checkpoint round-trip and external reference-parity
  // consumers. Centralised here so ParameterNames and FrozenTensorNames
  // overrides cannot drift, and so a future redesign that moves naming
  // policy out of CAIF leaves only this header to change.
  constexpr const char *g_caif_name_weight="weight";
  constexpr const char *g_caif_name_bias="bias";
  constexpr const char *g_caif_name_lora_a="lora_a";
  constexpr const char *g_caif_name_lora_b="lora_b";

  constexpr const char *g_caif_name_q_proj="q_proj.";
  constexpr const char *g_caif_name_k_proj="k_proj.";
  constexpr const char *g_caif_name_v_proj="v_proj.";
  constexpr const char *g_caif_name_o_proj="o_proj.";

  constexpr const char *g_caif_name_input_layernorm="input_layernorm.";
  constexpr const char *g_caif_name_self_attn="self_attn.";
  constexpr const char *g_caif_name_post_attention_layernorm="post_attention_layernorm.";
  constexpr const char *g_caif_name_mlp="mlp.";

  // Description() return strings for layer subclasses. Centralised so callers
  // (logging, bench tags, naming profiles) match exact-string expectations.
  constexpr const char *g_caif_description_frozen_linear="FrozenLinear";
  constexpr const char *g_caif_description_relu="ReLU";
  constexpr const char *g_caif_description_gelu="GELU";
  constexpr const char *g_caif_description_sigmoid="Sigmoid";
  constexpr const char *g_caif_description_tanh="Tanh";
  constexpr const char *g_caif_description_swish="Swish";
  constexpr const char *g_caif_description_leaky_relu="LeakyReLU";
  constexpr const char *g_caif_description_elu="ELU";
  constexpr const char *g_caif_description_linear="Linear";

  // Default activation hyperparameters.
  constexpr float g_caif_default_leaky_relu_alpha=0.01f;
  constexpr float g_caif_default_elu_alpha=1.0f;

  // Description() return strings for shape-op layers.
  constexpr const char *g_caif_description_flatten="CAIF_DeviceFlatten";
  constexpr const char *g_caif_description_reshape="CAIF_DeviceReshape";

  // Description() return strings for vision (cuDNN-backed) layers.
  constexpr const char *g_caif_description_average_pooling2d="CAIF_DeviceAveragePooling2D";
  constexpr const char *g_caif_description_max_pooling2d="CAIF_DeviceMaxPooling2D";
  constexpr const char *g_caif_description_conv2d="CAIF_DeviceConv2D";
  constexpr const char *g_caif_description_batch_norm="CAIF_DeviceBatchNorm";

  // Workspace cap (bytes) for cuDNN convolution algorithm selection.
  // The first algorithm whose workspace fits this cap is chosen.
  constexpr size_t g_caif_cudnn_workspace_max_bytes=256ULL*1024ULL*1024ULL;

  // Description() return strings / fragments for container layers.
  constexpr const char *g_caif_description_container_prefix="Container";
  constexpr const char *g_caif_description_pre_norm_block_prefix="PreNormBlock";
  constexpr const char *g_caif_description_transformer_block_prefix="TransformerBlock";
  constexpr const char *g_caif_description_transformer_model_prefix="TransformerModel";
  constexpr const char *g_caif_description_vit_prefix="ViT";

  // Naming prefixes for TransformerModel slot iteration.
  constexpr const char *g_caif_name_embed_tokens="embed_tokens.";
  constexpr const char *g_caif_name_embed_positions="embed_positions.";
  constexpr const char *g_caif_name_layers_prefix="layers.";
  constexpr const char *g_caif_name_norm="norm.";
  constexpr const char *g_caif_name_lm_head="lm_head.";

  // Naming prefixes for ViT slot iteration.
  constexpr const char *g_caif_name_vit_patch_embeddings="embeddings.patch_embeddings.";
  constexpr const char *g_caif_name_vit_position_embeddings="embeddings.position_embeddings.";
  constexpr const char *g_caif_name_vit_encoder_layer="encoder.layer.";
  constexpr const char *g_caif_name_vit_layernorm="layernorm.";
  constexpr const char *g_caif_name_vit_classifier="classifier.";

  // SafeTensors format limits (formerly `namespace SafeTensorsConstants`).
  constexpr uint64_t g_caif_safetensors_max_file_size=2ULL<<40;
  constexpr size_t g_caif_safetensors_max_tensors=4096;
  constexpr size_t g_caif_safetensors_max_header_size=100*1024*1024;
  constexpr size_t g_caif_safetensors_alignment=8;
}//end instance namespace
