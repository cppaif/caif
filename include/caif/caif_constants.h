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
#include <string>
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

  // Activation functions — enum moved to include/caif/caif_activation_type.h
  // (class-scoped: CAIF_ActivationType::CAIF_ActivationType_e). Moved 2026-05-13.

  // Loss functions — enum moved to include/caif/caif_loss_type.h
  // (class-scoped: CAIF_LossType::CAIF_LossType_e). Moved 2026-05-13.

  // Optimizer types — enum moved to include/caif/caif_optimizer_type.h
  // (class-scoped: CAIF_OptimizerType::CAIF_OptimizerType_e). Moved 2026-05-13.

  // Layer types — enum moved to include/caif/caif_layer_type.h
  // (class-scoped: CAIF_LayerType::CAIF_LayerType_e). Moved 2026-05-13.

  // Global constants
  constexpr uint32_t g_caif_max_tensor_dimensions=8;
  constexpr uint32_t g_caif_default_batch_size=32;
  constexpr float g_caif_default_learning_rate=0.001f;
  // Standard BatchNorm default eps
  constexpr float g_caif_epsilon=1e-5f;
  // Standard Adam default eps
  constexpr float g_caif_adam_epsilon=1e-8f;
  // Renormalization division-by-zero guard used by host TopK / MoE gating
  // helpers and matched by the device-kernel epsilon in
  // caif_cuda_kernels.cu.
  constexpr float g_caif_division_epsilon=1.0e-12f;
  // Sentinel rate value meaning "drop nothing" — dropout fast-path skips the
  // RNG / mask path entirely when rate equals this.
  constexpr float g_caif_dropout_full_keep_rate=0.0f;
  // Per-element seed mixer used by the host-side dropout SplitMix64 counter
  // to decorrelate adjacent indices when iterating a flat tensor.
  constexpr uint64_t g_caif_dropout_counter_mix=0xD1342543DE82EF95ULL;
  constexpr uint32_t g_caif_max_threads=16;

  // MatMul-family op labels for the RequireMatchingDtype diagnostic. Shared by
  // the host and device matmul backends so a dtype-mismatch error names the
  // operation that raised it.
  inline const std::string g_caif_op_matmul="MatMul";
  inline const std::string g_caif_op_matmul_bias="MatMulBias";
  inline const std::string g_caif_op_matmul_transpose_a="MatMulTransposeA";
  inline const std::string g_caif_op_matmul_transpose_b="MatMulTransposeB";
  inline const std::string g_caif_op_batched_matmul="BatchedMatMul";
  inline const std::string g_caif_op_batched_matmul_transpose_a="BatchedMatMulTransposeA";
  inline const std::string g_caif_op_batched_matmul_transpose_b="BatchedMatMulTransposeB";

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

  // Mixed-precision loss scaler defaults — match
  // torch.cuda.amp.GradScaler. init=2^16; double the scale every
  // growth_interval overflow-free steps; halve it the moment a step overflows.
  constexpr float g_caif_loss_scaler_init_scale=65536.0f;
  constexpr float g_caif_loss_scaler_growth_factor=2.0f;
  constexpr float g_caif_loss_scaler_backoff_factor=0.5f;
  constexpr uint32_t g_caif_loss_scaler_growth_interval=2000;
  // Standard RMSprop default eps
  constexpr float g_caif_rmsprop_default_epsilon=1e-8f;
  // Standard AdaGrad default eps
  constexpr float g_caif_adagrad_default_epsilon=1e-10f;
  constexpr float g_caif_weight_init_scale=0.1f;
  // CLS-token init uses a small offset on top of the golden-ratio
  // sequence so the deterministic seed differs from the W_proj table.
  constexpr float g_caif_cls_token_init_offset=0.3f;
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
  // Rank of a pooling-layer input tensor ([N, H, W, C]).
  constexpr size_t g_caif_pooling_input_rank=4;

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
  // algo bytes picked on cache miss. Flip these to true only for performance
  // profiling — they force a per-iter cudaEventSynchronize that adds overhead.
  constexpr bool g_caif_matmul_trace_enabled=false;
  constexpr bool g_caif_matmul_skip_probe=false;
  // Unit-conversion constants for the trace TFLOPS calculation.
  constexpr double g_caif_matmul_tflops_per_flop=1.0e-12;
  constexpr double g_caif_matmul_seconds_per_ms=1.0e-3;
  // Modulus used to print pointer-alignment residue in the matmul trace
  // (cuBLAS-Lt heuristics consider pointer alignment when ranking algos).
  constexpr unsigned int g_caif_matmul_trace_alignment_modulus=256u;
  // Optional cuBLAS-Lt workspace-size override for the MatMul probe-and-pick
  // path. 0 = use the device-context workspace as normal. Non-zero (in MiB)
  // allocates a separate workspace of that size and routes the perf-test
  // matmuls through it so we can measure whether bigger/smaller workspace
  // changes the heuristic's algo pick. Debug-only — set back to 0 after.
  constexpr size_t g_caif_matmul_workspace_override_mib=0;
  constexpr size_t g_caif_bytes_per_mebibyte=1024ULL*1024ULL;

  // MoE Forward stage-timing instrumentation. When enabled, each top-level
  // stage of CAIF_DeviceMoELayer::ForwardImpl reports event-timed ms to
  // stderr — router, dispatch-map, dispatch-gpu, per-expert, combine.
  // Debug-only; flip back to false after profiling because the per-stage
  // cudaEventSynchronize forces stream serialization and inflates the bench.
  constexpr bool g_caif_moe_forward_trace_enabled=false;

  // MLA decode-dispatch crossover. The standard decode path is compute-bound (the
  // per-step decompress GEMM, whose FLOPs grow with cache length); the absorbed
  // path is bandwidth-bound (the fixed folded-weight read). The crossover cache
  // length is dim/(qk_nope+v_head) times the GPU's effective compute:bandwidth
  // ratio, computed from queried device properties — no per-machine constant.
  // These describe the hardware/workload, not one machine.
  // FP32 CUDA cores per SM (Ampere/Ada/Hopper/Blackwell).
  constexpr uint32_t g_caif_cuda_fp32_cores_per_sm=128;
  // Two FLOPs per fused multiply-add.
  constexpr uint32_t g_caif_flops_per_fma=2;
  // (G)DDR transfers per memory clock.
  constexpr uint32_t g_caif_memory_transfers_per_clock=2;
  constexpr uint32_t g_caif_bits_per_byte=8;
  // Device clocks are reported in kHz.
  constexpr double g_caif_khz_to_hz=1000.0;
  // Fraction of FP32 peak the small-K decompress GEMM sustains — the single
  // workload factor (~0.5 across modern GPUs) that sets the crossover magnitude.
  constexpr double g_caif_mla_decode_gemm_efficiency=0.54;
  // Crossover ratio used when device properties are unavailable.
  constexpr uint32_t g_caif_mla_decode_fallback_ratio=120;


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
  // Exact (erf) GELU constants: f(x)=0.5*x*(1+erf(x/sqrt(2))); the backward
  // adds x*phi(x) with phi the standard-normal pdf (1/sqrt(2*pi))*exp(-x^2/2).
  constexpr float g_caif_gelu_inv_sqrt2=0.7071067812f;     // 1/sqrt(2)
  constexpr float g_caif_gelu_inv_sqrt2pi=0.3989422804f;   // 1/sqrt(2*pi)

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
  inline const std::string g_caif_name_weight="weight";
  inline const std::string g_caif_name_bias="bias";
  inline const std::string g_caif_name_lora_a="lora_a";
  inline const std::string g_caif_name_lora_b="lora_b";

  // Default activation hyperparameters.
  constexpr float g_caif_default_leaky_relu_alpha=0.01f;
  constexpr float g_caif_default_elu_alpha=1.0f;

  // Workspace cap (bytes) for cuDNN convolution algorithm selection.
  // The first algorithm whose workspace fits this cap is chosen.
  constexpr size_t g_caif_cudnn_workspace_max_bytes=256ULL*1024ULL*1024ULL;

  // Description() class-name tags live in caif_serialization_constants.h
  // under g_serial_tag_*.

  // Default caif-neutral names per CAIF_ParamRole::Role_e. These are the
  // factory defaults that CAIF_RoleRegistry seeds into each role's
  // CAIF_RoleInfo at startup. Callers override individual entries via
  // CAIF_RoleRegistry::SetName / LoadNamesFromJSON to map onto external
  // checkpoint formats (HF safetensors, GGUF, etc.); caif itself never
  // ships any external-format names.
  inline const std::string g_caif_role_name_attn_w_q="w_q";
  inline const std::string g_caif_role_name_attn_w_k="w_k";
  inline const std::string g_caif_role_name_attn_w_v="w_v";
  inline const std::string g_caif_role_name_attn_w_o="w_o";
  inline const std::string g_caif_role_name_attn_bias_q="bias_q";
  inline const std::string g_caif_role_name_attn_bias_k="bias_k";
  inline const std::string g_caif_role_name_attn_bias_v="bias_v";
  inline const std::string g_caif_role_name_attn_q_norm_gamma="q_norm_gamma";
  inline const std::string g_caif_role_name_attn_k_norm_gamma="k_norm_gamma";

  inline const std::string g_caif_role_name_mla_w_q_compress="w_q_compress";
  inline const std::string g_caif_role_name_mla_w_q_decompress="w_q_decompress";
  inline const std::string g_caif_role_name_mla_w_kv_compress="w_kv_compress";
  inline const std::string g_caif_role_name_mla_w_kv_decompress="w_kv_decompress";
  inline const std::string g_caif_role_name_mla_w_o="w_o";
  inline const std::string g_caif_role_name_mla_q_norm_gamma="q_norm_gamma";
  inline const std::string g_caif_role_name_mla_kv_norm_gamma="kv_norm_gamma";

  inline const std::string g_caif_role_name_ffn_w_gate="w_gate";
  inline const std::string g_caif_role_name_ffn_w_up="w_up";
  inline const std::string g_caif_role_name_ffn_w_down="w_down";
  inline const std::string g_caif_role_name_ffn_bias_gate="bias_gate";
  inline const std::string g_caif_role_name_ffn_bias_up="bias_up";
  inline const std::string g_caif_role_name_ffn_bias_down="bias_down";

  inline const std::string g_caif_role_name_moe_router_w="router_w";
  inline const std::string g_caif_role_name_moe_router_bias="router_bias";
  inline const std::string g_caif_role_name_moe_expert_w_gate="expert_w_gate";
  inline const std::string g_caif_role_name_moe_expert_w_up="expert_w_up";
  inline const std::string g_caif_role_name_moe_expert_w_down="expert_w_down";
  inline const std::string g_caif_role_name_moe_shared_expert_w_gate="shared_expert_w_gate";
  inline const std::string g_caif_role_name_moe_shared_expert_w_up="shared_expert_w_up";
  inline const std::string g_caif_role_name_moe_shared_expert_w_down="shared_expert_w_down";
  inline const std::string g_caif_role_name_moe_expert_b_gate="b_gate";
  inline const std::string g_caif_role_name_moe_expert_b_up="b_up";
  inline const std::string g_caif_role_name_moe_expert_b_down="b_down";

  inline const std::string g_caif_role_name_rmsnorm_gamma="rms_norm_gamma";
  inline const std::string g_caif_role_name_layernorm_gamma="layer_norm_gamma";
  inline const std::string g_caif_role_name_layernorm_beta="layer_norm_beta";
  inline const std::string g_caif_role_name_final_norm_gamma="final_norm_gamma";

  inline const std::string g_caif_role_name_token_embedding_table="token_embed";
  inline const std::string g_caif_role_name_position_embedding_table="position_embed";

  inline const std::string g_caif_role_name_linear_head_w="head_w";
  inline const std::string g_caif_role_name_linear_head_bias="head_bias";

  inline const std::string g_caif_role_name_lora_a="lora_a";
  inline const std::string g_caif_role_name_lora_b="lora_b";

  inline const std::string g_caif_role_name_conv_w="conv_w";
  inline const std::string g_caif_role_name_conv_bias="conv_bias";
  inline const std::string g_caif_role_name_bn_gamma="bn_gamma";
  inline const std::string g_caif_role_name_bn_beta="bn_beta";
  inline const std::string g_caif_role_name_bn_running_mean="bn_running_mean";
  inline const std::string g_caif_role_name_bn_running_var="bn_running_var";

  inline const std::string g_caif_role_name_relative_position_bias="rpb";

  inline const std::string g_caif_role_name_embed_proj_w="proj_w";
  inline const std::string g_caif_role_name_embed_proj_bias="proj_bias";
  inline const std::string g_caif_role_name_embed_cls_token="cls_token";

  inline const std::string g_caif_role_name_generic_weight="weight";
  inline const std::string g_caif_role_name_generic_bias="bias";

  // Structural path prefixes used by container layers when assembling
  // child parameter paths. Caif-structural (not external-format
  // specific); downstream model-builder code overrides only the leaf role
  // names — these container path components are always emitted as-is.
  inline const std::string g_caif_path_moe_router="router.";
  inline const std::string g_caif_path_moe_expert="expert_";
  inline const std::string g_caif_path_moe_shared_expert="shared_expert_";
  inline const std::string g_caif_path_transformer_blocks="blocks.";
  inline const std::string g_caif_path_vit_blocks="blocks.";
  inline const std::string g_caif_path_embed_in="embed_in.";
  inline const std::string g_caif_path_embed_pos="embed_pos.";
  inline const std::string g_caif_path_final_norm="final_norm.";
  inline const std::string g_caif_path_head="head.";
  inline const std::string g_caif_path_attn_norm="attn_norm.";
  inline const std::string g_caif_path_attn="attn.";
  inline const std::string g_caif_path_ffn_norm="ffn_norm.";
  inline const std::string g_caif_path_ffn="ffn.";
  inline const std::string g_caif_path_vit_patch_embed="patch_embed.";
  inline const std::string g_caif_path_generic_container_layer="layers.";

  // SafeTensors format limits (formerly `namespace SafeTensorsConstants`).
  constexpr uint64_t g_caif_safetensors_max_file_size=2ULL<<40;
  constexpr size_t g_caif_safetensors_max_tensors=4096;
  constexpr size_t g_caif_safetensors_max_header_size=100*1024*1024;
  constexpr size_t g_caif_safetensors_alignment=8;
}//end instance namespace
