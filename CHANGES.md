# CAIF — Changes

## 2026-05 — re-arch release

This is a major rework of the device-side runtime, picking up every
notable change since the previous public release on 2026-03-23. The
internal class structure, dtype handling, MoE machinery, attention
machinery, and offload substrate all moved substantially; downstream
users who pinned to an earlier build should expect compile-time API
changes and re-test their training/inference flows against the new
defaults.

### Highlights

- **Mixed-precision dtype templating end-to-end.** Every trainable
  device layer is now templated on `<ComputeT, StorageT>` with both
  ∈ {`float`, `__half`, `__nv_bfloat16`} — 9 instantiations per layer
  family. Mix bf16/fp16/fp32 storage with fp32 compute (or any other
  supported pair) per layer. AdamW carries fp32 master weights
  regardless of storage dtype. `CAIF_DeviceFrozenLinear` extends the
  storage spectrum further to `int8_t` and `caif_int4_packed_t` (3×5
  grid) for the pretrained base in LoRA / add-MoE fine-tunes; gradients
  do not flow through these tensors. `LoRAAdapter` stays pinned at
  `<float, float>` by `static_assert`. The previous runtime
  `storage_dtype` switch is gone — kernel dispatch is now compile-time
  constant-folded.
- **CPU offload substrate.** Pinned host tensor type, per-tensor
  offload policy on `CAIF_DeviceFrozenLinear`, a block-level
  scheduler, and an offloaded-Adam optimizer let a 27-layer
  DeepSeek-V2-Lite full-depth bf16 add-MoE land on a 32 GB GPU.
- **Activation (gradient) checkpointing** on
  `CAIF_DevicePreNormBlock` — opt-in, drops the per-block forward
  cache and recomputes during backward.
- **MoE Phase 4 layer surgery.** New `CAIF_DeviceMoEFrozenExpert`
  wraps pretrained expert weights as a frozen 3-projection block;
  external model builders can replace a `CAIF_DeviceFFN` slot in a
  transformer block with a fresh `CAIF_DeviceMoELayer` containing the
  existing experts wrapped frozen plus new trainable experts
  alongside, with the router widened to match.
- **`CAIF_DeviceMoELayer` gating expanded.** `SoftmaxTopK_e`
  unchanged; new `SigmoidNoauxTc_e` gating (DeepSeek-V2 / GLM-4-MoE
  "noaux_tc" — sigmoid scoring + bias-corrected top-k +
  `norm_topk_prob`) is supported on both forward and backward paths,
  with a gradcheck unit test.
- **MHA picks up QK-norm + partial-rotary RoPE.** Optional
  `LoadQNormGamma` / `LoadKNormGamma` add RMSNorm to Q and K post-
  projection (matches OLMoE, Olmo2, Qwen3). New `rope_dim` field on
  `AttentionConfig_t` enables partial-rotary models (Glm4Moe-style
  `partial_rotary_factor < 1.0`) by rotating only the first
  `rope_dim` head dimensions.
- **Optimizers.** Plain SGD, Momentum, RMSprop, and AdaGrad alongside
  the existing Adam/AdamW. Common
  `CAIF_DeviceOptimizer` abstraction; layers expose
  `ParameterTensor`/`GradientTensor` indexing rather than
  optimizer-specific hooks.
- **`CAIF_DevicePreNormBlock` is now a `CAIF_DeviceContainer`.** The
  per-block forward/backward state moved onto `CAIF_RunContext`;
  layer subclasses no longer mutate sideband fields via virtual
  setters.

### Correctness fixes

- **SwiGLU MoE expert.** Both `CAIF_DeviceMoEExpert` and
  `CAIF_DeviceMoEFrozenExpert` now compute the standard
  `silu(gate) * up` SwiGLU. The earlier implementation applied
  SiLU to the up-projection instead, breaking parity against the
  reference implementation on every loaded MoE checkpoint. Forward
  and backward both fixed.
- **MoE router respects `norm_topk_prob` for `SoftmaxTopK`.** The
  router was unconditionally re-normalizing top-k weights to sum
  to 1.0 even when the model config asked for raw softmax
  probabilities (OLMoE, Olmo2). Top-k normalization now reads from
  `Config().norm_topk_prob` independent of gating kind.
- **MHA Forward cache for GQA + FlashAttention combo.** The cached
  K/V layout was wrong when GQA was active and FlashAttention was
  used — backward saw stale shapes. Fixed by tightening the cache
  key.
- **MHA fused-QKV staleness.** A new `_w_qkv_dirty` flag drives
  on-demand rebuild of the fused `[Q|K|V]` weight after any
  per-projection load or in-place edit. The fast-path forward no
  longer consults a stale fused tensor when only one of W_q / W_k /
  W_v changed.
- **Safetensors single-shard fallback.** Models that ship a single
  top-level `model.safetensors` (no shard index) now load via both
  the eager and the lazy `OpenShardedHandle` paths.

### Public API surface (additive)

- New headers:
  - `caif_device_moe_frozen_expert.h` — frozen-weight expert wrapper.
  - `caif_device_moe_expert_base.h` — base for trainable and frozen
    experts.
  - `caif_device_frozen_linear_base.h` — base for the int8/int4-
    storage frozen linear layers.
  - `caif_device_container.h` — multi-sublayer container with
    transparent forward/backward composition (used by
    `CAIF_DevicePreNormBlock`).
  - `caif_device_layer_typed.h` — `<ComputeT, StorageT>` mixin.
- Optional `LoadQNormGamma` / `LoadKNormGamma` /
  `HasQNormGamma` / `HasKNormGamma` on
  `CAIF_DeviceMultiHeadAttention`.
- New `qk_norm_eps` and `rope_dim` fields on
  `CAIF_DeviceMultiHeadAttention::AttentionConfig_t`.
- New `CAIF_DeviceMoELayerFactory::GatingKind_e` enum value
  `SigmoidNoauxTc_e`; new `routed_scaling_factor`, `norm_topk_prob`,
  `gating_kind` fields on the router config.
- New `CAIF_DeviceFrozenLinearBase` virtual + `SetOffloadPolicy`
  surface for the offload scheduler.

### Internal cleanups (visible in source diff)

- Every member variable on the touched layers is now accessed
  through an inline accessor — `_member` direct reads in method
  bodies are gone. Subclasses can override the accessor via vtable
  to substitute behavior without modifying the base.
- `namespace instance` is the only named namespace; no anonymous
  `namespace {}` blocks in headers.
- One class per header / one class per .cpp file. No new `#define`
  dispatch macros — templated free functions + explicit
  enumeration instead.

### Removed

- `caif_categorical_cross_entropy_loss.h`, `caif_dense_layer.h`,
  `caif_convolution2d_layer.h`, `caif_batch_normalization_layer.h`,
  and the old non-device layer headers — superseded by the
  `caif_device_*` family with full `<ComputeT, StorageT>` templating.
- Legacy `caif_matrix_ops.h` / `caif_blas_backend.h` /
  `caif_eigen_backend.h` — host-side reference paths are now folded
  into `caif_ops.h` and dispatched per-tensor location.

