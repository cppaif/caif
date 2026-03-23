# CAIF - C++ AI Framework

A production-grade C++23/CUDA deep learning framework for LLM fine-tuning
and inference. Supports multi-dtype quantization, LoRA, MLA, MoE, and
SafeTensors I/O with HuggingFace model compatibility.

## Overview

CAIF provides GPU-accelerated neural network building blocks with a focus on
large language model architectures. All computation happens on device-resident
tensors with explicit host/device transfers. The framework is designed as a
static library consumed by downstream projects (retrainer, bingen).

**Repository:** Private development repo. Public release at `../caif_pub/`
(GitHub: cppaif/caif). All changes go here first, then sync to caif_pub.

## Building

```bash
cd /mnt/s/dev/ise/ai/caif
scons -j8              # Release build
scons debug=1          # Debug with symbols
scons cuda=0           # CPU-only (no CUDA)
scons -c               # Clean
```

**Output:**
- `libcaif.a` — Static library (deployed to `../../lib/`)
- Headers symlinked at `../../include/caif/`

**Compiler:** GCC, C++23, `-O3 -march=native -ffast-math` (release)
**CUDA:** nvcc with `compute_120/sm_120`, enabled by default

## Project Structure

```
caif/
├── include/caif/           # 83 header files
│   ├── Core               # Base, framework, exceptions, constants
│   ├── Tensor             # DeviceTensor, HostTensor, Tensor (legacy)
│   ├── Device layers      # 30+ GPU-resident layer implementations
│   ├── CUDA               # Kernels, stream, event, context
│   ├── Serialization      # SafeTensors, weight mapping
│   ├── Optimizers         # Adam, SGD
│   ├── Loss functions     # CrossEntropy, MSE, BCE
│   └── Legacy             # CPU backends, old layer types
├── src/                    # 69 .cpp files + 1 .cu file
├── tests/                  # 40+ test executables
├── benchmarks/             # Performance benchmarks
├── legacy/                 # Deprecated implementations
├── legacy_mds/             # Historical design documents
├── scripts/                # sync_pub.sh
├── SConstruct              # Build file
├── MLA.md                  # Multi-head Latent Attention design
├── FLASH_ATTENTION_DESIGN.md
├── TRANSFORMER_DESIGN.md
├── NCCL.md                 # Distributed training design
├── PERFORMANCE.md
├── GLMDEBUG.md             # GLM-4.7 debug tracking
└── CAIFPUB.md              # Public release notes
```

## Dependencies

| Dependency | Purpose | Location |
|------------|---------|----------|
| ISE library | Logging (ISE_Out), strings (ISE_UString), base classes | `../../include/`, `../../lib/` |
| CUDA Toolkit | GPU compute, cuBLAS, cuDNN | `../../third_party/include/cuda/` |
| cuBLAS/cuDNN | Matrix multiply, convolutions | Static linking |
| OpenBLAS | CPU matrix multiply fallback | System or third_party |
| Eigen | CPU backend (optional) | `../../third_party/include/` |
| OpenMP | CPU parallelization | Compiler flag `-fopenmp` |

---

## Architecture Overview

### Layer Composition

```
CAIF_DeviceTransformerModel (complete LM)
├── CAIF_DeviceTokenEmbedding
├── CAIF_DevicePositionalEncoding
├── N x CAIF_DevicePreNormBlock
│   ├── Stage 1: CAIF_DeviceRMSNorm + Attention
│   │   └── CAIF_DeviceMLAttention (GLM) or CAIF_DeviceMultiHeadAttention (Qwen)
│   │       └── Projections: CAIF_DeviceFrozenLinear wrapped by CAIF_DeviceLoRAAdapter
│   └── Stage 2: CAIF_DeviceRMSNorm + FFN
│       └── CAIF_DeviceFFN
│           └── Gate/Up/Down: CAIF_DeviceFrozenLinear wrapped by CAIF_DeviceLoRAAdapter
├── CAIF_DeviceRMSNorm (final norm)
└── CAIF_DeviceLinearHead (lm_head)
```

### Data Flow

```
Host (CPU)                              Device (GPU)
───────────                             ──────────────
SafeTensors file                        CAIF_DeviceTensor (move-only)
     │                                       │
     ├─ Load ──────────────────────────────► FrozenLinear (INT4/BF16)
     │                                       │
Token IDs ──── FromHost() ──────────────► TokenEmbedding.ForwardFromIds()
                                             │
                                        PreNormBlock.Forward() x N
                                             │
                                        RMSNorm → LinearHead
                                             │
                                        Logits ──── ToHost() ──────► Loss
```

---

## Class Reference

### Core Infrastructure

#### CAIF_Base
**File:** `caif_base.h` / `caif_base.cpp`
**Base class:** ISE_Object

Base class for all CAIF objects. Provides logging through ISE_Out integration.

#### CAIF_Framework
**File:** `caif_framework.h` / `caif_framework.cpp`
**Base class:** CAIF_Base

Lightweight framework singleton. Manages backend selection (BLAS, Eigen, CUDA).
Mostly vestigial — GPU operations now go through CAIF_DeviceOps directly.

#### CAIF_Exception / CAIF_Error
**Files:** `caif_exception.h`, `caif_error.h`

Exception classes with CAIF-specific error macros.

#### CAIF_Settings
**File:** `caif_settings.h` / `caif_settings.cpp`

Framework configuration settings.

---

### Data Types

#### CAIF_DataType
**File:** `caif_data_type.h`

Enum class for tensor element types with helper methods.

| Type | Enum | Size | Notes |
|------|------|------|-------|
| Float32 | `CAIF_DataType::Float32` | 4 bytes | Default compute type |
| Float16 | `CAIF_DataType::Float16` | 2 bytes | IEEE half precision |
| BFloat16 | `CAIF_DataType::BFloat16` | 2 bytes | Brain float |
| Int8 | `CAIF_DataType::Int8` | 1 byte | Quantized weights |
| Int4 | `CAIF_DataType::Int4` | 0.5 bytes | Packed 2 per byte, per-group scales |
| Int16/32/64 | `CAIF_DataType::Int*` | varies | General integer types |
| UInt8/16/32/64 | `CAIF_DataType::UInt*` | varies | Unsigned variants |
| Bool | `CAIF_DataType::Bool` | 1 byte | Boolean |
| Float64 | `CAIF_DataType::Float64` | 8 bytes | Double precision |

**Key methods:**
- `ElementSize()` — Bytes per element (INT4 returns 0, use special handling)
- `SafeTensorsName()` — Conversion to/from SafeTensors format strings
- `Name()` — Human-readable name

#### Constants
**File:** `caif_constants.h`

| Constant | Value | Purpose |
|----------|-------|---------|
| `g_caif_epsilon` | 1e-5 | BatchNorm epsilon |
| `g_caif_adam_epsilon` | 1e-8 | Adam optimizer |
| `g_caif_rope_base` | 10000.0 | RoPE theta base |
| `g_caif_ffn_multiplier` | 4 | Dense FFN hidden dim ratio |
| `g_caif_ffn_gated_frac` | 2.0/3.0 | Gated FFN ratio (LLaMA-style) |
| `g_caif_ffn_alignment` | 256 | Round FFN dim to nearest |
| `g_caif_gradient_clip` | 1e9 | Gradient clip (effectively disabled) |
| `g_caif_lora_default_rank` | 16 | Default LoRA rank |
| `g_caif_lora_default_alpha` | 32.0 | Default LoRA alpha |
| `g_caif_quant_group_size` | 128 | INT4 quantization group size |

---

### Tensor System

#### CAIF_DeviceTensor
**File:** `caif_device_tensor.h` / `caif_device_tensor.cpp`

**The core tensor type.** GPU-resident, move-only, no host buffer. All CAIF
device layers operate on this type.

**Construction (static factories only):**
- `Zeros(shape, dtype, stream)` — Zero-initialized tensor
- `Uninitialized(shape, dtype, stream)` — Uninitialized (fast allocation)
- `FromHost(host_data, shape, dtype, stream)` — Upload from CPU
- `FromHostData(CAIF_HostTensor, stream)` — Upload from HostTensor
- `FromHostRaw(pointer, num_elements, dtype, stream)` — Upload raw pointer

**Transfer:**
- `ToHost()` — Download to CAIF_HostTensor (blocking)
- `CopyToHost(pointer)` — Download to raw pointer
- `CopyFromHost(pointer)` — Upload from raw pointer

**Properties:**
- `Shape()` — Dimension vector
- `NumElements()` — Total element count
- `DataType()` — Element type
- `DataPtr()` — Raw GPU pointer (void*)
- `Stream()` — Associated CUDA stream

**Design decisions:**
- Move-only semantics prevent accidental device memory copies
- No dirty flags — transfers are always explicit
- Stream association enables operation ordering without global sync

#### CAIF_HostTensor
**File:** `caif_host_tensor.h` / `caif_host_tensor.cpp`

CPU-side tensor data. Simple wrapper around `std::vector<uint8_t>` with shape
and dtype metadata. Used as the transfer vehicle between host and device.

#### CAIF_Tensor (Legacy)
**File:** `caif_tensor.h` / `caif_tensor.cpp`

Legacy tensor type used by the CPU backend path (CAIF_NeuralNetwork, etc.).
Wraps CAIF_TensorData with shape tracking. Still compiled but not used by
device layer stack.

#### CAIF_CudaStream
**File:** `caif_cuda_stream.h` / `caif_cuda_stream.cpp`

RAII wrapper for `cudaStream_t`. Each layer and tensor is associated with a
stream for operation ordering.

**Key methods:**
- `Synchronize()` — Block until all stream operations complete
- `RecordEvent(CAIF_CudaEvent&)` — Record timeline event
- `WaitEvent(CAIF_CudaEvent&)` — Wait for event from another stream
- `Handle()` — Raw cudaStream_t

#### CAIF_CudaEvent
**File:** `caif_cuda_event.h` / `caif_cuda_event.cpp`

RAII wrapper for `cudaEvent_t`. Used for inter-stream synchronization.

---

### Device Layer Interface

#### CAIF_DeviceLayer
**File:** `caif_device_layer.h`
**Base class:** CAIF_Base

Abstract interface for all GPU-resident layers. Defines the contract that
CAIF_DeviceNetwork uses to manage heterogeneous layer stacks.

**Pure virtual interface:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `Forward` | `CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input, bool training)` | Forward pass |
| `Backward` | `CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)` | Backward pass, returns input gradient |
| `ZeroGradients` | `void ZeroGradients()` | Reset all gradient tensors to zero |
| `ParameterTensorCount` | `size_t ParameterTensorCount() const` | Number of parameter tensors |
| `ParameterTensor` | `CAIF_DeviceTensor &ParameterTensor(size_t index)` | Access parameter by index |
| `GradientTensor` | `CAIF_DeviceTensor &GradientTensor(size_t index)` | Access gradient by index |
| `TotalParameterCount` | `size_t TotalParameterCount() const` | Total scalar parameters |
| `Description` | `std::string Description() const` | Human-readable description |
| `ParameterNames` | `vector<string> ParameterNames(const string &prefix) const` | Serialization names |

**Properties:**
- Non-copyable (device tensors are move-only)
- Movable
- Constructed with a CUDA stream reference

---

### Linear Layers

#### CAIF_DeviceDenseLayer
**File:** `caif_device_dense_layer.h` / `caif_device_dense_layer.cpp`
**Base class:** CAIF_DeviceLayer

Standard trainable linear layer: `output = input @ W + b`.

**Constructor:** `CAIF_DeviceDenseLayer(input_dim, output_dim, use_bias, stream)`

**Parameters:**
- `_weight` [input_dim, output_dim]
- `_bias` [output_dim] (optional)

#### CAIF_DeviceFrozenLinear
**File:** `caif_device_frozen_linear.h` / `caif_device_frozen_linear.cpp`
**Base class:** CAIF_DeviceLayer

Non-trainable linear layer with multi-dtype weight storage. Stores weights in
any dtype (INT4, INT8, BF16, FP16, FP32) and dequantizes on-the-fly to FP32
for computation.

**Constructor:** `CAIF_DeviceFrozenLinear(input_dim, output_dim, storage_dtype, use_bias, stream)`

**Key features:**
- INT4: per-group FP16 scales (group_size=128)
- `cache_fp32` option: when false, avoids caching the FP32 dequantized
  weights (critical for large models — GLM-4.7 at 30B params would need
  120GB+ for FP32 cache)
- `SetScales()` — Load INT4 quantization scales
- Reports 0 trainable parameters (frozen)

#### CAIF_DeviceLoRAAdapter
**File:** `caif_device_lora_adapter.h` / `caif_device_lora_adapter.cpp`
**Base class:** CAIF_DeviceLayer

Low-rank adaptation wrapper. Wraps any base layer (typically FrozenLinear)
with trainable LoRA A and B matrices.

**Constructor:** `CAIF_DeviceLoRAAdapter(base_layer, rank, alpha, input_dim, output_dim, stream)`

**Forward:** `output = base_layer(input) + (input @ A^T @ B^T) * (alpha / rank)`

**Parameters (trainable):**
- `_lora_a` [rank, input_dim] — Kaiming uniform init
- `_lora_b` [output_dim, rank] — Zero init (identity at start)

Base layer parameters are hidden from the optimizer — only A and B are
exposed through `ParameterTensor()`.

#### CAIF_DeviceLinearHead
**File:** `caif_device_linear_head.h` / `caif_device_linear_head.cpp`
**Base class:** CAIF_DeviceLayer

Output projection layer for language model heads. Projects hidden state to
vocabulary logits: `logits = hidden @ W`.

**Constructor:** `CAIF_DeviceLinearHead(dim, vocab_size, stream)`

---

### Attention Layers

#### CAIF_DeviceMLAttention
**File:** `caif_device_ml_attention.h` / `caif_device_ml_attention.cpp`
**Base class:** CAIF_DeviceLayer

Multi-head Latent Attention (MLA). Used by GLM-4.7-Flash and DeepSeek-V2/V3.
Compresses Q and KV through low-rank bottlenecks. Splits head dimensions into
rope and non-rope portions.

**Constructor parameters:**
- `dim` — Model dimension
- `num_heads` — Number of attention heads
- `q_lora_rank` — Q compression rank
- `kv_lora_rank` — KV compression rank
- `qk_rope_head_dim` — Per-head dim for rotary-embedded portion
- `qk_nope_head_dim` — Per-head dim for non-rotary portion
- `v_head_dim` — Per-head value dimension
- `rope_base` — RoPE theta (default 10000)
- `stream` — CUDA stream

**Projections:**
- `q_compress` [dim → q_lora_rank] — Compress Q
- `q_decompress` [q_lora_rank → num_heads * (nope + rope)] — Expand Q
- `kv_compress` [dim → kv_lora_rank + rope] — Compress KV
- `kv_decompress` [kv_lora_rank → num_heads * (nope + v)] — Expand KV
- `o_proj` [num_heads * v_head_dim → dim] — Output projection

**KV-cache advantage:** Stores `compressed_kv + k_pe` (576 floats/position for
GLM-4.7) vs 10,240 for standard MHA. 18x compression.

#### CAIF_DeviceMultiHeadAttention
**File:** `caif_device_multi_head_attention.h` / `caif_device_multi_head_attention.cpp`
**Base class:** CAIF_DeviceLayer

Standard multi-head attention with optional grouped query attention (GQA) and
rotary position embeddings (RoPE).

**Constructor parameters:**
- `dim` — Model dimension
- `num_heads` — Query heads
- `num_kv_heads` — KV heads (< num_heads for GQA)
- `head_dim` — Per-head dimension
- `rope_base` — RoPE theta (0 = no RoPE)
- `has_qkv_bias` — Whether Q/K/V have bias terms
- `stream` — CUDA stream

**Projections:**
- `q_proj` [dim → num_heads * head_dim]
- `k_proj` [dim → num_kv_heads * head_dim]
- `v_proj` [dim → num_kv_heads * head_dim]
- `o_proj` [num_heads * head_dim → dim]

**Attention computation:**
- Flash attention for head_dim in {32, 64, 80, 96, 128}
- Standard attention fallback for other head_dims (e.g., GLM-4.7's 256)
- Causal mask applied automatically

---

### Normalization Layers

#### CAIF_DeviceRMSNorm
**File:** `caif_device_rmsnorm.h` / `caif_device_rmsnorm.cpp`
**Base class:** CAIF_DeviceLayer

Root Mean Square Layer Normalization. Preferred over LayerNorm for modern
transformers (no mean subtraction, fewer ops).

**Constructor:** `CAIF_DeviceRMSNorm(dim, epsilon, stream)`

**Formula:** `y = x / sqrt(mean(x^2) + eps) * gamma`

**Parameters:** `gamma` [dim] (initialized to 1.0)

#### CAIF_DeviceLayerNorm
**File:** `caif_device_layernorm.h` / `caif_device_layernorm.cpp`
**Base class:** CAIF_DeviceLayer

Standard Layer Normalization. Kept for compatibility with models that use it.

**Constructor:** `CAIF_DeviceLayerNorm(dim, epsilon, stream)`

**Formula:** `y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`

**Parameters:** `gamma` [dim], `beta` [dim]

---

### Feed-Forward Networks

#### CAIF_DeviceFFN
**File:** `caif_device_ffn.h` / `caif_device_ffn.cpp`
**Base class:** CAIF_DeviceLayer

Feed-forward network with pluggable activation. Supports both standard (2-weight)
and gated (3-weight, SwiGLU-style) configurations.

**Constructor:** `CAIF_DeviceFFN(input_dim, hidden_dim, activation, stream)`

**Standard mode** (pointwise activation):
```
hidden = input @ W_up
output = activation(hidden) @ W_down
```

**Gated mode** (gated activation like SwiGLU):
```
gate = input @ W_gate
up = input @ W_up
output = (activation(gate) * up) @ W_down
```

Auto-detects gated vs standard from the activation object's `IsGated()` method.

**Parameters:**
- Standard: `W_up` [input_dim, hidden_dim], `W_down` [hidden_dim, input_dim]
- Gated: `W_gate` [input_dim, hidden_dim], `W_up` [input_dim, hidden_dim],
  `W_down` [hidden_dim, input_dim]

---

### Activation Functions

#### CAIF_DeviceActivation (Abstract)
**File:** `caif_device_activation.h`

Abstract interface for activation functions.

**Methods:**
- `Forward(CAIF_DeviceTensor &input)` — Apply activation in-place
- `Backward(CAIF_DeviceTensor &grad, CAIF_DeviceTensor &input)` — Compute activation gradient
- `IsGated()` — Whether this is a binary gated activation
- `Description()` — Name string

#### Pointwise Activations
**File:** `caif_device_pointwise_activations.h`

Unary activations (input → output, same shape):

| Class | Formula |
|-------|---------|
| `CAIF_DeviceReLUActivation` | max(0, x) |
| `CAIF_DeviceSigmoidActivation` | 1 / (1 + exp(-x)) |
| `CAIF_DeviceTanhActivation` | tanh(x) |
| `CAIF_DeviceGELUActivation` | x * 0.5 * (1 + erf(x / sqrt(2))) |
| `CAIF_DeviceSwishActivation` | x * sigmoid(x) |
| `CAIF_DeviceLeakyReLUActivation` | x > 0 ? x : alpha * x |
| `CAIF_DeviceELUActivation` | x > 0 ? x : alpha * (exp(x) - 1) |
| `CAIF_DeviceLinearActivation` | x (identity) |

#### Gated Activations
**File:** `caif_device_gated_activations.h`

Binary activations (gate, up → output). Used by SwiGLU-style FFNs.
`IsGated()` returns true.

| Class | Formula |
|-------|---------|
| `CAIF_DeviceSwiGLU` | swish(gate) * up |
| `CAIF_DeviceGELUGLU` | gelu(gate) * up |
| `CAIF_DeviceSwishGLU` | swish(gate) * up |
| `CAIF_DeviceGLU` | sigmoid(gate) * up |

---

### Transformer Blocks

#### CAIF_DevicePreNormBlock
**File:** `caif_device_pre_norm_block.h` / `caif_device_pre_norm_block.cpp`
**Base class:** CAIF_DeviceLayer

**Generic pre-norm residual block.** The primary building block for all
transformer architectures. Accepts an arbitrary number of (norm, layer) stage
pairs and applies:

```
for each (norm, layer) stage:
    x = x + layer(norm(x))
```

**Constructor:** Takes a vector of `(CAIF_DeviceLayer* norm, CAIF_DeviceLayer* layer)` pairs.

**Replaces architecture-specific blocks.** A 2-stage block with
(RMSNorm, MHA) + (RMSNorm, FFN) is a standard transformer layer.
A 2-stage block with (RMSNorm, MLA) + (RMSNorm, MoE) is a GLM-4.7 layer.
Can scale to 3+ stages for future architectures.

#### CAIF_DeviceTransformerBlock
**File:** `caif_device_transformer_block.h` / `caif_device_transformer_block.cpp`
**Base class:** CAIF_DeviceLayer

Fixed 2-stage pre-norm block: (RMSNorm + MHA) + (RMSNorm + FFN) with
residual connections. Predates the generic PreNormBlock. Still compiled
but PreNormBlock is preferred for new code.

---

### Mixture of Experts

#### CAIF_DeviceMoERouter
**File:** `caif_device_moe_router.h` / `caif_device_moe_router.cpp`
**Base class:** CAIF_DeviceLayer

Expert routing network. Routes each token to top-k experts.

**Constructor:** `CAIF_DeviceMoERouter(input_dim, num_experts, top_k, stream)`

**Forward:** `input [batch, seq_len, dim] → Linear → Softmax → Top-k selection`

**Outputs:** expert_indices, expert_weights, router_logits, router_probs

**Routing types:** TopK (default), ExpertChoice, Soft

#### CAIF_DeviceMoEExpert
**File:** `caif_device_moe_expert.h` / `caif_device_moe_expert.cpp`
**Base class:** CAIF_DeviceLayer

Single expert FFN. Standard or gated (SwiGLU) configuration.

**Constructor:** `CAIF_DeviceMoEExpert(input_dim, hidden_dim, gated, stream)`

**Parameters:**
- Standard: `W_up`, `W_down`
- Gated: `W_gate`, `W_up`, `W_down`

#### CAIF_DeviceMoELayer
**File:** `caif_device_moe_layer.h` / `caif_device_moe_layer.cpp`
**Base class:** CAIF_DeviceLayer

Complete MoE layer. Composes router + multiple experts with sparse activation.

**Constructor parameters:**
- `input_dim`, `hidden_dim` — Expert dimensions
- `num_experts` — Total expert count
- `top_k` — Experts activated per token
- `expert_use_gated` — Whether experts use SwiGLU
- `stream` — CUDA stream

**Auxiliary losses:**
- `load_balance_loss` — Encourages even expert utilization
- `z_loss` — Stabilizes router logits

**Overflow strategies:** Drop, NoOp, Redistribute

#### CAIF_DeviceMoEBlock
**File:** `caif_device_moe_block.h` / `caif_device_moe_block.cpp`
**Base class:** CAIF_DeviceLayer

MoE transformer block: (RMSNorm + MHA) + (RMSNorm + MoELayer). Predates
PreNormBlock. Still compiled but PreNormBlock is preferred.

---

### Embedding Layers

#### CAIF_DeviceTokenEmbedding
**File:** `caif_device_token_embedding.h` / `caif_device_token_embedding.cpp`
**Base class:** CAIF_DeviceLayer

Token ID → vector lookup table.

**Constructor:** `CAIF_DeviceTokenEmbedding(vocab_size, dim, stream)`

**Forward paths:**
- `ForwardFromIds(token_ids)` — Primary: uint32 token IDs → embeddings
- `Forward(input)` — Legacy: one-hot or pre-embedded input

**Parameters:** `embedding_table` [vocab_size, dim]

#### CAIF_DevicePositionalEncoding
**File:** `caif_device_positional_encoding.h` / `caif_device_positional_encoding.cpp`
**Base class:** CAIF_DeviceLayer

Position encoding. Modes: None (when using RoPE), Sinusoidal, Learned.
RoPE is handled by the attention layers themselves, so most LLM configurations
use mode=None here.

#### CAIF_DevicePatchEmbedding
**File:** `caif_device_patch_embedding.h` / `caif_device_patch_embedding.cpp`
**Base class:** CAIF_DeviceLayer

Vision Transformer (ViT) patch embedding. Splits image into patches and
projects each to an embedding vector.

#### CAIF_DeviceSpectrogramEmbedding
**File:** `caif_device_spectrogram_embedding.h` / `caif_device_spectrogram_embedding.cpp`
**Base class:** CAIF_DeviceLayer

Audio spectrogram embedding for speech/audio models.

#### CAIF_DeviceTabularEmbedding
**File:** `caif_device_tabular_embedding.h` / `caif_device_tabular_embedding.cpp`
**Base class:** CAIF_DeviceLayer

Embedding layer for tabular/structured data.

---

### Complete Models

#### CAIF_DeviceTransformerModel
**File:** `caif_device_transformer_model.h` / `caif_device_transformer_model.cpp`
**Base class:** CAIF_DeviceLayer

Complete transformer language model. Sequential composition:

```
TokenEmbedding → PositionalEncoding → N × TransformerBlock → RMSNorm → LinearHead
```

**Constructor parameters:**
- `vocab_size`, `max_seq_len`, `dim`, `num_heads`, `num_layers`
- `ffn_dim`, `output_dim`, `tie_weights`
- `stream`

#### CAIF_DeviceMoETransformerModel
**File:** `caif_device_moe_transformer_model.h` / `caif_device_moe_transformer_model.cpp`
**Base class:** CAIF_DeviceLayer

Transformer with MoE layers replacing dense FFN.

#### CAIF_DeviceViTModel
**File:** `caif_device_vit_model.h` / `caif_device_vit_model.cpp`
**Base class:** CAIF_DeviceLayer

Vision Transformer model. PatchEmbedding → PositionalEncoding → N × TransformerBlock → classification head.

---

### Device Network

#### CAIF_DeviceNetwork
**File:** `caif_device_network.h` / `caif_device_network.cpp`

Sequential container for device layers. Manages the training loop interaction
(forward, loss, backward, optimizer step) across a heterogeneous stack of
CAIF_DeviceLayer subclasses.

**Key methods:**
- `AddLayer(CAIF_DeviceLayer*)` — Add a layer to the stack
- `Forward(input, training)` — Sequential forward through all layers
- `Backward(grad)` — Sequential backward through all layers
- `ZeroGradients()` — Zero all layer gradients
- `CollectParameters()` — Gather all trainable parameters and gradients
- `SetTrainable(layer_index, bool)` — Mark layers as trainable/frozen

---

### Device Operations

#### CAIF_DeviceOps
**File:** `caif_device_ops.h` / `caif_device_ops.cpp`

Static namespace with GPU operations on CAIF_DeviceTensor. All operations
use the output tensor's stream for ordering.

**Matrix operations:**
- `MatMul(A, B, out)` — C = A @ B
- `MatMulTransposeA(A, B, out)` — C = A^T @ B
- `MatMulTransposeB(A, B, out)` — C = A @ B^T
- `BatchedMatMul(A, B, out)` — Batched matrix multiply
- `BatchedMatMulTransposeB(A, B, out)` — Batched C = A @ B^T

**Element-wise:**
- `Add(A, B, out)` — Element-wise addition (used for residual connections)
- `Sub`, `Mul`, `Div`, `Sqrt` — Standard element-wise ops

**Reductions:**
- `ReduceSum`, `ReduceMean`, `ReduceMax`

**Specialized:**
- `Softmax(input, out)` — Row-wise softmax
- `CausalMask(attention_scores)` — Apply upper-triangle mask

---

### Device Context

#### CAIF_DeviceContext
**File:** `caif_device_context.h` / `caif_device_context.cpp`

Singleton managing cuBLAS and cuDNN handles. Lazy initialization on first
`Instance()` call. Thread-safe.

**Manages:**
- `cublasHandle_t` — cuBLAS handle
- `cublasLtHandle_t` — cuBLAS-Lt handle (matmul algorithm selection)
- Workspace memory for cuBLAS-Lt operations

---

### CUDA Kernels

#### caif_cuda_kernels.h / caif_cuda_kernels.cu
**Custom CUDA kernel declarations and implementations.**

| Category | Kernels |
|----------|---------|
| Activations | ReLU, Sigmoid, Tanh, LeakyReLU, GELU, Swish (forward + backward) |
| Gated activations | SwiGLU, GELU-GLU, Swish-GLU merge kernels |
| Flash attention | Forward + backward (head_dim 32/64/80/96/128 only) |
| Standard attention | Fallback for unsupported head_dims |
| Softmax | Row-wise with causal masking, numerical stability |
| RoPE | Rotary position embedding application |
| RMSNorm | Forward + backward |
| Quantization | INT4/INT8 dequantization with per-group scales |
| Element-wise | Vectorized add, mul, etc. |
| Reductions | Sum, mean, max |

**CPU stubs:** `caif_cuda_kernels_cpu.cpp` provides no-op implementations
for CPU-only builds (`cuda=0`).

**Important:** Flash attention kernel only supports head_dim in {32, 64, 80,
96, 128}. GLM-4.7 uses head_dim=256 (128 nope + 128 rope) which falls back
to standard attention. The previous `default:` case in the flash kernel
silently dispatched to the `<64>` template, causing out-of-bounds shared
memory access and NaN gradients. This was fixed by adding explicit fallback.

---

### Serialization

#### CAIF_SafeTensorsFormat
**File:** `caif_safetensors_format.h` / `caif_safetensors_format.cpp`

HuggingFace-compatible SafeTensors format implementation. Zero external
dependencies.

**File format:**
```
[8 bytes: header_size (uint64 LE)]
[header_size bytes: JSON metadata]
[tensor data (aligned)]
```

**Key methods:**
- `Save(path, tensors, metadata)` — Write tensors to file
- `Load(path)` — Read all tensors from file
- `Metadata(path)` — Read only JSON metadata (for sharded index)

**Supports:** F32, F16, BF16, I8, I4 dtypes. Sharded model loading via
`model.safetensors.index.json`.

#### CAIF_WeightMapper
**File:** `caif_weight_mapper.h` / `caif_weight_mapper.cpp`

Maps between HuggingFace and CAIF parameter names.

**Features:**
- Prefix rules: strip/add prefixes (e.g., "model." → "")
- Aliases: handle tied weights (lm_head.weight = embed_tokens.weight)
- Methods: `HfToAif()`, `AifToHf()`, `RequiredHfNames()`

---

### Optimizers

#### CAIF_AdamOptimizer
**File:** `caif_adam_optimizer.h` / `caif_adam_optimizer.cpp`
**Base class:** CAIF_Optimizer

Adam optimizer with bias correction.

**Constructor:** `CAIF_AdamOptimizer(learning_rate, beta1, beta2, epsilon, weight_decay)`

**Defaults:** lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0

**Methods:**
- `InitializeAdam(parameters, gradients, trainable_flags)` — Allocate moment tensors
- `Step()` — Update parameters using first/second moment estimates

#### CAIF_SGDOptimizer
**File:** `caif_sgd_optimizer.h` / `caif_sgd_optimizer.cpp`
**Base class:** CAIF_Optimizer

SGD with optional momentum.

**Constructor:** `CAIF_SGDOptimizer(learning_rate, momentum, weight_decay)`

**Defaults:** lr=0.001, momentum=0.1, weight_decay=0.0

#### CAIF_Optimizer (Abstract)
**File:** `caif_optimizer.h`

Base class with `OptimizerType` enum (Adam, SGD) and factory method.

---

### Loss Functions

#### CAIF_DeviceCrossEntropyLoss
**File:** `caif_device_cross_entropy_loss.h` / `caif_device_cross_entropy_loss.cpp`

Cross-entropy loss computed on device. Supports optional loss masking
(zero out loss for specific token positions).

#### CAIF_LossFunction (Abstract)
**File:** `caif_loss_function.h`

Abstract interface for loss functions. Methods: `Forward()`, `Backward()`.

#### Other Loss Functions
- `CAIF_CategoricalCrossEntropyLoss` — Legacy categorical CE
- `CAIF_CrossEntropyLoss` — Legacy CE
- `CAIF_LossFunctionBCELogits` — Binary cross-entropy with logits
- `CAIF_MeanSquaredErrorLoss` — MSE loss

---

### Activation-Aware Initialization

#### CAIF_ActivationAware
**File:** `caif_activation_aware.h` / `caif_activation_aware.cpp`

Provides activation-specific gain values for Kaiming initialization.
Each activation function has a corresponding gain that accounts for its
effect on variance during forward propagation.

| Activation | Gain |
|-----------|-------|
| ReLU | sqrt(2) ≈ 1.414 |
| Sigmoid | 1.0 |
| Tanh | 5/3 ≈ 1.667 |
| GELU | ~1.7 |
| Swish | ~1.67 |
| LeakyReLU | sqrt(2 / (1 + alpha^2)) |

---

### Legacy Classes (Compiled but Deprecated)

These classes predate the device-tensor architecture. They use `CAIF_Tensor`
with CPU backends (BLAS/Eigen). Still compiled for backward compatibility
with test code but not used in LLM workflows.

| Class | File | Purpose |
|-------|------|---------|
| `CAIF_NeuralNetwork` | `caif_neural_network.h` | CPU-based sequential network |
| `CAIF_DenseLayer` | `caif_dense_layer.h` | CPU trainable dense layer |
| `CAIF_Layer` | `caif_layer.h` | Legacy layer base class |
| `CAIF_TensorBackend` | `caif_tensor_backend.h` | Abstract computation backend |
| `CAIF_BLASBackend` | `caif_blas_backend.h` | OpenBLAS/MKL backend |
| `CAIF_EigenBackend` | `caif_eigen_backend.h` | Eigen backend |
| `CAIF_TensorData` | `caif_tensor_data.h` | Abstract tensor storage |
| `CAIF_CpuTensorData` | `caif_cpu_tensor_data.h` | CPU tensor data |
| `CAIF_TensorAdapter` | `caif_tensor_adapter.h` | Legacy→device bridge |
| `CAIF_Activations` | `caif_activations.h` | Enum-based activation factory |
| `CAIF_BatchNormalizationLayer` | `caif_batch_normalization_layer.h` | Batch norm |
| `CAIF_Convolution2DLayer` | `caif_convolution2d_layer.h` | 2D convolution |
| `CAIF_MaxPooling2DLayer` | `caif_max_pooling2d_layer.h` | Max pooling |
| `CAIF_AveragePooling2DLayer` | `caif_average_pooling2d_layer.h` | Average pooling |
| `CAIF_DropoutLayer` | `caif_dropout_layer.h` | Dropout |
| `CAIF_ReshapeLayer` | `caif_reshape_layer.h` | Tensor reshape |
| `CAIF_FlattenLayer` | `caif_flatten_layer.h` | Tensor flatten |

---

## How It All Fits Together

### Class Hierarchy

```
CAIF_Base (ISE_Object)
├── CAIF_Framework
├── CAIF_DeviceLayer (abstract)
│   ├── Linear
│   │   ├── CAIF_DeviceDenseLayer        (trainable W + b)
│   │   ├── CAIF_DeviceFrozenLinear      (frozen, multi-dtype)
│   │   ├── CAIF_DeviceLoRAAdapter       (wraps base + trainable A, B)
│   │   └── CAIF_DeviceLinearHead        (LM output projection)
│   ├── Attention
│   │   ├── CAIF_DeviceMLAttention       (MLA: GLM, DeepSeek)
│   │   └── CAIF_DeviceMultiHeadAttention (MHA: Qwen, LLaMA)
│   ├── Normalization
│   │   ├── CAIF_DeviceRMSNorm
│   │   └── CAIF_DeviceLayerNorm
│   ├── Feed-Forward
│   │   └── CAIF_DeviceFFN               (standard or gated/SwiGLU)
│   ├── Mixture of Experts
│   │   ├── CAIF_DeviceMoERouter         (token → expert routing)
│   │   ├── CAIF_DeviceMoEExpert         (single expert FFN)
│   │   ├── CAIF_DeviceMoELayer          (router + N experts)
│   │   └── CAIF_DeviceMoEBlock          (norm + attn + norm + MoE)
│   ├── Embeddings
│   │   ├── CAIF_DeviceTokenEmbedding
│   │   ├── CAIF_DevicePositionalEncoding
│   │   ├── CAIF_DevicePatchEmbedding
│   │   ├── CAIF_DeviceSpectrogramEmbedding
│   │   └── CAIF_DeviceTabularEmbedding
│   ├── Composite Blocks
│   │   ├── CAIF_DevicePreNormBlock      (generic N-stage residual)
│   │   └── CAIF_DeviceTransformerBlock  (fixed 2-stage: attn + FFN)
│   └── Complete Models
│       ├── CAIF_DeviceTransformerModel  (embed → blocks → head)
│       ├── CAIF_DeviceMoETransformerModel
│       └── CAIF_DeviceViTModel
├── CAIF_DeviceActivation (abstract)
│   ├── Pointwise: ReLU, Sigmoid, Tanh, GELU, Swish, LeakyReLU, ELU
│   └── Gated: SwiGLU, GELUGLU, SwishGLU, GLU
├── CAIF_Optimizer (abstract)
│   ├── CAIF_AdamOptimizer
│   └── CAIF_SGDOptimizer
├── CAIF_LossFunction (abstract)
│   ├── CAIF_DeviceCrossEntropyLoss
│   ├── CAIF_MeanSquaredErrorLoss
│   └── CAIF_LossFunctionBCELogits
└── CAIF_DeviceNetwork                   (sequential layer container)
```

### Typical LoRA Fine-Tuning Assembly

This is how the retrainer assembles a model for fine-tuning (e.g., Qwen2.5):

```
1. Load HuggingFace config.json → extract architecture params
2. Create CAIF_CudaStream

3. For each transformer layer (0..num_layers-1):
   a. Create attention projections as FrozenLinear (storage_dtype from CLI)
   b. Wrap LoRA-targeted projections with LoRAAdapter (rank, alpha from CLI)
   c. Create MHA with the projection layers
   d. Create FFN with FrozenLinear gate/up/down (wrapped with LoRA if targeted)
   e. Create 2 RMSNorm layers (attn_norm, ffn_norm)
   f. Assemble into PreNormBlock: [(attn_norm, MHA), (ffn_norm, FFN)]

4. Create TokenEmbedding, final RMSNorm, LinearHead
5. Load SafeTensors weights → WeightMapper → assign to FrozenLinear layers
6. Add all layers to DeviceNetwork
7. Mark embedding + final_norm + lm_head as non-trainable
8. Initialize AdamOptimizer with only trainable (LoRA) parameters
9. Training loop: forward → loss → backward → optimizer step
```

### Weight Flow During Forward Pass

```
Token IDs [batch, seq_len]
    │
    ▼
TokenEmbedding.ForwardFromIds()
    → lookup in embedding_table [vocab_size, dim]
    → output [batch, seq_len, dim]
    │
    ▼ (for each PreNormBlock)
RMSNorm(x)
    → normalized [batch, seq_len, dim]
    │
    ▼
Attention.Forward()
    → Q = LoRA_q(FrozenLinear_q(input))     # base + low-rank delta
    → K = LoRA_k(FrozenLinear_k(input))
    → V = LoRA_v(FrozenLinear_v(input))
    → Apply RoPE to Q, K
    → scores = Q @ K^T / sqrt(head_dim)
    → Apply causal mask
    → attn_weights = softmax(scores)
    → context = attn_weights @ V
    → output = LoRA_o(FrozenLinear_o(context))
    │
    ▼
x = x + attention_output                    # residual connection
    │
    ▼
RMSNorm(x)
    │
    ▼
FFN.Forward()
    → gate = LoRA_gate(FrozenLinear_gate(input))   # if gated (SwiGLU)
    → up = LoRA_up(FrozenLinear_up(input))
    → output = (swish(gate) * up) @ LoRA_down(FrozenLinear_down(...))
    │
    ▼
x = x + ffn_output                          # residual connection
    │
    ▼ (after all blocks)
RMSNorm(x)
    │
    ▼
LinearHead: logits = x @ W_head             # [batch, seq_len, vocab_size]
    │
    ▼
CrossEntropyLoss(logits, targets, mask)      # masked loss on assistant tokens only
```

---

## Test Suite

40+ test executables in `tests/`. Key tests:

| Test | What it validates |
|------|-------------------|
| `test_device_tensor` | DeviceTensor creation, transfer, move semantics |
| `test_device_attention` | MHA forward/backward correctness |
| `test_device_ml_attention` | MLA forward/backward |
| `test_device_gqa` | Grouped query attention |
| `test_device_rope` | Rotary position embeddings |
| `test_device_rmsnorm` | RMSNorm forward/backward |
| `test_device_layernorm` | LayerNorm forward/backward |
| `test_device_ffn` | FFN forward/backward (standard + gated) |
| `test_device_frozen_linear` | Multi-dtype weight storage + dequant |
| `test_device_lora_adapter` | LoRA wrapping + gradient flow |
| `test_device_moe` | MoE routing + expert dispatch |
| `test_device_pre_norm_block` | Generic residual block composition |
| `test_device_transformer_block` | Full transformer block |
| `test_device_transformer_model` | End-to-end transformer |
| `test_device_token_embedding` | Token ID → embedding lookup |
| `test_device_cross_entropy` | Loss computation |
| `test_device_kv_cache` | KV cache operations |
| `test_safetensors` | SafeTensors read/write |
| `test_flash_attn_backward_nan` | Regression: flash attn NaN fix |

Run all tests:
```bash
cd tests && ./run_all_tests
```

---

## Design Documents

| Document | Contents |
|----------|----------|
| `MLA.md` | Multi-head Latent Attention design, phases 0-7, progress log |
| `FLASH_ATTENTION_DESIGN.md` | Flash attention kernel design |
| `TRANSFORMER_DESIGN.md` | Transformer architecture details |
| `NCCL.md` | Distributed training (NCCL) design |
| `PERFORMANCE.md` | Performance analysis and benchmarks |
| `GLMDEBUG.md` | GLM-4.7 training debug tracking |
| `CAIFPUB.md` | Public release notes |
| `legacy_mds/` | Historical design docs (may be outdated) |
