# CAIF - C++ AI Framework

A production-grade C++23/CUDA deep learning framework for LLM fine-tuning
and inference. Supports multi-dtype quantization, LoRA, MLA, MoE, and
SafeTensors I/O with HuggingFace model compatibility. A single op surface
dispatches on tensor location, so the same layer stack runs on GPU or CPU.

## Overview

CAIF provides accelerated neural network building blocks with a focus on
large language model architectures. Layer code is written once against
`CAIF_DeviceLayer` and runs on either device- or host-backed tensors;
dispatch is internal to `CAIF_Ops`. The framework is a static library
consumed by downstream projects.

## Building

CAIF ships with a CMake build:

```bash
cd caif_pub
mkdir build && cd build
cmake ..
cmake --build . -j
```

See README.md for build options (CUDA toggle, tests, dependency paths).

**Output:**
- `libcaif.a` — static library
- `libcaif.so` — shared library

**Compiler:** GCC 13+ / Clang 17+, C++23, `-O3 -march=native -ffast-math` (release)
**CUDA:** nvcc with multi-architecture support, enabled by default when CUDA is present

Target architectures:
- `sm_75` — Turing (T4, RTX 2000) — scalar flash attention fallback
- `sm_80` — Ampere (A100, RTX 3000) — TF32 tensor core flash attention
- `sm_89` — Ada (RTX 4000) — TF32 tensor core flash attention
- `sm_90` — Hopper (H100) — TF32 tensor core flash attention
- `sm_120` — Blackwell (RTX 5000, B200) — TF32 tensor core flash attention

## Project Structure

```
caif_pub/
├── include/caif/           # public C++ headers
├── src/                    # implementation (.cpp / .cu)
├── tests/                  # test executables + shared harness
│   └── caif_cpu_reference/ # CPU reference impls for parity tests
├── examples/               # small demo programs
├── ise_lib/                # vendored ISE minimal subset (logging, strings, exceptions)
├── CMakeLists.txt
├── README.md
├── DESIGN.md
├── CHANGES.md
└── LICENSE                 # Apache 2.0
```

## Dependencies

| Dependency | Purpose | Location |
|------------|---------|----------|
| ISE library | Logging (ISE_Out), strings (ISE_String/ISE_UString), file I/O | Vendored under `ise_lib/` |
| CUDA Toolkit | GPU compute, cuBLAS, cuDNN | System install (auto-detected) |
| cuBLAS/cuDNN | Matrix multiply, convolutions | Linked with CUDA toolkit |
| OpenBLAS | CPU matrix multiply (host backend primary) | System or `-DOPENBLAS_*` paths |
| Eigen | CPU backend fallback | Header-only, optional |
| OpenMP | CPU parallelization | Compiler flag `-fopenmp` |

---

## Architecture Overview

### Run Context — The Single Source Of Truth Per Pass

`CAIF_RunContext` (`include/caif/caif_run_context.h`) carries every piece
of per-pass state that used to be scattered across static booleans and
layer-local sideband setters. It is constructed at the top of every
`CAIF_DeviceNetwork::Forward`/`Backward` call, passed by non-const
reference through the full layer tree, and destroyed on exit.

What it holds:

| Field | Purpose |
|-------|---------|
| `Pass_e Pass()` | `Forward_e` / `Backward_e`; flipped once per top-level call |
| `bool Training()` | Dropout and BatchNorm switch on this |
| `Subsystem_e` stack | Every `ForwardImpl`/`BackwardImpl` auto-pushes its tag via RAII |
| `CAIF_CudaStream &Stream()` | Active CUDA stream for every op in this pass |
| `const CAIF_DeviceTensor *EncoderContext()` | Decoder cross-attn reads this |
| `CAIF_DeviceTensor *GradEncoderContext()` | Decoder cross-attn backward accumulates here |
| `const CAIF_DeviceTensor *PositionBias()` | T5 attention reads this |
| `CAIF_DeviceTensor *GradPositionBias()` | T5 attention backward accumulates here |
| `const CAIF_DeviceTensor *PrefixLengths()` | MHA/MLA prefix-LM masking |
| `uint64_t RandomSeed()` + counter | Deterministic RNG for Dropout |
| `cublasComputeType_t ComputeTypeFor(dt)` | Single decision point for TF32 vs FP32 |

RAII scopes in `caif_run_context_scope.h`:
- `CAIF_RunContextPassScope` — flip `Pass_e`, restore on exit
- `CAIF_RunContextSubsystemScope` — push/pop `Subsystem_e` (dtor swallows
  throws — the one justified exception to `CAIF_CATCH_BLOCK` rules)

### Precision Modes — Performance vs Accuracy (**CORE DESIGN PRINCIPLE**)

The entire framework is built around a first-class distinction between two
precision modes, threaded through every FP32 matrix multiply in the system.
The authoritative control is `CAIF_Settings::MatmulMode_e` with two values:

- **`Performance_e`** — 10-bit-mantissa TF32 tensor-core matmuls on Ampere+
  GPUs (`CUBLAS_COMPUTE_32F_FAST_TF32`). Wins 2-10× throughput over full
  FP32 at the cost of ~3 decimal digits of precision per accumulation.
  Default mode — applied to **both forward and backward** symmetrically.
- **`Accuracy_e`** — 23-bit-mantissa full FP32 accumulation
  (`CUBLAS_COMPUTE_32F`) symmetrically across forward and backward.
  Required where TF32 drift breaks numerics: finite-difference gradient
  checks and reference-parity comparisons against a full-FP32 baseline.

**Symmetry is the invariant.** Every FP32 cuBLAS GEMM — forward or backward —
uses the same compute type for a given mode. The previous
`PreciseGradients`-plus-pass-direction design (TF32 forward, FP32 backward
under `PreciseGradients=true`) was deleted because it produced two bugs:
(1) a backward-pass perf regression when `PreciseGradients=true` was
defaulted on, and (2) a silent accuracy mismatch where TF32 forward
activations were consumed by a backward that assumed its own FP32 inputs
were canonical.

**Single decision point.** Every FP32 cuBLAS call flows through
`CAIF_RunContext::ComputeTypeFor(dt)`:
1. Non-FP32 dtypes (FP16, BF16, INT8, INT4) always use FP32 accumulation
   (`CUBLAS_COMPUTE_32F`) — storage is reduced-precision, accumulator is not.
2. FP32 dtype → read `CAIF_Settings::MatmulMode()`:
   - `Performance_e` → `CUBLAS_COMPUTE_32F_FAST_TF32`
   - `Accuracy_e` → `CUBLAS_COMPUTE_32F`
   Pass direction (forward vs backward) does **not** enter this decision.

Layer code never picks the compute type. `CAIF_Settings::MatmulMode` is
immutable during a pass; flipping mid-pass is undefined.

Tests and downstream tools select the mode through
`CAIF_Settings::SetMatmulMode(MatmulMode_e::Performance_e)` or
`Accuracy_e` at process start.

**Back-compat shim.** `CAIF_Settings::PreciseGradients()` and
`SetPreciseGradients(bool)` still exist as thin wrappers over
`MatmulMode()` — `PreciseGradients==true` is exactly `MatmulMode_e::Accuracy_e`.
No new call sites; existing callers migrate opportunistically.

**Host-backend note.** On `Location_e::Host_e` tensors there is no TF32 at
all — OpenBLAS uses full FP32 for every matmul. The mode flag is a no-op
on host; host and device are therefore independently self-consistent.
Cross-location parity tests use a looser tolerance to absorb expected
TF32 drift when running device in `Performance_e`.

**Always-full-precision locations.** Some places are intrinsically
precision-critical regardless of caller mode:
- **Finite-difference baselines in `CAIF_GradCheck::Check`** — FD in TF32
  is catastrophic cancellation (sign flips at `h~1e-3`); gradcheck pins
  `MatmulMode_e::Accuracy_e` for the FD window, restoring on exit.
- **Optimizer update math** — Adam moment accumulators are FP32.
- **Loss computations** — log-softmax + cross-entropy use FP32 reductions.

This mode selector is why `_in_backward_pass` statics, per-layer
`SetPreciseGradients` setters, and `Pass_e`-based compute-type flipping
were deleted — a single enum read once per op covers every case without
any layer knowing which mode it is in.

### Tensor Location Dispatch

`CAIF_DeviceTensor::Location_e` is `Device_e` (default, CUDA memory) or
`Host_e` (aligned host memory, `new[]`). Every `CAIF_Ops` entry point
asserts all operand tensors share the same location, then internally
dispatches to device (cuBLAS / cuDNN / custom kernels) or host
(OpenBLAS / OpenMP / Eigen) paths. Cross-location movement is explicit
via `ToDevice(stream)` / `ToHostLocation()`.

The name `CAIF_DeviceTensor` is a legacy misnomer now; renaming was
deferred because it ripples too broadly. Every instance of the class
carries its location field.

### Layer Composition

```
CAIF_DeviceTransformerModel (complete LM)  [CAIF_DeviceContainer]
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
                                        PreNormBlock.Forward(ctx) x N
                                             │
                                        RMSNorm → LinearHead
                                             │
                                        Logits ──── ToHost() ──────► Loss
```

---

## Per-Layer dtype Contract

dtype is **a type**, not a runtime config field. Every trainable layer is a
class template `Layer<ComputeT, StorageT>` and the dtype is selected at
construction time. There is no `storage_dtype` field on any layer Config_t
and no `switch(storage_dtype)` per kernel call — the kernel call site is a
direct `launch_xxx<StorageT>(...)` whose template argument is constant-folded
by the compiler.

- `StorageT` — the C++ type of every parameter tensor (weights, biases,
  saved activations, gradient tensors) the layer owns. The on-device
  byte layout matches `sizeof(StorageT)` per element.
- `ComputeT` — selects the cuBLAS-Lt compute type for every MatMul call
  in the layer. `float` → `CUBLAS_COMPUTE_32F`, `__half` →
  `CUBLAS_COMPUTE_32F_FAST_16F`, `__nv_bfloat16` →
  `CUBLAS_COMPUTE_32F_FAST_16BF`. Drives tensor-core path selection.

Both default to `float`. `Layer<>` is `Layer<float, float>`.

### Two boundaries: `CAIF_DeviceLayer` and `CAIF_DeviceLayerTyped`

```
CAIF_DeviceLayer                         (dtype-erased, polymorphic root)
  └── CAIF_DeviceLayerTyped<C, S>        (templated mid-base)
        └── CAIF_DeviceMultiHeadAttention<C, S>
        └── CAIF_DeviceFFN<C, S>
        └── ... (every templated layer)
```

- `CAIF_DeviceLayer` (untemplated) is what `CAIF_DeviceContainer`,
  `CAIF_DeviceNetwork`, `SaveModel`/`LoadModel`, and
  `std::unique_ptr<CAIF_DeviceLayer>` work with. It declares `Forward`,
  `Backward`, `ParameterTensor(i)`, etc. — all return / accept dtype-erased
  `CAIF_DeviceTensor`. Containers can hold any mix of templated layer
  instantiations because they only see this base.
- `CAIF_DeviceLayerTyped<ComputeT, StorageT>` is the templated mid-base
  every templated layer derives from. It exposes the static
  `StorageDtype()` / `ComputeDtype()` accessors (constexpr, return the
  matching `CAIF_DataType_e` value), plus shared protected helpers
  (`AssertInputDtype`, `AllocateOutput`, `CublasComputeType`,
  `StoragePtr`) that every templated layer would otherwise duplicate.

A templated layer reaches its inherited helpers via `using Base::X;`
declarations injected at the protected scope of the derived class — no
`this->` and no qualified `Base<C,S>::X(...)` clutter at every call site.

### Per-layer instantiation grid

| layer family | (ComputeT, StorageT) cells | constraint |
|---|---|---|
| **MHA, GQA, MLA, T5Attention, CrossAttention** | full 3×3 (fp32, fp16, bf16) | every cell legal; cuBLAS-Lt picks the compute_type via `ComputeT` |
| **FFN (any activation)** | full 3×3 | same |
| **DenseLayer, LinearHead** | full 3×3 | same |
| **MoE Router / Expert / Layer** | full 3×3 | same |
| **LayerNorm, RMSNorm, BatchNorm** | full 3×3 | gamma/beta stay fp32 by kernel sig; storage tensors follow `StorageT` |
| **TokenEmbedding, PositionalEncoding, PatchEmbedding, TabularEmbedding, SpectrogramEmbedding** | full 3×3 | non-fp32 cells stage via fp32 in `InitializeWeights` (cast-staging) |
| **CrossEntropyLoss** | full 3×3 | static utility class; logits dtype = StorageT, targets fp32 |
| **RelativePositionBias** | full 3×3 | embedding stays fp32 (atomicAdd safety); bias output goes at StorageT |
| **Dropout** | full 3×3 | mask staged through fp32 → StorageT |
| **Conv2D, Pooling2D family** | full 3×3 (declared) | non-fp32 cells throw at `ForwardImpl` until cuDNN device backend lands; fp32 path works today |
| **5 gated activations (SwiGLU, GeGLU, ReGLU, GLU, Bilinear)** | full 3×3 | strategy classes derived from the non-templated `CAIF_DeviceGatedActivation` polymorphic base; FFN holds them via `unique_ptr<CAIF_DeviceActivation>` |
| **LoRAAdapter** | only `<float, float>` | `static_assert((std::is_same_v<StorageT, float>)==true,...)` and same on `ComputeT`; LoRA A/B matrices are fp32 by design |
| **FrozenLinear** | 3 × 5 (`StorageT` ∈ {fp32, fp16, bf16, `int8_t`, `caif_int4_packed_t`}) | inference-only frozen weights; the two integer storage cells carry a per-row dequant scale and a packed 4-bit codeword type (`caif_int4_packed_t`). Used for the pretrained base in LoRA / add-MoE fine-tunes — gradients never flow through these tensors. |
| **MoEFrozenExpert** | full 3×3 | wraps three `CAIF_DeviceFrozenLinear` columns (gate/up/down) into a frozen expert; instantiation grid covers fp32/fp16/bf16 storage. INT8 / INT4 frozen experts are not currently in the extern-template set — call `CAIF_DeviceFrozenLinear` directly if you need quantized expert storage. |

Every templated layer derives from `CAIF_DeviceLayerTyped<ComputeT, StorageT>`
and ships a paired runtime factory class
`CAIF_DeviceXFactory::Create(..., compute_dtype, storage_dtype)` that
maps the runtime dtype enum to the correct template specialization. The
factory returns `std::unique_ptr<CAIF_DeviceLayer>` — that is the public
API for callers that have the dtype only as a runtime value.

Aggregate instantiation count: ~26 templated layer families at 3×3 cells +
5 gated activations at 3×3 + `CAIF_DeviceFrozenLinear` at 3×5 +
`CAIF_DeviceMoEFrozenExpert` at 3×3 + `LoRAAdapter` at 1 cell ≈ **~290
layer instantiations**. Estimated `libcaif.a` growth versus the
pre-templating runtime-dispatch design: negligible.

### Multi-dtype composition at the container boundary

`CAIF_DeviceContainer` (and the models built on top — Network, PreNormBlock,
TransformerBlock, TransformerModel, ViTModel) is **not** templated. It
holds `std::vector<std::unique_ptr<CAIF_DeviceLayer>>`. A model can have
fp32 norms feeding bf16 MHA feeding fp16 FFN — every sublayer is its own
templated instantiation, and dtype boundaries between sublayers are
crossed by explicit `CAIF_Ops::Cast(...)` operations inserted by the
strategy code at composition time. Containers never know which template
specialization they hold.

### Caller pattern

```cpp
// All-fp32, default templated path.
auto mha = std::make_unique<CAIF_DeviceMultiHeadAttention<>>(cfg, stream);

// bf16 storage, fp32 cuBLAS compute (tensor-core BF16 matmul).
auto mha = std::make_unique<
    CAIF_DeviceMultiHeadAttention<float, __nv_bfloat16>>(cfg, stream);

// Runtime dtype from a config object or model loader: factory bridge.
auto layer = CAIF_DeviceMultiHeadAttentionFactory::Create(
    dim, num_heads, num_kv_heads, head_dim,
    causal, use_rope, rope_base, rope_style, dropout_rate,
    stream,
    /*compute_dtype=*/CAIF_DataType::CAIF_DataType_e::Float32,
    /*storage_dtype=*/CAIF_DataType::CAIF_DataType_e::BFloat16);
// layer is a std::unique_ptr<CAIF_DeviceLayer> wrapping a
// CAIF_DeviceMultiHeadAttention<float, __nv_bfloat16> instance.
```

Downstream model-builder code uses the factory pattern uniformly so
that the dtype matrix is decided once from a config object and the
rest of the pipeline stays dtype-agnostic.

### Why this design

- **Zero hot-path branches.** The runtime `Dispatch*` switch is gone.
  At production scale (thousands of kernel dispatches per backward
  iter) the removed switch cost is measurable on every backward
  shape — most notably on Transformer, MLA, and FFN backward paths.
- **Compile-time correctness.** A layer that needs StorageT-typed
  pointers gets them: `tensor.template DevicePtr<StorageT>()` is a
  type-checked call. The previous bug class — passing
  `tensor.DevicePtr()` (fp32-only) into `launch_xxx<T>(const T*, ...)`
  whose `T` argument got deduced as `float` regardless of the tensor's
  actual storage — is now a compile error rather than silent corruption.
- **Polymorphic composition preserved.** Containers, optimizers, and
  serialization see only `CAIF_DeviceLayer`. Model-builder code, tests,
  and benchmarks that previously instantiated an un-templated class are
  pinned to `<float, float>` for fp32 paths — the factory bridge is the
  only place dtype is a runtime concept after this refactor.

---

## Class Reference

### Core Infrastructure

#### CAIF_Base
**File:** `caif_base.h` / `caif_base.cpp`
**Base class:** ISE_Object

Base class for all CAIF objects. Provides logging through ISE_Out integration.

#### CAIF_Exception / CAIF_Error
**Files:** `caif_exception.h`, `caif_error.h`

Exception classes with CAIF-specific error macros. `THROW_CAIFE` auto-captures
the function name; `CAIF_CATCH_BLOCK()` wraps every function body except
simple getters/setters.

#### CAIF_Settings
**File:** `caif_settings.h` / `caif_settings.cpp`

Framework configuration. Authoritative precision control:
- `MatmulMode()` / `SetMatmulMode(MatmulMode_e)` — enum
  `{Performance_e, Accuracy_e}` consulted once per FP32 cuBLAS GEMM by
  `CAIF_RunContext::ComputeTypeFor(dt)`. Applied symmetrically across
  forward and backward. Immutable during a pass.
- `PreciseGradients()` / `SetPreciseGradients(bool)` — back-compat shims.
  `true` ≡ `Accuracy_e`, `false` ≡ `Performance_e`. No new call sites;
  removal scheduled once all test/bench migrations are in.

#### CAIF_RunContext
**File:** `caif_run_context.h` / `caif_run_context.cpp`

See *Run Context* section above. The authoritative per-pass state object.

---

### Data Types

#### CAIF_DataType
**File:** `caif_data_type.h`

Enum class for tensor element types.

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

**The core tensor type.** Move-only. Carries a `Location_e` field selecting
device (CUDA memory) or host (aligned host memory) storage. Every layer
operates on this type regardless of backend.

**Location:**
- `Location_e::Device_e` — default; `cudaMalloc` storage, cuBLAS/cuDNN/custom ops
- `Location_e::Host_e` — aligned `new[]` storage, OpenBLAS/OpenMP/Eigen ops

**Device factories (static):**
- `Zeros(shape, dtype, stream)`
- `Uninitialized(shape, dtype, stream)`
- `FromHost(host_data, shape, dtype, stream)`
- `FromHostData(CAIF_HostTensor, stream)`
- `FromHostRaw(pointer, num_elements, dtype, stream)`

**Host factories (static):**
- `ZerosHost(shape, dtype)`
- `UninitializedHost(shape, dtype)`

**Cross-location:**
- `ToDevice(stream)` — returns a device-backed copy
- `ToHostLocation()` — returns a host-backed copy
- `ToHost()` — returns a `CAIF_HostTensor` (distinct from `ToHostLocation()`)

**Transfer (host pointers):**
- `CopyToHost(pointer)` / `CopyFromHost(pointer)`

**Properties:**
- `Shape()` — Dimension vector
- `NumElements()` — Total element count
- `DataType()` — Element type
- `Location()` — Storage location
- `DataPtr()` — Raw pointer (void*)
- `Stream()` — CUDA stream (throws on host-backed tensors)

#### CAIF_HostTensor
**File:** `caif_host_tensor.h` / `caif_host_tensor.cpp`

Non-move-only host-side tensor used as the transfer vehicle for SafeTensors
I/O and external data exchange. The host-backed `CAIF_DeviceTensor` (via
`Location_e::Host_e`) is what drives the host compute backend.

#### CAIF_CudaStream
**File:** `caif_cuda_stream.h` / `caif_cuda_stream.cpp`

RAII wrapper for `cudaStream_t`. A single stream is written into the
active `CAIF_RunContext` at the top of each forward/backward call;
every op reads `ctx.Stream()`. Layers retain a construction-time
stream reference used only for tensor allocation, checkpoint I/O,
and optimizer-state work that runs outside a pass.

#### CAIF_CudaEvent
**File:** `caif_cuda_event.h` / `caif_cuda_event.cpp`

RAII wrapper for `cudaEvent_t`. Used for inter-stream synchronization.

---

### Device Layer Interface

#### CAIF_DeviceLayer
**File:** `caif_device_layer.h`
**Base class:** CAIF_Base

Abstract interface for all layers. Composition-time stream reference
captured in the constructor; runtime stream comes from `ctx.Stream()`.

**Non-virtual entry points** (implemented in the base class):
- `Forward(input, ctx)` — pushes `SubsystemTag()`, calls `ForwardImpl`
- `Backward(grad_output, ctx)` — pushes `SubsystemTag()`, calls `BackwardImpl`

**Pure virtuals for subclasses:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `ForwardImpl` | `CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input, CAIF_RunContext &ctx)` | Per-layer forward |
| `BackwardImpl` | `CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output, CAIF_RunContext &ctx)` | Per-layer backward |
| `SubsystemTag` | `CAIF_RunContext::Subsystem_e SubsystemTag() const` | Tag auto-pushed on ctx stack |
| `ZeroGradients` | `void ZeroGradients()` | Reset all gradient tensors to zero |
| `ParameterTensorCount` | `size_t ParameterTensorCount() const` | Number of parameter tensors |
| `ParameterTensor` | `CAIF_DeviceTensor &ParameterTensor(size_t index)` | Access parameter by index |
| `GradientTensor` | `CAIF_DeviceTensor &GradientTensor(size_t index)` | Access gradient by index |
| `TotalParameterCount` | `size_t TotalParameterCount() const` | Total scalar parameters |
| `Description` | `std::string Description() const` | Human-readable description |
| `ParameterNames` | `std::vector<std::string> ParameterNames(const std::string &prefix)` | Serialization names |

**Non-pure virtuals:**
- `AuxLoss()` — default 0; MoE and similar layers override

**Properties:**
- Non-copyable (device tensors are move-only), movable
- `HasStream()` / `Stream()` — construction-time stream access (runtime
  code uses `ctx.Stream()` instead)

The seven sideband setters (`SetContext`, `SetGradContext`,
`SetPositionBias`, `SetGradPositionBias`, `SetPrefixLengths`,
`ClearPrefixLengths`, `HasPrefixLengths`) were deleted. Their
functionality moved to `CAIF_RunContext` fields.

#### CAIF_DeviceContainer
**File:** `caif_device_container.h` / `caif_device_container.cpp`
**Base class:** CAIF_DeviceLayer

A container IS-A layer. Owns `std::vector<std::unique_ptr<CAIF_DeviceLayer>>`
plus a parallel `std::vector<bool>` trainable flags. Provides:

- `AddLayer(std::unique_ptr<CAIF_DeviceLayer>)` — composition API
- `LayerCount()` / `Layer(index)` — heterogeneous access
- `SetLayerTrainable(index, bool)` / `IsLayerTrainable(index)`
- Default `ForwardImpl` chains sublayers in insertion order; default
  `BackwardImpl` reverses
- Aggregated parameter/gradient iteration via `ParameterTensorCount`,
  `ParameterTensor`, `GradientTensor`, `TotalParameterCount`,
  `ZeroGradients`, `AuxLoss`, `ParameterNames`
- Non-trainable sublayers are skipped for gradient zeroing and optimizer
  updates but still participate in forward/backward so gradient can flow

Subclasses: `CAIF_DeviceNetwork`, `CAIF_DeviceTransformerModel`,
`CAIF_DeviceViTModel`, `CAIF_DevicePreNormBlock`,
`CAIF_DeviceTransformerBlock`.

---

### Linear Layers

#### CAIF_DeviceDenseLayer
**File:** `caif_device_dense_layer.h` / `caif_device_dense_layer.cpp`

Standard trainable linear layer: `output = input @ W + b`.

#### CAIF_DeviceFrozenLinear
**File:** `caif_device_frozen_linear.h` / `caif_device_frozen_linear.cpp`

Non-trainable linear layer with multi-dtype weight storage. Stores weights
in any dtype (INT4, INT8, BF16, FP16, FP32) and dequantizes on-the-fly to
FP32 for computation.

- INT4: per-group FP16 scales (group_size=128)
- `cache_fp32` option: when false, avoids caching FP32 dequantized weights
  (critical for large models)
- Direct `cublasGemmEx` path routes through `ctx.ComputeTypeFor(dt)` like
  every other FP32 matmul

#### CAIF_DeviceLoRAAdapter
**File:** `caif_device_lora_adapter.h` / `caif_device_lora_adapter.cpp`

Low-rank adaptation wrapper around any base layer (typically
`CAIF_DeviceFrozenLinear`).

`output = base_layer(input) + (input @ A^T @ B^T) * (alpha / rank)`

Base layer parameters are hidden from the optimizer — only A and B are
exposed through `ParameterTensor()`.

#### CAIF_DeviceLinearHead
**File:** `caif_device_linear_head.h` / `caif_device_linear_head.cpp`

Output projection for language model heads: `logits = hidden @ W`.

---

### Attention Layers

#### CAIF_DeviceMLAttention
**File:** `caif_device_ml_attention.h` / `caif_device_ml_attention.cpp`

Multi-head Latent Attention (MLA). Used by GLM-4.7-Flash and DeepSeek-V2/V3.
Compresses Q and KV through low-rank bottlenecks; splits head dims into
rope and non-rope portions.

- Reads `ctx.PrefixLengths()`; `nullptr` means pure causal
- KV-cache lifecycle as layer-local methods: `EnableKVCache`, `DisableKVCache`,
  `ResetKVCache`, `IsKVCacheEnabled`, `KVCacheLength`

**KV-cache advantage:** Stores `compressed_kv + k_pe` (576 floats/position for
GLM-4.7) vs 10,240 for standard MHA. 18× compression.

#### CAIF_DeviceMultiHeadAttention
**File:** `caif_device_multi_head_attention.h` / `caif_device_multi_head_attention.cpp`

Standard multi-head attention with optional grouped query attention (GQA) and
rotary position embeddings (RoPE).

- Reads `ctx.PrefixLengths()`
- Reads `ctx.PositionBias()` — when set, skips flash attention (requires
  explicit score materialisation)
- KV-cache methods same as MLA

Flash attention kernels available for `head_dim ∈ {32, 64, 80, 96, 128}`
(TF32 tensor core on sm_80+, scalar warp-per-row elsewhere and for
backward). Standard attention fallback for other head_dims (e.g.,
GLM-4.7's 256).

#### CAIF_DeviceCrossAttention
**File:** `caif_device_cross_attention.h` / `caif_device_cross_attention.cpp`

Decoder cross-attention. Decoder hidden state arrives via `ForwardImpl(input, ctx)`;
encoder K/V come from `ctx.EncoderContext()`. Backward accumulates into
`ctx.GradEncoderContext()`. No constructor context parameter — the
relationship is carried by the ctx for that pass only.

#### CAIF_DeviceT5Attention
**File:** `caif_device_t5_attention.h` / `caif_device_t5_attention.cpp`

T5-style attention with additive relative position bias. Reads
`ctx.PositionBias()` and accumulates into `ctx.GradPositionBias()`.
Forces the non-flash code path whenever a position bias is present.

#### CAIF_DeviceRelativePositionBias
**File:** `caif_device_relative_position_bias.h` / `caif_device_relative_position_bias.cpp`

Owns the `[num_heads, num_buckets]` T5 relative-position bias table.
Sibling layer that feeds attention via `ctx.SetPositionBias(&bias_output)`
on the container's `ForwardImpl` entry and reads `ctx.GradPositionBias()`
on backward exit.

---

### Normalization Layers

#### CAIF_DeviceRMSNorm / CAIF_DeviceLayerNorm
**Files:** `caif_device_rmsnorm.{h,cpp}`, `caif_device_layernorm.{h,cpp}`

Root Mean Square Layer Norm (no mean subtraction) and standard Layer Norm.

#### CAIF_DeviceBatchNorm
**File:** `caif_device_batch_norm.h` / `caif_device_batch_norm.cpp`

Batch normalization with running mean/variance. Train/eval behaviour
switches on `ctx.Training()`.

---

### Feed-Forward Networks

#### CAIF_DeviceFFN
**File:** `caif_device_ffn.h` / `caif_device_ffn.cpp`

Feed-forward network with pluggable activation. Supports both standard
(2-weight) and gated (3-weight, SwiGLU-style) configurations — auto-detected
from the activation's `IsGated()` method.

---

### Activation Functions

#### CAIF_DeviceActivation (Abstract)
**File:** `caif_device_activation.h`

Abstract interface. Methods: `Forward`, `Backward`, `IsGated`, `Description`.

#### Pointwise Activations
**File:** `caif_device_pointwise_activations.h`

Unary activations (input → output, same shape):

| Class | Formula |
|-------|---------|
| `CAIF_DeviceReLUActivation` | max(0, x) |
| `CAIF_DeviceSigmoidActivation` | 1 / (1 + exp(-x)) |
| `CAIF_DeviceTanhActivation` | tanh(x) |
| `CAIF_DeviceGELUActivation` | x · 0.5 · (1 + erf(x / √2)) |
| `CAIF_DeviceSwishActivation` | x · sigmoid(x) |
| `CAIF_DeviceLeakyReLUActivation` | x > 0 ? x : α·x |
| `CAIF_DeviceELUActivation` | x > 0 ? x : α·(exp(x) − 1) |
| `CAIF_DeviceLinearActivation` | x (identity) |

#### Gated Activations
**File:** `caif_device_gated_activations.h`

Binary activations (gate, up → output):

| Class | Formula |
|-------|---------|
| `CAIF_DeviceSwiGLU` | swish(gate) · up |
| `CAIF_DeviceGELUGLU` | gelu(gate) · up |
| `CAIF_DeviceSwishGLU` | swish(gate) · up |
| `CAIF_DeviceGLU` | sigmoid(gate) · up |

---

### Regularisation & Shape

| Class | File | Purpose |
|-------|------|---------|
| `CAIF_DeviceDropout` | `caif_device_dropout.h/.cpp` | Inverted dropout; RNG state on ctx |
| `CAIF_DeviceFlatten` | `caif_device_flatten.h/.cpp` | View-only shape flatten |
| `CAIF_DeviceReshape` | `caif_device_reshape.h/.cpp` | View-only shape change |

### Vision Primitives

| Class | File | Purpose |
|-------|------|---------|
| `CAIF_DeviceConv2D` | `caif_device_conv2d.h/.cpp` | cuDNN on device; im2col+BLAS on host |
| `CAIF_DevicePooling2D` | `caif_device_pooling2d.h/.cpp` | Base + Max/Avg subclasses |

---

### Transformer Blocks

#### CAIF_DevicePreNormBlock
**File:** `caif_device_pre_norm_block.h` / `caif_device_pre_norm_block.cpp`
**Base class:** CAIF_DeviceContainer

Generic pre-norm residual block. Accepts an arbitrary number of
(norm, layer) stage pairs and applies:

```
for each (norm, layer) stage:
    x = x + layer(norm(x))
```

A 2-stage block with (RMSNorm, MHA) + (RMSNorm, FFN) is a standard
transformer layer; a 2-stage block with (RMSNorm, MLA) + (RMSNorm, MoE)
is a GLM-4.7 layer. Scales to 3+ stages for future architectures.
`SetNormsTrainable(bool)` toggles norm trainability as composition-time
config (not run-state, so it remains a direct method).

#### CAIF_DeviceTransformerBlock
**File:** `caif_device_transformer_block.h` / `caif_device_transformer_block.cpp`
**Base class:** CAIF_DeviceContainer

Fixed 2-stage pre-norm block: (RMSNorm + MHA) + (RMSNorm + FFN).
Predates `CAIF_DevicePreNormBlock`; still present because some test
paths wire it directly. `CAIF_DevicePreNormBlock` is preferred for
new code.

---

### Mixture of Experts

#### CAIF_DeviceMoERouter
**File:** `caif_device_moe_router.h` / `caif_device_moe_router.cpp`

Expert routing network. Routes each token to top-k experts. Routing types:
TopK (default), ExpertChoice, Soft.

#### CAIF_DeviceMoEExpert
**File:** `caif_device_moe_expert.h` / `caif_device_moe_expert.cpp`

Single expert FFN. Standard or gated (SwiGLU) configuration.

#### CAIF_DeviceMoELayer
**File:** `caif_device_moe_layer.h` / `caif_device_moe_layer.cpp`

Complete MoE layer. Composes router + multiple experts with sparse
activation via GPU dispatch/combine.

- Auxiliary losses: `LastZLoss()`, `LastBalanceLoss()` — summed into
  `AuxLoss()` via container aggregation
- Overflow strategies: only `Drop` is supported on the GPU dispatch path;
  `NoOp` and `Redistribute` throw `CAIF_Exception` explicitly

#### CAIF_MoEComposer
**File:** `caif_moe_composer.h` / `caif_moe_composer.cpp`

Convenience factory for common MoE block configurations. Produces a
`CAIF_DevicePreNormBlock` with attention + MoE stages wired in.

---

### Embedding Layers

| Class | File | Purpose |
|-------|------|---------|
| `CAIF_DeviceTokenEmbedding` | `caif_device_token_embedding.{h,cpp}` | Token ID → vector lookup |
| `CAIF_DevicePositionalEncoding` | `caif_device_positional_encoding.{h,cpp}` | None / Sinusoidal / Learned |
| `CAIF_DevicePatchEmbedding` | `caif_device_patch_embedding.{h,cpp}` | ViT patch embedding |
| `CAIF_DeviceSpectrogramEmbedding` | `caif_device_spectrogram_embedding.{h,cpp}` | Audio spectrogram |
| `CAIF_DeviceTabularEmbedding` | `caif_device_tabular_embedding.{h,cpp}` | Tabular/structured data |

`CAIF_DeviceTokenEmbedding` provides both `ForwardFromIds(token_ids)` (primary,
uint32 IDs) and `Forward(input, ctx)` (one-hot / pre-embedded).

---

### Complete Models

#### CAIF_DeviceTransformerModel
**File:** `caif_device_transformer_model.h` / `caif_device_transformer_model.cpp`
**Base class:** CAIF_DeviceContainer

Complete transformer language model. Sequential composition:

```
TokenEmbedding → PositionalEncoding → N × PreNormBlock → RMSNorm → LinearHead
```

#### CAIF_DeviceViTModel
**File:** `caif_device_vit_model.h` / `caif_device_vit_model.cpp`
**Base class:** CAIF_DeviceContainer

Vision Transformer model: PatchEmbedding → PositionalEncoding →
N × TransformerBlock → classification head.

---

### Device Network

#### CAIF_DeviceNetwork
**File:** `caif_device_network.h` / `caif_device_network.cpp`
**Base class:** CAIF_DeviceContainer

Top-level sequential container. Owns the per-pass `CAIF_RunContext`
construction and Adam-state bookkeeping. Manages the training loop
interaction (forward, loss, backward, optimizer step) across a
heterogeneous stack.

Key methods (unchanged public surface):
- `AddLayer(std::unique_ptr<CAIF_DeviceLayer>)` — composition
- `Forward(input, training)` — constructs ctx with `Pass_e::Forward_e`,
  chains sublayers
- `Backward(grad)` — flips ctx pass via `CAIF_RunContextPassScope`,
  chains sublayers in reverse
- `ZeroGradients()` — respects `SetLayerTrainable` flags
- `ClipGradientNorm(max_norm)` — returns pre-clip norm
- `AuxLoss()` — sums per-layer contributions
- `SaveModel(path)` / `LoadModel(path)` — SafeTensors round-trip

---

### Ops Surface

#### CAIF_Ops namespace
**Public header:** `caif_ops.h`
**Dispatch layer:** `src/caif_ops.cpp`
**Device backend:** `src/caif_ops_device.cpp` (cuBLAS / cuBLAS-Lt / cuDNN / custom kernels)
**Host backend:** `src/caif_ops_host.cpp` (OpenBLAS primary, Eigen fallback, OpenMP loops)

Every op takes a `CAIF_RunContext &ctx`. Every entry point asserts all
operand tensors share a location, then dispatches on that location.
Layer code is written once and runs on either backend.

Categories present:
- Matrix: `MatMul`, `MatMulTransposeA/B`, `BatchedMatMul*`, `MatMulBias`
- Element-wise: `Add`, `Sub`, `Mul`, `Div`, `Scale`, `AddScaled`, `Sqrt`, `AddScalar`
- Reductions: `ReduceSum`, `ReduceMean`, `ReduceMax`, `LogSumExp`
- Activations: ReLU/Sigmoid/Tanh/GELU/Swish/LeakyReLU/ELU/Linear + SwiGLU/GELUGLU/SwishGLU/GLU, forward + backward
- Softmax / causal mask / prefix mask / attention
- RoPE / GQA repeat/reduce / KV-cache append
- Flash attention (device) / standard attention (host)
- Norms: RMSNorm, LayerNorm, forward + backward
- Positional encoding, relative position bias
- MoE: BuildDispatchMap, Dispatch, Combine, top-k, scatter-add, normalise
- Losses: CrossEntropy, MSE, forward + backward
- Optimizer: fused_adam, fused_adam_clipped, fused_sgd_momentum
- Quantisation / cast: FP32↔FP16, FP32↔BF16, FP32↔INT8, FP32↔INT4 (per-group scales)
- ViT helpers: patch extract, cls prepend, embedding lookup
- Convolution (device cuDNN / host im2col+BLAS)

**Dtype dispatch** is driven by `CAIF_OPS_DISPATCH_FLOAT(dt, ...)` internally
so that unsupported dtypes fall through to `THROW_CAIFE` instead of silent
miscompute.

---

### Device Context

#### CAIF_DeviceContext
**File:** `caif_device_context.h` / `caif_device_context.cpp`

Singleton managing cuBLAS and cuDNN handles. Lazy initialization on first
`Instance()` call. Thread-safe.

---

### CUDA Kernels

#### caif_cuda_kernels.h / caif_cuda_kernels.cu
**Custom CUDA kernel declarations and implementations.**

| Category | Kernels |
|----------|---------|
| Activations | ReLU, Sigmoid, Tanh, LeakyReLU, GELU, Swish (forward + backward) |
| Gated activations | SwiGLU, GELU-GLU, Swish-GLU merge kernels |
| Flash attention (TC) | TF32 tensor core forward (sm_80+, head_dim 32/64/80/96/128) |
| Flash attention (scalar) | Warp-per-row forward + backward (all GPUs, head_dim 32/64/80/96/128) |
| Standard attention | Fallback for unsupported head_dims |
| Softmax | Row-wise with causal masking, numerical stability |
| RoPE | Rotary position embedding application |
| RMSNorm / LayerNorm | Forward + backward |
| Quantization | INT4/INT8 dequantization with per-group scales |
| Element-wise | Vectorized add, mul, etc. |
| Reductions | Sum, mean, max |
| MoE | Dispatch / combine / build dispatch map |
| Async copy | `cp_async_f4` (16-byte) and `cp_async_f2` (8-byte) global→shared helpers |

#### Flash Attention — Dual Kernel Architecture

Two flash attention forward paths are provided, selected at runtime based on
GPU compute capability and available shared memory:

**1. Tensor Core Path** (`flash_attention_forward_tc_kernel`, sm_80+)

TF32 tensor core fused kernel using `nvcuda::wmma` (16x16x8 tiles).

Template: `<int D, int BR, int BC>` (head_dim, Q rows per block, KV cols per tile)

Key optimisations:
- **Shared memory bank conflict elimination**: padded strides (D+2, BC+2)
- **cp_async_f2**: 8-byte async global→shared copies
- **Cooperative loading**: all warps participate in Q, K, V loads
- **Persistent O accumulator** in shared memory, rescaled per KV block

**2. Scalar Path** (`flash_attention_forward_kernel`, all GPUs)

Warp-per-row kernel using scalar FMA + warp shuffles. Used when sm < 80,
insufficient shared memory, or for the backward pass on any architecture.

**Runtime dispatch** in `launch_flash_attention_forward()`:
```
if (compute_capability_major >= 8 && smem_optin >= required)
    → tensor core kernel
else
    → scalar kernel
```

---

### Serialization

#### CAIF_SafeTensorsFormat
**File:** `caif_safetensors_format.h` / `caif_safetensors_format.cpp`

HuggingFace-compatible SafeTensors format. Zero external dependencies.

```
[8 bytes: header_size (uint64 LE)]
[header_size bytes: JSON metadata]
[tensor data (aligned)]
```

- `Save(path, tensors, metadata)` / `Load(path)` / `Metadata(path)`
- Supports F32, F16, BF16, I8, I4 dtypes; sharded loading via
  `model.safetensors.index.json`

#### CAIF_WeightMapper
**File:** `caif_weight_mapper.h` / `caif_weight_mapper.cpp`

Maps between HuggingFace and CAIF parameter names. Prefix rules, aliases
for tied weights (lm_head.weight ↔ embed_tokens.weight).

#### CAIF_ModelFormat
**File:** `caif_model_format.h`

Model-format enum and helpers for SafeTensors / checkpoint round-trips.

---

### Optimizers

Located outside the device-layer tree but part of the training surface:

#### CAIF_AdamOptimizer / CAIF_SGDOptimizer

Adam with bias correction and SGD with optional momentum. Initialised
against the layer stack's parameters + gradients + trainability flags;
walk skips non-trainable sublayers. All tensor ops called by the
optimizer run through `CAIF_Ops` with `ctx` threaded.

---

### Loss Functions

#### CAIF_DeviceCrossEntropyLoss
**File:** `caif_device_cross_entropy_loss.h` / `caif_device_cross_entropy_loss.cpp`

Cross-entropy loss computed on device (or host, with a host-backed input).
Supports optional loss masking (zero out loss for specific token positions).

---

### Activation-Aware Initialization

#### CAIF_ActivationAware
**File:** `caif_activation_aware.h` / `caif_activation_aware.cpp`

Provides activation-specific gain values for Kaiming initialization.

| Activation | Gain |
|-----------|-------|
| ReLU | √2 ≈ 1.414 |
| Sigmoid | 1.0 |
| Tanh | 5/3 ≈ 1.667 |
| GELU | ~1.7 |
| Swish | ~1.67 |
| LeakyReLU | √(2 / (1 + α²)) |

Also used downstream for activation-aware quantisation scales.

---

## How It All Fits Together

### Class Hierarchy

```
CAIF_Base (ISE_Object)
├── CAIF_RunContext                    (per-pass state carrier)
├── CAIF_DeviceLayer (abstract)
│   ├── Linear
│   │   ├── CAIF_DeviceDenseLayer
│   │   ├── CAIF_DeviceFrozenLinear
│   │   ├── CAIF_DeviceLoRAAdapter
│   │   └── CAIF_DeviceLinearHead
│   ├── Attention
│   │   ├── CAIF_DeviceMLAttention
│   │   ├── CAIF_DeviceMultiHeadAttention
│   │   ├── CAIF_DeviceCrossAttention
│   │   ├── CAIF_DeviceT5Attention
│   │   └── CAIF_DeviceRelativePositionBias
│   ├── Normalization
│   │   ├── CAIF_DeviceRMSNorm
│   │   ├── CAIF_DeviceLayerNorm
│   │   └── CAIF_DeviceBatchNorm
│   ├── Feed-Forward
│   │   └── CAIF_DeviceFFN
│   ├── Mixture of Experts
│   │   ├── CAIF_DeviceMoERouter
│   │   ├── CAIF_DeviceMoEExpert
│   │   └── CAIF_DeviceMoELayer
│   ├── Embeddings
│   │   ├── CAIF_DeviceTokenEmbedding
│   │   ├── CAIF_DevicePositionalEncoding
│   │   ├── CAIF_DevicePatchEmbedding
│   │   ├── CAIF_DeviceSpectrogramEmbedding
│   │   └── CAIF_DeviceTabularEmbedding
│   ├── Regularisation / Shape
│   │   ├── CAIF_DeviceDropout
│   │   ├── CAIF_DeviceFlatten
│   │   └── CAIF_DeviceReshape
│   ├── Vision
│   │   ├── CAIF_DeviceConv2D
│   │   └── CAIF_DevicePooling2D (Max / Avg subclasses)
│   └── CAIF_DeviceContainer (abstract)
│       ├── CAIF_DevicePreNormBlock
│       ├── CAIF_DeviceTransformerBlock
│       ├── CAIF_DeviceTransformerModel
│       ├── CAIF_DeviceViTModel
│       └── CAIF_DeviceNetwork
├── CAIF_DeviceActivation (abstract)
│   ├── Pointwise: ReLU, Sigmoid, Tanh, GELU, Swish, LeakyReLU, ELU, Linear
│   └── Gated: SwiGLU, GELUGLU, SwishGLU, GLU
└── CAIF_DeviceContext                 (cuBLAS/cuDNN handle singleton)
```

### Typical LoRA Fine-Tuning Assembly

```
1. Load HuggingFace config.json → extract architecture params
2. Create CAIF_CudaStream
3. Create CAIF_DeviceNetwork (uses stream for allocation)

4. For each transformer layer (0..num_layers-1):
   a. Create attention projections as FrozenLinear (storage_dtype from CLI)
   b. Wrap LoRA-targeted projections with LoRAAdapter (rank, alpha from CLI)
   c. Create MHA/MLA with the projection layers
   d. Create FFN with FrozenLinear gate/up/down (wrapped with LoRA if targeted)
   e. Create 2 RMSNorm layers (attn_norm, ffn_norm)
   f. Assemble into PreNormBlock: [(attn_norm, MHA), (ffn_norm, FFN)]
   g. network.AddLayer(std::move(block))

5. network.AddLayer(TokenEmbedding), final RMSNorm, LinearHead
6. Load SafeTensors weights → WeightMapper → assign to FrozenLinear layers
7. Mark embedding + final_norm + lm_head as non-trainable via
   network.SetLayerTrainable(idx, false)
8. Initialize AdamOptimizer with only trainable (LoRA) parameters
9. Training loop: forward(ctx) → loss → backward(ctx) → optimizer step
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
Attention.ForwardImpl(..., ctx)
    → Q = LoRA_q(FrozenLinear_q(input))
    → K = LoRA_k(FrozenLinear_k(input))
    → V = LoRA_v(FrozenLinear_v(input))
    → Apply RoPE to Q, K
    → scores = Q @ K^T / sqrt(head_dim)          (compute type per MatmulMode)
    → Apply causal mask (and ctx.PrefixLengths() if set)
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
FFN.ForwardImpl(..., ctx)
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

31 test executables in `tests/`. Shared infrastructure:

- `caif_test_harness.{h,cpp}` — `ReportResult`, `FloatEqual`, counters, summary
- `caif_gradcheck.{h,cpp}` — single FD entry with auto precision matching
  (scopes `MatmulMode_e::Accuracy_e` for the FD window, restoring on exit)
- `caif_tolerances.h` — canonical per-op-class tolerances
- `caif_cpu_reference/` — 14 extracted CPU reference implementations used as
  parity anchors
- `run_all_tests.cpp` — walks the test directory and executes `test_*` binaries

| Test | What it validates |
|------|-------------------|
| `test_device_tensor` | DeviceTensor creation, transfer, move semantics, Location_e |
| `test_device_attention` | MHA forward/backward correctness |
| `test_device_ml_attention` | MLA forward/backward |
| `test_device_gqa` | Grouped query attention |
| `test_device_rope` | Rotary position embeddings |
| `test_device_rmsnorm` / `test_device_layernorm` | Norm forward/backward |
| `test_device_ffn` / `test_device_geglu` | FFN forward/backward (standard + gated) |
| `test_device_frozen_linear` | Multi-dtype weight storage + dequant |
| `test_device_lora_adapter` | LoRA wrapping + gradient flow |
| `test_device_moe` | MoE routing + expert dispatch |
| `test_device_pre_norm_block` | Generic residual block composition |
| `test_device_transformer_block` | Full transformer block |
| `test_device_transformer_model` | End-to-end transformer |
| `test_device_token_embedding` / `test_device_patch_embedding` | Embedding lookup |
| `test_device_multimodal_embedding` | Tabular / spectrogram embeddings |
| `test_device_cross_entropy` | Loss computation |
| `test_device_kv_cache` | KV cache enable/disable/append |
| `test_device_positional_encoding` | Sinusoidal / learned position codes |
| `test_device_linear_head` | LM head projection |
| `test_device_matmul_dtype` | FP32/FP16/BF16 matmul coverage |
| `test_device_t5_attention` | T5-style additive position bias |
| `test_device_prefix_lm` | Prefix-LM masking via ctx |
| `test_ops_host_parity` | Host-backend ops vs device ops parity |
| `test_device_vision_ports` | Conv2D / Pool2D / BatchNorm / Dropout / Flatten / Reshape ports |
| `test_pluggable_projections` | FrozenLinear + LoRA composition |
| `test_flash_attn_backward_nan` | Regression: flash attn NaN fix |
| `test_safetensors` | SafeTensors read/write |
| `test_transformer_training` | End-to-end training smoke |

Run all tests:
```bash
cd tests && ./run_all_tests
```

---

## Public Documents

| Document | Contents |
|----------|----------|
| `README.md` | Build instructions, dependency setup, link recipes |
| `DESIGN.md` | This document — public architecture overview |
| `CHANGES.md` | Public release notes (versioned changelog) |
| `LICENSE` | Apache 2.0 license text |
