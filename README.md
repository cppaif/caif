# CAIF — C++ AI Framework

A C++23 GPU-accelerated framework for building, fine-tuning, and serving
modern large language models. Supports transformer LMs (dense and MoE),
multi-head and multi-head-latent attention, LoRA, MoE layer surgery with
frozen experts, mixed-precision dtype templating, CPU offload, and
SafeTensors I/O.

See `CHANGES.md` for the re-arch release notes.

## Features

- **Mixed-precision dtype templating** — every trainable device layer
  is `Layer<ComputeT, StorageT>`; mix bf16 / fp16 / fp32 storage with
  fp32 compute per layer. AdamW carries fp32 master weights regardless
  of storage dtype. Frozen weights extend the storage spectrum further:
  `CAIF_DeviceFrozenLinear` supports int8 and int4 quantized storage
  with on-the-fly dequant during matmul, suitable for the pretrained
  base in LoRA / add-MoE fine-tunes.
- **Device-resident tensors** — data lives on GPU, explicit host
  transfers only. Optional `Host_e` location runs the same op surface
  on aligned host memory via OpenBLAS / OpenMP.
- **CPU offload substrate** — pinned host tensors, per-tensor offload
  policy on frozen linears, a block-level scheduler, and an offloaded
  Adam optimizer so multi-billion-parameter add-MoE fine-tunes fit
  in a single consumer GPU.
- **Gradient (activation) checkpointing** on `CAIF_DevicePreNormBlock` —
  opt-in, drops the per-block forward cache and recomputes during
  backward.
- **Transformer stack** — multi-head attention, GQA, FlashAttention,
  optional QK-norm (OLMoE / Olmo2 / Qwen3), partial-rotary RoPE for
  Glm4Moe-style models, RMSNorm, LayerNorm. Optional per-config
  attention features: logit soft-cap (Gemma-2/3), sliding-window
  (Mistral), ALiBi (MPT / BLOOM), and training-time attention dropout.
- **Multi-head Latent Attention (MLA)** — DeepSeek-V2 / GLM-4.7-Flash
  style attention with compressed KV cache and a fused tensor-core
  flash-prefill kernel (O(seq) prefill memory for 16K+ context).
- **MoE** — `SoftmaxTopK_e` and `SigmoidNoauxTc_e` (DeepSeek-V2 /
  GLM-4-MoE) gating, optional DeepSeek-V3 group-limited routing
  (`n_group` / `topk_group`) and aux-loss-free load-balancing bias,
  top-k expert FFNs with the standard `silu(gate) * up` SwiGLU,
  load-balancing and z-loss auxiliaries, `norm_topk_prob` toggle.
- **Whole-model MoE composer** — `CAIF_MoEComposer` assembles a complete
  decoder-only MoE model (embedding + blocks + final norm + head) at
  fp32, fp16, or bf16 from a single config.
- **MoE layer surgery** — wrap pretrained experts as frozen via
  `CAIF_DeviceMoEFrozenExpert` and append new trainable experts to grow
  MoE capacity from a base checkpoint.
- **Mixed-precision loss scaling** — `CAIF_LossScaler` with dynamic
  loss-scale, overflow detection, and unscale-in-place.
- **Large-tensor 64-bit indexing** — element counts and indices are
  64-bit, so large-vocab logits at long context (>2.1B elements)
  compute correctly.
- **Optimizers** — Adam, AdamW (with master-weights), SGD, Momentum,
  RMSprop, AdaGrad, plus an offloaded-Adam variant for offload mode.
  All share `CAIF_DeviceOptimizer`.
- **LoRA** — low-rank adapters for parameter-efficient fine-tuning.
- **SafeTensors** — read/write `.safetensors` format with weight
  mapping; both sharded and single-file checkpoints supported.
- **Stream pipelining** — CUDA stream association for operation ordering.
- **CPU fallback** — OpenBLAS (or Eigen) backend for non-GPU environments.

## Requirements

- **C++23** compiler (GCC 13+, Clang 17+)
- **CMake** 3.18+
- **OpenBLAS** (recommended) or **Eigen3** (header-only fallback)
- **CUDA toolkit** 12.0+ (optional, for GPU support)
- **cuDNN** (optional, required if CUDA is enabled)

### Installing dependencies

**Ubuntu / Debian:**
```bash
sudo apt install build-essential cmake libopenblas-dev
# Optional: sudo apt install libeigen3-dev   (fallback if no OpenBLAS)
# Optional: install CUDA toolkit from https://developer.nvidia.com/cuda-downloads
```

**Fedora / RHEL:**
```bash
sudo dnf install gcc-c++ cmake openblas-devel
```

**openSUSE:**
```bash
sudo zypper install gcc-c++ cmake openblas-devel
```

**macOS:**
```bash
brew install cmake openblas
# You may need: export OPENBLAS_ROOT=$(brew --prefix openblas)
```

## Building

### Quick start

```bash
git clone https://github.com/your-repo/caif.git
cd caif
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This builds both `libcaif.a` (static) and `libcaif.so` (shared).

### Build options

Pass these with `-D` on the cmake command line:

| Option | Default | Description |
|--------|---------|-------------|
| `CAIF_BUILD_CUDA` | `ON` | Enable CUDA GPU support (auto-detected) |
| `CAIF_BUILD_TESTS` | `OFF` | Build test executables |
| `CAIF_BUILD_SHARED` | `ON` | Build shared library (.so) |
| `CAIF_BUILD_STATIC` | `ON` | Build static library (.a) |
| `CAIF_BUILD_EXAMPLES` | `OFF` | Build example programs in `examples/` (requires CUDA) |

### Custom dependency paths

If your libraries are installed in non-standard locations, tell cmake
where to find them.

**OpenBLAS** — if headers and lib are under one directory:
```bash
# Expects OPENBLAS_ROOT/include/openblas/cblas.h and OPENBLAS_ROOT/lib/libopenblas.so
cmake .. -DOPENBLAS_ROOT=/opt/OpenBLAS
```

**OpenBLAS** — if headers and lib are in separate directories:
```bash
# OPENBLAS_INCLUDE_DIR = directory containing openblas/cblas.h
# OPENBLAS_LIB_DIR     = directory containing libopenblas.a or .so
cmake .. \
  -DOPENBLAS_INCLUDE_DIR=/my/third_party/include \
  -DOPENBLAS_LIB_DIR=/my/third_party/lib/release/linux
```

**Eigen3** (header-only):
```bash
# EIGEN3_INCLUDE_DIR = directory containing Eigen/Core
cmake .. -DEIGEN3_INCLUDE_DIR=/my/third_party/include
```

**CUDA toolkit:**
```bash
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
```

**Combine them:**
```bash
cmake .. \
  -DOPENBLAS_INCLUDE_DIR=/my/includes \
  -DOPENBLAS_LIB_DIR=/my/libs \
  -DEIGEN3_INCLUDE_DIR=/my/includes \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCAIF_BUILD_TESTS=ON
```

### CPU-only build (no CUDA)

```bash
cmake .. -DCAIF_BUILD_CUDA=OFF
make -j$(nproc)
```

### Static-only or shared-only

```bash
# Static only
cmake .. -DCAIF_BUILD_SHARED=OFF

# Shared only
cmake .. -DCAIF_BUILD_STATIC=OFF
```

### Running tests

```bash
cmake .. -DCAIF_BUILD_TESTS=ON
make -j$(nproc)
./tests/run_all_tests
```

## Using CAIF in your project

### CMake (add_subdirectory)

```cmake
add_subdirectory(caif)
target_link_libraries(your_target PRIVATE caif_static)
```

### CMake (find_package after install)

```bash
cd build && sudo make install
```

Then in your CMakeLists.txt:
```cmake
find_library(CAIF_LIB caif)
target_link_libraries(your_target PRIVATE ${CAIF_LIB})
target_include_directories(your_target PRIVATE /usr/local/include)
```

### Manual linking

```bash
g++ -std=c++23 your_code.cpp -I/path/to/caif/include -L/path/to/caif/build -lcaif -lise_lib -lopenblas -fopenmp
```

## Project structure

```
caif/
  CMakeLists.txt          # Main build file
  include/caif/           # Public headers
  src/                    # Implementation (.cpp, .cu)
  tests/                  # Test source files
  examples/               # Small demo programs
  ise_lib/                # Bundled ISE base library (logging, exceptions)
    CMakeLists.txt
    include/ise_lib/
    src/
  DESIGN.md               # Architecture overview
  CHANGES.md              # Release notes
  LICENSE                 # Apache 2.0
```

## License

Copyright 2026 Eric Malloy

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for
the full text.
