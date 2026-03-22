# CAIF — C++ AI Framework

A C++ GPU-accelerated framework for building and training neural networks.
Supports dense networks, convolutional networks, transformers, vision
transformers, mixture-of-experts, multi-latent attention (MLA), LoRA
fine-tuning, and SafeTensors serialization.

## Features

- **Device-resident tensors** — data lives on GPU, explicit host transfers only
- **Multi-dtype** — fp32, fp16, bf16, int8, int4 with GPU conversion kernels
- **Transformer stack** — multi-head attention, GQA, MLA, RoPE, RMSNorm, LayerNorm
- **MoE** — top-k gating router, expert FFN blocks, load balancing loss
- **LoRA** — low-rank adapters for parameter-efficient fine-tuning
- **SafeTensors** — read/write `.safetensors` format with weight mapping
- **Stream pipelining** — CUDA stream association for operation ordering
- **CPU fallback** — OpenBLAS (or Eigen) backend for non-GPU environments

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
| `CAIF_BUILD_RETRAINER` | `OFF` | Build retrainer fine-tuning tool (requires CUDA) |

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
  retrainer/              # LoRA fine-tuning tool (optional, requires CUDA)
    CMakeLists.txt
    include/retrainer/
    src/
    scripts/              # Python pipeline scripts
    README.md             # Retrainer documentation
  ise_lib/                # Bundled ISE base library (logging, exceptions)
    CMakeLists.txt
    include/ise_lib/
    src/
  LICENSE                 # Apache 2.0
```

## License

Copyright 2026 Eric Malloy

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for
the full text.
