# Copyright 2026 Eric Malloy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===========================================================================
#
#  CAIF (C++ AI Framework) — CMake Build
#
#  Quick start:
#    mkdir build && cd build
#    cmake .. && make -j$(nproc)
#
#  This builds both libcaif.a (static) and libcaif.so (shared) by default.
#
# ---------------------------------------------------------------------------
#
#  BUILD OPTIONS (pass with -D on the cmake command line):
#
#    CAIF_BUILD_CUDA=ON|OFF      Enable CUDA GPU support             (default: ON)
#    CAIF_BUILD_TESTS=ON|OFF     Build test executables               (default: OFF)
#    CAIF_BUILD_SHARED=ON|OFF    Build shared library (.so)           (default: ON)
#    CAIF_BUILD_STATIC=ON|OFF    Build static library (.a)            (default: ON)
#    CAIF_BUILD_RETRAINER=ON|OFF Build retrainer fine-tuning tool     (default: OFF)
#                                Requires CUDA.
#
# ---------------------------------------------------------------------------
#
#  DEPENDENCY PATHS — if your libraries are not in standard system paths,
#  tell cmake where to find them.
#
#  OpenBLAS (default linear algebra backend):
#
#    If headers and lib are under one directory (e.g. /opt/OpenBLAS/include
#    and /opt/OpenBLAS/lib), use OPENBLAS_ROOT:
#
#      cmake .. -DOPENBLAS_ROOT=/opt/OpenBLAS
#
#    If headers and lib are in SEPARATE directories, set each individually:
#
#      -DOPENBLAS_INCLUDE_DIR=/path/to/dir-containing-openblas-folder
#          This directory must contain openblas/cblas.h
#
#      -DOPENBLAS_LIB_DIR=/path/to/dir-containing-libopenblas
#          This directory must contain libopenblas.a or libopenblas.so
#
#      Example:
#        cmake .. \
#          -DOPENBLAS_INCLUDE_DIR=/my/third_party/include \
#          -DOPENBLAS_LIB_DIR=/my/third_party/lib/release/linux
#
#  Eigen3 (fallback if OpenBLAS is not found, header-only):
#
#      -DEIGEN3_INCLUDE_DIR=/path/to/dir-containing-Eigen-folder
#          This directory must contain Eigen/Core
#
#      Example: cmake .. -DEIGEN3_INCLUDE_DIR=/my/third_party/include
#
#  CUDA toolkit (if not in /usr/local/cuda):
#
#      -DCUDAToolkit_ROOT=/path/to/cuda
#      Example: cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
#
#  To disable CUDA entirely (CPU-only build):
#      cmake .. -DCAIF_BUILD_CUDA=OFF
#
#  FULL EXAMPLES:
#
#    System-installed OpenBLAS (auto-detected, nothing to pass):
#      cmake ..
#
#    OpenBLAS in one directory:
#      cmake .. -DOPENBLAS_ROOT=/opt/OpenBLAS
#
#    OpenBLAS with headers and lib in separate directories:
#      cmake .. \
#        -DOPENBLAS_INCLUDE_DIR=/my/third_party/include \
#        -DOPENBLAS_LIB_DIR=/my/third_party/lib
#
#    Eigen-only (no OpenBLAS):
#      cmake .. -DEIGEN3_INCLUDE_DIR=/my/third_party/include
#
#    Everything custom:
#      cmake .. \
#        -DOPENBLAS_INCLUDE_DIR=/my/includes \
#        -DOPENBLAS_LIB_DIR=/my/libs \
#        -DEIGEN3_INCLUDE_DIR=/my/includes \
#        -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
#        -DCAIF_BUILD_TESTS=ON
#
# ===========================================================================

cmake_minimum_required(VERSION 3.18)
project(caif VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Build options (see header comment for documentation) ---
option(CAIF_BUILD_CUDA "Enable CUDA support" ON)
option(CAIF_BUILD_TESTS "Build test executables" OFF)
option(CAIF_BUILD_SHARED "Build shared library" ON)
option(CAIF_BUILD_STATIC "Build static library" ON)
option(CAIF_BUILD_RETRAINER "Build retrainer fine-tuning tool (requires CUDA)" OFF)

# --- ISE dependency (bundled, no user action needed) ---
add_subdirectory(ise_lib)

# ===========================================================================
#  Linear algebra: OpenBLAS (default) with Eigen3 fallback
#
#  CAIF uses OpenBLAS for CPU matrix operations. If OpenBLAS is not found,
#  it falls back to Eigen3 (header-only). At least one must be available.
#
#  User-settable variables (pass with -D on the cmake command line):
#
#    OPENBLAS_ROOT        — single directory with include/ and lib/ subdirs
#    OPENBLAS_INCLUDE_DIR — directory containing openblas/cblas.h
#    OPENBLAS_LIB_DIR     — directory containing libopenblas.a or .so
#    EIGEN3_INCLUDE_DIR   — directory containing Eigen/Core
#
#  OPENBLAS_INCLUDE_DIR and OPENBLAS_LIB_DIR take priority over OPENBLAS_ROOT.
#  This lets you handle the common case where headers and libraries are in
#  completely separate directory trees.
# ===========================================================================

# --- OpenBLAS ---

# Step 1: Find the header (openblas/cblas.h).
# If the user already set OPENBLAS_INCLUDE_DIR on the command line, skip search.
if(NOT OPENBLAS_INCLUDE_DIR)
  find_path(OPENBLAS_INCLUDE_DIR openblas/cblas.h
    HINTS
      ${OPENBLAS_ROOT}/include
      ENV OPENBLAS_ROOT
    PATHS
      /usr/include
      /usr/local/include
      /opt/OpenBLAS/include
  )
endif()

# Step 2: Find the library (libopenblas.a or libopenblas.so).
# If the user already set OPENBLAS_LIB_DIR, look there first.
# Otherwise try OPENBLAS_ROOT/lib, then system FindBLAS, then common paths.
if(OPENBLAS_INCLUDE_DIR)
  # Try user-provided OPENBLAS_LIB_DIR first
  if(OPENBLAS_LIB_DIR)
    find_library(OPENBLAS_LIBRARY openblas
      HINTS ${OPENBLAS_LIB_DIR}
      NO_DEFAULT_PATH
    )
  endif()

  # If not found via LIB_DIR, try OPENBLAS_ROOT/lib
  if(NOT OPENBLAS_LIBRARY AND OPENBLAS_ROOT)
    find_library(OPENBLAS_LIBRARY openblas
      HINTS ${OPENBLAS_ROOT}/lib
      NO_DEFAULT_PATH
    )
  endif()

  # If still not found, try cmake's FindBLAS (checks system paths)
  if(NOT OPENBLAS_LIBRARY)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
      set(OPENBLAS_LIBRARY ${BLAS_LIBRARIES})
    endif()
  endif()

  # Last resort: search common system library paths
  if(NOT OPENBLAS_LIBRARY)
    find_library(OPENBLAS_LIBRARY openblas
      PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /opt/OpenBLAS/lib
    )
  endif()

  if(OPENBLAS_LIBRARY)
    set(CAIF_HAS_OPENBLAS ON)
    set(BLAS_LIBRARIES ${OPENBLAS_LIBRARY})
    message(STATUS "CAIF: Found OpenBLAS headers at ${OPENBLAS_INCLUDE_DIR}")
    message(STATUS "CAIF: Found OpenBLAS library:  ${OPENBLAS_LIBRARY}")
  else()
    set(CAIF_HAS_OPENBLAS OFF)
    message(STATUS "CAIF: OpenBLAS headers found at ${OPENBLAS_INCLUDE_DIR}")
    message(STATUS "CAIF: but library (libopenblas) was NOT found.")
    message(STATUS "  Set -DOPENBLAS_LIB_DIR=/path/to/dir/containing/libopenblas")
  endif()
else()
  set(CAIF_HAS_OPENBLAS OFF)
  message(STATUS "CAIF: OpenBLAS not found — will try Eigen3 fallback")
  if(OPENBLAS_ROOT)
    message(STATUS "  (looked in OPENBLAS_ROOT=${OPENBLAS_ROOT}/include for openblas/cblas.h)")
  endif()
  if(OPENBLAS_INCLUDE_DIR)
    message(STATUS "  (OPENBLAS_INCLUDE_DIR=${OPENBLAS_INCLUDE_DIR} — did not find openblas/cblas.h there)")
  endif()
endif()

# --- Eigen3 (fallback) ---
# User can set -DEIGEN3_INCLUDE_DIR=/path on the cmake command line.
# This should be the directory that directly contains Eigen/Core.
# We first try cmake's built-in FindEigen3, then manually search.

if(NOT EIGEN3_INCLUDE_DIR)
  find_package(Eigen3 QUIET)
  if(Eigen3_FOUND)
    set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})
  else()
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core
      HINTS
        ENV EIGEN3_ROOT
      PATHS
        /usr/include/eigen3
        /usr/local/include/eigen3
        /opt/eigen3/include
    )
  endif()
endif()

if(EIGEN3_INCLUDE_DIR)
  set(CAIF_HAS_EIGEN ON)
  message(STATUS "CAIF: Found Eigen3 headers at ${EIGEN3_INCLUDE_DIR}")
else()
  set(CAIF_HAS_EIGEN OFF)
  if(NOT CAIF_HAS_OPENBLAS)
    message(STATUS "CAIF: Eigen3 not found either")
  endif()
endif()

# --- Require at least one backend ---
if(NOT CAIF_HAS_OPENBLAS AND NOT CAIF_HAS_EIGEN)
  message(FATAL_ERROR
    "CAIF requires OpenBLAS or Eigen3 for CPU linear algebra.\n"
    "\n"
    "Option 1 — Install OpenBLAS (recommended):\n"
    "  Ubuntu/Debian:  sudo apt install libopenblas-dev\n"
    "  Fedora/RHEL:    sudo dnf install openblas-devel\n"
    "  openSUSE:       sudo zypper install openblas-devel\n"
    "  macOS:          brew install openblas\n"
    "  From source:    https://github.com/OpenMathLib/OpenBLAS\n"
    "\n"
    "  If headers and lib are in different places:\n"
    "    cmake .. -DOPENBLAS_INCLUDE_DIR=/path/to/includes -DOPENBLAS_LIB_DIR=/path/to/libs\n"
    "\n"
    "Option 2 — Install Eigen3 (header-only fallback):\n"
    "  Ubuntu/Debian:  sudo apt install libeigen3-dev\n"
    "  Fedora/RHEL:    sudo dnf install eigen3-devel\n"
    "  openSUSE:       sudo zypper install eigen3-devel\n"
    "  macOS:          brew install eigen\n"
    "\n"
    "  If in a non-standard location:\n"
    "    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/dir-containing-Eigen\n"
  )
endif()

if(CAIF_HAS_OPENBLAS)
  message(STATUS "CAIF: Using OpenBLAS as linear algebra backend")
else()
  message(STATUS "CAIF: Using Eigen3 as linear algebra backend (OpenBLAS not available)")
endif()

# ===========================================================================
#  Gather sources
# ===========================================================================
file(GLOB CAIF_CPP_SOURCES src/*.cpp)

# ===========================================================================
#  CUDA (optional)
#
#  Enabled by default. Auto-detected — if nvcc is not found, CUDA is
#  silently disabled. To force off: -DCAIF_BUILD_CUDA=OFF
#
#  If your CUDA toolkit is not in /usr/local/cuda, pass:
#    -DCUDAToolkit_ROOT=/path/to/cuda
# ===========================================================================
if(CAIF_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    file(GLOB CAIF_CUDA_SOURCES src/*.cu)
    add_definitions(-DUSE_CAIF_CUDA)
    message(STATUS "CAIF: CUDA enabled (compiler: ${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING
      "CAIF: CUDA compiler (nvcc) not found — building CPU-only.\n"
      "  To fix: install CUDA toolkit, or pass -DCUDAToolkit_ROOT=/path/to/cuda\n"
      "  To silence: pass -DCAIF_BUILD_CUDA=OFF")
    set(CAIF_BUILD_CUDA OFF)
    set(CAIF_CUDA_SOURCES "")
  endif()
else()
  set(CAIF_CUDA_SOURCES "")
  message(STATUS "CAIF: CUDA disabled by user")
endif()

set(CAIF_ALL_SOURCES ${CAIF_CPP_SOURCES} ${CAIF_CUDA_SOURCES})

# --- Compiler flags ---
add_compile_options(
  -Wextra
  -Wpedantic
  -Werror=return-type
  -fdiagnostics-color=always
)

# --- Helper: configure common properties on a CAIF library target ---
function(caif_configure_target target_name)
  target_include_directories(${target_name}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include/caif
  )
  target_link_libraries(${target_name} PUBLIC ise_lib)
  target_link_libraries(${target_name} PRIVATE gomp)

  # Linear algebra backend
  if(CAIF_HAS_OPENBLAS)
    target_include_directories(${target_name} PRIVATE ${OPENBLAS_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE ${BLAS_LIBRARIES})
  endif()
  if(CAIF_HAS_EIGEN)
    target_include_directories(${target_name} PRIVATE ${EIGEN3_INCLUDE_DIR})
  endif()
endfunction()

# ===========================================================================
#  Library targets
# ===========================================================================

# --- Static library (libcaif.a) ---
if(CAIF_BUILD_STATIC)
  add_library(caif_static STATIC ${CAIF_ALL_SOURCES})
  set_target_properties(caif_static PROPERTIES OUTPUT_NAME caif)
  caif_configure_target(caif_static)

  if(CAIF_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(caif_static
      PRIVATE
        CUDA::cublas_static
        CUDA::cublasLt_static
        CUDA::cudart_static
        CUDA::cudnn
    )
    target_include_directories(caif_static PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(caif_static PROPERTIES CUDA_ARCHITECTURES "120")
  endif()
endif()

# --- Shared library (libcaif.so) ---
if(CAIF_BUILD_SHARED)
  add_library(caif_shared SHARED ${CAIF_ALL_SOURCES})
  set_target_properties(caif_shared PROPERTIES
    OUTPUT_NAME caif
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
  )
  caif_configure_target(caif_shared)

  if(CAIF_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(caif_shared
      PRIVATE
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cudart
        CUDA::cudnn
    )
    target_include_directories(caif_shared PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(caif_shared PROPERTIES CUDA_ARCHITECTURES "120")
  endif()
endif()

# ===========================================================================
#  Tests (off by default — pass -DCAIF_BUILD_TESTS=ON to enable)
# ===========================================================================
if(CAIF_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ===========================================================================
#  Retrainer (off by default — pass -DCAIF_BUILD_RETRAINER=ON to enable)
#  Requires CUDA. Fine-tunes transformer models using LoRA adapters.
# ===========================================================================
if(CAIF_BUILD_RETRAINER)
  if(NOT CAIF_BUILD_CUDA)
    message(FATAL_ERROR
      "CAIF_BUILD_RETRAINER=ON requires CAIF_BUILD_CUDA=ON.\n"
      "The retrainer uses CUDA for GPU-accelerated training.")
  endif()
  add_subdirectory(retrainer)
endif()

# ===========================================================================
#  Install
# ===========================================================================
install(DIRECTORY include/caif DESTINATION include)
if(CAIF_BUILD_STATIC)
  install(TARGETS caif_static ARCHIVE DESTINATION lib)
endif()
if(CAIF_BUILD_SHARED)
  install(TARGETS caif_shared LIBRARY DESTINATION lib)
endif()
