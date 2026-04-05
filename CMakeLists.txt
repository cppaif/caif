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
#  Quick start (system-installed deps, CUDA in /usr/local/cuda):
#
#    mkdir build && cd build
#    cmake .. && make -j$(nproc)
#
#  This builds both libcaif.a (static) and libcaif.so (shared) by default.
#
# ---------------------------------------------------------------------------
#  BUILD OPTIONS  (pass with -D on the cmake command line)
# ---------------------------------------------------------------------------
#
#    CAIF_BUILD_CUDA=ON|OFF      Enable CUDA GPU support              (default: ON)
#    CAIF_BUILD_TESTS=ON|OFF     Build test executables                (default: OFF)
#    CAIF_BUILD_SHARED=ON|OFF    Build shared library (.so)            (default: ON)
#    CAIF_BUILD_STATIC=ON|OFF    Build static library (.a)             (default: ON)
#    CAIF_BUILD_EXAMPLES=ON|OFF  Build example programs (requires CUDA) (default: OFF)
#
# ---------------------------------------------------------------------------
#  DEPENDENCY PATHS  — point cmake at your libraries
# ---------------------------------------------------------------------------
#
#  All dependency paths are optional.  If your libraries are in standard
#  system paths they will be auto-detected.  Pass these -D variables only
#  when auto-detection fails or you want to override with custom builds.
#
#  LINEAR ALGEBRA  (OpenBLAS required, Eigen3 optional supplement)
#  ~~~~~~~~~~~~~~~~
#
#  OpenBLAS (required):
#
#    Single root (has include/ and lib/ subdirs):
#      -DOPENBLAS_ROOT=/opt/OpenBLAS
#
#    Separate header and library directories:
#      -DOPENBLAS_INCLUDE_DIR=/path/to/dir-containing-openblas-folder
#          Must contain openblas/cblas.h
#      -DOPENBLAS_LIB_DIR=/path/to/dir-containing-libopenblas
#          Must contain libopenblas.a or libopenblas.so
#
#  Eigen3 (optional supplement, header-only):
#
#      -DEIGEN3_INCLUDE_DIR=/path/to/dir-containing-Eigen-folder
#          Must contain Eigen/Core
#
#  CUDA  (optional, enabled by default)
#  ~~~~~~
#
#  NVCC compiler — searched in this priority order (highest wins):
#
#    1. -DCMAKE_CUDA_COMPILER=/path/to/nvcc         (cmake variable)
#    2. NVCC_PATH environment variable               (export before cmake)
#    3. -DCUDAToolkit_ROOT=/path/to/cuda             (cmake variable)
#    4. Standard system locations                    (fallback)
#       /usr/local/cuda/bin/nvcc
#       /opt/cuda/bin/nvcc
#       /usr/bin/nvcc
#
#  CUDA header directory:
#
#    IMPORTANT:  CAIF headers include CUDA with a subdirectory prefix:
#
#      #include "cuda/cuda_runtime_api.h"
#      #include "cuda/cublas_v2.h"
#      #include "cudnn/cudnn.h"
#
#    This means CUDA_INCLUDE_DIR must point to a directory that CONTAINS
#    cuda/ and cudnn/ as subdirectories:
#
#      /my/includes/          <-- set CUDA_INCLUDE_DIR to this
#        cuda/
#          cuda_runtime_api.h
#          cublas_v2.h
#          cublasLt.h
#        cudnn/
#          cudnn.h
#
#      -DCUDA_INCLUDE_DIR=/my/includes
#
#    If your cuDNN headers are in a different tree from CUDA headers,
#    set CUDNN_INCLUDE_DIR separately:
#
#      -DCUDNN_INCLUDE_DIR=/other/path
#
#    If neither is set, falls back to CUDAToolkit_INCLUDE_DIRS from
#    cmake's find_package(CUDAToolkit).
#
#  CUDA libraries:
#
#      -DCUDA_LIB_DIR=/path/to/dir-containing-libcublas-etc
#          Prefers static libs (cublas_static) with shared fallback.
#          Also searches for cudadevrt, culibos.
#
#      -DCUDNN_LIB_DIR=/path/to/dir-containing-libcudnn
#          Only needed if cuDNN library is not in CUDA_LIB_DIR
#          or standard system paths.
#
#  Disable CUDA entirely (CPU-only build):
#
#      -DCAIF_BUILD_CUDA=OFF
#
# ---------------------------------------------------------------------------
#  EXAMPLES
# ---------------------------------------------------------------------------
#
#  System-installed OpenBLAS + system CUDA (auto-detected):
#
#    cmake ..
#
#  OpenBLAS in one root directory:
#
#    cmake .. -DOPENBLAS_ROOT=/opt/OpenBLAS
#
#  OpenBLAS with headers and lib in separate trees:
#
#    cmake .. \
#      -DOPENBLAS_INCLUDE_DIR=/my/third_party/include \
#      -DOPENBLAS_LIB_DIR=/my/third_party/lib
#
#  OpenBLAS + Eigen3, no CUDA:
#
#    cmake .. \
#      -DOPENBLAS_INCLUDE_DIR=/my/third_party/include \
#      -DOPENBLAS_LIB_DIR=/my/third_party/lib \
#      -DEIGEN3_INCLUDE_DIR=/my/third_party/include \
#      -DCAIF_BUILD_CUDA=OFF
#
#  Everything custom (BLAS, Eigen, CUDA headers + libs, tests):
#
#    cmake .. \
#      -DOPENBLAS_INCLUDE_DIR=/my/includes \
#      -DOPENBLAS_LIB_DIR=/my/libs \
#      -DEIGEN3_INCLUDE_DIR=/my/includes \
#      -DCUDA_INCLUDE_DIR=/my/includes \
#      -DCUDA_LIB_DIR=/my/libs/cuda \
#      -DCAIF_BUILD_TESTS=ON
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
option(CAIF_BUILD_EXAMPLES "Build example programs (requires CUDA)" OFF)

# --- ISE dependency (bundled, no user action needed) ---
add_subdirectory(ise_lib)

# ===========================================================================
#  Linear algebra: OpenBLAS (required) + Eigen3 (optional supplement)
#
#  CAIF uses OpenBLAS for CPU matrix operations (cblas_sgemm, etc.).
#  Eigen3 provides supplemental CPU math when available.
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
  message(STATUS "CAIF: OpenBLAS not found")
  if(OPENBLAS_ROOT)
    message(STATUS "  (looked in OPENBLAS_ROOT=${OPENBLAS_ROOT}/include for openblas/cblas.h)")
  endif()
  if(OPENBLAS_INCLUDE_DIR)
    message(STATUS "  (OPENBLAS_INCLUDE_DIR=${OPENBLAS_INCLUDE_DIR} — did not find openblas/cblas.h there)")
  endif()
endif()

# --- Eigen3 (optional supplement) ---
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
endif()

# --- Require OpenBLAS ---
# The BLAS backend (cblas_sgemm, etc.) is used unconditionally by the
# framework's matrix operations.  Eigen3 is an optional supplemental
# backend for additional CPU math, but does NOT replace OpenBLAS.
if(NOT CAIF_HAS_OPENBLAS)
  message(FATAL_ERROR
    "CAIF requires OpenBLAS for CPU linear algebra.\n"
    "\n"
    "Install OpenBLAS:\n"
    "  Ubuntu/Debian:  sudo apt install libopenblas-dev\n"
    "  Fedora/RHEL:    sudo dnf install openblas-devel\n"
    "  openSUSE:       sudo zypper install openblas-devel\n"
    "  macOS:          brew install openblas\n"
    "  From source:    https://github.com/OpenMathLib/OpenBLAS\n"
    "\n"
    "  If headers and lib are in different places:\n"
    "    cmake .. -DOPENBLAS_INCLUDE_DIR=/path/to/includes -DOPENBLAS_LIB_DIR=/path/to/libs\n"
    "\n"
    "Eigen3 is an optional supplement (not a replacement):\n"
    "  Ubuntu/Debian:  sudo apt install libeigen3-dev\n"
    "  cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/dir-containing-Eigen\n"
  )
endif()

message(STATUS "CAIF: Using OpenBLAS as primary linear algebra backend")
if(CAIF_HAS_EIGEN)
  message(STATUS "CAIF: Eigen3 also available as supplemental backend")
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
  # GCC 15+ rsqrt/rsqrtf declarations conflict with CUDA math_functions.h.
  # Work around with POSIX feature macros and --allow-unsupported-compiler.
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "15")
    set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -U_GNU_SOURCE -D_POSIX_C_SOURCE=200809L"
    )
  endif()

  # --- Locate nvcc ---
  #
  # check_language(CUDA) only searches PATH, which misses common installs
  # where the CUDA toolkit bin/ is not in PATH.  We search manually first.
  #
  # Priority (highest wins):
  #   1. -DCMAKE_CUDA_COMPILER=/path/to/nvcc   (user cmake variable)
  #   2. NVCC_PATH environment variable          (user env)
  #   3. -DCUDAToolkit_ROOT=/path/to/cuda        (user cmake variable)
  #   4. Standard system locations               (fallback)
  if(NOT CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{NVCC_PATH} AND EXISTS "$ENV{NVCC_PATH}")
      set(CMAKE_CUDA_COMPILER "$ENV{NVCC_PATH}")
      message(STATUS "CAIF: Found nvcc via NVCC_PATH env: ${CMAKE_CUDA_COMPILER}")
    elseif(CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
      message(STATUS "CAIF: Found nvcc via CUDAToolkit_ROOT: ${CMAKE_CUDA_COMPILER}")
    else()
      foreach(_nvcc_path
        /usr/local/cuda/bin/nvcc
        /opt/cuda/bin/nvcc
        /usr/bin/nvcc
      )
        if(EXISTS "${_nvcc_path}")
          set(CMAKE_CUDA_COMPILER "${_nvcc_path}")
          message(STATUS "CAIF: Found nvcc at ${CMAKE_CUDA_COMPILER}")
          break()
        endif()
      endforeach()
    endif()
  endif()

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
      "  To fix, try one of:\n"
      "    -DCMAKE_CUDA_COMPILER=/path/to/nvcc\n"
      "    export NVCC_PATH=/path/to/nvcc\n"
      "    -DCUDAToolkit_ROOT=/path/to/cuda\n"
      "  To silence: -DCAIF_BUILD_CUDA=OFF")
    set(CAIF_BUILD_CUDA OFF)
    set(CAIF_CUDA_SOURCES "")
  endif()
else()
  set(CAIF_CUDA_SOURCES "")
  message(STATUS "CAIF: CUDA disabled by user")
endif()

set(CAIF_ALL_SOURCES ${CAIF_CPP_SOURCES} ${CAIF_CUDA_SOURCES})

# --- Compiler flags (CXX only — nvcc does not understand these) ---
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>
  $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>
)

# --- Helper: configure common properties on a CAIF library target ---
#
# Link deps are PUBLIC so that consumers of the static library (tests,
# applications) automatically get the transitive dependencies they need.
function(caif_configure_target target_name)
  target_include_directories(${target_name}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include/caif
  )
  target_link_libraries(${target_name} PUBLIC ise_lib)
  target_link_libraries(${target_name} PUBLIC gomp)
  target_link_libraries(${target_name} PUBLIC pthread m dl)

  # Linear algebra backend
  if(CAIF_HAS_OPENBLAS)
    target_include_directories(${target_name} PRIVATE ${OPENBLAS_INCLUDE_DIR})
    target_link_libraries(${target_name} PUBLIC ${BLAS_LIBRARIES})
  endif()
  if(CAIF_HAS_EIGEN)
    target_include_directories(${target_name} PRIVATE ${EIGEN3_INCLUDE_DIR})
  endif()
endfunction()

# ===========================================================================
#  CUDA headers and libraries (cuBLAS, cuDNN)
#
#  CAIF headers include CUDA headers with a subdirectory prefix:
#
#    #include "cuda/cuda_runtime_api.h"
#    #include "cuda/cublas_v2.h"
#    #include "cudnn/cudnn.h"
#
#  This means CUDA_INCLUDE_DIR must point to a directory that CONTAINS
#  cuda/ and cudnn/ as subdirectories.  For example:
#
#    /my/includes/
#      cuda/
#        cuda_runtime_api.h
#        cublas_v2.h
#        cublasLt.h
#      cudnn/
#        cudnn.h
#
#    cmake .. -DCUDA_INCLUDE_DIR=/my/includes
#
#  If your CUDA and cuDNN headers are in separate trees, set both:
#
#    -DCUDA_INCLUDE_DIR=/path/to/parent-of-cuda-subdir
#    -DCUDNN_INCLUDE_DIR=/path/to/parent-of-cudnn-subdir
#
#  If neither is set, the build falls back to CUDAToolkit_INCLUDE_DIRS
#  from cmake's find_package(CUDAToolkit).  This works on systems where
#  the CUDA toolkit uses the cuda/ subdirectory layout.
#
#  CUDA libraries:
#
#    -DCUDA_LIB_DIR=/path/to/cublas-cudart-etc
#    -DCUDNN_LIB_DIR=/path/to/cudnn-lib
#
#  When CUDA_LIB_DIR is set, we link against whatever libs are found
#  there (.a or .so) instead of using the cmake CUDAToolkit targets.
# ===========================================================================
if(CAIF_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)

  # --- CUDA include directory ---
  # Must contain cuda/cuda_runtime_api.h (see comment block above).
  if(NOT CUDA_INCLUDE_DIR)
    set(CUDA_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CAIF: CUDA_INCLUDE_DIR not set, using CUDAToolkit: ${CUDA_INCLUDE_DIR}")
  else()
    message(STATUS "CAIF: Using CUDA headers from ${CUDA_INCLUDE_DIR}")
  endif()

  # --- cuDNN include directory ---
  # Must contain cudnn/cudnn.h OR cudnn.h directly.
  # If CUDA_INCLUDE_DIR already has a cudnn/ subdirectory, CUDNN_INCLUDE_DIR
  # is not needed.
  if(NOT CUDNN_INCLUDE_DIR)
    # Check if cudnn.h is inside CUDA_INCLUDE_DIR/cudnn/
    find_path(_cudnn_in_cuda_dir cudnn/cudnn.h HINTS ${CUDA_INCLUDE_DIR} NO_DEFAULT_PATH)
    if(_cudnn_in_cuda_dir)
      set(CUDNN_INCLUDE_DIR "${_cudnn_in_cuda_dir}")
      message(STATUS "CAIF: Found cuDNN headers inside CUDA_INCLUDE_DIR: ${CUDNN_INCLUDE_DIR}")
    else()
      # Search system paths
      find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        PATHS /usr/include /usr/local/include /usr/local/cuda/include
      )
    endif()
  else()
    message(STATUS "CAIF: Using cuDNN headers from ${CUDNN_INCLUDE_DIR}")
  endif()

  # --- cuDNN library ---
  if(NOT TARGET CUDA::cudnn)
    find_library(CUDNN_LIBRARY cudnn
      HINTS ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR}
      PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/cuda/lib64
    )
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
      message(STATUS "CAIF: Found cuDNN library: ${CUDNN_LIBRARY}")
    elseif(CUDNN_INCLUDE_DIR)
      set(CUDNN_LIBRARY "cudnn")
      message(STATUS "CAIF: cuDNN headers at ${CUDNN_INCLUDE_DIR}, linking -lcudnn")
    else()
      message(FATAL_ERROR
        "cuDNN not found. Install cuDNN or pass:\n"
        "  -DCUDNN_INCLUDE_DIR=/path/to/parent-of-cudnn-subdir\n"
        "  -DCUDNN_LIB_DIR=/path/to/cudnn-lib (optional)")
    endif()
  endif()

  # --- CUDA libraries (cuBLAS, cuBLAS-Lt, CUDA runtime, etc.) ---
  # If CUDA_LIB_DIR is set, find libs there (any format: .a or .so).
  # Prefers static libs (cublas_static) with shared fallback (cublas).
  if(CUDA_LIB_DIR)
    find_library(CAIF_CUBLAS cublas_static cublas HINTS ${CUDA_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CAIF_CUBLASLT cublasLt_static cublasLt HINTS ${CUDA_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CAIF_CUDART cudart_static cudart HINTS ${CUDA_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CAIF_CUDADEVRT cudadevrt HINTS ${CUDA_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CAIF_CULIBOS culibos HINTS ${CUDA_LIB_DIR} NO_DEFAULT_PATH)
    message(STATUS "CAIF: Using CUDA libs from ${CUDA_LIB_DIR}")
    message(STATUS "  cublas:    ${CAIF_CUBLAS}")
    message(STATUS "  cublasLt:  ${CAIF_CUBLASLT}")
    message(STATUS "  cudart:    ${CAIF_CUDART}")
    message(STATUS "  cudadevrt: ${CAIF_CUDADEVRT}")
    message(STATUS "  culibos:   ${CAIF_CULIBOS}")
    set(CAIF_USE_CUSTOM_CUDA_LIBS ON)
  else()
    set(CAIF_USE_CUSTOM_CUDA_LIBS OFF)
  endif()
endif()

# ===========================================================================
#  Library targets
# ===========================================================================

# --- Helper: link CUDA deps to a target ---
#
# CUDA include directories are PUBLIC because CAIF's public headers
# (caif_cuda_event.h, caif_device_context.h, etc.) include CUDA headers.
# Any target linking against CAIF needs these include paths too.
function(caif_link_cuda target_name)
  # Two levels of include paths are needed:
  #
  #   1. CUDA_INCLUDE_DIR itself — resolves CAIF's prefixed includes:
  #        #include "cuda/cuda_runtime_api.h"
  #        #include "cudnn/cudnn.h"
  #
  #   2. The cuda/ and cudnn/ subdirectories — resolves unprefixed
  #      includes used internally by CUDA's own headers:
  #        #include <cuda_fp16.h>   (from cublas_api.h)
  #        #include <cudnn_ops.h>   (from cudnn.h)
  target_include_directories(${target_name} PUBLIC ${CUDA_INCLUDE_DIR})
  if(IS_DIRECTORY "${CUDA_INCLUDE_DIR}/cuda")
    target_include_directories(${target_name} PUBLIC "${CUDA_INCLUDE_DIR}/cuda")
  endif()
  if(CUDNN_INCLUDE_DIR AND NOT "${CUDNN_INCLUDE_DIR}" STREQUAL "${CUDA_INCLUDE_DIR}")
    target_include_directories(${target_name} PUBLIC ${CUDNN_INCLUDE_DIR})
  endif()
  if(IS_DIRECTORY "${CUDNN_INCLUDE_DIR}/cudnn")
    target_include_directories(${target_name} PUBLIC "${CUDNN_INCLUDE_DIR}/cudnn")
  elseif(CUDNN_INCLUDE_DIR AND IS_DIRECTORY "${CUDA_INCLUDE_DIR}/cudnn")
    target_include_directories(${target_name} PUBLIC "${CUDA_INCLUDE_DIR}/cudnn")
  endif()

  if(CAIF_USE_CUSTOM_CUDA_LIBS)
    target_link_libraries(${target_name} PUBLIC
      ${CAIF_CUBLAS}
      ${CAIF_CUBLASLT}
      ${CAIF_CUDART})
    if(CAIF_CUDADEVRT)
      target_link_libraries(${target_name} PUBLIC ${CAIF_CUDADEVRT})
    endif()
    if(CAIF_CULIBOS)
      target_link_libraries(${target_name} PUBLIC ${CAIF_CULIBOS})
    endif()
  else()
    target_link_libraries(${target_name} PUBLIC
      CUDA::cublas
      CUDA::cublasLt
      CUDA::cudart)
  endif()

  # cuDNN
  if(TARGET CUDA::cudnn)
    target_link_libraries(${target_name} PUBLIC CUDA::cudnn)
  else()
    target_link_libraries(${target_name} PUBLIC ${CUDNN_LIBRARY})
  endif()

  # System libraries required when statically linking CUDA runtime
  target_link_libraries(${target_name} PUBLIC rt dl pthread)

  set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "75;80;89;90;120")
endfunction()

# --- Static library (libcaif.a) ---
if(CAIF_BUILD_STATIC)
  add_library(caif_static STATIC ${CAIF_ALL_SOURCES})
  set_target_properties(caif_static PROPERTIES OUTPUT_NAME caif)
  caif_configure_target(caif_static)
  if(CAIF_BUILD_CUDA)
    caif_link_cuda(caif_static)
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
    caif_link_cuda(caif_shared)
  endif()
endif()

# ===========================================================================
#  Tests (off by default — pass -DCAIF_BUILD_TESTS=ON to enable)
# ===========================================================================
if(CAIF_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ===========================================================================
#  Examples (off by default — pass -DCAIF_BUILD_EXAMPLES=ON to enable)
#  Requires CUDA.
# ===========================================================================
if(CAIF_BUILD_EXAMPLES)
  if(NOT CAIF_BUILD_CUDA)
    message(FATAL_ERROR
      "CAIF_BUILD_EXAMPLES=ON requires CAIF_BUILD_CUDA=ON.\n"
      "The examples use CUDA for GPU-accelerated operations.")
  endif()
  add_subdirectory(examples)
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
