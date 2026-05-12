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

#include "caif_device_context.h"
#include "caif_exception.h"
#include "caif_settings.h"

#include <cstdio>
#include <cstdlib>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#endif

namespace instance
{

CAIF_DeviceContext::CAIF_DeviceContext():
#ifdef USE_CAIF_CUDA
                                        _cublas_handle(nullptr),
                                        _cublaslt_handle(nullptr),
                                        _cudnn_handle(nullptr),
                                        _cublaslt_workspace(nullptr),
                                        _last_cublas_stream(nullptr),
#endif
                                        _cublaslt_workspace_size(0),
                                        _initialized(false),
                                        _device_id(0)
{
}

// Map (cc_major, cc_minor) to the workspace-size tier that unlocks the
// fastest cuBLAS-Lt algo set for that architecture. The mapping comes
// from NVIDIA's per-arch algo coverage notes plus empirical probing in
// the 2026-04-19/20 MatMul investigation (see CHANGES.md).
//
//   sm_80 (A100)             -> 32 MB (large)
//   sm_86 (RTX 3090 / A6000) -> 16 MB (medium)
//   sm_89 (RTX 4090 / L40)   -> 32 MB (large)
//   sm_90 (H100)             -> 32 MB (large)
//   sm_120 (RTX 5090)        -> 32 MB (large)
//   pre-Ampere (<sm_80)      ->  4 MB (small)
//   any unknown future arch  -> 32 MB (large) -- err toward coverage
size_t CAIF_DeviceContext::AutoSizeCublasLtWorkspace(const int cc_major,
                                                    const int cc_minor)const
{
  if(cc_major<8)
  {
    return _cublaslt_workspace_bytes_small;
  }
  if(cc_major==8&&cc_minor==6)
  {
    return _cublaslt_workspace_bytes_medium;
  }
  return _cublaslt_workspace_bytes_large;
}

size_t CAIF_DeviceContext::ResolveCublasLtWorkspaceSize(const int cc_major,
                                                        const int cc_minor,
                                                        const size_t free_vram_bytes,
                                                        bool &out_is_override)const
{
  const size_t override_bytes=CAIF_Settings::CublasLtWorkspaceBytes();
  if(override_bytes==0)
  {
    out_is_override=false;
    return AutoSizeCublasLtWorkspace(cc_major,cc_minor);
  }
  out_is_override=true;
  const size_t max_allowed=
                  free_vram_bytes/_cublaslt_workspace_free_vram_divisor;
  if(override_bytes>max_allowed)
  {
    THROW_CAIFE("CAIF_Settings::CublasLtWorkspaceBytes override exceeds "
                "free-VRAM safety bound");
  }
  return override_bytes;
}

CAIF_DeviceContext::~CAIF_DeviceContext()
{
  Cleanup();
}

CAIF_DeviceContext &CAIF_DeviceContext::Instance()
{
  static CAIF_DeviceContext instance;
  return instance;
}

void CAIF_DeviceContext::Initialize()
{
  if(_initialized==true)
  {
    return;
  }

#ifdef USE_CAIF_CUDA
  // Get device count
  int device_count=0;
  cudaError_t device_status=cudaGetDeviceCount(&device_count);
  if(device_status!=cudaSuccess)
  {
    THROW_CAIFE("cudaGetDeviceCount failed");
  }
  if(device_count==0)
  {
    THROW_CAIFE("No CUDA devices available");
  }

  // Set device
  if(_device_id>=device_count)
  {
    _device_id=0;
  }
  cudaError_t set_status=cudaSetDevice(_device_id);
  if(set_status!=cudaSuccess)
  {
    THROW_CAIFE("cudaSetDevice failed");
  }

  // Force cuBLAS to use a deterministic workspace pool. Must be set before
  // the first cuBLAS handle is created so the heuristic cache keys are
  // stable across runs (otherwise cuBLAS can pick different near-optimal
  // algos at dim=512..1024). See CHANGES.md — determinism step 4.
  setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8",1);

  // Create cuBLAS handle
  cublasStatus_t cublas_status=cublasCreate(&_cublas_handle);
  if(cublas_status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("Failed to create cuBLAS handle");
  }

  // Disable TF32 mode for full FP32 precision (TF32 is default on Ampere+ GPUs)
  // This ensures numerical consistency with CPU BLAS
  cublasSetMathMode(_cublas_handle,CUBLAS_DEFAULT_MATH);

  // Forbid atomic reductions inside cuBLAS kernels. atomicAdd accumulation
  // is order-dependent and produces run-to-run nondeterministic sums; the
  // non-atomic paths are deterministic (often slower on bwd split-K).
  // See CHANGES.md — determinism step 5.
  cublasSetAtomicsMode(_cublas_handle,CUBLAS_ATOMICS_NOT_ALLOWED);

  // Create cublasLt handle
  cublasStatus_t cublaslt_status=cublasLtCreate(&_cublaslt_handle);
  if(cublaslt_status!=CUBLAS_STATUS_SUCCESS)
  {
    cublasDestroy(_cublas_handle);
    _cublas_handle=nullptr;
    THROW_CAIFE("Failed to create cublasLt handle");
  }

  size_t free_vram=0;
  size_t total_vram=0;
  cudaError_t meminfo_status=cudaMemGetInfo(&free_vram,&total_vram);
  if(meminfo_status!=cudaSuccess)
  {
    cublasLtDestroy(_cublaslt_handle);
    _cublaslt_handle=nullptr;
    cublasDestroy(_cublas_handle);
    _cublas_handle=nullptr;
    THROW_CAIFE("cudaMemGetInfo failed");
  }
  CAIF_DeviceProperties &props=CAIF_DeviceProperties::ForDevice(_device_id);
  bool ws_is_override=false;
  _cublaslt_workspace_size=ResolveCublasLtWorkspaceSize(
                                    props.ComputeCapabilityMajor(),
                                    props.ComputeCapabilityMinor(),
                                    free_vram,
                                    ws_is_override);
  cudaError_t ws_status=cudaMalloc(&_cublaslt_workspace,_cublaslt_workspace_size);
  if(ws_status!=cudaSuccess)
  {
    cublasLtDestroy(_cublaslt_handle);
    _cublaslt_handle=nullptr;
    cublasDestroy(_cublas_handle);
    _cublas_handle=nullptr;
    THROW_CAIFE("Failed to allocate cublasLt workspace");
  }
  const char *basis_label="auto";
  if(ws_is_override==true)
  {
    basis_label="override";
  }
  const double workspace_megabytes=static_cast<double>(_cublaslt_workspace_size)/
                                   g_caif_bytes_per_megabyte_d;
  std::fprintf(stderr,
               "[CAIF] cuBLAS-Lt workspace: %.2f MB (%s, sm_%d%d)\n",
               workspace_megabytes,
               basis_label,
               props.ComputeCapabilityMajor(),
               props.ComputeCapabilityMinor());

  // Create cuDNN handle
  cudnnStatus_t cudnn_status=cudnnCreate(&_cudnn_handle);
  if(cudnn_status!=CUDNN_STATUS_SUCCESS)
  {
    cudaFree(_cublaslt_workspace);
    _cublaslt_workspace=nullptr;
    cublasLtDestroy(_cublaslt_handle);
    _cublaslt_handle=nullptr;
    cublasDestroy(_cublas_handle);
    _cublas_handle=nullptr;
    THROW_CAIFE("Failed to create cuDNN handle");
  }

  _initialized=true;
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceContext::Cleanup()
{
#ifdef USE_CAIF_CUDA
  if(_initialized==true)
  {
    if(_cudnn_handle!=nullptr)
    {
      cudnnDestroy(_cudnn_handle);
      _cudnn_handle=nullptr;
    }
    if(_cublaslt_workspace!=nullptr)
    {
      cudaFree(_cublaslt_workspace);
      _cublaslt_workspace=nullptr;
      _cublaslt_workspace_size=0;
    }
    if(_cublaslt_handle!=nullptr)
    {
      cublasLtDestroy(_cublaslt_handle);
      _cublaslt_handle=nullptr;
    }
    if(_cublas_handle!=nullptr)
    {
      cublasDestroy(_cublas_handle);
      _cublas_handle=nullptr;
    }
    _last_cublas_stream=nullptr;
    _initialized=false;
  }
#endif
}

#ifdef USE_CAIF_CUDA
cublasHandle_t CAIF_DeviceContext::CublasHandle()
{
  if(_initialized==false)
  {
    Initialize();
  }
  return _cublas_handle;
}

cudnnHandle_t CAIF_DeviceContext::CudnnHandle()
{
  if(_initialized==false)
  {
    Initialize();
  }
  return _cudnn_handle;
}

cublasLtHandle_t CAIF_DeviceContext::CublasLtHandle()
{
  if(_initialized==false)
  {
    Initialize();
  }
  return _cublaslt_handle;
}

void *CAIF_DeviceContext::CublasLtWorkspace()
{
  if(_initialized==false)
  {
    Initialize();
  }
  return _cublaslt_workspace;
}

size_t CAIF_DeviceContext::CublasLtWorkspaceSize()const
{
  return _cublaslt_workspace_size;
}

void CAIF_DeviceContext::SetCublasStream(cudaStream_t stream)
{
  if(_initialized==false)
  {
    Initialize();
  }
  if(stream==_last_cublas_stream)
  {
    return;
  }
  cublasStatus_t status=cublasSetStream(_cublas_handle,stream);
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("Failed to set cuBLAS stream");
  }
  _last_cublas_stream=stream;
}

void CAIF_DeviceContext::SetCudnnStream(cudaStream_t stream)
{
  if(_initialized==false)
  {
    Initialize();
  }
  cudnnStatus_t status=cudnnSetStream(_cudnn_handle,stream);
  if(status!=CUDNN_STATUS_SUCCESS)
  {
    THROW_CAIFE("Failed to set cuDNN stream");
  }
}
#endif

CAIF_DeviceProperties &CAIF_DeviceContext::DeviceProperties()
{
  if(_initialized==false)
  {
    Initialize();
  }
  return CAIF_DeviceProperties::ForDevice(_device_id);
}

}//end instance namespace
