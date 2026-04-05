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
#include "caif_constants.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#include "cuda/cublas_v2.h"
#include "cudnn/cudnn.h"
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
                                        _initialized(false),
                                        _device_id(0)
{
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

  // Create cuBLAS handle
  cublasStatus_t cublas_status=cublasCreate(&_cublas_handle);
  if(cublas_status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("Failed to create cuBLAS handle");
  }

  // Disable TF32 mode for full FP32 precision (TF32 is default on Ampere+ GPUs)
  // This ensures numerical consistency with CPU BLAS
  cublasSetMathMode(_cublas_handle,CUBLAS_DEFAULT_MATH);

  // Create cublasLt handle
  cublasStatus_t cublaslt_status=cublasLtCreate(&_cublaslt_handle);
  if(cublaslt_status!=CUBLAS_STATUS_SUCCESS)
  {
    cublasDestroy(_cublas_handle);
    _cublas_handle=nullptr;
    THROW_CAIFE("Failed to create cublasLt handle");
  }

  // Allocate cublasLt workspace (4 MB)
  cudaError_t ws_status=cudaMalloc(&_cublaslt_workspace,g_caif_cublaslt_workspace_size);
  if(ws_status!=cudaSuccess)
  {
    cublasLtDestroy(_cublaslt_handle);
    _cublaslt_handle=nullptr;
    cublasDestroy(_cublas_handle);
    _cublas_handle=nullptr;
    THROW_CAIFE("Failed to allocate cublasLt workspace");
  }

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
  return g_caif_cublaslt_workspace_size;
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
