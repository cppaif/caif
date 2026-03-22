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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// Device context singleton for cuBLAS/cuDNN handle management
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_CONTEXT_H
#define CAIF_DEVICE_CONTEXT_H

#include "caif_base.h"

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#include "cuda/cublas_v2.h"
#include "cuda/cublasLt.h"
#include "cudnn/cudnn.h"
#endif

namespace instance
{

/**
 * @brief Singleton for managing cuBLAS and cuDNN handles
 *
 * This class provides global access to CUDA library handles for device operations.
 * It follows the singleton pattern to ensure handles are created once and reused
 * across all device operations.
 *
 * Key features:
 * - Lazy initialization (handles created on first access)
 * - Thread-safe singleton access
 * - Automatic cleanup on destruction
 * - Manages cuBLAS and cuDNN handles
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
class CAIF_DeviceContext:public CAIF_Base
{
  public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the device context singleton
     */
    static CAIF_DeviceContext &Instance();

    /**
     * @brief Destructor - cleans up CUDA handles
     */
    ~CAIF_DeviceContext();

    // Non-copyable and non-movable
    CAIF_DeviceContext(const CAIF_DeviceContext &)=delete;
    CAIF_DeviceContext &operator=(const CAIF_DeviceContext &)=delete;
    CAIF_DeviceContext(CAIF_DeviceContext &&)=delete;
    CAIF_DeviceContext &operator=(CAIF_DeviceContext &&)=delete;

    /**
     * @brief Get the cuBLAS handle
     *
     * Initializes CUDA resources on first call if not already initialized.
     *
     * @return cuBLAS handle (or nullptr if CUDA not available)
     */
#ifdef USE_CAIF_CUDA
    cublasHandle_t CublasHandle();
    cublasLtHandle_t CublasLtHandle();
    void *CublasLtWorkspace();
    size_t CublasLtWorkspaceSize()const;
#else
    void *CublasHandle(){return nullptr;}
    void *CublasLtHandle(){return nullptr;}
    void *CublasLtWorkspace(){return nullptr;}
    size_t CublasLtWorkspaceSize()const{return 0;}
#endif

    /**
     * @brief Get the cuDNN handle
     *
     * Initializes CUDA resources on first call if not already initialized.
     *
     * @return cuDNN handle (or nullptr if CUDA not available)
     */
#ifdef USE_CAIF_CUDA
    cudnnHandle_t CudnnHandle();
#else
    void *CudnnHandle(){return nullptr;}
#endif

    /**
     * @brief Check if CUDA is initialized
     * @return true if CUDA resources have been initialized
     */
    bool IsInitialized()const{return _initialized;}

    /**
     * @brief Initialize CUDA resources
     *
     * Called automatically when handles are accessed, but can be called
     * explicitly to control when initialization happens.
     */
    void Initialize();

    /**
     * @brief Set the stream for cuBLAS operations
     *
     * Operations submitted through the cuBLAS handle will use this stream.
     * Call this before performing cuBLAS operations if you need to use
     * a specific stream.
     *
     * @param stream The CUDA stream handle
     */
#ifdef USE_CAIF_CUDA
    void SetCublasStream(cudaStream_t stream);
#endif

    /**
     * @brief Set the stream for cuDNN operations
     *
     * Operations submitted through the cuDNN handle will use this stream.
     * Call this before performing cuDNN operations if you need to use
     * a specific stream.
     *
     * @param stream The CUDA stream handle
     */
#ifdef USE_CAIF_CUDA
    void SetCudnnStream(cudaStream_t stream);
#endif

  protected:

  private:
    /**
     * @brief Private constructor for singleton pattern
     */
    CAIF_DeviceContext();

    /**
     * @brief Clean up CUDA resources
     */
    void Cleanup();

#ifdef USE_CAIF_CUDA
    cublasHandle_t _cublas_handle;
    cublasLtHandle_t _cublaslt_handle;
    cudnnHandle_t _cudnn_handle;
    void *_cublaslt_workspace;
    cudaStream_t _last_cublas_stream;
#endif
    bool _initialized;
    int _device_id;
};

}//end instance namespace

#endif  // CAIF_DEVICE_CONTEXT_H
