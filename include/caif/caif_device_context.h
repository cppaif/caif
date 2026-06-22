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
#include "caif_constants.h"
#include "caif_device_properties.h"

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
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
    size_t CublasLtWorkspaceSize()const{return _cublaslt_workspace_size;}
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
     * @brief Get device properties for the active device
     * @return Reference to cached properties
     */
    CAIF_DeviceProperties &DeviceProperties();

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

    // Private setters — keep Initialize() / Cleanup() bodies free of
    // direct _member writes. Reads through public accessors above.
    // Address-taking for C-API populate-by-pointer (e.g.
    // cublasCreate(&_cublas_handle)) is treated as construction and
    // left as direct member access.
    void SetDeviceId(const int v){_device_id=v;}
    int DeviceId()const{return _device_id;}
#ifdef USE_CAIF_CUDA
    cudaStream_t LastCublasStream()const{return _last_cublas_stream;}
#endif
    void SetInitialized(const bool v){_initialized=v;}
    void SetCublasLtWorkspaceSize(const size_t v){_cublaslt_workspace_size=v;}
    void SetCublasLtWorkspace(void *v){_cublaslt_workspace=v;}
#ifdef USE_CAIF_CUDA
    void SetCublasHandle(cublasHandle_t v){_cublas_handle=v;}
    void SetCublasLtHandle(cublasLtHandle_t v){_cublaslt_handle=v;}
    void SetCudnnHandle(cudnnHandle_t v){_cudnn_handle=v;}
    void SetLastCublasStream(cudaStream_t v){_last_cublas_stream=v;}
#endif

    /**
     * @brief Clean up CUDA resources
     */
    void Cleanup();

    static constexpr size_t _cublaslt_workspace_bytes_small=
                                    g_caif_cublaslt_workspace_bytes_small;
    static constexpr size_t _cublaslt_workspace_bytes_medium=
                                    g_caif_cublaslt_workspace_bytes_medium;
    static constexpr size_t _cublaslt_workspace_bytes_large=
                                    g_caif_cublaslt_workspace_bytes_large;
    static constexpr int _cublaslt_workspace_free_vram_divisor=
                                    g_caif_cublaslt_workspace_free_vram_divisor;

    size_t AutoSizeCublasLtWorkspace(const int cc_major,
                                     const int cc_minor)const;
    size_t ResolveCublasLtWorkspaceSize(const int cc_major,
                                        const int cc_minor,
                                        const size_t free_vram_bytes,
                                        bool &out_is_override)const;

    // RE-ENTRANCY HAZARD — read before "fixing" accessor discipline here.
    // The public accessors CublasHandle() / CublasLtHandle() /
    // CudnnHandle() / CublasLtWorkspace() are LAZY: each one calls
    // Initialize() when !IsInitialized(). Therefore Initialize() and
    // Cleanup() MUST touch these four members directly — `_cublas_handle`
    // etc. — never through their accessors. Routing Initialize()'s own
    // body through CublasHandle() makes it call Initialize() recursively
    // (IsInitialized() is still false mid-init), which re-enters,
    // re-creates handles, and exhausts cuBLAS until cublasCreate throws
    // "Failed to create cuBLAS handle" — cascading into a near-total
    // benchmark failure. Construction/teardown of these members is the
    // accessor-discipline carve-out; direct access is mandatory, not a
    // shortcut. (`_cublaslt_workspace_size` / `_initialized` / `_device_id`
    // have non-lazy accessors and are safe either way.)
#ifdef USE_CAIF_CUDA
    cublasHandle_t _cublas_handle;
    cublasLtHandle_t _cublaslt_handle;
    cudnnHandle_t _cudnn_handle;
    void *_cublaslt_workspace;
    cudaStream_t _last_cublas_stream;
#endif
    size_t _cublaslt_workspace_size;
    bool _initialized;
    int _device_id;
};

}//end instance namespace

#endif  // CAIF_DEVICE_CONTEXT_H
