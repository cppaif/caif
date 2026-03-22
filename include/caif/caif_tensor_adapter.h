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
// Tensor adapter for gradual migration from old to new tensor architecture
//------------------------------------------------------------------------------
#ifndef CAIF_TENSOR_ADAPTER_H
#define CAIF_TENSOR_ADAPTER_H

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include <vector>
#include <memory>
#include <cstdint>

namespace instance
{

/**
 * @brief Migration adapter that bridges old CAIF_Tensor API with new CAIF_DeviceTensor
 *
 * This adapter allows gradual migration of existing code:
 * - Old code using RawData()/MutableRawData() continues to work (with sync overhead)
 * - New code can access the underlying CAIF_DeviceTensor directly for optimal performance
 *
 * The adapter uses lazy host caching - host memory is only allocated when
 * old-style API methods are called. A performance warning is logged on first
 * slow-path access to help identify code that needs migration.
 *
 * Usage patterns:
 *
 * OLD CODE (works but slow - triggers sync):
 * ```cpp
 * CAIF_TensorAdapter tensor({batch, features}, stream);
 * float* data = tensor.MutableData<float>();  // Syncs from device, logs warning
 * data[0] = 1.0f;
 * tensor.MarkHostModified();  // Marks for re-upload on next device access
 * ```
 *
 * NEW CODE (fast - stays on device):
 * ```cpp
 * CAIF_TensorAdapter tensor({batch, features}, stream);
 * CAIF_DeviceOps::ReLU(other.Device(), tensor.Device());  // Direct device access
 * ```
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
class CAIF_TensorAdapter:public CAIF_Base
{
  public:
    /**
     * @brief Construct an adapter with device tensor initialized to zeros
     *
     * @param shape Tensor shape
     * @param stream CUDA stream for operations
     */
    CAIF_TensorAdapter(const std::vector<uint32_t> &shape,CAIF_CudaStream &stream);

    /**
     * @brief Construct an adapter wrapping an existing device tensor (move)
     *
     * @param device_tensor Device tensor to wrap (moved into adapter)
     */
    explicit CAIF_TensorAdapter(CAIF_DeviceTensor &&device_tensor);

    /**
     * @brief Destructor
     */
    ~CAIF_TensorAdapter()=default;

    // Move-only (due to device tensor ownership)
    CAIF_TensorAdapter(CAIF_TensorAdapter &&other)noexcept;
    CAIF_TensorAdapter &operator=(CAIF_TensorAdapter &&other)noexcept;
    CAIF_TensorAdapter(const CAIF_TensorAdapter &)=delete;
    CAIF_TensorAdapter &operator=(const CAIF_TensorAdapter &)=delete;

    //--------------------------------------------------------------------------
    // New API - Direct device tensor access (fast path)
    //--------------------------------------------------------------------------

    /**
     * @brief Get mutable reference to underlying device tensor
     *
     * Use this for new code that operates directly on device tensors.
     * This is the fast path - no synchronization overhead.
     */
    CAIF_DeviceTensor &Device(){return _device_tensor;}

    /**
     * @brief Get const reference to underlying device tensor
     */
    const CAIF_DeviceTensor &Device()const{return _device_tensor;}

    //--------------------------------------------------------------------------
    // Old API - Host data access (slow path with sync)
    //--------------------------------------------------------------------------

    /**
     * @brief Get immutable pointer to host data (SLOW - triggers device->host sync)
     *
     * WARNING: This is the slow migration path. It:
     * 1. Allocates host cache if not present
     * 2. Synchronizes device data to host
     * 3. Logs a performance warning on first call
     *
     * Prefer using Device() for new code.
     *
     * @return Const pointer to host data
     */
    const void *RawData()const;

    /**
     * @brief Get mutable pointer to host data (SLOW - triggers device->host sync)
     *
     * WARNING: This is the slow migration path. After modifying host data,
     * call MarkHostModified() to ensure changes are uploaded to device.
     *
     * Prefer using Device() for new code.
     *
     * @return Mutable pointer to host data
     */
    void *MutableRawData();

    /**
     * @brief Get typed const data pointer (SLOW - triggers sync)
     */
    template<typename T>
    const T *ConstData()const
    {
      return static_cast<const T*>(RawData());
    }

    /**
     * @brief Get typed mutable data pointer (SLOW - triggers sync)
     */
    template<typename T>
    T *MutableData()
    {
      return static_cast<T*>(MutableRawData());
    }

    /**
     * @brief Mark that host data has been modified and needs re-upload
     *
     * Call this after modifying data via MutableRawData() or MutableData().
     * The device tensor will be updated on next device access.
     */
    void MarkHostModified();

    /**
     * @brief Force upload host data to device immediately
     *
     * Normally upload happens lazily. Use this if you need to ensure
     * device has latest data before an external CUDA operation.
     */
    void SyncToDevice();

    /**
     * @brief Force download device data to host immediately
     *
     * Normally download happens lazily on RawData() access.
     */
    void SyncFromDevice()const;

    //--------------------------------------------------------------------------
    // Shape and metadata (no sync required)
    //--------------------------------------------------------------------------

    /**
     * @brief Get tensor shape
     */
    const std::vector<uint32_t> &Shape()const{return _device_tensor.Shape();}

    /**
     * @brief Get total number of elements
     */
    size_t NumElements()const{return _device_tensor.TotalElements();}

    /**
     * @brief Get size in bytes
     */
    size_t SizeBytes()const{return _device_tensor.TotalElements()*sizeof(float);}

    /**
     * @brief Check if tensor is valid (has allocated memory)
     */
    bool IsValid()const{return !_device_tensor.IsEmpty();}

    /**
     * @brief Check if host cache exists and is valid
     */
    bool HasHostCache()const{return _host_cache!=nullptr&&_host_cache_valid;}

    /**
     * @brief Check if host modifications are pending upload
     */
    bool HasPendingUpload()const{return _host_modified;}

    //--------------------------------------------------------------------------
    // Static factory methods
    //--------------------------------------------------------------------------

    /**
     * @brief Create adapter with device tensor initialized to zeros
     */
    static CAIF_TensorAdapter Zeros(const std::vector<uint32_t> &shape,CAIF_CudaStream &stream);

    /**
     * @brief Create adapter from host data (uploads to device)
     */
    static CAIF_TensorAdapter FromHostData(const float *data,
                                          const std::vector<uint32_t> &shape,
                                          CAIF_CudaStream &stream);

    /**
     * @brief Create adapter from existing host tensor (uploads to device)
     */
    static CAIF_TensorAdapter FromHost(const CAIF_HostTensor &host,CAIF_CudaStream &stream);

  protected:

  private:
    CAIF_DeviceTensor _device_tensor;
    mutable std::unique_ptr<CAIF_HostTensor> _host_cache;
    mutable bool _host_cache_valid;
    mutable bool _host_modified;
    mutable bool _slow_path_warned;

    /**
     * @brief Ensure host cache exists and is synchronized from device
     */
    void EnsureHostCache()const;

    /**
     * @brief Log performance warning on first slow-path access
     */
    void WarnSlowPath()const;
};

}//end instance namespace

#endif  // CAIF_TENSOR_ADAPTER_H
