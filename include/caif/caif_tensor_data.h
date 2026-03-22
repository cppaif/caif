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

/**
 * @file aif_tensor_data.h
 * @brief Abstract base class for tensor data storage and access
 */

#ifndef CAIF_TENSOR_DATA_H
#define CAIF_TENSOR_DATA_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "caif_constants.h"
#include "caif_base.h"
#include "caif_data_type.h"

namespace instance
{

// DEVICE_MIGRATION_REMOVE: This class hierarchy is deprecated.
// Replaced by CAIF_DeviceTensor (GPU) and CAIF_HostTensor (CPU).
// See DEPRECATED_FOR_DEVICE_MIGRATION.md for details.
/**
 * @brief Abstract base class for tensor data storage
 *
 * This class provides the interface for accessing tensor data across different
 * backends and storage strategies. Each backend implementation provides its own
 * derived class that manages memory allocation, device transfers, and data access
 * patterns optimized for that specific backend.
 *
 * @note All tensor data classes must inherit from this base class
 * @see CAIF_CPUTensorData, CAIF_CudaTensorData, CAIF_VulkanTensorData
 */
class CAIF_TensorData:public CAIF_Base
{
  public:
    /**
     * @brief Virtual destructor
     * 
     * Ensures proper cleanup of derived class resources when accessed through
     * the base class interface.
     */
    virtual ~CAIF_TensorData()=default;
    
    /**
     * @brief Get raw data pointer (immutable)
     * @return Pointer to raw data
     */
    virtual const void *RawData()const=0;
    
    /**
     * @brief Get shape of tensor data
     * @return Shape vector
     */
    virtual const std::vector<uint32_t> &Shape()const=0;
    
    /**
     * @brief Get data type
     * @return Data type enum
     */
    virtual CAIF_DataType Type()const=0;
    
    /**
     * @brief Get raw data pointer (mutable)
     * @return Pointer to raw data
     */
    virtual void *MutableRawData()=0;
    
    /**
     * @brief Fill tensor with single value
     * @param value Value to fill with
     */
    virtual void Fill(double value)=0;
    
    /**
     * @brief Get size in bytes
     * @return Total size in bytes
     */
    virtual size_t SizeBytes()const=0;

  protected:

  private:
};

}//end instance namespace

#endif  // CAIF_TENSOR_DATA_H 