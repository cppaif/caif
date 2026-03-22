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

#ifndef CAIF_HOST_TENSOR_H
#define CAIF_HOST_TENSOR_H

#include "caif_base.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

// Forward declaration for ToDevice()
class CAIF_DeviceTensor;
class CAIF_CudaStream;

/**
 * @brief Host-only tensor for CPU data storage
 *
 * This class manages tensor data exclusively on the host (CPU) side.
 * It is designed for:
 * - Loading data from disk (CSV, binary files)
 * - Storing final prediction results
 * - Model save/load operations
 *
 * Data transfer to GPU requires explicit ToDevice() call, which creates
 * an CAIF_DeviceTensor. There are NO dirty flags - data location is explicit.
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
class CAIF_HostTensor:public CAIF_Base
{
  public:
    /**
     * @brief Default constructor - creates empty tensor
     */
    CAIF_HostTensor();

    /**
     * @brief Destructor
     */
    ~CAIF_HostTensor();

    // Move semantics
    CAIF_HostTensor(CAIF_HostTensor &&other)noexcept;
    CAIF_HostTensor &operator=(CAIF_HostTensor &&other)noexcept;

    // Copy semantics (explicit copies)
    CAIF_HostTensor(const CAIF_HostTensor &other);
    CAIF_HostTensor &operator=(const CAIF_HostTensor &other);

    /**
     * @brief Create a tensor filled with zeros
     * @param shape The shape of the tensor
     * @return New tensor initialized to zero
     */
    static CAIF_HostTensor Zeros(const std::vector<uint32_t> &shape);

    /**
     * @brief Create a tensor from existing data (copies the data)
     * @param data Pointer to source data
     * @param shape The shape of the tensor
     * @return New tensor containing a copy of the data
     */
    static CAIF_HostTensor FromData(const float *data,const std::vector<uint32_t> &shape);

    /**
     * @brief Create a tensor with uninitialized memory
     * @param shape The shape of the tensor
     * @return New tensor with uninitialized values (for performance when will be overwritten)
     */
    static CAIF_HostTensor Uninitialized(const std::vector<uint32_t> &shape);

    /**
     * @brief Get mutable pointer to data
     * @return Pointer to float data array
     */
    float *Data(){return _data.get();}

    /**
     * @brief Get const pointer to data
     * @return Const pointer to float data array
     */
    const float *Data()const{return _data.get();}

    /**
     * @brief Element access with bounds checking
     * @param idx Linear index into the flattened tensor
     * @return Reference to the element
     */
    float &At(size_t idx);

    /**
     * @brief Const element access with bounds checking
     * @param idx Linear index into the flattened tensor
     * @return Const reference to the element
     */
    const float &At(size_t idx)const;

    /**
     * @brief Element access without bounds checking (for performance)
     * @param idx Linear index into the flattened tensor
     * @return Reference to the element
     */
    float &operator[](size_t idx){return _data[idx];}

    /**
     * @brief Const element access without bounds checking
     * @param idx Linear index into the flattened tensor
     * @return Const reference to the element
     */
    const float &operator[](size_t idx)const{return _data[idx];}

    /**
     * @brief Get the shape of the tensor
     * @return Vector of dimension sizes
     */
    const std::vector<uint32_t> &Shape()const{return _shape;}

    /**
     * @brief Get total number of elements
     * @return Product of all dimensions
     */
    size_t TotalElements()const{return _total_elements;}

    /**
     * @brief Get size in bytes
     * @return Total size of data in bytes
     */
    size_t SizeBytes()const{return _size_bytes;}

    /**
     * @brief Check if tensor is empty (has no elements)
     * @return true if tensor has zero elements
     */
    bool IsEmpty()const{return _total_elements==0;}

    /**
     * @brief Upload tensor to device (GPU)
     *
     * Creates a new CAIF_DeviceTensor with a copy of this tensor's data.
     * The host tensor remains valid and unchanged after this call.
     *
     * @param stream The CUDA stream to use for the copy operation
     * @return New device tensor containing a copy of the data
     */
    CAIF_DeviceTensor ToDevice(CAIF_CudaStream &stream)const;

    /**
     * @brief Fill tensor with a single value
     * @param value The value to fill with
     */
    void Fill(float value);

    /**
     * @brief Reshape the tensor (must have same total elements)
     * @param new_shape The new shape
     */
    void Reshape(const std::vector<uint32_t> &new_shape);

  protected:

  private:
    /**
     * @brief Private constructor for factory methods
     * @param shape The shape of the tensor
     * @param allocate Whether to allocate memory
     */
    explicit CAIF_HostTensor(const std::vector<uint32_t> &shape,bool allocate);

    std::unique_ptr<float[]> _data;
    std::vector<uint32_t> _shape;
    size_t _total_elements;
    size_t _size_bytes;
};

}//end instance namespace

#endif  // CAIF_HOST_TENSOR_H
