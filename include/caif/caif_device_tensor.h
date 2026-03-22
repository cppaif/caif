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

#ifndef CAIF_DEVICE_TENSOR_H
#define CAIF_DEVICE_TENSOR_H

#include "caif_base.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace instance
{

// Forward declaration
class CAIF_HostTensor;

/**
 * @brief Device-resident tensor for GPU computation
 *
 * This class manages tensor data exclusively on the GPU device.
 * It is the core of the device-resident tensor architecture, designed to:
 * - Eliminate dirty-flag synchronization bugs
 * - Enable stream-based pipelining
 * - Maximize GPU utilization
 *
 * Supports multiple data types (fp32, fp16, bf16, int8, int4) via
 * CAIF_DataType. Default dtype is FP32 for backward compatibility.
 * All existing code that does not specify a dtype continues to work
 * unchanged.
 *
 * Key design principles:
 * - NO host memory buffer (device-only)
 * - NO dirty flags (explicit transfers only)
 * - Stream association for operation ordering
 * - Move-only semantics (no accidental copies)
 *
 * Data transfer to/from host requires explicit method calls:
 * - FromHost() / CopyFromHost() to upload
 * - ToHost() / CopyToHost() to download
 */
class CAIF_DeviceTensor:public CAIF_Base
{
  public:
    /**
     * @brief Default constructor - creates empty tensor
     */
    CAIF_DeviceTensor();

    /**
     * @brief Destructor - frees device memory
     */
    ~CAIF_DeviceTensor();

    // Move-only semantics (no copies - explicit transfer required)
    CAIF_DeviceTensor(CAIF_DeviceTensor &&other);
    CAIF_DeviceTensor &operator=(CAIF_DeviceTensor &&other);
    CAIF_DeviceTensor(const CAIF_DeviceTensor &)=delete;
    CAIF_DeviceTensor &operator=(const CAIF_DeviceTensor &)=delete;

    /**
     * @brief Create a tensor filled with zeros on device (FP32)
     * @param shape The shape of the tensor
     * @param stream The CUDA stream for initialization
     * @return New device tensor initialized to zero
     */
    static CAIF_DeviceTensor Zeros(const std::vector<uint32_t> &shape,CAIF_CudaStream &stream);

    /**
     * @brief Create a tensor filled with zeros on device with specified dtype
     * @param shape The shape of the tensor
     * @param stream The CUDA stream for initialization
     * @param dtype The data type for the tensor
     * @return New device tensor initialized to zero
     */
    static CAIF_DeviceTensor Zeros(const std::vector<uint32_t> &shape,
                                  CAIF_CudaStream &stream,
                                  CAIF_DataType::CAIF_DataType_e dtype);

    /**
     * @brief Create a tensor with uninitialized device memory (FP32)
     * @param shape The shape of the tensor
     * @param stream The CUDA stream to associate with
     * @return New device tensor with uninitialized values
     */
    static CAIF_DeviceTensor Uninitialized(const std::vector<uint32_t> &shape,
                                          CAIF_CudaStream &stream);

    /**
     * @brief Create a tensor with uninitialized device memory with specified dtype
     * @param shape The shape of the tensor
     * @param stream The CUDA stream to associate with
     * @param dtype The data type for the tensor
     * @return New device tensor with uninitialized values
     */
    static CAIF_DeviceTensor Uninitialized(const std::vector<uint32_t> &shape,
                                          CAIF_CudaStream &stream,
                                          CAIF_DataType::CAIF_DataType_e dtype);

    /**
     * @brief Create a device tensor from host tensor (upload, FP32)
     * @param host The host tensor to copy from
     * @param stream The CUDA stream for the copy operation
     * @return New device tensor containing a copy of the host data
     */
    static CAIF_DeviceTensor FromHost(const CAIF_HostTensor &host,CAIF_CudaStream &stream);

    /**
     * @brief Create a device tensor from raw host data (upload, FP32)
     * @param host_data Pointer to host data
     * @param shape The shape of the tensor
     * @param stream The CUDA stream for the copy operation
     * @return New device tensor containing a copy of the data
     */
    static CAIF_DeviceTensor FromHostData(const float *host_data,
                                         const std::vector<uint32_t> &shape,
                                         CAIF_CudaStream &stream);

    /**
     * @brief Create a device tensor from raw host bytes with specified dtype
     * @param host_data Pointer to raw host data (size must match StorageSizeBytes)
     * @param shape The shape of the tensor
     * @param stream The CUDA stream for the copy operation
     * @param dtype The data type of the source data
     * @return New device tensor containing a copy of the data
     */
    static CAIF_DeviceTensor FromHostRaw(const void *host_data,
                                        const std::vector<uint32_t> &shape,
                                        CAIF_CudaStream &stream,
                                        CAIF_DataType::CAIF_DataType_e dtype);

    /**
     * @brief Copy fp32 data from host to this tensor (in-place upload)
     * @param host_data Pointer to host data
     * @param num_elements Number of elements to copy (must match tensor size)
     */
    void CopyFromHost(const float *host_data,size_t num_elements);

    /**
     * @brief Copy raw bytes from host to this tensor (in-place upload)
     * @param host_data Pointer to raw host bytes
     * @param num_bytes Number of bytes to copy (must match SizeBytes())
     */
    void CopyFromHostRaw(const void *host_data,size_t num_bytes);

    /**
     * @brief Copy data from this tensor to host buffer (download, FP32 only)
     * @param host_buffer Pointer to host buffer (caller-provided)
     */
    void CopyToHost(float *host_buffer)const;

    /**
     * @brief Copy raw bytes from this tensor to host buffer (download, any dtype)
     * @param host_buffer Pointer to host buffer (must have SizeBytes() space)
     */
    void CopyToHostRaw(void *host_buffer)const;

    /**
     * @brief Create a host tensor from this device tensor (download, FP32 only)
     * @return New host tensor containing a copy of the device data
     */
    CAIF_HostTensor ToHost()const;

    /**
     * @brief Convert tensor to a different data type
     *
     * Creates a new tensor with the target dtype. The conversion is
     * performed on the GPU via CUDA kernels. Supported conversions:
     * fp32<->fp16, fp32<->bf16. Other conversions will be added as needed.
     *
     * @param target_dtype The target data type
     * @return New device tensor with converted data
     */
    CAIF_DeviceTensor To(CAIF_DataType::CAIF_DataType_e target_dtype)const;

    /**
     * @brief Get mutable pointer to device memory (typed as float for FP32)
     * @return Pointer to device memory
     */
    float *DevicePtr(){return reinterpret_cast<float*>(_device_data);}

    /**
     * @brief Get const pointer to device memory (typed as float for FP32)
     * @return Const pointer to device memory
     */
    const float *DevicePtr()const{return reinterpret_cast<const float*>(_device_data);}

    /**
     * @brief Get mutable pointer to raw device memory (any dtype)
     * @return Void pointer to device memory
     */
    void *DeviceDataRaw(){return _device_data;}

    /**
     * @brief Get const pointer to raw device memory (any dtype)
     * @return Const void pointer to device memory
     */
    const void *DeviceDataRaw()const{return _device_data;}

    /**
     * @brief Get the data type of this tensor
     * @return The data type
     */
    CAIF_DataType::CAIF_DataType_e Dtype()const{return _dtype.Value();}

    /**
     * @brief Get the CAIF_DataType object for this tensor
     * @return The data type object
     */
    const CAIF_DataType &DtypeInfo()const{return _dtype;}

    /**
     * @brief Check if this tensor is FP32
     * @return true if dtype is Float32
     */
    bool IsFP32()const{return _dtype==CAIF_DataType::CAIF_DataType_e::Float32;}

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
     * @brief Get size in bytes (actual storage, handles INT4 packing)
     * @return Total size of data in bytes
     */
    size_t SizeBytes()const{return _size_bytes;}

    /**
     * @brief Check if tensor is empty (has no elements)
     * @return true if tensor has zero elements
     */
    bool IsEmpty()const{return _total_elements==0;}

    /**
     * @brief Check if device memory is allocated
     * @return true if device memory has been allocated
     */
    bool IsAllocated()const{return _device_data!=nullptr;}

    /**
     * @brief Get the associated CUDA stream
     * @return Reference to the CUDA stream
     */
    CAIF_CudaStream &Stream(){return *_stream;}

    /**
     * @brief Get the associated CUDA stream (const)
     * @return Const reference to the CUDA stream
     */
    const CAIF_CudaStream &Stream()const{return *_stream;}

    /**
     * @brief Set the associated CUDA stream
     * @param stream The new stream to associate with
     */
    void SetStream(CAIF_CudaStream &stream){_stream=&stream;}

    /**
     * @brief Synchronize the associated stream
     */
    void Synchronize()const;

    /**
     * @brief Fill tensor with a single value (on device, FP32 only)
     * @param value The value to fill with
     */
    void Fill(float value);

    /**
     * @brief Reshape the tensor (must have same total elements)
     * @param new_shape The new shape
     */
    void Reshape(const std::vector<uint32_t> &new_shape);

    /**
     * @brief Create a deep copy of this tensor on the same stream
     * @return New device tensor with copied data
     */
    CAIF_DeviceTensor Clone()const;

    /**
     * @brief Create a deep copy of this tensor on a different stream
     * @param stream The stream for the new tensor
     * @return New device tensor with copied data
     */
    CAIF_DeviceTensor CloneTo(CAIF_CudaStream &stream)const;

  protected:

  private:
    /**
     * @brief Private constructor for internal use (FP32)
     */
    CAIF_DeviceTensor(const std::vector<uint32_t> &shape,CAIF_CudaStream &stream,bool allocate);

    /**
     * @brief Private constructor with dtype
     */
    CAIF_DeviceTensor(const std::vector<uint32_t> &shape,
                     CAIF_CudaStream &stream,
                     bool allocate,
                     CAIF_DataType::CAIF_DataType_e dtype);

    void AllocateDevice();
    void FreeDevice();

    void *_device_data;  // Raw device pointer (was float*, now void* for multi-dtype)
    std::vector<uint32_t> _shape;
    size_t _total_elements;
    size_t _size_bytes;
    CAIF_CudaStream *_stream;
    CAIF_DataType _dtype;
};

}//end instance namespace

#endif  // CAIF_DEVICE_TENSOR_H
