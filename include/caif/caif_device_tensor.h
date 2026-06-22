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
#include "caif_exception.h"
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
     * @brief Storage location for tensor data.
     *
     * `Device_e` — CUDA device memory (cudaMalloc), operated on by the device
     * backend (cuBLAS/cuDNN/custom kernels). Default.
     *
     * `Host_e` — aligned host memory (`new[]`), operated on by the host
     * backend (OpenBLAS/OpenMP/Eigen). Calling `Stream()` on a host-backed
     * tensor throws; every op dispatches on `Location()` internally and
     * requires all operand tensors to share the same location.
     */
    enum class Location_e:uint32_t
    {
      Device_e=0,
      Host_e=1
    };

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
     * @brief Non-owning view over an existing device buffer with a chosen shape
     *
     * Zero-copy alternative to `Clone()+Reshape()` when callers only need a
     * different logical shape over data they do not own. The returned tensor
     * shares the pointer and will not free it on destruction. The caller is
     * responsible for ensuring the backing buffer outlives the view.
     */
    static CAIF_DeviceTensor WrapView(void *device_ptr,
                                     const std::vector<uint32_t> &shape,
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
     * @brief Host-backed zero-initialized tensor (no CUDA stream, FP32).
     *
     * Allocates aligned host memory (`new[]`) and zero-fills. Used by the
     * host backend of `CAIF_Ops` — dispatch branches on `Location()` so the
     * same layer code runs on either backend.
     */
    static CAIF_DeviceTensor ZerosHost(const std::vector<uint32_t> &shape);

    /**
     * @brief Host-backed zero-initialized tensor with explicit dtype.
     */
    static CAIF_DeviceTensor ZerosHost(const std::vector<uint32_t> &shape,
                                       CAIF_DataType::CAIF_DataType_e dtype);

    /**
     * @brief Host-backed uninitialized tensor (FP32).
     */
    static CAIF_DeviceTensor UninitializedHost(const std::vector<uint32_t> &shape);

    /**
     * @brief Host-backed uninitialized tensor with explicit dtype.
     */
    static CAIF_DeviceTensor UninitializedHost(const std::vector<uint32_t> &shape,
                                               CAIF_DataType::CAIF_DataType_e dtype);

    /**
     * @brief Copy a host-backed tensor to a new device-backed tensor.
     *
     * Errors if this tensor is already `Location_e::Device_e`.
     */
    CAIF_DeviceTensor ToDevice(CAIF_CudaStream &stream)const;

    /**
     * @brief Copy a device-backed tensor to a new host-backed tensor.
     *
     * Returned tensor is the same `CAIF_DeviceTensor` type but with
     * `Location_e::Host_e`. Distinct from `ToHost()` which returns a
     * legacy `CAIF_HostTensor`.
     */
    CAIF_DeviceTensor ToHostLocation()const;

    /**
     * @brief Copy fp32 data from host to this tensor (in-place upload)
     * @param host_data Pointer to host data
     * @param num_elements Number of elements to copy (must match tensor size)
     */
    void CopyFromHost(const float *host_data,size_t num_elements);

    /**
     * @brief Copy fp32 data from host into this tensor at whatever storage
     *        dtype this tensor was allocated at — performing the host-side
     *        conversion fp32 -> fp16 / fp32 -> bf16 inline before the
     *        single device upload.
     *
     * Use this on the random-init / Xavier-init path: the caller produces
     * an fp32 host sample buffer, this tensor was already allocated at the
     * target storage dtype, and the convert happens on the host side
     * without any GPU staging tensor. fp32 destinations short-circuit to
     * `CopyFromHost(...)`. Replaces the older `staging.To(sd)` pattern
     * which doubled transient device memory at construction time.
     *
     * @param host_data Pointer to fp32 host data
     * @param num_elements Number of elements (must match tensor size)
     */
    void CopyFromHostFp32(const float *host_data,size_t num_elements);

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
     * @brief Download this tensor's data to a host fp32 buffer regardless
     *        of the tensor's storage dtype, performing the host-side
     *        conversion fp16 -> fp32 / bf16 -> fp32 inline after the
     *        single device download.
     *
     * Symmetric counterpart to `CopyFromHostFp32(...)`. fp32 sources
     * short-circuit to `CopyToHost(...)`. fp16 / bf16 sources download
     * raw bytes via `CopyToHostRaw`, then expand to fp32 host-side. No
     * device-side staging tensor.
     *
     * @param host_buffer Pointer to fp32 host buffer (caller-provided,
     *                    must hold at least `TotalElements()` floats)
     */
    void CopyToHostFp32(float *host_buffer)const;

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
     * @brief Get mutable pointer to device memory (typed as float for FP32).
     * Caller is responsible for ensuring the tensor's storage_dtype is fp32
     * (or that the `float *` is being type-erased into a `void *` parameter
     * downstream). Use DevicePtr<T>() for typed access on non-fp32 storage,
     * or route through caif_kernel_dispatch.h for runtime dtype dispatch.
     */
    float *DevicePtr(){return reinterpret_cast<float*>(_device_data);}

    /**
     * @brief Get const pointer to device memory (typed as float for FP32).
     * Same caller contract as the non-const overload.
     */
    const float *DevicePtr()const{return reinterpret_cast<const float*>(_device_data);}

    /**
     * @brief Get typed device pointer for dtype-templated kernel launches.
     * Caller is responsible for passing T that matches the tensor's Dtype().
     */
    template<typename T>
    T *DevicePtr(){return reinterpret_cast<T*>(_device_data);}

    template<typename T>
    const T *DevicePtr()const{return reinterpret_cast<const T*>(_device_data);}

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
    std::vector<uint32_t> &Shape(){return _shape;}

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
     * @brief Storage location (device vs host).
     *
     * Default is `Location_e::Device_e` for every existing factory. The
     * host-backed factories (`ZerosHost`, `UninitializedHost`) produce
     * tensors with `Location_e::Host_e`. `ToDevice` / `ToHostLocation`
     * produce a new tensor in the requested location.
     */
    Location_e Location()const{return _location;}

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
     * @brief Zero the tensor's storage (works for any dtype).
     * Uses cudaMemsetAsync(0); the all-zero bit pattern is +0 in every
     * IEEE float and 0 in every integer encoding, so this is dtype-safe.
     */
    void FillZero();

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

    // Alignment (bytes) for host-backed tensor allocations made via
    // ::operator new[] in the UninitializedHost factory path.
    static constexpr size_t _host_alignment=64;

    // Private setters — every internal write to a member goes through one
    // of these so method bodies stay accessor-only. Address-taking sites
    // (`cudaMalloc(&_device_data)`) and ctor init lists are exempt.
    void SetDeviceData(void *p){_device_data=p;}
    void SetShape(std::vector<uint32_t> &&v){_shape=std::move(v);}
    void SetShape(const std::vector<uint32_t> &v){_shape=v;}
    void SetTotalElements(const size_t v){_total_elements=v;}
    void SetSizeBytes(const size_t v){_size_bytes=v;}
    void SetStreamPtr(CAIF_CudaStream *p){_stream=p;}
    void SetDtypeInfo(const CAIF_DataType &v){_dtype=v;}
    void SetOwnsData(const bool v){_owns_data=v;}
    void SetLocation(const Location_e v){_location=v;}

    // Internal reads for cases that need the raw stream pointer (not the
    // reference yielded by Stream()).
    CAIF_CudaStream *StreamPtr(){return _stream;}
    const CAIF_CudaStream *StreamPtr()const{return _stream;}
    bool OwnsData()const{return _owns_data;}

    void *_device_data;  // Raw device pointer (was float*, now void* for multi-dtype)
    std::vector<uint32_t> _shape;
    size_t _total_elements;
    size_t _size_bytes;
    CAIF_CudaStream *_stream;
    CAIF_DataType _dtype;
    bool _owns_data;  // False for WrapView — destructor skips free
    Location_e _location;
};

}//end instance namespace

#endif  // CAIF_DEVICE_TENSOR_H
