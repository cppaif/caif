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

#include "caif_device_tensor.h"
#include "caif_host_tensor.h"
#include "caif_exception.h"
#include "caif_cuda_kernels_attention_support.cuh"
#include "caif_cuda_kernels_quant.cuh"
#include "caif_host_dtype_convert.h"
#include "ise_lib/ise_out.h"
#include <cstring>
#include <new>
#include <vector>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#endif

namespace instance
{

CAIF_DeviceTensor::CAIF_DeviceTensor():_device_data(nullptr),
                                     _shape(),
                                     _total_elements(0),
                                     _size_bytes(0),
                                     _stream(nullptr),
                                     _dtype(CAIF_DataType::CAIF_DataType_e::Float32),
                                     _owns_data(false),
                                     _location(Location_e::Device_e)
{
}

CAIF_DeviceTensor::CAIF_DeviceTensor(const std::vector<uint32_t> &shape,
                                   CAIF_CudaStream &stream,
                                   bool allocate):_device_data(nullptr),
                                                  _shape(shape),
                                                  _total_elements(1),
                                                  _size_bytes(0),
                                                  _stream(&stream),
                                                  _dtype(CAIF_DataType::CAIF_DataType_e::Float32),
                                                  _owns_data(allocate),
                                                  _location(Location_e::Device_e)
{
  size_t total=1;
  for(const uint32_t dim:Shape())
  {
    total*=dim;
  }
  SetTotalElements(total);
  SetSizeBytes(total*sizeof(float));

  if(allocate==true&&TotalElements()>0)
  {
    AllocateDevice();
  }
}

CAIF_DeviceTensor::CAIF_DeviceTensor(const std::vector<uint32_t> &shape,
                                   CAIF_CudaStream &stream,
                                   bool allocate,
                                   CAIF_DataType::CAIF_DataType_e dtype):_device_data(nullptr),
                                                                       _shape(shape),
                                                                       _total_elements(1),
                                                                       _size_bytes(0),
                                                                       _stream(&stream),
                                                                       _dtype(dtype),
                                                                       _owns_data(allocate),
                                                                       _location(Location_e::Device_e)
{
  size_t total=1;
  for(const uint32_t dim:Shape())
  {
    total*=dim;
  }
  SetTotalElements(total);
  SetSizeBytes(DtypeInfo().StorageSizeBytes(total));

  if(allocate==true&&TotalElements()>0)
  {
    AllocateDevice();
  }
}

CAIF_DeviceTensor::~CAIF_DeviceTensor()
{
  FreeDevice();
}

CAIF_DeviceTensor::CAIF_DeviceTensor(CAIF_DeviceTensor &&other):_device_data(other._device_data),
                                                                     _shape(std::move(other._shape)),
                                                                     _total_elements(other._total_elements),
                                                                     _size_bytes(other._size_bytes),
                                                                     _stream(other._stream),
                                                                     _dtype(other._dtype),
                                                                     _owns_data(other._owns_data),
                                                                     _location(other._location)
{
  other.SetDeviceData(nullptr);
  other.SetTotalElements(0);
  other.SetSizeBytes(0);
  other.SetStreamPtr(nullptr);
  other.SetOwnsData(false);
  other.SetLocation(Location_e::Device_e);
}

CAIF_DeviceTensor &CAIF_DeviceTensor::operator=(CAIF_DeviceTensor &&other)
{
  if(this!=&other)
  {
    FreeDevice();

    SetDeviceData(other.DeviceDataRaw());
    SetShape(std::move(other.Shape()));
    SetTotalElements(other.TotalElements());
    SetSizeBytes(other.SizeBytes());
    SetStreamPtr(other.StreamPtr());
    SetDtypeInfo(other.DtypeInfo());
    SetOwnsData(other.OwnsData());
    SetLocation(other.Location());

    other.SetDeviceData(nullptr);
    other.SetTotalElements(0);
    other.SetSizeBytes(0);
    other.SetStreamPtr(nullptr);
    other.SetOwnsData(false);
    other.SetLocation(Location_e::Device_e);
  }
  return *this;
}

void CAIF_DeviceTensor::AllocateDevice()
{
#ifdef USE_CAIF_CUDA
  if(IsAllocated()==true)
  {
    THROW_CAIFE("Device memory already allocated");
  }
  if(SizeBytes()==0)
  {
    THROW_CAIFE("Cannot allocate zero-size device tensor");
  }

  // [ALLOC] debug logging commented out — kept for ad-hoc memory tracing.
  // Generates ~16K lines per training step at production scale; uncomment
  // only for a short window or it will OOM the log target.
  // if(_size_bytes>=1024*1024)
  // {
  //   size_t free_mem=0;
  //   size_t total_mem=0;
  //   cudaMemGetInfo(&free_mem,&total_mem);
  //   std::string shape_str="[";
  //   for(size_t si=0;si<_shape.size();++si)
  //   {
  //     if(si>0) shape_str+=",";
  //     shape_str+=std::to_string(_shape[si]);
  //   }
  //   shape_str+="]";
  //   ISE_Out::Out()<<"[ALLOC] "<<(_size_bytes/(1024*1024))<<"MB ("
  //                 <<_size_bytes<<" bytes) shape="<<shape_str
  //                 <<" dtype="<<_dtype.Name()
  //                 <<" free="<<(free_mem/(1024*1024))<<"MB"
  //                 <<" total="<<(total_mem/(1024*1024))<<"MB"<<std::endl;
  // }

  cudaError_t status=cudaMallocAsync(&_device_data,
                                     SizeBytes(),
                                     Stream().Handle());
  if(status!=cudaSuccess)
  {
    size_t err_free=0;
    size_t err_total=0;
    cudaMemGetInfo(&err_free,&err_total);
    std::string err_msg="Failed to allocate CUDA device memory for tensor: ";
    err_msg+=cudaGetErrorString(status);
    err_msg+=" (requested ";
    err_msg+=std::to_string(SizeBytes());
    err_msg+=" bytes, free=";
    err_msg+=std::to_string(err_free/(1024*1024));
    err_msg+="MB, total=";
    err_msg+=std::to_string(err_total/(1024*1024));
    err_msg+="MB)";
    THROW_CAIFE(err_msg);
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::FreeDevice()
{
  if(IsAllocated()==true&&OwnsData()==true)
  {
    if(Location()==Location_e::Host_e)
    {
      ::operator delete[](DeviceDataRaw(),std::align_val_t(64));
      SetDeviceData(nullptr);
      SetOwnsData(false);
      return;
    }
#ifdef USE_CAIF_CUDA
    if(StreamPtr()!=nullptr)
    {
      cudaFreeAsync(DeviceDataRaw(),Stream().Handle());
    }
    else
    {
      cudaFree(DeviceDataRaw());
    }
#endif
  }
  SetDeviceData(nullptr);
  SetOwnsData(false);
}

CAIF_DeviceTensor CAIF_DeviceTensor::Zeros(const std::vector<uint32_t> &shape,
                                         CAIF_CudaStream &stream)
{
  CAIF_DeviceTensor tensor(shape,stream,true);

#ifdef USE_CAIF_CUDA
  if(tensor.IsAllocated()==true&&tensor.SizeBytes()>0)
  {
    cudaError_t status=cudaMemsetAsync(tensor.DeviceDataRaw(),
                                       0,
                                       tensor.SizeBytes(),
                                       stream.Handle());
    if(status!=cudaSuccess)
    {
      THROW_CAIFE("Failed to zero-initialize device tensor");
    }
  }
#endif

  return tensor;
}

CAIF_DeviceTensor CAIF_DeviceTensor::Zeros(const std::vector<uint32_t> &shape,
                                         CAIF_CudaStream &stream,
                                         CAIF_DataType::CAIF_DataType_e dtype)
{
  CAIF_DeviceTensor tensor(shape,stream,true,dtype);

#ifdef USE_CAIF_CUDA
  if(tensor.IsAllocated()==true&&tensor.SizeBytes()>0)
  {
    cudaError_t status=cudaMemsetAsync(tensor.DeviceDataRaw(),
                                       0,
                                       tensor.SizeBytes(),
                                       stream.Handle());
    if(status!=cudaSuccess)
    {
      THROW_CAIFE("Failed to zero-initialize device tensor");
    }
  }
#endif

  return tensor;
}

CAIF_DeviceTensor CAIF_DeviceTensor::Uninitialized(const std::vector<uint32_t> &shape,
                                                 CAIF_CudaStream &stream)
{
  return CAIF_DeviceTensor(shape,stream,true);
}

CAIF_DeviceTensor CAIF_DeviceTensor::Uninitialized(const std::vector<uint32_t> &shape,
                                                 CAIF_CudaStream &stream,
                                                 CAIF_DataType::CAIF_DataType_e dtype)
{
  return CAIF_DeviceTensor(shape,stream,true,dtype);
}

CAIF_DeviceTensor CAIF_DeviceTensor::WrapView(void *device_ptr,
                                             const std::vector<uint32_t> &shape,
                                             CAIF_CudaStream &stream,
                                             CAIF_DataType::CAIF_DataType_e dtype)
{
  CAIF_DeviceTensor view(shape,stream,false,dtype);
  view.SetDeviceData(device_ptr);
  view.SetOwnsData(false);
  return view;
}

CAIF_DeviceTensor CAIF_DeviceTensor::FromHost(const CAIF_HostTensor &host,CAIF_CudaStream &stream)
{
  if(host.IsEmpty()==true)
  {
    return CAIF_DeviceTensor();
  }

  CAIF_DeviceTensor tensor(host.Shape(),stream,true);

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(tensor.DeviceDataRaw(),
                                     host.Data(),
                                     tensor.SizeBytes(),
                                     cudaMemcpyHostToDevice,
                                     stream.Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy tensor from host to device");
  }
#endif

  return tensor;
}

CAIF_DeviceTensor CAIF_DeviceTensor::FromHostData(const float *host_data,
                                                const std::vector<uint32_t> &shape,
                                                CAIF_CudaStream &stream)
{
  if(host_data==nullptr)
  {
    THROW_CAIFE("Cannot create device tensor from null host data");
  }

  CAIF_DeviceTensor tensor(shape,stream,true);

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(tensor.DeviceDataRaw(),
                                     host_data,
                                     tensor.SizeBytes(),
                                     cudaMemcpyHostToDevice,
                                     stream.Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy tensor from host data to device");
  }
#endif

  return tensor;
}

CAIF_DeviceTensor CAIF_DeviceTensor::FromHostRaw(const void *host_data,
                                               const std::vector<uint32_t> &shape,
                                               CAIF_CudaStream &stream,
                                               CAIF_DataType::CAIF_DataType_e dtype)
{
  if(host_data==nullptr)
  {
    THROW_CAIFE("Cannot create device tensor from null host data");
  }

  CAIF_DeviceTensor tensor(shape,stream,true,dtype);

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(tensor.DeviceDataRaw(),
                                     host_data,
                                     tensor.SizeBytes(),
                                     cudaMemcpyHostToDevice,
                                     stream.Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy tensor from host raw data to device");
  }
#endif

  return tensor;
}

void CAIF_DeviceTensor::CopyFromHost(const float *host_data,size_t num_elements)
{
  if(host_data==nullptr)
  {
    THROW_CAIFE("Cannot copy from null host data");
  }
  if(num_elements!=TotalElements())
  {
    THROW_CAIFE("Host data size does not match tensor size");
  }
  if(IsAllocated()==false)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(StreamPtr()==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }
  if(Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CopyFromHost(float*) requires FP32 tensor; use CopyFromHostRaw for other dtypes");
  }

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(DeviceDataRaw(),
                                     host_data,
                                     SizeBytes(),
                                     cudaMemcpyHostToDevice,
                                     Stream().Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy data from host to device tensor");
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::CopyFromHostFp32(const float *host_data,size_t num_elements)
{
  try
  {
    if(host_data==nullptr)
    {
      THROW_CAIFE("Cannot copy from null host data");
    }
    if(num_elements!=TotalElements())
    {
      THROW_CAIFE("Host fp32 element count does not match tensor element count");
    }
    if(IsAllocated()==false)
    {
      THROW_CAIFE("Device tensor not allocated");
    }
    if(Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      CopyFromHost(host_data,num_elements);
      return;
    }
    if(Dtype()==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      std::vector<uint16_t> packed(num_elements);
      CAIF_HostDtypeConvert::Fp32ToBf16Buffer(host_data,packed.data(),num_elements);
      CopyFromHostRaw(packed.data(),packed.size()*sizeof(uint16_t));
      return;
    }
    if(Dtype()==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      std::vector<uint16_t> packed(num_elements);
      CAIF_HostDtypeConvert::Fp32ToFp16Buffer(host_data,packed.data(),num_elements);
      CopyFromHostRaw(packed.data(),packed.size()*sizeof(uint16_t));
      return;
    }
    THROW_CAIFE("CopyFromHostFp32: unsupported target dtype");
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_DeviceTensor::CopyFromHostRaw(const void *host_data,size_t num_bytes)
{
  if(host_data==nullptr)
  {
    THROW_CAIFE("Cannot copy from null host data");
  }
  if(num_bytes!=SizeBytes())
  {
    THROW_CAIFE("Host data byte count does not match tensor storage size");
  }
  if(IsAllocated()==false)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(StreamPtr()==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(DeviceDataRaw(),
                                     host_data,
                                     SizeBytes(),
                                     cudaMemcpyHostToDevice,
                                     Stream().Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy raw data from host to device tensor");
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::CopyToHost(float *host_buffer)const
{
  if(host_buffer==nullptr)
  {
    THROW_CAIFE("Cannot copy to null host buffer");
  }
  if(IsAllocated()==false)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(StreamPtr()==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }
  if(Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CopyToHost(float*) requires FP32 tensor; use CopyToHostRaw for other dtypes");
  }

#ifdef USE_CAIF_CUDA
  Stream().Synchronize();

  cudaError_t status=cudaMemcpy(host_buffer,
                                DeviceDataRaw(),
                                SizeBytes(),
                                cudaMemcpyDeviceToHost);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy data from device tensor to host");
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::CopyToHostFp32(float *host_buffer)const
{
  try
  {
    if(host_buffer==nullptr)
    {
      THROW_CAIFE("Cannot copy to null host buffer");
    }
    if(IsAllocated()==false)
    {
      THROW_CAIFE("Device tensor not allocated");
    }
    if(Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      CopyToHost(host_buffer);
      return;
    }
    if(Dtype()==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      const size_t n=TotalElements();
      std::vector<uint16_t> packed(n);
      CopyToHostRaw(packed.data());
      CAIF_HostDtypeConvert::Bf16ToFp32Buffer(packed.data(),host_buffer,n);
      return;
    }
    if(Dtype()==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      const size_t n=TotalElements();
      std::vector<uint16_t> packed(n);
      CopyToHostRaw(packed.data());
      CAIF_HostDtypeConvert::Fp16ToFp32Buffer(packed.data(),host_buffer,n);
      return;
    }
    THROW_CAIFE("CopyToHostFp32: unsupported source dtype");
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_DeviceTensor::CopyToHostRaw(void *host_buffer)const
{
  if(host_buffer==nullptr)
  {
    THROW_CAIFE("Cannot copy to null host buffer");
  }
  if(IsAllocated()==false)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(StreamPtr()==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }

#ifdef USE_CAIF_CUDA
  Stream().Synchronize();

  cudaError_t status=cudaMemcpy(host_buffer,
                                DeviceDataRaw(),
                                SizeBytes(),
                                cudaMemcpyDeviceToHost);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy raw data from device tensor to host");
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

CAIF_HostTensor CAIF_DeviceTensor::ToHost()const
{
  if(IsEmpty()==true)
  {
    return CAIF_HostTensor();
  }
  if(Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("ToHost() requires FP32 tensor; convert with To(Float32) first");
  }

  CAIF_HostTensor host=CAIF_HostTensor::Uninitialized(Shape());
  CopyToHost(host.Data());
  return host;
}

CAIF_DeviceTensor CAIF_DeviceTensor::To(CAIF_DataType::CAIF_DataType_e target_dtype)const
{
  try
  {
    if(IsEmpty()==true||StreamPtr()==nullptr)
    {
      return CAIF_DeviceTensor();
    }
    if(Dtype()==target_dtype)
    {
      return Clone();
    }

    CAIF_DeviceTensor result(Shape(),*_stream,true,target_dtype);

#ifdef USE_CAIF_CUDA
    const CAIF_DataType::CAIF_DataType_e src=Dtype();
    const CAIF_DataType::CAIF_DataType_e dst=target_dtype;
    const int64_t n=static_cast<int64_t>(TotalElements());
    cudaStream_t raw_stream=Stream().Handle();

    // FP32 -> FP16
    if(src==CAIF_DataType::CAIF_DataType_e::Float32&&
       dst==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      launch_convert_fp32_to_fp16(static_cast<const float*>(DeviceDataRaw()),
                                  result.DeviceDataRaw(),
                                  n,
                                  raw_stream);
      return result;
    }
    // FP16 -> FP32
    if(src==CAIF_DataType::CAIF_DataType_e::Float16&&
       dst==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      launch_convert_fp16_to_fp32(DeviceDataRaw(),
                                  static_cast<float*>(result.DeviceDataRaw()),
                                  n,
                                  raw_stream);
      return result;
    }
    // FP32 -> BF16
    if(src==CAIF_DataType::CAIF_DataType_e::Float32&&
       dst==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      launch_convert_fp32_to_bf16(static_cast<const float*>(DeviceDataRaw()),
                                  result.DeviceDataRaw(),
                                  n,
                                  raw_stream);
      return result;
    }
    // BF16 -> FP32
    if(src==CAIF_DataType::CAIF_DataType_e::BFloat16&&
       dst==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      launch_convert_bf16_to_fp32(DeviceDataRaw(),
                                  static_cast<float*>(result.DeviceDataRaw()),
                                  n,
                                  raw_stream);
      return result;
    }
    // FP16 -> BF16 (via FP32 intermediate)
    if(src==CAIF_DataType::CAIF_DataType_e::Float16&&
       dst==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      CAIF_DeviceTensor fp32=To(CAIF_DataType::CAIF_DataType_e::Float32);
      return fp32.To(CAIF_DataType::CAIF_DataType_e::BFloat16);
    }
    // BF16 -> FP16 (via FP32 intermediate)
    if(src==CAIF_DataType::CAIF_DataType_e::BFloat16&&
       dst==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      CAIF_DeviceTensor fp32=To(CAIF_DataType::CAIF_DataType_e::Float32);
      return fp32.To(CAIF_DataType::CAIF_DataType_e::Float16);
    }
    // FP32 -> INT8
    if(src==CAIF_DataType::CAIF_DataType_e::Float32&&
       dst==CAIF_DataType::CAIF_DataType_e::Int8)
    {
      launch_convert_fp32_to_int8(static_cast<const float*>(DeviceDataRaw()),
                                   result.DeviceDataRaw(),
                                   n,
                                   raw_stream);
      return result;
    }
    // INT8 -> FP32
    if(src==CAIF_DataType::CAIF_DataType_e::Int8&&
       dst==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      launch_convert_int8_to_fp32(DeviceDataRaw(),
                                   static_cast<float*>(result.DeviceDataRaw()),
                                   n,
                                   raw_stream);
      return result;
    }

    THROW_CAIFE("Unsupported dtype conversion");
#else
    THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceTensor::Synchronize()const
{
  if(StreamPtr()!=nullptr)
  {
    Stream().Synchronize();
  }
}

void CAIF_DeviceTensor::FillZero()
{
  if(IsAllocated()==false||SizeBytes()==0)
  {
    return;
  }
  if(StreamPtr()==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }
#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemsetAsync(DeviceDataRaw(),0,SizeBytes(),Stream().Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to zero device tensor");
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::Fill(float value)
{
  if(IsAllocated()==false||TotalElements()==0)
  {
    return;
  }
  if(StreamPtr()==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }
  if(Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("Fill(float) requires FP32 tensor");
  }

#ifdef USE_CAIF_CUDA
  if(value==0.0f)
  {
    cudaError_t status=cudaMemsetAsync(DeviceDataRaw(),0,SizeBytes(),Stream().Handle());
    if(status!=cudaSuccess)
    {
      THROW_CAIFE("Failed to fill device tensor with zeros");
    }
  }
  else
  {
    launch_fill_fp32(DevicePtr(),value,static_cast<int64_t>(TotalElements()),Stream().Handle());
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::Reshape(const std::vector<uint32_t> &new_shape)
{
  size_t new_total=1;
  for(const uint32_t dim:new_shape)
  {
    new_total*=dim;
  }

  if(new_total!=TotalElements())
  {
    THROW_CAIFE("Reshape requires same total elements");
  }

  SetShape(new_shape);
}

CAIF_DeviceTensor CAIF_DeviceTensor::Clone()const
{
  if(IsEmpty()==true)
  {
    return CAIF_DeviceTensor();
  }
  if(Location()==Location_e::Host_e)
  {
    CAIF_DeviceTensor out=UninitializedHost(Shape(),Dtype());
    if(SizeBytes()>0)
    {
      std::memcpy(out.DeviceDataRaw(),DeviceDataRaw(),SizeBytes());
    }
    return out;
  }
  if(StreamPtr()==nullptr)
  {
    return CAIF_DeviceTensor();
  }
  return CloneTo(*_stream);
}

CAIF_DeviceTensor CAIF_DeviceTensor::CloneTo(CAIF_CudaStream &stream)const
{
  if(IsEmpty()==true)
  {
    return CAIF_DeviceTensor();
  }

  CAIF_DeviceTensor tensor(Shape(),stream,true,Dtype());

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(tensor.DeviceDataRaw(),
                                     _device_data,
                                     _size_bytes,
                                     cudaMemcpyDeviceToDevice,
                                     stream.Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to clone device tensor");
  }
#endif

  return tensor;
}

//------------------------------------------------------------------------------
// Host-backed factories
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_DeviceTensor::UninitializedHost(const std::vector<uint32_t> &shape)
{
  return UninitializedHost(shape,CAIF_DataType::CAIF_DataType_e::Float32);
}

CAIF_DeviceTensor CAIF_DeviceTensor::UninitializedHost(const std::vector<uint32_t> &shape,
                                                       CAIF_DataType::CAIF_DataType_e dtype)
{
  CAIF_DeviceTensor tensor;
  tensor.SetShape(shape);
  tensor.SetDtypeInfo(CAIF_DataType(dtype));
  size_t total=1;
  for(const uint32_t dim:shape)
  {
    total*=dim;
  }
  tensor.SetTotalElements(total);
  tensor.SetSizeBytes(tensor.DtypeInfo().StorageSizeBytes(total));
  tensor.SetStreamPtr(nullptr);
  tensor.SetLocation(Location_e::Host_e);
  tensor.SetOwnsData(false);
  if(tensor.SizeBytes()>0)
  {
    tensor.SetDeviceData(::operator new[](tensor.SizeBytes(),
                                          std::align_val_t(_host_alignment)));
    tensor.SetOwnsData(true);
  }
  return tensor;
}

CAIF_DeviceTensor CAIF_DeviceTensor::ZerosHost(const std::vector<uint32_t> &shape)
{
  return ZerosHost(shape,CAIF_DataType::CAIF_DataType_e::Float32);
}

CAIF_DeviceTensor CAIF_DeviceTensor::ZerosHost(const std::vector<uint32_t> &shape,
                                               CAIF_DataType::CAIF_DataType_e dtype)
{
  CAIF_DeviceTensor tensor=UninitializedHost(shape,dtype);
  if(tensor.IsAllocated()==true && tensor.SizeBytes()>0)
  {
    std::memset(tensor.DeviceDataRaw(),0,tensor.SizeBytes());
  }
  return tensor;
}

CAIF_DeviceTensor CAIF_DeviceTensor::ToDevice(CAIF_CudaStream &stream)const
{
  if(Location()==Location_e::Device_e)
  {
    THROW_CAIFE("CAIF_DeviceTensor::ToDevice called on an already-device tensor");
  }
  CAIF_DeviceTensor out(Shape(),stream,true,Dtype());
#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(out.DeviceDataRaw(),
                                     _device_data,
                                     _size_bytes,
                                     cudaMemcpyHostToDevice,
                                     stream.Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("CAIF_DeviceTensor::ToDevice: cudaMemcpyAsync failed");
  }
#endif
  return out;
}

CAIF_DeviceTensor CAIF_DeviceTensor::ToHostLocation()const
{
  if(Location()==Location_e::Host_e)
  {
    THROW_CAIFE("CAIF_DeviceTensor::ToHostLocation called on an already-host tensor");
  }
  CAIF_DeviceTensor out=UninitializedHost(Shape(),Dtype());
#ifdef USE_CAIF_CUDA
  if(StreamPtr()==nullptr)
  {
    THROW_CAIFE("CAIF_DeviceTensor::ToHostLocation: source tensor has no stream");
  }
  cudaError_t status=cudaMemcpyAsync(out.DeviceDataRaw(),
                                     _device_data,
                                     _size_bytes,
                                     cudaMemcpyDeviceToHost,
                                     Stream().Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("CAIF_DeviceTensor::ToHostLocation: cudaMemcpyAsync failed");
  }
  Stream().Synchronize();
#endif
  return out;
}

}//end instance namespace
