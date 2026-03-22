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
#include "caif_cuda_kernels.h"
#include "ise_lib/ise_out.h"
#include <cstring>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

using namespace instance;

CAIF_DeviceTensor::CAIF_DeviceTensor():_device_data(nullptr),
                                     _shape(),
                                     _total_elements(0),
                                     _size_bytes(0),
                                     _stream(nullptr),
                                     _dtype(CAIF_DataType::CAIF_DataType_e::Float32)
{
}

CAIF_DeviceTensor::CAIF_DeviceTensor(const std::vector<uint32_t> &shape,
                                   CAIF_CudaStream &stream,
                                   bool allocate):_device_data(nullptr),
                                                  _shape(shape),
                                                  _total_elements(1),
                                                  _size_bytes(0),
                                                  _stream(&stream),
                                                  _dtype(CAIF_DataType::CAIF_DataType_e::Float32)
{
  for(const uint32_t dim:_shape)
  {
    _total_elements*=dim;
  }
  _size_bytes=_total_elements*sizeof(float);

  if(allocate==true&&_total_elements>0)
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
                                                                       _dtype(dtype)
{
  for(const uint32_t dim:_shape)
  {
    _total_elements*=dim;
  }
  _size_bytes=_dtype.StorageSizeBytes(_total_elements);

  if(allocate==true&&_total_elements>0)
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
                                                                     _dtype(other._dtype)
{
  other._device_data=nullptr;
  other._total_elements=0;
  other._size_bytes=0;
  other._stream=nullptr;
}

CAIF_DeviceTensor &CAIF_DeviceTensor::operator=(CAIF_DeviceTensor &&other)
{
  if(this!=&other)
  {
    FreeDevice();

    _device_data=other._device_data;
    _shape=std::move(other._shape);
    _total_elements=other._total_elements;
    _size_bytes=other._size_bytes;
    _stream=other._stream;
    _dtype=other._dtype;

    other._device_data=nullptr;
    other._total_elements=0;
    other._size_bytes=0;
    other._stream=nullptr;
  }
  return *this;
}

void CAIF_DeviceTensor::AllocateDevice()
{
#ifdef USE_CAIF_CUDA
  if(_device_data!=nullptr)
  {
    THROW_CAIFE("Device memory already allocated");
  }
  if(_size_bytes==0)
  {
    THROW_CAIFE("Cannot allocate zero-size device tensor");
  }

  // [ALLOC] debug logging commented out -- see GLMDEBUG.md
  // Generates ~16K lines per training step, caused system RAM OOM via log growth
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
                                     _size_bytes,
                                     _stream->Handle());
  if(status!=cudaSuccess)
  {
    size_t err_free=0;
    size_t err_total=0;
    cudaMemGetInfo(&err_free,&err_total);
    std::string err_msg="Failed to allocate CUDA device memory for tensor: ";
    err_msg+=cudaGetErrorString(status);
    err_msg+=" (requested ";
    err_msg+=std::to_string(_size_bytes);
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
#ifdef USE_CAIF_CUDA
  if(_device_data!=nullptr)
  {
    // [FREE] debug logging commented out -- see GLMDEBUG.md
    // if(_size_bytes>=1024*1024)
    // {
    //   ISE_Out::Out()<<"[FREE] "<<(_size_bytes/(1024*1024))<<"MB ("
    //                 <<_size_bytes<<" bytes)"<<std::endl;
    // }
    if(_stream!=nullptr)
    {
      cudaFreeAsync(_device_data,_stream->Handle());
    }
    else
    {
      cudaFree(_device_data);
    }
    _device_data=nullptr;
  }
#endif
}

CAIF_DeviceTensor CAIF_DeviceTensor::Zeros(const std::vector<uint32_t> &shape,
                                         CAIF_CudaStream &stream)
{
  CAIF_DeviceTensor tensor(shape,stream,true);

#ifdef USE_CAIF_CUDA
  if(tensor._device_data!=nullptr&&tensor._size_bytes>0)
  {
    cudaError_t status=cudaMemsetAsync(tensor._device_data,
                                       0,
                                       tensor._size_bytes,
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
  if(tensor._device_data!=nullptr&&tensor._size_bytes>0)
  {
    cudaError_t status=cudaMemsetAsync(tensor._device_data,
                                       0,
                                       tensor._size_bytes,
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

CAIF_DeviceTensor CAIF_DeviceTensor::FromHost(const CAIF_HostTensor &host,CAIF_CudaStream &stream)
{
  if(host.IsEmpty()==true)
  {
    return CAIF_DeviceTensor();
  }

  CAIF_DeviceTensor tensor(host.Shape(),stream,true);

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(tensor._device_data,
                                     host.Data(),
                                     tensor._size_bytes,
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
  cudaError_t status=cudaMemcpyAsync(tensor._device_data,
                                     host_data,
                                     tensor._size_bytes,
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
  cudaError_t status=cudaMemcpyAsync(tensor._device_data,
                                     host_data,
                                     tensor._size_bytes,
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
  if(num_elements!=_total_elements)
  {
    THROW_CAIFE("Host data size does not match tensor size");
  }
  if(_device_data==nullptr)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(_stream==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }
  if(_dtype!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CopyFromHost(float*) requires FP32 tensor; use CopyFromHostRaw for other dtypes");
  }

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(_device_data,
                                     host_data,
                                     _size_bytes,
                                     cudaMemcpyHostToDevice,
                                     _stream->Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy data from host to device tensor");
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::CopyFromHostRaw(const void *host_data,size_t num_bytes)
{
  if(host_data==nullptr)
  {
    THROW_CAIFE("Cannot copy from null host data");
  }
  if(num_bytes!=_size_bytes)
  {
    THROW_CAIFE("Host data byte count does not match tensor storage size");
  }
  if(_device_data==nullptr)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(_stream==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(_device_data,
                                     host_data,
                                     _size_bytes,
                                     cudaMemcpyHostToDevice,
                                     _stream->Handle());
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
  if(_device_data==nullptr)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(_stream==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }
  if(_dtype!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CopyToHost(float*) requires FP32 tensor; use CopyToHostRaw for other dtypes");
  }

#ifdef USE_CAIF_CUDA
  _stream->Synchronize();

  cudaError_t status=cudaMemcpy(host_buffer,
                                _device_data,
                                _size_bytes,
                                cudaMemcpyDeviceToHost);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy data from device tensor to host");
  }
#else
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_DeviceTensor::CopyToHostRaw(void *host_buffer)const
{
  if(host_buffer==nullptr)
  {
    THROW_CAIFE("Cannot copy to null host buffer");
  }
  if(_device_data==nullptr)
  {
    THROW_CAIFE("Device tensor not allocated");
  }
  if(_stream==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }

#ifdef USE_CAIF_CUDA
  _stream->Synchronize();

  cudaError_t status=cudaMemcpy(host_buffer,
                                _device_data,
                                _size_bytes,
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
  if(_dtype!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("ToHost() requires FP32 tensor; convert with To(Float32) first");
  }

  CAIF_HostTensor host=CAIF_HostTensor::Uninitialized(_shape);
  CopyToHost(host.Data());
  return host;
}

CAIF_DeviceTensor CAIF_DeviceTensor::To(CAIF_DataType::CAIF_DataType_e target_dtype)const
{
  try
  {
    if(IsEmpty()==true||_stream==nullptr)
    {
      return CAIF_DeviceTensor();
    }
    if(_dtype==target_dtype)
    {
      return Clone();
    }

    CAIF_DeviceTensor result(_shape,*_stream,true,target_dtype);

#ifdef USE_CAIF_CUDA
    const CAIF_DataType::CAIF_DataType_e src=_dtype.Value();
    const CAIF_DataType::CAIF_DataType_e dst=target_dtype;
    const int n=static_cast<int>(_total_elements);
    cudaStream_t raw_stream=_stream->Handle();

    // FP32 -> FP16
    if(src==CAIF_DataType::CAIF_DataType_e::Float32&&
       dst==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      launch_convert_fp32_to_fp16(static_cast<const float*>(_device_data),
                                  result._device_data,
                                  n,
                                  raw_stream);
      return result;
    }
    // FP16 -> FP32
    if(src==CAIF_DataType::CAIF_DataType_e::Float16&&
       dst==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      launch_convert_fp16_to_fp32(_device_data,
                                  static_cast<float*>(result._device_data),
                                  n,
                                  raw_stream);
      return result;
    }
    // FP32 -> BF16
    if(src==CAIF_DataType::CAIF_DataType_e::Float32&&
       dst==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      launch_convert_fp32_to_bf16(static_cast<const float*>(_device_data),
                                  result._device_data,
                                  n,
                                  raw_stream);
      return result;
    }
    // BF16 -> FP32
    if(src==CAIF_DataType::CAIF_DataType_e::BFloat16&&
       dst==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      launch_convert_bf16_to_fp32(_device_data,
                                  static_cast<float*>(result._device_data),
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
      launch_convert_fp32_to_int8(static_cast<const float*>(_device_data),
                                   result._device_data,
                                   n,
                                   raw_stream);
      return result;
    }
    // INT8 -> FP32
    if(src==CAIF_DataType::CAIF_DataType_e::Int8&&
       dst==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      launch_convert_int8_to_fp32(_device_data,
                                   static_cast<float*>(result._device_data),
                                   n,
                                   raw_stream);
      return result;
    }

    THROW_CAIFE("Unsupported dtype conversion");
#else
    THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceTensor::Synchronize()const
{
  if(_stream!=nullptr)
  {
    _stream->Synchronize();
  }
}

void CAIF_DeviceTensor::Fill(float value)
{
  if(_device_data==nullptr||_total_elements==0)
  {
    return;
  }
  if(_stream==nullptr)
  {
    THROW_CAIFE("Device tensor has no associated stream");
  }
  if(_dtype!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("Fill(float) requires FP32 tensor");
  }

#ifdef USE_CAIF_CUDA
  if(value==0.0f)
  {
    cudaError_t status=cudaMemsetAsync(_device_data,0,_size_bytes,_stream->Handle());
    if(status!=cudaSuccess)
    {
      THROW_CAIFE("Failed to fill device tensor with zeros");
    }
  }
  else
  {
    std::vector<float> host_buffer(_total_elements,value);
    cudaError_t status=cudaMemcpyAsync(_device_data,
                                       host_buffer.data(),
                                       _size_bytes,
                                       cudaMemcpyHostToDevice,
                                       _stream->Handle());
    if(status!=cudaSuccess)
    {
      THROW_CAIFE("Failed to fill device tensor");
    }
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

  if(new_total!=_total_elements)
  {
    THROW_CAIFE("Reshape requires same total elements");
  }

  _shape=new_shape;
}

CAIF_DeviceTensor CAIF_DeviceTensor::Clone()const
{
  if(IsEmpty()==true||_stream==nullptr)
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

  CAIF_DeviceTensor tensor(_shape,stream,true,_dtype.Value());

#ifdef USE_CAIF_CUDA
  cudaError_t status=cudaMemcpyAsync(tensor._device_data,
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
