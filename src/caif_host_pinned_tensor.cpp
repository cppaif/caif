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

#include "caif_host_pinned_tensor.h"
#include "caif_exception.h"

#include "ise_lib/ise_out.h"

#ifdef USE_CAIF_CUDA
#include <cuda_runtime.h>
#endif

#include <string>

namespace instance
{


size_t ComputeBytes(const std::vector<uint32_t> &shape,
                    const CAIF_DataType::CAIF_DataType_e dtype)
{
  size_t elements=1u;
  for(size_t i=0;i<shape.size();++i)
  {
    elements*=static_cast<size_t>(shape[i]);
  }
  return CAIF_DataType(dtype).StorageSizeBytes(elements);
}


CAIF_HostPinnedTensor::CAIF_HostPinnedTensor(const std::vector<uint32_t> &shape,
                                             const CAIF_DataType::CAIF_DataType_e dtype):
                                            _shape(shape),
                                            _dtype(dtype),
                                            _host_ptr(nullptr),
                                            _bytes(ComputeBytes(shape,dtype))
{
  try
  {
    if(_bytes==0u)
    {
      THROW_CAIFE("CAIF_HostPinnedTensor: zero-byte shape");
    }
#ifdef USE_CAIF_CUDA
    cudaError_t status=cudaMallocHost(&_host_ptr,_bytes);
    if(status!=cudaSuccess)
    {
      std::string err_msg="cudaMallocHost failed: ";
      err_msg+=cudaGetErrorString(status);
      err_msg+=" (requested ";
      err_msg+=std::to_string(_bytes);
      err_msg+=" bytes)";
      THROW_CAIFE(err_msg);
    }
#else
    THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
  }
  CAIF_CATCH_BLOCK()
}

CAIF_HostPinnedTensor::~CAIF_HostPinnedTensor()
{
#ifdef USE_CAIF_CUDA
  if(_host_ptr!=nullptr)
  {
    cudaFreeHost(_host_ptr);
    _host_ptr=nullptr;
  }
#endif
}

CAIF_HostPinnedTensor::CAIF_HostPinnedTensor(CAIF_HostPinnedTensor &&other):
                                            _shape(std::move(other._shape)),
                                            _dtype(other._dtype),
                                            _host_ptr(other._host_ptr),
                                            _bytes(other._bytes)
{
  other._host_ptr=nullptr;
  other._bytes=0u;
}

CAIF_HostPinnedTensor &
CAIF_HostPinnedTensor::operator=(CAIF_HostPinnedTensor &&other)
{
  if(this!=&other)
  {
#ifdef USE_CAIF_CUDA
    if(_host_ptr!=nullptr)
    {
      cudaFreeHost(_host_ptr);
    }
#endif
    _shape=std::move(other._shape);
    _dtype=other._dtype;
    _host_ptr=other._host_ptr;
    _bytes=other._bytes;
    other._host_ptr=nullptr;
    other._bytes=0u;
  }
  return *this;
}

CAIF_DeviceTensor
CAIF_HostPinnedTensor::PrefetchToDevice(CAIF_CudaStream &stream)const
{
  try
  {
    if(IsAllocated()==false)
    {
      THROW_CAIFE("PrefetchToDevice: host buffer is not allocated");
    }
    CAIF_DeviceTensor out=CAIF_DeviceTensor::Uninitialized(Shape(),stream,Dtype());
    out.CopyFromHostRaw(HostPtr(),Bytes());
    return out;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_HostPinnedTensor::CopyFromDevice(const CAIF_DeviceTensor &src)
{
  try
  {
    if(IsAllocated()==false)
    {
      THROW_CAIFE("CopyFromDevice: host buffer is not allocated");
    }
    if(src.Shape()!=Shape())
    {
      THROW_CAIFE("CopyFromDevice: source shape does not match host buffer shape");
    }
    if(src.Dtype()!=Dtype())
    {
      THROW_CAIFE("CopyFromDevice: source dtype does not match host buffer dtype");
    }
    src.CopyToHostRaw(HostPtr());
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
