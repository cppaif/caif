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

#include "caif_host_tensor.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include <algorithm>
#include <cstring>
#include <numeric>

namespace instance
{

CAIF_HostTensor::CAIF_HostTensor():_data(nullptr),
                                 _shape(),
                                 _total_elements(0),
                                 _size_bytes(0)
{
}

CAIF_HostTensor::CAIF_HostTensor(const std::vector<uint32_t> &shape,bool allocate):_shape(shape),
                                                                                  _total_elements(1),
                                                                                  _size_bytes(0)
{
  // Calculate total elements
  for(const uint32_t dim:_shape)
  {
    _total_elements*=dim;
  }
  _size_bytes=_total_elements*sizeof(float);

  if(allocate==true&&_total_elements>0)
  {
    _data=std::make_unique<float[]>(_total_elements);
  }
}

CAIF_HostTensor::~CAIF_HostTensor()
{
  // unique_ptr handles cleanup
}

CAIF_HostTensor::CAIF_HostTensor(CAIF_HostTensor &&other)noexcept:_data(std::move(other._data)),
                                                               _shape(std::move(other._shape)),
                                                               _total_elements(other._total_elements),
                                                               _size_bytes(other._size_bytes)
{
  other._total_elements=0;
  other._size_bytes=0;
}

CAIF_HostTensor &CAIF_HostTensor::operator=(CAIF_HostTensor &&other)noexcept
{
  if(this!=&other)
  {
    _data=std::move(other._data);
    _shape=std::move(other._shape);
    _total_elements=other._total_elements;
    _size_bytes=other._size_bytes;
    other._total_elements=0;
    other._size_bytes=0;
  }
  return *this;
}

CAIF_HostTensor::CAIF_HostTensor(const CAIF_HostTensor &other):_shape(other._shape),
                                                            _total_elements(other._total_elements),
                                                            _size_bytes(other._size_bytes)
{
  if(other._data!=nullptr&&_total_elements>0)
  {
    _data=std::make_unique<float[]>(_total_elements);
    std::memcpy(_data.get(),other._data.get(),_size_bytes);
  }
}

CAIF_HostTensor &CAIF_HostTensor::operator=(const CAIF_HostTensor &other)
{
  if(this!=&other)
  {
    _shape=other._shape;
    _total_elements=other._total_elements;
    _size_bytes=other._size_bytes;

    if(other._data!=nullptr&&_total_elements>0)
    {
      _data=std::make_unique<float[]>(_total_elements);
      std::memcpy(_data.get(),other._data.get(),_size_bytes);
    }
    else
    {
      _data.reset();
    }
  }
  return *this;
}

CAIF_HostTensor CAIF_HostTensor::Zeros(const std::vector<uint32_t> &shape)
{
  CAIF_HostTensor tensor(shape,true);
  if(tensor._data!=nullptr)
  {
    std::memset(tensor._data.get(),0,tensor._size_bytes);
  }
  return tensor;
}

CAIF_HostTensor CAIF_HostTensor::FromData(const float *data,const std::vector<uint32_t> &shape)
{
  if(data==nullptr)
  {
    THROW_CAIFE("Cannot create tensor from null data pointer");
  }

  CAIF_HostTensor tensor(shape,true);
  if(tensor._data!=nullptr)
  {
    std::memcpy(tensor._data.get(),data,tensor._size_bytes);
  }
  return tensor;
}

CAIF_HostTensor CAIF_HostTensor::Uninitialized(const std::vector<uint32_t> &shape)
{
  return CAIF_HostTensor(shape,true);
}

float &CAIF_HostTensor::At(size_t idx)
{
  if(idx>=_total_elements)
  {
    THROW_CAIFE("Host tensor index out of bounds");
  }
  return _data[idx];
}

const float &CAIF_HostTensor::At(size_t idx)const
{
  if(idx>=_total_elements)
  {
    THROW_CAIFE("Host tensor index out of bounds");
  }
  return _data[idx];
}

CAIF_DeviceTensor CAIF_HostTensor::ToDevice(CAIF_CudaStream &stream)const
{
  return CAIF_DeviceTensor::FromHost(*this,stream);
}

void CAIF_HostTensor::Fill(float value)
{
  if(_data==nullptr)
  {
    return;
  }
  std::fill(_data.get(),_data.get()+_total_elements,value);
}

void CAIF_HostTensor::Reshape(const std::vector<uint32_t> &new_shape)
{
  // Calculate new total elements
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

}//end instance namespace
