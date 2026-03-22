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

#include "caif_cpu_tensor_data.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <cstring>

namespace instance
{

CAIF_CPUTensorData::CAIF_CPUTensorData(
                                      const std::vector<uint32_t> &shape,
                                      const CAIF_DataType &dtype
                                     ):_shape(shape),
                                       _dtype(dtype),
                                       _size_bytes(0),
                                       _total_elements(1)
{
  for(const uint32_t dim:_shape)
  {
    _total_elements*=dim;
  }
  _size_bytes=_total_elements*_dtype.ElementSizeBytes();
  _data=std::make_unique<float[]>(_total_elements);
  std::memset(_data.get(),0,_size_bytes);
}

CAIF_CPUTensorData::~CAIF_CPUTensorData()
{
}

const void *CAIF_CPUTensorData::RawData()const
{
  return _data.get();
}

void *CAIF_CPUTensorData::MutableRawData()
{
  return _data.get();
}

const std::vector<uint32_t> &CAIF_CPUTensorData::Shape()const
{
  return _shape;
}

CAIF_DataType CAIF_CPUTensorData::Type()const
{
  return _dtype;
}

size_t CAIF_CPUTensorData::SizeBytes()const
{
  return _size_bytes;
}

void CAIF_CPUTensorData::Fill(double value)
{
  float *data=_data.get();
  for(size_t i=0;i<_total_elements;++i)
  {
    data[i]=static_cast<float>(value);
  }
}

}//end instance namespace


