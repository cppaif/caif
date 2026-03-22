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
 * @file aif_tensor.cpp
 * @brief Implementation of the CAIF_Tensor class
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_tensor.h"
#include "caif_constants.h"

#include "ise_lib/ise_out.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include "caif_framework.h"
#include "caif_blas.h"

namespace instance
{

std::vector<size_t> CAIF_Tensor::CalculateStrides(const Shape_t &shape)
{
  std::vector<size_t> strides(shape.size());
  if(shape.empty()==false)
  {
    strides.back()=1;
    for(int i=static_cast<int>(shape.size())-2;i>=0;--i)
    {
      strides[i]=strides[i+1]*shape[i+1];
    }
  }
  return strides;
}

std::vector<size_t> CAIF_Tensor::LinearToMultiIndex(size_t linear_idx,
                                                   const Shape_t &shape,
                                                   const std::vector<size_t> &strides)
{
  std::vector<size_t> indices(shape.size());
  size_t temp_idx=linear_idx;
  for(size_t dim=0;dim<shape.size();++dim)
  {
    indices[dim]=temp_idx/strides[dim];
    temp_idx%=strides[dim];
  }
  return indices;
}

size_t CAIF_Tensor::MultiToLinearIndex(const std::vector<size_t> &indices,
                                      const std::vector<size_t> &strides)
{
  size_t linear_idx=0;
  for(size_t dim=0;dim<indices.size();++dim)
  {
    linear_idx+=indices[dim]*strides[dim];
  }
  return linear_idx;
}

uint32_t CAIF_Tensor::ValidateBatchDimension(const Shape_t &shape)
{
  if(shape.empty()==true){THROW_CAIFE("Empty shape");}
  return shape[0];
}

size_t CAIF_Tensor::ElementsPerSample(const Shape_t &shape)
{
  if(shape.empty()==true){THROW_CAIFE("Empty shape");}
  size_t elements=1;
  for(size_t i=1;i<shape.size();++i)  // Skip batch dimension
  {
    elements*=shape[i];
  }
  return elements;
}

uint32_t CAIF_Tensor::ValidateBatchOperation(const CAIF_Tensor &other,
                                            const std::string &operation)const
{
  const uint32_t this_batch=ValidateBatchDimension(_shape);
  const uint32_t other_batch=ValidateBatchDimension(other._shape);
  if(this_batch!=other_batch)
  {
    THROW_CAIFE(("Batch size mismatch for "+operation+": "+
                std::to_string(this_batch)+" != "+
                std::to_string(other_batch)).c_str());
  }
  return this_batch;
}

void CAIF_Tensor::EnsureBackendData()
{
  try
  {
    if(_tensor_data!=nullptr||_shape.empty()==true)
    {
      return;
    }
    auto td=_framework.CreateTensor(_shape,_data_type);
    if(td==nullptr)
    {
      return;
    }
    if(_buffer!=nullptr&&_buffer->empty()==false)
    {
      const size_t bytes=NumElements()*_element_size;
      std::memcpy(td->MutableRawData(),_buffer->data()+_byte_offset,bytes);
    }
    _tensor_data=std::shared_ptr<CAIF_TensorData>(std::move(td));
  }
  catch(...)
  {
    // Silent fallback to host buffer when backend allocation fails
    return;
  }
}

// Default constructor
CAIF_Tensor::CAIF_Tensor(CAIF_Framework &framework):_framework(framework),
                                                 _shape(),
                                                 _data_type(CAIF_DataType::CAIF_DataType_e::Float32),
                                                 _byte_offset(0),
                                                 _element_size(0)
{
}

// Parameterized constructor
CAIF_Tensor::CAIF_Tensor(CAIF_Framework &framework,
                       const Shape_t &shape,
                       const CAIF_DataType &type):_framework(framework)
{
  SetShape(shape);
  SetDataType(type);
  // Prefer backend-owned storage when available
  try
  {
    auto td=_framework.CreateTensor(shape,type);
    if(td!=nullptr)
    {
      _tensor_data=std::shared_ptr<CAIF_TensorData>(std::move(td));
    }
  }
  catch(...)
  {
    // Fallback to host buffer if backend creation fails
  }
  AllocateMemory();
}

// Convenience overload accepting nested enum
CAIF_Tensor::CAIF_Tensor(
                       CAIF_Framework &framework,
                       const Shape_t &shape,
                       const CAIF_DataType::CAIF_DataType_e type
                      ):_framework(framework)
{
  SetShape(shape);
  SetDataType(CAIF_DataType(type));
  try
  {
    auto td=_framework.CreateTensor(shape,_data_type);
    if(td!=nullptr)
    {
      _tensor_data=std::shared_ptr<CAIF_TensorData>(std::move(td));
    }
  }
  catch(...)
  {
  }
  AllocateMemory();
}

// Copy constructor
CAIF_Tensor::CAIF_Tensor(const CAIF_Tensor &other):_framework(other._framework),
                                                _shape(other._shape),
                                                _data_type(other._data_type),
                                                _tensor_data(other._tensor_data),
                                                _buffer(other._buffer),
                                                _byte_offset(other._byte_offset),
                                                _element_size(other._element_size)
{
}

// Move constructor
CAIF_Tensor::CAIF_Tensor(CAIF_Tensor &&other):_framework(other._framework),
                                           _shape(std::move(other._shape)),
                                           _data_type(other._data_type),
                                           _tensor_data(std::move(other._tensor_data)),
                                           _buffer(std::move(other._buffer)),
                                           _byte_offset(other._byte_offset),
                                           _element_size(other._element_size)
{
  other._data_type=CAIF_DataType(CAIF_DataType::CAIF_DataType_e::Float32);
  other._buffer.reset();
  other._byte_offset=0;
  other._element_size=0;
}

// Copy assignment operator
CAIF_Tensor &CAIF_Tensor::operator=(const CAIF_Tensor &other)
{
  if(this!=&other)
  {
    _shape=other._shape;
    _data_type=other._data_type;
    _tensor_data=other._tensor_data;
    _buffer=other._buffer;
    _byte_offset=other._byte_offset;
    _element_size=other._element_size;
  }
  return *this;
}

// Move assignment operator
CAIF_Tensor &CAIF_Tensor::operator=(CAIF_Tensor &&other)
{
  if(this!=&other)
  {
    _shape=std::move(other._shape);
    _data_type=other._data_type;
    _tensor_data=std::move(other._tensor_data);
    _buffer=std::move(other._buffer);
    _byte_offset=other._byte_offset;
    _element_size=other._element_size;
    
    other._data_type=CAIF_DataType(CAIF_DataType::CAIF_DataType_e::Float32);
    other._buffer.reset();
    other._byte_offset=0;
    other._element_size=0;
  }
  return *this;
}

void CAIF_Tensor::SetData(const void *data,size_t size)
{
  if(_tensor_data!=nullptr)
  {
    // Ensure host staging exists then copy into it; device upload is handled explicitly by backend operations
    if(_buffer==nullptr)
    {
      _buffer=std::make_shared<std::vector<uint8_t>>();
    }
    _buffer->resize(size);
    _byte_offset=0;
    if(data!=nullptr)
    {
      std::memcpy(_buffer->data(),data,size);
    }
    return;
  }

  if(data==nullptr)
  {
    ErrorLog()<<"[ERROR] SetData called with null pointer\n";
    throw std::invalid_argument("Data pointer cannot be null");
  }
  
  size_t expected_size=NumElements()*_element_size;
  if(size!=expected_size)
  {
    ErrorLog()<<"[ERROR] SetData size mismatch\n";
    ErrorLog()<<"[ERROR] Expected size: "<<expected_size<<"\n";
    ErrorLog()<<"[ERROR] Provided size: "<<size<<"\n";
    throw std::invalid_argument("Data size does not match tensor dimensions");
  }
  
  if(_buffer==nullptr)
  {
    _buffer=std::make_shared<std::vector<uint8_t>>();
  }
  _buffer->resize(size);
  _byte_offset=0;
  std::memcpy(_buffer->data(),data,size);
}

float CAIF_Tensor::Value(const std::vector<uint32_t> &coordinates)const
{
  if(coordinates.size()!=_shape.size())
  {
    ErrorLog()<<"[ERROR] Coordinate dimension mismatch\n";
    ErrorLog()<<"[ERROR] Expected dimensions: "<<_shape.size()<<"\n";
    ErrorLog()<<"[ERROR] Provided dimensions: "<<coordinates.size()<<"\n";
    THROW_CAIFE("Coordinate dimensions do not match tensor dimensions");
  }
  
  // Validate coordinates are within bounds
  for(size_t i=0; i<coordinates.size(); ++i)
  {
    if(coordinates[i]>=_shape[i])
    {
      THROW_CAIFE(("Coordinate "+
                  std::to_string(coordinates[i])+
                  " is out of bounds for dimension "+
                  std::to_string(i)+
                  " (max: "+
                  std::to_string(_shape[i]-1)+
                  ")").c_str());
    }
  }
  
  // Calculate linear index
  size_t linear_index=0;
  size_t stride=1;
  for(int i=static_cast<int>(_shape.size())-1; i>=0; --i)
  {
    linear_index+=coordinates[i]*stride;
    stride*=_shape[i];
  }
  
  if(linear_index>=NumElements())
  {
    THROW_CAIFE("Calculated index out of bounds");
  }
  
  // Return value converted to float based on data type
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *data_ptr=static_cast<const float*>(Data());
      return data_ptr[linear_index];
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *data_ptr=static_cast<const double*>(Data());
      return static_cast<float>(data_ptr[linear_index]);
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      const int32_t *data_ptr=static_cast<const int32_t*>(Data());
      return static_cast<float>(data_ptr[linear_index]);
    }
    case CAIF_DataType::CAIF_DataType_e::UInt32:
    {
      const uint32_t *data_ptr=static_cast<const uint32_t*>(Data());
      return static_cast<float>(data_ptr[linear_index]);
    }
    case CAIF_DataType::CAIF_DataType_e::Int8:
    {
      const int8_t *data_ptr=static_cast<const int8_t*>(Data());
      return static_cast<float>(data_ptr[linear_index]);
    }
    case CAIF_DataType::CAIF_DataType_e::UInt8:
    {
      const uint8_t *data_ptr=static_cast<const uint8_t*>(Data());
      return static_cast<float>(data_ptr[linear_index]);
    }
    default:
      THROW_CAIFE("Unsupported data type for Value operation");
  }
}

void CAIF_Tensor::SetValue(const std::vector<uint32_t> &coordinates,float value)
{
  if(coordinates.size()!=_shape.size())
  {
    THROW_CAIFE("Coordinate dimensions do not match tensor dimensions");
  }
  
  // Validate coordinates are within bounds
  for(size_t i=0; i<coordinates.size(); ++i)
  {
    if(coordinates[i]>=_shape[i])
    {
      THROW_CAIFE(("Coordinate "+
                  std::to_string(coordinates[i])+
                  " is out of bounds for dimension "+
                  std::to_string(i)+
                  " (max: "+
                  std::to_string(_shape[i]-1)+
                  ")").c_str());
    }
  }
  
  // Calculate linear index
  size_t linear_index=0;
  size_t stride=1;
  for(int i=static_cast<int>(_shape.size())-1; i>=0; --i)
  {
    linear_index+=coordinates[i]*stride;
    stride*=_shape[i];
  }
  
  if(linear_index>=NumElements())
  {
    THROW_CAIFE("Calculated index out of bounds");
  }
  
  // Set value converted from float based on data type
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      float *data_ptr=static_cast<float*>(Data());
      data_ptr[linear_index]=value;
      return;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      double *data_ptr=static_cast<double*>(Data());
      data_ptr[linear_index]=static_cast<double>(value);
      return;
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      int32_t *data_ptr=static_cast<int32_t*>(Data());
      data_ptr[linear_index]=static_cast<int32_t>(value);
      return;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt32:
    {
      uint32_t *data_ptr=static_cast<uint32_t*>(Data());
      data_ptr[linear_index]=static_cast<uint32_t>(value);
      return;
    }
    case CAIF_DataType::CAIF_DataType_e::Int8:
    {
      int8_t *data_ptr=static_cast<int8_t*>(Data());
      data_ptr[linear_index]=static_cast<int8_t>(value);
      return;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt8:
    {
      uint8_t *data_ptr=static_cast<uint8_t*>(Data());
      data_ptr[linear_index]=static_cast<uint8_t>(value);
      return;
    }
    default:
      THROW_CAIFE("Unsupported data type for SetValue operation");
  }
}

void CAIF_Tensor::AllocateMemory()
{
  if(_tensor_data!=nullptr)
  {
    // Backend storage exists; no host allocation required
    return;
  }
  // Attempt to obtain backend storage if available
  if(_shape.empty()==false)
  {
    try
    {
      auto td=_framework.CreateTensor(_shape,_data_type);
      if(td!=nullptr)
      {
        _tensor_data=std::shared_ptr<CAIF_TensorData>(std::move(td));
        return;
      }
    }
    catch(...)
    {
      // Fallback to host allocation below on failure
    }
  }
  if(_shape.empty())
  {
  if(_buffer!=nullptr)
  {
    _buffer->clear();
  }
    return;
  }
  
  if(_buffer==nullptr)
  {
    _buffer=std::make_shared<std::vector<uint8_t>>();
  }
  _buffer->resize(NumElements()*_element_size);
  _byte_offset=0;
  std::fill(_buffer->begin(),_buffer->end(),0);
}

size_t CAIF_Tensor::CalculateStrides()const
{
  if(_shape.empty())
  {
    return 0;
  }
  
  size_t stride=1;
  for(int i=static_cast<int>(_shape.size())-1;i>=0;--i)
  {
    stride*=_shape[i];
  }
  return stride;
}

void CAIF_Tensor::ValidateOperation(const CAIF_Tensor &other,const std::string &operation)const
{
  if(_shape!=other._shape)
  {
    ErrorLog()<<"[ERROR] Shape mismatch in "<<operation<<"\n";
    ErrorLog()<<"[ERROR] This shape: [";
    for(size_t i=0; i<_shape.size(); ++i)
    {
      if(i>0)ErrorLog()<<", ";
      ErrorLog()<<_shape[i];
    }
    ErrorLog()<<"]\n";
    ErrorLog()<<"[ERROR] Other shape: [";
    for(size_t i=0; i<other._shape.size(); ++i)
    {
      if(i>0)ErrorLog()<<", ";
      ErrorLog()<<other._shape[i];
    }
    ErrorLog()<<"]\n";
    throw std::invalid_argument("Tensor shapes are incompatible for "+operation);
  }
  
  if(_data_type!=other._data_type)
  {
    ErrorLog()<<"[ERROR] Data type mismatch in "<<operation<<"\n";
    ErrorLog()<<"[ERROR] This type: "<<static_cast<int>(_data_type.Value())<<"\n";
    ErrorLog()<<"[ERROR] Other type: "<<static_cast<int>(other._data_type.Value())<<"\n";
    throw std::invalid_argument("Tensor data types are incompatible for "+operation);
  }
}

CAIF_Tensor CAIF_Tensor::Reshape(const std::vector<uint32_t> &new_shape)const
{
  ValidateShape(new_shape);
  
  size_t new_elements=1;
  for(uint32_t dim:new_shape)
  {
    new_elements*=dim;
  }
  
  if(NumElements()!=new_elements)
  {
    throw std::invalid_argument("Cannot reshape tensor: element count mismatch");
  }
  
  // Create a new tensor (backend-backed when available) and copy raw bytes
  CAIF_Tensor result(_framework,new_shape,_data_type);
  const size_t bytes=new_elements*_element_size;
  const void *src=Data();
  void *dst=result.Data();
  if(src==nullptr||dst==nullptr)
  {
    THROW_CAIFE("Failed to access tensor data during reshape");
  }
  std::memcpy(dst,src,bytes);
  return result;
}

CAIF_Tensor CAIF_Tensor::Transpose(const std::vector<uint32_t> &permutation)const
{
  if(permutation.size()!=_shape.size())
  {
    throw std::invalid_argument("Permutation size must match tensor dimensions");
  }

  // Validate permutation
  std::vector<bool> used(_shape.size(),false);
  for(uint32_t idx:permutation)
  {
    if(idx>=_shape.size())
    {
      throw std::out_of_range("Invalid permutation index");
    }
    if(used[idx]==true)
    {
      throw std::invalid_argument("Duplicate index in permutation");
    }
    used[idx]=true;
  }

  // Create new shape based on permutation
  std::vector<uint32_t> new_shape(_shape.size());
  for(size_t i=0;i<_shape.size();++i)
  {
    new_shape[i]=_shape[permutation[i]];
  }

  CAIF_Tensor result(_framework,new_shape,_data_type);

  // Calculate strides for both tensors
  const auto src_strides=CalculateStrides(_shape);
  const auto dst_strides=CalculateStrides(new_shape);

  // Copy data with permuted indices
  const size_t num_elements=NumElements();
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      for(size_t src_idx=0;src_idx<num_elements;++src_idx)
      {
        const auto src_indices=LinearToMultiIndex(src_idx,_shape,src_strides);
        
        // Calculate destination index using permutation
        std::vector<size_t> dst_indices(_shape.size());
        for(size_t dim=0;dim<_shape.size();++dim)
        {
          dst_indices[dim]=src_indices[permutation[dim]];
        }
        const size_t dst_idx=MultiToLinearIndex(dst_indices,dst_strides);
        dst[dst_idx]=src[src_idx];
      }
      break;
    }
    default:
      throw std::runtime_error("Transpose not implemented for this data type");
  }

  return result;
}

CAIF_Tensor CAIF_Tensor::Slice(const std::vector<std::pair<uint32_t,uint32_t>> &ranges)const
{
  if(ranges.size()!=_shape.size())
  {
    throw std::invalid_argument("Number of ranges must match tensor dimensions");
  }

  // Validate ranges and create new shape
  std::vector<uint32_t> new_shape;
  for(size_t i=0;i<ranges.size();++i)
  {
    if(ranges[i].first>=ranges[i].second || ranges[i].second>_shape[i])
    {
      throw std::out_of_range("Invalid slice range");
    }
    new_shape.push_back(ranges[i].second-ranges[i].first);
  }

  CAIF_Tensor result(_framework,new_shape,_data_type);

  // Calculate strides for both tensors
  const auto src_strides=CalculateStrides(_shape);
  const auto dst_strides=CalculateStrides(new_shape);

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t result_elements=result.NumElements();

      for(size_t dst_idx=0;dst_idx<result_elements;++dst_idx)
      {
        const auto dst_indices=LinearToMultiIndex(dst_idx,new_shape,dst_strides);
        
        // Calculate source indices by adding range offsets
        std::vector<size_t> src_indices(_shape.size());
        for(size_t dim=0;dim<_shape.size();++dim)
        {
          src_indices[dim]=ranges[dim].first+dst_indices[dim];
        }
        const size_t src_idx=MultiToLinearIndex(src_indices,src_strides);
        dst[dst_idx]=src[src_idx];
      }
      break;
    }
    default:
      throw std::runtime_error("Slicing not implemented for this data type");
  }

  return result;
}

CAIF_Tensor CAIF_Tensor::SliceViewBatch(const std::pair<uint32_t,uint32_t> &batch_range)const
{
  if(_shape.empty())
  {
    throw std::invalid_argument("Tensor has empty shape");
  }
  if(batch_range.first>=batch_range.second||batch_range.second>_shape[0])
  {
    throw std::out_of_range("Invalid batch slice range");
  }
  // Elements per sample (excluding batch)
  const size_t elems_per_sample=std::accumulate(_shape.begin()+1,_shape.end(),1UL,std::multiplies<size_t>());
  const size_t bytes_per_sample=elems_per_sample*_element_size;

  // If backend-owned storage exists, fall back to a sliced copy to honor offsets.
  // This avoids misaligned device views during matmul when offsetting into backend buffers.
  if(_tensor_data!=nullptr)
  {
    std::vector<std::pair<uint32_t,uint32_t>> ranges;
    ranges.reserve(_shape.size());
    ranges.push_back(batch_range);
    for(size_t dim=1; dim<_shape.size(); ++dim)
    {
      ranges.push_back({0,_shape[dim]});
    }
    return Slice(ranges);
  }

  CAIF_Tensor view(_framework);
  view._shape=_shape;
  view._shape[0]=batch_range.second-batch_range.first;
  view._data_type=_data_type;
  view._element_size=_element_size;
  view._buffer=_buffer;
  view._tensor_data=_tensor_data;
  view._byte_offset=_byte_offset+batch_range.first*bytes_per_sample;
  // Propagate and slice batch mapping if present
  if(_batch_index_map!=nullptr && !_batch_index_map->empty())
  {
    auto sub_map=std::make_shared<std::vector<uint32_t>>();
    sub_map->reserve(view._shape[0]);
    for(uint32_t i=batch_range.first;i<batch_range.second;++i)
    {
      sub_map->push_back((*_batch_index_map)[i]);
    }
    view._batch_index_map=sub_map;
  }
  return view;
}

CAIF_Tensor CAIF_Tensor::Add(const CAIF_Tensor &other)const
{
  if(IsDynamicBatch()==true||other.IsDynamicBatch()==true)
  {
    return ExecuteTypedOperation(other,[](auto a,auto b){return a+b;});
  }

  // Non-dynamic case
  if(_shape!=other._shape)
  {
    throw std::invalid_argument("Tensor shapes must match exactly for addition");
  }

  // Use Framework for Float32 - enables GPU acceleration
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32&&
     other._data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    CAIF_Framework& framework=_framework;
    return framework.ElementwiseAdd(*this,other);
  }

  // Fallback for other types
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      const double *other_src=static_cast<const double*>(other.Data());
      double *dst=static_cast<double*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::plus<double>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      const int32_t *src=static_cast<const int32_t*>(Data());
      const int32_t *other_src=static_cast<const int32_t*>(other.Data());
      int32_t *dst=static_cast<int32_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::plus<int32_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt32:
    {
      const uint32_t *src=static_cast<const uint32_t*>(Data());
      const uint32_t *other_src=static_cast<const uint32_t*>(other.Data());
      uint32_t *dst=static_cast<uint32_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::plus<uint32_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int8:
    {
      const int8_t *src=static_cast<const int8_t*>(Data());
      const int8_t *other_src=static_cast<const int8_t*>(other.Data());
      int8_t *dst=static_cast<int8_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::plus<int8_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt8:
    {
      const uint8_t *src=static_cast<const uint8_t*>(Data());
      const uint8_t *other_src=static_cast<const uint8_t*>(other.Data());
      uint8_t *dst=static_cast<uint8_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::plus<uint8_t>());
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type for addition");
  }
  return result;
}

CAIF_Tensor CAIF_Tensor::Subtract(const CAIF_Tensor &other)const
{
  if(IsDynamicBatch()==true||other.IsDynamicBatch()==true)
  {
    return ExecuteTypedOperation(other,[](auto a,auto b){return a-b;});
  }

  // Non-dynamic case
  if(_shape!=other._shape)
  {
    throw std::invalid_argument("Tensor shapes must match exactly for subtraction");
  }

  // Use Framework for Float32 - enables GPU acceleration
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32&&
     other._data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    CAIF_Framework& framework=_framework;
    return framework.ElementwiseSub(*this,other);
  }

  // Fallback for other types
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      const double *other_src=static_cast<const double*>(other.Data());
      double *dst=static_cast<double*>(result.Data());
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=src[i]-other_src[i];
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      const int32_t *src=static_cast<const int32_t*>(Data());
      const int32_t *other_src=static_cast<const int32_t*>(other.Data());
      int32_t *dst=static_cast<int32_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::minus<int32_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt32:
    {
      const uint32_t *src=static_cast<const uint32_t*>(Data());
      const uint32_t *other_src=static_cast<const uint32_t*>(other.Data());
      uint32_t *dst=static_cast<uint32_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::minus<uint32_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int8:
    {
      const int8_t *src=static_cast<const int8_t*>(Data());
      const int8_t *other_src=static_cast<const int8_t*>(other.Data());
      int8_t *dst=static_cast<int8_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::minus<int8_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt8:
    {
      const uint8_t *src=static_cast<const uint8_t*>(Data());
      const uint8_t *other_src=static_cast<const uint8_t*>(other.Data());
      uint8_t *dst=static_cast<uint8_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::minus<uint8_t>());
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type for subtraction");
  }
  return result;
}

CAIF_Tensor CAIF_Tensor::Multiply(const CAIF_Tensor &other)const
{
  if(IsDynamicBatch()==true||other.IsDynamicBatch()==true)
  {
    return ExecuteTypedOperation(other,[](auto a,auto b){return a*b;});
  }

  // Non-dynamic case
  if(_shape!=other._shape)
  {
    throw std::invalid_argument("Tensor shapes must match exactly for multiplication");
  }

  // Use Framework for Float32 - enables GPU acceleration
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32&&
     other._data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    CAIF_Framework& framework=_framework;
    return framework.ElementwiseMul(*this,other);
  }

  // Fallback for other types
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      const double *other_src=static_cast<const double*>(other.Data());
      double *dst=static_cast<double*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::multiplies<double>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      const int32_t *src=static_cast<const int32_t*>(Data());
      const int32_t *other_src=static_cast<const int32_t*>(other.Data());
      int32_t *dst=static_cast<int32_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::multiplies<int32_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt32:
    {
      const uint32_t *src=static_cast<const uint32_t*>(Data());
      const uint32_t *other_src=static_cast<const uint32_t*>(other.Data());
      uint32_t *dst=static_cast<uint32_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::multiplies<uint32_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int8:
    {
      const int8_t *src=static_cast<const int8_t*>(Data());
      const int8_t *other_src=static_cast<const int8_t*>(other.Data());
      int8_t *dst=static_cast<int8_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::multiplies<int8_t>());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt8:
    {
      const uint8_t *src=static_cast<const uint8_t*>(Data());
      const uint8_t *other_src=static_cast<const uint8_t*>(other.Data());
      uint8_t *dst=static_cast<uint8_t*>(result.Data());
      std::transform(src,src+num_elements,other_src,dst,std::multiplies<uint8_t>());
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type for multiplication");
  }
  return result;
}

CAIF_Tensor CAIF_Tensor::Divide(float scalar)const
{
  if(scalar==0.0f)
  {
    throw std::runtime_error("Division by zero");
  }
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    CAIF_Tensor result(_framework,_shape,_data_type);
    const float *src=static_cast<const float*>(Data());
    float *dst=static_cast<float*>(result.Data());
    const size_t num_elements=NumElements();
    const float inv=1.0f/scalar;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(size_t i=0;i<num_elements;++i)
    {
      dst[i]=src[i]*inv;
    }
    return result;
  }
  return ElementWiseScalarOp(scalar,[](auto a,auto b){return a/b;});
}

CAIF_Tensor CAIF_Tensor::Abs()const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=std::abs(src[i]);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      double *dst=static_cast<double*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=std::abs(src[i]);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      const int32_t *src=static_cast<const int32_t*>(Data());
      int32_t *dst=static_cast<int32_t*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=std::abs(src[i]);
      }
      break;
    }
    default:
      throw std::runtime_error("Abs not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::Exp()const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=std::exp(src[i]);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      double *dst=static_cast<double*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=std::exp(src[i]);
      }
      break;
    }
    default:
      throw std::runtime_error("Exp not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::NatLog()const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]<=0.0f)
        {
          throw std::runtime_error("Logarithm of non-positive number");
        }
        dst[i]=std::log(src[i]);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      double *dst=static_cast<double*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]<=0.0)
        {
          throw std::runtime_error("Logarithm of non-positive number");
        }
        dst[i]=std::log(src[i]);
      }
      break;
    }
    default:
      throw std::runtime_error("Log not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::Sqrt()const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]<0.0f)
        {
          throw std::runtime_error("Square root of negative number");
        }
        dst[i]=std::sqrt(src[i]);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      double *dst=static_cast<double*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]<0.0)
        {
          throw std::runtime_error("Square root of negative number");
        }
        dst[i]=std::sqrt(src[i]);
      }
      break;
    }
    default:
      throw std::runtime_error("Sqrt not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::Pow(float exponent)const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=std::pow(src[i],exponent);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *src=static_cast<const double*>(Data());
      double *dst=static_cast<double*>(result.Data());
      
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=std::pow(src[i],static_cast<double>(exponent));
      }
      break;
    }
    default:
      throw std::runtime_error("Pow not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::Sum(uint32_t axis)const
{
  return ReduceAlongAxis(
                        axis,
                        [](auto a,auto b){return a+b;},
                        [](){return 0.0f;}
                       );
}

CAIF_Tensor CAIF_Tensor::Mean(uint32_t axis)const
{
  auto sum=Sum(axis);
  return sum.Multiply(1.0f/static_cast<float>(_shape[axis]));
}

CAIF_Tensor CAIF_Tensor::Max(uint32_t axis)const
{
  return ReduceAlongAxis(
                        axis,
                        [](auto a,auto b){return std::max(a,b);},
                        [](){return std::numeric_limits<float>::lowest();}
                       );
}

CAIF_Tensor CAIF_Tensor::Min(uint32_t axis)const
{
  return ReduceAlongAxis(
                        axis,
                        [](auto a,auto b){return std::min(a,b);},
                        [](){return std::numeric_limits<float>::max();}
                       );
}

CAIF_Tensor CAIF_Tensor::ReLU()const
{
  return ApplyActivation([](auto x){return std::max(0.0f,x);});
}

CAIF_Tensor CAIF_Tensor::Sigmoid()const
{
  return ApplyActivation([](auto x){return 1.0f/(1.0f+std::exp(-x));});
}

CAIF_Tensor CAIF_Tensor::Tanh()const
{
  return ApplyActivation([](auto x){return std::tanh(x);});
}

CAIF_Tensor CAIF_Tensor::Softmax()const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      
      if(_shape.empty())
      {
        return result;
      }
      
      const size_t classes=static_cast<size_t>(_shape.back());
      size_t rows=0;
      if(classes>0)
      {
        rows=num_elements/classes;
      }
      else
      {
        rows=0;
      }
      
      for(size_t r=0;r<rows;++r)
      {
        const size_t base=r*classes;
        
        // Find maximum for numerical stability (per row)
        float max_val=src[base];
        for(size_t i=1;i<classes;++i)
        {
          const float v=src[base+i];
          if(v>max_val)
          {
            max_val=v;
          }
        }
        
        // Compute exp(x - max) and sum (per row)
        float sum=0.0f;
        for(size_t i=0;i<classes;++i)
        {
          const float e=std::exp(src[base+i]-max_val);
          dst[base+i]=e;
          sum+=e;
        }
        
        // Normalize (per row)
        float inv_sum=0.0f;
        if(sum>0.0f)
        {
          inv_sum=1.0f/sum;
        }
        else
        {
          inv_sum=0.0f;
        }
        for(size_t i=0;i<classes;++i)
        {
          dst[base+i]*=inv_sum;
        }
      }
      break;
    }
    default:
      throw std::runtime_error("Softmax not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::LeakyReLU(float alpha)const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]>0.0f)
        {
          dst[i]=src[i];
        }
        else
        {
          dst[i]=alpha*src[i];
        }
      }
      break;
    }
    default:
      throw std::runtime_error("LeakyReLU not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::ELU(float alpha)const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=(src[i]>0.0f)?src[i]:alpha*(std::exp(src[i])-1.0f);
      }
      break;
    }
    default:
      throw std::runtime_error("ELU not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::GELU()const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      const float sqrt_2_pi=std::sqrt(2.0f/g_caif_pi);
      for(size_t i=0;i<num_elements;++i)
      {
        const float x=src[i];
        const float tanh_term=std::tanh(sqrt_2_pi*(x+0.044715f*x*x*x));
        dst[i]=0.5f*x*(1.0f+tanh_term);
      }
      break;
    }
    default:
      throw std::runtime_error("GELU not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::Swish()const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      for(size_t i=0;i<num_elements;++i)
      {
        const float x=src[i];
        float sigmoid;
        if(x>=0.0f)
        {
          const float z=std::exp(-x);
          sigmoid=1.0f/(1.0f+z);
        }
        else
        {
          const float z=std::exp(x);
          sigmoid=z/(1.0f+z);
        }
        dst[i]=x*sigmoid;
      }
      break;
    }
    default:
      throw std::runtime_error("Swish not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::Linear()const
{
  // Linear activation is identity function
  return *this;
}

CAIF_Tensor CAIF_Tensor::Multiply(float scalar)const
{
  // Use Framework for Float32 - enables GPU acceleration
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    CAIF_Framework& framework=_framework;
    return framework.ElementwiseMulScalar(*this,scalar);
  }
  return ElementWiseScalarOp(scalar,[](auto a,auto b){return a*b;});
}

CAIF_Tensor CAIF_Tensor::Add(float scalar)const
{
  // Use Framework for Float32 - enables GPU acceleration
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    CAIF_Framework& framework=_framework;
    return framework.ElementwiseAddScalar(*this,scalar);
  }
  return ElementWiseScalarOp(scalar,[](auto a,auto b){return a+b;});
}

CAIF_Tensor CAIF_Tensor::Divide(const CAIF_Tensor &other)const
{
  // Use Framework for Float32 - enables GPU acceleration
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32&&
     other._data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32&&
     _shape==other._shape)
  {
    CAIF_Framework& framework=_framework;
    return framework.ElementwiseDiv(*this,other);
  }
  return ExecuteTypedOperation(other,[](auto a,auto b){return a/b;});
}

CAIF_Tensor CAIF_Tensor::LinearDerivative(const CAIF_Tensor &gradient)const
{
  // Linear derivative is 1 - gradient passes through unchanged
  ValidateOperation(gradient,"Linear derivative");
  return CAIF_Tensor(gradient);
}

CAIF_Tensor CAIF_Tensor::ReLUDerivative(const CAIF_Tensor &gradient)const
{
  ValidateOperation(gradient,"ReLU derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      const float *grad=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]>0.0f)
        {
          dst[i]=grad[i];
        }
        else
        {
          dst[i]=0.0f;
        }
      }
      break;
    }
    default:
      throw std::runtime_error("ReLU derivative not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::SigmoidDerivative(const CAIF_Tensor &gradient)const
{
  ValidateOperation(gradient,"Sigmoid derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      const float *grad=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      // Previous implementation (with clamping) kept for reference:
      // for(size_t i=0;i<num_elements;++i)
      // {
      //   const float eps=1e-6f;
      //   float y=src[i];
      //   if(y<eps)
      //   {
      //     y=eps;
      //   }
      //   if(y>1.0f-eps)
      //   {
      //     y=1.0f-eps;
      //   }
      //   dst[i]=grad[i]*y*(1.0f-y);
      // }
      // Unclamped version to match PyTorch behavior:
      for(size_t i=0;i<num_elements;++i)
      {
        const float y=src[i];
        dst[i]=grad[i]*y*(1.0f-y);
      }
      break;
    }
    default:
      throw std::runtime_error("Sigmoid derivative not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::TanhDerivative(const CAIF_Tensor &gradient)const
{
  ValidateOperation(gradient,"Tanh derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());  // This contains tanh(x)
      const float *grad=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=grad[i]*(1.0f-src[i]*src[i]);  // src[i] is already tanh(x)
      }
      break;
    }
    default:
      throw std::runtime_error("Tanh derivative not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::SoftmaxDerivative(const CAIF_Tensor &gradient)const
{
  ValidateOperation(gradient,"Softmax derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *s=static_cast<const float*>(Data());  // Assume input already softmax probabilities
      const float *g=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      
      if(_shape.empty())
      {
        return result;
      }
      
      const size_t classes=static_cast<size_t>(_shape.back());
      size_t rows=0;
      if(classes>0)
      {
        rows=num_elements/classes;
      }
      else
      {
        rows=0;
      }
      
      for(size_t r=0;r<rows;++r)
      {
        const size_t base=r*classes;
        
        // dot = sum_j g_j * s_j (per row)
        float dot=0.0f;
        for(size_t j=0;j<classes;++j)
        {
          dot+=g[base+j]*s[base+j];
        }
        
        // dst_i = s_i * (g_i - dot)
        for(size_t i=0;i<classes;++i)
        {
          dst[base+i]=s[base+i]*(g[base+i]-dot);
        }
      }
      break;
    }
    default:
      throw std::runtime_error("Softmax derivative not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::LeakyReLUDerivative(const CAIF_Tensor &gradient,float alpha)const
{
  ValidateOperation(gradient,"LeakyReLU derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      const float *grad=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]>0.0f)
        {
          dst[i]=grad[i];
        }
        else
        {
          dst[i]=alpha*grad[i];
        }
      }
      break;
    }
    default:
      throw std::runtime_error("LeakyReLU derivative not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::ELUDerivative(const CAIF_Tensor &gradient,float alpha)const
{
  ValidateOperation(gradient,"ELU derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      const float *grad=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      for(size_t i=0;i<num_elements;++i)
      {
        if(src[i]>0.0f)
        {
          dst[i]=grad[i];
        }
        else
        {
          dst[i]=grad[i]*alpha*std::exp(src[i]);
        }
      }
      break;
    }
    default:
      throw std::runtime_error("ELU derivative not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::GELUDerivative(const CAIF_Tensor &gradient)const
{
  ValidateOperation(gradient,"GELU derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      const float *grad=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      const float sqrt_2_pi=std::sqrt(2.0f/g_caif_pi);
      for(size_t i=0;i<num_elements;++i)
      {
        const float x=src[i];
        const float x3=x*x*x;
        const float inner_term=sqrt_2_pi*(x+0.044715f*x3);
        const float tanh_val=std::tanh(inner_term);
        const float sech2=1.0f-tanh_val*tanh_val;
        const float inner_derivative=sqrt_2_pi*(1.0f+3.0f*0.044715f*x*x);
        dst[i]=grad[i]*(0.5f*(1.0f+tanh_val)+0.5f*x*sech2*inner_derivative);
      }
      break;
    }
    default:
      throw std::runtime_error("GELU derivative not implemented for this data type");
  }
  
  return result;
}

CAIF_Tensor CAIF_Tensor::SwishDerivative(const CAIF_Tensor &gradient)const
{
  ValidateOperation(gradient,"Swish derivative");
  CAIF_Tensor result(_framework,_shape,_data_type);
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      const float *grad=static_cast<const float*>(gradient.Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t num_elements=NumElements();
      for(size_t i=0;i<num_elements;++i)
      {
        const float x=src[i];
        float sigmoid;
        if(x>=0.0f)
        {
          const float z=std::exp(-x);
          sigmoid=1.0f/(1.0f+z);
        }
        else
        {
          const float z=std::exp(x);
          sigmoid=z/(1.0f+z);
        }
        dst[i]=grad[i]*(sigmoid+x*sigmoid*(1.0f-sigmoid));
      }
      break;
    }
    default:
      throw std::runtime_error("Swish derivative not implemented for this data type");
  }
  
  return result;
}

bool CAIF_Tensor::FillWithRandom(std::mt19937 &generator)
{
  if(!IsValid())
  {
    return false;
  }
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      float *data=static_cast<float*>(Data());
      const size_t num_elements=NumElements();
      std::uniform_real_distribution<float> dist(-1.0f,1.0f);
      for(size_t i=0;i<num_elements;++i)
      {
        data[i]=dist(generator);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      double *data=static_cast<double*>(Data());
      const size_t num_elements=NumElements();
      std::uniform_real_distribution<double> dist(-1.0,1.0);
      for(size_t i=0;i<num_elements;++i)
      {
        data[i]=dist(generator);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      int32_t *data=static_cast<int32_t*>(Data());
      const size_t num_elements=NumElements();
      std::uniform_int_distribution<int32_t> dist(std::numeric_limits<int32_t>::min(),
                                                std::numeric_limits<int32_t>::max());
      for(size_t i=0;i<num_elements;++i)
      {
        data[i]=dist(generator);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt32:
    {
      uint32_t *data=static_cast<uint32_t*>(Data());
      const size_t num_elements=NumElements();
      std::uniform_int_distribution<uint32_t> dist(0,std::numeric_limits<uint32_t>::max());
      for(size_t i=0;i<num_elements;++i)
      {
        data[i]=dist(generator);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int8:
    {
      int8_t *data=static_cast<int8_t*>(Data());
      const size_t num_elements=NumElements();
      std::uniform_int_distribution<int16_t> dist(std::numeric_limits<int8_t>::min(),
                                                std::numeric_limits<int8_t>::max());
      for(size_t i=0;i<num_elements;++i)
      {
        data[i]=static_cast<int8_t>(dist(generator));
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::UInt8:
    {
      uint8_t *data=static_cast<uint8_t*>(Data());
      const size_t num_elements=NumElements();
      std::uniform_int_distribution<uint16_t> dist(0,std::numeric_limits<uint8_t>::max());
      for(size_t i=0;i<num_elements;++i)
      {
        data[i]=static_cast<uint8_t>(dist(generator));
      }
      break;
    }
    default:
      return false;
  }
  
  return true;
}

bool CAIF_Tensor::FillWithRandom()
{
  std::random_device rd;
  std::mt19937 generator(rd());
  return FillWithRandom(generator);
}

std::string CAIF_Tensor::ToString()const
{
  std::ostringstream oss;
  oss<<"CAIF_Tensor(shape=[";
  
  for(size_t i=0;i<_shape.size();++i)
  {
    if(i>0)oss<<", ";
    oss<<_shape[i];
  }
  
  oss<<"], dtype=";
  
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:oss<<"Float32";break;
    case CAIF_DataType::CAIF_DataType_e::Float64:oss<<"Float64";break;
    case CAIF_DataType::CAIF_DataType_e::Int8:oss<<"Int8";break;
    case CAIF_DataType::CAIF_DataType_e::Int16:oss<<"Int16";break;
    case CAIF_DataType::CAIF_DataType_e::Int32:oss<<"Int32";break;
    case CAIF_DataType::CAIF_DataType_e::Int64:oss<<"Int64";break;
    case CAIF_DataType::CAIF_DataType_e::UInt8:oss<<"UInt8";break;
    case CAIF_DataType::CAIF_DataType_e::UInt16:oss<<"UInt16";break;
    case CAIF_DataType::CAIF_DataType_e::UInt32:oss<<"UInt32";break;
    case CAIF_DataType::CAIF_DataType_e::UInt64:oss<<"UInt64";break;
    case CAIF_DataType::CAIF_DataType_e::Bool:oss<<"Bool";break;
    default:oss<<"Unknown";break;
  }
  
  oss<<", elements="<<NumElements()<<")";
  
  return oss.str();
}

void CAIF_Tensor::SaveToFile(const std::string &filename)const
{
  std::ofstream file(filename,std::ios::binary);
  if(!file.is_open())
  {
    throw std::runtime_error("Cannot open file for writing: "+filename);
  }
  
  // Write header information
  uint32_t num_dims=static_cast<uint32_t>(_shape.size());
  file.write(reinterpret_cast<const char*>(&num_dims),sizeof(num_dims));
  
  for(uint32_t dim:_shape)
  {
    file.write(reinterpret_cast<const char*>(&dim),sizeof(dim));
  }
  
  uint8_t dtype=static_cast<uint8_t>(_data_type.Value());
  file.write(reinterpret_cast<const char*>(&dtype),sizeof(dtype));
  
  // Write data
  const size_t data_size=NumElements()*_element_size;
  file.write(reinterpret_cast<const char*>(&data_size),sizeof(data_size));
  if(data_size>0)
  {
    const uint8_t *src=nullptr;
    if(_buffer!=nullptr)
    {
      src=_buffer->data()+_byte_offset;
    }
    if(src==nullptr)
    {
      throw std::runtime_error("Tensor has no storage to save");
    }
    file.write(reinterpret_cast<const char*>(src),data_size);
  }
  
  if(!file.good())
  {
    throw std::runtime_error("Error writing to file: "+filename);
  }
}

CAIF_Tensor CAIF_Tensor::LoadFromFile(CAIF_Framework &framework,const std::string &filename)
{
  std::ifstream file(filename,std::ios::binary);
  if(!file.is_open())
  {
    THROW_CAIFE(("Cannot open file for reading: "+filename).c_str());
  }
  
  try
  {
    // Read header information
    uint32_t num_dims;
    file.read(reinterpret_cast<char*>(&num_dims),sizeof(num_dims));
    
    std::vector<uint32_t> shape(num_dims);
    for(uint32_t &dim:shape)
    {
      file.read(reinterpret_cast<char*>(&dim),sizeof(dim));
    }
    
    uint8_t dtype;
    file.read(reinterpret_cast<char*>(&dtype),sizeof(dtype));
    
    // Read data
    size_t data_size;
    file.read(reinterpret_cast<char*>(&data_size),sizeof(data_size));
    
    CAIF_Tensor tensor(framework,shape,static_cast<CAIF_DataType::CAIF_DataType_e>(dtype));
    const size_t expected_bytes=tensor.NumElements()*tensor._element_size;
    
    if(data_size!=expected_bytes)
    {
      THROW_CAIFE("Data size mismatch in file");
    }
    
    if(data_size>0)
    {
      if(tensor._buffer==nullptr)
      {
        tensor._buffer=std::make_shared<std::vector<uint8_t>>();
      }
      tensor._buffer->resize(data_size);
      tensor._byte_offset=0;
      file.read(reinterpret_cast<char*>(tensor._buffer->data()),data_size);
    }
    
    if(!file.good())
    {
      THROW_CAIFE(("Error reading from file: "+filename).c_str());
    }
    
    return tensor;
  }
  catch(const std::exception &e)
  {
    THROW_CAIFE((std::string("Error loading tensor: ")+e.what()).c_str());
  }
}

CAIF_Tensor CAIF_Tensor::MatMul(const CAIF_Tensor &other)const
{
  // Use framework-backed matmul for GPU awareness and backend flexibility
  // This allows CUDA backend to accelerate matrix multiplication when available
  if(_data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32&&
     other._data_type.Value()==CAIF_DataType::CAIF_DataType_e::Float32&&
     _shape.size()==g_caif_2d_matrix_dimensions&&
     other._shape.size()==g_caif_2d_matrix_dimensions)
  {
    CAIF_Framework& framework=_framework;
    return framework.MatrixMultiply(*this,other);
  }

  // Fallback to typed matmul for other data types
  return ExecuteTypedMatMul(other);
}

CAIF_Tensor CAIF_Tensor::ExecuteTypedMatMul(const CAIF_Tensor &other)const
{
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      return MatMulWithBroadcast<float>(other);
    case CAIF_DataType::CAIF_DataType_e::Float64:
      return MatMulWithBroadcast<double>(other);
    case CAIF_DataType::CAIF_DataType_e::Int32:
      return MatMulWithBroadcast<int32_t>(other);
    case CAIF_DataType::CAIF_DataType_e::UInt32:
      return MatMulWithBroadcast<uint32_t>(other);
    case CAIF_DataType::CAIF_DataType_e::Int8:
      return MatMulWithBroadcast<int8_t>(other);
    case CAIF_DataType::CAIF_DataType_e::UInt8:
      return MatMulWithBroadcast<uint8_t>(other);
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

CAIF_Tensor CAIF_Tensor::Convolution2D(const CAIF_Tensor &kernel,uint32_t stride_h,uint32_t stride_w)const
{
  // Validate input tensor shape (batch, height, width, channels)
  if(_shape.size()!=4)
  {
    throw std::invalid_argument("Input tensor must be 4D (batch, height, width, channels)");
  }

  // Validate kernel shape (kernel_h, kernel_w, in_channels, out_channels)
  if(kernel._shape.size()!=4)
  {
    throw std::invalid_argument("Kernel must be 4D (kernel_h, kernel_w, in_channels, out_channels)");
  }

  // Validate channel dimensions match
  if(_shape[3]!=kernel._shape[2])
  {
    throw std::invalid_argument("Input channels must match kernel input channels");
  }

  const uint32_t batch_size=_shape[0];
  const uint32_t input_h=_shape[1];
  const uint32_t input_w=_shape[2];
  const uint32_t in_channels=_shape[3];
  
  const uint32_t kernel_h=kernel._shape[0];
  const uint32_t kernel_w=kernel._shape[1];
  const uint32_t out_channels=kernel._shape[3];

  // Calculate output dimensions
  const uint32_t output_h=(input_h-kernel_h)/stride_h+1;
  const uint32_t output_w=(input_w-kernel_w)/stride_w+1;

  // Create output tensor
  Shape_t output_shape={batch_size,output_h,output_w,out_channels};
  CAIF_Tensor output(_framework,output_shape,_data_type);

  // Get data pointers
  const float *input_data=static_cast<const float*>(Data());
  const float *kernel_data=static_cast<const float*>(kernel.Data());
  float *output_data=static_cast<float*>(output.Data());

  // Perform convolution
  const size_t input_hw=input_h*input_w;
  const size_t output_hw=output_h*output_w;
  // removed unused kernel_hw to avoid warning

  for(uint32_t b=0;b<batch_size;++b)
  {
    for(uint32_t oh=0;oh<output_h;++oh)
    {
      for(uint32_t ow=0;ow<output_w;++ow)
      {
        for(uint32_t oc=0;oc<out_channels;++oc)
        {
          float sum=0.0f;
          
          for(uint32_t kh=0;kh<kernel_h;++kh)
          {
            for(uint32_t kw=0;kw<kernel_w;++kw)
            {
              const uint32_t ih=oh*stride_h+kh;
              const uint32_t iw=ow*stride_w+kw;
              
              for(uint32_t ic=0;ic<in_channels;++ic)
              {
                const size_t input_idx=b*input_hw*in_channels+
                                     ih*input_w*in_channels+
                                     iw*in_channels+
                                     ic;
                
                const size_t kernel_idx=kh*kernel_w*in_channels*out_channels+
                                      kw*in_channels*out_channels+
                                      ic*out_channels+
                                      oc;
                
                sum+=input_data[input_idx]*kernel_data[kernel_idx];
              }
            }
          }
          
          const size_t output_idx=b*output_hw*out_channels+
                                 oh*output_w*out_channels+
                                 ow*out_channels+
                                 oc;
          output_data[output_idx]=sum;
        }
      }
    }
  }

  return output;
}

}//end instance namespace
