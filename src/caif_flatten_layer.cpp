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
 * @file aif_flatten_layer.cpp
 * @brief Implementation of the CAIF_FlattenLayer class
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_flatten_layer.h"
#include "ise_lib/ise_out.h"
#include <sstream>
#include <numeric>
#include <iostream> // Added for debug printing
#include <cstddef>  // Added for size_t

namespace instance
{
CAIF_FlattenLayer::CAIF_FlattenLayer(CAIF_Framework &framework):CAIF_Layer(framework),
                                    _original_shape({0}),
                                    _last_input(framework),
                                    _last_output(framework),
                                    _layer_index(0)
{
  SetInitialized(false);
}

CAIF_FlattenLayer::CAIF_FlattenLayer(const CAIF_FlattenLayer &other):CAIF_Layer(other),
                                                                  _original_shape(other._original_shape),
                                                                  _last_input(other._last_input),
                                                                  _last_output(other._last_output),
                                                                  _layer_index(other._layer_index)
{
}

CAIF_FlattenLayer::CAIF_FlattenLayer(CAIF_FlattenLayer &&other):CAIF_Layer(std::move(other)),
                                                             _original_shape(std::move(other._original_shape)),
                                                             _last_input(std::move(other._last_input)),
                                                             _last_output(std::move(other._last_output)),
                                                             _layer_index(other._layer_index)
{
}

CAIF_FlattenLayer &CAIF_FlattenLayer::operator=(const CAIF_FlattenLayer &other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(other);
      _original_shape=other._original_shape;
      _last_input=other._last_input;
      _last_output=other._last_output;
      _layer_index=other._layer_index;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK(); 
}

CAIF_FlattenLayer &CAIF_FlattenLayer::operator=(CAIF_FlattenLayer &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(std::move(other));
      _original_shape=std::move(other._original_shape);
      _last_input=std::move(other._last_input);
      _last_output=std::move(other._last_output);
      _layer_index=other._layer_index;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK(); 
}

CAIF_Tensor CAIF_FlattenLayer::Forward(
                                     const CAIF_Tensor &input,
                                     [[maybe_unused]] const bool training
                                    )
{
  if(IsInitialized()==false)
  {
    ErrorLog()<<"[ERROR] Flatten layer not initialized\n";
    THROW_CAIFE("Flatten layer not initialized");
  }
  
  try
  {
    DbgLog()<<"[DEBUG] Layer "<<_layer_index<<" (Flatten Layer) - Input: "<<input.ToString()<<"\n";
    
    // Store original dimensions for backward pass
    _original_shape=input.Shape();
    
    // Calculate flattened size
    const uint32_t batch_size=input.Shape()[0];
    uint32_t flattened_size=1;
    for(size_t i=1; i<input.Shape().size(); ++i)
    {
      flattened_size*=input.Shape()[i];
    }
    
    // Reshape to [batch_size, flattened_size]
    std::vector<uint32_t> target_shape={batch_size,flattened_size};
    
    // Store input for backward pass
    _last_input=input;
    
    try
    {
      // Reshape returns a new tensor directly
      _last_output=input.Reshape(target_shape);
      return _last_output;
    }
    CAIF_CATCH_BLOCK()
  }
  CAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_FlattenLayer::Backward(const CAIF_Tensor &gradient)
{
  try
  {
    DbgLog()<<"[DEBUG] CAIF_FlattenLayer::Backward - Starting backward pass\n";
    DbgLog()<<"[DEBUG] Input gradient shape: "<<gradient.ToString()<<"\n";
    DbgLog()<<"[DEBUG] Original input dimensions: [";
    for(size_t i=0; i<_original_shape.size(); ++i)
    {
      if(i>0)DbgLog()<<", ";
      DbgLog()<<_original_shape[i];
    }
    DbgLog()<<"]\n";
    
    try
    {
      // Reshape returns a new tensor directly
      return gradient.Reshape(_original_shape);
    }
    CAIF_CATCH_BLOCK();
  }
  CAIF_CATCH_BLOCK(); 
}

void CAIF_FlattenLayer::Initialize(const std::vector<uint32_t> &input_shape,[[maybe_unused]] const uint32_t seed)
{
  try
  {
    if(input_shape.empty())
    {
      THROW_CAIFE("Input shape cannot be empty");
    }
    
    SetInputShape(input_shape);
    
    // Calculate output shape
    SetOutputShape(CalculateOutputShape(input_shape));
    SetInitialized(true);
    
    DbgLog()<<"FlattenLayer::Initialize - input_shape=[";
    const auto &input_shape_ref=InputShape();
    for(size_t i=0; i<input_shape_ref.size(); ++i)
    {
      if(i>0) DbgLog()<<", ";
      DbgLog()<<input_shape_ref[i];
    }
    DbgLog()<<"], output_shape=[";
    const auto &output_shape_ref=OutputShape();
    for(size_t i=0; i<output_shape_ref.size(); ++i)
    {
      if(i>0) DbgLog()<<", ";
      DbgLog()<<output_shape_ref[i];
    }
    DbgLog()<<"]"<<std::endl;
    
    return;
  }
  CAIF_CATCH_BLOCK(); 
}

std::vector<uint32_t> CAIF_FlattenLayer::CalculateOutputShape(const std::vector<uint32_t> &input_shape)const
{
  try
  {
    if(input_shape.empty())
    {
      THROW_CAIFE("Input shape cannot be empty");
    }
    
    // If input is already 1D, return as is
    if(input_shape.size()==1)
    {
      #pragma GCC diagnostic push
      #pragma GCC diagnostic ignored "-Wstringop-overflow"
      std::vector<uint32_t> out_shape;
      out_shape.reserve(1);
      out_shape.push_back(input_shape[0]);
      auto result=out_shape;
      #pragma GCC diagnostic pop
      return result;
    }
    
    // Calculate flattened size (preserve batch dimension if present)
    uint32_t flattened_size=1;
    
    // First dimension is assumed to be batch dimension in neural networks
    // If this is not a batch input, start from index 0
    size_t start_idx=0;
    
    if(input_shape.size()>1)
    {
      // For multi-dimensional input, preserve batch dimension (first dimension)
      start_idx=1;
      
      // Calculate product of all dimensions except batch
      for(size_t i=start_idx; i<input_shape.size(); ++i)
      {
        flattened_size*=input_shape[i];
      }
      
      // Output shape is [batch_size, flattened_size]
      #pragma GCC diagnostic push
      #pragma GCC diagnostic ignored "-Wstringop-overflow"
      std::vector<uint32_t> out_shape(2);
      out_shape[0]=input_shape[0];
      out_shape[1]=flattened_size;
      auto result=out_shape;
      #pragma GCC diagnostic pop
      return result;
    }
    {
      #pragma GCC diagnostic push
      #pragma GCC diagnostic ignored "-Wstringop-overflow"
      std::vector<uint32_t> out_shape;
      out_shape.reserve(input_shape.size());
      for(size_t i=0;i<input_shape.size();++i)
      {
        out_shape.push_back(input_shape[i]);
      }
      auto result=out_shape;
      #pragma GCC diagnostic pop
      return result;
    }
  }
  CAIF_CATCH_BLOCK(); 
}

std::unique_ptr<CAIF_Layer> CAIF_FlattenLayer::Clone()const
{
  // const_cast is safe here because we're creating a new layer that will reference
  // the same framework instance, and the framework itself is not const
  // Framework reference is copied from this layer via copy constructor
  return std::make_unique<CAIF_FlattenLayer>(*this);
}

std::string CAIF_FlattenLayer::Description()const
{
  try
  {
    std::ostringstream oss;
    oss<<"Flatten Layer";
    if(IsInitialized()==true)
    {
      oss<<" (";
      const auto &input_shape_ref=InputShape();
      for(size_t i=0;i<input_shape_ref.size();++i)
      {
        if(i>0)oss<<"x";
        oss<<input_shape_ref[i];
      }
      oss<<" -> ";
      const auto &output_shape_ref=OutputShape();
      for(size_t i=0;i<output_shape_ref.size();++i)
      {
        if(i>0)oss<<"x";
        oss<<output_shape_ref[i];
      }
      oss<<")";
    }
    return oss.str();
  }
  CAIF_CATCH_BLOCK(); 
}
}//end instance namespace
