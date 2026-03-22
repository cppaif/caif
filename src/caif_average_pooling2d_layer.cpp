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
 * @file aif_average_pooling2d_layer.cpp
 * @brief Implementation of the CAIF_AveragePooling2DLayer class
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_average_pooling2d_layer.h"
#include "caif_framework.h"
#include <sstream>
#include <algorithm>

namespace instance
{
  

  CAIF_AveragePooling2DLayer::CAIF_AveragePooling2DLayer(const CAIF_AveragePooling2DLayer &other)
    :CAIF_Layer(other),
     _pool_size(other._pool_size),
     _stride(other._stride),
     _padding(other._padding)
  {
  }

  CAIF_AveragePooling2DLayer::CAIF_AveragePooling2DLayer(CAIF_AveragePooling2DLayer &&other)noexcept
    :CAIF_Layer(std::move(other)),
     _pool_size(other._pool_size),
     _stride(other._stride),
     _padding(other._padding)
  {
  }

  CAIF_AveragePooling2DLayer &CAIF_AveragePooling2DLayer::operator=(const CAIF_AveragePooling2DLayer &other)
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(other);
      _pool_size=other._pool_size;
      _stride=other._stride;
      _padding=other._padding;
    }
    return *this;
  }

  CAIF_AveragePooling2DLayer &CAIF_AveragePooling2DLayer::operator=(CAIF_AveragePooling2DLayer &&other)noexcept
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(std::move(other));
      _pool_size=other._pool_size;
      _stride=other._stride;
      _padding=other._padding;
    }
    return *this;
  }

  CAIF_AveragePooling2DLayer::CAIF_AveragePooling2DLayer(
                                                       CAIF_Framework &framework,
                                                       const uint32_t pool_size,
                                                       const uint32_t stride,
                                                       const uint32_t padding
                                                      ):CAIF_Layer(framework),
                                                        _pool_size(pool_size),
                                                        _stride((stride==0)?pool_size:stride),
                                                        _padding(padding)
  {
    SetInitialized(false);
  }

  CAIF_Tensor CAIF_AveragePooling2DLayer::Forward(const CAIF_Tensor &input,
                                                const bool training)
  {
    (void)training;
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Average pooling layer not initialized");
    }
    
    const auto &input_shape=input.Shape();
    if(input_shape.size()!=4)
    {
      THROW_CAIFE("Average pooling requires 4D input [batch, height, width, channels]");
    }
    
    if(input.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Average pooling currently only supports Float32 data type");
    }
    
    // Store input shape for backward pass (use non-const accessor to modify)
    InputShape()=input_shape;
    
    // Use backend via framework
    CAIF_Framework &framework=Framework();
    CAIF_Tensor output=framework.AveragePooling2D(input,_pool_size,_stride,_padding);
    
    return output;
  }

  CAIF_Tensor CAIF_AveragePooling2DLayer::Backward(const CAIF_Tensor &gradient)
  {
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Average pooling layer not initialized");
    }
    
    // Ensure batch size matches incoming gradient
    std::vector<uint32_t> input_shape_fixed=InputShape();
    const auto &grad_shape=gradient.Shape();
    if(grad_shape.size()==4)
    {
      input_shape_fixed[0]=grad_shape[0];
    }
    
    // Use backend via framework
    CAIF_Framework &framework=Framework();
    CAIF_Tensor input_gradient=framework.AveragePooling2DBackward(gradient,input_shape_fixed,
                                                                  _pool_size,_stride,_padding);
    
    return input_gradient;
  }

  void CAIF_AveragePooling2DLayer::Initialize(
                                             const std::vector<uint32_t> &input_shape,
                                             const uint32_t seed
                                            )
  {
    (void)seed;
    if(input_shape.size()!=4)
    {
      THROW_CAIFE("Average pooling requires 4D input [batch, height, width, channels]");
    }
    
    if(_pool_size==0||_stride==0)
    {
      THROW_CAIFE("Pool size and stride must be greater than 0");
    }
    
    SetInputShape(input_shape);
    
    // Calculate output shape
    SetOutputShape(CalculateOutputShape(input_shape));
    SetInitialized(true);
  }

  std::vector<uint32_t> CAIF_AveragePooling2DLayer::CalculateOutputShape(
                                                                        const std::vector<uint32_t> &input_shape
                                                                       )const
  {
    if(input_shape.size()!=4)
    {
      THROW_CAIFE("Average pooling requires 4D input [batch, height, width, channels]");
    }
    
    const uint32_t batch_size=input_shape[0];
    const uint32_t input_height=input_shape[1];
    const uint32_t input_width=input_shape[2];
    const uint32_t channels=input_shape[3];
    
    const uint32_t output_height=CalculateOutputDim(input_height);
    const uint32_t output_width=CalculateOutputDim(input_width);
    
    return std::vector<uint32_t>{batch_size,output_height,output_width,channels};
  }

  std::unique_ptr<CAIF_Layer> CAIF_AveragePooling2DLayer::Clone()const
  {
    // Framework reference is copied from this layer via copy constructor
    return std::make_unique<CAIF_AveragePooling2DLayer>(*this);
  }

  std::string CAIF_AveragePooling2DLayer::Description()const
  {
    std::ostringstream oss;
    oss<<"Average Pooling 2D Layer (pool_size="<<_pool_size<<", stride="<<_stride<<", padding="<<_padding<<")";
    if(IsInitialized()==true)
    {
      oss<<" (";
      const auto &input_shape=InputShape();
      for(size_t i=0;i<input_shape.size();++i)
      {
        if(i>0)oss<<"x";
        oss<<input_shape[i];
      }
      oss<<" -> ";
      const auto &output_shape=OutputShape();
      for(size_t i=0;i<output_shape.size();++i)
      {
        if(i>0)oss<<"x";
        oss<<output_shape[i];
      }
      oss<<")";
    }
    return oss.str();
  }

  uint32_t CAIF_AveragePooling2DLayer::CalculateOutputDim(const uint32_t input_dim)const
  {
    if(_padding>0)
    {
      // Same padding
      return (input_dim+_stride-1)/_stride;
    }
    else
    {
      // Valid padding
      return (input_dim-_pool_size)/_stride+1;
    }
  }
}//end instance namespace
