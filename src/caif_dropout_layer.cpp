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
 * @file aif_dropout_layer.cpp
 * @brief Implementation of the CAIF_DropoutLayer class
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_dropout_layer.h"
#include "caif_framework.h"
#include <sstream>

namespace instance
{
  CAIF_DropoutLayer::CAIF_DropoutLayer(CAIF_Framework &framework,const float rate)
    :CAIF_Layer(framework),
     _rate(rate),
     _dropout_mask(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _rng(std::random_device{}())
  {
    SetInitialized(false);
  }

  CAIF_DropoutLayer::CAIF_DropoutLayer(const CAIF_DropoutLayer &other)
    :CAIF_Layer(other),
     _rate(other._rate),
     _dropout_mask(other._dropout_mask),
     _rng(other._rng)
  {
  }

  CAIF_DropoutLayer::CAIF_DropoutLayer(CAIF_DropoutLayer &&other)noexcept
    :CAIF_Layer(std::move(other)),
     _rate(other._rate),
     _dropout_mask(std::move(other._dropout_mask)),
     _rng(std::move(other._rng))
  {
  }

  CAIF_DropoutLayer &CAIF_DropoutLayer::operator=(const CAIF_DropoutLayer &other)
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(other);
      _rate=other._rate;
      _dropout_mask=other._dropout_mask;
      _rng=other._rng;
    }
    return *this;
  }

  CAIF_DropoutLayer &CAIF_DropoutLayer::operator=(CAIF_DropoutLayer &&other)noexcept
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(std::move(other));
      _rate=other._rate;
      _dropout_mask=std::move(other._dropout_mask);
      _rng=std::move(other._rng);
    }
    return *this;
  }

  CAIF_Tensor CAIF_DropoutLayer::Forward(
                                       const CAIF_Tensor &input,
                                       const bool training
                                      )
  {
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Dropout layer not initialized");
    }
    
    // During inference, no dropout is applied
    if(training==false||_rate==0.0f)
    {
      _dropout_mask=CAIF_Tensor(input.Framework(),input.Shape(),input.Type());
      // Set mask to all 1s for inference (pass-through)
      float *mask_data=_dropout_mask.MutableData<float>();
      for(uint32_t i=0;i<_dropout_mask.NumElements();++i)
      {
        mask_data[i]=1.0f;
      }
      return input;
    }
    
    // Use backend via framework
    CAIF_Framework &framework=Framework();
    CAIF_Tensor output=framework.DropoutForward(input,_rate,training,_dropout_mask);
    
    return output;
  }

  CAIF_Tensor CAIF_DropoutLayer::Backward(const CAIF_Tensor &gradient)
  {
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Dropout layer not initialized");
    }
    
    // During inference or rate=0, gradient passes through unchanged
    if(_rate==0.0f)
    {
      return gradient;
    }
    
    // Check if we have a valid dropout mask with matching shape
    if(gradient.Shape()!=_dropout_mask.Shape())
    {
      // If shapes don't match, pass gradient through unchanged
      // This handles cases where forward pass was in inference mode
      // but backward pass is called (shouldn't happen in normal training)
      return gradient;
    }
    
    // Use backend via framework
    CAIF_Framework &framework=Framework();
    CAIF_Tensor grad_input=framework.DropoutBackward(gradient,_dropout_mask,_rate);
    
    return grad_input;
  }

  void CAIF_DropoutLayer::Initialize(
                                    const std::vector<uint32_t> &input_shape,
                                    const uint32_t seed
                                   )
  {
    if(input_shape.empty())
    {
      THROW_CAIFE("Input shape cannot be empty");
    }
    
    if(_rate<0.0f||_rate>=1.0f)
    {
      THROW_CAIFE("Dropout rate must be between 0.0 and 1.0");
    }
    
    SetInputShape(input_shape);
    SetOutputShape(input_shape);  // Dropout doesn't change shape
    
    // Seed the random number generator
    _rng.seed(seed);
    
    SetInitialized(true);
  }

  std::vector<uint32_t> CAIF_DropoutLayer::CalculateOutputShape(
                                                                const std::vector<uint32_t> &input_shape
                                                               )const
  {
    if(input_shape.empty())
    {
      THROW_CAIFE("Input shape cannot be empty");
    }
    
    // Dropout doesn't change the shape
    return input_shape;
  }

  std::unique_ptr<CAIF_Layer> CAIF_DropoutLayer::Clone()const
  {
    // Framework reference is copied from this layer via copy constructor
    return std::make_unique<CAIF_DropoutLayer>(*this);
  }

  std::string CAIF_DropoutLayer::Description()const
  {
    std::ostringstream oss;
    oss<<"Dropout Layer (rate="<<_rate<<")";
    return oss.str();
  }

  CAIF_Tensor CAIF_DropoutLayer::GenerateDropoutMask(
                                                   const std::vector<uint32_t> &shape
                                                  )
  {
    // Create mask tensor
    CAIF_Framework &framework=Framework();
    CAIF_Tensor mask(framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
    
    // Get mutable data
    float *mask_data=mask.MutableData<float>();
    
    // Calculate total number of elements
    uint32_t total_elements=1;
    for(uint32_t dim:shape)
    {
      total_elements*=dim;
    }
    
    // Generate random mask
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    for(uint32_t i=0;i<total_elements;++i)
    {
      mask_data[i]=(dist(_rng)>_rate)?1.0f:0.0f;
    }
    
    return mask;
  }
}//end instance namespace
