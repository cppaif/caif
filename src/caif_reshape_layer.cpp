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

#include "caif_reshape_layer.h"
#include "ise_lib/ise_out.h"
#include <sstream>
#include <numeric>

namespace instance
{
  CAIF_ReshapeLayer::CAIF_ReshapeLayer(CAIF_Framework &framework,const std::vector<uint32_t> &target_shape)
    :CAIF_Layer(framework),
     _target_shape(target_shape),
     _original_shape({0})
  {
    SetInitialized(false);
  }

  CAIF_ReshapeLayer::CAIF_ReshapeLayer(const CAIF_ReshapeLayer &other)
    :CAIF_Layer(other),
     _target_shape(other._target_shape),
     _original_shape(other._original_shape)
  {
  }

  CAIF_ReshapeLayer::CAIF_ReshapeLayer(CAIF_ReshapeLayer &&other)noexcept
    :CAIF_Layer(std::move(other)),
     _target_shape(std::move(other._target_shape)),
     _original_shape(std::move(other._original_shape))
  {
  }

  CAIF_ReshapeLayer &CAIF_ReshapeLayer::operator=(const CAIF_ReshapeLayer &other)
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(other);
      _target_shape=other._target_shape;
      _original_shape=other._original_shape;
    }
    return *this;
  }

  CAIF_ReshapeLayer &CAIF_ReshapeLayer::operator=(CAIF_ReshapeLayer &&other)noexcept
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(std::move(other));
      _target_shape=std::move(other._target_shape);
      _original_shape=std::move(other._original_shape);
    }
    return *this;
  }

  CAIF_Tensor CAIF_ReshapeLayer::Forward(
                                       const CAIF_Tensor &input,
                                       const bool training
                                      )
  {
    (void)training;
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Layer not initialized");
    }

    // Store original shape for backward pass
    _original_shape=input.Shape();

    // Calculate output shape with batch dimension
    std::vector<uint32_t> output_shape=_target_shape;
    output_shape.insert(output_shape.begin(), input.Shape()[0]);

    // Create output tensor with new shape but same data
    return input.Reshape(output_shape);
  }

  CAIF_Tensor CAIF_ReshapeLayer::Backward(const CAIF_Tensor &gradient)
  {
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Layer not initialized");
    }

    // Reshape gradient back to original input shape
    return gradient.Reshape(_original_shape);
  }

  void CAIF_ReshapeLayer::Initialize(
                                    const std::vector<uint32_t> &input_shape,
                                    const uint32_t seed
                                   )
  {
    (void)seed;
    if(input_shape.empty())
    {
      THROW_CAIFE("Input shape cannot be empty");
    }

    // Calculate total elements in input (excluding batch)
    const uint32_t total_input_elements=std::accumulate(
                                                       input_shape.begin()+1,
                                                       input_shape.end(),
                                                       1u,
                                                       std::multiplies<uint32_t>()
                                                      );

    // Calculate total elements in target shape
    const uint32_t total_target_elements=std::accumulate(
                                                        _target_shape.begin(),
                                                        _target_shape.end(),
                                                        1u,
                                                        std::multiplies<uint32_t>()
                                                       );

    // Verify shapes are compatible
    if(total_input_elements!=total_target_elements)
    {
      std::stringstream ss;
      ss<<"Input shape has "<<total_input_elements<<" elements but target shape has "
        <<total_target_elements<<" elements";
      THROW_CAIFE(ss.str().c_str());
    }

    SetInitialized(true);
    return;
  }

  std::vector<uint32_t> CAIF_ReshapeLayer::CalculateOutputShape(
                                                               const std::vector<uint32_t> &input_shape
                                                              )const
  {
    if(input_shape.empty())
    {
      THROW_CAIFE("Input shape cannot be empty");
    }

    // Add batch dimension to target shape
    std::vector<uint32_t> output_shape=_target_shape;
    output_shape.insert(output_shape.begin(), input_shape[0]);

    return output_shape;
  }

  std::unique_ptr<CAIF_Layer> CAIF_ReshapeLayer::Clone()const
  {
    // Framework reference is copied from this layer via copy constructor
  return std::make_unique<CAIF_ReshapeLayer>(*this);
  }

  std::string CAIF_ReshapeLayer::Description()const
  {
    std::stringstream ss;
    ss<<"Reshape Layer: Target shape [";
    for(size_t i=0; i<_target_shape.size(); ++i)
    {
      if(i>0)
      {
        ss<<", ";
      }
      ss<<_target_shape[i];
    }
    ss<<"]";
    return ss.str();
  }
}//end instance namespace
