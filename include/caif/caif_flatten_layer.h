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
 * @file aif_flatten_layer.h
 * @brief Flatten layer for neural networks
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_layer.h"
#include "caif_exception.h"
#include "caif_framework.h"

namespace instance
{
  /**
   * @brief Flatten layer that reshapes multi-dimensional input to 1D
   * 
   * This layer flattens the input tensor while preserving the batch dimension.
   * For example, a tensor of shape [batch, height, width, channels] becomes
   * [batch, height*width*channels].
   */
  class CAIF_FlattenLayer:public CAIF_Layer
  {
    public:
      /**
       * @brief Constructor
       * @param framework Reference to CAIF framework instance
       */
      explicit CAIF_FlattenLayer(CAIF_Framework &framework);
      
      /**
       * @brief Copy constructor
       * @param other Layer to copy from
       */
      CAIF_FlattenLayer(const CAIF_FlattenLayer &other);
      
      /**
       * @brief Move constructor
       * @param other Layer to move from
       */
      CAIF_FlattenLayer(CAIF_FlattenLayer &&other);
      
      /**
       * @brief Destructor
       */
      virtual ~CAIF_FlattenLayer()=default;

      /**
       * @brief Copy assignment operator
       * @param other Layer to copy from
       * @return Reference to this layer
       */
      CAIF_FlattenLayer &operator=(const CAIF_FlattenLayer &other);
      
      /**
       * @brief Move assignment operator
       * @param other Layer to move from
       * @return Reference to this layer
       */
      CAIF_FlattenLayer &operator=(CAIF_FlattenLayer &&other);

      /**
       * @brief Perform forward pass
       * @param input Input tensor to flatten
       * @param training Whether in training mode
       * @return Flattened tensor
       */
      CAIF_Tensor Forward(
                         const CAIF_Tensor &input,
                         const bool training=false
                        )override;

      /**
       * @brief Perform backward pass
       * @param gradient Gradient from next layer
       * @return Reshaped gradient
       */
      CAIF_Tensor Backward(const CAIF_Tensor &gradient)override;

      /**
       * @brief Initialize the layer
       * @param input_shape Shape of input tensor
       * @param seed Random seed (unused for flatten layer)
       */
      void Initialize(
                      const std::vector<uint32_t> &input_shape,
                      const uint32_t seed=0
                     )override;

      /**
       * @brief Calculate output shape given input shape
       * @param input_shape Input tensor shape
       * @return Output shape
       */
      std::vector<uint32_t> CalculateOutputShape(
                                                  const std::vector<uint32_t> &input_shape
                                                 )const override;

      /**
       * @brief Create a copy of this layer
       * @return Unique pointer to cloned layer
       * @note Framework reference is copied from this layer (same framework instance)
       */
      std::unique_ptr<CAIF_Layer> Clone()const override;

      /**
       * @brief Get layer type
       * @return Layer type enum
       */
      CAIF_LayerType_e LayerType()const override{return CAIF_LayerType_e::Flatten;}

      /**
       * @brief Get string description of layer
       * @return Description string
       */
      std::string Description()const override;

    protected:

    private:
      std::vector<uint32_t> _original_shape;  ///< Original input shape for backward pass
      CAIF_Tensor _last_input;                 ///< Cached input for backward pass
      CAIF_Tensor _last_output;                ///< Cached output for backward pass
      uint32_t _layer_index;                  ///< Index of this layer in network
  };
}//end instance namespace
