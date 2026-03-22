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
 * @file aif_dropout_layer.h
 * @brief Dropout layer implementation for regularization
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_layer.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "caif_framework.h"
#include <random>

namespace instance
{
  /**
   * @brief Dropout layer for regularization
   * 
   * Randomly sets a fraction of input units to 0 during training
   * to prevent overfitting.
   */
  class CAIF_DropoutLayer final:public CAIF_Layer
  {
    public:
      /**
       * @brief Constructor for dropout layer
       * @param framework Reference to CAIF framework instance
       * @param rate Dropout rate (0.0 to 1.0)
       */
      explicit CAIF_DropoutLayer(CAIF_Framework &framework,const float rate=g_caif_default_dropout_rate);
      
      /**
       * @brief Destructor
       */
      ~CAIF_DropoutLayer()override=default;
      
      /**
       * @brief Copy constructor
       * @param other Layer to copy from
       */
      CAIF_DropoutLayer(const CAIF_DropoutLayer &other);
      
      /**
       * @brief Move constructor
       * @param other Layer to move from
       */
      CAIF_DropoutLayer(CAIF_DropoutLayer &&other)noexcept;
      
      /**
       * @brief Copy assignment operator
       * @param other Layer to copy from
       * @return Reference to this layer
       */
      CAIF_DropoutLayer &operator=(const CAIF_DropoutLayer &other);
      
      /**
       * @brief Move assignment operator
       * @param other Layer to move from
       * @return Reference to this layer
       */
      CAIF_DropoutLayer &operator=(CAIF_DropoutLayer &&other)noexcept;

      // Inherited pure virtual methods
      CAIF_Tensor Forward(
                         const CAIF_Tensor &input,
                         const bool training=false
                        )override;
      
      CAIF_Tensor Backward(const CAIF_Tensor &gradient)override;
      
      void Initialize(
                      const std::vector<uint32_t> &input_shape,
                      const uint32_t seed=0
                     )override;
      
      std::vector<uint32_t> CalculateOutputShape(
                                                  const std::vector<uint32_t> &input_shape
                                                 )const override;
      
      CAIF_LayerType_e LayerType()const override{return CAIF_LayerType_e::Dropout;}
      
      std::unique_ptr<CAIF_Layer> Clone()const override;
      
      std::string Description()const override;

      // Getters and setters specific to dropout layer
      float Rate()const{return _rate;}
      void SetRate(const float rate){_rate=rate;}

    protected:
      // Protected members go here

    private:
      float _rate;
      CAIF_Tensor _dropout_mask;
      std::mt19937 _rng;

      /**
       * @brief Generate dropout mask for training
       * @param shape Shape of the mask tensor
       * @return Expected with dropout mask or error message
       */
      CAIF_Tensor GenerateDropoutMask(
                                     const std::vector<uint32_t> &shape
                                    );
  };
}//end instance namespace
