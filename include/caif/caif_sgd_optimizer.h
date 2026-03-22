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
 * @file aif_sgd_optimizer.h
 * @brief SGD (Stochastic Gradient Descent) optimizer implementation
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_optimizer.h"
#include "caif_framework.h"
#include <vector>

namespace instance
{
  /**
   * @brief SGD (Stochastic Gradient Descent) optimizer implementation
   * 
   * Implements SGD with optional momentum and weight decay.
   * Update rule: param = param - learning_rate * gradient
   * With momentum: velocity = momentum * velocity - learning_rate * gradient
   *                param = param + velocity
   */
  class CAIF_SGDOptimizer:public CAIF_Optimizer
  {
    public:
      /**
       * @brief Constructor with optimizer parameters
       * @param framework Reference to AIF framework instance
       * @param learning_rate Learning rate for parameter updates
       * @param momentum Momentum factor (0.0 for no momentum)
       * @param weight_decay Weight decay factor for L2 regularization
       */
      explicit CAIF_SGDOptimizer(
                                CAIF_Framework &framework,
                                const float learning_rate=g_caif_default_learning_rate,
                                const float momentum=g_caif_default_momentum,
                                const float weight_decay=0.0f
                               );
      
      /**
       * @brief Virtual destructor
       */
      virtual ~CAIF_SGDOptimizer()=default;
      
      /**
       * @brief Copy constructor
       * @param other Optimizer to copy from
       */
      CAIF_SGDOptimizer(const CAIF_SGDOptimizer &other)=default;
      
      /**
       * @brief Move constructor
       * @param other Optimizer to move from
       */
      CAIF_SGDOptimizer(CAIF_SGDOptimizer &&other)noexcept=default;
      
      /**
       * @brief Copy assignment operator
       * @param other Optimizer to copy from
       * @return Reference to this optimizer
       */
      CAIF_SGDOptimizer &operator=(const CAIF_SGDOptimizer &other)=default;
      
      /**
       * @brief Move assignment operator
       * @param other Optimizer to move from
       * @return Reference to this optimizer
       */
      CAIF_SGDOptimizer &operator=(CAIF_SGDOptimizer &&other)noexcept=default;

      // Implementation of pure virtual methods from CAIF_Optimizer
      
      /**
       * @brief Update parameters using SGD algorithm
       * @param parameters Current parameter tensors
       * @param gradients Gradient tensors
       * @return Updated parameters
       */
      std::vector<CAIF_Tensor> UpdateParameters(
                                               const std::vector<CAIF_Tensor> &parameters,
                                               const std::vector<CAIF_Tensor> &gradients
                                              )override;
      
      /**
       * @brief Get optimizer type
       * @return SGD optimizer type
       */
      CAIF_OptimizerType_e OptimizerType()const override;
      
      /**
       * @brief Clone the optimizer (deep copy)
       * @return Unique pointer to cloned optimizer
       * @note Framework reference is copied from this optimizer (same framework instance)
       */
      std::unique_ptr<CAIF_Optimizer> Clone()const override;
      
      /**
       * @brief Reset optimizer state
       */
      void Reset()override;

      /**
       * @brief Apply gradients to parameters
       * @param parameters Parameters to update
       * @param gradients Gradients to apply
       */
      void ApplyGradients(
                          std::vector<CAIF_Tensor> &parameters,
                          const std::vector<CAIF_Tensor> &gradients
                         );
      
      /**
       * @brief Get optimizer state
       * @return Vector of state tensors
       */
      std::vector<CAIF_Tensor> State()const override;
      
      /**
       * @brief Set optimizer state
       * @param state Vector of state tensors
       */
      void SetState(const std::vector<CAIF_Tensor> &state)override;

      // Additional methods
      
      /**
       * @brief Get momentum value
       * @return Current momentum value
       */
      float Momentum()const;
      
      /**
       * @brief Set momentum value
       * @param momentum New momentum value
       */
      void SetMomentum(const float momentum);
      
      /**
       * @brief Get weight decay value
       * @return Current weight decay value
       */
      float WeightDecay()const;
      
      /**
       * @brief Set weight decay value
       * @param weight_decay New weight decay value
       */
      void SetWeightDecay(const float weight_decay);
      
      /**
       * @brief Get optimizer description
       * @return String describing the optimizer configuration
       */
      std::string Description()const;

    private:
      float _momentum;              ///< Momentum factor
      float _weight_decay;          ///< Weight decay factor for L2 regularization
      std::vector<CAIF_Tensor> _velocity;  ///< Velocity tensors for momentum
  };
}//end instance namespace
