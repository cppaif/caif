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
 * @file aif_adam_optimizer.h
 * @brief Adam optimizer implementation
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_optimizer.h"
#include "caif_exception.h"
#include "caif_framework.h"
#include <vector>

namespace instance
{
  /**
   * @brief Adam optimizer implementation
   * 
   * Implements Adam (Adaptive Moment Estimation) algorithm with:
   * - Adaptive learning rates
   * - Bias correction
   * - First and second moment estimates
   * 
   * Update rules:
   * m = beta1 * m + (1 - beta1) * grad
   * v = beta2 * v + (1 - beta2) * grad^2
   * m_hat = m / (1 - beta1^t)
   * v_hat = v / (1 - beta2^t)
   * param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
   */
  class CAIF_AdamOptimizer:public CAIF_Optimizer
  {
    public:
      /**
       * @brief Constructor with optimizer parameters
       * @param framework Reference to AIF framework instance
       * @param learning_rate Learning rate for parameter updates
       * @param beta1 Exponential decay rate for first moment estimates
       * @param beta2 Exponential decay rate for second moment estimates
       * @param epsilon Small constant for numerical stability
       * @param weight_decay Weight decay factor for L2 regularization
       */
      explicit CAIF_AdamOptimizer(
                                 CAIF_Framework &framework,
                                 const float learning_rate=g_caif_default_learning_rate,
                                 const float beta1=g_caif_default_beta1,
                                 const float beta2=g_caif_default_beta2,
                                 const float epsilon=g_caif_adam_epsilon,
                                 const float weight_decay=0.0f
                                );
      
      /**
       * @brief Virtual destructor
       */
      virtual ~CAIF_AdamOptimizer()=default;
      
      /**
       * @brief Copy constructor
       * @param other Optimizer to copy from
       */
      CAIF_AdamOptimizer(const CAIF_AdamOptimizer &other)=default;
      
      /**
       * @brief Move constructor
       * @param other Optimizer to move from
       */
      CAIF_AdamOptimizer(CAIF_AdamOptimizer &&other)noexcept=default;
      
      /**
       * @brief Copy assignment operator
       * @param other Optimizer to copy from
       * @return Reference to this optimizer
       */
      CAIF_AdamOptimizer &operator=(const CAIF_AdamOptimizer &other)=default;
      
      /**
       * @brief Move assignment operator
       * @param other Optimizer to move from
       * @return Reference to this optimizer
       */
      CAIF_AdamOptimizer &operator=(CAIF_AdamOptimizer &&other)noexcept=default;

      // Implementation of pure virtual methods from CAIF_Optimizer
      
      /**
       * @brief Update parameters using Adam algorithm
       * @param parameters Current parameter tensors
       * @param gradients Gradient tensors
       * @return Updated parameters
       */
      std::vector<CAIF_Tensor> UpdateParameters(
                                               const std::vector<CAIF_Tensor> &parameters,
                                               const std::vector<CAIF_Tensor> &gradients
                                              )override;

      /**
       * @brief In-place variant to update parameters without allocating new tensors
       */
      void ApplyGradients(std::vector<CAIF_Tensor> &parameters, const std::vector<CAIF_Tensor> &gradients);
      
      /**
       * @brief Get optimizer type
       * @return Adam optimizer type
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
       * @brief Get optimizer state for training resumption
       * @return Vector of tensors containing the optimizer state (m and v)
       */
      std::vector<CAIF_Tensor> State()const override;
      
      /**
       * @brief Set optimizer state for training resumption
       * @param state Vector of tensors containing the optimizer state
       */
      void SetState(const std::vector<CAIF_Tensor> &state)override;

      // Additional methods
      
      /**
       * @brief Get beta1 value
       * @return Current beta1 value
       */
      float Beta1()const;
      
      /**
       * @brief Set beta1 value
       * @param beta1 New beta1 value
       */
      void SetBeta1(const float beta1);
      
      /**
       * @brief Get beta2 value
       * @return Current beta2 value
       */
      float Beta2()const;
      
      /**
       * @brief Set beta2 value
       * @param beta2 New beta2 value
       */
      void SetBeta2(const float beta2);
      
      /**
       * @brief Get epsilon value
       * @return Current epsilon value
       */
      float Epsilon()const;
      
      /**
       * @brief Set epsilon value
       * @param epsilon New epsilon value
       */
      void SetEpsilon(const float epsilon);
      
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
      float _beta1;                 ///< Exponential decay rate for first moment estimates
      float _beta2;                 ///< Exponential decay rate for second moment estimates
      float _epsilon;               ///< Small constant for numerical stability
      float _weight_decay;          ///< Weight decay factor for L2 regularization
      
      std::vector<CAIF_Tensor> _m;   ///< First moment estimates (mean)
      std::vector<CAIF_Tensor> _v;   ///< Second moment estimates (variance)
  };
}//end instance namespace
