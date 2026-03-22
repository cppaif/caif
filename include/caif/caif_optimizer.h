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
 * @file aif_optimizer.h
 * @brief Base optimizer class for neural network parameter updates
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_constants.h"
#include "caif_tensor.h"
#include "caif_base.h"
#include <vector>
#include <memory>
#include <string>
#include "caif_exception.h"

namespace instance
{
  class CAIF_Framework;
  /**
   * @brief Abstract base class for all optimizers
   * 
   * The CAIF_Optimizer class provides the interface for parameter
   * optimization algorithms like SGD, Adam, etc.
   */
  class CAIF_Optimizer:public CAIF_Base
  {
    public:
      /**
       * @brief Parameterized constructor
       * @param framework Reference to CAIF framework instance
       * @param learning_rate Learning rate for parameter updates
       */
      explicit CAIF_Optimizer(CAIF_Framework &framework,
                             const float learning_rate=g_caif_default_learning_rate
                            )
        :_framework(framework),
         _learning_rate(learning_rate),
         _iteration(0){}
      
      /**
       * @brief Virtual destructor for proper inheritance
       */
      virtual ~CAIF_Optimizer()=default;
      
      /**
       * @brief Copy constructor
       * @param other Optimizer to copy from
       * @note The framework reference is copied from the source (same framework instance)
       */
      CAIF_Optimizer(const CAIF_Optimizer &other):_framework(other._framework),  // Copy framework reference
                                                _learning_rate(other._learning_rate),
                                                _iteration(other._iteration){}
      
      /**
       * @brief Move constructor
       * @param other Optimizer to move from
       */
      CAIF_Optimizer(CAIF_Optimizer &&other):_framework(other._framework),
                                           _learning_rate(other._learning_rate),
                                           _iteration(other._iteration){}
      
      /**
       * @brief Copy assignment operator
       * @param other Optimizer to copy from
       * @return Reference to this optimizer
       */
      CAIF_Optimizer &operator=(const CAIF_Optimizer &other)
      {
        if(this!=&other)
        {
          // Note: reference cannot be reassigned, so we only copy other members
          _learning_rate=other._learning_rate;
          _iteration=other._iteration;
        }
        return *this;
      }
      
      /**
       * @brief Move assignment operator
       * @param other Optimizer to move from
       * @return Reference to this optimizer
       */
      CAIF_Optimizer &operator=(CAIF_Optimizer &&other)noexcept
      {
        if(this!=&other)
        {
          // Note: reference cannot be reassigned, so we only move other members
          _learning_rate=other._learning_rate;
          _iteration=other._iteration;
        }
        return *this;
      }

      // Pure virtual methods
      
      /**
       * @brief Update parameters using gradients
       * @param parameters Current parameter tensors
       * @param gradients Gradient tensors
       * @return Updated parameters
       */
      virtual std::vector<CAIF_Tensor> UpdateParameters(
                                                       const std::vector<CAIF_Tensor> &parameters,
                                                       const std::vector<CAIF_Tensor> &gradients
                                                      )=0;
      
      /**
       * @brief Get optimizer type
       * @return Optimizer type enum
       */
      virtual CAIF_OptimizerType_e OptimizerType()const=0;
      
      /**
       * @brief Clone the optimizer
       * @return A new optimizer instance with the same configuration
       * @note The framework reference is copied from this optimizer (same framework instance)
       */
      virtual std::unique_ptr<CAIF_Optimizer> Clone()const=0;
      
      /**
       * @brief Reset optimizer state
       */
      virtual void Reset()=0;

      /**
       * @brief Get optimizer state for training resumption
       * @return Vector of tensors containing the optimizer state
       */
      virtual std::vector<CAIF_Tensor> State()const=0;
      
      /**
       * @brief Set optimizer state for training resumption
       * @param state Vector of tensors containing the optimizer state
       */
      virtual void SetState(const std::vector<CAIF_Tensor> &state)=0;

      // Virtual methods with default implementations
      
      /**
       * @brief Set learning rate
       * @param learning_rate New learning rate
       */
      virtual void SetLearningRate(const float learning_rate){_learning_rate=learning_rate;}
      
      /**
       * @brief Get current learning rate
       * @return Current learning rate
       */
      virtual float LearningRate()const{return _learning_rate;}
      
      /**
       * @brief Get current iteration count
       * @return Current iteration
       */
      virtual uint32_t Iteration()const{return _iteration;}
      
      /**
       * @brief Increment iteration counter
       */
      virtual void IncrementIteration(){++_iteration;}
      
      /**
       * @brief Set iteration counter
       * @param iteration New iteration value
       */
      virtual void SetIteration(const uint32_t iteration){_iteration=iteration;}

    protected:
      /**
       * @brief Get framework reference
       * @return Reference to CAIF framework instance
       */
      CAIF_Framework &Framework(){return _framework;}
      
      /**
       * @brief Get framework reference (const)
       * @return Const reference to CAIF framework instance
       */
      const CAIF_Framework &Framework()const{return _framework;}
      
    private:
      CAIF_Framework &_framework;
      float _learning_rate;
      uint32_t _iteration;

      // Private members go here
  };
}//end instance namespace
