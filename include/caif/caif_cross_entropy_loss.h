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
 * @file aif_cross_entropy_loss.h
 * @brief Cross Entropy loss function implementation
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_loss_function.h"
#include "caif_tensor_backend.h"

namespace instance
{
  /**
   * @brief Cross Entropy loss function implementation
   * 
   * Computes cross entropy loss: -sum(targets * log(predictions))
   * Typically used for multi-class classification with softmax outputs.
   */
  class CAIF_CrossEntropyLoss:public CAIF_LossFunction
  {
    public:
      /**
       * @brief Constructor with epsilon parameter
       * @param epsilon Small value to prevent log(0) and division by 0
       */
      explicit CAIF_CrossEntropyLoss(const float epsilon=1e-7f);
      
      /**
       * @brief Virtual destructor
       */
      virtual ~CAIF_CrossEntropyLoss()=default;
      
      /**
       * @brief Copy constructor
       * @param other Loss function to copy from
       */
      CAIF_CrossEntropyLoss(const CAIF_CrossEntropyLoss &other)=default;
      
      /**
       * @brief Move constructor
       * @param other Loss function to move from
       */
      CAIF_CrossEntropyLoss(CAIF_CrossEntropyLoss &&other)noexcept=default;
      
      /**
       * @brief Copy assignment operator
       * @param other Loss function to copy from
       * @return Reference to this loss function
       */
      CAIF_CrossEntropyLoss &operator=(const CAIF_CrossEntropyLoss &other)=default;
      
      /**
       * @brief Move assignment operator
       * @param other Loss function to move from
       * @return Reference to this loss function
       */
      CAIF_CrossEntropyLoss &operator=(CAIF_CrossEntropyLoss &&other)noexcept=default;

      // Implementation of pure virtual methods from CAIF_LossFunction
      
      /**
       * @brief Compute cross entropy loss between predictions and targets
       * @param predictions Network output predictions (softmax probabilities)
       * @param targets Ground truth target values (one-hot encoded)
       * @return Loss value tensor
       */
      CAIF_Tensor ComputeLoss(
                             const CAIF_Tensor &predictions,
                             const CAIF_Tensor &targets
                            )const override;
      
      /**
       * @brief Compute gradient of cross entropy loss with respect to predictions
       * @param predictions Network output predictions (softmax probabilities)
       * @param targets Ground truth target values (one-hot encoded)
       * @return Gradient tensor
       */
      CAIF_Tensor ComputeGradient(
                                 const CAIF_Tensor &predictions,
                                 const CAIF_Tensor &targets
                                )const override;
      
      /**
       * @brief Get loss function type
       * @return Cross entropy loss function type
       */
      CAIF_LossType_e LossType()const override;
      
      /**
       * @brief Clone the loss function (deep copy)
       * @return Unique pointer to cloned loss function
       */
      std::unique_ptr<CAIF_LossFunction> Clone()const override;
      
      /**
       * @brief Get loss function description
       * @return String describing the loss function
       */
      std::string Description()const override;

      // Additional methods
      
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

    private:
      float _epsilon;  ///< Small value to prevent numerical instability
  };
}//end instance namespace
