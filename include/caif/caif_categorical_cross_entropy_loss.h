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
 * @file aif_categorical_cross_entropy_loss.h
 * @brief Categorical Cross Entropy loss function implementation
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_loss_function.h"
#include "caif_tensor_backend.h"
#include "caif_exception.h"

namespace instance
{
  /**
   * @brief Categorical Cross Entropy loss function implementation
   * 
   * Computes categorical cross entropy loss: -sum(log(predictions[target_class]))
   * Typically used for multi-class classification with softmax outputs and class indices as targets.
   * Unlike regular CrossEntropy, this expects target class indices rather than one-hot encoded vectors.
   */
  class CAIF_CategoricalCrossEntropyLoss:public CAIF_LossFunction
  {
    public:
      /**
       * @brief Constructor with epsilon parameter
       * @param epsilon Small value to prevent log(0) and division by 0
       */
      explicit CAIF_CategoricalCrossEntropyLoss(const float epsilon=1e-7f);
      
      /**
       * @brief Virtual destructor
       */
      virtual ~CAIF_CategoricalCrossEntropyLoss()=default;
      
      /**
       * @brief Copy constructor
       * @param other Loss function to copy from
       */
      CAIF_CategoricalCrossEntropyLoss(const CAIF_CategoricalCrossEntropyLoss &other)=default;
      
      /**
       * @brief Move constructor
       * @param other Loss function to move from
       */
      CAIF_CategoricalCrossEntropyLoss(CAIF_CategoricalCrossEntropyLoss &&other)noexcept=default;
      
      /**
       * @brief Copy assignment operator
       * @param other Loss function to copy from
       * @return Reference to this loss function
       */
      CAIF_CategoricalCrossEntropyLoss &operator=(const CAIF_CategoricalCrossEntropyLoss &other)=default;
      
      /**
       * @brief Move assignment operator
       * @param other Loss function to move from
       * @return Reference to this loss function
       */
      CAIF_CategoricalCrossEntropyLoss &operator=(CAIF_CategoricalCrossEntropyLoss &&other)noexcept=default;

      // Implementation of pure virtual methods from CAIF_LossFunction
      
      /**
       * @brief Compute categorical cross entropy loss between predictions and class indices
       * @param predictions Network output predictions (softmax probabilities) [batch_size, num_classes]
       * @param targets Ground truth class indices [batch_size] or one-hot encoded [batch_size, num_classes]
       * @return Loss value tensor
       */
      CAIF_Tensor ComputeLoss(
                             const CAIF_Tensor &predictions,
                             const CAIF_Tensor &targets
                            )const override;
      
      /**
       * @brief Compute gradient of categorical cross entropy loss with respect to predictions
       * @param predictions Network output predictions (softmax probabilities) [batch_size, num_classes]
       * @param targets Ground truth class indices [batch_size] or one-hot encoded [batch_size, num_classes]
       * @return Gradient tensor
       */
      CAIF_Tensor ComputeGradient(
                                 const CAIF_Tensor &predictions,
                                 const CAIF_Tensor &targets
                                )const override;
      
      /**
       * @brief Get loss function type
       * @return Categorical cross entropy loss function type
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
      
      /**
       * @brief Check if targets are class indices (1D) or one-hot encoded (2D same as predictions)
       * @param predictions Predictions tensor [batch_size, num_classes]
       * @param targets Targets tensor
       * @return True if targets are class indices, false if one-hot encoded
       */
      bool AreTargetsClassIndices(
                                  const CAIF_Tensor &predictions,
                                  const CAIF_Tensor &targets
                                 )const;
  };
}//end instance namespace
