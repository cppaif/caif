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
 * @file aif_mean_squared_error_loss.h
 * @brief Mean Squared Error loss function implementation
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_loss_function.h"

namespace instance
{
  /**
   * @brief Mean Squared Error loss function implementation
   * 
   * Computes MSE loss: (1/N) * sum((predictions - targets)^2)
   * where N is the number of elements per sample.
   */
  class CAIF_MeanSquaredErrorLoss:public CAIF_LossFunction
  {
    public:
      /**
       * @brief Default constructor
       */
      CAIF_MeanSquaredErrorLoss();
      
      /**
       * @brief Virtual destructor
       */
      virtual ~CAIF_MeanSquaredErrorLoss()=default;
      
      /**
       * @brief Copy constructor
       * @param other Loss function to copy from
       */
      CAIF_MeanSquaredErrorLoss(const CAIF_MeanSquaredErrorLoss &other)=default;
      
      /**
       * @brief Move constructor
       * @param other Loss function to move from
       */
      CAIF_MeanSquaredErrorLoss(CAIF_MeanSquaredErrorLoss &&other)noexcept=default;
      
      /**
       * @brief Copy assignment operator
       * @param other Loss function to copy from
       * @return Reference to this loss function
       */
      CAIF_MeanSquaredErrorLoss &operator=(const CAIF_MeanSquaredErrorLoss &other)=default;
      
      /**
       * @brief Move assignment operator
       * @param other Loss function to move from
       * @return Reference to this loss function
       */
      CAIF_MeanSquaredErrorLoss &operator=(CAIF_MeanSquaredErrorLoss &&other)noexcept=default;

      // Implementation of pure virtual methods from CAIF_LossFunction
      
      /**
       * @brief Compute MSE loss between predictions and targets
       * @param predictions Network output predictions
       * @param targets Ground truth target values
       * @return Loss value tensor
       */
      CAIF_Tensor ComputeLoss(
                             const CAIF_Tensor &predictions,
                             const CAIF_Tensor &targets
                            )const override;
      
      /**
       * @brief Compute gradient of MSE loss with respect to predictions
       * @param predictions Network output predictions
       * @param targets Ground truth target values
       * @return Gradient tensor
       */
      CAIF_Tensor ComputeGradient(
                                 const CAIF_Tensor &predictions,
                                 const CAIF_Tensor &targets
                                )const override;

      std::pair<CAIF_Tensor,CAIF_Tensor> ComputeLossAndGradient(
                                                               const CAIF_Tensor &predictions,
                                                               const CAIF_Tensor &targets
                                                              )const override;
      
      /**
       * @brief Get loss function type
       * @return MSE loss function type
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

    private:
      // Private members go here
  };
}//end instance namespace
