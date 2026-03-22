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
 * @file aif_loss_function.h
 * @brief Base loss function class for neural network training
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_base.h"
#include "caif_constants.h"
#include "caif_tensor.h"

#include <memory>
#include <string>
#include "caif_exception.h"

namespace instance
{
  /**
   * @brief Abstract base class for all loss functions
   * 
   * The CAIF_LossFunction class provides the interface for computing
   * loss values and gradients for different loss functions.
   */
  class CAIF_LossFunction:public CAIF_Base
  {
    public:
      /**
       * @brief Default constructor
       */
      CAIF_LossFunction()=default;
      
      /**
       * @brief Virtual destructor for proper inheritance
       */
      virtual ~CAIF_LossFunction()=default;
      
      /**
       * @brief Copy constructor
       * @param other Loss function to copy from
       */
      CAIF_LossFunction(const CAIF_LossFunction &other)=default;
      
      /**
       * @brief Move constructor
       * @param other Loss function to move from
       */
      CAIF_LossFunction(CAIF_LossFunction &&other)noexcept=default;
      
      /**
       * @brief Copy assignment operator
       * @param other Loss function to copy from
       * @return Reference to this loss function
       */
      CAIF_LossFunction &operator=(const CAIF_LossFunction &other)=default;
      
      /**
       * @brief Move assignment operator
       * @param other Loss function to move from
       * @return Reference to this loss function
       */
      CAIF_LossFunction &operator=(CAIF_LossFunction &&other)noexcept=default;

      // Pure virtual methods
      
      /**
       * @brief Compute loss value between predictions and targets
       * @param predictions Network output predictions
       * @param targets Ground truth target values
       * @return Expected with loss value tensor or error message
       */
      virtual CAIF_Tensor ComputeLoss(
                                     const CAIF_Tensor &predictions,
                                     const CAIF_Tensor &targets
                                    )const=0;
      
      /**
       * @brief Compute gradient of loss with respect to predictions
       * @param predictions Network output predictions
       * @param targets Ground truth target values
       * @return Expected with gradient tensor or error message
       */
      virtual CAIF_Tensor ComputeGradient(
                                         const CAIF_Tensor &predictions,
                                         const CAIF_Tensor &targets
                                        )const=0;
      
      /**
       * @brief Get loss function type
       * @return Loss function type enum
       */
      virtual CAIF_LossType_e LossType()const=0;
      
      /**
       * @brief Clone the loss function (deep copy)
       * @return Unique pointer to cloned loss function
       */
      virtual std::unique_ptr<CAIF_LossFunction> Clone()const=0;
      
      /**
       * @brief Get loss function name/description
       * @return String describing the loss function
       */
      virtual std::string Description()const=0;

      // Virtual methods with default implementations
      
      /**
       * @brief Compute both loss and gradient in one call (for efficiency)
       * @param predictions Network output predictions
       * @param targets Ground truth target values
       * @return Expected with pair of (loss, gradient) or error message
       */
      virtual std::pair<CAIF_Tensor,CAIF_Tensor> ComputeLossAndGradient(
                                                                       const CAIF_Tensor &predictions,
                                                                       const CAIF_Tensor &targets
                                                                      )const
      {
        CAIF_Tensor loss=ComputeLoss(predictions,targets);
        CAIF_Tensor grad=ComputeGradient(predictions,targets);
        return std::make_pair(std::move(loss),std::move(grad));
      }

    protected:
      // Protected members go here

    private:
      // Private members go here
  };

  /**
   * @brief Binary Cross Entropy loss implementation
   */
  class CAIF_BinaryCrossEntropyLoss final:public CAIF_LossFunction
  {
    public:
      CAIF_BinaryCrossEntropyLoss()=default;
      ~CAIF_BinaryCrossEntropyLoss() override=default;

      CAIF_Tensor ComputeLoss(
                             const CAIF_Tensor &predictions,
                             const CAIF_Tensor &targets
                            )const override;

      CAIF_Tensor ComputeGradient(
                                 const CAIF_Tensor &predictions,
                                 const CAIF_Tensor &targets
                                )const override;

      CAIF_LossType_e LossType()const override{return CAIF_LossType_e::BinaryCrossEntropy;}
      std::unique_ptr<CAIF_LossFunction> Clone()const override
      {
        return std::make_unique<CAIF_BinaryCrossEntropyLoss>();
      }
      std::string Description()const override{return "Binary Cross Entropy";}
  };

  // BCE-with-logits declared in its own header per guidelines
}//end instance namespace
