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
 * @file aif_loss_function_bce_logits.h
 * @brief Binary Cross Entropy with logits loss declaration
 */

#pragma once

#include "caif_loss_function.h"
#include "caif_exception.h"

namespace instance
{
  /**
   * @brief Binary Cross Entropy with Logits loss implementation
   */
  class CAIF_BinaryCrossEntropyWithLogitsLoss final:public instance::CAIF_LossFunction
  {
    public:
      CAIF_BinaryCrossEntropyWithLogitsLoss()=default;
      ~CAIF_BinaryCrossEntropyWithLogitsLoss() override=default;

      instance::CAIF_Tensor ComputeLoss(
                                       const instance::CAIF_Tensor &predictions,
                                       const instance::CAIF_Tensor &targets
                                      )const override;

      instance::CAIF_Tensor ComputeGradient(
                                           const instance::CAIF_Tensor &predictions,
                                           const instance::CAIF_Tensor &targets
                                          )const override;

      instance::CAIF_LossType_e LossType()const override
      {
        return instance::CAIF_LossType_e::BinaryCrossEntropyWithLogits;
      }
      std::unique_ptr<instance::CAIF_LossFunction> Clone()const override
      {
        return std::make_unique<instance::CAIF_BinaryCrossEntropyWithLogitsLoss>();
      }
      std::string Description()const override{return "Binary Cross Entropy (with logits)";}

    protected:
    private:
  };
}//end instance namespace


