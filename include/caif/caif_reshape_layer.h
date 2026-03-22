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

#ifndef CAIF_RESHAPE_LAYER_H
#define CAIF_RESHAPE_LAYER_H

#include "caif_layer.h"
#include "caif_framework.h"
#include <vector>

namespace instance
{
  /**
   * @brief Layer that reshapes input tensor to specified shape while preserving elements
   */
  class CAIF_ReshapeLayer : public CAIF_Layer
  {
    public:
      /**
       * @brief Constructor
       * @param framework Reference to AIF framework instance
       * @param target_shape Shape to reshape to (excluding batch dimension)
       */
      explicit CAIF_ReshapeLayer(CAIF_Framework &framework,const std::vector<uint32_t> &target_shape);
      
      CAIF_ReshapeLayer(const CAIF_ReshapeLayer &other);
      CAIF_ReshapeLayer(CAIF_ReshapeLayer &&other)noexcept;
      CAIF_ReshapeLayer &operator=(const CAIF_ReshapeLayer &other);
      CAIF_ReshapeLayer &operator=(CAIF_ReshapeLayer &&other)noexcept;
      
      CAIF_LayerType_e LayerType()const override{return CAIF_LayerType_e::Reshape;}
      
      CAIF_Tensor Forward(
                         const CAIF_Tensor &input,
                         const bool training
                        )override;
      
      CAIF_Tensor Backward(
                          const CAIF_Tensor &gradient
                         )override;
      
      void Initialize(
                      const std::vector<uint32_t> &input_shape,
                      const uint32_t seed
                     )override;
      
      std::vector<uint32_t> CalculateOutputShape(
                                                 const std::vector<uint32_t> &input_shape
                                                )const override;
      
      std::vector<uint32_t> TargetShape()const{return _target_shape;}
      
      std::unique_ptr<CAIF_Layer> Clone()const override;
      
      std::string Description()const override;
      
    private:
      std::vector<uint32_t> _target_shape;  // Target shape (excluding batch dimension)
      std::vector<uint32_t> _original_shape; // Original input shape for backward pass
  };
}//end instance namespace

#endif // CAIF_RESHAPE_LAYER_H 