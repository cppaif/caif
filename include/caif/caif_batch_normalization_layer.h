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
 * @file aif_batch_normalization_layer.h
 * @brief Batch normalization layer for neural networks
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_layer.h"
#include "caif_framework.h"

namespace instance
{
  /**
   * @brief Batch normalization layer for stable training
   * 
   * This layer normalizes inputs across the batch dimension and applies
   * learnable scale and shift parameters. Helps with gradient flow and
   * allows higher learning rates.
   */
  class CAIF_BatchNormalizationLayer:public CAIF_Layer
  {
    public:
      /**
       * @brief Constructor with normalization parameters
       * @param framework Reference to CAIF framework instance
       * @param epsilon Small value to prevent division by zero
       * @param momentum Momentum for running statistics
       * @param affine Whether to use learnable scale and shift parameters
       */
      CAIF_BatchNormalizationLayer(
                                  CAIF_Framework &framework,
                                  const float epsilon=1e-5f,
                                  const float momentum=0.9f,
                                  const bool affine=true
                                 );
      
      /**
       * @brief Copy constructor
       * @param other Layer to copy from
       */
      CAIF_BatchNormalizationLayer(const CAIF_BatchNormalizationLayer &other);
      
      /**
       * @brief Move constructor
       * @param other Layer to move from
       */
      CAIF_BatchNormalizationLayer(CAIF_BatchNormalizationLayer &&other)noexcept;
      
      /**
       * @brief Destructor
       */
      virtual ~CAIF_BatchNormalizationLayer()=default;

      /**
       * @brief Copy assignment operator
       * @param other Layer to copy from
       * @return Reference to this layer
       */
      CAIF_BatchNormalizationLayer &operator=(const CAIF_BatchNormalizationLayer &other);
      
      /**
       * @brief Move assignment operator
       * @param other Layer to move from
       * @return Reference to this layer
       */
      CAIF_BatchNormalizationLayer &operator=(CAIF_BatchNormalizationLayer &&other)noexcept;

      /**
       * @brief Perform forward pass
       * @param input Input tensor to normalize
       * @param training Whether in training mode
       * @return Expected containing normalized tensor or error message
       */
      CAIF_Tensor Forward(
                         const CAIF_Tensor &input,
                         const bool training=false
                        )override;

      /**
       * @brief Perform backward pass
       * @param gradient Gradient from next layer
       * @return Expected containing input gradient or error message
       */
      CAIF_Tensor Backward(const CAIF_Tensor &gradient)override;

      /**
       * @brief Initialize the layer
       * @param input_shape Shape of input tensor
       * @param seed Random seed (unused for batch norm)
       * @return Expected containing void or error message
       */
      void Initialize(
                      const std::vector<uint32_t> &input_shape,
                      const uint32_t seed=0
                     )override;

      /**
       * @brief Calculate output shape given input shape
       * @param input_shape Input tensor shape
       * @return Expected containing output shape or error message
       */
      std::vector<uint32_t> CalculateOutputShape(
                                                  const std::vector<uint32_t> &input_shape
                                                 )const override;

      /**
       * @brief Create a copy of this layer
       * @return Unique pointer to cloned layer
       * @note Framework reference is copied from this layer (same framework instance)
       */
      std::unique_ptr<CAIF_Layer> Clone()const override;

      /**
       * @brief Get layer type
       * @return Layer type enum
       */
      CAIF_LayerType_e LayerType()const override{return CAIF_LayerType_e::BatchNormalization;}

      /**
       * @brief Get string description of layer
       * @return Description string
       */
      std::string Description()const override;

      /**
       * @brief Get epsilon value
       * @return Epsilon value
       */
      float Epsilon()const{return _epsilon;}
      
      /**
       * @brief Get momentum value
       * @return Momentum value
       */
      float Momentum()const{return _momentum;}

      /**
       * @brief Check if layer has trainable parameters
       * @return True if layer has parameters and affine is enabled
       */
      bool HasParameters()const override{return _affine;}

      /**
       * @brief Get layer parameters
       * @return Vector of parameter tensors
       */
      std::vector<CAIF_Tensor> Parameters()const override;

      /**
       * @brief Get parameter gradients
       * @return Vector of gradient tensors
       */
      std::vector<CAIF_Tensor> ParameterGradients()const override;

      /**
       * @brief Get number of trainable parameters
       * @return Number of parameter tensors
       */
      size_t ParameterCount()const override;

      /**
       * @brief Get mutable reference to parameter at index
       * @param index Parameter index
       * @return Mutable reference to parameter tensor
       */
      CAIF_Tensor &ParameterRef(const size_t index)override;

      /**
       * @brief Get const reference to parameter at index
       * @param index Parameter index
       * @return Const reference to parameter tensor
       */
      const CAIF_Tensor &ParameterRef(const size_t index)const override;

      /**
       * @brief Get mutable reference to gradient at index
       * @param index Gradient index
       * @return Mutable reference to gradient tensor
       */
      CAIF_Tensor &GradientRef(const size_t index)override;

      /**
       * @brief Get const reference to gradient at index
       * @param index Gradient index
       * @return Const reference to gradient tensor
       */
      const CAIF_Tensor &GradientRef(const size_t index)const override;

      /**
       * @brief Update layer parameters
       * @param new_parameters New parameter values
       * @return Expected containing void or error message
       */
      void UpdateParameters(
                            const std::vector<CAIF_Tensor> &new_parameters
                           )override;

      /**
       * @brief Reset layer parameters with default values
       * @param seed Random seed (unused)
       * @return Expected containing void or error message
       */
      void ResetParameters(const uint32_t seed=0)override;

    protected:

    private:
      float _epsilon;                       ///< Small value to prevent division by zero
      float _momentum;                      ///< Momentum for running statistics
      bool _affine;                         ///< Whether to use learnable parameters
      uint32_t _num_features;               ///< Number of features to normalize
      
      CAIF_Tensor _scale;                    ///< Scale parameter (gamma)
      CAIF_Tensor _shift;                    ///< Shift parameter (beta)
      CAIF_Tensor _running_mean;             ///< Running mean for inference
      CAIF_Tensor _running_var;              ///< Running variance for inference
      
      CAIF_Tensor _last_input;               ///< Last input for backward pass
      CAIF_Tensor _last_normalized;          ///< Last normalized values for backward pass
      CAIF_Tensor _last_mean;                ///< Last batch mean for backward pass
      CAIF_Tensor _last_variance;            ///< Last batch variance for backward pass
      
      CAIF_Tensor _scale_gradient;           ///< Gradient for scale parameter
      CAIF_Tensor _shift_gradient;           ///< Gradient for shift parameter

      /**
       * @brief Initialize parameters with default values
       */
      void InitializeParameters();

      /**
       * @brief Compute batch statistics
       * @param input Input tensor
       * @param mean Output mean tensor
       * @param variance Output variance tensor
       */
      void ComputeBatchStatistics(
                                  const CAIF_Tensor &input,
                                  CAIF_Tensor &mean,
                                  CAIF_Tensor &variance
                                 )const;

      /**
       * @brief Update running statistics
       * @param batch_mean Current batch mean
       * @param batch_var Current batch variance
       */
      void UpdateRunningStatistics(
                                   const CAIF_Tensor &batch_mean,
                                   const CAIF_Tensor &batch_var
                                  );

      /**
       * @brief Apply normalization
       * @param input Input tensor
       * @param mean Mean tensor
       * @param variance Variance tensor
       * @return Normalized tensor
       */
      CAIF_Tensor ApplyNormalization(
                                    const CAIF_Tensor &input,
                                    const CAIF_Tensor &mean,
                                    const CAIF_Tensor &variance
                                   )const;

      /**
       * @brief Apply scale and shift if affine is enabled
       * @param normalized Normalized tensor
       * @return Scaled and shifted tensor
       */
      CAIF_Tensor ApplyAffineTransform(const CAIF_Tensor &normalized)const;
  };
}//end instance namespace
