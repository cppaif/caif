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
 * @file aif_dense_layer.h
 * @brief Dense (fully connected) layer implementation
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_layer.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "caif_framework.h"
#include <random>

namespace instance
{
  /**
   * @brief Dense/fully connected layer implementation
   * 
   * Implements a standard dense layer with weight matrix and bias vector,
   * supporting various activation functions.
   */
  class CAIF_DenseLayer final:public CAIF_Layer
  {
    public:
      /**
       * @brief Constructor for dense layer
       * @param framework Reference to AIF framework instance
       * @param units Number of output units/neurons
       * @param activation Activation function type
       * @param use_bias Whether to use bias terms
       */
      CAIF_DenseLayer(
                     CAIF_Framework &framework,
                     const uint32_t units,
                     const CAIF_ActivationType_e activation=CAIF_ActivationType_e::ReLU,
                     const bool use_bias=true
                    );
      
      /**
       * @brief Destructor
       */
      ~CAIF_DenseLayer()override=default;
      
      /**
       * @brief Copy constructor
       * @param other Layer to copy from
       */
      CAIF_DenseLayer(const CAIF_DenseLayer &other);
      
      /**
       * @brief Move constructor
       * @param other Layer to move from
       */
      CAIF_DenseLayer(CAIF_DenseLayer &&other);
      
      /**
       * @brief Copy assignment operator
       * @param other Layer to copy from
       * @return Reference to this layer
       */
      CAIF_DenseLayer &operator=(const CAIF_DenseLayer &other);
      
      /**
       * @brief Move assignment operator
       * @param other Layer to move from
       * @return Reference to this layer
       */
      CAIF_DenseLayer &operator=(CAIF_DenseLayer &&other);

      // Inherited pure virtual methods
      CAIF_Tensor Forward(
                         const CAIF_Tensor &input,
                         const bool training=false
                        )override;
      
      CAIF_Tensor Backward(const CAIF_Tensor &gradient)override;
      
      void Initialize(
                      const std::vector<uint32_t> &input_shape,
                      const uint32_t seed=0
                     )override;
      
      std::vector<uint32_t> CalculateOutputShape(
                                                  const std::vector<uint32_t> &input_shape
                                                 )const override;
      
      CAIF_LayerType_e LayerType()const override{return CAIF_LayerType_e::Dense;}
      
      std::unique_ptr<CAIF_Layer> Clone()const override;

      // Inherited virtual methods with implementations
      bool HasParameters()const override{return true;}
      
      std::vector<CAIF_Tensor> Parameters()const override;
      
      std::vector<CAIF_Tensor> ParameterGradients()const override;
      
      size_t ParameterCount()const override;
      
      CAIF_Tensor &ParameterRef(const size_t index)override;
      
      const CAIF_Tensor &ParameterRef(const size_t index)const override;
      
      CAIF_Tensor &GradientRef(const size_t index)override;
      
      const CAIF_Tensor &GradientRef(const size_t index)const override;
      
      void UpdateParameters(
                            const std::vector<CAIF_Tensor> &new_parameters
                           )override;
      
      void ResetParameters(const uint32_t seed=0)override;
      
      std::string Description()const override;

      // Getters and setters specific to dense layer
      uint32_t Units()const{return _units;}
      CAIF_ActivationType_e Activation()const{return _activation;}
      bool UseBias()const{return _use_bias;}
      
      void SetActivation(const CAIF_ActivationType_e activation){_activation=activation;}

      /**
       * @brief Set all bias values to a constant
       * @param value Bias value to assign to each unit
       */
      void SetBias(const float value);

    protected:
      // Protected members go here

    private:
      uint32_t _units;
      CAIF_ActivationType_e _activation;
      bool _use_bias;
      
      // Parameters
      CAIF_Tensor _weights;
      CAIF_Tensor _bias;
      
      // Gradients
      CAIF_Tensor _weight_gradients;
      CAIF_Tensor _bias_gradients;
      
      // Cached values for backward pass
      CAIF_Tensor _last_input;
      CAIF_Tensor _last_linear;  // pre-activation (after bias add)
      CAIF_Tensor _last_output;

      /**
       * @brief Apply activation function to tensor
       * @param input Input tensor
       * @param activation Activation function type
       * @return Activated tensor
       */
      CAIF_Tensor ApplyActivation(
                                 const CAIF_Tensor &input,
                                 const CAIF_ActivationType_e activation
                                )const;
      
      /**
       * @brief Apply activation derivative for backward pass
       * @param input Input tensor (before activation)
       * @param gradient Gradient from next layer
       * @param activation Activation function type
       * @return Gradient tensor
       */
      CAIF_Tensor ApplyActivationDerivative(
                                           const CAIF_Tensor &input,
                                           const CAIF_Tensor &gradient,
                                           const CAIF_ActivationType_e activation
                                          )const;
      
      /**
       * @brief Initialize weights using Xavier/Glorot initialization
       * @param input_size Number of input units
       * @param output_size Number of output units
       * @param seed Random seed
       */
      void InitializeWeights(
                             const uint32_t input_size,
                             const uint32_t output_size,
                             const uint32_t seed
                            );

      /**
       * @brief Fused bias-add and activation application on output tensor
       * @param output Output tensor to be modified in-place (activated values)
       * @param out_linear Pre-activation tensor to write linear(bias-added) values for backward
       */
      void Fuse(
                CAIF_Tensor &output,
                CAIF_Tensor &out_linear
               )const;

      /**
       * @brief Point-wise activation application
       * @param activation Activation function to apply
       * @param value Input value
       * @return Activated value
       */
      static float PointWise(const CAIF_ActivationType_e activation,const float value);

  };
}//end instance namespace
