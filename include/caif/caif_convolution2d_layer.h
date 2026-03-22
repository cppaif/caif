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
 * @file aif_convolution2d_layer.h
 * @brief Convolution 2D layer for neural networks
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_layer.h"
#include "caif_exception.h"
#include "caif_framework.h"

namespace instance
{
  /**
   * @brief Convolution 2D layer for feature extraction
   * 
   * This layer performs 2D convolution operations on inputs using learnable filters.
   * Input format: [batch, height, width, input_channels]
   * Output format: [batch, output_height, output_width, output_channels]
   */
  class CAIF_Convolution2DLayer:public CAIF_Layer
  {
    public:
      /**
       * @brief Constructor with convolution parameters
       * @param framework Reference to AIF framework instance
       * @param filters Number of output filters (output channels)
       * @param kernel_size Size of convolution kernel (assumed square)
       * @param stride Stride for convolution operation
       * @param padding Padding amount
       * @param activation Activation function to apply
       * @param use_bias Whether to use bias terms
       */
      CAIF_Convolution2DLayer(
                             CAIF_Framework &framework,
                             const uint32_t filters,
                             const uint32_t kernel_size,
                             const uint32_t stride=1,
                             const uint32_t padding=0,
                             const CAIF_ActivationType_e activation=CAIF_ActivationType_e::ReLU,
                             const bool use_bias=true
                            );
      
      /**
       * @brief Copy constructor
       * @param other Layer to copy from
       */
      CAIF_Convolution2DLayer(const CAIF_Convolution2DLayer &other);
      
      /**
       * @brief Move constructor
       * @param other Layer to move from
       */
      CAIF_Convolution2DLayer(CAIF_Convolution2DLayer &&other);
      
      /**
       * @brief Destructor
       */
      virtual ~CAIF_Convolution2DLayer()=default;

      /**
       * @brief Copy assignment operator
       * @param other Layer to copy from
       * @return Reference to this layer
       */
      CAIF_Convolution2DLayer &operator=(const CAIF_Convolution2DLayer &other);
      
      /**
       * @brief Move assignment operator
       * @param other Layer to move from
       * @return Reference to this layer
       */
      CAIF_Convolution2DLayer &operator=(CAIF_Convolution2DLayer &&other);

      /**
       * @brief Perform forward pass
       * @param input Input tensor to convolve
       * @param training Whether in training mode
       * @return Expected containing convolved tensor or error message
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
       * @param seed Random seed for weight initialization
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
      CAIF_LayerType_e LayerType()const override{return CAIF_LayerType_e::Convolution2D;}

      /**
       * @brief Get string description of layer
       * @return Description string
       */
      std::string Description()const override;

      /**
       * @brief Check if layer has trainable parameters
       * @return True if layer has parameters
       */
      bool HasParameters()const override{return true;}

      /**
       * @brief Get layer parameters
       * @return Vector of parameter tensors
       */
      std::vector<CAIF_Tensor> Parameters()const override;

      /**
       * @brief Get parameter gradients
       * @return Vector of parameter gradient tensors
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
       * @brief Reset layer parameters with random values
       * @param seed Random seed
       * @return Expected containing void or error message
       */
      void ResetParameters(const uint32_t seed=0)override;

      /**
       * @brief Get last forward output (feature maps) for attribution methods
       * @return Const reference to last output tensor
       */
      const CAIF_Tensor &LastOutput()const{return _last_output;}

      /**
       * @brief Get last activation gradient computed during backward pass
       * @return Const reference to last activation gradient tensor
       */
      const CAIF_Tensor &LastActivationGradient()const{return _last_activation_gradient;}

      /**
       * @brief Get number of filters
       * @return Number of filters
       */
      uint32_t Filters()const{return _filters;}
      
      /**
       * @brief Get kernel size
       * @return Kernel size
       */
      uint32_t KernelSize()const{return _kernel_size;}
      
      /**
       * @brief Get stride
       * @return Stride value
       */
      uint32_t Stride()const{return _stride;}
      
      /**
       * @brief Get padding
       * @return Padding value
       */
      uint32_t Padding()const{return _padding;}
      
      /**
       * @brief Get activation type
       * @return Activation type
       */
      CAIF_ActivationType_e Activation()const{return _activation;}

    protected:

    private:
      uint32_t _filters;                    ///< Number of output filters
      uint32_t _kernel_size;                ///< Size of convolution kernel
      uint32_t _stride;                     ///< Stride for convolution
      uint32_t _padding;                    ///< Padding amount
      CAIF_ActivationType_e _activation;     ///< Activation function
      bool _use_bias;                       ///< Whether to use bias
      uint32_t _input_channels;             ///< Number of input channels
      
      ///< Convolution filters [filters,kernel_size,kernel_size,input_channels]
      CAIF_Tensor _weights;
      CAIF_Tensor _bias;                     ///< Bias terms [filters]
      CAIF_Tensor _weight_gradients;         ///< Weight gradients for parameter updates
      CAIF_Tensor _bias_gradients;           ///< Bias gradients for parameter updates
      CAIF_Tensor _last_input;               ///< Last input for backward pass
      CAIF_Tensor _last_pre_activation;      ///< Pre-activation output for backward pass
      CAIF_Tensor _last_output;              ///< Last output for backward pass
      CAIF_Tensor _last_activation_gradient; ///< Cached activation gradient for attribution

      /**
       * @brief Initialize weights using Xavier/Glorot initialization
       * @param seed Random seed
       */
      void InitializeWeights(const uint32_t seed);

      /**
       * @brief Apply activation function
       * @param tensor Input tensor
       * @return Activated tensor
       */
      CAIF_Tensor ApplyActivation(const CAIF_Tensor &tensor)const;

      /**
       * @brief Apply activation derivative
       * @param tensor Output tensor from forward pass
       * @param gradient Gradient from next layer
       * @return Gradient with activation derivative applied
       */
      CAIF_Tensor ApplyActivationDerivative(
                                           const CAIF_Tensor &tensor,
                                           const CAIF_Tensor &gradient
                                          )const;

      /**
       * @brief Calculate output dimensions after convolution
       * @param input_dim Input dimension size
       * @return Output dimension size
       */
      uint32_t CalculateOutputDim(const uint32_t input_dim)const;

      /**
       * @brief Perform convolution operation
       * @param input Input tensor
       * @param output Output tensor to store results
       * @return Expected containing void or error message
       */
      void PerformConvolution(
                              const CAIF_Tensor &input,
                              CAIF_Tensor &output
                             )const;

      /**
       * @brief Add bias to output tensor
       * @param output Output tensor to add bias to
       * @return Expected containing void or error message
       */
      void AddBias(CAIF_Tensor &output)const;

      /**
       * @brief Compute weight gradients for backpropagation
       * @param input Input tensor from forward pass
       * @param output_gradient Output gradient from next layer
       * @return Weight gradients tensor
       */
      CAIF_Tensor ComputeWeightGradients(
                                                      const CAIF_Tensor &input,
                                                      const CAIF_Tensor &output_gradient
                                                     )const;

      /**
       * @brief Compute bias gradients for backpropagation
       * @param output_gradient Output gradient from next layer
       * @return Bias gradients tensor
       */
      CAIF_Tensor ComputeBiasGradients(const CAIF_Tensor &output_gradient)const;

      /**
       * @brief Compute input gradients for backpropagation
       * @param output_gradient Output gradient from next layer
       * @return Expected containing input gradients tensor or error message
       */
      CAIF_Tensor ComputeInputGradients(
                                       const CAIF_Tensor &output_gradient
                                      )const;
  };
}//end instance namespace
