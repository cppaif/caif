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
 * @file aif_layer.h
 * @brief Base layer class for neural network layers
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_base.h"
#include "caif_constants.h"
#include "caif_tensor.h"
#include <vector>
#include <memory>
#include <string>
#include "caif_exception.h"

namespace instance
{
  class CAIF_Framework;
  /**
   * @brief Abstract base class for all neural network layers
   * 
   * The CAIF_Layer class provides the interface that all neural network
   * layers must implement, including forward pass, backward pass, and
   * parameter management.
   */
  class CAIF_Layer:public CAIF_Base
  {
    public:
      /**
       * @brief Constructor with framework reference
       * @param framework Reference to CAIF framework instance
       */
      explicit CAIF_Layer(CAIF_Framework &framework):_framework(framework),
                                                   _initialized(false),
                                                   _training_mode(false){}
      
      /**
       * @brief Virtual destructor for proper inheritance
       */
      virtual ~CAIF_Layer()=default;
      
      /**
       * @brief Copy constructor
       * @param other Layer to copy from
       * @note The framework reference is copied from the source (same framework instance)
       */
      CAIF_Layer(const CAIF_Layer &other):_framework(other._framework),  // Copy framework reference
                                        _input_shape(other._input_shape),
                                        _output_shape(other._output_shape),
                                        _initialized(other._initialized),
                                        _training_mode(other._training_mode){}
      
      /**
       * @brief Move constructor
       * @param other Layer to move from
       */
      CAIF_Layer(CAIF_Layer &&other):_framework(other._framework),
                                   _input_shape(std::move(other._input_shape)),
                                   _output_shape(std::move(other._output_shape)),
                                   _initialized(other._initialized),
                                   _training_mode(other._training_mode){}
      
      /**
       * @brief Copy assignment operator
       * @param other Layer to copy from
       * @return Reference to this layer
       */
      CAIF_Layer &operator=(const CAIF_Layer &other)
      {
        if(this!=&other)
        {
          // Note: reference cannot be reassigned, so we only copy other members
          _input_shape=other._input_shape;
          _output_shape=other._output_shape;
          _initialized=other._initialized;
          _training_mode=other._training_mode;
        }
        return *this;
      }
      
      /**
       * @brief Move assignment operator
       * @param other Layer to move from
       * @return Reference to this layer
       */
      CAIF_Layer &operator=(CAIF_Layer &&other)noexcept
      {
        if(this!=&other)
        {
          // Note: reference cannot be reassigned, so we only move other members
          _input_shape=std::move(other._input_shape);
          _output_shape=std::move(other._output_shape);
          _initialized=other._initialized;
          _training_mode=other._training_mode;
        }
        return *this;
      }

      // Pure virtual methods that must be implemented by derived classes
      
      /**
       * @brief Perform forward pass through the layer
       * @param input Input tensor
       * @param training Whether in training mode
       * @return Output tensor
       */
      virtual CAIF_Tensor Forward(
                                 const CAIF_Tensor &input,
                                 const bool training=false
                                )=0;
      
      /**
       * @brief Perform backward pass through the layer
       * @param gradient Gradient from next layer
       * @return Input gradient tensor
       */
      virtual CAIF_Tensor Backward(const CAIF_Tensor &gradient)=0;
      
      /**
       * @brief Initialize layer parameters
       * @param input_shape Shape of input tensor
       * @param seed Random seed for initialization
       */
      virtual void Initialize(
                              const std::vector<uint32_t> &input_shape,
                              const uint32_t seed=0
                             )=0;
      
      /**
       * @brief Calculate output shape given input shape
       * @param input_shape Shape of input tensor
       * @return Output shape
       */
      virtual std::vector<uint32_t> CalculateOutputShape(
                                                         const std::vector<uint32_t> &input_shape
                                                        )const=0;
      
      /**
       * @brief Get layer type
       * @return Layer type enum
       */
      virtual CAIF_LayerType_e LayerType()const=0;
      
      /**
       * @brief Clone the layer (deep copy)
       * @return Unique pointer to cloned layer
       * @note The framework reference is copied from this layer (same framework instance)
       */
      virtual std::unique_ptr<CAIF_Layer> Clone()const=0;

      // Virtual methods with default implementations
      
      /**
       * @brief Check if layer has trainable parameters
       * @return True if layer has parameters, false otherwise
       */
      virtual bool HasParameters()const{return false;}
      
      /**
       * @brief Get layer parameters (weights and biases)
       * @return Vector of parameter tensors
       */
      virtual std::vector<CAIF_Tensor> Parameters()const{return {};}
      
      /**
       * @brief Get parameter gradients
       * @return Vector of gradient tensors
       */
      virtual std::vector<CAIF_Tensor> ParameterGradients()const{return {};}
      
      /**
       * @brief Get number of trainable parameters (avoids allocating vectors)
       * @return Number of parameter tensors
       */
      virtual size_t ParameterCount()const{return 0;}
      
      /**
       * @brief Get mutable reference to parameter at index (avoids copies)
       * @param index Parameter index
       * @return Mutable reference to parameter tensor
       */
      virtual CAIF_Tensor &ParameterRef(const size_t index)
      {
        (void)index;
        THROW_CAIFE("Layer has no parameters");
      }
      
      /**
       * @brief Get const reference to parameter at index (avoids copies)
       * @param index Parameter index
       * @return Const reference to parameter tensor
       */
      virtual const CAIF_Tensor &ParameterRef(const size_t index)const
      {
        (void)index;
        THROW_CAIFE("Layer has no parameters");
      }
      
      /**
       * @brief Get mutable reference to gradient at index (avoids copies)
       * @param index Gradient index
       * @return Mutable reference to gradient tensor
       */
      virtual CAIF_Tensor &GradientRef(const size_t index)
      {
        (void)index;
        THROW_CAIFE("Layer has no parameters");
      }
      
      /**
       * @brief Get const reference to gradient at index (avoids copies)
       * @param index Gradient index
       * @return Const reference to gradient tensor
       */
      virtual const CAIF_Tensor &GradientRef(const size_t index)const
      {
        (void)index;
        THROW_CAIFE("Layer has no parameters");
      }
      
      /**
       * @brief Update layer parameters
       * @param new_parameters Vector of new parameter tensors
       */
      virtual void UpdateParameters(
                                    const std::vector<CAIF_Tensor> &new_parameters
                                   )
      {
        (void)new_parameters;
        THROW_CAIFE("Layer has no parameters");
      }
      
      /**
       * @brief Reset layer parameters to random values
       * @param seed Random seed for initialization
       */
      virtual void ResetParameters(const uint32_t seed=0)
      {
        (void)seed;
        THROW_CAIFE("Layer has no parameters");
      }
      
      /**
       * @brief Set layer to training or inference mode
       * @param training True for training mode, false for inference
       */
      virtual void SetTrainingMode(const bool training){_training_mode=training;}
      
      /**
       * @brief Check if layer is in training mode
       * @return True if in training mode, false otherwise
       */
      virtual bool IsTrainingMode()const{return _training_mode;}
      
      /**
       * @brief Get layer name/description
       * @return String describing the layer
       */
      virtual std::string Description()const{return "Base Layer";}

      // Getters and setters
      
      /**
       * @brief Get input shape of the layer
       * @return Const reference to input shape vector
       */
      const std::vector<uint32_t> &InputShape()const{return _input_shape;}
      
      /**
       * @brief Get output shape of the layer
       * @return Const reference to output shape vector
       */
      const std::vector<uint32_t> &OutputShape()const{return _output_shape;}
      
      /**
       * @brief Check if layer is initialized
       * @return True if initialized, false otherwise
       */
      bool IsInitialized()const{return _initialized;}

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
      
      /**
       * @brief Set input shape
       * @param input_shape New input shape
       */
      void SetInputShape(const std::vector<uint32_t> &input_shape){_input_shape=input_shape;}
      
      /**
       * @brief Set output shape
       * @param output_shape New output shape
       */
      void SetOutputShape(const std::vector<uint32_t> &output_shape){_output_shape=output_shape;}
      
      /**
       * @brief Set initialized flag
       * @param initialized New initialized value
       */
      void SetInitialized(const bool initialized){_initialized=initialized;}
      
      /**
       * @brief Get input shape (non-const for derived classes)
       * @return Reference to input shape vector
       */
      std::vector<uint32_t> &InputShape(){return _input_shape;}
      
      /**
       * @brief Get output shape (non-const for derived classes)
       * @return Reference to output shape vector
       */
      std::vector<uint32_t> &OutputShape(){return _output_shape;}
      
    private:
      CAIF_Framework &_framework;  // Must be first (reference must be initialized)
      std::vector<uint32_t> _input_shape;
      std::vector<uint32_t> _output_shape;
      bool _initialized;
      bool _training_mode;

      // Private members go here
  };
}//end instance namespace
