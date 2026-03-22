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
 * @file aif_average_pooling2d_layer.h
 * @brief Average pooling 2D layer for neural networks
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
 * @brief Average pooling 2D layer for downsampling spatial dimensions
 * 
 * This layer performs average pooling operations on 2D inputs, reducing
 * spatial dimensions by taking the average value in each pooling window.
 * Input format: [batch, height, width, channels]
 */
class CAIF_AveragePooling2DLayer:public CAIF_Layer
{
  public:
    /**
     * @brief Constructor with pooling parameters
     * @param framework Reference to CAIF framework instance
     * @param pool_size Size of pooling window (assumed square)
     * @param stride Stride for pooling operation (default: same as pool_size)
     * @param padding Padding type (0 = valid, >0 = same)
     */
    CAIF_AveragePooling2DLayer(CAIF_Framework &framework,
                              const uint32_t pool_size,
                              const uint32_t stride=0,
                              const uint32_t padding=0);
    
    /**
     * @brief Copy constructor
     * @param other Layer to copy from
     */
    CAIF_AveragePooling2DLayer(const CAIF_AveragePooling2DLayer &other);
    
    /**
     * @brief Move constructor
     * @param other Layer to move from
     */
    CAIF_AveragePooling2DLayer(CAIF_AveragePooling2DLayer &&other)noexcept;
    
    /**
     * @brief Destructor
     */
    virtual ~CAIF_AveragePooling2DLayer()=default;

    /**
     * @brief Copy assignment operator
     * @param other Layer to copy from
     * @return Reference to this layer
     */
    CAIF_AveragePooling2DLayer &operator=(const CAIF_AveragePooling2DLayer &other);
    
    /**
     * @brief Move assignment operator
     * @param other Layer to move from
     * @return Reference to this layer
     */
    CAIF_AveragePooling2DLayer &operator=(CAIF_AveragePooling2DLayer &&other)noexcept;

    /**
     * @brief Perform forward pass
     * @param input Input tensor to pool
     * @param training Whether in training mode
     * @return Pooled tensor
     */
    CAIF_Tensor Forward(const CAIF_Tensor &input,const bool training=false)override;

    /**
     * @brief Perform backward pass
     * @param gradient Gradient from next layer
     * @return Unpooled gradient
     */
    CAIF_Tensor Backward(const CAIF_Tensor &gradient)override;

    /**
     * @brief Initialize the layer
     * @param input_shape Shape of input tensor
     * @param seed Random seed (unused for pooling layer)
     */
    void Initialize(const std::vector<uint32_t> &input_shape,const uint32_t seed=0)override;

    /**
     * @brief Calculate output shape given input shape
     * @param input_shape Input tensor shape
     * @return Output shape
     */
    std::vector<uint32_t> CalculateOutputShape( const std::vector<uint32_t> &input_shape)const override;

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
    CAIF_LayerType_e LayerType()const override{return CAIF_LayerType_e::AveragePooling2D;}

    /**
     * @brief Get string description of layer
     * @return Description string
     */
    std::string Description()const override;
    
    /**
     * @brief Get pool size height
     * @return Pool size height
     */
    uint32_t PoolHeight()const{return _pool_size;}
    
    /**
     * @brief Get pool size width
     * @return Pool size width
     */
    uint32_t PoolWidth()const{return _pool_size;}
    
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

  protected:

  private:
    uint32_t _pool_size;  ///< Size of pooling window
    uint32_t _stride;     ///< Stride for pooling operation
    uint32_t _padding;    ///< Padding amount

    /**
     * @brief Calculate pooled output dimensions
     * @param input_dim Input dimension size
     * @return Output dimension size
     */
    uint32_t CalculateOutputDim(const uint32_t input_dim)const;
};
}//end instance namespace
