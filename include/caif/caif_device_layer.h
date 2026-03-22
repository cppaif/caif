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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// Abstract base class for device-resident layers
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_LAYER_H
#define CAIF_DEVICE_LAYER_H

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Abstract base class for all device-resident layers
 *
 * Defines the interface that CAIF_DeviceNetwork uses to manage
 * heterogeneous layer stacks. Every layer stores its parameters
 * and gradients as CAIF_DeviceTensor objects on the GPU.
 *
 * Subclasses must implement Forward, Backward, parameter access,
 * and gradient zeroing.
 */
class CAIF_DeviceLayer:public CAIF_Base
{
  public:
    virtual ~CAIF_DeviceLayer()=default;

    // Non-copyable (device tensors are move-only)
    CAIF_DeviceLayer(const CAIF_DeviceLayer &)=delete;
    CAIF_DeviceLayer &operator=(const CAIF_DeviceLayer &)=delete;

    // Movable
    CAIF_DeviceLayer(CAIF_DeviceLayer &&other)noexcept:_stream(other._stream)
    {
      other._stream=nullptr;
    }

    CAIF_DeviceLayer &operator=(CAIF_DeviceLayer &&other)noexcept
    {
      if(this!=&other)
      {
        _stream=other._stream;
        other._stream=nullptr;
      }
      return *this;
    }

    /**
     * @brief Forward pass
     * @param input Input tensor
     * @param training If true, cache intermediates for backward pass
     * @return Output tensor
     */
    virtual CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,
                                     bool training)=0;

    /**
     * @brief Backward pass
     * @param grad_output Gradient from the next layer
     * @return Gradient with respect to the input
     */
    virtual CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)=0;

    /**
     * @brief Zero all gradient tensors
     */
    virtual void ZeroGradients()=0;

    /**
     * @brief Number of parameter tensors (weights + biases)
     *
     * For example a dense layer with bias returns 2 (weight tensor + bias tensor).
     * A layer with no trainable parameters returns 0.
     */
    virtual size_t ParameterTensorCount()const=0;

    /**
     * @brief Access a parameter tensor by index
     * @param index Parameter tensor index (0-based)
     * @return Reference to the parameter tensor
     */
    virtual CAIF_DeviceTensor &ParameterTensor(size_t index)=0;
    virtual const CAIF_DeviceTensor &ParameterTensor(size_t index)const=0;

    /**
     * @brief Access a gradient tensor by index
     * @param index Gradient tensor index (same ordering as ParameterTensor)
     * @return Reference to the gradient tensor
     */
    virtual CAIF_DeviceTensor &GradientTensor(size_t index)=0;
    virtual const CAIF_DeviceTensor &GradientTensor(size_t index)const=0;

    /**
     * @brief Total number of scalar parameters across all tensors
     */
    virtual size_t TotalParameterCount()const=0;

    /**
     * @brief Human-readable layer description
     */
    virtual std::string Description()const=0;

    /**
     * @brief Get parameter names for serialization.
     *
     * Returns names for each parameter tensor in the same order as
     * ParameterTensor(). Names follow HuggingFace/PyTorch conventions
     * (e.g., "weight", "bias", "q_proj.weight").
     *
     * The prefix parameter allows building hierarchical names:
     * - For standalone layers: prefix="" -> "weight", "bias"
     * - For layers in a model: prefix="layers.0.self_attn." -> "layers.0.self_attn.q_proj.weight"
     *
     * @param prefix Prefix to prepend to each parameter name
     * @return Vector of parameter names (same size as ParameterTensorCount())
     */
    virtual std::vector<std::string> ParameterNames(const std::string &prefix="")const=0;

    /**
     * @brief Get the CUDA stream for this layer
     */
    CAIF_CudaStream *Stream()const{return _stream;}

  protected:
    /**
     * @brief Construct with a CUDA stream
     * @param stream Reference to CUDA stream (stored as pointer for move semantics)
     */
    explicit CAIF_DeviceLayer(CAIF_CudaStream &stream):_stream(&stream){}

    CAIF_CudaStream *_stream;

  private:
};

}//end instance namespace

#endif  // CAIF_DEVICE_LAYER_H
