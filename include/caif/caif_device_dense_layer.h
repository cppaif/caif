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
// AIF - AI Framework
// Device-resident dense layer using CAIF_DeviceTensor
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_DENSE_LAYER_H
#define CAIF_DEVICE_DENSE_LAYER_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <vector>
#include <cstdint>
#include <string>

namespace instance
{

/**
 * @brief Activation type for device layers
 */
enum class CAIF_DeviceActivation_e
{
  None,        // Linear/identity
  ReLU,
  Sigmoid,
  Tanh,
  Softmax,
  LeakyReLU,   // alpha=0.01 by default
  ELU,         // alpha=1.0 by default
  GELU,        // Gaussian Error Linear Unit
  Swish        // x * sigmoid(x)
};

/**
 * @brief Device-resident dense (fully connected) layer
 *
 * This layer uses CAIF_DeviceTensor exclusively for all data storage.
 * All operations are performed on the GPU without host memory allocation
 * during training.
 *
 * Key features:
 * - Weights and biases stored on device
 * - Forward/backward passes entirely on GPU
 * - No synchronization during forward/backward
 * - Uses CAIF_DeviceOps for all computations
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
class CAIF_DeviceDenseLayer:public CAIF_DeviceLayer
{
  public:
    /**
     * @brief Construct a dense layer
     *
     * @param input_size Number of input features
     * @param output_size Number of output units
     * @param activation Activation function to apply
     * @param stream CUDA stream for operations
     * @param use_bias Whether to use bias terms (default true)
     */
    CAIF_DeviceDenseLayer(uint32_t input_size,
                         uint32_t output_size,
                         CAIF_DeviceActivation_e activation,
                         CAIF_CudaStream &stream,
                         bool use_bias=true);

    /**
     * @brief Destructor
     */
    ~CAIF_DeviceDenseLayer()override=default;

    // Non-copyable (inherited from CAIF_DeviceLayer)

    // Movable
    CAIF_DeviceDenseLayer(CAIF_DeviceDenseLayer &&other);
    CAIF_DeviceDenseLayer &operator=(CAIF_DeviceDenseLayer &&other);

    // --- CAIF_DeviceLayer interface ---

    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training)override;

    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)override;

    void ZeroGradients()override;

    size_t ParameterTensorCount()const override;

    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;

    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;

    size_t TotalParameterCount()const override;

    std::string Description()const override;

    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // --- Dense-layer-specific accessors ---

    /**
     * @brief Initialize weights using Xavier/Glorot initialization
     * @param seed Random seed (0 for time-based)
     */
    void InitializeWeights(uint32_t seed=0);

    // Parameter access (for optimizer and serialization)
    CAIF_DeviceTensor &Weights(){return _weights;}
    const CAIF_DeviceTensor &Weights()const{return _weights;}

    CAIF_DeviceTensor &Bias(){return _bias;}
    const CAIF_DeviceTensor &Bias()const{return _bias;}

    // Gradient access (for optimizer and serialization)
    CAIF_DeviceTensor &WeightGradients(){return _weight_grads;}
    const CAIF_DeviceTensor &WeightGradients()const{return _weight_grads;}

    CAIF_DeviceTensor &BiasGradients(){return _bias_grads;}
    const CAIF_DeviceTensor &BiasGradients()const{return _bias_grads;}

    // Layer info
    uint32_t InputSize()const{return _input_size;}
    uint32_t OutputSize()const{return _output_size;}
    CAIF_DeviceActivation_e Activation()const{return _activation;}
    bool UseBias()const{return _use_bias;}

  protected:

  private:
    uint32_t _input_size;
    uint32_t _output_size;
    CAIF_DeviceActivation_e _activation;
    bool _use_bias;

    // Parameters (device-resident)
    CAIF_DeviceTensor _weights;       // [input_size x output_size]
    CAIF_DeviceTensor _bias;          // [output_size]

    // Gradients (device-resident)
    CAIF_DeviceTensor _weight_grads;  // [input_size x output_size]
    CAIF_DeviceTensor _bias_grads;    // [output_size]

    // Pre-allocated output buffer
    CAIF_DeviceTensor _output_buffer;      // [batch x output_size]
    uint32_t _output_batch;

    // Cached for backward pass (device-resident)
    CAIF_DeviceTensor _last_input;         // [batch x input_size]
    CAIF_DeviceTensor _last_preactivation; // [batch x output_size]
    CAIF_DeviceTensor _last_output;        // [batch x output_size] (for sigmoid/tanh backward)
};

}//end instance namespace

#endif  // CAIF_DEVICE_DENSE_LAYER_H
