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
// Device-resident neural network using CAIF_DeviceTensor
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_NETWORK_H
#define CAIF_DEVICE_NETWORK_H

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_device_layer.h"
#include "caif_device_dense_layer.h"
#include "caif_model_format.h"
#include "caif_cuda_stream.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <string>

namespace instance
{

/**
 * @brief Device-resident neural network
 *
 * A simplified neural network implementation that uses CAIF_DeviceTensor
 * exclusively. All data remains on the GPU during training.
 *
 * Key features:
 * - All layers and data on device
 * - No host memory allocation during training
 * - Simple sequential architecture
 * - Built-in Adam optimizer
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
class CAIF_DeviceNetwork:public CAIF_Base
{
  public:
    /**
     * @brief Construct a device network
     *
     * @param stream CUDA stream for all operations
     */
    explicit CAIF_DeviceNetwork(CAIF_CudaStream &stream);

    /**
     * @brief Destructor
     */
    ~CAIF_DeviceNetwork()=default;

    // Non-copyable
    CAIF_DeviceNetwork(const CAIF_DeviceNetwork &)=delete;
    CAIF_DeviceNetwork &operator=(const CAIF_DeviceNetwork &)=delete;

    // Movable
    CAIF_DeviceNetwork(CAIF_DeviceNetwork &&other);
    CAIF_DeviceNetwork &operator=(CAIF_DeviceNetwork &&other);

    /**
     * @brief Add a dense layer to the network
     *
     * Layers are added sequentially. The first layer's input_size must match
     * the data input size. Subsequent layers auto-connect.
     *
     * @param input_size Number of input features (or 0 to auto-infer from previous layer)
     * @param output_size Number of output units
     * @param activation Activation function
     * @param use_bias Whether to use bias
     */
    void AddDenseLayer(uint32_t input_size,
                       uint32_t output_size,
                       CAIF_DeviceActivation_e activation,
                       bool use_bias=true);

    /**
     * @brief Add a polymorphic layer to the network
     *
     * Allows adding any CAIF_DeviceLayer subclass (transformer, attention, etc.)
     *
     * @param layer Layer to add (ownership transferred)
     */
    void AddLayer(std::unique_ptr<CAIF_DeviceLayer> layer);

    /**
     * @brief Forward pass through all layers
     *
     * @param input Input tensor [batch x input_features]
     * @param training Whether in training mode
     * @return Output tensor [batch x output_features]
     */
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training=false);

    /**
     * @brief Backward pass through all layers
     *
     * Computes gradients for all parameters. Must call Forward with
     * training=true before calling Backward.
     *
     * @param grad_output Gradient from loss function [batch x output_features]
     */
    void Backward(const CAIF_DeviceTensor &grad_output);

    /**
     * @brief Zero all gradients in all layers
     */
    void ZeroGradients();

    /**
     * @brief Initialize Adam optimizer state
     *
     * Must be called before using AdamStep.
     *
     * @param lr Learning rate (default 0.001)
     * @param beta1 First moment decay (default 0.9)
     * @param beta2 Second moment decay (default 0.999)
     * @param epsilon Numerical stability constant (default 1e-8)
     */
    void InitializeAdam(float lr=0.001f,
                        float beta1=0.9f,
                        float beta2=0.999f,
                        float epsilon=1e-8f,
                        float weight_decay=0.0f);

    /**
     * @brief Perform one Adam optimizer update step
     *
     * Updates all parameters using their gradients and Adam state.
     * Must call InitializeAdam before first use.
     */
    void AdamStep();

    /**
     * @brief Set Adam learning rate (for LR scheduling)
     */
    void SetLearningRate(float lr){_adam_lr=lr;}

    /**
     * @brief Clip gradient norm across all trainable parameters
     *
     * Computes the global L2 norm of all trainable gradients and
     * scales them so the total norm does not exceed max_norm.
     * Equivalent to torch.nn.utils.clip_grad_norm_.
     *
     * @param max_norm Maximum allowed gradient norm
     * @return The total gradient norm before clipping
     */
    float ClipGradientNorm(float max_norm);

    /**
     * @brief Get the number of layers
     */
    size_t LayerCount()const{return _layers.size();}

    /**
     * @brief Get a layer by index (polymorphic)
     */
    CAIF_DeviceLayer &Layer(size_t index){return *_layers[index];}
    const CAIF_DeviceLayer &Layer(size_t index)const{return *_layers[index];}

    /**
     * @brief Get a dense layer by index (typed access)
     *
     * Throws if the layer at the given index is not a dense layer.
     */
    CAIF_DeviceDenseLayer &DenseLayer(size_t index);
    const CAIF_DeviceDenseLayer &DenseLayer(size_t index)const;

    /**
     * @brief Mark a layer as trainable or non-trainable
     *
     * Non-trainable layers are skipped by the Adam optimizer (no m/v tensors
     * allocated) and their gradients are not zeroed by ZeroGradients().
     * All layers default to trainable.
     *
     * @param index Layer index
     * @param trainable Whether the layer should be trained
     */
    void SetLayerTrainable(size_t index,bool trainable);

    /**
     * @brief Check if a layer is trainable
     */
    bool IsLayerTrainable(size_t index)const;

    /**
     * @brief Get total parameter count across all layers
     */
    size_t TotalParameterCount()const;

    /**
     * @brief Get the expected input size for the network
     *
     * Returns the input_size of the first layer, or 0 if no layers.
     */
    uint32_t InputSize()const;

    /**
     * @brief Get the output size of the network
     *
     * Returns the output_size of the last layer, or 0 if no layers.
     */
    uint32_t OutputSize()const;

    /**
     * @brief Save network parameters to SafeTensors format
     *
     * Saves all layer parameters using HuggingFace-compatible tensor names.
     * Works with any layer types (polymorphic). Model architecture is stored
     * in SafeTensors metadata.
     *
     * @param path Output file path (.safetensors)
     */
    void SaveSafeTensors(const std::string &path)const;

    /**
     * @brief Load network parameters from SafeTensors file
     *
     * Loads weights into existing network architecture. The network must
     * already have layers added that match the saved model. Parameters
     * are matched by name using ParameterNames().
     *
     * @param path Input file path (.safetensors)
     */
    void LoadSafeTensors(const std::string &path);

    /**
     * @brief Save network using specified format
     *
     * @param path Output file path
     * @param format Model format implementation
     */
    void Save(const std::string &path,const CAIF_ModelFormat &format)const;

    /**
     * @brief Load network parameters using specified format
     *
     * @param path Input file path
     * @param format Model format implementation
     */
    void Load(const std::string &path,const CAIF_ModelFormat &format);

    /**
     * @brief Save the network to legacy binary format (dense layers only)
     *
     * @deprecated Use SaveSafeTensors instead
     *
     * Saves layer architecture, weights, biases, and optionally Adam state.
     * The file format is CAIF device network binary format (.aifdn).
     * Only supports networks with dense layers.
     *
     * @param filepath Path to save the model
     * @param save_optimizer_state Whether to save Adam optimizer state
     */
    void SaveModel(const std::string &filepath,bool save_optimizer_state=true)const;

    /**
     * @brief Load network from legacy binary format (dense layers only)
     *
     * @deprecated Use LoadSafeTensors instead
     *
     * Restores layer architecture, weights, biases, and optionally Adam state.
     * The network must not have any layers added before loading.
     * Only supports networks with dense layers.
     *
     * @param filepath Path to the model file
     * @param load_optimizer_state Whether to load Adam optimizer state if present
     */
    void LoadModel(const std::string &filepath,bool load_optimizer_state=true);

  protected:

  private:
    CAIF_CudaStream *_stream;
    std::vector<std::unique_ptr<CAIF_DeviceLayer>> _layers;
    std::vector<bool> _trainable;
    uint32_t _input_size;
    uint32_t _output_size;

    // Adam optimizer state
    bool _adam_initialized;
    float _adam_lr;
    float _adam_beta1;
    float _adam_beta2;
    float _adam_epsilon;
    float _adam_weight_decay;
    int _adam_t;  // Timestep

    // Adam moment estimates (one pair per parameter tensor)
    std::vector<CAIF_DeviceTensor> _adam_m;  // First moments
    std::vector<CAIF_DeviceTensor> _adam_v;  // Second moments
};

}//end instance namespace

#endif  // CAIF_DEVICE_NETWORK_H
