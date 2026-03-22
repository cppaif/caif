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
 * @file aif_neural_network.h
 * @brief Neural network class for the CAIF (AI Framework)
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

#include "caif_base.h"
#include "caif_constants.h"
#include "caif_tensor.h"
#include "caif_layer.h"
#include "caif_optimizer.h"
#include "caif_loss_function.h"
#include "caif_framework.h"
#include <vector>
#include <memory>
#include <string>
#include "caif_exception.h"

namespace instance
{


  /**
   * @brief Comprehensive neural network implementation
   * 
   * The CAIF_NeuralNetwork class provides a flexible neural network
   * implementation supporting various layer types, optimizers, and
   * loss functions for both training and inference.
   */
  class CAIF_NeuralNetwork:public CAIF_Base
  {
    public:
      /**
       * @brief Training configuration structure
       */
      struct TrainingConfig_t
      {
        uint32_t epochs;
        uint32_t batch_size;
        float learning_rate;
        CAIF_OptimizerType_e optimizer_type;
        CAIF_LossType_e loss_type;
        bool shuffle_data;
        bool use_validation;
        float validation_split;
      };

      /**
       * @brief Structure to hold training metrics for each epoch
       */
      struct CAIF_TrainingMetrics_t
      {
        uint32_t epoch = 0;
        float loss = 0.0f;
        float accuracy = 0.0f;
        float val_loss = 0.0f;
        float val_accuracy = 0.0f;
        bool has_accuracy = false;
        bool has_validation = false;
        bool has_val_accuracy = false;
      };

      /**
       * @brief Default constructor - creates an empty neural network
       * Creates its own CAIF_Framework instance
       */
      CAIF_NeuralNetwork();
      
      /**
       * @brief Constructor with framework reference
       * @param framework Reference to CAIF framework instance
       */
      explicit CAIF_NeuralNetwork(CAIF_Framework &framework);
      
      /**
       * @brief Copy constructor
       * @param other Network to copy from
       * @note Creates a new framework instance for the copied network
       */
      CAIF_NeuralNetwork(const CAIF_NeuralNetwork &other);
      
      /**
       * @brief Move constructor
       * @param other Network to move from
       */
      CAIF_NeuralNetwork(CAIF_NeuralNetwork &&other);
      
      /**
       * @brief Destructor
       */
      ~CAIF_NeuralNetwork()=default;

      /**
       * @brief Copy assignment operator
       * @param other Network to copy from
       * @return Reference to this network
       */
      CAIF_NeuralNetwork &operator=(const CAIF_NeuralNetwork &other);
      
      /**
       * @brief Move assignment operator
       * @param other Network to move from
       * @return Reference to this network
       */
      CAIF_NeuralNetwork &operator=(CAIF_NeuralNetwork &&other);

      // Network construction methods
      
      /**
       * @brief Add a dense/fully connected layer
       * @param units Number of neurons in the layer
       * @param activation Activation function type
       * @param use_bias Whether to use bias in the layer
       * @return Expected with void on success or error message
       */
      void AddDenseLayer(const uint32_t units,
                         const CAIF_ActivationType_e activation,
                         const bool use_bias=true
                        );
      
      /**
       * @brief Add a 2D convolution layer
       * @param filters Number of filters/channels
       * @param kernel_size Size of convolution kernel
       * @param stride Stride of convolution
       * @param padding Amount of padding
       * @param activation Activation function type
       * @return Expected with void on success or error message
       */
      void AddConvolution2DLayer(const uint32_t filters,
                                 const uint32_t kernel_size,
                                 const uint32_t stride=1,
                                 const uint32_t padding=0,
                                 const CAIF_ActivationType_e activation=CAIF_ActivationType_e::ReLU
                                );
      
      /**
       * @brief Add a 2D max pooling layer
       * @param pool_size Size of pooling window
       * @param stride Stride of pooling operation
       * @return Expected with void on success or error message
       */
      void AddMaxPooling2DLayer(const uint32_t pool_size, const uint32_t stride=1);
      
      /**
       * @brief Add a 2D average pooling layer
       * @param pool_size Size of pooling window (square)
       * @param stride Stride of pooling operation
       * @return Expected with void on success or error message
       */
      void AddAveragePooling2DLayer(const uint32_t pool_size, const uint32_t stride=1);
      
      /**
       * @brief Add a dropout layer for regularization
       * @param rate Dropout rate (0.0 to 1.0)
       * @return Expected with void on success or error message
       */
      void AddDropoutLayer(const float rate=g_caif_default_dropout_rate);
      
      /**
       * @brief Add a batch normalization layer
       * @param momentum Momentum for running statistics
       * @param epsilon Small value to avoid division by zero
       * @return Expected with void on success or error message
       */
      void AddBatchNormalizationLayer(const float momentum=g_caif_default_momentum,
                                      const float epsilon=g_caif_epsilon
                                     );
      
      /**
       * @brief Add a flatten layer to convert multi-dimensional input to 1D
       * @return Expected with void on success or error message
       */
      void AddFlattenLayer();
      
      /**
       * @brief Add a reshape layer to the network
       * @param target_shape Shape to reshape to (excluding batch dimension)
       * @return Expected with void on success or error message
       */
      void AddReshapeLayer(const std::vector<uint32_t> &target_shape);
      
      // Training and inference methods
      
      /**
       * @brief Compile the network with optimizer and loss function
       * @param optimizer_type Type of optimizer to use
       * @param loss_type Type of loss function to use
       * @param learning_rate Learning rate for training
       * @return Expected with void on success or error message
       */
      void Compile(const CAIF_OptimizerType_e optimizer_type,
                   const CAIF_LossType_e loss_type,
                   const float learning_rate=g_caif_default_learning_rate
                  );
      
      /**
       * @brief Train the network on provided data
       * @param input_data Training input data
       * @param target_data Training target data
       * @param config Training configuration
       * @return Expected with training metrics or error message
       */
      std::vector<CAIF_NeuralNetwork::CAIF_TrainingMetrics_t> Train(const CAIF_Tensor &input_data,
                                                                  const CAIF_Tensor &target_data,
                                                                  const TrainingConfig_t &config
                                                                 );
      
      /**
       * @brief Perform forward pass through the network
       * @param input Input tensor
       * @param training Whether in training mode (affects dropout/batch norm)
       * @return Expected with output tensor or error message
       */
      CAIF_Tensor Forward( const CAIF_Tensor &input, const bool training=false);
      
      /**
       * @brief Perform inference on input data
       * @param input Input tensor for prediction
       * @return Expected with prediction tensor or error message
       */
      CAIF_Tensor Predict(const CAIF_Tensor &input)const;
      
      /**
       * @brief Evaluate network performance on test data
       * @param input_data Test input data
       * @param target_data Test target data
       * @return Expected with evaluation metrics or error message
       */
      CAIF_NeuralNetwork::CAIF_TrainingMetrics_t Evaluate(
        const CAIF_Tensor &input_data,
        const CAIF_Tensor &target_data);
      
      /**
       * @brief Extract features from a specific layer (typically penultimate layer)
       * @param input Input tensor for feature extraction
       * @param layer_index Index of layer to extract features from (-1 for penultimate layer)
       * @return Expected with feature tensor or error message
       */
      CAIF_Tensor ExtractFeatures( const CAIF_Tensor &input, const int32_t layer_index=-1);

      // Model persistence
      
      /**
       * @brief Save the neural network model to a file
       * @param filepath Path where to save the model
       * @return Expected with void on success or error message
       */
      void SaveModel(const std::string &filepath)const;
      
      /**
       * @brief Load a neural network model from a file
       * @param filepath Path to the model file
       * @return Expected with void on success or error message
       */
      void LoadModel(const std::string &filepath);
      
      /**
       * @brief Get the framework reference
       * @return Reference to CAIF_Framework instance
       */
      CAIF_Framework &Framework(){return _framework;}
      const CAIF_Framework &Framework()const{return _framework;}
      
      /**
       * @brief Export network architecture to string
       * @return String representation of network architecture
       */
      std::string ExportArchitecture()const;

      // ters and setters
      
      /**
       * @brief  number of layers in the network
       * @return Number of layers
       */
      uint32_t LayerCount()const{return static_cast<uint32_t>(_layers.size());}
      
      /**
       * @brief Check if network is compiled
       * @return True if compiled, false otherwise
       */
      bool IsCompiled()const{return _compiled;}
      
      /**
       * @brief Check if network is trained
       * @return True if trained, false otherwise
       */
      bool IsTrained()const{return _trained;}
      
      /**
       * @brief  input shape of the network
       * @return Const reference to input shape vector
       */
      const std::vector<uint32_t> &InputShape()const{return _input_shape;}
      
      /**
       * @brief  output shape of the network
       * @return Const reference to output shape vector
       */
      const std::vector<uint32_t> &OutputShape()const{return _output_shape;}
      
      /**
       * @brief Set input shape for the network
       * @param shape Input shape vector (must have at least 2 dimensions: [batch_size, features...])
       * @return Expected with void on success or error message
       */
      void SetInputShape(const std::vector<uint32_t> &shape);
      
      /**
       * @brief  current learning rate
       * @return Current learning rate
       */
      float LearningRate()const{return _learning_rate;}
      
      /**
       * @brief Set learning rate
       * @param rate New learning rate
       */
      void SetLearningRate(const float rate){_learning_rate=rate;}

      // Advanced features
      
      /**
       * @brief  gradients from last backward pass
       * @return Vector of gradient tensors for each layer
       */
      const std::vector<CAIF_Tensor> &Gradients()const{return _gradients;}
      
      /**
       * @brief Apply regularization to weights
       * @param l1_lambda L1 regularization strength
       * @param l2_lambda L2 regularization strength
       * @return Expected with void on success or error message
       */
      void ApplyRegularization( const float l1_lambda=0.0f, const float l2_lambda=0.0f);
      
      /**
       * @brief Reset network weights to random values
       * @param seed Random seed for reproducibility
       * @return Expected with void on success or error message
       */
      void ResetWeights(const uint32_t seed=0);

      /**
       * @brief  network's loss type
       * @return Loss type
       */
      CAIF_LossType_e LossType()const{return _loss_type;}
      
      /**
       * @brief  network's optimizer type
       * @return Optimizer type
       */
      CAIF_OptimizerType_e OptimizerType()const{return _optimizer_type;}
      
      /**
       * @brief Check if optimizer state is available
       * @return True if optimizer is initialized and state can be exported
       */
      bool IsOptimized()const{return _optimizer!=nullptr;}
      
      /**
       * @brief  training iterations (epochs) completed
       * @return Number of iterations
       */
      uint32_t TrainingIterations()const{return _training_iterations;}
      
      /**
       * @brief  model version string
       * @return Model version (e.g., "1.0.0")
       */
      std::string ModelVersion()const{return _model_version;}
      
      /**
       * @brief Set model version string
       * @param version Model version (e.g., "1.0.0")
       */
      void SetModelVersion(const std::string &version){_model_version=version;}
      
      /**
       * @brief Check if metrics history is available
       * @return True if metrics history exists
       */
      bool HasMetricsHistory()const{return _metrics_history.empty()==false;}
      
      /**
       * @brief  metrics history
       * @return Vector of training metrics
       */
      const std::vector<CAIF_NeuralNetwork::CAIF_TrainingMetrics_t>& MetricsHistory()const
      {
        return _metrics_history;
      }
      
      /**
       * @brief Set metrics history
       * @param metrics Vector of training metrics
       */
      void SetMetricsHistory(const std::vector<CAIF_NeuralNetwork::CAIF_TrainingMetrics_t>& metrics)
      {
        _metrics_history=metrics;
      }
      
      /**
       * @brief  optimizer state tensors
       * @return Vector of optimizer state tensors
       */
      std::vector<CAIF_Tensor> OptimizerState()const;
      
      /**
       * @brief Set optimizer state tensors
       * @param state Vector of optimizer state tensors
       * @return Expected with void on success or error message
       */
      void SetOptimizerState(const std::vector<CAIF_Tensor> &state);
      
      /**
       * @brief  a layer by index
       * @param index Index of the layer to get
       * @return Reference to the layer
       * @throws std::out_of_range if index is out of range
       */
      CAIF_Layer& Layer(const uint32_t index);
      
      /**
       * @brief  a layer by index (const version)
       * @param index Index of the layer to get
       * @return Const reference to the layer
       * @throws std::out_of_range if index is out of range
       */
      const CAIF_Layer& Layer(const uint32_t index)const;

      // Accessors for layers (read-only and mutable) for external builders
      const std::vector<std::unique_ptr<CAIF_Layer>> &Layers()const{return _layers;}
      std::vector<std::unique_ptr<CAIF_Layer>> &Layers(){return _layers;}

      /**
       * @brief Perform backward pass through the network (public)
       * @param output_gradient Gradient from loss function or target signal
       * @return Expected with void on success or error message
       */
      void BackwardPass(const CAIF_Tensor &output_gradient);

      /**
       * @brief Convenience API to compute gradients given an output gradient
       * @param output_gradient Gradient tensor shaped like model output
       * @return Expected with void on success or error message
       */
      void ComputeGradients(const CAIF_Tensor &output_gradient);

    protected:
      // Protected members go here

    private:
      CAIF_Framework _framework;  // Framework instance
      std::vector<std::unique_ptr<CAIF_Layer>> _layers;
      std::unique_ptr<CAIF_Optimizer> _optimizer;
      std::unique_ptr<CAIF_LossFunction> _loss_function;
      
      std::vector<uint32_t> _input_shape;
      std::vector<uint32_t> _output_shape;
      std::vector<CAIF_Tensor> _gradients;
      uint32_t _init_seed=1234;
      
      CAIF_OptimizerType_e _optimizer_type;
      CAIF_LossType_e _loss_type;
      float _learning_rate;
      
      bool _compiled;
      bool _trained;
      uint32_t _training_epochs;
      uint32_t _training_iterations;
      std::string _model_version;
      std::vector<CAIF_TrainingMetrics_t> _metrics_history;

      /**
       * @brief Perform backward pass through the network
       * @param output_gradient Gradient from loss function
       * @return Expected with void on success or error message
       */
      void Backward(const CAIF_Tensor &output_gradient);
      
      /**
       * @brief Initialize network weights
       * @param seed Random seed for initialization
       * @return Expected with void on success or error message
       */
      void InitializeWeights(const uint32_t seed=0);
      
      /**
       * @brief Validate network architecture before compilation
       * @return Expected with void on success or error message
       */
      void ValidateArchitecture()const;
      
      /**
       * @brief Calculate output shape based on current layers
       * @return Expected with output shape or error message
       */
      std::vector<uint32_t> CalculateOutputShape()const;
      
      /**
       * @brief Validate input shape for the network
       * @param shape Input shape vector to validate
       * @return Expected with void on success or error message
       */
      void ValidateInputShape(const std::vector<uint32_t> &shape)const;
      
      /**
       * @brief Shuffle training data
       * @param input_data Input data tensor to shuffle
       * @param target_data Target data tensor to shuffle
       * @param seed Random seed for shuffling
       * @return Expected with void on success or error message
       */
      void ShuffleData(CAIF_Tensor &input_data, CAIF_Tensor &target_data, const uint32_t seed=0)const;
      
      /**
       * @brief Split data into training and validation sets
       * @param input_data Full input dataset
       * @param target_data Full target dataset
       * @param validation_split Fraction of data for validation
       * @return Expected with tuple of train/val data or error message
       */
      std::tuple<CAIF_Tensor,CAIF_Tensor,CAIF_Tensor,CAIF_Tensor> SplitData(const CAIF_Tensor &input_data,
                                                                        const CAIF_Tensor &target_data,
                                                                        const float validation_split
                                                                       )const;
      
      
      
      /**
       * @brief Update network parameters using optimizer
       * @return Expected with void on success or error message
       */
      void UpdateNetworkParameters();
      
      /**
       * @brief Calculate accuracy between predictions and targets
       * @param predictions Model predictions
       * @param targets Ground truth targets
       * @return Number of correct predictions
       */
      uint32_t CalculateAccuracy(const CAIF_Tensor &predictions, const CAIF_Tensor &targets)const;
      
      /**
       * @brief Create optimizer instance based on type
       * @param optimizer_type Type of optimizer to create
       * @param learning_rate Learning rate for optimizer
       * @return Expected with optimizer instance or error message
       */
      std::unique_ptr<CAIF_Optimizer> CreateOptimizer(const CAIF_OptimizerType_e optimizer_type,
                                                     const float learning_rate
                                                    )const;
      
      /**
       * @brief Create loss function instance based on type
       * @param loss_type Type of loss function to create
       * @return Expected with loss function instance or error message
       */
      std::unique_ptr<CAIF_LossFunction> CreateLossFunction( const CAIF_LossType_e loss_type)const;
      
      /**
       * @brief Initialize layer shapes after network compilation
       * @return Expected with void on success or error message
       */
      void InitializeLayerShapes();

      void MaybeApplyBinaryCrossEntropyBias(const float computed_bias_val,bool &have_bias_val);

      bool ShouldLogBatchDetails(const uint32_t epoch,const uint32_t batch_index,const bool is_final_epoch)const;

      bool ShouldLogEpochSummary(const uint32_t epoch,const bool is_final_epoch)const;

      void LogResidualStats(const CAIF_Tensor &predictions,
                            const CAIF_Tensor &batch_target,
                            const uint32_t epoch)const;

      void LogBatchStatistics(const CAIF_Tensor &predictions,
                              const CAIF_Tensor &batch_target,
                              const uint32_t epoch)const;

      double ComputeFirstDenseLayerWeightL2()const;

      void LogWeightL2Delta(const double before,const double after,const uint32_t epoch)const;

      void LogEpochSummary(const CAIF_TrainingMetrics_t &metrics)const;
  };
}//end instance namespace
