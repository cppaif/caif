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
 * @file test_neural_network_training.cpp
 * @brief Test suite for CAIF neural network training functionality
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_tensor.h"
#include "caif_dense_layer.h"
#include "caif_neural_network.h"
#include "caif_optimizer.h"
#include "caif_loss_function.h"
#include "caif_constants.h"
#include "caif_base.h"
#include "caif_settings.h"
#include "caif_tensor_backend.h"
#include "caif_framework.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

using namespace instance;

// Global framework for tests
static CAIF_Framework g_test_framework;

static instance::CAIF_TensorBackend::BackendType_e BackendFromArg(const std::string &value)
{
  if(value=="cuda"||value=="CUDA")
  {
    return instance::CAIF_TensorBackend::BackendType_e::CUDA;
  }
  if(value=="cpu"||value=="CPU"||value=="blas"||value=="BLAS")
  {
    // BLAS is the default/preferred CPU backend (faster than Eigen)
    return instance::CAIF_TensorBackend::BackendType_e::BLAS;
  }
  if(value=="eigen"||value=="EIGEN")
  {
    return instance::CAIF_TensorBackend::BackendType_e::Eigen;
  }
  if(value=="blas"||value=="BLAS")
  {
    return instance::CAIF_TensorBackend::BackendType_e::BLAS;
  }
  if(value=="vulkan"||value=="VULKAN")
  {
    return instance::CAIF_TensorBackend::BackendType_e::Vulkan;
  }
  return instance::CAIF_TensorBackend::BackendType_e::Auto;
}

static void ConfigureBackendFromArgs(int argc,char *argv[])
{
  for(int i=1;i<argc;++i)
  {
    const std::string arg(argv[i]);
    const std::string prefix="--backend=";
    if(arg.rfind(prefix,0)==0)
    {
      const std::string value=arg.substr(prefix.size());
      const auto backend=BackendFromArg(value);
      instance::CAIF_Settings::SetBackendOverride(backend);
    }
  }
}

/**
 * @brief Helper function to create simple tensor with specified values
 */
CAIF_Tensor CreateTensor(
                        const std::vector<uint32_t> &shape,
                        const std::vector<float> &values
                       )
{
  CAIF_Tensor tensor(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  auto data_result=tensor.MutableData<float>();
  if(data_result!=nullptr)
  {
    float *data=data_result;
    for(size_t i=0;i<values.size()&&i<tensor.NumElements();++i)
    {
      data[i]=values[i];
    }
  }
  return tensor;
}

/**
 * @brief Helper function to create random tensor
 */
CAIF_Tensor CreateRandomTensor(
                              const std::vector<uint32_t> &shape,
                              const float min_val=-1.0f,
                              const float max_val=1.0f
                             )
{
  CAIF_Tensor tensor(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  auto data_result=tensor.MutableData<float>();
  if(data_result!=nullptr)
  {
    float *data=data_result;
    for(uint32_t i=0;i<tensor.NumElements();++i)
    {
      // Simple linear congruential generator for reproducible random values
      static uint32_t seed=12345;
      seed=(1103515245*seed+12345)&0x7fffffff;
      float rand_val=static_cast<float>(seed)/static_cast<float>(0x7fffffff);
      data[i]=min_val+(max_val-min_val)*rand_val;
    }
  }
  return tensor;
}

/**
 * @brief Test neural network compilation
 */
bool TestNeuralNetworkCompilation()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network Compilation... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Set input shape
    network.SetInputShape({1, 10});
    
    // Add layers
    network.AddDenseLayer(5,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(3,CAIF_ActivationType_e::Sigmoid);
    
    // Compile network
    network.Compile(CAIF_OptimizerType_e::SGD,CAIF_LossType_e::MeanSquaredError,0.01f);
    
    // Verify compilation state
    if(network.IsCompiled()==false)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Network should be compiled\n";
      return false;
    }
    
    // Verify input and output shapes
    const auto &input_shape=network.InputShape();
    const auto &output_shape=network.OutputShape();
    
    if(input_shape.size()!=2||input_shape[0]!=1||input_shape[1]!=10)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Input shape mismatch\n";
      return false;
    }
    
    if(output_shape.size()!=2||output_shape[0]!=1||output_shape[1]!=3)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Output shape mismatch\n";
      return false;
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test neural network forward pass
 */
bool TestNeuralNetworkForward()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network Forward Pass... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Set input shape and add layers
    network.SetInputShape({1, 4});  // batch_size=1, 4 features
    
    network.AddDenseLayer(2,CAIF_ActivationType_e::ReLU);
    
    // Compile network
    network.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::MeanSquaredError,0.001f);
    
    // Create input tensor
    CAIF_Tensor input=CreateTensor({1,4},{1.0f,2.0f,3.0f,4.0f});  // batch_size=1, 4 features
    
    // Forward pass
    CAIF_Tensor output=network.Forward(input,false);
    
    // Verify output shape
    const auto &output_shape=output.Shape();
    if(output_shape.size()!=2||output_shape[0]!=1||output_shape[1]!=2)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Output shape mismatch: expected [1,2] but got [";
      for(size_t i=0;i<output_shape.size();++i)
      {
        if(i>0)instance::CAIF_Base::SOut()<<",";
        instance::CAIF_Base::SOut()<<output_shape[i];
      }
      instance::CAIF_Base::SOut()<<"]\n";
      return false;
    }
    
    // Output should contain some values (not all zeros due to random initialization)
    auto output_data_result=output.ConstData<float>();
    if(output_data_result==nullptr)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Could not access output data\n";
      return false;
    }
    
    const float *output_data=output_data_result;
    bool has_non_zero=false;
    for(uint32_t i=0;i<output.NumElements();++i)
    {
      if(std::abs(output_data[i])>1e-6f)
      {
        has_non_zero=true;
        break;
      }
    }
    
    if(has_non_zero==false)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Output appears to be all zeros (likely initialization issue)\n";
      return false;
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test neural network prediction
 */
bool TestNeuralNetworkPrediction()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network Prediction... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Set input shape and add layers
    network.SetInputShape({1, 3});  // [batch_size=1, features=3]
    
    network.AddDenseLayer(1,CAIF_ActivationType_e::Sigmoid);
    
    // Compile network
    network.Compile(CAIF_OptimizerType_e::SGD,CAIF_LossType_e::MeanSquaredError,0.1f);
    
    // Create input tensor with batch_size=2
    CAIF_Tensor input=CreateTensor({2, 3},{1.0f,0.5f,0.2f,0.8f,0.3f,0.9f});  // [batch_size=2, features=3]
    
    // Predict
    CAIF_Tensor prediction=network.Predict(input);
    
    // Verify prediction shape
    const auto &pred_shape=prediction.Shape();
    if(pred_shape.size()!=2||pred_shape[0]!=2||pred_shape[1]!=1)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Prediction shape mismatch\n";
      return false;
    }
    
    // Verify prediction values are reasonable (sigmoid output should be [0,1])
    auto pred_data_result=prediction.ConstData<float>();
    if(pred_data_result==nullptr)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Could not access prediction data\n";
      return false;
    }
    
    const float *pred_data=pred_data_result;
    for(uint32_t i=0;i<prediction.NumElements();++i)
    {
      if(pred_data[i]<0.0f||pred_data[i]>1.0f)
      {
        instance::CAIF_Base::SOut()<<"FAILED - Sigmoid output should be between 0 and 1, got "<<pred_data[i]<<"\n";
        return false;
      }
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test simple neural network training (regression)
 */
bool TestNeuralNetworkTrainingRegression()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network Training - Regression... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Create simple regression problem: y = 2*x + 1
    network.SetInputShape({1, 1});  // [batch_size=1, features=1]
    
    network.AddDenseLayer(1,CAIF_ActivationType_e::Linear);
    
    // Compile with MSE loss for regression
    network.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::MeanSquaredError,0.01f);
    
    // Create training data: y = 2*x + 1
    CAIF_Tensor input_data=CreateTensor({4, 1},{1.0f,2.0f,3.0f,4.0f});  // [batch_size=4, features=1]
    CAIF_Tensor target_data=CreateTensor({4, 1},{3.0f,5.0f,7.0f,9.0f});  // [batch_size=4, features=1]
    
    // Training configuration
    CAIF_NeuralNetwork::TrainingConfig_t config;
    config.epochs=10;
    config.batch_size=4;
    config.learning_rate=0.01f;
    config.optimizer_type=CAIF_OptimizerType_e::Adam;
    config.loss_type=CAIF_LossType_e::MeanSquaredError;
    config.shuffle_data=false;
    config.use_validation=false;
    config.validation_split=0.0f;
    
    // Train the network
    instance::CAIF_Base::SOut()<<"  [Regression] Training...\n";
    const auto training_history=network.Train(input_data,target_data,config);
    instance::CAIF_Base::SOut()<<"  [Regression] Training done.\n";
    
    // Verify training history
    if(training_history.size()!=config.epochs)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Training history size mismatch\n";
      return false;
    }
    
    // Loss should decrease over time (at least from first to last epoch)
    if(training_history.back().loss>=training_history.front().loss)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Loss did not decrease during training\n";
      return false;
    }
    
    // Verify network is marked as trained
    if(network.IsTrained()==false)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Network should be marked as trained\n";
      return false;
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test neural network training with classification
 */
bool TestNeuralNetworkTrainingClassification()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network Training - Classification... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Create simple classification problem
    network.SetInputShape({1, 2});  // [batch_size=1, features=2]
    
    network.AddDenseLayer(2,CAIF_ActivationType_e::Softmax);  // 2 classes
    
    // Compile with Cross Entropy loss for classification
    network.Compile(CAIF_OptimizerType_e::SGD,CAIF_LossType_e::CrossEntropy,0.1f);
    
    // Create simple classification data
    // Class 0: points near origin, Class 1: points far from origin
    // [batch_size=4, features=2]
    CAIF_Tensor input_data=CreateTensor({4,2},
                                       {0.1f,0.1f,
                                        0.2f,0.2f,
                                        0.8f,0.8f,
                                        0.9f,0.9f});
    // [batch_size=4, classes=2] one-hot encoded
    CAIF_Tensor target_data=CreateTensor({4,2},
                                        {1.0f,0.0f,
                                         1.0f,0.0f,
                                         0.0f,1.0f,
                                         0.0f,1.0f});
    
    // Training configuration
    CAIF_NeuralNetwork::TrainingConfig_t config;
    config.epochs=5;
    config.batch_size=4;
    config.learning_rate=0.1f;
    config.optimizer_type=CAIF_OptimizerType_e::SGD;
    config.loss_type=CAIF_LossType_e::CrossEntropy;
    config.shuffle_data=false;
    config.use_validation=false;
    config.validation_split=0.0f;
    
    // Train the network
    const auto training_history=network.Train(input_data,target_data,config);
    
    // Verify training completed
    if(training_history.size()!=config.epochs)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Training history size mismatch\n";
      return false;
    }
    
    // Test prediction after training
    instance::CAIF_Base::SOut()<<"  [Classification] Predicting...\n";
    CAIF_Tensor predictions=network.Predict(input_data);
    instance::CAIF_Base::SOut()<<"  [Classification] Predict done.\n";
    
    // Verify prediction shape
    const auto &pred_shape=predictions.Shape();
    if(pred_shape.size()!=2||pred_shape[0]!=4||pred_shape[1]!=2)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Prediction shape mismatch after training\n";
      return false;
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test neural network error handling
 */
bool TestNeuralNetworkErrorHandling()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network Error Handling... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Test training without compilation
    CAIF_Tensor dummy_input=CreateTensor({2, 1},{1.0f,2.0f});  // [batch_size=2, features=1]
    CAIF_Tensor dummy_target=CreateTensor({2, 1},{3.0f,4.0f});  // [batch_size=2, features=1]
    
    CAIF_NeuralNetwork::TrainingConfig_t config;
    config.epochs=1;
    config.batch_size=2;
    config.learning_rate=0.01f;
    config.optimizer_type=CAIF_OptimizerType_e::SGD;
    config.loss_type=CAIF_LossType_e::MeanSquaredError;
    config.shuffle_data=false;
    config.use_validation=false;
    config.validation_split=0.0f;
    
    try
    {
      network.Train(dummy_input,dummy_target,config);
      instance::CAIF_Base::SOut()
        <<"FAILED - Should have failed training "
        <<"without compilation\n";
      return false;
    }
    catch(const instance::CAIF_Exception &)
    {
      /* expected */
    }
    
    // Test forward pass without compilation
    try
    {
      network.Forward(dummy_input,false);
      instance::CAIF_Base::SOut()
        <<"FAILED - Should have failed forward "
        <<"pass without compilation\n";
      return false;
    }
    catch(const instance::CAIF_Exception &)
    {
      /* expected */
    }
    
    // Test compilation without layers
    try
    {
      network.Compile(CAIF_OptimizerType_e::SGD,
                      CAIF_LossType_e::MeanSquaredError,
                      0.01f);
      instance::CAIF_Base::SOut()
        <<"FAILED - Should have failed compilation "
        <<"without layers\n";
      return false;
    }
    catch(const instance::CAIF_Exception &)
    {
      /* expected */
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test neural network with multiple layers
 */
bool TestNeuralNetworkMultipleLayers()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network - Multiple Layers... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Create multi-layer network
    network.SetInputShape({1, 4});  // [batch_size=1, features=4]
    
    network.AddDenseLayer(8,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(4,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(1,CAIF_ActivationType_e::Sigmoid);
    
    // Compile network
    network.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::MeanSquaredError,0.001f);
    
    // Verify layer count
    if(network.LayerCount()!=3)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Layer count mismatch: expected 3, got "<<network.LayerCount()<<"\n";
      return false;
    }
    
    // Test forward pass
    CAIF_Tensor input=CreateRandomTensor({4, 4},-1.0f,1.0f);  // [batch_size=4, features=4]
    instance::CAIF_Base::SOut()<<"  [MultiLayers] Forward...\n";
    CAIF_Tensor output=network.Forward(input,false);
    instance::CAIF_Base::SOut()<<"  [MultiLayers] Forward done.\n";
    
    // Verify output shape
    const auto &output_shape=output.Shape();
    if(output_shape.size()!=2||output_shape[0]!=4||output_shape[1]!=1)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Output shape mismatch\n";
      return false;
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    instance::CAIF_Base::SOut()<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test neural network feature extraction functionality
 */
bool TestNeuralNetworkFeatureExtraction()
{
  instance::CAIF_Base::SOut()<<"Testing Neural Network Feature Extraction... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Set input shape
    network.SetInputShape({1, 6});  // [batch_size=1, features=6]
    
    // Add multiple layers to create a feature hierarchy
    network.AddDenseLayer(12,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(8,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(4,CAIF_ActivationType_e::ReLU);  // Feature layer
    
    network.AddDenseLayer(2,CAIF_ActivationType_e::Softmax);  // Output layer
    
    // Compile network
    network.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::MeanSquaredError,0.001f);
    
    // Create test input
    CAIF_Tensor input=CreateTensor({1, 6},{1.0f,0.5f,-0.3f,0.8f,-0.2f,0.6f});  // [batch_size=1, features=6]
    
    // Test default feature extraction (penultimate layer - layer 2, index 2)
    CAIF_Tensor features=network.ExtractFeatures(input);
    
    // Verify feature shape (should be from penultimate layer - layer 3 with 4 units)
    const auto &feature_shape=features.Shape();
    if(feature_shape.size()!=2||feature_shape[0]!=1||feature_shape[1]!=4)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Default feature shape mismatch: expected [1,4] but got [";
      for(size_t i=0;i<feature_shape.size();++i)
      {
        if(i>0)instance::CAIF_Base::SOut()<<",";
        instance::CAIF_Base::SOut()<<feature_shape[i];
      }
      instance::CAIF_Base::SOut()<<"]\n";
      return false;
    }
    
    // Test explicit layer feature extraction (layer 1 - 8 units)
    CAIF_Tensor layer1_features=network.ExtractFeatures(input,1);
    const auto &layer1_shape=layer1_features.Shape();
    if(layer1_shape.size()!=2||layer1_shape[0]!=1||layer1_shape[1]!=8)
    {
      instance::CAIF_Base::SOut()<<"FAILED - Layer 1 feature shape mismatch: expected [1,8] but got [";
      for(size_t i=0;i<layer1_shape.size();++i)
      {
        if(i>0)instance::CAIF_Base::SOut()<<",";
        instance::CAIF_Base::SOut()<<layer1_shape[i];
      }
      instance::CAIF_Base::SOut()<<"]\n";
      return false;
    }
    
    // Test error handling - invalid layer index
    try
    {
      network.ExtractFeatures(input,10);
      instance::CAIF_Base::SOut()
        <<"FAILED - Should have failed with "
        <<"invalid layer index\n";
      return false;
    }
    catch(const instance::CAIF_Exception &)
    {
      /* expected */
    }
    
    // Test that negative layer index uses penultimate layer (should work)
    CAIF_Tensor negative_features=network.ExtractFeatures(input,-5);
    
    // Verify that negative index gives same result as default (both use penultimate layer)
    if(negative_features.Shape()!=features.Shape())
    {
      std::cout<<"FAILED - Negative layer index should give same shape as default penultimate layer\n";
      return false;
    }
    
    // Verify that features contain actual values (not all zeros)
    auto feature_data_result=features.ConstData<float>();
    if(feature_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not access feature data\n";
      return false;
    }
    
    const float *feature_data=feature_data_result;
    bool has_non_zero=false;
    for(uint32_t i=0;i<features.NumElements();++i)
    {
      if(std::abs(feature_data[i])>1e-6f)
      {
        has_non_zero=true;
        break;
      }
    }
    
    if(has_non_zero==false)
    {
      std::cout<<"FAILED - All feature values are zero (unexpected)\n";
      return false;
    }
    
    instance::CAIF_Base::SOut()<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    std::cout<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test to reproduce and verify the DEER program bug with dense layer shape handling
 */
bool TestDEERBugRepro()
{
  std::cout<<"Testing DEER Bug Reproduction... \n";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    std::cout<<"  Stage 1: Testing invalid input shape rejection...\n";
    // First ensure valid shape is used for subsequent steps
    // Then explicitly test invalid shape rejection separately
    try
    {
      network.SetInputShape({10});
      std::cout<<"    FAILED - Network incorrectly "
               <<"accepted input shape without "
               <<"batch dimension\n";
      return false;
    }
    catch(const instance::CAIF_Exception &)
    {
      /* expected */
    }
    std::cout<<"    PASSED - Network correctly rejected input shape without batch dimension\n";
    
    std::cout<<"  Stage 2: Testing valid input shape acceptance...\n";
    
    // Now set correct input shape with batch dimension
    network.SetInputShape({1,10});  // [batch_size=1, features=10]
    std::cout<<"    PASSED - Network accepted correct input shape with batch dimension\n";
    
    std::cout<<"  Stage 3: Testing network compilation...\n";
    
    // Add a dense layer
    network.AddDenseLayer(5,CAIF_ActivationType_e::ReLU);
    
    // Compile network
    network.Compile(CAIF_OptimizerType_e::SGD,CAIF_LossType_e::MeanSquaredError,0.01f);
    std::cout<<"    PASSED - Network compiled successfully\n";
    
    std::cout<<"  Stage 4: Testing forward pass with valid input...\n";
    
    // Create input tensor with correct shape
    CAIF_Tensor input=CreateTensor({1,10},{1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f});
    
    // Forward pass should work
    network.Forward(input,false);
    std::cout<<"    PASSED - Forward pass succeeded with valid input shape\n";
    
    std::cout<<"  Stage 5: Testing forward pass with invalid input...\n";
    
    // Create input tensor with wrong shape (missing batch dimension)
    CAIF_Tensor wrong_input=CreateTensor({10},{1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f});
    
    // Forward pass should fail
    try
    {
      network.Forward(wrong_input,false);
      std::cout<<"    FAILED - Forward pass incorrectly "
               <<"succeeded with invalid input "
               <<"shape\n";
      return false;
    }
    catch(const instance::CAIF_Exception &)
    {
      /* expected */
    }
    std::cout<<"    PASSED - Forward pass correctly rejected invalid input shape\n";
    
    std::cout<<"All stages PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    std::cout<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test neural network training with binary classification using BCE-with-logits
 */
bool TestNeuralNetworkTrainingBinaryLogits()
{
  std::cout<<"Testing Neural Network Training - Binary (BCE-with-logits)... ";
  try
  {
    CAIF_NeuralNetwork network;

    network.SetInputShape({1, 2});

    std::cout<<"  [BCELogits] Adding layer...\n";
    network.AddDenseLayer(1,CAIF_ActivationType_e::Linear);
    std::cout<<"  [BCELogits] Compiling...\n";
    network.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::BinaryCrossEntropyWithLogits,0.01f);

    // Generate larger separable dataset
    const uint32_t n_per_class=256;
    const uint32_t total=n_per_class*2;
    CAIF_Tensor input_data(network.Framework(),{total,2},CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Tensor target_data(network.Framework(),{total,1},CAIF_DataType::CAIF_DataType_e::Float32);
    auto ip=input_data.MutableData<float>();
    auto tp=target_data.MutableData<float>();
    if(ip==nullptr||tp==nullptr)
    {
      std::cout<<"FAILED - data pointers\n";
      return false;
    }
    float *x=ip;
    float *y=tp;

    uint32_t seed=1234567;
    for(uint32_t i=0;i<total;++i)
    {
      seed=(1103515245*seed+12345)&0x7fffffff;
      const float r1=static_cast<float>(seed)/static_cast<float>(0x7fffffff);
      seed=(1103515245*seed+12345)&0x7fffffff;
      const float r2=static_cast<float>(seed)/static_cast<float>(0x7fffffff);
      if((i%2U)==0U)
      {
        x[i*2+0]=0.15f+0.15f*r1;
        x[i*2+1]=0.15f+0.15f*r2;
        y[i]=1.0f;
      }
      else
      {
        x[i*2+0]=0.85f+0.15f*r1;
        x[i*2+1]=0.85f+0.15f*r2;
        y[i]=0.0f;
      }
    }

    CAIF_NeuralNetwork::TrainingConfig_t config;
    config.epochs=60;
    config.batch_size=64;
    config.learning_rate=0.001f;
    config.optimizer_type=CAIF_OptimizerType_e::Adam;
    config.loss_type=CAIF_LossType_e::BinaryCrossEntropyWithLogits;
    config.shuffle_data=true;
    config.use_validation=false;
    config.validation_split=0.0f;

    std::cout<<"  [Regression] Training...\n";
    const auto hist=network.Train(input_data,target_data,config);
    std::cout<<"  [Regression] Training done.\n";
    if(hist.empty()==true)
    {
      std::cout<<"FAILED - Empty training history\n";
      return false;
    }
    // Prefer accuracy as the acceptance, but also check for some loss improvement
    float first_loss=hist.front().loss;
    float min_loss=first_loss;
    for(size_t i=0;i<hist.size();++i)
    {
      if(hist[i].loss<min_loss)
      {
        min_loss=hist[i].loss;
      }
    }

    // Evaluate prediction accuracy on training set
    auto pred_tensor=network.Predict(input_data);
    auto pd=pred_tensor.ConstData<float>();
    if(pd==nullptr)
    {
      std::cout<<"FAILED - prediction data\n";
      return false;
    }
    const float *p=pd;
    uint32_t correct=0;
    for(uint32_t i=0;i<total;++i)
    {
      const float logit=p[i];
      const float prob=1.0f/(1.0f+std::exp(-logit));
      uint32_t label;
      if(i<n_per_class)
      {
        label=1U;
      }
      else
      {
        label=0U;
      }
      uint32_t pred;
      if(prob>0.5f)
      {
        pred=1U;
      }
      else
      {
        pred=0U;
      }
      if(pred==label)
      {
        ++correct;
      }
    }
    const float acc=static_cast<float>(correct)/static_cast<float>(total);
    if(acc<0.95f && !(min_loss<first_loss*0.99f))
    {
      std::cout<<"FAILED - Accuracy too low: "
               <<std::setprecision(3)
               <<acc
               <<" and insufficient loss improvement (first="
               <<first_loss
               <<", min="
               <<min_loss
               <<")\n";
      return false;
    }

    std::cout<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    std::cout<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test to assert bias init improves initial logits sign for positives vs negatives.
 */
bool TestBiasInitializationForBCELogits()
{
  std::cout<<"Testing BCE-with-logits bias initialization... ";
  try
  {
    CAIF_NeuralNetwork net;
    net.SetInputShape({1,2});
    net.AddDenseLayer(1,CAIF_ActivationType_e::Linear);
    net.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::BinaryCrossEntropyWithLogits,0.01f);
    // Imbalanced labels: 75% positives
    const uint32_t n=100;
    CAIF_Tensor X(net.Framework(),{n,2},CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Tensor Y(net.Framework(),{n,1},CAIF_DataType::CAIF_DataType_e::Float32);
    auto xp=X.MutableData<float>();
    auto yp=Y.MutableData<float>();
    if(xp==nullptr||yp==nullptr)
    {
      std::cout<<"FAILED - data ptr\n";
      return false;
    }
    float *xd=xp; float *yd=yp;
    for(uint32_t i=0;i<n;++i)
    {
      xd[i*2+0]=0.0f;
      xd[i*2+1]=0.0f;
      if(i<75)
      {
        yd[i]=1.0f;
      }
      else
      {
        yd[i]=0.0f;
      }
    }
    // One training step (epoch=1) should set bias ~ logit(0.75) ~ +1.098
    CAIF_NeuralNetwork::TrainingConfig_t cfg;
    cfg.epochs=1;
    cfg.batch_size=100;
    cfg.learning_rate=0.0f; // no weight updates
    cfg.optimizer_type=CAIF_OptimizerType_e::Adam;
    cfg.loss_type=CAIF_LossType_e::BinaryCrossEntropyWithLogits;
    cfg.shuffle_data=false;
    cfg.use_validation=false;
    cfg.validation_split=0.0f;
    net.Train(X,Y,cfg);
    // Expected bias = logit(0.75)
    const float expected_bias=std::log(0.75f/(1.0f-0.75f));
    // Inspect last dense layer bias directly
    const uint32_t last_index=net.LayerCount()-1;
    const CAIF_Layer &last_layer=net.Layer(last_index);
    const std::vector<CAIF_Tensor> params=last_layer.Parameters();
    if(params.size()<2)
    {
      std::cout<<"FAILED - last layer has no bias parameter\n";
      return false;
    }
    auto bptr=params[1].ConstData<float>();
    if(bptr==nullptr || params[1].NumElements()==0)
    {
      std::cout<<"FAILED - could not read bias parameter\n";
      return false;
    }
    const float actual_bias=bptr[0];
    if(std::abs(actual_bias-expected_bias)>0.05f)
    {
      std::cout<<"FAILED - bias mismatch expected="<<expected_bias
               <<" actual="<<actual_bias<<"\n";
      return false;
    }
    // With zero inputs, predictions should equal bias
    auto pred=net.Predict(X);
    auto pd=pred.ConstData<float>();
    if(pd==nullptr)
    {
      std::cout<<"FAILED - predict\n";
      return false;
    }
    bool all_match=true;
    for(uint32_t i=0;i<n;++i)
    {
      const float z=pd[i];
      if(std::abs(z-actual_bias)>1e-4f)
      {
        all_match=false;
        break;
      }
    }
    if(all_match==false)
    {
      std::cout<<"FAILED - predictions not equal to bias for zero inputs\n";
      return false;
    }
    std::cout<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    std::cout<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Reproduce DEER-like training on image-shaped tensors to catch crashes
 */
bool TestImageShapedTraining()
{
  std::cout<<"Testing Image-Shaped Training (DEER-like)... ";
  try
  {
    CAIF_NeuralNetwork net;
    // Input shape [batch,height,width,channels]
    net.SetInputShape({1,360,640,3});
    // Small conv stack to keep runtime reasonable
    std::cout<<"  [ImageTrain] Add conv1...\n";
    net.AddConvolution2DLayer(8,3,1,1,CAIF_ActivationType_e::LeakyReLU);
    std::cout<<"  [ImageTrain] Add maxpool1...\n";
    net.AddMaxPooling2DLayer(2,2);
    std::cout<<"  [ImageTrain] Add conv2...\n";
    net.AddConvolution2DLayer(16,3,1,1,CAIF_ActivationType_e::LeakyReLU);
    std::cout<<"  [ImageTrain] Add maxpool2...\n";
    net.AddMaxPooling2DLayer(2,2);
    std::cout<<"  [ImageTrain] Add flatten...\n";
    net.AddFlattenLayer();
    std::cout<<"  [ImageTrain] Add dense1...\n";
    net.AddDenseLayer(16,CAIF_ActivationType_e::LeakyReLU);
    std::cout<<"  [ImageTrain] Add out...\n";
    net.AddDenseLayer(1,CAIF_ActivationType_e::Linear);
    std::cout<<"  [ImageTrain] Compile...\n";
    net.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::BinaryCrossEntropyWithLogits,0.001f);
    // Build batch tensors N=12
    const uint32_t n=12;
    CAIF_Tensor X(net.Framework(),{n,360,640,3},CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Tensor Y(net.Framework(),{n,1},CAIF_DataType::CAIF_DataType_e::Float32);
    auto xp2=X.MutableData<float>();
    auto yp2=Y.MutableData<float>();
    if(xp2==nullptr||yp2==nullptr)
    {
      std::cout<<"FAILED - data ptr\n";
      return false;
    }
    float *xd=xp2;
    float *yd=yp2;
    const size_t elems=X.NumElements();
    for(size_t i=0;i<elems;++i)
    {
      xd[i]=static_cast<float>((i%1024))/1024.0f;
    }
    for(uint32_t i=0;i<n;++i)
    {
      if((i%2U)==0U)
      {
        yd[i]=1.0f;
      }
      else
      {
        yd[i]=0.0f;
      }
    }
    CAIF_NeuralNetwork::TrainingConfig_t cfg;
    cfg.epochs=1;
    cfg.batch_size=n;
    cfg.learning_rate=0.001f;
    cfg.optimizer_type=CAIF_OptimizerType_e::Adam;
    cfg.loss_type=CAIF_LossType_e::BinaryCrossEntropyWithLogits;
    cfg.shuffle_data=false;
    cfg.use_validation=false;
    cfg.validation_split=0.0f;
    std::cout<<"  [ImageTrain] Train...\n";
    net.Train(X,Y,cfg);
    std::cout<<"  [ImageTrain] Train done.\n";
    std::cout<<"PASSED\n";
    return true;
  }
  catch(const instance::CAIF_Exception &e)
  {
    std::cout<<"FAILED - CAIF Exception: "<<e<<"\n";
    return false;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Main test function
 */
int main(int argc,char *argv[])
{
  ConfigureBackendFromArgs(argc,argv);
  std::cout<<"=== CAIF Neural Network Training Test Suite ===\n\n";
  
  bool all_tests_passed=true;
  
  all_tests_passed&=(TestNeuralNetworkCompilation()==true);
  all_tests_passed&=(TestNeuralNetworkForward()==true);
  all_tests_passed&=(TestNeuralNetworkPrediction()==true);
  all_tests_passed&=(TestNeuralNetworkTrainingRegression()==true);
  all_tests_passed&=(TestNeuralNetworkTrainingClassification()==true);
  all_tests_passed&=(TestNeuralNetworkErrorHandling()==true);
  all_tests_passed&=(TestNeuralNetworkMultipleLayers()==true);
  all_tests_passed&=(TestNeuralNetworkFeatureExtraction()==true);
  all_tests_passed&=(TestDEERBugRepro()==true);
  all_tests_passed&=(TestNeuralNetworkTrainingBinaryLogits()==true);
  all_tests_passed&=(TestBiasInitializationForBCELogits()==true);
  all_tests_passed&=(TestImageShapedTraining()==true);
  
  std::cout<<"\n=== Test Summary ===\n";
  if(all_tests_passed==true)
  {
    std::cout<<"All tests PASSED!\n";
    return 0;
  }
  else
  {
    std::cout<<"Some tests FAILED!\n";
    return 1;
  }
} 