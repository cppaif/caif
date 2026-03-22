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
 * @file test_gradient_tensor_fix.cpp
 * @brief Test to verify the gradient tensor size bug fix
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 * 
 * This test verifies that the gradient tensor size bug is fixed:
 * - Convolution layers properly store _last_output during training=true forward pass
 * - Backward pass doesn't fail with "Gradient shape mismatch" errors
 * - Neural network training can proceed through CNN layers without shape issues
 */

#include "caif_neural_network.h"
#include "caif_tensor.h"
#include "caif_convolution2d_layer.h"
#include "caif_max_pooling2d_layer.h"
#include "caif_flatten_layer.h"
#include "caif_dense_layer.h"
#include "caif_framework.h"
#include <iostream>
#include <memory>

using namespace instance;
using CAIF_DataType_e=CAIF_DataType::CAIF_DataType_e;

/**
 * @brief Helper function to create test tensor with specified shape and values
 */
CAIF_Tensor CreateTensor(CAIF_Framework &framework,
                        const std::vector<uint32_t> &shape,
                        const std::vector<float> &data)
{
  CAIF_Tensor tensor(framework,shape,CAIF_DataType_e::Float32);
  auto tensor_data_result=tensor.MutableData<float>();
  if(tensor_data_result!=nullptr)
  {
    float *tensor_data=tensor_data_result;
    for(size_t i=0;i<data.size()&&i<tensor.NumElements();++i)
    {
      tensor_data[i]=data[i];
    }
  }
  return tensor;
}

/**
 * @brief Test that convolution layer properly stores _last_output during forward pass
 */
bool TestConvolutionLayerOutputStorage()
{
  std::cout<<"Testing Convolution Layer Output Storage... ";
  
  try
  {
    CAIF_Framework framework;
    
    // Create a convolution layer
    CAIF_Convolution2DLayer conv_layer(framework,32,3,1,1,CAIF_ActivationType_e::ReLU,true);
    
    // Initialize with 4D input shape [batch, height, width, channels]
    std::vector<uint32_t> input_shape={1,8,8,3};
    conv_layer.Initialize(input_shape,42);
    
    // Create test input tensor [1, 8, 8, 3]
    std::vector<float> input_data(1*8*8*3,0.5f);
    CAIF_Tensor input_tensor=CreateTensor(framework,input_shape,input_data);
    
    // Forward pass with training=true (should store _last_output)
    CAIF_Tensor output=conv_layer.Forward(input_tensor,true);
    const auto &output_shape=output.Shape();
    
    // Verify output has correct 4D shape
    if(output_shape.size()!=4)
    {
      std::cout<<"FAILED - Output should be 4D, got "<<output_shape.size()<<"D\n";
      return false;
    }
    
    // Create gradient tensor with same shape as output for backward pass
    std::vector<float> grad_data(output.NumElements(),0.1f);
    CAIF_Tensor gradient=CreateTensor(framework,output_shape,grad_data);
    
    // Backward pass should NOT fail with shape mismatch
    CAIF_Tensor input_gradient=conv_layer.Backward(gradient);
    if(input_gradient.Shape()!=input_shape)
    {
      std::cout<<"FAILED - Input gradient shape mismatch\n";
      return false;
    }
    
    std::cout<<"PASSED\n";
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - Exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test CNN training without gradient shape mismatch errors
 */
bool TestCNNTrainingGradientFlow()
{
  std::cout<<"Testing CNN Training Gradient Flow... ";
  
  try
  {
    CAIF_NeuralNetwork network;
    
    // Set input shape
    network.SetInputShape({1,8,8,3});
    
    // Add CNN layers
    network.AddConvolution2DLayer(16,3,1,1,CAIF_ActivationType_e::ReLU);
    
    network.AddMaxPooling2DLayer(2,2);
    
    network.AddFlattenLayer();
    
    network.AddDenseLayer(2,CAIF_ActivationType_e::Softmax);
    
    // Compile network
    network.Compile(CAIF_OptimizerType_e::SGD,CAIF_LossType_e::CategoricalCrossEntropy,0.01f);
    
    // Create training data
    std::vector<float> input_data(1*8*8*3,0.5f);
    CAIF_Tensor training_input=CreateTensor(network.Framework(),{1,8,8,3},input_data);
    
    // Create target data (one-hot encoded)
    CAIF_Tensor training_target=CreateTensor(network.Framework(),{1,2},{1.0f,0.0f});
    
    // Training configuration for minimal test
    CAIF_NeuralNetwork::TrainingConfig_t config;
    config.epochs=1;
    config.batch_size=1;
    config.learning_rate=0.01f;
    config.optimizer_type=CAIF_OptimizerType_e::SGD;
    config.loss_type=CAIF_LossType_e::CategoricalCrossEntropy;
    config.shuffle_data=false;
    config.use_validation=false;
    config.validation_split=0.0f;
    
    // This should NOT fail with gradient shape mismatch errors
    network.Train(training_input,training_target,config);
    
    std::cout<<"PASSED\n";
    return true;
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
int main()
{
  std::cout<<"=== Gradient Tensor Size Bug Fix Tests ===\n";
  
  bool all_passed=true;
  
  all_passed&=TestConvolutionLayerOutputStorage();
  all_passed&=TestCNNTrainingGradientFlow();
  
  std::cout<<"\n=== Test Summary ===\n";
  if(all_passed==true)
  {
    std::cout<<"✅ ALL TESTS PASSED\n";
    return 0;
  }
  else
  {
    std::cout<<"❌ SOME TESTS FAILED\n";
    return 1;
  }
}
