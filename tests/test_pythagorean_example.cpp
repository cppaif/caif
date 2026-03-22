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
 * @file test_pythagorean_example.cpp
 * @brief Educational neural network example using Pythagorean theorem
 * 
 * This example demonstrates how data flows through neural network layers
 * by training a network to learn the Pythagorean theorem: a² + b² = c²
 * 
 * The network learns to predict the hypotenuse (c) given two sides (a, b)
 * We'll trace the data through each layer to see how it transforms
 * 
 * @author CAIF Development Team
 * @version 1.0 
 * @date 2024
 */

#include "caif_neural_network.h"
#include "caif_tensor.h"
#include "caif_framework.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// Global framework for tests
static instance::CAIF_Framework g_test_framework;

// Helper function to print tensor contents
void PrintTensor(const instance::CAIF_Tensor &tensor, const std::string &name, const std::string &description)
{
  std::cout<<"  "<<name<<" "<<description<<":\n";
  
  auto shape=tensor.Shape();
  std::cout<<"    Shape: [";
  for(size_t i=0;i<shape.size();++i)
  {
    std::cout<<shape[i];
    if(i<shape.size()-1)std::cout<<", ";
  }
  std::cout<<"]\n";
  
  // Get tensor data for printing (first few values)
  auto data_result=tensor.ConstData<float>();
  if(data_result!=nullptr)
  {
    const float *data=data_result;
    uint32_t total_elements=1;
    for(uint32_t dim:shape)
    {
      total_elements*=dim;
    }
    
    std::cout<<"    Values: [";
    uint32_t print_count=std::min(total_elements,8u);  // Print first 8 values max
    for(uint32_t i=0;i<print_count;++i)
    {
      std::cout<<std::fixed<<std::setprecision(4)<<data[i];
      if(i<print_count-1)std::cout<<", ";
    }
    if(total_elements>8)
    {
      std::cout<<", ...";
    }
    std::cout<<"]\n";
  }
  std::cout<<"\n";
}

// Generate Pythagorean theorem training data
std::pair<instance::CAIF_Tensor,instance::CAIF_Tensor> GeneratePythagoreanData(uint32_t samples)
{
  std::cout<<"Generating Pythagorean theorem training data...\n";
  std::cout<<"Training the network to learn: c = sqrt(a² + b²)\n\n";
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(1.0f,10.0f);
  
  // Create input tensor (samples, 2) for (a, b) values
  instance::CAIF_Tensor inputs(g_test_framework,{samples,2},instance::CAIF_DataType::CAIF_DataType_e::Float32);
  
  // Create output tensor (samples, 1) for c values  
  instance::CAIF_Tensor targets(g_test_framework,{samples,1},instance::CAIF_DataType::CAIF_DataType_e::Float32);
  
  auto input_data_result=inputs.MutableData<float>();
  auto target_data_result=targets.MutableData<float>();
  
  if(input_data_result==nullptr||target_data_result==nullptr)
  {
    throw std::runtime_error("Failed to get tensor data for training set generation");
  }
  
  float *input_data=input_data_result;
  float *target_data=target_data_result;
  
  std::cout<<"Sample training data:\n";
  std::cout<<"  a     |   b     |   c (target)\n";
  std::cout<<"--------|---------|-------------\n";
  
  for(uint32_t i=0;i<samples;++i)
  {
    float a=dist(gen);
    float b=dist(gen);
    float c=std::sqrt(a*a+b*b);
    
    input_data[i*2+0]=a;     // First feature: a
    input_data[i*2+1]=b;     // Second feature: b
    target_data[i]=c;        // Target: c
    
    if(i<5)  // Print first 5 samples
    {
      std::cout<<"  "<<std::fixed<<std::setprecision(3)<<a
               <<"  |  "<<std::setprecision(3)<<b
               <<"  |  "<<std::setprecision(3)<<c<<"\n";
    }
  }
  
  if(samples>5)
  {
    std::cout<<"  ...   |  ...   |  ...\n";
  }
  
  std::cout<<"\n";
  return {inputs,targets};
}

// Custom prediction function that shows layer-by-layer processing
void DetailedPrediction(instance::CAIF_NeuralNetwork &network, const instance::CAIF_Tensor &input)
{
  std::cout<<"=== DETAILED FORWARD PASS ANALYSIS ===\n\n";
  
  std::cout<<"INPUT LAYER:\n";
  PrintTensor(input,"Input","(a, b values)");
  
  std::cout<<"The input represents two sides of a right triangle.\n";
  std::cout<<"The network must learn to compute the hypotenuse: c = sqrt(a² + b²)\n\n";
  
  // For now, we'll use the standard prediction and explain what should happen
  // In a more advanced implementation, we'd intercept each layer's output
  auto prediction=network.Predict(input);
  
  std::cout<<"LAYER-BY-LAYER PROCESSING EXPLANATION:\n\n";
  
  std::cout<<"DENSE LAYER 1 (Input -> Hidden1):\n";
  std::cout<<"  Purpose: Transform 2 input features into 8 hidden features\n";
  std::cout<<"  Operation: output = ReLU(input * weights + bias)\n";
  std::cout<<"  - Matrix multiplication: [1,2] * [2,8] = [1,8]\n";
  std::cout<<"  - Bias addition: [1,8] + [8] = [1,8]\n";
  std::cout<<"  - ReLU activation: max(0, x) for each element\n";
  std::cout<<"  Result: 8 features that capture non-linear combinations of a and b\n\n";
  
  std::cout<<"DROPOUT LAYER:\n";
  std::cout<<"  Purpose: Regularization during training (disabled during inference)\n";
  std::cout<<"  Operation: During training, randomly set 30% of values to 0\n";
  std::cout<<"  Result: Same shape [1,8], but some values may be zeroed\n\n";
  
  std::cout<<"DENSE LAYER 2 (Hidden1 -> Hidden2):\n";
  std::cout<<"  Purpose: Further process the 8 features into 4 features\n";
  std::cout<<"  Operation: output = ReLU(input * weights + bias)\n";
  std::cout<<"  - Matrix multiplication: [1,8] * [8,4] = [1,4]\n";
  std::cout<<"  - Bias addition: [1,4] + [4] = [1,4]\n";
  std::cout<<"  - ReLU activation: max(0, x) for each element\n";
  std::cout<<"  Result: 4 refined features\n\n";
  
  std::cout<<"OUTPUT LAYER (Hidden2 -> Output):\n";
  std::cout<<"  Purpose: Combine 4 features into final prediction\n";
  std::cout<<"  Operation: output = Linear(input * weights + bias)\n";
  std::cout<<"  - Matrix multiplication: [1,4] * [4,1] = [1,1]\n";
  std::cout<<"  - Bias addition: [1,1] + [1] = [1,1]\n";
  std::cout<<"  - No activation (linear output for regression)\n";
  std::cout<<"  Result: Single value representing predicted hypotenuse\n\n";
  
  std::cout<<"FINAL OUTPUT:\n";
  PrintTensor(prediction,"Output","(predicted c value)");
  
  // Calculate expected value for comparison
  auto input_data_result2=input.ConstData<float>();
  auto prediction_data_result=prediction.ConstData<float>();
  
  if(input_data_result2!=nullptr&&prediction_data_result!=nullptr)
  {
    const float *input_data=input_data_result2;
    const float *prediction_data=prediction_data_result;
    
    float a=input_data[0];
    float b=input_data[1];
    float expected_c=std::sqrt(a*a+b*b);
    float predicted_c=prediction_data[0];
    float error=std::abs(expected_c-predicted_c);
    
    std::cout<<"ANALYSIS:\n";
    std::cout<<"  Input a: "<<std::fixed<<std::setprecision(4)<<a<<"\n";
    std::cout<<"  Input b: "<<std::setprecision(4)<<b<<"\n";
    std::cout<<"  Expected c: "<<std::setprecision(4)<<expected_c<<"\n";
    std::cout<<"  Predicted c: "<<std::setprecision(4)<<predicted_c<<"\n";
    std::cout<<"  Error: "<<std::setprecision(4)<<error<<"\n";
    std::cout<<"  Error %: "<<std::setprecision(2)<<(error/expected_c*100.0f)<<"%\n\n";
  }
}

int main()
{
  std::cout<<"==================================================\n";
  std::cout<<"    NEURAL NETWORK LAYER-BY-LAYER EXAMPLE\n";
  std::cout<<"         Learning the Pythagorean Theorem\n";
  std::cout<<"==================================================\n\n";
  
  try
  {
    // Create neural network
    instance::CAIF_NeuralNetwork network;
    
    // Set input shape: batch_size=1, features=2 (a, b)
    std::vector<uint32_t> input_shape={1,2};
    network.SetInputShape(input_shape);
    
    std::cout<<"NETWORK ARCHITECTURE DESIGN:\n";
    std::cout<<"=============================\n\n";
    
    std::cout<<"Layer 1: Dense (2 -> 8) + ReLU\n";
    std::cout<<"  - Transforms 2 inputs (a, b) into 8 hidden features\n";
    std::cout<<"  - ReLU activation allows non-linear combinations\n";
    network.AddDenseLayer(8,instance::CAIF_ActivationType_e::ReLU);
    
    std::cout<<"Layer 2: Dropout (30%)\n";
    std::cout<<"  - Regularization to prevent overfitting\n";
    std::cout<<"  - Randomly zeros 30% of neurons during training\n";
    network.AddDropoutLayer(0.3f);
    
    std::cout<<"Layer 3: Dense (8 -> 4) + ReLU\n";
    std::cout<<"  - Compresses information into 4 key features\n";
    std::cout<<"  - Learns important combinations for Pythagorean calculation\n";
    network.AddDenseLayer(4,instance::CAIF_ActivationType_e::ReLU);
    
    std::cout<<"Layer 4: Dense (4 -> 1) + Linear\n";
    std::cout<<"  - Final output layer produces single prediction\n";
    std::cout<<"  - Linear activation for regression output\n";
    network.AddDenseLayer(1,instance::CAIF_ActivationType_e::Linear);
    
    std::cout<<"\n"<<network.ExportArchitecture()<<"\n";
    
    // Compile network
    std::cout<<"COMPILING NETWORK:\n";
    std::cout<<"==================\n";
    network.Compile(
      instance::CAIF_OptimizerType_e::Adam,
      instance::CAIF_LossType_e::MeanSquaredError,  // Correct loss for regression
      0.001f
    );
    std::cout<<"✅ Network compiled successfully!\n\n";
    
    // Generate training data
    std::cout<<"TRAINING DATA GENERATION:\n";
    std::cout<<"=========================\n";
    auto [train_inputs,train_targets]=GeneratePythagoreanData(100);
    
    // Show prediction BEFORE training
    std::cout<<"BEFORE TRAINING - RANDOM PREDICTIONS:\n";
    std::cout<<"=====================================\n";
    
    // Create a single test sample
    instance::CAIF_Tensor test_input(network.Framework(),{1,2},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    auto test_data_result=test_input.MutableData<float>();
    if(test_data_result!=nullptr)
    {
      float *test_data=test_data_result;
      test_data[0]=3.0f;  // a = 3
      test_data[1]=4.0f;  // b = 4
      // Expected c = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
    }
    
    DetailedPrediction(network,test_input);
    
    // Train the network
    std::cout<<"TRAINING PROCESS:\n";
    std::cout<<"=================\n";
    std::cout<<"Training network to learn Pythagorean theorem...\n";
    
    instance::CAIF_NeuralNetwork::TrainingConfig_t config{};
    config.epochs=50;
    config.batch_size=10;
    config.learning_rate=0.001f;
    config.optimizer_type=instance::CAIF_OptimizerType_e::Adam;
    config.loss_type=instance::CAIF_LossType_e::MeanSquaredError;
    config.shuffle_data=true;
    config.use_validation=false;
    config.validation_split=0.0f;
    
    auto training_history=network.Train(train_inputs,train_targets,config);
    std::cout<<"✅ Training completed for "<<training_history.size()<<" epochs\n\n";
    
    // Show prediction AFTER training
    std::cout<<"AFTER TRAINING - LEARNED PREDICTIONS:\n";
    std::cout<<"=====================================\n";
    
    DetailedPrediction(network,test_input);
    
    // Test with multiple examples
    std::cout<<"TESTING WITH MULTIPLE EXAMPLES:\n";
    std::cout<<"===============================\n";
    
    std::vector<std::pair<float,float>> test_cases={{3.0f,4.0f},{5.0f,12.0f},{1.0f,1.0f},{6.0f,8.0f}};
    
    for(const auto &test_case:test_cases)
    {
      instance::CAIF_Tensor single_test(network.Framework(),{1,2},instance::CAIF_DataType::CAIF_DataType_e::Float32);
      auto single_test_data_result=single_test.MutableData<float>();
      if(single_test_data_result!=nullptr)
      {
        float *single_test_data=single_test_data_result;
        single_test_data[0]=test_case.first;
        single_test_data[1]=test_case.second;
        
        auto pred_tensor=network.Predict(single_test);
        auto pred_data_result=pred_tensor.ConstData<float>();
        if(pred_data_result!=nullptr)
        {
          float a=test_case.first;
          float b=test_case.second;
          float expected=std::sqrt(a*a+b*b);
          float predicted=pred_data_result[0];
          float error=std::abs(expected-predicted);
          
          std::cout<<"  Test: a="<<std::fixed<<std::setprecision(1)<<a
                   <<", b="<<b
                   <<" → Expected="<<std::setprecision(3)<<expected
                   <<", Predicted="<<predicted
                   <<", Error="<<error<<"\n";
        }
      }
    }
    
    std::cout<<"\n✅ Educational example completed successfully!\n";
    std::cout<<"✅ You can now see how data flows through neural network layers!\n";
    
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception occurred: "<<e.what()<<"\n";
    return 1;
  }
  catch(...)
  {
    std::cout<<"Unknown exception occurred\n";
    return 1;
  }
} 