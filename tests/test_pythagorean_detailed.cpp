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
 * @file test_pythagorean_detailed.cpp
 * @brief Advanced neural network example showing actual layer-by-layer tensor values
 * 
 * This example demonstrates real data flow through neural network layers
 * by intercepting and displaying actual tensor values at each step
 * while learning the Pythagorean theorem: a² + b² = c²
 * 
 * @author CAIF Development Team
 * @version 1.0 
 * @date 2024
 */

#include "caif_neural_network.h"
#include "caif_tensor.h"
#include "caif_dense_layer.h"
#include "caif_dropout_layer.h"
#include "caif_framework.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <memory>

// Enhanced function to print tensor contents with more detail
void PrintDetailedTensor(const instance::CAIF_Tensor &tensor,
                         const std::string &name,
                         const std::string &description)
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
  
  // Calculate total elements
  uint32_t total_elements=1;
  for(uint32_t dim:shape)
  {
    total_elements*=dim;
  }
  std::cout<<"    Total elements: "<<total_elements<<"\n";
  
  // Get tensor data for printing
  auto data_result=tensor.ConstData<float>();
  if(data_result!=nullptr)
  {
    const float *data=data_result;
    
    std::cout<<"    Values: [";
    // For small tensors, print all values
    if(total_elements<=16)
    {
      for(uint32_t i=0;i<total_elements;++i)
      {
        std::cout<<std::fixed<<std::setprecision(4)<<data[i];
        if(i<total_elements-1)std::cout<<", ";
      }
    }
    else
    {
      // For larger tensors, print first 8 and statistics
      for(uint32_t i=0;i<8;++i)
      {
        std::cout<<std::fixed<<std::setprecision(4)<<data[i];
        if(i<7)std::cout<<", ";
      }
      std::cout<<", ...";
      
      // Calculate basic statistics
      float min_val=data[0], max_val=data[0], sum=0.0f;
      for(uint32_t i=0;i<total_elements;++i)
      {
        min_val=std::min(min_val,data[i]);
        max_val=std::max(max_val,data[i]);
        sum+=data[i];
      }
      float mean=sum/total_elements;
      
      std::cout<<"]\n";
      std::cout<<"    Statistics: min="<<std::setprecision(4)<<min_val
               <<", max="<<max_val
               <<", mean="<<mean;
    }
    std::cout<<"]\n";
  }
  std::cout<<"\n";
}

// Global framework for tests
static instance::CAIF_Framework g_test_framework;

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

// Manual layer-by-layer forward pass showing actual intermediate values
void DetailedLayerByLayerPrediction(const instance::CAIF_Tensor &input)
{
  std::cout<<"=== ACTUAL LAYER-BY-LAYER TENSOR VALUES ===\n\n";
  
  std::cout<<"STEP 0 - INPUT LAYER:\n";
  PrintDetailedTensor(input,"Input","(a, b values)");
  
  std::cout<<"The input represents two sides of a right triangle.\n";
  std::cout<<"Let's trace the actual data as it flows through each layer...\n\n";
  
  try
  {
    // Create individual layers for manual processing
    instance::CAIF_DenseLayer layer1(g_test_framework,8,instance::CAIF_ActivationType_e::ReLU);
    instance::CAIF_DropoutLayer dropout_layer(g_test_framework,0.3f);
    instance::CAIF_DenseLayer layer2(g_test_framework,4,instance::CAIF_ActivationType_e::ReLU);
    instance::CAIF_DenseLayer output_layer(g_test_framework,1,instance::CAIF_ActivationType_e::ReLU);
    
    // Initialize layers (normally done by the network)
    std::vector<uint32_t> input_shape={1,2};
    
    layer1.Initialize(input_shape,42);  // Fixed seed for reproducibility
    
    auto layer1_output_shape=layer1.CalculateOutputShape(input_shape);
    
    dropout_layer.Initialize(layer1_output_shape,43);
    
    layer2.Initialize(layer1_output_shape,44);
    
    auto layer2_output_shape=layer2.CalculateOutputShape(layer1_output_shape);
    
    output_layer.Initialize(layer2_output_shape,45);
    
    // STEP 1: First Dense Layer
    std::cout<<"STEP 1 - DENSE LAYER 1 (2 → 8 features):\n";
    std::cout<<"Operation: output = ReLU(input × weights + bias)\n";
    
    auto layer1_output=layer1.Forward(input,false);  // false = inference mode
    
    PrintDetailedTensor(layer1_output,"Layer1_Output","after ReLU(input×W1 + b1)");
    
    std::cout<<"Analysis:\n";
    std::cout<<"  - Input [1,2] was multiplied by weights [2,8] → [1,8]\n";
    std::cout<<"  - Bias [8] was added (broadcast to match [1,8])\n";
    std::cout<<"  - ReLU activation applied: max(0,x) to each element\n";
    std::cout<<"  - Result: 8 learned features representing combinations of a and b\n\n";
    
    // STEP 2: Dropout Layer
    std::cout<<"STEP 2 - DROPOUT LAYER (regularization):\n";
    std::cout<<"Operation: During inference, pass through unchanged\n";
    
    auto dropout_output=dropout_layer.Forward(layer1_output,false);  // false = inference mode
    
    PrintDetailedTensor(dropout_output,"Dropout_Output","(unchanged during inference)");
    
    std::cout<<"Analysis:\n";
    std::cout<<"  - During inference: dropout is disabled, data passes through unchanged\n";
    std::cout<<"  - During training: would randomly zero 30% of values for regularization\n";
    std::cout<<"  - Shape remains [1,8]\n\n";
    
    // STEP 3: Second Dense Layer
    std::cout<<"STEP 3 - DENSE LAYER 2 (8 → 4 features):\n";
    std::cout<<"Operation: output = ReLU(input × weights + bias)\n";
    
    auto layer2_output=layer2.Forward(dropout_output,false);
    
    PrintDetailedTensor(layer2_output,"Layer2_Output","after ReLU(hidden×W2 + b2)");
    
    std::cout<<"Analysis:\n";
    std::cout<<"  - Hidden features [1,8] multiplied by weights [8,4] → [1,4]\n";
    std::cout<<"  - Bias [4] added and ReLU applied\n";
    std::cout<<"  - Network is learning to compress 8 features into 4 key patterns\n";
    std::cout<<"  - These 4 features should capture essential info for Pythagorean calculation\n\n";
    
    // STEP 4: Output Layer
    std::cout<<"STEP 4 - OUTPUT LAYER (4 → 1 prediction):\n";
    std::cout<<"Operation: output = ReLU(input × weights + bias)\n";
    
    auto final_output=output_layer.Forward(layer2_output,false);
    
    PrintDetailedTensor(final_output,"Final_Output","(predicted hypotenuse)");
    
    // Calculate expected value for comparison
    auto input_data_result2=input.ConstData<float>();
    auto output_data_result=final_output.ConstData<float>();
    
    if(input_data_result2!=nullptr&&output_data_result!=nullptr)
    {
      const float *input_data=input_data_result2;
      const float *output_data=output_data_result;
      
      float a=input_data[0];
      float b=input_data[1];
      float expected_c=std::sqrt(a*a+b*b);
      float predicted_c=output_data[0];
      float error=std::abs(expected_c-predicted_c);
      
      std::cout<<"FINAL ANALYSIS:\n";
      std::cout<<"  Mathematical expectation: c = sqrt("
               <<std::fixed<<std::setprecision(1)
               <<a<<"² + "<<b
               <<"²) = sqrt("<<(a*a)
               <<" + "<<(b*b)
               <<") = sqrt("<<(a*a+b*b)
               <<") = "<<std::setprecision(4)
               <<expected_c<<"\n";
      std::cout<<"  Network prediction: "<<predicted_c<<"\n";
      std::cout<<"  Absolute error: "<<error<<"\n";
      std::cout<<"  Relative error: "<<std::setprecision(2)<<(error/expected_c*100.0f)<<"%\n\n";
      
      std::cout<<"Data Flow Summary:\n";
      std::cout<<"  [1,2] → Dense+ReLU → [1,8] → Dropout → [1,8] → Dense+ReLU → [1,4] → Dense+ReLU → [1,1]\n";
      std::cout<<"  Each layer learns increasingly abstract representations of the input\n";
      std::cout<<"  Random weights mean poor initial predictions, but this shows the data path!\n\n";
    }
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception during detailed prediction: "<<e.what()<<"\n";
  }
}

int main()
{
  std::cout<<"==================================================\n";
  std::cout<<"  NEURAL NETWORK DETAILED LAYER ANALYSIS\n";
  std::cout<<"    Real Tensor Values at Each Layer\n";
  std::cout<<"==================================================\n\n";
  
  try
  {
    std::cout<<"This example shows ACTUAL tensor values flowing through each layer\n";
    std::cout<<"We'll trace a single input through the network step-by-step\n\n";
    
    // Create a single test sample
    instance::CAIF_Tensor test_input(g_test_framework,{1,2},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    auto test_data_result=test_input.MutableData<float>();
    if(test_data_result!=nullptr)
    {
      float *test_data=test_data_result;
      test_data[0]=3.0f;  // a = 3
      test_data[1]=4.0f;  // b = 4
      // Expected c = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
    }
    
    std::cout<<"Test case: Classic 3-4-5 right triangle\n";
    std::cout<<"Input: a=3, b=4\n";
    std::cout<<"Expected output: c=5 (since 3² + 4² = 5²)\n\n";
    
    // Run detailed layer-by-layer analysis
    DetailedLayerByLayerPrediction(test_input);
    
    std::cout<<"KEY INSIGHTS:\n";
    std::cout<<"=============\n";
    std::cout<<"1. You can see the exact tensor values at each layer\n";
    std::cout<<"2. Matrix multiplication dimensions are clearly visible\n";
    std::cout<<"3. ReLU activation zeros out negative values\n";
    std::cout<<"4. Each layer transforms the representation\n";
    std::cout<<"5. Random weights give poor predictions initially\n";
    std::cout<<"6. Training would adjust these weights to minimize error\n\n";
    
    std::cout<<"✅ Layer-by-layer tensor analysis completed!\n";
    std::cout<<"✅ You can now see exactly how data transforms through the network!\n";
    
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