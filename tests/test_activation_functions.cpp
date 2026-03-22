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
 * @file test_activation_functions.cpp
 * @brief Comprehensive tests for activation functions in the AIF framework
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_tensor.h"
#include "caif_dense_layer.h"
#include "caif_neural_network.h"
#include "caif_optimizer.h"
#include "caif_loss_function.h"
#include "caif_constants.h"
#include "caif_framework.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

using namespace instance;
using CAIF_DataType_e=CAIF_DataType::CAIF_DataType_e;

/**
 * @brief Test Linear activation function
 * @return True if all tests pass, false otherwise
 */
bool TestLinearActivation()
{
  std::cout<<"Testing Linear activation function..."<<std::endl;
  
  try
  {
    CAIF_Framework framework;
    
    // Test 1: Basic Linear activation (identity function)
    std::vector<uint32_t> shape={2,3};
    CAIF_Tensor input(framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
    
    // Set test data
    auto input_data_result=input.MutableData<float>();
    if(input_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get input data"<<std::endl;
      return false;
    }
    float *input_data=input_data_result;
    input_data[0]=-2.0f;
    input_data[1]=-1.0f;
    input_data[2]=0.0f;
    input_data[3]=1.0f;
    input_data[4]=2.0f;
    input_data[5]=3.0f;
    
    // Apply Linear activation
    CAIF_Tensor output=input.Linear();
    
    // Verify output (should be identical to input)
    auto output_data_result=output.ConstData<float>();
    if(output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get output data"<<std::endl;
      return false;
    }
    const float *output_data=output_data_result;
    
    for(size_t i=0;i<6;++i)
    {
      if(std::abs(output_data[i]-input_data[i])>1e-6f)
      {
        std::cout<<"ERROR: Linear activation failed at index "<<i
                 <<" (expected "<<input_data[i]<<", got "<<output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    // Test 2: Linear derivative (should pass gradient unchanged)
    CAIF_Tensor gradient(framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
    auto grad_data_result=gradient.MutableData<float>();
    if(grad_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient data"<<std::endl;
      return false;
    }
    float *grad_data=grad_data_result;
    grad_data[0]=0.1f;
    grad_data[1]=0.2f;
    grad_data[2]=0.3f;
    grad_data[3]=0.4f;
    grad_data[4]=0.5f;
    grad_data[5]=0.6f;
    
    CAIF_Tensor grad_output=output.LinearDerivative(gradient);
    auto grad_output_data_result=grad_output.ConstData<float>();
    if(grad_output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient output data"<<std::endl;
      return false;
    }
    const float *grad_output_data=grad_output_data_result;
    
    for(size_t i=0;i<6;++i)
    {
      if(std::abs(grad_output_data[i]-grad_data[i])>1e-6f)
      {
        std::cout<<"ERROR: Linear derivative failed at index "<<i
                 <<" (expected "<<grad_data[i]<<", got "<<grad_output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    std::cout<<"✅ Linear activation tests passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"ERROR: Linear activation test failed with exception: "<<e.what()<<std::endl;
    return false;
  }
}

/**
 * @brief Test ReLU activation function
 * @return True if all tests pass, false otherwise
 */
bool TestReLUActivation()
{
  std::cout<<"Testing ReLU activation function..."<<std::endl;
  
  try
  {
    CAIF_Framework framework;
    
    // Test 1: Basic ReLU activation
    std::vector<uint32_t> shape={2,3};
    CAIF_Tensor input(framework,shape,CAIF_DataType_e::Float32);
    
    // Set test data
    auto input_data_result=input.MutableData<float>();
    if(input_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get input data"<<std::endl;
      return false;
    }
    float *input_data=input_data_result;
    input_data[0]=-2.0f;
    input_data[1]=-1.0f;
    input_data[2]=0.0f;
    input_data[3]=1.0f;
    input_data[4]=2.0f;
    input_data[5]=3.0f;
    
    // Apply ReLU activation
    CAIF_Tensor output=input.ReLU();
    
    // Verify output
    auto output_data_result=output.ConstData<float>();
    if(output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get output data"<<std::endl;
      return false;
    }
    const float *output_data=output_data_result;
    
    float expected[]={0.0f,0.0f,0.0f,1.0f,2.0f,3.0f};
    for(size_t i=0;i<6;++i)
    {
      if(std::abs(output_data[i]-expected[i])>1e-6f)
      {
        std::cout<<"ERROR: ReLU activation failed at index "<<i
                 <<" (expected "<<expected[i]<<", got "<<output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    // Test 2: ReLU derivative
    CAIF_Tensor gradient(framework,shape,CAIF_DataType_e::Float32);
    auto grad_data_result=gradient.MutableData<float>();
    if(grad_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient data"<<std::endl;
      return false;
    }
    float *grad_data=grad_data_result;
    for(size_t i=0;i<6;++i)
    {
      grad_data[i]=1.0f;  // Uniform gradient
    }
    
    CAIF_Tensor grad_output=input.ReLUDerivative(gradient);  // Use input for derivative
    auto grad_output_data_result=grad_output.ConstData<float>();
    if(grad_output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient output data"<<std::endl;
      return false;
    }
    const float *grad_output_data=grad_output_data_result;
    
    float expected_grad[]={0.0f,0.0f,0.0f,1.0f,1.0f,1.0f};
    for(size_t i=0;i<6;++i)
    {
      if(std::abs(grad_output_data[i]-expected_grad[i])>1e-6f)
      {
        std::cout<<"ERROR: ReLU derivative failed at index "<<i
                 <<" (expected "<<expected_grad[i]<<", got "<<grad_output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    std::cout<<"✅ ReLU activation tests passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"ERROR: ReLU activation test failed with exception: "<<e.what()<<std::endl;
    return false;
  }
}

/**
 * @brief Test Sigmoid activation function
 * @return True if all tests pass, false otherwise
 */
bool TestSigmoidActivation()
{
  std::cout<<"Testing Sigmoid activation function..."<<std::endl;
  
  try
  {
    CAIF_Framework framework;
    
    // Test 1: Basic Sigmoid activation
    std::vector<uint32_t> shape={1,4};
    CAIF_Tensor input(framework,shape,CAIF_DataType_e::Float32);
    
    // Set test data
    auto input_data_result=input.MutableData<float>();
    if(input_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get input data"<<std::endl;
      return false;
    }
    float *input_data=input_data_result;
    input_data[0]=-2.0f;
    input_data[1]=0.0f;
    input_data[2]=1.0f;
    input_data[3]=2.0f;
    
    // Apply Sigmoid activation
    CAIF_Tensor output=input.Sigmoid();
    
    // Verify output
    auto output_data_result=output.ConstData<float>();
    if(output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get output data"<<std::endl;
      return false;
    }
    const float *output_data=output_data_result;
    
    // Expected values: sigmoid(x) = 1/(1+exp(-x))
    float expected[]={
      1.0f/(1.0f+std::exp(2.0f)),   // sigmoid(-2)
      0.5f,                         // sigmoid(0)
      1.0f/(1.0f+std::exp(-1.0f)),  // sigmoid(1)
      1.0f/(1.0f+std::exp(-2.0f))   // sigmoid(2)
    };
    
    for(size_t i=0;i<4;++i)
    {
      if(std::abs(output_data[i]-expected[i])>1e-6f)
      {
        std::cout<<"ERROR: Sigmoid activation failed at index "<<i
                 <<" (expected "<<expected[i]<<", got "<<output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    // Test 2: Sigmoid derivative
    CAIF_Tensor gradient(framework,shape,CAIF_DataType_e::Float32);
    auto grad_data_result=gradient.MutableData<float>();
    if(grad_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient data"<<std::endl;
      return false;
    }
    float *grad_data=grad_data_result;
    for(size_t i=0;i<4;++i)
    {
      grad_data[i]=1.0f;  // Uniform gradient
    }
    
    CAIF_Tensor grad_output=output.SigmoidDerivative(gradient);  // Use output for derivative
    auto grad_output_data_result=grad_output.ConstData<float>();
    if(grad_output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient output data"<<std::endl;
      return false;
    }
    const float *grad_output_data=grad_output_data_result;
    
    // Expected derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    for(size_t i=0;i<4;++i)
    {
      float expected_grad=output_data[i]*(1.0f-output_data[i]);
      if(std::abs(grad_output_data[i]-expected_grad)>1e-6f)
      {
        std::cout<<"ERROR: Sigmoid derivative failed at index "<<i
                 <<" (expected "<<expected_grad<<", got "<<grad_output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    std::cout<<"✅ Sigmoid activation tests passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"ERROR: Sigmoid activation test failed with exception: "<<e.what()<<std::endl;
    return false;
  }
}

/**
 * @brief Test Tanh activation function
 * @return True if all tests pass, false otherwise
 */
bool TestTanhActivation()
{
  std::cout<<"Testing Tanh activation function..."<<std::endl;
  
  try
  {
    CAIF_Framework framework;
    
    // Test 1: Basic Tanh activation
    std::vector<uint32_t> shape={1,4};
    CAIF_Tensor input(framework,shape,CAIF_DataType_e::Float32);
    
    // Set test data
    auto input_data_result=input.MutableData<float>();
    if(input_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get input data"<<std::endl;
      return false;
    }
    float *input_data=input_data_result;
    input_data[0]=-1.0f;
    input_data[1]=0.0f;
    input_data[2]=0.5f;
    input_data[3]=1.0f;
    
    // Apply Tanh activation
    CAIF_Tensor output=input.Tanh();
    
    // Verify output
    auto output_data_result=output.ConstData<float>();
    if(output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get output data"<<std::endl;
      return false;
    }
    const float *output_data=output_data_result;
    
    // Expected values: tanh(x)
    float expected[]={
      std::tanh(-1.0f),
      std::tanh(0.0f),
      std::tanh(0.5f),
      std::tanh(1.0f)
    };
    
    for(size_t i=0;i<4;++i)
    {
      if(std::abs(output_data[i]-expected[i])>1e-6f)
      {
        std::cout<<"ERROR: Tanh activation failed at index "<<i
                 <<" (expected "<<expected[i]<<", got "<<output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    // Test 2: Tanh derivative
    CAIF_Tensor gradient(framework,shape,CAIF_DataType_e::Float32);
    auto grad_data_result=gradient.MutableData<float>();
    if(grad_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient data"<<std::endl;
      return false;
    }
    float *grad_data=grad_data_result;
    for(size_t i=0;i<4;++i)
    {
      grad_data[i]=1.0f;  // Uniform gradient
    }
    
    CAIF_Tensor grad_output=output.TanhDerivative(gradient);  // Use output for derivative
    auto grad_output_data_result=grad_output.ConstData<float>();
    if(grad_output_data_result==nullptr)
    {
      std::cout<<"ERROR: Failed to get gradient output data"<<std::endl;
      return false;
    }
    const float *grad_output_data=grad_output_data_result;
    
    // Expected derivative: tanh'(x) = 1 - tanh²(x)
    for(size_t i=0;i<4;++i)
    {
      float expected_grad=1.0f-output_data[i]*output_data[i];
      if(std::abs(grad_output_data[i]-expected_grad)>1e-6f)
      {
        std::cout<<"ERROR: Tanh derivative failed at index "<<i
                 <<" (expected "<<expected_grad<<", got "<<grad_output_data[i]<<")"<<std::endl;
        return false;
      }
    }
    
    std::cout<<"✅ Tanh activation tests passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"ERROR: Tanh activation test failed with exception: "<<e.what()<<std::endl;
    return false;
  }
}

/**
 * @brief Test activation functions in neural network layers
 * @return True if all tests pass, false otherwise
 */
bool TestActivationsInLayers()
{
  std::cout<<"Testing activation functions in neural network layers..."<<std::endl;
  
  try
  {
    // Test each activation type in a dense layer
    std::vector<CAIF_ActivationType_e> activations={
      CAIF_ActivationType_e::Linear,
      CAIF_ActivationType_e::ReLU,
      CAIF_ActivationType_e::Sigmoid,
      CAIF_ActivationType_e::Tanh
    };
    
    std::vector<std::string> activation_names={"Linear","ReLU","Sigmoid","Tanh"};
    
    for(size_t act_idx=0;act_idx<activations.size();++act_idx)
    {
      std::cout<<"  Testing "<<activation_names[act_idx]<<" in dense layer..."<<std::endl;
      
      // Create a simple network with one dense layer
      CAIF_NeuralNetwork network;
      network.AddDenseLayer(3,activations[act_idx]);
      
      // Set input shape
      std::vector<uint32_t> input_shape={1,2};  // Batch size 1, 2 features
      network.SetInputShape(input_shape);
      
      // Compile the network
      network.Compile(CAIF_OptimizerType_e::SGD,CAIF_LossType_e::MeanSquaredError,0.01f);
      
      // Create test input
      CAIF_Tensor input(network.Framework(),input_shape,CAIF_DataType_e::Float32);
      auto input_data_result=input.MutableData<float>();
      if(input_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get input data"<<std::endl;
        return false;
      }
      float *input_data=input_data_result;
      input_data[0]=1.0f;
      input_data[1]=-0.5f;
      
      // Forward pass
      CAIF_Tensor output=network.Forward(input,false);
      // Check output shape
      const auto &output_shape=output.Shape();
      if(output_shape.size()!=2||output_shape[0]!=1||output_shape[1]!=3)
      {
        std::cout<<"ERROR: Unexpected output shape for "<<activation_names[act_idx]<<std::endl;
        return false;
      }
      
      std::cout<<"    ✅ "<<activation_names[act_idx]<<" layer test passed"<<std::endl;
    }
    
    std::cout<<"✅ All activation functions work correctly in layers"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"ERROR: Layer activation test failed with exception: "<<e.what()<<std::endl;
    return false;
  }
}

/**
 * @brief Test activation function performance characteristics
 * @return True if all tests pass, false otherwise
 */
bool TestActivationCharacteristics()
{
  std::cout<<"Testing activation function characteristics..."<<std::endl;
  
  try
  {
    CAIF_Framework framework;
    std::vector<uint32_t> shape={1,1};
    
    // Test Linear characteristics
    {
      CAIF_Tensor input(framework,shape,CAIF_DataType_e::Float32);
      auto input_data_result=input.MutableData<float>();
      if(input_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get input data"<<std::endl;
        return false;
      }
      float *input_data=input_data_result;
      input_data[0]=5.0f;
      
      CAIF_Tensor output=input.Linear();
      auto output_data_result=output.ConstData<float>();
      if(output_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get output data"<<std::endl;
        return false;
      }
      const float *output_data=output_data_result;
      
      // Linear should preserve input exactly
      if(std::abs(output_data[0]-5.0f)>1e-6f)
      {
        std::cout<<"ERROR: Linear activation doesn't preserve input"<<std::endl;
        return false;
      }
    }
    
    // Test ReLU characteristics
    {
      CAIF_Tensor input(framework,shape,CAIF_DataType_e::Float32);
      auto input_data_result=input.MutableData<float>();
      if(input_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get input data"<<std::endl;
        return false;
      }
      float *input_data=input_data_result;
      
      // Test negative input
      input_data[0]=-3.0f;
      CAIF_Tensor output_neg=input.ReLU();
      auto output_neg_data_result=output_neg.ConstData<float>();
      if(output_neg_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get output data"<<std::endl;
        return false;
      }
      const float *output_neg_data=output_neg_data_result;
      
      if(output_neg_data[0]!=0.0f)
      {
        std::cout<<"ERROR: ReLU should output 0 for negative input"<<std::endl;
        return false;
      }
      
      // Test positive input
      input_data[0]=3.0f;
      CAIF_Tensor output_pos=input.ReLU();
      auto output_pos_data_result=output_pos.ConstData<float>();
      if(output_pos_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get output data"<<std::endl;
        return false;
      }
      const float *output_pos_data=output_pos_data_result;
      
      if(std::abs(output_pos_data[0]-3.0f)>1e-6f)
      {
        std::cout<<"ERROR: ReLU should preserve positive input"<<std::endl;
        return false;
      }
    }
    
    // Test Sigmoid characteristics (output range [0,1])
    {
      CAIF_Tensor input(framework,shape,CAIF_DataType_e::Float32);
      auto input_data_result=input.MutableData<float>();
      if(input_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get input data"<<std::endl;
        return false;
      }
      float *input_data=input_data_result;
      
      // Test extreme values
      input_data[0]=100.0f;  // Very large positive
      CAIF_Tensor output_large=input.Sigmoid();
      auto output_large_data_result=output_large.ConstData<float>();
      if(output_large_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get output data"<<std::endl;
        return false;
      }
      const float *output_large_data=output_large_data_result;
      
      if(output_large_data[0]<0.99f||output_large_data[0]>1.0f)
      {
        std::cout<<"ERROR: Sigmoid output should approach 1 for large positive input"<<std::endl;
        return false;
      }
      
      input_data[0]=-100.0f;  // Very large negative
      CAIF_Tensor output_small=input.Sigmoid();
      auto output_small_data_result=output_small.ConstData<float>();
      if(output_small_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get output data"<<std::endl;
        return false;
      }
      const float *output_small_data=output_small_data_result;
      
      if(output_small_data[0]<0.0f||output_small_data[0]>0.01f)
      {
        std::cout<<"ERROR: Sigmoid output should approach 0 for large negative input"<<std::endl;
        return false;
      }
    }
    
    // Test Tanh characteristics (output range [-1,1])
    {
      CAIF_Tensor input(framework,shape,CAIF_DataType_e::Float32);
      auto input_data_result=input.MutableData<float>();
      if(input_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get input data"<<std::endl;
        return false;
      }
      float *input_data=input_data_result;
      
      // Test extreme values
      input_data[0]=100.0f;  // Very large positive
      CAIF_Tensor output_large=input.Tanh();
      auto output_large_data_result=output_large.ConstData<float>();
      if(output_large_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get output data"<<std::endl;
        return false;
      }
      const float *output_large_data=output_large_data_result;
      
      if(output_large_data[0]<0.99f||output_large_data[0]>1.0f)
      {
        std::cout<<"ERROR: Tanh output should approach 1 for large positive input"<<std::endl;
        return false;
      }
      
      input_data[0]=-100.0f;  // Very large negative
      CAIF_Tensor output_small=input.Tanh();
      auto output_small_data_result=output_small.ConstData<float>();
      if(output_small_data_result==nullptr)
      {
        std::cout<<"ERROR: Failed to get output data"<<std::endl;
        return false;
      }
      const float *output_small_data=output_small_data_result;
      
      if(output_small_data[0]<-1.0f||output_small_data[0]>-0.99f)
      {
        std::cout<<"ERROR: Tanh output should approach -1 for large negative input"<<std::endl;
        return false;
      }
    }
    
    std::cout<<"✅ All activation function characteristics verified"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"ERROR: Activation characteristics test failed with exception: "<<e.what()<<std::endl;
    return false;
  }
}

/**
 * @brief Main test function
 * @return 0 if all tests pass, 1 otherwise
 */
int main()
{
  std::cout<<"=== AIF Activation Functions Test Suite ==="<<std::endl;
  std::cout<<std::endl;
  
  bool all_tests_passed=true;
  
  // Run individual activation function tests
  all_tests_passed&=TestLinearActivation();
  std::cout<<std::endl;
  
  all_tests_passed&=TestReLUActivation();
  std::cout<<std::endl;
  
  all_tests_passed&=TestSigmoidActivation();
  std::cout<<std::endl;
  
  all_tests_passed&=TestTanhActivation();
  std::cout<<std::endl;
  
  // Run integration tests
  all_tests_passed&=TestActivationsInLayers();
  std::cout<<std::endl;
  
  all_tests_passed&=TestActivationCharacteristics();
  std::cout<<std::endl;
  
  // Print final results
  std::cout<<"=== Test Results ==="<<std::endl;
  if(all_tests_passed==true)
  {
    std::cout<<"✅ ALL ACTIVATION FUNCTION TESTS PASSED!"<<std::endl;
    std::cout<<"  - Linear activation: Identity function working correctly"<<std::endl;
    std::cout<<"  - ReLU activation: Rectified linear unit working correctly"<<std::endl;
    std::cout<<"  - Sigmoid activation: Logistic function working correctly"<<std::endl;
    std::cout<<"  - Tanh activation: Hyperbolic tangent working correctly"<<std::endl;
    std::cout<<"  - All derivatives computed correctly"<<std::endl;
    std::cout<<"  - Integration with neural network layers successful"<<std::endl;
    return 0;
  }
  else
  {
    std::cout<<"✗ SOME ACTIVATION FUNCTION TESTS FAILED!"<<std::endl;
    std::cout<<"Please check the error messages above for details."<<std::endl;
    return 1;
  }
}
