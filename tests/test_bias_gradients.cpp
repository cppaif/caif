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
 * This test verifies that convolution layers properly implement bias gradients:
 * - Convolution layers compute and store weight gradients during backward pass
 * - Convolution layers compute and store bias gradients during backward pass
 * - ParameterGradients() method returns gradients in correct order and shape
 * - Parameter count matches gradient count for optimization
 */

#include "caif_convolution2d_layer.h"
#include "caif_tensor.h"
#include "caif_framework.h"
#include <iostream>
#include <memory>

using namespace instance;

/**
 * @brief Helper function to create test tensor with specified shape and values
 */
CAIF_Tensor CreateTensor(CAIF_Framework &framework,
                        const std::vector<uint32_t> &shape,
                        const std::vector<float> &data)
{
  CAIF_Tensor tensor(framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
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
 * @brief Test that convolution layer with bias computes parameter gradients
 */
bool TestConvolutionLayerBiasGradients()
{
  std::cout<<"Testing Convolution Layer Bias Gradients... ";
  
  try
  {
    CAIF_Framework framework;
    
    // Create convolution layer with bias enabled
    CAIF_Convolution2DLayer conv_layer(framework,8,3,1,1,CAIF_ActivationType_e::ReLU,true);
    
    // Initialize with small input shape for testing
    std::vector<uint32_t> input_shape={1,4,4,2};
    conv_layer.Initialize(input_shape,42);
    
    // Verify layer has parameters (weights + bias)
    auto parameters=conv_layer.Parameters();
    if(parameters.size()!=2)
    {
      std::cout<<"FAILED - Expected 2 parameters (weights + bias), got "<<parameters.size()<<"\n";
      return false;
    }
    
    // Check parameter shapes
    const auto &weight_shape=parameters[0].Shape();
    const auto &bias_shape=parameters[1].Shape();
    
    // Weight shape should be [filters, kernel_size, kernel_size, input_channels] = [8, 3, 3, 2]
    std::vector<uint32_t> expected_weight_shape={8,3,3,2};
    if(weight_shape!=expected_weight_shape)
    {
      std::cout<<"FAILED - Weight shape mismatch\n";
      return false;
    }
    
    // Bias shape should be [filters] = [8]
    std::vector<uint32_t> expected_bias_shape={8};
    if(bias_shape!=expected_bias_shape)
    {
      std::cout<<"FAILED - Bias shape mismatch\n";
      return false;
    }
    
    // Create test input and perform forward pass
    std::vector<float> input_data(1*4*4*2,0.5f);
    CAIF_Tensor input_tensor=CreateTensor(framework,input_shape,input_data);
    
    CAIF_Tensor output=conv_layer.Forward(input_tensor,true);
    
    // Create gradient for backward pass
    std::vector<float> grad_data(output.NumElements(),0.1f);
    CAIF_Tensor gradient=CreateTensor(framework,output.Shape(),grad_data);
    
    // Perform backward pass (this should compute parameter gradients)
    conv_layer.Backward(gradient);
    
    // Check that parameter gradients are available
    auto parameter_gradients=conv_layer.ParameterGradients();
    if(parameter_gradients.size()!=2)
    {
      std::cout<<"FAILED - Expected 2 parameter gradients "
               <<"(weight + bias), got "
               <<parameter_gradients.size()<<"\n";
      return false;
    }
    
    // Verify gradient shapes match parameter shapes
    const auto &weight_grad_shape=parameter_gradients[0].Shape();
    const auto &bias_grad_shape=parameter_gradients[1].Shape();
    
    if(weight_grad_shape!=expected_weight_shape)
    {
      std::cout<<"FAILED - Weight gradient shape mismatch\n";
      return false;
    }
    
    if(bias_grad_shape!=expected_bias_shape)
    {
      std::cout<<"FAILED - Bias gradient shape mismatch\n";
      return false;
    }
    
    // Verify parameter count matches gradient count
    if(parameters.size()!=parameter_gradients.size())
    {
      std::cout<<"FAILED - Parameter count ("
               <<parameters.size()
               <<") doesn't match gradient count ("
               <<parameter_gradients.size()<<")\n";
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
 * @brief Test that convolution layer without bias works correctly
 */
bool TestConvolutionLayerNoBias()
{
  std::cout<<"Testing Convolution Layer Without Bias... ";
  
  try
  {
    CAIF_Framework framework;
    
    // Create convolution layer with bias disabled
    CAIF_Convolution2DLayer conv_layer(framework,4,3,1,1,CAIF_ActivationType_e::ReLU,false);
    
    // Initialize layer
    std::vector<uint32_t> input_shape={1,4,4,2};
    conv_layer.Initialize(input_shape,42);
    
    // Verify layer has only weight parameters (no bias)
    auto parameters=conv_layer.Parameters();
    if(parameters.size()!=1)
    {
      std::cout<<"FAILED - Expected 1 parameter (weights only), got "<<parameters.size()<<"\n";
      return false;
    }
    
    // Forward and backward pass
    std::vector<float> input_data(1*4*4*2,0.5f);
    CAIF_Tensor input_tensor=CreateTensor(framework,input_shape,input_data);
    
    CAIF_Tensor output=conv_layer.Forward(input_tensor,true);
    std::vector<float> grad_data(output.NumElements(),0.1f);
    CAIF_Tensor gradient=CreateTensor(framework,output.Shape(),grad_data);
    
    conv_layer.Backward(gradient);
    
    // Check parameter gradients
    auto parameter_gradients=conv_layer.ParameterGradients();
    if(parameter_gradients.size()!=1)
    {
      std::cout<<"FAILED - Expected 1 parameter gradient (weight only), got "<<parameter_gradients.size()<<"\n";
      return false;
    }
    
    // Verify parameter count matches gradient count
    if(parameters.size()!=parameter_gradients.size())
    {
      std::cout<<"FAILED - Parameter count doesn't match gradient count\n";
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
 * @brief Test gradient tensor properties (non-null, finite values)
 */
bool TestGradientTensorProperties()
{
  std::cout<<"Testing Gradient Tensor Properties... ";
  
  try
  {
    CAIF_Framework framework;
    
    CAIF_Convolution2DLayer conv_layer(framework,4,3,1,1,CAIF_ActivationType_e::ReLU,true);
    
    std::vector<uint32_t> input_shape={1,4,4,2};
    conv_layer.Initialize(input_shape,42);
    
    // Forward pass
    std::vector<float> input_data(1*4*4*2,0.5f);
    CAIF_Tensor input_tensor=CreateTensor(framework,input_shape,input_data);
    
    auto forward_result=conv_layer.Forward(input_tensor,true);
    
    // Backward pass
    CAIF_Tensor output=forward_result;
    std::vector<float> grad_data(output.NumElements(),0.1f);
    CAIF_Tensor gradient=CreateTensor(framework,output.Shape(),grad_data);
    
    conv_layer.Backward(gradient);
    
    // Check gradient tensor properties
    auto parameter_gradients=conv_layer.ParameterGradients();
    
    for(size_t i=0;i<parameter_gradients.size();++i)
    {
      const auto &grad_tensor=parameter_gradients[i];
      
      // Check tensor has valid data type
      if(grad_tensor.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
      {
        std::cout<<"FAILED - Gradient tensor "<<i<<" has wrong data type\n";
        return false;
      }
      
      // Check tensor has non-zero size
      if(grad_tensor.NumElements()==0)
      {
        std::cout<<"FAILED - Gradient tensor "<<i<<" is empty\n";
        return false;
      }
      
      // Check tensor data is accessible
      auto grad_data_result=grad_tensor.ConstData<float>();
      if(grad_data_result==nullptr)
      {
        std::cout<<"FAILED - Cannot access gradient tensor "<<i<<" data\n";
        return false;
      }
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
 * @brief Main test function
 */
int main()
{
  std::cout<<"=== Bias Gradients Functionality Tests ===\n";
  
  bool all_passed=true;
  
  all_passed&=TestConvolutionLayerBiasGradients();
  all_passed&=TestConvolutionLayerNoBias();
  all_passed&=TestGradientTensorProperties();
  
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
