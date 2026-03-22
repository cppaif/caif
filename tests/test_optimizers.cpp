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
 * @file test_optimizers.cpp
 * @brief Test suite for AIF optimizers
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_sgd_optimizer.h"
#include "caif_adam_optimizer.h"
#include "caif_tensor.h"
#include "caif_framework.h"
#include "caif_exception.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace instance;

// Global framework for tests
static CAIF_Framework g_test_framework;

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
 * @brief Helper function to compare tensors with tolerance
 */
bool CompareTensors(
                    const CAIF_Tensor &tensor1,
                    const CAIF_Tensor &tensor2,
                    const float tolerance=1e-6f
                   )
{
  if((tensor1.Shape()==tensor2.Shape())==false||(tensor1.NumElements()==tensor2.NumElements())==false)
  {
    return false;
  }
  
  auto data1_result=tensor1.ConstData<float>();
  auto data2_result=tensor2.ConstData<float>();
  
  if(data1_result==nullptr||data2_result==nullptr)
  {
    return false;
  }
  
  const float *data1=data1_result;
  const float *data2=data2_result;
  
  for(uint32_t i=0;i<tensor1.NumElements();++i)
  {
    if((std::abs(data1[i]-data2[i])<=tolerance)==false)
    {
      return false;
    }
  }
  
  return true;
}

/**
 * @brief Test SGD optimizer basic functionality
 */
bool TestSGDBasic()
{
  std::cout<<"Testing SGD Optimizer - Basic... ";
  
  try
  {
    CAIF_SGDOptimizer sgd(g_test_framework,0.1f);  // Learning rate = 0.1
    
    // Create simple parameter and gradient tensors
    std::vector<uint32_t> shape={2,2};
    std::vector<CAIF_Tensor> parameters;
    std::vector<CAIF_Tensor> gradients;
    
    // Parameter: [[1.0, 2.0], [3.0, 4.0]]
    parameters.push_back(CreateTensor(shape,{1.0f,2.0f,3.0f,4.0f}));
    
    // Gradient: [[0.1, 0.2], [0.3, 0.4]]
    gradients.push_back(CreateTensor(shape,{0.1f,0.2f,0.3f,0.4f}));
    
    // Update parameters
    auto updated_params=sgd.UpdateParameters(parameters,gradients);
    
    // Expected: param - learning_rate * grad
    // = [[1.0, 2.0], [3.0, 4.0]] - 0.1 * [[0.1, 0.2], [0.3, 0.4]]
    // = [[1.0, 2.0], [3.0, 4.0]] - [[0.01, 0.02], [0.03, 0.04]]
    // = [[0.99, 1.98], [2.97, 3.96]]
    CAIF_Tensor expected=CreateTensor(shape,{0.99f,1.98f,2.97f,3.96f});
    
    if(CompareTensors(updated_params[0],expected,1e-6f)==false)
    {
      std::cout<<"FAILED - Parameter update mismatch\n";
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
 * @brief Test SGD optimizer with momentum
 */
bool TestSGDMomentum()
{
  std::cout<<"Testing SGD Optimizer - Momentum... ";
  
  try
  {
    CAIF_SGDOptimizer sgd(g_test_framework,0.1f,0.9f);  // Learning rate = 0.1, momentum = 0.9
    
    // Create simple parameter and gradient tensors
    std::vector<uint32_t> shape={2};
    std::vector<CAIF_Tensor> parameters;
    std::vector<CAIF_Tensor> gradients;
    
    // Initial parameters: [1.0, 2.0]
    parameters.push_back(CreateTensor(shape,{1.0f,2.0f}));
    
    // First gradient: [0.1, 0.2]
    gradients.push_back(CreateTensor(shape,{0.1f,0.2f}));
    
    // First update (no momentum yet)
    auto update1_result=sgd.UpdateParameters(parameters,gradients);
    
    // Second update with same gradient (should accumulate momentum)
    parameters=update1_result;
    auto final_params=sgd.UpdateParameters(parameters,gradients);
    
    // With momentum, the second update should have larger step size
    // First update:
    //   velocity = -0.1*[0.1,0.2] = [-0.01,-0.02]
    //   param = [1.0,2.0]+[-0.01,-0.02] = [0.99,1.98]
    // Second update:
    //   velocity = 0.9*[-0.01,-0.02]-0.1*[0.1,0.2]
    //            = [-0.009,-0.018]+[-0.01,-0.02]
    //            = [-0.019,-0.038]
    //                param = [0.99, 1.98] + [-0.019, -0.038] = [0.971, 1.942]
    
    auto final_data_result=final_params[0].ConstData<float>();
    if(final_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not access final parameter data\n";
      return false;
    }
    
    const float *final_data=final_data_result;
    float expected_0=0.971f;
    float expected_1=1.942f;
    float tolerance=1e-3f;
    
    if((std::abs(final_data[0]-expected_0)<=tolerance)==false ||
       (std::abs(final_data[1]-expected_1)<=tolerance)==false)
    {
      std::cout<<"FAILED - Expected ["
               <<expected_0<<", "<<expected_1
               <<"] but got ["
               <<final_data[0]<<", "
               <<final_data[1]<<"]\n";
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
 * @brief Test Adam optimizer basic functionality
 */
bool TestAdamBasic()
{
  std::cout<<"Testing Adam Optimizer - Basic... ";
  
  try
  {
    CAIF_AdamOptimizer adam(g_test_framework,0.001f,0.9f,0.999f,1e-8f);  // Default Adam parameters
    
    // Create simple parameter and gradient tensors
    std::vector<uint32_t> shape={2,2};
    std::vector<CAIF_Tensor> parameters;
    std::vector<CAIF_Tensor> gradients;
    
    // Parameter: [[1.0, 2.0], [3.0, 4.0]]
    parameters.push_back(CreateTensor(shape,{1.0f,2.0f,3.0f,4.0f}));
    
    // Gradient: [[0.1, 0.2], [0.3, 0.4]]
    gradients.push_back(CreateTensor(shape,{0.1f,0.2f,0.3f,0.4f}));
    
    // Update parameters
    auto updated_params=adam.UpdateParameters(parameters,gradients);
    
    // Verify that parameters were updated
    auto orig_data_result=parameters[0].ConstData<float>();
    auto updated_data_result=updated_params[0].ConstData<float>();
    
    if(orig_data_result==nullptr||updated_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not access tensor data\n";
      return false;
    }
    
    const float *orig_data=orig_data_result;
    const float *updated_data=updated_data_result;
    
    // Parameters should change but not by too much in a single update
    bool any_changed=false;
    bool all_reasonable=true;
    
    for(uint32_t i=0;i<parameters[0].NumElements();++i)
    {
      if((std::abs(updated_data[i]-orig_data[i])>1e-5f)==true)
      {
        any_changed=true;
      }
      
      // Increase tolerance for Adam updates
      if((std::abs(updated_data[i]-orig_data[i])<=0.01f)==false)
      {
        all_reasonable=false;
      }
    }
    
    if(any_changed==false)
    {
      std::cout<<"FAILED - Parameters did not change after update\n";
      return false;
    }
    
    if(all_reasonable==false)
    {
      std::cout<<"FAILED - Parameter updates too large for a single step\n";
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
 * @brief Test Adam optimizer convergence on a simple problem
 */
bool TestAdamConvergence()
{
  std::cout<<"Testing Adam Optimizer - Convergence... ";
  
  try
  {
    CAIF_AdamOptimizer adam(g_test_framework,0.1f);  // Slightly higher learning rate for faster convergence
    
    // Create a simple quadratic function: f(x) = x^2, with gradient df/dx = 2x
    // Starting from x = 5.0, should converge toward x = 0
    
    std::vector<uint32_t> shape={1};
    std::vector<CAIF_Tensor> parameters;
    parameters.push_back(CreateTensor(shape,{5.0f}));  // Starting point
    
    const int num_iterations=100;
    
    for(int i=0;i<num_iterations;++i)
    {
      // Get current parameter value
      auto param_data_result=parameters[0].ConstData<float>();
      if(param_data_result==nullptr)
      {
        std::cout<<"FAILED - Could not access parameter data\n";
        return false;
      }
      
      const float *param_data=param_data_result;
      float x=param_data[0];
      
      // Compute gradient: df/dx = 2x
      std::vector<CAIF_Tensor> gradients;
      gradients.push_back(CreateTensor(shape,{2.0f*x}));
      
      // Update parameters
      auto update_result=adam.UpdateParameters(parameters,gradients);
      parameters=update_result;
    }
    
    // Check final value - should be close to optimal (x = 0)
    auto final_data_result=parameters[0].ConstData<float>();
    if(final_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not access final parameter data\n";
      return false;
    }
    
    const float *final_data=final_data_result;
    float final_x=final_data[0];
    
    if((std::abs(final_x)<=0.1f)==false)  // Should be close to 0
    {
      std::cout<<"FAILED - Did not converge properly, final x = "<<final_x<<"\n";
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
 * @brief Test optimizer reset functionality
 */
bool TestOptimizerReset()
{
  std::cout<<"Testing Optimizer Reset... ";
  
  try
  {
    CAIF_SGDOptimizer sgd(g_test_framework,0.1f,0.9f);  // With momentum
    
    // Create parameter and gradient tensors
    std::vector<uint32_t> shape={2};
    std::vector<CAIF_Tensor> parameters;
    std::vector<CAIF_Tensor> gradients;
    
    parameters.push_back(CreateTensor(shape,{1.0f,2.0f}));
    gradients.push_back(CreateTensor(shape,{0.1f,0.2f}));
    
    // First update to build up momentum
    auto update1_result=sgd.UpdateParameters(parameters,gradients);
    if(false)
    {
      std::cout<<"FAILED - First update failed\n";
      return false;
    }
    parameters=update1_result;
    
    // Second update to further build momentum
    auto update2_result=sgd.UpdateParameters(parameters,gradients);
    if(false)
    {
      std::cout<<"FAILED - Second update failed\n";
      return false;
    }
    parameters=update2_result;
    
    // Store parameters after momentum has built up
    CAIF_Tensor params_with_momentum=parameters[0];
    
    // Reset optimizer
    sgd.Reset();
    if(false)
    {
      std::cout<<"FAILED - Reset failed: "<<"\n";
      return false;
    }
    
    // Update again (should be like first update, no momentum)
    auto update3_result=sgd.UpdateParameters(parameters,gradients);
    if(false)
    {
      std::cout<<"FAILED - Update after reset failed\n";
      return false;
    }
    CAIF_Tensor params_after_reset=(update3_result)[0];
    
    // Verify that the update after reset is different from the update with momentum
    auto data1_result=params_with_momentum.ConstData<float>();
    auto data2_result=params_after_reset.ConstData<float>();
    
    if(data1_result==nullptr||data2_result==nullptr)
    {
      std::cout<<"FAILED - Could not access tensor data\n";
      return false;
    }
    
    const float *data1=data1_result;
    const float *data2=data2_result;
    
    // The updates should be different due to momentum reset
    if((std::abs(data1[0]-data2[0])>1e-6f)==false||(std::abs(data1[1]-data2[1])>1e-6f)==false)
    {
      std::cout<<"FAILED - Reset did not clear optimizer state\n";
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
 * @brief Test optimizer error handling
 */
bool TestOptimizerErrorHandling()
{
  std::cout<<"Testing Optimizer Error Handling...\n";
  
  try
  {
    CAIF_SGDOptimizer sgd(g_test_framework,0.1f);
    bool all_tests_passed=true;
    
    // Test 1: Empty parameters
    std::cout<<"  Test 1: Empty parameters... ";
    {
      std::vector<CAIF_Tensor> empty_params;
      std::vector<CAIF_Tensor> gradients;
      gradients.push_back(CreateTensor({2,2},{0.1f,0.2f,0.3f,0.4f}));
      
      bool exception_caught=false;
      try
      {
        auto result=sgd.UpdateParameters(empty_params,gradients);
        std::cout<<"❌ FAILED - No exception thrown\n";
        all_tests_passed=false;
      }
      catch(const instance::CAIF_Exception &e)
      {
        // Check error message contains expected text
        if(e.Stack().empty()==false)
        {
          std::string error_msg=e.Stack()[0]._desc;
          if((error_msg.find("must have the same size")!=std::string::npos)==true)
          {
            exception_caught=true;
          }
          else
          {
            std::cout<<"❌ FAILED - Unexpected error message: "<<error_msg<<"\n";
            all_tests_passed=false;
          }
        }
        else
        {
          std::cout<<"❌ FAILED - Exception stack is empty\n";
          all_tests_passed=false;
        }
      }
      catch(...)
      {
        std::cout<<"❌ FAILED - Unexpected exception type\n";
        all_tests_passed=false;
      }
      
      if(exception_caught==true)
      {
        std::cout<<"✅ PASSED\n";
      }
      else if(exception_caught==false && all_tests_passed==false)
      {
        // Error message already printed
      }
      else
      {
        std::cout<<"❌ FAILED - Expected exception was not caught\n";
        all_tests_passed=false;
      }
    }
    
    // Test 2: Empty gradients
    std::cout<<"  Test 2: Empty gradients... ";
    {
      std::vector<CAIF_Tensor> parameters;
      parameters.push_back(CreateTensor({2,2},{1.0f,2.0f,3.0f,4.0f}));
      std::vector<CAIF_Tensor> empty_grads;
      
      bool exception_caught=false;
      try
      {
        auto result=sgd.UpdateParameters(parameters,empty_grads);
        std::cout<<"❌ FAILED - No exception thrown\n";
        all_tests_passed=false;
      }
      catch(const instance::CAIF_Exception &e)
      {
        // Check error message contains expected text
        if(e.Stack().empty()==false)
        {
          std::string error_msg=e.Stack()[0]._desc;
          if((error_msg.find("must have the same size")!=std::string::npos)==true)
          {
            exception_caught=true;
          }
          else
          {
            std::cout<<"❌ FAILED - Unexpected error message: "<<error_msg<<"\n";
            all_tests_passed=false;
          }
        }
        else
        {
          std::cout<<"❌ FAILED - Exception stack is empty\n";
          all_tests_passed=false;
        }
      }
      catch(...)
      {
        std::cout<<"❌ FAILED - Unexpected exception type\n";
        all_tests_passed=false;
      }
      
      if(exception_caught==true)
      {
        std::cout<<"✅ PASSED\n";
      }
      else if(exception_caught==false && all_tests_passed==false)
      {
        // Error message already printed
      }
      else
      {
        std::cout<<"❌ FAILED - Expected exception was not caught\n";
        all_tests_passed=false;
      }
    }
    
    // Test 3: Parameter/gradient count mismatch
    std::cout<<"  Test 3: Parameter/gradient count mismatch... ";
    {
      std::vector<CAIF_Tensor> parameters;
      parameters.push_back(CreateTensor({2,2},{1.0f,2.0f,3.0f,4.0f}));
      parameters.push_back(CreateTensor({2,2},{5.0f,6.0f,7.0f,8.0f}));
      
      std::vector<CAIF_Tensor> gradients;
      gradients.push_back(CreateTensor({2,2},{0.1f,0.2f,0.3f,0.4f}));
      
      bool exception_caught=false;
      try
      {
        auto result=sgd.UpdateParameters(parameters,gradients);
        std::cout<<"❌ FAILED - No exception thrown\n";
        all_tests_passed=false;
      }
      catch(const instance::CAIF_Exception &e)
      {
        // Check error message contains expected text
        if(e.Stack().empty()==false)
        {
          std::string error_msg=e.Stack()[0]._desc;
          if((error_msg.find("must have the same size")!=std::string::npos)==true)
          {
            exception_caught=true;
          }
          else
          {
            std::cout<<"❌ FAILED - Unexpected error message: "<<error_msg<<"\n";
            all_tests_passed=false;
          }
        }
        else
        {
          std::cout<<"❌ FAILED - Exception stack is empty\n";
          all_tests_passed=false;
        }
      }
      catch(...)
      {
        std::cout<<"❌ FAILED - Unexpected exception type\n";
        all_tests_passed=false;
      }
      
      if(exception_caught==true)
      {
        std::cout<<"✅ PASSED\n";
      }
      else if(exception_caught==false && all_tests_passed==false)
      {
        // Error message already printed
      }
      else
      {
        std::cout<<"❌ FAILED - Expected exception was not caught\n";
        all_tests_passed=false;
      }
    }
    
    // Test 4: Parameter/gradient shape mismatch
    std::cout<<"  Test 4: Parameter/gradient shape mismatch... ";
    {
      std::vector<CAIF_Tensor> parameters;
      parameters.push_back(CreateTensor({2,2},{1.0f,2.0f,3.0f,4.0f}));
      
      std::vector<CAIF_Tensor> gradients;
      gradients.push_back(CreateTensor({3,3},{0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f}));
      
      bool exception_caught=false;
      try
      {
        auto result=sgd.UpdateParameters(parameters,gradients);
        std::cout<<"❌ FAILED - No exception thrown\n";
        all_tests_passed=false;
      }
      catch(const instance::CAIF_Exception &e)
      {
        // Check error message contains expected text
        if(e.Stack().empty()==false)
        {
          std::string error_msg=e.Stack()[0]._desc;
          if((error_msg.find("shape")!=std::string::npos)==true)
          {
            exception_caught=true;
          }
          else
          {
            std::cout<<"❌ FAILED - Unexpected error message: "<<error_msg<<"\n";
            all_tests_passed=false;
          }
        }
        else
        {
          std::cout<<"❌ FAILED - Exception stack is empty\n";
          all_tests_passed=false;
        }
      }
      catch(...)
      {
        std::cout<<"❌ FAILED - Unexpected exception type\n";
        all_tests_passed=false;
      }
      
      if(exception_caught==true)
      {
        std::cout<<"✅ PASSED\n";
      }
      else if(exception_caught==false && all_tests_passed==false)
      {
        // Error message already printed
      }
      else
      {
        std::cout<<"❌ FAILED - Expected exception was not caught\n";
        all_tests_passed=false;
      }
    }
    
    // Summary
    std::cout<<"  Error Handling Test Summary: ";
    if(all_tests_passed==true)
    {
      std::cout<<"✅ PASSED\n";
    }
    else
    {
      std::cout<<"❌ FAILED\n";
    }
    
    return all_tests_passed;
  }
  catch(const std::exception &e)
  {
    std::cout<<"❌ FAILED - Unexpected exception: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Test optimizer cloning
 */
bool TestOptimizerCloning()
{
  std::cout<<"Testing Optimizer Cloning... ";
  
  try
  {
    // Test SGD cloning
    {
      CAIF_SGDOptimizer original_sgd(g_test_framework,0.05f,0.8f);
      std::unique_ptr<CAIF_Optimizer> cloned_sgd=original_sgd.Clone();
      
      if(cloned_sgd==nullptr)
      {
        std::cout<<"FAILED - SGD clone returned nullptr\n";
        return false;
      }
      
      // Verify cloned optimizer is of the correct type
      CAIF_SGDOptimizer *sgd_ptr=dynamic_cast<CAIF_SGDOptimizer*>(cloned_sgd.get());
      if(sgd_ptr==nullptr)
      {
        std::cout<<"FAILED - SGD clone is not of type CAIF_SGDOptimizer\n";
        return false;
      }
      
      // Verify cloned optimizer has the same parameters
      if((sgd_ptr->LearningRate()==original_sgd.LearningRate())==false||
         (sgd_ptr->Momentum()==original_sgd.Momentum())==false)
      {
        std::cout<<"FAILED - SGD clone parameters don't match original\n";
        return false;
      }
    }
    
    // Test Adam cloning
    {
      CAIF_AdamOptimizer original_adam(g_test_framework,0.002f,0.85f,0.995f,1e-7f);
      std::unique_ptr<CAIF_Optimizer> cloned_adam=original_adam.Clone();
      
      if(cloned_adam==nullptr)
      {
        std::cout<<"FAILED - Adam clone returned nullptr\n";
        return false;
      }
      
      // Verify cloned optimizer is of the correct type
      CAIF_AdamOptimizer *adam_ptr=dynamic_cast<CAIF_AdamOptimizer*>(cloned_adam.get());
      if(adam_ptr==nullptr)
      {
        std::cout<<"FAILED - Adam clone is not of type CAIF_AdamOptimizer\n";
        return false;
      }
      
      // Verify cloned optimizer has the same parameters
      if((adam_ptr->LearningRate()==original_adam.LearningRate())==false||
         (adam_ptr->Beta1()==original_adam.Beta1())==false||
         (adam_ptr->Beta2()==original_adam.Beta2())==false||
         (adam_ptr->Epsilon()==original_adam.Epsilon())==false)
      {
        std::cout<<"FAILED - Adam clone parameters don't match original\n";
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
 * @brief Test Adam gradient clipping
 */
bool TestAdamClipping()
{
  std::cout<<"Testing Adam Optimizer - Gradient Clipping... ";
  try
  {
    CAIF_AdamOptimizer adam(g_test_framework,0.001f,0.9f,0.999f,1e-8f);
    std::vector<uint32_t> shape={4};
    std::vector<CAIF_Tensor> parameters;
    std::vector<CAIF_Tensor> gradients;
    parameters.push_back(CreateTensor(shape,{1.0f,2.0f,3.0f,4.0f}));
    // Very large gradients to trigger clipping
    gradients.push_back(CreateTensor(shape,{1e6f,-1e6f,1e6f,-1e6f}));
    const auto &updated=adam.UpdateParameters(parameters,gradients);
    auto udata=updated[0].ConstData<float>();
    auto pdat=parameters[0].ConstData<float>();
    if(udata==nullptr||pdat==nullptr)
    {
      std::cout<<"FAILED - data access\n";
      return false;
    }
    const float *u=udata;
    const float *p=pdat;
    // Ensure update produced finite values and changed parameters
    for(uint32_t i=0;i<updated[0].NumElements();++i)
    {
      if(std::isfinite(u[i])==false)
      {
        std::cout<<"FAILED - non-finite param\n";
        return false;
      }
      if((std::abs(u[i]-p[i])>0.0f)==false)
      {
        std::cout<<"FAILED - no change\n";
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
 * @brief Main function to run all tests
 */
int main(int,char**)
{
  std::cout<<"=== Optimizer Tests ===\n";
  
  bool all_passed=true;
  
  all_passed&=TestSGDBasic();
  all_passed&=TestSGDMomentum();
  all_passed&=TestAdamBasic();
  all_passed&=TestAdamConvergence();
  all_passed&=TestOptimizerReset();
  all_passed&=TestOptimizerErrorHandling();
  all_passed&=TestOptimizerCloning();
  all_passed&=TestAdamClipping();
  
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