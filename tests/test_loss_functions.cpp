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
 * @file test_loss_functions.cpp
 * @brief Test suite for CAIF loss functions
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_mean_squared_error_loss.h"
#include "caif_cross_entropy_loss.h"
#include "caif_categorical_cross_entropy_loss.h"
#include "caif_loss_function.h"
#include "caif_loss_function_bce_logits.h"
#include "caif_tensor.h"
#include "caif_framework.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace instance;
using CAIF_DataType_e = CAIF_DataType::CAIF_DataType_e;

/**
 * @brief Test MSE loss function with simple case
 */
bool TestMSELossSimple()
{
  std::cout<<"Testing MSE Loss - Simple Case... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_MeanSquaredErrorLoss mse_loss;
    
    // Create simple 2x2 prediction and target tensors
    std::vector<uint32_t> shape={2,2};
    CAIF_Tensor predictions(framework,shape,CAIF_DataType_e::Float32);
    CAIF_Tensor targets(framework,shape,CAIF_DataType_e::Float32);
    
    // Set prediction data: [1.0, 2.0, 3.0, 4.0]
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    pred_data[0]=1.0f;
    pred_data[1]=2.0f;
    pred_data[2]=3.0f;
    pred_data[3]=4.0f;
    
    // Set target data: [1.5, 2.5, 2.5, 3.5]
    auto target_data_result=targets.MutableData<float>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    float *target_data=target_data_result;
    target_data[0]=1.5f;
    target_data[1]=2.5f;
    target_data[2]=2.5f;
    target_data[3]=3.5f;
    
    // Compute loss
    CAIF_Tensor loss=mse_loss.ComputeLoss(predictions,targets);
    auto loss_data_result=loss.ConstData<float>();
    if(loss_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get loss data\n";
      return false;
    }
    
    const float *loss_data=loss_data_result;
    
    // Expected: mean of [(1-1.5)^2, (2-2.5)^2, (3-2.5)^2, (4-3.5)^2] = mean of [0.25, 0.25, 0.25, 0.25] = 0.25
    float expected_loss=0.25f;
    float tolerance=1e-6f;
    
    // Check each batch element
    if(std::abs(loss_data[0]-expected_loss)>tolerance||std::abs(loss_data[1]-expected_loss)>tolerance)
    {
      std::cout<<"FAILED - Expected loss "<<expected_loss<<" but got ["<<loss_data[0]<<", "<<loss_data[1]<<"]\n";
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
 * @brief Test MSE gradient computation
 */
bool TestMSEGradient()
{
  std::cout<<"Testing MSE Gradient... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_MeanSquaredErrorLoss mse_loss;
    
    // Create simple 1x2 tensors for easier gradient verification
    std::vector<uint32_t> shape={1,2};
    CAIF_Tensor predictions(framework,shape,CAIF_DataType_e::Float32);
    CAIF_Tensor targets(framework,shape,CAIF_DataType_e::Float32);
    
    // Set prediction data: [2.0, 4.0]
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    pred_data[0]=2.0f;
    pred_data[1]=4.0f;
    
    // Set target data: [1.0, 3.0]
    auto target_data_result=targets.MutableData<float>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    float *target_data=target_data_result;
    target_data[0]=1.0f;
    target_data[1]=3.0f;
    
    // Compute gradient
    CAIF_Tensor gradient=mse_loss.ComputeGradient(predictions,targets);
    auto grad_data_result=gradient.ConstData<float>();
    if(grad_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get gradient data\n";
      return false;
    }
    
    const float *grad_data=grad_data_result;
    
    // Expected gradient: 2 * (predictions - targets) / N
    // = 2 * ([2,4] - [1,3]) / 2 = 2 * [1,1] / 2 = [1,1]
    float expected_grad_0=1.0f;
    float expected_grad_1=1.0f;
    float tolerance=1e-6f;
    
    if(std::abs(grad_data[0]-expected_grad_0)>tolerance||std::abs(grad_data[1]-expected_grad_1)>tolerance)
    {
      std::cout<<"FAILED - Expected gradient ["
               <<expected_grad_0<<", "
               <<expected_grad_1<<"] but got ["
               <<grad_data[0]<<", "
               <<grad_data[1]<<"]\n";
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
 * @brief Test Cross Entropy loss function
 */
bool TestCrossEntropyLoss()
{
  std::cout<<"Testing Cross Entropy Loss... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_CrossEntropyLoss ce_loss(1e-7f);
    
    // Create 2x3 tensors (2 samples, 3 classes)
    std::vector<uint32_t> shape={2,3};
    CAIF_Tensor predictions(framework,shape,CAIF_DataType_e::Float32);
    CAIF_Tensor targets(framework,shape,CAIF_DataType_e::Float32);
    
    // Set prediction data (softmax-like probabilities)
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    
    // Sample 1: [0.7, 0.2, 0.1] - confident prediction for class 0
    pred_data[0]=0.7f;
    pred_data[1]=0.2f;
    pred_data[2]=0.1f;
    
    // Sample 2: [0.1, 0.8, 0.1] - confident prediction for class 1
    pred_data[3]=0.1f;
    pred_data[4]=0.8f;
    pred_data[5]=0.1f;
    
    // Set target data (one-hot encoded)
    auto target_data_result=targets.MutableData<float>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    float *target_data=target_data_result;
    
    // Sample 1: [1, 0, 0] - true class is 0
    target_data[0]=1.0f;
    target_data[1]=0.0f;
    target_data[2]=0.0f;
    
    // Sample 2: [0, 1, 0] - true class is 1
    target_data[3]=0.0f;
    target_data[4]=1.0f;
    target_data[5]=0.0f;
    
    // Compute loss
    CAIF_Tensor loss=ce_loss.ComputeLoss(predictions,targets);
    auto loss_data_result=loss.ConstData<float>();
    if(loss_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get loss data\n";
      return false;
    }
    
    const float *loss_data=loss_data_result;
    
    // Expected loss for sample 1: -log(0.7) ≈ 0.357
    // Expected loss for sample 2: -log(0.8) ≈ 0.223
    float expected_loss_1=-std::log(0.7f);
    float expected_loss_2=-std::log(0.8f);
    float tolerance=1e-3f;
    
    if(std::abs(loss_data[0]-expected_loss_1)>tolerance ||
       std::abs(loss_data[1]-expected_loss_2)>tolerance)
    {
      std::cout<<"FAILED - Expected losses ["
               <<expected_loss_1<<", "
               <<expected_loss_2<<"] but got ["
               <<loss_data[0]<<", "
               <<loss_data[1]<<"]\n";
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
 * @brief Test Cross Entropy gradient computation
 */
bool TestCrossEntropyGradient()
{
  std::cout<<"Testing Cross Entropy Gradient... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_CrossEntropyLoss ce_loss(1e-7f);
    
    // Create 1x2 tensors for simple gradient verification
    std::vector<uint32_t> shape={1,2};
    CAIF_Tensor predictions(framework,shape,CAIF_DataType_e::Float32);
    CAIF_Tensor targets(framework,shape,CAIF_DataType_e::Float32);
    
    // Set prediction data
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    pred_data[0]=0.8f;
    pred_data[1]=0.2f;
    
    // Set target data (one-hot: true class is 0)
    auto target_data_result=targets.MutableData<float>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    float *target_data=target_data_result;
    target_data[0]=1.0f;
    target_data[1]=0.0f;
    
    // Compute gradient
    CAIF_Tensor gradient=ce_loss.ComputeGradient(predictions,targets);
    auto grad_data_result=gradient.ConstData<float>();
    if(grad_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get gradient data\n";
      return false;
    }
    
    const float *grad_data=grad_data_result;
    
    // Expected gradient: -target / (prediction * batch_size)
    // For class 0: -1.0 / (0.8 * 1) = -1.25
    // For class 1: -0.0 / (0.2 * 1) = 0.0
    float expected_grad_0=-1.25f;
    float expected_grad_1=0.0f;
    float tolerance=1e-6f;
    
    if(std::abs(grad_data[0]-expected_grad_0)>tolerance ||
       std::abs(grad_data[1]-expected_grad_1)>tolerance)
    {
      std::cout<<"FAILED - Expected gradient ["
               <<expected_grad_0<<", "
               <<expected_grad_1<<"] but got ["
               <<grad_data[0]<<", "
               <<grad_data[1]<<"]\n";
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
 * @brief Test Categorical Cross Entropy loss with class indices
 */
bool TestCategoricalCrossEntropyLossIndices()
{
  std::cout<<"Testing Categorical Cross Entropy Loss (Class Indices)... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_CategoricalCrossEntropyLoss cce_loss(1e-7f);
    
    // Create predictions: 2 samples, 3 classes
    std::vector<uint32_t> pred_shape={2,3};
    CAIF_Tensor predictions(framework,pred_shape,CAIF_DataType_e::Float32);
    
    // Create targets as class indices: 2 samples
    std::vector<uint32_t> target_shape={2};
    CAIF_Tensor targets(framework,target_shape,CAIF_DataType_e::UInt32);
    
    // Set prediction data (softmax-like probabilities)
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    
    // Sample 1: [0.7, 0.2, 0.1] - true class 0
    pred_data[0]=0.7f;
    pred_data[1]=0.2f;
    pred_data[2]=0.1f;
    
    // Sample 2: [0.1, 0.8, 0.1] - true class 1  
    pred_data[3]=0.1f;
    pred_data[4]=0.8f;
    pred_data[5]=0.1f;
    
    // Set target class indices
    auto target_data_result=targets.MutableData<uint32_t>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    uint32_t *target_data=target_data_result;
    target_data[0]=0;  // Sample 1: class 0
    target_data[1]=1;  // Sample 2: class 1
    
    // Compute loss
    CAIF_Tensor loss=cce_loss.ComputeLoss(predictions,targets);
    auto loss_data_result=loss.ConstData<float>();
    if(loss_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get loss data\n";
      return false;
    }
    
    const float *loss_data=loss_data_result;
    
    // Expected loss for sample 1: -log(0.7) ≈ 0.357
    // Expected loss for sample 2: -log(0.8) ≈ 0.223
    float expected_loss_1=-std::log(0.7f);
    float expected_loss_2=-std::log(0.8f);
    float tolerance=1e-3f;
    
    if(std::abs(loss_data[0]-expected_loss_1)>tolerance ||
       std::abs(loss_data[1]-expected_loss_2)>tolerance)
    {
      std::cout<<"FAILED - Expected losses ["
               <<expected_loss_1<<", "
               <<expected_loss_2<<"] but got ["
               <<loss_data[0]<<", "
               <<loss_data[1]<<"]\n";
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
 * @brief Test Categorical Cross Entropy loss with one-hot encoded targets
 */
bool TestCategoricalCrossEntropyLossOneHot()
{
  std::cout<<"Testing Categorical Cross Entropy Loss (One-Hot)... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_CategoricalCrossEntropyLoss cce_loss(1e-7f);
    
    // Create 2x3 tensors (2 samples, 3 classes) - both predictions and targets
    std::vector<uint32_t> shape={2,3};
    CAIF_Tensor predictions(framework,shape,CAIF_DataType_e::Float32);
    CAIF_Tensor targets(framework,shape,CAIF_DataType_e::Float32);
    
    // Set prediction data (softmax-like probabilities)
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    
    // Sample 1: [0.6, 0.3, 0.1]
    pred_data[0]=0.6f;
    pred_data[1]=0.3f;
    pred_data[2]=0.1f;
    
    // Sample 2: [0.2, 0.7, 0.1]
    pred_data[3]=0.2f;
    pred_data[4]=0.7f;
    pred_data[5]=0.1f;
    
    // Set target data (one-hot encoded)
    auto target_data_result=targets.MutableData<float>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    float *target_data=target_data_result;
    
    // Sample 1: [1, 0, 0] - true class is 0
    target_data[0]=1.0f;
    target_data[1]=0.0f;
    target_data[2]=0.0f;
    
    // Sample 2: [0, 1, 0] - true class is 1
    target_data[3]=0.0f;
    target_data[4]=1.0f;
    target_data[5]=0.0f;
    
    // Compute loss
    CAIF_Tensor loss=cce_loss.ComputeLoss(predictions,targets);
    auto loss_data_result=loss.ConstData<float>();
    if(loss_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get loss data\n";
      return false;
    }
    
    const float *loss_data=loss_data_result;
    
    // Expected loss for sample 1: -log(0.6) ≈ 0.511
    // Expected loss for sample 2: -log(0.7) ≈ 0.357
    float expected_loss_1=-std::log(0.6f);
    float expected_loss_2=-std::log(0.7f);
    float tolerance=1e-3f;
    
    if(std::abs(loss_data[0]-expected_loss_1)>tolerance ||
       std::abs(loss_data[1]-expected_loss_2)>tolerance)
    {
      std::cout<<"FAILED - Expected losses ["
               <<expected_loss_1<<", "
               <<expected_loss_2<<"] but got ["
               <<loss_data[0]<<", "
               <<loss_data[1]<<"]\n";
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
 * @brief Test Categorical Cross Entropy gradient with class indices
 */
bool TestCategoricalCrossEntropyGradientIndices()
{
  std::cout<<"Testing Categorical Cross Entropy Gradient (Class Indices)... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_CategoricalCrossEntropyLoss cce_loss(1e-7f);
    
    // Create predictions: 1 sample, 3 classes for simple gradient verification
    std::vector<uint32_t> pred_shape={1,3};
    CAIF_Tensor predictions(framework,pred_shape,CAIF_DataType_e::Float32);
    
    // Create targets as class indices: 1 sample
    std::vector<uint32_t> target_shape={1};
    CAIF_Tensor targets(framework,target_shape,CAIF_DataType_e::UInt32);
    
    // Set prediction data
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    pred_data[0]=0.5f;  // True class
    pred_data[1]=0.3f;
    pred_data[2]=0.2f;
    
    // Set target class index (class 0)
    auto target_data_result=targets.MutableData<uint32_t>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    uint32_t *target_data=target_data_result;
    target_data[0]=0;  // True class is 0
    
    // Compute gradient
    CAIF_Tensor gradient=cce_loss.ComputeGradient(predictions,targets);
    auto grad_data_result=gradient.ConstData<float>();
    if(grad_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get gradient data\n";
      return false;
    }
    
    const float *grad_data=grad_data_result;
    
    // Expected gradient using standard Softmax+CCE simplification: (predictions - targets)/batch
    // For class 0 (true): 0.5 - 1.0 = -0.5
    // For class 1: 0.3 - 0.0 = 0.3
    // For class 2: 0.2 - 0.0 = 0.2
    float expected_grad_0=-0.5f;
    float expected_grad_1=0.3f;
    float expected_grad_2=0.2f;
    float tolerance=1e-6f;
    
    if(std::abs(grad_data[0]-expected_grad_0)>tolerance||
       std::abs(grad_data[1]-expected_grad_1)>tolerance||
       std::abs(grad_data[2]-expected_grad_2)>tolerance)
    {
      std::cout<<"FAILED - Expected gradient ["
               <<expected_grad_0<<", "
               <<expected_grad_1<<", "
               <<expected_grad_2<<"] but got ["
               <<grad_data[0]<<", "
               <<grad_data[1]<<", "
               <<grad_data[2]<<"]\n";
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
 * @brief Test Categorical Cross Entropy error handling
 */
bool TestCategoricalCrossEntropyErrorHandling()
{
  std::cout<<"Testing Categorical Cross Entropy Error Handling... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_CategoricalCrossEntropyLoss cce_loss;
    
    // Test invalid class index (out of range)
    std::vector<uint32_t> pred_shape={1,3};
    CAIF_Tensor predictions(framework,pred_shape,CAIF_DataType_e::Float32);
    
    std::vector<uint32_t> target_shape={1};
    CAIF_Tensor targets(framework,target_shape,CAIF_DataType_e::UInt32);
    
    // Set valid prediction data
    auto pred_data_result=predictions.MutableData<float>();
    if(pred_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get prediction data\n";
      return false;
    }
    float *pred_data=pred_data_result;
    pred_data[0]=0.5f;
    pred_data[1]=0.3f;
    pred_data[2]=0.2f;
    
    // Set invalid class index (>=num_classes)
    auto target_data_result=targets.MutableData<uint32_t>();
    if(target_data_result==nullptr)
    {
      std::cout<<"FAILED - Could not get target data\n";
      return false;
    }
    uint32_t *target_data=target_data_result;
    target_data[0]=5;  // Invalid: only classes 0,1,2 exist
    
    // Should fail with out of range error
    std::cout<<"  [CCE ErrorHandling] Calling ComputeLoss with invalid class index...\n";
    try
    {
      cce_loss.ComputeLoss(predictions,targets);
      std::cout<<"FAILED - Should have failed with out of range class index\n";
      return false;
    }
    catch(const instance::CAIF_Exception &e)
    {
      std::cout<<"  [CCE ErrorHandling] Caught expected CAIF_Exception: "<<e<<"\n";
    }
    catch(...)
    {
      std::cout<<"  [CCE ErrorHandling] Caught non-CAIF exception (expected failure)\n";
    }
    
    // Test invalid target shape for class indices
    std::vector<uint32_t> bad_target_shape={2,2};  // Should be [batch_size] for indices
    CAIF_Tensor bad_targets(framework,bad_target_shape,CAIF_DataType_e::UInt32);
    
    std::cout<<"  [CCE ErrorHandling] Calling ComputeLoss with invalid target shape...\n";
    try
    {
      cce_loss.ComputeLoss(predictions,bad_targets);
      std::cout<<"FAILED - Should have failed with invalid target shape\n";
      return false;
    }
    catch(const instance::CAIF_Exception &e)
    {
      std::cout<<"  [CCE ErrorHandling] Caught expected CAIF_Exception: "<<e<<"\n";
    }
    catch(...)
    {
      std::cout<<"  [CCE ErrorHandling] Caught non-CAIF exception (expected failure)\n";
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
 * @brief Test loss function error handling
 */
bool TestLossFunctionErrorHandling()
{
  std::cout<<"Testing Loss Function Error Handling... ";
  
  try
  {
    CAIF_Framework framework;
    CAIF_MeanSquaredErrorLoss mse_loss;
    
    // Test shape mismatch
    std::vector<uint32_t> shape1={2,3};
    std::vector<uint32_t> shape2={2,4};
    CAIF_Tensor predictions(framework,shape1,CAIF_DataType_e::Float32);
    CAIF_Tensor targets(framework,shape2,CAIF_DataType_e::Float32);
    
    std::cout<<"  [MSE ErrorHandling] Calling ComputeLoss with mismatched shapes...\n";
    try
    {
      mse_loss.ComputeLoss(predictions,targets);
      std::cout<<"FAILED - Should have failed with shape mismatch\n";
      return false;
    }
    catch(const instance::CAIF_Exception &e)
    {
      std::cout<<"  [MSE ErrorHandling] Caught expected CAIF_Exception: "<<e<<"\n";
    }
    catch(...)
    {
      std::cout<<"  [MSE ErrorHandling] Caught non-CAIF exception (expected failure)\n";
    }
    
    // Test gradient with shape mismatch
    std::cout<<"  [MSE ErrorHandling] Calling ComputeGradient with mismatched shapes...\n";
    try
    {
      mse_loss.ComputeGradient(predictions,targets);
      std::cout<<"FAILED - Should have failed with gradient shape mismatch\n";
      return false;
    }
    catch(const instance::CAIF_Exception &e)
    {
      std::cout<<"  [MSE ErrorHandling] Caught expected CAIF_Exception: "<<e<<"\n";
    }
    catch(...)
    {
      std::cout<<"  [MSE ErrorHandling] Caught non-CAIF exception (expected failure)\n";
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
 * @brief Test Binary Cross Entropy loss function
 */
bool TestBinaryCrossEntropyLoss()
{
  std::cout<<"Testing Binary Cross Entropy Loss... ";
  try
  {
    CAIF_Framework framework;
    CAIF_BinaryCrossEntropyLoss bce;
    // 4 samples, single output
    CAIF_Tensor preds(framework,{4,1},CAIF_DataType_e::Float32);
    CAIF_Tensor targs(framework,{4,1},CAIF_DataType_e::Float32);
    auto pp=preds.MutableData<float>();
    auto tp=targs.MutableData<float>();
    if(pp==nullptr||tp==nullptr)
    {
      std::cout<<"FAILED - data ptr\n"; 
      return false;
    }
    float *p=pp; float *t=tp;
    p[0]=0.9f; t[0]=1.0f;
    p[1]=0.8f; t[1]=1.0f;
    p[2]=0.2f; t[2]=0.0f;
    p[3]=0.4f; t[3]=0.0f;
    auto loss=bce.ComputeLoss(preds,targs);
    auto ld=loss.ConstData<float>();
    if(ld==nullptr)
    {
      std::cout<<"FAILED - loss data\n"; 
      return false;
    }
    if(!std::isfinite(ld[0])||ld[0]>2.0f)
    {
      std::cout<<"FAILED - unreasonable loss\n"; 
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
 * @brief Test Binary Cross Entropy gradient
 */
bool TestBinaryCrossEntropyGradient()
{
  std::cout<<"Testing Binary Cross Entropy Gradient... ";
  try
  {
    CAIF_Framework framework;
    CAIF_BinaryCrossEntropyLoss bce;
    CAIF_Tensor preds(framework,{1,1},CAIF_DataType_e::Float32);
    CAIF_Tensor targs(framework,{1,1},CAIF_DataType_e::Float32);
    *preds.MutableData<float>()=0.8f;
    *targs.MutableData<float>()=1.0f;
    auto grad=bce.ComputeGradient(preds,targs);
    auto gd=grad.ConstData<float>();
    if(gd==nullptr)
    {
      std::cout<<"FAILED - gradient data\n"; 
      return false;
    }
    if(!std::isfinite(gd[0])||gd[0]>=0.0f)
    {
      std::cout<<"FAILED - unexpected gradient\n"; 
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
 * @brief Test BCE-with-Logits loss stability on large batches
 */
bool TestBinaryCrossEntropyWithLogitsLargeBatch()
{
  std::cout<<"Testing BCE-with-Logits Loss (large batch, stability)... ";
  try
  {
    CAIF_Framework framework;
    CAIF_BinaryCrossEntropyWithLogitsLoss bce_logits;
    const uint32_t samples=2048;
    CAIF_Tensor logits(framework,{samples,1},CAIF_DataType_e::Float32);
    CAIF_Tensor labels(framework,{samples,1},CAIF_DataType_e::Float32);

    auto lp=logits.MutableData<float>();
    auto tp=labels.MutableData<float>();
    if(lp==nullptr||tp==nullptr)
    {
      std::cout<<"FAILED - data pointers\n";
      return false;
    }
    float *z=lp;
    float *y=tp;

    // Deterministic pseudo-random for reproducibility; logits in [-20,20]
    uint32_t seed=123456789;
    for(uint32_t i=0;i<samples;++i)
    {
      seed=(1103515245*seed+12345)&0x7fffffff;
      const float u=static_cast<float>(seed)/static_cast<float>(0x7fffffff);
      z[i]=-20.0f+40.0f*u;
      // Alternate labels to avoid class collapse
      if((i%2)==0)
      {
        y[i]=1.0f;
      }
      else
      {
        y[i]=0.0f;
      }
    }

    auto loss=bce_logits.ComputeLoss(logits,labels);
    auto ld=loss.ConstData<float>();
    if(ld==nullptr)
    {
      std::cout<<"FAILED - loss data\n";
      return false;
    }
    const float v=ld[0];
    if(std::isfinite(v)==false||v<=0.0f||v>50.0f)
    {
      std::cout<<"FAILED - loss out of range: "<<v<<"\n";
      return false;
    }

    auto grad=bce_logits.ComputeGradient(logits,labels);
    auto gd=grad.ConstData<float>();
    if(gd==nullptr)
    {
      std::cout<<"FAILED - gradient data\n";
      return false;
    }
    // Ensure non-zero gradient magnitude across batch
    double l2=0.0;
    for(uint32_t i=0;i<samples;++i)
    {
      const float g=gd[i];
      l2+=static_cast<double>(g)*static_cast<double>(g);
    }
    if(l2<=0.0)
    {
      std::cout<<"FAILED - zero gradients\n";
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

static bool TestBCEWithLogitsSanity()
{
  std::cout<<"Testing BCE-with-Logits sanity... ";
  try
  {
    CAIF_Framework framework;
    CAIF_BinaryCrossEntropyWithLogitsLoss loss;
    CAIF_Tensor z(framework,{4,1},CAIF_DataType_e::Float32);
    CAIF_Tensor y(framework,{4,1},CAIF_DataType_e::Float32);
    auto zp=z.MutableData<float>();
    auto yp=y.MutableData<float>();
    if(zp==nullptr||yp==nullptr)
    {
      std::cout<<"FAILED - alloc\n";
      return false;
    }
    zp[0]=-2.0f;
    zp[1]=-1.0f;
    zp[2]=1.0f;
    zp[3]=2.0f;
    yp[0]=0.0f;
    yp[1]=0.0f;
    yp[2]=1.0f;
    yp[3]=1.0f;

    auto L=loss.ComputeLoss(z,y);
    if(L.Shape()!=std::vector<uint32_t>({1}))
    {
      std::cout<<"FAILED - loss shape\n";
      return false;
    }

    auto G=loss.ComputeGradient(z,y);
    if(G.Shape()!=std::vector<uint32_t>({4,1}))
    {
      std::cout<<"FAILED - grad shape\n";
      return false;
    }

    auto gp=G.ConstData<float>();
    if(gp==nullptr)
    {
      std::cout<<"FAILED - grad ptr\n";
      return false;
    }
    const float g_pos=gp[3]; // z=2,y=1
    const float g_neg=gp[0]; // z=-2,y=0
    if(!((g_pos<0.0f)==true&&((g_neg>0.0f)==true)))
    {
      std::cout<<"FAILED - gradient signs\n";
      return false;
    }
    std::cout<<"PASSED\n";
    return true;
  }
  catch(const std::exception &e)
  {
    std::cout<<"FAILED - ex: "<<e.what()<<"\n";
    return false;
  }
}

/**
 * @brief Main test function
 */
int main()
{
  std::cout<<"=== CAIF Loss Functions Test Suite ===\n\n";
  
  bool all_tests_passed=true;
  
  all_tests_passed&=TestBCEWithLogitsSanity();
  all_tests_passed&=TestMSELossSimple();
  all_tests_passed&=TestMSEGradient();
  all_tests_passed&=TestCrossEntropyLoss();
  all_tests_passed&=TestCrossEntropyGradient();
  all_tests_passed&=TestCategoricalCrossEntropyLossIndices();
  all_tests_passed&=TestCategoricalCrossEntropyLossOneHot();
  all_tests_passed&=TestCategoricalCrossEntropyGradientIndices();
  all_tests_passed&=TestCategoricalCrossEntropyErrorHandling();
  all_tests_passed&=TestLossFunctionErrorHandling();
  all_tests_passed&=TestBinaryCrossEntropyLoss();
  all_tests_passed&=TestBinaryCrossEntropyGradient();
  all_tests_passed&=TestBinaryCrossEntropyWithLogitsLargeBatch();
  
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
