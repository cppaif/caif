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
 * @file test_learning_process.cpp
 * @brief Educational example showing how neural networks learn through weight updates
 * 
 * This example demonstrates the iterative learning process using the Pythagorean theorem
 * Shows how weights change during training to minimize prediction error
 * 
 * We'll trace:
 * 1. Initial random weights and poor predictions
 * 2. Forward pass calculations
 * 3. Error computation
 * 4. Backward pass (gradient calculation)
 * 5. Weight updates
 * 6. Improved predictions over iterations
 * 
 * @author AIF Development Team
 * @version 1.0 
 * @date 2024
 */

#include "caif_neural_network.h"
#include "caif_tensor.h"
#include "caif_dense_layer.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// Simple neural network class for manual training demonstration
class SimpleNetwork
{
  private:
    instance::CAIF_Framework _framework;
    instance::CAIF_DenseLayer _layer1;
    instance::CAIF_DenseLayer _output_layer;
    bool _initialized;
    
  public:
    SimpleNetwork():_framework(),
                    _layer1(_framework,4,instance::CAIF_ActivationType_e::ReLU),
                    _output_layer(_framework,1,instance::CAIF_ActivationType_e::ReLU),
                    _initialized(false){}
    
    bool Initialize()
    {
      std::vector<uint32_t> input_shape={1,2};
      
      _layer1.Initialize(input_shape,42);  // Fixed seed for reproducibility
      
      auto layer1_output_shape=_layer1.CalculateOutputShape(input_shape);
      _output_layer.Initialize(layer1_output_shape,43);
      
      _initialized=true;
      return true;
    }
    
    std::pair<instance::CAIF_Tensor,instance::CAIF_Tensor> Forward(const instance::CAIF_Tensor &input)
    {
      auto layer1_output=_layer1.Forward(input,true);  // true = training mode
      auto final_output=_output_layer.Forward(layer1_output,true);
      return {layer1_output,final_output};
    }
    
    void Backward(const instance::CAIF_Tensor &target,const instance::CAIF_Tensor &prediction)
    {
      // Simple MSE loss gradient: 2 * (prediction - target)
      auto pred_data=prediction.ConstData<float>();
      auto target_data=target.ConstData<float>();
      if(pred_data==nullptr||target_data==nullptr)
      {
        throw std::runtime_error(
          "Failed to get prediction or target data");
      }
      
      // Create gradient tensor
      instance::CAIF_Tensor output_gradient(_framework,{1,1},instance::CAIF_DataType::CAIF_DataType_e::Float32);
      auto grad_data_result=output_gradient.MutableData<float>();
      if(grad_data_result==nullptr){throw std::runtime_error("Failed to get gradient data");}
      float *grad_data=grad_data_result;
      grad_data[0]=2.0f*(pred_data[0]-target_data[0]);  // MSE gradient
      
      // Backward through output layer
      auto layer2_grad_result=_output_layer.Backward(output_gradient);
      // Backward through first layer
      _layer1.Backward(layer2_grad_result);
    }
    
    void UpdateWeights(float learning_rate)
    {
      // Get current parameters and gradients
      auto layer1_params=_layer1.Parameters();
      auto layer1_grads=_layer1.ParameterGradients();
      auto output_params=_output_layer.Parameters();
      auto output_grads=_output_layer.ParameterGradients();
      
      // Update layer 1 weights
      UpdateLayerWeights(layer1_params,layer1_grads,learning_rate);
      _layer1.UpdateParameters(layer1_params);
      
      // Update output layer weights  
      UpdateLayerWeights(output_params,output_grads,learning_rate);
      _output_layer.UpdateParameters(output_params);
    }
    
  private:
    void UpdateLayerWeights(std::vector<instance::CAIF_Tensor> &params,
                           const std::vector<instance::CAIF_Tensor> &grads,
                           float learning_rate)
    {
      for(size_t i=0;i<params.size()&&i<grads.size();++i)
      {
        auto param_data_result=params[i].MutableData<float>();
        auto grad_data_result=grads[i].ConstData<float>();
        
        if(param_data_result!=nullptr&&grad_data_result!=nullptr)
        {
          float *param_data=param_data_result;
          const float *grad_data=grad_data_result;
          
          auto shape=params[i].Shape();
          uint32_t total_elements=1;
          for(uint32_t dim:shape)
          {
            total_elements*=dim;
          }
          
          // Gradient descent: param = param - learning_rate * gradient
          for(uint32_t j=0;j<total_elements;++j)
          {
            param_data[j]-=learning_rate*grad_data[j];
          }
        }
      }
    }
};

void PrintTensorValues(const instance::CAIF_Tensor &tensor,const std::string &name)
{
  std::cout<<"  "<<name<<": ";
  
  auto data_result=tensor.ConstData<float>();
  if(data_result!=nullptr)
  {
    const float *data=data_result;
    auto shape=tensor.Shape();
    uint32_t total_elements=1;
    for(uint32_t dim:shape)
    {
      total_elements*=dim;
    }
    
    std::cout<<"[";
    for(uint32_t i=0;i<std::min(total_elements,8u);++i)
    {
      std::cout<<std::fixed<<std::setprecision(4)<<data[i];
      if(i<std::min(total_elements,8u)-1)std::cout<<", ";
    }
    if(total_elements>8)std::cout<<", ...";
    std::cout<<"]";
  }
  std::cout<<"\n";
}

void PrintWeights(const std::vector<instance::CAIF_Tensor> &params,const std::string &layer_name)
{
  std::cout<<"  "<<layer_name<<" Weights:\n";
  if(params.size()>=1)
  {
    PrintTensorValues(params[0],"    W");
  }
  if(params.size()>=2)
  {
    PrintTensorValues(params[1],"    b");
  }
}

float CalculateLoss(const instance::CAIF_Tensor &prediction,const instance::CAIF_Tensor &target)
{
  auto pred_data=prediction.ConstData<float>();
  auto target_data=target.ConstData<float>();
  
  if(pred_data!=nullptr&&target_data!=nullptr)
  {
    float diff=pred_data[0]-target_data[0];
    return diff*diff;  // MSE loss
  }
  return 0.0f;
}

int main()
{
  std::cout<<"==================================================\n";
  std::cout<<"   NEURAL NETWORK LEARNING PROCESS VISUALIZATION\n";
  std::cout<<"      How Weights Change to Minimize Error\n";
  std::cout<<"==================================================\n\n";
  
  try
  {
    std::cout<<"LEARNING OBJECTIVE:\n";
    std::cout<<"===================\n";
    std::cout<<"Train a neural network to learn the Pythagorean theorem: c = sqrt(a² + b²)\n";
    std::cout<<"Architecture: Input(2) → Dense(4,ReLU) → Dense(1,ReLU) → Output\n\n";
    
    // Create framework for tensor operations
    instance::CAIF_Framework framework;
    
    // Create and initialize network
    SimpleNetwork network;
    if(!network.Initialize())
    {
      std::cout<<"Failed to initialize network\n";
      return 1;
    }
    
    // Create training example: 3-4-5 right triangle
    instance::CAIF_Tensor input(framework,{1,2},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    instance::CAIF_Tensor target(framework,{1,1},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    
    auto input_data_result=input.MutableData<float>();
    auto target_data_result=target.MutableData<float>();
    
    if(input_data_result==nullptr||target_data_result==nullptr)
    {
      std::cout<<"Failed to get tensor data\n";
      return 1;
    }
    
    float *input_data=input_data_result;
    float *target_data=target_data_result;
    
    input_data[0]=3.0f;   // a = 3
    input_data[1]=4.0f;   // b = 4
    target_data[0]=5.0f;  // c = 5 (since 3² + 4² = 5²)
    
    std::cout<<"TRAINING DATA:\n";
    std::cout<<"==============\n";
    std::cout<<"Input: a="<<input_data[0]<<", b="<<input_data[1]<<"\n";
    std::cout<<"Target: c="<<target_data[0]<<"\n";
    std::cout<<"Mathematical verification: sqrt("<<input_data[0]<<"² + "<<input_data[1]<<"²) = sqrt("
             <<(input_data[0]*input_data[0])<<" + "<<(input_data[1]*input_data[1])<<") = sqrt("
             <<(input_data[0]*input_data[0]+input_data[1]*input_data[1])<<") = "<<target_data[0]<<"\n\n";
    
    float learning_rate=0.01f;
    int num_iterations=10;
    
    std::cout<<"LEARNING PROCESS (Learning Rate = "<<learning_rate<<"):\n";
    std::cout<<"=====================================================\n\n";
    
    for(int iteration=0;iteration<num_iterations;++iteration)
    {
      std::cout<<"ITERATION "<<iteration<<"\n";
      std::cout<<"------------\n";
      
      // Forward pass
      auto [hidden_output,prediction]=network.Forward(input);
      float loss=CalculateLoss(prediction,target);
      
      auto pred_data=prediction.ConstData<float>();
      float pred_value=pred_data!=nullptr?pred_data[0]:0.0f;
      float error=pred_value-target_data[0];
      
      std::cout<<"1. Forward Pass:\n";
      PrintTensorValues(input,"  Input");
      PrintTensorValues(hidden_output,"  Hidden");
      PrintTensorValues(prediction,"  Output");
      std::cout<<"  Prediction: "<<std::fixed<<std::setprecision(4)<<pred_value<<"\n";
      std::cout<<"  Target: "<<target_data[0]<<"\n";
      std::cout<<"  Error: "<<error<<" (prediction - target)\n";
      std::cout<<"  Loss (MSE): "<<loss<<"\n\n";
      
      if(iteration==0)
      {
        std::cout<<"2. Initial Weights (random):\n";
        // Note: We can't easily access individual layer parameters in this simplified example
        // In a real implementation, you'd show the actual weight matrices here
        std::cout<<"  [Weights are randomly initialized - this is why prediction is poor]\n\n";
      }
      
      // Backward pass
      std::cout<<"3. Backward Pass (Gradient Calculation):\n";
      std::cout<<"  Loss gradient: d(Loss)/d(prediction) = 2 * (prediction - target) = 2 * "
               <<error<<" = "<<(2.0f*error)<<"\n";
      std::cout<<"  This gradient tells us:\n";
      if(error>0)
      {
        std::cout<<"    - Prediction is TOO HIGH, need to DECREASE weights that increase output\n";
      }
      else if(error<0)
      {
        std::cout<<"    - Prediction is TOO LOW, need to INCREASE weights that increase output\n";
      }
      else
      {
        std::cout<<"    - Prediction is PERFECT, no weight changes needed\n";
      }
      
      network.Backward(target,prediction);
      
      std::cout<<"  [Gradients computed for all weights using chain rule]\n\n";
      
      // Weight update
      std::cout<<"4. Weight Update:\n";
      std::cout<<"  Rule: new_weight = old_weight - learning_rate * gradient\n";
      std::cout<<"  Learning rate: "<<learning_rate<<"\n";
      
      network.UpdateWeights(learning_rate);
      
      std::cout<<"  [All weights updated to reduce error]\n\n";
      
      // Show improvement
      if(iteration>0)
      {
        std::cout<<"5. Improvement Analysis:\n";
        static float prev_loss=loss;  // This is a simplified tracking
        std::cout<<"  Loss change: "<<(loss-prev_loss)<<" ";
        if(loss<prev_loss)
        {
          std::cout<<"(IMPROVED! ✅)\n";
        }
        else
        {
          std::cout<<"(got worse)\n";
        }
        prev_loss=loss;
      }
      
      std::cout<<"========================================\n\n";
      
      // Early stopping if we get close enough
      if(std::abs(error)<0.1f)
      {
        std::cout<<"🎯 CONVERGENCE ACHIEVED! Error < 0.1\n\n";
        break;
      }
    }
    
    std::cout<<"LEARNING SUMMARY:\n";
    std::cout<<"=================\n";
    std::cout<<"Key Learning Principles:\n\n";
    
    std::cout<<"1. **Forward Pass**: Data flows through layers, each applying:\n";
    std::cout<<"   output = activation(input × weights + bias)\n\n";
    
    std::cout<<"2. **Error Calculation**: Compare prediction with target:\n";
    std::cout<<"   error = prediction - target\n";
    std::cout<<"   loss = error² (Mean Squared Error)\n\n";
    
    std::cout<<"3. **Backward Pass**: Calculate gradients using chain rule:\n";
    std::cout<<"   - How much does each weight contribute to the error?\n";
    std::cout<<"   - Gradients point in direction of steepest error increase\n\n";
    
    std::cout<<"4. **Weight Update**: Move weights in opposite direction of gradient:\n";
    std::cout<<"   new_weight = old_weight - learning_rate × gradient\n";
    std::cout<<"   - If gradient is positive → decrease weight\n";
    std::cout<<"   - If gradient is negative → increase weight\n\n";
    
    std::cout<<"5. **Iteration**: Repeat until error is minimized\n\n";
    
    std::cout<<"WHY IT WORKS:\n";
    std::cout<<"=============\n";
    std::cout<<"- Network starts with random weights (poor predictions)\n";
    std::cout<<"- Each iteration adjusts weights to reduce error\n";
    std::cout<<"- Over time, weights learn the mathematical relationship\n";
    std::cout<<"- Eventually: network approximates c = sqrt(a² + b²)\n\n";
    
    std::cout<<"✅ Learning process visualization completed!\n";
    std::cout<<"✅ You can now see how neural networks optimize through gradient descent!\n";
    
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception occurred: "<<e.what()<<"\n";
    return 1;
  }
} 