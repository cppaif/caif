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
 * @file test_shapes.cpp
 * @brief Test tensor shapes and operations
 */

#include "caif_tensor.h"
#include "caif_framework.h"
#include <iostream>
#include <vector>

int main()
{
  std::cout<<"Testing tensor shapes and operations...\n";
  
  try
  {
    instance::CAIF_Framework framework;
    
    // Test case 1: Matrix multiplication
    std::cout<<"Test 1: Matrix multiplication\n";
    instance::CAIF_Tensor input(framework,{1,784},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    instance::CAIF_Tensor weights(framework,{784,128},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    
    std::cout<<"Input shape: [";
    auto input_shape=input.Shape();
    for(size_t i=0;i<input_shape.size();++i)
    {
      std::cout<<input_shape[i];
      if(i<input_shape.size()-1)std::cout<<", ";
    }
    std::cout<<"]\n";
    
    std::cout<<"Weights shape: [";
    auto weights_shape=weights.Shape();
    for(size_t i=0;i<weights_shape.size();++i)
    {
      std::cout<<weights_shape[i];
      if(i<weights_shape.size()-1)std::cout<<", ";
    }
    std::cout<<"]\n";
    
    auto matmul_result=input.MatMul(weights);
    
    std::cout<<"MatMul result shape: [";
    auto result_shape=matmul_result.Shape();
    for(size_t i=0;i<result_shape.size();++i)
    {
      std::cout<<result_shape[i];
      if(i<result_shape.size()-1)std::cout<<", ";
    }
    std::cout<<"]\n";
    
    // Test case 2: Bias addition
    std::cout<<"\nTest 2: Bias addition\n";
    instance::CAIF_Tensor bias(framework,{128},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    
    std::cout<<"Bias shape: [";
    auto bias_shape=bias.Shape();
    for(size_t i=0;i<bias_shape.size();++i)
    {
      std::cout<<bias_shape[i];
      if(i<bias_shape.size()-1)std::cout<<", ";
    }
    std::cout<<"]\n";
    
    try
    {
      auto add_result=matmul_result.Add(bias);
      std::cout<<"Addition succeeded!\n";
      std::cout<<"Final result shape: [";
      auto final_shape=add_result.Shape();
      for(size_t i=0;i<final_shape.size();++i)
      {
        std::cout<<final_shape[i];
        if(i<final_shape.size()-1)std::cout<<", ";
      }
      std::cout<<"]\n";
    }
    catch(const std::exception &e)
    {
      std::cout<<"Addition failed: "<<e.what()<<"\n";
      
      // Try alternative: expand bias to match result shape
      std::cout<<"Trying to expand bias...\n";
      instance::CAIF_Tensor expanded_bias(framework,{1,128},instance::CAIF_DataType::CAIF_DataType_e::Float32);
      
      try
      {
        auto add_result2=matmul_result.Add(expanded_bias);
        std::cout<<"Expanded addition succeeded!\n";
        std::cout<<"Final result shape: [";
        auto final_shape=add_result2.Shape();
        for(size_t i=0;i<final_shape.size();++i)
        {
          std::cout<<final_shape[i];
          if(i<final_shape.size()-1)std::cout<<", ";
        }
        std::cout<<"]\n";
      }
      catch(const std::exception &e2)
      {
        std::cout<<"Expanded addition also failed: "<<e2.what()<<"\n";
        return 1;
      }
    }
    
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    return 1;
  }
} 