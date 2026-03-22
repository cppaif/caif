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

#include "caif_reshape_layer.h"
#include "caif_tensor.h"
#include "caif_framework.h"
#include "ise_lib/ise_out.h"
#include <vector>

using namespace instance;

// Global framework for tests
static CAIF_Framework g_test_framework;

bool TestReshapeLayer()
{
  try
  {
    ISE_Out::Log()<<"Testing reshape layer..."<<std::endl;

    // Test 1: Reshape 2D to 4D
    {
      ISE_Out::Log()<<"  Test 1: Reshape [32, 64] -> [32, 8, 8, 1]"<<std::endl;
      
      // Create input tensor [batch_size=32, features=64]
      std::vector<float> input_data(32*64, 1.0f);  // Fill with 1s for easy verification
      // Create tensor with shape
      CAIF_Tensor input(g_test_framework,
                       {32,64},
                       CAIF_DataType::CAIF_DataType_e::Float32);
      input.SetData(input_data.data(), input_data.size()*sizeof(float));  // Set data
      
      // Create reshape layer with target shape [8, 8, 1] (batch dimension handled automatically)
      CAIF_ReshapeLayer reshape_layer(g_test_framework,{8, 8, 1});
      
      // Initialize layer with seed=0
      reshape_layer.Initialize({32, 64}, 0);
      
      // Forward pass
      CAIF_Tensor output=reshape_layer.Forward(input, false);
      
      // Verify output shape
      const auto &output_shape=output.Shape();
      if(output_shape.size()!=4 || 
         output_shape[0]!=32 || 
         output_shape[1]!=8 || 
         output_shape[2]!=8 || 
         output_shape[3]!=1)
      {
        ISE_Out::Log()<<"❌ FAILED - Incorrect output shape: ["
                      <<output_shape[0]<<", "
                      <<output_shape[1]<<", "
                      <<output_shape[2]<<", "
                      <<output_shape[3]<<"]"<<std::endl;
        return false;
      }
      
      // Verify output data
      const float *output_data=output.ConstData<float>();
      if(output_data==nullptr)
      {
        ISE_Out::Log()<<"❌ FAILED - Could not get output data"<<std::endl;
        return false;
      }
      for(size_t i=0; i<output.NumElements(); ++i)
      {
        if(output_data[i]!=1.0f)
        {
          ISE_Out::Log()<<"❌ FAILED - Output data mismatch at index "<<i
                        <<": expected 1.0, got "<<output_data[i]<<std::endl;
          return false;
        }
      }
      
      // Test backward pass
      std::vector<float> gradient_data(32*8*8*1, 1.0f);  // Fill with 1s
      // Create gradient tensor
      CAIF_Tensor gradient(g_test_framework,
                          {32,8,8,1},
                          CAIF_DataType::CAIF_DataType_e::Float32);
      gradient.SetData(gradient_data.data(), gradient_data.size()*sizeof(float));
      
      CAIF_Tensor gradient_output=reshape_layer.Backward(gradient);
      
      // Verify gradient output shape
      const auto &gradient_output_shape=gradient_output.Shape();
      if(gradient_output_shape.size()!=2 || 
         gradient_output_shape[0]!=32 || 
         gradient_output_shape[1]!=64)
      {
        ISE_Out::Log()<<"❌ FAILED - Incorrect gradient output shape: ["
                      <<gradient_output_shape[0]<<", "
                      <<gradient_output_shape[1]<<"]"<<std::endl;
        return false;
      }
      
      // Verify gradient output data
      const float *gradient_output_data=gradient_output.ConstData<float>();
      if(gradient_output_data==nullptr)
      {
        ISE_Out::Log()<<"❌ FAILED - Could not get gradient output data"<<std::endl;
        return false;
      }
      for(size_t i=0; i<gradient_output.NumElements(); ++i)
      {
        if(gradient_output_data[i]!=1.0f)
        {
          ISE_Out::Log()<<"❌ FAILED - Gradient output data mismatch at index "<<i
                        <<": expected 1.0, got "<<gradient_output_data[i]<<std::endl;
          return false;
        }
      }
      
      ISE_Out::Log()<<"✅ PASSED - Test 1: 2D to 4D reshape"<<std::endl;
    }
    
    return true;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Log()<<"❌ FAILED - Unexpected exception: "<<e.what()<<std::endl;
    return false;
  }
}

int main()
{
  bool success=true;
  
  success&=TestReshapeLayer();
  
  return success ? 0 : 1;
} 