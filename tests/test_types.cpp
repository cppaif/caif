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

#include "caif_tensor.h"
#include "caif_framework.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace instance;
using CAIF_DataType_e = CAIF_DataType::CAIF_DataType_e;

// Global framework for tests
static CAIF_Framework g_test_framework;

void TestDataTypes()
{
  std::cout<<"Testing data type handling...\n";
  
  std::vector<uint32_t> shape={2,2};
  
  // Test Float32
  std::cout<<"Testing Float32...\n";
  CAIF_Tensor float32_tensor(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> float_data={1.0f,2.0f,3.0f,4.0f};
  float32_tensor.SetData(float_data.data(),float_data.size()*sizeof(float));
  
  const float *float32_data=float32_tensor.ConstData<float>();
  assert((float32_data!=nullptr));
  for(size_t i=0;i<float_data.size();++i)
  {
    assert((std::abs(float32_data[i]-float_data[i])<1e-6));
  }
  
  // Test Int32
  std::cout<<"Testing Int32...\n";
  CAIF_Tensor int32_tensor(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::Int32);
  std::vector<int32_t> int_data={1,2,3,4};
  int32_tensor.SetData(int_data.data(),int_data.size()*sizeof(int32_t));
  
  const int32_t *int32_data=int32_tensor.ConstData<int32_t>();
  assert((int32_data!=nullptr));
  for(size_t i=0;i<int_data.size();++i)
  {
    assert((int32_data[i]==int_data[i]));
  }
  
  // Test UInt8
  std::cout<<"Testing UInt8...\n";
  CAIF_Tensor uint8_tensor(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::UInt8);
  std::vector<uint8_t> uint8_data={1,2,3,4};
  uint8_tensor.SetData(uint8_data.data(),uint8_data.size()*sizeof(uint8_t));
  
  const uint8_t *uint8_data_result=uint8_tensor.ConstData<uint8_t>();
  assert((uint8_data_result!=nullptr));
  for(size_t i=0;i<uint8_data.size();++i)
  {
    assert((uint8_data_result[i]==uint8_data[i]));
  }
}

void TestTypeValidation()
{
  std::cout<<"Testing type validation...\n";
  
  std::vector<uint32_t> shape={2,2};
  CAIF_Tensor tensor(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  
  // Test invalid type access
  std::cout<<"Testing invalid type access...\n";
  const int32_t *invalid_data=tensor.ConstData<int32_t>();
  assert((invalid_data==nullptr));
  
  // Test invalid shape
  std::cout<<"Testing invalid shape...\n";
  try
  {
    std::vector<uint32_t> invalid_shape;  // Empty shape
    CAIF_Tensor invalid_tensor(g_test_framework,invalid_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    assert((false==true));  // Should not reach here
  }
  catch(const std::invalid_argument &e)
  {
    std::cout<<"Correctly caught invalid shape: "<<e.what()<<std::endl;
  }
  
  // Test zero dimension
  try
  {
    std::vector<uint32_t> zero_shape={0,2};  // Zero dimension
    CAIF_Tensor zero_tensor(g_test_framework,zero_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    assert((false==true));  // Should not reach here
  }
  catch(const std::invalid_argument &e)
  {
    std::cout<<"Correctly caught zero dimension: "<<e.what()<<std::endl;
  }
  
  // Test too many dimensions
  try
  {
    std::vector<uint32_t> large_shape(g_caif_max_tensor_dimensions+1,2);
    CAIF_Tensor large_tensor(g_test_framework,large_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    assert((false==true));  // Should not reach here
  }
  catch(const std::invalid_argument &e)
  {
    std::cout<<"Correctly caught too many dimensions: "<<e.what()<<std::endl;
  }
}

void TestSpanAccess()
{
  std::cout<<"Testing span access...\n";
  
  std::vector<uint32_t> shape={2,2};
  CAIF_Tensor tensor(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> data={1.0f,2.0f,3.0f,4.0f};
  tensor.SetData(data.data(),data.size()*sizeof(float));
  
  // Test mutable span
  auto mutable_span=tensor.AsSpan<float>();
  for(size_t i=0;i<data.size();++i)
  {
    assert((std::abs(mutable_span[i]-data[i])<1e-6));
  }
  
  // Test const span
  const auto const_span=tensor.AsSpan<float>();
  for(size_t i=0;i<data.size();++i)
  {
    assert((std::abs(const_span[i]-data[i])<1e-6));
  }
}

int main()
{
  try
  {
    TestDataTypes();
    TestTypeValidation();
    TestSpanAccess();
    
    std::cout<<"All type tests passed!\n";
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cerr<<"Error: "<<e.what()<<std::endl;
    return 1;
  }
} 