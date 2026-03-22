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
using CAIF_DataType_e=CAIF_DataType::CAIF_DataType_e;

void TestDynamicBatchSize()
{
  std::cout<<"Testing dynamic batch size support...\n";
  
  CAIF_Framework framework;
  
  // Test 1: Create tensor with dynamic batch size
  std::vector<uint32_t> shape={0,224,224,3};  // 0 indicates dynamic batch size
  CAIF_Tensor tensor(framework,shape,CAIF_DataType_e::Float32);
  
  std::cout<<"Created tensor with dynamic batch size: "<<tensor.ToString()<<"\n";
  assert((tensor.IsDynamicBatch()==true));
  assert((tensor.Shape()[0]==0));
  
  // Test 2: Set data with different batch sizes
  std::vector<float> data32(32*224*224*3,1.0f);  // Batch size 32
  std::vector<float> data64(64*224*224*3,1.0f);  // Batch size 64
  
  // Should succeed - setting batch size 32
  tensor.SetBatchData(data32.data(),32);
  assert((tensor.Shape()[0]==32));
  
  // Should succeed - changing to batch size 64
  tensor.SetBatchData(data64.data(),64);
  assert((tensor.Shape()[0]==64));
  
  // Test 3: Operations with dynamic batch sizes
  std::vector<uint32_t> kernel_shape={3,3,3,64};
  CAIF_Tensor kernel(framework,kernel_shape,CAIF_DataType_e::Float32);
  std::vector<float> kernel_data(3*3*3*64,1.0f);
  kernel.SetData(kernel_data.data(),kernel_data.size()*sizeof(float));
  
  // Convolution should work with any batch size
  auto conv_result=tensor.Convolution2D(kernel,1,1);
  assert((conv_result.Shape()[0]==64));  // Should preserve batch size
  
  std::cout<<"Dynamic batch size tests passed!\n";
}

void TestDynamicBatchValidation()
{
  std::cout<<"Testing dynamic batch size validation...\n";
  
  CAIF_Framework framework;
  
  // Test 1: Invalid dynamic dimension
  std::vector<uint32_t> invalid_shape={0,0,224,3};  // Only batch dimension can be dynamic
  try
  {
    CAIF_Tensor invalid_tensor(framework,invalid_shape,CAIF_DataType_e::Float32);
    assert((false==true));  // Should not reach here
  }
  catch(const std::exception &e)
  {
    std::cout<<"Correctly caught invalid dynamic shape: "<<e.what()<<"\n";
  }
  
  // Test 2: Invalid batch data size
  std::vector<uint32_t> shape={0,224,224,3};
  CAIF_Tensor tensor(framework,shape,CAIF_DataType_e::Float32);
  
  std::vector<float> invalid_data(15*224*224*3,1.0f);  // Invalid size
  try
  {
    tensor.SetBatchData(invalid_data.data(),32);
    assert(false);
  }
  catch(const std::exception &)
  {
    /* expected */
  }
  
  std::cout<<"Dynamic batch validation tests passed!\n";
}

void TestDynamicBatchOperations()
{
  std::cout<<"Testing operations with dynamic batch sizes...\n";
  
  CAIF_Framework framework;
  
  // Test 1: Addition with matching dynamic batch sizes
  std::vector<uint32_t> shape={0,10,10,3};
  CAIF_Tensor a(framework,shape,CAIF_DataType_e::Float32);
  CAIF_Tensor b(framework,shape,CAIF_DataType_e::Float32);
  
  std::vector<float> data_a(5*10*10*3,1.0f);  // Batch size 5, all ones
  std::vector<float> data_b(3*10*10*3,2.0f);  // Batch size 3, all twos
  
  a.SetBatchData(data_a.data(),5);
  b.SetBatchData(data_b.data(),3);
  assert((a.Shape()[0]==5));
  assert((b.Shape()[0]==3));
  
  auto add_result=a.Add(b);
  assert((add_result.Shape()[0]==5));  // Max batch size
  
  // Verify addition results - should broadcast b's batch
  auto result_data=add_result.ConstData<float>();
  assert((result_data!=nullptr));
  
  // First batch: 1 + 2 = 3
  assert((std::abs(result_data[0]-3.0f)<1e-6));
  // Fourth batch: 1 + 2 = 3 (b's batch 1 repeats)
  assert((std::abs(result_data[3*10*10*3]-3.0f)<1e-6));
  
  // Test 2: Matrix multiplication with dynamic batch sizes
  std::vector<uint32_t> shape_x={0,10,5};  // [batch,10,5]
  std::vector<uint32_t> shape_w={5,3};     // [5,3]
  
  CAIF_Tensor x(framework,shape_x,CAIF_DataType_e::Float32);
  CAIF_Tensor w(framework,shape_w,CAIF_DataType_e::Float32);
  
  std::vector<float> data_x(8*10*5,1.0f);  // Batch size 8, all ones
  std::vector<float> data_w(5*3,2.0f);     // All twos
  
  x.SetBatchData(data_x.data(),8);
  w.SetData(data_w.data(),data_w.size()*sizeof(float));
  
  auto matmul_result=x.MatMul(w);
  assert((matmul_result.Shape()[0]==8));   // Preserves batch size
  assert((matmul_result.Shape()[1]==10));  // Output rows
  assert((matmul_result.Shape()[2]==3));   // Output columns
  
  // Verify matrix multiplication results
  auto matmul_data=matmul_result.ConstData<float>();
  assert((matmul_data!=nullptr));
  
  // Each output element should be 5*2=10 (5 ones times 2)
  assert((std::abs(matmul_data[0]-10.0f)<1e-6));
  
  // Test 3: Element-wise multiplication with broadcasting
  std::vector<uint32_t> shape_a={0,4,4};
  std::vector<uint32_t> shape_b={0,4,4};
  
  CAIF_Tensor tensor_a(framework,shape_a,CAIF_DataType_e::Float32);
  CAIF_Tensor tensor_b(framework,shape_b,CAIF_DataType_e::Float32);
  
  std::vector<float> data_tensor_a(6*4*4,2.0f);  // Batch size 6, all twos
  std::vector<float> data_tensor_b(2*4*4,3.0f);  // Batch size 2, all threes
  
  tensor_a.SetBatchData(data_tensor_a.data(),6);
  tensor_b.SetBatchData(data_tensor_b.data(),2);
  assert((tensor_a.Shape()[0]==6));
  assert((tensor_b.Shape()[0]==2));
  
  auto mul_result=tensor_a.Multiply(tensor_b);
  assert((mul_result.Shape()[0]==6));  // Max batch size
  
  // Verify multiplication results with broadcasting
  auto mul_data=mul_result.ConstData<float>();
  assert((mul_data!=nullptr));
  
  // Each element should be 2*3=6
  assert((std::abs(mul_data[0]-6.0f)<1e-6));
  // Second batch should also be 2*3=6 (b's first batch)
  assert((std::abs(mul_data[4*4]-6.0f)<1e-6));
  // Third batch should be 2*3=6 (b's second batch)
  assert((std::abs(mul_data[2*4*4]-6.0f)<1e-6));
  // Fourth batch should be 2*3=6 (b's first batch repeats)
  assert((std::abs(mul_data[3*4*4]-6.0f)<1e-6));
  
  std::cout<<"Dynamic batch operations tests passed!\n";
}

int main()
{
  try
  {
    TestDynamicBatchSize();
    TestDynamicBatchValidation();
    TestDynamicBatchOperations();
    
    std::cout<<"All dynamic batch tests passed!\n";
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cerr<<"Error: "<<e.what()<<std::endl;
    return 1;
  }
}
