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
#include <cmath>

using namespace instance;
using CAIF_DataType_e = CAIF_DataType::CAIF_DataType_e;

void TestActivationFunctions()
{
  std::cout<<"Testing activation functions...\n";
  
  CAIF_Framework framework;
  
  // Create test tensor with known values
  std::vector<uint32_t> shape={2,3};
  CAIF_Tensor tensor(framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> data={-2.0f,-1.0f,0.0f,1.0f,2.0f,3.0f};
  tensor.SetData(data.data(),data.size()*sizeof(float));
  
  // Test Linear activation
  std::cout<<"Testing Linear activation...\n";
  CAIF_Tensor linear_result=tensor.Linear();
  const float *linear_data=linear_result.ConstData<float>();
  assert((linear_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    assert((std::abs(linear_data[i]-data[i])<1e-6));
  }
  
  // Test ReLU activation
  std::cout<<"Testing ReLU activation...\n";
  CAIF_Tensor relu_result=tensor.ReLU();
  const float *relu_data=relu_result.ConstData<float>();
  assert((relu_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=std::max(0.0f,data[i]);
    assert((std::abs(relu_data[i]-expected)<1e-6));
  }
  
  // Test Sigmoid activation
  std::cout<<"Testing Sigmoid activation...\n";
  CAIF_Tensor sigmoid_result=tensor.Sigmoid();
  const float *sigmoid_data=sigmoid_result.ConstData<float>();
  assert((sigmoid_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=1.0f/(1.0f+std::exp(-data[i]));
    assert((std::abs(sigmoid_data[i]-expected)<1e-6));
  }
  
  // Test Tanh activation
  std::cout<<"Testing Tanh activation...\n";
  CAIF_Tensor tanh_result=tensor.Tanh();
  const float *tanh_data=tanh_result.ConstData<float>();
  assert((tanh_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=std::tanh(data[i]);
    assert((std::abs(tanh_data[i]-expected)<1e-6));
  }
  
  // Test Softmax activation
  std::cout<<"Testing Softmax activation...\n";
  CAIF_Tensor softmax_result=tensor.Softmax();
  const float *softmax_data=softmax_result.ConstData<float>();
  assert((softmax_data!=nullptr));
  float sum=0.0f;
  for(size_t i=0;i<data.size();++i)
  {
    sum+=softmax_data[i];
  }
  assert((std::abs(sum-1.0f)<1e-6));  // Sum should be 1
  
  // Test LeakyReLU activation
  std::cout<<"Testing LeakyReLU activation...\n";
  float alpha=0.01f;
  CAIF_Tensor leaky_relu_result=tensor.LeakyReLU(alpha);
  const float *leaky_relu_data=leaky_relu_result.ConstData<float>();
  assert((leaky_relu_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=(data[i]>0.0f)?data[i]:alpha*data[i];
    assert((std::abs(leaky_relu_data[i]-expected)<1e-6));
  }
  
  // Test ELU activation
  std::cout<<"Testing ELU activation...\n";
  float elu_alpha=1.0f;
  CAIF_Tensor elu_result=tensor.ELU(elu_alpha);
  const float *elu_data=elu_result.ConstData<float>();
  assert((elu_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=(data[i]>0.0f)?data[i]:elu_alpha*(std::exp(data[i])-1.0f);
    assert((std::abs(elu_data[i]-expected)<1e-6));
  }
  
  // Test GELU activation
  std::cout<<"Testing GELU activation...\n";
  CAIF_Tensor gelu_result=tensor.GELU();
  const float *gelu_data=gelu_result.ConstData<float>();
  assert((gelu_data!=nullptr));
  
  // Test Swish activation
  std::cout<<"Testing Swish activation...\n";
  CAIF_Tensor swish_result=tensor.Swish();
  const float *swish_data=swish_result.ConstData<float>();
  assert((swish_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=data[i]/(1.0f+std::exp(-data[i]));
    assert((std::abs(swish_data[i]-expected)<1e-6));
  }
}

void TestActivationDerivatives()
{
  std::cout<<"Testing activation derivatives...\n";
  
  CAIF_Framework framework;
  
  // Create test tensor and gradient
  std::vector<uint32_t> shape={2,3};
  CAIF_Tensor tensor(framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_Tensor gradient(framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
  
  std::vector<float> data={-2.0f,-1.0f,0.0f,1.0f,2.0f,3.0f};
  std::vector<float> grad_data={0.5f,0.5f,0.5f,0.5f,0.5f,0.5f};
  
  tensor.SetData(data.data(),data.size()*sizeof(float));
  gradient.SetData(grad_data.data(),grad_data.size()*sizeof(float));
  
  // Test Linear derivative
  std::cout<<"Testing Linear derivative...\n";
  CAIF_Tensor linear_deriv=tensor.LinearDerivative(gradient);
  const float *linear_deriv_data=linear_deriv.ConstData<float>();
  assert((linear_deriv_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    assert((std::abs(linear_deriv_data[i]-grad_data[i])<1e-6));
  }
  
  // Test ReLU derivative
  std::cout<<"Testing ReLU derivative...\n";
  CAIF_Tensor relu_deriv=tensor.ReLUDerivative(gradient);
  const float *relu_deriv_data=relu_deriv.ConstData<float>();
  assert((relu_deriv_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=(data[i]>0.0f)?grad_data[i]:0.0f;
    assert((std::abs(relu_deriv_data[i]-expected)<1e-6));
  }
  
  // Test Sigmoid derivative
  std::cout<<"Testing Sigmoid derivative...\n";
  CAIF_Tensor sigmoid=tensor.Sigmoid();
  CAIF_Tensor sigmoid_deriv=sigmoid.SigmoidDerivative(gradient);
  const float *sigmoid_deriv_data=sigmoid_deriv.ConstData<float>();
  assert((sigmoid_deriv_data!=nullptr));
  
  // Test Tanh derivative
  std::cout<<"Testing Tanh derivative...\n";
  CAIF_Tensor tanh=tensor.Tanh();
  CAIF_Tensor tanh_deriv=tanh.TanhDerivative(gradient);
  const float *tanh_deriv_data=tanh_deriv.ConstData<float>();
  assert((tanh_deriv_data!=nullptr));
  
  // Test LeakyReLU derivative
  std::cout<<"Testing LeakyReLU derivative...\n";
  float alpha=0.01f;
  CAIF_Tensor leaky_relu_deriv=tensor.LeakyReLUDerivative(gradient,alpha);
  const float *leaky_relu_deriv_data=leaky_relu_deriv.ConstData<float>();
  assert((leaky_relu_deriv_data!=nullptr));
  for(size_t i=0;i<data.size();++i)
  {
    float expected=(data[i]>0.0f)?grad_data[i]:alpha*grad_data[i];
    assert((std::abs(leaky_relu_deriv_data[i]-expected)<1e-6));
  }
  
  // Test ELU derivative
  std::cout<<"Testing ELU derivative...\n";
  float elu_alpha=1.0f;
  CAIF_Tensor elu_deriv=tensor.ELUDerivative(gradient,elu_alpha);
  const float *elu_deriv_data=elu_deriv.ConstData<float>();
  assert((elu_deriv_data!=nullptr));
  
  // Test GELU derivative
  std::cout<<"Testing GELU derivative...\n";
  CAIF_Tensor gelu_deriv=tensor.GELUDerivative(gradient);
  const float *gelu_deriv_data=gelu_deriv.ConstData<float>();
  assert((gelu_deriv_data!=nullptr));
  
  // Test Swish derivative
  std::cout<<"Testing Swish derivative...\n";
  CAIF_Tensor swish_deriv=tensor.SwishDerivative(gradient);
  const float *swish_deriv_data=swish_deriv.ConstData<float>();
  assert((swish_deriv_data!=nullptr));
}

int main()
{
  try
  {
    TestActivationFunctions();
    TestActivationDerivatives();
    
    std::cout<<"All activation tests passed!\n";
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cerr<<"Error: "<<e.what()<<std::endl;
    return 1;
  }
}
