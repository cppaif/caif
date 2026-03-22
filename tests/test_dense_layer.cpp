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

#include "caif_dense_layer.h"
#include "caif_neural_network.h"
#include "caif_loss_function_bce_logits.h"
#include "caif_framework.h"
#include <iostream>

static int TestDenseForwardBackwardShapes()
{
  try
  {
    instance::CAIF_Framework framework;
    instance::CAIF_DenseLayer dense(framework,8,instance::CAIF_ActivationType_e::Sigmoid,true);
    std::vector<uint32_t> in_shape={3,5};
    std::cout<<"[DenseTest] Initializing dense layer...\n";
    dense.Initialize(in_shape);
    std::cout<<"[DenseTest] Initialized.\n";
    // Input batch [3,5]
    instance::CAIF_Tensor x(framework,in_shape,instance::CAIF_DataType::CAIF_DataType_e::Float32);
    auto xp=x.MutableData<float>();
    if(xp==nullptr)
    {
      return 1;
    }
    for(size_t i=0;i<x.NumElements();++i)
    {
      xp[i]=0.1f;
    }
    std::cout<<"[DenseTest] Running forward...\n";
    auto y=dense.Forward(x,true);
    std::cout<<"[DenseTest] Forward done.\n";
    if(false)
    {
      std::cout<<"forward failed\n";
      return 1;
    }
    if(y.Shape()!=std::vector<uint32_t>({3,8}))
    {
      std::cout<<"forward shape mismatch\n";
      return 1;
    }
    // Simple gradient coming from loss: same shape as output
    instance::CAIF_Tensor g(framework,{3,8},instance::CAIF_DataType::CAIF_DataType_e::Float32);
    auto gp=g.MutableData<float>();
    if(gp==nullptr)
    {
      return 1;
    }
    for(size_t i=0;i<g.NumElements();++i)
    {
      gp[i]=0.01f;
    }
    std::cout<<"[DenseTest] Running backward...\n";
    auto dx=dense.Backward(g);
    std::cout<<"[DenseTest] Backward done.\n";
    if(false)
    {
      std::cout<<"backward failed\n";
      return 1;
    }
    if(dx.Shape()!=std::vector<uint32_t>({3,5}))
    {
      std::cout<<"dx shape mismatch\n";
      return 1;
    }
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"dense fwd/bwd ex: "<<e.what()<<"\n";
    return 1;
  }
}

int main()
{
  std::cout<<"Testing Dense Layer Construction...\n";
  
  try
  {
    instance::CAIF_Framework framework;
    
    // Test 1: Direct dense layer construction
    std::cout<<"1. Testing direct dense layer construction...\n";
    instance::CAIF_DenseLayer dense_layer(framework,128,instance::CAIF_ActivationType_e::ReLU);
    std::cout<<"   ✅ Dense layer created successfully\n";
    
    // Test 2: Neural network layer addition
    std::cout<<"2. Testing neural network layer addition...\n";
    instance::CAIF_NeuralNetwork network;
    std::vector<uint32_t> input_shape={1,784};
    
    // Set input shape
    network.SetInputShape(input_shape);
    
    std::cout<<"   Adding dense layer to network...\n";
    network.AddDenseLayer(128,instance::CAIF_ActivationType_e::ReLU);
    std::cout<<"   ✅ AddDenseLayer returned success\n";
    std::cout<<"   Current layer count: "<<network.LayerCount()<<"\n";
    
    int rc=0;
    rc|=TestDenseForwardBackwardShapes();
    if(rc==0)
    {
      std::cout<<"\n✅ All dense layer tests passed!\n";
    }
    return rc;
  }
  catch(const std::exception &e)
  {
    std::cout<<"   ❌ Exception: "<<e.what()<<"\n";
    return 1;
  }
} 
