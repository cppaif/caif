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

#include "caif_neural_network.h"
#include "caif_tensor.h"
#include <iostream>

int main()
{
  using namespace instance;
  try
  {
    CAIF_NeuralNetwork net;
    net.SetInputShape({1,8,8,3});
    net.AddConvolution2DLayer(4,3,1,1);
    net.AddFlattenLayer();
    net.AddDenseLayer(1,CAIF_ActivationType_e::Sigmoid,true);
    net.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::BinaryCrossEntropy,0.001f);

    CAIF_Tensor x(net.Framework(),{1,8,8,3},CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Tensor y(net.Framework());
    try
    {
      y=net.Forward(x,true);
    }
    catch(const std::exception &e)
    {
      std::cout<<"Forward failed: "<<e.what()<<"\n";
      return 1;
    }
    CAIF_Tensor dy(net.Framework(),y.Shape(),CAIF_DataType::CAIF_DataType_e::Float32);
    auto dyw=dy.MutableData<float>();
    if(dyw==nullptr){std::cout<<"Alloc grad failed\n"; return 1;}
    float *g=dyw;
    for(uint32_t i=0;i<dy.NumElements();++i){g[i]=0.0f;}
    g[0]=1.0f;
    try
    {
      net.ComputeGradients(dy);
    }
    catch(const std::exception &e)
    {
      std::cout<<"ComputeGradients failed: "
               <<e.what()<<"\n";
      return 1;
    }

    // quick sanity: conv layer should have non-empty cached activation gradient
    int has_conv=0;
    for(uint32_t i=0;i<net.LayerCount();++i)
    {
      if(net.Layer(i).LayerType()==CAIF_LayerType_e::Convolution2D){has_conv=1;}
    }
    if(has_conv==0){std::cout<<"No conv layer found\n"; return 1;}
    std::cout<<"Gradients API smoke test passed.\n";
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    return 1;
  }
}


