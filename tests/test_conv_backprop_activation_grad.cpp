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
#include "caif_convolution2d_layer.h"
#include "caif_tensor.h"
#include <iostream>

int main()
{
  using namespace instance;
  try
  {
    std::cout<<"[Test] Conv backprop activation gradient non-zero check...\n";

    CAIF_NeuralNetwork net;
    net.SetInputShape({1,8,8,3});

    net.AddConvolution2DLayer(4,3,1,1,CAIF_ActivationType_e::ReLU);

    net.AddFlattenLayer();

    net.AddDenseLayer(1,CAIF_ActivationType_e::Sigmoid,true);

    net.Compile(CAIF_OptimizerType_e::Adam,CAIF_LossType_e::BinaryCrossEntropy,0.001f);

    CAIF_Tensor x(net.Framework(),{1,8,8,3},CAIF_DataType::CAIF_DataType_e::Float32);
    if(auto xp=x.MutableData<float>(); xp!=nullptr)
    {
      float *xd=xp;
      for(uint32_t i=0;i<x.NumElements();++i)
      {
        xd[i]=0.5f;
      }
    }
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
    if(dyw==nullptr)
    {
      std::cout<<"Alloc grad failed\n";
      return 1;
    }
    float *g=dyw;
    for(uint32_t i=0; i<dy.NumElements(); ++i)
    {
      g[i]=0.0f;
    }
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

    // Find last conv and inspect LastActivationGradient
    int last_conv=-1;
    for(uint32_t i=0; i<net.LayerCount(); ++i)
    {
      if(net.Layer(i).LayerType()==CAIF_LayerType_e::Convolution2D)
      {
        last_conv=static_cast<int>(i);
      }
    }
    if(last_conv<0)
    {
      std::cout<<"No convolution layer found\n";
      return 1;
    }
    auto *conv=dynamic_cast<CAIF_Convolution2DLayer*>(&net.Layer(static_cast<uint32_t>(last_conv)));
    if(conv==nullptr)
    {
      std::cout<<"Conv cast failed\n";
      return 1;
    }

    const CAIF_Tensor &act_grad=conv->LastActivationGradient();
    auto gp=act_grad.ConstData<float>();
    if(gp==nullptr)
    {
      std::cout<<"No activation gradient data\n";
      return 1;
    }
    const float *gd=gp;
    size_t n=act_grad.NumElements();
    size_t nz=0; float gmin=1e30f; float gmax=-1e30f;
    for(size_t i=0; i<n; ++i)
    {
      float v=gd[i];
      if(v!=0.0f)
      {
        ++nz;
      }
      if(v<gmin)
      {
        gmin=v;
      }
      if(v>gmax)
      {
        gmax=v;
      }
    }
    std::cout<<"[Test] Activation gradient stats: min="<<gmin<<" max="<<gmax<<" nz="<<nz<<"/"<<n<<"\n";

    if(n==0||nz==0)
    {
      std::cout<<"FAIL: Activation gradients are zero.\n";
      return 1;
    }

    std::cout<<"PASS: Activation gradients are non-zero.\n";
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    return 1;
  }
}


