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

#include "caif/caif_neural_network.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <iostream>

using namespace instance;

int main()
{
  ISE_Out::AddLogLevel(ISE_Out::ISE_LogLevel(ISE_Out::LOG));
  ISE_Out::AddLogLevel(ISE_Out::ISE_LogLevel(ISE_Out::DEBUG));
  ISE_Out::AddLogLevel(ISE_Out::ISE_LogLevel(ISE_Out::ERROR));

  // Simple binary classifier: input 2 features -> 1 output (sigmoid-like via softmax over 2)
  CAIF_NeuralNetwork net;
  try
  {
    net.SetInputShape({4,2});
  }
  catch(const std::exception &e)
  {
    std::cerr<<"SetInputShape failed: "
             <<e.what()<<"\n";
    return 1;
  }

  // One hidden dense layer
  try
  {
    net.AddDenseLayer(4,CAIF_ActivationType_e::ReLU);
  }
  catch(const std::exception &)
  {
    std::cerr<<"AddDenseLayer failed\n";
    return 1;
  }
  // Output 2 classes with softmax
  try
  {
    net.AddDenseLayer(2,CAIF_ActivationType_e::Softmax);
  }
  catch(const std::exception &)
  {
    std::cerr<<"AddDenseLayer failed\n";
    return 1;
  }

  try
  {
    net.Compile(CAIF_OptimizerType_e::SGD,
                CAIF_LossType_e::CrossEntropy,
                0.01f);
  }
  catch(const std::exception &)
  {
    std::cerr<<"Compile failed\n";
    return 1;
  }

  // Create a tiny dataset
  CAIF_Tensor X(net.Framework(),{4,2},CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_Tensor Y(net.Framework(),{4,2},CAIF_DataType::CAIF_DataType_e::Float32);

  // X: 4 samples, 2 features
  {
    float *x=X.MutableData<float>();
    float *y=Y.MutableData<float>();
    if(x==nullptr||y==nullptr){std::cerr<<"Data ptr failed\n"; return 1;}
    // Two samples of class 0, two of class 1 (one-hot targets)
    x[0]=0.0f; x[1]=0.0f; y[0]=1.0f; y[1]=0.0f;
    x[2]=0.1f; x[3]=0.2f; y[2]=1.0f; y[3]=0.0f;
    x[4]=1.0f; x[5]=1.0f; y[4]=0.0f; y[5]=1.0f;
    x[6]=0.9f; x[7]=0.8f; y[6]=0.0f; y[7]=1.0f;
  }

  // Evaluate before training (accuracy may be ~50%)
  CAIF_NeuralNetwork::CAIF_TrainingMetrics_t m;
  try
  {
    m=net.Evaluate(X,Y);
  }
  catch(const std::exception &e)
  {
    std::cerr<<"Evaluate failed: "<<e.what()<<"\n";
    return 1;
  }
  std::cout<<"Loss: "<<m.loss<<" Accuracy: "<<m.accuracy<<"%\n";
  return 0;
}

