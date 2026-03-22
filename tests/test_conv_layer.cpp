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

#include "caif_convolution2d_layer.h"
#include "caif_tensor.h"
#include "caif_settings.h"
#include "caif_tensor_backend.h"
#include "caif_framework.h"
#include <iostream>

static instance::CAIF_TensorBackend::BackendType_e BackendFromArg(const std::string &value)
{
  if(value=="cuda"||value=="CUDA")
  {
    return instance::CAIF_TensorBackend::BackendType_e::CUDA;
  }
  if(value=="cpu"||value=="CPU"||value=="blas"||value=="BLAS")
  {
    // BLAS is the default/preferred CPU backend (faster than Eigen)
    return instance::CAIF_TensorBackend::BackendType_e::BLAS;
  }
  if(value=="eigen"||value=="EIGEN")
  {
    return instance::CAIF_TensorBackend::BackendType_e::Eigen;
  }
  if(value=="blas"||value=="BLAS")
  {
    return instance::CAIF_TensorBackend::BackendType_e::BLAS;
  }
  if(value=="vulkan"||value=="VULKAN")
  {
    return instance::CAIF_TensorBackend::BackendType_e::Vulkan;
  }
  return instance::CAIF_TensorBackend::BackendType_e::Auto;
}

static void ConfigureBackendFromArgs(int argc,char *argv[])
{
  for(int i=1;i<argc;++i)
  {
    const std::string arg(argv[i]);
    const std::string prefix="--backend=";
    if(arg.rfind(prefix,0)==0)
    {
      const std::string value=arg.substr(prefix.size());
      const auto backend=BackendFromArg(value);
      instance::CAIF_Settings::SetBackendOverride(backend);
    }
  }
}

int main(int argc,char *argv[])
{
  ConfigureBackendFromArgs(argc,argv);
  std::cout<<"Testing Convolution2D Layer Construction...\n";
  
  try
  {
    instance::CAIF_Framework framework;
    
    // Test valid parameters
    std::cout<<"1. Testing valid parameters...\n";
    instance::CAIF_Convolution2DLayer conv_layer(framework,32,3,1,1);
    std::cout<<"   ✅ Convolution layer created successfully\n";
    
    // Test initialization
    std::cout<<"2. Testing layer initialization...\n";
    std::vector<uint32_t> input_shape={1,32,32,3};  // [batch, height, width, channels]
    conv_layer.Initialize(input_shape);
    std::cout<<"   ✅ Layer initialized successfully\n";
    
    // Test forward pass
    std::cout<<"3. Testing forward pass...\n";
    instance::CAIF_Tensor input_tensor(framework,input_shape,instance::CAIF_DataType::CAIF_DataType_e::Float32);
    auto output=conv_layer.Forward(input_tensor,false);
    auto output_shape=output.Shape();
    std::cout<<"   ✅ Forward pass successful\n";
    std::cout<<"   Output shape: [";
    for(size_t i=0;i<output_shape.size();++i)
    {
      if(i>0)
      {
        std::cout<<", ";
      }
      std::cout<<output_shape[i];
    }
    std::cout<<"]\n";
    
    std::cout<<"\n✅ All convolution layer tests passed!\n";
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"   ✗ Exception: "<<e.what()<<"\n";
    return 1;
  }
}
