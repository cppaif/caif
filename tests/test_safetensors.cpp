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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// SafeTensors Format Tests
//------------------------------------------------------------------------------
#include "caif_safetensors_format.h"
#include "caif_device_network.h"
#include "caif_device_dense_layer.h"
#include "caif_device_vit_model.h"
#include "caif_cuda_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    std::cout<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    std::cout<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
}

static bool FloatEqual(float a,float b,float tolerance=1e-5f)
{
  return std::fabs(a-b)<tolerance;
}

//------------------------------------------------------------------------------
// Test: Basic SafeTensors Save/Load
//------------------------------------------------------------------------------

static void TestBasicSaveLoad()
{
  try
  {
    CAIF_CudaStream stream;
    const std::string test_path="test_basic_safetensors.safetensors";

    // Create tensors
    CAIF_DeviceTensor tensor1=CAIF_DeviceTensor::Zeros({4,8},stream);
    CAIF_DeviceTensor tensor2=CAIF_DeviceTensor::Zeros({8},stream);

    // Fill with known values
    std::vector<float> data1(32);
    for(size_t i=0;i<data1.size();++i)
    {
      data1[i]=static_cast<float>(i)*0.1f;
    }
    tensor1.CopyFromHost(data1.data(),data1.size());

    std::vector<float> data2(8);
    for(size_t i=0;i<data2.size();++i)
    {
      data2[i]=static_cast<float>(i)*-0.5f;
    }
    tensor2.CopyFromHost(data2.data(),data2.size());

    // Save
    CAIF_SafeTensorsFormat format;
    std::vector<std::pair<std::string, const CAIF_DeviceTensor*>> tensors;
    tensors.push_back({"weight",&tensor1});
    tensors.push_back({"bias",&tensor2});

    std::map<std::string, std::string> metadata;
    metadata["test_key"]="test_value";

    format.Save(test_path,tensors,metadata);

    // Load
    auto loaded=format.Load(test_path,stream);

    // Verify
    bool passed=true;

    if(loaded.size()!=2)
    {
      passed=false;
    }

    if(loaded.find("weight")==loaded.end())
    {
      passed=false;
    }
    else
    {
      auto &loaded_weight=loaded.at("weight");
      if(loaded_weight.Shape()!=tensor1.Shape())
      {
        passed=false;
      }
      std::vector<float> loaded_data(32);
      loaded_weight.CopyToHost(loaded_data.data());
      for(size_t i=0;i<data1.size();++i)
      {
        if(FloatEqual(loaded_data[i],data1[i])==false)
        {
          passed=false;
          break;
        }
      }
    }

    if(loaded.find("bias")==loaded.end())
    {
      passed=false;
    }
    else
    {
      auto &loaded_bias=loaded.at("bias");
      if(loaded_bias.Shape()!=tensor2.Shape())
      {
        passed=false;
      }
      std::vector<float> loaded_data(8);
      loaded_bias.CopyToHost(loaded_data.data());
      for(size_t i=0;i<data2.size();++i)
      {
        if(FloatEqual(loaded_data[i],data2[i])==false)
        {
          passed=false;
          break;
        }
      }
    }

    // Verify metadata
    auto loaded_meta=format.Metadata(test_path);
    if(loaded_meta.find("test_key")==loaded_meta.end())
    {
      passed=false;
    }
    else if(loaded_meta["test_key"]!="test_value")
    {
      passed=false;
    }

    // Cleanup
    std::remove(test_path.c_str());

    ReportResult("BasicSaveLoad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("BasicSaveLoad",false);
  }
}

//------------------------------------------------------------------------------
// Test: Dense Network Save/Load
//------------------------------------------------------------------------------

static void TestDenseNetworkSaveLoad()
{
  try
  {
    CAIF_CudaStream stream;
    const std::string test_path="test_dense_network.safetensors";

    // Create network
    CAIF_DeviceNetwork net1(stream);
    net1.AddDenseLayer(4,8,CAIF_DeviceActivation_e::ReLU,true);
    net1.AddDenseLayer(8,2,CAIF_DeviceActivation_e::None,true);

    // Create test input
    CAIF_DeviceTensor input=CAIF_DeviceTensor::Zeros({2,4},stream);
    std::vector<float> input_data={1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f};
    input.CopyFromHost(input_data.data(),input_data.size());

    // Run forward pass
    CAIF_DeviceTensor output1=net1.Forward(input,false);
    std::vector<float> output1_data(4);
    output1.CopyToHost(output1_data.data());

    // Save
    net1.SaveSafeTensors(test_path);

    // Create new network with same architecture
    CAIF_DeviceNetwork net2(stream);
    net2.AddDenseLayer(4,8,CAIF_DeviceActivation_e::ReLU,true);
    net2.AddDenseLayer(8,2,CAIF_DeviceActivation_e::None,true);

    // Load weights
    net2.LoadSafeTensors(test_path);

    // Run forward pass
    CAIF_DeviceTensor output2=net2.Forward(input,false);
    std::vector<float> output2_data(4);
    output2.CopyToHost(output2_data.data());

    // Verify outputs match
    bool passed=true;
    for(size_t i=0;i<output1_data.size();++i)
    {
      if(FloatEqual(output1_data[i],output2_data[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": "<<output1_data[i]<<" vs "<<output2_data[i]<<"\n";
        passed=false;
      }
    }

    // Cleanup
    std::remove(test_path.c_str());

    ReportResult("DenseNetworkSaveLoad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DenseNetworkSaveLoad",false);
  }
}

//------------------------------------------------------------------------------
// Test: ViT Model Save/Load
//------------------------------------------------------------------------------

static void TestViTModelSaveLoad()
{
  try
  {
    CAIF_CudaStream stream;
    const std::string test_path="test_vit_model.safetensors";

    // Create small ViT config
    CAIF_DeviceViTModel::Config_t config;
    config.image_height=16;
    config.image_width=16;
    config.channels=3;
    config.patch_size=8;
    config.dim=32;
    config.num_heads=2;
    config.num_layers=1;
    config.ffn_hidden_dim=64;
    config.dropout_rate=0.0f;
    config.num_classes=4;
    config.use_rope=false;
    config.rope_base=10000.0f;

    // Create model
    CAIF_DeviceViTModel model1(config,stream);

    // Create test input: [batch=1, height=16, width=16, channels=3] (BHWC format)
    const uint32_t batch=1;
    const uint32_t height=16;
    const uint32_t width=16;
    const uint32_t channels=3;
    CAIF_DeviceTensor input=CAIF_DeviceTensor::Zeros({batch,height,width,channels},stream);

    // Fill with known pattern
    std::vector<float> input_data(batch*channels*height*width);
    for(size_t i=0;i<input_data.size();++i)
    {
      input_data[i]=static_cast<float>(i%256)/255.0f;
    }
    input.CopyFromHost(input_data.data(),input_data.size());

    // Run forward pass
    CAIF_DeviceTensor output1=model1.Forward(input,false);
    std::vector<float> output1_data(batch*config.num_classes);
    output1.CopyToHost(output1_data.data());

    // Save
    std::vector<std::pair<std::string, const CAIF_DeviceTensor*>> tensors;
    std::vector<std::string> param_names=model1.ParameterNames("");
    for(size_t i=0;i<model1.ParameterTensorCount();++i)
    {
      tensors.push_back({param_names[i],&model1.ParameterTensor(i)});
    }

    CAIF_SafeTensorsFormat format;
    std::map<std::string, std::string> metadata;
    metadata["model_type"]="vit";
    format.Save(test_path,tensors,metadata);

    // Create new model
    CAIF_DeviceViTModel model2(config,stream);

    // Load weights
    auto loaded=format.Load(test_path,stream);

    // Copy loaded weights to model2
    for(size_t i=0;i<model2.ParameterTensorCount();++i)
    {
      const std::string &name=param_names[i];
      auto it=loaded.find(name);
      if(it==loaded.end())
      {
        throw std::runtime_error("Missing tensor: "+name);
      }

      CAIF_DeviceTensor &param=model2.ParameterTensor(i);
      const CAIF_DeviceTensor &loaded_tensor=it->second;

      std::vector<float> data(loaded_tensor.TotalElements());
      loaded_tensor.CopyToHost(data.data());
      param.CopyFromHost(data.data(),data.size());
    }

    // Run forward pass
    CAIF_DeviceTensor output2=model2.Forward(input,false);
    std::vector<float> output2_data(batch*config.num_classes);
    output2.CopyToHost(output2_data.data());

    // Verify outputs match
    bool passed=true;
    for(size_t i=0;i<output1_data.size();++i)
    {
      if(FloatEqual(output1_data[i],output2_data[i],1e-4f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": "<<output1_data[i]<<" vs "<<output2_data[i]<<"\n";
        passed=false;
      }
    }

    // Cleanup
    std::remove(test_path.c_str());

    ReportResult("ViTModelSaveLoad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("ViTModelSaveLoad",false);
  }
}

//------------------------------------------------------------------------------
// Test: Parameter Names
//------------------------------------------------------------------------------

static void TestParameterNames()
{
  try
  {
    CAIF_CudaStream stream;

    // Create network with dense layers
    CAIF_DeviceNetwork net(stream);
    net.AddDenseLayer(4,8,CAIF_DeviceActivation_e::ReLU,true);
    net.AddDenseLayer(8,2,CAIF_DeviceActivation_e::None,false);

    bool passed=true;

    // Check layer 0 parameter names
    auto names0=net.DenseLayer(0).ParameterNames("layers.0.");
    if(names0.size()!=2)
    {
      passed=false;
    }
    if(names0[0]!="layers.0.weight")
    {
      passed=false;
    }
    if(names0[1]!="layers.0.bias")
    {
      passed=false;
    }

    // Check layer 1 parameter names (no bias)
    auto names1=net.DenseLayer(1).ParameterNames("layers.1.");
    if(names1.size()!=1)
    {
      passed=false;
    }
    if(names1[0]!="layers.1.weight")
    {
      passed=false;
    }

    ReportResult("ParameterNames",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("ParameterNames",false);
  }
}

//------------------------------------------------------------------------------
// Test: File Format Validation
//------------------------------------------------------------------------------

static void TestFileFormatValidation()
{
  try
  {
    CAIF_CudaStream stream;
    const std::string test_path="test_format.safetensors";

    // Create and save a tensor
    CAIF_DeviceTensor tensor=CAIF_DeviceTensor::Zeros({2,3},stream);
    std::vector<float> data={1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    tensor.CopyFromHost(data.data(),data.size());

    CAIF_SafeTensorsFormat format;
    std::vector<std::pair<std::string, const CAIF_DeviceTensor*>> tensors;
    tensors.push_back({"test_tensor",&tensor});

    std::map<std::string, std::string> metadata;
    metadata["format_version"]="1.0";
    metadata["model_name"]="test_model";

    format.Save(test_path,tensors,metadata);

    // Read file header manually to verify structure
    std::ifstream in(test_path,std::ios::binary);
    bool passed=true;

    if(in.is_open()==false)
    {
      passed=false;
    }
    else
    {
      // Read header size (8 bytes, little-endian)
      uint64_t header_size=0;
      in.read(reinterpret_cast<char*>(&header_size),sizeof(uint64_t));

      // Header should be reasonable size
      if(header_size==0||header_size>10000)
      {
        passed=false;
      }

      // Read header JSON
      std::string header(header_size,'\0');
      in.read(&header[0],static_cast<std::streamsize>(header_size));

      // Verify header contains expected keys
      if(header.find("\"test_tensor\"")==std::string::npos)
      {
        passed=false;
      }
      if(header.find("\"dtype\":\"F32\"")==std::string::npos)
      {
        passed=false;
      }
      if(header.find("\"shape\":[2,3]")==std::string::npos)
      {
        passed=false;
      }
      if(header.find("\"__metadata__\"")==std::string::npos)
      {
        passed=false;
      }
      if(header.find("\"format_version\":\"1.0\"")==std::string::npos)
      {
        passed=false;
      }

      in.close();
    }

    // Cleanup
    std::remove(test_path.c_str());

    ReportResult("FileFormatValidation",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("FileFormatValidation",false);
  }
}

//------------------------------------------------------------------------------
// Test: Python Serialization Compatibility
// Creates a file for Python verification - does NOT delete it
//------------------------------------------------------------------------------

static void TestPythonSerializationCompatibility()
{
  try
  {
    CAIF_CudaStream stream;
    const std::string test_path="caif_python_compat.safetensors";

    // Create tensors with known values
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::Zeros({2,3},stream);
    CAIF_DeviceTensor bias=CAIF_DeviceTensor::Zeros({3},stream);

    std::vector<float> weight_data={1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    std::vector<float> bias_data={0.1f,0.2f,0.3f};

    weight.CopyFromHost(weight_data.data(),weight_data.size());
    bias.CopyFromHost(bias_data.data(),bias_data.size());

    // Save
    CAIF_SafeTensorsFormat format;
    std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> tensors;
    tensors.push_back({"model.weight",&weight});
    tensors.push_back({"model.bias",&bias});

    std::map<std::string,std::string> metadata;
    metadata["framework"]="CAIF";
    metadata["version"]="1.0";

    format.Save(test_path,tensors,metadata);

    // Load back with CAIF and verify
    auto loaded=format.Load(test_path,stream);

    bool passed=true;

    // Verify weight
    if(loaded.find("model.weight")==loaded.end())
    {
      passed=false;
    }
    else
    {
      std::vector<float> loaded_weight(6);
      loaded.at("model.weight").CopyToHost(loaded_weight.data());
      for(size_t i=0;i<weight_data.size();++i)
      {
        if(FloatEqual(loaded_weight[i],weight_data[i])==false)
        {
          passed=false;
        }
      }
    }

    // Verify bias
    if(loaded.find("model.bias")==loaded.end())
    {
      passed=false;
    }
    else
    {
      std::vector<float> loaded_bias(3);
      loaded.at("model.bias").CopyToHost(loaded_bias.data());
      for(size_t i=0;i<bias_data.size();++i)
      {
        if(FloatEqual(loaded_bias[i],bias_data[i])==false)
        {
          passed=false;
        }
      }
    }

    // DO NOT cleanup - leave file for Python verification
    std::cout<<"  (File saved to "<<test_path<<" for Python verification)\n";

    ReportResult("PythonSerializationCompatibility",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("PythonSerializationCompatibility",false);
  }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main()
{
  std::cout<<"=== SafeTensors Format Tests ===\n\n";

  TestBasicSaveLoad();
  TestDenseNetworkSaveLoad();
  TestViTModelSaveLoad();
  TestParameterNames();
  TestFileFormatValidation();
  TestPythonSerializationCompatibility();

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  return (g_tests_failed>0)?1:0;
}
