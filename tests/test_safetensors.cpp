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
#include "caif_test_harness.h"
#include "caif_device_network.h"
#include "caif_device_dense_layer.h"
#include "caif_device_vit_model.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <fstream>
#include <cmath>
#include <cstdio>

namespace instance
{

constexpr float g_caif_safetensors_test_scale=0.1f;
constexpr float g_caif_safetensors_test_bias_scale=-0.5f;
constexpr float g_caif_safetensors_test_img_scale_inv=255.0f;
constexpr float g_caif_safetensors_test_vit_lr=0.01f;
constexpr float g_caif_safetensors_test_tol=1e-4f;
constexpr uint32_t g_caif_safetensors_test_header_max=10000;
constexpr float g_caif_safetensors_test_py_weight_0=1.0f;
constexpr float g_caif_safetensors_test_py_weight_1=2.0f;
constexpr float g_caif_safetensors_test_py_weight_2=3.0f;
constexpr float g_caif_safetensors_test_py_weight_3=4.0f;
constexpr float g_caif_safetensors_test_py_weight_4=5.0f;
constexpr float g_caif_safetensors_test_py_weight_5=6.0f;
constexpr float g_caif_safetensors_test_py_bias_0=0.1f;
constexpr float g_caif_safetensors_test_py_bias_1=0.2f;
constexpr float g_caif_safetensors_test_py_bias_2=0.3f;

//------------------------------------------------------------------------------
// SafeTensors format correctness tests.
//------------------------------------------------------------------------------
class CAIF_SafeTensorsTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestBasicSaveLoad();
    static void TestDenseNetworkSaveLoad();
    static void TestViTModelSaveLoad();
    static void TestParameterNames();
    static void TestFileFormatValidation();
    static void TestPythonSerializationCompatibility();
};

//------------------------------------------------------------------------------
// Test: Basic SafeTensors Save/Load
//------------------------------------------------------------------------------
void CAIF_SafeTensorsTests::TestBasicSaveLoad()
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
      data1[i]=static_cast<float>(i)*g_caif_safetensors_test_scale;
    }
    tensor1.CopyFromHost(data1.data(),data1.size());

    std::vector<float> data2(8);
    for(size_t i=0;i<data2.size();++i)
    {
      data2[i]=static_cast<float>(i)*g_caif_safetensors_test_bias_scale;
    }
    tensor2.CopyFromHost(data2.data(),data2.size());

    // Save
    CAIF_SafeTensorsFormat format;
    std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> tensors;
    tensors.push_back({"weight",&tensor1});
    tensors.push_back({"bias",&tensor2});

    std::map<std::string,std::string> metadata;
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
      if(passed==true)
      {
        std::vector<float> loaded_data(32);
        loaded_weight.CopyToHost(loaded_data.data());
        for(size_t i=0;i<data1.size();++i)
        {
          if(CAIF_TestHarness::FloatEqual(loaded_data[i],data1[i],g_caif_safetensors_test_tol)==false)
          {
            passed=false;
            break;
          }
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
      if(passed==true)
      {
        std::vector<float> loaded_data(8);
        loaded_bias.CopyToHost(loaded_data.data());
        for(size_t i=0;i<data2.size();++i)
        {
          if(CAIF_TestHarness::FloatEqual(loaded_data[i],data2[i],g_caif_safetensors_test_tol)==false)
          {
            passed=false;
            break;
          }
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

    CAIF_TestHarness::Report("SafeTensors::BasicSaveLoad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SafeTensors::BasicSaveLoad")
}

//------------------------------------------------------------------------------
// Test: Dense Network Save/Load
//------------------------------------------------------------------------------
void CAIF_SafeTensorsTests::TestDenseNetworkSaveLoad()
{
  try
  {
    CAIF_CudaStream stream;
    const std::string test_path="test_dense_network.safetensors";

    // Create network
    CAIF_DeviceNetwork net1(stream);
    net1.AddDenseLayer(4,8,CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU,true);
    net1.AddDenseLayer(8,2,CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,true);

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
    net2.AddDenseLayer(4,8,CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU,true);
    net2.AddDenseLayer(8,2,CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,true);

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
      if(CAIF_TestHarness::FloatEqual(output1_data[i],output2_data[i],g_caif_safetensors_test_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": "
                      <<output1_data[i]
                      <<" vs "
                      <<output2_data[i]
                      <<"\n";
        passed=false;
      }
    }

    // Cleanup
    std::remove(test_path.c_str());

    CAIF_TestHarness::Report("SafeTensors::DenseNetworkSaveLoad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SafeTensors::DenseNetworkSaveLoad")
}

//------------------------------------------------------------------------------
// Test: ViT Model Save/Load
//------------------------------------------------------------------------------
void CAIF_SafeTensorsTests::TestViTModelSaveLoad()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    const std::string test_path="test_vit_model.safetensors";

    // Create small ViT config
    CAIF_DeviceViTModelConfig config(16,16,3,8,32,1,2,64,0.0f,4,false,10000.0f);
    CAIF_DeviceViTModel<float,float> model1(config,stream);

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
      input_data[i]=static_cast<float>(i%256)/g_caif_safetensors_test_img_scale_inv;
    }
    input.CopyFromHost(input_data.data(),input_data.size());

    // Run forward pass
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output1=model1.Forward(input,ctx);
    std::vector<float> output1_data(batch*config.NumClasses());
    output1.CopyToHost(output1_data.data());

    // Save
    std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> tensors;
    std::vector<std::string> param_names=model1.ParameterNames("");
    for(size_t i=0;i<model1.ParameterTensorCount();++i)
    {
      tensors.push_back({param_names[i],&model1.ParameterTensor(i)});
    }

    CAIF_SafeTensorsFormat format;
    std::map<std::string,std::string> metadata;
    metadata["model_type"]="vit";
    format.Save(test_path,tensors,metadata);

    // Create new model
    CAIF_DeviceViTModel<float,float> model2(config,stream);

    // Load weights
    auto loaded=format.Load(test_path,stream);

    // Copy loaded weights to model2
    for(size_t i=0;i<model2.ParameterTensorCount();++i)
    {
      const std::string &name=param_names[i];
      auto it=loaded.find(name);
      if(it==loaded.end())
      {
        THROW_CAIFE("Missing tensor");
      }
      CAIF_DeviceTensor &param=model2.ParameterTensor(i);
      const CAIF_DeviceTensor &loaded_tensor=it->second;
      std::vector<float> data(loaded_tensor.TotalElements());
      loaded_tensor.CopyToHost(data.data());
      param.CopyFromHost(data.data(),data.size());
    }

    // Run forward pass
    CAIF_DeviceTensor output2=model2.Forward(input,ctx);
    std::vector<float> output2_data(batch*config.NumClasses());
    output2.CopyToHost(output2_data.data());

    // Verify outputs match
    bool passed=true;
    for(size_t i=0;i<output1_data.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(output1_data[i],output2_data[i],g_caif_safetensors_test_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": "
                      <<output1_data[i]
                      <<" vs "
                      <<output2_data[i]
                      <<"\n";
        passed=false;
      }
    }

    // Cleanup
    std::remove(test_path.c_str());

    CAIF_TestHarness::Report("SafeTensors::ViTModelSaveLoad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SafeTensors::ViTModelSaveLoad")
}

//------------------------------------------------------------------------------
// Test: Parameter Names
//------------------------------------------------------------------------------
void CAIF_SafeTensorsTests::TestParameterNames()
{
  try
  {
    CAIF_CudaStream stream;

    // Create network with dense layers
    CAIF_DeviceNetwork net(stream);
    net.AddDenseLayer(4,8,CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU,true);
    net.AddDenseLayer(8,2,CAIF_DeviceActivation::CAIF_DeviceActivation_e::None,false);

    bool passed=true;

    // Check layer 0 parameter names
    auto names0=net.DenseLayer(0).ParameterNames("layers.0.");
    if(names0.size()!=2)
    {
      passed=false;
    }
    if(passed==true&&names0[0]!="layers.0.weight")
    {
      passed=false;
    }
    if(passed==true&&names0[1]!="layers.0.bias")
    {
      passed=false;
    }

    // Check layer 1 parameter names (no bias)
    if(passed==true)
    {
      auto names1=net.DenseLayer(1).ParameterNames("layers.1.");
      if(names1.size()!=1)
      {
        passed=false;
      }
      if(passed==true&&names1[0]!="layers.1.weight")
      {
        passed=false;
      }
    }

    CAIF_TestHarness::Report("SafeTensors::ParameterNames",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SafeTensors::ParameterNames")
}

//------------------------------------------------------------------------------
// Test: File Format Validation
//------------------------------------------------------------------------------
void CAIF_SafeTensorsTests::TestFileFormatValidation()
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
    std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> tensors;
    tensors.push_back({"test_tensor",&tensor});

    std::map<std::string,std::string> metadata;
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
      if(header_size==0||header_size>g_caif_safetensors_test_header_max)
      {
        passed=false;
      }

      if(passed==true)
      {
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
      }

      in.close();
    }

    // Cleanup
    std::remove(test_path.c_str());

    CAIF_TestHarness::Report("SafeTensors::FileFormatValidation",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SafeTensors::FileFormatValidation")
}

//------------------------------------------------------------------------------
// Test: Python Serialization Compatibility.
// Creates a file for Python verification - does NOT delete it.
//------------------------------------------------------------------------------
void CAIF_SafeTensorsTests::TestPythonSerializationCompatibility()
{
  try
  {
    CAIF_CudaStream stream;
    const std::string test_path="caif_python_compat.safetensors";

    // Create tensors with known values
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::Zeros({2,3},stream);
    CAIF_DeviceTensor bias=CAIF_DeviceTensor::Zeros({3},stream);

    std::vector<float> weight_data={g_caif_safetensors_test_py_weight_0,
                                    g_caif_safetensors_test_py_weight_1,
                                    g_caif_safetensors_test_py_weight_2,
                                    g_caif_safetensors_test_py_weight_3,
                                    g_caif_safetensors_test_py_weight_4,
                                    g_caif_safetensors_test_py_weight_5};
    std::vector<float> bias_data={g_caif_safetensors_test_py_bias_0,
                                  g_caif_safetensors_test_py_bias_1,
                                  g_caif_safetensors_test_py_bias_2};

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
        if(CAIF_TestHarness::FloatEqual(loaded_weight[i],weight_data[i],g_caif_safetensors_test_tol)==false)
        {
          passed=false;
          break;
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
        if(CAIF_TestHarness::FloatEqual(loaded_bias[i],bias_data[i],g_caif_safetensors_test_tol)==false)
        {
          passed=false;
          break;
        }
      }
    }

    // DO NOT cleanup — leave file for Python verification
    ISE_Out::Out()<<"  (File saved to "
                  <<test_path
                  <<" for Python verification)\n";

    CAIF_TestHarness::Report("SafeTensors::PythonSerializationCompatibility",passed);
  }
  CAIF_TEST_CATCH_BLOCK("SafeTensors::PythonSerializationCompatibility")
}

void CAIF_SafeTensorsTests::RunAll()
{
  ISE_Out::Out()<<"=== SafeTensors Format Tests ==="
                <<"\n\n";
  TestBasicSaveLoad();
  TestDenseNetworkSaveLoad();
  TestViTModelSaveLoad();
  TestParameterNames();
  TestFileFormatValidation();
  TestPythonSerializationCompatibility();
  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

}//end instance namespace

int main()
{
  instance::CAIF_SafeTensorsTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
