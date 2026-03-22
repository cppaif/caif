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
#include "caif_exception.h"
#include "caif_settings.h"
#include "caif_tensor_backend.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include "ise_lib/ise_out.h"

using namespace instance;

// Type definitions
typedef std::vector<uint32_t> ShapeVec_t;

// Global constants
constexpr const char *g_test_models_dir="./test_network_models";

// Function declarations
bool TestNetworkCreation();
bool TestNetworkCompilation();
bool TestNetworkPrediction();
bool TestNetworkSerialization();
bool TestDenseWithConvOutput();

// Helper function to print tensor shape
void PrintShape(const ShapeVec_t &shape, const std::string &name)
{
  ISE_Out::Log()<<"SHAPE DEBUG - "<<name<<": [";
  for(size_t i=0; i<shape.size(); ++i)
  {
    if(i>0)
    {
      ISE_Out::Log()<<", ";
    }
    ISE_Out::Log()<<shape[i];
  }
  ISE_Out::Log()<<"]"<<std::endl;
}

/**
 * @brief Ensures that a directory exists, creating it if necessary
 * @param dirpath Path to the directory
 * @param error_message Error message prefix to use if creation fails
 * @return True if directory exists or was created, false otherwise
 */
bool EnsureDirectoryExists(const std::string &dirpath, std::string &error_message)
{
  try
  {
    if(dirpath.empty()==true)
    {
      error_message="Directory path is empty";
      return false;
    }
    
    if(std::filesystem::exists(dirpath)==true)
    {
      return std::filesystem::is_directory(dirpath);
    }
    
    bool result=std::filesystem::create_directories(dirpath);
    if(result==false)
    {
      error_message="Failed to create directory";
    }
    return result;
  }
  catch(const std::exception &e)
  {
    error_message=e.what();
    return false;
  }
}

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
  // Configure logging - using public AddLogLevel method
  ISE_Out::AddLogLevel(ISE_Out::ISE_LogLevel(ISE_Out::LOG));
  ISE_Out::AddLogLevel(ISE_Out::ISE_LogLevel(ISE_Out::DEBUG));
  ISE_Out::AddLogLevel(ISE_Out::ISE_LogLevel(ISE_Out::ERROR));
  
  ISE_Out::Log()<<"CAIF Neural Network Tests"<<std::endl
                <<"======================="<<std::endl;
  
  try
  {
    // Create a simple network
    ISE_Out::Log()<<"1. Creating network..."<<std::endl;
    CAIF_NeuralNetwork network;
    
    // Set input shape - we'll use a 1D input of size 784 to avoid the flattening issue
    // This matches what the flatten layer actually outputs during forward pass
    ShapeVec_t input_shape={1,784};  // Add batch dimension
    network.SetInputShape(input_shape);
    PrintShape(input_shape,"Network input shape");
    ISE_Out::Log()<<"   ✅ Input shape set to ["<<input_shape[0]<<", "<<input_shape[1]<<"]"<<std::endl;
    
    // Add layers
    ISE_Out::Log()<<"2. Adding layers..."<<std::endl;
    
    // Add dense layers directly to the flattened input
    network.AddDenseLayer(128,CAIF_ActivationType_e::ReLU);
    ISE_Out::Log()<<"   ✅ Added dense layer with 128 units and ReLU activation"<<std::endl;
    
    network.AddDropoutLayer(0.3);
    ISE_Out::Log()<<"   ✅ Added dropout layer with 0.3 dropout rate"<<std::endl;
    
    network.AddDenseLayer(64,CAIF_ActivationType_e::ReLU);
    ISE_Out::Log()<<"   ✅ Added dense layer with 64 units and ReLU activation"<<std::endl;
    
    network.AddDenseLayer(10,CAIF_ActivationType_e::Softmax);
    ISE_Out::Log()<<"   ✅ Added output layer with 10 units and softmax activation"<<std::endl;
    
    // Compile the network
    ISE_Out::Log()<<"3. Compiling network..."<<std::endl;
    network.Compile(
                                       CAIF_OptimizerType_e::Adam,
                                       CAIF_LossType_e::CategoricalCrossEntropy,
                                       0.001
                                     );
    ISE_Out::Log()<<"   ✅ Network compiled successfully!"<<std::endl;
    
    // Create a dummy input tensor
    ISE_Out::Log()<<std::endl<<"4. Creating dummy input tensor..."<<std::endl;
    ShapeVec_t input_tensor_shape={10,784}; // 10 samples of flattened 784 features
    CAIF_Tensor input_tensor(network.Framework(),input_tensor_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    input_tensor.FillWithRandom();
    
    PrintShape(input_tensor_shape,"Input tensor shape");
    ISE_Out::Log()<<"   ✅ Created input tensor with shape ["
                  <<input_tensor_shape[0]<<", "
                  <<input_tensor_shape[1]<<"]"<<std::endl;
    
    // Debug network layers
    ISE_Out::Log()<<"   Debug - Network expected input shape: ["
                  <<input_shape[0]<<", "<<input_shape[1]<<"]"<<std::endl;
    
    // Print layer information
    ISE_Out::Log()<<"   Debug - Layer count in network: "<<network.LayerCount()<<std::endl;
    for(uint32_t i=0; i<network.LayerCount(); ++i)
    {
      const auto &layer=network.Layer(i);
      ISE_Out::Log()<<"   Layer "<<i<<" ("<<layer.Description()<<"):"<<std::endl;
      
      auto input_shape=layer.InputShape();
      auto output_shape=layer.OutputShape();
      
      PrintShape(input_shape,"Layer "+std::to_string(i)+" input shape");
      PrintShape(output_shape,"Layer "+std::to_string(i)+" output shape");
      
      ISE_Out::Log()<<"     Input shape: [";
      for(size_t j=0; j<input_shape.size(); ++j)
      {
        if(j>0)
        {
          ISE_Out::Log()<<", ";
        }
        ISE_Out::Log()<<input_shape[j];
      }
      ISE_Out::Log()<<"]"<<std::endl;
      
      ISE_Out::Log()<<"     Output shape: [";
      for(size_t j=0; j<output_shape.size(); ++j)
      {
        if(j>0)
        {
          ISE_Out::Log()<<", ";
        }
        ISE_Out::Log()<<output_shape[j];
      }
      ISE_Out::Log()<<"]"<<std::endl;
    }
    
    // Create a target tensor
    ISE_Out::Log()<<std::endl<<"5. Creating target tensor..."<<std::endl;
    ShapeVec_t target_shape={10,10}; // 10 samples, 10 classes
    CAIF_Tensor target_tensor(network.Framework(),target_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    target_tensor.FillWithRandom();
    
    PrintShape(target_shape,"Target tensor shape");
    ISE_Out::Log()<<"   ✅ Created target tensor with shape ["
                  <<target_shape[0]<<", "
                  <<target_shape[1]<<"]"<<std::endl;
    
    // Make a prediction
    ISE_Out::Log()<<std::endl<<"6. Making prediction..."<<std::endl
                  <<"   Input tensor shape: ["
                  <<input_tensor_shape[0]<<", "
                  <<input_tensor_shape[1]<<"]"<<std::endl;
    
    CAIF_Tensor prediction=network.Predict(input_tensor);
    ShapeVec_t pred_shape=prediction.Shape();
    ISE_Out::Log()<<"   ✅ Prediction successful!"<<std::endl;
    ISE_Out::Log()<<"   Prediction shape: ["
                  <<pred_shape[0]<<", "
                  <<pred_shape[1]<<"]"<<std::endl;
    
    // Test serialization (deprecated — SaveModel/LoadModel moved to
    // CAIF_DeviceNetwork with SafeTensors format, tested in test_safetensors)
    ISE_Out::Log()<<std::endl<<"7. Testing serialization (deprecated path)..."<<std::endl;
    ISE_Out::Log()<<"   Skipped: SaveModel/LoadModel deprecated in CAIF_NeuralNetwork"<<std::endl;
    ISE_Out::Log()<<"   Use CAIF_DeviceNetwork with SafeTensors format instead"<<std::endl;
    ISE_Out::Log()<<"   (Covered by test_safetensors and test_model_serializer)"<<std::endl;
    
    // Add the new test to the test suite
    if(TestDenseWithConvOutput()==false)
    {
      return 1;
    }
    
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::ErrorLog()<<"CAIF Exception with stack trace: "<<e<<std::endl;
    return 1;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Log()<<"   ❌ FAILED - Exception: "<<e.what()<<std::endl;
    return 1;
  }
  
  return 0;
}

bool TestNetworkCreation()
{
  try
  {
    // Create a simple network
    ISE_Out::Log()<<"Testing network creation..."<<std::endl;
    CAIF_NeuralNetwork network;
    
    // Set input shape
    ShapeVec_t batch_input_shape={32,784};  // 32 samples, 784 features
    network.SetInputShape(batch_input_shape);
    
    // Add layers
    network.AddDenseLayer(128,CAIF_ActivationType_e::ReLU);
    
    network.AddDropoutLayer(0.3);
    
    network.AddDenseLayer(64,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(10,CAIF_ActivationType_e::Softmax);
    
    // Create dummy input tensor
    CAIF_Tensor input_tensor(network.Framework(),batch_input_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    input_tensor.FillWithRandom();
    
    // Create target tensor
    ShapeVec_t target_shape={32,10};  // 32 samples, 10 classes
    CAIF_Tensor target_tensor(network.Framework(),target_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    target_tensor.FillWithRandom();
    
    ISE_Out::Log()<<"✅ Network creation test passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Log()<<"❌ FAILED - Exception during network creation: "
                  <<e.what()<<std::endl;
    return false;
  }
}

bool TestNetworkCompilation()
{
  try
  {
    ISE_Out::Log()<<"Testing network compilation..."<<std::endl;
    CAIF_NeuralNetwork network;
    
    // Set input shape
    ShapeVec_t input_shape={32,784};
    network.SetInputShape(input_shape);
    
    // Add layers
    network.AddDenseLayer(128,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(10,CAIF_ActivationType_e::Softmax);
    
    // Compile with different optimizers and loss functions
    network.Compile(
                                       CAIF_OptimizerType_e::Adam,
                                       CAIF_LossType_e::CategoricalCrossEntropy,
                                       0.001
                                     );
    
    network.Compile(
                                  CAIF_OptimizerType_e::SGD,
                                  CAIF_LossType_e::MeanSquaredError,
                                  0.01
                                );
    
    ISE_Out::Log()<<"✅ Network compilation test passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Log()<<"❌ FAILED - Exception during network compilation: "
                  <<e.what()<<std::endl;
    return false;
  }
}

bool TestNetworkPrediction()
{
  try
  {
    ISE_Out::Log()<<"Testing network prediction..."<<std::endl;
    CAIF_NeuralNetwork network;
    
    // Set input shape
    ShapeVec_t input_shape={32,784};
    network.SetInputShape(input_shape);
    
    // Add layers
    network.AddDenseLayer(128,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(10,CAIF_ActivationType_e::Softmax);
    
    // Compile network
    network.Compile(
                                       CAIF_OptimizerType_e::Adam,
                                       CAIF_LossType_e::CategoricalCrossEntropy,
                                       0.001
                                     );
    
    // Create input tensor
    ShapeVec_t batch_input_shape={32,784};
    CAIF_Tensor input_tensor(network.Framework(),batch_input_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    input_tensor.FillWithRandom();
    
    // Create target tensor
    ShapeVec_t target_shape={32,10};
    CAIF_Tensor target_tensor(network.Framework(),target_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    target_tensor.FillWithRandom();
    
    // Make prediction
    CAIF_Tensor prediction=network.Predict(input_tensor);
    ShapeVec_t pred_shape=prediction.Shape();
    
    // Verify prediction shape
    if(pred_shape!=target_shape)
    {
      ISE_Out::Log()<<"❌ FAILED - Prediction shape mismatch"<<std::endl;
      return false;
    }
    
    ISE_Out::Log()<<"✅ Network prediction test passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Log()<<"❌ FAILED - Exception during network prediction: "
                  <<e.what()<<std::endl;
    return false;
  }
}

bool TestNetworkSerialization()
{
  try
  {
    ISE_Out::Log()<<"Testing network serialization..."<<std::endl;
    
    // Create test models directory if it doesn't exist
    std::string error_message;
    if(EnsureDirectoryExists(g_test_models_dir,error_message)==false)
    {
      ISE_Out::Log()<<"❌ FAILED - Could not create test models directory: "
                    <<error_message<<std::endl;
      return false;
    }
    
    // Create a network
    CAIF_NeuralNetwork network;
    
    // Set input shape
    ShapeVec_t input_shape={32,784};
    network.SetInputShape(input_shape);
    
    // Add layers
    network.AddDenseLayer(128,CAIF_ActivationType_e::ReLU);
    
    network.AddDenseLayer(10,CAIF_ActivationType_e::Softmax);
    
    // Compile network
    network.Compile(
                                       CAIF_OptimizerType_e::Adam,
                                       CAIF_LossType_e::CategoricalCrossEntropy,
                                       0.001
                                     );
    
    // Save network
    std::string model_path=std::string(g_test_models_dir)+"/test_network_model";
    network.SaveModel(model_path);
    
    // Load network
    CAIF_NeuralNetwork loaded_network;
    loaded_network.LoadModel(model_path);
    
    // Create input tensor
    ShapeVec_t batch_input_shape={32,784};
    CAIF_Tensor input_tensor(network.Framework(),batch_input_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    input_tensor.FillWithRandom();
    
    // Make predictions with both networks
    CAIF_Tensor prediction=network.Predict(input_tensor);
    CAIF_Tensor loaded_prediction=loaded_network.Predict(input_tensor);
    
    if(prediction.Shape()!=loaded_prediction.Shape())
    {
      ISE_Out::Log()<<"❌ FAILED - Prediction shapes do not match"<<std::endl;
      return false;
    }
    
    ISE_Out::Log()<<"✅ Network serialization test passed"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Log()<<"❌ FAILED - Exception during network serialization: "
                  <<e.what()<<std::endl;
    return false;
  }
}

/**
 * @brief Test dense layers working with convolutional layer output
 */
bool TestDenseWithConvOutput()
{
  try
  {
    ISE_Out::Log()<<"Testing dense with conv output..."<<std::endl;
    CAIF_NeuralNetwork network;
    
    // Set input shape
    ShapeVec_t input_shape={32,784};  // [batch_size, features]
    ISE_Out::Log()<<"Setting input shape: ["<<input_shape[0]<<", "<<input_shape[1]<<"]"<<std::endl;
    
    network.SetInputShape(input_shape);
    
    // Add first dense layer: 784 -> 128
    ISE_Out::Log()<<"Adding dense layer 1: 784 -> 128"<<std::endl;
    network.AddDenseLayer(128,CAIF_ActivationType_e::ReLU);
    
    // Add second dense layer: 128 -> 64
    ISE_Out::Log()<<"Adding dense layer 2: 128 -> 64"<<std::endl;
    network.AddDenseLayer(64,CAIF_ActivationType_e::ReLU);  // 8*8*1=64 outputs
    
    // Add reshape layer to convert [batch, 64] to [batch, 8, 8, 1]
    ISE_Out::Log()<<"Adding reshape layer: [batch, 64] -> [batch, 8, 8, 1]"<<std::endl;
    network.AddReshapeLayer({8,8,1});  // Target shape without batch dimension
    
    // Add convolution layer - it expects 4D input [batch, height, width, channels]
    ISE_Out::Log()<<"Adding conv2d layer: [batch, 8, 8, 1] -> [batch, 6, 6, 16]"<<std::endl;
    network.AddConvolution2DLayer(16,3,1,0,CAIF_ActivationType_e::ReLU);  // 16 3x3 filters, stride 1, no padding
    
    // Compile network
    ISE_Out::Log()<<"Compiling network..."<<std::endl;
    network.Compile(
                                      CAIF_OptimizerType_e::SGD,
                                      CAIF_LossType_e::MeanSquaredError,
                                      0.01f  // Learning rate
                                     );
    
    // Create input tensor
    ISE_Out::Log()<<"Creating input tensor..."<<std::endl;
    std::vector<float> input_data(32*784,1.0f);  // Fill with 1s for easy verification
    CAIF_Tensor input(network.Framework(),{32,784});  // [batch_size, features]
    input.SetData(input_data.data(),input_data.size()*sizeof(float));
    
    // Run prediction
    ISE_Out::Log()<<"Running prediction..."<<std::endl;
    CAIF_Tensor output=network.Predict(input);
    // Verify output shape
    const auto &output_shape=output.Shape();
    if(output_shape.size()!=4 || 
       output_shape[0]!=32 || 
       output_shape[1]!=6 || 
       output_shape[2]!=6 || 
       output_shape[3]!=16)
    {
      ISE_Out::Log()<<"❌ FAILED - Incorrect output shape: ["
                    <<output_shape[0]<<", "
                    <<output_shape[1]<<", "
                    <<output_shape[2]<<", "
                    <<output_shape[3]<<"]"<<std::endl;
      return false;
    }
    
    ISE_Out::Log()<<"✅ PASSED - Dense to conv layer test"<<std::endl;
    return true;
  }
  catch(const std::exception &e)
  {
    ISE_Out::Log()<<"❌ FAILED - Unexpected exception: "<<e.what()<<std::endl;
    return false;
  }
} 
