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

/**
 * @file aif_dense_layer.cpp
 * @brief Implementation of the CAIF_DenseLayer class
 * @author AIF Development Team
 * @version 1.0
 */

#include "caif_dense_layer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <iostream>
#include "ise_lib/ise_out.h"
#include "caif_framework.h"
#include "caif_blas.h"
#include "caif_settings.h"
#include "caif_activation_aware.h"
#include "caif_tensor_backend.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace instance;

namespace
{
  // Helper to convert high-level activation type to backend activation type
  CAIF_TensorBackend::ActivationType_e ToBackendActivation(const CAIF_ActivationType_e activation)
  {
    switch(activation)
    {
      case CAIF_ActivationType_e::Linear:
        return CAIF_TensorBackend::ActivationType_e::Identity;
      case CAIF_ActivationType_e::ReLU:
        return CAIF_TensorBackend::ActivationType_e::ReLU;
      case CAIF_ActivationType_e::Sigmoid:
        return CAIF_TensorBackend::ActivationType_e::Sigmoid;
      case CAIF_ActivationType_e::Tanh:
        return CAIF_TensorBackend::ActivationType_e::Tanh;
      case CAIF_ActivationType_e::LeakyReLU:
        return CAIF_TensorBackend::ActivationType_e::LeakyReLU;
      case CAIF_ActivationType_e::ELU:
        return CAIF_TensorBackend::ActivationType_e::ELU;
      case CAIF_ActivationType_e::GELU:
        return CAIF_TensorBackend::ActivationType_e::GELU;
      case CAIF_ActivationType_e::Swish:
        return CAIF_TensorBackend::ActivationType_e::Swish;
      default:
        return CAIF_TensorBackend::ActivationType_e::Identity;
    }
  }
}//end CAIF_TensorBackend namespace

CAIF_DenseLayer::CAIF_DenseLayer(
                               CAIF_Framework &framework,
                               const uint32_t units,
                               const CAIF_ActivationType_e activation,
                               const bool use_bias
                              ):CAIF_Layer(framework),
                               _units(units),
                               _activation(activation),
                               _use_bias(use_bias),
                               _weights(framework,{1,1},CAIF_DataType::CAIF_DataType_e::Float32),
                               _bias(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
                               _weight_gradients(framework,{1,1},CAIF_DataType::CAIF_DataType_e::Float32),
                               _bias_gradients(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
                               _last_input(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
                               _last_linear(framework),
                               _last_output(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32)
{
  SetInitialized(false);
}

CAIF_DenseLayer::CAIF_DenseLayer(const CAIF_DenseLayer &other):CAIF_Layer(other),
                                                            _units(other._units),
                                                            _activation(other._activation),
                                                            _use_bias(other._use_bias),
                                                            _weights(other._weights),
                                                            _bias(other._bias),
                                                            _weight_gradients(other._weight_gradients),
                                                            _bias_gradients(other._bias_gradients),
                                                            _last_input(other._last_input),
                                                            _last_linear(other._last_linear),
                                                            _last_output(other._last_output)
{
}

CAIF_DenseLayer::CAIF_DenseLayer(CAIF_DenseLayer &&other):CAIF_Layer(std::move(other)),
                                                       _units(other._units),
                                                       _activation(other._activation),
                                                       _use_bias(other._use_bias),
                                                       _weights(std::move(other._weights)),
                                                       _bias(std::move(other._bias)),
                                                       _weight_gradients(std::move(other._weight_gradients)),
                                                       _bias_gradients(std::move(other._bias_gradients)),
                                                       _last_input(std::move(other._last_input)),
                                                       _last_linear(std::move(other._last_linear)),
                                                       _last_output(std::move(other._last_output))
{
}

CAIF_DenseLayer &CAIF_DenseLayer::operator=(const CAIF_DenseLayer &other)
{
  if(this!=&other)
  {
    CAIF_Layer::operator=(other);
    _units=other._units;
    _activation=other._activation;
    _use_bias=other._use_bias;
    _weights=other._weights;
    _bias=other._bias;
    _weight_gradients=other._weight_gradients;
    _bias_gradients=other._bias_gradients;
    _last_input=other._last_input;
    _last_output=other._last_output;
    _last_linear=other._last_linear;
  }
  return *this;
}

CAIF_DenseLayer &CAIF_DenseLayer::operator=(CAIF_DenseLayer &&other)
{
  if(this!=&other)
  {
    CAIF_Layer::operator=(std::move(other));
    _units=other._units;
    _activation=other._activation;
    _use_bias=other._use_bias;
    _weights=std::move(other._weights);
    _bias=std::move(other._bias);
    _weight_gradients=std::move(other._weight_gradients);
    _bias_gradients=std::move(other._bias_gradients);
    _last_input=std::move(other._last_input);
    _last_output=std::move(other._last_output);
    _last_linear=std::move(other._last_linear);
  }
  return *this;
}

CAIF_Tensor CAIF_DenseLayer::Forward( const CAIF_Tensor &input, const bool training)
{
  (void)training;
  const bool gpu=Framework().IsGPUAccelerated();
  if(gpu==false)
  {
    DbgLog()<<"[DEBUG] CAIF_DenseLayer::Forward - Starting forward pass\n";
    DbgLog()<<"[DEBUG] Input tensor: "<<input.ToString()<<"\n";
    DbgLog()<<"[DEBUG] Weights tensor: "<<_weights.ToString()<<"\n";
    DbgLog()<<"[DEBUG] Bias tensor: "<<_bias.ToString()<<"\n";
  }
  
  if(IsInitialized()==false)
  {
    ErrorLog()<<"[ERROR] Dense layer not initialized\n";
    THROW_CAIFE("Dense layer not initialized");
  }
  
  const auto &input_shape=input.Shape();
  
  // Validate input shape
  if(input_shape.size()<2)
  {
    ErrorLog()<<"[ERROR] Input shape must have at least 2 dimensions [batch, features], got: "
             <<input_shape.size()
             <<"\n";
    THROW_CAIFE("Input shape must have at least 2 dimensions [batch, features]");
  }
  
  try
  {
    CAIF_Framework &framework=Framework();
    CAIF_Tensor reshaped_input(framework);
    
    // For 2D input [batch_size, features], reshape is not needed
    if(input_shape.size()==2)
    {
      reshaped_input=input;
      if(gpu==false){DbgLog()<<"[DEBUG] Using 2D input directly: "<<reshaped_input.ToString()<<"\n";}
    }
    else if(input_shape.size()==g_caif_conv_dimensions)
    {
      // For 4D input from conv layer [batch, height, width, channels]
      // Flatten to [batch_size, height * width * channels]
      uint32_t total_features=input_shape[g_caif_conv_height_idx]*
                              input_shape[g_caif_conv_width_idx]*
                              input_shape[g_caif_conv_channel_idx];
      std::vector<uint32_t> new_shape={input_shape[g_caif_conv_batch_idx],total_features};
      reshaped_input=input.Reshape(new_shape);
      
      if(gpu==false){DbgLog()<<"[DEBUG] Flattened 4D input to: "<<reshaped_input.ToString()<<"\n";}
    }
    else
    {
      // For other higher dimensional input, flatten all dimensions except the first one
      uint32_t total_features=1;
      for(size_t i=1; i<input_shape.size(); ++i)
      {
        total_features*=input_shape[i];
      }
      
      // Reshape to [batch_size, total_features]
      std::vector<uint32_t> new_shape={input_shape[0], total_features};
      reshaped_input=input.Reshape(new_shape);
      
      if(gpu==false)
      {
        DbgLog()<<"[DEBUG] Flattened "<<input_shape.size()
                <<"D input to: "<<reshaped_input.ToString()<<"\n";
      }
    }
    
    // Perform y = X(BxI) * W(IxU) + b  via backend framework for GPU acceleration
    if(gpu==false){DbgLog()<<"[DEBUG] Performing matrix multiplication via Framework (X * W)\n";}
    
    CAIF_Tensor output=framework.MatrixMultiply(reshaped_input,_weights);
    
    if(gpu==false){DbgLog()<<"[DEBUG] MatMul result: "<<output.ToString()<<"\n";}

    // Prefer fused path only on CPU (GPU fused path would sync host buffers)
    const bool can_fuse=(!gpu &&
                         output.Type()==CAIF_DataType::CAIF_DataType_e::Float32 &&
                         _activation!=CAIF_ActivationType_e::Softmax);
    if(can_fuse==true)
    {
      _last_linear=CAIF_Tensor(output.Framework(),output.Shape(),output.Type());
      Fuse(output,_last_linear);
      _last_input=reshaped_input;
      _last_output=output;
      
      if(gpu==false){DbgLog()<<"[DEBUG] Final output (fused): "<<_last_output.ToString()<<"\n";}
      return _last_output;
    }

    // Fallback: build linear then apply activation (Softmax case)
    CAIF_Tensor linear=output;
    if(_use_bias==true)
    {
      const auto &shape=linear.Shape();
      const uint32_t batch_size=shape[0];
      const uint32_t units=shape[1];
      const float *bias_data=_bias.ConstData<float>();
      float *lin_data=linear.MutableData<float>();
      for(uint32_t b=0;b<batch_size;++b)
      {
        float *row=lin_data+static_cast<size_t>(b)*units;
        for(uint32_t u=0;u<units;++u)
        {
          row[u]+=bias_data[u];
        }
      }
    }
    CAIF_Tensor activation_out(framework);
    if(_activation==CAIF_ActivationType_e::Softmax)
    {
      // Use framework for GPU-accelerated softmax
      activation_out=framework.SoftmaxForward(linear);
    }
    else
    {
      // Use framework for other activations
      activation_out=framework.ActivationForward(linear,ToBackendActivation(_activation));
    }
    _last_input=reshaped_input;
    _last_linear=linear;
    _last_output=activation_out;
    
    return _last_output;
  }
  catch(const std::exception &e)
  {
    ErrorLog()<<"[ERROR] Forward pass failed: "<<e.what()<<"\n";
    THROW_CAIFE((std::string("Forward pass failed: ")+e.what()).c_str());
  }
}

CAIF_Tensor CAIF_DenseLayer::Backward(const CAIF_Tensor &gradient)
{
  if(IsInitialized()==false)
  {
    THROW_CAIFE("Dense layer not initialized");
  }
  
  try
  {
    const bool gpu=Framework().IsGPUAccelerated();
    if(gpu==false)
    {
      DbgLog()<<"[DEBUG] CAIF_DenseLayer::Backward - Starting backward pass\n";
      DbgLog()<<"[DEBUG] Input gradient: "<<gradient.ToString()<<"\n";
      DbgLog()<<"[DEBUG] Last input: "<<_last_input.ToString()<<"\n";
      DbgLog()<<"[DEBUG] Last output: "<<_last_output.ToString()<<"\n";
      DbgLog()<<"[DEBUG] Weights: "<<_weights.ToString()<<"\n";
      if(_use_bias)
      {
        DbgLog()<<"[DEBUG] Bias: "<<_bias.ToString()<<"\n";
      }
    }
    
    // Apply activation derivative
    // Special-case: Softmax + CategoricalCrossEntropy uses simplified gradient from loss
    CAIF_Framework &framework=Framework();
    CAIF_Tensor linear_gradient(framework);
    
    if(_activation==CAIF_ActivationType_e::Softmax)
    {
      // gradient is already (predictions - targets) / batch
      linear_gradient=gradient;
    }
    else if(_activation==CAIF_ActivationType_e::Linear)
    {
      // Linear derivative is 1 - gradient passes through unchanged
      linear_gradient=gradient;
    }
    else
    {
      // Use pre-activation input for derivatives that depend on x (ReLU/LeakyReLU/ELU/GELU/Swish)
      // and post-activation output for those that depend on y (Sigmoid/Tanh)
      const bool uses_output=(_activation==CAIF_ActivationType_e::Sigmoid||
                              _activation==CAIF_ActivationType_e::Tanh);
      if(uses_output==true)
      {
        linear_gradient=ApplyActivationDerivative(_last_output,gradient,_activation);
      }
      else
      {
        linear_gradient=ApplyActivationDerivative(_last_linear,gradient,_activation);
      }
    }
    
    if(gpu==false){DbgLog()<<"[DEBUG] After activation derivative: "<<linear_gradient.ToString()<<"\n";}
    
    // Calculate weight gradients using GPU-accelerated matmul with transpose: dW = X^T * G
    _last_input.EnsureBackendData();
    linear_gradient.EnsureBackendData();
    
    _weight_gradients=framework.MatrixMultiplyEx(
                                                _last_input,
                                                linear_gradient,
                                                CAIF_TensorBackend::Transpose_e::Trans,
                                                CAIF_TensorBackend::Transpose_e::NoTrans
                                               );
    
    if(gpu==false){DbgLog()<<"[DEBUG] Weight gradients: "<<_weight_gradients.ToString()<<"\n";}
    
    // Calculate bias gradients if bias is used
    if(_use_bias==true)
    {
      const auto &lg_shape=linear_gradient.Shape();
      const uint32_t units_dim=lg_shape[1];
      _bias_gradients=CAIF_Tensor(linear_gradient.Framework(),{units_dim},linear_gradient.Type());
      const float *lg=linear_gradient.ConstData<float>();
      float *bg=_bias_gradients.MutableData<float>();
      std::fill(bg,bg+units_dim,0.0f);
      const uint32_t batch_dim=lg_shape[0];
      for(uint32_t b=0;b<batch_dim;++b)
      {
        const float *row=lg+static_cast<size_t>(b)*units_dim;
        for(uint32_t u=0;u<units_dim;++u)
        {
          bg[u]+=row[u];
        }
      }
      DbgLog()<<"[DEBUG] Bias gradients: "<<_bias_gradients.ToString()<<"\n";
      // Diagnostics disabled for performance
    }
    
    // Calculate input gradient using GPU-accelerated matmul with transpose: dX = G * W^T
    CAIF_Tensor input_gradient=framework.MatrixMultiplyEx(
                                                         linear_gradient,
                                                         _weights,
                                                         CAIF_TensorBackend::Transpose_e::NoTrans,
                                                         CAIF_TensorBackend::Transpose_e::Trans
                                                        );
    DbgLog()<<"[DEBUG] Input gradient: "<<input_gradient.ToString()<<"\n";
    // Diagnostics disabled for performance
    
    return input_gradient;
  }
  CCAIF_CATCH_BLOCK();
}

void CAIF_DenseLayer::Initialize( const std::vector<uint32_t> &input_shape, const uint32_t seed)
{
  if(input_shape.empty()==true)
  {
    THROW_CAIFE("Input shape cannot be empty");
  }
  
  SetInputShape(input_shape);
  
  // Calculate input size (product of all dimensions except the first/batch dimension)
  uint32_t input_size=1;
  for(size_t i=1; i<input_shape.size(); ++i)  // Start from 1 to skip batch dimension
  {
    input_size*=input_shape[i];
  }
  
  DbgLog()<<"DenseLayer::Initialize - input_shape=[";
  for(size_t i=0; i<input_shape.size(); ++i)
  {
    if(i>0) DbgLog()<<", ";
    DbgLog()<<input_shape[i];
  }
  DbgLog()<<"], input_size="<<input_size<<", units="<<_units<<std::endl;
  
  // Validate input shape
  if(input_shape.size()<2)
  {
    THROW_CAIFE("Input shape must have at least 2 dimensions [batch, features]");
  }
  
  // For 4D input (from conv layer), validate shape [batch, height, width, channels]
  if(input_shape.size()==g_caif_conv_dimensions)
  {
    // Input shape is valid, we'll flatten it during Forward pass
    DbgLog()<<"DenseLayer::Initialize - Detected 4D input from convolutional layer"<<std::endl;
  }
  else if(input_shape.size()>2)
  {
    // For other higher dimensional inputs, warn but proceed
    DbgLog()<<"DenseLayer::Initialize - Warning: Input has "
              <<input_shape.size() 
              <<" dimensions, will flatten all except batch dimension"
              <<std::endl;
  }
  
  // Initialize weights with correct dimensions for flattened input
  InitializeWeights(input_size,_units,seed);
  
  // Set output shape - will be [batch_size, units]
  SetOutputShape({input_shape[0],_units});
  SetInitialized(true);
  
  return;
}

std::vector<uint32_t> CAIF_DenseLayer::CalculateOutputShape( const std::vector<uint32_t> &input_shape)const
{
  if(input_shape.empty())
  {
    THROW_CAIFE("Input shape cannot be empty");
  }
  
  std::vector<uint32_t> output_shape=input_shape;
  output_shape.back()=_units;  // Replace last dimension with number of units
  
  return output_shape;
}

std::unique_ptr<CAIF_Layer> CAIF_DenseLayer::Clone()const
{
  // Framework reference is copied from this layer via copy constructor
  return std::make_unique<CAIF_DenseLayer>(*this);
}

std::vector<CAIF_Tensor> CAIF_DenseLayer::Parameters()const
{
  std::vector<CAIF_Tensor> params;
  params.push_back(_weights);
  if(_use_bias==true)
  {
    params.push_back(_bias);
  }
  return params;
}

std::vector<CAIF_Tensor> CAIF_DenseLayer::ParameterGradients()const
{
  std::vector<CAIF_Tensor> grads;
  grads.push_back(_weight_gradients);
  if(_use_bias==true)
  {
    grads.push_back(_bias_gradients);
  }
  return grads;
}

size_t CAIF_DenseLayer::ParameterCount()const
{
  if(_use_bias==true)
  {
    return 2;
  }
  return 1;
}

CAIF_Tensor &CAIF_DenseLayer::ParameterRef(const size_t index)
{
  try
  {
    if(index==0)
    {
      return _weights;
    }
    if(index==1&&_use_bias==true)
    {
      return _bias;
    }
    THROW_CAIFE("Parameter index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_Tensor &CAIF_DenseLayer::ParameterRef(const size_t index)const
{
  try
  {
    if(index==0)
    {
      return _weights;
    }
    if(index==1&&_use_bias==true)
    {
      return _bias;
    }
    THROW_CAIFE("Parameter index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor &CAIF_DenseLayer::GradientRef(const size_t index)
{
  try
  {
    if(index==0)
    {
      return _weight_gradients;
    }
    if(index==1&&_use_bias==true)
    {
      return _bias_gradients;
    }
    THROW_CAIFE("Gradient index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_Tensor &CAIF_DenseLayer::GradientRef(const size_t index)const
{
  try
  {
    if(index==0)
    {
      return _weight_gradients;
    }
    if(index==1&&_use_bias==true)
    {
      return _bias_gradients;
    }
    THROW_CAIFE("Gradient index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DenseLayer::UpdateParameters(const std::vector<CAIF_Tensor> &new_parameters)
{
  if(_use_bias==true&&new_parameters.size()!=2)
  {
    THROW_CAIFE("Expected 2 parameters (weights and bias)");
  }
  if(_use_bias==false&&new_parameters.size()!=1)
  {
    THROW_CAIFE("Expected 1 parameter (weights only)");
  }
  
  _weights=new_parameters[0];
  if(_use_bias==true&&new_parameters.size()>1)
  {
    _bias=new_parameters[1];
  }
  
  return;
}

void CAIF_DenseLayer::ResetParameters(const uint32_t seed)
{
  if(IsInitialized()==false)
  {
    THROW_CAIFE("Layer not initialized");
  }
  
  // Calculate input size as product of all dimensions except the first (batch) dimension
  uint32_t input_size=1;
  const auto &input_shape=InputShape();
  for(size_t i=1; i<input_shape.size(); ++i)  // Start from 1 to skip batch dimension
  {
    input_size*=input_shape[i];
  }
  
  InitializeWeights(input_size,_units,seed);
}

std::string CAIF_DenseLayer::Description()const
{
  std::ostringstream oss;
  oss<<"Dense Layer (units="<<_units;
  oss<<", activation=";
  switch(_activation)
  {
    case CAIF_ActivationType_e::Linear:oss<<"Linear";
      break;
    case CAIF_ActivationType_e::ReLU:oss<<"ReLU";
      break;
    case CAIF_ActivationType_e::Sigmoid:oss<<"Sigmoid";
      break;
    case CAIF_ActivationType_e::Tanh:oss<<"Tanh";
      break;
    case CAIF_ActivationType_e::Softmax:oss<<"Softmax";
      break;
    case CAIF_ActivationType_e::LeakyReLU:oss<<"LeakyReLU";
      break;
    case CAIF_ActivationType_e::ELU:oss<<"ELU";
      break;
    case CAIF_ActivationType_e::GELU:oss<<"GELU";
      break;
    case CAIF_ActivationType_e::Swish:oss<<"Swish";
      break;
    default:oss<<"Unknown";
      break;
  }
  oss<<", bias=";
  if(_use_bias==true)
  {
    oss<<"true";
  }
  else
  {
    oss<<"false";
  }
  oss<<")";
  return oss.str();
}

void CAIF_DenseLayer::SetBias(const float value)
{
  if(_use_bias==false)
  {
    THROW_CAIFE("Dense layer has no bias enabled");
  }
  if(_bias.Shape().empty())
  {
    THROW_CAIFE("Bias tensor not initialized");
  }
  float *bd=_bias.MutableData<float>();
  const uint32_t n=_bias.NumElements();
  for(uint32_t i=0;i<n;++i)
  {
    bd[i]=value;
  }
  return;
}

CAIF_Tensor CAIF_DenseLayer::ApplyActivation(const CAIF_Tensor &input,const CAIF_ActivationType_e activation)const
{
  switch(activation)
  {
    case CAIF_ActivationType_e::Linear:
      return input.Linear();
    case CAIF_ActivationType_e::ReLU:
      return input.ReLU();
    case CAIF_ActivationType_e::Sigmoid:
      return input.Sigmoid();
    case CAIF_ActivationType_e::Tanh:
      return input.Tanh();
    case CAIF_ActivationType_e::Softmax:
      return input.Softmax();
    case CAIF_ActivationType_e::LeakyReLU:
      return input.LeakyReLU(0.01f);
    case CAIF_ActivationType_e::ELU:
      return input.ELU(1.0f);
    case CAIF_ActivationType_e::GELU:
      return input.GELU();
    case CAIF_ActivationType_e::Swish:
      return input.Swish();
    default:
      THROW_CAIFE("Unknown activation function");
  }
}

CAIF_Tensor CAIF_DenseLayer::ApplyActivationDerivative(const CAIF_Tensor &input,
                                                     const CAIF_Tensor &gradient,
                                                     const CAIF_ActivationType_e activation
                                                    )const
{
  switch(activation)
  {
    case CAIF_ActivationType_e::Linear:
      return input.LinearDerivative(gradient);
    case CAIF_ActivationType_e::ReLU:
      return input.ReLUDerivative(gradient);
    case CAIF_ActivationType_e::Sigmoid:
      return input.SigmoidDerivative(gradient);
    case CAIF_ActivationType_e::Tanh:
      return input.TanhDerivative(gradient);
    case CAIF_ActivationType_e::Softmax:
      return input.SoftmaxDerivative(gradient);
    case CAIF_ActivationType_e::LeakyReLU:
      return input.LeakyReLUDerivative(gradient,0.01f);
    case CAIF_ActivationType_e::ELU:
      return input.ELUDerivative(gradient,1.0f);
    case CAIF_ActivationType_e::GELU:
      return input.GELUDerivative(gradient);
    case CAIF_ActivationType_e::Swish:
      return input.SwishDerivative(gradient);
    default:
      THROW_CAIFE("Unknown activation function");
  }
}

void CAIF_DenseLayer::InitializeWeights(const uint32_t input_size,
                                       const uint32_t output_size,
                                       const uint32_t seed
                                      )
{
  // Activation-aware initialization to match typical best practices and improve stability.
  // - Xavier uniform for Linear/Sigmoid/Tanh/Softmax (gain=1)
  // - Kaiming (He) uniform for ReLU-family with appropriate gain
  std::mt19937 rng(seed);
  const float fan_in=static_cast<float>(input_size);
  const float bound=CAIF_ActivationAware::UniformBound(
                                                      _activation,
                                                      input_size,
                                                      output_size,
                                                      CAIF_Settings::ActivationAwareInit()
                                                     );
  std::uniform_real_distribution<float> dist(-bound,bound);
  
  // Initialize weights
  CAIF_Framework &framework=Framework();
  std::vector<uint32_t> weight_shape={input_size,output_size};
  _weights=CAIF_Tensor(framework,weight_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  _weight_gradients=CAIF_Tensor(framework,weight_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  
  float *weight_data=_weights.MutableData<float>();
  for(uint32_t i=0;i<input_size*output_size;++i)
  {
    weight_data[i]=dist(rng);
  }
  
  // Initialize bias to uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) to mirror PyTorch nn.Linear
  if(_use_bias==true)
  {
    float safe_fan_in_for_bias=fan_in;
    if(fan_in<=0.0f)
    {
      safe_fan_in_for_bias=1.0f;
    }
    const float bias_bound=1.0f/std::sqrt(safe_fan_in_for_bias);
    std::uniform_real_distribution<float> bias_dist(-bias_bound,bias_bound);
    
    std::vector<uint32_t> bias_shape={output_size};
    _bias=CAIF_Tensor(framework,bias_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    _bias_gradients=CAIF_Tensor(framework,bias_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    
    float *bias_data=_bias.MutableData<float>();
    for(uint32_t i=0;i<output_size;++i)
    {
      bias_data[i]=bias_dist(rng);
    }
  }
  
  return;
}
float CAIF_DenseLayer::PointWise(const CAIF_ActivationType_e activation,const float value)
{
  switch(activation)
  {
    case CAIF_ActivationType_e::ReLU:
    {
      if(value>0.0f)
      {
        return value;
      }
      else
      {
        return 0.0f;
      }
    }
    case CAIF_ActivationType_e::Sigmoid:
    {
      const float denom=1.0f+std::exp(-value);
      return 1.0f/denom;
    }
    case CAIF_ActivationType_e::Tanh:
    {
      return std::tanh(value);
    }
    case CAIF_ActivationType_e::ELU:
    {
      const float alpha=1.0f;
      if(value>0.0f)
      {
        return value;
      }
      else
      {
        return alpha*(std::exp(value)-1.0f);
      }
    }
    case CAIF_ActivationType_e::LeakyReLU:
    {
      const float alpha=0.01f;
      if(value>0.0f)
      {
        return value;
      }
      else
      {
        return alpha*value;
      }
    }
    case CAIF_ActivationType_e::GELU:
    {
      const float x=value;
      const float x3=x*x*x;
      const float sqrt_2_pi=std::sqrt(2.0f/g_caif_pi);
      const float inner=sqrt_2_pi*(x+0.044715f*x3);
      return 0.5f*x*(1.0f+std::tanh(inner));
    }
    case CAIF_ActivationType_e::Swish:
    {
      const float denom=1.0f+std::exp(-value);
      return value/denom;
    }
    default:
    {
      return value;
    }
  }
}

void CAIF_DenseLayer::Fuse(
                          CAIF_Tensor &output,
                          CAIF_Tensor &out_linear
                         )const
{
  const auto &out_shape=output.Shape();
  const uint32_t batch_size=out_shape[0];
  const uint32_t units=out_shape[1];
  float *out_data=output.MutableData<float>();
  float *lin_data=out_linear.MutableData<float>();
  const float *bias_data=nullptr;
  if(_use_bias==true)
  {
    bias_data=_bias.ConstData<float>();
  }

  if(_activation==CAIF_ActivationType_e::Linear)
  {
    if(bias_data!=nullptr)
    {
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(uint32_t b=0;b<batch_size;++b)
      {
        float *row=out_data+static_cast<size_t>(b)*units;
        float *lrow=lin_data+static_cast<size_t>(b)*units;
        for(uint32_t u=0;u<units;++u)
        {
          const float v=row[u]+bias_data[u];
          lrow[u]=v;
          row[u]=v;
        }
      }
    }
    else
    {
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(uint32_t b=0;b<batch_size;++b)
      {
        float *row=out_data+static_cast<size_t>(b)*units;
        float *lrow=lin_data+static_cast<size_t>(b)*units;
        for(uint32_t u=0;u<units;++u)
        {
          const float v=row[u];
          lrow[u]=v;
        }
      }
    }
    return;
  }

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(uint32_t b=0;b<batch_size;++b)
  {
    float *row=out_data+static_cast<size_t>(b)*units;
    float *lrow=lin_data+static_cast<size_t>(b)*units;
    for(uint32_t u=0;u<units;++u)
    {
      const float base=row[u];
      float v=base;
      if(bias_data!=nullptr)
      {
        v=v+bias_data[u];
      }
      lrow[u]=v;
      row[u]=PointWise(_activation,v);
    }
  }
  
  return;
} 
