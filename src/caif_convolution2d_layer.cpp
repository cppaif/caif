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
 * @file aif_convolution2d_layer.cpp
 * @brief Implementation of the CAIF_Convolution2DLayer class
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_convolution2d_layer.h"
#include "caif_constants.h"
#include "caif_framework.h"
#include <sstream>
#include <random>
#include <cmath>
#include <algorithm>

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

CAIF_Convolution2DLayer::CAIF_Convolution2DLayer(
                                               CAIF_Framework &framework,
                                               const uint32_t filters,
                                               const uint32_t kernel_size,
                                               const uint32_t stride,
                                               const uint32_t padding,
                                               const CAIF_ActivationType_e activation,
                                               const bool use_bias
                                              )
  :CAIF_Layer(framework),
   _filters(filters),
   _kernel_size(kernel_size),
   _stride(stride),
   _padding(padding),
   _activation(activation),
   _use_bias(use_bias),
   _input_channels(0),
   _weights(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
   _bias(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
   _weight_gradients(framework),
   _bias_gradients(framework),
   _last_input(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
   _last_pre_activation(framework),
   _last_output(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
   _last_activation_gradient(framework)
{
  SetInitialized(false);
  
  // Enhanced parameter validation
  if(_filters==0)
  {
    throw std::invalid_argument("Number of filters must be greater than 0");
  }
  if(_kernel_size==0||_kernel_size%2==0)
  {
    throw std::invalid_argument("Kernel size must be odd and greater than 0");
  }
  if(_stride==0)
  {
    throw std::invalid_argument("Stride must be greater than 0");
  }
  if(_kernel_size>g_caif_max_kernel_size)
  {
    throw std::invalid_argument("Kernel size exceeds maximum allowed size");
  }
  if(_filters>g_caif_max_filters)
  {
    throw std::invalid_argument("Number of filters exceeds maximum allowed");
  }
}

CAIF_Convolution2DLayer::CAIF_Convolution2DLayer(const CAIF_Convolution2DLayer &other)
  :CAIF_Layer(other),
   _filters(other._filters),
   _kernel_size(other._kernel_size),
   _stride(other._stride),
   _padding(other._padding),
   _activation(other._activation),
   _use_bias(other._use_bias),
   _input_channels(other._input_channels),
   _weights(other._weights),
   _bias(other._bias),
   _weight_gradients(other._weight_gradients),
   _bias_gradients(other._bias_gradients),
   _last_input(other._last_input),
   _last_pre_activation(other._last_pre_activation),
   _last_output(other._last_output),
   _last_activation_gradient(other._last_activation_gradient)
{
}

CAIF_Convolution2DLayer::CAIF_Convolution2DLayer(CAIF_Convolution2DLayer &&other)
  :CAIF_Layer(std::move(other)),
   _filters(other._filters),
   _kernel_size(other._kernel_size),
   _stride(other._stride),
   _padding(other._padding),
   _activation(other._activation),
   _use_bias(other._use_bias),
   _input_channels(other._input_channels),
   _weights(std::move(other._weights)),
   _bias(std::move(other._bias)),
   _weight_gradients(std::move(other._weight_gradients)),
   _bias_gradients(std::move(other._bias_gradients)),
   _last_input(std::move(other._last_input)),
   _last_pre_activation(std::move(other._last_pre_activation)),
   _last_output(std::move(other._last_output)),
   _last_activation_gradient(std::move(other._last_activation_gradient))
{
}

CAIF_Convolution2DLayer &CAIF_Convolution2DLayer::operator=(const CAIF_Convolution2DLayer &other)
{
  if(this!=&other)
  {
    CAIF_Layer::operator=(other);
    _filters=other._filters;
    _kernel_size=other._kernel_size;
    _stride=other._stride;
    _padding=other._padding;
    _activation=other._activation;
    _use_bias=other._use_bias;
    _input_channels=other._input_channels;
    _weights=other._weights;
    _bias=other._bias;
    _last_input=other._last_input;
    _last_output=other._last_output;
  }
  return *this;
}

CAIF_Convolution2DLayer &CAIF_Convolution2DLayer::operator=(CAIF_Convolution2DLayer &&other)
{
  if(this!=&other)
  {
    CAIF_Layer::operator=(std::move(other));
    _filters=other._filters;
    _kernel_size=other._kernel_size;
    _stride=other._stride;
    _padding=other._padding;
    _activation=other._activation;
    _use_bias=other._use_bias;
    _input_channels=other._input_channels;
    _weights=std::move(other._weights);
    _bias=std::move(other._bias);
    _last_input=std::move(other._last_input);
    _last_output=std::move(other._last_output);
  }
  return *this;
}

CAIF_Tensor CAIF_Convolution2DLayer::Forward(const CAIF_Tensor &input,const bool training)
{
  // Enhanced input validation
  if(IsInitialized()==false)
  {
    THROW_CAIFE("Convolution layer not initialized");
  }
  
  const auto &input_shape=input.Shape();
  if(input_shape.size()!=4)
  {
    THROW_CAIFE("Convolution requires 4D input [batch, height, width, channels]");
  }
  
  // Validate input dimensions
  if(input_shape[0]==0||input_shape[1]==0||input_shape[2]==0||input_shape[3]==0)
  {
    THROW_CAIFE("Input dimensions must be greater than 0");
  }
  
  // Check minimum input size requirements
  const uint32_t min_input_size=_kernel_size;
  if(input_shape[1]<min_input_size||input_shape[2]<min_input_size)
  {
    std::ostringstream oss;
    oss<<"Input spatial dimensions ("<<input_shape[1]<<"x"<<input_shape[2]
       <<") too small for kernel size "<<_kernel_size;
    THROW_CAIFE(oss.str().c_str());
  }
  
  if(input.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("Convolution currently only supports Float32 data type");
  }
  
  // Validate input channels match expected
  if(input_shape[3]!=_input_channels&&_input_channels!=0)
  {
    std::ostringstream oss;
    oss<<"Input channels ("<<input_shape[3]<<") do not match expected ("<<_input_channels<<")";
    THROW_CAIFE(oss.str().c_str());
  }
  
  // Store input for backward pass (only if training)
  if(training==true)
  {
    _last_input=input;
  }
  
  // Create output tensor with enhanced error handling
  const auto output_shape=CalculateOutputShape(input_shape);
  
  // Validate output shape makes sense
  if(output_shape[1]==0||output_shape[2]==0)
  {
    THROW_CAIFE("Computed output spatial dimensions are zero - check stride/padding configuration");
  }
  
  CAIF_Framework &framework=Framework();
  CAIF_Tensor output(framework,output_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  
  // Perform convolution via backend framework
  {
    CAIF_Framework &framework=Framework();
    CAIF_Tensor backend_out=framework.Convolution2D(input,
                                                   _weights,
                                                   _stride,
                                                   _stride,
                                                   _padding,
                                                   _padding
                                                  );
    output=backend_out;
  }
  
  // Add bias if enabled with enhanced error handling
  if(_use_bias==true)
  {
    AddBias(output);
  }
  
  // Store pre-activation for backward pass (only if training)
  if(training==true)
  {
    _last_pre_activation=output;
  }
  
  // Apply activation
  CAIF_Tensor activated_output=ApplyActivation(output);
  
  // Store output for backward pass (only if training)
  if(training==true)
  {
    _last_output=activated_output;
  }
  
  return activated_output;
}

CAIF_Tensor CAIF_Convolution2DLayer::Backward(const CAIF_Tensor &gradient)
{
  if(IsInitialized()==false)
  {
    THROW_CAIFE("Convolution layer not initialized");
  }
  
  // Enhanced gradient validation
  if(gradient.Shape().size()!=4)
  {
    THROW_CAIFE("Gradient must be 4D tensor");
  }
  
  const auto &grad_shape=gradient.Shape();
  const auto &expected_shape=_last_output.Shape();
  
  if(grad_shape!=expected_shape)
  {
    std::ostringstream oss;
    oss<<"Gradient shape mismatch. Expected: [";
    for(size_t i=0;i<expected_shape.size();++i)
    {
      if(i>0)oss<<", ";
      oss<<expected_shape[i];
    }
    oss<<"], Got: [";
    for(size_t i=0;i<grad_shape.size();++i)
    {
      if(i>0)oss<<", ";
      oss<<grad_shape[i];
    }
    oss<<"]";
    THROW_CAIFE(oss.str().c_str());
  }
  
  // Apply activation derivative
  CAIF_Framework &framework=Framework();
  CAIF_Tensor activation_gradient(framework);
  if(_activation==CAIF_ActivationType_e::Linear)
  {
    // Linear derivative is 1 - gradient passes through unchanged
    activation_gradient=gradient;
  }
  else
  {
    // For CUDA backend, use framework.ActivationBackward() to leverage cuDNN
    // For CPU backends, use direct tensor operations to avoid overhead
    const auto backend_type=framework.CurrentBackend();
    if(backend_type==CAIF_TensorBackend::BackendType_e::CUDA)
    {
      activation_gradient=framework.ActivationBackward(
                                                       gradient,
                                                       _last_pre_activation,
                                                       _last_output,
                                                       ToBackendActivation(_activation)
                                                      );
    }
    else
    {
      activation_gradient=ApplyActivationDerivative(_last_output,gradient);
    }
  }
  // Cache activation gradient for attribution methods like Grad-CAM
  _last_activation_gradient=activation_gradient;
  
  // Compute and store parameter gradients for optimizer
  _weight_gradients=ComputeWeightGradients(_last_input,activation_gradient);
  
  if(_use_bias==true)
  {
    // Compute bias gradients by summing over spatial dimensions
    _bias_gradients=ComputeBiasGradients(activation_gradient);
  }
  
  // Compute input gradients for backpropagation to previous layer
  CAIF_Tensor input_gradient=ComputeInputGradients(activation_gradient);
  return input_gradient;
}

void CAIF_Convolution2DLayer::Initialize(const std::vector<uint32_t> &input_shape,const uint32_t seed)
{
  // Enhanced input shape validation
  if(input_shape.size()!=4)
  {
    THROW_CAIFE("Input shape must be 4D [batch, height, width, channels]");
  }
  
  if(input_shape[0]==0||input_shape[1]==0||input_shape[2]==0||input_shape[3]==0)
  {
    THROW_CAIFE("All input dimensions must be greater than 0");
  }
  
  // Check reasonable limits
  if(input_shape[1]>g_caif_max_tensor_dimension||input_shape[2]>g_caif_max_tensor_dimension)
  {
    THROW_CAIFE("Input dimensions exceed maximum allowed size");
  }
  
  if(input_shape[3]>g_caif_max_channels)
  {
    THROW_CAIFE("Input channels exceed maximum allowed");
  }
  
  // Validate convolution will produce valid output
  const uint32_t output_height=(input_shape[1]+2*_padding-_kernel_size)/_stride+1;
  const uint32_t output_width=(input_shape[2]+2*_padding-_kernel_size)/_stride+1;
  
  if(output_height==0||output_width==0)
  {
    std::ostringstream oss;
    oss<<"Invalid convolution parameters would produce zero output size. "
       <<"Input: "<<input_shape[1]<<"x"<<input_shape[2]
       <<", Kernel: "<<_kernel_size
       <<", Stride: "<<_stride
       <<", Padding: "<<_padding;
    THROW_CAIFE(oss.str().c_str());
  }
  
  _input_channels=input_shape[3];
  
  // Initialize weight tensor in HWIO order [kernel_h, kernel_w, in_channels, out_channels]
  // This matches the backend-bridged convolution expectation and yields NHWC output
  CAIF_Framework &framework=Framework();
  std::vector<uint32_t> weight_shape={_kernel_size,_kernel_size,_input_channels,_filters};
  _weights=CAIF_Tensor(framework,weight_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  _weight_gradients=CAIF_Tensor(framework,weight_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  
  // Initialize bias tensor if needed
  if(_use_bias==true)
  {
    std::vector<uint32_t> bias_shape={_filters};
    _bias=CAIF_Tensor(framework,bias_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    _bias_gradients=CAIF_Tensor(framework,bias_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  }
  
  // Initialize parameters
  InitializeWeights(seed);
  
  SetInitialized(true);
}

std::vector<uint32_t> CAIF_Convolution2DLayer::CalculateOutputShape(const std::vector<uint32_t> &input_shape
                                                                  )const
{
  if(input_shape.size()!=4)
  {
    THROW_CAIFE("Convolution requires 4D input [batch, height, width, channels]");
  }
  
  const uint32_t batch_size=input_shape[0];
  const uint32_t input_height=input_shape[1];
  const uint32_t input_width=input_shape[2];
  
  const uint32_t output_height=CalculateOutputDim(input_height);
  const uint32_t output_width=CalculateOutputDim(input_width);
  
  return std::vector<uint32_t>{batch_size,output_height,output_width,_filters};
}

std::unique_ptr<CAIF_Layer> CAIF_Convolution2DLayer::Clone()const
{
  // Framework reference is copied from this layer via copy constructor
  return std::make_unique<CAIF_Convolution2DLayer>(*this);
}

std::string CAIF_Convolution2DLayer::Description()const
{
  std::ostringstream oss;
  oss<<"Convolution 2D Layer (filters="<<_filters<<", kernel="<<_kernel_size
     <<"x"<<_kernel_size<<", stride="<<_stride<<", padding="<<_padding<<")";

  if(IsInitialized()==true)
  {
    oss<<" (";
    const auto &input_shape=InputShape();
    for(size_t i=0;i<input_shape.size();++i)
    {
      if(i>0)oss<<"x";
      oss<<input_shape[i];
    }

    oss<<" -> ";
    const auto &output_shape=OutputShape();
    for(size_t i=0;i<output_shape.size();++i)
    {
      if(i>0)oss<<"x";
      oss<<output_shape[i];
    }

    oss<<")";
  }
  return oss.str();
}

std::vector<CAIF_Tensor> CAIF_Convolution2DLayer::Parameters()const
{
  std::vector<CAIF_Tensor> parameters;
  parameters.push_back(_weights);
  if(_use_bias==true)
  {
    parameters.push_back(_bias);
  }
  return parameters;
}

std::vector<CAIF_Tensor> CAIF_Convolution2DLayer::ParameterGradients()const
{
  std::vector<CAIF_Tensor> gradients;
  gradients.push_back(_weight_gradients);
  if(_use_bias==true)
  {
    gradients.push_back(_bias_gradients);
  }
  return gradients;
}

size_t CAIF_Convolution2DLayer::ParameterCount()const
{
  if(_use_bias==true)
  {
    return 2;
  }
  return 1;
}

CAIF_Tensor &CAIF_Convolution2DLayer::ParameterRef(const size_t index)
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

const CAIF_Tensor &CAIF_Convolution2DLayer::ParameterRef(const size_t index)const
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

CAIF_Tensor &CAIF_Convolution2DLayer::GradientRef(const size_t index)
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

const CAIF_Tensor &CAIF_Convolution2DLayer::GradientRef(const size_t index)const
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

void CAIF_Convolution2DLayer::UpdateParameters(const std::vector<CAIF_Tensor> &new_parameters)
{
  const size_t expected_params=_use_bias?2:1;
  if(new_parameters.size()!=expected_params)
  {
    THROW_CAIFE("Invalid number of parameters");
  }
  
  _weights=new_parameters[0];
  if(_use_bias==true&&new_parameters.size()>1)
  {
    _bias=new_parameters[1];
  }
}

void CAIF_Convolution2DLayer::ResetParameters(const uint32_t seed)
{
  InitializeWeights(seed);
}

void CAIF_Convolution2DLayer::InitializeWeights(const uint32_t seed)
{
  std::mt19937 rng(seed);
  
  // Enhanced Xavier/Glorot initialization with bounds checking
  const float fan_in=static_cast<float>(_kernel_size*_kernel_size*_input_channels);
  const float fan_out=static_cast<float>(_kernel_size*_kernel_size*_filters);
  
  if(fan_in<=0.0f||fan_out<=0.0f)
  {
    throw std::runtime_error("Invalid fan_in or fan_out for weight initialization");
  }
  
  const float limit=std::sqrt(6.0f/(fan_in+fan_out));
  
  // Ensure reasonable bounds
  const float max_limit=2.0f;  // Prevent extremely large initial weights
  const float actual_limit=std::min(limit,max_limit);
  
  std::uniform_real_distribution<float> weight_dist(-actual_limit,actual_limit);
  
  // Initialize weights with error checking
  float *weight_data=_weights.MutableData<float>();
  const uint32_t weight_count=_weights.NumElements();
  
  for(uint32_t i=0;i<weight_count;++i)
  {
    weight_data[i]=weight_dist(rng);
  }
  
  // Initialize bias to zero with error checking
  if(_use_bias==true)
  {
    float *bias_data=_bias.MutableData<float>();
    std::fill_n(bias_data,_filters,0.0f);
  }
}

CAIF_Tensor CAIF_Convolution2DLayer::ApplyActivation(const CAIF_Tensor &tensor)const
{
  switch(_activation)
  {
    case CAIF_ActivationType_e::Linear:
      return tensor.Linear();
    case CAIF_ActivationType_e::ReLU:
      return tensor.ReLU();
    case CAIF_ActivationType_e::Sigmoid:
      return tensor.Sigmoid();
    case CAIF_ActivationType_e::Tanh:
      return tensor.Tanh();
    case CAIF_ActivationType_e::Softmax:
      return tensor.Softmax();
    case CAIF_ActivationType_e::LeakyReLU:
      return tensor.LeakyReLU();
    case CAIF_ActivationType_e::ELU:
      return tensor.ELU();
    case CAIF_ActivationType_e::GELU:
      return tensor.GELU();
    case CAIF_ActivationType_e::Swish:
      return tensor.Swish();
    default:
      return tensor;  // Return input unchanged for unknown activation
  }
}

CAIF_Tensor CAIF_Convolution2DLayer::ApplyActivationDerivative(
                                                            const CAIF_Tensor &tensor,
                                                            const CAIF_Tensor &gradient
                                                           )const
{
  switch(_activation)
  {
    case CAIF_ActivationType_e::Linear:
      return tensor.LinearDerivative(gradient);
    case CAIF_ActivationType_e::ReLU:
      return tensor.ReLUDerivative(gradient);
    case CAIF_ActivationType_e::Sigmoid:
      return tensor.SigmoidDerivative(gradient);
    case CAIF_ActivationType_e::Tanh:
      return tensor.TanhDerivative(gradient);
    case CAIF_ActivationType_e::Softmax:
      return tensor.SoftmaxDerivative(gradient);
    case CAIF_ActivationType_e::LeakyReLU:
      return tensor.LeakyReLUDerivative(gradient);
    case CAIF_ActivationType_e::ELU:
      return tensor.ELUDerivative(gradient);
    case CAIF_ActivationType_e::GELU:
      return tensor.GELUDerivative(gradient);
    case CAIF_ActivationType_e::Swish:
      return tensor.SwishDerivative(gradient);
    default:
      return gradient;  // No activation
  }
}

uint32_t CAIF_Convolution2DLayer::CalculateOutputDim(const uint32_t input_dim)const
{
  return (input_dim+2*_padding-_kernel_size)/_stride+1;
}

void CAIF_Convolution2DLayer::PerformConvolution(
                                                const CAIF_Tensor &input,
                                                CAIF_Tensor &output
                                               )const
{
  // Delegate to backend framework for convolution to enable acceleration
  // const_cast is safe here because we're calling non-const methods on framework
  // but the framework object itself is not const
  CAIF_Framework &framework=const_cast<CAIF_Framework&>(Framework());
  CAIF_Tensor backend_out=framework.Convolution2D(
                                                   input,
                                                   _weights,
                                                   _stride,
                                                   _stride,
                                                   _padding,
                                                   _padding
                                                  );
  output=backend_out;
}

void CAIF_Convolution2DLayer::AddBias(CAIF_Tensor &output)const
{
  const float *bias_data=_bias.ConstData<float>();
  float *output_data=output.MutableData<float>();
  
  const auto &out_shape=output.Shape();
  const uint32_t batch_size=out_shape[0];
  const uint32_t out_height=out_shape[1];
  const uint32_t out_width=out_shape[2];
  const uint32_t out_channels=out_shape[3];
  
  // Validate bias size matches output channels
  if(out_channels!=_filters)
  {
    THROW_CAIFE("Bias size mismatch with output channels");
  }
  
  // Efficient bias addition
  uint32_t bb=0;
  uint32_t hb=0;
  uint32_t wb=0;
  
  for(uint32_t b=0;b<batch_size;++b)
  {
    bb=b*out_height*out_width*out_channels;

    for(uint32_t h=0;h<out_height;++h)
    {
      hb=h*out_width*out_channels;

      for(uint32_t w=0;w<out_width;++w)
      {
        wb=w*out_channels;

        for(uint32_t c=0;c<out_channels;++c)
        {
          const uint32_t idx=bb+hb+wb+c;
          output_data[idx]+=bias_data[c];
        }
      }
    }
  }
  return;
}

CAIF_Tensor CAIF_Convolution2DLayer::ComputeWeightGradients(const CAIF_Tensor &input,
                                                          const CAIF_Tensor &output_gradient)const
{
  // Naive gradient computation: dW = sum_b,oh,ow( input_patched * dY )
  const auto &in_shape=input.Shape();
  const auto &w_shape=_weights.Shape();
  const auto &g_shape=output_gradient.Shape();
  
  const uint32_t batch=in_shape[0];
  const uint32_t in_h=in_shape[1];
  const uint32_t in_w=in_shape[2];
  const uint32_t in_c=in_shape[3];
  
  const uint32_t k_h=w_shape[0];
  const uint32_t k_w=w_shape[1];
  const uint32_t out_c=w_shape[3];
  
  const uint32_t out_h=g_shape[1];
  const uint32_t out_w=g_shape[2];
  
  CAIF_Framework &framework=const_cast<CAIF_Framework&>(Framework());
  CAIF_Tensor weight_gradient(framework,w_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  float *wg=weight_gradient.MutableData<float>();
  const float *in_data=input.ConstData<float>();
  const float *g_data=output_gradient.ConstData<float>();
  std::fill_n(wg,weight_gradient.NumElements(),0.0f);
  
  const size_t in_hw=static_cast<size_t>(in_h)*in_w;
  const size_t out_hw=static_cast<size_t>(out_h)*out_w;
  
  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t oh=0;oh<out_h;++oh)
    {
      for(uint32_t ow=0;ow<out_w;++ow)
      {
        for(uint32_t oc=0;oc<out_c;++oc)
        {
          const size_t g_idx=static_cast<size_t>(b)*out_hw*out_c+
                             static_cast<size_t>(oh)*out_w*out_c+
                             static_cast<size_t>(ow)*out_c+
                             oc;
          const float grad_val=g_data[g_idx];
          
          for(uint32_t kh=0;kh<k_h;++kh)
          {
            const int32_t ih=static_cast<int32_t>(oh)*
                             static_cast<int32_t>(_stride)-
                             static_cast<int32_t>(_padding)+
                             static_cast<int32_t>(kh);

            if(ih<0||ih>=static_cast<int32_t>(in_h))
            {
              continue;
            }

            for(uint32_t kw=0;kw<k_w;++kw)
            {
              const int32_t iw=static_cast<int32_t>(ow)*
                               static_cast<int32_t>(_stride)-
                               static_cast<int32_t>(_padding)+
                               static_cast<int32_t>(kw);

              if(iw<0||iw>=static_cast<int32_t>(in_w))
              {
                continue;
              }

              for(uint32_t ic=0;ic<in_c;++ic)
              {
                const size_t in_idx=static_cast<size_t>(b)*in_hw*in_c+
                                    static_cast<size_t>(ih)*in_w*in_c+
                                    static_cast<size_t>(iw)*in_c+
                                    ic;
                const size_t w_idx=static_cast<size_t>(kh)*k_w*in_c*out_c+
                                    static_cast<size_t>(kw)*in_c*out_c+
                                    static_cast<size_t>(ic)*out_c+
                                    oc;
                wg[w_idx]+=in_data[in_idx]*grad_val;
              }
            }
          }
        }
      }
    }
  }
  return weight_gradient;
}

CAIF_Tensor CAIF_Convolution2DLayer::ComputeBiasGradients(const CAIF_Tensor &output_gradient)const
{
  // Sum over batch and spatial dims for each output channel
  const auto &g_shape=output_gradient.Shape();
  const uint32_t batch=g_shape[0];
  const uint32_t out_h=g_shape[1];
  const uint32_t out_w=g_shape[2];
  const uint32_t out_c=g_shape[3];
  
  CAIF_Framework &framework=const_cast<CAIF_Framework&>(Framework());
  CAIF_Tensor bias_gradient(framework,_bias.Shape(),CAIF_DataType::CAIF_DataType_e::Float32);
  float *bg=bias_gradient.MutableData<float>();
  const float *g_data=output_gradient.ConstData<float>();
  std::fill_n(bg,bias_gradient.NumElements(),0.0f);
  
  const size_t out_hw=static_cast<size_t>(out_h)*out_w;
  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t oh=0;oh<out_h;++oh)
    {
      for(uint32_t ow=0;ow<out_w;++ow)
      {
        for(uint32_t oc=0;oc<out_c;++oc)
        {
          const size_t g_idx=static_cast<size_t>(b)*out_hw*out_c+
                             static_cast<size_t>(oh)*out_w*out_c+
                             static_cast<size_t>(ow)*out_c+
                             oc;
          bg[oc]+=g_data[g_idx];
        }
      }
    }
  }
  return bias_gradient;
}

CAIF_Tensor CAIF_Convolution2DLayer::ComputeInputGradients(
                                                         const CAIF_Tensor &output_gradient
                                                        )const
{
  // Backprop to input: dX = conv2d(dY, rotate180(W)) with appropriate stride/padding
  const auto &in_shape=_last_input.Shape();
  const auto &w_shape=_weights.Shape();
  const auto &g_shape=output_gradient.Shape();
  
  const uint32_t batch=in_shape[0];
  const uint32_t in_h=in_shape[1];
  const uint32_t in_w=in_shape[2];
  const uint32_t in_c=in_shape[3];
  
  const uint32_t k_h=w_shape[0];
  const uint32_t k_w=w_shape[1];
  const uint32_t out_c=w_shape[3];
  
  const uint32_t out_h=g_shape[1];
  const uint32_t out_w=g_shape[2];
  
  CAIF_Framework &framework=const_cast<CAIF_Framework&>(Framework());
  CAIF_Tensor input_gradient(framework,in_shape,CAIF_DataType::CAIF_DataType_e::Float32);
  float *ix=input_gradient.MutableData<float>();
  const float *w_data=_weights.ConstData<float>();
  const float *g_data=output_gradient.ConstData<float>();
  std::fill_n(ix,input_gradient.NumElements(),0.0f);
  
  const size_t in_hw=static_cast<size_t>(in_h)*in_w;
  const size_t out_hw=static_cast<size_t>(out_h)*out_w;
  
  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t oh=0;oh<out_h;++oh)
    {
      for(uint32_t ow=0;ow<out_w;++ow)
      {
        for(uint32_t oc=0;oc<out_c;++oc)
        {
          const size_t g_idx=static_cast<size_t>(b)*out_hw*out_c+
                             static_cast<size_t>(oh)*out_w*out_c+
                             static_cast<size_t>(ow)*out_c+
                             oc;
          const float grad_val=g_data[g_idx];
          
          for(uint32_t kh=0;kh<k_h;++kh)
          {
            const int32_t ih=static_cast<int32_t>(oh)*
                             static_cast<int32_t>(_stride)-
                             static_cast<int32_t>(_padding)+
                             static_cast<int32_t>(kh);

            if(ih<0||ih>=static_cast<int32_t>(in_h))
            {
              continue;
            }
            for(uint32_t kw=0;kw<k_w;++kw)
            {
              const int32_t iw=static_cast<int32_t>(ow)*
                               static_cast<int32_t>(_stride)-
                               static_cast<int32_t>(_padding)+
                               static_cast<int32_t>(kw);

              if(iw<0||iw>=static_cast<int32_t>(in_w))
              {
                continue;
              }
              for(uint32_t ic=0;ic<in_c;++ic)
              {
                const size_t in_idx=static_cast<size_t>(b)*in_hw*in_c+
                                    static_cast<size_t>(ih)*in_w*in_c+
                                    static_cast<size_t>(iw)*in_c+
                                    ic;
                const size_t w_idx=static_cast<size_t>(kh)*k_w*in_c*out_c+
                                    static_cast<size_t>(kw)*in_c*out_c+
                                    static_cast<size_t>(ic)*out_c+
                                    oc;
                ix[in_idx]+=grad_val*w_data[w_idx];
              }
            }
          }
        }
      }
    }
  }
  
  return input_gradient;
}

