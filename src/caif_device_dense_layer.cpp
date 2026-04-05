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

#include "caif_device_dense_layer.h"
#include "caif_device_ops.h"
#include "caif_host_tensor.h"
#include "caif_exception.h"
#include <random>
#include <cmath>
#include <ctime>

namespace instance
{

CAIF_DeviceDenseLayer::CAIF_DeviceDenseLayer(uint32_t input_size,
                                           uint32_t output_size,
                                           CAIF_DeviceActivation_e activation,
                                           CAIF_CudaStream &stream,
                                           bool use_bias):CAIF_DeviceLayer(stream),
                                                          _input_size(input_size),
                                                          _output_size(output_size),
                                                          _activation(activation),
                                                          _use_bias(use_bias),
                                                          _weights(),
                                                          _bias(),
                                                          _weight_grads(),
                                                          _bias_grads(),
                                                          _output_buffer(),
                                                          _output_batch(0),
                                                          _last_input(),
                                                          _last_preactivation(),
                                                          _last_output()
{
  try
  {
    if(input_size==0||output_size==0)
    {
      THROW_CAIFE("DeviceDenseLayer: input_size and output_size must be > 0");
    }

    // Allocate tensors using static factory methods
    _weights=CAIF_DeviceTensor::Uninitialized({input_size,output_size},stream);
    _weight_grads=CAIF_DeviceTensor::Zeros({input_size,output_size},stream);

    if(_use_bias==true)
    {
      _bias=CAIF_DeviceTensor::Zeros({output_size},stream);
      _bias_grads=CAIF_DeviceTensor::Zeros({output_size},stream);
    }

    // Initialize weights
    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceDenseLayer::CAIF_DeviceDenseLayer(
  CAIF_DeviceDenseLayer &&other):CAIF_DeviceLayer(std::move(other)),
                                _input_size(other._input_size),
                                _output_size(other._output_size),
                                _activation(other._activation),
                                _use_bias(other._use_bias),
                                _weights(std::move(other._weights)),
                                _bias(std::move(other._bias)),
                                _weight_grads(std::move(other._weight_grads)),
                                _bias_grads(std::move(other._bias_grads)),
                                _output_buffer(std::move(other._output_buffer)),
                                _output_batch(other._output_batch),
                                _last_input(std::move(other._last_input)),
                                _last_preactivation(std::move(other._last_preactivation)),
                                _last_output(std::move(other._last_output))
{
}

CAIF_DeviceDenseLayer &CAIF_DeviceDenseLayer::operator=(CAIF_DeviceDenseLayer &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _input_size=other._input_size;
      _output_size=other._output_size;
      _activation=other._activation;
      _use_bias=other._use_bias;
      _weights=std::move(other._weights);
      _bias=std::move(other._bias);
      _weight_grads=std::move(other._weight_grads);
      _bias_grads=std::move(other._bias_grads);
      _output_buffer=std::move(other._output_buffer);
      _output_batch=other._output_batch;
      _last_input=std::move(other._last_input);
      _last_preactivation=std::move(other._last_preactivation);
      _last_output=std::move(other._last_output);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceDenseLayer::InitializeWeights(uint32_t seed)
{
  try
  {
    if(seed==0)
    {
      seed=static_cast<uint32_t>(std::time(nullptr));
    }

    // Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
    const float scale=std::sqrt(2.0f/static_cast<float>(_input_size+_output_size));

    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f,scale);

    // Generate weights on host then upload
    const size_t weight_count=static_cast<size_t>(_input_size)*_output_size;
    std::vector<float> host_weights(weight_count);
    for(size_t i=0;i<weight_count;++i)
    {
      host_weights[i]=dist(gen);
    }
    _weights.CopyFromHost(host_weights.data(),weight_count);

    // Initialize bias to zero
    if(_use_bias==true)
    {
      _bias.Fill(0.0f);
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceDenseLayer::ZeroGradients()
{
  try
  {
    _weight_grads.Fill(0.0f);
    if(_use_bias==true)
    {
      _bias_grads.Fill(0.0f);
    }
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceDenseLayer::TotalParameterCount()const
{
  try
  {
    size_t count=static_cast<size_t>(_input_size)*_output_size;
    if(_use_bias==true)
    {
      count+=_output_size;
    }
    return count;
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceDenseLayer::ParameterTensorCount()const
{
  try
  {
    if(_use_bias==true)
    {
      return 2;
    }
    return 1;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceDenseLayer::ParameterTensor(size_t index)
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
    THROW_CAIFE("DeviceDenseLayer::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceDenseLayer::ParameterTensor(size_t index)const
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
    THROW_CAIFE("DeviceDenseLayer::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceDenseLayer::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _weight_grads;
    }
    if(index==1&&_use_bias==true)
    {
      return _bias_grads;
    }
    THROW_CAIFE("DeviceDenseLayer::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceDenseLayer::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _weight_grads;
    }
    if(index==1&&_use_bias==true)
    {
      return _bias_grads;
    }
    THROW_CAIFE("DeviceDenseLayer::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceDenseLayer::Description()const
{
  try
  {
    return "Dense("+std::to_string(_input_size)+","+std::to_string(_output_size)+")";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceDenseLayer::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"weight");
    if(_use_bias==true)
    {
      names.push_back(prefix+"bias");
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceDenseLayer::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceDenseLayer: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("DeviceDenseLayer::Forward: input must be 2D [batch x features]");
    }
    if(shape[1]!=_input_size)
    {
      THROW_CAIFE("DeviceDenseLayer::Forward: input features must match input_size");
    }

    const uint32_t batch_size=shape[0];

    // Ensure output buffer is allocated
    if(batch_size!=_output_batch)
    {
      _output_buffer=CAIF_DeviceTensor::Uninitialized({batch_size,_output_size},*_stream);
      _output_batch=batch_size;
    }

    // Compute linear = input * weights + bias (fused) or just input * weights
    if(_use_bias==true)
    {
      CAIF_DeviceOps::MatMulBias(input,_weights,_bias,_output_buffer,_stream->Handle());
    }
    else
    {
      CAIF_DeviceOps::MatMul(input,_weights,_output_buffer);
    }

    // Cache for backward if training
    if(training==true)
    {
      _last_input=input.Clone();
      _last_preactivation=_output_buffer.Clone();
    }

    // Apply activation
    if(_activation==CAIF_DeviceActivation_e::None)
    {
      return _output_buffer.Clone();
    }

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({batch_size,_output_size},*_stream);

    switch(_activation)
    {
      case CAIF_DeviceActivation_e::None:
        break;
      case CAIF_DeviceActivation_e::ReLU:
        CAIF_DeviceOps::ReLU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Sigmoid:
        CAIF_DeviceOps::Sigmoid(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Tanh:
        CAIF_DeviceOps::Tanh(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Softmax:
        CAIF_DeviceOps::Softmax(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::LeakyReLU:
        CAIF_DeviceOps::LeakyReLU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::ELU:
        CAIF_DeviceOps::ELU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::GELU:
        CAIF_DeviceOps::GELU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Swish:
        CAIF_DeviceOps::Swish(_output_buffer,output);
        break;
    }

    // Cache for backward: activations that need output for backward pass
    if(training==true)
    {
      if(_activation==CAIF_DeviceActivation_e::Sigmoid||
         _activation==CAIF_DeviceActivation_e::Tanh||
         _activation==CAIF_DeviceActivation_e::Softmax||
         _activation==CAIF_DeviceActivation_e::ELU||
         _activation==CAIF_DeviceActivation_e::Swish)
      {
        _last_output=output.Clone();
      }
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceDenseLayer::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceDenseLayer: layer has been moved from");
    }
    if(_last_input.IsEmpty()==true)
    {
      THROW_CAIFE("DeviceDenseLayer::Backward: must call Forward with training=true first");
    }

    const auto &shape=grad_output.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("DeviceDenseLayer::Backward: grad_output must be 2D [batch x output_size]");
    }
    if(shape[1]!=_output_size)
    {
      THROW_CAIFE("DeviceDenseLayer::Backward: grad_output must match output_size");
    }

    const uint32_t batch_size=shape[0];

    // Compute activation gradient
    CAIF_DeviceTensor linear_grad=CAIF_DeviceTensor::Uninitialized({batch_size,_output_size},*_stream);

    switch(_activation)
    {
      case CAIF_DeviceActivation_e::None:
        // Linear activation: gradient passes through unchanged
        linear_grad=grad_output.Clone();
        break;
      case CAIF_DeviceActivation_e::ReLU:
        CAIF_DeviceOps::ReLUBackward(grad_output,_last_preactivation,linear_grad);
        break;
      case CAIF_DeviceActivation_e::Sigmoid:
        CAIF_DeviceOps::SigmoidBackward(grad_output,_last_output,linear_grad);
        break;
      case CAIF_DeviceActivation_e::Tanh:
        CAIF_DeviceOps::TanhBackward(grad_output,_last_output,linear_grad);
        break;
      case CAIF_DeviceActivation_e::Softmax:
        CAIF_DeviceOps::SoftmaxBackward(grad_output,_last_output,linear_grad);
        break;
      case CAIF_DeviceActivation_e::LeakyReLU:
        CAIF_DeviceOps::LeakyReLUBackward(grad_output,_last_preactivation,linear_grad);
        break;
      case CAIF_DeviceActivation_e::ELU:
        CAIF_DeviceOps::ELUBackward(grad_output,_last_preactivation,_last_output,linear_grad);
        break;
      case CAIF_DeviceActivation_e::GELU:
        CAIF_DeviceOps::GELUBackward(grad_output,_last_preactivation,linear_grad);
        break;
      case CAIF_DeviceActivation_e::Swish:
        CAIF_DeviceOps::SwishBackward(grad_output,_last_preactivation,_last_output,linear_grad);
        break;
    }

    // Compute weight gradients: dW = input^T * linear_grad
    CAIF_DeviceOps::MatMulTransposeA(_last_input,linear_grad,_weight_grads);

    // Compute bias gradients: db = sum over batch of linear_grad
    if(_use_bias==true)
    {
      CAIF_DeviceOps::BiasGradient(linear_grad,_bias_grads);
    }

    // Compute input gradient: dX = linear_grad * weights^T
    CAIF_DeviceTensor input_grad=CAIF_DeviceTensor::Uninitialized({batch_size,_input_size},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(linear_grad,_weights,input_grad);

    return input_grad;
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
