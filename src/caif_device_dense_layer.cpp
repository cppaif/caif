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
#include "caif_ops.h"
#include "caif_host_tensor.h"
#include "caif_exception.h"
#include <random>
#include <cmath>
#include <ctime>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceDenseLayer<ComputeT,StorageT>::CAIF_DeviceDenseLayer(uint32_t input_size,
                                                                uint32_t output_size,
                                                                CAIF_DeviceActivation_e activation,
                                                                CAIF_CudaStream &stream,
                                                                bool use_bias):
                                            CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
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
      THROW_CAIFE("CAIF_DeviceDenseLayer: input_size and output_size must be > 0");
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    _weights=CAIF_DeviceTensor::Uninitialized({input_size,output_size},stream,sdt);
    _weight_grads=CAIF_DeviceTensor::Zeros({input_size,output_size},stream,sdt);

    if(_use_bias==true)
    {
      _bias=CAIF_DeviceTensor::Zeros({output_size},stream,sdt);
      _bias_grads=CAIF_DeviceTensor::Zeros({output_size},stream,sdt);
    }

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceDenseLayer<ComputeT,StorageT>::CAIF_DeviceDenseLayer(CAIF_DeviceDenseLayer &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceDenseLayer<ComputeT,StorageT> &
CAIF_DeviceDenseLayer<ComputeT,StorageT>::operator=(CAIF_DeviceDenseLayer &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
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

template<typename ComputeT,typename StorageT>
void CAIF_DeviceDenseLayer<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    if(seed==0)
    {
      seed=static_cast<uint32_t>(std::time(nullptr));
    }

    const float scale=std::sqrt(2.0f/static_cast<float>(_input_size+_output_size));
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f,scale);

    const size_t weight_count=static_cast<size_t>(_input_size)*_output_size;
    std::vector<float> host_weights(weight_count);
    for(size_t i=0;i<weight_count;++i)
    {
      host_weights[i]=dist(gen);
    }
    Weights().CopyFromHostFp32(host_weights.data(),weight_count);

    if(_use_bias==true)
    {
      _bias.FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceDenseLayer<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    _weight_grads.FillZero();
    if(_use_bias==true)
    {
      _bias_grads.FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceDenseLayer<ComputeT,StorageT>::TotalParameterCount()const
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

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceDenseLayer<ComputeT,StorageT>::ParameterTensorCount()const
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceDenseLayer<ComputeT,StorageT>::ParameterTensor(size_t index)
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
    THROW_CAIFE("CAIF_DeviceDenseLayer::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceDenseLayer<ComputeT,StorageT>::ParameterTensor(size_t index)const
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
    THROW_CAIFE("CAIF_DeviceDenseLayer::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceDenseLayer<ComputeT,StorageT>::GradientTensor(size_t index)
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
    THROW_CAIFE("CAIF_DeviceDenseLayer::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceDenseLayer<ComputeT,StorageT>::GradientTensor(size_t index)const
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
    THROW_CAIFE("CAIF_DeviceDenseLayer::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceDenseLayer<ComputeT,StorageT>::Description()const
{
  try
  {
    return "Dense("+std::to_string(_input_size)+","+std::to_string(_output_size)+")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceDenseLayer<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceDenseLayer<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                      CAIF_RunContext &ctx)
{
  try
  {
    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Forward: input must be 2D [batch x features]");
    }
    if(shape[1]!=_input_size)
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Forward: input features must match input_size");
    }

    AssertInputDtype(input);

    const uint32_t batch_size=shape[0];
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    if(batch_size!=_output_batch)
    {
      _output_buffer=CAIF_DeviceTensor::Uninitialized({batch_size,_output_size},ctx.Stream(),sdt);
      _output_batch=batch_size;
    }

    if(_use_bias==true)
    {
      CAIF_Ops::MatMulBias(input,_weights,_bias,_output_buffer,ctx.Stream().Handle(),ctx,cdt);
    }
    else
    {
      CAIF_Ops::MatMul(input,_weights,_output_buffer,ctx,cdt);
    }

    if(ctx.Training()==true)
    {
      _last_input=input.Clone();
      _last_preactivation=_output_buffer.Clone();
    }

    if(_activation==CAIF_DeviceActivation_e::None)
    {
      return _output_buffer.Clone();
    }

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({batch_size,_output_size},
                                                              ctx.Stream(),sdt);

    switch(_activation)
    {
      case CAIF_DeviceActivation_e::None:
        break;
      case CAIF_DeviceActivation_e::ReLU:
        CAIF_Ops::ReLU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Sigmoid:
        CAIF_Ops::Sigmoid(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Tanh:
        CAIF_Ops::Tanh(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Softmax:
        CAIF_Ops::Softmax(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::LeakyReLU:
        CAIF_Ops::LeakyReLU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::ELU:
        CAIF_Ops::ELU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::GELU:
        CAIF_Ops::GELU(_output_buffer,output);
        break;
      case CAIF_DeviceActivation_e::Swish:
        CAIF_Ops::Swish(_output_buffer,output);
        break;
    }

    if(ctx.Training()==true)
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceDenseLayer<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                       CAIF_RunContext &ctx)
{
  try
  {
    if(_last_input.IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Backward: must call Forward with training=true first");
    }

    const std::vector<uint32_t> &shape=grad_output.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Backward: grad_output must be 2D");
    }
    if(shape[1]!=_output_size)
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Backward: grad_output must match output_size");
    }

    const uint32_t batch_size=shape[0];
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    CAIF_DeviceTensor linear_grad;
    if(_activation==CAIF_DeviceActivation_e::None)
    {
      linear_grad=CAIF_DeviceTensor::WrapView(
                   const_cast<void *>(grad_output.DeviceDataRaw()),
                   {batch_size,_output_size},
                   ctx.Stream(),
                   grad_output.Dtype());
    }
    else
    {
      linear_grad=CAIF_DeviceTensor::Uninitialized({batch_size,_output_size},ctx.Stream(),sdt);
      switch(_activation)
      {
        case CAIF_DeviceActivation_e::None:
          break;
        case CAIF_DeviceActivation_e::ReLU:
          CAIF_Ops::ReLUBackward(grad_output,_last_preactivation,linear_grad);
          break;
        case CAIF_DeviceActivation_e::Sigmoid:
          CAIF_Ops::SigmoidBackward(grad_output,_last_output,linear_grad);
          break;
        case CAIF_DeviceActivation_e::Tanh:
          CAIF_Ops::TanhBackward(grad_output,_last_output,linear_grad);
          break;
        case CAIF_DeviceActivation_e::Softmax:
          CAIF_Ops::SoftmaxBackward(grad_output,_last_output,linear_grad);
          break;
        case CAIF_DeviceActivation_e::LeakyReLU:
          CAIF_Ops::LeakyReLUBackward(grad_output,_last_preactivation,linear_grad);
          break;
        case CAIF_DeviceActivation_e::ELU:
          CAIF_Ops::ELUBackward(grad_output,_last_preactivation,_last_output,linear_grad);
          break;
        case CAIF_DeviceActivation_e::GELU:
          CAIF_Ops::GELUBackward(grad_output,_last_preactivation,linear_grad);
          break;
        case CAIF_DeviceActivation_e::Swish:
          CAIF_Ops::SwishBackward(grad_output,_last_preactivation,_last_output,linear_grad);
          break;
      }
    }

    CAIF_Ops::MatMulTransposeA(_last_input,linear_grad,_weight_grads,ctx,cdt);

    if(_use_bias==true)
    {
      CAIF_Ops::BiasGradient(linear_grad,_bias_grads);
    }

    CAIF_DeviceTensor input_grad=CAIF_DeviceTensor::Uninitialized({batch_size,_input_size},
                                                                    ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeB(linear_grad,_weights,input_grad,ctx,cdt);

    return input_grad;
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceDenseLayer<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceDenseLayer<float,__half>;
template class CAIF_DeviceDenseLayer<float,__nv_bfloat16>;
template class CAIF_DeviceDenseLayer<__half,float>;
template class CAIF_DeviceDenseLayer<__half,__half>;
template class CAIF_DeviceDenseLayer<__half,__nv_bfloat16>;
template class CAIF_DeviceDenseLayer<__nv_bfloat16,float>;
template class CAIF_DeviceDenseLayer<__nv_bfloat16,__half>;
template class CAIF_DeviceDenseLayer<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
