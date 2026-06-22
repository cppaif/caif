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
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <random>
#include <cmath>
#include <ctime>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceDenseLayer<ComputeT,StorageT>::CAIF_DeviceDenseLayer(
                                                  uint32_t input_size,
                                                  uint32_t output_size,
                                                  CAIF_DeviceActivation::CAIF_DeviceActivation_e activation,
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
    SetWeights(CAIF_DeviceTensor::Uninitialized({input_size,output_size},stream,sdt));
    SetWeightGradients(CAIF_DeviceTensor::Zeros({input_size,output_size},stream,sdt));

    if(UseBias()==true)
    {
      SetBias(CAIF_DeviceTensor::Zeros({output_size},stream,sdt));
      SetBiasGradients(CAIF_DeviceTensor::Zeros({output_size},stream,sdt));
    }

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceDenseLayer<ComputeT,StorageT>::CAIF_DeviceDenseLayer(CAIF_DeviceDenseLayer &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _input_size(other.InputSize()),
                              _output_size(other.OutputSize()),
                              _activation(other.Activation()),
                              _use_bias(other.UseBias()),
                              _weights(std::move(other.Weights())),
                              _bias(std::move(other.Bias())),
                              _weight_grads(std::move(other.WeightGradients())),
                              _bias_grads(std::move(other.BiasGradients())),
                              _output_buffer(std::move(other.OutputBuffer())),
                              _output_batch(other.OutputBatch()),
                              _last_input(std::move(other.LastInput())),
                              _last_preactivation(std::move(other.LastPreactivation())),
                              _last_output(std::move(other.LastOutput()))
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
      SetInputSize(other.InputSize());
      SetOutputSize(other.OutputSize());
      SetActivation(other.Activation());
      SetUseBias(other.UseBias());
      SetWeights(std::move(other.Weights()));
      SetBias(std::move(other.Bias()));
      SetWeightGradients(std::move(other.WeightGradients()));
      SetBiasGradients(std::move(other.BiasGradients()));
      SetOutputBuffer(std::move(other.OutputBuffer()));
      SetOutputBatch(other.OutputBatch());
      SetLastInput(std::move(other.LastInput()));
      SetLastPreactivation(std::move(other.LastPreactivation()));
      SetLastOutput(std::move(other.LastOutput()));
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

    const float scale=std::sqrt(2.0f/static_cast<float>(InputSize()+OutputSize()));
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f,scale);

    const size_t weight_count=static_cast<size_t>(InputSize())*OutputSize();
    std::vector<float> host_weights(weight_count);
    for(size_t i=0;i<weight_count;++i)
    {
      host_weights[i]=dist(gen);
    }
    Weights().CopyFromHostFp32(host_weights.data(),weight_count);

    if(UseBias()==true)
    {
      Bias().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceDenseLayer<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    WeightGradients().FillZero();
    if(UseBias()==true)
    {
      BiasGradients().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceDenseLayer<ComputeT,StorageT>::TotalParameterCount()const
{
  try
  {
    size_t count=static_cast<size_t>(InputSize())*OutputSize();
    if(UseBias()==true)
    {
      count+=OutputSize();
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
    if(UseBias()==true)
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
      return Weights();
    }
    if(index==1&&UseBias()==true)
    {
      return Bias();
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
      return Weights();
    }
    if(index==1&&UseBias()==true)
    {
      return Bias();
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
      return WeightGradients();
    }
    if(index==1&&UseBias()==true)
    {
      return BiasGradients();
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
      return WeightGradients();
    }
    if(index==1&&UseBias()==true)
    {
      return BiasGradients();
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
    return g_serial_tag_dense+
           g_serial_open_paren+
           std::to_string(InputSize())+
           g_serial_comma+
           std::to_string(OutputSize())+
           g_serial_close_paren;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceDenseLayer<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
    std::vector<std::string> names;
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::GenericWeight_e));
    if(UseBias()==true)
    {
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::GenericBias_e));
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
    if(shape[1]!=InputSize())
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Forward: input features must match input_size");
    }

    AssertInputDtype(input);

    const uint32_t batch_size=shape[0];
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    if(batch_size!=OutputBatch())
    {
      SetOutputBuffer(CAIF_DeviceTensor::Uninitialized({batch_size,OutputSize()},
                                                       ctx.Stream(),sdt));
      SetOutputBatch(batch_size);
    }

    if(UseBias()==true)
    {
      CAIF_Ops::MatMulBias(input,Weights(),Bias(),OutputBuffer(),
                           ctx.Stream().Handle(),ctx,cdt);
    }
    else
    {
      CAIF_Ops::MatMul(input,Weights(),OutputBuffer(),ctx,cdt);
    }

    if(ctx.Training()==true)
    {
      SetLastInput(input.Clone());
      SetLastPreactivation(OutputBuffer().Clone());
    }

    if(Activation()==CAIF_DeviceActivation::CAIF_DeviceActivation_e::None)
    {
      return OutputBuffer().Clone();
    }

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({batch_size,OutputSize()},
                                                              ctx.Stream(),sdt);

    switch(Activation())
    {
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::None:
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU:
        CAIF_Ops::ReLU(OutputBuffer(),output);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Sigmoid:
        CAIF_Ops::Sigmoid(OutputBuffer(),output);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Tanh:
        CAIF_Ops::Tanh(OutputBuffer(),output);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Softmax:
        CAIF_Ops::Softmax(OutputBuffer(),output);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::LeakyReLU:
        CAIF_Ops::LeakyReLU(OutputBuffer(),output);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::ELU:
        CAIF_Ops::ELU(OutputBuffer(),output);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::GELU:
        CAIF_Ops::GELU(OutputBuffer(),
                       output,
                       CAIF_GELUApproximation::CAIF_GELUApproximation_e::Tanh);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Swish:
        CAIF_Ops::Swish(OutputBuffer(),output);
        break;
      case CAIF_DeviceActivation::CAIF_DeviceActivation_e::GELUExact:
        CAIF_Ops::GELU(OutputBuffer(),
                       output,
                       CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);
        break;
    }

    if(ctx.Training()==true)
    {
      if(Activation()==CAIF_DeviceActivation::CAIF_DeviceActivation_e::Sigmoid||
         Activation()==CAIF_DeviceActivation::CAIF_DeviceActivation_e::Tanh||
         Activation()==CAIF_DeviceActivation::CAIF_DeviceActivation_e::Softmax||
         Activation()==CAIF_DeviceActivation::CAIF_DeviceActivation_e::ELU||
         Activation()==CAIF_DeviceActivation::CAIF_DeviceActivation_e::Swish)
      {
        SetLastOutput(output.Clone());
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
    if(LastInput().IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Backward: must call Forward with training=true first");
    }

    const std::vector<uint32_t> &shape=grad_output.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Backward: grad_output must be 2D");
    }
    if(shape[1]!=OutputSize())
    {
      THROW_CAIFE("CAIF_DeviceDenseLayer::Backward: grad_output must match output_size");
    }

    const uint32_t batch_size=shape[0];
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    CAIF_DeviceTensor linear_grad;
    if(Activation()==CAIF_DeviceActivation::CAIF_DeviceActivation_e::None)
    {
      linear_grad=CAIF_DeviceTensor::WrapView(
                   const_cast<void *>(grad_output.DeviceDataRaw()),
                   {batch_size,OutputSize()},
                   ctx.Stream(),
                   grad_output.Dtype());
    }
    else
    {
      linear_grad=CAIF_DeviceTensor::Uninitialized({batch_size,OutputSize()},ctx.Stream(),sdt);
      switch(Activation())
      {
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::None:
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::ReLU:
          CAIF_Ops::ReLUBackward(grad_output,LastPreactivation(),linear_grad);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Sigmoid:
          CAIF_Ops::SigmoidBackward(grad_output,LastOutput(),linear_grad);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Tanh:
          CAIF_Ops::TanhBackward(grad_output,LastOutput(),linear_grad);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Softmax:
          CAIF_Ops::SoftmaxBackward(grad_output,LastOutput(),linear_grad);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::LeakyReLU:
          CAIF_Ops::LeakyReLUBackward(grad_output,LastPreactivation(),linear_grad);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::ELU:
          CAIF_Ops::ELUBackward(grad_output,LastPreactivation(),LastOutput(),linear_grad);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::GELU:
          CAIF_Ops::GELUBackward(grad_output,
                                 LastPreactivation(),
                                 linear_grad,
                                 CAIF_GELUApproximation::CAIF_GELUApproximation_e::Tanh);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::Swish:
          CAIF_Ops::SwishBackward(grad_output,LastPreactivation(),LastOutput(),linear_grad);
          break;
        case CAIF_DeviceActivation::CAIF_DeviceActivation_e::GELUExact:
          CAIF_Ops::GELUBackward(grad_output,
                                 LastPreactivation(),
                                 linear_grad,
                                 CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);
          break;
      }
    }

    // Accumulate (delta + Add) rather than overwrite, so several backward
    // passes before a ZeroGradients sum — matching CAIF_DeviceFFN and
    // CAIF_DeviceLinearHead and PyTorch-style gradient accumulation.
    CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized(WeightGradients().Shape(),
                                                                    ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeA(LastInput(),linear_grad,grad_w_delta,ctx,cdt);
    CAIF_Ops::Add(WeightGradients(),grad_w_delta,WeightGradients());

    if(UseBias()==true)
    {
      CAIF_DeviceTensor bias_grad_delta=CAIF_DeviceTensor::Uninitialized(BiasGradients().Shape(),
                                                                         ctx.Stream(),sdt);
      CAIF_Ops::BiasGradient(linear_grad,bias_grad_delta);
      CAIF_Ops::Add(BiasGradients(),bias_grad_delta,BiasGradients());
    }

    CAIF_DeviceTensor input_grad=CAIF_DeviceTensor::Uninitialized({batch_size,InputSize()},
                                                                  ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeB(linear_grad,Weights(),input_grad,ctx,cdt);

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
