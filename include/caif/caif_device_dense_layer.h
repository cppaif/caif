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
// Device-resident dense (fully-connected) layer (templated on
// <ComputeT, StorageT>).
//
// Storage tensors (weights/bias/grads/last_input/preactivation/output) follow
// StorageT. ComputeT selects the cuBLAS-Lt compute_type for MatMul. Every
// (ComputeT, StorageT) cell from the cuBLAS-Lt grid is a legal
// instantiation. The runtime factory CAIF_DeviceDenseLayerFactory::Create
// (in caif_device_dense_layer_factory.h) is the bridge for callers that
// have the dtypes only as runtime values.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <vector>
#include <cstdint>
#include <string>

namespace instance
{

enum class CAIF_DeviceActivation_e
{
  None,
  ReLU,
  Sigmoid,
  Tanh,
  Softmax,
  LeakyReLU,
  ELU,
  GELU,
  Swish
};

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceDenseLayer:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    CAIF_DeviceDenseLayer(uint32_t input_size,
                          uint32_t output_size,
                          CAIF_DeviceActivation_e activation,
                          CAIF_CudaStream &stream,
                          bool use_bias=true);

    ~CAIF_DeviceDenseLayer()override=default;

    CAIF_DeviceDenseLayer(CAIF_DeviceDenseLayer &&other);
    CAIF_DeviceDenseLayer &operator=(CAIF_DeviceDenseLayer &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;

    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Dense_e;
    }

    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    void InitializeWeights(uint32_t seed=0)override;

    CAIF_DeviceTensor &Weights(){return _weights;}
    const CAIF_DeviceTensor &Weights()const{return _weights;}
    CAIF_DeviceTensor &Bias(){return _bias;}
    const CAIF_DeviceTensor &Bias()const{return _bias;}
    CAIF_DeviceTensor &WeightGradients(){return _weight_grads;}
    const CAIF_DeviceTensor &WeightGradients()const{return _weight_grads;}
    CAIF_DeviceTensor &BiasGradients(){return _bias_grads;}
    const CAIF_DeviceTensor &BiasGradients()const{return _bias_grads;}

    uint32_t InputSize()const{return _input_size;}
    uint32_t OutputSize()const{return _output_size;}
    CAIF_DeviceActivation_e Activation()const{return _activation;}
    bool UseBias()const{return _use_bias;}

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

  private:
    uint32_t _input_size;
    uint32_t _output_size;
    CAIF_DeviceActivation_e _activation;
    bool _use_bias;

    CAIF_DeviceTensor _weights;
    CAIF_DeviceTensor _bias;
    CAIF_DeviceTensor _weight_grads;
    CAIF_DeviceTensor _bias_grads;

    CAIF_DeviceTensor _output_buffer;
    uint32_t _output_batch;

    CAIF_DeviceTensor _last_input;
    CAIF_DeviceTensor _last_preactivation;
    CAIF_DeviceTensor _last_output;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceDenseLayer<float,float>;
extern template class CAIF_DeviceDenseLayer<float,__half>;
extern template class CAIF_DeviceDenseLayer<float,__nv_bfloat16>;
extern template class CAIF_DeviceDenseLayer<__half,float>;
extern template class CAIF_DeviceDenseLayer<__half,__half>;
extern template class CAIF_DeviceDenseLayer<__half,__nv_bfloat16>;
extern template class CAIF_DeviceDenseLayer<__nv_bfloat16,float>;
extern template class CAIF_DeviceDenseLayer<__nv_bfloat16,__half>;
extern template class CAIF_DeviceDenseLayer<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceDenseLayer<float,float>;
#endif

}//end instance namespace
