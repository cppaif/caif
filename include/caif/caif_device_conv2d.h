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
// Device-resident 2D convolution layer (templated on <ComputeT, StorageT>).
//
// Stage 5b: host-located, fp32-only path. Non-fp32 cells throw at
// ForwardImpl until cuDNN device backend lands.
//
// Input:   [N, H, W, Cin]
// Kernel:  [Kh, Kw, Cin, Cout]
// Bias:    [Cout]
// Output:  [N, H_out, W_out, Cout]
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_conv2d_config.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceConv2D:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    typedef CAIF_DeviceLayerTyped<ComputeT,StorageT> Base_t;


    CAIF_DeviceConv2D(const CAIF_DeviceConv2DConfig &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceConv2D()override=default;

    CAIF_DeviceConv2D(CAIF_DeviceConv2D &&other);
    CAIF_DeviceConv2D &operator=(CAIF_DeviceConv2D &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Conv2D_e;
    }

    void ZeroGradients()override;
    size_t ParameterTensorCount()const override{return 2;}
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    const CAIF_DeviceConv2DConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DeviceConv2DConfig &c){_config=c;}

  public:
    using Base_t::StorageDtype;
    using Base_t::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using Base_t::AssertInputDtype;
    using Base_t::AllocateOutput;
    using Base_t::CublasComputeType;
    using Base_t::StoragePtr;

  private:
    // Internal accessors for own member access (per coding-guideline rule
    // "Member access via accessors, even from inside the class").
    CAIF_DeviceTensor &Weights(){return _weights;}
    const CAIF_DeviceTensor &Weights()const{return _weights;}
    void SetWeights(CAIF_DeviceTensor t){_weights=std::move(t);}
    CAIF_DeviceTensor &Bias(){return _bias;}
    const CAIF_DeviceTensor &Bias()const{return _bias;}
    void SetBias(CAIF_DeviceTensor t){_bias=std::move(t);}
    CAIF_DeviceTensor &WeightsGrad(){return _weights_grad;}
    const CAIF_DeviceTensor &WeightsGrad()const{return _weights_grad;}
    void SetWeightsGrad(CAIF_DeviceTensor t){_weights_grad=std::move(t);}
    CAIF_DeviceTensor &BiasGrad(){return _bias_grad;}
    const CAIF_DeviceTensor &BiasGrad()const{return _bias_grad;}
    void SetBiasGrad(CAIF_DeviceTensor t){_bias_grad=std::move(t);}

    const std::vector<uint32_t> &CachedInputShape()const{return _cached_input_shape;}
    void SetCachedInputShape(const std::vector<uint32_t> &shape){_cached_input_shape=shape;}

    const std::vector<float> &CachedInputHost()const{return _cached_input_host;}
    std::vector<float> &CachedInputHost(){return _cached_input_host;}
    void SetCachedInputHost(std::vector<float> &&v){_cached_input_host=std::move(v);}

    CAIF_DeviceTensor &CachedInputDevice(){return _cached_input_device;}
    const CAIF_DeviceTensor &CachedInputDevice()const{return _cached_input_device;}
    void SetCachedInputDevice(CAIF_DeviceTensor &&t){_cached_input_device=std::move(t);}

    CAIF_DeviceTensor &WeightsDeviceKRSC(){return _weights_device_krsc;}
    const CAIF_DeviceTensor &WeightsDeviceKRSC()const{return _weights_device_krsc;}
    void SetWeightsDeviceKRSC(CAIF_DeviceTensor &&t){_weights_device_krsc=std::move(t);}

    CAIF_DeviceTensor &BiasDevice(){return _bias_device;}
    const CAIF_DeviceTensor &BiasDevice()const{return _bias_device;}
    void SetBiasDevice(CAIF_DeviceTensor &&t){_bias_device=std::move(t);}

    CAIF_DeviceTensor &WeightsGradDeviceKRSC(){return _weights_grad_device_krsc;}
    const CAIF_DeviceTensor &WeightsGradDeviceKRSC()const{return _weights_grad_device_krsc;}
    void SetWeightsGradDeviceKRSC(CAIF_DeviceTensor &&t){_weights_grad_device_krsc=std::move(t);}

    CAIF_DeviceTensor &BiasGradDevice(){return _bias_grad_device;}
    const CAIF_DeviceTensor &BiasGradDevice()const{return _bias_grad_device;}
    void SetBiasGradDevice(CAIF_DeviceTensor &&t){_bias_grad_device=std::move(t);}

    CAIF_DeviceConv2DConfig _config;

    // Canonical host-side weights/bias/grads (HWCK layout for weights, fp32).
    // Used by host fp32 path and by optimizer integration.
    CAIF_DeviceTensor _weights;
    CAIF_DeviceTensor _bias;
    CAIF_DeviceTensor _weights_grad;
    CAIF_DeviceTensor _bias_grad;

    // Device-side mirror tensors used by the cuDNN path. Weights live in
    // KRSC layout (=[Cout, Kh, Kw, Cin]) to match cudnnSetFilter4dDescriptor
    // with CUDNN_TENSOR_NHWC. Synced from host on each device-path Forward;
    // grads synced back to host after each device-path Backward.
    CAIF_DeviceTensor _weights_device_krsc;
    CAIF_DeviceTensor _bias_device;
    CAIF_DeviceTensor _weights_grad_device_krsc;
    CAIF_DeviceTensor _bias_grad_device;

    std::vector<uint32_t> _cached_input_shape;
    std::vector<float> _cached_input_host;        // host-path cache
    CAIF_DeviceTensor _cached_input_device;       // device-path cache
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceConv2D<float,float>;
extern template class CAIF_DeviceConv2D<float,__half>;
extern template class CAIF_DeviceConv2D<float,__nv_bfloat16>;
extern template class CAIF_DeviceConv2D<__half,float>;
extern template class CAIF_DeviceConv2D<__half,__half>;
extern template class CAIF_DeviceConv2D<__half,__nv_bfloat16>;
extern template class CAIF_DeviceConv2D<__nv_bfloat16,float>;
extern template class CAIF_DeviceConv2D<__nv_bfloat16,__half>;
extern template class CAIF_DeviceConv2D<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceConv2D<float,float>;
#endif

}//end instance namespace
