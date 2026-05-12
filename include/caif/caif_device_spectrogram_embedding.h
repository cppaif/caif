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
// Device-resident spectrogram embedding layer (templated on
// <ComputeT, StorageT>).
//
// Uniform two-parameter signature with both defaulting to `float`.
// Storage tensors (W_proj / b_proj / cls_token / grads / cached input)
// follow StorageT. MatMul/BiasAdd/elementwise_add are dtype-aware so
// every (ComputeT, StorageT) cell runs natively.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_run_context.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceSpectrogramEmbedding:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    struct Config_t
    {
      uint32_t freq_bins;
      uint32_t dim;
      bool use_cls_token;
    };

    CAIF_DeviceSpectrogramEmbedding(const Config_t &config,
                                    CAIF_CudaStream &stream);
    ~CAIF_DeviceSpectrogramEmbedding()override=default;

    CAIF_DeviceSpectrogramEmbedding(CAIF_DeviceSpectrogramEmbedding &&other);
    CAIF_DeviceSpectrogramEmbedding &operator=(CAIF_DeviceSpectrogramEmbedding &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::SpectrogramEmbedding_e;
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

    uint32_t FreqBins()const{return _config.freq_bins;}
    uint32_t Dim()const{return _config.dim;}
    bool UseCLSToken()const{return _config.use_cls_token;}

    void InitializeWeights(uint32_t seed=0)override;

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
    const Config_t &Config()const{return _config;}
    CAIF_DeviceTensor &WProjMut(){return _w_proj;}
    CAIF_DeviceTensor &CLSTokenMut(){return _cls_token;}

    Config_t _config;

    CAIF_DeviceTensor _w_proj;
    CAIF_DeviceTensor _b_proj;
    CAIF_DeviceTensor _cls_token;

    CAIF_DeviceTensor _grad_w_proj;
    CAIF_DeviceTensor _grad_b_proj;
    CAIF_DeviceTensor _grad_cls;

    CAIF_DeviceTensor _cached_input;
    uint32_t _cached_batch;
    uint32_t _cached_time_frames;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceSpectrogramEmbedding<float,float>;
extern template class CAIF_DeviceSpectrogramEmbedding<float,__half>;
extern template class CAIF_DeviceSpectrogramEmbedding<float,__nv_bfloat16>;
extern template class CAIF_DeviceSpectrogramEmbedding<__half,float>;
extern template class CAIF_DeviceSpectrogramEmbedding<__half,__half>;
extern template class CAIF_DeviceSpectrogramEmbedding<__half,__nv_bfloat16>;
extern template class CAIF_DeviceSpectrogramEmbedding<__nv_bfloat16,float>;
extern template class CAIF_DeviceSpectrogramEmbedding<__nv_bfloat16,__half>;
extern template class CAIF_DeviceSpectrogramEmbedding<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceSpectrogramEmbedding<float,float>;
#endif

}//end instance namespace
