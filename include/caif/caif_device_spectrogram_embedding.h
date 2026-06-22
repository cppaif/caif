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
#include "caif_device_spectrogram_embedding_config.h"
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

    CAIF_DeviceSpectrogramEmbedding(const CAIF_DeviceSpectrogramEmbeddingConfig &config,
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

    uint32_t FreqBins()const{return Config().FreqBins();}
    uint32_t Dim()const{return Config().Dim();}
    bool UseCLSToken()const{return Config().UseCLSToken();}

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
    const CAIF_DeviceSpectrogramEmbeddingConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DeviceSpectrogramEmbeddingConfig &c){_config=c;}

    const CAIF_DeviceTensor &WProj()const{return _w_proj;}
    CAIF_DeviceTensor &WProjMut(){return _w_proj;}
    void SetWProj(CAIF_DeviceTensor &&t){_w_proj=std::move(t);}

    const CAIF_DeviceTensor &BProj()const{return _b_proj;}
    CAIF_DeviceTensor &BProj(){return _b_proj;}
    void SetBProj(CAIF_DeviceTensor &&t){_b_proj=std::move(t);}

    const CAIF_DeviceTensor &CLSToken()const{return _cls_token;}
    CAIF_DeviceTensor &CLSTokenMut(){return _cls_token;}
    void SetCLSToken(CAIF_DeviceTensor &&t){_cls_token=std::move(t);}

    const CAIF_DeviceTensor &GradWProj()const{return _grad_w_proj;}
    CAIF_DeviceTensor &GradWProj(){return _grad_w_proj;}
    void SetGradWProj(CAIF_DeviceTensor &&t){_grad_w_proj=std::move(t);}

    const CAIF_DeviceTensor &GradBProj()const{return _grad_b_proj;}
    CAIF_DeviceTensor &GradBProj(){return _grad_b_proj;}
    void SetGradBProj(CAIF_DeviceTensor &&t){_grad_b_proj=std::move(t);}

    const CAIF_DeviceTensor &GradCLS()const{return _grad_cls;}
    CAIF_DeviceTensor &GradCLS(){return _grad_cls;}
    void SetGradCLS(CAIF_DeviceTensor &&t){_grad_cls=std::move(t);}

    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    CAIF_DeviceTensor &CachedInput(){return _cached_input;}
    void SetCachedInput(CAIF_DeviceTensor &&t){_cached_input=std::move(t);}

    uint32_t CachedBatch()const{return _cached_batch;}
    void SetCachedBatch(const uint32_t v){_cached_batch=v;}
    uint32_t CachedTimeFrames()const{return _cached_time_frames;}
    void SetCachedTimeFrames(const uint32_t v){_cached_time_frames=v;}

    CAIF_DeviceSpectrogramEmbeddingConfig _config;

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
