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
// Device-resident tabular embedding layer (templated on
// <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_tabular_embedding_config.h"
#include "caif_run_context.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceTabularEmbedding:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    CAIF_DeviceTabularEmbedding(const CAIF_DeviceTabularEmbeddingConfig &config,
                                CAIF_CudaStream &stream);
    ~CAIF_DeviceTabularEmbedding()override=default;

    CAIF_DeviceTabularEmbedding(CAIF_DeviceTabularEmbedding &&other);
    CAIF_DeviceTabularEmbedding &operator=(CAIF_DeviceTabularEmbedding &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::TabularEmbedding_e;
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

    uint32_t NumFeatures()const{return Config().NumFeatures();}
    uint32_t Dim()const{return Config().Dim();}
    const CAIF_DeviceTabularEmbeddingConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DeviceTabularEmbeddingConfig &config){_config=config;}

    CAIF_DeviceTensor &WProj(){return _w_proj;}
    const CAIF_DeviceTensor &WProj()const{return _w_proj;}
    void SetWProj(CAIF_DeviceTensor &&t){_w_proj=std::move(t);}

    CAIF_DeviceTensor &BProj(){return _b_proj;}
    const CAIF_DeviceTensor &BProj()const{return _b_proj;}
    void SetBProj(CAIF_DeviceTensor &&t){_b_proj=std::move(t);}

    CAIF_DeviceTensor &GradWProj(){return _grad_w_proj;}
    const CAIF_DeviceTensor &GradWProj()const{return _grad_w_proj;}
    void SetGradWProj(CAIF_DeviceTensor &&t){_grad_w_proj=std::move(t);}

    CAIF_DeviceTensor &GradBProj(){return _grad_b_proj;}
    const CAIF_DeviceTensor &GradBProj()const{return _grad_b_proj;}
    void SetGradBProj(CAIF_DeviceTensor &&t){_grad_b_proj=std::move(t);}

    CAIF_DeviceTensor &CachedInput(){return _cached_input;}
    const CAIF_DeviceTensor &CachedInput()const{return _cached_input;}
    void SetCachedInput(CAIF_DeviceTensor &&t){_cached_input=std::move(t);}

    uint32_t CachedBatch()const{return _cached_batch;}
    void SetCachedBatch(uint32_t b){_cached_batch=b;}

    uint32_t CachedSeqLen()const{return _cached_seq_len;}
    void SetCachedSeqLen(uint32_t s){_cached_seq_len=s;}

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
    CAIF_DeviceTensor &WProjMut(){return _w_proj;}

    CAIF_DeviceTabularEmbeddingConfig _config;

    CAIF_DeviceTensor _w_proj;
    CAIF_DeviceTensor _b_proj;
    CAIF_DeviceTensor _grad_w_proj;
    CAIF_DeviceTensor _grad_b_proj;

    CAIF_DeviceTensor _cached_input;
    uint32_t _cached_batch;
    uint32_t _cached_seq_len;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceTabularEmbedding<float,float>;
extern template class CAIF_DeviceTabularEmbedding<float,__half>;
extern template class CAIF_DeviceTabularEmbedding<float,__nv_bfloat16>;
extern template class CAIF_DeviceTabularEmbedding<__half,float>;
extern template class CAIF_DeviceTabularEmbedding<__half,__half>;
extern template class CAIF_DeviceTabularEmbedding<__half,__nv_bfloat16>;
extern template class CAIF_DeviceTabularEmbedding<__nv_bfloat16,float>;
extern template class CAIF_DeviceTabularEmbedding<__nv_bfloat16,__half>;
extern template class CAIF_DeviceTabularEmbedding<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceTabularEmbedding<float,float>;
#endif

}//end instance namespace
