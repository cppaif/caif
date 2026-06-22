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
// Device-resident LayerNorm layer (templated on <ComputeT, StorageT>).
//
// Uniform two-parameter signature with both defaulting to `float`. Every
// (ComputeT, StorageT) cell from the cuBLAS-Lt-supported grid is a legal
// instantiation. The runtime factory CAIF_DeviceLayerNormFactory::Create
// (in caif_device_layernorm_factory.h) is the bridge for callers that
// have the dtypes only as runtime values.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceLayerNorm:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    CAIF_DeviceLayerNorm(uint32_t dim,
                         CAIF_CudaStream &stream,
                         float epsilon=g_caif_epsilon);
    ~CAIF_DeviceLayerNorm()override=default;

    // Move
    CAIF_DeviceLayerNorm(CAIF_DeviceLayerNorm &&other);
    CAIF_DeviceLayerNorm &operator=(CAIF_DeviceLayerNorm &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::LayerNorm_e;
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

    // Accessors (StorageDtype()/ComputeDtype() inherited from
    // CAIF_DeviceLayerTyped — no per-layer copy needed).
    uint32_t Dim()const{return _dim;}
    void SetDim(uint32_t dim){_dim=dim;}
    float Epsilon()const{return _epsilon;}
    void SetEpsilon(float epsilon){_epsilon=epsilon;}

    CAIF_DeviceTensor &Gamma(){return _gamma;}
    const CAIF_DeviceTensor &Gamma()const{return _gamma;}
    void SetGamma(CAIF_DeviceTensor &&gamma){_gamma=std::move(gamma);}

    CAIF_DeviceTensor &Beta(){return _beta;}
    const CAIF_DeviceTensor &Beta()const{return _beta;}
    void SetBeta(CAIF_DeviceTensor &&beta){_beta=std::move(beta);}

    CAIF_DeviceTensor &GammaGrad(){return _gamma_grad;}
    const CAIF_DeviceTensor &GammaGrad()const{return _gamma_grad;}
    void SetGammaGrad(CAIF_DeviceTensor &&gamma_grad){_gamma_grad=std::move(gamma_grad);}

    CAIF_DeviceTensor &BetaGrad(){return _beta_grad;}
    const CAIF_DeviceTensor &BetaGrad()const{return _beta_grad;}
    void SetBetaGrad(CAIF_DeviceTensor &&beta_grad){_beta_grad=std::move(beta_grad);}

    int CachedRows()const{return _cached_rows;}
    void SetCachedRows(int rows){_cached_rows=rows;}

    CAIF_DeviceTensor &LastInput(){return _last_input;}
    const CAIF_DeviceTensor &LastInput()const{return _last_input;}
    void SetLastInput(CAIF_DeviceTensor &&last_input){_last_input=std::move(last_input);}

    CAIF_DeviceTensor &MeanCache(){return _mean_cache;}
    const CAIF_DeviceTensor &MeanCache()const{return _mean_cache;}
    void SetMeanCache(CAIF_DeviceTensor &&mean_cache){_mean_cache=std::move(mean_cache);}

    CAIF_DeviceTensor &RstdCache(){return _rstd_cache;}
    const CAIF_DeviceTensor &RstdCache()const{return _rstd_cache;}
    void SetRstdCache(CAIF_DeviceTensor &&rstd_cache){_rstd_cache=std::move(rstd_cache);}

    /**
     * @brief Replace gamma (scale) with a tensor of shape [dim].
     */
    void LoadGamma(CAIF_DeviceTensor &&gamma);

    /**
     * @brief Replace beta (shift) with a tensor of shape [dim].
     */
    void LoadBeta(CAIF_DeviceTensor &&beta);

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
    uint32_t _dim;
    float _epsilon;
    CAIF_DeviceTensor _gamma;       // [dim] fp32 (kernel signature)
    CAIF_DeviceTensor _beta;        // [dim] fp32 (kernel signature)
    CAIF_DeviceTensor _gamma_grad;  // [dim] fp32
    CAIF_DeviceTensor _beta_grad;   // [dim] fp32

    int _cached_rows;

    // Cached for backward
    CAIF_DeviceTensor _last_input;  // [rows, dim] at StorageDtype()
    CAIF_DeviceTensor _mean_cache;  // [rows] fp32
    CAIF_DeviceTensor _rstd_cache;  // [rows] fp32
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceLayerNorm<float,float>;
extern template class CAIF_DeviceLayerNorm<float,__half>;
extern template class CAIF_DeviceLayerNorm<float,__nv_bfloat16>;
extern template class CAIF_DeviceLayerNorm<__half,float>;
extern template class CAIF_DeviceLayerNorm<__half,__half>;
extern template class CAIF_DeviceLayerNorm<__half,__nv_bfloat16>;
extern template class CAIF_DeviceLayerNorm<__nv_bfloat16,float>;
extern template class CAIF_DeviceLayerNorm<__nv_bfloat16,__half>;
extern template class CAIF_DeviceLayerNorm<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceLayerNorm<float,float>;
#endif

}//end instance namespace
