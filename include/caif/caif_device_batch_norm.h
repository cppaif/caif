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
// Device-resident batch-normalization layer (templated on
// <ComputeT, StorageT>).
//
// Uniform two-parameter signature with both defaulting to `float`.
// input/output follow StorageT (loaded as StorageT, math promoted to
// float, stored as StorageT); per-feature statistics
// (gamma/beta/running_mean/running_var) stay fp32 — the standard
// BatchNorm convention. Two paths:
//   - Host_e: existing CPU loop (works for any StorageT via cast-to-float).
//   - Device_e fp32 / fp16 / bf16: cuDNN device backend (CUDNN_BATCHNORM_SPATIAL
//     over [row_count, features, 1, 1] NCHW reshape).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_data_type.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceBatchNorm:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    CAIF_DeviceBatchNorm(uint32_t num_features,
                         CAIF_CudaStream &stream,
                         float epsilon=g_caif_epsilon,
                         float momentum=g_caif_default_momentum);
    ~CAIF_DeviceBatchNorm()override=default;

    CAIF_DeviceBatchNorm(CAIF_DeviceBatchNorm &&other);
    CAIF_DeviceBatchNorm &operator=(CAIF_DeviceBatchNorm &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::BatchNorm_e;
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

    uint32_t NumFeatures()const{return _num_features;}
    float Epsilon()const{return _epsilon;}
    float Momentum()const{return _momentum;}

    const CAIF_DeviceTensor &RunningMean()const{return _running_mean;}
    const CAIF_DeviceTensor &RunningVar()const{return _running_var;}

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
    CAIF_DeviceTensor ForwardHost(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx);
    CAIF_DeviceTensor BackwardHost(const CAIF_DeviceTensor &grad_output);
    CAIF_DeviceTensor ForwardDevice(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx);
    CAIF_DeviceTensor BackwardDevice(const CAIF_DeviceTensor &grad_output);

    // Internal accessors — every member access in method bodies routes
    // through these so subclasses can override storage / instrumentation.
    uint32_t NumFeaturesInternal()const{return _num_features;}
    void SetNumFeatures(const uint32_t v){_num_features=v;}
    float EpsilonInternal()const{return _epsilon;}
    void SetEpsilon(const float v){_epsilon=v;}
    float MomentumInternal()const{return _momentum;}
    void SetMomentum(const float v){_momentum=v;}
    void SetGamma(CAIF_DeviceTensor t){_gamma=std::move(t);}
    void SetBeta(CAIF_DeviceTensor t){_beta=std::move(t);}
    void SetGammaGrad(CAIF_DeviceTensor t){_gamma_grad=std::move(t);}
    void SetBetaGrad(CAIF_DeviceTensor t){_beta_grad=std::move(t);}
    void SetRunningMean(CAIF_DeviceTensor t){_running_mean=std::move(t);}
    void SetRunningVar(CAIF_DeviceTensor t){_running_var=std::move(t);}
    std::vector<uint32_t> &CachedInputShapeMutable(){return _cached_input_shape;}
    void SetCachedMean(std::vector<float> v){_cached_mean=std::move(v);}
    void SetCachedInvStd(std::vector<float> v){_cached_inv_std=std::move(v);}
    void SetCachedNormalized(std::vector<float> v){_cached_normalized=std::move(v);}

    CAIF_DeviceTensor &Gamma(){return _gamma;}
    const CAIF_DeviceTensor &Gamma()const{return _gamma;}
    CAIF_DeviceTensor &Beta(){return _beta;}
    const CAIF_DeviceTensor &Beta()const{return _beta;}
    CAIF_DeviceTensor &GammaGrad(){return _gamma_grad;}
    const CAIF_DeviceTensor &GammaGrad()const{return _gamma_grad;}
    CAIF_DeviceTensor &BetaGrad(){return _beta_grad;}
    const CAIF_DeviceTensor &BetaGrad()const{return _beta_grad;}
    CAIF_DeviceTensor &RunningMeanMutable(){return _running_mean;}
    CAIF_DeviceTensor &RunningVarMutable(){return _running_var;}

    const std::vector<uint32_t> &CachedInputShape()const{return _cached_input_shape;}
    void SetCachedInputShape(const std::vector<uint32_t> &s){_cached_input_shape=s;}
    std::vector<float> &CachedMean(){return _cached_mean;}
    std::vector<float> &CachedInvStd(){return _cached_inv_std;}
    std::vector<float> &CachedNormalized(){return _cached_normalized;}
    const std::vector<float> &CachedInvStd()const{return _cached_inv_std;}
    const std::vector<float> &CachedNormalized()const{return _cached_normalized;}

    CAIF_DeviceTensor &CachedInputDevice(){return _cached_input_device;}
    const CAIF_DeviceTensor &CachedInputDevice()const{return _cached_input_device;}
    void SetCachedInputDevice(CAIF_DeviceTensor &&t){_cached_input_device=std::move(t);}
    CAIF_DeviceTensor &CachedSaveMeanDevice(){return _cached_save_mean_device;}
    const CAIF_DeviceTensor &CachedSaveMeanDevice()const{return _cached_save_mean_device;}
    void SetCachedSaveMeanDevice(CAIF_DeviceTensor &&t){_cached_save_mean_device=std::move(t);}
    CAIF_DeviceTensor &CachedSaveInvVarDevice(){return _cached_save_inv_var_device;}
    const CAIF_DeviceTensor &CachedSaveInvVarDevice()const{return _cached_save_inv_var_device;}
    void SetCachedSaveInvVarDevice(CAIF_DeviceTensor &&t){_cached_save_inv_var_device=std::move(t);}

  private:
    uint32_t _num_features;
    float _epsilon;
    float _momentum;

    // Authoritative host-side fp32 parameters and running stats.
    CAIF_DeviceTensor _gamma;        // [features] fp32 host
    CAIF_DeviceTensor _beta;         // [features] fp32 host
    CAIF_DeviceTensor _gamma_grad;   // [features] fp32 host
    CAIF_DeviceTensor _beta_grad;    // [features] fp32 host
    CAIF_DeviceTensor _running_mean; // [features] fp32 host
    CAIF_DeviceTensor _running_var;  // [features] fp32 host

    // Host-path forward-to-backward caches.
    std::vector<uint32_t> _cached_input_shape;
    std::vector<float> _cached_mean;
    std::vector<float> _cached_inv_std;
    std::vector<float> _cached_normalized;

    // Device-path caches: input is needed by cuDNN backward, plus the
    // saved-mean / saved-inv-variance returned by ForwardTraining.
    CAIF_DeviceTensor _cached_input_device;        // StorageT [N, features, 1, 1]
    CAIF_DeviceTensor _cached_save_mean_device;    // [features] fp32
    CAIF_DeviceTensor _cached_save_inv_var_device; // [features] fp32
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceBatchNorm<float,float>;
extern template class CAIF_DeviceBatchNorm<float,__half>;
extern template class CAIF_DeviceBatchNorm<float,__nv_bfloat16>;
extern template class CAIF_DeviceBatchNorm<__half,float>;
extern template class CAIF_DeviceBatchNorm<__half,__half>;
extern template class CAIF_DeviceBatchNorm<__half,__nv_bfloat16>;
extern template class CAIF_DeviceBatchNorm<__nv_bfloat16,float>;
extern template class CAIF_DeviceBatchNorm<__nv_bfloat16,__half>;
extern template class CAIF_DeviceBatchNorm<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceBatchNorm<float,float>;
#endif

}//end instance namespace
