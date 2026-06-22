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
// Device-resident batch-normalisation layer implementation.
//
// Two paths, branched on input.Location():
//   - Host_e: existing CPU loop. Works for any StorageT (cast to float on
//     read, cast back on write). Per-feature stats stay fp32.
//   - Device_e: cuDNN CUDNN_BATCHNORM_SPATIAL over an NCHW reshape
//     [row_count, features, 1, 1]. cuDNN's spatial mode normalises over
//     the (N,H,W) axes per channel, exactly matching the host loop's
//     "average across rows for each feature" contract. Bn-scale / bn-bias
//     and running stats stay fp32 per cuDNN's API requirement.
//------------------------------------------------------------------------------

#include "caif_device_batch_norm.h"
#include "caif_device_batch_norm_factory.h"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_cudnn_util.h"
#include "caif_device_context.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include <cmath>
#include <cstring>
#include <vector>

#ifdef USE_CAIF_CUDA
#include "caif_batch_norm_cudnn_helpers.h"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{


template<typename ComputeT,typename StorageT>
CAIF_DeviceBatchNorm<ComputeT,StorageT>::CAIF_DeviceBatchNorm(uint32_t num_features,
                                                              CAIF_CudaStream &stream,
                                                              float epsilon,
                                                              float momentum):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _num_features(num_features),
                                          _epsilon(epsilon),
                                          _momentum(momentum),
                                          _gamma(),
                                          _beta(),
                                          _gamma_grad(),
                                          _beta_grad(),
                                          _running_mean(),
                                          _running_var(),
                                          _cached_input_shape(),
                                          _cached_mean(),
                                          _cached_inv_std(),
                                          _cached_normalized()
{
  try
  {
    if(num_features<1)
    {
      THROW_CAIFE("CAIF_DeviceBatchNorm: num_features must be >= 1");
    }
    const std::vector<uint32_t> feat_shape={num_features};

    Gamma()=CAIF_DeviceTensor::ZerosHost(feat_shape);
    Beta()=CAIF_DeviceTensor::ZerosHost(feat_shape);
    GammaGrad()=CAIF_DeviceTensor::ZerosHost(feat_shape);
    BetaGrad()=CAIF_DeviceTensor::ZerosHost(feat_shape);
    RunningMeanMutable()=CAIF_DeviceTensor::ZerosHost(feat_shape);
    RunningVarMutable()=CAIF_DeviceTensor::ZerosHost(feat_shape);

    std::vector<float> ones(num_features,1.0f);
    std::memcpy(Gamma().DeviceDataRaw(),ones.data(),num_features*sizeof(float));
    std::memcpy(RunningVarMutable().DeviceDataRaw(),
                ones.data(),
                num_features*sizeof(float));
  }
  CAIF_CATCH_BLOCK();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceBatchNorm<ComputeT,StorageT>::CAIF_DeviceBatchNorm(CAIF_DeviceBatchNorm &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _num_features(other.NumFeatures()),
                              _epsilon(other.Epsilon()),
                              _momentum(other.Momentum()),
                              _gamma(std::move(other.Gamma())),
                              _beta(std::move(other.Beta())),
                              _gamma_grad(std::move(other.GammaGrad())),
                              _beta_grad(std::move(other.BetaGrad())),
                              _running_mean(std::move(other.RunningMeanMutable())),
                              _running_var(std::move(other.RunningVarMutable())),
                              _cached_input_shape(std::move(other.CachedInputShapeMutable())),
                              _cached_mean(std::move(other.CachedMean())),
                              _cached_inv_std(std::move(other.CachedInvStd())),
                              _cached_normalized(std::move(other.CachedNormalized())),
                              _cached_input_device(std::move(other.CachedInputDevice())),
                              _cached_save_mean_device(std::move(other.CachedSaveMeanDevice())),
                              _cached_save_inv_var_device(std::move(other.CachedSaveInvVarDevice()))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceBatchNorm<ComputeT,StorageT> &
CAIF_DeviceBatchNorm<ComputeT,StorageT>::operator=(CAIF_DeviceBatchNorm &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      SetNumFeatures(other.NumFeatures());
      SetEpsilon(other.Epsilon());
      SetMomentum(other.Momentum());
      SetGamma(std::move(other.Gamma()));
      SetBeta(std::move(other.Beta()));
      SetGammaGrad(std::move(other.GammaGrad()));
      SetBetaGrad(std::move(other.BetaGrad()));
      SetRunningMean(std::move(other.RunningMeanMutable()));
      SetRunningVar(std::move(other.RunningVarMutable()));
      SetCachedInputShape(std::move(other.CachedInputShapeMutable()));
      SetCachedMean(std::move(other.CachedMean()));
      SetCachedInvStd(std::move(other.CachedInvStd()));
      SetCachedNormalized(std::move(other.CachedNormalized()));
      SetCachedInputDevice(std::move(other.CachedInputDevice()));
      SetCachedSaveMeanDevice(std::move(other.CachedSaveMeanDevice()));
      SetCachedSaveInvVarDevice(std::move(other.CachedSaveInvVarDevice()));
    }
    return *this;
  }
  CAIF_CATCH_BLOCK();
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceBatchNorm<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                     CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);
    const std::vector<uint32_t> &in_shape=input.Shape();
    if(in_shape.empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceBatchNorm: input shape must be non-empty");
    }
    const uint32_t feature_axis=in_shape.back();
    if(feature_axis!=NumFeatures())
    {
      THROW_CAIFE("CAIF_DeviceBatchNorm: input feature dim mismatch");
    }
    if(input.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      return ForwardHost(input,ctx);
    }
#ifdef USE_CAIF_CUDA
    return ForwardDevice(input,ctx);
#else
    THROW_CAIFE("CAIF_DeviceBatchNorm: device path requires USE_CAIF_CUDA");
#endif
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceBatchNorm<ComputeT,StorageT>::ForwardHost(const CAIF_DeviceTensor &input,
                                                     CAIF_RunContext &ctx)
{
  try
  {
    const std::vector<uint32_t> &in_shape=input.Shape();
    const uint32_t feature_axis=in_shape.back();
    const size_t total_elems=input.TotalElements();
    const size_t row_count=total_elems/feature_axis;
    const StorageT *x=StoragePtr(input);
    // fp32 by contract
    const float *gamma=static_cast<const float *>(Gamma().DeviceDataRaw());
    // fp32 by contract
    const float *beta=static_cast<const float *>(Beta().DeviceDataRaw());
    // fp32 by contract
    float *running_mean=static_cast<float *>(RunningMeanMutable().DeviceDataRaw());
    // fp32 by contract
    float *running_var=static_cast<float *>(RunningVarMutable().DeviceDataRaw());
    const float momentum=MomentumInternal();
    const float epsilon=EpsilonInternal();

    CAIF_DeviceTensor output=CAIF_DeviceTensor::ZerosHost(in_shape,StorageDtype());
    StorageT *y=StoragePtr(output);

    SetCachedInputShape(in_shape);
    CachedMean().assign(feature_axis,0.0f);
    CachedInvStd().assign(feature_axis,0.0f);
    CachedNormalized().assign(total_elems,0.0f);

    if(ctx.Training()==true)
    {
      for(size_t f=0;f<feature_axis;++f)
      {
        double sum=0.0;
        double sum_sq=0.0;
        for(size_t r=0;r<row_count;++r)
        {
          const float v=static_cast<float>(x[r*feature_axis+f]);
          sum+=static_cast<double>(v);
          sum_sq+=static_cast<double>(v)*static_cast<double>(v);
        }
        const double mean=sum/static_cast<double>(row_count);
        const double var=sum_sq/static_cast<double>(row_count)-mean*mean;
        const double inv_std=1.0/std::sqrt(var+static_cast<double>(epsilon));
        CachedMean()[f]=static_cast<float>(mean);
        CachedInvStd()[f]=static_cast<float>(inv_std);
        running_mean[f]=(1.0f-momentum)*running_mean[f]+
                        momentum*static_cast<float>(mean);
        running_var[f]=(1.0f-momentum)*running_var[f]+
                       momentum*static_cast<float>(var);
        for(size_t r=0;r<row_count;++r)
        {
          const size_t idx=r*feature_axis+f;
          const float xv=static_cast<float>(x[idx]);
          const float normalized=static_cast<float>((static_cast<double>(xv)-mean)*inv_std);
          CachedNormalized()[idx]=normalized;
          y[idx]=static_cast<StorageT>(gamma[f]*normalized+beta[f]);
        }
      }
    }
    else
    {
      for(size_t f=0;f<feature_axis;++f)
      {
        const double inv_std=1.0/std::sqrt(static_cast<double>(running_var[f])+
                                            static_cast<double>(epsilon));
        for(size_t r=0;r<row_count;++r)
        {
          const size_t idx=r*feature_axis+f;
          const float xv=static_cast<float>(x[idx]);
          const float normalized=static_cast<float>((static_cast<double>(xv)-
                                                      static_cast<double>(running_mean[f]))*inv_std);
          y[idx]=static_cast<StorageT>(gamma[f]*normalized+beta[f]);
        }
      }
    }
    return output;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

#ifdef USE_CAIF_CUDA

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceBatchNorm<ComputeT,StorageT>::ForwardDevice(const CAIF_DeviceTensor &input,
                                                       CAIF_RunContext &ctx)
{
  try
  {
    const std::vector<uint32_t> &in_shape=input.Shape();
    const uint32_t feature_axis=in_shape.back();
    const size_t total_elems=input.TotalElements();
    const uint32_t row_count=static_cast<uint32_t>(total_elems/feature_axis);

    // Sync host fp32 params and running stats into device fp32 mirrors.
    CAIF_DeviceTensor gamma_dev;
    CAIF_DeviceTensor beta_dev;
    CAIF_DeviceTensor running_mean_dev;
    CAIF_DeviceTensor running_var_dev;
    CAIF_BatchNormCudnnHelpers::SyncFp32HostToDevice(Gamma(),Stream(),gamma_dev);
    CAIF_BatchNormCudnnHelpers::SyncFp32HostToDevice(Beta(),Stream(),beta_dev);
    CAIF_BatchNormCudnnHelpers::SyncFp32HostToDevice(RunningMeanMutable(),Stream(),running_mean_dev);
    CAIF_BatchNormCudnnHelpers::SyncFp32HostToDevice(RunningVarMutable(),Stream(),running_var_dev);

    // Allocate output and per-call save buffers (shape [features] fp32).
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(in_shape,
                                                              Stream(),
                                                              StorageDtype());
    const std::vector<uint32_t> param_shape={feature_axis};
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    CAIF_DeviceTensor save_mean=CAIF_DeviceTensor::Uninitialized(param_shape,Stream(),fp32);
    CAIF_DeviceTensor save_inv_var=CAIF_DeviceTensor::Uninitialized(param_shape,
                                                                    Stream(),
                                                                    fp32);

    cudnnHandle_t handle=CAIF_DeviceContext::Instance().CudnnHandle();
    CAIF_DeviceContext::Instance().SetCudnnStream(Stream().Handle());

    const cudnnDataType_t cudnn_dt=CAIF_CudnnUtil::CudnnDtypeFromStorage(StorageDtype());
    cudnnTensorDescriptor_t data_desc=nullptr;
    cudnnTensorDescriptor_t param_desc=nullptr;
    CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&data_desc),"create data_desc");
    CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&param_desc),"create param_desc");
    CAIF_BatchNormCudnnHelpers::SetupBnDataDescriptor(data_desc,row_count,feature_axis,cudnn_dt);
    CAIF_BatchNormCudnnHelpers::SetupBnParamDescriptor(param_desc,feature_axis);

    const float alpha=1.0f;
    const float beta=0.0f;
    const double mom=static_cast<double>(MomentumInternal());
    const double eps=static_cast<double>(EpsilonInternal());
    if(ctx.Training()==true)
    {
      CAIF_CudnnUtil::CheckCudnn(cudnnBatchNormalizationForwardTraining(handle,
                                                                        CUDNN_BATCHNORM_SPATIAL,
                                                                        &alpha,
                                                                        &beta,
                                                                        data_desc,
                                                                        input.DeviceDataRaw(),
                                                                        data_desc,
                                                                        output.DeviceDataRaw(),
                                                                        param_desc,
                                                                        gamma_dev.DeviceDataRaw(),
                                                                        beta_dev.DeviceDataRaw(),
                                                                        mom,
                                                                        running_mean_dev.DeviceDataRaw(),
                                                                        running_var_dev.DeviceDataRaw(),
                                                                        eps,
                                                                        save_mean.DeviceDataRaw(),
                                                                        save_inv_var.DeviceDataRaw()),
                                 "BNForwardTraining");
      // cuDNN updated running stats — copy back to host.
      CAIF_BatchNormCudnnHelpers::SyncFp32DeviceToHostOverwrite(running_mean_dev,Stream(),RunningMeanMutable());
      CAIF_BatchNormCudnnHelpers::SyncFp32DeviceToHostOverwrite(running_var_dev,Stream(),RunningVarMutable());
      SetCachedInputShape(in_shape);
      SetCachedInputDevice(input.Clone());
      SetCachedSaveMeanDevice(std::move(save_mean));
      SetCachedSaveInvVarDevice(std::move(save_inv_var));
    }
    else
    {
      CAIF_CudnnUtil::CheckCudnn(cudnnBatchNormalizationForwardInference(handle,
                                                                         CUDNN_BATCHNORM_SPATIAL,
                                                                         &alpha,
                                                                         &beta,
                                                                         data_desc,
                                                                         input.DeviceDataRaw(),
                                                                         data_desc,
                                                                         output.DeviceDataRaw(),
                                                                         param_desc,
                                                                         gamma_dev.DeviceDataRaw(),
                                                                         beta_dev.DeviceDataRaw(),
                                                                         running_mean_dev.DeviceDataRaw(),
                                                                         running_var_dev.DeviceDataRaw(),
                                                                         eps),
                                 "BNForwardInference");
    }

    cudnnDestroyTensorDescriptor(param_desc);
    cudnnDestroyTensorDescriptor(data_desc);
    return output;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

#endif // USE_CAIF_CUDA

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceBatchNorm<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                      CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    AssertInputDtype(grad_output);
    if(CachedInputShape().empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceBatchNorm: backward called before forward");
    }
    if(grad_output.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      return BackwardHost(grad_output);
    }
#ifdef USE_CAIF_CUDA
    return BackwardDevice(grad_output);
#else
    THROW_CAIFE("CAIF_DeviceBatchNorm: device path requires USE_CAIF_CUDA");
#endif
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceBatchNorm<ComputeT,StorageT>::BackwardHost(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    const std::vector<uint32_t> &cached_shape=CachedInputShape();
    const uint32_t feature_axis=cached_shape.back();
    const size_t total_elems=grad_output.TotalElements();
    const size_t row_count=total_elems/feature_axis;
    const StorageT *g_out=StoragePtr(grad_output);
    // fp32 by contract
    const float *gamma=static_cast<const float *>(Gamma().DeviceDataRaw());
    // fp32 by contract
    float *grad_gamma=static_cast<float *>(GammaGrad().DeviceDataRaw());
    // fp32 by contract
    float *grad_beta=static_cast<float *>(BetaGrad().DeviceDataRaw());

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::ZerosHost(cached_shape,StorageDtype());
    StorageT *g_in=StoragePtr(grad_input);
    const float m_inv=1.0f/static_cast<float>(row_count);
    const std::vector<float> &cached_norm=CachedNormalized();
    const std::vector<float> &cached_inv_std=CachedInvStd();

    for(size_t f=0;f<feature_axis;++f)
    {
      double sum_dy=0.0;
      double sum_dy_xhat=0.0;
      for(size_t r=0;r<row_count;++r)
      {
        const size_t idx=r*feature_axis+f;
        const float dy=static_cast<float>(g_out[idx]);
        sum_dy+=static_cast<double>(dy);
        sum_dy_xhat+=static_cast<double>(dy)*static_cast<double>(cached_norm[idx]);
      }
      grad_beta[f]+=static_cast<float>(sum_dy);
      grad_gamma[f]+=static_cast<float>(sum_dy_xhat);
      const float inv_std=cached_inv_std[f];
      const float g=gamma[f];
      const float sum_dy_f=static_cast<float>(sum_dy);
      const float sum_dy_xhat_f=static_cast<float>(sum_dy_xhat);
      for(size_t r=0;r<row_count;++r)
      {
        const size_t idx=r*feature_axis+f;
        const float dy=static_cast<float>(g_out[idx]);
        const float xhat=cached_norm[idx];
        const float term=m_inv*inv_std*g*(static_cast<float>(row_count)*dy-
                                            sum_dy_f-xhat*sum_dy_xhat_f);
        g_in[idx]=static_cast<StorageT>(term);
      }
    }
    return grad_input;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

#ifdef USE_CAIF_CUDA

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceBatchNorm<ComputeT,StorageT>::BackwardDevice(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(CachedInputDevice().IsAllocated()==false)
    {
      THROW_CAIFE("CAIF_DeviceBatchNorm: device backward called before device forward");
    }
    const std::vector<uint32_t> &cached_shape=CachedInputShape();
    const uint32_t feature_axis=cached_shape.back();
    const size_t total_elems=grad_output.TotalElements();
    const uint32_t row_count=static_cast<uint32_t>(total_elems/feature_axis);

    // Sync host gamma into device fp32 mirror; allocate device grads too.
    CAIF_DeviceTensor gamma_dev;
    CAIF_BatchNormCudnnHelpers::SyncFp32HostToDevice(Gamma(),Stream(),gamma_dev);
    const std::vector<uint32_t> param_shape={feature_axis};
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;
    CAIF_DeviceTensor gamma_grad_dev=CAIF_DeviceTensor::Uninitialized(param_shape,
                                                                      Stream(),
                                                                      fp32);
    CAIF_DeviceTensor beta_grad_dev=CAIF_DeviceTensor::Uninitialized(param_shape,
                                                                     Stream(),
                                                                     fp32);

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized(cached_shape,
                                                                  Stream(),
                                                                  StorageDtype());

    cudnnHandle_t handle=CAIF_DeviceContext::Instance().CudnnHandle();
    CAIF_DeviceContext::Instance().SetCudnnStream(Stream().Handle());

    const cudnnDataType_t cudnn_dt=CAIF_CudnnUtil::CudnnDtypeFromStorage(StorageDtype());
    cudnnTensorDescriptor_t data_desc=nullptr;
    cudnnTensorDescriptor_t param_desc=nullptr;
    CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&data_desc),"create data_desc");
    CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&param_desc),"create param_desc");
    CAIF_BatchNormCudnnHelpers::SetupBnDataDescriptor(data_desc,row_count,feature_axis,cudnn_dt);
    CAIF_BatchNormCudnnHelpers::SetupBnParamDescriptor(param_desc,feature_axis);

    const float alpha_data=1.0f;
    const float beta_data=0.0f;
    const float alpha_param=1.0f;
    const float beta_param=0.0f;
    const double eps=static_cast<double>(EpsilonInternal());
    CAIF_CudnnUtil::CheckCudnn(cudnnBatchNormalizationBackward(handle,
                                                               CUDNN_BATCHNORM_SPATIAL,
                                                               &alpha_data,
                                                               &beta_data,
                                                               &alpha_param,
                                                               &beta_param,
                                                               data_desc,
                                                               CachedInputDevice().DeviceDataRaw(),
                                                               data_desc,
                                                               grad_output.DeviceDataRaw(),
                                                               data_desc,
                                                               grad_input.DeviceDataRaw(),
                                                               param_desc,
                                                               gamma_dev.DeviceDataRaw(),
                                                               gamma_grad_dev.DeviceDataRaw(),
                                                               beta_grad_dev.DeviceDataRaw(),
                                                               eps,
                                                               CachedSaveMeanDevice().DeviceDataRaw(),
                                                               CachedSaveInvVarDevice().DeviceDataRaw()),
                               "BNBackward");

    cudnnDestroyTensorDescriptor(param_desc);
    cudnnDestroyTensorDescriptor(data_desc);

    // Accumulate device fp32 grads back into host fp32 grad accumulators.
    CAIF_BatchNormCudnnHelpers::AccumulateFp32DeviceToHost(gamma_grad_dev,Stream(),GammaGrad());
    CAIF_BatchNormCudnnHelpers::AccumulateFp32DeviceToHost(beta_grad_dev,Stream(),BetaGrad());
    return grad_input;
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

#endif // USE_CAIF_CUDA

template<typename ComputeT,typename StorageT>
void CAIF_DeviceBatchNorm<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    const uint32_t n=NumFeaturesInternal();
    // fp32 by contract
    float *gg=static_cast<float *>(GammaGrad().DeviceDataRaw());
    // fp32 by contract
    float *bg=static_cast<float *>(BetaGrad().DeviceDataRaw());
    std::memset(gg,0,n*sizeof(float));
    std::memset(bg,0,n*sizeof(float));
  }
  CAIF_CATCH_BLOCK();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceBatchNorm<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return Gamma();
    }
    if(index==1)
    {
      return Beta();
    }
    THROW_CAIFE("CAIF_DeviceBatchNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK();
  return Gamma();
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceBatchNorm<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return Gamma();
    }
    if(index==1)
    {
      return Beta();
    }
    THROW_CAIFE("CAIF_DeviceBatchNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK();
  return Gamma();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceBatchNorm<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return GammaGrad();
    }
    if(index==1)
    {
      return BetaGrad();
    }
    THROW_CAIFE("CAIF_DeviceBatchNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK();
  return GammaGrad();
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceBatchNorm<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return GammaGrad();
    }
    if(index==1)
    {
      return BetaGrad();
    }
    THROW_CAIFE("CAIF_DeviceBatchNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK();
  return GammaGrad();
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceBatchNorm<ComputeT,StorageT>::TotalParameterCount()const
{
  return static_cast<size_t>(NumFeaturesInternal())*2u;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceBatchNorm<ComputeT,StorageT>::Description()const
{
  return g_serial_tag_batch_norm;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceBatchNorm<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
  std::vector<std::string> names;
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::BNGamma_e));
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::BNBeta_e));
  return names;
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid. Each cell
// runs its declared storage in DRAM via the cuDNN device path; the host
// fp32 path remains for Host_e tensors regardless of StorageT.
template class CAIF_DeviceBatchNorm<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceBatchNorm<float,__half>;
template class CAIF_DeviceBatchNorm<float,__nv_bfloat16>;
template class CAIF_DeviceBatchNorm<__half,float>;
template class CAIF_DeviceBatchNorm<__half,__half>;
template class CAIF_DeviceBatchNorm<__half,__nv_bfloat16>;
template class CAIF_DeviceBatchNorm<__nv_bfloat16,float>;
template class CAIF_DeviceBatchNorm<__nv_bfloat16,__half>;
template class CAIF_DeviceBatchNorm<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
