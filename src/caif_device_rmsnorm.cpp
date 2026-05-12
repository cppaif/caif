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

// `_gamma`, `_gamma_grad`, `rms_cache`, `_rms_cache` are fp32 by RMSNorm
// reference convention. Per-site `DevicePtr<float>()` reads name this
// contract inline, per the type-dispatch full plan (Phase 2).

#include "caif_device_rmsnorm.h"
#include "caif_device_rmsnorm_factory.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceRMSNorm<ComputeT,StorageT>::CAIF_DeviceRMSNorm(uint32_t dim,
                                                          CAIF_CudaStream &stream,
                                                          float epsilon):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _dim(dim),
                                          _epsilon(epsilon),
                                          _gamma(),
                                          _gamma_grad(),
                                          _last_input(),
                                          _rms_cache()
{
  try
  {
    if(dim==0)
    {
      THROW_CAIFE("CAIF_DeviceRMSNorm: dim must be > 0");
    }

    // gamma is always fp32: the kernel takes `const float *gamma` regardless
    // of T. Init to ones.
    _gamma=CAIF_DeviceTensor::Uninitialized({dim},stream);
    std::vector<float> ones(dim,1.0f);
    _gamma.CopyFromHost(ones.data(),dim);

    _gamma_grad=CAIF_DeviceTensor::Zeros({dim},stream);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceRMSNorm<ComputeT,StorageT>::CAIF_DeviceRMSNorm(CAIF_DeviceRMSNorm &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _dim(other._dim),
                              _epsilon(other._epsilon),
                              _gamma(std::move(other._gamma)),
                              _gamma_grad(std::move(other._gamma_grad)),
                              _last_input(std::move(other._last_input)),
                              _rms_cache(std::move(other._rms_cache))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceRMSNorm<ComputeT,StorageT> &
CAIF_DeviceRMSNorm<ComputeT,StorageT>::operator=(CAIF_DeviceRMSNorm &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      _dim=other._dim;
      _epsilon=other._epsilon;
      _gamma=std::move(other._gamma);
      _gamma_grad=std::move(other._gamma_grad);
      _last_input=std::move(other._last_input);
      _rms_cache=std::move(other._rms_cache);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceRMSNorm<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                   CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);

    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceRMSNorm::Forward: input tensor is empty");
    }
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("CAIF_DeviceRMSNorm::Forward: last dimension must match dim");
    }

    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    CAIF_DeviceTensor output=AllocateOutput(shape,ctx);
    CAIF_DeviceTensor rms_cache=CAIF_DeviceTensor::Uninitialized(
                                  {static_cast<uint32_t>(rows)},ctx.Stream());

    launch_rmsnorm_forward<StorageT>(StoragePtr(input),
                                     _gamma.DevicePtr<float>(),       // fp32: scale param
                                     StoragePtr(output),
                                     rms_cache.DevicePtr<float>(),    // fp32: reduction cache
                                     _epsilon,
                                     rows,
                                     static_cast<int>(_dim),
                                     ctx.Stream().Handle());

    if(ctx.Training()==true)
    {
      _last_input=input.Clone();
      _rms_cache=std::move(rms_cache);
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceRMSNorm<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    if(_last_input.IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceRMSNorm::Backward: must call Forward with training=true first");
    }
    AssertInputDtype(grad_output);

    const std::vector<uint32_t> &shape=grad_output.Shape();
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("CAIF_DeviceRMSNorm::Backward: last dimension must match dim");
    }

    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    CAIF_DeviceTensor grad_input=AllocateOutput(shape,ctx);

    // Zero grad_gamma before atomicAdd accumulation. _gamma_grad is fp32
    // so Fill(0.0f) is safe; FillZero() is dtype-agnostic and equivalent.
    _gamma_grad.FillZero();

    launch_rmsnorm_backward<StorageT>(StoragePtr(grad_output),
                                      StoragePtr(_last_input),
                                      _gamma.DevicePtr<float>(),       // fp32: scale param
                                      _rms_cache.DevicePtr<float>(),   // fp32: reduction cache
                                      StoragePtr(grad_input),
                                      _gamma_grad.DevicePtr<float>(),  // fp32: scale-param grad
                                      _epsilon,
                                      rows,
                                      static_cast<int>(_dim),
                                      ctx.Stream().Handle());

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceRMSNorm<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    _gamma_grad.FillZero();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceRMSNorm<ComputeT,StorageT>::ParameterTensorCount()const
{
  return 1;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceRMSNorm<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    THROW_CAIFE("CAIF_DeviceRMSNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceRMSNorm<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    THROW_CAIFE("CAIF_DeviceRMSNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceRMSNorm<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    THROW_CAIFE("CAIF_DeviceRMSNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceRMSNorm<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    THROW_CAIFE("CAIF_DeviceRMSNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceRMSNorm<ComputeT,StorageT>::TotalParameterCount()const
{
  return static_cast<size_t>(_dim);
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceRMSNorm<ComputeT,StorageT>::Description()const
{
  try
  {
    return "RMSNorm("+std::to_string(_dim)+")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceRMSNorm<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"weight");
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceRMSNorm<ComputeT,StorageT>::LoadGamma(CAIF_DeviceTensor &&gamma)
{
  try
  {
    const std::vector<uint32_t> &shape=gamma.Shape();
    if(shape.size()!=1||shape[0]!=_dim)
    {
      THROW_CAIFE("CAIF_DeviceRMSNorm::LoadGamma: shape mismatch, expected [dim]");
    }
    _gamma=std::move(gamma);
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid for shape
// uniformity with GEMM-bearing layers. ComputeT has no semantic meaning
// inside RMSNorm (no MatMul) but the cells exist so RMSNorm has the same
// surface as MHA / FFN / etc.
template class CAIF_DeviceRMSNorm<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceRMSNorm<float,__half>;
template class CAIF_DeviceRMSNorm<float,__nv_bfloat16>;
template class CAIF_DeviceRMSNorm<__half,float>;
template class CAIF_DeviceRMSNorm<__half,__half>;
template class CAIF_DeviceRMSNorm<__half,__nv_bfloat16>;
template class CAIF_DeviceRMSNorm<__nv_bfloat16,float>;
template class CAIF_DeviceRMSNorm<__nv_bfloat16,__half>;
template class CAIF_DeviceRMSNorm<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
