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

// `_gamma`, `_beta`, `_gamma_grad`, `_beta_grad` are fp32 affine params
// (the standard LayerNorm convention). `_mean_cache`, `_rstd_cache` are
// fp32 reduction caches (mean/inv-std must accumulate in fp32 even when
// activations are fp16/bf16). Per-site `DevicePtr<float>()` reads name
// this contract inline.

#include "caif_device_layernorm.h"
#include "caif_device_layernorm_factory.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceLayerNorm<ComputeT,StorageT>::CAIF_DeviceLayerNorm(uint32_t dim,
                                                              CAIF_CudaStream &stream,
                                                              float epsilon):
                                            CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                            _dim(dim),
                                            _epsilon(epsilon),
                                            _gamma(),
                                            _beta(),
                                            _gamma_grad(),
                                            _beta_grad(),
                                            _cached_rows(0),
                                            _last_input(),
                                            _mean_cache(),
                                            _rstd_cache()
{
  try
  {
    if(dim==0)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm: dim must be > 0");
    }

    _gamma=CAIF_DeviceTensor::Uninitialized({dim},stream);
    std::vector<float> ones(dim,1.0f);
    _gamma.CopyFromHost(ones.data(),dim);

    _beta=CAIF_DeviceTensor::Zeros({dim},stream);

    _gamma_grad=CAIF_DeviceTensor::Zeros({dim},stream);
    _beta_grad=CAIF_DeviceTensor::Zeros({dim},stream);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLayerNorm<ComputeT,StorageT>::CAIF_DeviceLayerNorm(CAIF_DeviceLayerNorm &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _dim(other._dim),
                              _epsilon(other._epsilon),
                              _gamma(std::move(other._gamma)),
                              _beta(std::move(other._beta)),
                              _gamma_grad(std::move(other._gamma_grad)),
                              _beta_grad(std::move(other._beta_grad)),
                              _cached_rows(other._cached_rows),
                              _last_input(std::move(other._last_input)),
                              _mean_cache(std::move(other._mean_cache)),
                              _rstd_cache(std::move(other._rstd_cache))
{
  other._cached_rows=0;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLayerNorm<ComputeT,StorageT> &
CAIF_DeviceLayerNorm<ComputeT,StorageT>::operator=(CAIF_DeviceLayerNorm &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      _dim=other._dim;
      _epsilon=other._epsilon;
      _gamma=std::move(other._gamma);
      _beta=std::move(other._beta);
      _gamma_grad=std::move(other._gamma_grad);
      _beta_grad=std::move(other._beta_grad);
      _cached_rows=other._cached_rows;
      other._cached_rows=0;
      _last_input=std::move(other._last_input);
      _mean_cache=std::move(other._mean_cache);
      _rstd_cache=std::move(other._rstd_cache);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceLayerNorm<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                     CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);

    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::Forward: input tensor is empty");
    }
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::Forward: last dimension must match dim");
    }

    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    if(rows!=_cached_rows)
    {
      _mean_cache=CAIF_DeviceTensor::Uninitialized(
                    {static_cast<uint32_t>(rows)},ctx.Stream());
      _rstd_cache=CAIF_DeviceTensor::Uninitialized(
                    {static_cast<uint32_t>(rows)},ctx.Stream());
      _cached_rows=rows;
    }

    CAIF_DeviceTensor output=AllocateOutput(shape,ctx);

    launch_layernorm_forward<StorageT>(StoragePtr(input),
                                       _gamma.DevicePtr<float>(),       // fp32: affine
                                       _beta.DevicePtr<float>(),        // fp32: affine
                                       StoragePtr(output),
                                       _mean_cache.DevicePtr<float>(),  // fp32: reduction cache
                                       _rstd_cache.DevicePtr<float>(),  // fp32: reduction cache
                                       _epsilon,
                                       rows,
                                       static_cast<int>(_dim),
                                       ctx.Stream().Handle());

    if(ctx.Training()==true)
    {
      _last_input=input.Clone();
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceLayerNorm<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                      CAIF_RunContext &ctx)
{
  try
  {
    if(_last_input.IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::Backward: must call Forward with training=true first");
    }
    AssertInputDtype(grad_output);

    const std::vector<uint32_t> &shape=grad_output.Shape();
    if(shape.back()!=_dim)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::Backward: last dimension must match dim");
    }

    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    CAIF_DeviceTensor grad_input=AllocateOutput(shape,ctx);

    // Zero grad_gamma and grad_beta before atomicAdd accumulation. They
    // are fp32 so Fill(0.0f) is legal; FillZero() is dtype-agnostic and
    // equivalent.
    _gamma_grad.FillZero();
    _beta_grad.FillZero();

    launch_layernorm_backward<StorageT>(StoragePtr(grad_output),
                                        StoragePtr(_last_input),
                                        _gamma.DevicePtr<float>(),        // fp32: affine
                                        _mean_cache.DevicePtr<float>(),   // fp32: reduction cache
                                        _rstd_cache.DevicePtr<float>(),   // fp32: reduction cache
                                        StoragePtr(grad_input),
                                        _gamma_grad.DevicePtr<float>(),   // fp32: affine grad
                                        _beta_grad.DevicePtr<float>(),    // fp32: affine grad
                                        rows,
                                        static_cast<int>(_dim),
                                        ctx.Stream().Handle());

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLayerNorm<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    _gamma_grad.FillZero();
    _beta_grad.FillZero();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLayerNorm<ComputeT,StorageT>::ParameterTensorCount()const
{
  return 2;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceLayerNorm<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    if(index==1)
    {
      return _beta;
    }
    THROW_CAIFE("CAIF_DeviceLayerNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceLayerNorm<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma;
    }
    if(index==1)
    {
      return _beta;
    }
    THROW_CAIFE("CAIF_DeviceLayerNorm::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceLayerNorm<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    if(index==1)
    {
      return _beta_grad;
    }
    THROW_CAIFE("CAIF_DeviceLayerNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceLayerNorm<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _gamma_grad;
    }
    if(index==1)
    {
      return _beta_grad;
    }
    THROW_CAIFE("CAIF_DeviceLayerNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLayerNorm<ComputeT,StorageT>::TotalParameterCount()const
{
  return static_cast<size_t>(_dim)*2;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceLayerNorm<ComputeT,StorageT>::Description()const
{
  try
  {
    return "LayerNorm("+std::to_string(_dim)+")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceLayerNorm<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"weight");
    names.push_back(prefix+"bias");
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLayerNorm<ComputeT,StorageT>::LoadGamma(CAIF_DeviceTensor &&gamma)
{
  try
  {
    const std::vector<uint32_t> &shape=gamma.Shape();
    if(shape.size()!=1||shape[0]!=_dim)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::LoadGamma: shape mismatch, expected [dim]");
    }
    _gamma=std::move(gamma);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLayerNorm<ComputeT,StorageT>::LoadBeta(CAIF_DeviceTensor &&beta)
{
  try
  {
    const std::vector<uint32_t> &shape=beta.Shape();
    if(shape.size()!=1||shape[0]!=_dim)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::LoadBeta: shape mismatch, expected [dim]");
    }
    _beta=std::move(beta);
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid for
// shape uniformity with GEMM-bearing layers. ComputeT has no semantic
// effect on LayerNorm's kernel.
template class CAIF_DeviceLayerNorm<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceLayerNorm<float,__half>;
template class CAIF_DeviceLayerNorm<float,__nv_bfloat16>;
template class CAIF_DeviceLayerNorm<__half,float>;
template class CAIF_DeviceLayerNorm<__half,__half>;
template class CAIF_DeviceLayerNorm<__half,__nv_bfloat16>;
template class CAIF_DeviceLayerNorm<__nv_bfloat16,float>;
template class CAIF_DeviceLayerNorm<__nv_bfloat16,__half>;
template class CAIF_DeviceLayerNorm<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
