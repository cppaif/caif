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
#include "caif_cuda_kernels_normalization.cuh"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
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

    SetGamma(CAIF_DeviceTensor::Uninitialized({dim},stream));
    std::vector<float> ones(dim,1.0f);
    Gamma().CopyFromHost(ones.data(),dim);

    SetBeta(CAIF_DeviceTensor::Zeros({dim},stream));

    SetGammaGrad(CAIF_DeviceTensor::Zeros({dim},stream));
    SetBetaGrad(CAIF_DeviceTensor::Zeros({dim},stream));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLayerNorm<ComputeT,StorageT>::CAIF_DeviceLayerNorm(CAIF_DeviceLayerNorm &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _dim(other.Dim()),
                              _epsilon(other.Epsilon()),
                              _gamma(std::move(other.Gamma())),
                              _beta(std::move(other.Beta())),
                              _gamma_grad(std::move(other.GammaGrad())),
                              _beta_grad(std::move(other.BetaGrad())),
                              _cached_rows(other.CachedRows()),
                              _last_input(std::move(other.LastInput())),
                              _mean_cache(std::move(other.MeanCache())),
                              _rstd_cache(std::move(other.RstdCache()))
{
  other.SetCachedRows(0);
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
      SetDim(other.Dim());
      SetEpsilon(other.Epsilon());
      SetGamma(std::move(other.Gamma()));
      SetBeta(std::move(other.Beta()));
      SetGammaGrad(std::move(other.GammaGrad()));
      SetBetaGrad(std::move(other.BetaGrad()));
      SetCachedRows(other.CachedRows());
      other.SetCachedRows(0);
      SetLastInput(std::move(other.LastInput()));
      SetMeanCache(std::move(other.MeanCache()));
      SetRstdCache(std::move(other.RstdCache()));
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
    if(shape.back()!=Dim())
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::Forward: last dimension must match dim");
    }

    int rows=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      rows*=static_cast<int>(shape[i]);
    }

    if(rows!=CachedRows())
    {
      SetMeanCache(CAIF_DeviceTensor::Uninitialized({static_cast<uint32_t>(rows)},ctx.Stream()));
      SetRstdCache(CAIF_DeviceTensor::Uninitialized({static_cast<uint32_t>(rows)},ctx.Stream()));
      SetCachedRows(rows);
    }

    CAIF_DeviceTensor output=AllocateOutput(shape,ctx);

    launch_layernorm_forward<StorageT>(StoragePtr(input),
                                       Gamma().template DevicePtr<float>(),       // fp32: affine
                                       Beta().template DevicePtr<float>(),        // fp32: affine
                                       StoragePtr(output),
                                       MeanCache().template DevicePtr<float>(),  // fp32: reduction cache
                                       RstdCache().template DevicePtr<float>(),  // fp32: reduction cache
                                       Epsilon(),
                                       rows,
                                       static_cast<int>(Dim()),
                                       ctx.Stream().Handle());

    if(ctx.Training()==true)
    {
      SetLastInput(input.Clone());
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
    if(LastInput().IsEmpty()==true)
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::Backward: must call Forward with training=true first");
    }
    AssertInputDtype(grad_output);

    const std::vector<uint32_t> &shape=grad_output.Shape();
    if(shape.back()!=Dim())
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
    GammaGrad().FillZero();
    BetaGrad().FillZero();

    launch_layernorm_backward<StorageT>(StoragePtr(grad_output),
                                        StoragePtr(LastInput()),
                                        Gamma().template DevicePtr<float>(),        // fp32: affine
                                        MeanCache().template DevicePtr<float>(),   // fp32: reduction cache
                                        RstdCache().template DevicePtr<float>(),   // fp32: reduction cache
                                        StoragePtr(grad_input),
                                        GammaGrad().template DevicePtr<float>(),   // fp32: affine grad
                                        BetaGrad().template DevicePtr<float>(),    // fp32: affine grad
                                        rows,
                                        static_cast<int>(Dim()),
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
    GammaGrad().FillZero();
    BetaGrad().FillZero();
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
      return Gamma();
    }
    if(index==1)
    {
      return Beta();
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
      return Gamma();
    }
    if(index==1)
    {
      return Beta();
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
      return GammaGrad();
    }
    if(index==1)
    {
      return BetaGrad();
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
      return GammaGrad();
    }
    if(index==1)
    {
      return BetaGrad();
    }
    THROW_CAIFE("CAIF_DeviceLayerNorm::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLayerNorm<ComputeT,StorageT>::TotalParameterCount()const
{
  return static_cast<size_t>(Dim())*2;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceLayerNorm<ComputeT,StorageT>::Description()const
{
  try
  {
    return g_serial_tag_layernorm+
           g_serial_open_paren+
           std::to_string(Dim())+
           g_serial_close_paren;
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
    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::LayerNormGamma_e));
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::LayerNormBeta_e));
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
    if(shape.size()!=1||shape[0]!=Dim())
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::LoadGamma: shape mismatch, expected [dim]");
    }
    SetGamma(std::move(gamma));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLayerNorm<ComputeT,StorageT>::LoadBeta(CAIF_DeviceTensor &&beta)
{
  try
  {
    const std::vector<uint32_t> &shape=beta.Shape();
    if(shape.size()!=1||shape[0]!=Dim())
    {
      THROW_CAIFE("CAIF_DeviceLayerNorm::LoadBeta: shape mismatch, expected [dim]");
    }
    SetBeta(std::move(beta));
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
