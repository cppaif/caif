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

#include "caif_device_frozen_linear.h"
#include "caif_ops.h"
#include "caif_device_context.h"
#include "caif_exception.h"
#include "caif_cuda_kernels.h"
#ifdef USE_CAIF_CUDA
#include "cublas_v2.h"
#include "cuda_runtime_api.h"
#include "library_types.h"
#endif

// Per-site dispatch convention (TYPE_DISPATCH_FULL_PLAN Phase 8.5.A):
//   - `_weight.DeviceDataRaw()` and `_scales.DeviceDataRaw()` reads are
//     legitimate non-fp32 reinterprets — `_weight` is INT4-packed bytes
//     (UInt8 storage) or INT8 (Int8 storage) and `_scales` carries fp16
//     scales (INT4 path) or fp32 scales (INT8 per-tensor / per-channel
//     paths). The matching launchers (`launch_dequantize_int4`,
//     `launch_dequantize_int8_per_tensor`, `launch_dequantize_int8_per_channel`)
//     all carry `fp32-only by contract` markers per Phase 5.6 and take
//     `void *` for the packed/scale buffers.
//   - `fp32.template DevicePtr<float>()` reads are the fp32 dequantize
//     output (the dequantize launcher's fp32 master OUT contract).
//   - `compute_weight.DeviceDataRaw()` / `input_reduced.DeviceDataRaw()` /
//     `output_2d.DeviceDataRaw()` (and their backward analogues) feed
//     `cublasGemmEx` which takes `void *` plus an explicit
//     `cudaDataType` argument; the dtype is provided by the
//     `compute_type` / `input_cuda_type` switch directly above each
//     call, so the void-pointer reads are dtype-justified per-site.

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::CAIF_DeviceFrozenLinear(
  uint32_t input_dim,
  uint32_t output_dim,
  CAIF_CudaStream &stream,
  uint32_t group_size,
  bool cache_fp32,
  CAIF_Ops::QuantScheme_e int8_scheme):Base_t(stream),
                                       _input_dim(input_dim),
                                       _output_dim(output_dim),
                                       _int8_scheme(int8_scheme),
                                       _group_size(group_size),
                                       _weight(),
                                       _scales(),
                                       _cache_fp32(cache_fp32),
                                       _cached_compute_weight(),
                                       _cached_input(),
                                       _offload_policy(
                                         CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::GpuResident_e),
                                       _host_weight(),
                                       _is_prefetched(false)
{
  try
  {
    if(input_dim==0||output_dim==0)
    {
      THROW_CAIFE("FrozenLinear: input_dim and output_dim must be > 0");
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::CAIF_DeviceFrozenLinear(
  CAIF_DeviceFrozenLinear &&other):Base_t(std::move(other)),
                                   _input_dim(other._input_dim),
                                   _output_dim(other._output_dim),
                                   _int8_scheme(other._int8_scheme),
                                   _group_size(other._group_size),
                                   _weight(std::move(other._weight)),
                                   _scales(std::move(other._scales)),
                                   _cache_fp32(other._cache_fp32),
                                   _cached_compute_weight(std::move(other._cached_compute_weight)),
                                   _cached_input(std::move(other._cached_input)),
                                   _offload_policy(other._offload_policy),
                                   _host_weight(std::move(other._host_weight)),
                                   _is_prefetched(other._is_prefetched)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceFrozenLinear<ComputeT,StorageT> &
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::operator=(CAIF_DeviceFrozenLinear &&other)
{
  try
  {
    if(this!=&other)
    {
      Base_t::operator=(std::move(other));
      _input_dim=other._input_dim;
      _output_dim=other._output_dim;
      _int8_scheme=other._int8_scheme;
      _group_size=other._group_size;
      _weight=std::move(other._weight);
      _scales=std::move(other._scales);
      _cache_fp32=other._cache_fp32;
      _cached_compute_weight=std::move(other._cached_compute_weight);
      _cached_input=std::move(other._cached_input);
      _offload_policy=other._offload_policy;
      _host_weight=std::move(other._host_weight);
      _is_prefetched=other._is_prefetched;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::LoadFromTensor(CAIF_DeviceTensor &&weight)
{
  try
  {
    SetWeight(std::move(weight));
    if(OffloadPolicy()==CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e)
    {
      MigrateWeightToHost();
    }
    else
    {
      SetPrefetched(true);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::SetOffloadPolicy(
  const CAIF_OffloadPolicy::CAIF_OffloadPolicy_e p)
{
  try
  {
    if(OffloadPolicy()==p)
    {
      return;
    }
    if(p==CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e)
    {
      _offload_policy=p;
      if(HasWeight()==true)
      {
        MigrateWeightToHost();
      }
      return;
    }
    if(p==CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::GpuResident_e)
    {
      if(HasHostWeight()==true && HasWeight()==false)
      {
        SetWeight(HostWeight().PrefetchToDevice(Stream()));
        Stream().Synchronize();
      }
      _host_weight.reset();
      SetPrefetched(true);
      _offload_policy=p;
      return;
    }
    THROW_CAIFE("FrozenLinear::SetOffloadPolicy: unsupported policy");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::MigrateWeightToHost()
{
  try
  {
    if(HasWeight()==false)
    {
      SetPrefetched(false);
      return;
    }
    _host_weight.reset(new CAIF_HostPinnedTensor(Weight().Shape(),Weight().Dtype()));
    HostWeight().CopyFromDevice(Weight());
    ClearWeight();
    ClearFP32Cache();
    SetPrefetched(false);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::Prefetch(CAIF_CudaStream &stream)
{
  try
  {
    if(OffloadPolicy()!=CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e)
    {
      return;
    }
    if(IsPrefetched()==true)
    {
      return;
    }
    if(HasHostWeight()==false)
    {
      THROW_CAIFE("FrozenLinear::Prefetch: HostPinned policy but no host weight");
    }
    SetWeight(HostWeight().PrefetchToDevice(stream));
    SetPrefetched(true);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::Evict()
{
  if(OffloadPolicy()!=CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e)
  {
    return;
  }
  ClearWeight();
  ClearFP32Cache();
  SetPrefetched(false);
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::LoadScalesFromHost(const void *data,
                                                                   size_t num_bytes)
{
  try
  {
    constexpr CAIF_DataType::CAIF_DataType_e sd=Base_t::StorageDtype();
    if(sd==CAIF_DataType::CAIF_DataType_e::Int4)
    {
      const size_t total_elements=static_cast<size_t>(_input_dim)*_output_dim;
      const uint32_t num_groups=
        static_cast<uint32_t>((total_elements+_group_size-1)/_group_size);
      _scales=CAIF_DeviceTensor::Zeros({num_groups},
                                       Stream(),
                                       CAIF_DataType::CAIF_DataType_e::Float16);
      _scales.CopyFromHostRaw(data,num_bytes);
      return;
    }
    if(sd==CAIF_DataType::CAIF_DataType_e::Int8)
    {
      if(_int8_scheme==CAIF_Ops::QuantScheme_e::PerTensor_e)
      {
        _scales=CAIF_DeviceTensor::Zeros({1u},
                                         Stream(),
                                         CAIF_DataType::CAIF_DataType_e::Float32);
      }
      else
      {
        _scales=CAIF_DeviceTensor::Zeros({_output_dim},
                                         Stream(),
                                         CAIF_DataType::CAIF_DataType_e::Float32);
      }
      _scales.CopyFromHostRaw(data,num_bytes);
      return;
    }
    THROW_CAIFE("FrozenLinear::LoadScalesFromHost: storage dtype does not use scales");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
bool CAIF_DeviceFrozenLinear<ComputeT,StorageT>::NeedsScales()const
{
  constexpr CAIF_DataType::CAIF_DataType_e sd=Base_t::StorageDtype();
  return sd==CAIF_DataType::CAIF_DataType_e::Int4
         ||sd==CAIF_DataType::CAIF_DataType_e::Int8;
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ClearFP32Cache()
{
  ClearCachedComputeWeight();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ConvertToComputeDtype()const
{
  try
  {
    if(_weight.IsEmpty()==true)
    {
      THROW_CAIFE("FrozenLinear: weight not loaded");
    }

    constexpr CAIF_DataType::CAIF_DataType_e target=Base_t::ComputeDtype();
    constexpr CAIF_DataType::CAIF_DataType_e sd=Base_t::StorageDtype();

    if(sd==target)
    {
      // Same-dtype path: no conversion needed and no copy. Return a
      // non-owning view of `_weight` reshaped to [output_dim, input_dim]
      // (the shape `MatMulTransposeB` expects). Cloning here was a real
      // ~17 MB-per-expert GPU allocation per Forward call; for a
      // 27-layer DSv2-Lite with 64 frozen experts × 3 sublayers per
      // MoE block that's ~3 GB of redundant GPU per layer per step,
      // and the offload-prefetch path can't keep up with the wasted
      // headroom. WrapView shares `_weight`'s pointer; the view stays
      // valid for the duration of the Forward call (the caller doesn't
      // mutate _weight during forward).
      CAIF_DeviceTensor view=CAIF_DeviceTensor::WrapView(const_cast<void*>(_weight.DeviceDataRaw()),
                                                          {_output_dim,_input_dim},
                                                          Stream(),
                                                          _weight.Dtype());
      return view;
    }
    if(sd==CAIF_DataType::CAIF_DataType_e::Int4)
    {
#ifdef USE_CAIF_CUDA
      const size_t total_elements=static_cast<size_t>(_input_dim)*_output_dim;
      CAIF_DeviceTensor fp32=CAIF_DeviceTensor::Uninitialized({_output_dim,_input_dim},Stream());
      launch_dequantize_int4(_weight.DeviceDataRaw(),
                             _scales.DeviceDataRaw(),
                             fp32.template DevicePtr<float>(),
                             static_cast<int>(total_elements),
                             static_cast<int>(_group_size),
                             Stream().Handle());
      if(target==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        return fp32;
      }
      return fp32.To(target);
#else
      THROW_CAIFE("INT4 dequantization requires CUDA");
#endif
    }

    if(sd==CAIF_DataType::CAIF_DataType_e::Int8
       &&_scales.IsEmpty()==false)
    {
#ifdef USE_CAIF_CUDA
      const size_t total_elements=static_cast<size_t>(_input_dim)*_output_dim;
      CAIF_DeviceTensor fp32=CAIF_DeviceTensor::Uninitialized({_output_dim,_input_dim},Stream());
      if(_int8_scheme==CAIF_Ops::QuantScheme_e::PerTensor_e)
      {
        launch_dequantize_int8_per_tensor(_weight.DeviceDataRaw(),
                                          fp32.template DevicePtr<float>(),
                                          _scales.DeviceDataRaw(),
                                          static_cast<int>(total_elements),
                                          Stream().Handle());
      }
      else
      {
        launch_dequantize_int8_per_channel(_weight.DeviceDataRaw(),
                                           fp32.template DevicePtr<float>(),
                                           _scales.DeviceDataRaw(),
                                           static_cast<int>(_input_dim),
                                           static_cast<int>(_output_dim),
                                           Stream().Handle());
      }
      if(target==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        return fp32;
      }
      return fp32.To(target);
#else
      THROW_CAIFE("INT8 dequantization requires CUDA");
#endif
    }

    CAIF_DeviceTensor result=_weight.To(target);
    result.Reshape({_output_dim,_input_dim});
    return result;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                        CAIF_RunContext &ctx)
{
  try
  {
    if(HasStream()==false)
    {
      THROW_CAIFE("FrozenLinear: stream is null");
    }
    if(OffloadPolicy()==CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e &&
       IsPrefetched()==false)
    {
      THROW_CAIFE("FrozenLinear::ForwardImpl: HostPinned policy but layer is not prefetched."
                  " The block-level scheduler must call Prefetch(stream) before forward.");
    }
    if(_weight.IsEmpty()==true)
    {
      THROW_CAIFE("FrozenLinear: weight not loaded");
    }

    const auto &shape=input.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("FrozenLinear::Forward: input must be at least 2D");
    }
    if(shape.back()!=_input_dim)
    {
      THROW_CAIFE("FrozenLinear::Forward: last dim must match input_dim");
    }

    CAIF_DeviceTensor local_weight;
    const CAIF_DeviceTensor *weight_ptr=nullptr;
    if(CacheFP32()==true)
    {
      if(HasCachedComputeWeight()==false)
      {
        SetCachedComputeWeight(ConvertToComputeDtype());
      }
      weight_ptr=&CachedComputeWeight();
    }
    else
    {
      local_weight=ConvertToComputeDtype();
      weight_ptr=&local_weight;
    }
    const CAIF_DeviceTensor &compute_weight=*weight_ptr;

    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }
    CAIF_DeviceTensor input_2d=input.Clone();
    input_2d.Reshape({n,_input_dim});

    if(ctx.Training()==true)
    {
      _cached_input=input_2d.Clone();
    }

    CAIF_DeviceTensor output_2d=CAIF_DeviceTensor::Uninitialized({n,_output_dim},Stream());

    constexpr CAIF_DataType::CAIF_DataType_e cd=Base_t::ComputeDtype();
    if(cd==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      // Mixed cell <ComputeT=float, StorageT=fp16/bf16>: input arrives at
      // StorageT but compute_weight was just upcast to ComputeT=fp32 by
      // ConvertToComputeDtype(). MatMulTransposeB requires all 3 tensors
      // at matching dtype, so promote input to fp32 here. Same-cell
      // <float,float> takes the no-op branch and skips the .To() cost.
      CAIF_DeviceTensor input_compute;
      if(input_2d.Dtype()==cd)
      {
        CAIF_Ops::MatMulTransposeB(input_2d,compute_weight,output_2d,ctx);
      }
      else
      {
        input_compute=input_2d.To(cd);
        CAIF_Ops::MatMulTransposeB(input_compute,compute_weight,output_2d,ctx);
      }
    }
    else
    {
#ifdef USE_CAIF_CUDA
      CAIF_DeviceTensor input_reduced=input_2d.To(cd);

      CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
      device_ctx.SetCublasStream(Stream().Handle());

      const int m=static_cast<int>(n);
      const int nn=static_cast<int>(_output_dim);
      const int k=static_cast<int>(_input_dim);

      cudaDataType input_cuda_type=CUDA_R_16BF;
      if(cd==CAIF_DataType::CAIF_DataType_e::Float16)
      {
        input_cuda_type=CUDA_R_16F;
      }

      const float alpha=1.0f;
      const float beta=0.0f;

      const cublasComputeType_t compute_type=
        static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(cd));

      cublasStatus_t status=cublasGemmEx(device_ctx.CublasHandle(),
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         nn,
                                         m,
                                         k,
                                         &alpha,
                                         compute_weight.DeviceDataRaw(),
                                         input_cuda_type,
                                         k,
                                         input_reduced.DeviceDataRaw(),
                                         input_cuda_type,
                                         k,
                                         &beta,
                                         output_2d.DeviceDataRaw(),
                                         CUDA_R_32F,
                                         nn,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT);
      if(status!=CUBLAS_STATUS_SUCCESS)
      {
        THROW_CAIFE("FrozenLinear: cublasGemmEx forward failed");
      }
#else
      THROW_CAIFE("FrozenLinear: reduced precision requires CUDA");
#endif
    }

    if(shape.size()>2)
    {
      std::vector<uint32_t> out_shape(shape.begin(),shape.end()-1);
      out_shape.push_back(_output_dim);
      output_2d.Reshape(out_shape);
    }
    return output_2d;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                         CAIF_RunContext &ctx)
{
  try
  {
    if(HasStream()==false)
    {
      THROW_CAIFE("FrozenLinear: stream is null");
    }
    if(OffloadPolicy()==CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e &&
       IsPrefetched()==false)
    {
      THROW_CAIFE("FrozenLinear::BackwardImpl: HostPinned policy but layer is not prefetched."
                  " The block-level scheduler must call Prefetch(stream) before backward.");
    }
    if(_cached_input.IsEmpty()==true)
    {
      THROW_CAIFE("FrozenLinear::Backward: must call Forward with training=true first");
    }
    CAIF_DeviceTensor local_weight;
    const CAIF_DeviceTensor *weight_ptr=nullptr;
    if(CacheFP32()==true)
    {
      // Lazy (re)populate. With offload, eviction during forward
      // clears both the GPU `_weight` and the cached compute view;
      // backward's OnEnterBackwardStage re-prefetches `_weight` but
      // doesn't re-fill the compute cache, so we rebuild it here on
      // demand. For the sd==target cell ConvertToComputeDtype returns
      // a non-owning WrapView, so the rebuild costs nothing on GPU.
      if(HasCachedComputeWeight()==false)
      {
        SetCachedComputeWeight(ConvertToComputeDtype());
      }
      weight_ptr=&CachedComputeWeight();
    }
    else
    {
      local_weight=ConvertToComputeDtype();
      weight_ptr=&local_weight;
    }
    const CAIF_DeviceTensor &compute_weight=*weight_ptr;

    const auto &shape=grad_output.Shape();

    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }
    CAIF_DeviceTensor grad_2d=grad_output.Clone();
    grad_2d.Reshape({n,_output_dim});

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({n,_input_dim},Stream());

    constexpr CAIF_DataType::CAIF_DataType_e cd=Base_t::ComputeDtype();
    if(cd==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      // Mixed cell <ComputeT=float, StorageT=fp16/bf16>: grad_2d arrives
      // at StorageT, compute_weight is fp32 after ConvertToComputeDtype().
      // Promote grad to fp32 to satisfy MatMul's matching-dtype gate.
      CAIF_DeviceTensor grad_compute;
      if(grad_2d.Dtype()==cd)
      {
        CAIF_Ops::MatMul(grad_2d,compute_weight,grad_input,ctx);
      }
      else
      {
        grad_compute=grad_2d.To(cd);
        CAIF_Ops::MatMul(grad_compute,compute_weight,grad_input,ctx);
      }
    }
    else
    {
#ifdef USE_CAIF_CUDA
      CAIF_DeviceTensor grad_reduced=grad_2d.To(cd);

      CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
      device_ctx.SetCublasStream(Stream().Handle());

      const int m=static_cast<int>(n);
      const int nn=static_cast<int>(_input_dim);
      const int k=static_cast<int>(_output_dim);

      cudaDataType input_cuda_type=CUDA_R_16BF;
      if(cd==CAIF_DataType::CAIF_DataType_e::Float16)
      {
        input_cuda_type=CUDA_R_16F;
      }

      const float alpha=1.0f;
      const float beta=0.0f;

      const cublasComputeType_t compute_type=
        static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(cd));

      cublasStatus_t status=cublasGemmEx(device_ctx.CublasHandle(),
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         nn,
                                         m,
                                         k,
                                         &alpha,
                                         compute_weight.DeviceDataRaw(),
                                         input_cuda_type,
                                         nn,
                                         grad_reduced.DeviceDataRaw(),
                                         input_cuda_type,
                                         k,
                                         &beta,
                                         grad_input.DeviceDataRaw(),
                                         CUDA_R_32F,
                                         nn,
                                         compute_type,
                                         CUBLAS_GEMM_DEFAULT);
      if(status!=CUBLAS_STATUS_SUCCESS)
      {
        THROW_CAIFE("FrozenLinear: cublasGemmEx backward failed");
      }
#else
      THROW_CAIFE("FrozenLinear: reduced precision requires CUDA");
#endif
    }

    if(shape.size()>2)
    {
      std::vector<uint32_t> in_shape(shape.begin(),shape.end()-1);
      in_shape.push_back(_input_dim);
      grad_input.Reshape(in_shape);
    }
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ZeroGradients()
{
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ParameterTensorCount()const
{
  return 0;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    static_cast<void>(index);
    THROW_CAIFE("FrozenLinear: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceFrozenLinear<ComputeT,StorageT>::TotalParameterCount()const
{
  return 0;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceFrozenLinear<ComputeT,StorageT>::Description()const
{
  try
  {
    CAIF_DataType dt(Base_t::StorageDtype());
    return std::string(g_caif_description_frozen_linear)+
           "("+std::to_string(_input_dim)+
           ","+std::to_string(_output_dim)+
           ","+dt.Name()+")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    static_cast<void>(prefix);
    return {};
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceFrozenLinear<ComputeT,StorageT>::FrozenTensorCount()const
{
  return 1;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::FrozenTensorFP32(size_t index)const
{
  try
  {
    if(index!=0)
    {
      THROW_CAIFE("FrozenLinear: frozen tensor index out of range");
    }
    if(_weight.IsEmpty()==true)
    {
      THROW_CAIFE("FrozenLinear: weight not loaded");
    }

    constexpr CAIF_DataType::CAIF_DataType_e sd=Base_t::StorageDtype();
    if(sd==CAIF_DataType::CAIF_DataType_e::Int4)
    {
#ifdef USE_CAIF_CUDA
      const size_t total_elements=static_cast<size_t>(_input_dim)*_output_dim;
      CAIF_DeviceTensor fp32=CAIF_DeviceTensor::Uninitialized({_output_dim,_input_dim},Stream());
      launch_dequantize_int4(_weight.DeviceDataRaw(),
                             _scales.DeviceDataRaw(),
                             fp32.template DevicePtr<float>(),
                             static_cast<int>(total_elements),
                             static_cast<int>(_group_size),
                             Stream().Handle());
      return fp32;
#else
      THROW_CAIFE("INT4 dequantization requires CUDA");
#endif
    }

    if(sd==CAIF_DataType::CAIF_DataType_e::Int8
       &&_scales.IsEmpty()==false)
    {
#ifdef USE_CAIF_CUDA
      const size_t total_elements=static_cast<size_t>(_input_dim)*_output_dim;
      CAIF_DeviceTensor fp32=CAIF_DeviceTensor::Uninitialized({_output_dim,_input_dim},Stream());
      if(_int8_scheme==CAIF_Ops::QuantScheme_e::PerTensor_e)
      {
        launch_dequantize_int8_per_tensor(_weight.DeviceDataRaw(),
                                          fp32.template DevicePtr<float>(),
                                          _scales.DeviceDataRaw(),
                                          static_cast<int>(total_elements),
                                          Stream().Handle());
      }
      else
      {
        launch_dequantize_int8_per_channel(_weight.DeviceDataRaw(),
                                           fp32.template DevicePtr<float>(),
                                           _scales.DeviceDataRaw(),
                                           static_cast<int>(_input_dim),
                                           static_cast<int>(_output_dim),
                                           Stream().Handle());
      }
      return fp32;
#else
      THROW_CAIFE("INT8 dequantization requires CUDA");
#endif
    }

    CAIF_DeviceTensor result=_weight.To(CAIF_DataType::CAIF_DataType_e::Float32);
    result.Reshape({_output_dim,_input_dim});
    return result;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceFrozenLinear<ComputeT,StorageT>::FrozenTensorNames(const std::string &prefix)const
{
  return {prefix+g_caif_name_weight};
}

// Explicit instantiations — full 5×3 (StorageT, ComputeT) grid.
// 9 float-storage cells + 6 int-storage cells = 15 cells.
template class CAIF_DeviceFrozenLinear<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceFrozenLinear<float,__half>;
template class CAIF_DeviceFrozenLinear<float,__nv_bfloat16>;
template class CAIF_DeviceFrozenLinear<__half,float>;
template class CAIF_DeviceFrozenLinear<__half,__half>;
template class CAIF_DeviceFrozenLinear<__half,__nv_bfloat16>;
template class CAIF_DeviceFrozenLinear<__nv_bfloat16,float>;
template class CAIF_DeviceFrozenLinear<__nv_bfloat16,__half>;
template class CAIF_DeviceFrozenLinear<__nv_bfloat16,__nv_bfloat16>;
template class CAIF_DeviceFrozenLinear<float,int8_t>;
template class CAIF_DeviceFrozenLinear<__half,int8_t>;
template class CAIF_DeviceFrozenLinear<__nv_bfloat16,int8_t>;
template class CAIF_DeviceFrozenLinear<float,caif_int4_packed_t>;
template class CAIF_DeviceFrozenLinear<__half,caif_int4_packed_t>;
template class CAIF_DeviceFrozenLinear<__nv_bfloat16,caif_int4_packed_t>;
#endif

}//end instance namespace
