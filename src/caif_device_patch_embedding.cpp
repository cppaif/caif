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

#include "caif_device_patch_embedding.h"
#include "caif_device_patch_embedding_factory.h"
#include "caif_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>
#include <cmath>

namespace instance
{


// CLS-token init uses a small offset on top of the golden-ratio
// sequence so the deterministic seed differs from the W_proj table.
constexpr float g_caif_cls_token_init_offset=0.3f;


template<typename ComputeT,typename StorageT>
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::CAIF_DevicePatchEmbedding(
                                                  const Config_t &config,
                                                  CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _num_patches_h(0),
                                          _num_patches_w(0),
                                          _num_patches(0),
                                          _patch_flat_dim(0),
                                          _w_proj(),
                                          _b_proj(),
                                          _cls_token(),
                                          _grad_w_proj(),
                                          _grad_b_proj(),
                                          _grad_cls(),
                                          _cached_input(),
                                          _cached_patches(),
                                          _cached_batch(0)
{
  try
  {
    if(config.patch_size==0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: patch_size must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: dim must be > 0");
    }
    if(config.image_height%config.patch_size!=0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: image_height must be divisible by patch_size");
    }
    if(config.image_width%config.patch_size!=0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: image_width must be divisible by patch_size");
    }

    _num_patches_h=config.image_height/config.patch_size;
    _num_patches_w=config.image_width/config.patch_size;
    _num_patches=_num_patches_h*_num_patches_w;
    _patch_flat_dim=config.patch_size*config.patch_size*config.channels;

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    // Xavier uniform init for W_proj.
    const float w_limit=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(_patch_flat_dim+config.dim));
    const size_t w_size=static_cast<size_t>(_patch_flat_dim)*config.dim;
    std::vector<float> w_init(w_size);
    for(size_t i=0;i<w_size;++i)
    {
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      w_init[i]=(t-std::floor(t))*2.0f*w_limit-w_limit;
    }
    SetWProj(CAIF_DeviceTensor::Uninitialized({PatchFlatDim(),config.dim},stream,sdt));
    WProjMut().CopyFromHostFp32(w_init.data(),w_size);

    SetBProj(CAIF_DeviceTensor::Zeros({config.dim},stream,sdt));
    SetGradWProj(CAIF_DeviceTensor::Zeros({PatchFlatDim(),config.dim},stream,sdt));
    SetGradBProj(CAIF_DeviceTensor::Zeros({config.dim},stream,sdt));

    if(config.use_cls_token==true)
    {
      const float cls_limit=std::sqrt(g_caif_xavier_uniform_scale/
                                       static_cast<float>(1+config.dim));
      std::vector<float> cls_init(config.dim);
      for(uint32_t i=0;i<config.dim;++i)
      {
        const float t=static_cast<float>(i)*g_caif_golden_ratio_frac+
                       g_caif_cls_token_init_offset;
        cls_init[i]=(t-std::floor(t))*2.0f*cls_limit-cls_limit;
      }
      SetCLSToken(CAIF_DeviceTensor::Uninitialized({1,config.dim},stream,sdt));
      CLSTokenMut().CopyFromHostFp32(cls_init.data(),config.dim);
      SetGradCls(CAIF_DeviceTensor::Zeros({1,config.dim},stream,sdt));
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::CAIF_DevicePatchEmbedding(
                                              CAIF_DevicePatchEmbedding &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _config(other._config),
                              _num_patches_h(other._num_patches_h),
                              _num_patches_w(other._num_patches_w),
                              _num_patches(other._num_patches),
                              _patch_flat_dim(other._patch_flat_dim),
                              _w_proj(std::move(other._w_proj)),
                              _b_proj(std::move(other._b_proj)),
                              _cls_token(std::move(other._cls_token)),
                              _grad_w_proj(std::move(other._grad_w_proj)),
                              _grad_b_proj(std::move(other._grad_b_proj)),
                              _grad_cls(std::move(other._grad_cls)),
                              _cached_input(std::move(other._cached_input)),
                              _cached_patches(std::move(other._cached_patches)),
                              _cached_batch(other._cached_batch)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePatchEmbedding<ComputeT,StorageT> &
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::operator=(CAIF_DevicePatchEmbedding &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
      _config=other._config;
      _num_patches_h=other._num_patches_h;
      _num_patches_w=other._num_patches_w;
      _num_patches=other._num_patches;
      _patch_flat_dim=other._patch_flat_dim;
      _w_proj=std::move(other._w_proj);
      _b_proj=std::move(other._b_proj);
      _cls_token=std::move(other._cls_token);
      _grad_w_proj=std::move(other._grad_w_proj);
      _grad_b_proj=std::move(other._grad_b_proj);
      _grad_cls=std::move(other._grad_cls);
      _cached_input=std::move(other._cached_input);
      _cached_patches=std::move(other._cached_patches);
      _cached_batch=other._cached_batch;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                          CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);

    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.size()!=4)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding::Forward: input must be 4D [batch,H,W,C]");
    }

    const uint32_t batch=shape[0];
    const uint32_t h=shape[1];
    const uint32_t w=shape[2];
    const uint32_t c=shape[3];

    if(h!=_config.image_height||w!=_config.image_width||c!=_config.channels)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding::Forward: input dimensions mismatch config");
    }

    const uint32_t total_patches=batch*_num_patches;
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    CAIF_DeviceTensor patches=CAIF_DeviceTensor::Uninitialized({total_patches,_patch_flat_dim},
                                                                ctx.Stream(),sdt);

    launch_extract_patches<StorageT>(input.template DevicePtr<StorageT>(),
                                     patches.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(h),
                                     static_cast<int>(w),
                                     static_cast<int>(c),
                                     static_cast<int>(_config.patch_size),
                                     static_cast<int>(_num_patches_h),
                                     static_cast<int>(_num_patches_w),
                                     static_cast<int>(_patch_flat_dim),
                                     ctx.Stream().Handle());

    CAIF_DeviceTensor projected=CAIF_DeviceTensor::Uninitialized({total_patches,_config.dim},
                                                                  ctx.Stream(),sdt);
    CAIF_Ops::MatMul(patches,_w_proj,projected,ctx);
    CAIF_Ops::BiasAdd(projected,_b_proj,projected);
    projected.Reshape({batch,_num_patches,_config.dim});

    CAIF_DeviceTensor output;
    if(_config.use_cls_token==true)
    {
      const uint32_t out_seq=_num_patches+1;
      output=CAIF_DeviceTensor::Uninitialized({batch,out_seq,_config.dim},ctx.Stream(),sdt);
      launch_cls_prepend<StorageT>(projected.template DevicePtr<StorageT>(),
                                   _cls_token.template DevicePtr<StorageT>(),
                                   output.template DevicePtr<StorageT>(),
                                   static_cast<int>(batch),
                                   static_cast<int>(_num_patches),
                                   static_cast<int>(_config.dim),
                                   ctx.Stream().Handle());
    }
    else
    {
      output=std::move(projected);
    }

    if(ctx.Training()==true)
    {
      _cached_input=input.Clone();
      _cached_patches=std::move(patches);
      _cached_batch=batch;
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                            CAIF_RunContext &ctx)
{
  try
  {
    if(_cached_batch==0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding::Backward: must call Forward with training=true first");
    }

    const uint32_t batch=_cached_batch;
    const uint32_t total_patches=batch*_num_patches;
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    CAIF_DeviceTensor grad_patches;
    if(_config.use_cls_token==true)
    {
      _grad_cls.FillZero();
      grad_patches=CAIF_DeviceTensor::Uninitialized({batch,_num_patches,_config.dim},
                                                     ctx.Stream(),sdt);
      launch_cls_grad_extract<StorageT>(grad_output.template DevicePtr<StorageT>(),
                                        _grad_cls.template DevicePtr<StorageT>(),
                                        grad_patches.template DevicePtr<StorageT>(),
                                        static_cast<int>(batch),
                                        static_cast<int>(_num_patches),
                                        static_cast<int>(_config.dim),
                                        ctx.Stream().Handle());
    }
    else
    {
      grad_patches=grad_output.Clone();
    }
    grad_patches.Reshape({total_patches,_config.dim});

    _grad_b_proj.FillZero();
    CAIF_Ops::BiasGradient(grad_patches,_grad_b_proj);

    CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized({_patch_flat_dim,_config.dim},
                                                                     ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeA(_cached_patches,grad_patches,grad_w_delta,ctx);
    CAIF_Ops::Add(_grad_w_proj,grad_w_delta,_grad_w_proj);

    CAIF_DeviceTensor grad_flat=CAIF_DeviceTensor::Uninitialized({total_patches,_patch_flat_dim},
                                                                  ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeB(grad_patches,_w_proj,grad_flat,ctx);

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Zeros(
                                  {batch,
                                   _config.image_height,
                                   _config.image_width,
                                   _config.channels},
                                  ctx.Stream(),sdt);

    launch_extract_patches_backward<StorageT>(grad_flat.template DevicePtr<StorageT>(),
                                              grad_input.template DevicePtr<StorageT>(),
                                              static_cast<int>(batch),
                                              static_cast<int>(_config.image_height),
                                              static_cast<int>(_config.image_width),
                                              static_cast<int>(_config.channels),
                                              static_cast<int>(_config.patch_size),
                                              static_cast<int>(_num_patches_h),
                                              static_cast<int>(_num_patches_w),
                                              static_cast<int>(_patch_flat_dim),
                                              ctx.Stream().Handle());

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    _grad_w_proj.FillZero();
    _grad_b_proj.FillZero();
    if(_config.use_cls_token==true)
    {
      _grad_cls.FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(_config.use_cls_token==true)
  {
    return 3;
  }
  return 2;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _w_proj;
    }
    if(index==1)
    {
      return _b_proj;
    }
    if(index==2&&_config.use_cls_token==true)
    {
      return _cls_token;
    }
    THROW_CAIFE("CAIF_DevicePatchEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _w_proj;
    }
    if(index==1)
    {
      return _b_proj;
    }
    if(index==2&&_config.use_cls_token==true)
    {
      return _cls_token;
    }
    THROW_CAIFE("CAIF_DevicePatchEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return _grad_w_proj;
    }
    if(index==1)
    {
      return _grad_b_proj;
    }
    if(index==2&&_config.use_cls_token==true)
    {
      return _grad_cls;
    }
    THROW_CAIFE("CAIF_DevicePatchEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return _grad_w_proj;
    }
    if(index==1)
    {
      return _grad_b_proj;
    }
    if(index==2&&_config.use_cls_token==true)
    {
      return _grad_cls;
    }
    THROW_CAIFE("CAIF_DevicePatchEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DevicePatchEmbedding<ComputeT,StorageT>::TotalParameterCount()const
{
  size_t total=static_cast<size_t>(_patch_flat_dim)*_config.dim;
  total+=_config.dim;
  if(_config.use_cls_token==true)
  {
    total+=_config.dim;
  }
  return total;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DevicePatchEmbedding<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc="PatchEmbedding(patch="+std::to_string(_config.patch_size)+
                     ",ch="+std::to_string(_config.channels)+
                     ",dim="+std::to_string(_config.dim);
    if(_config.use_cls_token==true)
    {
      desc+=",cls=true";
    }
    desc+=")";
    return desc;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    names.push_back(prefix+"proj.weight");
    names.push_back(prefix+"proj.bias");
    if(_config.use_cls_token==true)
    {
      names.push_back(prefix+"cls_token");
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DevicePatchEmbedding<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DevicePatchEmbedding<float,__half>;
template class CAIF_DevicePatchEmbedding<float,__nv_bfloat16>;
template class CAIF_DevicePatchEmbedding<__half,float>;
template class CAIF_DevicePatchEmbedding<__half,__half>;
template class CAIF_DevicePatchEmbedding<__half,__nv_bfloat16>;
template class CAIF_DevicePatchEmbedding<__nv_bfloat16,float>;
template class CAIF_DevicePatchEmbedding<__nv_bfloat16,__half>;
template class CAIF_DevicePatchEmbedding<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
