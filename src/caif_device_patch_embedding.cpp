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
#include "caif_device_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <vector>
#include <cmath>

namespace instance
{

CAIF_DevicePatchEmbedding::CAIF_DevicePatchEmbedding(const CAIF_DevicePatchEmbedding::Config_t &config,
                                                   CAIF_CudaStream &stream):
                                                   CAIF_DeviceLayer(stream),
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
      THROW_CAIFE("DevicePatchEmbedding: patch_size must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("DevicePatchEmbedding: dim must be > 0");
    }
    if(config.image_height%config.patch_size!=0)
    {
      THROW_CAIFE("DevicePatchEmbedding: image_height must be divisible by patch_size");
    }
    if(config.image_width%config.patch_size!=0)
    {
      THROW_CAIFE("DevicePatchEmbedding: image_width must be divisible by patch_size");
    }

    _num_patches_h=config.image_height/config.patch_size;
    _num_patches_w=config.image_width/config.patch_size;
    _num_patches=_num_patches_h*_num_patches_w;
    _patch_flat_dim=config.patch_size*config.patch_size*config.channels;

    // Xavier uniform init for W_proj
    const float w_limit=std::sqrt(6.0f/static_cast<float>(_patch_flat_dim+config.dim));
    const size_t w_size=static_cast<size_t>(_patch_flat_dim)*config.dim;
    std::vector<float> w_init(w_size);
    for(size_t i=0;i<w_size;++i)
    {
      const float t=static_cast<float>(i)*0.6180339887f;
      w_init[i]=(t-std::floor(t))*2.0f*w_limit-w_limit;
    }
    _w_proj=CAIF_DeviceTensor::Uninitialized({_patch_flat_dim,config.dim},stream);
    _w_proj.CopyFromHost(w_init.data(),w_size);

    // b_proj = zeros
    _b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream);

    // Gradients
    _grad_w_proj=CAIF_DeviceTensor::Zeros({_patch_flat_dim,config.dim},stream);
    _grad_b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream);

    // CLS token (if enabled)
    if(config.use_cls_token==true)
    {
      const float cls_limit=std::sqrt(6.0f/static_cast<float>(1+config.dim));
      std::vector<float> cls_init(config.dim);
      for(uint32_t i=0;i<config.dim;++i)
      {
        const float t=static_cast<float>(i)*0.6180339887f+0.3f;
        cls_init[i]=(t-std::floor(t))*2.0f*cls_limit-cls_limit;
      }
      _cls_token=CAIF_DeviceTensor::Uninitialized({1,config.dim},stream);
      _cls_token.CopyFromHost(cls_init.data(),config.dim);
      _grad_cls=CAIF_DeviceTensor::Zeros({1,config.dim},stream);
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DevicePatchEmbedding::CAIF_DevicePatchEmbedding(CAIF_DevicePatchEmbedding &&other):
                                                   CAIF_DeviceLayer(std::move(other)),
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

CAIF_DevicePatchEmbedding &CAIF_DevicePatchEmbedding::operator=(CAIF_DevicePatchEmbedding &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
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
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DevicePatchEmbedding::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DevicePatchEmbedding: layer has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.size()!=4)
    {
      THROW_CAIFE("DevicePatchEmbedding::Forward: input must be 4D [batch,H,W,C]");
    }

    const uint32_t batch=shape[0];
    const uint32_t h=shape[1];
    const uint32_t w=shape[2];
    const uint32_t c=shape[3];

    if(h!=_config.image_height||w!=_config.image_width||c!=_config.channels)
    {
      THROW_CAIFE("DevicePatchEmbedding::Forward: input dimensions mismatch config");
    }

    // Step 1: Extract patches -> [batch*num_patches, patch_flat_dim]
    const uint32_t total_patches=batch*_num_patches;
    CAIF_DeviceTensor patches=CAIF_DeviceTensor::Uninitialized({total_patches,_patch_flat_dim},*_stream);

    launch_extract_patches(input.DevicePtr(),
                           patches.DevicePtr(),
                           static_cast<int>(batch),
                           static_cast<int>(h),
                           static_cast<int>(w),
                           static_cast<int>(c),
                           static_cast<int>(_config.patch_size),
                           static_cast<int>(_num_patches_h),
                           static_cast<int>(_num_patches_w),
                           static_cast<int>(_patch_flat_dim),
                           _stream->Handle());

    // Step 2: MatMul(patches, W_proj) -> [total_patches, dim]
    CAIF_DeviceTensor projected=CAIF_DeviceTensor::Uninitialized({total_patches,_config.dim},*_stream);
    CAIF_DeviceOps::MatMul(patches,_w_proj,projected);

    // Step 3: BiasAdd(projected, b_proj)
    CAIF_DeviceOps::BiasAdd(projected,_b_proj,projected);

    // Step 4: Reshape to [batch, num_patches, dim]
    projected.Reshape({batch,_num_patches,_config.dim});

    // Step 5: Optional CLS prepend
    CAIF_DeviceTensor output;
    if(_config.use_cls_token==true)
    {
      const uint32_t out_seq=_num_patches+1;
      output=CAIF_DeviceTensor::Uninitialized({batch,out_seq,_config.dim},*_stream);

      launch_cls_prepend(projected.DevicePtr(),
                         _cls_token.DevicePtr(),
                         output.DevicePtr(),
                         static_cast<int>(batch),
                         static_cast<int>(_num_patches),
                         static_cast<int>(_config.dim),
                         _stream->Handle());
    }
    else
    {
      output=std::move(projected);
    }

    // Cache for backward
    if(training==true)
    {
      _cached_input=input.Clone();
      _cached_patches=std::move(patches);
      _cached_batch=batch;
    }

    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DevicePatchEmbedding::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DevicePatchEmbedding: layer has been moved from");
    }
    if(_cached_batch==0)
    {
      THROW_CAIFE(
        "DevicePatchEmbedding::Backward: must call Forward with training=true first");
    }

    const uint32_t batch=_cached_batch;
    const uint32_t total_patches=batch*_num_patches;

    // Step 1: Handle CLS token gradient if needed
    CAIF_DeviceTensor grad_patches;
    if(_config.use_cls_token==true)
    {
      _grad_cls.Fill(0.0f);
      grad_patches=CAIF_DeviceTensor::Uninitialized({batch,_num_patches,_config.dim},*_stream);

      launch_cls_grad_extract(grad_output.DevicePtr(),
                              _grad_cls.DevicePtr(),
                              grad_patches.DevicePtr(),
                              static_cast<int>(batch),
                              static_cast<int>(_num_patches),
                              static_cast<int>(_config.dim),
                              _stream->Handle());
    }
    else
    {
      grad_patches=grad_output.Clone();
    }

    // Reshape to [total_patches, dim]
    grad_patches.Reshape({total_patches,_config.dim});

    // Step 2: Bias gradient
    _grad_b_proj.Fill(0.0f);
    CAIF_DeviceOps::BiasGradient(grad_patches,_grad_b_proj);

    // Step 3: Weight gradient: grad_W = patches^T @ grad_patches
    CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized({_patch_flat_dim,_config.dim},*_stream);
    CAIF_DeviceOps::MatMulTransposeA(_cached_patches,grad_patches,grad_w_delta);
    CAIF_DeviceOps::Add(_grad_w_proj,grad_w_delta,_grad_w_proj);

    // Step 4: Gradient w.r.t. patches: grad_flat = grad_patches @ W_proj^T
    CAIF_DeviceTensor grad_flat=CAIF_DeviceTensor::Uninitialized({total_patches,_patch_flat_dim},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(grad_patches,_w_proj,grad_flat);

    // Step 5: extract_patches_backward -> grad_input [batch, H, W, C]
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Zeros(
                                  {batch,
                                   _config.image_height,
                                   _config.image_width,
                                   _config.channels},
                                  *_stream);

    launch_extract_patches_backward(grad_flat.DevicePtr(),
                                    grad_input.DevicePtr(),
                                    static_cast<int>(batch),
                                    static_cast<int>(_config.image_height),
                                    static_cast<int>(_config.image_width),
                                    static_cast<int>(_config.channels),
                                    static_cast<int>(_config.patch_size),
                                    static_cast<int>(_num_patches_h),
                                    static_cast<int>(_num_patches_w),
                                    static_cast<int>(_patch_flat_dim),
                                    _stream->Handle());

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DevicePatchEmbedding::ZeroGradients()
{
  try
  {
    _grad_w_proj.Fill(0.0f);
    _grad_b_proj.Fill(0.0f);
    if(_config.use_cls_token==true)
    {
      _grad_cls.Fill(0.0f);
    }
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DevicePatchEmbedding::ParameterTensorCount()const
{
  try
  {
    if(_config.use_cls_token==true)
    {
      return 3;
    }
    else
    {
      return 2;
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DevicePatchEmbedding::ParameterTensor(size_t index)
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
    THROW_CAIFE("DevicePatchEmbedding::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DevicePatchEmbedding::ParameterTensor(size_t index)const
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
    THROW_CAIFE("DevicePatchEmbedding::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DevicePatchEmbedding::GradientTensor(size_t index)
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
    THROW_CAIFE("DevicePatchEmbedding::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DevicePatchEmbedding::GradientTensor(size_t index)const
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
    THROW_CAIFE("DevicePatchEmbedding::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DevicePatchEmbedding::TotalParameterCount()const
{
  try
  {
    size_t total=static_cast<size_t>(_patch_flat_dim)*_config.dim;  // W_proj
    total+=_config.dim;  // b_proj
    if(_config.use_cls_token==true)
    {
      total+=_config.dim;  // cls_token
    }
    return total;
  }
  CCAIF_CATCH_BLOCK()
}

std::string CAIF_DevicePatchEmbedding::Description()const
{
  try
  {
    std::string desc="PatchEmbedding(patch="+std::to_string(_config.patch_size)+
                     ",ch="+std::to_string(_config.channels)+
                     ",dim="+std::to_string(_config.dim)+")";
    if(_config.use_cls_token==true)
    {
      desc="PatchEmbedding(patch="+std::to_string(_config.patch_size)+
           ",ch="+std::to_string(_config.channels)+
           ",dim="+std::to_string(_config.dim)+
           ",cls=true)";
    }
    return desc;
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DevicePatchEmbedding::ParameterNames(const std::string &prefix)const
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
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
