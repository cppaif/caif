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
// Device-resident spectrogram embedding implementation
//------------------------------------------------------------------------------
#include "caif_device_spectrogram_embedding.h"
#include "caif_device_ops.h"
#include <cstring>
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <cmath>
#include <random>

namespace instance
{

CAIF_DeviceSpectrogramEmbedding::CAIF_DeviceSpectrogramEmbedding(const Config_t &config,
                                                               CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                               _config(config),
                                                               _cached_batch(0),
                                                               _cached_time_frames(0)
{
  try
  {
    if(config.freq_bins==0)
    {
      THROW_CAIFE("DeviceSpectrogramEmbedding: freq_bins must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceSpectrogramEmbedding: dim must be > 0");
    }

    // Allocate parameters
    _w_proj=CAIF_DeviceTensor::Zeros({config.freq_bins,config.dim},stream);
    _b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream);

    // Allocate gradients
    _grad_w_proj=CAIF_DeviceTensor::Zeros({config.freq_bins,config.dim},stream);
    _grad_b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream);

    // CLS token (optional)
    if(config.use_cls_token==true)
    {
      _cls_token=CAIF_DeviceTensor::Zeros({1,config.dim},stream);
      _grad_cls=CAIF_DeviceTensor::Zeros({1,config.dim},stream);
    }

    // Initialize weights
    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceSpectrogramEmbedding::CAIF_DeviceSpectrogramEmbedding(
  CAIF_DeviceSpectrogramEmbedding &&other):CAIF_DeviceLayer(std::move(other)),
                                          _config(other._config),
                                          _w_proj(std::move(other._w_proj)),
                                          _b_proj(std::move(other._b_proj)),
                                          _cls_token(std::move(other._cls_token)),
                                          _grad_w_proj(std::move(other._grad_w_proj)),
                                          _grad_b_proj(std::move(other._grad_b_proj)),
                                          _grad_cls(std::move(other._grad_cls)),
                                          _cached_input(std::move(other._cached_input)),
                                          _cached_batch(other._cached_batch),
                                          _cached_time_frames(other._cached_time_frames)
{
}

CAIF_DeviceSpectrogramEmbedding &CAIF_DeviceSpectrogramEmbedding::operator=(CAIF_DeviceSpectrogramEmbedding &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _w_proj=std::move(other._w_proj);
    _b_proj=std::move(other._b_proj);
    _cls_token=std::move(other._cls_token);
    _grad_w_proj=std::move(other._grad_w_proj);
    _grad_b_proj=std::move(other._grad_b_proj);
    _grad_cls=std::move(other._grad_cls);
    _cached_input=std::move(other._cached_input);
    _cached_batch=other._cached_batch;
    _cached_time_frames=other._cached_time_frames;
  }
  return *this;
}

CAIF_DeviceTensor CAIF_DeviceSpectrogramEmbedding::Forward(const CAIF_DeviceTensor &input,
                                                         bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceSpectrogramEmbedding: layer has been moved from");
    }

    // Input: [batch, time_frames, freq_bins]
    const auto &shape=input.Shape();
    if(shape.size()!=3)
    {
      THROW_CAIFE("DeviceSpectrogramEmbedding: input must be 3D [batch, time_frames, freq_bins]");
    }

    const uint32_t batch=shape[0];
    const uint32_t time_frames=shape[1];
    const uint32_t freq_bins=shape[2];

    if(freq_bins!=_config.freq_bins)
    {
      THROW_CAIFE("DeviceSpectrogramEmbedding: freq_bins mismatch");
    }

    const uint32_t dim=_config.dim;

    // Cache for backward
    if(training==true)
    {
      _cached_input=input.Clone();
      _cached_batch=batch;
      _cached_time_frames=time_frames;
    }

    // Flatten to [batch*time_frames, freq_bins]
    const uint32_t total_rows=batch*time_frames;
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({total_rows,freq_bins});

    // Project: output = input @ W_proj
    CAIF_DeviceTensor proj_output=CAIF_DeviceTensor::Uninitialized({total_rows,dim},*_stream);
    CAIF_DeviceOps::MatMul(flat_input,_w_proj,proj_output);

    // Add bias
    launch_bias_add_2d(proj_output.DevicePtr(),_b_proj.DevicePtr(),
                       proj_output.DevicePtr(),
                       static_cast<int>(total_rows),
                       static_cast<int>(dim),
                       _stream->Handle());

    // Reshape to [batch, time_frames, dim]
    proj_output.Reshape({batch,time_frames,dim});

    // Optionally prepend CLS token
    if(_config.use_cls_token==true)
    {
      const uint32_t out_seq_len=time_frames+1;
      CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({batch,out_seq_len,dim},*_stream);

      // Copy CLS token to position 0 for each batch
      for(uint32_t b=0;b<batch;++b)
      {
        const size_t dst_offset=b*out_seq_len*dim;
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(output.DevicePtr()+dst_offset,
                        _cls_token.DevicePtr(),
                        dim*sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        _stream->Handle());
#else
        std::memcpy(output.DevicePtr()+dst_offset,
                    _cls_token.DevicePtr(),
                    dim*sizeof(float));
#endif
      }

      // Copy projected frames to positions 1..time_frames for each batch
      for(uint32_t b=0;b<batch;++b)
      {
        const size_t src_offset=b*time_frames*dim;
        const size_t dst_offset=b*out_seq_len*dim+dim;  // +dim to skip CLS
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(output.DevicePtr()+dst_offset,
                        proj_output.DevicePtr()+src_offset,
                        time_frames*dim*sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        _stream->Handle());
#else
        std::memcpy(output.DevicePtr()+dst_offset,
                    proj_output.DevicePtr()+src_offset,
                    time_frames*dim*sizeof(float));
#endif
      }

      return output;
    }

    return proj_output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceSpectrogramEmbedding::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceSpectrogramEmbedding: layer has been moved from");
    }

    const uint32_t batch=_cached_batch;
    const uint32_t time_frames=_cached_time_frames;
    const uint32_t freq_bins=_config.freq_bins;
    const uint32_t dim=_config.dim;
    const uint32_t total_rows=batch*time_frames;

    CAIF_DeviceTensor grad_proj;

    if(_config.use_cls_token==true)
    {
      // grad_output is [batch, time_frames+1, dim]
      const uint32_t out_seq_len=time_frames+1;

      // Extract gradient for CLS token (position 0) and accumulate
      for(uint32_t b=0;b<batch;++b)
      {
        const size_t src_offset=b*out_seq_len*dim;
        // Accumulate gradient for CLS token
        launch_elementwise_add(_grad_cls.DevicePtr(),
                               grad_output.DevicePtr()+src_offset,
                               _grad_cls.DevicePtr(),
                               static_cast<int>(dim),
                               _stream->Handle());
      }

      // Extract gradients for projected frames (positions 1..time_frames)
      grad_proj=CAIF_DeviceTensor::Uninitialized({batch,time_frames,dim},*_stream);
      for(uint32_t b=0;b<batch;++b)
      {
        const size_t src_offset=b*out_seq_len*dim+dim;  // +dim to skip CLS
        const size_t dst_offset=b*time_frames*dim;
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(grad_proj.DevicePtr()+dst_offset,
                        grad_output.DevicePtr()+src_offset,
                        time_frames*dim*sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        _stream->Handle());
#else
        std::memcpy(grad_proj.DevicePtr()+dst_offset,
                    grad_output.DevicePtr()+src_offset,
                    time_frames*dim*sizeof(float));
#endif
      }
    }
    else
    {
      grad_proj=grad_output.Clone();
    }

    // Flatten to [total_rows, dim]
    grad_proj.Reshape({total_rows,dim});

    // grad_b_proj = sum(grad_proj, axis=0)
    launch_bias_grad_2d(grad_proj.DevicePtr(),_grad_b_proj.DevicePtr(),
                        static_cast<int>(total_rows),
                        static_cast<int>(dim),
                        _stream->Handle());

    // grad_w_proj = input^T @ grad_proj
    CAIF_DeviceTensor flat_input=_cached_input.Clone();
    flat_input.Reshape({total_rows,freq_bins});
    CAIF_DeviceOps::MatMulTransposeA(flat_input,grad_proj,_grad_w_proj);

    // grad_input = grad_proj @ W_proj^T
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({total_rows,freq_bins},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(grad_proj,_w_proj,grad_input);

    // Reshape to [batch, time_frames, freq_bins]
    grad_input.Reshape({batch,time_frames,freq_bins});

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceSpectrogramEmbedding::ZeroGradients()
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
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceSpectrogramEmbedding::ParameterTensorCount()const
{
  if(_config.use_cls_token==true)
  {
    return 3;  // w_proj, b_proj, cls_token
  }
  return 2;  // w_proj, b_proj
}

CAIF_DeviceTensor &CAIF_DeviceSpectrogramEmbedding::ParameterTensor(size_t index)
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
  THROW_CAIFE("DeviceSpectrogramEmbedding: parameter index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceSpectrogramEmbedding::ParameterTensor(size_t index)const
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
  THROW_CAIFE("DeviceSpectrogramEmbedding: parameter index out of range");
}

CAIF_DeviceTensor &CAIF_DeviceSpectrogramEmbedding::GradientTensor(size_t index)
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
  THROW_CAIFE("DeviceSpectrogramEmbedding: gradient index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceSpectrogramEmbedding::GradientTensor(size_t index)const
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
  THROW_CAIFE("DeviceSpectrogramEmbedding: gradient index out of range");
}

size_t CAIF_DeviceSpectrogramEmbedding::TotalParameterCount()const
{
  size_t count=_config.freq_bins*_config.dim+_config.dim;
  if(_config.use_cls_token==true)
  {
    count+=_config.dim;  // CLS token
  }
  return count;
}

std::string CAIF_DeviceSpectrogramEmbedding::Description()const
{
  try
  {
    std::string desc="SpectrogramEmbedding(freq_bins="+std::to_string(_config.freq_bins)+
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

std::vector<std::string> CAIF_DeviceSpectrogramEmbedding::ParameterNames(const std::string &prefix)const
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

void CAIF_DeviceSpectrogramEmbedding::InitializeWeights(uint32_t seed)
{
  try
  {
    std::mt19937 rng(seed);

    // Xavier uniform for W_proj
    const float limit=std::sqrt(6.0f/static_cast<float>(_config.freq_bins+_config.dim));
    std::uniform_real_distribution<float> dist(-limit,limit);

    std::vector<float> w_data(_config.freq_bins*_config.dim);
    for(size_t i=0;i<w_data.size();++i)
    {
      w_data[i]=dist(rng);
    }
    _w_proj.CopyFromHost(w_data.data(),w_data.size());

    // Bias initialized to zero (already done)

    // CLS token with Xavier uniform
    if(_config.use_cls_token==true)
    {
      const float cls_limit=std::sqrt(6.0f/static_cast<float>(_config.dim+_config.dim));
      std::uniform_real_distribution<float> cls_dist(-cls_limit,cls_limit);

      std::vector<float> cls_data(_config.dim);
      for(size_t i=0;i<cls_data.size();++i)
      {
        cls_data[i]=cls_dist(rng);
      }
      _cls_token.CopyFromHost(cls_data.data(),cls_data.size());
    }
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
