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
// Device-resident spectrogram embedding implementation (templated).
//------------------------------------------------------------------------------
#include "caif_device_spectrogram_embedding.h"
#include "caif_device_spectrogram_embedding_factory.h"
#include "caif_ops.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <cstring>
#include <cmath>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::CAIF_DeviceSpectrogramEmbedding(
                                                  const Config_t &config,
                                                  CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _w_proj(),
                                          _b_proj(),
                                          _cls_token(),
                                          _grad_w_proj(),
                                          _grad_b_proj(),
                                          _grad_cls(),
                                          _cached_input(),
                                          _cached_batch(0),
                                          _cached_time_frames(0)
{
  try
  {
    if(config.freq_bins==0)
    {
      THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding: freq_bins must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding: dim must be > 0");
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    _w_proj=CAIF_DeviceTensor::Zeros({config.freq_bins,config.dim},stream,sdt);
    _b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream,sdt);

    _grad_w_proj=CAIF_DeviceTensor::Zeros({config.freq_bins,config.dim},stream,sdt);
    _grad_b_proj=CAIF_DeviceTensor::Zeros({config.dim},stream,sdt);

    if(config.use_cls_token==true)
    {
      _cls_token=CAIF_DeviceTensor::Zeros({1,config.dim},stream,sdt);
      _grad_cls=CAIF_DeviceTensor::Zeros({1,config.dim},stream,sdt);
    }

    InitializeWeights(0);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::CAIF_DeviceSpectrogramEmbedding(
                                              CAIF_DeviceSpectrogramEmbedding &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT> &
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::operator=(CAIF_DeviceSpectrogramEmbedding &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
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
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                                 CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);

    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.size()!=3)
    {
      THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding: input must be 3D [batch, time_frames, freq_bins]");
    }

    const uint32_t batch=shape[0];
    const uint32_t time_frames=shape[1];
    const uint32_t freq_bins=shape[2];

    if(freq_bins!=_config.freq_bins)
    {
      THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding: freq_bins mismatch");
    }

    const uint32_t dim=_config.dim;

    if(ctx.Training()==true)
    {
      _cached_input=input.Clone();
      _cached_batch=batch;
      _cached_time_frames=time_frames;
    }

    const uint32_t total_rows=batch*time_frames;
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({total_rows,freq_bins});

    CAIF_DeviceTensor proj_output=AllocateOutput({total_rows,dim},ctx);
    CAIF_Ops::MatMul(flat_input,_w_proj,proj_output,ctx);
    CAIF_Ops::BiasAdd(proj_output,_b_proj,proj_output);

    proj_output.Reshape({batch,time_frames,dim});

    if(_config.use_cls_token==true)
    {
      const uint32_t out_seq_len=time_frames+1;
      CAIF_DeviceTensor output=AllocateOutput({batch,out_seq_len,dim},ctx);

      StorageT *output_ptr=StoragePtr(output);
      const StorageT *cls_ptr=StoragePtr(_cls_token);
      const StorageT *proj_ptr=StoragePtr(proj_output);

      for(uint32_t b=0;b<batch;++b)
      {
        const size_t dst_offset=static_cast<size_t>(b)*out_seq_len*dim;
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(output_ptr+dst_offset,
                        cls_ptr,
                        dim*sizeof(StorageT),
                        cudaMemcpyDeviceToDevice,
                        ctx.Stream().Handle());
#else
        std::memcpy(output_ptr+dst_offset,
                    cls_ptr,
                    dim*sizeof(StorageT));
#endif
      }

      for(uint32_t b=0;b<batch;++b)
      {
        const size_t src_offset=static_cast<size_t>(b)*time_frames*dim;
        const size_t dst_offset=static_cast<size_t>(b)*out_seq_len*dim+dim;
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(output_ptr+dst_offset,
                        proj_ptr+src_offset,
                        static_cast<size_t>(time_frames)*dim*sizeof(StorageT),
                        cudaMemcpyDeviceToDevice,
                        ctx.Stream().Handle());
#else
        std::memcpy(output_ptr+dst_offset,
                    proj_ptr+src_offset,
                    static_cast<size_t>(time_frames)*dim*sizeof(StorageT));
#endif
      }

      return output;
    }

    return proj_output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                                  CAIF_RunContext &ctx)
{
  try
  {
    const uint32_t batch=_cached_batch;
    const uint32_t time_frames=_cached_time_frames;
    const uint32_t freq_bins=_config.freq_bins;
    const uint32_t dim=_config.dim;
    const uint32_t total_rows=batch*time_frames;

    CAIF_DeviceTensor grad_proj;

    if(_config.use_cls_token==true)
    {
      const uint32_t out_seq_len=time_frames+1;
      StorageT *grad_cls_ptr=StoragePtr(_grad_cls);
      const StorageT *grad_out_ptr=StoragePtr(grad_output);

      for(uint32_t b=0;b<batch;++b)
      {
        const size_t src_offset=static_cast<size_t>(b)*out_seq_len*dim;
        launch_elementwise_add<StorageT>(grad_cls_ptr,
                                          grad_out_ptr+src_offset,
                                          grad_cls_ptr,
                                          static_cast<int>(dim),
                                          ctx.Stream().Handle());
      }

      grad_proj=AllocateOutput({batch,time_frames,dim},ctx);
      StorageT *grad_proj_ptr=StoragePtr(grad_proj);
      for(uint32_t b=0;b<batch;++b)
      {
        const size_t src_offset=static_cast<size_t>(b)*out_seq_len*dim+dim;
        const size_t dst_offset=static_cast<size_t>(b)*time_frames*dim;
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(grad_proj_ptr+dst_offset,
                        grad_out_ptr+src_offset,
                        static_cast<size_t>(time_frames)*dim*sizeof(StorageT),
                        cudaMemcpyDeviceToDevice,
                        ctx.Stream().Handle());
#else
        std::memcpy(grad_proj_ptr+dst_offset,
                    grad_out_ptr+src_offset,
                    static_cast<size_t>(time_frames)*dim*sizeof(StorageT));
#endif
      }
    }
    else
    {
      grad_proj=grad_output.Clone();
    }

    grad_proj.Reshape({total_rows,dim});

    CAIF_Ops::BiasGradient(grad_proj,_grad_b_proj);

    CAIF_DeviceTensor flat_input=_cached_input.Clone();
    flat_input.Reshape({total_rows,freq_bins});
    CAIF_Ops::MatMulTransposeA(flat_input,grad_proj,_grad_w_proj,ctx);

    CAIF_DeviceTensor grad_input=AllocateOutput({total_rows,freq_bins},ctx);
    CAIF_Ops::MatMulTransposeB(grad_proj,_w_proj,grad_input,ctx);

    grad_input.Reshape({batch,time_frames,freq_bins});

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::ZeroGradients()
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
size_t CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(_config.use_cls_token==true)
  {
    return 3;
  }
  return 2;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)
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
    THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::ParameterTensor(size_t index)const
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
    THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)
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
    THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::GradientTensor(size_t index)const
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
    THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::TotalParameterCount()const
{
  size_t count=static_cast<size_t>(_config.freq_bins)*_config.dim+_config.dim;
  if(_config.use_cls_token==true)
  {
    count+=_config.dim;
  }
  return count;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::Description()const
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

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
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

template<typename ComputeT,typename StorageT>
void CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;

    const float w_limit=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(_config.freq_bins+_config.dim));
    const size_t w_size=static_cast<size_t>(_config.freq_bins)*_config.dim;
    std::vector<float> w_data(w_size);
    for(size_t i=0;i<w_size;++i)
    {
      const float t=static_cast<float>(i+seed)*g_caif_golden_ratio_frac;
      w_data[i]=(t-std::floor(t))*2.0f*w_limit-w_limit;
    }

    WProjMut().CopyFromHostFp32(w_data.data(),w_size);

    if(UseCLSToken()==true)
    {
      const float cls_limit=std::sqrt(g_caif_xavier_uniform_scale/
                                       static_cast<float>(Dim()+Dim()));
      std::vector<float> cls_data(Dim());
      for(uint32_t i=0;i<Dim();++i)
      {
        const float t=static_cast<float>(i+seed)*g_caif_golden_ratio_frac;
        cls_data[i]=(t-std::floor(t))*2.0f*cls_limit-cls_limit;
      }
      CLSTokenMut().CopyFromHostFp32(cls_data.data(),Dim());
    }
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceSpectrogramEmbedding<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceSpectrogramEmbedding<float,__half>;
template class CAIF_DeviceSpectrogramEmbedding<float,__nv_bfloat16>;
template class CAIF_DeviceSpectrogramEmbedding<__half,float>;
template class CAIF_DeviceSpectrogramEmbedding<__half,__half>;
template class CAIF_DeviceSpectrogramEmbedding<__half,__nv_bfloat16>;
template class CAIF_DeviceSpectrogramEmbedding<__nv_bfloat16,float>;
template class CAIF_DeviceSpectrogramEmbedding<__nv_bfloat16,__half>;
template class CAIF_DeviceSpectrogramEmbedding<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
