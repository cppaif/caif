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
#include "caif_cuda_kernels_elementwise.cuh"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <cstring>
#include <cmath>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::CAIF_DeviceSpectrogramEmbedding(
                                                  const CAIF_DeviceSpectrogramEmbeddingConfig &config,
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
    if(config.FreqBins()==0)
    {
      THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding: freq_bins must be > 0");
    }
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding: dim must be > 0");
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    SetWProj(CAIF_DeviceTensor::Zeros({config.FreqBins(),config.Dim()},stream,sdt));
    SetBProj(CAIF_DeviceTensor::Zeros({config.Dim()},stream,sdt));

    SetGradWProj(CAIF_DeviceTensor::Zeros({config.FreqBins(),config.Dim()},stream,sdt));
    SetGradBProj(CAIF_DeviceTensor::Zeros({config.Dim()},stream,sdt));

    if(config.UseCLSToken()==true)
    {
      SetCLSToken(CAIF_DeviceTensor::Zeros({1,config.Dim()},stream,sdt));
      SetGradCLS(CAIF_DeviceTensor::Zeros({1,config.Dim()},stream,sdt));
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
      SetConfig(other.Config());
      SetWProj(std::move(other.WProjMut()));
      SetBProj(std::move(other.BProj()));
      SetCLSToken(std::move(other.CLSTokenMut()));
      SetGradWProj(std::move(other.GradWProj()));
      SetGradBProj(std::move(other.GradBProj()));
      SetGradCLS(std::move(other.GradCLS()));
      SetCachedInput(std::move(other.CachedInput()));
      SetCachedBatch(other.CachedBatch());
      SetCachedTimeFrames(other.CachedTimeFrames());
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

    if(freq_bins!=Config().FreqBins())
    {
      THROW_CAIFE("CAIF_DeviceSpectrogramEmbedding: freq_bins mismatch");
    }

    const uint32_t dim=Config().Dim();

    if(ctx.Training()==true)
    {
      SetCachedInput(input.Clone());
      SetCachedBatch(batch);
      SetCachedTimeFrames(time_frames);
    }

    const uint32_t total_rows=batch*time_frames;
    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({total_rows,freq_bins});

    CAIF_DeviceTensor proj_output=AllocateOutput({total_rows,dim},ctx);
    CAIF_Ops::MatMul(flat_input,WProjMut(),proj_output,ctx);
    CAIF_Ops::BiasAdd(proj_output,BProj(),proj_output);

    proj_output.Reshape({batch,time_frames,dim});

    if(Config().UseCLSToken()==true)
    {
      const uint32_t out_seq_len=time_frames+1;
      CAIF_DeviceTensor output=AllocateOutput({batch,out_seq_len,dim},ctx);

      StorageT *output_ptr=StoragePtr(output);
      const StorageT *cls_ptr=StoragePtr(CLSToken());
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
    const uint32_t batch=CachedBatch();
    const uint32_t time_frames=CachedTimeFrames();
    const uint32_t freq_bins=Config().FreqBins();
    const uint32_t dim=Config().Dim();
    const uint32_t total_rows=batch*time_frames;

    CAIF_DeviceTensor grad_proj;

    if(Config().UseCLSToken()==true)
    {
      const uint32_t out_seq_len=time_frames+1;
      StorageT *grad_cls_ptr=StoragePtr(GradCLS());
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

    CAIF_Ops::BiasGradient(grad_proj,GradBProj());

    CAIF_DeviceTensor flat_input=CachedInput().Clone();
    flat_input.Reshape({total_rows,freq_bins});
    CAIF_Ops::MatMulTransposeA(flat_input,grad_proj,GradWProj(),ctx);

    CAIF_DeviceTensor grad_input=AllocateOutput({total_rows,freq_bins},ctx);
    CAIF_Ops::MatMulTransposeB(grad_proj,WProjMut(),grad_input,ctx);

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
    GradWProj().FillZero();
    GradBProj().FillZero();
    if(Config().UseCLSToken()==true)
    {
      GradCLS().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
  size_t count=static_cast<size_t>(Config().FreqBins())*Config().Dim()+Config().Dim();
  if(Config().UseCLSToken()==true)
  {
    count+=Config().Dim();
  }
  return count;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceSpectrogramEmbedding<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc=std::string(g_serial_tag_spectrogram_embedding)+
                     g_serial_open_paren+
                     g_serial_kv_freq_bins+
                     std::to_string(Config().FreqBins())+
                     g_serial_comma+
                     g_serial_kv_dim+
                     std::to_string(Config().Dim());
    if(Config().UseCLSToken()==true)
    {
      desc+=g_serial_flag_cls_true;
    }
    desc+=g_serial_close_paren;
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
    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
    std::vector<std::string> names;
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::EmbedProjWeight_e));
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::EmbedProjBias_e));
    if(Config().UseCLSToken()==true)
    {
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::EmbedClsToken_e));
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
    const float w_limit=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(Config().FreqBins()+Config().Dim()));
    const size_t w_size=static_cast<size_t>(Config().FreqBins())*Config().Dim();
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
