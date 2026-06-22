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
#include "caif_cuda_kernels_embeddings.cuh"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include "caif_serialization_constants.h"
#include <vector>
#include <cmath>

namespace instance
{


template<typename ComputeT,typename StorageT>
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::CAIF_DevicePatchEmbedding(
                                                  const CAIF_DevicePatchEmbeddingConfig &config,
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
    if(config.PatchSize()==0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: patch_size must be > 0");
    }
    if(config.Dim()==0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: dim must be > 0");
    }
    if(config.ImageHeight()%config.PatchSize()!=0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: image_height must be divisible by patch_size");
    }
    if(config.ImageWidth()%config.PatchSize()!=0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding: image_width must be divisible by patch_size");
    }

    SetNumPatchesH(config.ImageHeight()/config.PatchSize());
    SetNumPatchesW(config.ImageWidth()/config.PatchSize());
    SetNumPatches(NumPatchesH()*NumPatchesW());
    SetPatchFlatDim(config.PatchSize()*config.PatchSize()*config.Channels());

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    // Xavier uniform init for W_proj.
    const float w_limit=std::sqrt(g_caif_xavier_uniform_scale/
                                   static_cast<float>(PatchFlatDim()+config.Dim()));
    const size_t w_size=static_cast<size_t>(PatchFlatDim())*config.Dim();
    std::vector<float> w_init(w_size);
    for(size_t i=0;i<w_size;++i)
    {
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      w_init[i]=(t-std::floor(t))*2.0f*w_limit-w_limit;
    }
    SetWProj(CAIF_DeviceTensor::Uninitialized({PatchFlatDim(),config.Dim()},stream,sdt));
    WProjMut().CopyFromHostFp32(w_init.data(),w_size);

    SetBProj(CAIF_DeviceTensor::Zeros({config.Dim()},stream,sdt));
    SetGradWProj(CAIF_DeviceTensor::Zeros({PatchFlatDim(),config.Dim()},stream,sdt));
    SetGradBProj(CAIF_DeviceTensor::Zeros({config.Dim()},stream,sdt));

    if(config.UseCLSToken()==true)
    {
      const float cls_limit=std::sqrt(g_caif_xavier_uniform_scale/
                                       static_cast<float>(1+config.Dim()));
      std::vector<float> cls_init(config.Dim());
      for(uint32_t i=0;i<config.Dim();++i)
      {
        const float t=static_cast<float>(i)*g_caif_golden_ratio_frac+
                       g_caif_cls_token_init_offset;
        cls_init[i]=(t-std::floor(t))*2.0f*cls_limit-cls_limit;
      }
      SetCLSToken(CAIF_DeviceTensor::Uninitialized({1,config.Dim()},stream,sdt));
      CLSTokenMut().CopyFromHostFp32(cls_init.data(),config.Dim());
      SetGradCls(CAIF_DeviceTensor::Zeros({1,config.Dim()},stream,sdt));
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
      SetConfig(other.Config());
      SetNumPatchesH(other.NumPatchesH());
      SetNumPatchesW(other.NumPatchesW());
      SetNumPatches(other.NumPatches());
      SetPatchFlatDim(other.PatchFlatDim());
      SetWProj(std::move(other.WProjMut()));
      SetBProj(std::move(other.BProj()));
      SetCLSToken(std::move(other.CLSTokenMut()));
      SetGradWProj(std::move(other.GradWProj()));
      SetGradBProj(std::move(other.GradBProj()));
      SetGradCls(std::move(other.GradCls()));
      SetCachedInput(std::move(other.CachedInput()));
      SetCachedPatches(std::move(other.CachedPatches()));
      SetCachedBatch(other.CachedBatch());
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

    if(h!=Config().ImageHeight()||w!=Config().ImageWidth()||c!=Config().Channels())
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding::Forward: input dimensions mismatch config");
    }

    const uint32_t total_patches=batch*NumPatches();
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    CAIF_DeviceTensor patches=CAIF_DeviceTensor::Uninitialized({total_patches,PatchFlatDim()},
                                                                ctx.Stream(),sdt);

    launch_extract_patches<StorageT>(input.template DevicePtr<StorageT>(),
                                     patches.template DevicePtr<StorageT>(),
                                     static_cast<int>(batch),
                                     static_cast<int>(h),
                                     static_cast<int>(w),
                                     static_cast<int>(c),
                                     static_cast<int>(Config().PatchSize()),
                                     static_cast<int>(NumPatchesH()),
                                     static_cast<int>(NumPatchesW()),
                                     static_cast<int>(PatchFlatDim()),
                                     ctx.Stream().Handle());

    CAIF_DeviceTensor projected=CAIF_DeviceTensor::Uninitialized({total_patches,Config().Dim()},
                                                                  ctx.Stream(),sdt);
    CAIF_Ops::MatMul(patches,WProjMut(),projected,ctx);
    CAIF_Ops::BiasAdd(projected,BProj(),projected);
    projected.Reshape({batch,NumPatches(),Config().Dim()});

    CAIF_DeviceTensor output;
    if(Config().UseCLSToken()==true)
    {
      const uint32_t out_seq=NumPatches()+1;
      output=CAIF_DeviceTensor::Uninitialized({batch,out_seq,Config().Dim()},ctx.Stream(),sdt);
      launch_cls_prepend<StorageT>(projected.template DevicePtr<StorageT>(),
                                   CLSToken().template DevicePtr<StorageT>(),
                                   output.template DevicePtr<StorageT>(),
                                   static_cast<int>(batch),
                                   static_cast<int>(NumPatches()),
                                   static_cast<int>(Config().Dim()),
                                   ctx.Stream().Handle());
    }
    else
    {
      output=std::move(projected);
    }

    if(ctx.Training()==true)
    {
      SetCachedInput(input.Clone());
      SetCachedPatches(std::move(patches));
      SetCachedBatch(batch);
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
    if(CachedBatch()==0)
    {
      THROW_CAIFE("CAIF_DevicePatchEmbedding::Backward: must call Forward with training=true first");
    }

    const uint32_t batch=CachedBatch();
    const uint32_t total_patches=batch*NumPatches();
    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();

    CAIF_DeviceTensor grad_patches;
    if(Config().UseCLSToken()==true)
    {
      GradCls().FillZero();
      grad_patches=CAIF_DeviceTensor::Uninitialized({batch,NumPatches(),Config().Dim()},
                                                     ctx.Stream(),sdt);
      launch_cls_grad_extract<StorageT>(grad_output.template DevicePtr<StorageT>(),
                                        GradCls().template DevicePtr<StorageT>(),
                                        grad_patches.template DevicePtr<StorageT>(),
                                        static_cast<int>(batch),
                                        static_cast<int>(NumPatches()),
                                        static_cast<int>(Config().Dim()),
                                        ctx.Stream().Handle());
    }
    else
    {
      grad_patches=grad_output.Clone();
    }
    grad_patches.Reshape({total_patches,Config().Dim()});

    GradBProj().FillZero();
    CAIF_Ops::BiasGradient(grad_patches,GradBProj());

    CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized({PatchFlatDim(),Config().Dim()},
                                                                     ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeA(CachedPatches(),grad_patches,grad_w_delta,ctx);
    CAIF_Ops::Add(GradWProj(),grad_w_delta,GradWProj());

    CAIF_DeviceTensor grad_flat=CAIF_DeviceTensor::Uninitialized({total_patches,PatchFlatDim()},
                                                                  ctx.Stream(),sdt);
    CAIF_Ops::MatMulTransposeB(grad_patches,WProjMut(),grad_flat,ctx);

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Zeros(
                                  {batch,
                                   Config().ImageHeight(),
                                   Config().ImageWidth(),
                                   Config().Channels()},
                                  ctx.Stream(),sdt);

    launch_extract_patches_backward<StorageT>(grad_flat.template DevicePtr<StorageT>(),
                                              grad_input.template DevicePtr<StorageT>(),
                                              static_cast<int>(batch),
                                              static_cast<int>(Config().ImageHeight()),
                                              static_cast<int>(Config().ImageWidth()),
                                              static_cast<int>(Config().Channels()),
                                              static_cast<int>(Config().PatchSize()),
                                              static_cast<int>(NumPatchesH()),
                                              static_cast<int>(NumPatchesW()),
                                              static_cast<int>(PatchFlatDim()),
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
    GradWProj().FillZero();
    GradBProj().FillZero();
    if(Config().UseCLSToken()==true)
    {
      GradCls().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
    if(index==2&&Config().UseCLSToken()==true)
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
  size_t total=static_cast<size_t>(PatchFlatDim())*Config().Dim();
  total+=Config().Dim();
  if(Config().UseCLSToken()==true)
  {
    total+=Config().Dim();
  }
  return total;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DevicePatchEmbedding<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc=std::string(g_serial_tag_patch_embedding)+
                     g_serial_open_paren+
                     g_serial_kv_patch+
                     std::to_string(Config().PatchSize())+
                     g_serial_comma+
                     g_serial_kv_ch+
                     std::to_string(Config().Channels())+
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
CAIF_DevicePatchEmbedding<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
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
