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
// Device-resident Vision Transformer implementation
//------------------------------------------------------------------------------
#include "caif_device_vit_model.h"
#include "caif_ops.h"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"
#include "caif_role_registry.h"

#include <cstring>
#include <cmath>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceViTModel<ComputeT,StorageT>::CAIF_DeviceViTModel(const CAIF_DeviceViTModelConfig &config,
                                                             CAIF_CudaStream &stream):
                                                             CAIF_DeviceContainer(stream),
                                                             _config(config),
                                                             _cached_cls_output(),
                                                             _cached_batch(0)
{
  try
  {
    // Validate config
    if(config.ImageHeight()%config.PatchSize()!=0)
    {
      THROW_CAIFE("DeviceViTModel: image_height must be divisible by patch_size");
    }
    if(config.ImageWidth()%config.PatchSize()!=0)
    {
      THROW_CAIFE("DeviceViTModel: image_width must be divisible by patch_size");
    }
    if(config.Dim()==0||config.NumLayers()==0||config.NumHeads()==0)
    {
      THROW_CAIFE("DeviceViTModel: dim, num_layers, num_heads must be > 0");
    }
    if(config.Dim()%config.NumHeads()!=0)
    {
      THROW_CAIFE("DeviceViTModel: dim must be divisible by num_heads");
    }

    // [0] Patch embedding (with CLS token)
    CAIF_DevicePatchEmbeddingConfig patch_config(config.ImageHeight(),
                                                 config.ImageWidth(),
                                                 config.Channels(),
                                                 config.PatchSize(),
                                                 config.Dim(),
                                                 true);
    AddLayer(std::make_unique<CAIF_DevicePatchEmbedding<ComputeT,StorageT>>(patch_config,stream));

    // [1] Positional encoding — ViT uses learnable position embeddings
    const uint32_t num_patches_h=config.ImageHeight()/config.PatchSize();
    const uint32_t num_patches_w=config.ImageWidth()/config.PatchSize();
    const uint32_t seq_len=num_patches_h*num_patches_w+1;  // +1 for CLS

    CAIF_DevicePositionalEncodingConfig pe_config(
      seq_len,
      config.Dim(),
      CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::Learned);
    AddLayer(std::make_unique<CAIF_DevicePositionalEncoding<ComputeT,StorageT>>(pe_config,stream));

    // [2 .. 1+N] Transformer blocks
    for(uint32_t i=0;i<config.NumLayers();++i)
    {
      CAIF_DeviceTransformerBlockConfig block_config(config.Dim(),
                                                     config.NumHeads(),
                                                     config.NumHeads(),
                                                     config.FfnHiddenDim(),
                                                     config.DropoutRate(),
                                                     false,
                                                     config.UseRope(),
                                                     config.RopeBase());
      block_config.SetRopeStyle(config.RopeStyle());
      AddLayer(std::make_unique<CAIF_DeviceTransformerBlock<ComputeT,StorageT>>(block_config,stream));
    }

    // [2+N] Final layer norm
    AddLayer(std::make_unique<CAIF_DeviceLayerNorm<ComputeT,StorageT>>(config.Dim(),
                                                                        stream,
                                                                        g_caif_vit_layernorm_eps));

    // [3+N] Classification head (CLS token -> num_classes)
    CAIF_DeviceLinearHeadConfig head_config(config.Dim(),config.NumClasses(),true);
    AddLayer(std::make_unique<CAIF_DeviceLinearHead<ComputeT,StorageT>>(head_config,stream));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceViTModel<ComputeT,StorageT>::CAIF_DeviceViTModel(CAIF_DeviceViTModel &&other):
                                  CAIF_DeviceContainer(std::move(other)),
                                  _config(other.Config()),
                                  _cached_cls_output(std::move(other.CachedClsOutputMutable())),
                                  _cached_batch(other.CachedBatch())
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceViTModel<ComputeT,StorageT> &
CAIF_DeviceViTModel<ComputeT,StorageT>::operator=(CAIF_DeviceViTModel &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceContainer::operator=(std::move(other));
    SetConfig(other.Config());
    SetCachedClsOutput(std::move(other.CachedClsOutputMutable()));
    SetCachedBatch(other.CachedBatch());
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceViTModel<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                     CAIF_RunContext &ctx)
{
  try
  {
    // Validate input shape [batch, H, W, C]
    const auto &shape=input.Shape();
    if(shape.size()!=4)
    {
      THROW_CAIFE("DeviceViTModel: input must be 4D [batch, H, W, C]");
    }
    if(shape[1]!=Config().ImageHeight()||shape[2]!=Config().ImageWidth())
    {
      THROW_CAIFE("DeviceViTModel: input image size mismatch");
    }
    if(shape[3]!=Config().Channels())
    {
      THROW_CAIFE("DeviceViTModel: input channels mismatch");
    }

    const uint32_t batch=shape[0];
    SetCachedBatch(batch);

    // Step 1: Patch embedding -> [batch, seq_len, dim]
    CAIF_DeviceTensor x=Layer(PatchEmbeddingSlot()).Forward(input,ctx);

    // Step 2: Add positional encoding
    x=Layer(PositionalEncodingSlot()).Forward(x,ctx);

    // Step 3: Pass through transformer blocks
    for(uint32_t i=0;i<Config().NumLayers();++i)
    {
      x=Layer(FirstBlockSlot()+i).Forward(x,ctx);
    }

    // Step 4: Final layer norm
    x=Layer(FinalNormSlot()).Forward(x,ctx);

    // Step 5: Extract CLS token (position 0) -> [batch, dim]
    constexpr CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const uint32_t seq_len=SequenceLength();
    const uint32_t dim=Config().Dim();

    CAIF_DeviceTensor cls_output=CAIF_DeviceTensor::Uninitialized({batch,dim},ctx.Stream(),sd);

    // Pointer arithmetic and memcpy size must follow StorageT, not fp32.
    StorageT *cls_ptr=cls_output.template DevicePtr<StorageT>();
    const StorageT *x_ptr=x.template DevicePtr<StorageT>();
    for(uint32_t b=0;b<batch;++b)
    {
      const size_t src_offset=b*seq_len*dim;
      const size_t dst_offset=b*dim;
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(cls_ptr+dst_offset,
                      x_ptr+src_offset,
                      dim*sizeof(StorageT),
                      cudaMemcpyDeviceToDevice,
                      ctx.Stream().Handle());
#else
      std::memcpy(cls_ptr+dst_offset,
                  x_ptr+src_offset,
                  dim*sizeof(StorageT));
#endif
    }

    if(ctx.Training()==true)
    {
      SetCachedClsOutput(cls_output.Clone());
    }

    // Step 6: Classification head -> [batch, num_classes]
    CAIF_DeviceTensor logits=Layer(ClassificationHeadSlot()).Forward(cls_output,ctx);

    return logits;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceViTModel<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                      CAIF_RunContext &ctx)
{
  try
  {
    constexpr CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const uint32_t batch=CachedBatch();
    const uint32_t seq_len=SequenceLength();
    const uint32_t dim=Config().Dim();

    // Step 1: Backward through classification head
    CAIF_DeviceTensor grad_cls=Layer(ClassificationHeadSlot()).Backward(grad_output,ctx);

    // Step 2: Expand CLS gradient to full sequence ([batch, seq_len, dim])
    CAIF_DeviceTensor grad_seq=CAIF_DeviceTensor::Zeros({batch,seq_len,dim},ctx.Stream(),sd);

    // StorageT-typed pointer arithmetic so the memcpy size and stride
    // match the actual element size (fp16/bf16 = 2 bytes, fp32 = 4 bytes).
    StorageT *gs_ptr=grad_seq.template DevicePtr<StorageT>();
    const StorageT *gc_ptr=grad_cls.template DevicePtr<StorageT>();
    for(uint32_t b=0;b<batch;++b)
    {
      const size_t src_offset=b*dim;
      const size_t dst_offset=b*seq_len*dim;
#ifdef USE_CAIF_CUDA
      cudaMemcpyAsync(gs_ptr+dst_offset,
                      gc_ptr+src_offset,
                      dim*sizeof(StorageT),
                      cudaMemcpyDeviceToDevice,
                      ctx.Stream().Handle());
#else
      std::memcpy(gs_ptr+dst_offset,
                  gc_ptr+src_offset,
                  dim*sizeof(StorageT));
#endif
    }

    // Step 3: Backward through final norm
    CAIF_DeviceTensor grad_x=Layer(FinalNormSlot()).Backward(grad_seq,ctx);

    // Step 4: Backward through transformer blocks (in reverse order)
    for(int i=static_cast<int>(Config().NumLayers())-1;i>=0;--i)
    {
      grad_x=Layer(FirstBlockSlot()+static_cast<size_t>(i)).Backward(grad_x,ctx);
    }

    // Step 5: Backward through positional encoding
    grad_x=Layer(PositionalEncodingSlot()).Backward(grad_x,ctx);

    // Step 6: Backward through patch embedding
    CAIF_DeviceTensor grad_input=Layer(PatchEmbeddingSlot()).Backward(grad_x,ctx);

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceViTModel<ComputeT,StorageT>::Description()const
{
  try
  {
    const uint32_t num_patches=NumPatches();
    return std::string(g_serial_tag_vit)+
           g_serial_open_paren+
           g_serial_kv_img+
           std::to_string(Config().ImageHeight())+
           g_serial_dim_separator+
           std::to_string(Config().ImageWidth())+
           g_serial_comma+
           g_serial_kv_patch+
           std::to_string(Config().PatchSize())+
           g_serial_comma+
           g_serial_kv_patches+
           std::to_string(num_patches)+
           g_serial_comma+
           g_serial_kv_dim+
           std::to_string(Config().Dim())+
           g_serial_comma+
           g_serial_kv_layers+
           std::to_string(Config().NumLayers())+
           g_serial_comma+
           g_serial_kv_heads+
           std::to_string(Config().NumHeads())+
           g_serial_comma+
           g_serial_kv_classes+
           std::to_string(Config().NumClasses())+
           g_serial_close_paren;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceViTModel<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    std::vector<std::string> slot_names;

    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();

    // Patch embedding
    slot_names=Layer(PatchEmbeddingSlot()).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathViTPatchEmbed_e));
    names.insert(names.end(),slot_names.begin(),slot_names.end());

    // Positional encoding
    slot_names=Layer(PositionalEncodingSlot()).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathEmbedPos_e));
    names.insert(names.end(),slot_names.begin(),slot_names.end());

    // Transformer blocks
    for(uint32_t i=0;i<Config().NumLayers();++i)
    {
      std::string block_prefix=prefix+reg.Name(CAIF_ParamRole::Role_e::PathViTBlocks_e)+std::to_string(i)+".";
      slot_names=Layer(FirstBlockSlot()+i).ParameterNames(block_prefix);
      names.insert(names.end(),slot_names.begin(),slot_names.end());
    }

    // Final norm
    slot_names=Layer(FinalNormSlot()).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathFinalNorm_e));
    names.insert(names.end(),slot_names.begin(),slot_names.end());

    // Classification head
    slot_names=Layer(ClassificationHeadSlot()).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathHead_e));
    names.insert(names.end(),slot_names.begin(),slot_names.end());

    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
uint32_t CAIF_DeviceViTModel<ComputeT,StorageT>::NumPatches()const
{
  const uint32_t num_patches_h=Config().ImageHeight()/Config().PatchSize();
  const uint32_t num_patches_w=Config().ImageWidth()/Config().PatchSize();
  return num_patches_h*num_patches_w;
}

template<typename ComputeT,typename StorageT>
uint32_t CAIF_DeviceViTModel<ComputeT,StorageT>::SequenceLength()const
{
  return NumPatches()+1;  // +1 for CLS token
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceViTModel<ComputeT,StorageT>::InitializeWeights(uint32_t seed)
{
  try
  {
    static_cast<void>(seed);  // Each sublayer uses its own default initialization
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePatchEmbedding<ComputeT,StorageT> &
CAIF_DeviceViTModel<ComputeT,StorageT>::PatchEmbedding()
{
  try
  {
    return static_cast<CAIF_DevicePatchEmbedding<ComputeT,StorageT> &>(Layer(PatchEmbeddingSlot()));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePositionalEncoding<ComputeT,StorageT> &
CAIF_DeviceViTModel<ComputeT,StorageT>::PositionalEncoding()
{
  try
  {
    return static_cast<CAIF_DevicePositionalEncoding<ComputeT,StorageT> &>(
              Layer(PositionalEncodingSlot()));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerBlock<ComputeT,StorageT> &
CAIF_DeviceViTModel<ComputeT,StorageT>::TransformerBlock(size_t index)
{
  try
  {
    if(index>=Config().NumLayers())
    {
      THROW_CAIFE("DeviceViTModel: transformer block index out of range");
    }
    return static_cast<CAIF_DeviceTransformerBlock<ComputeT,StorageT> &>(
              Layer(FirstBlockSlot()+index));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLayerNorm<ComputeT,StorageT> &
CAIF_DeviceViTModel<ComputeT,StorageT>::FinalNorm()
{
  try
  {
    return static_cast<CAIF_DeviceLayerNorm<ComputeT,StorageT> &>(Layer(FinalNormSlot()));
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT> &
CAIF_DeviceViTModel<ComputeT,StorageT>::ClassificationHead()
{
  try
  {
    return static_cast<CAIF_DeviceLinearHead<ComputeT,StorageT> &>(
              Layer(ClassificationHeadSlot()));
  }
  CAIF_CATCH_BLOCK()
}

template class CAIF_DeviceViTModel<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceViTModel<float,__half>;
template class CAIF_DeviceViTModel<float,__nv_bfloat16>;
template class CAIF_DeviceViTModel<__half,float>;
template class CAIF_DeviceViTModel<__half,__half>;
template class CAIF_DeviceViTModel<__half,__nv_bfloat16>;
template class CAIF_DeviceViTModel<__nv_bfloat16,float>;
template class CAIF_DeviceViTModel<__nv_bfloat16,__half>;
template class CAIF_DeviceViTModel<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
