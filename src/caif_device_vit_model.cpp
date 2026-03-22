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
// AIF - AI Framework
// Device-resident Vision Transformer implementation
//------------------------------------------------------------------------------
#include "caif_device_vit_model.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <cmath>

namespace instance
{

CAIF_DeviceViTModel::CAIF_DeviceViTModel(const Config_t &config,
                                       CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                       _config(config),
                                       _cached_batch(0)
{
  try
  {
    // Validate config
    if(config.image_height%config.patch_size!=0)
    {
      THROW_CAIFE("DeviceViTModel: image_height must be divisible by patch_size");
    }
    if(config.image_width%config.patch_size!=0)
    {
      THROW_CAIFE("DeviceViTModel: image_width must be divisible by patch_size");
    }
    if(config.dim==0||config.num_layers==0||config.num_heads==0)
    {
      THROW_CAIFE("DeviceViTModel: dim, num_layers, num_heads must be > 0");
    }
    if(config.dim%config.num_heads!=0)
    {
      THROW_CAIFE("DeviceViTModel: dim must be divisible by num_heads");
    }

    // Create patch embedding (with CLS token)
    CAIF_DevicePatchEmbedding::Config_t patch_config;
    patch_config.image_height=config.image_height;
    patch_config.image_width=config.image_width;
    patch_config.channels=config.channels;
    patch_config.patch_size=config.patch_size;
    patch_config.dim=config.dim;
    patch_config.use_cls_token=true;  // ViT uses CLS token
    _patch_embedding=std::make_unique<CAIF_DevicePatchEmbedding>(patch_config,stream);

    // Compute sequence length (num_patches + 1 for CLS)
    const uint32_t num_patches_h=config.image_height/config.patch_size;
    const uint32_t num_patches_w=config.image_width/config.patch_size;
    const uint32_t seq_len=num_patches_h*num_patches_w+1;

    // Create positional encoding
    CAIF_DevicePositionalEncoding::Config_t pe_config;
    pe_config.max_seq_len=seq_len;
    pe_config.dim=config.dim;
    pe_config.mode=PositionalEncodingMode_e::Learned;  // ViT uses learnable position embeddings
    _positional_encoding=std::make_unique<CAIF_DevicePositionalEncoding>(pe_config,stream);

    // Create transformer blocks
    for(uint32_t i=0;i<config.num_layers;++i)
    {
      CAIF_DeviceTransformerBlock::TransformerBlockConfig_t block_config;
      block_config.dim=config.dim;
      block_config.num_heads=config.num_heads;
      block_config.num_kv_heads=config.num_heads;  // Standard MHA for ViT
      block_config.ffn_dim=config.ffn_hidden_dim;
      block_config.causal=false;  // ViT is bidirectional
      block_config.use_rope=config.use_rope;
      block_config.rope_base=config.rope_base;
      block_config.dropout_rate=config.dropout_rate;

      _transformer_blocks.push_back(
        std::make_unique<CAIF_DeviceTransformerBlock>(block_config,stream));
    }

    // Create final layer norm
    constexpr float kLayerNormEps=1e-6f;
    _final_norm=std::make_unique<CAIF_DeviceLayerNorm>(config.dim,stream,kLayerNormEps);

    // Create classification head (CLS token -> num_classes)
    CAIF_DeviceLinearHead::Config_t head_config;
    head_config.input_dim=config.dim;
    head_config.output_dim=config.num_classes;
    head_config.use_bias=true;
    _classification_head=std::make_unique<CAIF_DeviceLinearHead>(head_config,stream);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceViTModel::CAIF_DeviceViTModel(CAIF_DeviceViTModel &&other):CAIF_DeviceLayer(std::move(other)),
                                       _config(other._config),
                                       _patch_embedding(std::move(other._patch_embedding)),
                                       _positional_encoding(std::move(other._positional_encoding)),
                                       _transformer_blocks(std::move(other._transformer_blocks)),
                                       _final_norm(std::move(other._final_norm)),
                                       _classification_head(std::move(other._classification_head)),
                                       _cached_cls_output(std::move(other._cached_cls_output)),
                                       _cached_batch(other._cached_batch)
{
}

CAIF_DeviceViTModel &CAIF_DeviceViTModel::operator=(CAIF_DeviceViTModel &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _patch_embedding=std::move(other._patch_embedding);
    _positional_encoding=std::move(other._positional_encoding);
    _transformer_blocks=std::move(other._transformer_blocks);
    _final_norm=std::move(other._final_norm);
    _classification_head=std::move(other._classification_head);
    _cached_cls_output=std::move(other._cached_cls_output);
    _cached_batch=other._cached_batch;
  }
  return *this;
}

CAIF_DeviceTensor CAIF_DeviceViTModel::Forward(const CAIF_DeviceTensor &input,
                                              bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceViTModel: layer has been moved from");
    }

    // Validate input shape [batch, H, W, C]
    const auto &shape=input.Shape();
    if(shape.size()!=4)
    {
      THROW_CAIFE("DeviceViTModel: input must be 4D [batch, H, W, C]");
    }
    if(shape[1]!=_config.image_height||shape[2]!=_config.image_width)
    {
      THROW_CAIFE("DeviceViTModel: input image size mismatch");
    }
    if(shape[3]!=_config.channels)
    {
      THROW_CAIFE("DeviceViTModel: input channels mismatch");
    }

    const uint32_t batch=shape[0];
    _cached_batch=batch;

    // Step 1: Patch embedding -> [batch, seq_len, dim]
    CAIF_DeviceTensor x=_patch_embedding->Forward(input,training);

    // Step 2: Add positional encoding
    x=_positional_encoding->Forward(x,training);

    // Step 3: Pass through transformer blocks
    for(auto &block:_transformer_blocks)
    {
      x=block->Forward(x,training);
    }

    // Step 4: Final layer norm
    x=_final_norm->Forward(x,training);

    // Step 5: Extract CLS token (position 0) -> [batch, dim]
    // x is [batch, seq_len, dim], we need [batch, 1, dim] then squeeze to [batch, dim]
    const uint32_t seq_len=SequenceLength();
    const uint32_t dim=_config.dim;

    // Copy just the first position (CLS token)
    CAIF_DeviceTensor cls_output=CAIF_DeviceTensor::Uninitialized(
                                  {batch,dim},*_stream);

    // x is contiguous [batch, seq_len, dim], CLS is at positions [b*seq_len*dim]
    for(uint32_t b=0;b<batch;++b)
    {
      const size_t src_offset=b*seq_len*dim;
      const size_t dst_offset=b*dim;
      cudaMemcpyAsync(cls_output.DevicePtr()+dst_offset,
                      x.DevicePtr()+src_offset,
                      dim*sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      _stream->Handle());
    }

    // Cache for backward
    if(training==true)
    {
      _cached_cls_output=cls_output.Clone();
    }

    // Step 6: Classification head -> [batch, num_classes]
    CAIF_DeviceTensor logits=_classification_head->Forward(cls_output,training);

    return logits;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceViTModel::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceViTModel: layer has been moved from");
    }

    const uint32_t batch=_cached_batch;
    const uint32_t seq_len=SequenceLength();
    const uint32_t dim=_config.dim;

    // Step 1: Backward through classification head
    CAIF_DeviceTensor grad_cls=_classification_head->Backward(grad_output);

    // Step 2: Expand CLS gradient to full sequence
    // grad_cls is [batch, dim], need [batch, seq_len, dim] with zeros except at position 0
    CAIF_DeviceTensor grad_seq=CAIF_DeviceTensor::Zeros({batch,seq_len,dim},*_stream);

    for(uint32_t b=0;b<batch;++b)
    {
      const size_t src_offset=b*dim;
      const size_t dst_offset=b*seq_len*dim;
      cudaMemcpyAsync(grad_seq.DevicePtr()+dst_offset,
                      grad_cls.DevicePtr()+src_offset,
                      dim*sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      _stream->Handle());
    }

    // Step 3: Backward through final norm
    CAIF_DeviceTensor grad_x=_final_norm->Backward(grad_seq);

    // Step 4: Backward through transformer blocks (in reverse order)
    for(int i=static_cast<int>(_transformer_blocks.size())-1;i>=0;--i)
    {
      grad_x=_transformer_blocks[static_cast<size_t>(i)]->Backward(grad_x);
    }

    // Step 5: Backward through positional encoding
    grad_x=_positional_encoding->Backward(grad_x);

    // Step 6: Backward through patch embedding
    CAIF_DeviceTensor grad_input=_patch_embedding->Backward(grad_x);

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceViTModel::ZeroGradients()
{
  try
  {
    _patch_embedding->ZeroGradients();
    _positional_encoding->ZeroGradients();
    for(auto &block:_transformer_blocks)
    {
      block->ZeroGradients();
    }
    _final_norm->ZeroGradients();
    _classification_head->ZeroGradients();
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceViTModel::ParameterTensorCount()const
{
  size_t count=0;
  count+=_patch_embedding->ParameterTensorCount();
  count+=_positional_encoding->ParameterTensorCount();
  for(const auto &block:_transformer_blocks)
  {
    count+=block->ParameterTensorCount();
  }
  count+=_final_norm->ParameterTensorCount();
  count+=_classification_head->ParameterTensorCount();
  return count;
}

CAIF_DeviceTensor &CAIF_DeviceViTModel::ParameterTensor(size_t index)
{
  size_t offset=0;

  // Patch embedding
  size_t patch_count=_patch_embedding->ParameterTensorCount();
  if(index<offset+patch_count)
  {
    return _patch_embedding->ParameterTensor(index-offset);
  }
  offset+=patch_count;

  // Positional encoding
  size_t pe_count=_positional_encoding->ParameterTensorCount();
  if(index<offset+pe_count)
  {
    return _positional_encoding->ParameterTensor(index-offset);
  }
  offset+=pe_count;

  // Transformer blocks
  for(auto &block:_transformer_blocks)
  {
    size_t block_count=block->ParameterTensorCount();
    if(index<offset+block_count)
    {
      return block->ParameterTensor(index-offset);
    }
    offset+=block_count;
  }

  // Final norm
  size_t norm_count=_final_norm->ParameterTensorCount();
  if(index<offset+norm_count)
  {
    return _final_norm->ParameterTensor(index-offset);
  }
  offset+=norm_count;

  // Classification head
  return _classification_head->ParameterTensor(index-offset);
}

const CAIF_DeviceTensor &CAIF_DeviceViTModel::ParameterTensor(size_t index)const
{
  size_t offset=0;

  size_t patch_count=_patch_embedding->ParameterTensorCount();
  if(index<offset+patch_count)
  {
    return _patch_embedding->ParameterTensor(index-offset);
  }
  offset+=patch_count;

  size_t pe_count=_positional_encoding->ParameterTensorCount();
  if(index<offset+pe_count)
  {
    return _positional_encoding->ParameterTensor(index-offset);
  }
  offset+=pe_count;

  for(const auto &block:_transformer_blocks)
  {
    size_t block_count=block->ParameterTensorCount();
    if(index<offset+block_count)
    {
      return block->ParameterTensor(index-offset);
    }
    offset+=block_count;
  }

  size_t norm_count=_final_norm->ParameterTensorCount();
  if(index<offset+norm_count)
  {
    return _final_norm->ParameterTensor(index-offset);
  }
  offset+=norm_count;

  return _classification_head->ParameterTensor(index-offset);
}

CAIF_DeviceTensor &CAIF_DeviceViTModel::GradientTensor(size_t index)
{
  size_t offset=0;

  size_t patch_count=_patch_embedding->ParameterTensorCount();
  if(index<offset+patch_count)
  {
    return _patch_embedding->GradientTensor(index-offset);
  }
  offset+=patch_count;

  size_t pe_count=_positional_encoding->ParameterTensorCount();
  if(index<offset+pe_count)
  {
    return _positional_encoding->GradientTensor(index-offset);
  }
  offset+=pe_count;

  for(auto &block:_transformer_blocks)
  {
    size_t block_count=block->ParameterTensorCount();
    if(index<offset+block_count)
    {
      return block->GradientTensor(index-offset);
    }
    offset+=block_count;
  }

  size_t norm_count=_final_norm->ParameterTensorCount();
  if(index<offset+norm_count)
  {
    return _final_norm->GradientTensor(index-offset);
  }
  offset+=norm_count;

  return _classification_head->GradientTensor(index-offset);
}

const CAIF_DeviceTensor &CAIF_DeviceViTModel::GradientTensor(size_t index)const
{
  size_t offset=0;

  size_t patch_count=_patch_embedding->ParameterTensorCount();
  if(index<offset+patch_count)
  {
    return _patch_embedding->GradientTensor(index-offset);
  }
  offset+=patch_count;

  size_t pe_count=_positional_encoding->ParameterTensorCount();
  if(index<offset+pe_count)
  {
    return _positional_encoding->GradientTensor(index-offset);
  }
  offset+=pe_count;

  for(const auto &block:_transformer_blocks)
  {
    size_t block_count=block->ParameterTensorCount();
    if(index<offset+block_count)
    {
      return block->GradientTensor(index-offset);
    }
    offset+=block_count;
  }

  size_t norm_count=_final_norm->ParameterTensorCount();
  if(index<offset+norm_count)
  {
    return _final_norm->GradientTensor(index-offset);
  }
  offset+=norm_count;

  return _classification_head->GradientTensor(index-offset);
}

size_t CAIF_DeviceViTModel::TotalParameterCount()const
{
  size_t total=0;
  total+=_patch_embedding->TotalParameterCount();
  total+=_positional_encoding->TotalParameterCount();
  for(const auto &block:_transformer_blocks)
  {
    total+=block->TotalParameterCount();
  }
  total+=_final_norm->TotalParameterCount();
  total+=_classification_head->TotalParameterCount();
  return total;
}

std::string CAIF_DeviceViTModel::Description()const
{
  try
  {
    const uint32_t num_patches=NumPatches();
    return "ViT(img="+std::to_string(_config.image_height)+"x"+
           std::to_string(_config.image_width)+
           ",patch="+std::to_string(_config.patch_size)+
           ",patches="+std::to_string(num_patches)+
           ",dim="+std::to_string(_config.dim)+
           ",layers="+std::to_string(_config.num_layers)+
           ",heads="+std::to_string(_config.num_heads)+
           ",classes="+std::to_string(_config.num_classes)+")";
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceViTModel::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;

    // Patch embedding
    auto patch_names=_patch_embedding->ParameterNames(prefix+"embeddings.patch_embeddings.");
    names.insert(names.end(),patch_names.begin(),patch_names.end());

    // Positional encoding
    auto pe_names=_positional_encoding->ParameterNames(prefix+"embeddings.position_embeddings.");
    names.insert(names.end(),pe_names.begin(),pe_names.end());

    // Transformer blocks
    for(size_t i=0;i<_transformer_blocks.size();++i)
    {
      std::string block_prefix=prefix+"encoder.layer."+std::to_string(i)+".";
      auto block_names=_transformer_blocks[i]->ParameterNames(block_prefix);
      names.insert(names.end(),block_names.begin(),block_names.end());
    }

    // Final norm
    auto norm_names=_final_norm->ParameterNames(prefix+"layernorm.");
    names.insert(names.end(),norm_names.begin(),norm_names.end());

    // Classification head
    auto head_names=_classification_head->ParameterNames(prefix+"classifier.");
    names.insert(names.end(),head_names.begin(),head_names.end());

    return names;
  }
  CCAIF_CATCH_BLOCK()
}

uint32_t CAIF_DeviceViTModel::NumPatches()const
{
  const uint32_t num_patches_h=_config.image_height/_config.patch_size;
  const uint32_t num_patches_w=_config.image_width/_config.patch_size;
  return num_patches_h*num_patches_w;
}

uint32_t CAIF_DeviceViTModel::SequenceLength()const
{
  return NumPatches()+1;  // +1 for CLS token
}

void CAIF_DeviceViTModel::InitializeWeights(uint32_t seed)
{
  try
  {
    // Each sublayer initializes its own weights
    // For reproducibility, we could pass different seeds to each
    (void)seed;  // Currently each sublayer uses its default initialization
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTransformerBlock &CAIF_DeviceViTModel::TransformerBlock(size_t index)
{
  if(index>=_transformer_blocks.size())
  {
    THROW_CAIFE("DeviceViTModel: transformer block index out of range");
  }
  return *_transformer_blocks[index];
}

}//end instance namespace
