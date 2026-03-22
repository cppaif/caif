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
// Generic pre-norm residual block layer implementation
//------------------------------------------------------------------------------
#include "caif_device_pre_norm_block.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <sstream>

using namespace instance;

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

CAIF_DevicePreNormBlock::CAIF_DevicePreNormBlock(SubLayerVec_t sub_layers,
                                               CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                                        _sub_layers(std::move(sub_layers))
{
  try
  {
    if(_sub_layers.empty()==true)
    {
      THROW_CAIFE("PreNormBlock: must have at least one sub-layer");
    }

    for(size_t i=0;i<_sub_layers.size();++i)
    {
      if(_sub_layers[i].layer==nullptr)
      {
        THROW_CAIFE("PreNormBlock: sub-layer "
                    +std::to_string(i)+" has null layer");
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Move semantics
//------------------------------------------------------------------------------

CAIF_DevicePreNormBlock::CAIF_DevicePreNormBlock(
  CAIF_DevicePreNormBlock &&other):CAIF_DeviceLayer(std::move(other)),
                                  _sub_layers(std::move(other._sub_layers))
{
}

CAIF_DevicePreNormBlock &CAIF_DevicePreNormBlock::operator=(CAIF_DevicePreNormBlock &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _sub_layers=std::move(other._sub_layers);
  }
  return *this;
}

//------------------------------------------------------------------------------
// Forward pass
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_DevicePreNormBlock::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    CAIF_DeviceTensor x=input.Clone();

    for(size_t i=0;i<_sub_layers.size();++i)
    {
      CAIF_DeviceTensor normed;
      if(_sub_layers[i].norm!=nullptr)
      {
        normed=_sub_layers[i].norm->Forward(x,training);
      }
      else
      {
        normed=x.Clone();
      }

      CAIF_DeviceTensor sub_out=_sub_layers[i].layer->Forward(normed,training);

      CAIF_DeviceTensor residual=CAIF_DeviceTensor::Uninitialized(x.Shape(),*_stream);
      CAIF_DeviceOps::Add(x,sub_out,residual);
      x=std::move(residual);
    }

    return x;
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Backward pass
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_DevicePreNormBlock::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    CAIF_DeviceTensor grad=grad_output.Clone();

    for(size_t ri=0;ri<_sub_layers.size();++ri)
    {
      const size_t i=_sub_layers.size()-1-ri;

      CAIF_DeviceTensor d_sub=_sub_layers[i].layer->Backward(grad);

      CAIF_DeviceTensor d_normed;
      if(_sub_layers[i].norm!=nullptr)
      {
        d_normed=_sub_layers[i].norm->Backward(d_sub);
      }
      else
      {
        d_normed=std::move(d_sub);
      }

      CAIF_DeviceTensor d_residual=CAIF_DeviceTensor::Uninitialized(grad.Shape(),*_stream);
      CAIF_DeviceOps::Add(grad,d_normed,d_residual);
      grad=std::move(d_residual);
    }

    return grad;
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Parameter management
//------------------------------------------------------------------------------

void CAIF_DevicePreNormBlock::ZeroGradients()
{
  try
  {
    for(size_t i=0;i<_sub_layers.size();++i)
    {
      if(_sub_layers[i].norm!=nullptr)
      {
        _sub_layers[i].norm->ZeroGradients();
      }
      _sub_layers[i].layer->ZeroGradients();
    }
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DevicePreNormBlock::ParameterTensorCount()const
{
  size_t total=0;
  for(size_t i=0;i<_sub_layers.size();++i)
  {
    if(_sub_layers[i].norm!=nullptr)
    {
      total+=_sub_layers[i].norm->ParameterTensorCount();
    }
    total+=_sub_layers[i].layer->ParameterTensorCount();
  }
  return total;
}

CAIF_DevicePreNormBlock::SubLayerMapping_t CAIF_DevicePreNormBlock::MapIndex(size_t index)const
{
  size_t remaining=index;
  for(size_t i=0;i<_sub_layers.size();++i)
  {
    if(_sub_layers[i].norm!=nullptr)
    {
      const size_t norm_count=_sub_layers[i].norm->ParameterTensorCount();
      if(remaining<norm_count)
      {
        SubLayerMapping_t mapping;
        mapping.stage_idx=i;
        mapping.is_norm=true;
        mapping.local_idx=remaining;
        return mapping;
      }
      remaining-=norm_count;
    }

    const size_t layer_count=_sub_layers[i].layer->ParameterTensorCount();
    if(remaining<layer_count)
    {
      SubLayerMapping_t mapping;
      mapping.stage_idx=i;
      mapping.is_norm=false;
      mapping.local_idx=remaining;
      return mapping;
    }
    remaining-=layer_count;
  }

  THROW_CAIFE("PreNormBlock::MapIndex: index out of range");
}

CAIF_DeviceLayer &CAIF_DevicePreNormBlock::LayerByMapping(const SubLayerMapping_t &mapping)
{
  if(mapping.is_norm==true)
  {
    return *_sub_layers[mapping.stage_idx].norm;
  }
  return *_sub_layers[mapping.stage_idx].layer;
}

const CAIF_DeviceLayer &CAIF_DevicePreNormBlock::LayerByMapping(const SubLayerMapping_t &mapping)const
{
  if(mapping.is_norm==true)
  {
    return *_sub_layers[mapping.stage_idx].norm;
  }
  return *_sub_layers[mapping.stage_idx].layer;
}

CAIF_DeviceTensor &CAIF_DevicePreNormBlock::ParameterTensor(size_t index)
{
  try
  {
    const auto mapping=MapIndex(index);
    return LayerByMapping(mapping).ParameterTensor(mapping.local_idx);
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DevicePreNormBlock::ParameterTensor(size_t index)const
{
  try
  {
    const auto mapping=MapIndex(index);
    return LayerByMapping(mapping).ParameterTensor(mapping.local_idx);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DevicePreNormBlock::GradientTensor(size_t index)
{
  try
  {
    const auto mapping=MapIndex(index);
    return LayerByMapping(mapping).GradientTensor(mapping.local_idx);
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DevicePreNormBlock::GradientTensor(size_t index)const
{
  try
  {
    const auto mapping=MapIndex(index);
    return LayerByMapping(mapping).GradientTensor(mapping.local_idx);
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DevicePreNormBlock::TotalParameterCount()const
{
  size_t total=0;
  for(size_t i=0;i<_sub_layers.size();++i)
  {
    if(_sub_layers[i].norm!=nullptr)
    {
      total+=_sub_layers[i].norm->TotalParameterCount();
    }
    total+=_sub_layers[i].layer->TotalParameterCount();
  }
  return total;
}

std::string CAIF_DevicePreNormBlock::Description()const
{
  std::ostringstream ss;
  ss<<"PreNormBlock(stages="
    <<_sub_layers.size()
    <<",params="
    <<TotalParameterCount()
    <<")";
  return ss.str();
}

std::vector<std::string> CAIF_DevicePreNormBlock::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    for(size_t i=0;i<_sub_layers.size();++i)
    {
      if(_sub_layers[i].norm!=nullptr)
      {
        auto norm_names=_sub_layers[i].norm->ParameterNames(prefix+_sub_layers[i].norm_prefix);
        names.insert(names.end(),norm_names.begin(),norm_names.end());
      }

      auto layer_names=_sub_layers[i].layer->ParameterNames(prefix+_sub_layers[i].layer_prefix);
      names.insert(names.end(),layer_names.begin(),layer_names.end());
    }
    return names;
  }
  CCAIF_CATCH_BLOCK()
}
