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

#include "caif_device_container.h"
#include "caif_constants.h"
#include "caif_exception.h"

namespace instance
{

CAIF_DeviceContainer::CAIF_DeviceContainer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                                    _sublayers(),
                                                                    _trainable()
{
}

CAIF_DeviceContainer::CAIF_DeviceContainer(CAIF_DeviceContainer &&other):
                                          CAIF_DeviceLayer(std::move(other)),
                                          _sublayers(std::move(other._sublayers)),
                                          _trainable(std::move(other._trainable))
{
}

CAIF_DeviceContainer &CAIF_DeviceContainer::operator=(CAIF_DeviceContainer &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _sublayers=std::move(other._sublayers);
    _trainable=std::move(other._trainable);
  }
  return *this;
}

void CAIF_DeviceContainer::AddLayer(std::unique_ptr<CAIF_DeviceLayer> layer)
{
  try
  {
    if(layer==nullptr)
    {
      THROW_CAIFE("DeviceContainer::AddLayer: layer is null");
    }
    _sublayers.push_back(std::move(layer));
    _trainable.push_back(true);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceContainer::ReplaceLayer(size_t index,std::unique_ptr<CAIF_DeviceLayer> layer)
{
  try
  {
    if(layer==nullptr)
    {
      THROW_CAIFE("DeviceContainer::ReplaceLayer: layer is null");
    }
    if(index>=LayerCount())
    {
      THROW_CAIFE("DeviceContainer::ReplaceLayer: index out of range");
    }
    _sublayers[index]=std::move(layer);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceLayer &CAIF_DeviceContainer::Layer(size_t index)
{
  try
  {
    if(index>=_sublayers.size())
    {
      THROW_CAIFE("DeviceContainer::Layer: index out of range");
    }
    return *_sublayers[index];
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceLayer &CAIF_DeviceContainer::Layer(size_t index)const
{
  try
  {
    if(index>=_sublayers.size())
    {
      THROW_CAIFE("DeviceContainer::Layer: index out of range");
    }
    return *_sublayers[index];
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceContainer::SetLayerTrainable(size_t index,bool trainable)
{
  try
  {
    if(index>=_trainable.size())
    {
      THROW_CAIFE("DeviceContainer::SetLayerTrainable: index out of range");
    }
    _trainable[index]=trainable;
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_DeviceContainer::IsLayerTrainable(size_t index)const
{
  try
  {
    if(index>=_trainable.size())
    {
      THROW_CAIFE("DeviceContainer::IsLayerTrainable: index out of range");
    }
    return _trainable[index];
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceContainer::ForwardImpl(const CAIF_DeviceTensor &input,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    if(_sublayers.empty()==true)
    {
      return input.Clone();
    }

    CAIF_DeviceTensor current=_sublayers[0]->Forward(input,ctx);
    for(size_t i=1;i<_sublayers.size();++i)
    {
      current=_sublayers[i]->Forward(current,ctx);
    }
    return current;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceContainer::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                     CAIF_RunContext &ctx)
{
  try
  {
    if(_sublayers.empty()==true)
    {
      return grad_output.Clone();
    }

    CAIF_DeviceTensor current=_sublayers.back()->Backward(grad_output,ctx);
    for(size_t i=_sublayers.size()-1;i>0;--i)
    {
      current=_sublayers[i-1]->Backward(current,ctx);
    }
    return current;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceContainer::ZeroGradients()
{
  try
  {
    for(size_t i=0;i<_sublayers.size();++i)
    {
      if(_trainable[i]==true)
      {
        _sublayers[i]->ZeroGradients();
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceContainer::ParameterTensorCount()const
{
  try
  {
    size_t total=0;
    for(const auto &layer:_sublayers)
    {
      total+=layer->ParameterTensorCount();
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

std::pair<CAIF_DeviceLayer *,size_t>
CAIF_DeviceContainer::ResolveParameterIndex(size_t index)
{
  try
  {
    size_t offset=0;
    for(auto &layer:_sublayers)
    {
      const size_t count=layer->ParameterTensorCount();
      if(index<offset+count)
      {
        return std::make_pair(layer.get(),index-offset);
      }
      offset+=count;
    }
    THROW_CAIFE("DeviceContainer::ResolveParameterIndex: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

std::pair<const CAIF_DeviceLayer *,size_t>
CAIF_DeviceContainer::ResolveParameterIndex(size_t index)const
{
  try
  {
    size_t offset=0;
    for(const auto &layer:_sublayers)
    {
      const size_t count=layer->ParameterTensorCount();
      if(index<offset+count)
      {
        return std::make_pair(layer.get(),index-offset);
      }
      offset+=count;
    }
    THROW_CAIFE("DeviceContainer::ResolveParameterIndex: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceContainer::ParameterTensor(size_t index)
{
  try
  {
    auto resolved=ResolveParameterIndex(index);
    return resolved.first->ParameterTensor(resolved.second);
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceContainer::ParameterTensor(size_t index)const
{
  try
  {
    auto resolved=ResolveParameterIndex(index);
    return resolved.first->ParameterTensor(resolved.second);
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_DeviceContainer::IsParameterTrainable(size_t index)const
{
  try
  {
    size_t offset=0;
    const size_t n=LayerCount();
    for(size_t i=0;i<n;++i)
    {
      const CAIF_DeviceLayer &child=Layer(i);
      const size_t count=child.ParameterTensorCount();
      if(index<offset+count)
      {
        if(IsLayerTrainable(i)==false)
        {
          return false;
        }
        return child.IsParameterTrainable(index-offset);
      }
      offset+=count;
    }
    THROW_CAIFE("DeviceContainer::IsParameterTrainable: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceContainer::GradientTensor(size_t index)
{
  try
  {
    auto resolved=ResolveParameterIndex(index);
    return resolved.first->GradientTensor(resolved.second);
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceContainer::GradientTensor(size_t index)const
{
  try
  {
    auto resolved=ResolveParameterIndex(index);
    return resolved.first->GradientTensor(resolved.second);
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceContainer::TotalParameterCount()const
{
  try
  {
    size_t total=0;
    for(const auto &layer:_sublayers)
    {
      total+=layer->TotalParameterCount();
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceContainer::Description()const
{
  try
  {
    return std::string(g_caif_description_container_prefix)+
           "("+std::to_string(_sublayers.size())+" sublayers)";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceContainer::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    for(size_t i=0;i<_sublayers.size();++i)
    {
      const std::string layer_prefix=prefix+"layers."+std::to_string(i)+".";
      std::vector<std::string> sublayer_names=_sublayers[i]->ParameterNames(layer_prefix);
      for(auto &name:sublayer_names)
      {
        names.push_back(std::move(name));
      }
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceContainer::FrozenTensorCount()const
{
  try
  {
    size_t total=0;
    const size_t n=LayerCount();
    for(size_t i=0;i<n;++i)
    {
      total+=Layer(i).FrozenTensorCount();
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceContainer::FrozenTensorFP32(size_t index)const
{
  try
  {
    size_t offset=0;
    const size_t n=LayerCount();
    for(size_t i=0;i<n;++i)
    {
      const CAIF_DeviceLayer &layer=Layer(i);
      const size_t count=layer.FrozenTensorCount();
      if(index<offset+count)
      {
        return layer.FrozenTensorFP32(index-offset);
      }
      offset+=count;
    }
    THROW_CAIFE("DeviceContainer::FrozenTensorFP32: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceContainer::FrozenTensorNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    const size_t n=LayerCount();
    for(size_t i=0;i<n;++i)
    {
      const std::string layer_prefix=prefix+"layers."+std::to_string(i)+".";
      std::vector<std::string> sub=Layer(i).FrozenTensorNames(layer_prefix);
      for(auto &nm:sub)
      {
        names.push_back(std::move(nm));
      }
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

float CAIF_DeviceContainer::AuxLoss()const
{
  try
  {
    float total=0.0f;
    for(const auto &layer:_sublayers)
    {
      total+=layer->AuxLoss();
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
