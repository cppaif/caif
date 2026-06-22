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
#include "caif_serialization_constants.h"
#include "caif_exception.h"
#include "caif_role_registry.h"

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
    SetSublayers(std::move(other.Sublayers()));
    SetTrainable(std::move(other.Trainable()));
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
    Sublayers().push_back(std::move(layer));
    Trainable().push_back(true);
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
    Sublayers()[index]=std::move(layer);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceLayer &CAIF_DeviceContainer::Layer(size_t index)
{
  try
  {
    if(index>=Sublayers().size())
    {
      THROW_CAIFE("DeviceContainer::Layer: index out of range");
    }
    return *Sublayers()[index];
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceLayer &CAIF_DeviceContainer::Layer(size_t index)const
{
  try
  {
    if(index>=Sublayers().size())
    {
      THROW_CAIFE("DeviceContainer::Layer: index out of range");
    }
    return *Sublayers()[index];
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceContainer::SetLayerTrainable(size_t index,bool trainable)
{
  try
  {
    if(index>=Trainable().size())
    {
      THROW_CAIFE("DeviceContainer::SetLayerTrainable: index out of range");
    }
    Trainable()[index]=trainable;
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_DeviceContainer::IsLayerTrainable(size_t index)const
{
  try
  {
    if(index>=Trainable().size())
    {
      THROW_CAIFE("DeviceContainer::IsLayerTrainable: index out of range");
    }
    return Trainable()[index];
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceContainer::ForwardImpl(const CAIF_DeviceTensor &input,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    if(Sublayers().empty()==true)
    {
      return input.Clone();
    }

    CAIF_DeviceTensor current=Sublayers()[0]->Forward(input,ctx);
    for(size_t i=1;i<Sublayers().size();++i)
    {
      current=Sublayers()[i]->Forward(current,ctx);
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
    if(Sublayers().empty()==true)
    {
      return grad_output.Clone();
    }

    CAIF_DeviceTensor current=Sublayers().back()->Backward(grad_output,ctx);
    for(size_t i=Sublayers().size()-1;i>0;--i)
    {
      current=Sublayers()[i-1]->Backward(current,ctx);
    }
    return current;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceContainer::ZeroGradients()
{
  try
  {
    for(size_t i=0;i<Sublayers().size();++i)
    {
      if(Trainable()[i]==true)
      {
        Sublayers()[i]->ZeroGradients();
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
    for(const auto &layer:Sublayers())
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
    for(auto &layer:Sublayers())
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
    for(const auto &layer:Sublayers())
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
    for(const auto &layer:Sublayers())
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
    return std::string(g_serial_tag_container)+
           g_serial_open_paren+
           std::to_string(Sublayers().size())+
           g_serial_suffix_sublayers;
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceContainer::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    for(size_t i=0;i<Sublayers().size();++i)
    {
      const std::string layer_prefix=prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathGenericContainerLayer_e)+std::to_string(i)+".";
      std::vector<std::string> sublayer_names=Sublayers()[i]->ParameterNames(layer_prefix);
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
      const std::string layer_prefix=prefix+CAIF_RoleRegistry::Instance().Name(CAIF_ParamRole::Role_e::PathGenericContainerLayer_e)+std::to_string(i)+".";
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
    for(const auto &layer:Sublayers())
    {
      total+=layer->AuxLoss();
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
