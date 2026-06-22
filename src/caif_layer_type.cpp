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

#include "caif_layer_type.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

const std::map<CAIF_LayerType::CAIF_LayerType_e,std::string>
CAIF_LayerType::_name_map=
{
  {CAIF_LayerType_e::Embedding,g_serial_layer_type_embedding},
  {CAIF_LayerType_e::Dense,g_serial_layer_type_dense},
  {CAIF_LayerType_e::Convolution2D,g_serial_layer_type_conv2d},
  {CAIF_LayerType_e::MaxPooling2D,g_serial_layer_type_maxpool2d},
  {CAIF_LayerType_e::AveragePooling2D,g_serial_layer_type_avgpool2d},
  {CAIF_LayerType_e::BatchNormalization,g_serial_layer_type_batchnorm},
  {CAIF_LayerType_e::Dropout,g_serial_layer_type_dropout},
  {CAIF_LayerType_e::Flatten,g_serial_layer_type_flatten},
  {CAIF_LayerType_e::Reshape,g_serial_layer_type_reshape},
  {CAIF_LayerType_e::MultiHeadAttention,g_serial_layer_type_multi_head_attention},
  {CAIF_LayerType_e::TransformerEncoder,g_serial_layer_type_transformer_encoder}
};

const std::string &CAIF_LayerType::Name(const CAIF_LayerType_e v)
{
  try
  {
    const auto it=NameMap().find(v);
    if(it==NameMap().end())
    {
      THROW_CAIFE("CAIF_LayerType::Name: unsupported layer type");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_LayerType::CAIF_LayerType_e CAIF_LayerType::FromName(const std::string &name)
{
  try
  {
    for(const auto &kv:NameMap())
    {
      if(kv.second==name)
      {
        return kv.first;
      }
    }
    THROW_CAIFE("CAIF_LayerType::FromName: unrecognised layer type name");
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
