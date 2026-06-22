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

#include "caif_activation_type.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

const std::map<CAIF_ActivationType::CAIF_ActivationType_e,std::string>
CAIF_ActivationType::_name_map=
{
  {CAIF_ActivationType_e::Linear,g_serial_activation_linear},
  {CAIF_ActivationType_e::ReLU,g_serial_activation_relu},
  {CAIF_ActivationType_e::Sigmoid,g_serial_activation_sigmoid},
  {CAIF_ActivationType_e::Tanh,g_serial_activation_tanh},
  {CAIF_ActivationType_e::Softmax,g_serial_activation_softmax},
  {CAIF_ActivationType_e::LeakyReLU,g_serial_activation_leakyrelu},
  {CAIF_ActivationType_e::ELU,g_serial_activation_elu},
  {CAIF_ActivationType_e::GELU,g_serial_activation_gelu},
  {CAIF_ActivationType_e::Swish,g_serial_activation_swish}
};

const std::string &CAIF_ActivationType::Name(const CAIF_ActivationType_e v)
{
  try
  {
    const auto it=NameMap().find(v);
    if(it==NameMap().end())
    {
      THROW_CAIFE("CAIF_ActivationType::Name: unsupported activation");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_ActivationType::CAIF_ActivationType_e CAIF_ActivationType::FromName(const std::string &name)
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
    THROW_CAIFE("CAIF_ActivationType::FromName: unrecognised activation name");
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
