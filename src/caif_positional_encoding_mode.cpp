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

#include "caif_positional_encoding_mode.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

const std::map<CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e,std::string>
CAIF_PositionalEncodingMode::_name_map=
{
  {CAIF_PositionalEncodingMode_e::Learned,g_serial_pe_mode_learned},
  {CAIF_PositionalEncodingMode_e::Sinusoidal,g_serial_pe_mode_sinusoidal},
  {CAIF_PositionalEncodingMode_e::None,g_serial_pe_mode_none}
};

const std::string &CAIF_PositionalEncodingMode::Name(const CAIF_PositionalEncodingMode_e v)
{
  try
  {
    const auto it=NameMap().find(v);
    if(it==NameMap().end())
    {
      THROW_CAIFE("CAIF_PositionalEncodingMode::Name: unsupported mode");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e
CAIF_PositionalEncodingMode::FromName(const std::string &name)
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
    THROW_CAIFE("CAIF_PositionalEncodingMode::FromName: unrecognised mode name");
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
