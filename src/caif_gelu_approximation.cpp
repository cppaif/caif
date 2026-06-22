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

#include "caif_gelu_approximation.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

const std::map<CAIF_GELUApproximation::CAIF_GELUApproximation_e,std::string>
CAIF_GELUApproximation::_name_map=
{
  {CAIF_GELUApproximation_e::Tanh,g_serial_gelu_approx_tanh},
  {CAIF_GELUApproximation_e::Exact,g_serial_gelu_approx_exact}
};

const std::string &CAIF_GELUApproximation::Name(const CAIF_GELUApproximation_e v)
{
  try
  {
    const auto it=NameMap().find(v);
    if(it==NameMap().end())
    {
      THROW_CAIFE("CAIF_GELUApproximation::Name: unsupported approximation");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_GELUApproximation::CAIF_GELUApproximation_e
CAIF_GELUApproximation::FromName(const std::string &name)
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
    THROW_CAIFE("CAIF_GELUApproximation::FromName: unknown approximation name");
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
