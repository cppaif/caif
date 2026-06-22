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

#include "caif_optimizer_type.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

const std::map<CAIF_OptimizerType::CAIF_OptimizerType_e,std::string>
CAIF_OptimizerType::_name_map=
{
  {CAIF_OptimizerType_e::SGD,g_serial_optimizer_sgd},
  {CAIF_OptimizerType_e::Adam,g_serial_optimizer_adam},
  {CAIF_OptimizerType_e::AdaGrad,g_serial_optimizer_adagrad},
  {CAIF_OptimizerType_e::RMSprop,g_serial_optimizer_rmsprop},
  {CAIF_OptimizerType_e::Momentum,g_serial_optimizer_momentum}
};

const std::string &CAIF_OptimizerType::Name(const CAIF_OptimizerType_e v)
{
  try
  {
    const auto it=NameMap().find(v);
    if(it==NameMap().end())
    {
      THROW_CAIFE("CAIF_OptimizerType::Name: unsupported optimizer");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_OptimizerType::CAIF_OptimizerType_e CAIF_OptimizerType::FromName(const std::string &name)
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
    THROW_CAIFE("CAIF_OptimizerType::FromName: unrecognised optimizer name");
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
