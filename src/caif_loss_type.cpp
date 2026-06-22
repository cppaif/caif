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

#include "caif_loss_type.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

const std::map<CAIF_LossType::CAIF_LossType_e,std::string>
CAIF_LossType::_name_map=
{
  {CAIF_LossType_e::MeanSquaredError,g_serial_loss_mse},
  {CAIF_LossType_e::CrossEntropy,g_serial_loss_cross_entropy},
  {CAIF_LossType_e::BinaryCrossEntropy,g_serial_loss_binary_crossentropy},
  {CAIF_LossType_e::BinaryCrossEntropyWithLogits,g_serial_loss_binary_crossentropy_logits},
  {CAIF_LossType_e::CategoricalCrossEntropy,g_serial_loss_categorical_crossentropy},
  {CAIF_LossType_e::Huber,g_serial_loss_huber},
  {CAIF_LossType_e::MeanAbsoluteError,g_serial_loss_mae}
};

const std::string &CAIF_LossType::Name(const CAIF_LossType_e v)
{
  try
  {
    const auto it=NameMap().find(v);
    if(it==NameMap().end())
    {
      THROW_CAIFE("CAIF_LossType::Name: unsupported loss");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_LossType::CAIF_LossType_e CAIF_LossType::FromName(const std::string &name)
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
    THROW_CAIFE("CAIF_LossType::FromName: unrecognised loss name");
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
