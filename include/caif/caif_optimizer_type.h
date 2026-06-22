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
// CAIF_OptimizerType — closed-value enumeration of optimizer kinds plus the
// bidirectional kind ↔ name mapping used by SafeTensors metadata and any
// serializer that needs to round-trip an optimizer kind through a string
// form. Strings come from g_serial_optimizer_* in
// caif_serialization_constants.h.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "caif_base.h"

namespace instance
{

class CAIF_OptimizerType:public CAIF_Base
{
  public:
    enum class CAIF_OptimizerType_e:uint8_t
    {
      SGD,
      Adam,
      AdaGrad,
      RMSprop,
      Momentum
    };

    static const std::string &Name(const CAIF_OptimizerType_e v);
    static CAIF_OptimizerType_e FromName(const std::string &name);

    static const std::map<CAIF_OptimizerType_e,std::string> &NameMap(){return _name_map;}

  protected:

  private:
    CAIF_OptimizerType()=delete;

    static const std::map<CAIF_OptimizerType_e,std::string> _name_map;
};

}//end instance namespace
