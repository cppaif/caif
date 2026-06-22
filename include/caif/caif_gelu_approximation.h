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
// CAIF_GELUApproximation — closed-value enumeration of the two GELU formulas
// plus the bidirectional kind <-> name mapping. Mirrors CAIF_ActivationType.
//   Tanh  : f(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))  (the
//           "gelu_new" / "gelu_pytorch_tanh" approximation; CAIF's historical
//           default, so every existing call site keeps its numerics).
//   Exact : f(x) = 0.5*x*(1 + erf(x/sqrt(2)))                       (PyTorch
//           nn.GELU() default; BERT / GPT-2 original).
// Strings come from g_serial_gelu_approx_* in caif_serialization_constants.h
// (one source of truth for the vocabulary).
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "caif_base.h"

namespace instance
{

class CAIF_GELUApproximation:public CAIF_Base
{
  public:
    enum class CAIF_GELUApproximation_e:uint8_t
    {
      Tanh,
      Exact
    };

    static const std::string &Name(const CAIF_GELUApproximation_e v);
    static CAIF_GELUApproximation_e FromName(const std::string &name);

    static const std::map<CAIF_GELUApproximation_e,std::string> &NameMap(){return _name_map;}

  protected:

  private:
    CAIF_GELUApproximation()=delete;

    static const std::map<CAIF_GELUApproximation_e,std::string> _name_map;
};

}//end instance namespace
