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
// Host-side dropout RNG primitives. Pure-function helpers used by the host
// dropout fallback (the device path uses curand on-GPU). SplitMix64 is the
// per-counter scrambler, UniformFromBits converts the scrambled 64-bit word
// into a uniform float in [0, 1).
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include <cstdint>

namespace instance
{

class CAIF_DropoutRng:public CAIF_Base
{
  public:
    static uint64_t SplitMix64(const uint64_t x);
    static float UniformFromBits(const uint64_t bits);

  protected:

  private:
    CAIF_DropoutRng()=delete;
};

}//end instance namespace
