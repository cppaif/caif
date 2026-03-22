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

/**
 * @file aif_activation_aware.h
 * @brief Helper for activation-aware initialization bounds.
 */

#pragma once

#include "caif_constants.h"

namespace instance
{
  class CAIF_ActivationAware
  {
    public:
      static bool UsesKaiming(const CAIF_ActivationType_e activation);

      static float Gain(const CAIF_ActivationType_e activation);

      static float UniformBound(
                                const CAIF_ActivationType_e activation,
                                const uint32_t fan_in,
                                const uint32_t fan_out,
                                const bool activation_aware_enabled
                               );
  };
}//end instance namespace


