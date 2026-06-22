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
// Host-side bit-level FP16 / BF16 <-> FP32 conversion plus tensor-level
// up-cast / down-cast helpers. Lives in its own class so the host-only
// ops backend (and host RAII float views) can route every dtype conversion
// through a single, testable surface instead of duplicating the bit-level
// arithmetic at every callsite.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

class CAIF_HostFpCast:public CAIF_Base
{
  public:
    static float Fp16ToFloat(const uint16_t h);
    static uint16_t FloatToFp16(const float f);
    static float Bf16ToFloat(const uint16_t b);
    static uint16_t FloatToBf16(const float f);

    static std::vector<float> UpcastToFloat(const CAIF_DeviceTensor &t);
    static void DowncastFromFloat(const std::vector<float> &src,
                                  CAIF_DeviceTensor &out);

    static float *HostFp32(CAIF_DeviceTensor &t,const std::string &op);
    static const float *HostFp32(const CAIF_DeviceTensor &t,const std::string &op);

  protected:

  private:
    CAIF_HostFpCast()=delete;
};

}//end instance namespace
