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

#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"
#include <cstdint>

namespace instance
{

class CAIF_SysDebug:public CAIF_Base
{
  public:

    // Enable/disable debug output globally
    static void SetEnabled(bool enabled);
    static bool IsEnabled();

    // Check tensor for NaN/Inf values and optionally print summary
    // Returns true if tensor contains NaN or Inf values
    static bool CheckTensor(const std::string &label,
                            const CAIF_DeviceTensor &tensor,
                            bool print_summary=true);

    // Print raw memory values from device tensor (first N elements)
    static void PrintRawValues(const std::string &label,
                               const CAIF_DeviceTensor &tensor,
                               uint32_t count=16);

    // Print a debug message (only if enabled)
    static void DebugLog(const std::string &message);

  protected:

  private:

    static bool _enabled;
};

}//end instance namespace
