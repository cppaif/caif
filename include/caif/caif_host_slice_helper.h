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
// Host-side slice / concat dimension helpers. Collapses every leading axis
// of a tensor into a single "rows" count while preserving the last
// dimension as "cols", so a host loop can iterate row-by-row regardless of
// the tensor's actual rank.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"

#include <cstddef>

namespace instance
{

class CAIF_HostSliceHelper:public CAIF_Base
{
  public:
    static void SliceDimsForLastDim(const CAIF_DeviceTensor &t,
                                    size_t &rows,
                                    size_t &cols);

  protected:

  private:
    CAIF_HostSliceHelper()=delete;
};

}//end instance namespace
