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

#include "caif_host_slice_helper.h"
#include "caif_exception.h"

namespace instance
{

void CAIF_HostSliceHelper::SliceDimsForLastDim(const CAIF_DeviceTensor &t,
                                               size_t &rows,
                                               size_t &cols)
{
  try
  {
    const auto &s=t.Shape();
    if(s.size()==0)
    {
      THROW_CAIFE("CAIF_HostSliceHelper::SliceDimsForLastDim: tensor has no dimensions");
    }
    cols=s.back();
    rows=1;
    for(size_t i=0;i+1<s.size();++i)
    {
      rows*=s[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
