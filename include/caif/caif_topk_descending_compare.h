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
// Top-K descending index comparator. Orders indices so that higher-valued
// entries come first, breaking ties by ascending index for determinism. Used
// by host TopK / MoE gating callers as a named functor (the "no lambdas"
// coding-guideline replacement for an inline comparator lambda).
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace instance
{

class CAIF_TopKDescendingCompare
{
  public:
    explicit CAIF_TopKDescendingCompare(const float *values):_values(values){}

    bool operator()(const uint32_t x,const uint32_t y)const
    {
      if(Values()[x]!=Values()[y])
      {
        return Values()[x]>Values()[y];
      }
      return x<y;
    }

  protected:

  private:
    const float *Values()const{return _values;}

    const float *_values;
};

}//end instance namespace
