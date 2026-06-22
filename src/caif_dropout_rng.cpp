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

#include "caif_dropout_rng.h"
#include "caif_exception.h"

namespace instance
{

uint64_t CAIF_DropoutRng::SplitMix64(const uint64_t x)
{
  try
  {
    uint64_t z=(x+0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
  }
  CAIF_CATCH_BLOCK();
}

float CAIF_DropoutRng::UniformFromBits(const uint64_t bits)
{
  try
  {
    const uint64_t mantissa=(bits>>11)&((1ULL<<53)-1ULL);
    const double unit_double=static_cast<double>(mantissa)/static_cast<double>(1ULL<<53);
    return static_cast<float>(unit_double);
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
