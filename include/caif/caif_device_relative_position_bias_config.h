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
// Configuration for CAIF_DeviceRelativePositionBias. All four fields
// (num_heads, num_buckets, max_distance, bidirectional) are required by the
// constructor so the bias table can never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceRelativePositionBiasConfig:public CAIF_Base
{
  public:
    // All four fields are required: head count, bucket count, the maximum
    // relative distance, and whether the bucketing is bidirectional.
    CAIF_DeviceRelativePositionBiasConfig(const uint32_t num_heads,
                                          const uint32_t num_buckets,
                                          const uint32_t max_distance,
                                          const bool bidirectional);

    // Number of attention heads (one bias row per head).
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}

    // Number of relative-position buckets.
    uint32_t NumBuckets()const{return _num_buckets;}
    void SetNumBuckets(const uint32_t num_buckets){_num_buckets=num_buckets;}

    // Maximum relative distance mapped into a bucket.
    uint32_t MaxDistance()const{return _max_distance;}
    void SetMaxDistance(const uint32_t max_distance){_max_distance=max_distance;}

    // Bidirectional bucketing (encoder) vs unidirectional (decoder).
    bool Bidirectional()const{return _bidirectional;}
    void SetBidirectional(const bool bidirectional){_bidirectional=bidirectional;}

  protected:

  private:
    uint32_t _num_heads;
    uint32_t _num_buckets;
    uint32_t _max_distance;
    bool _bidirectional;
};

}//end instance namespace
