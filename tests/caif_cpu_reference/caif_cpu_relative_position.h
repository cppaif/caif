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
// Shared CPU-reference T5 relative-position bucket + bias primitive.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_RELATIVE_POSITION_H
#define CAIF_CPU_REFERENCE_RELATIVE_POSITION_H

#include <cmath>
#include <cstdint>

namespace instance
{

class CAIF_CpuRelativePosition
{
  public:
    static int32_t Bucket(const int32_t relative_position,
                          const bool bidirectional,
                          const int32_t num_buckets_in,
                          const int32_t max_distance)
    {
      int32_t num_buckets=num_buckets_in;
      int32_t offset=0;
      int32_t n=-relative_position;
      if(bidirectional==true)
      {
        num_buckets=num_buckets/2;
        if(n<0)
        {
          offset=num_buckets;
          n=-n;
        }
      }
      else
      {
        if(n<0)
        {
          n=0;
        }
      }

      const int32_t max_exact=num_buckets/2;
      int32_t bucket=0;
      if(n<max_exact)
      {
        bucket=n;
      }
      else
      {
        const float log_ratio=std::log(static_cast<float>(n)/
                                         static_cast<float>(max_exact));
        const float log_max=std::log(static_cast<float>(max_distance)/
                                       static_cast<float>(max_exact));
        bucket=max_exact+static_cast<int32_t>(log_ratio/log_max*
                                                static_cast<float>(num_buckets-
                                                                     max_exact));
        if(bucket>num_buckets-1)
        {
          bucket=num_buckets-1;
        }
      }
      return bucket+offset;
    }

    static void BiasForward(const float *embedding,float *output,
                            const int32_t num_heads,const int32_t q_len,
                            const int32_t k_len,const int32_t num_buckets,
                            const int32_t max_distance,const bool bidirectional)
    {
      for(int32_t h=0;h<num_heads;++h)
      {
        for(int32_t q=0;q<q_len;++q)
        {
          for(int32_t k=0;k<k_len;++k)
          {
            const int32_t rel_pos=k-q;
            const int32_t bucket=Bucket(rel_pos,bidirectional,num_buckets,
                                          max_distance);
            output[h*q_len*k_len+q*k_len+k]=embedding[h*num_buckets+bucket];
          }
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
