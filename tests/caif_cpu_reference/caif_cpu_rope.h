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
// Shared CPU-reference RoPE rotation primitive.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_ROPE_H
#define CAIF_CPU_REFERENCE_ROPE_H

#include <cmath>
#include <cstdint>

namespace instance
{

class CAIF_CpuRoPE
{
  public:
    static void Apply(float *data,const int32_t batch_heads,
                      const int32_t seq_len,const int32_t head_dim,
                      const float base)
    {
      const int32_t half_dim=head_dim/2;
      for(int32_t bh=0;bh<batch_heads;++bh)
      {
        for(int32_t s=0;s<seq_len;++s)
        {
          for(int32_t p=0;p<half_dim;++p)
          {
            const float freq_exp=2.0f*static_cast<float>(p)/
                                   static_cast<float>(head_dim);
            const float theta=static_cast<float>(s)/std::pow(base,freq_exp);
            const float cos_t=std::cos(theta);
            const float sin_t=std::sin(theta);

            const int32_t idx=(bh*seq_len+s)*head_dim+p*2;
            const float x0=data[idx];
            const float x1=data[idx+1];
            data[idx]=x0*cos_t-x1*sin_t;
            data[idx+1]=x0*sin_t+x1*cos_t;
          }
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
