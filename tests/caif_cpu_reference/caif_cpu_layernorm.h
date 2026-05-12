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
// Shared CPU-reference LayerNorm primitive.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_LAYERNORM_H
#define CAIF_CPU_REFERENCE_LAYERNORM_H

#include <cmath>
#include <cstdint>

namespace instance
{

class CAIF_CpuLayerNorm
{
  public:
    static void Apply(const float *input,const float *gamma,const float *beta,
                      float *output,const int32_t rows,const int32_t dim,
                      const float epsilon)
    {
      for(int32_t r=0;r<rows;++r)
      {
        float sum=0.0f;
        for(int32_t c=0;c<dim;++c)
        {
          sum+=input[r*dim+c];
        }
        const float mean=sum/static_cast<float>(dim);

        float var_sum=0.0f;
        for(int32_t c=0;c<dim;++c)
        {
          const float diff=input[r*dim+c]-mean;
          var_sum+=diff*diff;
        }
        const float variance=var_sum/static_cast<float>(dim);
        const float rstd=1.0f/std::sqrt(variance+epsilon);

        for(int32_t c=0;c<dim;++c)
        {
          const float x_hat=(input[r*dim+c]-mean)*rstd;
          output[r*dim+c]=x_hat*gamma[c]+beta[c];
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
