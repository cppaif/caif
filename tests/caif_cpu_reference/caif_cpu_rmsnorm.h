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
// Shared CPU-reference RMSNorm primitive.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_RMSNORM_H
#define CAIF_CPU_REFERENCE_RMSNORM_H

#include <cmath>
#include <cstdint>

namespace instance
{

class CAIF_CpuRMSNorm
{
  public:
    static void Apply(const float *input,const float *gamma,float *output,
                      const int32_t rows,const int32_t dim,const float epsilon)
    {
      for(int32_t r=0;r<rows;++r)
      {
        float sum_sq=0.0f;
        for(int32_t c=0;c<dim;++c)
        {
          const float val=input[r*dim+c];
          sum_sq+=val*val;
        }
        const float rms=std::sqrt(sum_sq/static_cast<float>(dim)+epsilon);
        for(int32_t c=0;c<dim;++c)
        {
          output[r*dim+c]=input[r*dim+c]/rms*gamma[c];
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
