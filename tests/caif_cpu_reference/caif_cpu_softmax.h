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
// Shared CPU-reference softmax primitive used across test files.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_SOFTMAX_H
#define CAIF_CPU_REFERENCE_SOFTMAX_H

#include <cmath>
#include <cstdint>

namespace instance
{

class CAIF_CpuSoftmax
{
  public:
    static void Apply(float *data,const int32_t rows,const int32_t cols)
    {
      for(int32_t r=0;r<rows;++r)
      {
        float *row=data+r*cols;
        float max_val=row[0];
        for(int32_t c=1;c<cols;++c)
        {
          if(row[c]>max_val)
          {
            max_val=row[c];
          }
        }
        float sum=0.0f;
        for(int32_t c=0;c<cols;++c)
        {
          row[c]=std::exp(row[c]-max_val);
          sum+=row[c];
        }
        for(int32_t c=0;c<cols;++c)
        {
          row[c]/=sum;
        }
      }
    }

    static void ApplyRow(float *row,const int32_t cols)
    {
      Apply(row,1,cols);
    }

  protected:
  private:
};

}//end instance namespace

#endif
