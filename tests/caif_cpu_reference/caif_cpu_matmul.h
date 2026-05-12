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
// Shared CPU-reference matrix-multiply primitives used across test files.
// Extracted verbatim from previously-duplicated test-local statics.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_MATMUL_H
#define CAIF_CPU_REFERENCE_MATMUL_H

#include <cstdint>

namespace instance
{

class CAIF_CpuMatMul
{
  public:
    static void Apply(const float *a,const float *b,float *c,
                      const int32_t m,const int32_t k,const int32_t n)
    {
      for(int32_t i=0;i<m;++i)
      {
        for(int32_t j=0;j<n;++j)
        {
          float sum=0.0f;
          for(int32_t p=0;p<k;++p)
          {
            sum+=a[i*k+p]*b[p*n+j];
          }
          c[i*n+j]=sum;
        }
      }
    }

    static void TransposeB(const float *a,const float *b,float *c,
                           const int32_t m,const int32_t k,const int32_t n)
    {
      for(int32_t i=0;i<m;++i)
      {
        for(int32_t j=0;j<n;++j)
        {
          float sum=0.0f;
          for(int32_t p=0;p<k;++p)
          {
            sum+=a[i*k+p]*b[j*k+p];
          }
          c[i*n+j]=sum;
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
