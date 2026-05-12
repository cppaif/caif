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
// Shared CPU-reference pointwise activation primitives.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_ACTIVATIONS_H
#define CAIF_CPU_REFERENCE_ACTIVATIONS_H

#include <cmath>
#include <cstdint>

namespace instance
{

class CAIF_CpuActivations
{
  public:
    static float GELU(const float x)
    {
      const float sqrt_2_over_pi=0.7978845608f;
      const float coeff=0.044715f;
      const float inner=sqrt_2_over_pi*(x+coeff*x*x*x);
      return 0.5f*x*(1.0f+std::tanh(inner));
    }

    static void GELUArray(float *data,const int32_t n)
    {
      for(int32_t i=0;i<n;++i)
      {
        data[i]=GELU(data[i]);
      }
    }

    static float Swish(const float x)
    {
      return x/(1.0f+std::exp(-x));
    }

    static void SwishArray(float *data,const int32_t n)
    {
      for(int32_t i=0;i<n;++i)
      {
        data[i]=Swish(data[i]);
      }
    }

    static void SwiGLU(const float *gate,const float *up,float *out,
                       const int32_t n)
    {
      for(int32_t i=0;i<n;++i)
      {
        out[i]=Swish(gate[i])*up[i];
      }
    }

    static void GeGLU(const float *gate,const float *up,float *out,
                      const int32_t n)
    {
      for(int32_t i=0;i<n;++i)
      {
        out[i]=GELU(gate[i])*up[i];
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
