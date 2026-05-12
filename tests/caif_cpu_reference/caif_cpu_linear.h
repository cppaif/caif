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
// Shared CPU-reference linear / bias-add primitives.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_LINEAR_H
#define CAIF_CPU_REFERENCE_LINEAR_H

#include <cstdint>

namespace instance
{

class CAIF_CpuLinear
{
  public:
    static void Apply(const float *input,const float *weight,const float *bias,
                      float *output,const int32_t n,const int32_t input_dim,
                      const int32_t output_dim,const bool use_bias)
    {
      for(int32_t i=0;i<n;++i)
      {
        for(int32_t o=0;o<output_dim;++o)
        {
          float sum=0.0f;
          for(int32_t j=0;j<input_dim;++j)
          {
            sum+=input[i*input_dim+j]*weight[j*output_dim+o];
          }
          if(use_bias==true)
          {
            sum+=bias[o];
          }
          output[i*output_dim+o]=sum;
        }
      }
    }

    static void ApplyTranspose(const float *input,const float *weight_t,
                               const float *bias,float *output,
                               const int32_t n,const int32_t input_dim,
                               const int32_t output_dim,const bool use_bias)
    {
      for(int32_t i=0;i<n;++i)
      {
        for(int32_t o=0;o<output_dim;++o)
        {
          float sum=0.0f;
          for(int32_t j=0;j<input_dim;++j)
          {
            sum+=input[i*input_dim+j]*weight_t[o*input_dim+j];
          }
          if(use_bias==true)
          {
            sum+=bias[o];
          }
          output[i*output_dim+o]=sum;
        }
      }
    }

    static void BiasAdd(float *data,const float *bias,const int32_t rows,
                        const int32_t cols)
    {
      for(int32_t r=0;r<rows;++r)
      {
        for(int32_t c=0;c<cols;++c)
        {
          data[r*cols+c]+=bias[c];
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
