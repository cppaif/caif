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

/**
 * @file aif_activations.h
 * @brief High-performance activation function operations
 * 
 * Each activation function is in its own static class for modularity.
 * All operations work DIRECTLY on tensor data pointers for maximum performance.
 */

#ifndef CAIF_ACTIVATIONS_H
#define CAIF_ACTIVATIONS_H

#include "caif_tensor.h"
#include "caif_exception.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace instance
{

class CAIF_Framework;

//==============================================================================
// CAIF_ReLU - Rectified Linear Unit
//==============================================================================

class CAIF_ReLU
{
  public:
    CAIF_ReLU()=delete;
    
    static void ForwardInPlace(float *data,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(data[i]<0.0f)
          {
            data[i]=0.0f;
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Forward(const float *input,float *output,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(input[i]>0.0f)
          {
            output[i]=input[i];
          }
          else
          {
            output[i]=0.0f;
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(const float *input,const float *grad_output,float *grad_input,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(input[i]>0.0f)
          {
            grad_input[i]=grad_output[i];
          }
          else
          {
            grad_input[i]=0.0f;
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input);
    static CAIF_Tensor Backward(const CAIF_Tensor &input,const CAIF_Tensor &grad_output);
};

//==============================================================================
// CAIF_Sigmoid
//==============================================================================

class CAIF_Sigmoid
{
  public:
    CAIF_Sigmoid()=delete;
    
    static void ForwardInPlace(float *data,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          data[i]=1.0f/(1.0f+std::exp(-data[i]));
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Forward(const float *input,float *output,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          output[i]=1.0f/(1.0f+std::exp(-input[i]));
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(const float *output,const float *grad_output,float *grad_input,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          grad_input[i]=grad_output[i]*output[i]*(1.0f-output[i]);
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input);
    static CAIF_Tensor Backward(const CAIF_Tensor &output,const CAIF_Tensor &grad_output);
};

//==============================================================================
// CAIF_Tanh
//==============================================================================

class CAIF_Tanh
{
  public:
    CAIF_Tanh()=delete;
    
    static void ForwardInPlace(float *data,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          data[i]=std::tanh(data[i]);
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Forward(const float *input,float *output,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          output[i]=std::tanh(input[i]);
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(const float *output,const float *grad_output,float *grad_input,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          grad_input[i]=grad_output[i]*(1.0f-output[i]*output[i]);
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input);
    static CAIF_Tensor Backward(const CAIF_Tensor &output,const CAIF_Tensor &grad_output);
};

//==============================================================================
// CAIF_LeakyReLU
//==============================================================================

class CAIF_LeakyReLU
{
  public:
    CAIF_LeakyReLU()=delete;
    
    static constexpr float g_default_alpha=0.01f;
    
    static void ForwardInPlace(float *data,const size_t n,const float alpha=g_default_alpha)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(data[i]<0.0f)
          {
            data[i]=alpha*data[i];
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Forward(const float *input,float *output,const size_t n,const float alpha=g_default_alpha)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(input[i]>0.0f)
          {
            output[i]=input[i];
          }
          else
          {
            output[i]=alpha*input[i];
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(
                         const float *input,
                         const float *grad_output,
                         float *grad_input,
                         const size_t n,
                         const float alpha=g_default_alpha
                        )
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(input[i]>0.0f)
          {
            grad_input[i]=grad_output[i];
          }
          else
          {
            grad_input[i]=alpha*grad_output[i];
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input,const float alpha=g_default_alpha);
    static CAIF_Tensor Backward(
                               const CAIF_Tensor &input,
                               const CAIF_Tensor &grad_output,
                               const float alpha=g_default_alpha
                              );
};

//==============================================================================
// CAIF_ELU - Exponential Linear Unit
//==============================================================================

class CAIF_ELU
{
  public:
    CAIF_ELU()=delete;
    
    static constexpr float g_default_alpha=1.0f;
    
    static void Forward(const float *input,float *output,const size_t n,const float alpha=g_default_alpha)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(input[i]>0.0f)
          {
            output[i]=input[i];
          }
          else
          {
            output[i]=alpha*(std::exp(input[i])-1.0f);
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(
                         const float *input,
                         const float *output,
                         const float *grad_output,
                         float *grad_input,
                         const size_t n,
                         const float alpha=g_default_alpha
                        )
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          if(input[i]>0.0f)
          {
            grad_input[i]=grad_output[i];
          }
          else
          {
            grad_input[i]=grad_output[i]*(output[i]+alpha);
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input,const float alpha=g_default_alpha);
    static CAIF_Tensor Backward(
                               const CAIF_Tensor &input,
                               const CAIF_Tensor &output,
                               const CAIF_Tensor &grad_output,
                               const float alpha=g_default_alpha
                              );
};

//==============================================================================
// CAIF_GELU - Gaussian Error Linear Unit
//==============================================================================

class CAIF_GELU
{
  public:
    CAIF_GELU()=delete;
    
    static constexpr float g_sqrt_2_over_pi=0.7978845608f;
    static constexpr float g_gelu_coeff=0.044715f;
    
    static void Forward(const float *input,float *output,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          const float x=input[i];
          const float cdf=0.5f*(1.0f+std::tanh(g_sqrt_2_over_pi*(x+g_gelu_coeff*x*x*x)));
          output[i]=x*cdf;
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(const float *input,const float *grad_output,float *grad_input,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          const float x=input[i];
          const float x3=x*x*x;
          const float inner=g_sqrt_2_over_pi*(x+g_gelu_coeff*x3);
          const float tanh_inner=std::tanh(inner);
          const float sech2=1.0f-tanh_inner*tanh_inner;
          const float cdf=0.5f*(1.0f+tanh_inner);
          const float pdf=0.5f*g_sqrt_2_over_pi*(1.0f+3.0f*g_gelu_coeff*x*x)*sech2;
          grad_input[i]=grad_output[i]*(cdf+x*pdf);
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input);
    static CAIF_Tensor Backward(const CAIF_Tensor &input,const CAIF_Tensor &grad_output);
};

//==============================================================================
// CAIF_Swish (SiLU)
//==============================================================================

class CAIF_Swish
{
  public:
    CAIF_Swish()=delete;
    
    static void Forward(const float *input,float *output,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          const float sigmoid=1.0f/(1.0f+std::exp(-input[i]));
          output[i]=input[i]*sigmoid;
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(
                         const float *input,
                         const float *output,
                         const float *grad_output,
                         float *grad_input,
                         const size_t n
                        )
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          const float x=input[i];
          const float sigmoid=1.0f/(1.0f+std::exp(-x));
          grad_input[i]=grad_output[i]*(output[i]+sigmoid*(1.0f-output[i]));
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input);
    static CAIF_Tensor Backward(const CAIF_Tensor &input,const CAIF_Tensor &output,const CAIF_Tensor &grad_output);
};

//==============================================================================
// CAIF_Softmax
//==============================================================================

class CAIF_Softmax
{
  public:
    CAIF_Softmax()=delete;
    
    static void Forward(const float *input,float *output,const size_t batch_size,const size_t num_classes)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t b=0;b<batch_size;++b)
        {
          const float *in_row=input+b*num_classes;
          float *out_row=output+b*num_classes;
          
          // Find max for numerical stability
          float max_val=in_row[0];
          for(size_t i=1;i<num_classes;++i)
          {
            if(in_row[i]>max_val)
            {
              max_val=in_row[i];
            }
          }
          
          // Compute exp and sum
          float sum=0.0f;
          for(size_t i=0;i<num_classes;++i)
          {
            out_row[i]=std::exp(in_row[i]-max_val);
            sum+=out_row[i];
          }
          
          // Normalize
          const float inv_sum=1.0f/sum;
          for(size_t i=0;i<num_classes;++i)
          {
            out_row[i]*=inv_sum;
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static void Backward(
                         const float *output,
                         const float *grad_output,
                         float *grad_input,
                         const size_t batch_size,
                         const size_t num_classes
                        )
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t b=0;b<batch_size;++b)
        {
          const float *s=output+b*num_classes;
          const float *g=grad_output+b*num_classes;
          float *gi=grad_input+b*num_classes;
          
          float dot=0.0f;
          for(size_t i=0;i<num_classes;++i)
          {
            dot+=g[i]*s[i];
          }
          
          for(size_t i=0;i<num_classes;++i)
          {
            gi[i]=s[i]*(g[i]-dot);
          }
        }
      }
      CAIF_CATCH_BLOCK()
    }
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input);
    static CAIF_Tensor Backward(const CAIF_Tensor &output,const CAIF_Tensor &grad_output);
};

}//end instance namespace

#endif  // CAIF_ACTIVATIONS_H

