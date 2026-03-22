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
 * @file aif_element_ops.h
 * @brief High-performance element-wise tensor operations
 * 
 * All operations work DIRECTLY on tensor data pointers for maximum performance.
 * No intermediate tensor creation or copying.
 */

#ifndef CAIF_ELEMENT_OPS_H
#define CAIF_ELEMENT_OPS_H

#include "caif_tensor.h"
#include "caif_tensor_backend.h"
#include "caif_exception.h"
#include <cstdint>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace instance
{

/**
 * @brief Static class for high-performance element-wise operations
 * 
 * All methods operate directly on tensor data pointers, avoiding
 * unnecessary memory allocation and copying.
 */
class CAIF_ElementOps
{
  public:
    CAIF_ElementOps()=delete;  // Static class - no instances
    
    //--------------------------------------------------------------------------
    // In-place operations (fastest - modify data directly)
    //--------------------------------------------------------------------------
    
    /**
     * @brief In-place addition: a += b
     */
    static void AddInPlace(float *a,const float *b,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]+=b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place scalar addition: a += scalar
     */
    static void AddScalarInPlace(float *a,const float scalar,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]+=scalar;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place subtraction: a -= b
     */
    static void SubInPlace(float *a,const float *b,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]-=b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place scalar subtraction: a -= scalar
     */
    static void SubScalarInPlace(float *a,const float scalar,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]-=scalar;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place multiplication: a *= b
     */
    static void MulInPlace(float *a,const float *b,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]*=b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place scalar multiplication: a *= scalar
     */
    static void MulScalarInPlace(float *a,const float scalar,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]*=scalar;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place division: a /= b
     */
    static void DivInPlace(float *a,const float *b,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]/=b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place scalar division: a /= scalar
     */
    static void DivScalarInPlace(float *a,const float scalar,const size_t n)
    {
      try
      {
        const float inv=1.0f/scalar;
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]*=inv;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief In-place square root: a = sqrt(a)
     */
    static void SqrtInPlace(float *a,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          a[i]=std::sqrt(a[i]);
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    //--------------------------------------------------------------------------
    // Output operations (write to separate destination)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Addition: out = a + b
     */
    static void Add(const float *a,const float *b,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]+b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Scalar addition: out = a + scalar
     */
    static void AddScalar(const float *a,const float scalar,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]+scalar;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Subtraction: out = a - b
     */
    static void Sub(const float *a,const float *b,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]-b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Scalar subtraction: out = a - scalar
     */
    static void SubScalar(const float *a,const float scalar,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]-scalar;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Multiplication: out = a * b
     */
    static void Mul(const float *a,const float *b,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]*b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Scalar multiplication: out = a * scalar
     */
    static void MulScalar(const float *a,const float scalar,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]*scalar;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Division: out = a / b
     */
    static void Div(const float *a,const float *b,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]/b[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Scalar division: out = a / scalar
     */
    static void DivScalar(const float *a,const float scalar,float *out,const size_t n)
    {
      try
      {
        const float inv=1.0f/scalar;
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=a[i]*inv;
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Square root: out = sqrt(a)
     */
    static void Sqrt(const float *a,float *out,const size_t n)
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          out[i]=std::sqrt(a[i]);
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    //--------------------------------------------------------------------------
    // Reduction operations
    //--------------------------------------------------------------------------
    
    /**
     * @brief Sum all elements
     */
    static float Sum(const float *a,const size_t n)
    {
      try
      {
        float sum=0.0f;
        #ifdef _OPENMP
        #pragma omp parallel for simd reduction(+:sum)
        #endif
        for(size_t i=0;i<n;++i)
        {
          sum+=a[i];
        }
        return sum;
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Mean of all elements
     */
    static float Mean(const float *a,const size_t n)
    {
      try
      {
        return Sum(a,n)/static_cast<float>(n);
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Max element value
     */
    static float Max(const float *a,const size_t n)
    {
      try
      {
        float max_val=a[0];
        #ifdef _OPENMP
        #pragma omp parallel for reduction(max:max_val)
        #endif
        for(size_t i=1;i<n;++i)
        {
          if(a[i]>max_val)
          {
            max_val=a[i];
          }
        }
        return max_val;
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Min element value
     */
    static float Min(const float *a,const size_t n)
    {
      try
      {
        float min_val=a[0];
        #ifdef _OPENMP
        #pragma omp parallel for reduction(min:min_val)
        #endif
        for(size_t i=1;i<n;++i)
        {
          if(a[i]<min_val)
          {
            min_val=a[i];
          }
        }
        return min_val;
      }
      CCAIF_CATCH_BLOCK()
    }
    
    //--------------------------------------------------------------------------
    // CAIF_Tensor convenience wrappers (allocate output tensor)
    // Implementations in aif_element_ops.cpp
    //--------------------------------------------------------------------------
    
    static CAIF_Tensor Add(const CAIF_Tensor &a,const CAIF_Tensor &b);
    static CAIF_Tensor AddScalar(const CAIF_Tensor &a,const float scalar);
    static CAIF_Tensor Sub(const CAIF_Tensor &a,const CAIF_Tensor &b);
    static CAIF_Tensor SubScalar(const CAIF_Tensor &a,const float scalar);
    static CAIF_Tensor Mul(const CAIF_Tensor &a,const CAIF_Tensor &b);
    static CAIF_Tensor MulScalar(const CAIF_Tensor &a,const float scalar);
    static CAIF_Tensor Div(const CAIF_Tensor &a,const CAIF_Tensor &b);
    static CAIF_Tensor DivScalar(const CAIF_Tensor &a,const float scalar);
    static CAIF_Tensor Sqrt(const CAIF_Tensor &a);
    static float Sum(const CAIF_Tensor &a);
    static float Mean(const CAIF_Tensor &a);
};

}//end instance namespace

#endif  // CAIF_ELEMENT_OPS_H
