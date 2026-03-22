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
 * @file aif_matrix_ops.cpp
 * @brief Implementation of high-performance matrix operations
 */

#include "caif_matrix_ops.h"
#include "caif_framework.h"
#include "caif_constants.h"
#include "openblas/cblas.h"
#include "ise_lib/ise_out.h"
#include <cstring>

using namespace instance;

void CAIF_MatrixOps::Multiply(
                             const float *a,
                             const float *b,
                             float *c,
                             const uint32_t m,
                             const uint32_t k,
                             const uint32_t n
                            )
{
  try
  {
    // C = A * B where A is m x k and B is k x n
    cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(m),
                static_cast<int>(n),
                static_cast<int>(k),
                1.0f,
                a,
                static_cast<int>(k),
                b,
                static_cast<int>(n),
                0.0f,
                c,
                static_cast<int>(n)
               );
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_MatrixOps::MultiplyEx(
                               const float *a,
                               const float *b,
                               float *c,
                               const uint32_t m,
                               const uint32_t k,
                               const uint32_t n,
                               const uint32_t lda,
                               const uint32_t ldb,
                               const uint32_t ldc,
                               const bool trans_a,
                               const bool trans_b
                              )
{
  try
  {
    const CBLAS_TRANSPOSE op_a=(trans_a==true)?CblasTrans:CblasNoTrans;
    const CBLAS_TRANSPOSE op_b=(trans_b==true)?CblasTrans:CblasNoTrans;
    
    cblas_sgemm(
                CblasRowMajor,
                op_a,
                op_b,
                static_cast<int>(m),
                static_cast<int>(n),
                static_cast<int>(k),
                1.0f,
                a,
                static_cast<int>(lda),
                b,
                static_cast<int>(ldb),
                0.0f,
                c,
                static_cast<int>(ldc)
               );
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_MatrixOps::Multiply(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    const auto &shape_a=a.Shape();
    const auto &shape_b=b.Shape();
    
    if(shape_a.size()!=g_caif_2d_matrix_dimensions||shape_b.size()!=g_caif_2d_matrix_dimensions)
    {
      THROW_CAIFE("Matrix multiplication requires 2D tensors");
    }
    if(shape_a[1]!=shape_b[0])
    {
      THROW_CAIFE("Matrix A columns must equal matrix B rows");
    }
    
    const uint32_t m=shape_a[0];
    const uint32_t k=shape_a[1];
    const uint32_t n=shape_b[1];
    
    CAIF_Tensor result(a.Framework(),{m,n},a.Type());
    Multiply(a.ConstData<float>(),b.ConstData<float>(),result.MutableData<float>(),m,k,n);
    
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_MatrixOps::MultiplyEx(
                                     const CAIF_Tensor &a,
                                     const CAIF_Tensor &b,
                                     const CAIF_TensorBackend::Transpose_e trans_a,
                                     const CAIF_TensorBackend::Transpose_e trans_b
                                    )
{
  try
  {
    const auto &shape_a=a.Shape();
    const auto &shape_b=b.Shape();
    
    if(shape_a.size()!=g_caif_2d_matrix_dimensions||shape_b.size()!=g_caif_2d_matrix_dimensions)
    {
      THROW_CAIFE("Matrix multiplication requires 2D tensors");
    }
    
    const bool ta=(trans_a==CAIF_TensorBackend::Transpose_e::Trans);
    const bool tb=(trans_b==CAIF_TensorBackend::Transpose_e::Trans);
    
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t k_b;
    
    if(ta==true)
    {
      m=shape_a[1];
      k=shape_a[0];
    }
    else
    {
      m=shape_a[0];
      k=shape_a[1];
    }
    
    if(tb==true)
    {
      n=shape_b[0];
      k_b=shape_b[1];
    }
    else
    {
      n=shape_b[1];
      k_b=shape_b[0];
    }
    
    if(k!=k_b)
    {
      THROW_CAIFE("Inner dimensions mismatch for MatrixMultiplyEx");
    }
    
    CAIF_Tensor result(a.Framework(),{m,n},a.Type());
    MultiplyEx(
               a.ConstData<float>(),
               b.ConstData<float>(),
               result.MutableData<float>(),
               m,
               k,
               n,
               shape_a[1],
               shape_b[1],
               n,
               ta,
               tb
              );
    
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_MatrixOps::Multiply(CAIF_Framework &framework,const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    const auto &shape_a=a.Shape();
    const auto &shape_b=b.Shape();
    
    if(shape_a.size()!=g_caif_2d_matrix_dimensions||shape_b.size()!=g_caif_2d_matrix_dimensions)
    {
      THROW_CAIFE("Matrix multiplication requires 2D tensors");
    }
    if(shape_a[1]!=shape_b[0])
    {
      THROW_CAIFE("Matrix A columns must equal matrix B rows");
    }
    
    const uint32_t m=shape_a[0];
    const uint32_t n=shape_b[1];
    
    // Ensure backend storage exists for inputs and output
    const_cast<CAIF_Tensor&>(a).EnsureBackendData();
    const_cast<CAIF_Tensor&>(b).EnsureBackendData();
    CAIF_Tensor result(framework,{m,n},a.Type());
    result.EnsureBackendData();

    CAIF_TensorBackend *backend=framework.Backend();
    if(backend==nullptr)
    {
      THROW_CAIFE("No backend initialized");
    }
    backend->MatrixMultiply(*a.TensorData(),*b.TensorData(),*result.TensorData());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_MatrixOps::MultiplyEx(
                                     CAIF_Framework &framework,
                                     const CAIF_Tensor &a,
                                     const CAIF_Tensor &b,
                                     const CAIF_TensorBackend::Transpose_e trans_a,
                                     const CAIF_TensorBackend::Transpose_e trans_b
                                    )
{
  try
  {
    const auto &shape_a=a.Shape();
    const auto &shape_b=b.Shape();
    
    if(shape_a.size()!=g_caif_2d_matrix_dimensions||shape_b.size()!=g_caif_2d_matrix_dimensions)
    {
      THROW_CAIFE("Matrix multiplication requires 2D tensors");
    }
    
    const bool ta=(trans_a==CAIF_TensorBackend::Transpose_e::Trans);
    const bool tb=(trans_b==CAIF_TensorBackend::Transpose_e::Trans);
    
    uint32_t m=(ta==true)?shape_a[1]:shape_a[0];
    uint32_t k=(ta==true)?shape_a[0]:shape_a[1];
    uint32_t n=(tb==true)?shape_b[0]:shape_b[1];
    uint32_t k_b=(tb==true)?shape_b[1]:shape_b[0];
    
    if(k!=k_b)
    {
      THROW_CAIFE("Inner dimensions mismatch for MatrixMultiplyEx");
    }
    
    // Ensure backend storage exists for inputs and output
    const_cast<CAIF_Tensor&>(a).EnsureBackendData();
    const_cast<CAIF_Tensor&>(b).EnsureBackendData();
    CAIF_Tensor result(framework,{m,n},a.Type());
    result.EnsureBackendData();

    CAIF_TensorBackend *backend=framework.Backend();
    if(backend==nullptr)
    {
      THROW_CAIFE("No backend initialized");
    }
    backend->MatrixMultiplyEx(*a.TensorData(),*b.TensorData(),*result.TensorData(),trans_a,trans_b);
    return result;
  }
  CCAIF_CATCH_BLOCK()
}
