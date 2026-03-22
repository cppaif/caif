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

#include "caif_blas.h"
#include "caif_exception.h"
#include "openblas/cblas.h"

namespace instance
{
  CAIF_Tensor CAIF_BLAS::MatMul(
                               const CAIF_Tensor &a,
                               const CAIF_Tensor &b,
                               const std::vector<uint32_t> &shape_a,
                               const std::vector<uint32_t> &shape_b
                              )
  {
    if(a.Type()!=CAIF_DataType::CAIF_DataType_e::Float32||b.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Only Float32 matmul is supported by BLAS");
    }
    const float *a_data=static_cast<const float*>(a.Data());
    const float *b_data=static_cast<const float*>(b.Data());
    if(a_data==nullptr||b_data==nullptr)
    {
      THROW_CAIFE("Invalid tensor data");
    }
    const int m=static_cast<int>(shape_a[0]);
    const int k=static_cast<int>(shape_a[1]);
    const int n=static_cast<int>(shape_b[1]);
    CAIF_Tensor out(a.Framework(),{shape_a[0],shape_b[1]},a.Type());
    float *c_data=static_cast<float*>(out.Data());
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                m,n,k,
                1.0f,
                a_data,k,
                b_data,n,
                0.0f,
                c_data,n);
    return out;
  }

  CAIF_Tensor CAIF_BLAS::MatMulEx(
                                 const CAIF_Tensor &a,
                                 const CAIF_Tensor &b,
                                 const std::vector<uint32_t> &shape_a,
                                 const std::vector<uint32_t> &shape_b,
                                 const Transpose_e trans_a,
                                 const Transpose_e trans_b
                                )
  {
    if(a.Type()!=CAIF_DataType::CAIF_DataType_e::Float32||b.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Only Float32 matmul is supported by BLAS");
    }
    const float *a_data=static_cast<const float*>(a.Data());
    const float *b_data=static_cast<const float*>(b.Data());
    if(a_data==nullptr||b_data==nullptr)
    {
      THROW_CAIFE("Invalid tensor data");
    }

    const int a_rows=static_cast<int>(shape_a[0]);
    const int a_cols=static_cast<int>(shape_a[1]);
    const int b_rows=static_cast<int>(shape_b[0]);
    const int b_cols=static_cast<int>(shape_b[1]);

    const bool ta=(trans_a==Transpose_e::Trans);
    const bool tb=(trans_b==Transpose_e::Trans);

    int m;   // rows of op(A)
    if(ta==true)
    {
      m=a_cols;
    }
    else
    {
      m=a_rows;
    }

    int k;   // cols of op(A)
    if(ta==true)
    {
      k=a_rows;
    }
    else
    {
      k=a_cols;
    }

    int n;   // cols of op(B)
    if(tb==true)
    {
      n=b_rows;
    }
    else
    {
      n=b_cols;
    }

    // Validate inner dims
    int k_b; // rows of op(B)
    if(tb==true)
    {
      k_b=b_cols;
    }
    else
    {
      k_b=b_rows;
    }
    if(k!=k_b)
    {
      THROW_CAIFE("Inner dimensions mismatch for MatMulEx");
    }

    CAIF_Tensor out(a.Framework(),{static_cast<uint32_t>(m),static_cast<uint32_t>(n)},a.Type());
    float *c_data=static_cast<float*>(out.Data());

    CBLAS_TRANSPOSE cta;
    if(ta==true)
    {
      cta=CblasTrans;
    }
    else
    {
      cta=CblasNoTrans;
    }
    CBLAS_TRANSPOSE ctb;
    if(tb==true)
    {
      ctb=CblasTrans;
    }
    else
    {
      ctb=CblasNoTrans;
    }

    // For RowMajor: lda=(TransA? M : K), ldb=(TransB? K : N), ldc=N where M,N,K are sgemm args
    int lda;
    if(ta==true)
    {
      lda=m;
    }
    else
    {
      lda=k;
    }
    int ldb;
    if(tb==true)
    {
      ldb=k;
    }
    else
    {
      ldb=n;
    }
    const int ldc=n;

    cblas_sgemm(CblasRowMajor,cta,ctb,
                m,n,k,
                1.0f,
                a_data,lda,
                b_data,ldb,
                0.0f,
                c_data,ldc);
    return out;
  }
}//end instance namespace


