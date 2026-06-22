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

#include "caif_host_gemm.h"
#include "caif_host_fp_cast.h"
#include "caif_exception.h"
#include "openblas/cblas.h"

#include <cstddef>
#include <vector>

namespace instance
{

void CAIF_HostGemm::GemmFloat(const float *a_data,
                              const float *b_data,
                              float *c_data,
                              const int m,
                              const int n,
                              const int k,
                              const int lda,
                              const int ldb,
                              const int ldc,
                              const bool trans_a,
                              const bool trans_b,
                              const float alpha,
                              const float beta)
{
  try
  {
    CBLAS_TRANSPOSE op_a=CblasNoTrans;
    if(trans_a==true)
    {
      op_a=CblasTrans;
    }
    CBLAS_TRANSPOSE op_b=CblasNoTrans;
    if(trans_b==true)
    {
      op_b=CblasTrans;
    }
    cblas_sgemm(CblasRowMajor,
                op_a,
                op_b,
                m,
                n,
                k,
                alpha,
                a_data,
                lda,
                b_data,
                ldb,
                beta,
                c_data,
                ldc);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_HostGemm::MatMul2DFloat(const CAIF_DeviceTensor &a,
                                  const CAIF_DeviceTensor &b,
                                  CAIF_DeviceTensor &output,
                                  const bool trans_a,
                                  const bool trans_b)
{
  try
  {
    const auto &sa=a.Shape();
    const auto &sb=b.Shape();
    const int a_rows=static_cast<int>(sa[0]);
    const int a_cols=static_cast<int>(sa[1]);
    const int b_rows=static_cast<int>(sb[0]);
    const int b_cols=static_cast<int>(sb[1]);
    int m=a_rows;
    int k_a=a_cols;
    if(trans_a==true)
    {
      m=a_cols;
      k_a=a_rows;
    }
    int k_b=b_rows;
    int n=b_cols;
    if(trans_b==true)
    {
      k_b=b_cols;
      n=b_rows;
    }
    if(k_a!=k_b)
    {
      THROW_CAIFE("CAIF_HostGemm::MatMul2DFloat: inner dimensions do not match");
    }
    const int lda=a_cols;
    const int ldb=b_cols;
    const int ldc=n;
    if(a.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32
       && b.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32
       && output.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      // fp32 by branch gate (a/b/output dtypes verified above)
      GemmFloat(static_cast<const float*>(a.DeviceDataRaw()),
                static_cast<const float*>(b.DeviceDataRaw()),
                static_cast<float*>(output.DeviceDataRaw()),
                m,
                n,
                k_a,
                lda,
                ldb,
                ldc,
                trans_a,
                trans_b,
                1.0f,
                0.0f);
      return;
    }
    std::vector<float> af=CAIF_HostFpCast::UpcastToFloat(a);
    std::vector<float> bf=CAIF_HostFpCast::UpcastToFloat(b);
    std::vector<float> cf(static_cast<size_t>(m)*static_cast<size_t>(n));
    GemmFloat(af.data(),
              bf.data(),
              cf.data(),
              m,
              n,
              k_a,
              lda,
              ldb,
              ldc,
              trans_a,
              trans_b,
              1.0f,
              0.0f);
    CAIF_HostFpCast::DowncastFromFloat(cf,output);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_HostGemm::BatchedMatMulFloatInternal(const CAIF_DeviceTensor &a,
                                               const CAIF_DeviceTensor &b,
                                               CAIF_DeviceTensor &output,
                                               const int m,
                                               const int k,
                                               const int n,
                                               const int batch_count,
                                               const bool trans_a,
                                               const bool trans_b)
{
  try
  {
    const size_t a_slice=static_cast<size_t>(m)*static_cast<size_t>(k);
    const size_t b_slice=static_cast<size_t>(k)*static_cast<size_t>(n);
    const size_t c_slice=static_cast<size_t>(m)*static_cast<size_t>(n);
    const int lda_notrans=k;
    const int lda_trans=m;
    const int ldb_notrans=n;
    const int ldb_trans=k;
    int lda=lda_notrans;
    int ldb=ldb_notrans;
    if(trans_a==true)
    {
      lda=lda_trans;
    }
    if(trans_b==true)
    {
      ldb=ldb_trans;
    }
    const int ldc=n;
    if(a.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32
       && b.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32
       && output.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      // fp32 by branch gate (a/b/output dtypes verified above)
      const float *a_base=static_cast<const float*>(a.DeviceDataRaw());
      const float *b_base=static_cast<const float*>(b.DeviceDataRaw());
      float *c_base=static_cast<float*>(output.DeviceDataRaw());
      #pragma omp parallel for
      for(int batch=0;batch<batch_count;++batch)
      {
        GemmFloat(a_base+static_cast<size_t>(batch)*a_slice,
                  b_base+static_cast<size_t>(batch)*b_slice,
                  c_base+static_cast<size_t>(batch)*c_slice,
                  m,
                  n,
                  k,
                  lda,
                  ldb,
                  ldc,
                  trans_a,
                  trans_b,
                  1.0f,
                  0.0f);
      }
      return;
    }
    std::vector<float> af=CAIF_HostFpCast::UpcastToFloat(a);
    std::vector<float> bf=CAIF_HostFpCast::UpcastToFloat(b);
    std::vector<float> cf(static_cast<size_t>(batch_count)*c_slice);
    #pragma omp parallel for
    for(int batch=0;batch<batch_count;++batch)
    {
      GemmFloat(af.data()+static_cast<size_t>(batch)*a_slice,
                bf.data()+static_cast<size_t>(batch)*b_slice,
                cf.data()+static_cast<size_t>(batch)*c_slice,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                trans_a,
                trans_b,
                1.0f,
                0.0f);
    }
    CAIF_HostFpCast::DowncastFromFloat(cf,output);
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
