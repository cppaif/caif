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
// Host-side BLAS GEMM wrappers used by the CAIF_Ops host backend. Wraps
// OpenBLAS `cblas_sgemm` with row-major / trans-a / trans-b conventions,
// plus higher-level FP32-with-upcast helpers used by the public host
// MatMul / BatchedMatMul dispatchers.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"

namespace instance
{

class CAIF_HostGemm:public CAIF_Base
{
  public:
    // C = alpha * op(A) * op(B) + beta * C, row-major, FP32 only.
    static void GemmFloat(const float *a_data,
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
                          const float beta);

    // 2D MatMul with up/down-cast around an FP32 BLAS call. Output may be
    // FP16 / BF16; A and B may individually be any supported host dtype.
    static void MatMul2DFloat(const CAIF_DeviceTensor &a,
                              const CAIF_DeviceTensor &b,
                              CAIF_DeviceTensor &output,
                              const bool trans_a,
                              const bool trans_b);

    // Batched MatMul with up/down-cast. `m`/`k`/`n` describe the per-batch
    // op(A) × op(B) shape; `batch_count` is the outer batch dimension.
    static void BatchedMatMulFloatInternal(const CAIF_DeviceTensor &a,
                                           const CAIF_DeviceTensor &b,
                                           CAIF_DeviceTensor &output,
                                           const int m,
                                           const int k,
                                           const int n,
                                           const int batch_count,
                                           const bool trans_a,
                                           const bool trans_b);

  protected:

  private:
    CAIF_HostGemm()=delete;
};

}//end instance namespace
