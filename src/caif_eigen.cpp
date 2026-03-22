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

#include "caif_eigen.h"
#include "caif_exception.h"
#include <Eigen/Core>

namespace instance
{
  void CAIF_EIGEN::SetNumThreads(unsigned int nthreads)
  {
    if(nthreads==0)
    {
      nthreads=1;
    }
    Eigen::setNbThreads(static_cast<int>(nthreads));
    return;
  }

  CAIF_Tensor CAIF_EIGEN::MatMul(
                                const CAIF_Tensor &a,
                                const CAIF_Tensor &b,
                                const std::vector<uint32_t> &shape_a,
                                const std::vector<uint32_t> &shape_b
                              )
  {
    if(a.Type()!=CAIF_DataType::CAIF_DataType_e::Float32||b.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Only Float32 matmul is supported by Eigen");
    }
    using RowMajorMatrix=Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const float *a_data=static_cast<const float*>(a.Data());
    const float *b_data=static_cast<const float*>(b.Data());
    if(a_data==nullptr||b_data==nullptr)
    {
      THROW_CAIFE("Invalid tensor data");
    }
    const int rows_a=static_cast<int>(shape_a[0]);
    const int cols_a=static_cast<int>(shape_a[1]);
    const int rows_b=static_cast<int>(shape_b[0]);
    const int cols_b=static_cast<int>(shape_b[1]);
    (void)rows_b;
    Eigen::Map<const RowMajorMatrix> A(a_data, rows_a, cols_a);
    Eigen::Map<const RowMajorMatrix> B(b_data, rows_b, cols_b);
    CAIF_Tensor out(a.Framework(),{shape_a[0],shape_b[1]},a.Type());
    float *out_ptr=static_cast<float*>(out.Data());
    Eigen::Map<RowMajorMatrix> C(out_ptr, rows_a, cols_b);
    C.noalias()=A*B;
    return out;
  }
}//end instance namespace


