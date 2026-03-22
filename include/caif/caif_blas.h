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
 * @file aif_blas.h
 * @brief BLAS-specific operations wrapper
 */

#pragma once

#include "caif_tensor.h"
#include <vector>

namespace instance
{
  class CAIF_BLAS
  {
    public:
      enum Transpose_e:uint8_t
      {
        NoTrans=0,
        Trans=1
      };
      // MatMul using BLAS (program must link a BLAS library)
      static CAIF_Tensor MatMul(
                               const CAIF_Tensor &a,
                               const CAIF_Tensor &b,
                               const std::vector<uint32_t> &shape_a,
                               const std::vector<uint32_t> &shape_b
                              );

      // Extended MatMul with transpose flags (avoids explicit transposes)
      static CAIF_Tensor MatMulEx(
                                 const CAIF_Tensor &a,
                                 const CAIF_Tensor &b,
                                 const std::vector<uint32_t> &shape_a,
                                 const std::vector<uint32_t> &shape_b,
                                 const Transpose_e trans_a,
                                 const Transpose_e trans_b
                                );
  };
}//end instance namespace


