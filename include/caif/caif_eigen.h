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
 * @file aif_eigen.h
 * @brief Eigen-specific operations wrapper
 */

#pragma once

#include "caif_tensor.h"
#include <vector>

namespace instance
{
  class CAIF_EIGEN
  {
    public:
      // Configure Eigen worker thread count
      static void SetNumThreads(unsigned int nthreads);

      // MatMul using Eigen zero-copy Maps
      static CAIF_Tensor MatMul(
                               const CAIF_Tensor &a,
                               const CAIF_Tensor &b,
                               const std::vector<uint32_t> &shape_a,
                               const std::vector<uint32_t> &shape_b
                              );
  };
}//end instance namespace


