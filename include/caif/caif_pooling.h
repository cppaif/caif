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
 * @file aif_pooling.h
 * @brief High-performance pooling operations
 */

#ifndef CAIF_POOLING_H
#define CAIF_POOLING_H

#include "caif_tensor.h"
#include "caif_exception.h"
#include <cstdint>
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace instance
{

class CAIF_Framework;

/**
 * @brief Static class for high-performance pooling operations
 */
class CAIF_Pooling
{
  public:
    CAIF_Pooling()=delete;
    
    //--------------------------------------------------------------------------
    // Max Pooling 2D
    //--------------------------------------------------------------------------
    
    static void MaxPool2D(
                          const float *input,
                          float *output,
                          uint32_t *indices,
                          const uint32_t batch,
                          const uint32_t in_h,
                          const uint32_t in_w,
                          const uint32_t channels,
                          const uint32_t pool_size,
                          const uint32_t stride,
                          const uint32_t padding
                         );
    
    static void MaxPool2DBackward(
                                  const float *grad_output,
                                  const uint32_t *indices,
                                  float *grad_input,
                                  const uint32_t batch,
                                  const uint32_t in_h,
                                  const uint32_t in_w,
                                  const uint32_t out_h,
                                  const uint32_t out_w,
                                  const uint32_t channels
                                 );
    
    //--------------------------------------------------------------------------
    // Average Pooling 2D
    //--------------------------------------------------------------------------
    
    static void AvgPool2D(
                          const float *input,
                          float *output,
                          const uint32_t batch,
                          const uint32_t in_h,
                          const uint32_t in_w,
                          const uint32_t channels,
                          const uint32_t pool_size,
                          const uint32_t stride,
                          const uint32_t padding
                         );
    
    static void AvgPool2DBackward(
                                  const float *grad_output,
                                  float *grad_input,
                                  const uint32_t batch,
                                  const uint32_t in_h,
                                  const uint32_t in_w,
                                  const uint32_t out_h,
                                  const uint32_t out_w,
                                  const uint32_t channels,
                                  const uint32_t pool_size,
                                  const uint32_t stride,
                                  const uint32_t padding
                                 );
    
    //--------------------------------------------------------------------------
    // CAIF_Tensor convenience wrappers
    //--------------------------------------------------------------------------
    
    static CAIF_Tensor MaxPool2D(
                                const CAIF_Tensor &input,
                                const uint32_t pool_size,
                                const uint32_t stride,
                                const uint32_t padding,
                                CAIF_Tensor *indices=nullptr
                               );
    
    static CAIF_Tensor MaxPool2DBackward(
                                        const CAIF_Tensor &grad_output,
                                        const CAIF_Tensor &indices,
                                        const CAIF_Tensor &input,
                                        const uint32_t pool_size,
                                        const uint32_t stride,
                                        const uint32_t padding
                                       );
    
    static CAIF_Tensor AvgPool2D(
                                const CAIF_Tensor &input,
                                const uint32_t pool_size,
                                const uint32_t stride,
                                const uint32_t padding
                               );
    
    static CAIF_Tensor AvgPool2DBackward(
                                        const CAIF_Tensor &grad_output,
                                        const std::vector<uint32_t> &input_shape,
                                        const uint32_t pool_size,
                                        const uint32_t stride,
                                        const uint32_t padding
                                       );
};

}//end instance namespace

#endif  // CAIF_POOLING_H
