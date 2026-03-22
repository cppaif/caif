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
 * @file aif_convolution_ops.h
 * @brief High-performance convolution operations
 */

#ifndef CAIF_CONVOLUTION_OPS_H
#define CAIF_CONVOLUTION_OPS_H

#include "caif_tensor.h"
#include "caif_exception.h"
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace instance
{

class CAIF_Framework;

/**
 * @brief Static class for high-performance convolution operations
 */
class CAIF_ConvolutionOps
{
  public:
    CAIF_ConvolutionOps()=delete;
    
    /**
     * @brief 2D Convolution forward pass (NHWC format)
     */
    static void Conv2DForward(
                              const float *input,
                              const float *kernel,
                              float *output,
                              const uint32_t batch,
                              const uint32_t in_h,
                              const uint32_t in_w,
                              const uint32_t in_c,
                              const uint32_t out_c,
                              const uint32_t k_h,
                              const uint32_t k_w,
                              const uint32_t stride_h,
                              const uint32_t stride_w,
                              const uint32_t pad_h,
                              const uint32_t pad_w
                             );
    
    /**
     * @brief 2D Convolution backward pass for input gradients
     */
    static void Conv2DBackwardInput(
                                    const float *grad_output,
                                    const float *kernel,
                                    float *grad_input,
                                    const uint32_t batch,
                                    const uint32_t in_h,
                                    const uint32_t in_w,
                                    const uint32_t in_c,
                                    const uint32_t out_h,
                                    const uint32_t out_w,
                                    const uint32_t out_c,
                                    const uint32_t k_h,
                                    const uint32_t k_w,
                                    const uint32_t stride_h,
                                    const uint32_t stride_w,
                                    const uint32_t pad_h,
                                    const uint32_t pad_w
                                   );
    
    /**
     * @brief 2D Convolution backward pass for weight gradients
     */
    static void Conv2DBackwardWeight(
                                     const float *input,
                                     const float *grad_output,
                                     float *grad_kernel,
                                     const uint32_t batch,
                                     const uint32_t in_h,
                                     const uint32_t in_w,
                                     const uint32_t in_c,
                                     const uint32_t out_h,
                                     const uint32_t out_w,
                                     const uint32_t out_c,
                                     const uint32_t k_h,
                                     const uint32_t k_w,
                                     const uint32_t stride_h,
                                     const uint32_t stride_w,
                                     const uint32_t pad_h,
                                     const uint32_t pad_w
                                    );
    
    //--------------------------------------------------------------------------
    // CAIF_Tensor convenience wrappers
    //--------------------------------------------------------------------------
    
    static CAIF_Tensor Conv2DForward(
                                    const CAIF_Tensor &input,
                                    const CAIF_Tensor &kernel,
                                    const uint32_t stride_h,
                                    const uint32_t stride_w,
                                    const uint32_t pad_h,
                                    const uint32_t pad_w
                                   );
    
    static CAIF_Tensor Conv2DForward(
                                    CAIF_Framework &framework,
                                    const CAIF_Tensor &input,
                                    const CAIF_Tensor &kernel,
                                    const uint32_t stride_h,
                                    const uint32_t stride_w,
                                    const uint32_t pad_h,
                                    const uint32_t pad_w
                                   );
};

}//end instance namespace

#endif  // CAIF_CONVOLUTION_OPS_H

