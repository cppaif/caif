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
 * @file aif_batch_ops.h
 * @brief High-performance batch normalization and dropout operations
 */

#ifndef CAIF_BATCH_OPS_H
#define CAIF_BATCH_OPS_H

#include "caif_tensor.h"
#include "caif_exception.h"
#include <cstdint>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace instance
{

class CAIF_Framework;

/**
 * @brief Static class for batch normalization operations
 */
class CAIF_BatchNorm
{
  public:
    CAIF_BatchNorm()=delete;
    
    static void Forward(
                        const float *input,
                        float *output,
                        const float *scale,
                        const float *bias,
                        float *running_mean,
                        float *running_var,
                        float *saved_mean,
                        float *saved_inv_var,
                        const uint32_t batch,
                        const uint32_t features,
                        const float epsilon,
                        const float momentum,
                        const bool training
                       );
    
    static void Backward(
                         const float *grad_output,
                         const float *input,
                         const float *scale,
                         const float *saved_mean,
                         const float *saved_inv_var,
                         float *grad_input,
                         float *grad_scale,
                         float *grad_bias,
                         const uint32_t batch,
                         const uint32_t features,
                         const float epsilon
                        );
    
    static CAIF_Tensor Forward(
                              const CAIF_Tensor &input,
                              const CAIF_Tensor &scale,
                              const CAIF_Tensor &bias,
                              CAIF_Tensor &running_mean,
                              CAIF_Tensor &running_var,
                              const float epsilon,
                              const float momentum,
                              const bool training,
                              CAIF_Tensor &saved_mean,
                              CAIF_Tensor &saved_inv_var
                             );
    
    static CAIF_Tensor Backward(
                               const CAIF_Tensor &grad_output,
                               const CAIF_Tensor &input,
                               const CAIF_Tensor &scale,
                               const CAIF_Tensor &saved_mean,
                               const CAIF_Tensor &saved_inv_var,
                               const float epsilon,
                               CAIF_Tensor &grad_scale,
                               CAIF_Tensor &grad_bias
                              );
};

/**
 * @brief Static class for dropout operations
 */
class CAIF_Dropout
{
  public:
    CAIF_Dropout()=delete;
    
    static void Forward(
                        const float *input,
                        float *output,
                        float *mask,
                        const size_t n,
                        const float dropout_rate,
                        const bool training,
                        const uint32_t seed
                       );
    
    static void Backward(const float *grad_output,const float *mask,float *grad_input,const size_t n,
                         const float dropout_rate);
    
    static CAIF_Tensor Forward(const CAIF_Tensor &input,const float dropout_rate,const bool training,
                              CAIF_Tensor &mask);
    static CAIF_Tensor Backward(const CAIF_Tensor &grad_output,const CAIF_Tensor &mask,const float dropout_rate);
};

}//end instance namespace

#endif  // CAIF_BATCH_OPS_H

