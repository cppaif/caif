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
 * @file aif_optimizer_ops.h
 * @brief High-performance fused optimizer operations
 */

#ifndef CAIF_OPTIMIZER_OPS_H
#define CAIF_OPTIMIZER_OPS_H

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
 * @brief Static class for fused optimizer update operations
 */
class CAIF_OptimizerOps
{
  public:
    CAIF_OptimizerOps()=delete;
    
    //--------------------------------------------------------------------------
    // Adam Optimizer
    //--------------------------------------------------------------------------
    
    /**
     * @brief Fused Adam update on raw data pointers
     */
    static void AdamUpdate(
                           float *param,
                           const float *grad,
                           float *m,
                           float *v,
                           const size_t n,
                           const float lr,
                           const float beta1,
                           const float beta2,
                           const float epsilon,
                           const float weight_decay,
                           const float bias_correction1,
                           const float bias_correction2
                          )
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          float g=grad[i];
          if(weight_decay>0.0f)
          {
            g+=weight_decay*param[i];
          }
          
          m[i]=beta1*m[i]+(1.0f-beta1)*g;
          v[i]=beta2*v[i]+(1.0f-beta2)*g*g;
          
          const float m_hat=m[i]/bias_correction1;
          const float v_hat=v[i]/bias_correction2;
          
          param[i]-=lr*m_hat/(std::sqrt(v_hat)+epsilon);
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Fused Adam update on CAIF_Tensor
     */
    static void AdamUpdate(
                           CAIF_Tensor &param,
                           const CAIF_Tensor &grad,
                           CAIF_Tensor &m,
                           CAIF_Tensor &v,
                           const float lr,
                           const float beta1,
                           const float beta2,
                           const float epsilon,
                           const float weight_decay,
                           const float bias_correction1,
                           const float bias_correction2
                          );
    
    //--------------------------------------------------------------------------
    // SGD with Momentum
    //--------------------------------------------------------------------------
    
    /**
     * @brief Fused SGD with momentum update on raw data pointers
     */
    static void SGDMomentumUpdate(
                                  float *param,
                                  const float *grad,
                                  float *velocity,
                                  const size_t n,
                                  const float lr,
                                  const float momentum,
                                  const float weight_decay
                                 )
    {
      try
      {
        #ifdef _OPENMP
        #pragma omp parallel for simd
        #endif
        for(size_t i=0;i<n;++i)
        {
          float g=grad[i];
          if(weight_decay>0.0f)
          {
            g+=weight_decay*param[i];
          }
          
          velocity[i]=momentum*velocity[i]-lr*g;
          param[i]+=velocity[i];
        }
      }
      CCAIF_CATCH_BLOCK()
    }
    
    /**
     * @brief Fused SGD with momentum update on CAIF_Tensor
     */
    static void SGDMomentumUpdate(
                                  CAIF_Tensor &param,
                                  const CAIF_Tensor &grad,
                                  CAIF_Tensor &velocity,
                                  const float lr,
                                  const float momentum,
                                  const float weight_decay
                                 );
};

}//end instance namespace

#endif  // CAIF_OPTIMIZER_OPS_H

