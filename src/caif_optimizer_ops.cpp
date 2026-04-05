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
 * @file aif_optimizer_ops.cpp
 * @brief Implementation of fused optimizer operations for CAIF_Tensor
 */

#include "caif_optimizer_ops.h"
#include "caif_framework.h"

using namespace instance;

void CAIF_OptimizerOps::AdamUpdate(
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
                                 )
{
  try
  {
    AdamUpdate(
               param.MutableData<float>(),
               grad.ConstData<float>(),
               m.MutableData<float>(),
               v.MutableData<float>(),
               param.NumElements(),
               lr,
               beta1,
               beta2,
               epsilon,
               weight_decay,
               bias_correction1,
               bias_correction2
              );
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_OptimizerOps::SGDMomentumUpdate(
                                         CAIF_Tensor &param,
                                         const CAIF_Tensor &grad,
                                         CAIF_Tensor &velocity,
                                         const float lr,
                                         const float momentum,
                                         const float weight_decay
                                        )
{
  try
  {
    SGDMomentumUpdate(
                      param.MutableData<float>(),
                      grad.ConstData<float>(),
                      velocity.MutableData<float>(),
                      param.NumElements(),
                      lr,
                      momentum,
                      weight_decay
                     );
  }
  CAIF_CATCH_BLOCK()
}

