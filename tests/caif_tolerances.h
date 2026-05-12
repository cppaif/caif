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
// Canonical tolerance table for tests. Values sourced from caif_test_constants.h
// and exposed through accessor methods on CAIF_Tolerances. Tightening any value
// is a separate, reviewed commit - not incidental test-file churn.
//------------------------------------------------------------------------------
#ifndef CAIF_TOLERANCES_H
#define CAIF_TOLERANCES_H

#include "caif_test_constants.h"

namespace instance
{

class CAIF_Tolerances
{
  public:
    static float Fp32Elementwise(){return g_caif_tol_fp32_elementwise;}
    static float Fp32MatmulSameLoc(){return g_caif_tol_fp32_matmul_same_loc;}
    static float Fp32MatmulCrossLoc(){return g_caif_tol_fp32_matmul_cross_loc;}
    static float Fp32Softmax(){return g_caif_tol_fp32_softmax;}
    static float Fp32Norm(){return g_caif_tol_fp32_norm;}
    static float Fp32Rope(){return g_caif_tol_fp32_rope;}
    static float FdStep(){return g_caif_tol_fd_step;}
    static float GradcheckRel(){return g_caif_tol_gradcheck_rel;}
    static float Fp32Rel(){return g_caif_tol_fp32_rel;}
    static float ShapeIdentity(){return g_caif_tol_shape_identity;}
    static float GradcheckAbsFloor(){return g_caif_tol_gradcheck_abs_floor;}

  protected:
  private:
};

}//end instance namespace

#endif
