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
// Shared gradient-check mode for the device backward tests. Consolidates the
// per-file `struct GradMode_t` + `kGradMode*` copies that used to live in
// test_device_attention/ffn/gqa/rope/transformer_block/transformer_model.
//
// `tol` is applied relative to |analytical| for values > 1, absolute
// otherwise: diff_tol = tol * max(1, |analytical|).
//
// `fd_h` is the finite-difference step. TF32's reduced mantissa precision
// causes catastrophic cancellation in (sum_plus - sum_minus) when h is small
// relative to accumulated TF32 noise across a full transformer block
// (RMSNorm+MHA+residual+RMSNorm+FFN), so the TF32 mode uses a larger step to
// keep signal above the noise floor. Tests that do not exercise a full block
// (single-op backward checks) simply ignore fd_h.
//------------------------------------------------------------------------------
#pragma once

#include <string>

namespace instance
{

struct GradMode_t
{
  public:
    GradMode_t(const bool precise,
               const float tol,
               const float fd_h,
               const std::string &label):_precise(precise),
                                         _tol(tol),
                                         _fd_h(fd_h),
                                         _label(label){}

    bool Precise()const{return _precise;}
    float Tol()const{return _tol;}
    float FdH()const{return _fd_h;}
    const std::string &Label()const{return _label;}

  private:
    bool _precise;
    float _tol;
    float _fd_h;
    std::string _label;
};

inline const GradMode_t g_caif_grad_mode_precise(true,8e-2f,1e-3f,"Precise");
inline const GradMode_t g_caif_grad_mode_tf32(false,1.5e-1f,1e-2f,"TF32");

}//end instance namespace
