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
// Test-suite constants. One source of truth; no magic numbers in test files.
//------------------------------------------------------------------------------
#ifndef CAIF_TEST_CONSTANTS_H
#define CAIF_TEST_CONSTANTS_H

#include <cstdint>

namespace instance
{

constexpr float g_caif_tol_fp32_elementwise=5.0e-5f;
constexpr float g_caif_tol_fp32_matmul_same_loc=1.0e-4f;
constexpr float g_caif_tol_fp32_matmul_cross_loc=2.0e-3f;
constexpr float g_caif_tol_fp32_softmax=1.0e-4f;
constexpr float g_caif_tol_fp32_norm=1.0e-4f;
constexpr float g_caif_tol_fp32_rope=1.0e-4f;
constexpr float g_caif_tol_fd_step=1.0e-3f;
constexpr float g_caif_tol_gradcheck_rel=5.0e-3f;
constexpr float g_caif_tol_fp32_rel=5.0e-5f;
constexpr float g_caif_tol_shape_identity=1.0e-6f;
constexpr float g_caif_tol_gradcheck_abs_floor=1.0e-5f;

// MoE frozen-expert test constants.
constexpr uint32_t g_caif_moe_fexp_test_input_dim=16;
constexpr uint32_t g_caif_moe_fexp_test_hidden_dim=32;
constexpr uint32_t g_caif_moe_fexp_test_batch=2;

constexpr uint32_t g_caif_moe_fexp_seed_cell_ff=101u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_fh=111u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_fb=121u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_hf=131u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_hh=141u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_hb=151u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_bf=161u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_bh=171u;
constexpr uint32_t g_caif_moe_fexp_seed_cell_bb=181u;
constexpr uint32_t g_caif_moe_fexp_seed_parity_ff=201u;
constexpr uint32_t g_caif_moe_fexp_seed_parity_fh=211u;
constexpr uint32_t g_caif_moe_fexp_seed_parity_hh=221u;
constexpr uint32_t g_caif_moe_fexp_seed_parity_bb=231u;

constexpr float g_caif_moe_fexp_parity_tol=1.0e-3f;
constexpr float g_caif_moe_fexp_weight_init_lo=-0.5f;
constexpr float g_caif_moe_fexp_weight_init_hi=0.5f;

constexpr uint32_t g_caif_moe_fexp_seed_offset_gate=1u;
constexpr uint32_t g_caif_moe_fexp_seed_offset_up=2u;
constexpr uint32_t g_caif_moe_fexp_seed_offset_down=3u;
constexpr uint32_t g_caif_moe_fexp_seed_offset_input_mul=7u;
constexpr uint32_t g_caif_moe_fexp_seed_offset_input_add=11u;
constexpr uint32_t g_caif_moe_fexp_seed_offset_parity_mul=13u;
constexpr uint32_t g_caif_moe_fexp_seed_offset_parity_add=5u;

// MoE router biased-init test constants (Phase 8.5.E).
constexpr uint32_t g_caif_moe_init_favor_test_tokens=8;
constexpr uint32_t g_caif_moe_init_favor_test_dim=8;
constexpr uint32_t g_caif_moe_init_favor_test_experts=4;
constexpr uint32_t g_caif_moe_init_favor_test_topk=2;
constexpr uint32_t g_caif_moe_init_favor_target_expert=0u;
constexpr float g_caif_moe_init_favor_bias_magnitude=5.0f;
// softmax(5,0,0,0) = e^5 / (e^5+3) = 148.4/151.4 ≈ 0.9802; bumping
// magnitude to 5.0 yields >0.98 on the favored expert and <0.01 each on
// the others. Using a slightly looser threshold than 0.99 (which the
// plan suggests) so the test isn't brittle to fp16/bf16 storage drift.
constexpr float g_caif_moe_init_favor_min_prob=0.97f;
constexpr float g_caif_moe_init_favor_max_other_prob=0.02f;
constexpr float g_caif_moe_init_favor_input_value=0.1f;

// MoE mixed-size experts test constants (Phase 8.5.D).
constexpr uint32_t g_caif_moe_mixed_test_input_dim=32;
constexpr uint32_t g_caif_moe_mixed_test_num_tokens=4;
constexpr uint32_t g_caif_moe_mixed_test_topk=2;
constexpr float g_caif_moe_mixed_test_capacity_factor=2.0f;
constexpr float g_caif_moe_mixed_test_input_value=0.05f;
constexpr float g_caif_moe_mixed_test_grad_value=1.0f;
// Per-expert hidden dims chosen to be visibly different so a uniform
// fallback (e.g. all sized to max) would over-allocate.
constexpr uint32_t g_caif_moe_mixed_test_hidden_a=32u;
constexpr uint32_t g_caif_moe_mixed_test_hidden_b=16u;
constexpr uint32_t g_caif_moe_mixed_test_hidden_c=8u;

}//end instance namespace

#endif
