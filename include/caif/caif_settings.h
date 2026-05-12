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
 * @file caif_settings.h
 * @brief Global runtime configuration for the CAIF framework.
 *
 * ============================================================================
 *                       FP32 MATMUL-PRECISION MODES
 * ============================================================================
 *
 * `MatmulMode_e` is the *authoritative* control for CAIF's FP32 matrix-
 * multiply precision. It replaces the older boolean `PreciseGradients`
 * (which only toggled the backward pass) with a symmetric, explicitly-
 * named regime that applies to BOTH forward and backward passes.
 *
 * ---------------------------------------------------------------------------
 * WHY TWO MODES
 * ---------------------------------------------------------------------------
 *
 * Ampere+ hardware exposes two GEMM code paths for FP32 inputs:
 *
 *   1. TF32 ("TensorFloat-32"): inputs are rounded to a 10-bit mantissa
 *      before multiplication; accumulation is in FP32. This uses the
 *      tensor cores and is the substantially faster path on Ampere+
 *      hardware. Relative error is on the order of 1e-3.
 *
 *   2. Full FP32 ("IEEE single"): inputs retain all 23 mantissa bits;
 *      multiplication and accumulation are both in FP32. This uses the
 *      CUDA cores. Relative error is on the order of 1e-7.
 *
 * For production inference and most training, TF32's error is entirely
 * invisible — SGD/Adam stochastic noise is already well above 1e-3 per
 * step, so per-GEMM matmul noise at that level is absorbed. For
 * finite-difference gradient checks, however, TF32 is catastrophic: the
 * classic central-difference `(f(x+h) - f(x-h)) / (2h)` with h=1e-3
 * amplifies a 1e-3 forward error into an O(1) relative backward error.
 *
 * The two modes exist so the same library binary can serve both:
 * timing runs use TF32 (fast, realistic), gradient-correctness tests
 * use full FP32 (accurate enough to validate analytical backward
 * formulas).
 *
 * ---------------------------------------------------------------------------
 * MODE DEFINITIONS
 * ---------------------------------------------------------------------------
 *
 *   Performance_e  (default for benchmarks and production training)
 *   --------------
 *     CAIF side:
 *       - Every FP32-input cuBLAS GEMM uses CUBLAS_COMPUTE_32F_FAST_TF32,
 *         forward AND backward symmetrically. Selection happens in exactly
 *         one place: `CAIF_RunContext::ComputeTypeFor(dt)`.
 *       - FP16/BF16 inputs are unaffected — they always use
 *         CUBLAS_COMPUTE_32F (FP32 accumulate) regardless of this flag.
 *       - cuDNN convolutions respect the same policy via their own allow-
 *         TF32 hint path.
 *     Numerical contract:
 *       - ~10-bit mantissa on FP32 inputs during multiply.
 *       - ~1e-3 relative error per GEMM.
 *       - FD gradient checks DO NOT converge at this precision.
 *     Use cases:
 *       - All performance-oriented timing runs.
 *       - Production training runs where the small loss-trajectory
 *         delta vs full FP32 is dominated by SGD/Adam noise and batch
 *         sampling variance.
 *       - All inference (inference performs no FD, and downstream task
 *         metrics are unaffected by TF32 at this scale).
 *
 *   Accuracy_e  (default for gradient-correctness tests)
 *   ----------
 *     CAIF side:
 *       - Every FP32-input cuBLAS GEMM uses CUBLAS_COMPUTE_32F, forward
 *         AND backward symmetrically.
 *       - FP16/BF16 inputs continue to use CUBLAS_COMPUTE_32F as before.
 *       - cuDNN TF32 is disabled.
 *     Numerical contract:
 *       - Full 23-bit mantissa on FP32 inputs.
 *       - ~1e-7 relative error per GEMM.
 *       - FD gradient checks converge to canonical tolerances.
 *     Use cases:
 *       - `tests/test_device_*` gradient-check harnesses.
 *       - Reference-implementation parity investigations where the
 *         reference is also configured for full FP32.
 *       - Any timing run where bit-exact reproducibility of FP32 arithmetic
 *         is required across hardware generations.
 *
 * ---------------------------------------------------------------------------
 * SYMMETRY (critical invariant)
 * ---------------------------------------------------------------------------
 *
 * This setting is applied symmetrically across pass direction. The
 * previous `PreciseGradients` boolean only upgraded the BACKWARD pass to
 * full FP32 while leaving the forward in TF32. That split design produced
 * two distinct bugs:
 *
 *   (1) Perf regression: with `PreciseGradients=true` as the default
 *       and `CAIF_RunContext::Pass_e::Backward_e` set uniformly during
 *       the backward walk, every backward GEMM flipped to full FP32 —
 *       a substantial across-the-board slowdown.
 *
 *   (2) Silent accuracy mismatch: forward activations computed in TF32
 *       are then consumed by a backward that assumes its own FP32 inputs
 *       are canonical. The 1e-3 forward drift shows up as a consistent
 *       analytical-gradient bias that no amount of tolerance-tightening
 *       can fix.
 *
 * The symmetric contract here eliminates both. The two modes are the
 * only two valid configurations; no code path mixes them.
 *
 * ---------------------------------------------------------------------------
 * IMPACT SUMMARY
 * ---------------------------------------------------------------------------
 *
 *   Surface                  | Performance_e      | Accuracy_e
 *   -------------------------|--------------------|---------------------
 *   cuBLAS compute type      | 32F_FAST_TF32      | 32F
 *   cuDNN TF32               | allowed            | disabled
 *   FP16/BF16 GEMMs          | 32F accumulate     | 32F accumulate
 *   Integer/bool ops         | unaffected         | unaffected
 *   Per-GEMM rel error       | ~1e-3              | ~1e-7
 *   FD gradient checks       | fail               | pass
 *   Training loss trajectory | small drift        | baseline
 *   Inference output         | indistinguishable  | indistinguishable
 *   End-to-end throughput    | substantially up   | baseline (slowest)
 *
 * ---------------------------------------------------------------------------
 * PRE-AMPERE BEHAVIOR
 * ---------------------------------------------------------------------------
 *
 * On hardware without tensor-core TF32 support (Volta, Pascal, Turing),
 * cuBLAS falls back to full FP32 regardless of this flag. The setting is
 * still honored — `MatmulMode_e::Performance_e` is simply a no-op on those
 * devices. No error is raised. Benchmarks on legacy hardware will show
 * identical numbers across the two modes; this is expected.
 *
 * ---------------------------------------------------------------------------
 * SELECTING A MODE
 * ---------------------------------------------------------------------------
 *
 * A caller chooses the mode once at process start with
 *
 *   CAIF_Settings::Instance().SetMatmulMode(MatmulMode_e::Performance_e);
 *   CAIF_Settings::Instance().SetMatmulMode(MatmulMode_e::Accuracy_e);
 *
 * and the choice is honored by every `CAIF_Ops::MatMul` call thereafter.
 * Timing harnesses should record the mode alongside their result so perf
 * and accuracy runs are never compared against each other.
 *
 * ---------------------------------------------------------------------------
 * WHERE THE DECISION IS MADE (SINGLE SOURCE OF TRUTH)
 * ---------------------------------------------------------------------------
 *
 * `CAIF_RunContext::ComputeTypeFor(CAIF_DataType::CAIF_DataType_e dt)` is
 * the one and only call site that translates this flag into a cuBLAS
 * compute type. Op implementations never read `CAIF_Settings` directly.
 * This keeps the policy in one place and makes it mechanically auditable.
 * ============================================================================
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace instance
{
  class CAIF_Settings
  {
    public:
      enum class MatmulMode_e:uint32_t
      {
        Performance_e=0,  // TF32 tensor cores on FP32 GEMMs, both passes
        Accuracy_e=1      // Full IEEE FP32 on FP32 GEMMs, both passes
      };

      /**
       * @brief Enable or disable verbose training diagnostics.
       */
      static void SetTrainLog(const bool enabled){g_train_log=enabled;}
      static bool TrainLog(){return g_train_log;}

      /**
       * @brief Enable or disable activation-aware initialization.
       */
      static void SetActivationAwareInit(const bool enabled){g_activation_aware_init=enabled;}
      static bool ActivationAwareInit(){return g_activation_aware_init;}

      /**
       * @brief Select the FP32 matmul precision regime.
       *
       * This is the authoritative, symmetric control for FP32 matmul
       * precision. See the file-level docblock above for the full
       * contract and impact summary. Quick reference:
       *
       *   Performance_e -> CUBLAS_COMPUTE_32F_FAST_TF32 (TF32 tensor cores)
       *                    ~1e-3 rel error; FD gradchecks will NOT converge.
       *                    Use for all timing runs and production training.
       *
       *   Accuracy_e    -> CUBLAS_COMPUTE_32F (full IEEE FP32)
       *                    ~1e-7 rel error; FD gradchecks converge.
       *                    Use for test harness and numerical-parity work.
       *
       * Symmetry: applied to BOTH forward and backward passes. Never split.
       *
       * Scope:
       *   - FP32-input cuBLAS GEMMs: controlled by this flag.
       *   - FP16/BF16 GEMMs: always CUBLAS_COMPUTE_32F (FP32 accumulate),
       *     unaffected.
       *   - Integer/bool ops: unaffected.
       *   - cuDNN convolutions: respect the same policy via their
       *     allow-TF32 hint.
       *
       * Single source of truth: `CAIF_RunContext::ComputeTypeFor(dt)` is
       * the only call site that reads this flag. Op implementations
       * must NEVER read `CAIF_Settings::MatmulMode()` directly.
       *
       * Pre-Ampere: no-op (cuBLAS falls back to full FP32 regardless).
       *
       * Default: `Accuracy_e` (test-suite-safe; bench binaries explicitly
       * call `ApplyMode(Performance_e)` at startup).
       */
      static void SetMatmulMode(const MatmulMode_e mode){g_matmul_mode=mode;}

      /**
       * @brief Read the current FP32 matmul precision regime.
       *
       * Returned value governs every FP32-input GEMM and every cuDNN
       * allow-TF32 decision for the duration of the call. See
       * `SetMatmulMode` above and the file-level docblock for the full
       * contract.
       */
      static MatmulMode_e MatmulMode(){return g_matmul_mode;}

      /**
       * @brief Back-compat read alias for `MatmulMode`.
       *
       * Returns `true` iff the current mode is `Accuracy_e`. Provided so
       * legacy test-harness and gradcheck code that still gates FD
       * perturbation windows on a boolean can continue to work during
       * the rearchitecture transition.
       *
       * DO NOT introduce new call sites. New code reads `MatmulMode()`
       * directly and switches on the enum — the name `PreciseGradients`
       * is now misleading because the flag is symmetric across passes,
       * not backward-only as the old field implied.
       *
       * This alias will be removed once the test harness migrates to
       * `MatmulMode()` (Stage 6 test-harness consolidation).
       */
      static bool PreciseGradients(){return g_matmul_mode==MatmulMode_e::Accuracy_e;}

      /**
       * @brief Back-compat write alias for `SetMatmulMode`.
       *
       * `true`  -> `Accuracy_e`   (full FP32 both passes)
       * `false` -> `Performance_e` (TF32 both passes)
       *
       * Exists purely to let pre-rearchitecture call sites
       * (`SetPreciseGradients(true/false)` scattered across tests)
       * compile unchanged while the new enum-based API is rolled out.
       * Every new call site must use `SetMatmulMode(MatmulMode_e::...)`.
       *
       * Like `PreciseGradients()`, this alias will be removed at the end
       * of the Stage 6 test-harness consolidation.
       */
      static void SetPreciseGradients(const bool enabled)
      {
        if(enabled==true)
        {
          g_matmul_mode=MatmulMode_e::Accuracy_e;
        }
        else
        {
          g_matmul_mode=MatmulMode_e::Performance_e;
        }
      }

      /**
       * @brief Override the cuBLAS-Lt workspace size (bytes).
       *
       * Must be called BEFORE `CAIF_DeviceContext::Initialize()` runs
       * (i.e. before the first cuBLAS-Lt handle access). The override
       * is validated at Initialize() time against free VRAM and the
       * configured safety divisor; an oversized request throws.
       *
       * `bytes==0` (the default) selects auto-detection from the active
       * GPU's compute capability. Non-zero bypasses auto-detection and
       * uses the literal value.
       *
       * Typical call:
       *   CAIF_Settings::SetCublasLtWorkspaceBytes(64ULL*1024ULL*1024ULL);
       */
      static void SetCublasLtWorkspaceBytes(const size_t bytes)
      {
        g_cublaslt_workspace_bytes=bytes;
      }

      /**
       * @brief Read the current cuBLAS-Lt workspace override (bytes).
       *
       * Returns 0 when no override is set (auto-detect mode).
       */
      static size_t CublasLtWorkspaceBytes(){return g_cublaslt_workspace_bytes;}

    private:
      static bool g_train_log;
      static bool g_activation_aware_init;
      static MatmulMode_e g_matmul_mode;
      static size_t g_cublaslt_workspace_bytes;
  };
}//end instance namespace
