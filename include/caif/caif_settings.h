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
 * @file aif_settings.h
 * @brief Global runtime configuration for CAIF framework
 */

#pragma once

#include "caif_tensor_backend.h"

namespace instance
{
  /**
   * @brief Static configuration flags for CAIF runtime behaviour.
   *
   * External programs (like Solve) may set these flags once at process
   * startup instead of relying on environment variables.
   */
  class CAIF_Settings
  {
    public:
      /**
       * @brief Enable or disable verbose training diagnostics.
       */
      static void SetTrainLog(const bool enabled){g_train_log=enabled;}

      /**
       * @brief Query whether verbose training diagnostics are enabled.
       */
      static bool TrainLog(){return g_train_log;}

      /**
       * @brief Enable or disable activation-aware initialization.
       */
      static void SetActivationAwareInit(const bool enabled){g_activation_aware_init=enabled;}

      /**
       * @brief Query whether activation-aware initialization is enabled.
       */
      static bool ActivationAwareInit(){return g_activation_aware_init;}

      /**
       * @brief Force a specific backend for the current process.
       *
       * Set to BackendType_e::Auto to allow normal auto-selection.
       */
      static void SetBackendOverride(const CAIF_TensorBackend::BackendType_e backend){g_backend_override=backend;}

      /**
       * @brief Retrieve configured backend override.
       */
      static CAIF_TensorBackend::BackendType_e BackendOverride(){return g_backend_override;}

    private:
      static bool g_train_log;
      static bool g_activation_aware_init;
      static CAIF_TensorBackend::BackendType_e g_backend_override;
  };
}//end instance namespace

