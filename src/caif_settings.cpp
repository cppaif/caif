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

#include "caif_settings.h"

namespace instance
{

bool CAIF_Settings::g_train_log=false;
bool CAIF_Settings::g_activation_aware_init=false;
// Default regime: Accuracy_e (full FP32 both passes).
// Conservative default that keeps gradient-test behaviour identical
// to the pre-rearch baseline's network-level backward pass. Timing
// runs should opt into Performance_e explicitly.
CAIF_Settings::MatmulMode_e CAIF_Settings::g_matmul_mode=CAIF_Settings::MatmulMode_e::Accuracy_e;
// cuBLAS-Lt workspace override (bytes). Zero selects auto-detection from
// the active GPU's compute capability at CAIF_DeviceContext::Initialize().
size_t CAIF_Settings::g_cublaslt_workspace_bytes=0;

}//end instance namespace
