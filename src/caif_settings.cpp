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
 * @file aif_settings.cpp
 * @brief Definition of global runtime configuration flags.
 */

#include "caif_settings.h"

namespace instance
{

bool CAIF_Settings::g_train_log=false;
bool CAIF_Settings::g_activation_aware_init=false;
CAIF_TensorBackend::BackendType_e CAIF_Settings::g_backend_override=CAIF_TensorBackend::BackendType_e::Auto;

}//end instance namespace

