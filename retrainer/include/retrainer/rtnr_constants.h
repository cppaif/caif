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
 * @file rtnr_constants.h
 * @brief Constants for the Retrainer project
 */

#pragma once

#include <cstdint>

namespace instance
{

// Token file format
constexpr uint32_t g_rtnr_token_magic=0x52544E54;  // "RTNT"
constexpr uint32_t g_rtnr_token_version=1;

// Default training parameters
constexpr float g_rtnr_default_learning_rate=2e-5f;
constexpr uint32_t g_rtnr_default_batch_size=4;
constexpr uint32_t g_rtnr_default_max_seq_len=512;
constexpr uint32_t g_rtnr_default_epochs=3;
constexpr float g_rtnr_default_warmup_ratio=0.1f;
constexpr float g_rtnr_default_weight_decay=0.01f;

// LoRA defaults
constexpr uint32_t g_rtnr_default_lora_r=16;
constexpr uint32_t g_rtnr_default_lora_alpha=32;
constexpr float g_rtnr_default_lora_dropout=0.05f;

// Gradient accumulation
constexpr uint32_t g_rtnr_default_grad_accum_steps=4;

// Checkpointing
constexpr uint32_t g_rtnr_default_checkpoint_interval=1000;

// Logging
constexpr uint32_t g_rtnr_default_log_interval=10;

}
