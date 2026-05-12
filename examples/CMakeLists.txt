# Copyright 2026 Eric Malloy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CAIF example programs (CUDA required)

# Prefer static lib if available, otherwise shared
if(TARGET caif_static)
  set(CAIF_EXAMPLE_LIB caif_static)
elseif(TARGET caif_shared)
  set(CAIF_EXAMPLE_LIB caif_shared)
else()
  message(FATAL_ERROR "No CAIF library target available for examples")
endif()

# Examples include CAIF headers without the caif/ prefix
set(CAIF_EXAMPLE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/caif)

# Train a small transformer from scratch
add_executable(train_transformer train_transformer.cpp)
target_link_libraries(train_transformer PRIVATE ${CAIF_EXAMPLE_LIB})
target_include_directories(train_transformer PRIVATE ${CAIF_EXAMPLE_INCLUDE_DIR})

# Fine-tune with LoRA adapters
add_executable(finetune_lora finetune_lora.cpp)
target_link_libraries(finetune_lora PRIVATE ${CAIF_EXAMPLE_LIB})
target_include_directories(finetune_lora PRIVATE ${CAIF_EXAMPLE_INCLUDE_DIR})

# Multi-head attention with QK-norm + partial-rotary RoPE
add_executable(mha_qknorm_partial_rope mha_qknorm_partial_rope.cpp)
target_link_libraries(mha_qknorm_partial_rope PRIVATE ${CAIF_EXAMPLE_LIB})
target_include_directories(mha_qknorm_partial_rope PRIVATE ${CAIF_EXAMPLE_INCLUDE_DIR})

# MoE router norm_topk_prob toggle (OLMoE-style raw softmax vs HF normalised)
add_executable(moe_router_norm_topk_prob moe_router_norm_topk_prob.cpp)
target_link_libraries(moe_router_norm_topk_prob PRIVATE ${CAIF_EXAMPLE_LIB})
target_include_directories(moe_router_norm_topk_prob PRIVATE ${CAIF_EXAMPLE_INCLUDE_DIR})

# MoE-extension via CAIF_DeviceMoEFrozenExpert (frozen pretrained expert wrapper)
add_executable(moe_frozen_expert_extension moe_frozen_expert_extension.cpp)
target_link_libraries(moe_frozen_expert_extension PRIVATE ${CAIF_EXAMPLE_LIB})
target_include_directories(moe_frozen_expert_extension PRIVATE ${CAIF_EXAMPLE_INCLUDE_DIR})

# Inference with int8 + int4 quantized FrozenLinear (fp32 reference for error comparison)
add_executable(inference_quantized inference_quantized.cpp)
target_link_libraries(inference_quantized PRIVATE ${CAIF_EXAMPLE_LIB})
target_include_directories(inference_quantized PRIVATE ${CAIF_EXAMPLE_INCLUDE_DIR})
