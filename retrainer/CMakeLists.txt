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

# ===========================================================================
#
#  Retrainer — CAIF-based LLM fine-tuning tool
#
#  This builds the retrainer executable, which fine-tunes transformer models
#  using LoRA adapters with CUDA acceleration.
#
#  Requires CAIF to be built with CUDA support (CAIF_BUILD_CUDA=ON).
#
# ===========================================================================

# Gather all .cpp sources except main.cpp for the static library
file(GLOB RTNR_LIB_SOURCES src/*.cpp)
list(FILTER RTNR_LIB_SOURCES EXCLUDE REGEX "src/main\\.cpp$")

# Static library: libretrainer.a
add_library(retrainer_lib STATIC ${RTNR_LIB_SOURCES})
set_target_properties(retrainer_lib PROPERTIES OUTPUT_NAME retrainer)

target_include_directories(retrainer_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/retrainer
)

# Link against CAIF (prefer static)
if(TARGET caif_static)
  target_link_libraries(retrainer_lib PUBLIC caif_static)
elseif(TARGET caif_shared)
  target_link_libraries(retrainer_lib PUBLIC caif_shared)
else()
  message(FATAL_ERROR "No CAIF library target available for retrainer")
endif()

# Retrainer executable
add_executable(retrainer src/main.cpp)
target_link_libraries(retrainer PRIVATE retrainer_lib)
target_include_directories(retrainer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/retrainer
)
