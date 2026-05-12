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

# Build each test_*.cpp as a separate executable linked against caif.
#
# All include directories and link libraries propagate from the CAIF
# library target via PUBLIC dependencies — no need to duplicate them here.

file(GLOB TEST_SOURCES test_*.cpp)

# Prefer static lib if available, otherwise shared
if(TARGET caif_static)
  set(CAIF_TEST_LIB caif_static)
elseif(TARGET caif_shared)
  set(CAIF_TEST_LIB caif_shared)
else()
  message(FATAL_ERROR "No CAIF library target available for tests")
endif()

# Test sources include CAIF headers without the caif/ prefix
# (e.g. #include "caif_tensor.h" rather than "caif/caif_tensor.h").
# The library exposes include/ as PUBLIC (for the caif/ prefix path) and
# include/caif/ as PRIVATE (internal shorthand).  Tests need the shorthand.
set(CAIF_TEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/caif)

# Shared test support objects compiled once and linked into every test.
# Contains CAIF_TestHarness (Report / Pass-Fail counters / FinalExitCode)
# and CAIF_GradCheck (finite-difference gradient verifier).
set(CAIF_TEST_SUPPORT_SOURCES
  caif_test_harness.cpp
  caif_gradcheck.cpp
)
add_library(caif_test_support STATIC ${CAIF_TEST_SUPPORT_SOURCES})
target_link_libraries(caif_test_support PUBLIC ${CAIF_TEST_LIB})
target_include_directories(caif_test_support PUBLIC ${CAIF_TEST_INCLUDE_DIR})

foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE caif_test_support)
endforeach()

# run_all_tests master runner
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/run_all_tests.cpp)
  add_executable(run_all_tests run_all_tests.cpp)
  target_link_libraries(run_all_tests PRIVATE caif_test_support)
endif()
