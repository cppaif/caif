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
// CAIF - Per-tensor / per-layer offload policy.
//
// Tells offload-aware caif primitives where the canonical authoritative copy
// of a tensor lives. The default (`GpuResident_e`) preserves all current
// behavior — every existing layer, bench, and test runs unchanged. Opt-in
// via `HostPinned_e` makes host pinned RAM authoritative; the GPU copy
// becomes a transient scratch populated by `Prefetch(stream)` and freed by
// `Evict()`. Reserved values (`Nvme_e`, `Unified_e`) are placeholders for
// later tiers (ZeRO-Infinity-style disk paging, CUDA Unified Memory).
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace instance
{

class CAIF_OffloadPolicy
{
  public:
    CAIF_OffloadPolicy()=delete;
    ~CAIF_OffloadPolicy()=delete;

    enum class CAIF_OffloadPolicy_e:uint8_t
    {
      GpuResident_e=0,
      HostPinned_e=1
    };

  protected:

  private:
};

}//end instance namespace