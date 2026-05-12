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
// caif_int4_packed_t — distinct C++ storage type for INT4-packed weights.
//
// CUDA / C++ have no native 4-bit integer. CAIF stores INT4 weights two
// elements per byte (low nibble first, high nibble second). This struct
// is a one-byte-wide tag type used as a template StorageT parameter so
// the templated layer machinery (CAIF_DeviceLayerTyped, FrozenLinear,
// kernel launchers) can distinguish INT4 storage from INT8 / fp32 / fp16
// / bf16 cells at compile time.
//
// The struct intentionally exposes nothing beyond its packed byte; the
// element-extraction / element-pack logic lives in the kernel side
// (launch_dequantize_int4 / launch_quantize_to_int4) and is selected
// by `if constexpr(std::is_same_v<StorageT, caif_int4_packed_t>==true)`
// in the layer dispatch path.
//
// Sized one byte so allocation arithmetic is straightforward:
//     bytes = (num_elements + 1) / 2
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace instance
{

struct caif_int4_packed_t
{
  public:
    caif_int4_packed_t()=default;
    ~caif_int4_packed_t()=default;

    explicit caif_int4_packed_t(const uint8_t packed):_packed(packed)
    {
    }

    uint8_t Packed()const{return _packed;}
    void SetPacked(const uint8_t packed){_packed=packed;}

  private:
    uint8_t _packed=0;
};

}//end instance namespace
