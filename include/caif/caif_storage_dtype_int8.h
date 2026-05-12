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
// CAIF_StorageDtype_t<int8_t> — INT8 specialization of the storage-dtype
// trait declared in caif_storage_dtype.h.
//
// Used as the StorageT template parameter for CAIF_DeviceFrozenLinear's
// INT8 cells: <float, int8_t>, <__half, int8_t>, <__nv_bfloat16, int8_t>.
//------------------------------------------------------------------------------
#pragma once

#include "caif_storage_dtype.h"
#include "caif_data_type.h"

#include <cstdint>

namespace instance
{

template<>
class CAIF_StorageDtype_t<int8_t>
{
  public:
    CAIF_StorageDtype_t()=delete;
    ~CAIF_StorageDtype_t()=delete;

    static constexpr CAIF_DataType::CAIF_DataType_e Value=
      CAIF_DataType::CAIF_DataType_e::Int8;

  protected:

  private:
};

}//end instance namespace
