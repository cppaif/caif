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
// CAIF_StorageDtype_t<caif_int4_packed_t> — INT4-packed specialization of
// the storage-dtype trait declared in caif_storage_dtype.h.
//
// Used as the StorageT template parameter for CAIF_DeviceFrozenLinear's
// INT4 cells: <float, caif_int4_packed_t>, <__half, caif_int4_packed_t>,
// <__nv_bfloat16, caif_int4_packed_t>. Storage element size is 1 byte
// (two packed 4-bit values per byte); allocation arithmetic is
// `(num_elements + 1) / 2` bytes — handled by CAIF_DataType::ByteCount()
// for the Int4 enum value resolved here.
//------------------------------------------------------------------------------
#pragma once

#include "caif_storage_dtype.h"
#include "caif_int4_packed_t.h"
#include "caif_data_type.h"

namespace instance
{

template<>
class CAIF_StorageDtype_t<caif_int4_packed_t>
{
  public:
    CAIF_StorageDtype_t()=delete;
    ~CAIF_StorageDtype_t()=delete;

    static constexpr CAIF_DataType::CAIF_DataType_e Value=
      CAIF_DataType::CAIF_DataType_e::Int4;

  protected:

  private:
};

}//end instance namespace
