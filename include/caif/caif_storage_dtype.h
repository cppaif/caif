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
// Compile-time mapping between C++ storage types and CAIF_DataType_e.
//
// This header declares the PRIMARY template only. The fp32 / fp16 / bf16
// specializations live in separate headers per the one-class-per-header
// rule:
//   caif_storage_dtype_float.h
//   caif_storage_dtype_half.h
//   caif_storage_dtype_bfloat16.h
//
// Consumers that need any specialization include the matching specialization
// header (or all three via caif_storage_dtype_all.h, which is the
// convenience aggregator). The primary template here has no `Value` member
// on purpose — using an unspecialized type triggers a compile error,
// which is the desired behavior.
//------------------------------------------------------------------------------
#pragma once

#include "caif_data_type.h"

namespace instance
{

template<typename T>
class CAIF_StorageDtype_t
{
  public:
    CAIF_StorageDtype_t()=delete;
    ~CAIF_StorageDtype_t()=delete;

  protected:

  private:
};

}//end instance namespace
