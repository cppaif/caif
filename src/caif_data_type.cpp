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

#include "caif_data_type.h"
#include "caif_serialization_constants.h"

namespace instance
{

std::string CAIF_DataType::Name()const
{
  switch(Value())
  {
    case CAIF_DataType_e::Float32:
      return g_serial_dtype_name_fp32;
    case CAIF_DataType_e::Float64:
      return g_serial_dtype_name_fp64;
    case CAIF_DataType_e::Float16:
      return g_serial_dtype_name_fp16;
    case CAIF_DataType_e::BFloat16:
      return g_serial_dtype_name_bf16;
    case CAIF_DataType_e::Int4:
      return g_serial_dtype_name_int4;
    case CAIF_DataType_e::Int8:
      return g_serial_dtype_name_int8;
    case CAIF_DataType_e::Int16:
      return g_serial_dtype_name_int16;
    case CAIF_DataType_e::Int32:
      return g_serial_dtype_name_int32;
    case CAIF_DataType_e::Int64:
      return g_serial_dtype_name_int64;
    case CAIF_DataType_e::UInt8:
      return g_serial_dtype_name_uint8;
    case CAIF_DataType_e::UInt16:
      return g_serial_dtype_name_uint16;
    case CAIF_DataType_e::UInt32:
      return g_serial_dtype_name_uint32;
    case CAIF_DataType_e::UInt64:
      return g_serial_dtype_name_uint64;
    case CAIF_DataType_e::Bool:
      return g_serial_dtype_name_bool;
  }
  return g_serial_dtype_name_unknown;
}

}//end instance namespace
