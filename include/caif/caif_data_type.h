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
// Data type enumeration and utilities
//------------------------------------------------------------------------------
#ifndef CAIF_DATA_TYPE_H
#define CAIF_DATA_TYPE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <type_traits>

namespace instance
{
  class CAIF_DataType
  {
    public:
      enum class CAIF_DataType_e:uint8_t
      {
        Float32,
        Float64,
        Float16,
        BFloat16,
        Int4,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Bool
      };

      CAIF_DataType():_v(CAIF_DataType_e::Float32){}
      explicit CAIF_DataType(const CAIF_DataType_e v):_v(v){}

      size_t ElementSizeBytes()const
      {
        switch(_v)
        {
          case CAIF_DataType_e::Float32:return 4;
          case CAIF_DataType_e::Float64:return 8;
          case CAIF_DataType_e::Float16:return 2;
          case CAIF_DataType_e::BFloat16:return 2;
          case CAIF_DataType_e::Int4:return 1;  // packed: use StorageSizeBytes() for actual size
          case CAIF_DataType_e::Int8:return 1;
          case CAIF_DataType_e::Int16:return 2;
          case CAIF_DataType_e::Int32:return 4;
          case CAIF_DataType_e::Int64:return 8;
          case CAIF_DataType_e::UInt8:return 1;
          case CAIF_DataType_e::UInt16:return 2;
          case CAIF_DataType_e::UInt32:return 4;
          case CAIF_DataType_e::UInt64:return 8;
          case CAIF_DataType_e::Bool:return 1;
        }
        return 0;
      }

      /**
       * @brief Compute actual storage size in bytes for a given number of elements.
       * Handles INT4 packing (2 elements per byte). For all other types,
       * this is simply num_elements * ElementSizeBytes().
       */
      size_t StorageSizeBytes(size_t num_elements)const
      {
        if(_v==CAIF_DataType_e::Int4)
        {
          return (num_elements+1)/2;
        }
        return num_elements*ElementSizeBytes();
      }

      /**
       * @brief Human-readable type name (e.g. "fp32", "bf16", "int4")
       */
      std::string Name()const
      {
        switch(_v)
        {
          case CAIF_DataType_e::Float32:return "fp32";
          case CAIF_DataType_e::Float64:return "fp64";
          case CAIF_DataType_e::Float16:return "fp16";
          case CAIF_DataType_e::BFloat16:return "bf16";
          case CAIF_DataType_e::Int4:return "int4";
          case CAIF_DataType_e::Int8:return "int8";
          case CAIF_DataType_e::Int16:return "int16";
          case CAIF_DataType_e::Int32:return "int32";
          case CAIF_DataType_e::Int64:return "int64";
          case CAIF_DataType_e::UInt8:return "uint8";
          case CAIF_DataType_e::UInt16:return "uint16";
          case CAIF_DataType_e::UInt32:return "uint32";
          case CAIF_DataType_e::UInt64:return "uint64";
          case CAIF_DataType_e::Bool:return "bool";
        }
        return "unknown";
      }

      /**
       * @brief SafeTensors dtype string (e.g. "F32", "BF16", "I8")
       */
      std::string SafeTensorsName()const
      {
        switch(_v)
        {
          case CAIF_DataType_e::Float32:return "F32";
          case CAIF_DataType_e::Float64:return "F64";
          case CAIF_DataType_e::Float16:return "F16";
          case CAIF_DataType_e::BFloat16:return "BF16";
          case CAIF_DataType_e::Int8:return "I8";
          case CAIF_DataType_e::Int16:return "I16";
          case CAIF_DataType_e::Int32:return "I32";
          case CAIF_DataType_e::Int64:return "I64";
          case CAIF_DataType_e::UInt8:return "U8";
          case CAIF_DataType_e::UInt16:return "U16";
          case CAIF_DataType_e::UInt32:return "U32";
          case CAIF_DataType_e::UInt64:return "U64";
          case CAIF_DataType_e::Bool:return "BOOL";
          case CAIF_DataType_e::Int4:return "I4";
        }
        return "F32";
      }

      /**
       * @brief Parse SafeTensors dtype string to CAIF_DataType_e.
       * Returns Float32 for unrecognized strings.
       */
      static CAIF_DataType_e FromSafeTensorsName(const std::string &name)
      {
        if(name=="F32")
        {
          return CAIF_DataType_e::Float32;
        }
        if(name=="F64")
        {
          return CAIF_DataType_e::Float64;
        }
        if(name=="F16")
        {
          return CAIF_DataType_e::Float16;
        }
        if(name=="BF16")
        {
          return CAIF_DataType_e::BFloat16;
        }
        if(name=="I8")
        {
          return CAIF_DataType_e::Int8;
        }
        if(name=="I16")
        {
          return CAIF_DataType_e::Int16;
        }
        if(name=="I32")
        {
          return CAIF_DataType_e::Int32;
        }
        if(name=="I64")
        {
          return CAIF_DataType_e::Int64;
        }
        if(name=="U8")
        {
          return CAIF_DataType_e::UInt8;
        }
        if(name=="U16")
        {
          return CAIF_DataType_e::UInt16;
        }
        if(name=="U32")
        {
          return CAIF_DataType_e::UInt32;
        }
        if(name=="U64")
        {
          return CAIF_DataType_e::UInt64;
        }
        if(name=="BOOL")
        {
          return CAIF_DataType_e::Bool;
        }
        if(name=="I4")
        {
          return CAIF_DataType_e::Int4;
        }
        return CAIF_DataType_e::Float32;
      }

      CAIF_DataType_e Value()const{return _v;}

      // Implicit conversion to nested enum for legacy switch/casts
      operator CAIF_DataType_e()const{return _v;}

      // Comparisons
      bool operator==(const CAIF_DataType &other)const{return _v==other._v;}
      bool operator!=(const CAIF_DataType &other)const{return _v!=other._v;}
      bool operator==(const CAIF_DataType_e other)const{return _v==other;}
      bool operator!=(const CAIF_DataType_e other)const{return _v!=other;}

    protected:

    private:
      CAIF_DataType_e _v;
  };

}//end instance namespace

#endif  // CAIF_DATA_TYPE_H
