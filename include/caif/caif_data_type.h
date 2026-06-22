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
        switch(Value())
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
        if(Value()==CAIF_DataType_e::Int4)
        {
          return (num_elements+1)/2;
        }
        return num_elements*ElementSizeBytes();
      }

      /**
       * @brief Human-readable type name (e.g. "fp32", "bf16", "int4").
       * Defined out-of-line in src/caif_data_type.cpp.
       */
      std::string Name()const;

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
