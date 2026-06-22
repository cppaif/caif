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
// Minimal recursive-descent JSON parser for SafeTensors headers and shard
// index files. Internal to CAIF_SafeTensorsFormat — extracted from
// caif_safetensors_format.cpp for the one-class-per-file structural refactor.
//------------------------------------------------------------------------------
#pragma once

#include "caif_safetensors_format.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace instance
{

// Simple JSON parser for SafeTensors header
class SafeTensorsJsonParser
{
  public:
    explicit SafeTensorsJsonParser(const std::string &json):_json(json),_pos(0){}

    std::map<std::string,CAIF_SafeTensorsFormat::SafeTensorInfo_t> ParseTensors()
    {
      std::map<std::string,CAIF_SafeTensorsFormat::SafeTensorInfo_t> result;
      SkipWhitespace();
      Expect('{');

      while(Peek()!='}')
      {
        SkipWhitespace();
        std::string key=ParseString();
        SkipWhitespace();
        Expect(':');
        SkipWhitespace();

        if(key==g_serial_key_metadata_outer)
        {
          // Skip metadata object for tensor parsing
          SkipObject();
        }
        else
        {
          CAIF_SafeTensorsFormat::SafeTensorInfo_t info=ParseTensorInfo();
          result[key]=info;
        }

        SkipWhitespace();
        if(Peek()==',')
        {
          Advance();
        }
      }

      Expect('}');
      return result;
    }

    std::map<std::string,std::string> ParseMetadata()
    {
      std::map<std::string,std::string> result;
      SkipWhitespace();
      Expect('{');

      while(Peek()!='}')
      {
        SkipWhitespace();
        std::string key=ParseString();
        SkipWhitespace();
        Expect(':');
        SkipWhitespace();

        if(key==g_serial_key_metadata_outer)
        {
          // Parse metadata object
          Expect('{');
          while(Peek()!='}')
          {
            SkipWhitespace();
            std::string meta_key=ParseString();
            SkipWhitespace();
            Expect(':');
            SkipWhitespace();
            std::string meta_value=ParseString();
            result[meta_key]=meta_value;
            SkipWhitespace();
            if(Peek()==',')
            {
              Advance();
            }
          }
          Expect('}');
        }
        else
        {
          // Skip tensor info
          SkipObject();
        }

        SkipWhitespace();
        if(Peek()==',')
        {
          Advance();
        }
      }

      Expect('}');
      return result;
    }

    /**
     * @brief Parse a shard index file's weight_map.
     * Format: {"metadata":{...},"weight_map":{"tensor_name":"shard_file",...}}
     */
    std::map<std::string,std::string> ParseWeightMap()
    {
      std::map<std::string,std::string> result;
      SkipWhitespace();
      Expect('{');

      while(Peek()!='}')
      {
        SkipWhitespace();
        std::string key=ParseString();
        SkipWhitespace();
        Expect(':');
        SkipWhitespace();

        if(key==g_serial_key_weight_map)
        {
          Expect('{');
          while(Peek()!='}')
          {
            SkipWhitespace();
            std::string tensor_name=ParseString();
            SkipWhitespace();
            Expect(':');
            SkipWhitespace();
            std::string shard_file=ParseString();
            result[tensor_name]=shard_file;
            SkipWhitespace();
            if(Peek()==',')
            {
              Advance();
            }
          }
          Expect('}');
        }
        else
        {
          SkipValue();
        }

        SkipWhitespace();
        if(Peek()==',')
        {
          Advance();
        }
      }

      Expect('}');
      return result;
    }

  private:
    const std::string &Json()const{return _json;}
    size_t Pos()const{return _pos;}
    void SetPos(const size_t p){_pos=p;}
    void IncPos(){++_pos;}

    char Peek()const
    {
      if(Pos()>=Json().size())
      {
        return '\0';
      }
      return Json()[Pos()];
    }

    void Advance()
    {
      if(Pos()<Json().size())
      {
        IncPos();
      }
    }

    void Expect(char c)
    {
      SkipWhitespace();
      if(Peek()!=c)
      {
        std::string msg="SafeTensors JSON parse error: expected '";
        msg+=c;
        msg+="' at position ";
        msg+=std::to_string(Pos());
        THROW_CAIFE(msg.c_str());
      }
      Advance();
    }

    void SkipWhitespace()
    {
      while(Pos()<Json().size())
      {
        char c=Json()[Pos()];
        if(c==' '||c=='\n'||c=='\r'||c=='\t')
        {
          IncPos();
        }
        else
        {
          break;
        }
      }
    }

    const std::string &_json;
    size_t _pos;

    std::string ParseString()
    {
      Expect('"');
      std::string result;
      while(Peek()!='"'&&Peek()!='\0')
      {
        if(Peek()=='\\')
        {
          Advance();
          char escaped=Peek();
          if(escaped=='n')
          {
            result+='\n';
          }
          else if(escaped=='r')
          {
            result+='\r';
          }
          else if(escaped=='t')
          {
            result+='\t';
          }
          else if(escaped=='u')
          {
            // Skip unicode escape
            Advance();
            Advance();
            Advance();
            Advance();
          }
          else
          {
            result+=escaped;
          }
          Advance();
        }
        else
        {
          result+=Peek();
          Advance();
        }
      }
      Expect('"');
      return result;
    }

    int64_t ParseNumber()
    {
      SkipWhitespace();
      bool negative=false;
      if(Peek()=='-')
      {
        negative=true;
        Advance();
      }
      int64_t value=0;
      while(std::isdigit(Peek()))
      {
        value=value*10+(Peek()-'0');
        Advance();
      }
      if(negative==true)
      {
        value=-value;
      }
      return value;
    }

    std::vector<uint32_t> ParseShapeArray()
    {
      std::vector<uint32_t> result;
      Expect('[');
      SkipWhitespace();
      while(Peek()!=']')
      {
        int64_t val=ParseNumber();
        result.push_back(static_cast<uint32_t>(val));
        SkipWhitespace();
        if(Peek()==',')
        {
          Advance();
        }
        SkipWhitespace();
      }
      Expect(']');
      return result;
    }

    std::pair<uint64_t,uint64_t> ParseDataOffsets()
    {
      Expect('[');
      SkipWhitespace();
      uint64_t start=static_cast<uint64_t>(ParseNumber());
      SkipWhitespace();
      Expect(',');
      SkipWhitespace();
      uint64_t end=static_cast<uint64_t>(ParseNumber());
      SkipWhitespace();
      Expect(']');
      return {start,end};
    }

    CAIF_SafeTensorsFormat::SafeTensorInfo_t ParseTensorInfo()
    {
      CAIF_SafeTensorsFormat::SafeTensorInfo_t info;
      Expect('{');

      while(Peek()!='}')
      {
        SkipWhitespace();
        std::string key=ParseString();
        SkipWhitespace();
        Expect(':');
        SkipWhitespace();

        if(key==g_serial_key_dtype)
        {
          info.dtype=ParseString();
        }
        else if(key==g_serial_key_shape)
        {
          info.shape=ParseShapeArray();
        }
        else if(key==g_serial_key_data_offsets)
        {
          auto offsets=ParseDataOffsets();
          info.data_offset_start=offsets.first;
          info.data_offset_end=offsets.second;
        }
        else
        {
          // Skip unknown field
          SkipValue();
        }

        SkipWhitespace();
        if(Peek()==',')
        {
          Advance();
        }
      }

      Expect('}');
      return info;
    }

    void SkipObject()
    {
      Expect('{');
      int depth=1;
      while(depth>0&&Pos()<Json().size())
      {
        if(Peek()=='{')
        {
          ++depth;
        }
        else if(Peek()=='}')
        {
          --depth;
        }
        else if(Peek()=='"')
        {
          // Skip string
          Advance();
          while(Peek()!='"'&&Peek()!='\0')
          {
            if(Peek()=='\\')
            {
              Advance();
            }
            Advance();
          }
        }
        Advance();
      }
    }

    void SkipValue()
    {
      char c=Peek();
      if(c=='"')
      {
        ParseString();
      }
      else if(c=='[')
      {
        SkipArray();
      }
      else if(c=='{')
      {
        SkipObject();
      }
      else
      {
        // Number or literal
        while(Pos()<Json().size())
        {
          char ch=Peek();
          if(ch==','||ch=='}'||ch==']')
          {
            break;
          }
          Advance();
        }
      }
    }

    void SkipArray()
    {
      Expect('[');
      int depth=1;
      while(depth>0&&Pos()<Json().size())
      {
        if(Peek()=='[')
        {
          ++depth;
        }
        else if(Peek()==']')
        {
          --depth;
        }
        else if(Peek()=='"')
        {
          Advance();
          while(Peek()!='"'&&Peek()!='\0')
          {
            if(Peek()=='\\')
            {
              Advance();
            }
            Advance();
          }
        }
        Advance();
      }
    }
};

}//end instance namespace
