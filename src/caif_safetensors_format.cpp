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
// AIF - AI Framework
// SafeTensors format implementation
//------------------------------------------------------------------------------
#include "caif_safetensors_format.h"
#include "caif_exception.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <set>

namespace instance
{

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

uint64_t CAIF_SafeTensorsFormat::AlignTo8(uint64_t offset)
{
  return (offset+7)&~static_cast<uint64_t>(7);
}

std::string CAIF_SafeTensorsFormat::EscapeJsonString(const std::string &s)
{
  std::string result;
  result.reserve(s.size()+10);
  for(char c:s)
  {
    if(c=='"')
    {
      result+="\\\"";
    }
    else if(c=='\\')
    {
      result+="\\\\";
    }
    else if(c=='\n')
    {
      result+="\\n";
    }
    else if(c=='\r')
    {
      result+="\\r";
    }
    else if(c=='\t')
    {
      result+="\\t";
    }
    else
    {
      result+=c;
    }
  }
  return result;
}

//------------------------------------------------------------------------------
// JSON Header Building (dtype-aware)
//------------------------------------------------------------------------------

std::string CAIF_SafeTensorsFormat::BuildJsonHeader(
    const std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> &tensors,
    const std::map<std::string,std::string> &metadata,
    std::vector<uint64_t> &data_offsets)
{
  std::ostringstream json;
  json<<"{";

  // Calculate data offsets and build tensor entries
  uint64_t current_offset=0;
  data_offsets.clear();
  data_offsets.reserve(tensors.size()*2);

  bool first_entry=true;

  for(const auto &pair:tensors)
  {
    const std::string &name=pair.first;
    const CAIF_DeviceTensor *tensor=pair.second;

    if(tensor==nullptr)
    {
      continue;
    }

    // Align offset to 8 bytes
    current_offset=AlignTo8(current_offset);
    uint64_t start_offset=current_offset;
    uint64_t tensor_bytes=static_cast<uint64_t>(tensor->SizeBytes());
    uint64_t end_offset=start_offset+tensor_bytes;

    data_offsets.push_back(start_offset);
    data_offsets.push_back(end_offset);

    if(first_entry==false)
    {
      json<<",";
    }
    first_entry=false;

    // Write tensor entry with native dtype
    json<<"\""<<EscapeJsonString(name)<<"\":{";
    json<<"\"dtype\":\""<<tensor->DtypeInfo().SafeTensorsName()<<"\",";
    json<<"\"shape\":[";

    const auto &shape=tensor->Shape();
    for(size_t i=0;i<shape.size();++i)
    {
      if(i>0)
      {
        json<<",";
      }
      json<<shape[i];
    }
    json<<"],";
    json<<"\"data_offsets\":["<<start_offset<<","<<end_offset<<"]}";

    current_offset=end_offset;
  }

  // Write metadata if present
  if(metadata.empty()==false)
  {
    if(first_entry==false)
    {
      json<<",";
    }
    json<<"\"__metadata__\":{";

    bool first_meta=true;
    for(const auto &kv:metadata)
    {
      if(first_meta==false)
      {
        json<<",";
      }
      first_meta=false;
      json<<"\""<<EscapeJsonString(kv.first)<<"\":\""
          <<EscapeJsonString(kv.second)<<"\"";
    }
    json<<"}";
  }

  json<<"}";
  return json.str();
}

//------------------------------------------------------------------------------
// JSON Parsing
//------------------------------------------------------------------------------

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

        if(key=="__metadata__")
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

        if(key=="__metadata__")
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

        if(key=="weight_map")
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
    const std::string &_json;
    size_t _pos;

    char Peek()const
    {
      if(_pos>=_json.size())
      {
        return '\0';
      }
      return _json[_pos];
    }

    void Advance()
    {
      if(_pos<_json.size())
      {
        ++_pos;
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
        msg+=std::to_string(_pos);
        THROW_CAIFE(msg.c_str());
      }
      Advance();
    }

    void SkipWhitespace()
    {
      while(_pos<_json.size())
      {
        char c=_json[_pos];
        if(c==' '||c=='\n'||c=='\r'||c=='\t')
        {
          ++_pos;
        }
        else
        {
          break;
        }
      }
    }

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

        if(key=="dtype")
        {
          info.dtype=ParseString();
        }
        else if(key=="shape")
        {
          info.shape=ParseShapeArray();
        }
        else if(key=="data_offsets")
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
      while(depth>0&&_pos<_json.size())
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
        while(_pos<_json.size())
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
      while(depth>0&&_pos<_json.size())
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

std::map<std::string,CAIF_SafeTensorsFormat::SafeTensorInfo_t>
CAIF_SafeTensorsFormat::ParseTensorInfos(const std::string &json)
{
  SafeTensorsJsonParser parser(json);
  return parser.ParseTensors();
}

std::map<std::string,std::string> CAIF_SafeTensorsFormat::ParseMetadata(const std::string &json)
{
  SafeTensorsJsonParser parser(json);
  return parser.ParseMetadata();
}

//------------------------------------------------------------------------------
// ReadHeader — shared helper to read the JSON header from a SafeTensors file
//------------------------------------------------------------------------------

std::string CAIF_SafeTensorsFormat::ReadHeader(const std::string &path,uint64_t &data_start)
{
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  if(in.is_open()==false)
  {
    THROW_CAIFE(("SafeTensors: cannot open file: "+path).c_str());
  }

  std::streamsize file_size=in.tellg();
  in.seekg(0,std::ios::beg);

  if(file_size<8)
  {
    THROW_CAIFE("SafeTensors: file too small");
  }
  if(static_cast<uint64_t>(file_size)>SafeTensorsConstants::g_max_file_size)
  {
    THROW_CAIFE("SafeTensors: file too large");
  }

  uint64_t header_size=0;
  in.read(reinterpret_cast<char*>(&header_size),sizeof(uint64_t));

  if(header_size>SafeTensorsConstants::g_max_header_size)
  {
    THROW_CAIFE("SafeTensors: header too large");
  }
  if(8+header_size>static_cast<uint64_t>(file_size))
  {
    THROW_CAIFE("SafeTensors: invalid header size");
  }

  std::string json_header(static_cast<size_t>(header_size),'\0');
  in.read(&json_header[0],static_cast<std::streamsize>(header_size));
  in.close();

  data_start=8+header_size;
  return json_header;
}

//------------------------------------------------------------------------------
// LoadSingleTensor — load one tensor from file given its metadata
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_SafeTensorsFormat::LoadSingleTensor(const std::string &path,
                                                         uint64_t data_start,
                                                         const CAIF_SafeTensorsFormat::SafeTensorInfo_t &info,
                                                         CAIF_CudaStream &stream)
{
  // Determine dtype
  CAIF_DataType::CAIF_DataType_e dtype=CAIF_DataType::FromSafeTensorsName(info.dtype);
  CAIF_DataType dtype_obj(dtype);

  // Calculate expected byte count
  size_t num_elements=1;
  for(uint32_t dim:info.shape)
  {
    num_elements*=dim;
  }

  size_t expected_bytes=dtype_obj.StorageSizeBytes(num_elements);
  uint64_t actual_bytes=info.data_offset_end-info.data_offset_start;

  if(actual_bytes!=static_cast<uint64_t>(expected_bytes))
  {
    THROW_CAIFE("SafeTensors: data size mismatch");
  }

  // Read raw bytes from file
  std::ifstream in(path,std::ios::binary);
  if(in.is_open()==false)
  {
    THROW_CAIFE(("SafeTensors: cannot open file: "+path).c_str());
  }

  in.seekg(static_cast<std::streamoff>(data_start+info.data_offset_start));

  std::vector<char> host_data(expected_bytes);
  in.read(host_data.data(),static_cast<std::streamsize>(expected_bytes));
  in.close();

  // Create device tensor with native dtype and upload raw bytes
  CAIF_DeviceTensor tensor=CAIF_DeviceTensor::FromHostRaw(host_data.data(),
                                                        info.shape,
                                                        stream,
                                                        dtype);
  return tensor;
}

//------------------------------------------------------------------------------
// ParseShardIndex — parse model.safetensors.index.json
//------------------------------------------------------------------------------

std::map<std::string,std::string>
CAIF_SafeTensorsFormat::ParseShardIndex(const std::string &index_path)
{
  std::ifstream in(index_path);
  if(in.is_open()==false)
  {
    THROW_CAIFE(("SafeTensors: cannot open index file: "+index_path).c_str());
  }

  std::ostringstream ss;
  ss<<in.rdbuf();
  in.close();

  std::string json=ss.str();
  SafeTensorsJsonParser parser(json);
  return parser.ParseWeightMap();
}

//------------------------------------------------------------------------------
// Save Implementation (dtype-aware)
//------------------------------------------------------------------------------

void CAIF_SafeTensorsFormat::Save(const std::string &path,
                                 const std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> &tensors,
                                 const std::map<std::string,std::string> &metadata)const
{
  try
  {
    if(tensors.size()>SafeTensorsConstants::g_max_tensors)
    {
      THROW_CAIFE("SafeTensors: too many tensors");
    }

    // Build JSON header
    std::vector<uint64_t> data_offsets;
    std::string json_header=BuildJsonHeader(tensors,metadata,data_offsets);

    // Pad header to 8-byte alignment
    while((json_header.size()%8)!=0)
    {
      json_header+=' ';
    }

    if(json_header.size()>SafeTensorsConstants::g_max_header_size)
    {
      THROW_CAIFE("SafeTensors: header too large");
    }

    // Open output file
    std::ofstream out(path,std::ios::binary);
    if(out.is_open()==false)
    {
      THROW_CAIFE(("SafeTensors: cannot open file for writing: "+path).c_str());
    }

    // Write header size (8 bytes, little-endian)
    uint64_t header_size=static_cast<uint64_t>(json_header.size());
    out.write(reinterpret_cast<const char*>(&header_size),sizeof(uint64_t));

    // Write JSON header
    out.write(json_header.data(),static_cast<std::streamsize>(json_header.size()));

    // Write tensor data
    size_t offset_idx=0;
    std::vector<char> padding_buffer(8,0);

    for(const auto &pair:tensors)
    {
      const CAIF_DeviceTensor *tensor=pair.second;
      if(tensor==nullptr)
      {
        continue;
      }

      // Align to 8 bytes if needed
      uint64_t current_pos=static_cast<uint64_t>(out.tellp())-8-header_size;
      uint64_t expected_pos=data_offsets[offset_idx*2];
      if(current_pos<expected_pos)
      {
        uint64_t padding_needed=expected_pos-current_pos;
        out.write(padding_buffer.data(),static_cast<std::streamsize>(padding_needed));
      }

      // Copy tensor raw bytes to host and write (dtype-aware)
      size_t byte_count=tensor->SizeBytes();
      std::vector<char> host_data(byte_count);
      tensor->CopyToHostRaw(host_data.data());

      out.write(host_data.data(),static_cast<std::streamsize>(byte_count));

      ++offset_idx;
    }

    out.close();
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Load Implementation (native dtype)
//------------------------------------------------------------------------------

std::map<std::string,CAIF_DeviceTensor>
CAIF_SafeTensorsFormat::Load(const std::string &path,CAIF_CudaStream &stream)const
{
  try
  {
    std::map<std::string,CAIF_DeviceTensor> result;

    // Read header
    uint64_t data_start=0;
    std::string json_header=ReadHeader(path,data_start);

    // Parse tensor infos
    auto tensor_infos=ParseTensorInfos(json_header);

    if(tensor_infos.size()>SafeTensorsConstants::g_max_tensors)
    {
      THROW_CAIFE("SafeTensors: too many tensors");
    }

    // Load each tensor in its native dtype
    for(const auto &kv:tensor_infos)
    {
      const std::string &name=kv.first;
      const CAIF_SafeTensorsFormat::SafeTensorInfo_t &info=kv.second;

      CAIF_DeviceTensor tensor=LoadSingleTensor(path,data_start,info,stream);
      result.emplace(name,std::move(tensor));
    }

    return result;
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// TensorInfos — tensor metadata without loading data
//------------------------------------------------------------------------------

std::map<std::string,CAIF_SafeTensorsFormat::SafeTensorInfo_t>
CAIF_SafeTensorsFormat::TensorInfos(const std::string &path)const
{
  try
  {
    uint64_t data_start=0;
    std::string json_header=ReadHeader(path,data_start);
    return ParseTensorInfos(json_header);
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Metadata Implementation
//------------------------------------------------------------------------------

std::map<std::string,std::string> CAIF_SafeTensorsFormat::Metadata(const std::string &path)const
{
  try
  {
    uint64_t data_start=0;
    std::string json_header=ReadHeader(path,data_start);
    return ParseMetadata(json_header);
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// LoadSharded — load all tensors from a sharded model directory
//------------------------------------------------------------------------------

std::map<std::string,CAIF_DeviceTensor>
CAIF_SafeTensorsFormat::LoadSharded(const std::string &directory,CAIF_CudaStream &stream)const
{
  try
  {
    std::map<std::string,CAIF_DeviceTensor> result;

    // Parse the index file to get tensor->shard mapping
    std::string index_path=directory+"/model.safetensors.index.json";
    std::map<std::string,std::string> weight_map=ParseShardIndex(index_path);

    if(weight_map.empty()==true)
    {
      THROW_CAIFE("SafeTensors: empty weight map in index file");
    }

    // Group tensors by shard file for efficient loading
    std::map<std::string,std::vector<std::string>> shard_tensors;
    for(const auto &kv:weight_map)
    {
      shard_tensors[kv.second].push_back(kv.first);
    }

    // Load one shard at a time to limit CPU memory
    for(const auto &shard_kv:shard_tensors)
    {
      const std::string &shard_name=shard_kv.first;
      const std::vector<std::string> &tensor_names=shard_kv.second;

      std::string shard_path=directory+"/"+shard_name;

      // Read shard header once
      uint64_t data_start=0;
      std::string json_header=ReadHeader(shard_path,data_start);
      auto tensor_infos=ParseTensorInfos(json_header);

      // Load requested tensors from this shard
      for(const std::string &name:tensor_names)
      {
        auto it=tensor_infos.find(name);
        if(it==tensor_infos.end())
        {
          std::string msg="SafeTensors: tensor '"+name+
                          "' not found in shard '"+shard_name+"'";
          THROW_CAIFE(msg.c_str());
        }

        CAIF_DeviceTensor tensor=LoadSingleTensor(shard_path,data_start,it->second,stream);
        result.emplace(name,std::move(tensor));
      }
    }

    return result;
  }
  CCAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// LoadTensorByName — load a single tensor from a sharded model directory
//------------------------------------------------------------------------------

CAIF_DeviceTensor CAIF_SafeTensorsFormat::LoadTensorByName(const std::string &directory,
                                                         const std::string &tensor_name,
                                                         CAIF_CudaStream &stream)const
{
  try
  {
    // Try sharded model first (model.safetensors.index.json)
    std::string index_path=directory+"/model.safetensors.index.json";
    std::string shard_path;

    std::ifstream index_test(index_path);
    if(index_test.is_open()==true)
    {
      index_test.close();

      std::map<std::string,std::string> weight_map=ParseShardIndex(index_path);

      auto it=weight_map.find(tensor_name);
      if(it==weight_map.end())
      {
        std::string msg="SafeTensors: tensor '"+tensor_name+
                        "' not found in index";
        THROW_CAIFE(msg.c_str());
      }

      shard_path=directory+"/"+it->second;
    }
    else
    {
      // Fall back to single-file model
      shard_path=directory+"/model.safetensors";
      std::ifstream single_test(shard_path);
      if(single_test.is_open()==false)
      {
        std::string msg="SafeTensors: no index file and no model.safetensors"
                        " in "+directory;
        THROW_CAIFE(msg.c_str());
      }
      single_test.close();
    }

    uint64_t data_start=0;
    std::string json_header=ReadHeader(shard_path,data_start);
    auto tensor_infos=ParseTensorInfos(json_header);

    auto tensor_it=tensor_infos.find(tensor_name);
    if(tensor_it==tensor_infos.end())
    {
      std::string msg="SafeTensors: tensor '"+tensor_name+
                      "' not found in '"+shard_path+"'";
      THROW_CAIFE(msg.c_str());
    }

    return LoadSingleTensor(shard_path,data_start,tensor_it->second,stream);
  }
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
