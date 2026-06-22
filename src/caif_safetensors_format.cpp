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
// SafeTensors format implementation
//------------------------------------------------------------------------------
#include "caif_safetensors_format.h"
#include "caif_safetensors_json_parser.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "caif_serialization_constants.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <set>

namespace instance
{

//------------------------------------------------------------------------------
// Static const dtype ↔ SafeTensors-name map. Initialised at file scope so it
// is available before any caller runs. Strings come from g_serial_dtype_*
// (caif_serialization_constants.h) — the single source of truth for the
// SafeTensors dtype vocabulary.
//------------------------------------------------------------------------------

const std::map<CAIF_DataType::CAIF_DataType_e,std::string>
CAIF_SafeTensorsFormat::_dtype_to_safetensors_name=
{
  {CAIF_DataType::CAIF_DataType_e::Float32,g_serial_dtype_f32},
  {CAIF_DataType::CAIF_DataType_e::Float64,g_serial_dtype_f64},
  {CAIF_DataType::CAIF_DataType_e::Float16,g_serial_dtype_f16},
  {CAIF_DataType::CAIF_DataType_e::BFloat16,g_serial_dtype_bf16},
  {CAIF_DataType::CAIF_DataType_e::Int4,g_serial_dtype_i4},
  {CAIF_DataType::CAIF_DataType_e::Int8,g_serial_dtype_i8},
  {CAIF_DataType::CAIF_DataType_e::Int16,g_serial_dtype_i16},
  {CAIF_DataType::CAIF_DataType_e::Int32,g_serial_dtype_i32},
  {CAIF_DataType::CAIF_DataType_e::Int64,g_serial_dtype_i64},
  {CAIF_DataType::CAIF_DataType_e::UInt8,g_serial_dtype_u8},
  {CAIF_DataType::CAIF_DataType_e::UInt16,g_serial_dtype_u16},
  {CAIF_DataType::CAIF_DataType_e::UInt32,g_serial_dtype_u32},
  {CAIF_DataType::CAIF_DataType_e::UInt64,g_serial_dtype_u64},
  {CAIF_DataType::CAIF_DataType_e::Bool,g_serial_dtype_bool}
};

const std::string &CAIF_SafeTensorsFormat::DtypeToSafeTensorsName(const CAIF_DataType::CAIF_DataType_e dt)
{
  try
  {
    const auto it=DtypeToSafeTensorsNameMap().find(dt);
    if(it==DtypeToSafeTensorsNameMap().end())
    {
      THROW_CAIFE("CAIF_SafeTensorsFormat::DtypeToSafeTensorsName: unsupported dtype");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_DataType::CAIF_DataType_e CAIF_SafeTensorsFormat::DtypeFromSafeTensorsName(const std::string &name)
{
  try
  {
    for(const auto &kv:DtypeToSafeTensorsNameMap())
    {
      if(kv.second==name)
      {
        return kv.first;
      }
    }
    THROW_CAIFE("CAIF_SafeTensorsFormat::DtypeFromSafeTensorsName: unrecognised name");
  }
  CAIF_CATCH_BLOCK();
}

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

    // safetensors spec requires tensors packed contiguously - the
    // reference Rust validator rejects any file where tensor[i].start
    // != tensor[i-1].end with "invalid offset for tensor ...".
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
    json<<"\""
        <<EscapeJsonString(name)
        <<"\":{";
    json<<"\""
        <<g_serial_key_dtype
        <<"\":\""
        <<DtypeToSafeTensorsName(tensor->DtypeInfo().Value())
        <<"\",";
    json<<"\""
        <<g_serial_key_shape
        <<"\":[";

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
    json<<"\""
        <<g_serial_key_data_offsets
        <<"\":["
        <<start_offset
        <<","
        <<end_offset
        <<"]}";

    current_offset=end_offset;
  }

  // Write metadata if present
  if(metadata.empty()==false)
  {
    if(first_entry==false)
    {
      json<<",";
    }
    json<<"\""
        <<g_serial_key_metadata_outer
        <<"\":{";

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
  if(static_cast<uint64_t>(file_size)>g_caif_safetensors_max_file_size)
  {
    THROW_CAIFE("SafeTensors: file too large");
  }

  uint64_t header_size=0;
  in.read(reinterpret_cast<char*>(&header_size),sizeof(uint64_t));

  if(header_size>g_caif_safetensors_max_header_size)
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
  CAIF_DataType::CAIF_DataType_e dtype=DtypeFromSafeTensorsName(info.dtype);
  CAIF_DataType dtype_obj(dtype);

  // Calculate expected byte count
  size_t num_elements=1;
  for(uint32_t dim:info.shape)
  {
    num_elements*=dim;
  }

  size_t expected_bytes=dtype_obj.StorageSizeBytes(num_elements);
  if(info.data_offset_start>info.data_offset_end)
  {
    THROW_CAIFE("SafeTensors: data_offset_start exceeds data_offset_end");
  }
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

  in.seekg(0,std::ios::end);
  const std::streamoff file_end=in.tellg();
  if(file_end<0)
  {
    THROW_CAIFE("SafeTensors: cannot determine file size");
  }
  const uint64_t file_size=static_cast<uint64_t>(file_end);
  if(data_start>file_size || info.data_offset_end>file_size-data_start)
  {
    THROW_CAIFE("SafeTensors: tensor data extends past end of file");
  }

  in.seekg(static_cast<std::streamoff>(data_start+info.data_offset_start));

  std::vector<char> host_data(expected_bytes);
  in.read(host_data.data(),static_cast<std::streamsize>(expected_bytes));
  if(in.gcount()!=static_cast<std::streamsize>(expected_bytes))
  {
    THROW_CAIFE("SafeTensors: short read loading tensor data");
  }
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
    if(tensors.size()>g_caif_safetensors_max_tensors)
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

    if(json_header.size()>g_caif_safetensors_max_header_size)
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

    // Write tensor data contiguously - BuildJsonHeader assigns
    // offsets back-to-back per the safetensors spec.
    for(const auto &pair:tensors)
    {
      const CAIF_DeviceTensor *tensor=pair.second;
      if(tensor==nullptr)
      {
        continue;
      }

      size_t byte_count=tensor->SizeBytes();
      std::vector<char> host_data(byte_count);
      tensor->CopyToHostRaw(host_data.data());

      out.write(host_data.data(),static_cast<std::streamsize>(byte_count));
    }

    out.close();
  }
  CAIF_CATCH_BLOCK()
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

    if(tensor_infos.size()>g_caif_safetensors_max_tensors)
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
  CAIF_CATCH_BLOCK()
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
  CAIF_CATCH_BLOCK()
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
  CAIF_CATCH_BLOCK()
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
    std::string index_path=directory+g_serial_index_relative;
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

      std::string shard_path=directory+g_serial_dir_separator+shard_name;

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
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// OpenShardedHandle / LoadFromHandle — lazy load surface
//
// Parses the index file and every shard's JSON header up front, then
// streams individual tensors on demand without re-parsing. Together these
// replace the eager `LoadSharded` for callers (add-MoE / finetune
// loaders) that walk a HF safetensors directory but only consume a
// subset of the tensors per session.
//------------------------------------------------------------------------------

CAIF_SafeTensorsFormat::ShardedHandle_t
CAIF_SafeTensorsFormat::OpenShardedHandle(const std::string &directory)const
{
  try
  {
    ShardedHandle_t handle;
    handle.SetDirectory(directory);

    const std::string index_path=directory+g_serial_index_relative;
    ShardedHandle_t::WeightMap_t wm;
    {
      std::ifstream index_test(index_path);
      const bool has_index=index_test.is_open();
      index_test.close();
      if(has_index==true)
      {
        wm=ParseShardIndex(index_path);
        if(wm.empty()==true)
        {
          THROW_CAIFE("SafeTensors: empty weight map in index file");
        }
      }
      else
      {
        // Single-shard fallback: no index file, but a top-level
        // `model.safetensors` carrying every tensor. Read the shard's
        // header, synthesize a weight map mapping every tensor name to
        // the single shard. Matches the fallback already implemented
        // for the eager LoadTensorByName path.
        const std::string single_path=directory+g_serial_single_shard_relative;
        std::ifstream single_test(single_path);
        if(single_test.is_open()==false)
        {
          THROW_CAIFE("SafeTensors: no index file and no model.safetensors in "+directory);
        }
        single_test.close();
        uint64_t hdr_data_start=0;
        const std::string hdr_json=ReadHeader(single_path,hdr_data_start);
        const auto infos=ParseTensorInfos(hdr_json);
        for(const auto &kv:infos)
        {
          wm.emplace(kv.first,g_serial_single_shard_name);
        }
        if(wm.empty()==true)
        {
          THROW_CAIFE("SafeTensors: single-shard model.safetensors has no tensors in "+directory);
        }
      }
    }
    handle.SetWeightMap(std::move(wm));

    // Parse each unique shard's JSON header once.
    std::map<std::string,std::vector<std::string>> shard_tensors;
    for(const auto &kv:handle.WeightMap())
    {
      shard_tensors[kv.second].push_back(kv.first);
    }
    for(const auto &shard_kv:shard_tensors)
    {
      const std::string &shard_name=shard_kv.first;
      ShardedHandle_t::PerShard_t per;
      per.SetPath(directory+g_serial_dir_separator+shard_name);
      uint64_t data_start=0;
      const std::string json_header=ReadHeader(per.Path(),data_start);
      per.SetDataStart(data_start);
      per.SetTensorInfos(ParseTensorInfos(json_header));
      handle.MutableShards().emplace(shard_name,std::move(per));
    }
    return handle;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor
CAIF_SafeTensorsFormat::LoadFromHandle(const ShardedHandle_t &handle,
                                       const std::string &tensor_name,
                                       CAIF_CudaStream &stream)const
{
  try
  {
    const ShardedHandle_t::WeightMap_t &wm=handle.WeightMap();
    auto wm_it=wm.find(tensor_name);
    if(wm_it==wm.end())
    {
      const std::string msg="SafeTensors: tensor '"+tensor_name+
                            "' not found in sharded handle index";
      THROW_CAIFE(msg.c_str());
    }
    const std::string &shard_name=wm_it->second;
    const ShardedHandle_t::ShardMap_t &shards=handle.Shards();
    auto sh_it=shards.find(shard_name);
    if(sh_it==shards.end())
    {
      const std::string msg="SafeTensors: shard '"+shard_name+
                            "' not present in handle (index/shard mismatch)";
      THROW_CAIFE(msg.c_str());
    }
    const ShardedHandle_t::PerShard_t &per=sh_it->second;
    const ShardedHandle_t::InfoMap_t &infos=per.TensorInfos();
    auto info_it=infos.find(tensor_name);
    if(info_it==infos.end())
    {
      const std::string msg="SafeTensors: tensor '"+tensor_name+
                            "' not found in shard '"+shard_name+"'";
      THROW_CAIFE(msg.c_str());
    }
    return LoadSingleTensor(per.Path(),per.DataStart(),info_it->second,stream);
  }
  CAIF_CATCH_BLOCK()
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
    std::string index_path=directory+g_serial_index_relative;
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
      shard_path=directory+g_serial_single_shard_relative;
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
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
