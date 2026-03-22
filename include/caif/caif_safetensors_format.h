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
#ifndef CAIF_SAFETENSORS_FORMAT_H
#define CAIF_SAFETENSORS_FORMAT_H

#include "caif_model_format.h"
#include "caif_device_tensor.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"
#include <vector>
#include <map>
#include <string>
#include <cstdint>

namespace instance
{

/**
 * @brief SafeTensors format constants
 */
namespace SafeTensorsConstants
{
  constexpr uint64_t g_max_file_size=2ULL<<40;  // 2 TiB
  constexpr size_t g_max_tensors=4096;
  constexpr size_t g_max_header_size=100*1024*1024;  // 100 MB header limit
  constexpr size_t g_alignment=8;  // Data alignment
}//end SafeTensorsConstants namespace

/**
 * @brief SafeTensors format implementation.
 *
 * Industry standard format, HuggingFace compatible, zero dependencies.
 * Supports multi-dtype tensors (F32, F16, BF16, I8, I4, etc.) and
 * sharded model loading (model.safetensors.index.json).
 *
 * File format:
 *   [8 bytes] header_size (uint64 little-endian)
 *   [N bytes] JSON header with tensor metadata
 *   [M bytes] raw tensor data (8-byte aligned)
 *
 * JSON header format:
 *   {
 *     "tensor_name": {"dtype": "F32", "shape": [d1,d2,...], "data_offsets": [start,end]},
 *     ...
 *     "__metadata__": {"key": "value", ...}
 *   }
 */
class CAIF_SafeTensorsFormat:public CAIF_ModelFormat
{
  public:
    CAIF_SafeTensorsFormat()=default;
    ~CAIF_SafeTensorsFormat() override=default;

    /**
     * @brief Tensor metadata for SafeTensors format
     */
    struct SafeTensorInfo_t
    {
      std::string dtype;
      std::vector<uint32_t> shape;
      uint64_t data_offset_start;
      uint64_t data_offset_end;
    };

    /**
     * @brief Save tensors with metadata to SafeTensors file.
     * Preserves each tensor's native dtype.
     */
    void Save(const std::string &path,
              const std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> &tensors,
              const std::map<std::string,std::string> &metadata)const override;

    /**
     * @brief Load tensors from a single SafeTensors file.
     * Tensors are loaded in their native dtype (bf16 stays bf16, etc.).
     */
    std::map<std::string,CAIF_DeviceTensor> Load(const std::string &path,
                                                 CAIF_CudaStream &stream)const override;

    /**
     * @brief Metadata without loading tensor data.
     */
    std::map<std::string,std::string> Metadata(const std::string &path)const override;

    /**
     * @brief Load all tensors from a sharded model directory.
     *
     * Parses model.safetensors.index.json to discover shards, then loads
     * shard-by-shard (one at a time to limit CPU memory). Tensors are loaded
     * in their native dtype.
     *
     * @param directory Path to the model directory containing shard files
     * @param stream CUDA stream for device tensor creation
     * @return Map of tensor name to loaded tensor
     */
    std::map<std::string,CAIF_DeviceTensor> LoadSharded(const std::string &directory,
                                                        CAIF_CudaStream &stream)const;

    /**
     * @brief Load a single tensor by name from a sharded model directory.
     *
     * Parses the index file to find which shard contains the tensor,
     * then loads only that tensor. Efficient for selective loading.
     *
     * @param directory Path to the model directory
     * @param tensor_name The name of the tensor to load
     * @param stream CUDA stream for device tensor creation
     * @return The loaded device tensor
     */
    CAIF_DeviceTensor LoadTensorByName(const std::string &directory,
                                      const std::string &tensor_name,
                                      CAIF_CudaStream &stream)const;

    /**
     * @brief Tensor metadata from a single SafeTensors file without loading data.
     * @return Map of tensor name to tensor info (dtype, shape, offsets)
     */
    std::map<std::string,SafeTensorInfo_t> TensorInfos(const std::string &path)const;

    std::string Extension()const override{return ".safetensors";}
    std::string FormatName()const override{return "SafeTensors";}

  protected:

  private:
    // JSON building helpers
    static std::string BuildJsonHeader(
        const std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> &tensors,
        const std::map<std::string,std::string> &metadata,
        std::vector<uint64_t> &data_offsets);

    // JSON parsing helpers
    static std::map<std::string,SafeTensorInfo_t> ParseTensorInfos(const std::string &json);
    static std::map<std::string,std::string> ParseMetadata(const std::string &json);

    /**
     * @brief Parse a sharded model index JSON file.
     * Returns map of tensor_name -> shard_filename.
     */
    static std::map<std::string,std::string> ParseShardIndex(const std::string &index_path);

    /**
     * @brief Read the JSON header from a SafeTensors file.
     * Returns the header string and sets data_start to the byte offset
     * where tensor data begins.
     */
    static std::string ReadHeader(const std::string &path,uint64_t &data_start);

    /**
     * @brief Load a single tensor from a SafeTensors file given its info.
     */
    static CAIF_DeviceTensor LoadSingleTensor(const std::string &path,
                                             uint64_t data_start,
                                             const SafeTensorInfo_t &info,
                                             CAIF_CudaStream &stream);

    // Utility
    static std::string EscapeJsonString(const std::string &s);
    static uint64_t AlignTo8(uint64_t offset);
};

}//end instance namespace

#endif  // CAIF_SAFETENSORS_FORMAT_H
