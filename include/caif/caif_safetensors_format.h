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
#include "caif_serialization_constants.h"
#include <vector>
#include <map>
#include <string>
#include <cstdint>

namespace instance
{

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

    /**
     * @brief Cached metadata for one shard of a sharded SafeTensors model.
     *
     * `path` is the full filesystem path to the .safetensors shard.
     * `data_start` is the byte offset where tensor data begins within
     * the shard (the 8-byte length prefix + JSON header are above
     * this). `tensor_infos` maps each tensor name in this shard to
     * its dtype / shape / data offsets — populated once at handle-
     * open time so subsequent `LoadFromHandle` calls do not re-parse
     * the JSON header.
     */
    struct ShardedHandle_t
    {
      public:
        typedef std::map<std::string,SafeTensorInfo_t> InfoMap_t;
        typedef std::map<std::string,std::string> WeightMap_t;

        struct PerShard_t
        {
          public:
            PerShard_t():_path(),_data_start(0),_tensor_infos(){}
            const std::string &Path()const{return _path;}
            uint64_t DataStart()const{return _data_start;}
            const InfoMap_t &TensorInfos()const{return _tensor_infos;}
            void SetPath(const std::string &p){_path=p;}
            void SetDataStart(const uint64_t o){_data_start=o;}
            void SetTensorInfos(InfoMap_t m){_tensor_infos=std::move(m);}
          private:
            std::string _path;
            uint64_t _data_start;
            InfoMap_t _tensor_infos;
        };

        typedef std::map<std::string,PerShard_t> ShardMap_t;

        ShardedHandle_t():_directory(),_weight_map(),_shards(){}

        const std::string &Directory()const{return _directory;}
        const WeightMap_t &WeightMap()const{return _weight_map;}
        const ShardMap_t &Shards()const{return _shards;}

        void SetDirectory(const std::string &d){_directory=d;}
        void SetWeightMap(WeightMap_t m){_weight_map=std::move(m);}
        WeightMap_t &MutableWeightMap(){return _weight_map;}
        ShardMap_t &MutableShards(){return _shards;}

      private:
        std::string _directory;
        // tensor_name -> shard_filename (relative)
        WeightMap_t _weight_map;
        // shard_filename -> per-shard cache
        ShardMap_t _shards;
    };

    /**
     * @brief Open a sharded model directory lazily — parse the index file
     * and every shard's JSON header but load no tensor data. Returns a
     * `ShardedHandle_t` the caller passes to `LoadFromHandle` to stream
     * individual tensors as needed. Memory cost: O(num_tensors * sizeof
     * SafeTensorInfo_t) host bytes; zero device memory.
     */
    ShardedHandle_t OpenShardedHandle(const std::string &directory)const;

    /**
     * @brief Stream a single tensor out of a previously-opened sharded
     * handle. Throws if the tensor name is not in the handle's index. Uses
     * the cached shard path + data offset, so this call only does:
     *   open shard fd -> seek -> read tensor bytes -> upload to device.
     * No JSON re-parse, no repeated index lookup.
     */
    CAIF_DeviceTensor LoadFromHandle(const ShardedHandle_t &handle,
                                     const std::string &tensor_name,
                                     CAIF_CudaStream &stream)const;

    std::string Extension()const override{return g_serial_extension;}
    std::string FormatName()const override{return g_serial_format_name;}

    /**
     * @brief Map a CAIF dtype to its SafeTensors string form (e.g.
     * Float32 → "F32", BFloat16 → "BF16"). Throws on the unsupported
     * cell; SafeTensors does not have a designated name for every CAIF
     * dtype.
     */
    static const std::string &DtypeToSafeTensorsName(const CAIF_DataType::CAIF_DataType_e dt);

    /**
     * @brief Map a SafeTensors string form back to a CAIF dtype.
     * Throws if `name` is not a recognised SafeTensors dtype.
     */
    static CAIF_DataType::CAIF_DataType_e DtypeFromSafeTensorsName(const std::string &name);

    /**
     * @brief Read-only access to the static const dtype→name map. Used
     * by callers that need to iterate all pairs (e.g. enumerating
     * supported dtypes for diagnostics).
     */
    static const std::map<CAIF_DataType::CAIF_DataType_e,std::string> &
    DtypeToSafeTensorsNameMap(){return _dtype_to_safetensors_name;}

  protected:

  private:
    // Static const map of CAIF dtype → SafeTensors string form. Defined
    // at file scope in caif_safetensors_format.cpp; initialised before
    // main() via the static-storage rules. Strings come from
    // g_serial_dtype_* in caif_serialization_constants.h.
    static const std::map<CAIF_DataType::CAIF_DataType_e,std::string> _dtype_to_safetensors_name;

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
