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
// Abstract interface for model serialization formats
//------------------------------------------------------------------------------
#ifndef CAIF_MODEL_FORMAT_H
#define CAIF_MODEL_FORMAT_H

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include <vector>
#include <map>
#include <string>
#include <utility>

namespace instance
{

/**
 * @brief Abstract interface for model serialization formats.
 *
 * Derive from this to implement SafeTensors, ONNX, or any other format.
 * This enables format-agnostic save/load APIs in CAIF_DeviceNetwork.
 */
class CAIF_ModelFormat:public CAIF_Base
{
  public:
    virtual ~CAIF_ModelFormat()=default;

    /**
     * @brief Save tensors with metadata to file.
     *
     * @param path Output file path
     * @param tensors Named tensors to save (name, tensor pointer pairs)
     * @param metadata Model metadata (architecture config, etc.)
     */
    virtual void Save(const std::string &path,
                      const std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> &tensors,
                      const std::map<std::string,std::string> &metadata)const=0;

    /**
     * @brief Load tensors from file.
     *
     * @param path Input file path
     * @param stream CUDA stream for device tensor creation
     * @return Map of tensor name to loaded tensor
     */
    virtual std::map<std::string,CAIF_DeviceTensor> Load(const std::string &path,
                                                       CAIF_CudaStream &stream)const=0;

    /**
     * @brief Metadata without loading tensor data.
     *
     * @param path Input file path
     * @return Metadata map
     */
    virtual std::map<std::string,std::string> Metadata(const std::string &path)const=0;

    /**
     * @brief File extension for this format (e.g., ".safetensors").
     */
    virtual std::string Extension()const=0;

    /**
     * @brief Format name for logging (e.g., "SafeTensors").
     */
    virtual std::string FormatName()const=0;

  protected:
    CAIF_ModelFormat()=default;

  private:
    // Non-copyable
    CAIF_ModelFormat(const CAIF_ModelFormat &)=delete;
    CAIF_ModelFormat &operator=(const CAIF_ModelFormat &)=delete;
};

}//end instance namespace

#endif  // CAIF_MODEL_FORMAT_H
