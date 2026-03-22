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
// Device-resident tabular embedding layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_TABULAR_EMBEDDING_H
#define CAIF_DEVICE_TABULAR_EMBEDDING_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Projects tabular features to embedding dimension for transformer input.
 *
 * Input:  [batch, num_features] or [batch, seq_len, num_features]
 * Output: [batch, 1, dim] or [batch, seq_len, dim]
 *
 * This is a simple linear projection that converts raw feature vectors
 * into embeddings suitable for transformer processing.
 *
 * Parameters:
 *   0: W_proj  [num_features, dim]  Xavier uniform
 *   1: b_proj  [dim]                Zeros
 */
class CAIF_DeviceTabularEmbedding:public CAIF_DeviceLayer
{
  public:
    struct Config_t
    {
      uint32_t num_features;  // Input feature dimension
      uint32_t dim;           // Output embedding dimension
    };

    CAIF_DeviceTabularEmbedding(const Config_t &config,
                               CAIF_CudaStream &stream);
    ~CAIF_DeviceTabularEmbedding()override=default;

    // Move
    CAIF_DeviceTabularEmbedding(CAIF_DeviceTabularEmbedding &&other);
    CAIF_DeviceTabularEmbedding &operator=(CAIF_DeviceTabularEmbedding &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training)override;
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)override;
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors
    uint32_t NumFeatures()const{return _config.num_features;}
    uint32_t Dim()const{return _config.dim;}

    // Weight initialization
    void InitializeWeights(uint32_t seed=0);

  protected:

  private:
    Config_t _config;

    CAIF_DeviceTensor _w_proj;       // [num_features, dim]
    CAIF_DeviceTensor _b_proj;       // [dim]

    CAIF_DeviceTensor _grad_w_proj;  // [num_features, dim]
    CAIF_DeviceTensor _grad_b_proj;  // [dim]

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    uint32_t _cached_batch;
    uint32_t _cached_seq_len;
};

}//end instance namespace

#endif  // CAIF_DEVICE_TABULAR_EMBEDDING_H
