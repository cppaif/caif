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
// Device-resident token embedding layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_TOKEN_EMBEDDING_H
#define CAIF_DEVICE_TOKEN_EMBEDDING_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Maps integer token IDs to dense vectors via table lookup.
 *
 * Input:  [batch, seq_len] (uint32 token IDs)
 * Output: [batch, seq_len, dim]
 *
 * Provides two forward paths:
 * 1. ForwardFromIds — primary path using uint32 host IDs
 * 2. Forward — compatibility path interpreting float values as token IDs
 *
 * Parameters: embedding_table [vocab_size, dim] (Xavier uniform init)
 */
class CAIF_DeviceTokenEmbedding:public CAIF_DeviceLayer
{
  public:
    struct Config_t
    {
      uint32_t vocab_size;
      uint32_t dim;
    };

    CAIF_DeviceTokenEmbedding(const Config_t &config,
                             CAIF_CudaStream &stream);
    ~CAIF_DeviceTokenEmbedding()override;

    // Move
    CAIF_DeviceTokenEmbedding(CAIF_DeviceTokenEmbedding &&other)noexcept;
    CAIF_DeviceTokenEmbedding &operator=(CAIF_DeviceTokenEmbedding &&other)noexcept;

    /**
     * @brief Primary forward path using uint32 host token IDs
     */
    CAIF_DeviceTensor ForwardFromIds(const uint32_t *host_token_ids,
                                    uint32_t batch,
                                    uint32_t seq_len,
                                    bool training);

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
    uint32_t VocabSize()const{return _config.vocab_size;}
    uint32_t Dim()const{return _config.dim;}

  protected:

  private:
    Config_t _config;

    CAIF_DeviceTensor _embedding_table;       // [vocab_size, dim]
    CAIF_DeviceTensor _embedding_table_grad;  // [vocab_size, dim]
    CAIF_DeviceTensor _output_buffer;         // Reusable [batch, seq_len, dim]

    // Internal uint32 device buffer for token IDs
    uint32_t *_token_ids_device;
    size_t _token_ids_capacity;

    // Cached for backward
    uint32_t _cached_num_tokens;

    // Cached output shape for buffer reuse
    uint32_t _output_batch;
    uint32_t _output_seq_len;

    void EnsureTokenIdCapacity(size_t num_tokens);
    void EnsureOutputBuffer(uint32_t batch,uint32_t seq_len);
    void FreeTokenIdBuffer();
};

}//end instance namespace

#endif  // CAIF_DEVICE_TOKEN_EMBEDDING_H
