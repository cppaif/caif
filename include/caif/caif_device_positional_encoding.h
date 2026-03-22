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
// Device-resident positional encoding layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_POSITIONAL_ENCODING_H
#define CAIF_DEVICE_POSITIONAL_ENCODING_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

enum class PositionalEncodingMode_e:uint8_t
{
  Learned,
  Sinusoidal
};

/**
 * @brief Adds position information to embeddings.
 *
 * Input:  [batch, seq_len, dim]
 * Output: [batch, seq_len, dim] (input + positional encoding)
 *
 * Two modes:
 * - Learned: trainable pe_table [max_seq_len, dim]
 * - Sinusoidal: fixed sin/cos table computed at construction
 */
class CAIF_DevicePositionalEncoding:public CAIF_DeviceLayer
{
  public:
    struct Config_t
    {
      uint32_t max_seq_len;
      uint32_t dim;
      PositionalEncodingMode_e mode;
    };

    CAIF_DevicePositionalEncoding(const Config_t &config,
                                 CAIF_CudaStream &stream);
    ~CAIF_DevicePositionalEncoding()override=default;

    // Move
    CAIF_DevicePositionalEncoding(CAIF_DevicePositionalEncoding &&other);
    CAIF_DevicePositionalEncoding &operator=(CAIF_DevicePositionalEncoding &&other);

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
    uint32_t MaxSeqLen()const{return _config.max_seq_len;}
    uint32_t Dim()const{return _config.dim;}
    PositionalEncodingMode_e Mode()const{return _config.mode;}

  protected:

  private:
    Config_t _config;

    // Learned mode: trainable table + gradient
    CAIF_DeviceTensor _pe_table;      // [max_seq_len, dim]
    CAIF_DeviceTensor _pe_table_grad; // [max_seq_len, dim]

    // Sinusoidal mode: fixed table (non-trainable)
    CAIF_DeviceTensor _sinusoidal_table; // [max_seq_len, dim]

    // Cached for backward
    uint32_t _cached_batch;
    uint32_t _cached_seq_len;
};

}//end instance namespace

#endif  // CAIF_DEVICE_POSITIONAL_ENCODING_H
