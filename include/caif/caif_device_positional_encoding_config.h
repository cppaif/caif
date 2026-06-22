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
// Configuration for CAIF_DevicePositionalEncoding. All three fields
// (max_seq_len, dim, mode) are required by the constructor so the encoding can
// never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"
#include "caif_positional_encoding_mode.h"

namespace instance
{

class CAIF_DevicePositionalEncodingConfig:public CAIF_Base
{
  public:
    // All three fields are required: the table shape (max_seq_len, dim) and the
    // positional-encoding mode (learned / sinusoidal / none).
    CAIF_DevicePositionalEncodingConfig(const uint32_t max_seq_len,
                                        const uint32_t dim,
                                        const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &mode);

    // Maximum sequence length (rows of the positional table).
    uint32_t MaxSeqLen()const{return _max_seq_len;}
    void SetMaxSeqLen(const uint32_t max_seq_len){_max_seq_len=max_seq_len;}

    // Encoding dimension (columns of the positional table).
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Positional-encoding mode (learned / sinusoidal / none).
    CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e Mode()const{return _mode;}
    void SetMode(const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &mode){_mode=mode;}

  protected:

  private:
    uint32_t _max_seq_len;
    uint32_t _dim;
    CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e _mode;
};

}//end instance namespace
