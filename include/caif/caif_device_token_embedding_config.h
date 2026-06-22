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
// Configuration for CAIF_DeviceTokenEmbedding. vocab_size and dim define the
// embedding table shape and are required by the constructor so a table can
// never be sized half-way. output_scale carries a documented no-op default
// (1.0, set in the constructor's initializer list, not as an in-class
// initializer) and is adjusted through its setter.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceTokenEmbeddingConfig:public CAIF_Base
{
  public:
    // vocab_size and dim are the required, table-defining fields. output_scale
    // defaults to 1.0 (no-op) and is configured via SetOutputScale().
    CAIF_DeviceTokenEmbeddingConfig(const uint32_t vocab_size,const uint32_t dim);

    // Vocabulary size (rows of the embedding table).
    uint32_t VocabSize()const{return _vocab_size;}
    void SetVocabSize(const uint32_t vocab_size){_vocab_size=vocab_size;}

    // Embedding dimension (columns of the embedding table).
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Multiplies the embedding lookup output (and scales the gradient back on
    // the backward path). 1.0 = no-op; sqrt(dim) for Gemma-style models (F6).
    float OutputScale()const{return _output_scale;}
    void SetOutputScale(const float output_scale){_output_scale=output_scale;}

  protected:

  private:
    uint32_t _vocab_size;
    uint32_t _dim;
    float _output_scale;
};

}//end instance namespace
