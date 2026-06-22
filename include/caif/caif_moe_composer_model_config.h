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
// Configuration for a full CAIF_MoEComposer MoE model. The eight defining
// fields (embedding/head sizing, positional mode, final-norm epsilon, layer
// count, and the per-layer block template) are required by the constructor so a
// model can never be built half-configured; only the storage/compute dtypes
// carry documented defaults (set in the constructor's initializer list).
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"
#include "caif_data_type.h"
#include "caif_positional_encoding_mode.h"
#include "caif_moe_composer_block_config.h"

namespace instance
{

class CAIF_MoEComposerModelConfig:public CAIF_Base
{
  public:
    // All eight fields are required, including the per-layer block_template.
    // storage_dtype and compute_dtype default to Float32 and are set via the
    // setters.
    CAIF_MoEComposerModelConfig(const uint32_t vocab_size,
                                const uint32_t max_seq_len,
                                const uint32_t output_dim,
                                const bool tie_weights,
                                const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &pe_mode,
                                const float final_norm_eps,
                                const uint32_t num_layers,
                                const CAIF_MoEComposerBlockConfig &block_template);

    // Vocabulary size for the token embedding.
    uint32_t VocabSize()const{return _vocab_size;}
    void SetVocabSize(const uint32_t vocab_size){_vocab_size=vocab_size;}

    // Maximum sequence length.
    uint32_t MaxSeqLen()const{return _max_seq_len;}
    void SetMaxSeqLen(const uint32_t max_seq_len){_max_seq_len=max_seq_len;}

    // Output head dimension; 0 means equal to VocabSize().
    uint32_t OutputDim()const{return _output_dim;}
    void SetOutputDim(const uint32_t output_dim){_output_dim=output_dim;}

    // Tie the output head weights to the embedding table.
    bool TieWeights()const{return _tie_weights;}
    void SetTieWeights(const bool tie_weights){_tie_weights=tie_weights;}

    // Positional-encoding mode (skipped when the block uses RoPE).
    CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e PeMode()const{return _pe_mode;}
    void SetPeMode(const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &pe_mode){_pe_mode=pe_mode;}

    // Final RMSNorm epsilon.
    float FinalNormEps()const{return _final_norm_eps;}
    void SetFinalNormEps(const float final_norm_eps){_final_norm_eps=final_norm_eps;}

    // Number of MoE block layers.
    uint32_t NumLayers()const{return _num_layers;}
    void SetNumLayers(const uint32_t num_layers){_num_layers=num_layers;}

    // Per-layer MoE block template, applied to every layer.
    const CAIF_MoEComposerBlockConfig &BlockTemplate()const{return _block_template;}
    void SetBlockTemplate(const CAIF_MoEComposerBlockConfig &block_template){_block_template=block_template;}

    // Precision (default Float32).
    CAIF_DataType::CAIF_DataType_e StorageDtype()const{return _storage_dtype;}
    void SetStorageDtype(const CAIF_DataType::CAIF_DataType_e &storage_dtype){_storage_dtype=storage_dtype;}
    CAIF_DataType::CAIF_DataType_e ComputeDtype()const{return _compute_dtype;}
    void SetComputeDtype(const CAIF_DataType::CAIF_DataType_e &compute_dtype){_compute_dtype=compute_dtype;}

  protected:

  private:
    uint32_t _vocab_size;
    uint32_t _max_seq_len;
    uint32_t _output_dim;
    bool _tie_weights;
    CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e _pe_mode;
    float _final_norm_eps;
    uint32_t _num_layers;
    CAIF_MoEComposerBlockConfig _block_template;
    CAIF_DataType::CAIF_DataType_e _storage_dtype;
    CAIF_DataType::CAIF_DataType_e _compute_dtype;
};

}//end instance namespace
