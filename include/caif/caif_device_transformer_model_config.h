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
// Configuration for CAIF_DeviceTransformerModel. The architecture-defining
// fields are required by the constructor so a model can never be built
// half-configured. The remaining fields carry documented auto / no-op defaults
// (set in the constructor's initializer list, not as in-class initializers) and
// are adjusted through the setters.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"
#include "caif_positional_encoding_mode.h"

namespace instance
{

class CAIF_DeviceTransformerModelConfig:public CAIF_Base
{
  public:
    // Required, architecture-defining fields. The optional fields (KV-head
    // count, FFN dim, output dim, RoPE style, embed/logit scale) default to
    // their auto / no-op values and are configured via the setters below.
    CAIF_DeviceTransformerModelConfig(const uint32_t vocab_size,
                                      const uint32_t max_seq_len,
                                      const uint32_t dim,
                                      const uint32_t num_heads,
                                      const uint32_t num_layers,
                                      const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &pe_mode,
                                      const bool causal,
                                      const bool use_rope,
                                      const bool tie_weights);

    // Vocabulary size for the token embedding.
    uint32_t VocabSize()const{return _vocab_size;}
    void SetVocabSize(const uint32_t vocab_size){_vocab_size=vocab_size;}

    // Maximum sequence length.
    uint32_t MaxSeqLen()const{return _max_seq_len;}
    void SetMaxSeqLen(const uint32_t max_seq_len){_max_seq_len=max_seq_len;}

    // Model (hidden) dimension.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Number of attention heads.
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}

    // Number of KV heads for GQA; 0 means equal to NumHeads().
    uint32_t NumKvHeads()const{return _num_kv_heads;}
    void SetNumKvHeads(const uint32_t num_kv_heads){_num_kv_heads=num_kv_heads;}

    // Number of transformer blocks.
    uint32_t NumLayers()const{return _num_layers;}
    void SetNumLayers(const uint32_t num_layers){_num_layers=num_layers;}

    // FFN hidden dimension; 0 means auto-compute from Dim().
    uint32_t FfnDim()const{return _ffn_dim;}
    void SetFfnDim(const uint32_t ffn_dim){_ffn_dim=ffn_dim;}

    // Causal attention mask.
    bool Causal()const{return _causal;}
    void SetCausal(const bool causal){_causal=causal;}

    // Use rotary position embeddings.
    bool UseRope()const{return _use_rope;}
    void SetUseRope(const bool use_rope){_use_rope=use_rope;}

    // RoPE layout: 0 = interleaved, 1 = half-split.
    int RopeStyle()const{return _rope_style;}
    void SetRopeStyle(const int rope_style){_rope_style=rope_style;}

    // Positional-encoding mode, used when UseRope() is false.
    CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e PeMode()const{return _pe_mode;}
    void SetPeMode(const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &pe_mode){_pe_mode=pe_mode;}

    // Output head dimension; 0 means equal to VocabSize() (LM head).
    uint32_t OutputDim()const{return _output_dim;}
    void SetOutputDim(const uint32_t output_dim){_output_dim=output_dim;}

    // Tie the output head weights to the embedding table.
    bool TieWeights()const{return _tie_weights;}
    void SetTieWeights(const bool tie_weights){_tie_weights=tie_weights;}

    // Multiplies the embedding lookup (sqrt(dim) archs like Gemma); 1.0 = no-op.
    float EmbedScale()const{return _embed_scale;}
    void SetEmbedScale(const float embed_scale){_embed_scale=embed_scale;}

    // Multiplies the head output / logits; 1.0 = no-op.
    float LogitScale()const{return _logit_scale;}
    void SetLogitScale(const float logit_scale){_logit_scale=logit_scale;}

  protected:

  private:
    uint32_t _vocab_size;
    uint32_t _max_seq_len;
    uint32_t _dim;
    uint32_t _num_heads;
    uint32_t _num_kv_heads;
    uint32_t _num_layers;
    uint32_t _ffn_dim;
    bool _causal;
    bool _use_rope;
    int _rope_style;
    CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e _pe_mode;
    uint32_t _output_dim;
    bool _tie_weights;
    float _embed_scale;
    float _logit_scale;
};

}//end instance namespace
