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
// Configuration for CAIF_DeviceSpectrogramEmbedding. All three fields
// (freq_bins, dim, use_cls_token) are required by the constructor so the
// embedding can never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceSpectrogramEmbeddingConfig:public CAIF_Base
{
  public:
    // All three fields are required: the input frequency-bin count, the
    // embedding dimension, and whether a leading CLS token is prepended.
    CAIF_DeviceSpectrogramEmbeddingConfig(const uint32_t freq_bins,
                                          const uint32_t dim,
                                          const bool use_cls_token);

    // Number of input frequency bins per frame.
    uint32_t FreqBins()const{return _freq_bins;}
    void SetFreqBins(const uint32_t freq_bins){_freq_bins=freq_bins;}

    // Embedding dimension each frame is projected to.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Prepend a learnable CLS token to the sequence.
    bool UseCLSToken()const{return _use_cls_token;}
    void SetUseCLSToken(const bool use_cls_token){_use_cls_token=use_cls_token;}

  protected:

  private:
    uint32_t _freq_bins;
    uint32_t _dim;
    bool _use_cls_token;
};

}//end instance namespace
