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
// Configuration for CAIF_DeviceTabularEmbedding. Both fields (num_features,
// dim) are required by the constructor so the embedding can never be built
// half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceTabularEmbeddingConfig:public CAIF_Base
{
  public:
    // Both fields are required: the number of tabular features and the
    // embedding dimension each feature is projected to.
    CAIF_DeviceTabularEmbeddingConfig(const uint32_t num_features,const uint32_t dim);

    // Number of input (tabular) features.
    uint32_t NumFeatures()const{return _num_features;}
    void SetNumFeatures(const uint32_t num_features){_num_features=num_features;}

    // Embedding dimension each feature is projected to.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

  protected:

  private:
    uint32_t _num_features;
    uint32_t _dim;
};

}//end instance namespace
