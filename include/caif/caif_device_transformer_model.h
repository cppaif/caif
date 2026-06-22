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
// CAIF_DeviceTransformerModel<ComputeT, StorageT> — complete transformer
// model. Assembles embedding, positional encoding, N transformer blocks,
// final norm, and output head into a single trainable unit.
//
// Input:  [batch, seq_len] (token IDs as float)
// Output: [batch, seq_len, output_dim]
//
// Components (appended to the container in this order):
//   - TokenEmbedding<C, S>           [batch, seq_len] -> [batch, seq_len, dim]
//   - PositionalEncoding<C, S>       (optional, omitted when use_rope=true)
//   - N x TransformerBlock<C, S>
//   - RMSNorm<C, S>                  (final normalization)
//   - LinearHead<C, S>               [batch, seq_len, dim] -> [batch, seq_len, output_dim]
//
// Forward / backward chaining is inherited from CAIF_DeviceContainer.
// Parameter / gradient iteration, zero-grad, total-param-count, and aux-loss
// summation all come from the container base. The model templates on
// <ComputeT, StorageT> for the cell every internal sublayer is built at.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_container.h"
#include "caif_device_layer_typed.h"
#include "caif_run_context.h"
#include "caif_device_token_embedding.h"
#include "caif_device_positional_encoding.h"
#include "caif_device_transformer_block.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_linear_head.h"
#include "caif_device_transformer_model_config.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceTransformerModel:public CAIF_DeviceContainer
{
  public:
    typedef CAIF_DeviceLayerTyped<ComputeT,StorageT> Typed_t;

    CAIF_DeviceTransformerModel(const CAIF_DeviceTransformerModelConfig &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceTransformerModel()override=default;

    // Move
    CAIF_DeviceTransformerModel(CAIF_DeviceTransformerModel &&other);
    CAIF_DeviceTransformerModel &operator=(CAIF_DeviceTransformerModel &&other);

    // CAIF_DeviceLayer interface — only the tag and the name-prefixing override
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::TransformerModel_e;
    }
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors
    const CAIF_DeviceTransformerModelConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DeviceTransformerModelConfig &c){_config=c;}
    uint32_t NumLayers()const{return Config().NumLayers();}
    bool PosEncPresent()const{return _pos_enc_present;}
    void SetPosEncPresent(const bool b){_pos_enc_present=b;}

    static constexpr CAIF_DataType::CAIF_DataType_e ComputeDtype()
    {
      return CAIF_StorageDtype_t<ComputeT>::Value;
    }
    static constexpr CAIF_DataType::CAIF_DataType_e StorageDtype()
    {
      return CAIF_StorageDtype_t<StorageT>::Value;
    }

  protected:

  private:
    // Slot indices within _sublayers for name-prefix resolution.
    // _pos_enc_present toggles whether a PositionalEncoding sublayer
    // sits between the embedding and the first block.
    CAIF_DeviceTransformerModelConfig _config;
    bool _pos_enc_present;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceTransformerModel<float,float>;
extern template class CAIF_DeviceTransformerModel<float,__half>;
extern template class CAIF_DeviceTransformerModel<float,__nv_bfloat16>;
extern template class CAIF_DeviceTransformerModel<__half,float>;
extern template class CAIF_DeviceTransformerModel<__half,__half>;
extern template class CAIF_DeviceTransformerModel<__half,__nv_bfloat16>;
extern template class CAIF_DeviceTransformerModel<__nv_bfloat16,float>;
extern template class CAIF_DeviceTransformerModel<__nv_bfloat16,__half>;
extern template class CAIF_DeviceTransformerModel<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceTransformerModel<float,float>;
#endif

}//end instance namespace
