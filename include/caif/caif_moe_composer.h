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
// MoE convenience composer — builds MoE blocks and models from primitives
//------------------------------------------------------------------------------
#ifndef CAIF_MOE_COMPOSER_H
#define CAIF_MOE_COMPOSER_H

#include "caif_base.h"
#include "caif_device_pre_norm_block.h"
#include "caif_device_network.h"
#include "caif_device_moe_layer.h"
#include "caif_moe_composer_block_config.h"
#include "caif_moe_composer_model_config.h"
#include "caif_device_positional_encoding.h"
#include "caif_cuda_stream.h"
#include "caif_data_type.h"
#include <cstdint>
#include <memory>
#include <string>

namespace instance
{

/**
 * @brief Convenience factory for MoE blocks and full MoE decoder models.
 *
 * Produces stock primitives — CAIF_DevicePreNormBlock<float,float> for blocks and
 * CAIF_DeviceNetwork for whole models — from a single configuration.
 * The composer is a shortcut, not a required abstraction layer: callers
 * that need something the configuration does not expose should bypass it
 * and hand-roll the block/model from the underlying primitives
 * (CAIF_DeviceRMSNorm, CAIF_DeviceMultiHeadAttention<float,float>, CAIF_DeviceMoELayer,
 * CAIF_DevicePreNormBlock<float,float>, CAIF_DeviceNetwork).
 *
 * Block composition matches what ANVL_TransformerBuilder assembles today:
 *
 *   CAIF_DevicePreNormBlock
 *     stage 0: (RMSNorm, CAIF_DeviceMultiHeadAttention<float,float>)
 *     stage 1: (RMSNorm, CAIF_DeviceMoELayer)
 *
 * Model composition (BuildModel):
 *
 *   TokenEmbedding
 *   [PositionalEncoding iff pe_mode != None and use_rope == false]
 *   num_layers x MoE block (via BuildMoEBlock)
 *   Final RMSNorm
 *   LinearHead (tied to embedding when tie_weights == true)
 */
class CAIF_MoEComposer:public CAIF_Base
{
  public:

    /**
     * @brief Build a single MoE transformer block.
     *
     * Equivalent to the four-line hand composition of
     * (RMSNorm, MHA, RMSNorm, MoELayer) wrapped in a CAIF_DevicePreNormBlock<float,float>.
     */
    static std::unique_ptr<CAIF_DevicePreNormBlock<float,float>>
      BuildMoEBlock(const CAIF_MoEComposerBlockConfig &cfg,CAIF_CudaStream &stream);

    /**
     * @brief Build a full MoE decoder-only model.
     *
     * Every layer is an MoE block. For interleaved dense/MoE models,
     * hand-compose directly from primitives (see class comment).
     *
     * Aggregate aux loss (balance+z summed) is available via
     * CAIF_DeviceNetwork::AuxLoss(); callers that need the split should
     * bypass the composer and access the primitives directly.
     */
    static std::unique_ptr<CAIF_DeviceNetwork>
      BuildModel(const CAIF_MoEComposerModelConfig &cfg,CAIF_CudaStream &stream);

  protected:

  private:

    CAIF_MoEComposer()=delete;

    // Dtype-templated builders. The public BuildModel/BuildMoEBlock dispatch on
    // the config's compute/storage dtype to the matching instantiation so a
    // bf16 (or fp16) model assembles end to end, not just <float,float>.
    template<typename ComputeT,typename StorageT>
    static std::unique_ptr<CAIF_DevicePreNormBlock<ComputeT,StorageT>> BuildMoEBlockImpl(
                                                                           const CAIF_MoEComposerBlockConfig &cfg,
                                                                           CAIF_CudaStream &stream);

    template<typename ComputeT,typename StorageT>
    static std::unique_ptr<CAIF_DeviceNetwork> BuildModelImpl(const CAIF_MoEComposerModelConfig &cfg,
                                                              CAIF_CudaStream &stream);
};

}//end instance namespace

#endif  // CAIF_MOE_COMPOSER_H
