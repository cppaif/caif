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
// Device-resident token embedding layer (templated on <ComputeT, StorageT>).
//
// Uniform two-parameter signature with both defaulting to `float`. The
// embedding table itself is allocated at StorageT; lookup kernel reads
// the table and writes the output at StorageT.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceTokenEmbedding:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
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
    CAIF_DeviceTokenEmbedding(CAIF_DeviceTokenEmbedding &&other);
    CAIF_DeviceTokenEmbedding &operator=(CAIF_DeviceTokenEmbedding &&other);

    /**
     * @brief Primary forward path using uint32 host token IDs
     */
    CAIF_DeviceTensor ForwardFromIds(const uint32_t *host_token_ids,
                                     uint32_t batch,
                                     uint32_t seq_len,
                                     bool training);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::TokenEmbedding_e;
    }
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors (StorageDtype()/ComputeDtype() inherited from
    // CAIF_DeviceLayerTyped — no per-layer copy needed).
    uint32_t VocabSize()const{return _config.vocab_size;}
    uint32_t Dim()const{return _config.dim;}

    /**
     * @brief Replace the embedding table with a tensor of shape [vocab_size, dim].
     */
    void LoadEmbeddingTable(CAIF_DeviceTensor &&table);

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

  private:
    CAIF_DeviceTensor &EmbeddingTableMut(){return _embedding_table;}
    void SetEmbeddingTable(CAIF_DeviceTensor &&t){_embedding_table=std::move(t);}

    Config_t _config;

    CAIF_DeviceTensor _embedding_table;       // [vocab_size, dim] at StorageT
    CAIF_DeviceTensor _embedding_table_grad;  // [vocab_size, dim] at StorageT

    // Internal uint32 device buffer for token IDs
    uint32_t *_token_ids_device;
    size_t _token_ids_capacity;

    // Cached for backward
    uint32_t _cached_num_tokens;

    void EnsureTokenIdCapacity(size_t num_tokens);
    void FreeTokenIdBuffer();
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceTokenEmbedding<float,float>;
extern template class CAIF_DeviceTokenEmbedding<float,__half>;
extern template class CAIF_DeviceTokenEmbedding<float,__nv_bfloat16>;
extern template class CAIF_DeviceTokenEmbedding<__half,float>;
extern template class CAIF_DeviceTokenEmbedding<__half,__half>;
extern template class CAIF_DeviceTokenEmbedding<__half,__nv_bfloat16>;
extern template class CAIF_DeviceTokenEmbedding<__nv_bfloat16,float>;
extern template class CAIF_DeviceTokenEmbedding<__nv_bfloat16,__half>;
extern template class CAIF_DeviceTokenEmbedding<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceTokenEmbedding<float,float>;
#endif

}//end instance namespace
