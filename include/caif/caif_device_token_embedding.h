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
#include "caif_device_token_embedding_config.h"
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
    CAIF_DeviceTokenEmbedding(const CAIF_DeviceTokenEmbeddingConfig &config,CAIF_CudaStream &stream);
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
    uint32_t VocabSize()const{return Config().VocabSize();}
    uint32_t Dim()const{return Config().Dim();}
    float OutputScale()const{return Config().OutputScale();}
    const CAIF_DeviceTokenEmbeddingConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DeviceTokenEmbeddingConfig &config){_config=config;}

    CAIF_DeviceTensor &EmbeddingTable(){return *_embedding_table;}
    const CAIF_DeviceTensor &EmbeddingTable()const{return *_embedding_table;}

    CAIF_DeviceTensor &EmbeddingTableGrad(){return *_embedding_table_grad;}
    const CAIF_DeviceTensor &EmbeddingTableGrad()const{return *_embedding_table_grad;}
    void SetEmbeddingTableGrad(CAIF_DeviceTensor &&t){*_embedding_table_grad=std::move(t);}

    // GPU scratch buffer holding the current batch's token IDs (one
    // uint32_t per token) for the embedding-lookup CUDA kernel, which
    // takes a raw `const unsigned int *` argument. Hand-managed via
    // cudaMalloc / cudaFree (see EnsureTokenIdCapacity / FreeTokenIdBuffer)
    // because it's a kernel-API scratch buffer, not a parameter or
    // gradient — no shape, dtype, save/load, or grad accumulation is
    // needed. The `uint32_t *` type spells "pointer to the start of a
    // contiguous device buffer of uint32_t"; the buffer's length is
    // tracked separately in `_token_ids_capacity` (the standard C idiom
    // for a runtime-sized buffer).
    uint32_t *TokenIdsDevice(){return _token_ids_device;}
    const uint32_t *TokenIdsDevice()const{return _token_ids_device;}
    void SetTokenIdsDevice(uint32_t *p){_token_ids_device=p;}

    size_t TokenIdsCapacity()const{return _token_ids_capacity;}
    void SetTokenIdsCapacity(size_t cap){_token_ids_capacity=cap;}

    uint32_t CachedNumTokens()const{return _cached_num_tokens;}
    void SetCachedNumTokens(uint32_t n){_cached_num_tokens=n;}

    /**
     * @brief Replace the embedding table with a tensor of shape [vocab_size, dim].
     * Virtual so CAIF_DeviceSharedTokenEmbedding can override and throw — a
     * borrower cannot replace the donor's storage through its tied pointer.
     */
    virtual void LoadEmbeddingTable(CAIF_DeviceTensor &&table);

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

    /**
     * @brief Subclass-only constructor that borrows the table + grad from a
     * donor instance (T5-style shared encoder/decoder embeddings). Stores
     * the donor's pointers; allocates nothing of its own. The subclass
     * (CAIF_DeviceSharedTokenEmbedding) is responsible for nulling these
     * pointers before this base destructor runs, so the base destructor's
     * `delete` becomes a `delete nullptr` no-op rather than freeing the
     * donor's storage. Shape contract: shared_table.Shape() == [vocab_size, dim].
     */
    CAIF_DeviceTokenEmbedding(const CAIF_DeviceTokenEmbeddingConfig &config,
                              CAIF_DeviceTensor &shared_table,
                              CAIF_DeviceTensor &shared_grad,
                              CAIF_CudaStream &stream);

    // Pointer-rebinding setters + getters used by CAIF_DeviceSharedTokenEmbedding's
    // destructor (and the base's own move ctor / op=) to inspect and null these
    // out before storage gets freed (see the `_embedding_table` data-member
    // comment below for the full rationale). Protected because the borrower
    // subclass needs to call them; not public because external callers must not
    // rebind ownership behind the layer's back.
    CAIF_DeviceTensor *EmbeddingTablePtr()const{return _embedding_table;}
    CAIF_DeviceTensor *EmbeddingTableGradPtr()const{return _embedding_table_grad;}
    void SetEmbeddingTablePtr(CAIF_DeviceTensor *p){_embedding_table=p;}
    void SetEmbeddingTableGradPtr(CAIF_DeviceTensor *p){_embedding_table_grad=p;}

  private:
    CAIF_DeviceTensor &EmbeddingTableMut(){return *_embedding_table;}
    void SetEmbeddingTable(CAIF_DeviceTensor &&t){*_embedding_table=std::move(t);}

    CAIF_DeviceTokenEmbeddingConfig _config;

    // `_embedding_table` and `_embedding_table_grad` are raw pointers rather
    // than value members so that CAIF_DeviceSharedTokenEmbedding can borrow
    // these from a donor instance (the shared encoder/decoder embedding
    // case) without duplicating the storage. The standard owning constructor
    // `new`s these tensors and the destructor `delete`s them; the protected
    // borrower constructor stores donor pointers and the borrower subclass
    // nulls them before the base destructor runs so `delete nullptr` leaves
    // the donor's storage intact. Every read/write of the embedding table
    // and its gradient goes through EmbeddingTable() / EmbeddingTableGrad()
    // accessors, which deref these pointers — no call site needs to know
    // whether storage is owned or borrowed.
    CAIF_DeviceTensor *_embedding_table;       // [vocab_size, dim] at StorageT
    CAIF_DeviceTensor *_embedding_table_grad;  // [vocab_size, dim] at StorageT

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
