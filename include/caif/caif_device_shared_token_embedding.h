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

#pragma once

#include "caif_device_token_embedding.h"

#include <cstddef>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief T5-style shared encoder/decoder token embedding — borrows its
 * `[vocab_size, dim]` table and gradient from a donor CAIF_DeviceTokenEmbedding
 * instance instead of allocating its own.
 *
 * Forward lookups read from the donor's table through the inherited
 * `_embedding_table` pointer; backward gradients atomically accumulate
 * into the donor's `_embedding_table_grad` (via the inherited
 * BackwardImpl, which intentionally does NOT FillZero — per-step zeroing
 * is the donor's ZeroGradients). The donor owns the storage, the
 * parameter (for the optimizer), the save/load surface, and the
 * per-step zero. The borrower exposes a forward/backward kernel
 * pathway through the same physical tensors, and nothing else.
 *
 * Typical use: encoder.embedding is an owning CAIF_DeviceTokenEmbedding;
 * decoder.embedding is a CAIF_DeviceSharedTokenEmbedding pointing at the
 * encoder's table and grad. Both networks then see the same logical
 * parameter without two physical copies.
 */
template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceSharedTokenEmbedding:public CAIF_DeviceTokenEmbedding<ComputeT,StorageT>
{
  public:
    /**
     * @brief Construct a borrower tied to a donor's table + grad. The
     * donor must outlive this object — typically the donor is the
     * encoder's TokenEmbedding and this is the decoder's. Shape
     * contract: donor_table.Shape() == [config.vocab_size, config.dim].
     */
    CAIF_DeviceSharedTokenEmbedding(const CAIF_DeviceTokenEmbeddingConfig &config,
                                    CAIF_DeviceTensor &donor_table,
                                    CAIF_DeviceTensor &donor_grad,
                                    CAIF_CudaStream &stream);

    /**
     * @brief Destructor — nulls the inherited table/grad pointers BEFORE
     * the base destructor runs, so the base's `delete EmbeddingTablePtr()`
     * is a `delete nullptr` no-op rather than freeing the donor's storage
     * out from under the donor on scope exit.
     */
    ~CAIF_DeviceSharedTokenEmbedding()override;

    // Move
    CAIF_DeviceSharedTokenEmbedding(CAIF_DeviceSharedTokenEmbedding &&other);
    CAIF_DeviceSharedTokenEmbedding &operator=(CAIF_DeviceSharedTokenEmbedding &&other);

    /**
     * @brief No-op in the borrower. The donor's ZeroGradients clears the
     * shared gradient table once per training step; if the borrower also
     * zeroed (after the donor's backward had already accumulated into the
     * shared grad for this step) we would wipe the donor's contribution.
     * Atomic-add accumulation in the borrower's BackwardImpl still works
     * — it writes through the shared grad pointer.
     */
    void ZeroGradients()override;

    /**
     * @brief Returns 0 — the borrower exposes no parameters of its own.
     * The donor's CAIF_DeviceTokenEmbedding owns the embedding table for
     * parameter enumeration; if the borrower also reported it as a
     * parameter, the optimizer (which iterates each network's layers and
     * runs Adam on every ParameterTensor) would step the same physical
     * tensor twice per training step — double-bumping Adam's m/v moments
     * and applying two updates worth of step to the shared table.
     */
    size_t ParameterTensorCount()const override;

    /**
     * @brief Throws — the borrower has no own parameters; see
     * ParameterTensorCount() above. The donor's ParameterTensor(0) is
     * the authoritative handle on the shared table for the optimizer
     * and save/load; code walking the network's parameters reaches the
     * table through the donor, not here.
     */
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;

    /**
     * @brief Throws — same rationale as ParameterTensor: the donor owns
     * the gradient tensor for parameter-enumeration purposes. Gradient
     * accumulation into the shared grad still works correctly because
     * the inherited BackwardImpl writes through the deref'd
     * `_embedding_table_grad` pointer (atomicAdd into the donor's grad),
     * a path that does not go through this GradientTensor() accessor.
     */
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;

    /**
     * @brief Returns 0. The shared table is counted once by the donor's
     * TotalParameterCount; double-counting it from the borrower would
     * inflate the network's reported parameter count and any per-layer
     * memory accounting that derives from it.
     */
    size_t TotalParameterCount()const override;

    /**
     * @brief Returns an empty vector. Save/load walks ParameterNames to
     * build the safetensors key list; if the borrower contributed a
     * name, the same logical parameter would be written/read twice
     * (wasted IO, double-write) or — if the borrower-side and
     * donor-side prefixes happen to resolve to the same safetensors
     * key — the save would fail outright on a duplicate-key error.
     */
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    /**
     * @brief Throws — the borrower cannot replace the donor's storage
     * through its tied pointer. To load weights into the shared table,
     * the caller must invoke LoadEmbeddingTable on the donor instance.
     */
    void LoadEmbeddingTable(CAIF_DeviceTensor &&table)override;

  protected:
    using CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::SetEmbeddingTablePtr;
    using CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::SetEmbeddingTableGradPtr;

  private:
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceSharedTokenEmbedding<float,float>;
extern template class CAIF_DeviceSharedTokenEmbedding<float,__half>;
extern template class CAIF_DeviceSharedTokenEmbedding<float,__nv_bfloat16>;
extern template class CAIF_DeviceSharedTokenEmbedding<__half,float>;
extern template class CAIF_DeviceSharedTokenEmbedding<__half,__half>;
extern template class CAIF_DeviceSharedTokenEmbedding<__half,__nv_bfloat16>;
extern template class CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,float>;
extern template class CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,__half>;
extern template class CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceSharedTokenEmbedding<float,float>;
#endif

}//end instance namespace
