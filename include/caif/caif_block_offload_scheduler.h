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
// CAIF - Block-level scheduler that drives Prefetch / Evict on offload-aware
// frozen sublayers.
//
// Owned by a `CAIF_DevicePreNormBlock` (or any container that wants to
// support offload). The block calls `OnEnterStage(i)` / `OnExitStage(i)`
// from its forward / backward sweep; the scheduler issues prefetches for
// the upcoming stage on a separate copy stream so the host->device DMA
// overlaps with the current stage's compute, and evicts a stage's
// pinned-host weights as soon as that stage's backward is done.
//
// MVP scheduling (lookahead=1, lag=0):
//   forward stage i  : prefetch stage i+1 (async on copy stream)
//                      do compute on stage i (already prefetched)
//                      compute stream waits on copy stream's i+1 event
//                      [does not evict during forward — backward will need
//                       these stages too]
//   backward stage i : do compute on stage i (still prefetched from forward)
//                      evict stage i (free GPU scratch)
//                      prefetch stage i-1 if not already in
//
// Stages register themselves at block-construction time. When no
// frozen-offloadable sublayers are registered, the scheduler is a no-op.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_cuda_stream.h"
#include "caif_device_frozen_linear_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace instance
{

class CAIF_BlockOffloadScheduler:public CAIF_Base
{
  public:
    typedef std::vector<std::vector<CAIF_DeviceFrozenLinearBase*>> StageLayersVec_t;

    CAIF_BlockOffloadScheduler();
    ~CAIF_BlockOffloadScheduler();

    CAIF_BlockOffloadScheduler(const CAIF_BlockOffloadScheduler &)=delete;
    CAIF_BlockOffloadScheduler &operator=(const CAIF_BlockOffloadScheduler &)=delete;
    CAIF_BlockOffloadScheduler(CAIF_BlockOffloadScheduler &&other);
    CAIF_BlockOffloadScheduler &operator=(CAIF_BlockOffloadScheduler &&other);

    // Register a frozen-offloadable sublayer at stage `stage_index`. A
    // single stage may have multiple offloadable layers (e.g. an MoE block
    // has many frozen experts). Layers must outlive the scheduler.
    void RegisterAtStage(const size_t stage_index,CAIF_DeviceFrozenLinearBase &layer);

    // Block-level hooks. `compute_stream` is the stream the block is
    // computing on (ctx.Stream() in normal use). The scheduler issues its
    // DMA on a separate copy stream and inserts cross-stream waits so the
    // compute stream sees the prefetched weight as a happens-before
    // dependency.
    void OnEnterForwardStage(const size_t stage_index,CAIF_CudaStream &compute_stream);
    void OnExitForwardStage(const size_t stage_index);
    void OnEnterBackwardStage(const size_t stage_index,CAIF_CudaStream &compute_stream);
    void OnExitBackwardStage(const size_t stage_index);

    // Evict every registered stage. Called at block destruction or when
    // disabling offload.
    void EvictAll();

    size_t StageCount()const{return _stage_layers.size();}
    bool HasAnyOffloadedLayer()const;

    // Layer-level read accessors. The scheduler stores raw non-owning
    // pointers internally (the layers belong to the owning PreNormBlock,
    // not to the scheduler), but the public surface only ever exposes
    // them by reference — the underlying `T*` element type never crosses
    // the accessor boundary. Out-of-bounds indices throw.
    size_t LayerCountAtStage(const size_t stage_index)const;
    bool HasLayerAt(const size_t stage_index,const size_t layer_index)const;
    CAIF_DeviceFrozenLinearBase &LayerAt(const size_t stage_index,
                                          const size_t layer_index);
    const CAIF_DeviceFrozenLinearBase &LayerAt(const size_t stage_index,
                                                const size_t layer_index)const;

  protected:

  private:
    void PrefetchStage(const size_t stage_index,CAIF_CudaStream &stream);
    void EvictStage(const size_t stage_index);
    CAIF_CudaStream &CopyStream();

    // Single-purpose private setters/probes that own all direct touches
    // of `_stage_layers`. Every non-trivial method body routes through
    // these; method bodies never touch `_stage_layers` themselves.
    bool IsLayerRegistered(CAIF_DeviceFrozenLinearBase *layer)const;
    void EnsureStageCapacity(const size_t stage_index);
    void AppendLayerAtStage(const size_t stage_index,CAIF_DeviceFrozenLinearBase &layer);

    StageLayersVec_t _stage_layers;
    std::unique_ptr<CAIF_CudaStream> _copy_stream;
};

}//end instance namespace