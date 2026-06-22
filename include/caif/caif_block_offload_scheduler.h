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
// Scheduling (lookahead=1, lag=0):
//   forward stage i  : compute stream waits on stage i's H2D event
//                      prefetch stage i+1 (async on copy stream, event
//                      recorded after the H2D)
//                      do compute on stage i
//                      on exit: evict stage i — the eviction's stream-side
//                      frees are ordered after compute's reads via a
//                      compute-stream event the copy stream waits on
//                      (read-before-free guard); GPU peak stays bounded by
//                      one stage's working set, backward re-prefetches
//                      lazily
//   backward stage i : compute stream waits on stage i's H2D event
//                      (cold-started on the copy stream if forward evicted
//                      it); prefetch stage i-1 (backward walks high -> low)
//                      do compute on stage i; on exit: guarded evict
//
// All synchronization is GPU-side (cudaStreamWaitEvent) — the host never
// blocks. Stages register themselves at block-construction time. When no
// frozen-offloadable sublayers are registered, the scheduler is a no-op.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_cuda_event.h"
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
    void OnExitForwardStage(const size_t stage_index,CAIF_CudaStream &compute_stream);
    void OnEnterBackwardStage(const size_t stage_index,CAIF_CudaStream &compute_stream);
    void OnExitBackwardStage(const size_t stage_index,CAIF_CudaStream &compute_stream);

    // Evict every registered stage and drop pending prefetch events.
    // Teardown path (block destruction / disabling offload) — the caller
    // must have quiesced compute on the stages first; there is no
    // read-before-free guard here.
    void EvictAll();

    size_t StageCount()const{return StageLayers().size();}
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

    // Issue stage `stage_index`'s H2D on the copy stream and record its
    // completion event. No-op when the stage has no layers, is already
    // fully prefetched, or already has a pending event.
    void IssuePrefetch(const size_t stage_index);
    // GPU-side wait: `compute_stream` waits on the stage's pending H2D
    // event (consumed). Cold-starts the prefetch when none was issued
    // (stage 0 of forward; lazy re-prefetch in backward after the forward
    // sweep evicted the stage).
    void AwaitPrefetch(const size_t stage_index,CAIF_CudaStream &compute_stream);
    // Read-before-free guard: the copy stream waits on an event recorded
    // on `compute_stream`, then the stage is evicted. The weights were
    // allocated on the copy stream, so their cudaFreeAsync lands there —
    // ordered after compute's reads.
    void GuardedEvict(const size_t stage_index,CAIF_CudaStream &compute_stream);
    bool StageFullyPrefetched(const size_t stage_index)const;

    // Single-purpose private setters/probes that own all direct touches
    // of `_stage_layers`. Every non-trivial method body routes through
    // these; method bodies never touch `_stage_layers` themselves.
    bool IsLayerRegistered(CAIF_DeviceFrozenLinearBase *layer)const;
    void EnsureStageCapacity(const size_t stage_index);
    void AppendLayerAtStage(const size_t stage_index,CAIF_DeviceFrozenLinearBase &layer);

    // Single point of access for the stage-layer table; every method
    // body routes through this rather than touching `_stage_layers`.
    const StageLayersVec_t &StageLayers()const{return _stage_layers;}
    StageLayersVec_t &StageLayers(){return _stage_layers;}

    // Single point of access for the per-stage pending H2D-complete
    // events. A null entry means no prefetch is in flight for that stage.
    const CAIF_CudaEvent::PtrVec_t &PrefetchEvents()const{return _prefetch_events;}
    CAIF_CudaEvent::PtrVec_t &PrefetchEvents(){return _prefetch_events;}

    StageLayersVec_t _stage_layers;
    CAIF_CudaEvent::PtrVec_t _prefetch_events;
    std::unique_ptr<CAIF_CudaStream> _copy_stream;
};

}//end instance namespace