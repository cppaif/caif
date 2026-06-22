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

#include "caif_block_offload_scheduler.h"
#include "caif_exception.h"

namespace instance
{

CAIF_BlockOffloadScheduler::CAIF_BlockOffloadScheduler()
try:_stage_layers(),
    _prefetch_events(),
    _copy_stream()
{
}
CAIF_CATCH_BLOCK()

CAIF_BlockOffloadScheduler::~CAIF_BlockOffloadScheduler()
{
}

CAIF_BlockOffloadScheduler::CAIF_BlockOffloadScheduler(CAIF_BlockOffloadScheduler &&other)
try:_stage_layers(std::move(other._stage_layers)),
    _prefetch_events(std::move(other._prefetch_events)),
    _copy_stream(std::move(other._copy_stream))
{
}
CAIF_CATCH_BLOCK()

CAIF_BlockOffloadScheduler &
CAIF_BlockOffloadScheduler::operator=(CAIF_BlockOffloadScheduler &&other)
{
  try
  {
    if(this!=&other)
    {
      _stage_layers=std::move(other._stage_layers);
      _prefetch_events=std::move(other._prefetch_events);
      _copy_stream=std::move(other._copy_stream);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
  return *this;
}

void CAIF_BlockOffloadScheduler::RegisterAtStage(const size_t stage_index,
                                                  CAIF_DeviceFrozenLinearBase &layer)
{
  try
  {
    EnsureStageCapacity(stage_index);
    // Idempotent: skip if `layer` is already registered at any stage.
    // Prevents double-prefetch / double-evict if the caller registers
    // the same layer twice (e.g. during-build pre-registration plus a
    // redundant post-build sweep).
    if(IsLayerRegistered(&layer)==true)
    {
      return;
    }
    AppendLayerAtStage(stage_index,layer);
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_BlockOffloadScheduler::LayerCountAtStage(const size_t stage_index)const
{
  try
  {
    if(stage_index>=StageLayers().size())
    {
      return 0u;
    }
    return StageLayers()[stage_index].size();
  }
  CAIF_CATCH_BLOCK()
  return 0u;
}

bool CAIF_BlockOffloadScheduler::HasLayerAt(const size_t stage_index,
                                             const size_t layer_index)const
{
  try
  {
    if(stage_index>=StageLayers().size())
    {
      return false;
    }
    return layer_index<StageLayers()[stage_index].size();
  }
  CAIF_CATCH_BLOCK()
  return false;
}

CAIF_DeviceFrozenLinearBase &
CAIF_BlockOffloadScheduler::LayerAt(const size_t stage_index,const size_t layer_index)
{
  try
  {
    if(HasLayerAt(stage_index,layer_index)==false)
    {
      THROW_CAIFE("CAIF_BlockOffloadScheduler::LayerAt: index out of range");
    }
    CAIF_DeviceFrozenLinearBase *p=StageLayers()[stage_index][layer_index];
    if(p==nullptr)
    {
      THROW_CAIFE("CAIF_BlockOffloadScheduler::LayerAt: registered layer is null");
    }
    return *p;
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceFrozenLinearBase &
CAIF_BlockOffloadScheduler::LayerAt(const size_t stage_index,const size_t layer_index)const
{
  try
  {
    if(HasLayerAt(stage_index,layer_index)==false)
    {
      THROW_CAIFE("CAIF_BlockOffloadScheduler::LayerAt: index out of range");
    }
    const CAIF_DeviceFrozenLinearBase *p=StageLayers()[stage_index][layer_index];
    if(p==nullptr)
    {
      THROW_CAIFE("CAIF_BlockOffloadScheduler::LayerAt: registered layer is null");
    }
    return *p;
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_BlockOffloadScheduler::IsLayerRegistered(CAIF_DeviceFrozenLinearBase *layer)const
{
  try
  {
    const size_t ns=StageCount();
    for(size_t s=0;s<ns;++s)
    {
      const size_t nl=LayerCountAtStage(s);
      for(size_t i=0;i<nl;++i)
      {
        if(&LayerAt(s,i)==layer)
        {
          return true;
        }
      }
    }
    return false;
  }
  CAIF_CATCH_BLOCK()
  return false;
}

void CAIF_BlockOffloadScheduler::EnsureStageCapacity(const size_t stage_index)
{
  try
  {
    if(StageLayers().size()<=stage_index)
    {
      StageLayers().resize(stage_index+1u);
    }
    if(PrefetchEvents().size()<=stage_index)
    {
      PrefetchEvents().resize(stage_index+1u);
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::AppendLayerAtStage(const size_t stage_index,
                                                     CAIF_DeviceFrozenLinearBase &layer)
{
  try
  {
    StageLayers()[stage_index].push_back(&layer);
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_BlockOffloadScheduler::HasAnyOffloadedLayer()const
{
  try
  {
    const size_t ns=StageCount();
    for(size_t s=0;s<ns;++s)
    {
      const size_t nl=LayerCountAtStage(s);
      for(size_t i=0;i<nl;++i)
      {
        if(LayerAt(s,i).OffloadPolicy()==
           CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e)
        {
          return true;
        }
      }
    }
    return false;
  }
  CAIF_CATCH_BLOCK()
  return false;
}

CAIF_CudaStream &CAIF_BlockOffloadScheduler::CopyStream()
{
  try
  {
    if(_copy_stream==nullptr)
    {
      _copy_stream.reset(new CAIF_CudaStream());
    }
    return *_copy_stream;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::PrefetchStage(const size_t stage_index,CAIF_CudaStream &stream)
{
  try
  {
    const size_t nl=LayerCountAtStage(stage_index);
    for(size_t i=0;i<nl;++i)
    {
      LayerAt(stage_index,i).Prefetch(stream);
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::EvictStage(const size_t stage_index)
{
  try
  {
    const size_t nl=LayerCountAtStage(stage_index);
    for(size_t i=0;i<nl;++i)
    {
      LayerAt(stage_index,i).Evict();
    }
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_BlockOffloadScheduler::StageFullyPrefetched(const size_t stage_index)const
{
  try
  {
    const size_t nl=LayerCountAtStage(stage_index);
    for(size_t i=0;i<nl;++i)
    {
      if(LayerAt(stage_index,i).IsPrefetched()==false)
      {
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK()
  return true;
}

void CAIF_BlockOffloadScheduler::IssuePrefetch(const size_t stage_index)
{
  try
  {
    if(LayerCountAtStage(stage_index)==0u)
    {
      return;
    }
    if(PrefetchEvents()[stage_index]!=nullptr)
    {
      return;
    }
    if(StageFullyPrefetched(stage_index)==true)
    {
      return;
    }
    PrefetchStage(stage_index,CopyStream());
    PrefetchEvents()[stage_index]=std::make_unique<CAIF_CudaEvent>(CopyStream().RecordEvent());
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::AwaitPrefetch(const size_t stage_index,
                                               CAIF_CudaStream &compute_stream)
{
  try
  {
    if(LayerCountAtStage(stage_index)==0u)
    {
      return;
    }
    if(StageFullyPrefetched(stage_index)==false &&
       PrefetchEvents()[stage_index]==nullptr)
    {
      // Cold start: forward stage 0, or backward re-entering a stage the
      // forward sweep evicted. Issue the H2D now; the wait below covers it.
      IssuePrefetch(stage_index);
    }
    if(PrefetchEvents()[stage_index]!=nullptr)
    {
      compute_stream.WaitFor(*PrefetchEvents()[stage_index]);
      PrefetchEvents()[stage_index].reset();
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::GuardedEvict(const size_t stage_index,
                                              CAIF_CudaStream &compute_stream)
{
  try
  {
    if(LayerCountAtStage(stage_index)==0u)
    {
      return;
    }
    // Read-before-free guard: the stage's weights were allocated on the
    // copy stream, so their cudaFreeAsync lands there. Making the copy
    // stream wait on compute's reads orders the frees after them with no
    // host stall.
    CAIF_CudaEvent compute_done=compute_stream.RecordEvent();
    CopyStream().WaitFor(compute_done);
    EvictStage(stage_index);
    PrefetchEvents()[stage_index].reset();
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::OnEnterForwardStage(const size_t stage_index,
                                                     CAIF_CudaStream &compute_stream)
{
  try
  {
    if(HasAnyOffloadedLayer()==false)
    {
      return;
    }
    // Wait (GPU-side) on this stage's H2D — cold-started on the copy
    // stream for stage 0 — then issue the next stage's prefetch so its
    // DMA overlaps this stage's compute.
    AwaitPrefetch(stage_index,compute_stream);
    if(stage_index+1u<StageCount())
    {
      IssuePrefetch(stage_index+1u);
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::OnExitForwardStage(const size_t stage_index,
                                                    CAIF_CudaStream &compute_stream)
{
  try
  {
    if(HasAnyOffloadedLayer()==false)
    {
      return;
    }
    // Evict this stage's frozen weights as soon as its forward compute
    // completes (guarded — the frees are stream-ordered behind compute's
    // reads). Keeping every block's weights resident through the forward
    // sweep only works when the frozen base fits on GPU; full-depth bf16
    // add-MoE on 27-layer DSv2-Lite has ~30 GB of frozen routed-expert
    // weights — past a 32 GB card. Backward re-prefetches lazily, so the
    // round-trip cost is one extra host->device DMA per block per step in
    // exchange for a GPU peak bounded by one block's working set.
    GuardedEvict(stage_index,compute_stream);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::OnEnterBackwardStage(const size_t stage_index,
                                                      CAIF_CudaStream &compute_stream)
{
  try
  {
    if(HasAnyOffloadedLayer()==false)
    {
      return;
    }
    // The forward sweep evicted this stage; AwaitPrefetch cold-starts the
    // re-prefetch when no lookahead H2D is pending. Backward walks stages
    // high -> low, so the lookahead target is stage_index-1.
    AwaitPrefetch(stage_index,compute_stream);
    if(stage_index>0u)
    {
      IssuePrefetch(stage_index-1u);
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::OnExitBackwardStage(const size_t stage_index,
                                                     CAIF_CudaStream &compute_stream)
{
  try
  {
    if(HasAnyOffloadedLayer()==false)
    {
      return;
    }
    // Stage's backward is done — its frozen weight is no longer needed in
    // this iteration. Free the GPU scratch (guarded).
    GuardedEvict(stage_index,compute_stream);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::EvictAll()
{
  try
  {
    const size_t ns=StageCount();
    for(size_t s=0;s<ns;++s)
    {
      EvictStage(s);
    }
    const size_t ne=PrefetchEvents().size();
    for(size_t s=0;s<ne;++s)
    {
      PrefetchEvents()[s].reset();
    }
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace