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
    _copy_stream()
{
}
CAIF_CATCH_BLOCK()

CAIF_BlockOffloadScheduler::~CAIF_BlockOffloadScheduler()
{
}

CAIF_BlockOffloadScheduler::CAIF_BlockOffloadScheduler(CAIF_BlockOffloadScheduler &&other)
try:_stage_layers(std::move(other._stage_layers)),
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
    if(stage_index>=_stage_layers.size())
    {
      return 0u;
    }
    return _stage_layers[stage_index].size();
  }
  CAIF_CATCH_BLOCK()
  return 0u;
}

bool CAIF_BlockOffloadScheduler::HasLayerAt(const size_t stage_index,
                                             const size_t layer_index)const
{
  try
  {
    if(stage_index>=_stage_layers.size())
    {
      return false;
    }
    return layer_index<_stage_layers[stage_index].size();
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
    CAIF_DeviceFrozenLinearBase *p=_stage_layers[stage_index][layer_index];
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
    const CAIF_DeviceFrozenLinearBase *p=_stage_layers[stage_index][layer_index];
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
    if(_stage_layers.size()<=stage_index)
    {
      _stage_layers.resize(stage_index+1u);
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::AppendLayerAtStage(const size_t stage_index,
                                                     CAIF_DeviceFrozenLinearBase &layer)
{
  try
  {
    _stage_layers[stage_index].push_back(&layer);
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

void CAIF_BlockOffloadScheduler::OnEnterForwardStage(const size_t stage_index,
                                                      CAIF_CudaStream &compute_stream)
{
  try
  {
    if(HasAnyOffloadedLayer()==false)
    {
      return;
    }
    // MVP: prefetch this stage on the compute stream (synchronous wrt
    // compute, so the weight is guaranteed visible by the time the stage
    // runs). Async-overlap with prior stage's compute is a follow-up
    // optimization (would issue on copy_stream and insert a cross-stream
    // wait); this simpler path is correct first, fast later.
    PrefetchStage(stage_index,compute_stream);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::OnExitForwardStage(const size_t stage_index)
{
  try
  {
    if(HasAnyOffloadedLayer()==false)
    {
      return;
    }
    // Evict this stage's frozen weights immediately after its forward
    // compute completes. The earlier "no eviction during forward"
    // policy assumed the entire frozen base fits on GPU; that only
    // holds for small models. Full-depth bf16 add-MoE on 27-layer
    // DSv2-Lite has ~30 GB of frozen routed-expert weights — way past
    // a 32 GB card budget when ALL blocks' weights stay resident
    // through the forward sweep. Backward's `OnEnterBackwardStage`
    // re-prefetches lazily (PrefetchStage is idempotent), so the
    // round-trip cost is one extra host->device DMA per block per
    // step in exchange for keeping GPU peak bounded by the working
    // set of one block at a time.
    EvictStage(stage_index);
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
    // Forward kept the weight prefetched; backward only needs to ensure the
    // weight is still loaded. If a prior backward stage evicted this stage
    // (e.g. via EvictAll on context boundary), re-prefetch.
    PrefetchStage(stage_index,compute_stream);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BlockOffloadScheduler::OnExitBackwardStage(const size_t stage_index)
{
  try
  {
    if(HasAnyOffloadedLayer()==false)
    {
      return;
    }
    // Stage's backward is done — its frozen weight is no longer needed in
    // this iteration. Free the GPU scratch.
    EvictStage(stage_index);
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
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace