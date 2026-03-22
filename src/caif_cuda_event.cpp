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

#include "caif_cuda_event.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

namespace instance
{

CAIF_CudaEvent::CAIF_CudaEvent():
#ifdef USE_CAIF_CUDA
                              _event(nullptr),
#endif
                              _valid(false)
{
#ifdef USE_CAIF_CUDA
  // Create event with timing disabled for better performance
  cudaError_t status=cudaEventCreateWithFlags(&_event,cudaEventDisableTiming);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to create CUDA event");
  }
  _valid=true;
#endif
}

CAIF_CudaEvent::~CAIF_CudaEvent()
{
#ifdef USE_CAIF_CUDA
  if(_valid==true&&_event!=nullptr)
  {
    cudaEventDestroy(_event);
    _event=nullptr;
  }
#endif
  _valid=false;
}

CAIF_CudaEvent::CAIF_CudaEvent(CAIF_CudaEvent &&other)noexcept:
#ifdef USE_CAIF_CUDA
                                                           _event(other._event),
#endif
                                                           _valid(other._valid)
{
#ifdef USE_CAIF_CUDA
  other._event=nullptr;
#endif
  other._valid=false;
}

CAIF_CudaEvent &CAIF_CudaEvent::operator=(CAIF_CudaEvent &&other)noexcept
{
  if(this!=&other)
  {
#ifdef USE_CAIF_CUDA
    // Destroy current event if valid
    if(_valid==true&&_event!=nullptr)
    {
      cudaEventDestroy(_event);
    }
    _event=other._event;
    other._event=nullptr;
#endif
    _valid=other._valid;
    other._valid=false;
  }
  return *this;
}

void CAIF_CudaEvent::Synchronize()const
{
#ifdef USE_CAIF_CUDA
  if(_valid==false)
  {
    THROW_CAIFE("Cannot synchronize invalid CUDA event");
  }
  cudaError_t status=cudaEventSynchronize(_event);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to synchronize CUDA event");
  }
#endif
}

bool CAIF_CudaEvent::IsComplete()const
{
#ifdef USE_CAIF_CUDA
  if(_valid==false)
  {
    return true;  // Invalid events are considered complete
  }
  cudaError_t status=cudaEventQuery(_event);
  if(status==cudaSuccess)
  {
    return true;
  }
  else if(status==cudaErrorNotReady)
  {
    return false;
  }
  else
  {
    THROW_CAIFE("Failed to query CUDA event status");
  }
#else
  return true;  // Always complete when CUDA not available
#endif
}

}//end instance namespace
