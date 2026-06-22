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

#include "caif_cuda_stream.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#endif

namespace instance
{

// Static singleton default stream
std::unique_ptr<CAIF_CudaStream> CAIF_CudaStream::_default_stream=nullptr;

CAIF_CudaStream::CAIF_CudaStream():
#ifdef USE_CAIF_CUDA
                                _stream(nullptr),
#endif
                                _valid(false),
                                _owns_stream(true)
{
#ifdef USE_CAIF_CUDA
  // Create a non-blocking stream for concurrent execution
  cudaError_t status=cudaStreamCreateWithFlags(&_stream,cudaStreamNonBlocking);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to create CUDA stream");
  }
  SetValid(true);
#endif
}

#ifdef USE_CAIF_CUDA
CAIF_CudaStream::CAIF_CudaStream(cudaStream_t stream,bool owns_stream):_stream(stream),
                                                                     _valid(true),
                                                                     _owns_stream(owns_stream)
{
}
#endif

CAIF_CudaStream::~CAIF_CudaStream()
{
#ifdef USE_CAIF_CUDA
  if(IsValid()==true&&OwnsStream()==true&&Handle()!=nullptr)
  {
    cudaStreamDestroy(Handle());
    SetStream(nullptr);
  }
#endif
  SetValid(false);
}

CAIF_CudaStream::CAIF_CudaStream(CAIF_CudaStream &&other)noexcept:
#ifdef USE_CAIF_CUDA
                                                              _stream(other._stream),
#endif
                                                              _valid(other._valid),
                                                              _owns_stream(other._owns_stream)
{
#ifdef USE_CAIF_CUDA
  other.SetStream(nullptr);
#endif
  other.SetValid(false);
  other.SetOwnsStream(false);
}

CAIF_CudaStream &CAIF_CudaStream::operator=(CAIF_CudaStream &&other)noexcept
{
  if(this!=&other)
  {
#ifdef USE_CAIF_CUDA
    // Destroy current stream if we own it
    if(IsValid()==true&&OwnsStream()==true&&Handle()!=nullptr)
    {
      cudaStreamDestroy(Handle());
    }
    SetStream(other.Handle());
    other.SetStream(nullptr);
#endif
    SetValid(other.IsValid());
    SetOwnsStream(other.OwnsStream());
    other.SetValid(false);
    other.SetOwnsStream(false);
  }
  return *this;
}

CAIF_CudaEvent CAIF_CudaStream::RecordEvent()const
{
  CAIF_CudaEvent event;
#ifdef USE_CAIF_CUDA
  if(IsValid()==false)
  {
    THROW_CAIFE("Cannot record event on invalid CUDA stream");
  }
  cudaError_t status=cudaEventRecord(event.Handle(),Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to record CUDA event on stream");
  }
#endif
  return event;
}

void CAIF_CudaStream::WaitFor(const CAIF_CudaEvent &event)
{
#ifdef USE_CAIF_CUDA
  if(IsValid()==false)
  {
    THROW_CAIFE("Cannot wait on invalid CUDA stream");
  }
  if(event.IsValid()==false)
  {
    THROW_CAIFE("Cannot wait for invalid CUDA event");
  }
  cudaError_t status=cudaStreamWaitEvent(Handle(),event.Handle(),0);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to make CUDA stream wait for event");
  }
#endif
}

void CAIF_CudaStream::Synchronize()const
{
#ifdef USE_CAIF_CUDA
  if(IsValid()==false)
  {
    THROW_CAIFE("Cannot synchronize invalid CUDA stream");
  }
  cudaError_t status=cudaStreamSynchronize(Handle());
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to synchronize CUDA stream");
  }
#endif
}

CAIF_CudaStream &CAIF_CudaStream::Default()
{
#ifdef USE_CAIF_CUDA
  if(_default_stream==nullptr)
  {
    // Use the CUDA default stream (0/nullptr) - we don't own it
    _default_stream=std::unique_ptr<CAIF_CudaStream>(new CAIF_CudaStream(nullptr,false));
  }
  return *_default_stream;
#else
  // When CUDA not available, return a dummy stream
  if(_default_stream==nullptr)
  {
    _default_stream=std::make_unique<CAIF_CudaStream>();
  }
  return *_default_stream;
#endif
}

}//end instance namespace
