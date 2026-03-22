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
#include "cuda/cuda_runtime_api.h"
#endif

namespace instance
{

// Static singleton default stream
std::unique_ptr<CAIF_CudaStream> CAIF_CudaStream::s_default_stream=nullptr;

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
  _valid=true;
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
  if(_valid==true&&_owns_stream==true&&_stream!=nullptr)
  {
    cudaStreamDestroy(_stream);
    _stream=nullptr;
  }
#endif
  _valid=false;
}

CAIF_CudaStream::CAIF_CudaStream(CAIF_CudaStream &&other)noexcept:
#ifdef USE_CAIF_CUDA
                                                              _stream(other._stream),
#endif
                                                              _valid(other._valid),
                                                              _owns_stream(other._owns_stream)
{
#ifdef USE_CAIF_CUDA
  other._stream=nullptr;
#endif
  other._valid=false;
  other._owns_stream=false;
}

CAIF_CudaStream &CAIF_CudaStream::operator=(CAIF_CudaStream &&other)noexcept
{
  if(this!=&other)
  {
#ifdef USE_CAIF_CUDA
    // Destroy current stream if we own it
    if(_valid==true&&_owns_stream==true&&_stream!=nullptr)
    {
      cudaStreamDestroy(_stream);
    }
    _stream=other._stream;
    other._stream=nullptr;
#endif
    _valid=other._valid;
    _owns_stream=other._owns_stream;
    other._valid=false;
    other._owns_stream=false;
  }
  return *this;
}

CAIF_CudaEvent CAIF_CudaStream::RecordEvent()const
{
  CAIF_CudaEvent event;
#ifdef USE_CAIF_CUDA
  if(_valid==false)
  {
    THROW_CAIFE("Cannot record event on invalid CUDA stream");
  }
  cudaError_t status=cudaEventRecord(event.Handle(),_stream);
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
  if(_valid==false)
  {
    THROW_CAIFE("Cannot wait on invalid CUDA stream");
  }
  if(event.IsValid()==false)
  {
    THROW_CAIFE("Cannot wait for invalid CUDA event");
  }
  cudaError_t status=cudaStreamWaitEvent(_stream,event.Handle(),0);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to make CUDA stream wait for event");
  }
#endif
}

void CAIF_CudaStream::Synchronize()const
{
#ifdef USE_CAIF_CUDA
  if(_valid==false)
  {
    THROW_CAIFE("Cannot synchronize invalid CUDA stream");
  }
  cudaError_t status=cudaStreamSynchronize(_stream);
  if(status!=cudaSuccess)
  {
    THROW_CAIFE("Failed to synchronize CUDA stream");
  }
#endif
}

CAIF_CudaStream &CAIF_CudaStream::Default()
{
#ifdef USE_CAIF_CUDA
  if(s_default_stream==nullptr)
  {
    // Use the CUDA default stream (0/nullptr) - we don't own it
    s_default_stream=std::unique_ptr<CAIF_CudaStream>(new CAIF_CudaStream(nullptr,false));
  }
  return *s_default_stream;
#else
  // When CUDA not available, return a dummy stream
  if(s_default_stream==nullptr)
  {
    s_default_stream=std::make_unique<CAIF_CudaStream>();
  }
  return *s_default_stream;
#endif
}

}//end instance namespace
