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

#ifndef CAIF_CUDA_STREAM_H
#define CAIF_CUDA_STREAM_H

#include "caif_base.h"
#include "caif_cuda_event.h"
#include <cstdint>
#include <memory>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

namespace instance
{

/**
 * @brief RAII wrapper for CUDA streams
 *
 * Provides safe management of cudaStream_t resources with automatic cleanup.
 * Streams enable concurrent execution of CUDA operations and are fundamental
 * to the device-resident tensor architecture.
 *
 * Key features:
 * - RAII resource management
 * - Event-based synchronization via RecordEvent() and WaitFor()
 * - Static default stream for backward compatibility
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
class CAIF_CudaStream:public CAIF_Base
{
  public:
    /**
     * @brief Construct a new CUDA stream
     *
     * Creates a non-blocking stream that can execute concurrently with other streams.
     */
    CAIF_CudaStream();

    /**
     * @brief Destructor - destroys the CUDA stream
     */
    ~CAIF_CudaStream();

    // Move-only semantics (no copies allowed)
    CAIF_CudaStream(CAIF_CudaStream &&other)noexcept;
    CAIF_CudaStream &operator=(CAIF_CudaStream &&other)noexcept;
    CAIF_CudaStream(const CAIF_CudaStream &)=delete;
    CAIF_CudaStream &operator=(const CAIF_CudaStream &)=delete;

    /**
     * @brief Get the underlying CUDA stream handle
     * @return The cudaStream_t handle (or nullptr/0 if CUDA not available)
     */
#ifdef USE_CAIF_CUDA
    cudaStream_t Handle()const{return _stream;}
#else
    void *Handle()const{return nullptr;}
#endif

    /**
     * @brief Record an event on this stream
     *
     * The returned event marks the completion of all operations submitted
     * to this stream before the RecordEvent call.
     *
     * @return A new event recorded on this stream
     */
    CAIF_CudaEvent RecordEvent()const;

    /**
     * @brief Make this stream wait for an event from another stream
     *
     * All operations submitted to this stream after this call will wait
     * until the specified event completes. This enables cross-stream
     * synchronization without blocking the CPU.
     *
     * @param event The event to wait for
     */
    void WaitFor(const CAIF_CudaEvent &event);

    /**
     * @brief Block the calling CPU thread until all operations on this stream complete
     *
     * This is a synchronization point - use sparingly in performance-critical code.
     * Prefer event-based synchronization (RecordEvent/WaitFor) when possible.
     */
    void Synchronize()const;

    /**
     * @brief Check if the stream is valid (has been created)
     * @return true if the stream handle is valid
     */
    bool IsValid()const{return _valid;}

    /**
     * @brief Get the default stream (lazily created)
     *
     * The default stream is provided for backward compatibility during migration.
     * New code should prefer creating explicit streams for better control over
     * operation ordering and concurrency.
     *
     * @return Reference to the singleton default stream
     */
    static CAIF_CudaStream &Default();

  protected:

  private:
    /**
     * @brief Private constructor for wrapping existing stream handle
     *
     * Used internally for the default stream wrapper.
     *
     * @param stream Existing CUDA stream handle
     * @param owns_stream Whether this object owns the stream (and should destroy it)
     */
#ifdef USE_CAIF_CUDA
    CAIF_CudaStream(cudaStream_t stream,bool owns_stream);
    cudaStream_t _stream;
#endif
    bool _valid;
    bool _owns_stream;

    // Singleton default stream
    static std::unique_ptr<CAIF_CudaStream> s_default_stream;
};

}//end instance namespace

#endif  // CAIF_CUDA_STREAM_H
