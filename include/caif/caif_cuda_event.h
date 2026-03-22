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

#ifndef CAIF_CUDA_EVENT_H
#define CAIF_CUDA_EVENT_H

#include "caif_base.h"
#include <cstdint>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

namespace instance
{

/**
 * @brief RAII wrapper for CUDA events
 *
 * Provides safe management of cudaEvent_t resources with automatic cleanup.
 * Events are used for fine-grained synchronization between CUDA streams
 * without requiring full device synchronization.
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
class CAIF_CudaEvent:public CAIF_Base
{
  public:
    /**
     * @brief Construct a new CUDA event
     *
     * Creates an event with cudaEventDisableTiming for better performance
     * when timing information is not needed.
     */
    CAIF_CudaEvent();

    /**
     * @brief Destructor - destroys the CUDA event
     */
    ~CAIF_CudaEvent();

    // Move-only semantics (no copies allowed)
    CAIF_CudaEvent(CAIF_CudaEvent &&other)noexcept;
    CAIF_CudaEvent &operator=(CAIF_CudaEvent &&other)noexcept;
    CAIF_CudaEvent(const CAIF_CudaEvent &)=delete;
    CAIF_CudaEvent &operator=(const CAIF_CudaEvent &)=delete;

    /**
     * @brief Get the underlying CUDA event handle
     * @return The cudaEvent_t handle (or nullptr if CUDA not available)
     */
#ifdef USE_CAIF_CUDA
    cudaEvent_t Handle()const{return _event;}
#else
    void *Handle()const{return nullptr;}
#endif

    /**
     * @brief Block the calling CPU thread until this event completes
     *
     * This is a synchronization point - use sparingly in performance-critical code.
     */
    void Synchronize()const;

    /**
     * @brief Check if all work captured by this event has completed
     * @return true if completed, false if still pending
     */
    bool IsComplete()const;

    /**
     * @brief Check if the event is valid (has been created)
     * @return true if the event handle is valid
     */
    bool IsValid()const{return _valid;}

  protected:

  private:
#ifdef USE_CAIF_CUDA
    cudaEvent_t _event;
#endif
    bool _valid;
};

}//end instance namespace

#endif  // CAIF_CUDA_EVENT_H
