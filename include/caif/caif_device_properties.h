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
// CAIF - AI Framework
// GPU device properties cache for runtime kernel tuning
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_PROPERTIES_H
#define CAIF_DEVICE_PROPERTIES_H

#include "caif_base.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Per-device GPU capability cache for runtime kernel tuning
 *
 * Queries and caches CUDA device properties on first access. Supports
 * multiple devices — each device_id gets its own instance. Thread-safe
 * static accessors provide lazy initialization with caching.
 *
 * Kernel launchers use these properties to select optimal tile sizes,
 * block dimensions, and shared memory configurations for the target GPU.
 *
 * No CUDA headers are exposed — all CUDA types stay in the .cpp file.
 */
class CAIF_DeviceProperties:public CAIF_Base
{
  public:
    typedef std::vector<std::unique_ptr<CAIF_DeviceProperties>>
      DevicePropertiesVec_t;

    /**
     * @brief Construct properties for a specific device
     *
     * Calls cudaGetDeviceProperties and cudaDeviceGetAttribute to query
     * all relevant properties. Does NOT call cudaSetDevice — safe to
     * query any installed GPU without changing the active device.
     *
     * @param device_id CUDA device index (0-based)
     */
    CAIF_DeviceProperties(const int device_id);

    ~CAIF_DeviceProperties()=default;

    // Non-copyable, movable
    CAIF_DeviceProperties(const CAIF_DeviceProperties &)=delete;
    CAIF_DeviceProperties &operator=(const CAIF_DeviceProperties &)=delete;
    CAIF_DeviceProperties(CAIF_DeviceProperties &&)=default;
    CAIF_DeviceProperties &operator=(CAIF_DeviceProperties &&)=default;

    // Identity
    int DeviceId()const{return _device_id;}
    std::string DeviceName()const{return _name;}
    int ComputeCapabilityMajor()const{return _compute_major;}
    int ComputeCapabilityMinor()const{return _compute_minor;}

    // Memory
    size_t TotalGlobalMemory()const{return _total_global_mem;}
    size_t SharedMemoryPerBlock()const{return _shared_mem_per_block;}
    size_t SharedMemoryPerBlockOptin()const
    {
      return _shared_mem_per_block_optin;
    }
    size_t SharedMemoryPerSM()const{return _shared_mem_per_sm;}

    // Compute
    int MultiprocessorCount()const{return _sm_count;}
    int MaxThreadsPerBlock()const{return _max_threads_per_block;}
    int MaxThreadsPerSM()const{return _max_threads_per_sm;}
    int WarpSize()const{return _warp_size;}
    int RegistersPerSM()const{return _regs_per_sm;}
    int RegistersPerBlock()const{return _regs_per_block;}
    int MaxBlocksPerSM()const{return _max_blocks_per_sm;}

    /**
     * @brief Get the number of installed CUDA devices
     * @return Device count (0 if no CUDA or no GPUs)
     */
    static int DeviceCount();

    /**
     * @brief Get cached properties for a specific device
     *
     * Lazily creates and caches properties on first call per device_id.
     * Thread-safe via internal mutex.
     *
     * @param device_id CUDA device index (0-based)
     * @return Reference to cached properties
     */
    static CAIF_DeviceProperties &ForDevice(const int device_id);

    /**
     * @brief Get properties for the currently active CUDA device
     *
     * Calls cudaGetDevice() to determine active device, then ForDevice().
     *
     * @return Reference to cached properties for the active device
     */
    static CAIF_DeviceProperties &Current();

  protected:

  private:
    int _device_id;
    std::string _name;
    int _compute_major;
    int _compute_minor;
    size_t _total_global_mem;
    size_t _shared_mem_per_block;
    size_t _shared_mem_per_block_optin;
    size_t _shared_mem_per_sm;
    int _sm_count;
    int _max_threads_per_block;
    int _max_threads_per_sm;
    int _warp_size;
    int _regs_per_sm;
    int _regs_per_block;
    int _max_blocks_per_sm;

    static DevicePropertiesVec_t s_device_cache;
    static std::mutex s_cache_mutex;
};

}//end instance namespace

#endif  // CAIF_DEVICE_PROPERTIES_H
