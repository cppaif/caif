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

#include "caif_device_properties.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

using namespace instance;

// Static members
CAIF_DeviceProperties::DevicePropertiesVec_t
  CAIF_DeviceProperties::s_device_cache;
std::mutex CAIF_DeviceProperties::s_cache_mutex;

CAIF_DeviceProperties::CAIF_DeviceProperties(const int device_id):_device_id(device_id),
                                                                  _name("CPU"),
                                                                  _compute_major(0),
                                                                  _compute_minor(0),
                                                                  _total_global_mem(0),
                                                                  _shared_mem_per_block(49152),
                                                                  _shared_mem_per_block_optin(49152),
                                                                  _shared_mem_per_sm(49152),
                                                                  _sm_count(1),
                                                                  _max_threads_per_block(1024),
                                                                  _max_threads_per_sm(2048),
                                                                  _warp_size(32),
                                                                  _regs_per_sm(65536),
                                                                  _regs_per_block(65536),
                                                                  _max_blocks_per_sm(16)
{
  try
  {
#ifdef USE_CAIF_CUDA
    cudaDeviceProp props;
    cudaError_t status=cudaGetDeviceProperties(&props,device_id);
    if(status!=cudaSuccess)
    {
      THROW_CAIFE("cudaGetDeviceProperties failed for device "
                  +std::to_string(device_id));
    }

    _name=props.name;
    _compute_major=props.major;
    _compute_minor=props.minor;
    _total_global_mem=props.totalGlobalMem;
    _shared_mem_per_block=props.sharedMemPerBlock;
    _shared_mem_per_sm=props.sharedMemPerMultiprocessor;
    _sm_count=props.multiProcessorCount;
    _max_threads_per_block=props.maxThreadsPerBlock;
    _max_threads_per_sm=props.maxThreadsPerMultiProcessor;
    _warp_size=props.warpSize;
    _regs_per_sm=props.regsPerMultiprocessor;
    _regs_per_block=props.regsPerBlock;

    // Max dynamic shared memory per block after opt-in
    int optin_smem=0;
    cudaError_t attr_status=cudaDeviceGetAttribute(&optin_smem,
                                                   cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                   device_id);
    if(attr_status==cudaSuccess)
    {
      _shared_mem_per_block_optin=static_cast<size_t>(optin_smem);
    }
    else
    {
      _shared_mem_per_block_optin=_shared_mem_per_block;
    }

    // Max blocks per SM
    int max_blocks=0;
    attr_status=cudaDeviceGetAttribute(&max_blocks,
                                       cudaDevAttrMaxBlocksPerMultiprocessor,
                                       device_id);
    if(attr_status==cudaSuccess)
    {
      _max_blocks_per_sm=max_blocks;
    }
#endif
  }
  CAIF_CATCH_BLOCK();
}

int CAIF_DeviceProperties::DeviceCount()
{
  int count=0;
  try
  {
#ifdef USE_CAIF_CUDA
    cudaError_t status=cudaGetDeviceCount(&count);
    if(status!=cudaSuccess)
    {
      count=0;
    }
#endif
  }
  CAIF_CATCH_BLOCK();
  return count;
}

CAIF_DeviceProperties &CAIF_DeviceProperties::ForDevice(const int device_id)
{
  std::lock_guard<std::mutex> lock(s_cache_mutex);

  // Grow cache vector if needed
  const size_t idx=static_cast<size_t>(device_id);
  if(idx>=s_device_cache.size())
  {
    s_device_cache.resize(idx+1);
  }

  // Lazily create
  if(s_device_cache[idx]==nullptr)
  {
    s_device_cache[idx]=std::make_unique<CAIF_DeviceProperties>(device_id);
  }

  return *s_device_cache[idx];
}

CAIF_DeviceProperties &CAIF_DeviceProperties::Current()
{
  int device_id=0;
#ifdef USE_CAIF_CUDA
  cudaGetDevice(&device_id);
#endif
  return ForDevice(device_id);
}
