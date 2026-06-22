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
// CAIF - CPU-side pinned host tensor for layered offload (O1 of
// CPU_OFFLOAD_DESIGN.md).
//
// Owns a `cudaMallocHost`-pinned buffer for one tensor's worth of bytes, sized
// at construction by (shape, dtype). The pinned attribute is what makes
// `cudaMemcpyAsync` actually run async from a copy stream — pageable host
// memory forces the driver to stage through an internal pinned region and
// blocks the calling stream. This class is the host-pinned authority for
// CPU-offload of frozen weights and offloaded optimizer state.
//
// Lifecycle for the frozen-weight case:
//   1. Construct (allocates pinned host buffer)
//   2. Caller writes the weight bytes into the host buffer (via HostPtr())
//   3. Layer's Prefetch(stream) calls PrefetchToDevice(stream) -> CAIF_DeviceTensor
//      living on `stream`'s associated device, returned by value
//   4. Compute runs against the device tensor on `stream`
//   5. Layer's Evict() drops the device tensor; pinned host buffer stays
//   6. Repeat 3-5 each forward/backward; host buffer never moves
//
// For optimizer state (m, v, fp32 master):
//   - Construct one pinned tensor per per-param-group buffer
//   - At optimizer step: PrefetchToDevice -> kernel step -> CopyFromDevice
//     to write the updated master back to host -> drop the GPU scratch
//
// Move-only — the pinned-RAM buffer is owned, not shared. cudaFreeHost the
// buffer in the dtor; don't double-free on move.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_data_type.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace instance
{

class CAIF_HostPinnedTensor:public CAIF_Base
{
  public:
    CAIF_HostPinnedTensor()=delete;

    // Allocate `shape.product() * sizeof(dtype)` bytes of pinned host memory.
    // Throws on cudaMallocHost failure with a free/total RAM diagnostic.
    CAIF_HostPinnedTensor(const std::vector<uint32_t> &shape,
                          const CAIF_DataType::CAIF_DataType_e dtype);

    ~CAIF_HostPinnedTensor();

    // Move-only — the pinned-RAM allocation is owned.
    CAIF_HostPinnedTensor(const CAIF_HostPinnedTensor &)=delete;
    CAIF_HostPinnedTensor &operator=(const CAIF_HostPinnedTensor &)=delete;
    CAIF_HostPinnedTensor(CAIF_HostPinnedTensor &&other);
    CAIF_HostPinnedTensor &operator=(CAIF_HostPinnedTensor &&other);

    const std::vector<uint32_t> &Shape()const{return _shape;}
    CAIF_DataType::CAIF_DataType_e Dtype()const{return _dtype;}
    size_t Bytes()const{return _bytes;}
    void *HostPtr(){return _host_ptr;}
    const void *HostPtr()const{return _host_ptr;}
    bool IsAllocated()const{return _host_ptr!=nullptr;}

    // Bring this host buffer onto the device as a fresh CAIF_DeviceTensor.
    // Allocates a new device tensor on `stream`'s device at our (shape, dtype)
    // and DMAs `_bytes` from the pinned host buffer to it via cudaMemcpyAsync.
    // Async — caller must order subsequent compute on `stream`.
    CAIF_DeviceTensor PrefetchToDevice(CAIF_CudaStream &stream)const;

    // Pull the bytes from `src` back into this host buffer (writeback path
    // for offloaded optimizer state). Shape and dtype must match. The
    // download is sync via CAIF_DeviceTensor::CopyToHostRaw (stream-sync
    // then cudaMemcpy device->host) — fine for the optimizer-step boundary.
    void CopyFromDevice(const CAIF_DeviceTensor &src);

    // Async writeback: DMA `_bytes` from `src` (device) into this pinned host
    // buffer via cudaMemcpyAsync on `stream`, with NO stream sync. Correct for
    // the offloaded-optimizer step loop — the next step's PrefetchToDevice H2D
    // is ordered after this copy on the same stream, so the per-parameter
    // stalls of the synchronizing CopyFromDevice are removed. A host-side read
    // of HostPtr() (e.g. a checkpoint save) must sync `stream` first.
    void CopyFromDeviceAsync(const CAIF_DeviceTensor &src,CAIF_CudaStream &stream);

  protected:

  private:
    // Internal setters — method bodies route through these per
    // CODING_GUIDELINES.md §Member Access.
    void SetShape(std::vector<uint32_t> &&v){_shape=std::move(v);}
    void SetDtype(const CAIF_DataType::CAIF_DataType_e v){_dtype=v;}
    void SetHostPtr(void *p){_host_ptr=p;}
    void SetBytes(const size_t v){_bytes=v;}

    std::vector<uint32_t> _shape;
    CAIF_DataType::CAIF_DataType_e _dtype;
    void *_host_ptr;
    size_t _bytes;
};

}//end instance namespace
