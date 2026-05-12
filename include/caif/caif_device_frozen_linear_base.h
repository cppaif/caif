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
// CAIF_DeviceFrozenLinearBase — pure virtual mix-in interface that exposes
// FrozenLinear-specific methods (LoadFromTensor, LoadScalesFromHost,
// ClearFP32Cache, NeedsScales, dim accessors) on a non-templated handle.
//
// CAIF_DeviceFrozenLinear<ComputeT, StorageT> inherits both
// CAIF_DeviceLayerTyped<ComputeT, StorageT> (for the polymorphic layer
// interface) and CAIF_DeviceFrozenLinearBase (for the FrozenLinear-
// specific interface). The two paths do not share a common ancestor, so
// no diamond inheritance.
//
// The runtime factory returns std::unique_ptr<CAIF_DeviceLayer>; callers
// that need FrozenLinear-specific methods dynamic_cast to
// CAIF_DeviceFrozenLinearBase* (or to the concrete cell). This matches the
// CAIF_DeviceDenseLayer dispatch pattern from templating-final.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_tensor.h"
#include "caif_ops.h"
#include "caif_offload_policy.h"
#include "caif_cuda_stream.h"

#include <cstddef>
#include <cstdint>

namespace instance
{

class CAIF_DeviceFrozenLinearBase
{
  public:
    virtual ~CAIF_DeviceFrozenLinearBase()=default;

    virtual void LoadFromTensor(CAIF_DeviceTensor &&weight)=0;
    virtual void LoadScalesFromHost(const void *data,size_t num_bytes)=0;
    virtual void ClearFP32Cache()=0;
    virtual bool NeedsScales()const=0;
    virtual CAIF_Ops::QuantScheme_e Int8Scheme()const=0;
    virtual bool CacheFP32()const=0;
    virtual uint32_t InputDim()const=0;
    virtual uint32_t OutputDim()const=0;

    // CPU offload surface — non-templated entry points for the
    // block-level scheduler. Default no-ops on this interface, so
    // subclasses that don't support offload don't need to override.
    virtual void SetOffloadPolicy(const CAIF_OffloadPolicy::CAIF_OffloadPolicy_e p)
    {
      static_cast<void>(p);
    }
    virtual CAIF_OffloadPolicy::CAIF_OffloadPolicy_e OffloadPolicy()const
    {
      return CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::GpuResident_e;
    }
    virtual void Prefetch(CAIF_CudaStream &stream)
    {
      static_cast<void>(stream);
    }
    virtual void Evict(){}
    virtual bool IsPrefetched()const{return true;}

  protected:

  private:
};

}//end instance namespace
