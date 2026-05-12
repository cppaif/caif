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
// Templated mid-level base for every dtype-aware device layer.
//
// Inherits from CAIF_DeviceLayer (the dtype-erased polymorphic root used
// by CAIF_DeviceContainer's std::unique_ptr<CAIF_DeviceLayer> sublayer
// vector). Adds a single uniform <ComputeT, StorageT=ComputeT> template
// signature, plus the dtype-introspection accessors and shared protected
// helpers (boundary check, output allocation, cuBLAS compute-type
// resolution, typed pointer accessor) that every templated layer would
// otherwise duplicate.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_run_context.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif
#include "caif_data_type.h"
#include "caif_exception.h"

#include <cstdint>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT=ComputeT>
class CAIF_DeviceLayerTyped:public CAIF_DeviceLayer
{
  public:
    typedef ComputeT compute_t;
    typedef StorageT storage_t;

    CAIF_DeviceLayerTyped(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream){}
    CAIF_DeviceLayerTyped():CAIF_DeviceLayer(){}
    virtual ~CAIF_DeviceLayerTyped()=default;

    // Move-only — base class is move-only. Defaulted here so derived
    // classes can construct/assign through us.
    CAIF_DeviceLayerTyped(const CAIF_DeviceLayerTyped &)=delete;
    CAIF_DeviceLayerTyped &operator=(const CAIF_DeviceLayerTyped &)=delete;
    CAIF_DeviceLayerTyped(CAIF_DeviceLayerTyped &&other):
                          CAIF_DeviceLayer(std::move(other)){}
    CAIF_DeviceLayerTyped &operator=(CAIF_DeviceLayerTyped &&other)
    {
      if(this!=&other)
      {
        CAIF_DeviceLayer::operator=(std::move(other));
      }
      return *this;
    }

    // Inherited dtype introspection — every templated layer gets these
    // for free. Replaces the per-layer `_config.compute_dtype` /
    // `_config.storage_dtype` reads.
    static constexpr CAIF_DataType::CAIF_DataType_e ComputeDtype()
    {
      return CAIF_StorageDtype_t<ComputeT>::Value;
    }
    static constexpr CAIF_DataType::CAIF_DataType_e StorageDtype()
    {
      return CAIF_StorageDtype_t<StorageT>::Value;
    }

  protected:
    // Shared boundary check — every templated layer's ForwardImpl
    // / BackwardImpl asserts input dtype matches StorageT once per
    // call. Today every layer opens its body with the same boilerplate;
    // moving it here makes the per-layer code drop the assert.
    void AssertInputDtype(const CAIF_DeviceTensor &input)const
    {
      if(input.Dtype()!=StorageDtype())
      {
        THROW_CAIFE("CAIF_DeviceLayerTyped::AssertInputDtype: input dtype mismatch");
      }
    }

    // Shared output allocation — every layer today writes
    // `CAIF_DeviceTensor::Uninitialized(shape, ctx.Stream(),
    // _config.storage_dtype)`. This shorthand removes the trailing
    // dtype argument and the `_config.` read.
    CAIF_DeviceTensor AllocateOutput(const std::vector<uint32_t> &shape,
                                     CAIF_RunContext &ctx)const
    {
      return CAIF_DeviceTensor::Uninitialized(shape,ctx.Stream(),StorageDtype());
    }

    // Shared compute-type resolution — wraps `ctx.ComputeTypeFor(...)`
    // and forwards `ComputeDtype()`, so layer code reads
    // `CublasComputeType(ctx)` instead of repeating the
    // `ctx.ComputeTypeFor(_config.compute_dtype)` chain at every
    // MatMul site.
    int32_t CublasComputeType(CAIF_RunContext &ctx)const
    {
      return ctx.ComputeTypeFor(ComputeDtype());
    }

    // Shared typed-pointer accessors — wrap the `.template DevicePtr<T>()`
    // syntax so layer code can write `StoragePtr(input)` instead of the
    // awkward `input.template DevicePtr<StorageT>()`.
    static StorageT *StoragePtr(CAIF_DeviceTensor &t)
    {
      return t.template DevicePtr<StorageT>();
    }
    static const StorageT *StoragePtr(const CAIF_DeviceTensor &t)
    {
      return t.template DevicePtr<StorageT>();
    }

  private:
};

}//end instance namespace
