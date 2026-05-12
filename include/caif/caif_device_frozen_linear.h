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
// Device-resident frozen (non-trainable) linear layer, templated on
// <ComputeT, StorageT>.
//
// StorageT ∈ {float, __half, __nv_bfloat16, int8_t, caif_int4_packed_t}
// ComputeT ∈ {float, __half, __nv_bfloat16}
// Total grid: 5 × 3 = 15 cells.
//
// Storage dtype controls VRAM footprint; compute dtype controls the
// MatMul precision. INT4 / INT8 cells dequantize via launch_dequantize_*
// kernels with per-group / per-channel / per-tensor scales. Float storage
// cells cast through CAIF_DeviceTensor::To().
//
// Zero trainable parameters — TotalParameterCount() always returns 0.
// FrozenLinear is the storage-quantized counterpart of CAIF_DeviceDenseLayer.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer.h"
#include "caif_device_layer_typed.h"
#include "caif_device_frozen_linear_base.h"
#include "caif_host_pinned_tensor.h"
#include "caif_offload_policy.h"
#include "caif_exception.h"
#include "caif_run_context.h"
#include "caif_constants.h"
#include "caif_ops.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#include "caif_storage_dtype_int8.h"
#include "caif_storage_dtype_int4.h"
#include "caif_int4_packed_t.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#include <memory>
#include <vector>
#include <cstdint>
#include <string>

namespace instance
{

/**
 * @brief Frozen linear layer that stores weights in any dtype.
 *
 * The caller chooses the storage dtype (FP32, FP16, BF16, INT8, INT4)
 * based on VRAM budget. Weights are converted to ComputeT on-the-fly for
 * Forward/Backward. The layer has zero trainable parameters.
 *
 * For INT4, per-group FP16 scales must be provided via LoadScalesFromHost().
 * For all other dtypes, To() is used for conversion.
 */
template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceFrozenLinear:public CAIF_DeviceLayerTyped<ComputeT,StorageT>,
                              public CAIF_DeviceFrozenLinearBase
{
  public:
    typedef CAIF_DeviceLayerTyped<ComputeT,StorageT> Base_t;

    // Inject inherited helpers from CAIF_DeviceLayerTyped + CAIF_DeviceLayer
    // so derived-class method bodies can call them unqualified (no `this->`).
    using Base_t::Stream;
    using Base_t::HasStream;

    CAIF_DeviceFrozenLinear(uint32_t input_dim,
                            uint32_t output_dim,
                            CAIF_CudaStream &stream,
                            uint32_t group_size=g_caif_quant_default_group_size,
                            bool cache_fp32=true,
                            CAIF_Ops::QuantScheme_e int8_scheme=
                              CAIF_Ops::QuantScheme_e::PerTensor_e);

    ~CAIF_DeviceFrozenLinear()override=default;

    CAIF_DeviceFrozenLinear(CAIF_DeviceFrozenLinear &&other);
    CAIF_DeviceFrozenLinear &operator=(CAIF_DeviceFrozenLinear &&other);

    // --- CAIF_DeviceLayer interface ---
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::FrozenLinear_e;
    }
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;
    size_t FrozenTensorCount()const override;
    CAIF_DeviceTensor FrozenTensorFP32(size_t index)const override;
    std::vector<std::string> FrozenTensorNames(const std::string &prefix="")const override;

    // --- CAIF_DeviceFrozenLinearBase interface ---

    /**
     * @brief Move a weight tensor into this layer.
     * The tensor must have shape [input_dim, output_dim] and matching dtype.
     */
    void LoadFromTensor(CAIF_DeviceTensor &&weight)override;

    /**
     * @brief Upload quantization scales from host.
     *
     * Scale layout depends on storage dtype and INT8 scheme:
     *   - INT4 per-group: FP16, num_groups = ceil(N / group_size) elements
     *   - INT8 per-tensor: FP32, 1 element
     *   - INT8 per-channel: FP32, output_dim elements
     *
     * @param data Pointer to scale data on host (dtype per above)
     * @param num_bytes Size in bytes of the data region
     */
    void LoadScalesFromHost(const void *data,size_t num_bytes)override;

    /**
     * @brief True for INT4 and INT8 (when explicit scales drive dequant).
     *
     * If scales are not loaded, INT8 storage falls back to a direct
     * element-wise cast via CAIF_DeviceTensor::To().
     */
    bool NeedsScales()const override;

    CAIF_Ops::QuantScheme_e Int8Scheme()const override{return _int8_scheme;}

    /**
     * @brief Clear the cached FP32 weight to free VRAM.
     */
    void ClearFP32Cache()override;

    bool CacheFP32()const override{return _cache_fp32;}
    uint32_t InputDim()const override{return _input_dim;}
    uint32_t OutputDim()const override{return _output_dim;}

    // --- CPU offload surface (O2 of CPU_OFFLOAD_DESIGN.md) ---
    //
    // Default policy is GpuResident_e — every existing code path is byte-
    // identical, no host pinned memory is allocated, no DMA happens. Opt-in
    // to HostPinned_e migrates the canonical weight to a pinned host buffer:
    // the GPU `_weight` becomes a transient scratch populated by Prefetch
    // and freed by Evict. Forward / Backward require the layer to be
    // prefetched; calling them between Evict and the next Prefetch throws.
    //
    // The block-level scheduler (CAIF_BlockOffloadScheduler) is responsible
    // for issuing Prefetch / Evict at the right points so layer-level callers
    // never see the host-pinned state directly.
    void SetOffloadPolicy(const CAIF_OffloadPolicy::CAIF_OffloadPolicy_e p)override;
    CAIF_OffloadPolicy::CAIF_OffloadPolicy_e OffloadPolicy()const override
    {
      return _offload_policy;
    }

    void Prefetch(CAIF_CudaStream &stream)override;
    void Evict()override;
    bool IsPrefetched()const override{return _is_prefetched;}

    // Inherited from CAIF_DeviceLayerTyped:
    //   static constexpr CAIF_DataType_e StorageDtype()
    //   static constexpr CAIF_DataType_e ComputeDtype()

  protected:

  private:
    CAIF_DeviceTensor ConvertToComputeDtype()const;

    // Migrate the GPU `_weight` to a fresh pinned host buffer and free the
    // GPU side. Called by SetOffloadPolicy(HostPinned_e) when a weight is
    // already loaded.
    void MigrateWeightToHost();

    // Private accessors for the offload-state members. Per the
    // accessor-only-discipline; all method bodies in this file (offload
    // path) read/write through these instead of direct `_member` access.
    bool HasWeight()const{return _weight.IsAllocated();}
    const CAIF_DeviceTensor &Weight()const{return _weight;}
    bool HasHostWeight()const{return _host_weight!=nullptr;}
    CAIF_HostPinnedTensor &HostWeight()
    {
      if(_host_weight==nullptr)
      {
        THROW_CAIFE("HostWeight: not allocated");
      }
      return *_host_weight;
    }
    const CAIF_HostPinnedTensor &HostWeight()const
    {
      if(_host_weight==nullptr)
      {
        THROW_CAIFE("HostWeight: not allocated");
      }
      return *_host_weight;
    }
    void SetWeight(CAIF_DeviceTensor t){_weight=std::move(t);}
    void ClearWeight(){_weight=CAIF_DeviceTensor();}
    void SetPrefetched(const bool b){_is_prefetched=b;}

    bool HasCachedComputeWeight()const{return _cached_compute_weight.IsAllocated();}
    const CAIF_DeviceTensor &CachedComputeWeight()const{return _cached_compute_weight;}
    void SetCachedComputeWeight(CAIF_DeviceTensor t){_cached_compute_weight=std::move(t);}
    void ClearCachedComputeWeight(){_cached_compute_weight=CAIF_DeviceTensor();}

    uint32_t _input_dim;
    uint32_t _output_dim;
    CAIF_Ops::QuantScheme_e _int8_scheme;
    uint32_t _group_size;

    CAIF_DeviceTensor _weight;
    CAIF_DeviceTensor _scales;
    bool _cache_fp32;
    CAIF_DeviceTensor _cached_compute_weight;
    CAIF_DeviceTensor _cached_input;

    CAIF_OffloadPolicy::CAIF_OffloadPolicy_e _offload_policy;
    std::unique_ptr<CAIF_HostPinnedTensor> _host_weight;
    bool _is_prefetched;
};

}//end instance namespace
