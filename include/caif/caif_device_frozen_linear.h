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
// AIF - AI Framework
// Device-resident frozen (non-trainable) linear layer with dtype-agnostic
// weight storage
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_FROZEN_LINEAR_H
#define CAIF_DEVICE_FROZEN_LINEAR_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <vector>
#include <cstdint>
#include <string>

namespace instance
{

/**
 * @brief Frozen linear layer that stores weights in any dtype
 *
 * The caller chooses the storage dtype (FP32, FP16, BF16, INT8, INT4)
 * based on VRAM budget. Weights are converted to FP32 on-the-fly for
 * Forward/Backward. The layer has zero trainable parameters.
 *
 * For INT4, per-group FP16 scales must be provided via LoadScalesFromHost().
 * For all other dtypes, To() is used for conversion.
 */
class CAIF_DeviceFrozenLinear:public CAIF_DeviceLayer
{
  public:
    CAIF_DeviceFrozenLinear(uint32_t input_dim,
                           uint32_t output_dim,
                           CAIF_DataType::CAIF_DataType_e storage_dtype,
                           CAIF_CudaStream &stream,
                           uint32_t group_size=g_caif_quant_default_group_size,
                           bool cache_fp32=true);

    ~CAIF_DeviceFrozenLinear()override=default;

    // Movable
    CAIF_DeviceFrozenLinear(CAIF_DeviceFrozenLinear &&other);
    CAIF_DeviceFrozenLinear &operator=(CAIF_DeviceFrozenLinear &&other);

    // --- CAIF_DeviceLayer interface ---
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training)override;
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)override;
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // --- FrozenLinear-specific ---

    /**
     * @brief Move a weight tensor into this layer
     * The tensor must have shape [input_dim, output_dim] and matching dtype.
     */
    void LoadFromTensor(CAIF_DeviceTensor &&weight);

    /**
     * @brief Upload INT4 per-group scales from host
     * @param data Pointer to FP16 scale data on host
     * @param num_bytes Size in bytes (num_groups * 2)
     */
    void LoadScalesFromHost(const void *data,size_t num_bytes);

    /**
     * @brief True only for INT4 (requires per-group scales)
     */
    bool NeedsScales()const;

    /**
     * @brief Clear the cached FP32 weight to free VRAM
     */
    void ClearFP32Cache();

    bool CacheFP32()const{return _cache_fp32;}

    uint32_t InputDim()const{return _input_dim;}
    uint32_t OutputDim()const{return _output_dim;}
    CAIF_DataType::CAIF_DataType_e StorageDtype()const{return _storage_dtype;}

  protected:

  private:
    CAIF_DeviceTensor ConvertToFP32()const;

    uint32_t _input_dim;
    uint32_t _output_dim;
    CAIF_DataType::CAIF_DataType_e _storage_dtype;
    uint32_t _group_size;

    CAIF_DeviceTensor _weight;
    CAIF_DeviceTensor _scales;
    bool _cache_fp32;
    CAIF_DeviceTensor _cached_fp32_weight;
    CAIF_DeviceTensor _cached_input;
};

}//end instance namespace

#endif  // CAIF_DEVICE_FROZEN_LINEAR_H
