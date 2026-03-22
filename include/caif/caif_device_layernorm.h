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
// Device-resident LayerNorm layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_LAYERNORM_H
#define CAIF_DEVICE_LAYERNORM_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Standard Layer Normalization (Ba, Kiros, Hinton 2016)
 *
 *   y = (x - mean(x)) / sqrt(var(x) + epsilon) * gamma + beta
 *
 * Normalizes along the last dimension. For 3D inputs [batch, seq_len, dim],
 * leading dimensions are flattened to rows (batch * seq_len), normalizing
 * each row of length dim.
 *
 * Parameters: gamma [dim] (initialized to 1.0), beta [dim] (initialized to 0.0)
 *
 * Provided for compatibility with BERT-style and ViT-style models.
 */
class CAIF_DeviceLayerNorm:public CAIF_DeviceLayer
{
  public:
    CAIF_DeviceLayerNorm(uint32_t dim,
                        CAIF_CudaStream &stream,
                        float epsilon=g_caif_epsilon);
    ~CAIF_DeviceLayerNorm()override=default;

    // Move
    CAIF_DeviceLayerNorm(CAIF_DeviceLayerNorm &&other);
    CAIF_DeviceLayerNorm &operator=(CAIF_DeviceLayerNorm &&other);

    // CAIF_DeviceLayer interface
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

    // Accessors
    uint32_t Dim()const{return _dim;}
    float Epsilon()const{return _epsilon;}

  protected:

  private:
    uint32_t _dim;
    float _epsilon;
    CAIF_DeviceTensor _gamma;       // [dim], initialized to 1.0
    CAIF_DeviceTensor _beta;        // [dim], initialized to 0.0
    CAIF_DeviceTensor _gamma_grad;  // [dim]
    CAIF_DeviceTensor _beta_grad;   // [dim]

    // Pre-allocated buffers, reused when row count matches
    int _cached_rows;              // Row count for current buffer sizes (0=not allocated)

    // Cached for backward
    CAIF_DeviceTensor _last_input;  // [rows, dim]
    CAIF_DeviceTensor _mean_cache;  // [rows]
    CAIF_DeviceTensor _rstd_cache;  // [rows]
};

}//end instance namespace

#endif  // CAIF_DEVICE_LAYERNORM_H
