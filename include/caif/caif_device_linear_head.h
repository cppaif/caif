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
// Device-resident Linear Head (output projection layer)
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_LINEAR_HEAD_H
#define CAIF_DEVICE_LINEAR_HEAD_H

#include "caif_device_layer.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Linear projection layer for transformer outputs
 *
 * Projects from input_dim to output_dim via learned weight matrix.
 * Supports weight tying with embedding tables for language model heads.
 *
 * Input:  [batch, seq_len, input_dim] or [batch, input_dim]
 * Output: [batch, seq_len, output_dim] or [batch, output_dim]
 *
 * Parameters (untied):
 *   - W [input_dim, output_dim] Xavier uniform init
 *   - b [output_dim] zeros (only when use_bias==true)
 *
 * Parameters (tied):
 *   - Uses external weight tensor transposed
 *   - b [output_dim] zeros (only when use_bias==true)
 */
class CAIF_DeviceLinearHead:public CAIF_DeviceLayer
{
  public:
    /**
     * @brief Configuration for LinearHead
     */
    struct Config_t
    {
      uint32_t input_dim;   // Input dimension
      uint32_t output_dim;  // Output dimension
      bool use_bias;        // Whether to add bias after projection
    };

    // Standard constructor (owns weight)
    CAIF_DeviceLinearHead(const Config_t &config,CAIF_CudaStream &stream);

    // Weight-tied constructor (uses external weight transposed)
    CAIF_DeviceLinearHead(const Config_t &config,
                         CAIF_DeviceTensor &tied_weight,
                         CAIF_DeviceTensor &tied_weight_grad,
                         CAIF_CudaStream &stream);

    ~CAIF_DeviceLinearHead()override=default;

    // Move
    CAIF_DeviceLinearHead(CAIF_DeviceLinearHead &&other)noexcept;
    CAIF_DeviceLinearHead &operator=(CAIF_DeviceLinearHead &&other)noexcept;

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
    uint32_t InputDim()const{return _config.input_dim;}
    uint32_t OutputDim()const{return _config.output_dim;}
    bool UseBias()const{return _config.use_bias;}
    bool IsWeightTied()const{return _weight_tied;}
    bool Frozen()const{return _frozen;}
    void SetFrozen(bool frozen){_frozen=frozen;}

  protected:

  private:
    Config_t _config;
    bool _weight_tied;
    bool _frozen;

    // Owned weight (only when not tied)
    CAIF_DeviceTensor _weight;       // [input_dim, output_dim]
    CAIF_DeviceTensor _weight_grad;  // [input_dim, output_dim]

    // External weight pointers (only when tied)
    CAIF_DeviceTensor *_tied_weight;       // [output_dim, input_dim] (embedding table)
    CAIF_DeviceTensor *_tied_weight_grad;

    // Bias (always owned, if enabled)
    CAIF_DeviceTensor _bias;       // [output_dim]
    CAIF_DeviceTensor _bias_grad;  // [output_dim]

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    std::vector<uint32_t> _cached_shape;
};

}//end instance namespace

#endif  // CAIF_DEVICE_LINEAR_HEAD_H
