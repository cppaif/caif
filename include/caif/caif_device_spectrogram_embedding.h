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
// Device-resident spectrogram embedding layer for audio
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_SPECTROGRAM_EMBEDDING_H
#define CAIF_DEVICE_SPECTROGRAM_EMBEDDING_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Projects audio spectrograms to embedding dimension for transformer input.
 *
 * Input:  [batch, time_frames, freq_bins] (spectrogram)
 * Output: [batch, time_frames, dim] or [batch, time_frames+1, dim] (with CLS)
 *
 * Each time frame's frequency bins are linearly projected to the embedding
 * dimension. Optionally prepends a CLS token for classification tasks.
 *
 * Parameters:
 *   0: W_proj     [freq_bins, dim]  Xavier uniform
 *   1: b_proj     [dim]             Zeros
 *   2: cls_token  [1, dim]          Xavier uniform (only with use_cls_token)
 */
class CAIF_DeviceSpectrogramEmbedding:public CAIF_DeviceLayer
{
  public:
    struct Config_t
    {
      uint32_t freq_bins;       // Number of frequency bins (e.g., 80 for mel)
      uint32_t dim;             // Output embedding dimension
      bool use_cls_token;       // Prepend CLS token for classification
    };

    CAIF_DeviceSpectrogramEmbedding(const Config_t &config,
                                   CAIF_CudaStream &stream);
    ~CAIF_DeviceSpectrogramEmbedding()override=default;

    // Move
    CAIF_DeviceSpectrogramEmbedding(CAIF_DeviceSpectrogramEmbedding &&other);
    CAIF_DeviceSpectrogramEmbedding &operator=(CAIF_DeviceSpectrogramEmbedding &&other);

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
    uint32_t FreqBins()const{return _config.freq_bins;}
    uint32_t Dim()const{return _config.dim;}
    bool UseCLSToken()const{return _config.use_cls_token;}

    // Weight initialization
    void InitializeWeights(uint32_t seed=0);

  protected:

  private:
    Config_t _config;

    CAIF_DeviceTensor _w_proj;       // [freq_bins, dim]
    CAIF_DeviceTensor _b_proj;       // [dim]
    CAIF_DeviceTensor _cls_token;    // [1, dim] (only when use_cls_token)

    CAIF_DeviceTensor _grad_w_proj;  // [freq_bins, dim]
    CAIF_DeviceTensor _grad_b_proj;  // [dim]
    CAIF_DeviceTensor _grad_cls;     // [1, dim] (only when use_cls_token)

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    uint32_t _cached_batch;
    uint32_t _cached_time_frames;
};

}//end instance namespace

#endif  // CAIF_DEVICE_SPECTROGRAM_EMBEDDING_H
