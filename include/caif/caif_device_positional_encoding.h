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
// Device-resident positional encoding layer (templated on
// <ComputeT, StorageT>).
//
// Uniform two-parameter signature with both defaulting to `float`. The
// pe_table / sinusoidal_table follow StorageT; runtime dispatch over
// dtype is replaced by the static type.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include "caif_positional_encoding_mode.h"
#include "caif_device_positional_encoding_config.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace instance
{

// enum moved to include/caif/caif_positional_encoding_mode.h
// (class-scoped: CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e).
// Moved 2026-05-13.

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DevicePositionalEncoding:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    CAIF_DevicePositionalEncoding(const CAIF_DevicePositionalEncodingConfig &config,CAIF_CudaStream &stream);
    ~CAIF_DevicePositionalEncoding()override=default;

    // Move
    CAIF_DevicePositionalEncoding(CAIF_DevicePositionalEncoding &&other);
    CAIF_DevicePositionalEncoding &operator=(CAIF_DevicePositionalEncoding &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::PositionalEncoding_e;
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

    uint32_t MaxSeqLen()const{return Config().MaxSeqLen();}
    uint32_t Dim()const{return Config().Dim();}
    CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e Mode()const{return Config().Mode();}

    /**
     * @brief Replace the learned positional-embedding table with a tensor of
     * shape [max_seq_len, dim]. Only valid in Learned mode.
     */
    void LoadPETable(CAIF_DeviceTensor &&table);

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

  private:
    const CAIF_DevicePositionalEncodingConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DevicePositionalEncodingConfig &c){_config=c;}

    const CAIF_DeviceTensor &PETable()const{return _pe_table;}
    CAIF_DeviceTensor &PETableMut(){return _pe_table;}
    void SetPETable(CAIF_DeviceTensor &&t){_pe_table=std::move(t);}

    const CAIF_DeviceTensor &PETableGrad()const{return _pe_table_grad;}
    CAIF_DeviceTensor &PETableGrad(){return _pe_table_grad;}
    void SetPETableGrad(CAIF_DeviceTensor &&t){_pe_table_grad=std::move(t);}

    const CAIF_DeviceTensor &SinusoidalTable()const{return _sinusoidal_table;}
    CAIF_DeviceTensor &SinusoidalTableMut(){return _sinusoidal_table;}
    void SetSinusoidalTable(CAIF_DeviceTensor &&t){_sinusoidal_table=std::move(t);}

    uint32_t CachedBatch()const{return _cached_batch;}
    void SetCachedBatch(const uint32_t v){_cached_batch=v;}
    uint32_t CachedSeqLen()const{return _cached_seq_len;}
    void SetCachedSeqLen(const uint32_t v){_cached_seq_len=v;}

    CAIF_DevicePositionalEncodingConfig _config;

    CAIF_DeviceTensor _pe_table;          // [max_seq_len, dim] at StorageT
    CAIF_DeviceTensor _pe_table_grad;     // [max_seq_len, dim] at StorageT
    CAIF_DeviceTensor _sinusoidal_table;  // [max_seq_len, dim] at StorageT

    uint32_t _cached_batch;
    uint32_t _cached_seq_len;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DevicePositionalEncoding<float,float>;
extern template class CAIF_DevicePositionalEncoding<float,__half>;
extern template class CAIF_DevicePositionalEncoding<float,__nv_bfloat16>;
extern template class CAIF_DevicePositionalEncoding<__half,float>;
extern template class CAIF_DevicePositionalEncoding<__half,__half>;
extern template class CAIF_DevicePositionalEncoding<__half,__nv_bfloat16>;
extern template class CAIF_DevicePositionalEncoding<__nv_bfloat16,float>;
extern template class CAIF_DevicePositionalEncoding<__nv_bfloat16,__half>;
extern template class CAIF_DevicePositionalEncoding<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DevicePositionalEncoding<float,float>;
#endif

}//end instance namespace
