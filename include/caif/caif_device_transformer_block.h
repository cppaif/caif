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
// CAIF_DeviceTransformerBlock<ComputeT, StorageT> — pre-norm transformer
// block. Composes RMSNorm, Multi-Head Attention, and FFN sub-layers with
// residual connections:
//
//   h   = x + attention(norm1(x))    // residual 1
//   out = h + ffn(norm2(h))          // residual 2
//
// All four sub-layers share the block's <ComputeT, StorageT> cell. The
// residual buffers also allocate at the block's StorageT.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_container.h"
#include "caif_device_layer_typed.h"
#include "caif_run_context.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_multi_head_attention.h"
#include "caif_device_ffn.h"
#include "caif_device_activation.h"
#include "caif_constants.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceTransformerBlock:public CAIF_DeviceContainer
{
  public:
    typedef CAIF_DeviceLayerTyped<ComputeT,StorageT> Typed_t;

    struct TransformerBlockConfig_t
    {
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      uint32_t ffn_dim;
      float dropout_rate;
      bool causal;
      bool use_rope;
      float rope_base;
      int rope_style=0;
    };

    CAIF_DeviceTransformerBlock(const TransformerBlockConfig_t &config,
                                std::unique_ptr<CAIF_DeviceActivation> activation,
                                CAIF_CudaStream &stream);

    // Convenience constructor: defaults to SwiGLU activation at the same cell.
    CAIF_DeviceTransformerBlock(const TransformerBlockConfig_t &config,
                                CAIF_CudaStream &stream);

    ~CAIF_DeviceTransformerBlock()override=default;

    CAIF_DeviceTransformerBlock(CAIF_DeviceTransformerBlock &&other);
    CAIF_DeviceTransformerBlock &operator=(CAIF_DeviceTransformerBlock &&other);

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::TransformerBlock_e;
    }
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;
    size_t FrozenTensorCount()const override;
    CAIF_DeviceTensor FrozenTensorFP32(size_t index)const override;
    std::vector<std::string> FrozenTensorNames(const std::string &prefix="")const override;

    const TransformerBlockConfig_t &Config()const{return _config;}
    uint32_t EffectiveFFNDim()const{return _effective_ffn_dim;}

    static constexpr CAIF_DataType::CAIF_DataType_e ComputeDtype()
    {
      return CAIF_StorageDtype_t<ComputeT>::Value;
    }
    static constexpr CAIF_DataType::CAIF_DataType_e StorageDtype()
    {
      return CAIF_StorageDtype_t<StorageT>::Value;
    }

    CAIF_DeviceRMSNorm<ComputeT,StorageT> &Norm1()
    {
      if(_norm1==nullptr)
      {
        THROW_CAIFE("TransformerBlock: norm1 is null");
      }
      return *_norm1;
    }
    const CAIF_DeviceRMSNorm<ComputeT,StorageT> &Norm1()const
    {
      if(_norm1==nullptr)
      {
        THROW_CAIFE("TransformerBlock: norm1 is null");
      }
      return *_norm1;
    }
    CAIF_DeviceMultiHeadAttention<ComputeT,StorageT> &Attention()
    {
      if(_attention==nullptr)
      {
        THROW_CAIFE("TransformerBlock: attention is null");
      }
      return *_attention;
    }
    const CAIF_DeviceMultiHeadAttention<ComputeT,StorageT> &Attention()const
    {
      if(_attention==nullptr)
      {
        THROW_CAIFE("TransformerBlock: attention is null");
      }
      return *_attention;
    }
    CAIF_DeviceRMSNorm<ComputeT,StorageT> &Norm2()
    {
      if(_norm2==nullptr)
      {
        THROW_CAIFE("TransformerBlock: norm2 is null");
      }
      return *_norm2;
    }
    const CAIF_DeviceRMSNorm<ComputeT,StorageT> &Norm2()const
    {
      if(_norm2==nullptr)
      {
        THROW_CAIFE("TransformerBlock: norm2 is null");
      }
      return *_norm2;
    }
    CAIF_DeviceFFN<ComputeT,StorageT> &FFN()
    {
      if(_ffn==nullptr)
      {
        THROW_CAIFE("TransformerBlock: ffn is null");
      }
      return *_ffn;
    }
    const CAIF_DeviceFFN<ComputeT,StorageT> &FFN()const
    {
      if(_ffn==nullptr)
      {
        THROW_CAIFE("TransformerBlock: ffn is null");
      }
      return *_ffn;
    }

  protected:

  private:
    static uint32_t ComputeDefaultFFNDim(uint32_t dim);

    TransformerBlockConfig_t _config;
    uint32_t _effective_ffn_dim;

    // Non-owning typed pointers into _sublayers (owned by container base).
    CAIF_DeviceRMSNorm<ComputeT,StorageT> *_norm1;
    CAIF_DeviceMultiHeadAttention<ComputeT,StorageT> *_attention;
    CAIF_DeviceRMSNorm<ComputeT,StorageT> *_norm2;
    CAIF_DeviceFFN<ComputeT,StorageT> *_ffn;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceTransformerBlock<float,float>;
extern template class CAIF_DeviceTransformerBlock<float,__half>;
extern template class CAIF_DeviceTransformerBlock<float,__nv_bfloat16>;
extern template class CAIF_DeviceTransformerBlock<__half,float>;
extern template class CAIF_DeviceTransformerBlock<__half,__half>;
extern template class CAIF_DeviceTransformerBlock<__half,__nv_bfloat16>;
extern template class CAIF_DeviceTransformerBlock<__nv_bfloat16,float>;
extern template class CAIF_DeviceTransformerBlock<__nv_bfloat16,__half>;
extern template class CAIF_DeviceTransformerBlock<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceTransformerBlock<float,float>;
#endif

}//end instance namespace
