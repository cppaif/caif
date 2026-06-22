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

#include "caif_device_token_embedding_factory.h"
#include "caif_device_token_embedding.h"
#include "caif_device_shared_token_embedding.h"
#include "caif_exception.h"

namespace instance
{

std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceTokenEmbeddingFactory::Create(uint32_t vocab_size,
                                          uint32_t dim,
                                          CAIF_CudaStream &stream,
                                          CAIF_DataType::CAIF_DataType_e compute_dtype,
                                          CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    // Token embedding is a gather, not a MatMul — ComputeT has no
    // semantic effect on the kernel — but the factory still dispatches
    // on both so the built layer's RuntimeStorageDtype() /
    // RuntimeComputeDtype() introspect the same (compute, storage)
    // pair the caller asked for. See the matching comment in
    // caif_device_rmsnorm_factory.cpp.
    if(storage_dtype==Dtype_e::Float32 && compute_dtype==Dtype_e::Float32)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<float,float>>(cfg,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16 && compute_dtype==Dtype_e::Float32)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<float,__half>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16 && compute_dtype==Dtype_e::Float32)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<float,__nv_bfloat16>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::Float32 && compute_dtype==Dtype_e::Float16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<__half,float>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::Float16 && compute_dtype==Dtype_e::Float16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<__half,__half>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16 && compute_dtype==Dtype_e::Float16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<__half,__nv_bfloat16>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::Float32 && compute_dtype==Dtype_e::BFloat16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<__nv_bfloat16,float>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::Float16 && compute_dtype==Dtype_e::BFloat16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<__nv_bfloat16,__half>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16 && compute_dtype==Dtype_e::BFloat16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceTokenEmbedding<__nv_bfloat16,__nv_bfloat16>>(cfg,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceTokenEmbeddingFactory::Create: unsupported (compute, storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceTokenEmbeddingFactory::CreateShared(uint32_t vocab_size,
                                                uint32_t dim,
                                                CAIF_DeviceTensor &donor_table,
                                                CAIF_DeviceTensor &donor_grad,
                                                CAIF_CudaStream &stream,
                                                CAIF_DataType::CAIF_DataType_e compute_dtype,
                                                CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32 && compute_dtype==Dtype_e::Float32)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<float,float>>(cfg,
                                                                            donor_table,
                                                                            donor_grad,
                                                                            stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16 && compute_dtype==Dtype_e::Float32)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<float,__half>>(cfg,
                                                                             donor_table,
                                                                             donor_grad,
                                                                             stream);
    }
    if(storage_dtype==Dtype_e::BFloat16 && compute_dtype==Dtype_e::Float32)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<float,__nv_bfloat16>>(cfg,
                                                                                    donor_table,
                                                                                    donor_grad,
                                                                                    stream);
    }
    if(storage_dtype==Dtype_e::Float32 && compute_dtype==Dtype_e::Float16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<__half,float>>(cfg,
                                                                             donor_table,
                                                                             donor_grad,
                                                                             stream);
    }
    if(storage_dtype==Dtype_e::Float16 && compute_dtype==Dtype_e::Float16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<__half,__half>>(cfg,
                                                                              donor_table,
                                                                              donor_grad,
                                                                              stream);
    }
    if(storage_dtype==Dtype_e::BFloat16 && compute_dtype==Dtype_e::Float16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<__half,__nv_bfloat16>>(cfg,
                                                                                     donor_table,
                                                                                     donor_grad,
                                                                                     stream);
    }
    if(storage_dtype==Dtype_e::Float32 && compute_dtype==Dtype_e::BFloat16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,float>>(cfg,
                                                                                    donor_table,
                                                                                    donor_grad,
                                                                                    stream);
    }
    if(storage_dtype==Dtype_e::Float16 && compute_dtype==Dtype_e::BFloat16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,__half>>(cfg,
                                                                                     donor_table,
                                                                                     donor_grad,
                                                                                     stream);
    }
    if(storage_dtype==Dtype_e::BFloat16 && compute_dtype==Dtype_e::BFloat16)
    {
      CAIF_DeviceTokenEmbeddingConfig cfg{vocab_size,dim};
      return std::make_unique<CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,
                                                              __nv_bfloat16>>(cfg,
                                                                              donor_table,
                                                                              donor_grad,
                                                                              stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceTokenEmbeddingFactory::CreateShared: unsupported"
                " (compute, storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
