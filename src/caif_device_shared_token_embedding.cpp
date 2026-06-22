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

#include "caif_device_shared_token_embedding.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::CAIF_DeviceSharedTokenEmbedding(
                                          const CAIF_DeviceTokenEmbeddingConfig &config,
                                          CAIF_DeviceTensor &donor_table,
                                          CAIF_DeviceTensor &donor_grad,
                                          CAIF_CudaStream &stream):
                                          CAIF_DeviceTokenEmbedding<ComputeT,StorageT>(config,
                                                                                       donor_table,
                                                                                       donor_grad,
                                                                                       stream)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::~CAIF_DeviceSharedTokenEmbedding()
{
  // Null the inherited pointers BEFORE the base destructor runs so its
  // `delete EmbeddingTablePtr()` is a `delete nullptr` no-op rather than
  // freeing the donor's storage.
  SetEmbeddingTablePtr(nullptr);
  SetEmbeddingTableGradPtr(nullptr);
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::CAIF_DeviceSharedTokenEmbedding(
                                          CAIF_DeviceSharedTokenEmbedding &&other):
                                          CAIF_DeviceTokenEmbedding<ComputeT,StorageT>(
                                                                          std::move(other))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT> &
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::operator=(CAIF_DeviceSharedTokenEmbedding &&other)
{
  if(this!=&other)
  {
    // Null our donor pointers BEFORE the base op= so its `delete` is a
    // no-op rather than freeing the donor's storage; then let the base
    // op= take over the rest of the move.
    SetEmbeddingTablePtr(nullptr);
    SetEmbeddingTableGradPtr(nullptr);
    CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::operator=(std::move(other));
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::ZeroGradients()
{
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::ParameterTensorCount()const
{
  return 0;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::ParameterTensor(size_t)
{
  try
  {
    THROW_CAIFE("CAIF_DeviceSharedTokenEmbedding::ParameterTensor: borrower"
                " exposes no parameters of its own; reach the shared table"
                " through the donor's ParameterTensor(0)");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::ParameterTensor(size_t)const
{
  try
  {
    THROW_CAIFE("CAIF_DeviceSharedTokenEmbedding::ParameterTensor: borrower"
                " exposes no parameters of its own; reach the shared table"
                " through the donor's ParameterTensor(0)");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::GradientTensor(size_t)
{
  try
  {
    THROW_CAIFE("CAIF_DeviceSharedTokenEmbedding::GradientTensor: borrower"
                " exposes no gradient of its own; the donor owns the shared"
                " gradient tensor");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::GradientTensor(size_t)const
{
  try
  {
    THROW_CAIFE("CAIF_DeviceSharedTokenEmbedding::GradientTensor: borrower"
                " exposes no gradient of its own; the donor owns the shared"
                " gradient tensor");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::TotalParameterCount()const
{
  return 0;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::ParameterNames(const std::string &)const
{
  return std::vector<std::string>();
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceSharedTokenEmbedding<ComputeT,StorageT>::LoadEmbeddingTable(CAIF_DeviceTensor &&)
{
  try
  {
    THROW_CAIFE("CAIF_DeviceSharedTokenEmbedding::LoadEmbeddingTable: borrower"
                " cannot replace the donor's storage; invoke LoadEmbeddingTable"
                " on the donor instance instead");
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid, matching
// the parent CAIF_DeviceTokenEmbedding's instantiation list.
template class CAIF_DeviceSharedTokenEmbedding<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceSharedTokenEmbedding<float,__half>;
template class CAIF_DeviceSharedTokenEmbedding<float,__nv_bfloat16>;
template class CAIF_DeviceSharedTokenEmbedding<__half,float>;
template class CAIF_DeviceSharedTokenEmbedding<__half,__half>;
template class CAIF_DeviceSharedTokenEmbedding<__half,__nv_bfloat16>;
template class CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,float>;
template class CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,__half>;
template class CAIF_DeviceSharedTokenEmbedding<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
