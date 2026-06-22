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
// Cross-entropy loss from logits implementation (templated).
//
// Per-site dispositions:
//   - `targets` are float-encoded target indices (CE kernel signature
//     contract; kernel casts them to int internally) —
//     `DevicePtr<float>()`.
//   - `losses` / `per_losses` / `reduction` / `result` are fp32
//     accumulators (per-token loss, mean, grad scalar) —
//     `DevicePtr<float>()`.
//   - All logits-tensor reads use `tensor.template DevicePtr<StorageT>()`.
//------------------------------------------------------------------------------
#include "caif_device_cross_entropy_loss.h"
#include "caif_host_tensor.h"
#include "caif_cuda_kernels_loss.cuh"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#endif

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceCrossEntropyLoss<ComputeT,StorageT>::ComputePerPositionLoss(
    const CAIF_DeviceTensor &logits,
    const CAIF_DeviceTensor &targets,
    CAIF_CudaStream &stream,
    int ignore_index)
{
  try
  {
    if(logits.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceCrossEntropyLoss: logits dtype != StorageT");
    }

    const std::vector<uint32_t> &logits_shape=logits.Shape();
    const std::vector<uint32_t> &targets_shape=targets.Shape();

    if(logits_shape.size()<2)
    {
      THROW_CAIFE("CAIF_DeviceCrossEntropyLoss: logits must be at least 2D");
    }

    uint32_t n=1;
    for(size_t i=0;i<logits_shape.size()-1;++i)
    {
      n*=logits_shape[i];
    }
    const uint32_t vocab_size=logits_shape.back();

    uint32_t target_n=1;
    for(size_t i=0;i<targets_shape.size();++i)
    {
      target_n*=targets_shape[i];
    }
    if(target_n!=n)
    {
      THROW_CAIFE("CAIF_DeviceCrossEntropyLoss: targets size mismatch (expected "+
                  std::to_string(n)+", got "+std::to_string(target_n)+")");
    }

    CAIF_DeviceTensor losses=CAIF_DeviceTensor::Uninitialized({n},stream);

#ifdef USE_CAIF_CUDA
    launch_cross_entropy_logits_forward<StorageT>(logits.template DevicePtr<StorageT>(),
                                                   targets.DevicePtr<float>(),
                                                   losses.DevicePtr<float>(),
                                                   static_cast<int>(n),
                                                   static_cast<int>(vocab_size),
                                                   ignore_index,
                                                   stream.Handle());
#endif

    return losses;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
float CAIF_DeviceCrossEntropyLoss<ComputeT,StorageT>::ComputeLoss(
    const CAIF_DeviceTensor &logits,
    const CAIF_DeviceTensor &targets,
    CAIF_CudaStream &stream,
    int ignore_index)
{
  try
  {
    CAIF_DeviceTensor losses=ComputePerPositionLoss(logits,targets,stream,ignore_index);

    const std::vector<uint32_t> &logits_shape=logits.Shape();
    uint32_t n=1;
    for(size_t i=0;i<logits_shape.size()-1;++i)
    {
      n*=logits_shape[i];
    }

#ifdef USE_CAIF_CUDA
    CAIF_DeviceTensor reduction=CAIF_DeviceTensor::Zeros({2},stream);

    launch_cross_entropy_reduce_mean(losses.DevicePtr<float>(),
                                     targets.DevicePtr<float>(),
                                     reduction.DevicePtr<float>(),
                                     static_cast<int>(n),
                                     ignore_index,
                                     stream.Handle());

    CAIF_HostTensor host_result=reduction.ToHost();
    const float total_loss=host_result.At({0});
    const float count=host_result.At({1});

    if(count<1.0f)
    {
      return 0.0f;
    }
    return total_loss/count;
#else
    (void)ignore_index;
    return 0.0f;
#endif
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceCrossEntropyLoss<ComputeT,StorageT>::ComputeGradient(
    const CAIF_DeviceTensor &logits,
    const CAIF_DeviceTensor &targets,
    CAIF_CudaStream &stream,
    int ignore_index)
{
  try
  {
    if(logits.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceCrossEntropyLoss: logits dtype != StorageT");
    }

    const std::vector<uint32_t> &logits_shape=logits.Shape();
    uint32_t n=1;
    for(size_t i=0;i<logits_shape.size()-1;++i)
    {
      n*=logits_shape[i];
    }
    const uint32_t vocab_size=logits_shape.back();

    CAIF_DeviceTensor grad=CAIF_DeviceTensor::Uninitialized(logits_shape,stream,logits.Dtype());

#ifdef USE_CAIF_CUDA
    CAIF_DeviceTensor per_losses=ComputePerPositionLoss(logits,targets,stream,ignore_index);
    CAIF_DeviceTensor reduction=CAIF_DeviceTensor::Zeros({2},stream);
    launch_cross_entropy_reduce_mean(per_losses.DevicePtr<float>(),
                                     targets.DevicePtr<float>(),
                                     reduction.DevicePtr<float>(),
                                     static_cast<int>(n),
                                     ignore_index,
                                     stream.Handle());
    CAIF_HostTensor host_result=reduction.ToHost();
    const float valid_count=host_result.At({1});

    float scale=0.0f;
    if(valid_count>=1.0f)
    {
      scale=1.0f/valid_count;
    }

    launch_cross_entropy_logits_backward<StorageT>(logits.template DevicePtr<StorageT>(),
                                                    targets.DevicePtr<float>(),
                                                    grad.template DevicePtr<StorageT>(),
                                                    static_cast<int>(n),
                                                    static_cast<int>(vocab_size),
                                                    ignore_index,
                                                    scale,
                                                    stream.Handle());
#endif

    return grad;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
float CAIF_DeviceCrossEntropyLoss<ComputeT,StorageT>::ComputeLossAndGradient(
    const CAIF_DeviceTensor &logits,
    const CAIF_DeviceTensor &targets,
    CAIF_DeviceTensor &grad_logits,
    CAIF_CudaStream &stream,
    int ignore_index)
{
  try
  {
    if(logits.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceCrossEntropyLoss: logits dtype != StorageT");
    }

    const std::vector<uint32_t> &logits_shape=logits.Shape();
    uint32_t n=1;
    for(size_t i=0;i<logits_shape.size()-1;++i)
    {
      n*=logits_shape[i];
    }
    const uint32_t vocab_size=logits_shape.back();

#ifdef USE_CAIF_CUDA
    CAIF_DeviceTensor per_losses=CAIF_DeviceTensor::Uninitialized({n},stream);
    grad_logits=CAIF_DeviceTensor::Uninitialized(logits_shape,stream,logits.Dtype());
    CAIF_DeviceTensor result=CAIF_DeviceTensor::Zeros({2},stream);

    launch_cross_entropy_fused<StorageT>(logits.template DevicePtr<StorageT>(),
                                          targets.DevicePtr<float>(),
                                          per_losses.DevicePtr<float>(),
                                          grad_logits.template DevicePtr<StorageT>(),
                                          result.DevicePtr<float>(),
                                          static_cast<int>(n),
                                          static_cast<int>(vocab_size),
                                          ignore_index,
                                          stream.Handle());

    CAIF_HostTensor host_result=result.ToHost();
    const float valid_count=host_result.At({0});
    const float loss_sum=host_result.At({1});

    float loss=0.0f;
    if(valid_count>=1.0f)
    {
      loss=loss_sum/valid_count;
    }

    return loss;
#else
    (void)targets;
    (void)grad_logits;
    (void)ignore_index;
    (void)n;
    (void)vocab_size;
    return 0.0f;
#endif
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceCrossEntropyLoss<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceCrossEntropyLoss<float,__half>;
template class CAIF_DeviceCrossEntropyLoss<float,__nv_bfloat16>;
template class CAIF_DeviceCrossEntropyLoss<__half,float>;
template class CAIF_DeviceCrossEntropyLoss<__half,__half>;
template class CAIF_DeviceCrossEntropyLoss<__half,__nv_bfloat16>;
template class CAIF_DeviceCrossEntropyLoss<__nv_bfloat16,float>;
template class CAIF_DeviceCrossEntropyLoss<__nv_bfloat16,__half>;
template class CAIF_DeviceCrossEntropyLoss<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
