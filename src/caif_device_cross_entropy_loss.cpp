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
// Cross-entropy loss from logits implementation
//------------------------------------------------------------------------------
#include "caif_device_cross_entropy_loss.h"
#include "caif_host_tensor.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

namespace instance
{

CAIF_DeviceTensor CAIF_DeviceCrossEntropyLoss::ComputePerPositionLoss(
    const CAIF_DeviceTensor &logits,
    const CAIF_DeviceTensor &targets,
    CAIF_CudaStream &stream,
    int ignore_index)
{
  try
  {
    const auto &logits_shape=logits.Shape();
    const auto &targets_shape=targets.Shape();

    // Logits should be 2D [N, vocab_size] or 3D [batch, seq_len, vocab_size]
    if(logits_shape.size()<2)
    {
      THROW_CAIFE("CrossEntropyLoss: logits must be at least 2D");
    }

    // Flatten logits to [N, vocab_size]
    uint32_t n=1;
    for(size_t i=0;i<logits_shape.size()-1;++i)
    {
      n*=logits_shape[i];
    }
    const uint32_t vocab_size=logits_shape.back();

    // Verify targets shape
    uint32_t target_n=1;
    for(size_t i=0;i<targets_shape.size();++i)
    {
      target_n*=targets_shape[i];
    }
    if(target_n!=n)
    {
      THROW_CAIFE("CrossEntropyLoss: targets size mismatch (expected "+
                 std::to_string(n)+", got "+std::to_string(target_n)+")");
    }

    // Allocate output
    CAIF_DeviceTensor losses=CAIF_DeviceTensor::Uninitialized({n},stream);

#ifdef USE_CAIF_CUDA
    launch_cross_entropy_logits_forward(logits.DevicePtr(),
                                        targets.DevicePtr(),
                                        losses.DevicePtr(),
                                        static_cast<int>(n),
                                        static_cast<int>(vocab_size),
                                        ignore_index,
                                        stream.Handle());
#endif

    return losses;
  }
  CCAIF_CATCH_BLOCK()
}

float CAIF_DeviceCrossEntropyLoss::ComputeLoss(const CAIF_DeviceTensor &logits,
                                              const CAIF_DeviceTensor &targets,
                                              CAIF_CudaStream &stream,
                                              int ignore_index)
{
  try
  {
    // Get per-position losses
    CAIF_DeviceTensor losses=ComputePerPositionLoss(logits,targets,stream,ignore_index);

    const auto &logits_shape=logits.Shape();
    uint32_t n=1;
    for(size_t i=0;i<logits_shape.size()-1;++i)
    {
      n*=logits_shape[i];
    }

#ifdef USE_CAIF_CUDA
    // Allocate output buffer for sum and count (2 floats)
    CAIF_DeviceTensor reduction=CAIF_DeviceTensor::Zeros({2},stream);

    launch_cross_entropy_reduce_mean(losses.DevicePtr(),
                                     targets.DevicePtr(),
                                     reduction.DevicePtr(),
                                     static_cast<int>(n),
                                     ignore_index,
                                     stream.Handle());

    // Copy result to host
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
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceCrossEntropyLoss::ComputeGradient(
    const CAIF_DeviceTensor &logits,
    const CAIF_DeviceTensor &targets,
    CAIF_CudaStream &stream,
    int ignore_index)
{
  try
  {
    const auto &logits_shape=logits.Shape();

    // Flatten to [N, vocab_size]
    uint32_t n=1;
    for(size_t i=0;i<logits_shape.size()-1;++i)
    {
      n*=logits_shape[i];
    }
    const uint32_t vocab_size=logits_shape.back();

    // Allocate gradient with same shape as logits
    CAIF_DeviceTensor grad=CAIF_DeviceTensor::Uninitialized(logits_shape,stream);

#ifdef USE_CAIF_CUDA
    launch_cross_entropy_logits_backward(logits.DevicePtr(),
                                         targets.DevicePtr(),
                                         grad.DevicePtr(),
                                         static_cast<int>(n),
                                         static_cast<int>(vocab_size),
                                         ignore_index,
                                         stream.Handle());
#endif

    return grad;
  }
  CCAIF_CATCH_BLOCK()
}

float CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(
    const CAIF_DeviceTensor &logits,
    const CAIF_DeviceTensor &targets,
    CAIF_DeviceTensor &grad_logits,
    CAIF_CudaStream &stream,
    int ignore_index)
{
  try
  {
    // Compute loss first
    float loss=ComputeLoss(logits,targets,stream,ignore_index);

    // Compute gradient
    grad_logits=ComputeGradient(logits,targets,stream,ignore_index);

    return loss;
  }
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
