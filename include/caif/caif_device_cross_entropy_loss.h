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
// Cross-entropy loss from logits (templated on <ComputeT, StorageT>).
//
// Utility class with static methods. Uniform two-parameter signature with
// both defaulting to `float`. Logits are stored at StorageT; targets are
// always fp32-encoded token IDs. Every legal (ComputeT, StorageT) cell from
// the cuBLAS-Lt grid is instantiated.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"

#ifdef USE_CAIF_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceCrossEntropyLoss:public CAIF_Base
{
  public:
    CAIF_DeviceCrossEntropyLoss()=delete;

    static constexpr int g_default_ignore_index=-100;

    static CAIF_DeviceTensor ComputePerPositionLoss(const CAIF_DeviceTensor &logits,
                                                    const CAIF_DeviceTensor &targets,
                                                    CAIF_CudaStream &stream,
                                                    int ignore_index=g_default_ignore_index);

    static float ComputeLoss(const CAIF_DeviceTensor &logits,
                             const CAIF_DeviceTensor &targets,
                             CAIF_CudaStream &stream,
                             int ignore_index=g_default_ignore_index);

    static CAIF_DeviceTensor ComputeGradient(const CAIF_DeviceTensor &logits,
                                             const CAIF_DeviceTensor &targets,
                                             CAIF_CudaStream &stream,
                                             int ignore_index=g_default_ignore_index);

    static float ComputeLossAndGradient(const CAIF_DeviceTensor &logits,
                                        const CAIF_DeviceTensor &targets,
                                        CAIF_DeviceTensor &grad_logits,
                                        CAIF_CudaStream &stream,
                                        int ignore_index=g_default_ignore_index);
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceCrossEntropyLoss<float,float>;
extern template class CAIF_DeviceCrossEntropyLoss<float,__half>;
extern template class CAIF_DeviceCrossEntropyLoss<float,__nv_bfloat16>;
extern template class CAIF_DeviceCrossEntropyLoss<__half,float>;
extern template class CAIF_DeviceCrossEntropyLoss<__half,__half>;
extern template class CAIF_DeviceCrossEntropyLoss<__half,__nv_bfloat16>;
extern template class CAIF_DeviceCrossEntropyLoss<__nv_bfloat16,float>;
extern template class CAIF_DeviceCrossEntropyLoss<__nv_bfloat16,__half>;
extern template class CAIF_DeviceCrossEntropyLoss<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceCrossEntropyLoss<float,float>;
#endif

}//end instance namespace
