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
// cuDNN-side helpers used by CAIF_DeviceBatchNorm. Two responsibilities:
//   - SyncFp32* / AccumulateFp32* move fp32 BN params and gradients between
//     the host-side master copies and freshly-allocated device buffers that
//     cuDNN expects, since cuDNN's CUDNN_BATCHNORM_SPATIAL kernel requires
//     fp32 scale / bias / running-mean / running-var even when the data
//     tensor is fp16 or bf16.
//   - SetupBn{Param,Data}Descriptor configures the cuDNN tensor descriptors
//     for the [1, C, 1, 1] param layout and the [N, C, 1, 1] data layout
//     that the spatial-mode kernel expects.
//------------------------------------------------------------------------------
#pragma once

#ifdef USE_CAIF_CUDA

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"

#include <cudnn.h>
#include <cstdint>

namespace instance
{

class CAIF_BatchNormCudnnHelpers:public CAIF_Base
{
  public:
    static void SyncFp32HostToDevice(const CAIF_DeviceTensor &host,
                                     CAIF_CudaStream &stream,
                                     CAIF_DeviceTensor &device_out);

    static void SyncFp32DeviceToHostOverwrite(const CAIF_DeviceTensor &device,
                                              CAIF_CudaStream &stream,
                                              CAIF_DeviceTensor &host_out);

    static void AccumulateFp32DeviceToHost(const CAIF_DeviceTensor &device_grad,
                                           CAIF_CudaStream &stream,
                                           CAIF_DeviceTensor &host_grad_inout);

    static void SetupBnParamDescriptor(cudnnTensorDescriptor_t desc,
                                       const uint32_t features);

    static void SetupBnDataDescriptor(cudnnTensorDescriptor_t desc,
                                      const uint32_t row_count,
                                      const uint32_t features,
                                      const cudnnDataType_t cudnn_dt);

  protected:

  private:
    CAIF_BatchNormCudnnHelpers()=delete;
};

}//end instance namespace

#endif // USE_CAIF_CUDA
