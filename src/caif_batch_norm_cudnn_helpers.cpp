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

#ifdef USE_CAIF_CUDA

#include "caif_batch_norm_cudnn_helpers.h"
#include "caif_cudnn_util.h"
#include "caif_exception.h"

#include <cstring>
#include <vector>

namespace instance
{

void CAIF_BatchNormCudnnHelpers::SyncFp32HostToDevice(const CAIF_DeviceTensor &host,
                                                      CAIF_CudaStream &stream,
                                                      CAIF_DeviceTensor &device_out)
{
  try
  {
    device_out=CAIF_DeviceTensor::Uninitialized(host.Shape(),
                                                stream,
                                                CAIF_DataType::CAIF_DataType_e::Float32);
    // fp32 by helper contract
    device_out.CopyFromHost(static_cast<const float*>(host.DeviceDataRaw()),
                            host.TotalElements());
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_BatchNormCudnnHelpers::SyncFp32DeviceToHostOverwrite(const CAIF_DeviceTensor &device,
                                                                CAIF_CudaStream &stream,
                                                                CAIF_DeviceTensor &host_out)
{
  try
  {
    std::vector<float> staging(device.TotalElements());
    device.CopyToHost(staging.data());
    stream.Synchronize();
    std::memcpy(host_out.DeviceDataRaw(),
                staging.data(),
                staging.size()*sizeof(float));
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_BatchNormCudnnHelpers::AccumulateFp32DeviceToHost(const CAIF_DeviceTensor &device_grad,
                                                             CAIF_CudaStream &stream,
                                                             CAIF_DeviceTensor &host_grad_inout)
{
  try
  {
    std::vector<float> staging(device_grad.TotalElements());
    device_grad.CopyToHost(staging.data());
    stream.Synchronize();
    // fp32 by helper contract
    float *acc=static_cast<float*>(host_grad_inout.DeviceDataRaw());
    for(size_t i=0;i<staging.size();++i)
    {
      acc[i]+=staging[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_BatchNormCudnnHelpers::SetupBnParamDescriptor(cudnnTensorDescriptor_t desc,
                                                         const uint32_t features)
{
  try
  {
    CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(desc,
                                                          CUDNN_TENSOR_NCHW,
                                                          CUDNN_DATA_FLOAT,
                                                          1,
                                                          static_cast<int>(features),
                                                          1,
                                                          1),
                               "set bn_param_desc");
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_BatchNormCudnnHelpers::SetupBnDataDescriptor(cudnnTensorDescriptor_t desc,
                                                        const uint32_t row_count,
                                                        const uint32_t features,
                                                        const cudnnDataType_t cudnn_dt)
{
  try
  {
    CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(desc,
                                                          CUDNN_TENSOR_NCHW,
                                                          cudnn_dt,
                                                          static_cast<int>(row_count),
                                                          static_cast<int>(features),
                                                          1,
                                                          1),
                               "set bn_data_desc");
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace

#endif // USE_CAIF_CUDA
