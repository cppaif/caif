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
// Conv2D backend primitives used by CAIF_DeviceConv2D. Extracted from former
// file-scope free functions in caif_device_conv2d.cpp (structural refactor: all
// functions must be class methods). Three groups:
//   - Conv2DForwardHost / Conv2DBackwardHost: the fp32 CPU reference path.
//   - Conv2DForwardDevice / Conv2DBackwardDevice: the cuDNN conv path.
//   - Sync* : move fp32 host master weights/bias and their gradients to and
//     from the StorageT device buffers cuDNN operates on, transposing
//     between the host HWCK layout and the device KRSC filter layout.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_data_type.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"

#include <cstdint>
#include <vector>

namespace instance
{

class CAIF_Conv2dBackend:public CAIF_Base
{
  public:
    static CAIF_DeviceTensor Conv2DForwardHost(const CAIF_DeviceTensor &input,
                                               const CAIF_DeviceTensor &weights_host_hwck,
                                               const CAIF_DeviceTensor &bias_host,
                                               uint32_t Kh,
                                               uint32_t Kw,
                                               uint32_t Sh,
                                               uint32_t Sw,
                                               uint32_t Cin,
                                               uint32_t Cout,
                                               std::vector<uint32_t> &cached_input_shape_out,
                                               std::vector<float> &cached_input_host_out);

    static CAIF_DeviceTensor Conv2DBackwardHost(const CAIF_DeviceTensor &grad_output,
                                                const std::vector<uint32_t> &cached_input_shape,
                                                const std::vector<float> &cached_input_host,
                                                const CAIF_DeviceTensor &weights_host_hwck,
                                                uint32_t Kh,
                                                uint32_t Kw,
                                                uint32_t Sh,
                                                uint32_t Sw,
                                                uint32_t Cin,
                                                uint32_t Cout,
                                                CAIF_DeviceTensor &weights_grad_host_hwck,
                                                CAIF_DeviceTensor &bias_grad_host);

#ifdef USE_CAIF_CUDA
    // Sync host fp32 HWCK weights -> device StorageT KRSC weights.
    // Ensures `device_krsc_out` is allocated with matching shape and StorageT.
    static void SyncWeightsHostToDevice(const CAIF_DeviceTensor &weights_host_hwck,
                                        CAIF_CudaStream &stream,
                                        CAIF_DataType::CAIF_DataType_e storage_dtype,
                                        uint32_t Kh,
                                        uint32_t Kw,
                                        uint32_t Cin,
                                        uint32_t Cout,
                                        CAIF_DeviceTensor &device_krsc_out);

    // Sync host fp32 bias -> device StorageT bias.
    static void SyncBiasHostToDevice(const CAIF_DeviceTensor &bias_host,
                                     CAIF_CudaStream &stream,
                                     CAIF_DataType::CAIF_DataType_e storage_dtype,
                                     CAIF_DeviceTensor &bias_device_out);

    // Sync device StorageT KRSC grad-weights back to host fp32 HWCK,
    // accumulating onto the existing host grad (matches the host-path
    // `g_w[w_idx]+=...` accumulation contract).
    static void SyncWeightsGradDeviceToHostAccumulate(const CAIF_DeviceTensor &device_grad_krsc,
                                                      CAIF_CudaStream &stream,
                                                      CAIF_DataType::CAIF_DataType_e storage_dtype,
                                                      uint32_t Kh,
                                                      uint32_t Kw,
                                                      uint32_t Cin,
                                                      uint32_t Cout,
                                                      CAIF_DeviceTensor &weights_grad_host_hwck);

    // Sync device StorageT bias-grad back to host fp32 bias-grad accumulator.
    static void SyncBiasGradDeviceToHostAccumulate(const CAIF_DeviceTensor &bias_grad_device,
                                                   CAIF_CudaStream &stream,
                                                   CAIF_DataType::CAIF_DataType_e storage_dtype,
                                                   CAIF_DeviceTensor &bias_grad_host);

    // Run the full cuDNN forward (conv + bias add). Returns output tensor in
    // NHWC layout, StorageT dtype, on device. Caches the input device tensor
    // for backward via cached_input_out.
    static CAIF_DeviceTensor Conv2DForwardDevice(const CAIF_DeviceTensor &input,
                                                 CAIF_CudaStream &stream,
                                                 const CAIF_DeviceTensor &weights_device_krsc,
                                                 const CAIF_DeviceTensor &bias_device,
                                                 uint32_t Kh,
                                                 uint32_t Kw,
                                                 uint32_t Sh,
                                                 uint32_t Sw,
                                                 uint32_t Cin,
                                                 uint32_t Cout,
                                                 CAIF_DataType::CAIF_DataType_e storage_dtype,
                                                 CAIF_DeviceTensor &cached_input_out);

    // Run cuDNN backward (data + filter + bias) and write grads into the
    // supplied device tensors. Returns grad_input in NHWC StorageT.
    static CAIF_DeviceTensor Conv2DBackwardDevice(const CAIF_DeviceTensor &grad_output,
                                                  CAIF_CudaStream &stream,
                                                  const CAIF_DeviceTensor &cached_input,
                                                  const CAIF_DeviceTensor &weights_device_krsc,
                                                  uint32_t Kh,
                                                  uint32_t Kw,
                                                  uint32_t Sh,
                                                  uint32_t Sw,
                                                  uint32_t Cin,
                                                  uint32_t Cout,
                                                  CAIF_DataType::CAIF_DataType_e storage_dtype,
                                                  CAIF_DeviceTensor &weights_grad_device_krsc_out,
                                                  CAIF_DeviceTensor &bias_grad_device_out);
#endif // USE_CAIF_CUDA

  protected:

  private:
    CAIF_Conv2dBackend()=delete;

    // HWCK (host master layout) <-> KRSC (cuDNN filter layout) fp32
    // transposes. Internal to the Sync* weight helpers.
    static void TransposeHwckToKrscFp32(const float *src_hwck,
                                        float *dst_krsc,
                                        uint32_t Kh,
                                        uint32_t Kw,
                                        uint32_t Cin,
                                        uint32_t Cout);

    static void TransposeKrscToHwckFp32Accumulate(const float *src_krsc,
                                                  float *dst_hwck,
                                                  uint32_t Kh,
                                                  uint32_t Kw,
                                                  uint32_t Cin,
                                                  uint32_t Cout);
};

}//end instance namespace
