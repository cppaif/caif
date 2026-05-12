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
// Device-resident 2D convolution. Two paths:
//   - Host_e fp32: existing CPU loop (canonical weights live in host
//     fp32 HWCK = [Kh,Kw,Cin,Cout]).
//   - Device_e fp32 / fp16 / bf16: cuDNN convolution. Weights are
//     mirrored to device in KRSC = [Cout,Kh,Kw,Cin] (the layout cuDNN
//     expects for CUDNN_TENSOR_NHWC filters), in StorageT dtype, on
//     every device-path Forward. Grads are written back to host fp32
//     HWCK after every device-path Backward.
//------------------------------------------------------------------------------

#include "caif_device_conv2d.h"
#include "caif_device_conv2d_factory.h"
#include "caif_constants.h"
#include "caif_cudnn_util.h"
#include "caif_device_context.h"
#include "caif_exception.h"
#include <cmath>
#include <cstring>
#include <vector>

#ifdef USE_CAIF_CUDA
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{


constexpr size_t CONV2D_INPUT_RANK=4;

void TransposeHwckToKrscFp32(const float *src_hwck,
                              float *dst_krsc,
                              uint32_t Kh,
                              uint32_t Kw,
                              uint32_t Cin,
                              uint32_t Cout)
{
  for(uint32_t cout=0;cout<Cout;++cout)
  {
    for(uint32_t kh=0;kh<Kh;++kh)
    {
      for(uint32_t kw=0;kw<Kw;++kw)
      {
        for(uint32_t cin=0;cin<Cin;++cin)
        {
          const size_t hwck=((static_cast<size_t>(kh)*Kw+kw)*Cin+cin)*Cout+cout;
          const size_t krsc=((static_cast<size_t>(cout)*Kh+kh)*Kw+kw)*Cin+cin;
          dst_krsc[krsc]=src_hwck[hwck];
        }
      }
    }
  }
}

void TransposeKrscToHwckFp32Accumulate(const float *src_krsc,
                                        float *dst_hwck,
                                        uint32_t Kh,
                                        uint32_t Kw,
                                        uint32_t Cin,
                                        uint32_t Cout)
{
  for(uint32_t cout=0;cout<Cout;++cout)
  {
    for(uint32_t kh=0;kh<Kh;++kh)
    {
      for(uint32_t kw=0;kw<Kw;++kw)
      {
        for(uint32_t cin=0;cin<Cin;++cin)
        {
          const size_t hwck=((static_cast<size_t>(kh)*Kw+kw)*Cin+cin)*Cout+cout;
          const size_t krsc=((static_cast<size_t>(cout)*Kh+kh)*Kw+kw)*Cin+cin;
          dst_hwck[hwck]+=src_krsc[krsc];
        }
      }
    }
  }
}

#ifdef USE_CAIF_CUDA

// Sync host fp32 HWCK weights -> device StorageT KRSC weights.
// Ensures `device_krsc_out` is allocated with matching shape and StorageT.
void SyncWeightsHostToDevice(const CAIF_DeviceTensor &weights_host_hwck,
                              CAIF_CudaStream &stream,
                              CAIF_DataType::CAIF_DataType_e storage_dtype,
                              uint32_t Kh,
                              uint32_t Kw,
                              uint32_t Cin,
                              uint32_t Cout,
                              CAIF_DeviceTensor &device_krsc_out)
{
  const std::vector<uint32_t> krsc_shape={Cout,Kh,Kw,Cin};
  std::vector<float> staging(static_cast<size_t>(Cout)*Kh*Kw*Cin);
  // fp32 by helper contract: weights_host_hwck is the conv2d host-side
  // fp32 master weight tensor; storage_dtype dictates only the device copy.
  TransposeHwckToKrscFp32(static_cast<const float*>(weights_host_hwck.DeviceDataRaw()),
                           staging.data(),
                           Kh,
                           Kw,
                           Cin,
                           Cout);
  CAIF_DeviceTensor device_fp32=CAIF_DeviceTensor::Uninitialized(krsc_shape,
                                                                  stream,
                                                                  CAIF_DataType::CAIF_DataType_e::Float32);
  device_fp32.CopyFromHost(staging.data(),staging.size());
  if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    device_krsc_out=std::move(device_fp32);
  }
  else
  {
    device_krsc_out=device_fp32.To(storage_dtype);
  }
}

// Sync host fp32 bias -> device StorageT bias.
void SyncBiasHostToDevice(const CAIF_DeviceTensor &bias_host,
                           CAIF_CudaStream &stream,
                           CAIF_DataType::CAIF_DataType_e storage_dtype,
                           CAIF_DeviceTensor &bias_device_out)
{
  const std::vector<uint32_t> bias_shape=bias_host.Shape();
  CAIF_DeviceTensor device_fp32=CAIF_DeviceTensor::Uninitialized(bias_shape,
                                                                  stream,
                                                                  CAIF_DataType::CAIF_DataType_e::Float32);
  // fp32 by helper contract
  device_fp32.CopyFromHost(static_cast<const float*>(bias_host.DeviceDataRaw()),
                            bias_host.TotalElements());
  if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    bias_device_out=std::move(device_fp32);
  }
  else
  {
    bias_device_out=device_fp32.To(storage_dtype);
  }
}

// Sync device StorageT KRSC grad-weights back to host fp32 HWCK,
// accumulating onto the existing host grad (matches the host-path
// `g_w[w_idx]+=...` accumulation contract).
void SyncWeightsGradDeviceToHostAccumulate(const CAIF_DeviceTensor &device_grad_krsc,
                                            CAIF_CudaStream &stream,
                                            CAIF_DataType::CAIF_DataType_e storage_dtype,
                                            uint32_t Kh,
                                            uint32_t Kw,
                                            uint32_t Cin,
                                            uint32_t Cout,
                                            CAIF_DeviceTensor &weights_grad_host_hwck)
{
  CAIF_DeviceTensor device_fp32_krsc;
  if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    device_fp32_krsc=device_grad_krsc.Clone();
  }
  else
  {
    device_fp32_krsc=device_grad_krsc.To(CAIF_DataType::CAIF_DataType_e::Float32);
  }
  std::vector<float> staging(static_cast<size_t>(Cout)*Kh*Kw*Cin);
  device_fp32_krsc.CopyToHost(staging.data());
  stream.Synchronize();
  // fp32 by helper contract
  TransposeKrscToHwckFp32Accumulate(staging.data(),
                                     static_cast<float*>(weights_grad_host_hwck.DeviceDataRaw()),
                                     Kh,
                                     Kw,
                                     Cin,
                                     Cout);
}

// Sync device StorageT bias-grad back to host fp32 bias-grad accumulator.
void SyncBiasGradDeviceToHostAccumulate(const CAIF_DeviceTensor &bias_grad_device,
                                         CAIF_CudaStream &stream,
                                         CAIF_DataType::CAIF_DataType_e storage_dtype,
                                         CAIF_DeviceTensor &bias_grad_host)
{
  CAIF_DeviceTensor device_fp32;
  if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    device_fp32=bias_grad_device.Clone();
  }
  else
  {
    device_fp32=bias_grad_device.To(CAIF_DataType::CAIF_DataType_e::Float32);
  }
  const size_t n=bias_grad_host.TotalElements();
  std::vector<float> staging(n);
  device_fp32.CopyToHost(staging.data());
  stream.Synchronize();
  // fp32 by helper contract
  float *dst=static_cast<float*>(bias_grad_host.DeviceDataRaw());
  for(size_t i=0;i<n;++i)
  {
    dst[i]+=staging[i];
  }
}

// Run the full cuDNN forward (conv + bias add). Returns output tensor
// in NHWC layout, StorageT dtype, on device. Caches input device tensor
// for backward via cached_input_out.
CAIF_DeviceTensor Conv2DForwardDevice(const CAIF_DeviceTensor &input,
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
                                       CAIF_DeviceTensor &cached_input_out)
{
  const std::vector<uint32_t> &in_shape=input.Shape();
  const uint32_t N=in_shape[0];
  const uint32_t H=in_shape[1];
  const uint32_t W=in_shape[2];
  if(in_shape[3]!=Cin)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: input channel dim mismatch");
  }
  if(H<Kh||W<Kw)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: kernel larger than input");
  }
  const uint32_t H_out=(H-Kh)/Sh+1;
  const uint32_t W_out=(W-Kw)/Sw+1;

  CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({N,H_out,W_out,Cout},
                                                             stream,
                                                             storage_dtype);

  cudnnHandle_t handle=CAIF_DeviceContext::Instance().CudnnHandle();
  CAIF_DeviceContext::Instance().SetCudnnStream(stream.Handle());

  const cudnnDataType_t cudnn_dt=CAIF_CudnnUtil::CudnnDtypeFromStorage(storage_dtype);
  // Compute type: cuDNN uses fp32 accumulators for fp16/bf16 conv to
  // avoid overflow, and fp32 for fp32 conv. Match that.
  const cudnnDataType_t conv_compute_dt=CUDNN_DATA_FLOAT;

  cudnnTensorDescriptor_t in_desc=nullptr;
  cudnnTensorDescriptor_t out_desc=nullptr;
  cudnnTensorDescriptor_t bias_desc=nullptr;
  cudnnFilterDescriptor_t filter_desc=nullptr;
  cudnnConvolutionDescriptor_t conv_desc=nullptr;
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&in_desc),"create in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&out_desc),"create out_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&bias_desc),"create bias_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateFilterDescriptor(&filter_desc),"create filter_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateConvolutionDescriptor(&conv_desc),"create conv_desc");

  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(in_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(Cin),
                                                        static_cast<int>(H),
                                                        static_cast<int>(W)),
                             "set in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(out_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(Cout),
                                                        static_cast<int>(H_out),
                                                        static_cast<int>(W_out)),
                             "set out_desc");
  // Bias is broadcast over N/H/W, applied per-Cout channel. Use NHWC
  // descriptor [1,Cout,1,1]; cuDNN broadcasts across the spatial dims.
  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(bias_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        1,
                                                        static_cast<int>(Cout),
                                                        1,
                                                        1),
                             "set bias_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetFilter4dDescriptor(filter_desc,
                                                        cudnn_dt,
                                                        CUDNN_TENSOR_NHWC,
                                                        static_cast<int>(Cout),
                                                        static_cast<int>(Cin),
                                                        static_cast<int>(Kh),
                                                        static_cast<int>(Kw)),
                             "set filter_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                             0,
                                                             0,
                                                             static_cast<int>(Sh),
                                                             static_cast<int>(Sw),
                                                             1,
                                                             1,
                                                             CUDNN_CROSS_CORRELATION,
                                                             conv_compute_dt),
                             "set conv_desc");

  // Algorithm selection — pick first algo that fits the workspace cap.
  int algo_count=0;
  cudnnConvolutionFwdAlgoPerf_t algo_perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  CAIF_CudnnUtil::CheckCudnn(cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                                                    in_desc,
                                                                    filter_desc,
                                                                    conv_desc,
                                                                    out_desc,
                                                                    CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                                                    &algo_count,
                                                                    algo_perf),
                             "GetForwardAlgorithm_v7");
  cudnnConvolutionFwdAlgo_t fwd_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t workspace_bytes=0;
  bool found=false;
  for(int i=0;i<algo_count;++i)
  {
    if(algo_perf[i].status==CUDNN_STATUS_SUCCESS&&
       algo_perf[i].memory<=g_caif_cudnn_workspace_max_bytes)
    {
      fwd_algo=algo_perf[i].algo;
      workspace_bytes=algo_perf[i].memory;
      found=true;
      break;
    }
  }
  if(found==false)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: no cuDNN forward algorithm fits workspace cap");
  }

  void *workspace=nullptr;
  if(workspace_bytes>0)
  {
    if(cudaMalloc(&workspace,workspace_bytes)!=cudaSuccess)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: workspace cudaMalloc failed");
    }
  }

  const float alpha=1.0f;
  const float beta=0.0f;
  CAIF_CudnnUtil::CheckCudnn(cudnnConvolutionForward(handle,
                                                     &alpha,
                                                     in_desc,
                                                     input.DeviceDataRaw(),
                                                     filter_desc,
                                                     weights_device_krsc.DeviceDataRaw(),
                                                     conv_desc,
                                                     fwd_algo,
                                                     workspace,
                                                     workspace_bytes,
                                                     &beta,
                                                     out_desc,
                                                     output.DeviceDataRaw()),
                             "ConvolutionForward");

  // Bias add: out = out + bias (per-Cout broadcast)
  const float bias_alpha=1.0f;
  const float bias_beta=1.0f;
  CAIF_CudnnUtil::CheckCudnn(cudnnAddTensor(handle,
                                            &bias_alpha,
                                            bias_desc,
                                            bias_device.DeviceDataRaw(),
                                            &bias_beta,
                                            out_desc,
                                            output.DeviceDataRaw()),
                             "AddTensor (bias)");

  if(workspace!=nullptr)
  {
    cudaFree(workspace);
  }
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(in_desc);

  cached_input_out=input.Clone();
  return output;
}

// Run cuDNN backward (data + filter + bias) and write grads into the
// supplied device tensors. Returns grad_input in NHWC StorageT.
CAIF_DeviceTensor Conv2DBackwardDevice(const CAIF_DeviceTensor &grad_output,
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
                                        CAIF_DeviceTensor &bias_grad_device_out)
{
  const std::vector<uint32_t> &in_shape=cached_input.Shape();
  const uint32_t N=in_shape[0];
  const uint32_t H=in_shape[1];
  const uint32_t W=in_shape[2];
  const std::vector<uint32_t> &grad_out_shape=grad_output.Shape();
  const uint32_t H_out=grad_out_shape[1];
  const uint32_t W_out=grad_out_shape[2];

  CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized(in_shape,
                                                                 stream,
                                                                 storage_dtype);
  weights_grad_device_krsc_out=CAIF_DeviceTensor::Uninitialized({Cout,Kh,Kw,Cin},
                                                                 stream,
                                                                 storage_dtype);
  bias_grad_device_out=CAIF_DeviceTensor::Uninitialized({Cout},
                                                         stream,
                                                         storage_dtype);

  cudnnHandle_t handle=CAIF_DeviceContext::Instance().CudnnHandle();
  CAIF_DeviceContext::Instance().SetCudnnStream(stream.Handle());

  const cudnnDataType_t cudnn_dt=CAIF_CudnnUtil::CudnnDtypeFromStorage(storage_dtype);
  const cudnnDataType_t conv_compute_dt=CUDNN_DATA_FLOAT;

  cudnnTensorDescriptor_t in_desc=nullptr;
  cudnnTensorDescriptor_t out_desc=nullptr;
  cudnnTensorDescriptor_t bias_desc=nullptr;
  cudnnFilterDescriptor_t filter_desc=nullptr;
  cudnnConvolutionDescriptor_t conv_desc=nullptr;
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&in_desc),"create in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&out_desc),"create out_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&bias_desc),"create bias_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateFilterDescriptor(&filter_desc),"create filter_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateConvolutionDescriptor(&conv_desc),"create conv_desc");

  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(in_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(Cin),
                                                        static_cast<int>(H),
                                                        static_cast<int>(W)),
                             "set in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(out_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(Cout),
                                                        static_cast<int>(H_out),
                                                        static_cast<int>(W_out)),
                             "set out_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(bias_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        1,
                                                        static_cast<int>(Cout),
                                                        1,
                                                        1),
                             "set bias_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetFilter4dDescriptor(filter_desc,
                                                        cudnn_dt,
                                                        CUDNN_TENSOR_NHWC,
                                                        static_cast<int>(Cout),
                                                        static_cast<int>(Cin),
                                                        static_cast<int>(Kh),
                                                        static_cast<int>(Kw)),
                             "set filter_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                             0,
                                                             0,
                                                             static_cast<int>(Sh),
                                                             static_cast<int>(Sw),
                                                             1,
                                                             1,
                                                             CUDNN_CROSS_CORRELATION,
                                                             conv_compute_dt),
                             "set conv_desc");

  // BackwardData algorithm
  int data_count=0;
  cudnnConvolutionBwdDataAlgoPerf_t data_perf[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
  CAIF_CudnnUtil::CheckCudnn(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,
                                                                         filter_desc,
                                                                         out_desc,
                                                                         conv_desc,
                                                                         in_desc,
                                                                         CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                                                                         &data_count,
                                                                         data_perf),
                             "GetBackwardDataAlgorithm_v7");
  cudnnConvolutionBwdDataAlgo_t data_algo=CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  size_t data_ws_bytes=0;
  bool data_found=false;
  for(int i=0;i<data_count;++i)
  {
    if(data_perf[i].status==CUDNN_STATUS_SUCCESS&&
       data_perf[i].memory<=g_caif_cudnn_workspace_max_bytes)
    {
      data_algo=data_perf[i].algo;
      data_ws_bytes=data_perf[i].memory;
      data_found=true;
      break;
    }
  }
  if(data_found==false)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: no cuDNN backward-data algorithm fits workspace cap");
  }

  // BackwardFilter algorithm
  int filt_count=0;
  cudnnConvolutionBwdFilterAlgoPerf_t filt_perf[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
  CAIF_CudnnUtil::CheckCudnn(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle,
                                in_desc,
                                out_desc,
                                conv_desc,
                                filter_desc,
                                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                                &filt_count,
                                filt_perf),
                             "GetBackwardFilterAlgorithm_v7");
  cudnnConvolutionBwdFilterAlgo_t filt_algo=CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  size_t filt_ws_bytes=0;
  bool filt_found=false;
  for(int i=0;i<filt_count;++i)
  {
    if(filt_perf[i].status==CUDNN_STATUS_SUCCESS&&
       filt_perf[i].memory<=g_caif_cudnn_workspace_max_bytes)
    {
      filt_algo=filt_perf[i].algo;
      filt_ws_bytes=filt_perf[i].memory;
      filt_found=true;
      break;
    }
  }
  if(filt_found==false)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: no cuDNN backward-filter algorithm fits workspace cap");
  }

  const size_t ws_bytes=data_ws_bytes>filt_ws_bytes?data_ws_bytes:filt_ws_bytes;
  void *workspace=nullptr;
  if(ws_bytes>0)
  {
    if(cudaMalloc(&workspace,ws_bytes)!=cudaSuccess)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: backward workspace cudaMalloc failed");
    }
  }

  const float alpha=1.0f;
  const float beta=0.0f;

  CAIF_CudnnUtil::CheckCudnn(cudnnConvolutionBackwardData(handle,
                                                          &alpha,
                                                          filter_desc,
                                                          weights_device_krsc.DeviceDataRaw(),
                                                          out_desc,
                                                          grad_output.DeviceDataRaw(),
                                                          conv_desc,
                                                          data_algo,
                                                          workspace,
                                                          ws_bytes,
                                                          &beta,
                                                          in_desc,
                                                          grad_input.DeviceDataRaw()),
                             "ConvolutionBackwardData");

  CAIF_CudnnUtil::CheckCudnn(cudnnConvolutionBackwardFilter(handle,
                                                            &alpha,
                                                            in_desc,
                                                            cached_input.DeviceDataRaw(),
                                                            out_desc,
                                                            grad_output.DeviceDataRaw(),
                                                            conv_desc,
                                                            filt_algo,
                                                            workspace,
                                                            ws_bytes,
                                                            &beta,
                                                            filter_desc,
                                                            weights_grad_device_krsc_out.DeviceDataRaw()),
                             "ConvolutionBackwardFilter");

  CAIF_CudnnUtil::CheckCudnn(cudnnConvolutionBackwardBias(handle,
                                                          &alpha,
                                                          out_desc,
                                                          grad_output.DeviceDataRaw(),
                                                          &beta,
                                                          bias_desc,
                                                          bias_grad_device_out.DeviceDataRaw()),
                             "ConvolutionBackwardBias");

  if(workspace!=nullptr)
  {
    cudaFree(workspace);
  }
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(in_desc);

  return grad_input;
}

#endif // USE_CAIF_CUDA

CAIF_DeviceTensor Conv2DForwardHost(const CAIF_DeviceTensor &input,
                                     const CAIF_DeviceTensor &weights_host_hwck,
                                     const CAIF_DeviceTensor &bias_host,
                                     uint32_t Kh,
                                     uint32_t Kw,
                                     uint32_t Sh,
                                     uint32_t Sw,
                                     uint32_t Cin,
                                     uint32_t Cout,
                                     std::vector<uint32_t> &cached_input_shape_out,
                                     std::vector<float> &cached_input_host_out)
{
  // host path is fp32-only by Stage 5b contract (CPU im2col + GEMM
  // backend operates on fp32 master tensors; non-fp32 storage stays on
  // device for the cuDNN forward path above this function).
  if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CAIF_DeviceConv2D host path: only fp32 supported on host_e");
  }
  // fp32 host path: input/output (or grad_*) are Float32 by the throw
  // above; per-site markers on each cast below.
  const std::vector<uint32_t> &in_shape=input.Shape();
  const uint32_t N=in_shape[0];
  const uint32_t H=in_shape[1];
  const uint32_t W=in_shape[2];
  if(in_shape[3]!=Cin)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: input channel dim mismatch");
  }
  if(H<Kh||W<Kw)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: kernel larger than input");
  }
  const uint32_t H_out=(H-Kh)/Sh+1;
  const uint32_t W_out=(W-Kw)/Sw+1;

  cached_input_shape_out=in_shape;
  // fp32 host path
  const float *x=static_cast<const float *>(input.DeviceDataRaw());
  const size_t in_total=input.TotalElements();
  cached_input_host_out.assign(x,x+in_total);

  CAIF_DeviceTensor output=CAIF_DeviceTensor::ZerosHost({N,H_out,W_out,Cout});
  // fp32 host path
  float *y=static_cast<float *>(output.DeviceDataRaw());
  // fp32 by contract
  const float *w=static_cast<const float *>(weights_host_hwck.DeviceDataRaw());
  // fp32 by contract
  const float *b=static_cast<const float *>(bias_host.DeviceDataRaw());

  for(uint32_t n=0;n<N;++n)
  {
    for(uint32_t oh=0;oh<H_out;++oh)
    {
      for(uint32_t ow=0;ow<W_out;++ow)
      {
        for(uint32_t oc=0;oc<Cout;++oc)
        {
          double acc=static_cast<double>(b[oc]);
          for(uint32_t kh=0;kh<Kh;++kh)
          {
            for(uint32_t kw=0;kw<Kw;++kw)
            {
              const uint32_t ih=oh*Sh+kh;
              const uint32_t iw=ow*Sw+kw;
              for(uint32_t ic=0;ic<Cin;++ic)
              {
                const size_t in_idx=((static_cast<size_t>(n)*H+ih)*W+iw)*Cin+ic;
                const size_t w_idx=((static_cast<size_t>(kh)*Kw+kw)*Cin+ic)*Cout+oc;
                acc+=static_cast<double>(x[in_idx])*static_cast<double>(w[w_idx]);
              }
            }
          }
          const size_t out_idx=((static_cast<size_t>(n)*H_out+oh)*W_out+ow)*Cout+oc;
          y[out_idx]=static_cast<float>(acc);
        }
      }
    }
  }
  return output;
}

CAIF_DeviceTensor Conv2DBackwardHost(const CAIF_DeviceTensor &grad_output,
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
                                      CAIF_DeviceTensor &bias_grad_host)
{
  // host path is fp32-only by Stage 5b contract.
  if(grad_output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CAIF_DeviceConv2D host path: only fp32 supported on host_e");
  }
  // fp32 host path: input/output (or grad_*) are Float32 by the throw
  // above; per-site markers on each cast below.
  if(cached_input_shape.empty()==true)
  {
    THROW_CAIFE("CAIF_DeviceConv2D: backward called before forward");
  }
  const uint32_t N=cached_input_shape[0];
  const uint32_t H=cached_input_shape[1];
  const uint32_t W=cached_input_shape[2];
  const uint32_t H_out=(H-Kh)/Sh+1;
  const uint32_t W_out=(W-Kw)/Sw+1;

  CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::ZerosHost(cached_input_shape);
  // fp32 host path
  float *g_in=static_cast<float *>(grad_input.DeviceDataRaw());
  // fp32 host path
  const float *g_out=static_cast<const float *>(grad_output.DeviceDataRaw());
  // fp32 by contract
  const float *w=static_cast<const float *>(weights_host_hwck.DeviceDataRaw());
  // fp32 by contract
  float *g_w=static_cast<float *>(weights_grad_host_hwck.DeviceDataRaw());
  // fp32 by contract
  float *g_b=static_cast<float *>(bias_grad_host.DeviceDataRaw());
  const float *x=cached_input_host.data();

  for(uint32_t n=0;n<N;++n)
  {
    for(uint32_t oh=0;oh<H_out;++oh)
    {
      for(uint32_t ow=0;ow<W_out;++ow)
      {
        for(uint32_t oc=0;oc<Cout;++oc)
        {
          const size_t out_idx=((static_cast<size_t>(n)*H_out+oh)*W_out+ow)*Cout+oc;
          const float dy=g_out[out_idx];
          g_b[oc]+=dy;
          for(uint32_t kh=0;kh<Kh;++kh)
          {
            for(uint32_t kw=0;kw<Kw;++kw)
            {
              const uint32_t ih=oh*Sh+kh;
              const uint32_t iw=ow*Sw+kw;
              for(uint32_t ic=0;ic<Cin;++ic)
              {
                const size_t in_idx=((static_cast<size_t>(n)*H+ih)*W+iw)*Cin+ic;
                const size_t w_idx=((static_cast<size_t>(kh)*Kw+kw)*Cin+ic)*Cout+oc;
                g_w[w_idx]+=x[in_idx]*dy;
                g_in[in_idx]+=w[w_idx]*dy;
              }
            }
          }
        }
      }
    }
  }
  return grad_input;
}


template<typename ComputeT,typename StorageT>
CAIF_DeviceConv2D<ComputeT,StorageT>::CAIF_DeviceConv2D(const Config_t &config,
                                                        CAIF_CudaStream &stream):
                                          Base_t(stream),
                                          _config(config),
                                          _weights(),
                                          _bias(),
                                          _weights_grad(),
                                          _bias_grad(),
                                          _weights_device_krsc(),
                                          _bias_device(),
                                          _weights_grad_device_krsc(),
                                          _bias_grad_device(),
                                          _cached_input_shape(),
                                          _cached_input_host(),
                                          _cached_input_device()
{
  try
  {
    if(config.in_channels==0||config.out_channels==0)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: channel dims must be > 0");
    }
    if(config.kernel_height==0||config.kernel_width==0)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: kernel dims must be > 0");
    }
    if(config.stride_height==0||config.stride_width==0)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: stride dims must be > 0");
    }

    const std::vector<uint32_t> w_shape={config.kernel_height,
                                         config.kernel_width,
                                         config.in_channels,
                                         config.out_channels};
    const std::vector<uint32_t> b_shape={config.out_channels};
    _weights=CAIF_DeviceTensor::ZerosHost(w_shape);
    _bias=CAIF_DeviceTensor::ZerosHost(b_shape);
    _weights_grad=CAIF_DeviceTensor::ZerosHost(w_shape);
    _bias_grad=CAIF_DeviceTensor::ZerosHost(b_shape);

    const size_t fan_in=static_cast<size_t>(config.kernel_height)*
                         config.kernel_width*config.in_channels;
    const size_t fan_out=static_cast<size_t>(config.kernel_height)*
                          config.kernel_width*config.out_channels;
    const float limit=std::sqrt(g_caif_xavier_uniform_scale/static_cast<float>(fan_in+fan_out));
    const size_t total=fan_in*config.out_channels;
    // fp32 by contract
    float *w=static_cast<float *>(_weights.DeviceDataRaw());
    for(size_t i=0;i<total;++i)
    {
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      w[i]=(t-std::floor(t))*2.0f*limit-limit;
    }
  }
  CAIF_CATCH_BLOCK();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceConv2D<ComputeT,StorageT>::CAIF_DeviceConv2D(CAIF_DeviceConv2D &&other):
                                Base_t(std::move(other)),
                                _config(other._config),
                                _weights(std::move(other._weights)),
                                _bias(std::move(other._bias)),
                                _weights_grad(std::move(other._weights_grad)),
                                _bias_grad(std::move(other._bias_grad)),
                                _weights_device_krsc(std::move(other._weights_device_krsc)),
                                _bias_device(std::move(other._bias_device)),
                                _weights_grad_device_krsc(std::move(other._weights_grad_device_krsc)),
                                _bias_grad_device(std::move(other._bias_grad_device)),
                                _cached_input_shape(std::move(other._cached_input_shape)),
                                _cached_input_host(std::move(other._cached_input_host)),
                                _cached_input_device(std::move(other._cached_input_device))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceConv2D<ComputeT,StorageT> &
CAIF_DeviceConv2D<ComputeT,StorageT>::operator=(CAIF_DeviceConv2D &&other)
{
  try
  {
    if(this!=&other)
    {
      Base_t::operator=(std::move(other));
      _config=other._config;
      _weights=std::move(other._weights);
      _bias=std::move(other._bias);
      _weights_grad=std::move(other._weights_grad);
      _bias_grad=std::move(other._bias_grad);
      _weights_device_krsc=std::move(other._weights_device_krsc);
      _bias_device=std::move(other._bias_device);
      _weights_grad_device_krsc=std::move(other._weights_grad_device_krsc);
      _bias_grad_device=std::move(other._bias_grad_device);
      _cached_input_shape=std::move(other._cached_input_shape);
      _cached_input_host=std::move(other._cached_input_host);
      _cached_input_device=std::move(other._cached_input_device);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK();
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceConv2D<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                   CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    const std::vector<uint32_t> &in_shape=input.Shape();
    if(in_shape.size()!=CONV2D_INPUT_RANK)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: expects rank-4 input [N,H,W,Cin]");
    }
    const uint32_t Kh=Config().kernel_height;
    const uint32_t Kw=Config().kernel_width;
    const uint32_t Sh=Config().stride_height;
    const uint32_t Sw=Config().stride_width;
    const uint32_t Cin=Config().in_channels;
    const uint32_t Cout=Config().out_channels;

    if(input.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      SetCachedInputDevice(CAIF_DeviceTensor());
      std::vector<uint32_t> shape_out;
      std::vector<float> cached_in_host;
      CAIF_DeviceTensor result=Conv2DForwardHost(input,
                                                  Weights(),
                                                  Bias(),
                                                  Kh,
                                                  Kw,
                                                  Sh,
                                                  Sw,
                                                  Cin,
                                                  Cout,
                                                  shape_out,
                                                  cached_in_host);
      SetCachedInputShape(shape_out);
      SetCachedInputHost(std::move(cached_in_host));
      return result;
    }
#ifdef USE_CAIF_CUDA
    SetCachedInputShape(in_shape);
    SetCachedInputHost(std::vector<float>());

    // Sync canonical host weights/bias to device every Forward (correct
    // regardless of optimizer pattern; sub-millisecond for typical sizes).
    CAIF_DeviceTensor weights_dev;
    CAIF_DeviceTensor bias_dev;
    SyncWeightsHostToDevice(Weights(),Stream(),StorageDtype(),Kh,Kw,Cin,Cout,weights_dev);
    SyncBiasHostToDevice(Bias(),Stream(),StorageDtype(),bias_dev);
    SetWeightsDeviceKRSC(std::move(weights_dev));
    SetBiasDevice(std::move(bias_dev));

    CAIF_DeviceTensor cached_in_dev;
    CAIF_DeviceTensor result=Conv2DForwardDevice(input,
                                                  Stream(),
                                                  WeightsDeviceKRSC(),
                                                  BiasDevice(),
                                                  Kh,
                                                  Kw,
                                                  Sh,
                                                  Sw,
                                                  Cin,
                                                  Cout,
                                                  StorageDtype(),
                                                  cached_in_dev);
    SetCachedInputDevice(std::move(cached_in_dev));
    return result;
#else
    THROW_CAIFE("CAIF_DeviceConv2D: device path requires USE_CAIF_CUDA");
#endif
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceConv2D<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    const uint32_t Kh=Config().kernel_height;
    const uint32_t Kw=Config().kernel_width;
    const uint32_t Sh=Config().stride_height;
    const uint32_t Sw=Config().stride_width;
    const uint32_t Cin=Config().in_channels;
    const uint32_t Cout=Config().out_channels;

    if(grad_output.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      return Conv2DBackwardHost(grad_output,
                                CachedInputShape(),
                                CachedInputHost(),
                                Weights(),
                                Kh,
                                Kw,
                                Sh,
                                Sw,
                                Cin,
                                Cout,
                                WeightsGrad(),
                                BiasGrad());
    }
#ifdef USE_CAIF_CUDA
    if(CachedInputDevice().IsAllocated()==false)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: backward called before forward (device path)");
    }
    if(WeightsDeviceKRSC().IsAllocated()==false)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: device weights not synced");
    }
    CAIF_DeviceTensor wgrad_dev;
    CAIF_DeviceTensor bgrad_dev;
    CAIF_DeviceTensor grad_input=Conv2DBackwardDevice(grad_output,
                                                       Stream(),
                                                       CachedInputDevice(),
                                                       WeightsDeviceKRSC(),
                                                       Kh,
                                                       Kw,
                                                       Sh,
                                                       Sw,
                                                       Cin,
                                                       Cout,
                                                       StorageDtype(),
                                                       wgrad_dev,
                                                       bgrad_dev);
    // Accumulate device grads back into canonical host fp32 grads. The
    // host grad tensors are kept in fp32 HWCK so the optimizer (which
    // currently runs on host fp32) sees a consistent view.
    SyncWeightsGradDeviceToHostAccumulate(wgrad_dev,
                                           Stream(),
                                           StorageDtype(),
                                           Kh,
                                           Kw,
                                           Cin,
                                           Cout,
                                           WeightsGrad());
    SyncBiasGradDeviceToHostAccumulate(bgrad_dev,Stream(),StorageDtype(),BiasGrad());
    SetWeightsGradDeviceKRSC(std::move(wgrad_dev));
    SetBiasGradDevice(std::move(bgrad_dev));
    return grad_input;
#else
    THROW_CAIFE("CAIF_DeviceConv2D: device path requires USE_CAIF_CUDA");
#endif
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceConv2D<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    const size_t w_total=WeightsGrad().TotalElements();
    const size_t b_total=BiasGrad().TotalElements();
    std::memset(WeightsGrad().DeviceDataRaw(),0,w_total*sizeof(float));
    std::memset(BiasGrad().DeviceDataRaw(),0,b_total*sizeof(float));
  }
  CAIF_CATCH_BLOCK();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceConv2D<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return Weights();
    }
    if(index==1)
    {
      return Bias();
    }
    THROW_CAIFE("CAIF_DeviceConv2D: parameter index out of range");
  }
  CAIF_CATCH_BLOCK();
  return Weights();
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceConv2D<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return Weights();
    }
    if(index==1)
    {
      return Bias();
    }
    THROW_CAIFE("CAIF_DeviceConv2D: parameter index out of range");
  }
  CAIF_CATCH_BLOCK();
  return Weights();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceConv2D<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(index==0)
    {
      return WeightsGrad();
    }
    if(index==1)
    {
      return BiasGrad();
    }
    THROW_CAIFE("CAIF_DeviceConv2D: gradient index out of range");
  }
  CAIF_CATCH_BLOCK();
  return WeightsGrad();
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceConv2D<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(index==0)
    {
      return WeightsGrad();
    }
    if(index==1)
    {
      return BiasGrad();
    }
    THROW_CAIFE("CAIF_DeviceConv2D: gradient index out of range");
  }
  CAIF_CATCH_BLOCK();
  return WeightsGrad();
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceConv2D<ComputeT,StorageT>::TotalParameterCount()const
{
  return Weights().TotalElements()+Bias().TotalElements();
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceConv2D<ComputeT,StorageT>::Description()const
{
  return g_caif_description_conv2d;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceConv2D<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  std::vector<std::string> names;
  names.push_back(prefix+"weights");
  names.push_back(prefix+"bias");
  return names;
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceConv2D<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceConv2D<float,__half>;
template class CAIF_DeviceConv2D<float,__nv_bfloat16>;
template class CAIF_DeviceConv2D<__half,float>;
template class CAIF_DeviceConv2D<__half,__half>;
template class CAIF_DeviceConv2D<__half,__nv_bfloat16>;
template class CAIF_DeviceConv2D<__nv_bfloat16,float>;
template class CAIF_DeviceConv2D<__nv_bfloat16,__half>;
template class CAIF_DeviceConv2D<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
