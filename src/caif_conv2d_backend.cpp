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

#include "caif_conv2d_backend.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include <vector>

#ifdef USE_CAIF_CUDA
#include "caif_cudnn_util.h"
#include "caif_device_context.h"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{

void CAIF_Conv2dBackend::TransposeHwckToKrscFp32(const float *src_hwck,
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

void CAIF_Conv2dBackend::TransposeKrscToHwckFp32Accumulate(const float *src_krsc,
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

void CAIF_Conv2dBackend::SyncWeightsHostToDevice(const CAIF_DeviceTensor &weights_host_hwck,
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

void CAIF_Conv2dBackend::SyncBiasHostToDevice(const CAIF_DeviceTensor &bias_host,
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

void CAIF_Conv2dBackend::SyncWeightsGradDeviceToHostAccumulate(const CAIF_DeviceTensor &device_grad_krsc,
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

void CAIF_Conv2dBackend::SyncBiasGradDeviceToHostAccumulate(const CAIF_DeviceTensor &bias_grad_device,
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

CAIF_DeviceTensor CAIF_Conv2dBackend::Conv2DForwardDevice(const CAIF_DeviceTensor &input,
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

CAIF_DeviceTensor CAIF_Conv2dBackend::Conv2DBackwardDevice(const CAIF_DeviceTensor &grad_output,
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

CAIF_DeviceTensor CAIF_Conv2dBackend::Conv2DForwardHost(const CAIF_DeviceTensor &input,
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

CAIF_DeviceTensor CAIF_Conv2dBackend::Conv2DBackwardHost(const CAIF_DeviceTensor &grad_output,
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

}//end instance namespace
