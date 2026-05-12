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
// 2D Max pooling (templated). Two paths:
//   - Host_e fp32: existing CPU loop with argmax cache.
//   - Device_e fp32 / fp16 / bf16: cuDNN device backend.
//------------------------------------------------------------------------------

#include "caif_device_max_pooling2d.h"
#include "caif_constants.h"
#include "caif_cudnn_util.h"
#include "caif_device_context.h"
#include "caif_exception.h"
#include <limits>
#include <vector>

#ifdef USE_CAIF_CUDA
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{


constexpr size_t POOLING_INPUT_RANK=4;

#ifdef USE_CAIF_CUDA

CAIF_DeviceTensor MaxPool2DForwardDevice(const CAIF_DeviceTensor &input,
                                         CAIF_CudaStream &stream,
                                         uint32_t pH,
                                         uint32_t pW,
                                         uint32_t sH,
                                         uint32_t sW,
                                         CAIF_DataType::CAIF_DataType_e storage_dtype,
                                         CAIF_DeviceTensor &cached_input_out,
                                         CAIF_DeviceTensor &cached_output_out)
{
  const std::vector<uint32_t> &in_shape=input.Shape();
  const uint32_t N=in_shape[0];
  const uint32_t H=in_shape[1];
  const uint32_t W=in_shape[2];
  const uint32_t C=in_shape[3];
  if(H<pH||W<pW)
  {
    THROW_CAIFE("CAIF_DeviceMaxPooling2D: window larger than input");
  }
  const uint32_t H_out=(H-pH)/sH+1;
  const uint32_t W_out=(W-pW)/sW+1;

  CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized({N,H_out,W_out,C},
                                                            stream,
                                                            storage_dtype);

  cudnnHandle_t handle=CAIF_DeviceContext::Instance().CudnnHandle();
  CAIF_DeviceContext::Instance().SetCudnnStream(stream.Handle());

  const cudnnDataType_t cudnn_dt=CAIF_CudnnUtil::CudnnDtypeFromStorage(storage_dtype);

  cudnnTensorDescriptor_t in_desc=nullptr;
  cudnnTensorDescriptor_t out_desc=nullptr;
  cudnnPoolingDescriptor_t pool_desc=nullptr;
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&in_desc),"create in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&out_desc),"create out_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreatePoolingDescriptor(&pool_desc),"create pool_desc");

  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(in_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(C),
                                                        static_cast<int>(H),
                                                        static_cast<int>(W)),
                             "set in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(out_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(C),
                                                        static_cast<int>(H_out),
                                                        static_cast<int>(W_out)),
                             "set out_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetPooling2dDescriptor(pool_desc,
                                                         CUDNN_POOLING_MAX_DETERMINISTIC,
                                                         CUDNN_PROPAGATE_NAN,
                                                         static_cast<int>(pH),
                                                         static_cast<int>(pW),
                                                         0,
                                                         0,
                                                         static_cast<int>(sH),
                                                         static_cast<int>(sW)),
                             "set pool_desc");

  const float alpha=1.0f;
  const float beta=0.0f;
  CAIF_CudnnUtil::CheckCudnn(cudnnPoolingForward(handle,
                                                 pool_desc,
                                                 &alpha,
                                                 in_desc,
                                                 input.DeviceDataRaw(),
                                                 &beta,
                                                 out_desc,
                                                 output.DeviceDataRaw()),
                             "PoolingForward");

  cudnnDestroyPoolingDescriptor(pool_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(in_desc);

  cached_input_out=input.Clone();
  cached_output_out=output.Clone();
  return output;
}

CAIF_DeviceTensor MaxPool2DBackwardDevice(const CAIF_DeviceTensor &grad_output,
                                          CAIF_CudaStream &stream,
                                          const CAIF_DeviceTensor &cached_input,
                                          const CAIF_DeviceTensor &cached_output,
                                          uint32_t pH,
                                          uint32_t pW,
                                          uint32_t sH,
                                          uint32_t sW,
                                          CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  const std::vector<uint32_t> &in_shape=cached_input.Shape();
  const uint32_t N=in_shape[0];
  const uint32_t H=in_shape[1];
  const uint32_t W=in_shape[2];
  const uint32_t C=in_shape[3];
  const std::vector<uint32_t> &out_shape=grad_output.Shape();
  const uint32_t H_out=out_shape[1];
  const uint32_t W_out=out_shape[2];

  CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized(in_shape,
                                                                stream,
                                                                storage_dtype);

  cudnnHandle_t handle=CAIF_DeviceContext::Instance().CudnnHandle();
  CAIF_DeviceContext::Instance().SetCudnnStream(stream.Handle());

  const cudnnDataType_t cudnn_dt=CAIF_CudnnUtil::CudnnDtypeFromStorage(storage_dtype);

  cudnnTensorDescriptor_t in_desc=nullptr;
  cudnnTensorDescriptor_t out_desc=nullptr;
  cudnnPoolingDescriptor_t pool_desc=nullptr;
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&in_desc),"create in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreateTensorDescriptor(&out_desc),"create out_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnCreatePoolingDescriptor(&pool_desc),"create pool_desc");

  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(in_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(C),
                                                        static_cast<int>(H),
                                                        static_cast<int>(W)),
                             "set in_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetTensor4dDescriptor(out_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        cudnn_dt,
                                                        static_cast<int>(N),
                                                        static_cast<int>(C),
                                                        static_cast<int>(H_out),
                                                        static_cast<int>(W_out)),
                             "set out_desc");
  CAIF_CudnnUtil::CheckCudnn(cudnnSetPooling2dDescriptor(pool_desc,
                                                         CUDNN_POOLING_MAX_DETERMINISTIC,
                                                         CUDNN_PROPAGATE_NAN,
                                                         static_cast<int>(pH),
                                                         static_cast<int>(pW),
                                                         0,
                                                         0,
                                                         static_cast<int>(sH),
                                                         static_cast<int>(sW)),
                             "set pool_desc");

  const float alpha=1.0f;
  const float beta=0.0f;
  CAIF_CudnnUtil::CheckCudnn(cudnnPoolingBackward(handle,
                                                  pool_desc,
                                                  &alpha,
                                                  out_desc,
                                                  cached_output.DeviceDataRaw(),
                                                  out_desc,
                                                  grad_output.DeviceDataRaw(),
                                                  in_desc,
                                                  cached_input.DeviceDataRaw(),
                                                  &beta,
                                                  in_desc,
                                                  grad_input.DeviceDataRaw()),
                             "PoolingBackward");

  cudnnDestroyPoolingDescriptor(pool_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(in_desc);

  return grad_input;
}

#endif // USE_CAIF_CUDA

CAIF_DeviceTensor MaxPool2DForwardHost(const CAIF_DeviceTensor &input,
                                       uint32_t pH,
                                       uint32_t pW,
                                       uint32_t sH,
                                       uint32_t sW,
                                       std::vector<uint32_t> &cached_input_shape_out,
                                       std::vector<int32_t> &cached_max_indices_out)
{
  // host path is fp32-only by Stage 5b contract (CPU pooling backend
  // operates on fp32 master tensors; non-fp32 storage stays on device).
  if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CAIF_DeviceMaxPooling2D host path: only fp32 supported on host_e");
  }
  // fp32 host path: input/output are Float32 by the throw above; per-site
  // markers on each cast below.
  const std::vector<uint32_t> &in_shape=input.Shape();
  const uint32_t N=in_shape[0];
  const uint32_t H=in_shape[1];
  const uint32_t W=in_shape[2];
  const uint32_t C=in_shape[3];
  if(H<pH||W<pW)
  {
    THROW_CAIFE("CAIF_DeviceMaxPooling2D: window larger than input");
  }
  const uint32_t H_out=(H-pH)/sH+1;
  const uint32_t W_out=(W-pW)/sW+1;

  cached_input_shape_out=in_shape;
  const size_t out_elems=static_cast<size_t>(N)*H_out*W_out*C;
  cached_max_indices_out.assign(out_elems,-1);

  CAIF_DeviceTensor output=CAIF_DeviceTensor::ZerosHost({N,H_out,W_out,C});
  // fp32 host path
  const float *x=static_cast<const float *>(input.DeviceDataRaw());
  // fp32 host path
  float *y=static_cast<float *>(output.DeviceDataRaw());

  for(uint32_t n=0;n<N;++n)
  {
    for(uint32_t oh=0;oh<H_out;++oh)
    {
      for(uint32_t ow=0;ow<W_out;++ow)
      {
        for(uint32_t c=0;c<C;++c)
        {
          float max_val=-std::numeric_limits<float>::infinity();
          int32_t max_idx=-1;
          for(uint32_t kh=0;kh<pH;++kh)
          {
            for(uint32_t kw=0;kw<pW;++kw)
            {
              const uint32_t ih=oh*sH+kh;
              const uint32_t iw=ow*sW+kw;
              const size_t in_idx=((static_cast<size_t>(n)*H+ih)*W+iw)*C+c;
              const float v=x[in_idx];
              if(v>max_val)
              {
                max_val=v;
                max_idx=static_cast<int32_t>(in_idx);
              }
            }
          }
          const size_t out_idx=((static_cast<size_t>(n)*H_out+oh)*W_out+ow)*C+c;
          y[out_idx]=max_val;
          cached_max_indices_out[out_idx]=max_idx;
        }
      }
    }
  }
  return output;
}

CAIF_DeviceTensor MaxPool2DBackwardHost(const CAIF_DeviceTensor &grad_output,
                                        const std::vector<uint32_t> &cached_input_shape,
                                        const std::vector<int32_t> &cached_max_indices)
{
  // host path is fp32-only by Stage 5b contract.
  if(grad_output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("CAIF_DeviceMaxPooling2D host path: only fp32 supported on host_e");
  }
  // fp32 host path: grad_output is Float32 by the throw above; per-site
  // markers on each cast below.
  if(cached_input_shape.empty()==true)
  {
    THROW_CAIFE("CAIF_DeviceMaxPooling2D: backward called before forward");
  }
  CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::ZerosHost(cached_input_shape);
  // fp32 host path
  float *g_in=static_cast<float *>(grad_input.DeviceDataRaw());
  // fp32 host path
  const float *g_out=static_cast<const float *>(grad_output.DeviceDataRaw());
  const size_t out_elems=grad_output.TotalElements();
  if(out_elems!=cached_max_indices.size())
  {
    THROW_CAIFE("CAIF_DeviceMaxPooling2D: backward size mismatch");
  }
  for(size_t i=0;i<out_elems;++i)
  {
    const int32_t idx=cached_max_indices[i];
    if(idx>=0)
    {
      g_in[idx]+=g_out[i];
    }
  }
  return grad_input;
}


template<typename ComputeT,typename StorageT>
CAIF_DeviceMaxPooling2D<ComputeT,StorageT>::CAIF_DeviceMaxPooling2D(const Config_t &config,
                                                                    CAIF_CudaStream &stream):
                                                CAIF_DevicePooling2D<ComputeT,StorageT>(config,stream),
                                                _cached_max_indices()
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMaxPooling2D<ComputeT,StorageT>::CAIF_DeviceMaxPooling2D(CAIF_DeviceMaxPooling2D &&other):
                                CAIF_DevicePooling2D<ComputeT,StorageT>(std::move(other)),
                                _cached_max_indices(std::move(other._cached_max_indices))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMaxPooling2D<ComputeT,StorageT> &
CAIF_DeviceMaxPooling2D<ComputeT,StorageT>::operator=(CAIF_DeviceMaxPooling2D &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DevicePooling2D<ComputeT,StorageT>::operator=(std::move(other));
      _cached_max_indices=std::move(other._cached_max_indices);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK();
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceMaxPooling2D<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                         CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    const std::vector<uint32_t> &in_shape=input.Shape();
    if(in_shape.size()!=POOLING_INPUT_RANK)
    {
      THROW_CAIFE("CAIF_DeviceMaxPooling2D: expects rank-4 input [N,H,W,C]");
    }
    const uint32_t pH=Config().pool_height;
    const uint32_t pW=Config().pool_width;
    const uint32_t sH=Config().stride_height;
    const uint32_t sW=Config().stride_width;

    if(input.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      SetCachedInput(CAIF_DeviceTensor());
      SetCachedOutput(CAIF_DeviceTensor());
      std::vector<uint32_t> shape_out;
      std::vector<int32_t> indices_out;
      CAIF_DeviceTensor result=MaxPool2DForwardHost(input,pH,pW,sH,sW,shape_out,indices_out);
      SetCachedInputShape(shape_out);
      SetCachedMaxIndices(std::move(indices_out));
      return result;
    }
#ifdef USE_CAIF_CUDA
    SetCachedInputShape(in_shape);
    SetCachedMaxIndices(std::vector<int32_t>());
    CAIF_DeviceTensor cached_in;
    CAIF_DeviceTensor cached_out;
    CAIF_DeviceTensor result=MaxPool2DForwardDevice(input,
                                                    Stream(),
                                                    pH,
                                                    pW,
                                                    sH,
                                                    sW,
                                                    StorageDtype(),
                                                    cached_in,
                                                    cached_out);
    SetCachedInput(std::move(cached_in));
    SetCachedOutput(std::move(cached_out));
    return result;
#else
    THROW_CAIFE("CAIF_DeviceMaxPooling2D: device path requires USE_CAIF_CUDA");
#endif
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceMaxPooling2D<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                          CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    const uint32_t pH=Config().pool_height;
    const uint32_t pW=Config().pool_width;
    const uint32_t sH=Config().stride_height;
    const uint32_t sW=Config().stride_width;

    if(grad_output.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      return MaxPool2DBackwardHost(grad_output,CachedInputShape(),CachedMaxIndices());
    }
#ifdef USE_CAIF_CUDA
    if(CachedInput().IsAllocated()==false||CachedOutput().IsAllocated()==false)
    {
      THROW_CAIFE("CAIF_DeviceMaxPooling2D: backward called before forward (device path)");
    }
    return MaxPool2DBackwardDevice(grad_output,
                                   Stream(),
                                   CachedInput(),
                                   CachedOutput(),
                                   pH,
                                   pW,
                                   sH,
                                   sW,
                                   StorageDtype());
#else
    THROW_CAIFE("CAIF_DeviceMaxPooling2D: device path requires USE_CAIF_CUDA");
#endif
  }
  CAIF_CATCH_BLOCK();
  return CAIF_DeviceTensor();
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceMaxPooling2D<ComputeT,StorageT>::Description()const
{
  return g_caif_description_max_pooling2d;
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceMaxPooling2D<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceMaxPooling2D<float,__half>;
template class CAIF_DeviceMaxPooling2D<float,__nv_bfloat16>;
template class CAIF_DeviceMaxPooling2D<__half,float>;
template class CAIF_DeviceMaxPooling2D<__half,__half>;
template class CAIF_DeviceMaxPooling2D<__half,__nv_bfloat16>;
template class CAIF_DeviceMaxPooling2D<__nv_bfloat16,float>;
template class CAIF_DeviceMaxPooling2D<__nv_bfloat16,__half>;
template class CAIF_DeviceMaxPooling2D<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
