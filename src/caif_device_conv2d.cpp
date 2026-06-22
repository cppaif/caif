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
#include "caif_conv2d_backend.h"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_cudnn_util.h"
#include "caif_device_context.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
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

template<typename ComputeT,typename StorageT>
CAIF_DeviceConv2D<ComputeT,StorageT>::CAIF_DeviceConv2D(const CAIF_DeviceConv2DConfig &config,
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
    if(config.InChannels()==0||config.OutChannels()==0)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: channel dims must be > 0");
    }
    if(config.KernelHeight()==0||config.KernelWidth()==0)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: kernel dims must be > 0");
    }
    if(config.StrideHeight()==0||config.StrideWidth()==0)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: stride dims must be > 0");
    }

    const std::vector<uint32_t> w_shape={config.KernelHeight(),
                                         config.KernelWidth(),
                                         config.InChannels(),
                                         config.OutChannels()};
    const std::vector<uint32_t> b_shape={config.OutChannels()};
    SetWeights(CAIF_DeviceTensor::ZerosHost(w_shape));
    SetBias(CAIF_DeviceTensor::ZerosHost(b_shape));
    SetWeightsGrad(CAIF_DeviceTensor::ZerosHost(w_shape));
    SetBiasGrad(CAIF_DeviceTensor::ZerosHost(b_shape));

    const size_t fan_in=static_cast<size_t>(config.KernelHeight())*
                         config.KernelWidth()*config.InChannels();
    const size_t fan_out=static_cast<size_t>(config.KernelHeight())*
                          config.KernelWidth()*config.OutChannels();
    const float limit=std::sqrt(g_caif_xavier_uniform_scale/static_cast<float>(fan_in+fan_out));
    const size_t total=fan_in*config.OutChannels();
    // fp32 by contract
    float *w=static_cast<float *>(Weights().DeviceDataRaw());
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
                                _config(other.Config()),
                                _weights(std::move(other.Weights())),
                                _bias(std::move(other.Bias())),
                                _weights_grad(std::move(other.WeightsGrad())),
                                _bias_grad(std::move(other.BiasGrad())),
                                _weights_device_krsc(std::move(other.WeightsDeviceKRSC())),
                                _bias_device(std::move(other.BiasDevice())),
                                _weights_grad_device_krsc(std::move(other.WeightsGradDeviceKRSC())),
                                _bias_grad_device(std::move(other.BiasGradDevice())),
                                _cached_input_shape(std::move(other.CachedInputShape())),
                                _cached_input_host(std::move(other.CachedInputHost())),
                                _cached_input_device(std::move(other.CachedInputDevice()))
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
      SetConfig(other.Config());
      SetWeights(std::move(other.Weights()));
      SetBias(std::move(other.Bias()));
      SetWeightsGrad(std::move(other.WeightsGrad()));
      SetBiasGrad(std::move(other.BiasGrad()));
      SetWeightsDeviceKRSC(std::move(other.WeightsDeviceKRSC()));
      SetBiasDevice(std::move(other.BiasDevice()));
      SetWeightsGradDeviceKRSC(std::move(other.WeightsGradDeviceKRSC()));
      SetBiasGradDevice(std::move(other.BiasGradDevice()));
      SetCachedInputShape(std::move(other.CachedInputShape()));
      SetCachedInputHost(std::move(other.CachedInputHost()));
      SetCachedInputDevice(std::move(other.CachedInputDevice()));
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
    if(in_shape.size()!=g_caif_conv_dimensions)
    {
      THROW_CAIFE("CAIF_DeviceConv2D: expects rank-4 input [N,H,W,Cin]");
    }
    const uint32_t Kh=Config().KernelHeight();
    const uint32_t Kw=Config().KernelWidth();
    const uint32_t Sh=Config().StrideHeight();
    const uint32_t Sw=Config().StrideWidth();
    const uint32_t Cin=Config().InChannels();
    const uint32_t Cout=Config().OutChannels();

    if(input.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      SetCachedInputDevice(CAIF_DeviceTensor());
      std::vector<uint32_t> shape_out;
      std::vector<float> cached_in_host;
      CAIF_DeviceTensor result=CAIF_Conv2dBackend::Conv2DForwardHost(input,
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
    CAIF_Conv2dBackend::SyncWeightsHostToDevice(Weights(),Stream(),StorageDtype(),Kh,Kw,Cin,Cout,weights_dev);
    CAIF_Conv2dBackend::SyncBiasHostToDevice(Bias(),Stream(),StorageDtype(),bias_dev);
    SetWeightsDeviceKRSC(std::move(weights_dev));
    SetBiasDevice(std::move(bias_dev));

    CAIF_DeviceTensor cached_in_dev;
    CAIF_DeviceTensor result=CAIF_Conv2dBackend::Conv2DForwardDevice(input,
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
    const uint32_t Kh=Config().KernelHeight();
    const uint32_t Kw=Config().KernelWidth();
    const uint32_t Sh=Config().StrideHeight();
    const uint32_t Sw=Config().StrideWidth();
    const uint32_t Cin=Config().InChannels();
    const uint32_t Cout=Config().OutChannels();

    if(grad_output.Location()==CAIF_DeviceTensor::Location_e::Host_e)
    {
      return CAIF_Conv2dBackend::Conv2DBackwardHost(grad_output,
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
    CAIF_DeviceTensor grad_input=CAIF_Conv2dBackend::Conv2DBackwardDevice(grad_output,
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
    CAIF_Conv2dBackend::SyncWeightsGradDeviceToHostAccumulate(wgrad_dev,
                                                              Stream(),
                                                              StorageDtype(),
                                                              Kh,
                                                              Kw,
                                                              Cin,
                                                              Cout,
                                                              WeightsGrad());
    CAIF_Conv2dBackend::SyncBiasGradDeviceToHostAccumulate(bgrad_dev,Stream(),StorageDtype(),BiasGrad());
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
  return g_serial_tag_conv2d;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceConv2D<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
  std::vector<std::string> names;
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::GenericWeight_e));
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::GenericBias_e));
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
