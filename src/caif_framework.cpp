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

/**
 * @file aif_framework.cpp
 * @brief Slim framework implementation - backend management and delegation to operation classes
 */

#include "caif_framework.h"
#include "caif_tensor.h"
#include "caif_eigen_backend.h"
#include "caif_blas_backend.h"
#include "caif_cpu_tensor_data.h"
#include "caif_settings.h"
#include "caif_matrix_ops.h"
#include "caif_element_ops.h"
#include "caif_activations.h"
#include "caif_pooling.h"
#include "caif_optimizer_ops.h"
#include "caif_batch_ops.h"
#include "caif_convolution_ops.h"
#include <cstring>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif


using namespace instance;

//==============================================================================
// Static member initialization
//==============================================================================

CAIF_TensorBackend::BackendType_e CAIF_Framework::_default_backend=CAIF_TensorBackend::BackendType_e::Auto;

//==============================================================================
// Constructors
//==============================================================================

CAIF_Framework::CAIF_Framework():_backend_type(CAIF_TensorBackend::BackendType_e::Auto)
{
  try
  {
    // Priority order:
    // 1. CAIF_Settings::BackendOverride() (set by application via command line)
    // 2. _default_backend (set by SetDefaultBackend())
    // 3. Auto-select (CUDA if available, otherwise BLAS)
    const CAIF_TensorBackend::BackendType_e settings_override=CAIF_Settings::BackendOverride();
    if(settings_override!=CAIF_TensorBackend::BackendType_e::Auto)
    {
      SetBackend(settings_override);
    }
    else if(_default_backend!=CAIF_TensorBackend::BackendType_e::Auto)
    {
      SetBackend(_default_backend);
    }
    else
    {
      AutoSelectBackend();
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Framework::CAIF_Framework(const CAIF_Framework &other):CAIF_Base(other),
                                                         _backend_type(other._backend_type)
{
  try
  {
    if(other._current_backend!=nullptr)
    {
      _current_backend=CreateBackend(_backend_type);
    }
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// Backend Management
//==============================================================================

void CAIF_Framework::SetBackend(const CAIF_TensorBackend::BackendType_e backend)
{
  try
  {
    _backend_type=backend;
    _current_backend=CreateBackend(backend);
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_Framework::AutoSelectBackend()
{
  try
  {
    // GPU training uses CAIF_DeviceNetwork directly, not CAIF_Framework backends
    SetBackend(CAIF_TensorBackend::BackendType_e::BLAS);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_TensorBackend::BackendType_e CAIF_Framework::CurrentBackend()const
{
  return _backend_type;
}

bool CAIF_Framework::IsGPUAccelerated()const
{
  // GPU training uses CAIF_DeviceNetwork, not CAIF_Framework backends
  return (_backend_type==CAIF_TensorBackend::BackendType_e::CUDA);
}

std::unique_ptr<CAIF_TensorData> CAIF_Framework::CreateTensor(
                                                            const std::vector<uint32_t> &shape,
                                                            const CAIF_DataType &dtype
                                                           )
{
  try
  {
    if(_current_backend==nullptr)
    {
      THROW_CAIFE("No backend initialized");
    }
    return _current_backend->CreateTensor(shape,dtype);
  }
  CCAIF_CATCH_BLOCK()
}

std::unique_ptr<CAIF_TensorBackend> CAIF_Framework::CreateBackend(
                                                                const CAIF_TensorBackend::BackendType_e backend
                                                               )
{
  try
  {
    switch(backend)
    {
      case CAIF_TensorBackend::BackendType_e::BLAS:
        return std::make_unique<CAIF_BLASBackend>();
        
      case CAIF_TensorBackend::BackendType_e::Eigen:
        return std::make_unique<CAIF_EigenBackend>();
        
      case CAIF_TensorBackend::BackendType_e::CUDA:
        THROW_CAIFE("CUDA backend removed - use CAIF_DeviceNetwork for GPU training");

      case CAIF_TensorBackend::BackendType_e::Vulkan:
        THROW_CAIFE("Vulkan backend removed");

      case CAIF_TensorBackend::BackendType_e::Auto:
        return std::make_unique<CAIF_BLASBackend>();
        
      default:
        THROW_CAIFE("Unknown backend type");
    }
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// Backend Availability Checks
//==============================================================================

bool CAIF_Framework::IsCudaAvailable()
{
#ifdef USE_CAIF_CUDA
  int device_count=0;
  cudaError_t err=cudaGetDeviceCount(&device_count);
  return (err==cudaSuccess&&device_count>0);
#else
  return false;
#endif
}


//==============================================================================
// LEGACY INTERFACE - Matrix Operations (delegate to CAIF_MatrixOps)
//==============================================================================

CAIF_Tensor CAIF_Framework::MatrixMultiply(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    return CAIF_MatrixOps::Multiply(*this,a,b);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::MatrixMultiplyEx(
                                           const CAIF_Tensor &a,
                                           const CAIF_Tensor &b,
                                           const CAIF_TensorBackend::Transpose_e trans_a,
                                           const CAIF_TensorBackend::Transpose_e trans_b
                                          )
{
  try
  {
    return CAIF_MatrixOps::MultiplyEx(*this,a,b,trans_a,trans_b);
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_Framework::MatrixMultiply(const CAIF_TensorData &a,const CAIF_TensorData &b,CAIF_TensorData &result)
{
  try
  {
    if(_current_backend==nullptr)
    {
      THROW_CAIFE("No backend initialized");
    }
    _current_backend->MatrixMultiply(a,b,result);
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// LEGACY INTERFACE - Element-wise Operations (delegate to CAIF_ElementOps)
//==============================================================================

CAIF_Tensor CAIF_Framework::ElementwiseAdd(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    return CAIF_ElementOps::Add(a,b);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseAddScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    return CAIF_ElementOps::AddScalar(a,scalar);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseSub(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    return CAIF_ElementOps::Sub(a,b);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseSubScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    return CAIF_ElementOps::SubScalar(a,scalar);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseMul(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    return CAIF_ElementOps::Mul(a,b);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseMulScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    return CAIF_ElementOps::MulScalar(a,scalar);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseDiv(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    return CAIF_ElementOps::Div(a,b);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseDivScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    return CAIF_ElementOps::DivScalar(a,scalar);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ElementwiseSqrt(const CAIF_Tensor &a)
{
  try
  {
    return CAIF_ElementOps::Sqrt(a);
  }
  CCAIF_CATCH_BLOCK()
}

float CAIF_Framework::ReduceSum(const CAIF_Tensor &a)
{
  try
  {
    return CAIF_ElementOps::Sum(a);
  }
  CCAIF_CATCH_BLOCK()
}

float CAIF_Framework::ReduceMean(const CAIF_Tensor &a)
{
  try
  {
    return CAIF_ElementOps::Mean(a);
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// LEGACY INTERFACE - Activation Functions (delegate to activation classes)
//==============================================================================

CAIF_Tensor CAIF_Framework::ActivationForward(
                                            const CAIF_Tensor &input,
                                            const CAIF_TensorBackend::ActivationType_e activation_type
                                           )
{
  try
  {
    // Use backend when available for GPU-accelerated activation
    if(_current_backend!=nullptr)
    {
      CAIF_Tensor output(input.Framework(),input.Shape(),input.Type());
      output.EnsureBackendData();
      const_cast<CAIF_Tensor&>(input).EnsureBackendData();
      _current_backend->ActivationForward(*input.TensorData(),*output.TensorData(),activation_type);
      return output;
    }

    // Fallback to host computation when no backend
    switch(activation_type)
    {
      case CAIF_TensorBackend::ActivationType_e::ReLU:
        return CAIF_ReLU::Forward(input);
      case CAIF_TensorBackend::ActivationType_e::Sigmoid:
        return CAIF_Sigmoid::Forward(input);
      case CAIF_TensorBackend::ActivationType_e::Tanh:
        return CAIF_Tanh::Forward(input);
      case CAIF_TensorBackend::ActivationType_e::LeakyReLU:
        return CAIF_LeakyReLU::Forward(input);
      case CAIF_TensorBackend::ActivationType_e::ELU:
        return CAIF_ELU::Forward(input);
      case CAIF_TensorBackend::ActivationType_e::GELU:
        return CAIF_GELU::Forward(input);
      case CAIF_TensorBackend::ActivationType_e::Swish:
        return CAIF_Swish::Forward(input);
      case CAIF_TensorBackend::ActivationType_e::Identity:
      default:
        return input;
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::ActivationBackward(
                                             const CAIF_Tensor &grad_output,
                                             const CAIF_Tensor &input,
                                             const CAIF_Tensor &output,
                                             const CAIF_TensorBackend::ActivationType_e activation_type
                                            )
{
  try
  {
    // Use backend when available for GPU-accelerated activation backward
    if(_current_backend!=nullptr)
    {
      CAIF_Tensor grad_input(grad_output.Framework(),grad_output.Shape(),grad_output.Type());
      grad_input.EnsureBackendData();
      const_cast<CAIF_Tensor&>(grad_output).EnsureBackendData();
      const_cast<CAIF_Tensor&>(input).EnsureBackendData();
      const_cast<CAIF_Tensor&>(output).EnsureBackendData();
      _current_backend->ActivationBackward(
                                           *grad_output.TensorData(),
                                           *input.TensorData(),
                                           *output.TensorData(),
                                           *grad_input.TensorData(),
                                           activation_type
                                          );
      return grad_input;
    }

    // Fallback to host computation when no backend
    switch(activation_type)
    {
      case CAIF_TensorBackend::ActivationType_e::ReLU:
        return CAIF_ReLU::Backward(input,grad_output);
      case CAIF_TensorBackend::ActivationType_e::Sigmoid:
        return CAIF_Sigmoid::Backward(output,grad_output);
      case CAIF_TensorBackend::ActivationType_e::Tanh:
        return CAIF_Tanh::Backward(output,grad_output);
      case CAIF_TensorBackend::ActivationType_e::LeakyReLU:
        return CAIF_LeakyReLU::Backward(input,grad_output);
      case CAIF_TensorBackend::ActivationType_e::ELU:
        return CAIF_ELU::Backward(input,output,grad_output);
      case CAIF_TensorBackend::ActivationType_e::GELU:
        return CAIF_GELU::Backward(input,grad_output);
      case CAIF_TensorBackend::ActivationType_e::Swish:
        return CAIF_Swish::Backward(input,output,grad_output);
      case CAIF_TensorBackend::ActivationType_e::Identity:
      default:
        return grad_output;
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::SoftmaxForward(const CAIF_Tensor &input)
{
  try
  {
    return CAIF_Softmax::Forward(input);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::SoftmaxBackward(const CAIF_Tensor &grad_output,const CAIF_Tensor &output)
{
  try
  {
    return CAIF_Softmax::Backward(output,grad_output);
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// LEGACY INTERFACE - Pooling Operations (delegate to CAIF_Pooling)
//==============================================================================

CAIF_Tensor CAIF_Framework::MaxPooling2D(
                                       const CAIF_Tensor &input,
                                       const uint32_t pool_size,
                                       const uint32_t stride,
                                       const uint32_t padding,
                                       CAIF_Tensor *indices
                                      )
{
  try
  {
    return CAIF_Pooling::MaxPool2D(input,pool_size,stride,padding,indices);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::AveragePooling2D(
                                           const CAIF_Tensor &input,
                                           const uint32_t pool_size,
                                           const uint32_t stride,
                                           const uint32_t padding
                                          )
{
  try
  {
    return CAIF_Pooling::AvgPool2D(input,pool_size,stride,padding);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::MaxPooling2DBackward(
                                               const CAIF_Tensor &grad_output,
                                               const CAIF_Tensor &indices,
                                               const CAIF_Tensor &input,
                                               const uint32_t pool_size,
                                               const uint32_t stride,
                                               const uint32_t padding
                                              )
{
  try
  {
    return CAIF_Pooling::MaxPool2DBackward(grad_output,indices,input,pool_size,stride,padding);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::AveragePooling2DBackward(
                                                   const CAIF_Tensor &grad_output,
                                                   const std::vector<uint32_t> &input_shape,
                                                   const uint32_t pool_size,
                                                   const uint32_t stride,
                                                   const uint32_t padding
                                                  )
{
  try
  {
    return CAIF_Pooling::AvgPool2DBackward(grad_output,input_shape,pool_size,stride,padding);
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// LEGACY INTERFACE - Optimizer Operations (delegate to CAIF_OptimizerOps)
//==============================================================================

void CAIF_Framework::FusedAdamUpdate(
                                    CAIF_Tensor &param,
                                    const CAIF_Tensor &grad,
                                    CAIF_Tensor &m,
                                    CAIF_Tensor &v,
                                    const float lr,
                                    const float beta1,
                                    const float beta2,
                                    const float epsilon,
                                    const float weight_decay,
                                    const float bias_correction1,
                                    const float bias_correction2
                                   )
{
  try
  {
    if(_current_backend!=nullptr)
    {
      param.EnsureBackendData();
      const_cast<CAIF_Tensor&>(grad).EnsureBackendData();
      m.EnsureBackendData();
      v.EnsureBackendData();
      _current_backend->FusedAdamUpdate(
                                        *param.TensorData(),
                                        *grad.TensorData(),
                                        *m.TensorData(),
                                        *v.TensorData(),
                                        lr,
                                        beta1,
                                        beta2,
                                        epsilon,
                                        weight_decay,
                                        bias_correction1,
                                        bias_correction2
                                       );
      return;
    }
    CAIF_OptimizerOps::AdamUpdate(
                                 param,
                                 grad,
                                 m,
                                 v,
                                 lr,
                                 beta1,
                                 beta2,
                                 epsilon,
                                 weight_decay,
                                 bias_correction1,
                                 bias_correction2
                                );
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_Framework::FusedSGDMomentumUpdate(
                                           CAIF_Tensor &param,
                                           const CAIF_Tensor &grad,
                                           CAIF_Tensor &velocity,
                                           const float lr,
                                           const float momentum,
                                           const float weight_decay
                                          )
{
  try
  {
    if(_current_backend!=nullptr)
    {
      param.EnsureBackendData();
      const_cast<CAIF_Tensor&>(grad).EnsureBackendData();
      velocity.EnsureBackendData();
      _current_backend->FusedSGDMomentumUpdate(
                                               *param.TensorData(),
                                               *grad.TensorData(),
                                               *velocity.TensorData(),
                                               lr,
                                               momentum,
                                               weight_decay
                                              );
      return;
    }
    CAIF_OptimizerOps::SGDMomentumUpdate(
                                        param,
                                        grad,
                                        velocity,
                                        lr,
                                        momentum,
                                        weight_decay
                                       );
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// LEGACY INTERFACE - Convolution (delegate to CAIF_ConvolutionOps)
//==============================================================================

CAIF_Tensor CAIF_Framework::Convolution2D(
                                        const CAIF_Tensor &input,
                                        const CAIF_Tensor &kernel,
                                        const uint32_t stride_y,
                                        const uint32_t stride_x,
                                        const uint32_t padding_y,
                                        const uint32_t padding_x
                                       )
{
  try
  {
    return CAIF_ConvolutionOps::Conv2DForward(*this,input,kernel,stride_y,stride_x,padding_y,padding_x);
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_Framework::Convolution2D(
                                  const CAIF_TensorData &input,
                                  const CAIF_TensorData &kernel,
                                  CAIF_TensorData &output,
                                  const CAIF_TensorBackend::ConvolutionParams &params
                                 )
{
  try
  {
    if(_current_backend==nullptr)
    {
      THROW_CAIFE("No backend initialized");
    }
    _current_backend->Convolution2D(input,kernel,output,params);
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// LEGACY INTERFACE - Batch Operations (delegate to CAIF_BatchNorm/CAIF_Dropout)
//==============================================================================

CAIF_Tensor CAIF_Framework::BatchNormForward(
                                           const CAIF_Tensor &input,
                                           const CAIF_Tensor &scale,
                                           const CAIF_Tensor &bias,
                                           CAIF_Tensor &running_mean,
                                           CAIF_Tensor &running_var,
                                           const float epsilon,
                                           const float momentum,
                                           const bool training,
                                           CAIF_Tensor &saved_mean,
                                           CAIF_Tensor &saved_inv_var
                                          )
{
  try
  {
    return CAIF_BatchNorm::Forward(input,scale,bias,running_mean,running_var,epsilon,momentum,training,
                                  saved_mean,saved_inv_var);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::BatchNormBackward(
                                            const CAIF_Tensor &grad_output,
                                            const CAIF_Tensor &input,
                                            const CAIF_Tensor &scale,
                                            const CAIF_Tensor &saved_mean,
                                            const CAIF_Tensor &saved_inv_var,
                                            const float epsilon,
                                            CAIF_Tensor &grad_scale,
                                            CAIF_Tensor &grad_bias
                                           )
{
  try
  {
    return CAIF_BatchNorm::Backward(grad_output,input,scale,saved_mean,saved_inv_var,epsilon,
                                   grad_scale,grad_bias);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::DropoutForward(
                                         const CAIF_Tensor &input,
                                         const float dropout_rate,
                                         const bool training,
                                         CAIF_Tensor &mask
                                        )
{
  try
  {
    return CAIF_Dropout::Forward(input,dropout_rate,training,mask);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Framework::DropoutBackward(
                                          const CAIF_Tensor &grad_output,
                                          const CAIF_Tensor &mask,
                                          const float dropout_rate
                                         )
{
  try
  {
    return CAIF_Dropout::Backward(grad_output,mask,dropout_rate);
  }
  CCAIF_CATCH_BLOCK()
}
