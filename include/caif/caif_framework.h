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
 * @file aif_framework.h
 * @brief Slim framework class - backend management only
 * 
 * All tensor operations have been moved to specialized static operation classes:
 * - CAIF_MatrixOps: Matrix multiplication operations
 * - CAIF_ElementOps: Element-wise operations
 * - CAIF_Pooling: Pooling operations
 * - CAIF_BatchOps: Batch normalization and dropout
 * - CAIF_Convolution: Convolution operations
 * - CAIF_Softmax: Softmax operations
 * - CAIF_ReLU, CAIF_Sigmoid, etc.: Individual activation functions
 * - CAIF_OptimizerOps: Fused optimizer updates
 */

#ifndef CAIF_FRAMEWORK_H
#define CAIF_FRAMEWORK_H

#include "caif_tensor_backend.h"
#include "caif_constants.h"
#include "caif_base.h"
#include "caif_exception.h"
#include "caif_error.h"
#include <memory>
#include <vector>
#include <string>

namespace instance
{

// Forward declaration
class CAIF_Tensor;

/**
 * @brief Slim framework class for backend management only
 * 
 * This class manages backend selection and provides access to the current backend.
 * All tensor operations are handled by specialized static operation classes.
 */
class CAIF_Framework:public CAIF_Base
{
  public:
    CAIF_Framework();
    CAIF_Framework(const CAIF_Framework &other);
    ~CAIF_Framework()=default;
    
    //--------------------------------------------------------------------------
    // Backend Management
    //--------------------------------------------------------------------------
    
    /**
     * @brief Set the computation backend explicitly
     */
    void SetBackend(const CAIF_TensorBackend::BackendType_e backend);
    
    /**
     * @brief Automatically select the best available backend
     */
    void AutoSelectBackend();
    
    /**
     * @brief Get the currently active backend type
     */
    CAIF_TensorBackend::BackendType_e CurrentBackend()const;
    
    /**
     * @brief Check if current backend uses GPU acceleration
     */
    bool IsGPUAccelerated()const;
    
    /**
     * @brief Get raw pointer to current backend (for operation classes)
     */
    CAIF_TensorBackend *Backend()const{return _current_backend.get();}
    
    //--------------------------------------------------------------------------
    // Tensor Creation
    //--------------------------------------------------------------------------
    
    /**
     * @brief Create a new tensor data object with current backend
     */
    std::unique_ptr<CAIF_TensorData> CreateTensor(
                                                 const std::vector<uint32_t> &shape,
                                                 const CAIF_DataType &dtype
                                                );
    
    //--------------------------------------------------------------------------
    // Backend Availability Checks
    //--------------------------------------------------------------------------
    
    static bool IsCudaAvailable();
    static bool IsEigenAvailable(){return true;}
    static bool IsBLASAvailable(){return true;}
    
    //--------------------------------------------------------------------------
    // Default Backend
    //--------------------------------------------------------------------------
    
    static void SetDefaultBackend(const CAIF_TensorBackend::BackendType_e backend){_default_backend=backend;}
    static CAIF_TensorBackend::BackendType_e DefaultBackend(){return _default_backend;}
    
    //==========================================================================
    // LEGACY INTERFACE - Delegates to operation classes
    // These methods exist for backward compatibility during transition
    //==========================================================================
    
    // Matrix operations - delegate to CAIF_MatrixOps
    CAIF_Tensor MatrixMultiply(const CAIF_Tensor &a,const CAIF_Tensor &b);
    CAIF_Tensor MatrixMultiplyEx(const CAIF_Tensor &a,const CAIF_Tensor &b,
                                const CAIF_TensorBackend::Transpose_e trans_a,
                                const CAIF_TensorBackend::Transpose_e trans_b);
    
    // Convolution - delegate to CAIF_Convolution
    CAIF_Tensor Convolution2D(const CAIF_Tensor &input,const CAIF_Tensor &kernel,
                             const uint32_t stride_y,const uint32_t stride_x,
                             const uint32_t padding_y=0,const uint32_t padding_x=0);
    
    // Pooling - delegate to CAIF_Pooling
    CAIF_Tensor MaxPooling2D(const CAIF_Tensor &input,const uint32_t pool_size,
                            const uint32_t stride,const uint32_t padding,
                            CAIF_Tensor *indices=nullptr);
    CAIF_Tensor AveragePooling2D(const CAIF_Tensor &input,const uint32_t pool_size,
                                const uint32_t stride,const uint32_t padding);
    CAIF_Tensor MaxPooling2DBackward(const CAIF_Tensor &grad_output,const CAIF_Tensor &indices,
                                    const CAIF_Tensor &input,const uint32_t pool_size,
                                    const uint32_t stride,const uint32_t padding);
    CAIF_Tensor AveragePooling2DBackward(const CAIF_Tensor &grad_output,
                                        const std::vector<uint32_t> &input_shape,
                                        const uint32_t pool_size,const uint32_t stride,
                                        const uint32_t padding);
    
    // Batch operations - delegate to CAIF_BatchOps
    CAIF_Tensor BatchNormForward(const CAIF_Tensor &input,const CAIF_Tensor &scale,
                                const CAIF_Tensor &bias,CAIF_Tensor &running_mean,
                                CAIF_Tensor &running_var,const float epsilon,
                                const float momentum,const bool training,
                                CAIF_Tensor &saved_mean,CAIF_Tensor &saved_inv_var);
    CAIF_Tensor BatchNormBackward(const CAIF_Tensor &grad_output,const CAIF_Tensor &input,
                                 const CAIF_Tensor &scale,const CAIF_Tensor &saved_mean,
                                 const CAIF_Tensor &saved_inv_var,const float epsilon,
                                 CAIF_Tensor &grad_scale,CAIF_Tensor &grad_bias);
    CAIF_Tensor DropoutForward(const CAIF_Tensor &input,const float dropout_rate,
                              const bool training,CAIF_Tensor &mask);
    CAIF_Tensor DropoutBackward(const CAIF_Tensor &grad_output,const CAIF_Tensor &mask,
                               const float dropout_rate);
    
    // Activation - delegate to activation classes
    CAIF_Tensor ActivationForward(const CAIF_Tensor &input,
                                 const CAIF_TensorBackend::ActivationType_e activation_type);
    CAIF_Tensor ActivationBackward(const CAIF_Tensor &grad_output,const CAIF_Tensor &input,
                                  const CAIF_Tensor &output,
                                  const CAIF_TensorBackend::ActivationType_e activation_type);
    
    // Softmax - delegate to CAIF_Softmax
    CAIF_Tensor SoftmaxForward(const CAIF_Tensor &input);
    CAIF_Tensor SoftmaxBackward(const CAIF_Tensor &grad_output,const CAIF_Tensor &output);
    
    // Element-wise - delegate to CAIF_ElementOps
    CAIF_Tensor ElementwiseAdd(const CAIF_Tensor &a,const CAIF_Tensor &b);
    CAIF_Tensor ElementwiseAddScalar(const CAIF_Tensor &a,const float scalar);
    CAIF_Tensor ElementwiseSub(const CAIF_Tensor &a,const CAIF_Tensor &b);
    CAIF_Tensor ElementwiseSubScalar(const CAIF_Tensor &a,const float scalar);
    CAIF_Tensor ElementwiseMul(const CAIF_Tensor &a,const CAIF_Tensor &b);
    CAIF_Tensor ElementwiseMulScalar(const CAIF_Tensor &a,const float scalar);
    CAIF_Tensor ElementwiseDiv(const CAIF_Tensor &a,const CAIF_Tensor &b);
    CAIF_Tensor ElementwiseDivScalar(const CAIF_Tensor &a,const float scalar);
    CAIF_Tensor ElementwiseSqrt(const CAIF_Tensor &a);
    float ReduceSum(const CAIF_Tensor &a);
    float ReduceMean(const CAIF_Tensor &a);
    
    // Optimizer - delegate to CAIF_OptimizerOps
    void FusedAdamUpdate(CAIF_Tensor &param,const CAIF_Tensor &grad,CAIF_Tensor &m,CAIF_Tensor &v,
                         const float lr,const float beta1,const float beta2,const float epsilon,
                         const float weight_decay,const float bias_correction1,
                         const float bias_correction2);
    void FusedSGDMomentumUpdate(CAIF_Tensor &param,const CAIF_Tensor &grad,CAIF_Tensor &velocity,
                                const float lr,const float momentum,const float weight_decay);
    
    // Low-level backend delegation (for backward compatibility)
    void MatrixMultiply(const CAIF_TensorData &a,const CAIF_TensorData &b,CAIF_TensorData &result);
    void Convolution2D(const CAIF_TensorData &input,const CAIF_TensorData &kernel,
                       CAIF_TensorData &output,const CAIF_TensorBackend::ConvolutionParams &params);

  private:
    std::unique_ptr<CAIF_TensorBackend> _current_backend;
    CAIF_TensorBackend::BackendType_e _backend_type;
    static CAIF_TensorBackend::BackendType_e _default_backend;
    
    std::unique_ptr<CAIF_TensorBackend> CreateBackend(const CAIF_TensorBackend::BackendType_e backend);
};

}//end instance namespace

#endif  // CAIF_FRAMEWORK_H
