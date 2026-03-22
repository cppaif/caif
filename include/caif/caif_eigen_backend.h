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
 * @file aif_eigen_backend.h
 * @brief CPU-based tensor computation backend using Eigen library
 */

#ifndef CAIF_EIGEN_BACKEND_H
#define CAIF_EIGEN_BACKEND_H

#include "caif_tensor_backend.h"
#include "caif_cpu_tensor_data.h"
#include <memory>
#include <vector>
#include "caif_exception.h"
#include <string>

namespace instance
{

/**
 * @brief CPU-based tensor computation backend using Eigen library
 * 
 * This backend provides CPU-optimized tensor operations using the Eigen linear
 * algebra library. It offers excellent performance for moderate-sized tensors
 * and is the fallback backend available on all systems.
 * 
 * Key features:
 * - Optimized CPU matrix operations using Eigen
 * - SIMD acceleration where available
 * - Memory-efficient algorithms
 * - Cross-platform compatibility
 * - No external dependencies beyond Eigen
 * 
 * This backend is automatically selected when GPU backends are not available
 * or when explicitly requested for CPU-only computation.
 * 
 * @note All operations are performed on CPU using optimized Eigen algorithms
 * @see CAIF_TensorBackend, CAIF_CPUTensorData
 */
class CAIF_EigenBackend:public CAIF_TensorBackend
{
  public:
    /**
     * @brief Constructor
     * 
     * Initializes the Eigen backend. No special setup is required as Eigen
     * is a header-only library.
     */
    CAIF_EigenBackend();
    
    /**
     * @brief Destructor
     * 
     * Cleans up any resources. No special cleanup is required for Eigen.
     */
    virtual ~CAIF_EigenBackend();
    
    /**
     * @brief Create a new tensor with specified shape and data type
     * 
     * Creates an Eigen-compatible tensor data object with CPU memory allocation.
     * 
     * @param shape Vector defining the dimensions of the tensor
     * @param dtype Data type for tensor elements
     * @return Unique pointer to CAIF_CPUTensorData object
     * @throws std::bad_alloc if memory allocation fails
     */
    virtual std::unique_ptr<CAIF_TensorData> CreateTensor(
                                                         const std::vector<uint32_t> &shape,
                                                         const CAIF_DataType &dtype
                                                        )override;
    
    /**
     * @brief Perform matrix multiplication using Eigen optimizations
     * 
     * Performs highly optimized matrix multiplication using Eigen's internal
     * algorithms. Automatically uses SIMD instructions where available.
     * 
     * @param a Left matrix operand (2D tensor)
     * @param b Right matrix operand (2D tensor)
     * @param result Pre-allocated result matrix
     * @return Expected void on success, error string on failure
     * 
     * @pre All tensors must be CAIF_CPUTensorData objects
     * @pre Matrix dimensions must be compatible for multiplication
     * 
     * @note Uses Eigen's optimized matrix multiplication algorithms
     */
    virtual void MatrixMultiply(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )override;

    /**
     * @brief Perform matrix multiplication with transpose using Eigen
     * 
     * CPU-based matrix multiplication with optional transpose on either
     * or both matrices. Uses optimized Eigen matrix operations.
     * 
     * @param a Left matrix operand
     * @param b Right matrix operand
     * @param result Pre-allocated result matrix
     * @param trans_a Transpose flag for matrix a
     * @param trans_b Transpose flag for matrix b
     */
    virtual void MatrixMultiplyEx(
                                  const CAIF_TensorData &a,
                                  const CAIF_TensorData &b,
                                  CAIF_TensorData &result,
                                  const Transpose_e trans_a,
                                  const Transpose_e trans_b
                                 )override;
    
    /**
     * @brief Perform 2D convolution using CPU algorithms
     * 
     * Implements 2D convolution using optimized CPU loops with support for
     * arbitrary stride and padding configurations. Uses NHWC tensor format.
     * 
     * @param input Input tensor in NHWC format
     * @param kernel Convolution kernel tensor
     * @param output Pre-allocated output tensor
     * @param params Convolution parameters (stride, padding)
     * @return Expected void on success, error string on failure
     * 
     * @pre All tensors must be 4D in NHWC format
     * @pre All tensors must be CAIF_CPUTensorData objects
     * @pre Output dimensions must match expected convolution result
     * 
     * @note This is a reference implementation optimized for correctness
     */
    virtual void Convolution2D(
                               const CAIF_TensorData &input,
                               const CAIF_TensorData &kernel,
                               CAIF_TensorData &output,
                               const ConvolutionParams &params
                              )override;
    
    /**
     * @brief Perform 2D max pooling using CPU algorithms
     */
    virtual void MaxPooling2D(
                              const CAIF_TensorData &input,
                              CAIF_TensorData &output,
                              CAIF_TensorData *indices,
                              const PoolingParams &params
                             )override;

    /**
     * @brief Perform 2D average pooling using CPU algorithms
     */
    virtual void AveragePooling2D(
                                  const CAIF_TensorData &input,
                                  CAIF_TensorData &output,
                                  const PoolingParams &params
                                 )override;

    /**
     * @brief Perform batch normalization forward pass using CPU
     */
    virtual void BatchNormForward(
                                  const CAIF_TensorData &input,
                                  CAIF_TensorData &output,
                                  const CAIF_TensorData &scale,
                                  const CAIF_TensorData &bias,
                                  CAIF_TensorData &running_mean,
                                  CAIF_TensorData &running_var,
                                  CAIF_TensorData &saved_mean,
                                  CAIF_TensorData &saved_inv_var,
                                  const BatchNormParams &params,
                                  const bool training
                                 )override;

    /**
     * @brief Apply activation function using CPU
     */
    virtual void ActivationForward(
                                   const CAIF_TensorData &input,
                                   CAIF_TensorData &output,
                                   const ActivationType_e activation_type
                                  )override;

    /**
     * @brief Apply softmax using CPU
     */
    virtual void SoftmaxForward(
                                const CAIF_TensorData &input,
                                CAIF_TensorData &output
                               )override;

    /**
     * @brief Compute gradient of convolution w.r.t. input using CPU
     */
    virtual void Convolution2DBackwardData(
                                           const CAIF_TensorData &grad_output,
                                           const CAIF_TensorData &kernel,
                                           CAIF_TensorData &grad_input,
                                           const ConvolutionParams &params
                                          )override;

    /**
     * @brief Compute gradient of convolution w.r.t. filter using CPU
     */
    virtual void Convolution2DBackwardFilter(
                                             const CAIF_TensorData &input,
                                             const CAIF_TensorData &grad_output,
                                             CAIF_TensorData &grad_kernel,
                                             const ConvolutionParams &params
                                            )override;

    /**
     * @brief Backward pass for max pooling using CPU
     */
    virtual void MaxPooling2DBackward(
                                      const CAIF_TensorData &grad_output,
                                      const CAIF_TensorData *indices,
                                      const CAIF_TensorData &input,
                                      CAIF_TensorData &grad_input,
                                      const PoolingParams &params
                                     )override;

    /**
     * @brief Backward pass for average pooling using CPU
     */
    virtual void AveragePooling2DBackward(
                                          const CAIF_TensorData &grad_output,
                                          CAIF_TensorData &grad_input,
                                          const PoolingParams &params
                                         )override;

    /**
     * @brief Backward pass for batch normalization using CPU
     */
    virtual void BatchNormBackward(
                                   const CAIF_TensorData &grad_output,
                                   const CAIF_TensorData &input,
                                   const CAIF_TensorData &scale,
                                   const CAIF_TensorData &saved_mean,
                                   const CAIF_TensorData &saved_inv_var,
                                   CAIF_TensorData &grad_input,
                                   CAIF_TensorData &grad_scale,
                                   CAIF_TensorData &grad_bias,
                                   const BatchNormParams &params
                                  )override;

    /**
     * @brief Backward pass for activation function using CPU
     */
    virtual void ActivationBackward(
                                    const CAIF_TensorData &grad_output,
                                    const CAIF_TensorData &input,
                                    const CAIF_TensorData &output,
                                    CAIF_TensorData &grad_input,
                                    const ActivationType_e activation_type
                                   )override;

    /**
     * @brief Backward pass for softmax using CPU
     */
    virtual void SoftmaxBackward(
                                 const CAIF_TensorData &grad_output,
                                 const CAIF_TensorData &output,
                                 CAIF_TensorData &grad_input
                                )override;

    /**
     * @brief Apply dropout using CPU
     */
    virtual void DropoutForward(
                                const CAIF_TensorData &input,
                                CAIF_TensorData &output,
                                CAIF_TensorData &mask,
                                const float dropout_rate,
                                const bool training
                               )override;

    /**
     * @brief Backward pass for dropout using CPU
     */
    virtual void DropoutBackward(
                                 const CAIF_TensorData &grad_output,
                                 const CAIF_TensorData &mask,
                                 CAIF_TensorData &grad_input,
                                 const float dropout_rate
                                )override;

    //--------------------------------------------------------------------------
    // Element-wise Operations
    //--------------------------------------------------------------------------

    virtual void ElementwiseAdd(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )override;

    virtual void ElementwiseAddScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )override;

    virtual void ElementwiseSub(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )override;

    virtual void ElementwiseSubScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )override;

    virtual void ElementwiseMul(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )override;

    virtual void ElementwiseMulScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )override;

    virtual void ElementwiseDiv(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )override;

    virtual void ElementwiseDivScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )override;

    virtual void ElementwiseSqrt(
                                 const CAIF_TensorData &a,
                                 CAIF_TensorData &result
                                )override;

    virtual float ReduceSum(const CAIF_TensorData &a)override;

    virtual float ReduceMean(const CAIF_TensorData &a)override;

    //--------------------------------------------------------------------------
    // Loss Function Operations
    //--------------------------------------------------------------------------

    virtual void CrossEntropyLoss(
                                  const CAIF_TensorData &predictions,
                                  const CAIF_TensorData &targets,
                                  CAIF_TensorData &loss_per_sample,
                                  const float epsilon
                                 )override;

    virtual void CrossEntropyGradient(
                                      const CAIF_TensorData &predictions,
                                      const CAIF_TensorData &targets,
                                      CAIF_TensorData &gradient,
                                      const float epsilon
                                     )override;

    virtual void MSELoss(
                         const CAIF_TensorData &predictions,
                         const CAIF_TensorData &targets,
                         CAIF_TensorData &loss_elements
                        )override;

    virtual void MSEGradient(
                             const CAIF_TensorData &predictions,
                             const CAIF_TensorData &targets,
                             CAIF_TensorData &gradient
                            )override;

    //--------------------------------------------------------------------------
    // Optimizer Operations
    //--------------------------------------------------------------------------

    virtual void FusedAdamUpdate(
                                 CAIF_TensorData &param,
                                 const CAIF_TensorData &grad,
                                 CAIF_TensorData &m,
                                 CAIF_TensorData &v,
                                 const float lr,
                                 const float beta1,
                                 const float beta2,
                                 const float epsilon,
                                 const float weight_decay,
                                 const float bias_correction1,
                                 const float bias_correction2
                                )override;

    virtual void FusedSGDMomentumUpdate(
                                        CAIF_TensorData &param,
                                        const CAIF_TensorData &grad,
                                        CAIF_TensorData &velocity,
                                        const float lr,
                                        const float momentum,
                                        const float weight_decay
                                       )override;

    /**
     * @brief Get backend type identifier
     * 
     * @return CAIF_TensorBackend::BackendType_e::Eigen
     */
    virtual BackendType_e Type()const override{return BackendType_e::Eigen;}
    
    /**
     * @brief Check if backend uses GPU acceleration
     * 
     * @return false (Eigen backend is CPU-only)
     */
    virtual bool IsGPUAccelerated()const override{return false;}

  protected:

  private:
    /**
     * @brief Validate matrix dimensions for multiplication
     * 
     * Checks that the provided tensors have compatible dimensions for
     * matrix multiplication and are properly allocated.
     * 
     * @param a Left matrix operand
     * @param b Right matrix operand
     * @param result Result matrix
     * @return Expected void on success, error string on failure
     * 
     * @note This method performs comprehensive validation of tensor compatibility
     */
    void ValidateMatrixDimensions(
                                  const CAIF_TensorData &a,
                                  const CAIF_TensorData &b,
                                  const CAIF_TensorData &result
                                 )const;
};

}//end instance namespace

#endif  // CAIF_EIGEN_BACKEND_H 