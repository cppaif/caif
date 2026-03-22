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
 * @file aif_tensor_backend.h
 * @brief Abstract base class for tensor computation backends
 */

#ifndef CAIF_TENSOR_BACKEND_H
#define CAIF_TENSOR_BACKEND_H

#include "caif_constants.h"
#include "caif_tensor_data.h"
#include "caif_data_type.h"
#include "caif_base.h"
#include <memory>
#include <vector>
#include "caif_exception.h"
#include <string>

namespace instance
{

// DEVICE_MIGRATION_REMOVE: This backend abstraction is deprecated for GPU operations.
// GPU operations now use CAIF_DeviceOps directly on CAIF_DeviceTensor.
// CPU backends (Eigen, BLAS) may be retained for CPU-only workflows.
// See DEPRECATED_FOR_DEVICE_MIGRATION.md for details.
/**
 * @brief Abstract base class for tensor computation backends
 *
 * This class defines the interface that all tensor computation backends must implement.
 * Backends provide hardware-specific optimizations for tensor operations like matrix
 * multiplication and convolution. The framework can automatically select the best
 * available backend or allow manual selection.
 *
 * @note All derived classes must implement the pure virtual methods
 * @see CAIF_EigenBackend, CAIF_CudaBackend, CAIF_VulkanBackend
 */
class CAIF_TensorBackend:public CAIF_Base
{
  public:
    /**
     * @brief Enumeration of available backend types
     * 
     * Defines the different computation backends available for tensor operations.
     * Each backend provides different performance characteristics and hardware support.
     */
    enum class BackendType_e:uint8_t
    {
      Eigen,   ///< CPU-based computation using Eigen library
      BLAS,    ///< CPU computation using BLAS (e.g., OpenBLAS/MKL)
      CUDA,    ///< GPU computation using NVIDIA CUDA
      Vulkan,  ///< GPU computation using Vulkan compute shaders
      Auto     ///< Automatic backend selection based on availability
    };

    /**
     * @brief Parameters for 2D convolution operations
     * 
     * Contains all necessary parameters to configure a 2D convolution operation
     * including stride and padding settings.
     */
    struct ConvolutionParams
    {
      uint32_t stride_x;   ///< Horizontal stride (step size) for convolution
      uint32_t stride_y;   ///< Vertical stride (step size) for convolution
      uint32_t padding_x;  ///< Horizontal padding to apply to input
      uint32_t padding_y;  ///< Vertical padding to apply to input
    };

    /**
     * @brief Parameters for 2D pooling operations
     */
    struct PoolingParams
    {
      uint32_t pool_height;  ///< Pooling window height
      uint32_t pool_width;   ///< Pooling window width
      uint32_t stride_y;     ///< Vertical stride
      uint32_t stride_x;     ///< Horizontal stride
      uint32_t padding_y;    ///< Vertical padding
      uint32_t padding_x;    ///< Horizontal padding
    };

    /**
     * @brief Activation function types
     */
    enum class ActivationType_e:uint8_t
    {
      Identity,   ///< No activation (pass-through) / Linear
      ReLU,       ///< Rectified Linear Unit: max(0, x)
      Sigmoid,    ///< Sigmoid: 1 / (1 + exp(-x))
      Tanh,       ///< Hyperbolic tangent
      LeakyReLU,  ///< Leaky ReLU: max(alpha*x, x)
      ELU,        ///< Exponential Linear Unit
      GELU,       ///< Gaussian Error Linear Unit
      Swish       ///< Swish: x * sigmoid(x)
    };

    /**
     * @brief Transpose operation flags for matrix multiplication
     */
    enum class Transpose_e:uint8_t
    {
      NoTrans=0,  ///< Use matrix as-is
      Trans=1     ///< Transpose the matrix
    };

    /**
     * @brief Parameters for batch normalization operations
     */
    struct BatchNormParams
    {
      float epsilon;   ///< Small constant for numerical stability
      float momentum;  ///< Momentum for running statistics update
    };

    /**
     * @brief Virtual destructor
     */
    virtual ~CAIF_TensorBackend()=default;
    
    /**
     * @brief Create a new tensor with specified shape and data type
     * 
     * Creates a tensor data object appropriate for this backend. The tensor
     * will be initialized with zero values and allocated according to the
     * backend's memory management strategy.
     * 
     * @param shape Vector defining the dimensions of the tensor
     * @param dtype Data type for tensor elements
     * @return Unique pointer to the created tensor data object
     * @throws std::bad_alloc if memory allocation fails
     * 
     * @note The returned tensor is owned by the caller and must be managed appropriately
     */
    virtual std::unique_ptr<CAIF_TensorData> CreateTensor(
                                                         const std::vector<uint32_t> &shape,
                                                         const CAIF_DataType &dtype
                                                        )=0;
    
    /**
     * @brief Perform matrix multiplication: result = a * b
     * 
     * Computes the matrix multiplication of two 2D tensors using the backend's
     * optimized implementation. The operation follows standard matrix multiplication
     * rules: A(m,n) * B(n,p) = C(m,p).
     * 
     * @param a Left matrix operand (must be 2D)
     * @param b Right matrix operand (must be 2D)
     * @param result Output matrix to store the result (must be pre-allocated)
     * @return Expected void on success, error string on failure
     * 
     * @pre All tensors must be 2D (matrices)
     * @pre a.columns must equal b.rows
     * @pre result dimensions must be (a.rows, b.columns)
     * @pre All tensors must have the same data type
     * 
     * @note This operation may be performed on GPU or CPU depending on backend
     */
    virtual void MatrixMultiply(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )=0;

    /**
     * @brief Perform matrix multiplication with optional transpose: result = op(a) * op(b)
     * 
     * Extended matrix multiplication supporting transpose operations on either
     * or both input matrices. Uses cuBLAS on GPU or optimized BLAS on CPU.
     * This is critical for backward pass gradient computation.
     * 
     * @param a Left matrix operand (must be 2D)
     * @param b Right matrix operand (must be 2D)
     * @param result Output matrix to store the result (must be pre-allocated)
     * @param trans_a Transpose flag for matrix a
     * @param trans_b Transpose flag for matrix b
     * 
     * @pre All tensors must be 2D (matrices)
     * @pre op(a).columns must equal op(b).rows where op() applies transpose if specified
     * @pre result dimensions must be (op(a).rows, op(b).columns)
     * @pre All tensors must have the same data type
     */
    virtual void MatrixMultiplyEx(
                                  const CAIF_TensorData &a,
                                  const CAIF_TensorData &b,
                                  CAIF_TensorData &result,
                                  const Transpose_e trans_a,
                                  const Transpose_e trans_b
                                 )=0;
    
    /**
     * @brief Perform 2D convolution operation
     * 
     * Applies a 2D convolution operation using the specified kernel and parameters.
     * Expects tensors in NHWC format (batch, height, width, channels) for optimal
     * performance across different backends.
     * 
     * @param input Input tensor (4D: batch, height, width, input_channels)
     * @param kernel Convolution kernel (4D: kernel_height, kernel_width, input_channels, output_channels)
     * @param output Pre-allocated output tensor for results
     * @param params Convolution parameters (stride, padding)
     * @return Expected void on success, error string on failure
     * 
     * @pre All tensors must be 4D in NHWC format
     * @pre input.channels must equal kernel.input_channels
     * @pre kernel.output_channels must equal output.channels
     * @pre Output dimensions must match expected convolution result
     * 
     * @note Performance varies significantly between CPU and GPU backends
     */
    virtual void Convolution2D(
                               const CAIF_TensorData &input,
                               const CAIF_TensorData &kernel,
                               CAIF_TensorData &output,
                               const ConvolutionParams &params
                              )=0;
    
    /**
     * @brief Perform 2D max pooling operation
     * 
     * @param input Input tensor (4D: batch, height, width, channels) in NHWC format
     * @param output Pre-allocated output tensor for results
     * @param indices Optional tensor to store max indices for backward pass (can be nullptr)
     * @param params Pooling parameters (pool size, stride, padding)
     */
    virtual void MaxPooling2D(
                              const CAIF_TensorData &input,
                              CAIF_TensorData &output,
                              CAIF_TensorData *indices,
                              const PoolingParams &params
                             )=0;

    /**
     * @brief Perform 2D average pooling operation
     * 
     * @param input Input tensor (4D: batch, height, width, channels) in NHWC format
     * @param output Pre-allocated output tensor for results
     * @param params Pooling parameters (pool size, stride, padding)
     */
    virtual void AveragePooling2D(
                                  const CAIF_TensorData &input,
                                  CAIF_TensorData &output,
                                  const PoolingParams &params
                                 )=0;

    /**
     * @brief Perform batch normalization forward pass (training mode)
     * 
     * @param input Input tensor
     * @param output Output tensor (normalized)
     * @param scale Scale parameter (gamma)
     * @param bias Bias parameter (beta)
     * @param running_mean Running mean (updated during training)
     * @param running_var Running variance (updated during training)
     * @param saved_mean Output: batch mean saved for backward pass
     * @param saved_inv_var Output: batch inverse variance saved for backward pass
     * @param params Batch normalization parameters
     * @param training True for training mode, false for inference
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
                                 )=0;

    /**
     * @brief Apply activation function forward pass
     * 
     * @param input Input tensor
     * @param output Output tensor
     * @param activation_type Type of activation function
     */
    virtual void ActivationForward(
                                   const CAIF_TensorData &input,
                                   CAIF_TensorData &output,
                                   const ActivationType_e activation_type
                                  )=0;

    /**
     * @brief Apply softmax forward pass
     * 
     * @param input Input tensor
     * @param output Output tensor
     */
    virtual void SoftmaxForward(
                                const CAIF_TensorData &input,
                                CAIF_TensorData &output
                               )=0;

    /**
     * @brief Compute gradient of convolution w.r.t. input (backprop through conv)
     * 
     * @param grad_output Gradient flowing from output (4D: batch, height, width, out_channels)
     * @param kernel Convolution kernel (4D: kernel_h, kernel_w, in_channels, out_channels)
     * @param grad_input Output: gradient w.r.t. input (4D: batch, height, width, in_channels)
     * @param params Convolution parameters (stride, padding)
     */
    virtual void Convolution2DBackwardData(
                                           const CAIF_TensorData &grad_output,
                                           const CAIF_TensorData &kernel,
                                           CAIF_TensorData &grad_input,
                                           const ConvolutionParams &params
                                          )=0;

    /**
     * @brief Compute gradient of convolution w.r.t. filter/kernel weights
     * 
     * @param input Original input to convolution (4D: batch, height, width, in_channels)
     * @param grad_output Gradient flowing from output (4D: batch, height, width, out_channels)
     * @param grad_kernel Output: gradient w.r.t. kernel (4D: kernel_h, kernel_w, in_channels, out_channels)
     * @param params Convolution parameters (stride, padding)
     */
    virtual void Convolution2DBackwardFilter(
                                             const CAIF_TensorData &input,
                                             const CAIF_TensorData &grad_output,
                                             CAIF_TensorData &grad_kernel,
                                             const ConvolutionParams &params
                                            )=0;

    /**
     * @brief Backward pass for max pooling
     * 
     * @param grad_output Gradient flowing from output
     * @param indices Max indices from forward pass (or nullptr to recompute)
     * @param input Original input (needed if indices is nullptr)
     * @param grad_input Output: gradient w.r.t. input
     * @param params Pooling parameters
     */
    virtual void MaxPooling2DBackward(
                                      const CAIF_TensorData &grad_output,
                                      const CAIF_TensorData *indices,
                                      const CAIF_TensorData &input,
                                      CAIF_TensorData &grad_input,
                                      const PoolingParams &params
                                     )=0;

    /**
     * @brief Backward pass for average pooling
     * 
     * @param grad_output Gradient flowing from output
     * @param grad_input Output: gradient w.r.t. input
     * @param params Pooling parameters
     */
    virtual void AveragePooling2DBackward(
                                          const CAIF_TensorData &grad_output,
                                          CAIF_TensorData &grad_input,
                                          const PoolingParams &params
                                         )=0;

    /**
     * @brief Backward pass for batch normalization
     * 
     * @param grad_output Gradient flowing from output
     * @param input Original input
     * @param scale Scale parameter (gamma)
     * @param saved_mean Saved batch mean from forward pass
     * @param saved_inv_var Saved inverse variance from forward pass
     * @param grad_input Output: gradient w.r.t. input
     * @param grad_scale Output: gradient w.r.t. scale
     * @param grad_bias Output: gradient w.r.t. bias
     * @param params Batch normalization parameters
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
                                  )=0;

    /**
     * @brief Backward pass for activation function
     * 
     * @param grad_output Gradient flowing from output
     * @param input Original input (or output for some activations)
     * @param output Original output (for activations like ReLU, Sigmoid)
     * @param grad_input Output: gradient w.r.t. input
     * @param activation_type Type of activation function
     */
    virtual void ActivationBackward(
                                    const CAIF_TensorData &grad_output,
                                    const CAIF_TensorData &input,
                                    const CAIF_TensorData &output,
                                    CAIF_TensorData &grad_input,
                                    const ActivationType_e activation_type
                                   )=0;

    /**
     * @brief Backward pass for softmax
     * 
     * @param grad_output Gradient flowing from output
     * @param output Softmax output from forward pass
     * @param grad_input Output: gradient w.r.t. input
     */
    virtual void SoftmaxBackward(
                                 const CAIF_TensorData &grad_output,
                                 const CAIF_TensorData &output,
                                 CAIF_TensorData &grad_input
                                )=0;

    /**
     * @brief Apply dropout forward pass
     * 
     * Randomly zeros elements with probability dropout_rate during training.
     * Scales remaining elements by 1/(1-dropout_rate) for inverted dropout.
     * 
     * @param input Input tensor
     * @param output Output tensor
     * @param mask Output: binary mask tensor (same shape as input) for backward pass
     * @param dropout_rate Probability of dropping an element (0.0 to 1.0)
     * @param training True for training mode (apply dropout), false for inference (pass through)
     */
    virtual void DropoutForward(
                                const CAIF_TensorData &input,
                                CAIF_TensorData &output,
                                CAIF_TensorData &mask,
                                const float dropout_rate,
                                const bool training
                               )=0;

    /**
     * @brief Backward pass for dropout
     * 
     * @param grad_output Gradient flowing from output
     * @param mask Binary mask from forward pass
     * @param grad_input Output: gradient w.r.t. input
     * @param dropout_rate Dropout rate used in forward pass (for scaling)
     */
    virtual void DropoutBackward(
                                 const CAIF_TensorData &grad_output,
                                 const CAIF_TensorData &mask,
                                 CAIF_TensorData &grad_input,
                                 const float dropout_rate
                                )=0;

    //--------------------------------------------------------------------------
    // Element-wise Operations
    //--------------------------------------------------------------------------

    /**
     * @brief Element-wise addition of two tensors
     * @param a First input tensor
     * @param b Second input tensor
     * @param result Output tensor (same shape as inputs)
     */
    virtual void ElementwiseAdd(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )=0;

    /**
     * @brief Element-wise addition of tensor and scalar
     * @param a Input tensor
     * @param scalar Scalar value to add
     * @param result Output tensor (same shape as input)
     */
    virtual void ElementwiseAddScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )=0;

    /**
     * @brief Element-wise subtraction of two tensors
     * @param a First input tensor
     * @param b Second input tensor
     * @param result Output tensor (same shape as inputs)
     */
    virtual void ElementwiseSub(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )=0;

    /**
     * @brief Element-wise subtraction of tensor and scalar
     * @param a Input tensor
     * @param scalar Scalar value to subtract
     * @param result Output tensor (same shape as input)
     */
    virtual void ElementwiseSubScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )=0;

    /**
     * @brief Element-wise multiplication of two tensors
     * @param a First input tensor
     * @param b Second input tensor
     * @param result Output tensor (same shape as inputs)
     */
    virtual void ElementwiseMul(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )=0;

    /**
     * @brief Element-wise multiplication of tensor and scalar
     * @param a Input tensor
     * @param scalar Scalar value to multiply
     * @param result Output tensor (same shape as input)
     */
    virtual void ElementwiseMulScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )=0;

    /**
     * @brief Element-wise division of two tensors
     * @param a First input tensor (numerator)
     * @param b Second input tensor (denominator)
     * @param result Output tensor (same shape as inputs)
     */
    virtual void ElementwiseDiv(
                                const CAIF_TensorData &a,
                                const CAIF_TensorData &b,
                                CAIF_TensorData &result
                               )=0;

    /**
     * @brief Element-wise division of tensor by scalar
     * @param a Input tensor
     * @param scalar Scalar divisor
     * @param result Output tensor (same shape as input)
     */
    virtual void ElementwiseDivScalar(
                                      const CAIF_TensorData &a,
                                      const float scalar,
                                      CAIF_TensorData &result
                                     )=0;

    /**
     * @brief Element-wise square root
     * @param a Input tensor
     * @param result Output tensor (same shape as input)
     */
    virtual void ElementwiseSqrt(
                                 const CAIF_TensorData &a,
                                 CAIF_TensorData &result
                                )=0;

    //--------------------------------------------------------------------------
    // Reduction Operations
    //--------------------------------------------------------------------------

    /**
     * @brief Compute sum of all elements in tensor
     * @param a Input tensor
     * @return Sum of all elements
     */
    virtual float ReduceSum(const CAIF_TensorData &a)=0;

    /**
     * @brief Compute mean of all elements in tensor
     * @param a Input tensor
     * @return Mean of all elements
     */
    virtual float ReduceMean(const CAIF_TensorData &a)=0;

    //--------------------------------------------------------------------------
    // Loss Function Operations
    //--------------------------------------------------------------------------

    /**
     * @brief Compute cross entropy loss
     * @param predictions Predicted probabilities [batch_size, num_classes]
     * @param targets Target labels (one-hot) [batch_size, num_classes]
     * @param loss_per_sample Output: loss for each sample [batch_size]
     * @param epsilon Numerical stability constant
     */
    virtual void CrossEntropyLoss(
                                  const CAIF_TensorData &predictions,
                                  const CAIF_TensorData &targets,
                                  CAIF_TensorData &loss_per_sample,
                                  const float epsilon
                                 )=0;

    /**
     * @brief Compute cross entropy gradient
     * @param predictions Predicted probabilities
     * @param targets Target labels (one-hot)
     * @param gradient Output: gradient tensor
     * @param epsilon Numerical stability constant
     */
    virtual void CrossEntropyGradient(
                                      const CAIF_TensorData &predictions,
                                      const CAIF_TensorData &targets,
                                      CAIF_TensorData &gradient,
                                      const float epsilon
                                     )=0;

    /**
     * @brief Compute mean squared error loss
     * @param predictions Predicted values
     * @param targets Target values
     * @param loss_elements Output: squared error for each element
     */
    virtual void MSELoss(
                         const CAIF_TensorData &predictions,
                         const CAIF_TensorData &targets,
                         CAIF_TensorData &loss_elements
                        )=0;

    /**
     * @brief Compute MSE gradient
     * @param predictions Predicted values
     * @param targets Target values
     * @param gradient Output: gradient tensor
     */
    virtual void MSEGradient(
                             const CAIF_TensorData &predictions,
                             const CAIF_TensorData &targets,
                             CAIF_TensorData &gradient
                            )=0;

    //--------------------------------------------------------------------------
    // Optimizer Operations
    //--------------------------------------------------------------------------

    /**
     * @brief Fused Adam optimizer update
     * Combines all Adam operations into a single efficient pass
     * @param param Parameter tensor (in/out)
     * @param grad Gradient tensor
     * @param m First moment estimate (in/out)
     * @param v Second moment estimate (in/out)
     * @param lr Learning rate
     * @param beta1 First moment decay rate
     * @param beta2 Second moment decay rate
     * @param epsilon Numerical stability constant
     * @param weight_decay L2 regularization coefficient
     * @param bias_correction1 Bias correction for first moment (1 - beta1^t)
     * @param bias_correction2 Bias correction for second moment (1 - beta2^t)
     */
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
                                )=0;

    /**
     * @brief Fused SGD with momentum update
     * @param param Parameter tensor (in/out)
     * @param grad Gradient tensor
     * @param velocity Velocity tensor (in/out)
     * @param lr Learning rate
     * @param momentum Momentum coefficient
     * @param weight_decay L2 regularization coefficient
     */
    virtual void FusedSGDMomentumUpdate(
                                        CAIF_TensorData &param,
                                        const CAIF_TensorData &grad,
                                        CAIF_TensorData &velocity,
                                        const float lr,
                                        const float momentum,
                                        const float weight_decay
                                       )=0;

    /**
     * @brief Get the backend type identifier
     * @return BackendType_e enum value
     */
    virtual BackendType_e Type()const=0;
    
    /**
     * @brief Check if this backend uses GPU acceleration
     * 
     * @return true if operations are performed on GPU, false for CPU-only backends
     */
    virtual bool IsGPUAccelerated()const=0;

  protected:

  private:
};

}//end instance namespace

#endif  // CAIF_TENSOR_BACKEND_H 