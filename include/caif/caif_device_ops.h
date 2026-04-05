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
// Device operations for CAIF_DeviceTensor
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_OPS_H
#define CAIF_DEVICE_OPS_H

#include "caif_device_tensor.h"
#include <cstdint>
#include <vector>

#ifndef USE_CAIF_CUDA
typedef void *cudaStream_t;
#endif

namespace instance
{

/**
 * @brief Device operations namespace
 *
 * This namespace contains operations that work exclusively with CAIF_DeviceTensor.
 * All operations are designed to:
 * - Take device tensors by reference
 * - Use the stream from the output tensor for operation ordering
 * - Never call cudaDeviceSynchronize() (except for scalar returns)
 *
 * Scalar return operations (ReduceSum, ReduceMean) are the only sync points
 * since they must return values to the CPU.
 *
 * Part of the device-resident tensor architecture (see DEVICE_TENSOR_MIGRATION.md)
 */
namespace CAIF_DeviceOps
{
  //----------------------------------------------------------------------------
  // Matrix Operations
  //----------------------------------------------------------------------------

  /**
   * @brief Standard matrix multiplication: output = a * b
   *
   * Computes C = A * B where:
   * - A is [M x K]
   * - B is [K x N]
   * - C is [M x N]
   *
   * Uses the output tensor's stream for the operation.
   *
   * @param a Input matrix A [M x K]
   * @param b Input matrix B [K x N]
   * @param output Output matrix C [M x N] (must be pre-allocated)
   */
  void MatMul(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief Matrix multiplication with A transposed: output = a^T * b
   *
   * Computes C = A^T * B where:
   * - A is [K x M] (transposed to [M x K])
   * - B is [K x N]
   * - C is [M x N]
   *
   * @param a Input matrix A [K x M]
   * @param b Input matrix B [K x N]
   * @param output Output matrix C [M x N] (must be pre-allocated)
   */
  void MatMulTransposeA(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief Matrix multiplication with B transposed: output = a * b^T
   *
   * Computes C = A * B^T where:
   * - A is [M x K]
   * - B is [N x K] (transposed to [K x N])
   * - C is [M x N]
   *
   * @param a Input matrix A [M x K]
   * @param b Input matrix B [N x K]
   * @param output Output matrix C [M x N] (must be pre-allocated)
   */
  void MatMulTransposeB(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  //----------------------------------------------------------------------------
  // Batched Matrix Operations (for multi-head attention)
  //----------------------------------------------------------------------------

  /**
   * @brief Batched matrix multiplication: output[i] = a[i] * b[i]
   *
   * Each matrix multiplication computes C = A * B where:
   * - A is [M x K], B is [K x N], C is [M x N]
   * - Stride between consecutive matrices = M*K, K*N, M*N respectively
   *
   * Uses cublasSgemmStridedBatched.
   *
   * @param a Input tensor containing batch_count stacked [M x K] matrices
   * @param b Input tensor containing batch_count stacked [K x N] matrices
   * @param output Output tensor for batch_count stacked [M x N] matrices
   * @param m Rows of A / rows of C
   * @param k Columns of A / rows of B
   * @param n Columns of B / columns of C
   * @param batch_count Number of independent matrix multiplications
   */
  void BatchedMatMul(const CAIF_DeviceTensor &a,
                     const CAIF_DeviceTensor &b,
                     CAIF_DeviceTensor &output,
                     int m,
                     int k,
                     int n,
                     int batch_count);

  /**
   * @brief Batched matrix multiplication with A transposed: output[i] = a[i]^T * b[i]
   *
   * Each multiplication computes C = A^T * B where:
   * - A is physically [K x M], transposed to [M x K]
   * - B is [K x N], C is [M x N]
   *
   * @param a Input tensor containing batch_count stacked [K x M] matrices
   * @param b Input tensor containing batch_count stacked [K x N] matrices
   * @param output Output tensor for batch_count stacked [M x N] matrices
   * @param k Rows of physical A (columns of A^T)
   * @param m Columns of physical A (rows of A^T)
   * @param n Columns of B / columns of C
   * @param batch_count Number of independent matrix multiplications
   */
  void BatchedMatMulTransposeA(const CAIF_DeviceTensor &a,
                               const CAIF_DeviceTensor &b,
                               CAIF_DeviceTensor &output,
                               int k,
                               int m,
                               int n,
                               int batch_count);

  /**
   * @brief Batched matrix multiplication with B transposed: output[i] = a[i] * b[i]^T
   *
   * Each multiplication computes C = A * B^T where:
   * - A is [M x K], B is physically [N x K], transposed to [K x N]
   * - C is [M x N]
   *
   * @param a Input tensor containing batch_count stacked [M x K] matrices
   * @param b Input tensor containing batch_count stacked [N x K] matrices
   * @param output Output tensor for batch_count stacked [M x N] matrices
   * @param m Rows of A / rows of C
   * @param k Columns of A / columns of B
   * @param n Rows of physical B (columns of B^T)
   * @param batch_count Number of independent matrix multiplications
   */
  void BatchedMatMulTransposeB(const CAIF_DeviceTensor &a,
                               const CAIF_DeviceTensor &b,
                               CAIF_DeviceTensor &output,
                               int m,
                               int k,
                               int n,
                               int batch_count);

  //----------------------------------------------------------------------------
  // Tensor Manipulation
  //----------------------------------------------------------------------------

  /**
   * @brief Transpose a 2D tensor: output = input^T
   *
   * @param input Input tensor [M x N]
   * @param output Output tensor [N x M] (must be pre-allocated)
   */
  void Transpose(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  //----------------------------------------------------------------------------
  // Element-wise Operations
  //----------------------------------------------------------------------------

  /**
   * @brief Element-wise addition: output = a + b
   *
   * Both input tensors must have the same shape, or the operation
   * must be a valid broadcast (same total elements).
   *
   * @param a First input tensor
   * @param b Second input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void Add(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief In-place scalar multiplication: tensor = tensor * scale
   *
   * Modifies the tensor in place.
   *
   * @param tensor Tensor to scale (modified in place)
   * @param scale Scalar multiplier
   */
  void Scale(CAIF_DeviceTensor &tensor,float scale);

  /**
   * @brief Scaled addition: target = target + source * scale
   *
   * Computes target += source * scale in place.
   * Commonly used for gradient accumulation with learning rate.
   *
   * @param target Target tensor (modified in place)
   * @param source Source tensor
   * @param scale Scalar multiplier for source
   */
  void AddScaled(CAIF_DeviceTensor &target,const CAIF_DeviceTensor &source,float scale);

  //----------------------------------------------------------------------------
  // Bias Operations (for dense/conv layers)
  //----------------------------------------------------------------------------

  /**
   * @brief Add bias to 2D tensor: output[b][u] = input[b][u] + bias[u]
   *
   * Broadcasts bias vector across batch dimension.
   * Can operate in-place (output == input).
   *
   * @param input Input tensor [batch x units]
   * @param bias Bias tensor [units]
   * @param output Output tensor [batch x units] (must be pre-allocated)
   */
  void BiasAdd(const CAIF_DeviceTensor &input,const CAIF_DeviceTensor &bias,CAIF_DeviceTensor &output);

  /**
   * @brief Fused matrix multiply + bias add using cublasLt
   *
   * Computes output = (a * b) + bias in a single kernel via epilogue fusion.
   * More efficient than separate MatMul + BiasAdd calls.
   *
   * @param a Input matrix A [M x K]
   * @param b Input matrix B [K x N]
   * @param bias Bias vector [N]
   * @param output Output matrix [M x N] (must be pre-allocated)
   * @param stream CUDA stream for the operation
   */
  void MatMulBias(const CAIF_DeviceTensor &a,
                  const CAIF_DeviceTensor &b,
                  const CAIF_DeviceTensor &bias,
                  CAIF_DeviceTensor &output,
                  cudaStream_t stream);

  /**
   * @brief Compute bias gradient from 2D gradient tensor
   *
   * Computes bias_grad[u] = sum over batch of grad[b][u]
   *
   * @param grad Gradient tensor [batch x units]
   * @param bias_grad Output bias gradient [units] (must be pre-allocated)
   */
  void BiasGradient(const CAIF_DeviceTensor &grad,CAIF_DeviceTensor &bias_grad);

  //----------------------------------------------------------------------------
  // Activation Functions (Forward)
  //----------------------------------------------------------------------------

  /**
   * @brief ReLU activation: output = max(0, input)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void ReLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Sigmoid activation: output = 1 / (1 + exp(-input))
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void Sigmoid(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Tanh activation: output = tanh(input)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void Tanh(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Softmax activation: output[i] = exp(input[i]) / sum(exp(input))
   *
   * Applies softmax along the last dimension (columns) for each row.
   * For 2D input [batch x classes], computes softmax over classes for each sample.
   *
   * @param input Input tensor [batch x classes]
   * @param output Output tensor (must be pre-allocated) [batch x classes]
   */
  void Softmax(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Leaky ReLU activation: output = max(alpha * input, input)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   * @param alpha Negative slope (default 0.01)
   */
  void LeakyReLU(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output,
                 float alpha=0.01f);

  /**
   * @brief ELU activation: output = x if x > 0, else alpha * (exp(x) - 1)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   * @param alpha Scale for negative values (default 1.0)
   */
  void ELU(const CAIF_DeviceTensor &input,
           CAIF_DeviceTensor &output,
           float alpha=1.0f);

  /**
   * @brief GELU activation: output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   *
   * Gaussian Error Linear Unit - approximation using tanh.
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void GELU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Swish activation: output = x * sigmoid(x)
   *
   * Also known as SiLU (Sigmoid Linear Unit).
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void Swish(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  //----------------------------------------------------------------------------
  // Activation Functions (Backward)
  //----------------------------------------------------------------------------

  /**
   * @brief ReLU backward pass
   *
   * Computes grad_input = grad_output * (input > 0 ? 1 : 0)
   *
   * @param grad_output Gradient from the next layer
   * @param input Original input to ReLU (for computing mask)
   * @param grad_input Output gradient (must be pre-allocated)
   */
  void ReLUBackward(const CAIF_DeviceTensor &grad_output,
                    const CAIF_DeviceTensor &input,
                    CAIF_DeviceTensor &grad_input);

  /**
   * @brief Sigmoid backward pass
   *
   * Computes grad_input = grad_output * output * (1 - output)
   * where output = sigmoid(input)
   *
   * @param grad_output Gradient from the next layer
   * @param output The forward pass output (sigmoid(input))
   * @param grad_input Output gradient (must be pre-allocated)
   */
  void SigmoidBackward(const CAIF_DeviceTensor &grad_output,
                       const CAIF_DeviceTensor &output,
                       CAIF_DeviceTensor &grad_input);

  /**
   * @brief Tanh backward pass
   *
   * Computes grad_input = grad_output * (1 - output^2)
   * where output = tanh(input)
   *
   * @param grad_output Gradient from the next layer
   * @param output The forward pass output (tanh(input))
   * @param grad_input Output gradient (must be pre-allocated)
   */
  void TanhBackward(const CAIF_DeviceTensor &grad_output,
                    const CAIF_DeviceTensor &output,
                    CAIF_DeviceTensor &grad_input);

  /**
   * @brief Softmax backward pass
   *
   * Computes the gradient through softmax using the Jacobian.
   * For each sample: grad_input = softmax * (grad_output - sum(grad_output * softmax))
   *
   * @param grad_output Gradient from the next layer [batch x classes]
   * @param output The forward pass output (softmax(input)) [batch x classes]
   * @param grad_input Output gradient (must be pre-allocated) [batch x classes]
   */
  void SoftmaxBackward(const CAIF_DeviceTensor &grad_output,
                       const CAIF_DeviceTensor &output,
                       CAIF_DeviceTensor &grad_input);

  /**
   * @brief Leaky ReLU backward pass
   *
   * Computes grad_input = grad_output * (input > 0 ? 1 : alpha)
   *
   * @param grad_output Gradient from the next layer
   * @param input Original input to LeakyReLU
   * @param grad_input Output gradient (must be pre-allocated)
   * @param alpha Negative slope (default 0.01)
   */
  void LeakyReLUBackward(const CAIF_DeviceTensor &grad_output,
                         const CAIF_DeviceTensor &input,
                         CAIF_DeviceTensor &grad_input,
                         float alpha=0.01f);

  /**
   * @brief ELU backward pass
   *
   * Computes grad_input = grad_output * (input > 0 ? 1 : output + alpha)
   *
   * @param grad_output Gradient from the next layer
   * @param input Original input to ELU
   * @param output The forward pass output
   * @param grad_input Output gradient (must be pre-allocated)
   * @param alpha Scale for negative values (default 1.0)
   */
  void ELUBackward(const CAIF_DeviceTensor &grad_output,
                   const CAIF_DeviceTensor &input,
                   const CAIF_DeviceTensor &output,
                   CAIF_DeviceTensor &grad_input,
                   float alpha=1.0f);

  /**
   * @brief GELU backward pass
   *
   * @param grad_output Gradient from the next layer
   * @param input Original input to GELU
   * @param grad_input Output gradient (must be pre-allocated)
   */
  void GELUBackward(const CAIF_DeviceTensor &grad_output,
                    const CAIF_DeviceTensor &input,
                    CAIF_DeviceTensor &grad_input);

  /**
   * @brief Swish backward pass
   *
   * Computes grad_input = grad_output * (output + sigmoid(input) * (1 - output))
   *
   * @param grad_output Gradient from the next layer
   * @param input Original input to Swish
   * @param output The forward pass output (x * sigmoid(x))
   * @param grad_input Output gradient (must be pre-allocated)
   */
  void SwishBackward(const CAIF_DeviceTensor &grad_output,
                     const CAIF_DeviceTensor &input,
                     const CAIF_DeviceTensor &output,
                     CAIF_DeviceTensor &grad_input);

  //----------------------------------------------------------------------------
  // Reduction Operations (synchronous - returns scalar)
  //----------------------------------------------------------------------------

  /**
   * @brief Sum all elements in the tensor
   *
   * WARNING: This operation synchronizes the stream and returns a scalar.
   * It is a sync point in the computation graph.
   *
   * @param tensor Input tensor
   * @return Sum of all elements
   */
  float ReduceSum(const CAIF_DeviceTensor &tensor);

  /**
   * @brief Compute mean of all elements in the tensor
   *
   * WARNING: This operation synchronizes the stream and returns a scalar.
   * It is a sync point in the computation graph.
   *
   * @param tensor Input tensor
   * @return Mean of all elements
   */
  float ReduceMean(const CAIF_DeviceTensor &tensor);

  //----------------------------------------------------------------------------
  // Loss Functions
  //----------------------------------------------------------------------------

  /**
   * @brief Mean Squared Error loss: loss = mean((pred - target)^2)
   *
   * @param pred Prediction tensor
   * @param target Target tensor
   * @param loss Scalar output tensor (1 element, must be pre-allocated)
   */
  void MSELoss(const CAIF_DeviceTensor &pred,const CAIF_DeviceTensor &target,CAIF_DeviceTensor &loss);

  /**
   * @brief MSE loss backward: grad = 2 * (pred - target) / n
   *
   * @param pred Prediction tensor
   * @param target Target tensor
   * @param grad Output gradient tensor (must be pre-allocated, same shape as pred)
   */
  void MSELossBackward(const CAIF_DeviceTensor &pred,
                       const CAIF_DeviceTensor &target,
                       CAIF_DeviceTensor &grad);

  //----------------------------------------------------------------------------
  // Optimizer Operations
  //----------------------------------------------------------------------------

  /**
   * @brief Adam optimizer update
   *
   * Updates parameters using the Adam algorithm:
   * - m = beta1 * m + (1 - beta1) * grad
   * - v = beta2 * v + (1 - beta2) * grad^2
   * - m_hat = m / (1 - beta1^t)
   * - v_hat = v / (1 - beta2^t)
   * - param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
   *
   * All state tensors (param, m, v) are modified in place.
   *
   * @param param Parameter tensor (modified in place)
   * @param grad Gradient tensor
   * @param m First moment estimate (modified in place)
   * @param v Second moment estimate (modified in place)
   * @param lr Learning rate
   * @param beta1 First moment decay rate (default 0.9)
   * @param beta2 Second moment decay rate (default 0.999)
   * @param epsilon Small constant for numerical stability (default 1e-8)
   * @param t Timestep (for bias correction)
   */
  void AdamUpdate(CAIF_DeviceTensor &param,
                  const CAIF_DeviceTensor &grad,
                  CAIF_DeviceTensor &m,
                  CAIF_DeviceTensor &v,
                  float lr,
                  float beta1,
                  float beta2,
                  float epsilon,
                  float weight_decay,
                  int t);

  //----------------------------------------------------------------------------
  // Additional Element-wise Operations
  //----------------------------------------------------------------------------

  /**
   * @brief Element-wise multiplication: output = a * b
   *
   * @param a First input tensor
   * @param b Second input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void Multiply(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief Scale tensor with output: output = input * scale
   *
   * @param input Input tensor
   * @param scale Scalar multiplier
   * @param output Output tensor (must be pre-allocated)
   */
  void Scale(const CAIF_DeviceTensor &input,float scale,CAIF_DeviceTensor &output);

  /**
   * @brief SiLU (Swish) activation: output = x * sigmoid(x)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  void SiLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief SiLU backward pass
   *
   * @param input Original input to SiLU
   * @param grad_output Gradient from the next layer
   * @param grad_input Output gradient (must be pre-allocated)
   */
  void SiLUBackward(const CAIF_DeviceTensor &input,
                    const CAIF_DeviceTensor &grad_output,
                    CAIF_DeviceTensor &grad_input);

  /**
   * @brief Add bias to tensor (alias for BiasAdd)
   *
   * @param input Input tensor [batch x units]
   * @param bias Bias tensor [units]
   * @param output Output tensor (must be pre-allocated)
   */
  void AddBias(const CAIF_DeviceTensor &input,const CAIF_DeviceTensor &bias,CAIF_DeviceTensor &output);

  //----------------------------------------------------------------------------
  // Reduction Operations (tensor output)
  //----------------------------------------------------------------------------

  /**
   * @brief Sum along specified axis
   *
   * @param input Input tensor
   * @param axis Axis to sum along
   * @param output Output tensor (must be pre-allocated)
   */
  void SumAxis(const CAIF_DeviceTensor &input,uint32_t axis,CAIF_DeviceTensor &output);

  /**
   * @brief Sum all elements to a single-element tensor
   *
   * @param input Input tensor
   * @param output Single-element output tensor (must be pre-allocated)
   */
  void Sum(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Log-sum-exp along last axis: output[i] = log(sum(exp(input[i,:])))
   *
   * @param input Input tensor [batch x dim]
   * @param output Output tensor [batch] (must be pre-allocated)
   */
  void LogSumExp(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  //----------------------------------------------------------------------------
  // Top-K and Scatter Operations
  //----------------------------------------------------------------------------

  /**
   * @brief Select top-k values and indices along last axis
   *
   * @param input Input tensor [batch x dim]
   * @param k Number of top values to select
   * @param indices Output indices tensor [batch x k] (int32, must be pre-allocated)
   * @param values Output values tensor [batch x k] (must be pre-allocated)
   */
  void TopK(const CAIF_DeviceTensor &input,
            uint32_t k,
            CAIF_DeviceTensor &indices,
            CAIF_DeviceTensor &values);

  /**
   * @brief Normalize each row to sum to 1
   *
   * @param input Input tensor [batch x dim]
   * @param output Output tensor (must be pre-allocated)
   */
  void NormalizeRows(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Scatter-add values to output using indices
   *
   * For each position, adds values[i,j] to output[i, indices[i,j]]
   *
   * @param values Values tensor [batch x k]
   * @param indices Indices tensor [batch x k] (int32)
   * @param output Output tensor [batch x dim] (must be pre-allocated and zeroed)
   */
  void ScatterAdd(const CAIF_DeviceTensor &values,
                  const CAIF_DeviceTensor &indices,
                  CAIF_DeviceTensor &output);

  //----------------------------------------------------------------------------
  // MoE-specific Operations
  //----------------------------------------------------------------------------

  /**
   * @brief Dispatch tokens to experts based on routing indices
   *
   * Gathers tokens for each expert based on the routing decisions.
   *
   * @param input Input tokens [num_tokens x dim]
   * @param expert_indices Routing indices [num_tokens x top_k] (int32)
   * @param top_k Number of experts per token
   * @param token_counts Number of tokens per expert
   * @param expert_inputs Output: vector of [tokens_for_expert x dim] tensors
   */
  void MoEDispatch(const CAIF_DeviceTensor &input,
                   const CAIF_DeviceTensor &expert_indices,
                   uint32_t top_k,
                   const std::vector<uint32_t> &token_counts,
                   std::vector<CAIF_DeviceTensor> &expert_inputs);

  /**
   * @brief Combine expert outputs back to original token positions
   *
   * Scatter-adds weighted expert outputs to the original token positions.
   *
   * @param expert_outputs Vector of [tokens_for_expert x dim] tensors
   * @param expert_indices Routing indices [num_tokens x top_k] (int32)
   * @param expert_weights Routing weights [num_tokens x top_k]
   * @param top_k Number of experts per token
   * @param token_counts Number of tokens per expert
   * @param output Combined output [num_tokens x dim]
   */
  void MoECombine(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                  const CAIF_DeviceTensor &expert_indices,
                  const CAIF_DeviceTensor &expert_weights,
                  uint32_t top_k,
                  const std::vector<uint32_t> &token_counts,
                  CAIF_DeviceTensor &output);

  /**
   * @brief Backward pass for MoE combine operation
   *
   * Distributes gradients back to expert outputs and routing weights.
   *
   * @param grad_output Gradient w.r.t. combined output [num_tokens x dim]
   * @param expert_outputs Forward pass expert outputs (for weight gradient)
   * @param expert_indices Routing indices [num_tokens x top_k] (int32)
   * @param expert_weights Routing weights [num_tokens x top_k]
   * @param top_k Number of experts per token
   * @param token_counts Number of tokens per expert
   * @param grad_expert_outputs Output: gradients for each expert
   * @param grad_weights Output: gradient w.r.t. routing weights
   */
  void MoECombineBackward(const CAIF_DeviceTensor &grad_output,
                          const std::vector<CAIF_DeviceTensor> &expert_outputs,
                          const CAIF_DeviceTensor &expert_indices,
                          const CAIF_DeviceTensor &expert_weights,
                          uint32_t top_k,
                          const std::vector<uint32_t> &token_counts,
                          std::vector<CAIF_DeviceTensor> &grad_expert_outputs,
                          CAIF_DeviceTensor &grad_weights);

  /**
   * @brief Backward pass for MoE dispatch operation
   *
   * Combines gradients from experts back to original input positions.
   *
   * @param grad_expert_inputs Gradients from each expert
   * @param expert_indices Routing indices [num_tokens x top_k] (int32)
   * @param top_k Number of experts per token
   * @param token_counts Number of tokens per expert
   * @param grad_input Output: gradient w.r.t. input [num_tokens x dim]
   */
  void MoEDispatchBackward(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
                           const CAIF_DeviceTensor &expert_indices,
                           uint32_t top_k,
                           const std::vector<uint32_t> &token_counts,
                           CAIF_DeviceTensor &grad_input);

  //----------------------------------------------------------------------------
  // GPU-Optimized MoE Operations (Phase 6)
  //
  // These functions use contiguous expert buffers for better GPU utilization.
  // The contiguous buffer approach enables:
  // - Coalesced memory access patterns
  // - Single kernel launch for dispatch/combine
  // - Reduced memory allocation overhead
  //----------------------------------------------------------------------------

  /**
   * @brief GPU-optimized fused softmax + top-k gating for MoE router
   *
   * Performs softmax over experts and selects top-k experts per token.
   * More efficient than separate Softmax + TopK calls.
   *
   * @param router_logits Raw router output [num_tokens x num_experts]
   * @param num_experts Total number of experts
   * @param top_k Number of experts per token
   * @param expert_indices Output: selected expert indices [num_tokens x top_k] (floats)
   * @param expert_weights Output: normalized routing weights [num_tokens x top_k]
   * @param router_probs Output: full softmax probs [num_tokens x num_experts] (for aux loss)
   */
  void MoETopKGating(const CAIF_DeviceTensor &router_logits,
                     uint32_t num_experts,
                     uint32_t top_k,
                     CAIF_DeviceTensor &expert_indices,
                     CAIF_DeviceTensor &expert_weights,
                     CAIF_DeviceTensor &router_probs);

  /**
   * @brief Count tokens per expert for GPU MoE dispatch
   *
   * Uses atomic operations to count how many tokens are routed to each expert.
   *
   * @param expert_indices Expert indices [num_tokens x top_k] (floats)
   * @param num_experts Total number of experts
   * @param top_k Number of experts per token
   * @param expert_counts Output: count per expert [num_experts] (int32)
   */
  void MoECountPerExpert(const CAIF_DeviceTensor &expert_indices,
                         uint32_t num_experts,
                         uint32_t top_k,
                         CAIF_DeviceTensor &expert_counts);

  /**
   * @brief GPU dispatch with contiguous expert buffer
   *
   * Gathers tokens into a single contiguous buffer organized by expert.
   * Buffer layout: [expert_0_tokens..., expert_1_tokens..., ...]
   *
   * @param input Input tokens [num_tokens x dim]
   * @param expert_indices Expert indices [num_tokens x top_k] (floats)
   * @param dispatch_map Position within expert buffer [num_tokens x top_k] (int32)
   * @param expert_offsets Cumulative counts [num_experts+1] (int32)
   * @param top_k Number of experts per token
   * @param expert_buffer Output: contiguous buffer [total_assigned x dim]
   */
  void MoEDispatchGPU(const CAIF_DeviceTensor &input,
                      const CAIF_DeviceTensor &expert_indices,
                      const CAIF_DeviceTensor &dispatch_map,
                      const CAIF_DeviceTensor &expert_offsets,
                      uint32_t top_k,
                      CAIF_DeviceTensor &expert_buffer);

  /**
   * @brief GPU combine with contiguous expert buffer
   *
   * Scatters and combines expert outputs from contiguous buffer to token positions.
   *
   * @param expert_buffer Expert outputs in contiguous buffer [total_assigned x dim]
   * @param expert_indices Expert indices [num_tokens x top_k] (floats)
   * @param expert_weights Routing weights [num_tokens x top_k]
   * @param dispatch_map Position within expert buffer [num_tokens x top_k] (int32)
   * @param expert_offsets Cumulative counts [num_experts+1] (int32)
   * @param top_k Number of experts per token
   * @param output Combined output [num_tokens x dim]
   */
  void MoECombineGPU(const CAIF_DeviceTensor &expert_buffer,
                     const CAIF_DeviceTensor &expert_indices,
                     const CAIF_DeviceTensor &expert_weights,
                     const CAIF_DeviceTensor &dispatch_map,
                     const CAIF_DeviceTensor &expert_offsets,
                     uint32_t top_k,
                     CAIF_DeviceTensor &output);

  /**
   * @brief Build dispatch map for GPU MoE operations
   *
   * Creates the dispatch_map and expert_offsets tensors needed for GPU dispatch/combine.
   * This function must run on CPU since it builds position assignments.
   *
   * @param expert_indices Expert indices [num_tokens x top_k] (floats)
   * @param num_experts Total number of experts
   * @param top_k Number of experts per token
   * @param capacity_per_expert Maximum tokens per expert (0 = no limit)
   * @param dispatch_map Output: position within expert buffer [num_tokens x top_k] (int32)
   * @param expert_offsets Output: cumulative counts [num_experts+1] (int32)
   * @return Total number of assigned tokens (sum of min(count, capacity) per expert)
   */
  uint32_t MoEBuildDispatchMap(const CAIF_DeviceTensor &expert_indices,
                               uint32_t num_experts,
                               uint32_t top_k,
                               uint32_t capacity_per_expert,
                               CAIF_DeviceTensor &dispatch_map,
                               CAIF_DeviceTensor &expert_offsets);

}//end CAIF_DeviceOps namespace

}//end instance namespace

#endif  // CAIF_DEVICE_OPS_H
