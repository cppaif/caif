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
// Tensor operations (device + host backends, dispatched on tensor location)
//------------------------------------------------------------------------------
#ifndef CAIF_OPS_H
#define CAIF_OPS_H

#include "caif_device_tensor.h"
#include <cstdint>
#include <vector>
#include "caif_base.h"

#ifndef USE_CAIF_CUDA
typedef void *cudaStream_t;
#endif

namespace instance
{


class CAIF_RunContext;


/**
 * @brief Tensor operations — all-static utility class.
 *
 * Public API entries (CAIF_Ops::MatMul etc.) dispatch on the input
 * tensors' location and call the matching Device/Host backend
 * implementation. Layer code is written once and runs on either
 * backend. Every op takes a CAIF_RunContext &ctx.
 *
 * Pure utility class — never instantiated. All methods are public
 * static. Inherits CAIF_Base for framework-wide logging integration.
 */
class CAIF_Ops:public CAIF_Base
{
  public:
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
  static void MatMul(const CAIF_DeviceTensor &a,
              const CAIF_DeviceTensor &b,
              CAIF_DeviceTensor &output,
              CAIF_RunContext &ctx,
              const CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32);

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
  static void MatMulTransposeA(const CAIF_DeviceTensor &a,
                        const CAIF_DeviceTensor &b,
                        CAIF_DeviceTensor &output,
                        CAIF_RunContext &ctx,
                        const CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32);

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
  static void MatMulTransposeB(const CAIF_DeviceTensor &a,
                        const CAIF_DeviceTensor &b,
                        CAIF_DeviceTensor &output,
                        CAIF_RunContext &ctx,
                        const CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32);

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
  static void BatchedMatMul(const CAIF_DeviceTensor &a,
                     const CAIF_DeviceTensor &b,
                     CAIF_DeviceTensor &output,
                     int m,
                     int k,
                     int n,
                     int batch_count,
                     CAIF_RunContext &ctx,
                     const CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32);

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
  static void BatchedMatMulTransposeA(const CAIF_DeviceTensor &a,
                               const CAIF_DeviceTensor &b,
                               CAIF_DeviceTensor &output,
                               int k,
                               int m,
                               int n,
                               int batch_count,
                               CAIF_RunContext &ctx,
                               const CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32);

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
  static void BatchedMatMulTransposeB(const CAIF_DeviceTensor &a,
                               const CAIF_DeviceTensor &b,
                               CAIF_DeviceTensor &output,
                               int m,
                               int k,
                               int n,
                               int batch_count,
                               CAIF_RunContext &ctx,
                               const CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32);

  //----------------------------------------------------------------------------
  // Tensor Manipulation
  //----------------------------------------------------------------------------

  /**
   * @brief Transpose a 2D tensor: output = input^T
   *
   * @param input Input tensor [M x N]
   * @param output Output tensor [N x M] (must be pre-allocated)
   */
  static void Transpose(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

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
  static void Add(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief In-place scalar multiplication: tensor = tensor * scale
   *
   * Modifies the tensor in place.
   *
   * @param tensor Tensor to scale (modified in place)
   * @param scale Scalar multiplier
   */
  static void Scale(CAIF_DeviceTensor &tensor,float scale);

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
  static void AddScaled(CAIF_DeviceTensor &target,const CAIF_DeviceTensor &source,float scale);

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
  static void BiasAdd(const CAIF_DeviceTensor &input,const CAIF_DeviceTensor &bias,CAIF_DeviceTensor &output);

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
  static void MatMulBias(const CAIF_DeviceTensor &a,
                  const CAIF_DeviceTensor &b,
                  const CAIF_DeviceTensor &bias,
                  CAIF_DeviceTensor &output,
                  cudaStream_t stream,
                  CAIF_RunContext &ctx,
                  const CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32);

  /**
   * @brief Compute bias gradient from 2D gradient tensor
   *
   * Computes bias_grad[u] = sum over batch of grad[b][u]
   *
   * @param grad Gradient tensor [batch x units]
   * @param bias_grad Output bias gradient [units] (must be pre-allocated)
   */
  static void BiasGradient(const CAIF_DeviceTensor &grad,CAIF_DeviceTensor &bias_grad);

  /**
   * @brief Add positional encoding table rows to input (broadcast over batch).
   *
   * output[b,s,d] = input[b,s,d] + pe_table[s,d]
   *
   * @param input 3D tensor [batch, seq_len, dim] (or 2D [seq_len, dim])
   * @param pe_table 2D table [seq_len, dim]; rows 0..seq_len-1 of
   *                 [max_seq_len, dim] are used
   * @param output Output tensor, same shape as input (pre-allocated)
   */
  static void AddPositionalEncoding(const CAIF_DeviceTensor &input,
                             const CAIF_DeviceTensor &pe_table,
                             CAIF_DeviceTensor &output);

  /**
   * @brief Accumulate positional encoding gradient from per-batch gradients.
   *
   * grad_table[s,d] = sum_b grad_output[b,s,d]
   * Sum accumulates in float for numerical stability at fp16/bf16.
   *
   * @param grad_output 3D gradient tensor [batch, seq_len, dim]
   * @param grad_table 2D table gradient [seq_len, dim] (written, not accumulated)
   */
  static void PositionalEncodingBackward(const CAIF_DeviceTensor &grad_output,
                                  CAIF_DeviceTensor &grad_table);

  /**
   * @brief Compute T5-style relative position bias matrix.
   *
   * Embedding table is always fp32 (autocast convention); output dtype T
   * (float / __half / __nv_bfloat16) is dispatched on output.Dtype().
   *
   * @param embedding 2D table [num_heads, num_buckets], must be Float32
   * @param output 3D bias tensor [num_heads, q_len, k_len] (pre-allocated)
   * @param max_distance T5 log-bucketing clamp
   * @param bidirectional true for encoder-style (bucket includes negative distances)
   */
  static void ComputeRelativePositionBias(const CAIF_DeviceTensor &embedding,
                                   CAIF_DeviceTensor &output,
                                   uint32_t max_distance,
                                   bool bidirectional);

  /**
   * @brief Accumulate gradient into relative position bias embedding table.
   *
   * grad_embedding stays fp32 so atomicAdd is always on float (sm_75 safe);
   * grad_output dtype T is up-cast during atomicAdd.
   *
   * @param grad_output 3D grad tensor [num_heads, q_len, k_len]
   * @param grad_embedding 2D accumulator [num_heads, num_buckets], Float32
   * @param max_distance T5 log-bucketing clamp
   * @param bidirectional true for encoder-style
   */
  static void AccumulateRelativePositionBiasGradient(const CAIF_DeviceTensor &grad_output,
                                              CAIF_DeviceTensor &grad_embedding,
                                              uint32_t max_distance,
                                              bool bidirectional);

  //----------------------------------------------------------------------------
  // Quantization / dequantization (weight-only, training via fp32 shadow)
  //----------------------------------------------------------------------------

  /**
   * @brief Quantization scheme for INT8 quantize/dequantize.
   */
  enum class QuantScheme_e:uint8_t
  {
    PerTensor_e,
    PerChannel_e
  };

  /**
   * @brief Symmetric INT8 quantization from fp32 input.
   *
   * Writes quantized bytes into `output` (dtype Int8, same shape as input)
   * and scales into `scales` (dtype Float32). For PerTensor, scales is
   * shape {1}. For PerChannel, scales is shape {cols} where cols is the
   * last dim of a 2D input.
   *
   * Input must be fp32; output must be pre-allocated Int8 with matching
   * shape; scales must be pre-allocated Float32 with the shape above.
   */
  static void QuantizeInt8(const CAIF_DeviceTensor &input,
                    CAIF_DeviceTensor &output,
                    CAIF_DeviceTensor &scales,
                    QuantScheme_e scheme,
                    CAIF_RunContext &ctx);

  /**
   * @brief Dequantize INT8 back to fp32 using stored scales.
   */
  static void DequantizeInt8(const CAIF_DeviceTensor &input,
                      CAIF_DeviceTensor &output,
                      const CAIF_DeviceTensor &scales,
                      QuantScheme_e scheme,
                      CAIF_RunContext &ctx);

  /**
   * @brief Symmetric per-group INT4 quantization from fp32 input.
   *
   * `output` holds packed uint8 bytes (2 values per byte, low nibble first).
   * `scales` is fp16 with shape {num_groups} where num_groups = ceil(n/group_size).
   */
  static void QuantizeInt4PerGroup(const CAIF_DeviceTensor &input,
                            CAIF_DeviceTensor &output,
                            CAIF_DeviceTensor &scales,
                            uint32_t group_size,
                            CAIF_RunContext &ctx);

  /** Dequantize per-group INT4 back to fp32. */
  static void DequantizeInt4PerGroup(const CAIF_DeviceTensor &input,
                              CAIF_DeviceTensor &output,
                              const CAIF_DeviceTensor &scales,
                              uint32_t group_size,
                              CAIF_RunContext &ctx);

  /**
   * @brief QAT fake-quantization for INT8 (FP32 -> FP32 round-trip).
   *
   * Simulates INT8 quantization noise in the forward pass: quantizes input
   * to INT8 using the given scheme, then dequantizes back to FP32 in one
   * shot. Output is FP32 with the same shape as input. No external scales
   * tensor is required — scales are computed on-the-fly and discarded.
   *
   * Backward is straight-through: gradient flows through unchanged. Caller
   * is responsible for wiring STE (by not treating this op as differentiable
   * in the graph), or by composing it inline before a differentiable op.
   *
   * For PerChannel, input must be 2D (scales computed along last dim).
   */
  static void FakeQuantInt8(const CAIF_DeviceTensor &input,
                     CAIF_DeviceTensor &output,
                     QuantScheme_e scheme,
                     CAIF_RunContext &ctx);

  /**
   * @brief QAT fake-quantization for per-group INT4 (FP32 -> FP32 round-trip).
   *
   * Simulates INT4 quantization noise in the forward pass using symmetric
   * per-group scaling with FP16 scales. Input and output are both FP32 with
   * matching shape; intermediate packed INT4 and FP16 scales are allocated
   * and discarded inside the call.
   */
  static void FakeQuantInt4PerGroup(const CAIF_DeviceTensor &input,
                             CAIF_DeviceTensor &output,
                             uint32_t group_size,
                             CAIF_RunContext &ctx);

  /**
   * @brief Element-wise dtype cast between floating-point types.
   *
   * Writes a cast copy of `input` into pre-allocated `output`. Input and
   * output must share shape. Supported pairs:
   *   fp32 <-> fp16
   *   fp32 <-> bf16
   *   fp16 <-> bf16 (via fp32 intermediate)
   *   fp32 <-> fp32 (identity copy)
   * Integer casts (int8/int4) go through CAIF_Ops::Quantize /
   * CAIF_Ops::Dequantize which require scale/zero-point metadata and are
   * therefore not valid Cast pairs.
   *
   * @param input  source tensor (any floating-point dtype)
   * @param output pre-allocated destination tensor (floating-point dtype)
   * @param ctx    run context carrying the stream
   */
  static void Cast(const CAIF_DeviceTensor &input,
            CAIF_DeviceTensor &output,
            CAIF_RunContext &ctx);

  /**
   * @brief Slice a contiguous column range from a 2D (logically) tensor.
   *
   * input/output must share dtype (Float32/Float16/BFloat16).
   * Copies columns [col_start, col_start+out_cols) from each row.
   *
   * @param input [rows, in_cols]
   * @param output [rows, out_cols] (must be pre-allocated, same dtype as input)
   * @param col_start starting column offset
   */
  static void SliceLastDim(const CAIF_DeviceTensor &input,
                    CAIF_DeviceTensor &output,
                    uint32_t col_start);

  /**
   * @brief Scatter-add a slice back into the last dimension (backward of slice).
   *
   * grad_input must be pre-zeroed or pre-filled; grad_output is added into
   * grad_input[:, col_start:col_start+out_cols].
   *
   * @param grad_output [rows, out_cols]
   * @param grad_input [rows, in_cols] (same dtype as grad_output)
   * @param col_start starting column offset
   */
  static void SliceLastDimBackward(const CAIF_DeviceTensor &grad_output,
                            CAIF_DeviceTensor &grad_input,
                            uint32_t col_start);

  /**
   * @brief Concatenate two tensors along the last dimension.
   *
   * a/b/output must share dtype. output shape: [rows, cols_a+cols_b].
   *
   * @param a [rows, cols_a]
   * @param b [rows, cols_b]
   * @param output [rows, cols_a+cols_b] (must be pre-allocated, same dtype)
   */
  static void ConcatLastDim(const CAIF_DeviceTensor &a,
                     const CAIF_DeviceTensor &b,
                     CAIF_DeviceTensor &output);

  //----------------------------------------------------------------------------
  // Activation Functions (Forward)
  //----------------------------------------------------------------------------

  /**
   * @brief ReLU activation: output = max(0, input)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  static void ReLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Sigmoid activation: output = 1 / (1 + exp(-input))
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  static void Sigmoid(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Tanh activation: output = tanh(input)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  static void Tanh(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Softmax activation: output[i] = exp(input[i]) / sum(exp(input))
   *
   * Applies softmax along the last dimension (columns) for each row.
   * For 2D input [batch x classes], computes softmax over classes for each sample.
   *
   * @param input Input tensor [batch x classes]
   * @param output Output tensor (must be pre-allocated) [batch x classes]
   */
  static void Softmax(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Leaky ReLU activation: output = max(alpha * input, input)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   * @param alpha Negative slope (default 0.01)
   */
  static void LeakyReLU(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output,
                 float alpha=0.01f);

  /**
   * @brief ELU activation: output = x if x > 0, else alpha * (exp(x) - 1)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   * @param alpha Scale for negative values (default 1.0)
   */
  static void ELU(const CAIF_DeviceTensor &input,
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
  static void GELU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Swish activation: output = x * sigmoid(x)
   *
   * Also known as SiLU (Sigmoid Linear Unit).
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  static void Swish(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

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
  static void ReLUBackward(const CAIF_DeviceTensor &grad_output,
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
  static void SigmoidBackward(const CAIF_DeviceTensor &grad_output,
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
  static void TanhBackward(const CAIF_DeviceTensor &grad_output,
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
  static void SoftmaxBackward(const CAIF_DeviceTensor &grad_output,
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
  static void LeakyReLUBackward(const CAIF_DeviceTensor &grad_output,
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
  static void ELUBackward(const CAIF_DeviceTensor &grad_output,
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
  static void GELUBackward(const CAIF_DeviceTensor &grad_output,
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
  static void SwishBackward(const CAIF_DeviceTensor &grad_output,
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
  static float ReduceSum(const CAIF_DeviceTensor &tensor);

  /**
   * @brief Compute mean of all elements in the tensor
   *
   * WARNING: This operation synchronizes the stream and returns a scalar.
   * It is a sync point in the computation graph.
   *
   * @param tensor Input tensor
   * @return Mean of all elements
   */
  static float ReduceMean(const CAIF_DeviceTensor &tensor);

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
  static void MSELoss(const CAIF_DeviceTensor &pred,const CAIF_DeviceTensor &target,CAIF_DeviceTensor &loss);

  /**
   * @brief MSE loss backward: grad = 2 * (pred - target) / n
   *
   * @param pred Prediction tensor
   * @param target Target tensor
   * @param grad Output gradient tensor (must be pre-allocated, same shape as pred)
   */
  static void MSELossBackward(const CAIF_DeviceTensor &pred,
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
  static void AdamUpdate(CAIF_DeviceTensor &param,
                  const CAIF_DeviceTensor &grad,
                  CAIF_DeviceTensor &m,
                  CAIF_DeviceTensor &v,
                  float lr,
                  float beta1,
                  float beta2,
                  float epsilon,
                  float weight_decay,
                  int t);

  /**
   * @brief Plain SGD update: param = param - lr * (grad + weight_decay * param)
   */
  static void SgdUpdate(CAIF_DeviceTensor &param,
                 const CAIF_DeviceTensor &grad,
                 float lr,
                 float weight_decay);

  /**
   * @brief SGD with momentum: velocity = momentum*velocity + (grad + wd*param);
   *        param = param - lr * velocity
   */
  static void MomentumUpdate(CAIF_DeviceTensor &param,
                      const CAIF_DeviceTensor &grad,
                      CAIF_DeviceTensor &velocity,
                      float lr,
                      float momentum,
                      float weight_decay);

  /**
   * @brief RMSprop: avg_sq = alpha*avg_sq + (1-alpha)*grad^2;
   *        param = param - lr * (grad + wd*param) / (sqrt(avg_sq) + epsilon)
   */
  static void RmspropUpdate(CAIF_DeviceTensor &param,
                     const CAIF_DeviceTensor &grad,
                     CAIF_DeviceTensor &avg_sq,
                     float lr,
                     float alpha,
                     float epsilon,
                     float weight_decay);

  /**
   * @brief AdaGrad: accum += grad^2;
   *        param = param - lr * (grad + wd*param) / (sqrt(accum) + epsilon)
   */
  static void AdaGradUpdate(CAIF_DeviceTensor &param,
                     const CAIF_DeviceTensor &grad,
                     CAIF_DeviceTensor &accum,
                     float lr,
                     float epsilon,
                     float weight_decay);

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
  static void Multiply(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief Scale tensor with output: output = input * scale
   *
   * @param input Input tensor
   * @param scale Scalar multiplier
   * @param output Output tensor (must be pre-allocated)
   */
  static void Scale(const CAIF_DeviceTensor &input,float scale,CAIF_DeviceTensor &output);

  /**
   * @brief SiLU (Swish) activation: output = x * sigmoid(x)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  static void SiLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief SiLU backward pass
   *
   * @param input Original input to SiLU
   * @param grad_output Gradient from the next layer
   * @param grad_input Output gradient (must be pre-allocated)
   */
  static void SiLUBackward(const CAIF_DeviceTensor &input,
                    const CAIF_DeviceTensor &grad_output,
                    CAIF_DeviceTensor &grad_input);

  /**
   * @brief Add bias to tensor (alias for BiasAdd)
   *
   * @param input Input tensor [batch x units]
   * @param bias Bias tensor [units]
   * @param output Output tensor (must be pre-allocated)
   */
  static void AddBias(const CAIF_DeviceTensor &input,const CAIF_DeviceTensor &bias,CAIF_DeviceTensor &output);

  /**
   * @brief Element-wise addition with scalar: output = input + scalar
   *
   * @param input Input tensor
   * @param scalar Scalar addend
   * @param output Output tensor (must be pre-allocated)
   */
  static void AddScalar(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output);

  /**
   * @brief Element-wise subtraction: output = a - b
   *
   * @param a First input tensor
   * @param b Second input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  static void Subtract(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief Element-wise subtraction with scalar: output = input - scalar
   *
   * @param input Input tensor
   * @param scalar Scalar subtrahend
   * @param output Output tensor (must be pre-allocated)
   */
  static void SubtractScalar(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output);

  /**
   * @brief Element-wise division: output = a / b
   *
   * @param a First input tensor (numerator)
   * @param b Second input tensor (denominator)
   * @param output Output tensor (must be pre-allocated)
   */
  static void Divide(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output);

  /**
   * @brief Element-wise division by scalar: output = input / scalar
   *
   * @param input Input tensor (numerator)
   * @param scalar Scalar denominator
   * @param output Output tensor (must be pre-allocated)
   */
  static void DivideScalar(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output);

  /**
   * @brief Element-wise square root: output = sqrt(input)
   *
   * @param input Input tensor
   * @param output Output tensor (must be pre-allocated)
   */
  static void Sqrt(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

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
  static void SumAxis(const CAIF_DeviceTensor &input,uint32_t axis,CAIF_DeviceTensor &output);

  /**
   * @brief Sum all elements to a single-element tensor
   *
   * @param input Input tensor
   * @param output Single-element output tensor (must be pre-allocated)
   */
  static void Sum(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief Log-sum-exp along last axis: output[i] = log(sum(exp(input[i,:])))
   *
   * @param input Input tensor [batch x dim]
   * @param output Output tensor [batch] (must be pre-allocated)
   */
  static void LogSumExp(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

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
  static void TopK(const CAIF_DeviceTensor &input,
            uint32_t k,
            CAIF_DeviceTensor &indices,
            CAIF_DeviceTensor &values);

  /**
   * @brief Normalize each row to sum to 1
   *
   * @param input Input tensor [batch x dim]
   * @param output Output tensor (must be pre-allocated)
   */
  static void NormalizeRows(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);

  /**
   * @brief NormalizeRows backward Jacobian on top-k rows, gather variant.
   *
   * Gathers p[k] = probs[t, indices[t,k]] from the full softmax cache and
   * computes s/w/dot/grad_p_topk inline — no separate w_norm/row_sum caches
   * are required beyond the probs+indices the router already keeps for the
   * softmax backward.
   *
   * @param grad_w       Upstream grads wrt normalized weights [N x K]
   * @param probs        Cached softmax probabilities           [N x E]
   * @param indices      Top-k expert indices (float-encoded)   [N x K]
   * @param grad_p_topk  Output grads in top-k expert space     [N x K]
   *
   * top_k must be <= 32.
   */
  static void NormalizeRowsBackwardTopKGather(const CAIF_DeviceTensor &grad_w,
                                       const CAIF_DeviceTensor &probs,
                                       const CAIF_DeviceTensor &indices,
                                       CAIF_DeviceTensor &grad_p_topk);

  /**
   * @brief Gather per-row scores at top-k indices
   *
   * out[t,k] = scores[t, indices[t,k]]
   *
   * Used by SigmoidNoauxTc routing (HF noaux_tc) Phase 1b: selection
   * happens on bias-corrected sigmoid scores (CAIF_Ops::AddBias +
   * CAIF_Ops::TopK), but the combine weights must be the ORIGINAL
   * (uncorrected) sigmoid scores at the chosen indices to match HF
   * `topk_method=noaux_tc`.
   *
   * @param scores  Per-row scores [num_tokens x num_experts]
   * @param indices Chosen indices [num_tokens x top_k] (int32)
   * @param out     Output gathered values [num_tokens x top_k], same dtype as scores
   *
   * top_k must be <= 32.
   */
  static void GatherTopKValues(const CAIF_DeviceTensor &scores,
                        const CAIF_DeviceTensor &indices,
                        CAIF_DeviceTensor &out);

  /**
   * @brief Scatter-add values to output using indices
   *
   * For each position, adds values[i,j] to output[i, indices[i,j]]
   *
   * @param values Values tensor [batch x k]
   * @param indices Indices tensor [batch x k] (int32)
   * @param output Output tensor [batch x dim] (must be pre-allocated and zeroed)
   */
  static void ScatterAdd(const CAIF_DeviceTensor &values,
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
  static void MoEDispatch(const CAIF_DeviceTensor &input,
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
  static void MoECombine(const std::vector<CAIF_DeviceTensor> &expert_outputs,
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
  static void MoECombineBackward(const CAIF_DeviceTensor &grad_output,
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
  static void MoEDispatchBackward(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
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
  static void MoETopKGating(const CAIF_DeviceTensor &router_logits,
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
  static void MoECountPerExpert(const CAIF_DeviceTensor &expert_indices,
                         uint32_t num_experts,
                         uint32_t top_k,
                         CAIF_DeviceTensor &expert_counts);

  /**
   * @brief ST-MoE router z-loss gradient contribution (in-place add).
   *
   * grad_logits[t,e] += logsumexp_scaled[t] * probs[t,e]
   *
   * logsumexp_scaled must be pre-multiplied by (2 * z_loss_weight / N) so this
   * op is a single [N, E] row-broadcast multiply-add.
   *
   * @param logsumexp_scaled Pre-scaled per-token logsumexp [num_tokens]
   * @param probs Router softmax probabilities [num_tokens x num_experts]
   * @param grad_logits Gradient w.r.t. router logits [num_tokens x num_experts] (in-place)
   */
  static void MoEZLossGradAdd(const CAIF_DeviceTensor &logsumexp_scaled,
                       const CAIF_DeviceTensor &probs,
                       CAIF_DeviceTensor &grad_logits);

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
  static void MoEDispatchGPU(const CAIF_DeviceTensor &input,
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
  static void MoECombineGPU(const CAIF_DeviceTensor &expert_buffer,
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
  static uint32_t MoEBuildDispatchMap(const CAIF_DeviceTensor &expert_indices,
                               uint32_t num_experts,
                               uint32_t top_k,
                               uint32_t capacity_per_expert,
                               CAIF_DeviceTensor &dispatch_map,
                               CAIF_DeviceTensor &expert_offsets);

  /**
   * @brief GPU backward pass for MoE combine
   *
   * Computes grad_expert_buffer and grad_weights from grad_output given the
   * dispatch map and expert buffer used in the forward combine.
   *
   * @param grad_output Incoming gradient [num_tokens x dim]
   * @param expert_buffer Expert outputs from forward [total_assigned x dim]
   * @param expert_indices Expert indices [num_tokens x top_k] (floats)
   * @param expert_weights Routing weights [num_tokens x top_k]
   * @param dispatch_map Position within expert buffer [num_tokens x top_k] (int32)
   * @param expert_offsets Cumulative counts [num_experts+1] (int32)
   * @param top_k Number of experts per token
   * @param grad_expert_buffer Output: gradient w.r.t. expert outputs [total_assigned x dim]
   * @param grad_weights Output: gradient w.r.t. routing weights [num_tokens x top_k]
   */
  static void MoECombineBackwardGPU(const CAIF_DeviceTensor &grad_output,
                             const CAIF_DeviceTensor &expert_buffer,
                             const CAIF_DeviceTensor &expert_indices,
                             const CAIF_DeviceTensor &expert_weights,
                             const CAIF_DeviceTensor &dispatch_map,
                             const CAIF_DeviceTensor &expert_offsets,
                             uint32_t top_k,
                             CAIF_DeviceTensor &grad_expert_buffer,
                             CAIF_DeviceTensor &grad_weights);

  /**
   * @brief GPU backward pass for MoE dispatch
   *
   * Gathers gradients from the contiguous expert buffer back to per-token input gradients.
   *
   * @param grad_expert_buffer Gradient w.r.t. expert inputs [total_assigned x dim]
   * @param expert_indices Expert indices [num_tokens x top_k] (floats)
   * @param dispatch_map Position within expert buffer [num_tokens x top_k] (int32)
   * @param expert_offsets Cumulative counts [num_experts+1] (int32)
   * @param top_k Number of experts per token
   * @param grad_input Output: gradient w.r.t. input [num_tokens x dim]
   */
  static void MoEDispatchBackwardGPU(const CAIF_DeviceTensor &grad_expert_buffer,
                              const CAIF_DeviceTensor &expert_indices,
                              const CAIF_DeviceTensor &dispatch_map,
                              const CAIF_DeviceTensor &expert_offsets,
                              uint32_t top_k,
                              CAIF_DeviceTensor &grad_input);


    //----------------------------------------------------------------------------
    // Internal backend helpers (called by the public dispatch entries above).
    // Public for in-class access from caif_ops.cpp / caif_ops_device.cpp /
    // caif_ops_host.cpp; not part of the API consumers should call.
    //----------------------------------------------------------------------------


  //----------------------------------------------------------------------------
  // Shared helpers (defined in caif_ops.cpp)
  //----------------------------------------------------------------------------

  static void RequireSameLocation(const CAIF_DeviceTensor &a,
                           const CAIF_DeviceTensor &b,
                           const CAIF_DeviceTensor &c,
                           const char *op_name);
  static void RequireSameLocation(const CAIF_DeviceTensor &a,
                           const CAIF_DeviceTensor &b,
                           const char *op_name);
  static void RequireSameLocation(const CAIF_DeviceTensor &a,
                           const char *op_name);

  //----------------------------------------------------------------------------
  // Device-backend entry points (caif_ops_device.cpp)
  //----------------------------------------------------------------------------

  static void MatMulDevice(const CAIF_DeviceTensor &a,
                    const CAIF_DeviceTensor &b,
                    CAIF_DeviceTensor &output,
                    CAIF_RunContext &ctx,
                    const CAIF_DataType::CAIF_DataType_e compute_dtype);
  static void MatMulTransposeADevice(const CAIF_DeviceTensor &a,
                              const CAIF_DeviceTensor &b,
                              CAIF_DeviceTensor &output,
                              CAIF_RunContext &ctx,
                              const CAIF_DataType::CAIF_DataType_e compute_dtype);
  static void MatMulTransposeBDevice(const CAIF_DeviceTensor &a,
                              const CAIF_DeviceTensor &b,
                              CAIF_DeviceTensor &output,
                              CAIF_RunContext &ctx,
                              const CAIF_DataType::CAIF_DataType_e compute_dtype);
  static void BatchedMatMulDevice(const CAIF_DeviceTensor &a,
                           const CAIF_DeviceTensor &b,
                           CAIF_DeviceTensor &output,
                           int m,
                           int k,
                           int n,
                           int batch_count,
                           CAIF_RunContext &ctx,
                           const CAIF_DataType::CAIF_DataType_e compute_dtype);
  static void BatchedMatMulTransposeADevice(const CAIF_DeviceTensor &a,
                                     const CAIF_DeviceTensor &b,
                                     CAIF_DeviceTensor &output,
                                     int k,
                                     int m,
                                     int n,
                                     int batch_count,
                                     CAIF_RunContext &ctx,
                                     const CAIF_DataType::CAIF_DataType_e compute_dtype);
  static void BatchedMatMulTransposeBDevice(const CAIF_DeviceTensor &a,
                                     const CAIF_DeviceTensor &b,
                                     CAIF_DeviceTensor &output,
                                     int m,
                                     int k,
                                     int n,
                                     int batch_count,
                                     CAIF_RunContext &ctx,
                                     const CAIF_DataType::CAIF_DataType_e compute_dtype);
  static void TransposeDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void AddDevice(const CAIF_DeviceTensor &a,
                 const CAIF_DeviceTensor &b,
                 CAIF_DeviceTensor &output);
  static void ScaleDevice(CAIF_DeviceTensor &tensor,float scale);
  static void ScaleDevice(const CAIF_DeviceTensor &input,
                   float scale,
                   CAIF_DeviceTensor &output);
  static void AddScaledDevice(CAIF_DeviceTensor &target,
                       const CAIF_DeviceTensor &source,
                       float scale);
  static void BiasAddDevice(const CAIF_DeviceTensor &input,
                     const CAIF_DeviceTensor &bias,
                     CAIF_DeviceTensor &output);
  static void MatMulBiasDevice(const CAIF_DeviceTensor &a,
                        const CAIF_DeviceTensor &b,
                        const CAIF_DeviceTensor &bias,
                        CAIF_DeviceTensor &output,
                        cudaStream_t stream,
                        CAIF_RunContext &ctx,
                        const CAIF_DataType::CAIF_DataType_e compute_dtype);
  static void BiasGradientDevice(const CAIF_DeviceTensor &grad,
                          CAIF_DeviceTensor &bias_grad);
  static void AddPositionalEncodingDevice(const CAIF_DeviceTensor &input,
                                   const CAIF_DeviceTensor &pe_table,
                                   CAIF_DeviceTensor &output);
  static void PositionalEncodingBackwardDevice(const CAIF_DeviceTensor &grad_output,
                                        CAIF_DeviceTensor &grad_table);
  static void ComputeRelativePositionBiasDevice(const CAIF_DeviceTensor &embedding,
                                         CAIF_DeviceTensor &output,
                                         uint32_t max_distance,
                                         bool bidirectional);
  static void AccumulateRelativePositionBiasGradientDevice(const CAIF_DeviceTensor &grad_output,
                                                    CAIF_DeviceTensor &grad_embedding,
                                                    uint32_t max_distance,
                                                    bool bidirectional);
  static void CastDevice(const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &output,
                  CAIF_RunContext &ctx);
  static void QuantizeInt8Device(const CAIF_DeviceTensor &input,
                          CAIF_DeviceTensor &output,
                          CAIF_DeviceTensor &scales,
                          CAIF_Ops::QuantScheme_e scheme,
                          CAIF_RunContext &ctx);
  static void DequantizeInt8Device(const CAIF_DeviceTensor &input,
                            CAIF_DeviceTensor &output,
                            const CAIF_DeviceTensor &scales,
                            CAIF_Ops::QuantScheme_e scheme,
                            CAIF_RunContext &ctx);
  static void QuantizeInt4PerGroupDevice(const CAIF_DeviceTensor &input,
                                  CAIF_DeviceTensor &output,
                                  CAIF_DeviceTensor &scales,
                                  uint32_t group_size,
                                  CAIF_RunContext &ctx);
  static void DequantizeInt4PerGroupDevice(const CAIF_DeviceTensor &input,
                                    CAIF_DeviceTensor &output,
                                    const CAIF_DeviceTensor &scales,
                                    uint32_t group_size,
                                    CAIF_RunContext &ctx);
  static void SliceLastDimDevice(const CAIF_DeviceTensor &input,
                          CAIF_DeviceTensor &output,
                          uint32_t col_start);
  static void SliceLastDimBackwardDevice(const CAIF_DeviceTensor &grad_output,
                                  CAIF_DeviceTensor &grad_input,
                                  uint32_t col_start);
  static void ConcatLastDimDevice(const CAIF_DeviceTensor &a,
                           const CAIF_DeviceTensor &b,
                           CAIF_DeviceTensor &output);
  static void ReLUDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SigmoidDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void TanhDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SoftmaxDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void LeakyReLUDevice(const CAIF_DeviceTensor &input,
                       CAIF_DeviceTensor &output,
                       float alpha);
  static void ELUDevice(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output,
                 float alpha);
  static void GELUDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SwishDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void ReLUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                          const CAIF_DeviceTensor &input,
                          CAIF_DeviceTensor &grad_input);
  static void SigmoidBackwardDevice(const CAIF_DeviceTensor &grad_output,
                             const CAIF_DeviceTensor &output,
                             CAIF_DeviceTensor &grad_input);
  static void TanhBackwardDevice(const CAIF_DeviceTensor &grad_output,
                          const CAIF_DeviceTensor &output,
                          CAIF_DeviceTensor &grad_input);
  static void SoftmaxBackwardDevice(const CAIF_DeviceTensor &grad_output,
                             const CAIF_DeviceTensor &output,
                             CAIF_DeviceTensor &grad_input);
  static void LeakyReLUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                               const CAIF_DeviceTensor &input,
                               CAIF_DeviceTensor &grad_input,
                               float alpha);
  static void ELUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                         const CAIF_DeviceTensor &input,
                         const CAIF_DeviceTensor &output,
                         CAIF_DeviceTensor &grad_input,
                         float alpha);
  static void GELUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                          const CAIF_DeviceTensor &input,
                          CAIF_DeviceTensor &grad_input);
  static void SwishBackwardDevice(const CAIF_DeviceTensor &grad_output,
                           const CAIF_DeviceTensor &input,
                           const CAIF_DeviceTensor &output,
                           CAIF_DeviceTensor &grad_input);
  static float ReduceSumDevice(const CAIF_DeviceTensor &tensor);
  static float ReduceMeanDevice(const CAIF_DeviceTensor &tensor);
  static void MSELossDevice(const CAIF_DeviceTensor &pred,
                     const CAIF_DeviceTensor &target,
                     CAIF_DeviceTensor &loss);
  static void MSELossBackwardDevice(const CAIF_DeviceTensor &pred,
                             const CAIF_DeviceTensor &target,
                             CAIF_DeviceTensor &grad);
  static void AdamUpdateDevice(CAIF_DeviceTensor &param,
                        const CAIF_DeviceTensor &grad,
                        CAIF_DeviceTensor &m,
                        CAIF_DeviceTensor &v,
                        float lr,
                        float beta1,
                        float beta2,
                        float epsilon,
                        float weight_decay,
                        int t);
  static void SgdUpdateDevice(CAIF_DeviceTensor &param,
                       const CAIF_DeviceTensor &grad,
                       float lr,
                       float weight_decay);
  static void MomentumUpdateDevice(CAIF_DeviceTensor &param,
                            const CAIF_DeviceTensor &grad,
                            CAIF_DeviceTensor &velocity,
                            float lr,
                            float momentum,
                            float weight_decay);
  static void RmspropUpdateDevice(CAIF_DeviceTensor &param,
                           const CAIF_DeviceTensor &grad,
                           CAIF_DeviceTensor &avg_sq,
                           float lr,
                           float alpha,
                           float epsilon,
                           float weight_decay);
  static void AdaGradUpdateDevice(CAIF_DeviceTensor &param,
                           const CAIF_DeviceTensor &grad,
                           CAIF_DeviceTensor &accum,
                           float lr,
                           float epsilon,
                           float weight_decay);
  static void MultiplyDevice(const CAIF_DeviceTensor &a,
                      const CAIF_DeviceTensor &b,
                      CAIF_DeviceTensor &output);
  static void SiLUDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SiLUBackwardDevice(const CAIF_DeviceTensor &input,
                          const CAIF_DeviceTensor &grad_output,
                          CAIF_DeviceTensor &grad_input);
  static void AddBiasDevice(const CAIF_DeviceTensor &input,
                     const CAIF_DeviceTensor &bias,
                     CAIF_DeviceTensor &output);
  static void AddScalarDevice(const CAIF_DeviceTensor &input,
                       float scalar,
                       CAIF_DeviceTensor &output);
  static void SubtractDevice(const CAIF_DeviceTensor &a,
                      const CAIF_DeviceTensor &b,
                      CAIF_DeviceTensor &output);
  static void SubtractScalarDevice(const CAIF_DeviceTensor &input,
                            float scalar,
                            CAIF_DeviceTensor &output);
  static void DivideDevice(const CAIF_DeviceTensor &a,
                    const CAIF_DeviceTensor &b,
                    CAIF_DeviceTensor &output);
  static void DivideScalarDevice(const CAIF_DeviceTensor &input,
                          float scalar,
                          CAIF_DeviceTensor &output);
  static void SqrtDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SumAxisDevice(const CAIF_DeviceTensor &input,
                     uint32_t axis,
                     CAIF_DeviceTensor &output);
  static void SumDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void LogSumExpDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void TopKDevice(const CAIF_DeviceTensor &input,
                  uint32_t k,
                  CAIF_DeviceTensor &indices,
                  CAIF_DeviceTensor &values);
  static void NormalizeRowsDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void NormalizeRowsBackwardTopKGatherDevice(const CAIF_DeviceTensor &grad_w,
                                             const CAIF_DeviceTensor &probs,
                                             const CAIF_DeviceTensor &indices,
                                             CAIF_DeviceTensor &grad_p_topk);
  static void GatherTopKValuesDevice(const CAIF_DeviceTensor &scores,
                              const CAIF_DeviceTensor &indices,
                              CAIF_DeviceTensor &out);
  static void ScatterAddDevice(const CAIF_DeviceTensor &values,
                        const CAIF_DeviceTensor &indices,
                        CAIF_DeviceTensor &output);
  static void MoEDispatchDevice(const CAIF_DeviceTensor &input,
                         const CAIF_DeviceTensor &expert_indices,
                         uint32_t top_k,
                         const std::vector<uint32_t> &token_counts,
                         std::vector<CAIF_DeviceTensor> &expert_inputs);
  static void MoECombineDevice(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                        const CAIF_DeviceTensor &expert_indices,
                        const CAIF_DeviceTensor &expert_weights,
                        uint32_t top_k,
                        const std::vector<uint32_t> &token_counts,
                        CAIF_DeviceTensor &output);
  static void MoECombineBackwardDevice(const CAIF_DeviceTensor &grad_output,
                                const std::vector<CAIF_DeviceTensor> &expert_outputs,
                                const CAIF_DeviceTensor &expert_indices,
                                const CAIF_DeviceTensor &expert_weights,
                                uint32_t top_k,
                                const std::vector<uint32_t> &token_counts,
                                std::vector<CAIF_DeviceTensor> &grad_expert_outputs,
                                CAIF_DeviceTensor &grad_weights);
  static void MoEDispatchBackwardDevice(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
                                 const CAIF_DeviceTensor &expert_indices,
                                 uint32_t top_k,
                                 const std::vector<uint32_t> &token_counts,
                                 CAIF_DeviceTensor &grad_input);
  static void MoETopKGatingDevice(const CAIF_DeviceTensor &router_logits,
                           uint32_t num_experts,
                           uint32_t top_k,
                           CAIF_DeviceTensor &expert_indices,
                           CAIF_DeviceTensor &expert_weights,
                           CAIF_DeviceTensor &router_probs);
  static void MoECountPerExpertDevice(const CAIF_DeviceTensor &expert_indices,
                               uint32_t num_experts,
                               uint32_t top_k,
                               CAIF_DeviceTensor &expert_counts);
  static void MoEZLossGradAddDevice(const CAIF_DeviceTensor &logsumexp_scaled,
                             const CAIF_DeviceTensor &probs,
                             CAIF_DeviceTensor &grad_logits);
  static void MoEDispatchGPUDevice(const CAIF_DeviceTensor &input,
                            const CAIF_DeviceTensor &expert_indices,
                            const CAIF_DeviceTensor &dispatch_map,
                            const CAIF_DeviceTensor &expert_offsets,
                            uint32_t top_k,
                            CAIF_DeviceTensor &expert_buffer);
  static void MoECombineGPUDevice(const CAIF_DeviceTensor &expert_buffer,
                           const CAIF_DeviceTensor &expert_indices,
                           const CAIF_DeviceTensor &expert_weights,
                           const CAIF_DeviceTensor &dispatch_map,
                           const CAIF_DeviceTensor &expert_offsets,
                           uint32_t top_k,
                           CAIF_DeviceTensor &output);
  static uint32_t MoEBuildDispatchMapDevice(const CAIF_DeviceTensor &expert_indices,
                                     uint32_t num_experts,
                                     uint32_t top_k,
                                     uint32_t capacity_per_expert,
                                     CAIF_DeviceTensor &dispatch_map,
                                     CAIF_DeviceTensor &expert_offsets);
  static void MoECombineBackwardGPUDevice(const CAIF_DeviceTensor &grad_output,
                                   const CAIF_DeviceTensor &expert_buffer,
                                   const CAIF_DeviceTensor &expert_indices,
                                   const CAIF_DeviceTensor &expert_weights,
                                   const CAIF_DeviceTensor &dispatch_map,
                                   const CAIF_DeviceTensor &expert_offsets,
                                   uint32_t top_k,
                                   CAIF_DeviceTensor &grad_expert_buffer,
                                   CAIF_DeviceTensor &grad_weights);
  static void MoEDispatchBackwardGPUDevice(const CAIF_DeviceTensor &grad_expert_buffer,
                                    const CAIF_DeviceTensor &expert_indices,
                                    const CAIF_DeviceTensor &dispatch_map,
                                    const CAIF_DeviceTensor &expert_offsets,
                                    uint32_t top_k,
                                    CAIF_DeviceTensor &grad_input);

  //----------------------------------------------------------------------------
  // Host-backend entry points (caif_ops_host.cpp)
  //----------------------------------------------------------------------------

  static void MatMulHost(const CAIF_DeviceTensor &a,
                  const CAIF_DeviceTensor &b,
                  CAIF_DeviceTensor &output,
                  CAIF_RunContext &ctx);
  static void MatMulTransposeAHost(const CAIF_DeviceTensor &a,
                            const CAIF_DeviceTensor &b,
                            CAIF_DeviceTensor &output,
                            CAIF_RunContext &ctx);
  static void MatMulTransposeBHost(const CAIF_DeviceTensor &a,
                            const CAIF_DeviceTensor &b,
                            CAIF_DeviceTensor &output,
                            CAIF_RunContext &ctx);
  static void BatchedMatMulHost(const CAIF_DeviceTensor &a,
                         const CAIF_DeviceTensor &b,
                         CAIF_DeviceTensor &output,
                         int m,
                         int k,
                         int n,
                         int batch_count,
                         CAIF_RunContext &ctx);
  static void BatchedMatMulTransposeAHost(const CAIF_DeviceTensor &a,
                                   const CAIF_DeviceTensor &b,
                                   CAIF_DeviceTensor &output,
                                   int k,
                                   int m,
                                   int n,
                                   int batch_count,
                                   CAIF_RunContext &ctx);
  static void BatchedMatMulTransposeBHost(const CAIF_DeviceTensor &a,
                                   const CAIF_DeviceTensor &b,
                                   CAIF_DeviceTensor &output,
                                   int m,
                                   int k,
                                   int n,
                                   int batch_count,
                                   CAIF_RunContext &ctx);
  static void TransposeHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void AddHost(const CAIF_DeviceTensor &a,
               const CAIF_DeviceTensor &b,
               CAIF_DeviceTensor &output);
  static void ScaleHost(CAIF_DeviceTensor &tensor,float scale);
  static void ScaleHost(const CAIF_DeviceTensor &input,
                 float scale,
                 CAIF_DeviceTensor &output);
  static void AddScaledHost(CAIF_DeviceTensor &target,
                     const CAIF_DeviceTensor &source,
                     float scale);
  static void BiasAddHost(const CAIF_DeviceTensor &input,
                   const CAIF_DeviceTensor &bias,
                   CAIF_DeviceTensor &output);
  static void MatMulBiasHost(const CAIF_DeviceTensor &a,
                      const CAIF_DeviceTensor &b,
                      const CAIF_DeviceTensor &bias,
                      CAIF_DeviceTensor &output,
                      cudaStream_t stream,
                      CAIF_RunContext &ctx);
  static void BiasGradientHost(const CAIF_DeviceTensor &grad,
                        CAIF_DeviceTensor &bias_grad);
  static void AddPositionalEncodingHost(const CAIF_DeviceTensor &input,
                                 const CAIF_DeviceTensor &pe_table,
                                 CAIF_DeviceTensor &output);
  static void PositionalEncodingBackwardHost(const CAIF_DeviceTensor &grad_output,
                                      CAIF_DeviceTensor &grad_table);
  static void ComputeRelativePositionBiasHost(const CAIF_DeviceTensor &embedding,
                                       CAIF_DeviceTensor &output,
                                       uint32_t max_distance,
                                       bool bidirectional);
  static void AccumulateRelativePositionBiasGradientHost(const CAIF_DeviceTensor &grad_output,
                                                  CAIF_DeviceTensor &grad_embedding,
                                                  uint32_t max_distance,
                                                  bool bidirectional);
  static void CastHost(const CAIF_DeviceTensor &input,
                CAIF_DeviceTensor &output,
                CAIF_RunContext &ctx);
  static void SliceLastDimHost(const CAIF_DeviceTensor &input,
                        CAIF_DeviceTensor &output,
                        uint32_t col_start);
  static void SliceLastDimBackwardHost(const CAIF_DeviceTensor &grad_output,
                                CAIF_DeviceTensor &grad_input,
                                uint32_t col_start);
  static void ConcatLastDimHost(const CAIF_DeviceTensor &a,
                         const CAIF_DeviceTensor &b,
                         CAIF_DeviceTensor &output);
  static void ReLUHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SigmoidHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void TanhHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SoftmaxHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void LeakyReLUHost(const CAIF_DeviceTensor &input,
                     CAIF_DeviceTensor &output,
                     float alpha);
  static void ELUHost(const CAIF_DeviceTensor &input,
               CAIF_DeviceTensor &output,
               float alpha);
  static void GELUHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SwishHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void ReLUBackwardHost(const CAIF_DeviceTensor &grad_output,
                        const CAIF_DeviceTensor &input,
                        CAIF_DeviceTensor &grad_input);
  static void SigmoidBackwardHost(const CAIF_DeviceTensor &grad_output,
                           const CAIF_DeviceTensor &output,
                           CAIF_DeviceTensor &grad_input);
  static void TanhBackwardHost(const CAIF_DeviceTensor &grad_output,
                        const CAIF_DeviceTensor &output,
                        CAIF_DeviceTensor &grad_input);
  static void SoftmaxBackwardHost(const CAIF_DeviceTensor &grad_output,
                           const CAIF_DeviceTensor &output,
                           CAIF_DeviceTensor &grad_input);
  static void LeakyReLUBackwardHost(const CAIF_DeviceTensor &grad_output,
                             const CAIF_DeviceTensor &input,
                             CAIF_DeviceTensor &grad_input,
                             float alpha);
  static void ELUBackwardHost(const CAIF_DeviceTensor &grad_output,
                       const CAIF_DeviceTensor &input,
                       const CAIF_DeviceTensor &output,
                       CAIF_DeviceTensor &grad_input,
                       float alpha);
  static void GELUBackwardHost(const CAIF_DeviceTensor &grad_output,
                        const CAIF_DeviceTensor &input,
                        CAIF_DeviceTensor &grad_input);
  static void SwishBackwardHost(const CAIF_DeviceTensor &grad_output,
                         const CAIF_DeviceTensor &input,
                         const CAIF_DeviceTensor &output,
                         CAIF_DeviceTensor &grad_input);
  static float ReduceSumHost(const CAIF_DeviceTensor &tensor);
  static float ReduceMeanHost(const CAIF_DeviceTensor &tensor);
  static void MSELossHost(const CAIF_DeviceTensor &pred,
                   const CAIF_DeviceTensor &target,
                   CAIF_DeviceTensor &loss);
  static void MSELossBackwardHost(const CAIF_DeviceTensor &pred,
                           const CAIF_DeviceTensor &target,
                           CAIF_DeviceTensor &grad);
  static void AdamUpdateHost(CAIF_DeviceTensor &param,
                      const CAIF_DeviceTensor &grad,
                      CAIF_DeviceTensor &m,
                      CAIF_DeviceTensor &v,
                      float lr,
                      float beta1,
                      float beta2,
                      float epsilon,
                      float weight_decay,
                      int t);
  static void SgdUpdateHost(CAIF_DeviceTensor &param,
                     const CAIF_DeviceTensor &grad,
                     float lr,
                     float weight_decay);
  static void MomentumUpdateHost(CAIF_DeviceTensor &param,
                          const CAIF_DeviceTensor &grad,
                          CAIF_DeviceTensor &velocity,
                          float lr,
                          float momentum,
                          float weight_decay);
  static void RmspropUpdateHost(CAIF_DeviceTensor &param,
                         const CAIF_DeviceTensor &grad,
                         CAIF_DeviceTensor &avg_sq,
                         float lr,
                         float alpha,
                         float epsilon,
                         float weight_decay);
  static void AdaGradUpdateHost(CAIF_DeviceTensor &param,
                         const CAIF_DeviceTensor &grad,
                         CAIF_DeviceTensor &accum,
                         float lr,
                         float epsilon,
                         float weight_decay);
  static void MultiplyHost(const CAIF_DeviceTensor &a,
                    const CAIF_DeviceTensor &b,
                    CAIF_DeviceTensor &output);
  static void SiLUHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SiLUBackwardHost(const CAIF_DeviceTensor &input,
                        const CAIF_DeviceTensor &grad_output,
                        CAIF_DeviceTensor &grad_input);
  static void AddBiasHost(const CAIF_DeviceTensor &input,
                   const CAIF_DeviceTensor &bias,
                   CAIF_DeviceTensor &output);
  static void AddScalarHost(const CAIF_DeviceTensor &input,
                     float scalar,
                     CAIF_DeviceTensor &output);
  static void SubtractHost(const CAIF_DeviceTensor &a,
                    const CAIF_DeviceTensor &b,
                    CAIF_DeviceTensor &output);
  static void SubtractScalarHost(const CAIF_DeviceTensor &input,
                          float scalar,
                          CAIF_DeviceTensor &output);
  static void DivideHost(const CAIF_DeviceTensor &a,
                  const CAIF_DeviceTensor &b,
                  CAIF_DeviceTensor &output);
  static void DivideScalarHost(const CAIF_DeviceTensor &input,
                        float scalar,
                        CAIF_DeviceTensor &output);
  static void SqrtHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void SumAxisHost(const CAIF_DeviceTensor &input,
                   uint32_t axis,
                   CAIF_DeviceTensor &output);
  static void SumHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void LogSumExpHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void TopKHost(const CAIF_DeviceTensor &input,
                uint32_t k,
                CAIF_DeviceTensor &indices,
                CAIF_DeviceTensor &values);
  static void NormalizeRowsHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output);
  static void NormalizeRowsBackwardTopKGatherHost(const CAIF_DeviceTensor &grad_w,
                                           const CAIF_DeviceTensor &probs,
                                           const CAIF_DeviceTensor &indices,
                                           CAIF_DeviceTensor &grad_p_topk);
  static void GatherTopKValuesHost(const CAIF_DeviceTensor &scores,
                            const CAIF_DeviceTensor &indices,
                            CAIF_DeviceTensor &out);
  static void ScatterAddHost(const CAIF_DeviceTensor &values,
                      const CAIF_DeviceTensor &indices,
                      CAIF_DeviceTensor &output);
  static void MoEDispatchHost(const CAIF_DeviceTensor &input,
                       const CAIF_DeviceTensor &expert_indices,
                       uint32_t top_k,
                       const std::vector<uint32_t> &token_counts,
                       std::vector<CAIF_DeviceTensor> &expert_inputs);
  static void MoECombineHost(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                      const CAIF_DeviceTensor &expert_indices,
                      const CAIF_DeviceTensor &expert_weights,
                      uint32_t top_k,
                      const std::vector<uint32_t> &token_counts,
                      CAIF_DeviceTensor &output);
  static void MoECombineBackwardHost(const CAIF_DeviceTensor &grad_output,
                              const std::vector<CAIF_DeviceTensor> &expert_outputs,
                              const CAIF_DeviceTensor &expert_indices,
                              const CAIF_DeviceTensor &expert_weights,
                              uint32_t top_k,
                              const std::vector<uint32_t> &token_counts,
                              std::vector<CAIF_DeviceTensor> &grad_expert_outputs,
                              CAIF_DeviceTensor &grad_weights);
  static void MoEDispatchBackwardHost(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
                               const CAIF_DeviceTensor &expert_indices,
                               uint32_t top_k,
                               const std::vector<uint32_t> &token_counts,
                               CAIF_DeviceTensor &grad_input);
  static void MoETopKGatingHost(const CAIF_DeviceTensor &router_logits,
                         uint32_t num_experts,
                         uint32_t top_k,
                         CAIF_DeviceTensor &expert_indices,
                         CAIF_DeviceTensor &expert_weights,
                         CAIF_DeviceTensor &router_probs);
  static void MoECountPerExpertHost(const CAIF_DeviceTensor &expert_indices,
                             uint32_t num_experts,
                             uint32_t top_k,
                             CAIF_DeviceTensor &expert_counts);
  static void MoEZLossGradAddHost(const CAIF_DeviceTensor &logsumexp_scaled,
                           const CAIF_DeviceTensor &probs,
                           CAIF_DeviceTensor &grad_logits);
  static void MoEDispatchGPUHost(const CAIF_DeviceTensor &input,
                          const CAIF_DeviceTensor &expert_indices,
                          const CAIF_DeviceTensor &dispatch_map,
                          const CAIF_DeviceTensor &expert_offsets,
                          uint32_t top_k,
                          CAIF_DeviceTensor &expert_buffer);
  static void MoECombineGPUHost(const CAIF_DeviceTensor &expert_buffer,
                         const CAIF_DeviceTensor &expert_indices,
                         const CAIF_DeviceTensor &expert_weights,
                         const CAIF_DeviceTensor &dispatch_map,
                         const CAIF_DeviceTensor &expert_offsets,
                         uint32_t top_k,
                         CAIF_DeviceTensor &output);
  static uint32_t MoEBuildDispatchMapHost(const CAIF_DeviceTensor &expert_indices,
                                   uint32_t num_experts,
                                   uint32_t top_k,
                                   uint32_t capacity_per_expert,
                                   CAIF_DeviceTensor &dispatch_map,
                                   CAIF_DeviceTensor &expert_offsets);
  static void MoECombineBackwardGPUHost(const CAIF_DeviceTensor &grad_output,
                                 const CAIF_DeviceTensor &expert_buffer,
                                 const CAIF_DeviceTensor &expert_indices,
                                 const CAIF_DeviceTensor &expert_weights,
                                 const CAIF_DeviceTensor &dispatch_map,
                                 const CAIF_DeviceTensor &expert_offsets,
                                 uint32_t top_k,
                                 CAIF_DeviceTensor &grad_expert_buffer,
                                 CAIF_DeviceTensor &grad_weights);
  static void MoEDispatchBackwardGPUHost(const CAIF_DeviceTensor &grad_expert_buffer,
                                  const CAIF_DeviceTensor &expert_indices,
                                  const CAIF_DeviceTensor &dispatch_map,
                                  const CAIF_DeviceTensor &expert_offsets,
                                  uint32_t top_k,
                                  CAIF_DeviceTensor &grad_input);


  protected:

  private:

    CAIF_Ops()=delete;
};//end CAIF_Ops class

}//end instance namespace

#endif  // CAIF_OPS_H
