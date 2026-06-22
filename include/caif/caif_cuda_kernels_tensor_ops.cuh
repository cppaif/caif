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
// CAIF - C++ AI Framework
// Tensor-op CUDA kernels: slice/concat on the last dim (+backward)
// and T5-style relative position bias forward/backward.
// Launcher declarations for src/caif_cuda_kernels_tensor_ops.cu
//. CPU-only builds link the no-op stubs in
// legacy/src/caif_cuda_kernels_cpu.cpp, same contract as the old
// caif_cuda_kernels.h umbrella header.
//------------------------------------------------------------------------------

#pragma once

#include <cstdint>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#else
// Placeholder for non-CUDA builds
typedef void *cudaStream_t;
#endif

//------------------------------------------------------------------------------
// Tensor slice and concatenation kernels
//------------------------------------------------------------------------------

/**
 * Slice along the last dimension of a 2D (logically) tensor.
 * input: [rows, in_cols], output: [rows, out_cols]
 * Copies columns [col_start, col_start + out_cols) from each row.
 */
template<typename T>
void launch_slice_last_dim(const T *input,
                           T *output,
                           int rows,
                           int in_cols,
                           int col_start,
                           int out_cols,
                           cudaStream_t stream);

/**
 * Scatter a slice back into the last dimension (backward of slice).
 * grad_input: [rows, in_cols] (must be pre-zeroed or pre-filled)
 * grad_output: [rows, out_cols]
 * Adds grad_output into grad_input[:, col_start:col_start+out_cols].
 */
template<typename T>
void launch_slice_last_dim_backward(const T *grad_output,
                                    T *grad_input,
                                    int rows,
                                    int in_cols,
                                    int col_start,
                                    int out_cols,
                                    cudaStream_t stream);

/**
 * Concatenate two tensors along the last dimension.
 * a: [rows, cols_a], b: [rows, cols_b], output: [rows, cols_a + cols_b]
 */
template<typename T>
void launch_concat_last_dim(const T *a,
                            const T *b,
                            T *output,
                            int rows,
                            int cols_a,
                            int cols_b,
                            cudaStream_t stream);

/**
 * Compute T5-style relative position bias matrix.
 * embedding: [num_heads, num_buckets] — always fp32 (autocast convention)
 * output: [num_heads, q_len, k_len] — dtype T (matches attention dtype)
 * Uses logarithmic bucketing for distances > num_buckets/2.
 */
template<typename T>
void launch_relative_position_bias_forward(const float *embedding,
                                           T *output,
                                           int num_heads,
                                           int q_len,
                                           int k_len,
                                           int num_buckets,
                                           int max_distance,
                                           int bidirectional,
                                           cudaStream_t stream);

/**
 * Backward for relative position bias — accumulate embedding gradient.
 * grad_output: [num_heads, q_len, k_len] — dtype T
 * grad_embedding: [num_heads, num_buckets] — fp32 (atomicAdd stays on float,
 * sm_75 safe; grads up-cast from T before accumulation).
 */
template<typename T>
void launch_relative_position_bias_backward(const T *grad_output,
                                            float *grad_embedding,
                                            int num_heads,
                                            int q_len,
                                            int k_len,
                                            int num_buckets,
                                            int max_distance,
                                            int bidirectional,
                                            cudaStream_t stream);

