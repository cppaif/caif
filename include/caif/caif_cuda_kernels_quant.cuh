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
// Quantization / dtype conversion CUDA kernels: fp32<->fp16/bf16,
// int8 (plain + scaled per-tensor/per-channel), int4 per-group
// quantize/dequantize.
// Launcher declarations for src/caif_cuda_kernels_quant.cu
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
// Data type conversion kernels
//------------------------------------------------------------------------------

/**
 * Convert FP32 array to FP16 (__half) on device.
 * input: [n] float, output: [n] half (2 bytes each)
 */
void launch_convert_fp32_to_fp16(const float *input,
                                 void *output,
                                 int64_t n,
                                 cudaStream_t stream);

/**
 * Convert FP16 (__half) array to FP32 on device.
 * input: [n] half, output: [n] float
 */
void launch_convert_fp16_to_fp32(const void *input,
                                 float *output,
                                 int64_t n,
                                 cudaStream_t stream);

/**
 * Convert FP32 array to BF16 (__nv_bfloat16) on device.
 * input: [n] float, output: [n] bfloat16 (2 bytes each)
 *
 * fp32-only by contract: dtype-conversion utility — fp32 IN, bf16 OUT
 * is the operation's name. Templating on T is meaningless.
 */
void launch_convert_fp32_to_bf16(const float *input,
                                 void *output,
                                 int64_t n,
                                 cudaStream_t stream);

/**
 * Convert BF16 (__nv_bfloat16) array to FP32 on device.
 * input: [n] bfloat16, output: [n] float
 *
 * fp32-only by contract: dtype-conversion utility — bf16 IN, fp32 OUT
 * is the operation's name.
 */
void launch_convert_bf16_to_fp32(const void *input,
                                 float *output,
                                 int64_t n,
                                 cudaStream_t stream);

//------------------------------------------------------------------------------
// INT8 conversion kernels
//------------------------------------------------------------------------------

/**
 * Convert FP32 to INT8 on device. Clamps to [-127,127] and casts.
 * input: [n] float, output: [n] int8_t
 *
 * fp32-only by contract: dtype-conversion utility — fp32 IN, int8 OUT
 * is the operation's name.
 */
void launch_convert_fp32_to_int8(const float *input,
                                 void *output,
                                 int64_t n,
                                 cudaStream_t stream);

/**
 * Convert INT8 to FP32 on device. Simple cast.
 * input: [n] int8_t, output: [n] float
 *
 * fp32-only by contract: dtype-conversion utility — int8 IN, fp32 OUT
 * is the operation's name.
 */
void launch_convert_int8_to_fp32(const void *input,
                                 float *output,
                                 int64_t n,
                                 cudaStream_t stream);

//------------------------------------------------------------------------------
// INT8 scaled quantization kernels (symmetric, per-tensor and per-channel)
//------------------------------------------------------------------------------

/**
 * Symmetric per-tensor INT8 quantization.
 *
 * Computes scale = max(|input|)/127, then q = round(input/scale) clamped to
 * [-127, 127]. Writes quantized bytes into `output` and the scale value into
 * `scale[0]`. Caller allocates `output` as int8_t[n] and `scale` as float[1].
 *
 * If the input is identically zero the scale is set to 1.0f so dequantization
 * returns zero.
 *
 * fp32-only by contract: quantization utility — fp32 master IN, int8
 * packed OUT. The fp32 input is the canonical representation; int8 is
 * the quantized output. Not templated.
 */
void launch_quantize_int8_per_tensor(const float *input,
                                     void *output,
                                     void *scale,
                                     int64_t n,
                                     cudaStream_t stream);

/**
 * Dequantize per-tensor INT8 back to fp32.
 *
 * fp32-only by contract: dequantization utility — int8 packed IN, fp32
 * master OUT.
 */
void launch_dequantize_int8_per_tensor(const void *input,
                                       float *output,
                                       const void *scale,
                                       int64_t n,
                                       cudaStream_t stream);

/**
 * Symmetric per-channel INT8 quantization on a 2D tensor [rows, cols] with
 * scales computed along `cols` (the output-channel axis for weight tensors).
 *
 * scales[c] = max_r |input[r, c]| / 127
 * q[r, c]   = round(input[r, c] / scales[c]) clamped to [-127, 127]
 *
 * `output` is int8_t[rows*cols], `scales` is float[cols].
 *
 * fp32-only by contract: quantization utility — fp32 master IN, int8
 * packed OUT (per-channel scales).
 */
void launch_quantize_int8_per_channel(const float *input,
                                      void *output,
                                      void *scales,
                                      int rows,
                                      int cols,
                                      cudaStream_t stream);

/**
 * Dequantize per-channel INT8 back to fp32.
 *
 * fp32-only by contract: dequantization utility — int8 packed IN, fp32
 * master OUT.
 */
void launch_dequantize_int8_per_channel(const void *input,
                                        float *output,
                                        const void *scales,
                                        int rows,
                                        int cols,
                                        cudaStream_t stream);

//------------------------------------------------------------------------------
// INT4 quantization kernels (symmetric per-group with FP16 scales)
//------------------------------------------------------------------------------

/**
 * Dequantize INT4 packed data to FP32.
 * Packed format: 2 elements per byte (low nibble=even, high nibble=odd).
 * Symmetric per-group: val = int4_val * scale[group_idx]
 *
 * packed_data: [(num_elements+1)/2] bytes of packed INT4
 * scales: [num_groups] FP16 per-group scales
 * output: [num_elements] float
 * num_elements: total number of logical elements
 * group_size: elements per quantization group
 *
 * fp32-only by contract: dequantization utility — int4 packed IN with
 * fp16 scales, fp32 master OUT.
 */
void launch_dequantize_int4(const void *packed_data,
                            const void *scales,
                            float *output,
                            int64_t num_elements,
                            int group_size,
                            cudaStream_t stream);

/**
 * Quantize FP32 to INT4 packed format with per-group FP16 scales.
 * Computes scales from data, packs 2 elements per byte.
 *
 * input: [num_elements] float
 * packed_output: [(num_elements+1)/2] bytes
 * scales_output: [num_groups] FP16
 * num_elements: total number of logical elements
 * group_size: elements per quantization group
 *
 * fp32-only by contract: quantization utility — fp32 master IN, int4
 * packed OUT (with per-group fp16 scales).
 */
void launch_quantize_to_int4(const float *input,
                             void *packed_output,
                             void *scales_output,
                             int64_t num_elements,
                             int group_size,
                             cudaStream_t stream);

