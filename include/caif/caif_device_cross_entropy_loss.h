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
// Cross-entropy loss from logits for language modeling
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_CROSS_ENTROPY_LOSS_H
#define CAIF_DEVICE_CROSS_ENTROPY_LOSS_H

#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"

namespace instance
{

/**
 * @brief Cross-entropy loss computed directly from logits (numerically stable)
 *
 * For language modeling, computes:
 *   loss = -log(softmax(logits)[target])
 *        = -logits[target] + log(sum(exp(logits)))
 *
 * Uses max subtraction for numerical stability.
 */
class CAIF_DeviceCrossEntropyLoss
{
  public:
    // Default ignore index (positions with this target are excluded from loss)
    static constexpr int g_default_ignore_index=-100;

    /**
     * @brief Compute per-position loss from logits
     * @param logits [batch * seq_len, vocab_size] or [batch, seq_len, vocab_size]
     * @param targets [batch * seq_len] or [batch, seq_len] (float-encoded token IDs)
     * @param stream CUDA stream
     * @param ignore_index Target value to ignore (default -100)
     * @return Tensor of per-position losses [N]
     */
    static CAIF_DeviceTensor ComputePerPositionLoss(const CAIF_DeviceTensor &logits,
                                                   const CAIF_DeviceTensor &targets,
                                                   CAIF_CudaStream &stream,
                                                   int ignore_index=g_default_ignore_index);

    /**
     * @brief Compute mean loss (scalar)
     * @param logits [batch * seq_len, vocab_size] or [batch, seq_len, vocab_size]
     * @param targets [batch * seq_len] or [batch, seq_len]
     * @param stream CUDA stream
     * @param ignore_index Target value to ignore
     * @return Mean loss as float (synchronizes stream)
     */
    static float ComputeLoss(const CAIF_DeviceTensor &logits,
                             const CAIF_DeviceTensor &targets,
                             CAIF_CudaStream &stream,
                             int ignore_index=g_default_ignore_index);

    /**
     * @brief Compute gradient of loss w.r.t. logits
     * grad[i,j] = (softmax(logits)[i,j] - (j == target[i] ? 1 : 0)) / N
     * @param logits [N, vocab_size]
     * @param targets [N]
     * @param stream CUDA stream
     * @param ignore_index Target value to ignore
     * @return Gradient tensor [N, vocab_size]
     */
    static CAIF_DeviceTensor ComputeGradient(const CAIF_DeviceTensor &logits,
                                            const CAIF_DeviceTensor &targets,
                                            CAIF_CudaStream &stream,
                                            int ignore_index=g_default_ignore_index);

    /**
     * @brief Compute both loss and gradient in one pass
     * More efficient than calling ComputeLoss and ComputeGradient separately.
     * @param logits [N, vocab_size]
     * @param targets [N]
     * @param grad_logits Output gradient tensor [N, vocab_size]
     * @param stream CUDA stream
     * @param ignore_index Target value to ignore
     * @return Mean loss as float
     */
    static float ComputeLossAndGradient(const CAIF_DeviceTensor &logits,
                                        const CAIF_DeviceTensor &targets,
                                        CAIF_DeviceTensor &grad_logits,
                                        CAIF_CudaStream &stream,
                                        int ignore_index=g_default_ignore_index);
};

}//end instance namespace

#endif  // CAIF_DEVICE_CROSS_ENTROPY_LOSS_H
