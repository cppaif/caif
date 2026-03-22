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

#ifndef CAIF_BLAS_BACKEND_H
#define CAIF_BLAS_BACKEND_H

#include "caif_tensor_backend.h"
#include "caif_cpu_tensor_data.h"

namespace instance
{
class CAIF_BLASBackend:public CAIF_TensorBackend
{
  public:
    CAIF_BLASBackend()=default;
    ~CAIF_BLASBackend() override=default;

    std::unique_ptr<CAIF_TensorData> CreateTensor(
                                                 const std::vector<uint32_t> &shape,
                                                 const CAIF_DataType &dtype
                                                ) override
    {
      return std::make_unique<CAIF_CPUTensorData>(shape,dtype);
    }

    void MatrixMultiply(
                        const CAIF_TensorData &a,
                        const CAIF_TensorData &b,
                        CAIF_TensorData &result
                       ) override;

    void MatrixMultiplyEx(
                          const CAIF_TensorData &a,
                          const CAIF_TensorData &b,
                          CAIF_TensorData &result,
                          const Transpose_e trans_a,
                          const Transpose_e trans_b
                         ) override;

    void Convolution2D(
                       const CAIF_TensorData &input,
                       const CAIF_TensorData &kernel,
                       CAIF_TensorData &output,
                       const ConvolutionParams &params
                      )override;

    void MaxPooling2D(
                      const CAIF_TensorData &input,
                      CAIF_TensorData &output,
                      CAIF_TensorData *indices,
                      const PoolingParams &params
                     )override;

    void AveragePooling2D(
                          const CAIF_TensorData &input,
                          CAIF_TensorData &output,
                          const PoolingParams &params
                         )override;

    void BatchNormForward(
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

    void ActivationForward(
                           const CAIF_TensorData &input,
                           CAIF_TensorData &output,
                           const ActivationType_e activation_type
                          )override;

    void SoftmaxForward(
                        const CAIF_TensorData &input,
                        CAIF_TensorData &output
                       )override;

    void Convolution2DBackwardData(
                                   const CAIF_TensorData &grad_output,
                                   const CAIF_TensorData &kernel,
                                   CAIF_TensorData &grad_input,
                                   const ConvolutionParams &params
                                  )override;

    void Convolution2DBackwardFilter(
                                     const CAIF_TensorData &input,
                                     const CAIF_TensorData &grad_output,
                                     CAIF_TensorData &grad_kernel,
                                     const ConvolutionParams &params
                                    )override;

    void MaxPooling2DBackward(
                              const CAIF_TensorData &grad_output,
                              const CAIF_TensorData *indices,
                              const CAIF_TensorData &input,
                              CAIF_TensorData &grad_input,
                              const PoolingParams &params
                             )override;

    void AveragePooling2DBackward(
                                  const CAIF_TensorData &grad_output,
                                  CAIF_TensorData &grad_input,
                                  const PoolingParams &params
                                 )override;

    void BatchNormBackward(
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

    void ActivationBackward(
                            const CAIF_TensorData &grad_output,
                            const CAIF_TensorData &input,
                            const CAIF_TensorData &output,
                            CAIF_TensorData &grad_input,
                            const ActivationType_e activation_type
                           )override;

    void SoftmaxBackward(
                         const CAIF_TensorData &grad_output,
                         const CAIF_TensorData &output,
                         CAIF_TensorData &grad_input
                        )override;

    void DropoutForward(
                        const CAIF_TensorData &input,
                        CAIF_TensorData &output,
                        CAIF_TensorData &mask,
                        const float dropout_rate,
                        const bool training
                       )override;

    void DropoutBackward(
                         const CAIF_TensorData &grad_output,
                         const CAIF_TensorData &mask,
                         CAIF_TensorData &grad_input,
                         const float dropout_rate
                        )override;

    //--------------------------------------------------------------------------
    // Element-wise Operations
    //--------------------------------------------------------------------------

    void ElementwiseAdd(
                        const CAIF_TensorData &a,
                        const CAIF_TensorData &b,
                        CAIF_TensorData &result
                       )override;

    void ElementwiseAddScalar(
                              const CAIF_TensorData &a,
                              const float scalar,
                              CAIF_TensorData &result
                             )override;

    void ElementwiseSub(
                        const CAIF_TensorData &a,
                        const CAIF_TensorData &b,
                        CAIF_TensorData &result
                       )override;

    void ElementwiseSubScalar(
                              const CAIF_TensorData &a,
                              const float scalar,
                              CAIF_TensorData &result
                             )override;

    void ElementwiseMul(
                        const CAIF_TensorData &a,
                        const CAIF_TensorData &b,
                        CAIF_TensorData &result
                       )override;

    void ElementwiseMulScalar(
                              const CAIF_TensorData &a,
                              const float scalar,
                              CAIF_TensorData &result
                             )override;

    void ElementwiseDiv(
                        const CAIF_TensorData &a,
                        const CAIF_TensorData &b,
                        CAIF_TensorData &result
                       )override;

    void ElementwiseDivScalar(
                              const CAIF_TensorData &a,
                              const float scalar,
                              CAIF_TensorData &result
                             )override;

    void ElementwiseSqrt(
                         const CAIF_TensorData &a,
                         CAIF_TensorData &result
                        )override;

    float ReduceSum(const CAIF_TensorData &a)override;

    float ReduceMean(const CAIF_TensorData &a)override;

    //--------------------------------------------------------------------------
    // Loss Function Operations
    //--------------------------------------------------------------------------

    void CrossEntropyLoss(
                          const CAIF_TensorData &predictions,
                          const CAIF_TensorData &targets,
                          CAIF_TensorData &loss_per_sample,
                          const float epsilon
                         )override;

    void CrossEntropyGradient(
                              const CAIF_TensorData &predictions,
                              const CAIF_TensorData &targets,
                              CAIF_TensorData &gradient,
                              const float epsilon
                             )override;

    void MSELoss(
                 const CAIF_TensorData &predictions,
                 const CAIF_TensorData &targets,
                 CAIF_TensorData &loss_elements
                )override;

    void MSEGradient(
                     const CAIF_TensorData &predictions,
                     const CAIF_TensorData &targets,
                     CAIF_TensorData &gradient
                    )override;

    //--------------------------------------------------------------------------
    // Optimizer Operations
    //--------------------------------------------------------------------------

    void FusedAdamUpdate(
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

    void FusedSGDMomentumUpdate(
                                CAIF_TensorData &param,
                                const CAIF_TensorData &grad,
                                CAIF_TensorData &velocity,
                                const float lr,
                                const float momentum,
                                const float weight_decay
                               )override;

    BackendType_e Type()const override {return BackendType_e::BLAS;}
    bool IsGPUAccelerated()const override {return false;}
};
}//end instance namespace

#endif


