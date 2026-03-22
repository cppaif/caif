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
 * @file aif_mean_squared_error_loss.cpp
 * @brief Implementation of the CAIF_MeanSquaredErrorLoss class
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_mean_squared_error_loss.h"
#include <cmath>
#include "caif_framework.h"
#include "caif_tensor_backend.h"

namespace instance
{
  CAIF_MeanSquaredErrorLoss::CAIF_MeanSquaredErrorLoss()
    :CAIF_LossFunction()
  {
  }

  CAIF_Tensor CAIF_MeanSquaredErrorLoss::ComputeLoss(
                                                   const CAIF_Tensor &predictions,
                                                   const CAIF_Tensor &targets
                                                  )const
  {
      // Validate input shapes
      if(predictions.Shape()!=targets.Shape())
      {
        THROW_CAIFE("Predictions and targets must have the same shape");
      }

      if(predictions.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32 ||
         targets.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("MSE loss currently only supports Float32 data type");
      }

      // Direct fused computation: per-sample mean of squared error
      const auto &shape=predictions.Shape();
      if(shape.size()<2)
      {
        THROW_CAIFE("MSE expects at least 2D tensors [batch, features...]");
      }
      if(predictions.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32 ||
         targets.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("MSE loss currently only supports Float32 data type");
      }
      const uint32_t batch=shape[0];
      uint32_t elems_per_sample=1;
      for(size_t i=1;i<shape.size();++i)
      {
        elems_per_sample*=shape[i];
      }
      CAIF_Tensor mean_losses(predictions.Framework(),{batch},predictions.Type());
      const float *p=predictions.ConstData<float>();
      const float *t=targets.ConstData<float>();
      float *out=mean_losses.MutableData<float>();
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(uint32_t b=0;b<batch;++b)
      {
        double sum=0.0;
        const size_t base=static_cast<size_t>(b)*elems_per_sample;
        for(uint32_t i=0;i<elems_per_sample;++i)
        {
          const float d=p[base+i]-t[base+i];
          sum+=static_cast<double>(d)*static_cast<double>(d);
        }
        out[b]=static_cast<float>(sum/static_cast<double>(elems_per_sample));
      }
      return mean_losses;
  }

  CAIF_Tensor CAIF_MeanSquaredErrorLoss::ComputeGradient(
                                                       const CAIF_Tensor &predictions,
                                                       const CAIF_Tensor &targets
                                                      )const
  {
      // Validate input shapes
      if(predictions.Shape()!=targets.Shape())
      {
        THROW_CAIFE("Predictions and targets must have the same shape");
      }

      if(predictions.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32 ||
         targets.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("MSE gradient currently only supports Float32 data type");
      }

      // Fused gradient: grad = 2*(pred-target)/(N_features * batch_size) per element
      const auto &shape=predictions.Shape();
      if(shape.size()<2)
      {
        THROW_CAIFE("MSE expects at least 2D tensors [batch, features...]");
      }
      if(predictions.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32 ||
         targets.Type()!=
         CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("MSE gradient currently only supports Float32 data type");
      }
      uint32_t elems_per_sample=1;
      for(size_t i=1;i<shape.size();++i)
      {
        elems_per_sample*=shape[i];
      }
      const uint32_t batch=shape[0];
      CAIF_Tensor grad(predictions.Framework(),shape,predictions.Type());
      const float *p=predictions.ConstData<float>();
      const float *t=targets.ConstData<float>();
      float *g=grad.MutableData<float>();
      const float invN=2.0f/(static_cast<float>(elems_per_sample)*static_cast<float>(batch));
      const size_t total=grad.NumElements();
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(size_t i=0;i<total;++i)
      {
        g[i]=(p[i]-t[i])*invN;
      }
      return grad;
  }

  std::pair<CAIF_Tensor,CAIF_Tensor> CAIF_MeanSquaredErrorLoss::ComputeLossAndGradient(
                                                                                    const CAIF_Tensor &predictions,
                                                                                    const CAIF_Tensor &targets
                                                                                   )const
  {
    // Validate inputs
    if(predictions.Shape()!=targets.Shape())
    {
      THROW_CAIFE("Predictions and targets must have the same shape");
    }
    if(predictions.Type()!=
       CAIF_DataType::CAIF_DataType_e::Float32 ||
       targets.Type()!=
       CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("MSE loss currently only supports Float32 data type");
    }
    const auto &shape=predictions.Shape();
    if(shape.size()<2)
    {
      THROW_CAIFE("MSE expects at least 2D tensors [batch, features...]");
    }

    const uint32_t batch=shape[0];
    uint32_t elems_per_sample=1;
    for(size_t i=1;i<shape.size();++i)
    {
      elems_per_sample*=shape[i];
    }
    const float invN=2.0f/(static_cast<float>(elems_per_sample)*static_cast<float>(batch));

    CAIF_Tensor mean_losses(predictions.Framework(),{batch},predictions.Type());
    CAIF_Tensor grad(predictions.Framework(),shape,predictions.Type());

    const float *p=predictions.ConstData<float>();
    const float *t=targets.ConstData<float>();
    float *out=mean_losses.MutableData<float>();
    float *g=grad.MutableData<float>();

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(uint32_t b=0;b<batch;++b)
    {
      double sum=0.0;
      const size_t base=static_cast<size_t>(b)*elems_per_sample;
      for(uint32_t i=0;i<elems_per_sample;++i)
      {
        const float d=p[base+i]-t[base+i];
        sum+=static_cast<double>(d)*static_cast<double>(d);
        g[base+i]=d*invN;
      }
      out[b]=static_cast<float>(sum/static_cast<double>(elems_per_sample));
    }
    // temp debug removed
    return {std::move(mean_losses),std::move(grad)};
  }

  CAIF_LossType_e CAIF_MeanSquaredErrorLoss::LossType()const
  {
    return CAIF_LossType_e::MeanSquaredError;
  }

  std::unique_ptr<CAIF_LossFunction> CAIF_MeanSquaredErrorLoss::Clone()const
  {
    return std::make_unique<CAIF_MeanSquaredErrorLoss>(*this);
  }

  std::string CAIF_MeanSquaredErrorLoss::Description()const
  {
    return "Mean Squared Error Loss";
  }
}//end instance namespace
