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
 * @file aif_loss_function_bce.cpp
 * @brief Binary Cross Entropy loss implementation
 */

#include "caif_loss_function.h"
#include <cmath>
#include <algorithm>

namespace instance
{
  static inline float ClampProbability(const float p)
  {
    // Numerical stability epsilon consistent with framework constants
    constexpr float eps=1e-6f;
    return std::min(std::max(p,eps),1.0f-eps);
  }

  CAIF_Tensor CAIF_BinaryCrossEntropyLoss::ComputeLoss(
                                                     const CAIF_Tensor &predictions,
                                                     const CAIF_Tensor &targets
                                                    )const
  {
    // Expect Float32 predictions and targets with same shape [batch, 1] or [batch]
    if(predictions.Type()!=CAIF_DataType::CAIF_DataType_e::Float32 ||
       targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Binary cross entropy expects Float32 predictions and targets");
    }

    const auto &p_shape=predictions.Shape();
    const auto &t_shape=targets.Shape();
    if(p_shape!=t_shape)
    {
      THROW_CAIFE("Predictions and targets must have the same shape for BCE");
    }
    if(p_shape.empty()==true)
    {
      THROW_CAIFE("Empty input to BCE");
    }

    const float *p_data=predictions.ConstData<float>();
    const float *t_data=targets.ConstData<float>();

    // Output is a single scalar loss per batch: [1]
    CAIF_Tensor loss(predictions.Framework(),{1},CAIF_DataType::CAIF_DataType_e::Float32);
    float *loss_ptr=loss.MutableData<float>();

    const size_t n=predictions.NumElements();

    // Class-weighting (inverse frequency) computed per batch
    size_t positive_count=0;
    size_t negative_count=0;
    for(size_t i=0;i<n;++i)
    {
      const float y_count_src=t_data[i];
      if(y_count_src>0.5f)
      {
        positive_count+=1;
      }
      else
      {
        negative_count+=1;
      }
    }

    float w_pos=1.0f;
    float w_neg=1.0f;
    if(positive_count>0&&negative_count>0)
    {
      const float n_float=static_cast<float>(n);
      w_pos=n_float/(2.0f*static_cast<float>(positive_count));
      w_neg=n_float/(2.0f*static_cast<float>(negative_count));
    }

    float sum=0.0f;
    for(size_t i=0;i<n;++i)
    {
      const float y=t_data[i];
      const float py=ClampProbability(p_data[i]);
      float w=1.0f;
      if(y>0.5f)
      {
        w=w_pos;
      }
      else
      {
        w=w_neg;
      }
      sum+=-w*(y*std::log(py)+(1.0f-y)*std::log(1.0f-py));
    }
    const float batch_loss=sum/static_cast<float>(n);
    loss_ptr[0]=batch_loss;
    return loss;
  }

  CAIF_Tensor CAIF_BinaryCrossEntropyLoss::ComputeGradient(
                                                         const CAIF_Tensor &predictions,
                                                         const CAIF_Tensor &targets
                                                        )const
  {
    if(predictions.Type()!=CAIF_DataType::CAIF_DataType_e::Float32 ||
       targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Binary cross entropy expects Float32 predictions and targets");
    }
    const auto &p_shape=predictions.Shape();
    const auto &t_shape=targets.Shape();
    if(p_shape!=t_shape)
    {
      THROW_CAIFE("Predictions and targets must have the same shape for BCE gradient");
    }
    const float *p_data=predictions.ConstData<float>();
    const float *t_data=targets.ConstData<float>();

    CAIF_Tensor grad(predictions.Framework(),p_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    float *g_ptr=grad.MutableData<float>();

    const size_t n=predictions.NumElements();

    // Class-weighting (inverse frequency) computed per batch
    size_t positive_count=0;
    size_t negative_count=0;
    for(size_t i=0;i<n;++i)
    {
      const float y_count_src=t_data[i];
      if(y_count_src>0.5f)
      {
        positive_count+=1;
      }
      else
      {
        negative_count+=1;
      }
    }

    float w_pos=1.0f;
    float w_neg=1.0f;
    if(positive_count>0&&negative_count>0)
    {
      const float n_float=static_cast<float>(n);
      w_pos=n_float/(2.0f*static_cast<float>(positive_count));
      w_neg=n_float/(2.0f*static_cast<float>(negative_count));
    }

    for(size_t i=0;i<n;++i)
    {
      const float y=t_data[i];
      const float py=ClampProbability(p_data[i]);
      float w=1.0f;
      if(y>0.5f)
      {
        w=w_pos;
      }
      else
      {
        w=w_neg;
      }
      // d/dp [-y log p - (1-y) log (1-p)] = -(y/p) + ((1-y)/(1-p))
      float gi=w*(-(y/py)+((1.0f-y)/(1.0f-py)));
      // Clamp gradient to avoid exploding values
      if(gi>1e3f){gi=1e3f;}
      if(gi<-1e3f){gi=-1e3f;}
      g_ptr[i]=gi;
    }
    // Average over elements
    const float scale=1.0f/static_cast<float>(n);
    for(size_t i=0;i<n;++i)
    {
      g_ptr[i]*=scale;
    }
    return grad;
  }
}//end instance namespace

