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
 * @file aif_loss_function_bce_logits.cpp
 * @brief Binary Cross Entropy with logits (numerically stable)
 */

#include "caif_loss_function_bce_logits.h"
#include <cmath>
#include <algorithm>

namespace instance
{
// Stable helpers
static inline float SigmoidFromLogitStable(const float z)
{
  if(z>=0.0f)
  {
    const float ez=std::exp(-z);
    return 1.0f/(1.0f+ez);
  }
  else
  {
    const float ez=std::exp(z);
    return ez/(1.0f+ez);
  }
}

static inline float SoftplusStable(const float x)
{
  if(x>0.0f)
  {
    return x+std::log1pf(std::exp(-x));
  }
  else
  {
    return std::log1pf(std::exp(x));
  }
}

CAIF_Tensor CAIF_BinaryCrossEntropyWithLogitsLoss::ComputeLoss(
                                                             const CAIF_Tensor &logits,
                                                             const CAIF_Tensor &targets
                                                            )const
{
  if(logits.Type()!=CAIF_DataType::CAIF_DataType_e::Float32||targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("BCE-with-logits expects Float32 logits and targets");
  }
  if(logits.Shape()!=targets.Shape())
  {
    THROW_CAIFE("Logits and targets must have the same shape");
  }

  const float *z=logits.ConstData<float>();
  const float *y=targets.ConstData<float>();

  const size_t n=logits.NumElements();

  // Class balancing weights per batch (same as BCE)
  size_t positive_count=0;
  size_t negative_count=0;
  for(size_t i=0;i<n;++i)
  {
    const float yi=y[i];
    if(yi>0.5f)
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
    const float nf=static_cast<float>(n);
    w_pos=nf/(2.0f*static_cast<float>(positive_count));
    w_neg=nf/(2.0f*static_cast<float>(negative_count));
  }

  // Output: scalar loss [1]
  CAIF_Tensor loss(logits.Framework(),{1},CAIF_DataType::CAIF_DataType_e::Float32);
  float *loss_out=loss.MutableData<float>();

  double sum=0.0;
  for(size_t i=0;i<n;++i)
  {
    const float yi=y[i];
    const float zi=z[i];
    float w=1.0f;
    if(yi>0.5f)
    {
      w=w_pos;
    }
    else
    {
      w=w_neg;
    }
    // Stable BCE-with-logits: softplus(z) - z*y
    const float term=SoftplusStable(zi)-zi*yi;
    sum+=static_cast<double>(w)*static_cast<double>(term);
  }
  const float batch_loss=static_cast<float>(sum/static_cast<double>(n));
  loss_out[0]=batch_loss;
  return loss;
}

CAIF_Tensor CAIF_BinaryCrossEntropyWithLogitsLoss::ComputeGradient(
                                                                 const CAIF_Tensor &logits,
                                                                 const CAIF_Tensor &targets
                                                                )const
{
  if(logits.Type()!=CAIF_DataType::CAIF_DataType_e::Float32||targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("BCE-with-logits expects Float32 logits and targets");
  }
  if(logits.Shape()!=targets.Shape())
  {
    THROW_CAIFE("Logits and targets must have the same shape for gradient");
  }

  const float *z=logits.ConstData<float>();
  const float *y=targets.ConstData<float>();

  CAIF_Tensor grad(logits.Framework(),logits.Shape(),CAIF_DataType::CAIF_DataType_e::Float32);
  float *g=grad.MutableData<float>();

  const size_t n=logits.NumElements();

  // Class balancing weights per batch (same as BCE)
  size_t positive_count=0;
  size_t negative_count=0;
  for(size_t i=0;i<n;++i)
  {
    const float yi=y[i];
    if(yi>0.5f)
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
    const float nf=static_cast<float>(n);
    w_pos=nf/(2.0f*static_cast<float>(positive_count));
    w_neg=nf/(2.0f*static_cast<float>(negative_count));
  }

  for(size_t i=0;i<n;++i)
  {
    const float yi=y[i];
    const float zi=z[i];
    float w=1.0f;
    if(yi>0.5f)
    {
      w=w_pos;
    }
    else
    {
      w=w_neg;
    }
    const float si=SigmoidFromLogitStable(zi);
    g[i]=w*(si-yi);
  }

  // Average over batch elements
  const float scale=1.0f/static_cast<float>(n);
  for(size_t i=0;i<n;++i)
  {
    g[i]*=scale;
  }
  return grad;
}
}//end instance namespace


