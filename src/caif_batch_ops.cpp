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
 * @file aif_batch_ops.cpp
 * @brief Implementation of batch normalization and dropout operations
 */

#include "caif_batch_ops.h"
#include "caif_framework.h"
#include <cstring>
#include <random>

using namespace instance;

//==============================================================================
// CAIF_BatchNorm
//==============================================================================

void CAIF_BatchNorm::Forward(
                            const float *input,
                            float *output,
                            const float *scale,
                            const float *bias,
                            float *running_mean,
                            float *running_var,
                            float *saved_mean,
                            float *saved_inv_var,
                            const uint32_t batch,
                            const uint32_t features,
                            const float epsilon,
                            const float momentum,
                            const bool training
                           )
{
  try
  {
    if(training==true)
    {
      // Compute batch mean
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(uint32_t f=0;f<features;++f)
      {
        float sum=0.0f;
        for(uint32_t b=0;b<batch;++b)
        {
          sum+=input[b*features+f];
        }
        saved_mean[f]=sum/static_cast<float>(batch);
      }
      
      // Compute batch variance
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(uint32_t f=0;f<features;++f)
      {
        float sum=0.0f;
        const float mean=saved_mean[f];
        for(uint32_t b=0;b<batch;++b)
        {
          const float diff=input[b*features+f]-mean;
          sum+=diff*diff;
        }
        const float var=sum/static_cast<float>(batch);
        saved_inv_var[f]=1.0f/std::sqrt(var+epsilon);
        
        // Update running stats
        running_mean[f]=momentum*running_mean[f]+(1.0f-momentum)*mean;
        running_var[f]=momentum*running_var[f]+(1.0f-momentum)*var;
      }
      
      // Normalize and scale
      #ifdef _OPENMP
      #pragma omp parallel for collapse(2)
      #endif
      for(uint32_t b=0;b<batch;++b)
      {
        for(uint32_t f=0;f<features;++f)
        {
          const float normalized=(input[b*features+f]-saved_mean[f])*saved_inv_var[f];
          output[b*features+f]=normalized*scale[f]+bias[f];
        }
      }
    }
    else
    {
      // Inference mode - use running stats
      #ifdef _OPENMP
      #pragma omp parallel for collapse(2)
      #endif
      for(uint32_t b=0;b<batch;++b)
      {
        for(uint32_t f=0;f<features;++f)
        {
          const float inv_std=1.0f/std::sqrt(running_var[f]+epsilon);
          const float normalized=(input[b*features+f]-running_mean[f])*inv_std;
          output[b*features+f]=normalized*scale[f]+bias[f];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_BatchNorm::Backward(
                             const float *grad_output,
                             const float *input,
                             const float *scale,
                             const float *saved_mean,
                             const float *saved_inv_var,
                             float *grad_input,
                             float *grad_scale,
                             float *grad_bias,
                             const uint32_t batch,
                             const uint32_t features,
                             const float /*epsilon*/
                            )
{
  try
  {
    const float batch_f=static_cast<float>(batch);
    
    // Compute gradients for scale and bias
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(uint32_t f=0;f<features;++f)
    {
      float sum_grad_bias=0.0f;
      float sum_grad_scale=0.0f;
      const float mean=saved_mean[f];
      const float inv_var=saved_inv_var[f];
      
      for(uint32_t b=0;b<batch;++b)
      {
        const float g=grad_output[b*features+f];
        sum_grad_bias+=g;
        sum_grad_scale+=g*(input[b*features+f]-mean)*inv_var;
      }
      grad_bias[f]=sum_grad_bias;
      grad_scale[f]=sum_grad_scale;
    }
    
    // Compute gradient for input
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(uint32_t f=0;f<features;++f)
    {
      const float mean=saved_mean[f];
      const float inv_var=saved_inv_var[f];
      const float sc=scale[f];
      
      float sum1=0.0f;
      float sum2=0.0f;
      for(uint32_t b=0;b<batch;++b)
      {
        const float g=grad_output[b*features+f];
        sum1+=g;
        sum2+=g*(input[b*features+f]-mean);
      }
      
      for(uint32_t b=0;b<batch;++b)
      {
        const float x_hat=(input[b*features+f]-mean)*inv_var;
        grad_input[b*features+f]=sc*inv_var*(grad_output[b*features+f]-
                                              sum1/batch_f-
                                              x_hat*sum2*inv_var*inv_var/batch_f);
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_BatchNorm::Forward(
                                  const CAIF_Tensor &input,
                                  const CAIF_Tensor &scale,
                                  const CAIF_Tensor &bias,
                                  CAIF_Tensor &running_mean,
                                  CAIF_Tensor &running_var,
                                  const float epsilon,
                                  const float momentum,
                                  const bool training,
                                  CAIF_Tensor &saved_mean,
                                  CAIF_Tensor &saved_inv_var
                                 )
{
  try
  {
    const auto &shape=input.Shape();
    const uint32_t batch=shape[0];
    uint32_t features=1;
    for(size_t i=1;i<shape.size();++i)
    {
      features*=shape[i];
    }
    
    CAIF_Tensor output(input.Framework(),shape,input.Type());
    
    Forward(
            input.ConstData<float>(),
            output.MutableData<float>(),
            scale.ConstData<float>(),
            bias.ConstData<float>(),
            running_mean.MutableData<float>(),
            running_var.MutableData<float>(),
            saved_mean.MutableData<float>(),
            saved_inv_var.MutableData<float>(),
            batch,
            features,
            epsilon,
            momentum,
            training
           );
    
    return output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_BatchNorm::Backward(
                                   const CAIF_Tensor &grad_output,
                                   const CAIF_Tensor &input,
                                   const CAIF_Tensor &scale,
                                   const CAIF_Tensor &saved_mean,
                                   const CAIF_Tensor &saved_inv_var,
                                   const float epsilon,
                                   CAIF_Tensor &grad_scale,
                                   CAIF_Tensor &grad_bias
                                  )
{
  try
  {
    const auto &shape=input.Shape();
    const uint32_t batch=shape[0];
    uint32_t features=1;
    for(size_t i=1;i<shape.size();++i)
    {
      features*=shape[i];
    }
    
    CAIF_Tensor grad_input(input.Framework(),shape,input.Type());
    
    Backward(
             grad_output.ConstData<float>(),
             input.ConstData<float>(),
             scale.ConstData<float>(),
             saved_mean.ConstData<float>(),
             saved_inv_var.ConstData<float>(),
             grad_input.MutableData<float>(),
             grad_scale.MutableData<float>(),
             grad_bias.MutableData<float>(),
             batch,
             features,
             epsilon
            );
    
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_Dropout
//==============================================================================

void CAIF_Dropout::Forward(
                          const float *input,
                          float *output,
                          float *mask,
                          const size_t n,
                          const float dropout_rate,
                          const bool training,
                          const uint32_t seed
                         )
{
  try
  {
    if(training==true&&dropout_rate>0.0f)
    {
      std::mt19937 gen(seed);
      std::uniform_real_distribution<float> dist(0.0f,1.0f);
      const float scale=1.0f/(1.0f-dropout_rate);
      
      for(size_t i=0;i<n;++i)
      {
        if(dist(gen)<dropout_rate)
        {
          mask[i]=0.0f;
          output[i]=0.0f;
        }
        else
        {
          mask[i]=scale;
          output[i]=input[i]*scale;
        }
      }
    }
    else
    {
      std::memcpy(output,input,n*sizeof(float));
      for(size_t i=0;i<n;++i)
      {
        mask[i]=1.0f;
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_Dropout::Backward(
                           const float *grad_output,
                           const float *mask,
                           float *grad_input,
                           const size_t n,
                           const float /*dropout_rate*/
                          )
{
  try
  {
    #ifdef _OPENMP
    #pragma omp parallel for simd
    #endif
    for(size_t i=0;i<n;++i)
    {
      grad_input[i]=grad_output[i]*mask[i];
    }
  }
  CAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Dropout::Forward(
                                const CAIF_Tensor &input,
                                const float dropout_rate,
                                const bool training,
                                CAIF_Tensor &mask
                               )
{
  try
  {
    const auto &shape=input.Shape();
    CAIF_Tensor output(input.Framework(),shape,input.Type());
    mask=CAIF_Tensor(input.Framework(),shape,input.Type());
    
    static std::random_device rd;
    const uint32_t seed=rd();
    
    Forward(
            input.ConstData<float>(),
            output.MutableData<float>(),
            mask.MutableData<float>(),
            input.NumElements(),
            dropout_rate,
            training,
            seed
           );
    
    return output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Dropout::Backward(
                                 const CAIF_Tensor &grad_output,
                                 const CAIF_Tensor &mask,
                                 const float dropout_rate
                                )
{
  try
  {
    CAIF_Tensor grad_input(grad_output.Framework(),grad_output.Shape(),grad_output.Type());
    
    Backward(
             grad_output.ConstData<float>(),
             mask.ConstData<float>(),
             grad_input.MutableData<float>(),
             grad_output.NumElements(),
             dropout_rate
            );
    
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

