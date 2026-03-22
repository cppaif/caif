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
 * @file aif_cross_entropy_loss.cpp
 * @brief Implementation of the CAIF_CrossEntropyLoss class
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_cross_entropy_loss.h"

#include "ise_lib/ise_out.h"
#include "caif_tensor_backend.h"
#include "caif_framework.h"

#include <cmath>
#include <algorithm>
#include <iostream> // Added for detailed logging

namespace instance
{
  CAIF_CrossEntropyLoss::CAIF_CrossEntropyLoss(const float epsilon)
    :CAIF_LossFunction(),_epsilon(epsilon)
  {
  }

  CAIF_Tensor CAIF_CrossEntropyLoss::ComputeLoss(
                                               const CAIF_Tensor &predictions,
                                               const CAIF_Tensor &targets
                                              )const
  {
      // Validate input shapes
      if(predictions.Shape()!=targets.Shape())
      {
        THROW_CAIFE("Predictions and targets must have the same shape");
      }
      
      if(predictions.Type()!=CAIF_DataType::CAIF_DataType_e::Float32 ||
         targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("Cross entropy loss currently only supports Float32 data type");
      }
      
      const auto &shape=predictions.Shape();
      if(shape.size()!=2)
      {
        THROW_CAIFE("Cross entropy loss expects 2D tensors [batch_size, num_classes]");
      }
      
      const uint32_t batch_size=shape[0];
      const uint32_t num_classes=shape[1];
      
      // Get data pointers
      const float *pred_data=predictions.ConstData<float>();
      const float *target_data=targets.ConstData<float>();
      
      // Create loss tensor [batch_size]
      std::vector<uint32_t> loss_shape={batch_size};
      CAIF_Tensor loss_tensor(predictions.Framework(),loss_shape,CAIF_DataType::CAIF_DataType_e::Float32);
      
      float *loss_data=loss_tensor.MutableData<float>();
      
      // Compute cross entropy loss for each sample
      for(uint32_t b=0;b<batch_size;++b)
      {
        float sample_loss=0.0f;
        
        for(uint32_t c=0;c<num_classes;++c)
        {
          const uint32_t idx=b*num_classes+c;
          const float pred=std::max(_epsilon,std::min(1.0f-_epsilon,pred_data[idx]));  // Clip to avoid log(0)
          const float target=target_data[idx];
          
          if(target>0.0f)  // Only compute loss for non-zero targets
          {
            sample_loss-=target*std::log(pred);
          }
        }
        
        loss_data[b]=sample_loss;
      }
      
      return loss_tensor;
  }

  CAIF_Tensor CAIF_CrossEntropyLoss::ComputeGradient(
                                                   const CAIF_Tensor &predictions,
                                                   const CAIF_Tensor &targets
                                                  )const
  {
      DbgLog()<<"[DEBUG] CAIF_CrossEntropyLoss::ComputeGradient - Starting gradient computation\n";
      DbgLog()<<"[DEBUG] Predictions: "<<predictions.ToString()<<"\n";
      DbgLog()<<"[DEBUG] Targets: "<<targets.ToString()<<"\n";
      
      // Validate input shapes
      if(predictions.Shape()!=targets.Shape())
      {
        ErrorLog()<<"[ERROR] Shape mismatch - Predictions: "
                  <<predictions.ToString()
                  <<", Targets: "<<targets.ToString()<<"\n";
        THROW_CAIFE("Predictions and targets must have the same shape");
      }
      
      if(predictions.Type()!=CAIF_DataType::CAIF_DataType_e::Float32 ||
         targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
      {
        ErrorLog()<<"[ERROR] Invalid data type - Predictions: "
                  <<static_cast<int>(predictions.Type().Value())
                  <<", Targets: "
                  <<static_cast<int>(targets.Type().Value())<<"\n";
        THROW_CAIFE("Cross entropy gradient currently only supports Float32 data type");
      }
      
      const auto &shape=predictions.Shape();
      if(shape.size()!=2)
      {
        ErrorLog()<<"[ERROR] Invalid tensor dimensions: "<<predictions.ToString()<<"\n";
        THROW_CAIFE("Cross entropy gradient expects 2D tensors [batch_size, num_classes]");
      }
      
      const uint32_t batch_size=shape[0];
      const uint32_t num_classes=shape[1];
      
      // Get data pointers
      const float *pred_data=predictions.ConstData<float>();
      const float *target_data=targets.ConstData<float>();
      
      // Create gradient tensor with same shape as predictions
      CAIF_Tensor gradient_tensor(predictions.Framework(),shape,CAIF_DataType::CAIF_DataType_e::Float32);
      
      float *grad_data=gradient_tensor.MutableData<float>();
      
      // Compute gradient: -targets / predictions (normalized by batch size)
      for(uint32_t b=0;b<batch_size;++b)
      {
        for(uint32_t c=0;c<num_classes;++c)
        {
          const uint32_t idx=b*num_classes+c;
          // Clip to avoid division by 0
          const float pred=std::max(_epsilon,
            std::min(1.0f-_epsilon,pred_data[idx]));
          const float target=target_data[idx];
          
          // Gradient: -target / prediction (normalized by batch size)
          grad_data[idx]=-target/(pred*static_cast<float>(batch_size));
        }
      }
      
      DbgLog()<<"[DEBUG] Computed gradient: "<<gradient_tensor.ToString()<<"\n";
      return gradient_tensor;
  }

  CAIF_LossType_e CAIF_CrossEntropyLoss::LossType()const
  {
    return CAIF_LossType_e::CrossEntropy;
  }

  std::unique_ptr<CAIF_LossFunction> CAIF_CrossEntropyLoss::Clone()const
  {
    return std::make_unique<CAIF_CrossEntropyLoss>(*this);
  }

  std::string CAIF_CrossEntropyLoss::Description()const
  {
    return "Cross Entropy Loss (epsilon="+std::to_string(_epsilon)+")";
  }

  float CAIF_CrossEntropyLoss::Epsilon()const
  {
    return _epsilon;
  }

  void CAIF_CrossEntropyLoss::SetEpsilon(const float epsilon)
  {
    _epsilon=epsilon;
  }
}//end instance namespace
