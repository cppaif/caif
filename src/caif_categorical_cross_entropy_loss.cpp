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
 * @file aif_categorical_cross_entropy_loss.cpp
 * @brief Implementation of the CAIF_CategoricalCrossEntropyLoss class
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_categorical_cross_entropy_loss.h"
#include "caif_tensor_backend.h"
#include "caif_framework.h"
#include <cmath>
#include <algorithm>

namespace instance
{
  CAIF_CategoricalCrossEntropyLoss::CAIF_CategoricalCrossEntropyLoss(const float epsilon)
    :CAIF_LossFunction(),_epsilon(epsilon)
  {
  }

  CAIF_Tensor CAIF_CategoricalCrossEntropyLoss::ComputeLoss(
                                                          const CAIF_Tensor &predictions,
                                                          const CAIF_Tensor &targets
                                                         )const
  {
    try
    {
      // Validate predictions
      if(predictions.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("Categorical cross entropy loss currently only supports Float32 predictions");
      }
      
      const auto &pred_shape=predictions.Shape();
      if(pred_shape.size()!=2)
      {
        THROW_CAIFE("Categorical cross entropy loss expects 2D predictions [batch_size, num_classes]");
      }
      
      const uint32_t batch_size=pred_shape[0];
      const uint32_t num_classes=pred_shape[1];
      
      // Get predictions data
      const float *pred_data=predictions.ConstData<float>();
      
      // Check if targets are class indices or one-hot encoded
      bool targets_are_indices=AreTargetsClassIndices(predictions,targets);
      
      // Create loss tensor [batch_size]
      std::vector<uint32_t> loss_shape={batch_size};
      CAIF_Tensor loss_tensor(predictions.Framework(),loss_shape,CAIF_DataType::CAIF_DataType_e::Float32);
      
      float *loss_data=loss_tensor.MutableData<float>();
      
      if(targets_are_indices)
      {
        // Targets are class indices [batch_size]
        if(targets.Type()!=CAIF_DataType::CAIF_DataType_e::UInt32 &&
           targets.Type()!=CAIF_DataType::CAIF_DataType_e::Int32)
        {
          THROW_CAIFE("Class index targets must be UInt32 or Int32 type");
        }

        const uint32_t *target_indices=targets.ConstData<uint32_t>();

        // Compute loss for each sample
        for(uint32_t b=0;b<batch_size;++b)
        {
          const uint32_t target_class=target_indices[b];

          if(target_class>=num_classes)
          {
            THROW_CAIFE(("Target class index "+
              std::to_string(target_class)+
              " out of range [0, "+
              std::to_string(num_classes-1)+"]").c_str());
          }

          const uint32_t pred_idx=b*num_classes+target_class;
          // Clip to avoid log(0)
          const float pred=std::max(_epsilon,
            std::min(1.0f-_epsilon,pred_data[pred_idx]));
          
          loss_data[b]=-std::log(pred);
        }
      }
      else
      {
        // Targets are one-hot encoded [batch_size, num_classes]
        if(targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
        {
          THROW_CAIFE("One-hot encoded targets must be Float32 type");
        }
        
        if(targets.Shape()!=pred_shape)
        {
          THROW_CAIFE("One-hot encoded targets must have same shape as predictions");
        }
        
        const float *target_data=targets.ConstData<float>();
        
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
      }
      
      return loss_tensor;
    }
    CCAIF_CATCH_BLOCK()
  }

  CAIF_Tensor CAIF_CategoricalCrossEntropyLoss::ComputeGradient(
                                                              const CAIF_Tensor &predictions,
                                                              const CAIF_Tensor &targets
                                                             )const
  {
    try
    {
      // Validate predictions
      if(predictions.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("Categorical cross entropy gradient currently only supports Float32 predictions");
      }
      
      const auto &pred_shape=predictions.Shape();
      if(pred_shape.size()!=2)
      {
        THROW_CAIFE("Categorical cross entropy gradient expects 2D predictions [batch_size, num_classes]");
      }
      
      const uint32_t batch_size=pred_shape[0];
      const uint32_t num_classes=pred_shape[1];
      
      // Get predictions data
      const float *pred_data=predictions.ConstData<float>();
      
      // Create gradient tensor = predictions - targets (normalized by batch)
      CAIF_Tensor gradient_tensor(predictions.Framework(),pred_shape,CAIF_DataType::CAIF_DataType_e::Float32);

      float *grad_data=gradient_tensor.MutableData<float>();

      const float inv_batch=1.0f/static_cast<float>(batch_size);

      // Check if targets are class indices or one-hot encoded
      bool targets_are_indices=AreTargetsClassIndices(predictions,targets);

      if(targets_are_indices)
      {
        if(targets.Type()!=CAIF_DataType::CAIF_DataType_e::UInt32 &&
           targets.Type()!=CAIF_DataType::CAIF_DataType_e::Int32)
        {
          THROW_CAIFE("Class index targets must be UInt32 or Int32 type");
        }
        const uint32_t *target_indices=targets.ConstData<uint32_t>();

        for(uint32_t b=0;b<batch_size;++b)
        {
          // copy predictions row
          for(uint32_t c=0;c<num_classes;++c)
          {
            const uint32_t idx=b*num_classes+c;
            grad_data[idx]=pred_data[idx];
          }
          // subtract 1 at the target class
          const uint32_t t=target_indices[b];
          if(t>=num_classes)
          {
            THROW_CAIFE(("Target class index "+
              std::to_string(t)+
              " out of range [0, "+
              std::to_string(num_classes-1)+"]").c_str());
          }
          grad_data[b*num_classes+t]-=1.0f;
        }
      }
      else
      {
        if(targets.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
        {
          THROW_CAIFE("One-hot encoded targets must be Float32 type");
        }
        if(targets.Shape()!=pred_shape)
        {
          THROW_CAIFE("One-hot encoded targets must have same shape as predictions");
        }
        const float *target_data=targets.ConstData<float>();

        const size_t n=static_cast<size_t>(batch_size)*static_cast<size_t>(num_classes);
        for(size_t i=0;i<n;++i)
        {
          grad_data[i]=pred_data[i]-target_data[i];
        }
      }

      // normalize by batch size
      const size_t n_all=static_cast<size_t>(batch_size)*static_cast<size_t>(num_classes);
      for(size_t i=0;i<n_all;++i)
      {
        grad_data[i]*=inv_batch;
      }

      return gradient_tensor;
    }
    CCAIF_CATCH_BLOCK()
  }

  CAIF_LossType_e CAIF_CategoricalCrossEntropyLoss::LossType()const
  {
    return CAIF_LossType_e::CategoricalCrossEntropy;
  }

  std::unique_ptr<CAIF_LossFunction> CAIF_CategoricalCrossEntropyLoss::Clone()const
  {
    return std::make_unique<CAIF_CategoricalCrossEntropyLoss>(*this);
  }

  std::string CAIF_CategoricalCrossEntropyLoss::Description()const
  {
    return "Categorical Cross Entropy Loss (epsilon="+std::to_string(_epsilon)+")";
  }

  float CAIF_CategoricalCrossEntropyLoss::Epsilon()const
  {
    return _epsilon;
  }

  void CAIF_CategoricalCrossEntropyLoss::SetEpsilon(const float epsilon)
  {
    _epsilon=epsilon;
  }

  bool CAIF_CategoricalCrossEntropyLoss::AreTargetsClassIndices(
                                                              const CAIF_Tensor &predictions,
                                                              const CAIF_Tensor &targets
                                                             )const
  {
    const auto &pred_shape=predictions.Shape();
    const auto &target_shape=targets.Shape();
    
    // If targets have 1 dimension and batch size matches, they are class indices
    if(target_shape.size()==1&&target_shape[0]==pred_shape[0])
    {
      return true;
    }
    
    // If targets have same shape as predictions, they are one-hot encoded
    if(target_shape==pred_shape)
    {
      return false;
    }
    
    // Invalid target shape - let other validation catch this
    return false;
  }
}//end instance namespace
