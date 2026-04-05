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
 * @file aif_batch_normalization_layer.cpp
 * @brief Implementation of the CAIF_BatchNormalizationLayer class
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_batch_normalization_layer.h"
#include "caif_framework.h"
#include <sstream>
#include <cmath>

namespace instance
{
  CAIF_BatchNormalizationLayer::CAIF_BatchNormalizationLayer(
                                                           CAIF_Framework &framework,
                                                           const float epsilon,
                                                           const float momentum,
                                                           const bool affine
                                                          )
    :CAIF_Layer(framework),
     _epsilon(epsilon),
     _momentum(momentum),
     _affine(affine),
     _num_features(0),
     _scale(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _shift(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _running_mean(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _running_var(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _last_input(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _last_normalized(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _last_mean(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _last_variance(framework,{1},CAIF_DataType::CAIF_DataType_e::Float32),
     _scale_gradient(framework),
     _shift_gradient(framework)
  {
    SetInitialized(false);
  }

  CAIF_BatchNormalizationLayer::CAIF_BatchNormalizationLayer(const CAIF_BatchNormalizationLayer &other)
    :CAIF_Layer(other),
     _epsilon(other._epsilon),
     _momentum(other._momentum),
     _affine(other._affine),
     _num_features(other._num_features),
     _scale(other._scale),
     _shift(other._shift),
     _running_mean(other._running_mean),
     _running_var(other._running_var),
     _last_input(other._last_input),
     _last_normalized(other._last_normalized),
     _last_mean(other._last_mean),
     _last_variance(other._last_variance),
     _scale_gradient(other._scale_gradient),
     _shift_gradient(other._shift_gradient)
  {
  }

  CAIF_BatchNormalizationLayer::CAIF_BatchNormalizationLayer(CAIF_BatchNormalizationLayer &&other)noexcept
    :CAIF_Layer(std::move(other)),
     _epsilon(other._epsilon),
     _momentum(other._momentum),
     _affine(other._affine),
     _num_features(other._num_features),
     _scale(std::move(other._scale)),
     _shift(std::move(other._shift)),
     _running_mean(std::move(other._running_mean)),
     _running_var(std::move(other._running_var)),
     _last_input(std::move(other._last_input)),
     _last_normalized(std::move(other._last_normalized)),
     _last_mean(std::move(other._last_mean)),
     _last_variance(std::move(other._last_variance)),
     _scale_gradient(std::move(other._scale_gradient)),
     _shift_gradient(std::move(other._shift_gradient))
  {
  }

  CAIF_BatchNormalizationLayer &CAIF_BatchNormalizationLayer::operator=(const CAIF_BatchNormalizationLayer &other)
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(other);
      _epsilon=other._epsilon;
      _momentum=other._momentum;
      _affine=other._affine;
      _num_features=other._num_features;
      _scale=other._scale;
      _shift=other._shift;
      _running_mean=other._running_mean;
      _running_var=other._running_var;
      _last_input=other._last_input;
      _last_normalized=other._last_normalized;
      _last_mean=other._last_mean;
      _last_variance=other._last_variance;
      _scale_gradient=other._scale_gradient;
      _shift_gradient=other._shift_gradient;
    }
    return *this;
  }

  CAIF_BatchNormalizationLayer &CAIF_BatchNormalizationLayer::operator=(CAIF_BatchNormalizationLayer &&other)noexcept
  {
    if(this!=&other)
    {
      CAIF_Layer::operator=(std::move(other));
      _epsilon=other._epsilon;
      _momentum=other._momentum;
      _affine=other._affine;
      _num_features=other._num_features;
      _scale=std::move(other._scale);
      _shift=std::move(other._shift);
      _running_mean=std::move(other._running_mean);
      _running_var=std::move(other._running_var);
      _last_input=std::move(other._last_input);
      _last_normalized=std::move(other._last_normalized);
      _last_mean=std::move(other._last_mean);
      _last_variance=std::move(other._last_variance);
      _scale_gradient=std::move(other._scale_gradient);
      _shift_gradient=std::move(other._shift_gradient);
    }
    return *this;
  }

  CAIF_Tensor CAIF_BatchNormalizationLayer::Forward(
                                                  const CAIF_Tensor &input,
                                                  const bool training
                                                 )
  {
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Batch normalization layer not initialized");
    }
    
    if(input.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Batch normalization currently only supports Float32 data type");
    }
    
    // Store input for backward pass
    _last_input=input;
    
    // Use backend via framework
    CAIF_Framework &framework=Framework();
    CAIF_Tensor saved_mean(framework);
    CAIF_Tensor saved_inv_var(framework);
    
    CAIF_Tensor output=framework.BatchNormForward(input,_scale,_shift,
                                                  _running_mean,_running_var,
                                                  _epsilon,_momentum,training,
                                                  saved_mean,saved_inv_var);
    
    // Store for backward pass
    _last_mean=saved_mean;
    _last_variance=saved_inv_var;  // Actually stores inverse variance
    _last_normalized=output;
    
    return output;
  }

  CAIF_Tensor CAIF_BatchNormalizationLayer::Backward(const CAIF_Tensor &gradient)
  {
    if(IsInitialized()==false)
    {
      THROW_CAIFE("Batch normalization layer not initialized");
    }
    
    // Use backend via framework
    CAIF_Framework &framework=Framework();
    CAIF_Tensor grad_scale(framework);
    CAIF_Tensor grad_bias(framework);
    
    CAIF_Tensor input_gradient=framework.BatchNormBackward(gradient,_last_input,_scale,
                                                          _last_mean,_last_variance,
                                                          _epsilon,
                                                          grad_scale,grad_bias);
    
    // Store parameter gradients for optimizer
    if(_affine==true)
    {
      _scale_gradient=grad_scale;
      _shift_gradient=grad_bias;
    }
    
    return input_gradient;
  }

  void CAIF_BatchNormalizationLayer::Initialize(
                                               const std::vector<uint32_t> &input_shape,
                                               const uint32_t seed
                                              )
  {
    (void)seed;
    if(input_shape.empty())
    {
      THROW_CAIFE("Input shape cannot be empty");
    }
    
    SetInputShape(input_shape);
    SetOutputShape(input_shape);  // Output has same shape as input
    
    // Determine number of features (last dimension)
    _num_features=input_shape.back();
    
    // Initialize parameter tensors
    std::vector<uint32_t> param_shape={_num_features};
    CAIF_Framework &framework=Framework();
    
    _running_mean=CAIF_Tensor(framework,param_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    _running_var=CAIF_Tensor(framework,param_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    
    if(_affine==true)
    {
      _scale=CAIF_Tensor(framework,param_shape,CAIF_DataType::CAIF_DataType_e::Float32);
      _shift=CAIF_Tensor(framework,param_shape,CAIF_DataType::CAIF_DataType_e::Float32);
    }
    
    // Initialize parameters
    InitializeParameters();
    
    SetInitialized(true);
  }

  std::vector<uint32_t> CAIF_BatchNormalizationLayer::CalculateOutputShape(
    const std::vector<uint32_t> &input_shape)const
  {
    // Batch normalization preserves input shape
    return input_shape;
  }

  std::unique_ptr<CAIF_Layer> CAIF_BatchNormalizationLayer::Clone()const
  {
    // Framework reference is copied from this layer via copy constructor
    return std::make_unique<CAIF_BatchNormalizationLayer>(*this);
  }

  std::string CAIF_BatchNormalizationLayer::Description()const
  {
    std::ostringstream oss;
    oss<<"Batch Normalization Layer (features="<<_num_features<<", epsilon="<<_epsilon
       <<", momentum="<<_momentum<<", affine="<<(_affine?"true":"false")<<")";
    if(IsInitialized()==true)
    {
      oss<<" (";
      const auto &input_shape=InputShape();
      for(size_t i=0;i<input_shape.size();++i)
      {
        if(i>0)oss<<"x";
        oss<<input_shape[i];
      }
      oss<<")";
    }
    return oss.str();
  }

  std::vector<CAIF_Tensor> CAIF_BatchNormalizationLayer::Parameters()const
  {
    if(_affine==false)
    {
      return std::vector<CAIF_Tensor>{};  // No learnable parameters
    }
    
    std::vector<CAIF_Tensor> parameters;
    parameters.push_back(_scale);
    parameters.push_back(_shift);
    return parameters;
  }

  std::vector<CAIF_Tensor> CAIF_BatchNormalizationLayer::ParameterGradients()const
  {
    if(_affine==false)
    {
      return std::vector<CAIF_Tensor>{};  // No learnable parameters
    }
    
    std::vector<CAIF_Tensor> gradients;
    gradients.push_back(_scale_gradient);
    gradients.push_back(_shift_gradient);
    return gradients;
  }

  size_t CAIF_BatchNormalizationLayer::ParameterCount()const
  {
    if(_affine==true)
    {
      return 2;
    }
    return 0;
  }

  CAIF_Tensor &CAIF_BatchNormalizationLayer::ParameterRef(const size_t index)
  {
    try
    {
      if(_affine==false)
      {
        THROW_CAIFE("No parameters when affine is disabled");
      }
      if(index==0)
      {
        return _scale;
      }
      if(index==1)
      {
        return _shift;
      }
      THROW_CAIFE("Parameter index out of range");
    }
    CAIF_CATCH_BLOCK()
  }

  const CAIF_Tensor &CAIF_BatchNormalizationLayer::ParameterRef(const size_t index)const
  {
    try
    {
      if(_affine==false)
      {
        THROW_CAIFE("No parameters when affine is disabled");
      }
      if(index==0)
      {
        return _scale;
      }
      if(index==1)
      {
        return _shift;
      }
      THROW_CAIFE("Parameter index out of range");
    }
    CAIF_CATCH_BLOCK()
  }

  CAIF_Tensor &CAIF_BatchNormalizationLayer::GradientRef(const size_t index)
  {
    try
    {
      if(_affine==false)
      {
        THROW_CAIFE("No gradients when affine is disabled");
      }
      if(index==0)
      {
        return _scale_gradient;
      }
      if(index==1)
      {
        return _shift_gradient;
      }
      THROW_CAIFE("Gradient index out of range");
    }
    CAIF_CATCH_BLOCK()
  }

  const CAIF_Tensor &CAIF_BatchNormalizationLayer::GradientRef(const size_t index)const
  {
    try
    {
      if(_affine==false)
      {
        THROW_CAIFE("No gradients when affine is disabled");
      }
      if(index==0)
      {
        return _scale_gradient;
      }
      if(index==1)
      {
        return _shift_gradient;
      }
      THROW_CAIFE("Gradient index out of range");
    }
    CAIF_CATCH_BLOCK()
  }

  void CAIF_BatchNormalizationLayer::UpdateParameters(
                                                     const std::vector<CAIF_Tensor> &new_parameters
                                                    )
  {
    if(_affine==false)
    {
      if(new_parameters.empty()==false)
      {
        THROW_CAIFE("No parameters expected when affine is disabled");
      }
      return;
    }
    
    if(new_parameters.size()!=2)
    {
      THROW_CAIFE("Expected 2 parameters (scale and shift)");
    }
    
    _scale=new_parameters[0];
    _shift=new_parameters[1];
  }

  void CAIF_BatchNormalizationLayer::ResetParameters(const uint32_t seed)
  {
    (void)seed;
    InitializeParameters();
  }

  void CAIF_BatchNormalizationLayer::InitializeParameters()
  {
    // Initialize running statistics to 0 (mean) and 1 (variance)
    float *mean_data=_running_mean.MutableData<float>();
    float *var_data=_running_var.MutableData<float>();
    for(uint32_t i=0;i<_num_features;++i)
    {
      mean_data[i]=0.0f;
      var_data[i]=1.0f;
    }

    // Initialize affine parameters if enabled
    if(_affine==true)
    {
      float *scale_data=_scale.MutableData<float>();
      float *shift_data=_shift.MutableData<float>();
      for(uint32_t i=0;i<_num_features;++i)
      {
        scale_data[i]=1.0f;  // Initialize scale to 1
        shift_data[i]=0.0f;  // Initialize shift to 0
      }
    }
  }

  void CAIF_BatchNormalizationLayer::ComputeBatchStatistics(
                                                           const CAIF_Tensor &input,
                                                           CAIF_Tensor &mean,
                                                           CAIF_Tensor &variance
                                                          )const
  {
    const float *input_data=input.ConstData<float>();
    float *mean_data=mean.MutableData<float>();
    float *var_data=variance.MutableData<float>();
    
    const uint32_t num_elements_per_feature=input.NumElements()/_num_features;
    
    // Initialize mean and variance to zero
    for(uint32_t f=0;f<_num_features;++f)
    {
      mean_data[f]=0.0f;
      var_data[f]=0.0f;
    }
    
    // Compute mean
    for(uint32_t i=0;i<input.NumElements();++i)
    {
      const uint32_t feature_idx=i%_num_features;
      mean_data[feature_idx]+=input_data[i];
    }
    
    for(uint32_t f=0;f<_num_features;++f)
    {
      mean_data[f]/=static_cast<float>(num_elements_per_feature);
    }
    
    // Compute variance
    for(uint32_t i=0;i<input.NumElements();++i)
    {
      const uint32_t feature_idx=i%_num_features;
      const float diff=input_data[i]-mean_data[feature_idx];
      var_data[feature_idx]+=diff*diff;
    }
    
    for(uint32_t f=0;f<_num_features;++f)
    {
      var_data[f]/=static_cast<float>(num_elements_per_feature);
    }
  }

  void CAIF_BatchNormalizationLayer::UpdateRunningStatistics(
                                                            const CAIF_Tensor &batch_mean,
                                                            const CAIF_Tensor &batch_var
                                                           )
  {
    float *running_mean_data=_running_mean.MutableData<float>();
    float *running_var_data=_running_var.MutableData<float>();
    const float *batch_mean_data=batch_mean.ConstData<float>();
    const float *batch_var_data=batch_var.ConstData<float>();
    for(uint32_t f=0;f<_num_features;++f)
    {
      running_mean_data[f]=_momentum*running_mean_data[f]+(1.0f-_momentum)*batch_mean_data[f];
      running_var_data[f]=_momentum*running_var_data[f]+(1.0f-_momentum)*batch_var_data[f];
    }
  }

  CAIF_Tensor CAIF_BatchNormalizationLayer::ApplyNormalization(
                                                             const CAIF_Tensor &input,
                                                             const CAIF_Tensor &mean,
                                                             const CAIF_Tensor &variance
                                                            )const
  {
    CAIF_Tensor output=input;  // Copy input shape and data
    
    const float *input_data=input.ConstData<float>();
    float *output_data=output.MutableData<float>();
    const float *mean_data=mean.ConstData<float>();
    const float *var_data=variance.ConstData<float>();
    
    // Apply normalization: (x - mean) / sqrt(var + epsilon)
    for(uint32_t i=0;i<input.NumElements();++i)
    {
      const uint32_t feature_idx=i%_num_features;
      const float normalized=(input_data[i]-mean_data[feature_idx])/
                              std::sqrt(var_data[feature_idx]+_epsilon);
      output_data[i]=normalized;
    }
    
    return output;
  }

  CAIF_Tensor CAIF_BatchNormalizationLayer::ApplyAffineTransform(const CAIF_Tensor &normalized)const
  {
    CAIF_Tensor output=normalized;  // Copy input
    
    const float *normalized_data=normalized.ConstData<float>();
    float *output_data=output.MutableData<float>();
    const float *scale_data=_scale.ConstData<float>();
    const float *shift_data=_shift.ConstData<float>();
    
    // Apply affine transformation: scale * normalized + shift
    for(uint32_t i=0;i<normalized.NumElements();++i)
    {
      const uint32_t feature_idx=i%_num_features;
      output_data[i]=scale_data[feature_idx]*normalized_data[i]+shift_data[feature_idx];
    }
    
    return output;
  }
}//end instance namespace
