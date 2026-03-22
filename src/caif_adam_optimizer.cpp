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
 * @file aif_adam_optimizer.cpp
 * @brief Implementation of the CAIF_AdamOptimizer class
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_adam_optimizer.h"
#include "caif_framework.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace instance
{
  CAIF_AdamOptimizer::CAIF_AdamOptimizer(
                                       CAIF_Framework &framework,
                                       const float learning_rate,
                                       const float beta1,
                                       const float beta2,
                                       const float epsilon,
                                       const float weight_decay
                                      )
    :CAIF_Optimizer(framework,learning_rate),
     _beta1(beta1),
     _beta2(beta2),
     _epsilon(epsilon),
     _weight_decay(weight_decay)
  {
  }

  std::vector<CAIF_Tensor> CAIF_AdamOptimizer::UpdateParameters(
                                                               const std::vector<CAIF_Tensor> &parameters,
                                                               const std::vector<CAIF_Tensor> &gradients
                                                              )
  {
    if(parameters.size()!=gradients.size())
    {
      THROW_CAIFE("Parameters and gradients must have the same size");
    }
      
    // Initialize moment estimates if this is the first update
    if(_m.empty())
    {
      _m.reserve(parameters.size());
      _v.reserve(parameters.size());
      
      for(const auto &param:parameters)
      {
        // Initialize first moment estimate (mean) to zero
        CAIF_Tensor m(param.Framework(),param.Shape(),param.Type());
        float *m_data=m.MutableData<float>();
        std::fill_n(m_data,m.NumElements(),0.0f);
        _m.push_back(std::move(m));
        
        // Initialize second moment estimate (variance) to zero
        CAIF_Tensor v(param.Framework(),param.Shape(),param.Type());
        float *v_data=v.MutableData<float>();
        std::fill_n(v_data,v.NumElements(),0.0f);
        _v.push_back(std::move(v));
      }
    }
    
    std::vector<CAIF_Tensor> updated_parameters;
    updated_parameters.reserve(parameters.size());
    
    // Increment iteration for bias correction
    IncrementIteration();
    const float bias_correction1=1.0f-std::pow(_beta1,static_cast<float>(Iteration()));
    const float bias_correction2=1.0f-std::pow(_beta2,static_cast<float>(Iteration()));
    
    // Get framework for GPU-accelerated updates
    CAIF_Framework &framework=Framework();
    
    for(size_t i=0;i<parameters.size();++i)
    {
      const auto &param=parameters[i];
      const auto &grad=gradients[i];
      
      // Validate shapes match
      if(param.Shape()!=grad.Shape())
      {
        THROW_CAIFE("Parameter and gradient shapes must match");
      }
      
      if(param.Type()!=CAIF_DataType::CAIF_DataType_e::Float32||
         grad.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("Adam optimizer currently only supports Float32 tensors");
      }
      
      // Handle gradient clipping if enabled
      CAIF_Tensor effective_grad=grad;
      if(g_caif_grad_clip_threshold<1e8f)
      {
        const float *gd=effective_grad.ConstData<float>();
        const size_t gn=effective_grad.NumElements();
        double sumsq=0.0;
        for(size_t k=0;k<gn;++k)
        {
          const double v=static_cast<double>(gd[k]);
          sumsq+=v*v;
        }
        const double l2=std::sqrt(sumsq);
        const double clip_threshold=static_cast<double>(g_caif_grad_clip_threshold);
        if(l2>clip_threshold&&l2>0.0)
        {
          const float scale=static_cast<float>(clip_threshold/l2);
          effective_grad=effective_grad.Multiply(scale);
        }
      }
      
      // Create a new tensor with its own buffer (deep copy) for the updated parameter
      CAIF_Tensor updated_param(param.Framework(),param.Shape(),param.Type());
      std::memcpy(updated_param.MutableData<float>(),param.ConstData<float>(),
                  param.NumElements()*sizeof(float));
      
      // Use fused Adam update - handles weight decay, moment updates, and param update in one pass
      framework.FusedAdamUpdate(updated_param,effective_grad,_m[i],_v[i],
                                LearningRate(),_beta1,_beta2,_epsilon,_weight_decay,
                                bias_correction1,bias_correction2);
      
      updated_parameters.push_back(std::move(updated_param));
    }
    
    return updated_parameters;
  }

  CAIF_OptimizerType_e CAIF_AdamOptimizer::OptimizerType()const
  {
    return CAIF_OptimizerType_e::Adam;
  }

  std::unique_ptr<CAIF_Optimizer> CAIF_AdamOptimizer::Clone()const
  {
    // Framework reference is copied from this optimizer via copy constructor
    auto cloned=std::make_unique<CAIF_AdamOptimizer>(*this);
    cloned->SetIteration(Iteration());
    cloned->_m=_m;
    cloned->_v=_v;
    return cloned;
  }

  void CAIF_AdamOptimizer::Reset()
  {
    SetIteration(0);
      
    // Reset first moment estimates to zero
    for(auto &m:_m)
    {
      float *m_data=m.MutableData<float>();
      std::fill_n(m_data,m.NumElements(),0.0f);
    }
      
    // Reset second moment estimates to zero
    for(auto &v:_v)
    {
      float *v_data=v.MutableData<float>();
      std::fill_n(v_data,v.NumElements(),0.0f);
    }
  }

  void CAIF_AdamOptimizer::ApplyGradients(
                                         std::vector<CAIF_Tensor> &parameters,
                                         const std::vector<CAIF_Tensor> &gradients
                                        )
  {
    if(parameters.size()!=gradients.size())
    {
      THROW_CAIFE("Parameters and gradients must have the same size");
    }

    // Initialize state if first time
    if(_m.empty())
    {
      _m.reserve(parameters.size());
      _v.reserve(parameters.size());
      for(const auto &param:parameters)
      {
        CAIF_Tensor m(param.Framework(),param.Shape(),param.Type());
        float *m_data=m.MutableData<float>();
        std::fill_n(m_data,m.NumElements(),0.0f);
        _m.push_back(std::move(m));

        CAIF_Tensor v(param.Framework(),param.Shape(),param.Type());
        float *v_data=v.MutableData<float>();
        std::fill_n(v_data,v.NumElements(),0.0f);
        _v.push_back(std::move(v));
      }
    }

    IncrementIteration();
    const float bias_correction1=1.0f-std::pow(_beta1,static_cast<float>(Iteration()));
    const float bias_correction2=1.0f-std::pow(_beta2,static_cast<float>(Iteration()));

    // Get framework for GPU-accelerated updates
    CAIF_Framework &framework=Framework();

    for(size_t i=0;i<parameters.size();++i)
    {
      auto &parameter_tensor=parameters[i];
      const auto &gradient_tensor=gradients[i];
      if(parameter_tensor.Shape()!=gradient_tensor.Shape())
      {
        THROW_CAIFE("Parameter and gradient shapes must match");
      }
      if(parameter_tensor.Type()!=CAIF_DataType::CAIF_DataType_e::Float32||
         gradient_tensor.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
      {
        THROW_CAIFE("Adam optimizer currently only supports Float32 tensors");
      }

      // Handle gradient clipping if enabled
      CAIF_Tensor effective_grad=gradient_tensor;
      if(g_caif_grad_clip_threshold<1e8f)
      {
        const float *grad_data=gradient_tensor.ConstData<float>();
        const size_t element_count=gradient_tensor.NumElements();

        double sumsq=0.0;
        for(size_t j=0;j<element_count;++j)
        {
          const double g=static_cast<double>(grad_data[j]);
          sumsq+=g*g;
        }
        const double l2=std::sqrt(sumsq);
        const double th=static_cast<double>(g_caif_grad_clip_threshold);
        if(l2>th&&l2>0.0)
        {
          const float clip_scale=static_cast<float>(th/l2);
          effective_grad=gradient_tensor.Multiply(clip_scale);
        }
      }

      // Use fused Adam update - handles weight decay, moment updates, and param update in one pass
      framework.FusedAdamUpdate(parameter_tensor,effective_grad,_m[i],_v[i],
                                LearningRate(),_beta1,_beta2,_epsilon,_weight_decay,
                                bias_correction1,bias_correction2);
    }
  }

  std::vector<CAIF_Tensor> CAIF_AdamOptimizer::State()const
  {
    std::vector<CAIF_Tensor> state;
    CAIF_Framework &framework=const_cast<CAIF_Framework&>(Framework());
    
    // First, add the iteration count as a single-element tensor
    CAIF_Tensor iteration_tensor(framework,{1},CAIF_DataType::CAIF_DataType_e::UInt32);
    uint32_t *iter_data=iteration_tensor.MutableData<uint32_t>();
    iter_data[0]=Iteration();
    state.push_back(iteration_tensor);
    
    // Add all first moment (m) tensors
    for(const auto &m:_m)
    {
      state.push_back(m);
    }
    
    // Add all second moment (v) tensors
    for(const auto &v:_v)
    {
      state.push_back(v);
    }
    
    return state;
  }
  
  void CAIF_AdamOptimizer::SetState(const std::vector<CAIF_Tensor> &state)
  {
    if(state.empty()==true)
    {
      THROW_CAIFE("Empty optimizer state provided");
    }
      
      // First tensor should be the iteration count
      if(state[0].Shape().size()!=1 || state[0].Shape()[0]!=1 || 
         state[0].Type()!=CAIF_DataType::CAIF_DataType_e::UInt32)
      {
        THROW_CAIFE("Invalid iteration tensor in optimizer state");
      }
      
      const uint32_t *iter_data=static_cast<const uint32_t*>(state[0].Data());
      SetIteration(iter_data[0]);
      
      // Remaining tensors should be m and v tensors
      // The number should be even (half m, half v)
      if((state.size()-1)%2!=0)
      {
        THROW_CAIFE("Invalid optimizer state: uneven number of m and v tensors");
      }
      
      const size_t param_count=(state.size()-1)/2;
      
      // Clear existing state
      _m.clear();
      _v.clear();
      
      // Copy m tensors
      for(size_t i=0; i<param_count; ++i)
      {
        _m.push_back(state[i+1]);
      }
      
      // Copy v tensors
      for(size_t i=0; i<param_count; ++i)
      {
        _v.push_back(state[i+1+param_count]);
      }
      
    return;
  }

  float CAIF_AdamOptimizer::Beta1()const
  {
    return _beta1;
  }

  void CAIF_AdamOptimizer::SetBeta1(const float beta1)
  {
    _beta1=beta1;
  }

  float CAIF_AdamOptimizer::Beta2()const
  {
    return _beta2;
  }

  void CAIF_AdamOptimizer::SetBeta2(const float beta2)
  {
    _beta2=beta2;
  }

  float CAIF_AdamOptimizer::Epsilon()const
  {
    return _epsilon;
  }

  void CAIF_AdamOptimizer::SetEpsilon(const float epsilon)
  {
    _epsilon=epsilon;
  }

  float CAIF_AdamOptimizer::WeightDecay()const
  {
    return _weight_decay;
  }

  void CAIF_AdamOptimizer::SetWeightDecay(const float weight_decay)
  {
    _weight_decay=weight_decay;
  }

  std::string CAIF_AdamOptimizer::Description()const
  {
    return "Adam Optimizer (lr="+std::to_string(LearningRate())+
           ", beta1="+std::to_string(_beta1)+
           ", beta2="+std::to_string(_beta2)+
           ", epsilon="+std::to_string(_epsilon)+
           ", weight_decay="+std::to_string(_weight_decay)+")";
  }
}//end instance namespace
