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
 * @file aif_sgd_optimizer.cpp
 * @brief Implementation of the CAIF_SGDOptimizer class
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_sgd_optimizer.h"
#include "caif_exception.h"
#include <algorithm>

namespace instance
{
  CAIF_SGDOptimizer::CAIF_SGDOptimizer(
                                     CAIF_Framework &framework,
                                     const float learning_rate,
                                     const float momentum,
                                     const float weight_decay
                                    )
    :CAIF_Optimizer(framework,learning_rate),
     _momentum(momentum),
     _weight_decay(weight_decay)
  {
  }

  std::vector<CAIF_Tensor> CAIF_SGDOptimizer::UpdateParameters(
                                                             const std::vector<CAIF_Tensor> &parameters,
                                                             const std::vector<CAIF_Tensor> &gradients
                                                            )
  {
      if(parameters.size()!=gradients.size())
      {
        THROW_CAIFE("Parameters and gradients must have the same size");
      }
      
      // Initialize velocity if this is the first update
      if(_velocity.empty()&&_momentum>0.0f)
      {
        _velocity.reserve(parameters.size());
        for(const auto &param:parameters)
        {
          CAIF_Tensor velocity(Framework(),param.Shape(),param.Type());
          // Initialize velocity to zero
          float *velocity_data=velocity.MutableData<float>();
          std::fill_n(velocity_data,velocity.NumElements(),0.0f);
          _velocity.push_back(std::move(velocity));
        }
      }
      
      std::vector<CAIF_Tensor> updated_parameters;
      updated_parameters.reserve(parameters.size());
      
      for(size_t i=0;i<parameters.size();++i)
      {
        const auto &param=parameters[i];
        const auto &grad=gradients[i];
        
        // Validate shapes match
        if(param.Shape()!=grad.Shape())
        {
          THROW_CAIFE("Parameter and gradient shapes must match");
        }
        
        if(param.Type()!=CAIF_DataType::CAIF_DataType_e::Float32 ||
           grad.Type()!=CAIF_DataType::CAIF_DataType_e::Float32)
        {
          THROW_CAIFE("SGD optimizer currently only supports Float32 tensors");
        }
        
        // Apply weight decay if specified (L2 regularization)
        CAIF_Tensor effective_grad=grad;
        if(_weight_decay>0.0f)
        {
          CAIF_Tensor weight_decay_term=param.Multiply(_weight_decay);
          effective_grad=grad.Add(weight_decay_term);
        }
        
        CAIF_Tensor updated_param(Framework());
        
        if(_momentum>0.0f&&i<_velocity.size())
        {
          // Update velocity: v=momentum * v - learning_rate * grad
          CAIF_Tensor momentum_term=_velocity[i].Multiply(_momentum);
          CAIF_Tensor grad_term=effective_grad.Multiply(LearningRate());
          _velocity[i]=momentum_term.Subtract(grad_term);
          
          // Update parameter: param=param + velocity
          updated_param=param.Add(_velocity[i]);
        }
        else
        {
          // Standard SGD: param=param - learning_rate * grad
          CAIF_Tensor update=effective_grad.Multiply(LearningRate());
          updated_param=param.Subtract(update);
        }
        
        updated_parameters.push_back(std::move(updated_param));
      }
      
      // Increment iteration counter
      IncrementIteration();
      
      return updated_parameters;
  }

  CAIF_OptimizerType_e CAIF_SGDOptimizer::OptimizerType()const
  {
    return CAIF_OptimizerType_e::SGD;
  }

  std::unique_ptr<CAIF_Optimizer> CAIF_SGDOptimizer::Clone()const
  {
    // Framework reference is copied from this optimizer via copy constructor
    auto cloned=std::make_unique<CAIF_SGDOptimizer>(*this);
    cloned->SetIteration(Iteration());
    cloned->_velocity=_velocity;
    return cloned;
  }

  void CAIF_SGDOptimizer::Reset()
  {
      SetIteration(0);
      
      // Reset velocity to zero
      for(auto &velocity:_velocity)
      {
        float *velocity_data=velocity.MutableData<float>();
        std::fill_n(velocity_data,velocity.NumElements(),0.0f);
      }
      return;
  }

  std::vector<CAIF_Tensor> CAIF_SGDOptimizer::State()const
  {
    // Return velocity tensors as the optimizer state
    std::vector<CAIF_Tensor> state=_velocity;
    
    // Add iteration count as a scalar tensor
    std::vector<uint32_t> scalar_shape={1};
    CAIF_Framework &framework_ref=const_cast<CAIF_Framework&>(Framework());
    CAIF_Tensor iteration_tensor(framework_ref,scalar_shape, CAIF_DataType::CAIF_DataType_e::UInt32);
    uint32_t* data=iteration_tensor.MutableData<uint32_t>();
    data[0]=Iteration();
    state.push_back(iteration_tensor);
    
    return state;
  }

  void CAIF_SGDOptimizer::SetState(const std::vector<CAIF_Tensor> &state)
  {
      // Check if state vector is valid
      if(state.empty())
      {
        THROW_CAIFE("Empty state provided to SGD optimizer");
      }
      
      // Last tensor should be the iteration count
      if(state.size()<_velocity.size()+1)
      {
        THROW_CAIFE("Incomplete state provided to SGD optimizer");
      }
      
      // Copy velocity tensors
      for(size_t i=0; i < _velocity.size(); ++i)
      {
        if(state[i].Shape()!=_velocity[i].Shape() || state[i].Type()!=_velocity[i].Type())
        {
          THROW_CAIFE("Incompatible velocity tensor shape or type");
        }
        _velocity[i]=state[i];
      }
      
      // Extract iteration count from the last tensor
      const CAIF_Tensor& iteration_tensor=state[state.size() - 1];
      if(iteration_tensor.Shape().size()!=1 || iteration_tensor.Shape()[0]!=1 || 
         iteration_tensor.Type()!=CAIF_DataType::CAIF_DataType_e::UInt32)
      {
        THROW_CAIFE("Invalid iteration tensor format");
      }
      
      const uint32_t* idata=iteration_tensor.ConstData<uint32_t>();
      SetIteration(idata[0]);
      return;
  }

  float CAIF_SGDOptimizer::Momentum()const
  {
    return _momentum;
  }

  void CAIF_SGDOptimizer::SetMomentum(const float momentum)
  {
    _momentum=momentum;
  }

  float CAIF_SGDOptimizer::WeightDecay()const
  {
    return _weight_decay;
  }

  void CAIF_SGDOptimizer::SetWeightDecay(const float weight_decay)
  {
    _weight_decay=weight_decay;
  }

  std::string CAIF_SGDOptimizer::Description()const
  {
    return "SGD Optimizer (lr="+std::to_string(LearningRate())+
           ", momentum="+std::to_string(_momentum)+
           ", weight_decay="+std::to_string(_weight_decay)+")";
  }

  void CAIF_SGDOptimizer::ApplyGradients(
                                        std::vector<CAIF_Tensor> &parameters,
                                        const std::vector<CAIF_Tensor> &gradients
                                       )
  {
      // Check input validity
      if(parameters.size()!=gradients.size())
      {
        THROW_CAIFE("Parameters and gradients size mismatch");
      }
      
      // Initialize velocity tensors if not already done
      if(_velocity.empty())
      {
        _velocity.reserve(parameters.size());
        for(const auto &param:parameters)
        {
          _velocity.push_back(CAIF_Tensor(Framework(),param.Shape(),param.Type()));
          
          // Initialize velocity to zero
          float *velocity_data=_velocity.back().MutableData<float>();
          std::fill_n(velocity_data,_velocity.back().NumElements(),0.0f);
        }
      }
      
      // Apply SGD update rule with momentum
      for(size_t i=0;i<parameters.size();++i)
      {
        float *param_data=parameters[i].MutableData<float>();
        const float *grad_data=gradients[i].ConstData<float>();
        float *velocity_data=_velocity[i].MutableData<float>();
        
        if(i==0)
        {
          ISE_Out::Out()<<"[SGD] param[0]="<<param_data[0]
                       <<" grad[0]="<<grad_data[0]
                       <<" lr="<<LearningRate()
                       <<"\n";
        }
        
        const size_t num_elements=parameters[i].NumElements();
        for(size_t j=0;j<num_elements;++j)
        {
          // v=momentum * v - learning_rate * grad
          velocity_data[j]=_momentum*velocity_data[j]-LearningRate()*grad_data[j];
          // param=param + v
          param_data[j]+=velocity_data[j];
        }
        
        if(i==0)
        {
          ISE_Out::Out()<<"[SGD] after update param[0]="<<param_data[0]<<"\n";
        }
      }
      
      IncrementIteration();
      return;
  }
}//end instance namespace
