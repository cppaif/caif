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
 * @file aif_activations.cpp
 * @brief Implementation of CAIF_Tensor convenience wrappers for activation functions
 */

#include "caif_activations.h"
#include "caif_framework.h"

using namespace instance;

//==============================================================================
// CAIF_ReLU
//==============================================================================

CAIF_Tensor CAIF_ReLU::Forward(const CAIF_Tensor &input)
{
  try
  {
    CAIF_Tensor result(input.Framework(),input.Shape(),input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),input.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ReLU::Backward(const CAIF_Tensor &input,const CAIF_Tensor &grad_output)
{
  try
  {
    CAIF_Tensor grad_input(input.Framework(),input.Shape(),input.Type());
    Backward(
             input.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             input.NumElements()
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_Sigmoid
//==============================================================================

CAIF_Tensor CAIF_Sigmoid::Forward(const CAIF_Tensor &input)
{
  try
  {
    CAIF_Tensor result(input.Framework(),input.Shape(),input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),input.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Sigmoid::Backward(const CAIF_Tensor &output,const CAIF_Tensor &grad_output)
{
  try
  {
    CAIF_Tensor grad_input(output.Framework(),output.Shape(),output.Type());
    Backward(
             output.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             output.NumElements()
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_Tanh
//==============================================================================

CAIF_Tensor CAIF_Tanh::Forward(const CAIF_Tensor &input)
{
  try
  {
    CAIF_Tensor result(input.Framework(),input.Shape(),input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),input.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Tanh::Backward(const CAIF_Tensor &output,const CAIF_Tensor &grad_output)
{
  try
  {
    CAIF_Tensor grad_input(output.Framework(),output.Shape(),output.Type());
    Backward(
             output.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             output.NumElements()
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_LeakyReLU
//==============================================================================

CAIF_Tensor CAIF_LeakyReLU::Forward(const CAIF_Tensor &input,const float alpha)
{
  try
  {
    CAIF_Tensor result(input.Framework(),input.Shape(),input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),input.NumElements(),alpha);
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_LeakyReLU::Backward(const CAIF_Tensor &input,const CAIF_Tensor &grad_output,const float alpha)
{
  try
  {
    CAIF_Tensor grad_input(input.Framework(),input.Shape(),input.Type());
    Backward(
             input.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             input.NumElements(),
             alpha
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_ELU
//==============================================================================

CAIF_Tensor CAIF_ELU::Forward(const CAIF_Tensor &input,const float alpha)
{
  try
  {
    CAIF_Tensor result(input.Framework(),input.Shape(),input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),input.NumElements(),alpha);
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ELU::Backward(
                             const CAIF_Tensor &input,
                             const CAIF_Tensor &output,
                             const CAIF_Tensor &grad_output,
                             const float alpha
                            )
{
  try
  {
    CAIF_Tensor grad_input(input.Framework(),input.Shape(),input.Type());
    Backward(
             input.ConstData<float>(),
             output.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             input.NumElements(),
             alpha
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_GELU
//==============================================================================

CAIF_Tensor CAIF_GELU::Forward(const CAIF_Tensor &input)
{
  try
  {
    CAIF_Tensor result(input.Framework(),input.Shape(),input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),input.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_GELU::Backward(const CAIF_Tensor &input,const CAIF_Tensor &grad_output)
{
  try
  {
    CAIF_Tensor grad_input(input.Framework(),input.Shape(),input.Type());
    Backward(
             input.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             input.NumElements()
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_Swish
//==============================================================================

CAIF_Tensor CAIF_Swish::Forward(const CAIF_Tensor &input)
{
  try
  {
    CAIF_Tensor result(input.Framework(),input.Shape(),input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),input.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Swish::Backward(const CAIF_Tensor &input,const CAIF_Tensor &output,const CAIF_Tensor &grad_output)
{
  try
  {
    CAIF_Tensor grad_input(input.Framework(),input.Shape(),input.Type());
    Backward(
             input.ConstData<float>(),
             output.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             input.NumElements()
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

//==============================================================================
// CAIF_Softmax
//==============================================================================

CAIF_Tensor CAIF_Softmax::Forward(const CAIF_Tensor &input)
{
  try
  {
    const auto &shape=input.Shape();
    if(shape.size()<1)
    {
      THROW_CAIFE("Softmax requires at least 1D tensor");
    }
    
    size_t batch_size=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      batch_size*=shape[i];
    }
    const size_t num_classes=shape[shape.size()-1];
    
    CAIF_Tensor result(input.Framework(),shape,input.Type());
    Forward(input.ConstData<float>(),result.MutableData<float>(),batch_size,num_classes);
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Softmax::Backward(const CAIF_Tensor &output,const CAIF_Tensor &grad_output)
{
  try
  {
    const auto &shape=output.Shape();
    if(shape.size()<1)
    {
      THROW_CAIFE("Softmax backward requires at least 1D tensor");
    }
    
    size_t batch_size=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      batch_size*=shape[i];
    }
    const size_t num_classes=shape[shape.size()-1];
    
    CAIF_Tensor grad_input(output.Framework(),shape,output.Type());
    Backward(
             output.ConstData<float>(),
             grad_output.ConstData<float>(),
             grad_input.MutableData<float>(),
             batch_size,
             num_classes
            );
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

