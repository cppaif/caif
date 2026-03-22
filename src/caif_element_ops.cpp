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
 * @file aif_element_ops.cpp
 * @brief Implementation of CAIF_Tensor convenience wrappers for element-wise operations
 */

#include "caif_element_ops.h"
#include "caif_framework.h"

using namespace instance;

CAIF_Tensor CAIF_ElementOps::Add(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    if(a.Shape()!=b.Shape())
    {
      THROW_CAIFE("Tensor shapes must match for element-wise addition");
    }
    
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    Add(a.ConstData<float>(),b.ConstData<float>(),result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::AddScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    AddScalar(a.ConstData<float>(),scalar,result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::Sub(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    if(a.Shape()!=b.Shape())
    {
      THROW_CAIFE("Tensor shapes must match for element-wise subtraction");
    }
    
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    Sub(a.ConstData<float>(),b.ConstData<float>(),result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::SubScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    SubScalar(a.ConstData<float>(),scalar,result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::Mul(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    if(a.Shape()!=b.Shape())
    {
      THROW_CAIFE("Tensor shapes must match for element-wise multiplication");
    }
    
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    Mul(a.ConstData<float>(),b.ConstData<float>(),result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::MulScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    MulScalar(a.ConstData<float>(),scalar,result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::Div(const CAIF_Tensor &a,const CAIF_Tensor &b)
{
  try
  {
    if(a.Shape()!=b.Shape())
    {
      THROW_CAIFE("Tensor shapes must match for element-wise division");
    }
    
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    Div(a.ConstData<float>(),b.ConstData<float>(),result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::DivScalar(const CAIF_Tensor &a,const float scalar)
{
  try
  {
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    DivScalar(a.ConstData<float>(),scalar,result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ElementOps::Sqrt(const CAIF_Tensor &a)
{
  try
  {
    CAIF_Tensor result(a.Framework(),a.Shape(),a.Type());
    Sqrt(a.ConstData<float>(),result.MutableData<float>(),a.NumElements());
    return result;
  }
  CCAIF_CATCH_BLOCK()
}

float CAIF_ElementOps::Sum(const CAIF_Tensor &a)
{
  try
  {
    return Sum(a.ConstData<float>(),a.NumElements());
  }
  CCAIF_CATCH_BLOCK()
}

float CAIF_ElementOps::Mean(const CAIF_Tensor &a)
{
  try
  {
    return Mean(a.ConstData<float>(),a.NumElements());
  }
  CCAIF_CATCH_BLOCK()
}

