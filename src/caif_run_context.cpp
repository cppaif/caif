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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// CAIF_RunContext implementation.
//------------------------------------------------------------------------------
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include <cublas_v2.h>
#endif

namespace instance
{

CAIF_RunContext::Subsystem_e CAIF_RunContext::CurrentSubsystem()const
{
  if(_subsystem_stack.empty()==true)
  {
    return Subsystem_e::None_e;
  }
  return _subsystem_stack.back();
}

void CAIF_RunContext::PopSubsystem()
{
  try
  {
    if(_subsystem_stack.empty()==true)
    {
      THROW_CAIFE("CAIF_RunContext::PopSubsystem: subsystem stack is empty");
    }
    _subsystem_stack.pop_back();
  }
  CAIF_CATCH_BLOCK();
}

CAIF_CudaStream &CAIF_RunContext::Stream()const
{
  if(_stream==nullptr)
  {
    THROW_CAIFE("CAIF_RunContext::Stream: stream is not set on run context");
  }
  return *_stream;
}

const CAIF_DeviceTensor &CAIF_RunContext::EncoderContext()const
{
  if(_encoder_context==nullptr)
  {
    THROW_CAIFE("EncoderContext is not set; gate reads with HasEncoderContext()");
  }
  return *_encoder_context;
}

CAIF_DeviceTensor &CAIF_RunContext::GradEncoderContext()const
{
  if(_grad_encoder_context==nullptr)
  {
    THROW_CAIFE("GradEncoderContext is not set; gate reads with HasGradEncoderContext()");
  }
  return *_grad_encoder_context;
}

const CAIF_DeviceTensor &CAIF_RunContext::PositionBias()const
{
  if(_position_bias==nullptr)
  {
    THROW_CAIFE("PositionBias is not set; gate reads with HasPositionBias()");
  }
  return *_position_bias;
}

CAIF_DeviceTensor &CAIF_RunContext::GradPositionBias()const
{
  if(_grad_position_bias==nullptr)
  {
    THROW_CAIFE("GradPositionBias is not set; gate reads with HasGradPositionBias()");
  }
  return *_grad_position_bias;
}

const CAIF_DeviceTensor &CAIF_RunContext::PrefixLengths()const
{
  if(_prefix_lengths==nullptr)
  {
    THROW_CAIFE("PrefixLengths is not set; gate reads with HasPrefixLengths()");
  }
  return *_prefix_lengths;
}

int32_t CAIF_RunContext::ComputeTypeFor(const CAIF_DataType::CAIF_DataType_e dt)const
{
#ifdef USE_CAIF_CUDA
  if(dt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    if(CAIF_Settings::MatmulMode()==CAIF_Settings::MatmulMode_e::Accuracy_e)
    {
      return static_cast<int32_t>(CUBLAS_COMPUTE_32F);
    }
    return static_cast<int32_t>(CUBLAS_COMPUTE_32F_FAST_TF32);
  }
  return static_cast<int32_t>(CUBLAS_COMPUTE_32F);
#else
  (void)dt;
  return 0;
#endif
}

int32_t CAIF_RunContext::ComputeTypeFor(const CAIF_DataType::CAIF_DataType_e input_dt,
                                        const CAIF_DataType::CAIF_DataType_e compute_dt)const
{
#ifdef USE_CAIF_CUDA
  if(compute_dt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    return ComputeTypeFor(input_dt);
  }
  if(input_dt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    if(compute_dt==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      return static_cast<int32_t>(CUBLAS_COMPUTE_32F_FAST_16BF);
    }
    if(compute_dt==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      return static_cast<int32_t>(CUBLAS_COMPUTE_32F_FAST_16F);
    }
  }
  return static_cast<int32_t>(CUBLAS_COMPUTE_32F);
#else
  (void)input_dt;
  (void)compute_dt;
  return 0;
#endif
}

}//end instance namespace
