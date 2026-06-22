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
// Shared cuDNN-backend helpers used by every cudnn-backed layer
// (BatchNorm, Conv2D, MaxPooling2D, AveragePooling2D). Static utility
// class — never instantiated. Each cudnn-backed cpp used to carry a
// private copy of these helpers inside an anonymous namespace; those
// copies were consolidated here after the no-anonymous-namespace rule
// landed (2026-05-10).
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_data_type.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include <cudnn.h>
#endif

namespace instance
{

class CAIF_CudnnUtil:public CAIF_Base
{
  public:
    CAIF_CudnnUtil()=delete;

#ifdef USE_CAIF_CUDA
    static cudnnDataType_t CudnnDtypeFromStorage(CAIF_DataType::CAIF_DataType_e dt)
    {
      if(dt==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        return CUDNN_DATA_FLOAT;
      }
      if(dt==CAIF_DataType::CAIF_DataType_e::Float16)
      {
        return CUDNN_DATA_HALF;
      }
      if(dt==CAIF_DataType::CAIF_DataType_e::BFloat16)
      {
        return CUDNN_DATA_BFLOAT16;
      }
      THROW_CAIFE("CAIF_CudnnUtil: unsupported storage dtype for cuDNN");
    }

    static void CheckCudnn(cudnnStatus_t status,const std::string &what)
    {
      if(status!=CUDNN_STATUS_SUCCESS)
      {
        (void)what;
        THROW_CAIFE("CAIF_CudnnUtil: cuDNN call failed");
      }
    }
#endif

  protected:

  private:
};

}//end instance namespace
