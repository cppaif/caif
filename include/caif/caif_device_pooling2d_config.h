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
// Configuration for CAIF_DevicePooling2D. All four fields (pool/stride height
// and width) are required by the constructor so a pooling window can never be
// built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DevicePooling2DConfig:public CAIF_Base
{
  public:
    // All four fields are required: the pooling-window and stride extents.
    CAIF_DevicePooling2DConfig(const uint32_t pool_height,
                               const uint32_t pool_width,
                               const uint32_t stride_height,
                               const uint32_t stride_width);

    // Pooling-window height.
    uint32_t PoolHeight()const{return _pool_height;}
    void SetPoolHeight(const uint32_t pool_height){_pool_height=pool_height;}

    // Pooling-window width.
    uint32_t PoolWidth()const{return _pool_width;}
    void SetPoolWidth(const uint32_t pool_width){_pool_width=pool_width;}

    // Vertical stride.
    uint32_t StrideHeight()const{return _stride_height;}
    void SetStrideHeight(const uint32_t stride_height){_stride_height=stride_height;}

    // Horizontal stride.
    uint32_t StrideWidth()const{return _stride_width;}
    void SetStrideWidth(const uint32_t stride_width){_stride_width=stride_width;}

  protected:

  private:
    uint32_t _pool_height;
    uint32_t _pool_width;
    uint32_t _stride_height;
    uint32_t _stride_width;
};

}//end instance namespace
