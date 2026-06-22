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
// Configuration for CAIF_DeviceConv2D. All six fields (channel counts, kernel
// extents, stride extents) are required by the constructor so a convolution can
// never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceConv2DConfig:public CAIF_Base
{
  public:
    // All six fields are required: input/output channel counts, the kernel
    // height/width, and the stride height/width.
    CAIF_DeviceConv2DConfig(const uint32_t in_channels,
                            const uint32_t out_channels,
                            const uint32_t kernel_height,
                            const uint32_t kernel_width,
                            const uint32_t stride_height,
                            const uint32_t stride_width);

    // Number of input channels.
    uint32_t InChannels()const{return _in_channels;}
    void SetInChannels(const uint32_t in_channels){_in_channels=in_channels;}

    // Number of output channels (filters).
    uint32_t OutChannels()const{return _out_channels;}
    void SetOutChannels(const uint32_t out_channels){_out_channels=out_channels;}

    // Kernel height.
    uint32_t KernelHeight()const{return _kernel_height;}
    void SetKernelHeight(const uint32_t kernel_height){_kernel_height=kernel_height;}

    // Kernel width.
    uint32_t KernelWidth()const{return _kernel_width;}
    void SetKernelWidth(const uint32_t kernel_width){_kernel_width=kernel_width;}

    // Vertical stride.
    uint32_t StrideHeight()const{return _stride_height;}
    void SetStrideHeight(const uint32_t stride_height){_stride_height=stride_height;}

    // Horizontal stride.
    uint32_t StrideWidth()const{return _stride_width;}
    void SetStrideWidth(const uint32_t stride_width){_stride_width=stride_width;}

  protected:

  private:
    uint32_t _in_channels;
    uint32_t _out_channels;
    uint32_t _kernel_height;
    uint32_t _kernel_width;
    uint32_t _stride_height;
    uint32_t _stride_width;
};

}//end instance namespace
