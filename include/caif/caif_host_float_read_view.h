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
// Read-only RAII float view over a host-backed CAIF_DeviceTensor. The FP32
// path is zero-copy (the view aliases the tensor's storage). The FP16 / BF16
// path up-casts to FP32 on construction so element-wise / reduction ops can
// operate on a uniform float buffer regardless of the tensor's storage dtype.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_host_fp_cast.h"

#include <cstddef>
#include <vector>

namespace instance
{

class CAIF_HostFloatReadView:public CAIF_Base
{
  public:
    explicit CAIF_HostFloatReadView(const CAIF_DeviceTensor &t):_ptr(nullptr),
                                                                _scratch(),
                                                                _size(t.TotalElements())
    {
      if(t.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        // fp32 by branch gate
        SetPtr(static_cast<const float*>(t.DeviceDataRaw()));
      }
      else
      {
        SetScratch(CAIF_HostFpCast::UpcastToFloat(t));
        SetPtr(Scratch().data());
      }
    }

    const float *Data()const{return _ptr;}
    size_t Size()const{return _size;}

  protected:

  private:
    void SetPtr(const float *p){_ptr=p;}
    const std::vector<float> &Scratch()const{return _scratch;}
    void SetScratch(std::vector<float> v){_scratch=std::move(v);}

    const float *_ptr;
    std::vector<float> _scratch;
    size_t _size;
};

}//end instance namespace
