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
// Read-write RAII float view over a host-backed CAIF_DeviceTensor. The FP32
// path is zero-copy (the view aliases the tensor's storage). The FP16 / BF16
// path up-casts to FP32 on construction (so the body can read the current
// values) and down-casts back into the target tensor at destruction.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_host_fp_cast.h"

#include <cstddef>
#include <vector>

namespace instance
{

class CAIF_HostFloatReadWriteView:public CAIF_Base
{
  public:
    explicit CAIF_HostFloatReadWriteView(CAIF_DeviceTensor &t):_tensor(t),
                                                                _ptr(nullptr),
                                                                _scratch(),
                                                                _size(t.TotalElements()),
                                                                _needs_writeback(false)
    {
      if(t.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        // fp32 by branch gate
        SetPtr(static_cast<float*>(t.DeviceDataRaw()));
      }
      else
      {
        SetScratch(CAIF_HostFpCast::UpcastToFloat(t));
        SetPtr(ScratchMutable().data());
        SetNeedsWriteback(true);
      }
    }

    ~CAIF_HostFloatReadWriteView()
    {
      if(NeedsWriteback()==true)
      {
        try
        {
          CAIF_HostFpCast::DowncastFromFloat(ScratchMutable(),TensorMutable());
        }
        catch(...)
        {
        }
      }
    }

    float *Data(){return _ptr;}
    size_t Size()const{return _size;}

  protected:

  private:
    CAIF_DeviceTensor &TensorMutable(){return _tensor;}
    void SetPtr(float *p){_ptr=p;}
    std::vector<float> &ScratchMutable(){return _scratch;}
    void SetScratch(std::vector<float> v){_scratch=std::move(v);}
    bool NeedsWriteback()const{return _needs_writeback;}
    void SetNeedsWriteback(const bool b){_needs_writeback=b;}

    CAIF_DeviceTensor &_tensor;
    float *_ptr;
    std::vector<float> _scratch;
    size_t _size;
    bool _needs_writeback;
};

}//end instance namespace
