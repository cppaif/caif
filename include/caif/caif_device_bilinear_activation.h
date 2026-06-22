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
// Bilinear gated activation strategy (templated on <ComputeT, StorageT>).
// output = gate * up (identity gate activation)
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_gated_activation.h"
#include "caif_serialization_constants.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceBilinearActivation:public CAIF_DeviceGatedActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &gate_input,
                 const CAIF_DeviceTensor &up_input,
                 CAIF_DeviceTensor &output)const override;

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &cached_gate_input,
                  const CAIF_DeviceTensor &cached_up_input,
                  CAIF_DeviceTensor &grad_gate,
                  CAIF_DeviceTensor &grad_up)const override;

    std::string Description()const override{return g_serial_gated_bilinear;}
    std::unique_ptr<CAIF_DeviceActivation> Clone()const override;

  protected:

  private:
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceBilinearActivation<float,float>;
extern template class CAIF_DeviceBilinearActivation<float,__half>;
extern template class CAIF_DeviceBilinearActivation<float,__nv_bfloat16>;
extern template class CAIF_DeviceBilinearActivation<__half,float>;
extern template class CAIF_DeviceBilinearActivation<__half,__half>;
extern template class CAIF_DeviceBilinearActivation<__half,__nv_bfloat16>;
extern template class CAIF_DeviceBilinearActivation<__nv_bfloat16,float>;
extern template class CAIF_DeviceBilinearActivation<__nv_bfloat16,__half>;
extern template class CAIF_DeviceBilinearActivation<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceBilinearActivation<float,float>;
#endif

}//end instance namespace
