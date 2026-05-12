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
// CAIF_DeviceReLUActivation — pointwise ReLU: f(x) = max(0, x).
// Templated on <ComputeT, StorageT>; calls the templated kernel launcher
// directly with StorageT-typed pointers.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_pointwise_activation.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceReLUActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override;

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override;

    std::string Description()const override;
    std::unique_ptr<CAIF_DeviceActivation> Clone()const override;

  protected:

  private:
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceReLUActivation<float,float>;
extern template class CAIF_DeviceReLUActivation<float,__half>;
extern template class CAIF_DeviceReLUActivation<float,__nv_bfloat16>;
extern template class CAIF_DeviceReLUActivation<__half,float>;
extern template class CAIF_DeviceReLUActivation<__half,__half>;
extern template class CAIF_DeviceReLUActivation<__half,__nv_bfloat16>;
extern template class CAIF_DeviceReLUActivation<__nv_bfloat16,float>;
extern template class CAIF_DeviceReLUActivation<__nv_bfloat16,__half>;
extern template class CAIF_DeviceReLUActivation<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceReLUActivation<float,float>;
#endif

}//end instance namespace
