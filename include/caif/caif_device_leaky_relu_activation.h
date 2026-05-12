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
// CAIF_DeviceLeakyReLUActivation — pointwise LeakyReLU: f(x) = max(alpha*x, x).
// Templated on <ComputeT, StorageT>; the alpha hyperparameter defaults to
// g_caif_default_leaky_relu_alpha.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_pointwise_activation.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#include "caif_constants.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceLeakyReLUActivation:public CAIF_DevicePointwiseActivation
{
  public:
    CAIF_DeviceLeakyReLUActivation()=default;
    explicit CAIF_DeviceLeakyReLUActivation(const float alpha):_alpha(alpha){}

    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override;

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override;

    std::string Description()const override;
    std::unique_ptr<CAIF_DeviceActivation> Clone()const override;

    float Alpha()const{return _alpha;}

  protected:

  private:
    float _alpha=g_caif_default_leaky_relu_alpha;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceLeakyReLUActivation<float,float>;
extern template class CAIF_DeviceLeakyReLUActivation<float,__half>;
extern template class CAIF_DeviceLeakyReLUActivation<float,__nv_bfloat16>;
extern template class CAIF_DeviceLeakyReLUActivation<__half,float>;
extern template class CAIF_DeviceLeakyReLUActivation<__half,__half>;
extern template class CAIF_DeviceLeakyReLUActivation<__half,__nv_bfloat16>;
extern template class CAIF_DeviceLeakyReLUActivation<__nv_bfloat16,float>;
extern template class CAIF_DeviceLeakyReLUActivation<__nv_bfloat16,__half>;
extern template class CAIF_DeviceLeakyReLUActivation<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceLeakyReLUActivation<float,float>;
#endif

}//end instance namespace
