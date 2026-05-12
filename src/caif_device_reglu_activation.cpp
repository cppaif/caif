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

#include "caif_device_reglu_activation.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
void CAIF_DeviceReGLUActivation<ComputeT,StorageT>::Forward(
                                          const CAIF_DeviceTensor &gate_input,
                                          const CAIF_DeviceTensor &up_input,
                                          CAIF_DeviceTensor &output)const
{
  try
  {
    if(output.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceReGLUActivation: output dtype != StorageT");
    }
    const int n=static_cast<int>(gate_input.TotalElements());
    launch_gated_activation_forward<StorageT>(gate_input.template DevicePtr<StorageT>(),
                                               up_input.template DevicePtr<StorageT>(),
                                               output.template DevicePtr<StorageT>(),
                                               CAIF_GATED_OP_RELU,
                                               n,
                                               output.Stream().Handle());
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceReGLUActivation<ComputeT,StorageT>::Backward(
                                          const CAIF_DeviceTensor &grad_output,
                                          const CAIF_DeviceTensor &cached_gate_input,
                                          const CAIF_DeviceTensor &cached_up_input,
                                          CAIF_DeviceTensor &grad_gate,
                                          CAIF_DeviceTensor &grad_up)const
{
  try
  {
    if(grad_output.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceReGLUActivation: grad_output dtype != StorageT");
    }
    const int n=static_cast<int>(grad_output.TotalElements());
    launch_gated_activation_backward<StorageT>(grad_output.template DevicePtr<StorageT>(),
                                                cached_gate_input.template DevicePtr<StorageT>(),
                                                cached_up_input.template DevicePtr<StorageT>(),
                                                grad_gate.template DevicePtr<StorageT>(),
                                                grad_up.template DevicePtr<StorageT>(),
                                                CAIF_GATED_OP_RELU,
                                                n,
                                                grad_gate.Stream().Handle());
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceActivation>
CAIF_DeviceReGLUActivation<ComputeT,StorageT>::Clone()const
{
  return std::make_unique<CAIF_DeviceReGLUActivation<ComputeT,StorageT>>();
}

template class CAIF_DeviceReGLUActivation<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceReGLUActivation<float,__half>;
template class CAIF_DeviceReGLUActivation<float,__nv_bfloat16>;
template class CAIF_DeviceReGLUActivation<__half,float>;
template class CAIF_DeviceReGLUActivation<__half,__half>;
template class CAIF_DeviceReGLUActivation<__half,__nv_bfloat16>;
template class CAIF_DeviceReGLUActivation<__nv_bfloat16,float>;
template class CAIF_DeviceReGLUActivation<__nv_bfloat16,__half>;
template class CAIF_DeviceReGLUActivation<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
