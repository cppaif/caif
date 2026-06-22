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

#include "caif_device_bilinear_activation.h"
#include "caif_cuda_kernels_activations.cuh"
#include "caif_exception.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
void CAIF_DeviceBilinearActivation<ComputeT,StorageT>::Forward(
                                          const CAIF_DeviceTensor &gate_input,
                                          const CAIF_DeviceTensor &up_input,
                                          CAIF_DeviceTensor &output)const
{
  try
  {
    if(output.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceBilinearActivation: output dtype != StorageT");
    }
    const int64_t n=static_cast<int64_t>(gate_input.TotalElements());
    launch_gated_activation_forward<StorageT>(gate_input.template DevicePtr<StorageT>(),
                                               up_input.template DevicePtr<StorageT>(),
                                               output.template DevicePtr<StorageT>(),
                                               CAIF_GATED_OP_LINEAR,
                                               n,
                                               output.Stream().Handle());
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceBilinearActivation<ComputeT,StorageT>::Backward(
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
      THROW_CAIFE("CAIF_DeviceBilinearActivation: grad_output dtype != StorageT");
    }
    const int64_t n=static_cast<int64_t>(grad_output.TotalElements());
    launch_gated_activation_backward<StorageT>(grad_output.template DevicePtr<StorageT>(),
                                                cached_gate_input.template DevicePtr<StorageT>(),
                                                cached_up_input.template DevicePtr<StorageT>(),
                                                grad_gate.template DevicePtr<StorageT>(),
                                                grad_up.template DevicePtr<StorageT>(),
                                                CAIF_GATED_OP_LINEAR,
                                                n,
                                                grad_gate.Stream().Handle());
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceActivation>
CAIF_DeviceBilinearActivation<ComputeT,StorageT>::Clone()const
{
  return std::make_unique<CAIF_DeviceBilinearActivation<ComputeT,StorageT>>();
}

template class CAIF_DeviceBilinearActivation<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceBilinearActivation<float,__half>;
template class CAIF_DeviceBilinearActivation<float,__nv_bfloat16>;
template class CAIF_DeviceBilinearActivation<__half,float>;
template class CAIF_DeviceBilinearActivation<__half,__half>;
template class CAIF_DeviceBilinearActivation<__half,__nv_bfloat16>;
template class CAIF_DeviceBilinearActivation<__nv_bfloat16,float>;
template class CAIF_DeviceBilinearActivation<__nv_bfloat16,__half>;
template class CAIF_DeviceBilinearActivation<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
