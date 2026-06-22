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

#include "caif_device_swish_activation.h"
#include "caif_cuda_kernels_activations.cuh"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
void CAIF_DeviceSwishActivation<ComputeT,StorageT>::Forward(const CAIF_DeviceTensor &input,
                                                             CAIF_DeviceTensor &output)const
{
  try
  {
    if(output.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceSwishActivation: output dtype != StorageT");
    }
    const int64_t n=static_cast<int64_t>(input.TotalElements());
    launch_swish_forward<StorageT>(input.template DevicePtr<StorageT>(),
                                    output.template DevicePtr<StorageT>(),
                                    n,
                                    output.Stream().Handle());
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceSwishActivation<ComputeT,StorageT>::Backward(const CAIF_DeviceTensor &grad_output,
                                                              const CAIF_DeviceTensor &pre_activation,
                                                              const CAIF_DeviceTensor &post_activation,
                                                              CAIF_DeviceTensor &grad_input)const
{
  try
  {
    if(grad_output.Dtype()!=CAIF_StorageDtype_t<StorageT>::Value)
    {
      THROW_CAIFE("CAIF_DeviceSwishActivation: grad_output dtype != StorageT");
    }
    const int64_t n=static_cast<int64_t>(grad_output.TotalElements());
    launch_swish_backward<StorageT>(grad_output.template DevicePtr<StorageT>(),
                                     pre_activation.template DevicePtr<StorageT>(),
                                     post_activation.template DevicePtr<StorageT>(),
                                     grad_input.template DevicePtr<StorageT>(),
                                     n,
                                     grad_input.Stream().Handle());
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceSwishActivation<ComputeT,StorageT>::Description()const
{
  return g_serial_tag_swish;
}

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceActivation>
CAIF_DeviceSwishActivation<ComputeT,StorageT>::Clone()const
{
  return std::make_unique<CAIF_DeviceSwishActivation<ComputeT,StorageT>>();
}

template class CAIF_DeviceSwishActivation<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceSwishActivation<float,__half>;
template class CAIF_DeviceSwishActivation<float,__nv_bfloat16>;
template class CAIF_DeviceSwishActivation<__half,float>;
template class CAIF_DeviceSwishActivation<__half,__half>;
template class CAIF_DeviceSwishActivation<__half,__nv_bfloat16>;
template class CAIF_DeviceSwishActivation<__nv_bfloat16,float>;
template class CAIF_DeviceSwishActivation<__nv_bfloat16,__half>;
template class CAIF_DeviceSwishActivation<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
