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

// Per-site `tensor.DevicePtr<float>()` reads in this file are all on
// `fp32_view`, which is built one line above each call via
// `.To(Float32)` / `.Clone()`. Comments at each call site name this
// explicitly.

#include "caif_sys_debug.h"
#include <vector>
#include <cstring>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#endif

namespace instance
{

bool CAIF_SysDebug::_enabled=false;

void CAIF_SysDebug::SetEnabled(bool enabled)
{
  _enabled=enabled;
}

bool CAIF_SysDebug::IsEnabled()
{
  return _enabled;
}

bool CAIF_SysDebug::CheckTensor(const std::string &label,
                               const CAIF_DeviceTensor &tensor,
                               bool print_summary)
{
  if(IsEnabled()==false)
  {
    return false;
  }

#ifdef USE_CAIF_CUDA
  cudaStreamSynchronize(tensor.Stream().Handle());

  // Cast non-fp32 storage to fp32 first so the NaN/inf scan reads typed
  // float bits. The debug helper isn't perf-critical; the extra .To()
  // round-trip is negligible vs the host-side scan.
  CAIF_DeviceTensor fp32_view;
  if(tensor.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    fp32_view=tensor.To(CAIF_DataType::CAIF_DataType_e::Float32);
  }
  else
  {
    fp32_view=tensor.Clone();
  }

  const size_t total_elements=fp32_view.TotalElements();
  std::vector<float> host_data(total_elements);
  // fp32: fp32_view was just cast/cloned to Float32 above.
  cudaMemcpy(host_data.data(),
             fp32_view.DevicePtr<float>(),
             total_elements*sizeof(float),
             cudaMemcpyDeviceToHost);

  uint32_t nan_count=0;
  uint32_t inf_count=0;
  float min_val=host_data[0];
  float max_val=host_data[0];

  for(size_t i=0;i<total_elements;++i)
  {
    const float val=host_data[i];
    // Note: Using bitwise check since -ffast-math breaks std::isnan
    uint32_t bits=0;
    std::memcpy(&bits,&val,sizeof(float));
    constexpr uint32_t exp_mask=0x7F800000;
    constexpr uint32_t mantissa_mask=0x007FFFFF;
    if((bits&exp_mask)==exp_mask)
    {
      if((bits&mantissa_mask)!=0)
      {
        ++nan_count;
      }
      else
      {
        ++inf_count;
      }
    }
    else
    {
      if(val<min_val)
      {
        min_val=val;
      }
      if(val>max_val)
      {
        max_val=val;
      }
    }
  }

  const bool has_bad_values=(nan_count>0||inf_count>0);

  if(print_summary==true)
  {
    SDbgLog()<<"[CAIF_SysDebug] "
             <<label
             <<": elements="
             <<total_elements
             <<" nan="
             <<nan_count
             <<" inf="
             <<inf_count
             <<" range=["
             <<min_val
             <<","
             <<max_val
             <<"]"
             <<std::endl;
  }

  return has_bad_values;
#else
  (void)label;
  (void)tensor;
  (void)print_summary;
  return false;
#endif
}

void CAIF_SysDebug::PrintRawValues(const std::string &label,
                                  const CAIF_DeviceTensor &tensor,
                                  uint32_t count)
{
  if(IsEnabled()==false)
  {
    return;
  }

#ifdef USE_CAIF_CUDA
  cudaStreamSynchronize(tensor.Stream().Handle());

  // Cast non-fp32 storage to fp32 first so the print reads typed float
  // bits. Debug helper, perf irrelevant.
  CAIF_DeviceTensor fp32_view;
  if(tensor.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    fp32_view=tensor.To(CAIF_DataType::CAIF_DataType_e::Float32);
  }
  else
  {
    fp32_view=tensor.Clone();
  }

  const size_t total_elements=fp32_view.TotalElements();
  uint32_t print_count=count;
  if(count>total_elements)
  {
    print_count=static_cast<uint32_t>(total_elements);
  }

  std::vector<float> host_data(print_count);
  // fp32: fp32_view was just cast/cloned to Float32 above.
  cudaMemcpy(host_data.data(),
             fp32_view.DevicePtr<float>(),
             print_count*sizeof(float),
             cudaMemcpyDeviceToHost);

  SDbgLog()<<"[CAIF_SysDebug] "
           <<label
           <<" (first "
           <<print_count
           <<"): ";
  for(uint32_t i=0;i<print_count;++i)
  {
    SDbgLog()<<host_data[i]<<" ";
  }
  SDbgLog()<<std::endl;
#else
  (void)label;
  (void)tensor;
  (void)count;
#endif
}

void CAIF_SysDebug::DebugLog(const std::string &message)
{
  if(IsEnabled()==true)
  {
    SDbgLog()<<"[CAIF_SysDebug] "<<message<<std::endl;
  }
}

}//end instance namespace
