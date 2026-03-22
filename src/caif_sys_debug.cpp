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

#include "caif_sys_debug.h"
#include <vector>
#include <cstring>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#endif

using namespace instance;

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
  if(_enabled==false)
  {
    return false;
  }

#ifdef USE_CAIF_CUDA
  cudaStreamSynchronize(tensor.Stream().Handle());

  const size_t total_elements=tensor.TotalElements();
  std::vector<float> host_data(total_elements);
  cudaMemcpy(host_data.data(),
             tensor.DevicePtr(),
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
  if(_enabled==false)
  {
    return;
  }

#ifdef USE_CAIF_CUDA
  cudaStreamSynchronize(tensor.Stream().Handle());

  const size_t total_elements=tensor.TotalElements();
  uint32_t print_count=count;
  if(count>total_elements)
  {
    print_count=static_cast<uint32_t>(total_elements);
  }

  std::vector<float> host_data(print_count);
  cudaMemcpy(host_data.data(),
             tensor.DevicePtr(),
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
  if(_enabled==true)
  {
    SDbgLog()<<"[CAIF_SysDebug] "<<message<<std::endl;
  }
}
