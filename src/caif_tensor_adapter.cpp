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

#include "caif_tensor_adapter.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <cstring>
#include <sstream>

namespace instance
{

CAIF_TensorAdapter::CAIF_TensorAdapter(const std::vector<uint32_t> &shape,CAIF_CudaStream &stream):
                                     _device_tensor(CAIF_DeviceTensor::Zeros(shape,stream)),
                                     _host_cache(nullptr),
                                     _host_cache_valid(false),
                                     _host_modified(false),
                                     _slow_path_warned(false)
{
}

CAIF_TensorAdapter::CAIF_TensorAdapter(CAIF_DeviceTensor &&device_tensor):
                                     _device_tensor(std::move(device_tensor)),
                                     _host_cache(nullptr),
                                     _host_cache_valid(false),
                                     _host_modified(false),
                                     _slow_path_warned(false)
{
}

CAIF_TensorAdapter::CAIF_TensorAdapter(CAIF_TensorAdapter &&other)noexcept:
                                     _device_tensor(std::move(other._device_tensor)),
                                     _host_cache(std::move(other._host_cache)),
                                     _host_cache_valid(other._host_cache_valid),
                                     _host_modified(other._host_modified),
                                     _slow_path_warned(other._slow_path_warned)
{
  other._host_cache_valid=false;
  other._host_modified=false;
}

CAIF_TensorAdapter &CAIF_TensorAdapter::operator=(CAIF_TensorAdapter &&other)noexcept
{
  if(this!=&other)
  {
    _device_tensor=std::move(other._device_tensor);
    _host_cache=std::move(other._host_cache);
    _host_cache_valid=other._host_cache_valid;
    _host_modified=other._host_modified;
    _slow_path_warned=other._slow_path_warned;
    other._host_cache_valid=false;
    other._host_modified=false;
  }
  return *this;
}

void CAIF_TensorAdapter::WarnSlowPath()const
{
  if(_slow_path_warned==false)
  {
    _slow_path_warned=true;
    std::ostringstream msg;
    msg<<"[PERF WARNING] CAIF_TensorAdapter: Host data access triggers device sync. "
       <<"Consider migrating to CAIF_DeviceTensor API for better performance. "
       <<"Shape: [";
    const auto &shape=_device_tensor.Shape();
    for(size_t i=0;i<shape.size();++i)
    {
      if(i>0)
      {
        msg<<", ";
      }
      msg<<shape[i];
    }
    msg<<"]";
    ErrorLog()<<msg.str()<<"\n";
  }
}

void CAIF_TensorAdapter::EnsureHostCache()const
{
  // Allocate host cache if not present
  if(_host_cache==nullptr)
  {
    const auto &shape=_device_tensor.Shape();
    _host_cache=std::make_unique<CAIF_HostTensor>(
      CAIF_HostTensor::Uninitialized(std::vector<uint32_t>(shape.begin(),shape.end()))
    );
    _host_cache_valid=false;
  }

  // Sync from device if cache is invalid and no pending host modifications
  if(_host_cache_valid==false&&_host_modified==false)
  {
    WarnSlowPath();
    _device_tensor.CopyToHost(_host_cache->Data());
    _host_cache_valid=true;
  }
}

const void *CAIF_TensorAdapter::RawData()const
{
  if(_device_tensor.IsEmpty()==true)
  {
    return nullptr;
  }

  EnsureHostCache();
  return _host_cache->Data();
}

void *CAIF_TensorAdapter::MutableRawData()
{
  if(_device_tensor.IsEmpty()==true)
  {
    return nullptr;
  }

  EnsureHostCache();
  // Note: We don't automatically mark as modified here because
  // the caller might just be reading. They must call MarkHostModified()
  // after actual modifications.
  return _host_cache->Data();
}

void CAIF_TensorAdapter::MarkHostModified()
{
  if(_host_cache==nullptr)
  {
    THROW_CAIFE("TensorAdapter: Cannot mark host modified without host cache");
  }
  _host_modified=true;
  _host_cache_valid=true;  // Host cache is valid (it's the source of truth now)
}

void CAIF_TensorAdapter::SyncToDevice()
{
  if(_host_modified==true&&_host_cache!=nullptr)
  {
    _device_tensor.CopyFromHost(_host_cache->Data(),_device_tensor.TotalElements());
    _host_modified=false;
    // Host cache remains valid since it matches device
  }
}

void CAIF_TensorAdapter::SyncFromDevice()const
{
  if(_device_tensor.IsEmpty()==true)
  {
    return;
  }

  // Force sync even if cache is valid
  if(_host_cache==nullptr)
  {
    const auto &shape=_device_tensor.Shape();
    _host_cache=std::make_unique<CAIF_HostTensor>(
      CAIF_HostTensor::Uninitialized(std::vector<uint32_t>(shape.begin(),shape.end()))
    );
  }

  WarnSlowPath();
  _device_tensor.CopyToHost(_host_cache->Data());
  _host_cache_valid=true;
  _host_modified=false;  // Device is now source of truth
}

CAIF_TensorAdapter CAIF_TensorAdapter::Zeros(const std::vector<uint32_t> &shape,CAIF_CudaStream &stream)
{
  return CAIF_TensorAdapter(shape,stream);
}

CAIF_TensorAdapter CAIF_TensorAdapter::FromHostData(const float *data,
                                                  const std::vector<uint32_t> &shape,
                                                  CAIF_CudaStream &stream)
{
  CAIF_DeviceTensor device=CAIF_DeviceTensor::FromHostData(data,shape,stream);
  return CAIF_TensorAdapter(std::move(device));
}

CAIF_TensorAdapter CAIF_TensorAdapter::FromHost(const CAIF_HostTensor &host,CAIF_CudaStream &stream)
{
  CAIF_DeviceTensor device=CAIF_DeviceTensor::FromHost(host,stream);
  return CAIF_TensorAdapter(std::move(device));
}

}//end instance namespace
