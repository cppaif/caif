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

/**
 * @file aif_cpu_tensor_data.h
 * @brief CPU-based tensor data implementation shared by CPU backends
 */

#ifndef CAIF_CPU_TENSOR_DATA_H
#define CAIF_CPU_TENSOR_DATA_H

#include "caif_tensor_data.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include <memory>
#include <vector>

namespace instance
{

class CAIF_CPUTensorData:public CAIF_TensorData
{
  public:
    CAIF_CPUTensorData(
                       const std::vector<uint32_t> &shape,
                       const CAIF_DataType &dtype
                      );
    virtual ~CAIF_CPUTensorData();

    virtual const void *RawData()const override;
    virtual void *MutableRawData()override;
    virtual size_t SizeBytes()const override;
    virtual const std::vector<uint32_t> &Shape()const override;
    virtual CAIF_DataType Type()const override;
    virtual void Fill(double value)override;

    size_t TotalElements()const{return _total_elements;}

  protected:

  private:
    std::vector<uint32_t> _shape;
    CAIF_DataType _dtype;
    std::unique_ptr<float[]> _data;
    size_t _size_bytes;
    size_t _total_elements;
};

}//end instance namespace

#endif  // CAIF_CPU_TENSOR_DATA_H


