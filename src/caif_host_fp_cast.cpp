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

#include "caif_host_fp_cast.h"
#include "caif_exception.h"

#include <cstdint>
#include <cstring>

namespace instance
{

float CAIF_HostFpCast::Fp16ToFloat(const uint16_t h)
{
  try
  {
    const uint32_t sign=static_cast<uint32_t>((h>>15)&0x1);
    const uint32_t exp=static_cast<uint32_t>((h>>10)&0x1F);
    const uint32_t mant=static_cast<uint32_t>(h&0x3FF);
    uint32_t bits=0;
    if(exp==0)
    {
      if(mant==0)
      {
        bits=sign<<31;
      }
      else
      {
        uint32_t m=mant;
        uint32_t e=1;
        while((m&0x400)==0)
        {
          m<<=1;
          e++;
        }
        m&=0x3FF;
        bits=(sign<<31)|((127-15-e+1)<<23)|(m<<13);
      }
    }
    else
    {
      if(exp==31)
      {
        bits=(sign<<31)|(0xFF<<23)|(mant<<13);
      }
      else
      {
        bits=(sign<<31)|((exp+(127-15))<<23)|(mant<<13);
      }
    }
    float out;
    std::memcpy(&out,&bits,sizeof(out));
    return out;
  }
  CAIF_CATCH_BLOCK();
}

uint16_t CAIF_HostFpCast::FloatToFp16(const float f)
{
  try
  {
    uint32_t bits;
    std::memcpy(&bits,&f,sizeof(bits));
    const uint32_t sign=(bits>>31)&0x1;
    const int32_t exp=static_cast<int32_t>((bits>>23)&0xFF);
    const uint32_t mant=bits&0x7FFFFF;
    uint16_t out=0;
    if(exp==0xFF)
    {
      const uint32_t half_mant=mant>>13;
      out=static_cast<uint16_t>((sign<<15)|(0x1F<<10)|half_mant);
      return out;
    }
    const int32_t unbiased=exp-127;
    if(unbiased>15)
    {
      out=static_cast<uint16_t>((sign<<15)|(0x1F<<10));
      return out;
    }
    if(unbiased<-14)
    {
      if(unbiased<-24)
      {
        out=static_cast<uint16_t>(sign<<15);
        return out;
      }
      const uint32_t m=(mant|0x800000)>>(-unbiased-14+13);
      out=static_cast<uint16_t>((sign<<15)|m);
      return out;
    }
    out=static_cast<uint16_t>((sign<<15)|((unbiased+15)<<10)|(mant>>13));
    return out;
  }
  CAIF_CATCH_BLOCK();
}

float CAIF_HostFpCast::Bf16ToFloat(const uint16_t b)
{
  try
  {
    uint32_t bits=static_cast<uint32_t>(b)<<16;
    float out;
    std::memcpy(&out,&bits,sizeof(out));
    return out;
  }
  CAIF_CATCH_BLOCK();
}

uint16_t CAIF_HostFpCast::FloatToBf16(const float f)
{
  try
  {
    uint32_t bits;
    std::memcpy(&bits,&f,sizeof(bits));
    const uint32_t rounded=bits+0x7FFF+((bits>>16)&0x1);
    return static_cast<uint16_t>(rounded>>16);
  }
  CAIF_CATCH_BLOCK();
}

std::vector<float> CAIF_HostFpCast::UpcastToFloat(const CAIF_DeviceTensor &t)
{
  try
  {
    const size_t n=t.TotalElements();
    std::vector<float> out(n);
    const CAIF_DataType::CAIF_DataType_e dt=t.Dtype();
    if(dt==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      std::memcpy(out.data(),t.DeviceDataRaw(),n*sizeof(float));
      return out;
    }
    if(dt==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      const uint16_t *p=static_cast<const uint16_t*>(t.DeviceDataRaw());
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        out[i]=Fp16ToFloat(p[i]);
      }
      return out;
    }
    if(dt==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      const uint16_t *p=static_cast<const uint16_t*>(t.DeviceDataRaw());
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        out[i]=Bf16ToFloat(p[i]);
      }
      return out;
    }
    THROW_CAIFE("CAIF_HostFpCast::UpcastToFloat: unsupported dtype");
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_HostFpCast::DowncastFromFloat(const std::vector<float> &src,
                                        CAIF_DeviceTensor &out)
{
  try
  {
    const size_t n=out.TotalElements();
    if(src.size()!=n)
    {
      THROW_CAIFE("CAIF_HostFpCast::DowncastFromFloat: size mismatch");
    }
    const CAIF_DataType::CAIF_DataType_e dt=out.Dtype();
    if(dt==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      std::memcpy(out.DeviceDataRaw(),src.data(),n*sizeof(float));
      return;
    }
    if(dt==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      uint16_t *p=static_cast<uint16_t*>(out.DeviceDataRaw());
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        p[i]=FloatToFp16(src[i]);
      }
      return;
    }
    if(dt==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      uint16_t *p=static_cast<uint16_t*>(out.DeviceDataRaw());
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        p[i]=FloatToBf16(src[i]);
      }
      return;
    }
    THROW_CAIFE("CAIF_HostFpCast::DowncastFromFloat: unsupported dtype");
  }
  CAIF_CATCH_BLOCK();
}

float *CAIF_HostFpCast::HostFp32(CAIF_DeviceTensor &t,const std::string &op)
{
  try
  {
    if(t.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      (void)op;
      THROW_CAIFE("CAIF_HostFpCast::HostFp32: op requires Float32 dtype");
    }
    // fp32 by gate above
    return static_cast<float*>(t.DeviceDataRaw());
  }
  CAIF_CATCH_BLOCK();
}

const float *CAIF_HostFpCast::HostFp32(const CAIF_DeviceTensor &t,
                                       const std::string &op)
{
  try
  {
    if(t.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      (void)op;
      THROW_CAIFE("CAIF_HostFpCast::HostFp32: op requires Float32 dtype");
    }
    // fp32 by gate above
    return static_cast<const float*>(t.DeviceDataRaw());
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
