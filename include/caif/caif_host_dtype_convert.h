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
// CAIF - Host-side fp32 -> half / bfloat16 conversion utilities.
//
// Layer constructors that randomly initialize weights produce host-side fp32
// samples and then need to land them in a target-dtype device tensor. The old
// pattern allocated a same-size fp32 staging tensor on the device, uploaded
// the samples, and ran a device-side conversion kernel via
// `CAIF_DeviceTensor::To(target_dtype)`. That doubled the transient device
// memory at the moment of init and tripped OOM on large MoE models with many
// experts (1408 x 2048 weights -> 11.5 MB of fp32 staging per weight x
// 3 weights x 64 experts x 27 layers -> wall-clock peak collides with
// permanent footprint at ~32 GB).
//
// This class replaces that pattern with host-side conversion. The fp32
// host buffer is converted in-place into a target-dtype host buffer, then
// uploaded once into the already-allocated target-dtype device tensor.
// Zero device-side staging.
//
// Call sites use these via `CAIF_DeviceTensor::CopyFromHostFp32(...)`
// rather than going through this class directly.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <cstring>

namespace instance
{

class CAIF_HostDtypeConvert
{
  public:
    CAIF_HostDtypeConvert()=delete;
    ~CAIF_HostDtypeConvert()=delete;

    // fp32 -> bf16 with round-to-nearest-even. bf16 stores the upper 16 bits
    // of fp32 (same exponent layout, 7-bit mantissa). The
    // `+ ((bits>>16) & 1)` rounding bias is the standard banker's-rounding
    // trick: when the truncated bit is exactly halfway, round to whichever
    // side keeps the LSB even.
    static uint16_t Fp32ToBf16(const float f)
    {
      uint32_t bits;
      std::memcpy(&bits,&f,sizeof(bits));
      // NaN: preserve mantissa-non-zero so bf16 stays NaN
      if(((bits&0x7f800000)==0x7f800000)&&((bits&0x007fffff)!=0))
      {
        return static_cast<uint16_t>((bits>>16)|0x40);
      }
      const uint32_t rounding_bias=0x7fff+((bits>>16)&1u);
      return static_cast<uint16_t>((bits+rounding_bias)>>16);
    }

    // fp32 -> fp16 (IEEE 754 half) with round-to-nearest-even. Handles
    // inf / NaN, denormalized fp16, and overflow-to-inf. Mirrors the
    // algorithm CUDA's `__float2half_rn` uses on host so caif's host-side
    // init matches what device kernels would have produced.
    static uint16_t Fp32ToFp16(const float f)
    {
      uint32_t bits;
      std::memcpy(&bits,&f,sizeof(bits));
      const uint32_t sign=(bits>>16)&0x8000u;
      int32_t exp=static_cast<int32_t>((bits>>23)&0xffu)-127+15;
      uint32_t mant=bits&0x007fffffu;
      if(exp>=31)
      {
        // inf or NaN
        if(((bits&0x7f800000)==0x7f800000)&&(mant!=0))
        {
          return static_cast<uint16_t>(sign|0x7c00u|(mant>>13)|((mant>>13)==0?1u:0u));
        }
        return static_cast<uint16_t>(sign|0x7c00u);
      }
      if(exp<=0)
      {
        if(exp<-10)
        {
          return static_cast<uint16_t>(sign);
        }
        // Denormalized: shift mantissa with implicit 1 reattached
        mant|=0x00800000u;
        const int32_t shift=14-exp;
        const uint32_t shifted=mant>>shift;
        const uint32_t rem=mant&((1u<<shift)-1u);
        const uint32_t halfway=1u<<(shift-1);
        uint32_t rounded=shifted;
        if(rem>halfway||(rem==halfway&&(shifted&1u)))
        {
          rounded+=1u;
        }
        return static_cast<uint16_t>(sign|rounded);
      }
      // Normalized: round-to-nearest-even on bits 12..0
      const uint32_t rounded=mant+0x00001000u;
      if(rounded&0x00800000u)
      {
        // mantissa carry: bump exponent
        if(exp+1>=31)
        {
          return static_cast<uint16_t>(sign|0x7c00u);
        }
        return static_cast<uint16_t>(sign|(static_cast<uint32_t>(exp+1)<<10));
      }
      return static_cast<uint16_t>(sign|(static_cast<uint32_t>(exp)<<10)|(rounded>>13));
    }

    // Buffer variants. `dst` must have space for at least `count` 16-bit
    // values; `src` is `count` fp32 values. No allocation, no exceptions.
    static void Fp32ToBf16Buffer(const float *src,uint16_t *dst,const size_t count)
    {
      for(size_t i=0;i<count;++i)
      {
        dst[i]=Fp32ToBf16(src[i]);
      }
    }

    static void Fp32ToFp16Buffer(const float *src,uint16_t *dst,const size_t count)
    {
      for(size_t i=0;i<count;++i)
      {
        dst[i]=Fp32ToFp16(src[i]);
      }
    }

    // bf16 -> fp32 (no rounding — bf16 has fewer mantissa bits but every
    // bf16 representable value has an exact fp32 equivalent: just shift
    // the 16-bit pattern into the upper half of a 32-bit float). Used
    // by the host-side download path when the source tensor is bf16.
    static float Bf16ToFp32(const uint16_t bits)
    {
      const uint32_t expanded=static_cast<uint32_t>(bits)<<16;
      float result;
      std::memcpy(&result,&expanded,sizeof(result));
      return result;
    }

    // fp16 -> fp32 (IEEE 754 expansion). Inverse of Fp32ToFp16; used by
    // the host-side download path when the source tensor is fp16.
    static float Fp16ToFp32(const uint16_t bits)
    {
      const uint32_t sign=(static_cast<uint32_t>(bits)&0x8000u)<<16;
      const uint32_t exp=static_cast<uint32_t>(bits>>10)&0x1fu;
      const uint32_t mant=static_cast<uint32_t>(bits)&0x3ffu;
      uint32_t result;
      if(exp==0)
      {
        if(mant==0)
        {
          result=sign;
        }
        else
        {
          // Denormalized fp16: shift mantissa to find leading 1, then
          // re-bias exponent.
          uint32_t m=mant;
          int32_t e=-14;
          while((m&0x400u)==0)
          {
            m<<=1;
            --e;
          }
          m&=0x3ffu;
          const uint32_t exp_fp32=static_cast<uint32_t>(e+127);
          result=sign|(exp_fp32<<23)|(m<<13);
        }
      }
      else if(exp==0x1f)
      {
        // inf or NaN — preserve mantissa-non-zero
        result=sign|(0xffu<<23)|(mant<<13);
      }
      else
      {
        const uint32_t exp_fp32=exp-15+127;
        result=sign|(exp_fp32<<23)|(mant<<13);
      }
      float f;
      std::memcpy(&f,&result,sizeof(f));
      return f;
    }

    static void Bf16ToFp32Buffer(const uint16_t *src,float *dst,const size_t count)
    {
      for(size_t i=0;i<count;++i)
      {
        dst[i]=Bf16ToFp32(src[i]);
      }
    }

    static void Fp16ToFp32Buffer(const uint16_t *src,float *dst,const size_t count)
    {
      for(size_t i=0;i<count;++i)
      {
        dst[i]=Fp16ToFp32(src[i]);
      }
    }

  protected:

  private:
};

}//end instance namespace
