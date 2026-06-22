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

#include "caif_mla_decode_threshold.h"
#include "caif_constants.h"
#ifdef USE_CAIF_CUDA
#include <cuda_runtime.h>
#endif

namespace instance
{

uint32_t CAIF_MlaDecodeThreshold::For(const uint32_t dim,
                                      const uint32_t qk_nope_head_dim,
                                      const uint32_t v_head_dim)
{
  const uint32_t per_token_dim=qk_nope_head_dim+v_head_dim;
  if(per_token_dim==0u)
  {
    return 0u;
  }
#ifdef USE_CAIF_CUDA
  int device=0;
  cudaGetDevice(&device);
  int sm_count=0;
  int clock_khz=0;
  int mem_clock_khz=0;
  int mem_bus_bits=0;
  cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,device);
  cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,device);
  cudaDeviceGetAttribute(&mem_clock_khz,cudaDevAttrMemoryClockRate,device);
  cudaDeviceGetAttribute(&mem_bus_bits,cudaDevAttrGlobalMemoryBusWidth,device);
  const double bandwidth=static_cast<double>(g_caif_memory_transfers_per_clock)*
                         static_cast<double>(mem_clock_khz)*g_caif_khz_to_hz*
                         static_cast<double>(mem_bus_bits)/g_caif_bits_per_byte;
  const double peak_compute=static_cast<double>(sm_count)*
                            static_cast<double>(clock_khz)*g_caif_khz_to_hz*
                            g_caif_cuda_fp32_cores_per_sm*g_caif_flops_per_fma;
  if(bandwidth>0.0 && peak_compute>0.0)
  {
    const double ratio=static_cast<double>(sizeof(float))*
                       peak_compute*g_caif_mla_decode_gemm_efficiency/
                       bandwidth;
    return static_cast<uint32_t>(ratio*static_cast<double>(dim)/per_token_dim);
  }
#endif
  return (g_caif_mla_decode_fallback_ratio*dim)/per_token_dim;
}

}//end instance namespace
