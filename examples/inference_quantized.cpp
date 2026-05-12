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

//--------------------------------------------------------------------------
// Example: inference with quantized frozen weights
//
// Demonstrates the public quantization workflow for inference:
//
//   1. Take a fp32 weight matrix W.
//   2. Quantize it once: CAIF_Ops::QuantizeInt8 (PerTensor) or
//      CAIF_Ops::QuantizeInt4PerGroup (per-group scales).
//   3. Hold the quantized weights in a CAIF_DeviceFrozenLinear
//      instantiated at the matching storage dtype:
//         - <float, int8_t>                — 4x smaller than fp32
//         - <float, caif_int4_packed_t>    — 8x smaller than fp32
//   4. LoadFromTensor(quantized_weight) + LoadScalesFromHost(scales).
//      The layer dequantises on-the-fly inside each forward pass.
//
// This example builds three layers with the SAME underlying fp32
// weight (fp32 reference / int8 PerTensor / int4 per-group), runs
// the same synthetic input through each, and prints the max-abs and
// mean-abs error of int8 / int4 outputs vs the fp32 reference. As
// expected, int4 has larger error than int8, both small relative to
// the output magnitude. No real model or external file required —
// just a quick numeric sanity check that the dequant kernel produces
// outputs close to the full-precision baseline.
//--------------------------------------------------------------------------

#include "caif_device_frozen_linear.h"
#include "caif_device_context.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_run_context_scope.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include "caif_int4_packed_t.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <memory>

using namespace instance;

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF quantized inference example ==="<<std::endl;

    typedef CAIF_DeviceFrozenLinear<float,float> LinearFP32_t;
    typedef CAIF_DeviceFrozenLinear<float,int8_t> LinearInt8_t;
    typedef CAIF_DeviceFrozenLinear<float,caif_int4_packed_t> LinearInt4_t;

    CAIF_DeviceContext::Instance().Initialize();
    CAIF_CudaStream stream;

    const uint32_t input_dim=64;
    const uint32_t output_dim=128;
    const uint32_t batch_size=8;
    const uint32_t group_size=128;
    const uint32_t total_weight_elements=input_dim*output_dim;
    const uint32_t num_int4_groups=(total_weight_elements+group_size-1)/group_size;

    ISE_Out::Out()<<"Weight shape: ["<<input_dim<<", "<<output_dim<<"] "
                  <<"(="<<total_weight_elements<<" fp32 elements, "
                  <<(total_weight_elements*sizeof(float))<<" bytes fp32)"
                  <<std::endl;

    // Synthesise a fp32 weight matrix with a structured pattern so
    // every run is deterministic and the quant error has something
    // non-trivial to round.
    const float weight_scale=0.05f;
    std::vector<float> w_host(total_weight_elements);
    for(uint32_t i=0;i<total_weight_elements;++i)
    {
      const float v=static_cast<float>((i*11+7)%41)/41.0f-0.5f;
      w_host[i]=weight_scale*v;
    }
    CAIF_DeviceTensor w_fp32_src=CAIF_DeviceTensor::FromHostData(w_host.data(),
                                                                 {input_dim,output_dim},
                                                                 stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    CAIF_RunContextPassScope forward_scope(ctx,CAIF_RunContext::Pass_e::Forward_e);

    // ---- fp32 reference --------------------------------------------------
    LinearFP32_t lin_fp32(input_dim,output_dim,stream);
    {
      CAIF_DeviceTensor w_copy=w_fp32_src.Clone();
      lin_fp32.LoadFromTensor(std::move(w_copy));
    }

    // ---- int8 PerTensor --------------------------------------------------
    CAIF_DeviceTensor w_int8=CAIF_DeviceTensor::Uninitialized({input_dim,output_dim},
                                                              stream,
                                                              CAIF_DataType::CAIF_DataType_e::Int8);
    CAIF_DeviceTensor scales_int8=CAIF_DeviceTensor::Uninitialized({1},
                                                                   stream,
                                                                   CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Ops::QuantizeInt8(w_fp32_src,
                           w_int8,
                           scales_int8,
                           CAIF_Ops::QuantScheme_e::PerTensor_e,
                           ctx);
    stream.Synchronize();

    LinearInt8_t lin_int8(input_dim,
                          output_dim,
                          stream,
                          group_size,
                          true,
                          CAIF_Ops::QuantScheme_e::PerTensor_e);
    lin_int8.LoadFromTensor(std::move(w_int8));

    float scales_int8_host=0.0f;
    scales_int8.CopyToHost(&scales_int8_host);
    stream.Synchronize();
    lin_int8.LoadScalesFromHost(&scales_int8_host,sizeof(float));

    ISE_Out::Out()<<"int8 PerTensor scale: "<<scales_int8_host<<std::endl;

    // ---- int4 per-group --------------------------------------------------
    CAIF_DeviceTensor w_int4=CAIF_DeviceTensor::Uninitialized({input_dim,output_dim},
                                                              stream,
                                                              CAIF_DataType::CAIF_DataType_e::Int4);
    CAIF_DeviceTensor scales_int4=CAIF_DeviceTensor::Uninitialized({num_int4_groups},
                                                                   stream,
                                                                   CAIF_DataType::CAIF_DataType_e::Float16);
    CAIF_Ops::QuantizeInt4PerGroup(w_fp32_src,
                                   w_int4,
                                   scales_int4,
                                   group_size,
                                   ctx);
    stream.Synchronize();

    LinearInt4_t lin_int4(input_dim,output_dim,stream,group_size);
    lin_int4.LoadFromTensor(std::move(w_int4));

    // fp16 scales are 2 bytes each; download as raw bytes so this
    // example doesn't need <cuda_fp16.h> for the __half type.
    std::vector<uint8_t> scales_int4_host(num_int4_groups*sizeof(uint16_t));
    scales_int4.CopyToHostRaw(scales_int4_host.data());
    stream.Synchronize();
    lin_int4.LoadScalesFromHost(scales_int4_host.data(),scales_int4_host.size());

    ISE_Out::Out()<<"int4 per-group: "<<num_int4_groups<<" groups of "<<group_size
                  <<" elements each"<<std::endl;

    // ---- Forward pass through all three with the same input -------------
    const uint32_t total_input_elements=batch_size*input_dim;
    std::vector<float> x_host(total_input_elements);
    for(uint32_t i=0;i<total_input_elements;++i)
    {
      x_host[i]=static_cast<float>((i*13+5)%19)/19.0f-0.5f;
    }
    CAIF_DeviceTensor x=CAIF_DeviceTensor::FromHostData(x_host.data(),
                                                        {batch_size,input_dim},
                                                        stream);

    CAIF_DeviceTensor y_fp32=lin_fp32.ForwardImpl(x,ctx);
    CAIF_DeviceTensor y_int8=lin_int8.ForwardImpl(x,ctx);
    CAIF_DeviceTensor y_int4=lin_int4.ForwardImpl(x,ctx);
    stream.Synchronize();

    const uint32_t total_output_elements=batch_size*output_dim;
    std::vector<float> y_fp32_host(total_output_elements);
    std::vector<float> y_int8_host(total_output_elements);
    std::vector<float> y_int4_host(total_output_elements);
    y_fp32.CopyToHost(y_fp32_host.data());
    y_int8.CopyToHost(y_int8_host.data());
    y_int4.CopyToHost(y_int4_host.data());
    stream.Synchronize();

    // ---- Compare error ---------------------------------------------------
    float max_err_int8=0.0f;
    float max_err_int4=0.0f;
    float sum_err_int8=0.0f;
    float sum_err_int4=0.0f;
    float max_abs_fp32=0.0f;
    for(uint32_t i=0;i<total_output_elements;++i)
    {
      const float y_ref=y_fp32_host[i];
      const float e8=std::fabs(y_ref-y_int8_host[i]);
      const float e4=std::fabs(y_ref-y_int4_host[i]);
      if(e8>max_err_int8)
      {
        max_err_int8=e8;
      }
      if(e4>max_err_int4)
      {
        max_err_int4=e4;
      }
      sum_err_int8+=e8;
      sum_err_int4+=e4;
      const float abs_ref=std::fabs(y_ref);
      if(abs_ref>max_abs_fp32)
      {
        max_abs_fp32=abs_ref;
      }
    }
    const float mean_err_int8=sum_err_int8/static_cast<float>(total_output_elements);
    const float mean_err_int4=sum_err_int4/static_cast<float>(total_output_elements);

    ISE_Out::Out()<<"Output shape per layer: ["<<batch_size<<", "<<output_dim<<"]"<<std::endl;
    ISE_Out::Out()<<"fp32 reference |output|_max = "<<max_abs_fp32<<std::endl;
    ISE_Out::Out()<<"int8 vs fp32  : max_abs_err="<<max_err_int8
                  <<"  mean_abs_err="<<mean_err_int8
                  <<std::endl;
    ISE_Out::Out()<<"int4 vs fp32  : max_abs_err="<<max_err_int4
                  <<"  mean_abs_err="<<mean_err_int4
                  <<std::endl;

    ISE_Out::Out()<<"=== Done ==="<<std::endl;
    return 0;
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception: "<<e<<std::endl;
    return 1;
  }
}
