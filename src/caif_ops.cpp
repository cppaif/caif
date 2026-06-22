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
// CAIF - AI Framework
// CAIF_Ops public dispatch.
//
// Every public CAIF_Ops::Foo(...) branches on tensor location and forwards to
// FooDevice(...) in caif_ops_device.cpp or FooHost(...) in caif_ops_host.cpp.
// The host backend is stubbed in Stage 5c and filled in during Stage 5d.
//------------------------------------------------------------------------------
#include "caif_ops.h"
#include "caif_exception.h"

namespace instance
{


//------------------------------------------------------------------------------
// Location guards
//------------------------------------------------------------------------------

void CAIF_Ops::RequireSameLocation(const CAIF_DeviceTensor &a,
                         const CAIF_DeviceTensor &b,
                         const CAIF_DeviceTensor &c,
                         const std::string &op_name)
{
  try
  {
    if(a.Location()!=b.Location() || a.Location()!=c.Location())
    {
      (void)op_name;
      THROW_CAIFE("CAIF_Ops: tensor-location mismatch across inputs/output");
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::RequireSameLocation(const CAIF_DeviceTensor &a,
                         const CAIF_DeviceTensor &b,
                         const std::string &op_name)
{
  try
  {
    if(a.Location()!=b.Location())
    {
      (void)op_name;
      THROW_CAIFE("CAIF_Ops: tensor-location mismatch across inputs/output");
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::RequireSameLocation(const CAIF_DeviceTensor &a,const std::string &op_name)
{
  (void)a;
  (void)op_name;
}

void CAIF_Ops::RequireMatchingDtype(const CAIF_DeviceTensor &a,
                                    const CAIF_DeviceTensor &b,
                                    const CAIF_DeviceTensor &out,
                                    const std::string &op)
{
  try
  {
    if(a.Dtype()!=b.Dtype() || a.Dtype()!=out.Dtype())
    {
      THROW_CAIFE("CAIF_Ops::"+op+": matmul inputs and output must share a dtype");
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Matrix operations
//------------------------------------------------------------------------------

void CAIF_Ops::MatMul(const CAIF_DeviceTensor &a,
            const CAIF_DeviceTensor &b,
            CAIF_DeviceTensor &output,
            CAIF_RunContext &ctx,
            const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
  try
  {
    RequireSameLocation(a,b,output,"MatMul");
    if(IsHost(a)==true)
    {
      MatMulHost(a,b,output,ctx);
    }
    else
    {
      MatMulDevice(a,b,output,ctx,compute_dtype);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MatMulTransposeA(const CAIF_DeviceTensor &a,
                      const CAIF_DeviceTensor &b,
                      CAIF_DeviceTensor &output,
                      CAIF_RunContext &ctx,
                      const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
  try
  {
    RequireSameLocation(a,b,output,"MatMulTransposeA");
    if(IsHost(a)==true)
    {
      MatMulTransposeAHost(a,b,output,ctx);
    }
    else
    {
      MatMulTransposeADevice(a,b,output,ctx,compute_dtype);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MatMulTransposeB(const CAIF_DeviceTensor &a,
                      const CAIF_DeviceTensor &b,
                      CAIF_DeviceTensor &output,
                      CAIF_RunContext &ctx,
                      const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
  try
  {
    RequireSameLocation(a,b,output,"MatMulTransposeB");
    if(IsHost(a)==true)
    {
      MatMulTransposeBHost(a,b,output,ctx);
    }
    else
    {
      MatMulTransposeBDevice(a,b,output,ctx,compute_dtype);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BatchedMatMul(const CAIF_DeviceTensor &a,
                   const CAIF_DeviceTensor &b,
                   CAIF_DeviceTensor &output,
                   int m,
                   int k,
                   int n,
                   int batch_count,
                   CAIF_RunContext &ctx,
                   const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
  try
  {
    RequireSameLocation(a,b,output,"BatchedMatMul");
    if(IsHost(a)==true)
    {
      BatchedMatMulHost(a,b,output,m,k,n,batch_count,ctx);
    }
    else
    {
      BatchedMatMulDevice(a,b,output,m,k,n,batch_count,ctx,compute_dtype);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BatchedMatMulTransposeA(const CAIF_DeviceTensor &a,
                             const CAIF_DeviceTensor &b,
                             CAIF_DeviceTensor &output,
                             int k,
                             int m,
                             int n,
                             int batch_count,
                             CAIF_RunContext &ctx,
                             const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
  try
  {
    RequireSameLocation(a,b,output,"BatchedMatMulTransposeA");
    if(IsHost(a)==true)
    {
      BatchedMatMulTransposeAHost(a,b,output,k,m,n,batch_count,ctx);
    }
    else
    {
      BatchedMatMulTransposeADevice(a,b,output,k,m,n,batch_count,ctx,compute_dtype);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BatchedMatMulTransposeB(const CAIF_DeviceTensor &a,
                             const CAIF_DeviceTensor &b,
                             CAIF_DeviceTensor &output,
                             int m,
                             int k,
                             int n,
                             int batch_count,
                             CAIF_RunContext &ctx,
                             const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
  try
  {
    RequireSameLocation(a,b,output,"BatchedMatMulTransposeB");
    if(IsHost(a)==true)
    {
      BatchedMatMulTransposeBHost(a,b,output,m,k,n,batch_count,ctx);
    }
    else
    {
      BatchedMatMulTransposeBDevice(a,b,output,m,k,n,batch_count,ctx,compute_dtype);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Tensor manipulation
//------------------------------------------------------------------------------

void CAIF_Ops::Transpose(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Transpose");
    if(IsHost(input)==true)
    {
      TransposeHost(input,output);
    }
    else
    {
      TransposeDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Element-wise
//------------------------------------------------------------------------------

void CAIF_Ops::Add(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(a,b,output,"Add");
    if(IsHost(a)==true)
    {
      AddHost(a,b,output);
    }
    else
    {
      AddDevice(a,b,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Scale(CAIF_DeviceTensor &tensor,float scale)
{
  try
  {
    if(IsHost(tensor)==true)
    {
      ScaleHost(tensor,scale);
    }
    else
    {
      ScaleDevice(tensor,scale);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::UnscaleCheckInf(CAIF_DeviceTensor &grad,
                               float inv_scale,
                               CAIF_DeviceTensor &found_inf)
{
  try
  {
    if(IsHost(grad)==true)
    {
      UnscaleCheckInfHost(grad,inv_scale,found_inf);
    }
    else
    {
      UnscaleCheckInfDevice(grad,inv_scale,found_inf);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Scale(const CAIF_DeviceTensor &input,float scale,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Scale");
    if(IsHost(input)==true)
    {
      ScaleHost(input,scale,output);
    }
    else
    {
      ScaleDevice(input,scale,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddScaled(CAIF_DeviceTensor &target,const CAIF_DeviceTensor &source,float scale)
{
  try
  {
    RequireSameLocation(target,source,"AddScaled");
    if(IsHost(target)==true)
    {
      AddScaledHost(target,source,scale);
    }
    else
    {
      AddScaledDevice(target,source,scale);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Bias
//------------------------------------------------------------------------------

void CAIF_Ops::BiasAdd(const CAIF_DeviceTensor &input,
             const CAIF_DeviceTensor &bias,
             CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,bias,output,"BiasAdd");
    if(IsHost(input)==true)
    {
      BiasAddHost(input,bias,output);
    }
    else
    {
      BiasAddDevice(input,bias,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MatMulBias(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                const CAIF_DeviceTensor &bias,
                CAIF_DeviceTensor &output,
                cudaStream_t stream,
                CAIF_RunContext &ctx,
                const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
  try
  {
    RequireSameLocation(a,b,output,"MatMulBias");
    RequireSameLocation(a,bias,"MatMulBias");
    if(IsHost(a)==true)
    {
      MatMulBiasHost(a,b,bias,output,stream,ctx);
    }
    else
    {
      MatMulBiasDevice(a,b,bias,output,stream,ctx,compute_dtype);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BiasGradient(const CAIF_DeviceTensor &grad,CAIF_DeviceTensor &bias_grad)
{
  try
  {
    RequireSameLocation(grad,bias_grad,"BiasGradient");
    if(IsHost(grad)==true)
    {
      BiasGradientHost(grad,bias_grad);
    }
    else
    {
      BiasGradientDevice(grad,bias_grad);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddPositionalEncoding(const CAIF_DeviceTensor &input,
                           const CAIF_DeviceTensor &pe_table,
                           CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,pe_table,output,"AddPositionalEncoding");
    if(IsHost(input)==true)
    {
      AddPositionalEncodingHost(input,pe_table,output);
    }
    else
    {
      AddPositionalEncodingDevice(input,pe_table,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::PositionalEncodingBackward(const CAIF_DeviceTensor &grad_output,
                                CAIF_DeviceTensor &grad_table)
{
  try
  {
    RequireSameLocation(grad_output,grad_table,"PositionalEncodingBackward");
    if(IsHost(grad_output)==true)
    {
      PositionalEncodingBackwardHost(grad_output,grad_table);
    }
    else
    {
      PositionalEncodingBackwardDevice(grad_output,grad_table);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ComputeRelativePositionBias(const CAIF_DeviceTensor &embedding,
                                 CAIF_DeviceTensor &output,
                                 uint32_t max_distance,
                                 bool bidirectional)
{
  try
  {
    RequireSameLocation(embedding,output,"ComputeRelativePositionBias");
    if(IsHost(embedding)==true)
    {
      ComputeRelativePositionBiasHost(embedding,output,max_distance,bidirectional);
    }
    else
    {
      ComputeRelativePositionBiasDevice(embedding,output,max_distance,bidirectional);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AccumulateRelativePositionBiasGradient(const CAIF_DeviceTensor &grad_output,
                                            CAIF_DeviceTensor &grad_embedding,
                                            uint32_t max_distance,
                                            bool bidirectional)
{
  try
  {
    RequireSameLocation(grad_output,grad_embedding,"AccumulateRelativePositionBiasGradient");
    if(IsHost(grad_output)==true)
    {
      AccumulateRelativePositionBiasGradientHost(grad_output,
                                                 grad_embedding,
                                                 max_distance,
                                                 bidirectional);
    }
    else
    {
      AccumulateRelativePositionBiasGradientDevice(grad_output,
                                                   grad_embedding,
                                                   max_distance,
                                                   bidirectional);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Cast(const CAIF_DeviceTensor &input,
          CAIF_DeviceTensor &output,
          CAIF_RunContext &ctx)
{
  try
  {
    RequireSameLocation(input,output,"Cast");
    if(IsHost(input)==true)
    {
      CastHost(input,output,ctx);
    }
    else
    {
      CastDevice(input,output,ctx);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::QuantizeInt8(const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &output,
                  CAIF_DeviceTensor &scales,
                  QuantScheme_e scheme,
                  CAIF_RunContext &ctx)
{
  try
  {
    RequireSameLocation(input,output,"QuantizeInt8");
    RequireSameLocation(input,scales,"QuantizeInt8");
    if(IsHost(input)==true)
    {
      THROW_CAIFE("QuantizeInt8: host backend not supported");
    }
    QuantizeInt8Device(input,output,scales,scheme,ctx);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::DequantizeInt8(const CAIF_DeviceTensor &input,
                    CAIF_DeviceTensor &output,
                    const CAIF_DeviceTensor &scales,
                    QuantScheme_e scheme,
                    CAIF_RunContext &ctx)
{
  try
  {
    RequireSameLocation(input,output,"DequantizeInt8");
    RequireSameLocation(input,scales,"DequantizeInt8");
    if(IsHost(input)==true)
    {
      THROW_CAIFE("DequantizeInt8: host backend not supported");
    }
    DequantizeInt8Device(input,output,scales,scheme,ctx);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::QuantizeInt4PerGroup(const CAIF_DeviceTensor &input,
                          CAIF_DeviceTensor &output,
                          CAIF_DeviceTensor &scales,
                          uint32_t group_size,
                          CAIF_RunContext &ctx)
{
  try
  {
    RequireSameLocation(input,output,"QuantizeInt4PerGroup");
    RequireSameLocation(input,scales,"QuantizeInt4PerGroup");
    if(IsHost(input)==true)
    {
      THROW_CAIFE("QuantizeInt4PerGroup: host backend not supported");
    }
    QuantizeInt4PerGroupDevice(input,output,scales,group_size,ctx);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::DequantizeInt4PerGroup(const CAIF_DeviceTensor &input,
                            CAIF_DeviceTensor &output,
                            const CAIF_DeviceTensor &scales,
                            uint32_t group_size,
                            CAIF_RunContext &ctx)
{
  try
  {
    RequireSameLocation(input,output,"DequantizeInt4PerGroup");
    RequireSameLocation(input,scales,"DequantizeInt4PerGroup");
    if(IsHost(input)==true)
    {
      THROW_CAIFE("DequantizeInt4PerGroup: host backend not supported");
    }
    DequantizeInt4PerGroupDevice(input,output,scales,group_size,ctx);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::FakeQuantInt8(const CAIF_DeviceTensor &input,
                   CAIF_DeviceTensor &output,
                   QuantScheme_e scheme,
                   CAIF_RunContext &ctx)
{
  try
  {
    RequireSameLocation(input,output,"FakeQuantInt8");
    if(IsHost(input)==true)
    {
      THROW_CAIFE("FakeQuantInt8: host backend not supported");
    }
    // fp32-only by op contract: FakeQuant simulates fp32-master ↔ int8
    // round-trip noise; both ends must be fp32 master tensors. The
    // quantize/dequantize launchers (Phase 5.6 docstring) take the same
    // fp32-only contract.
    if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("FakeQuantInt8: input must be fp32");
    }
    if(output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("FakeQuantInt8: output must be fp32");
    }
    if(input.Shape()!=output.Shape())
    {
      THROW_CAIFE("FakeQuantInt8: input/output shape mismatch");
    }

    CAIF_CudaStream &stream=output.Stream();
    CAIF_DeviceTensor q_int8=CAIF_DeviceTensor::Zeros(input.Shape(),
                                                      stream,
                                                      CAIF_DataType::CAIF_DataType_e::Int8);
    uint32_t scale_count=1u;
    if(scheme==QuantScheme_e::PerChannel_e)
    {
      if(input.Shape().size()!=2)
      {
        THROW_CAIFE("FakeQuantInt8 PerChannel: input must be 2D");
      }
      scale_count=input.Shape()[1];
    }
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Zeros({scale_count},
                                                      stream,
                                                      CAIF_DataType::CAIF_DataType_e::Float32);

    QuantizeInt8Device(input,q_int8,scales,scheme,ctx);
    DequantizeInt8Device(q_int8,output,scales,scheme,ctx);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::FakeQuantInt4PerGroup(const CAIF_DeviceTensor &input,
                           CAIF_DeviceTensor &output,
                           uint32_t group_size,
                           CAIF_RunContext &ctx)
{
  try
  {
    RequireSameLocation(input,output,"FakeQuantInt4PerGroup");
    if(IsHost(input)==true)
    {
      THROW_CAIFE("FakeQuantInt4PerGroup: host backend not supported");
    }
    // fp32-only by op contract: FakeQuant simulates fp32-master ↔ int4
    // round-trip noise; both ends must be fp32 master tensors. The
    // int4 quantize/dequantize launchers carry the matching contract.
    if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("FakeQuantInt4PerGroup: input must be fp32");
    }
    if(output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("FakeQuantInt4PerGroup: output must be fp32");
    }
    if(input.Shape()!=output.Shape())
    {
      THROW_CAIFE("FakeQuantInt4PerGroup: input/output shape mismatch");
    }
    if(group_size==0u)
    {
      THROW_CAIFE("FakeQuantInt4PerGroup: group_size must be > 0");
    }

    CAIF_CudaStream &stream=output.Stream();
    const size_t total_elements=input.TotalElements();
    const uint32_t num_groups=
      static_cast<uint32_t>((total_elements+group_size-1u)/group_size);

    CAIF_DeviceTensor packed=CAIF_DeviceTensor::Zeros(input.Shape(),
                                                      stream,
                                                      CAIF_DataType::CAIF_DataType_e::Int4);
    CAIF_DeviceTensor scales=CAIF_DeviceTensor::Zeros({num_groups},
                                                      stream,
                                                      CAIF_DataType::CAIF_DataType_e::Float16);

    QuantizeInt4PerGroupDevice(input,packed,scales,group_size,ctx);
    DequantizeInt4PerGroupDevice(packed,output,scales,group_size,ctx);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SliceLastDim(const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &output,
                  uint32_t col_start)
{
  try
  {
    RequireSameLocation(input,output,"SliceLastDim");
    if(IsHost(input)==true)
    {
      SliceLastDimHost(input,output,col_start);
    }
    else
    {
      SliceLastDimDevice(input,output,col_start);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SliceLastDimBackward(const CAIF_DeviceTensor &grad_output,
                          CAIF_DeviceTensor &grad_input,
                          uint32_t col_start)
{
  try
  {
    RequireSameLocation(grad_output,grad_input,"SliceLastDimBackward");
    if(IsHost(grad_output)==true)
    {
      SliceLastDimBackwardHost(grad_output,grad_input,col_start);
    }
    else
    {
      SliceLastDimBackwardDevice(grad_output,grad_input,col_start);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ConcatLastDim(const CAIF_DeviceTensor &a,
                   const CAIF_DeviceTensor &b,
                   CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(a,b,output,"ConcatLastDim");
    if(IsHost(a)==true)
    {
      ConcatLastDimHost(a,b,output);
    }
    else
    {
      ConcatLastDimDevice(a,b,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Activations (forward)
//------------------------------------------------------------------------------

void CAIF_Ops::ReLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"ReLU");
    if(IsHost(input)==true)
    {
      ReLUHost(input,output);
    }
    else
    {
      ReLUDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Sigmoid(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Sigmoid");
    if(IsHost(input)==true)
    {
      SigmoidHost(input,output);
    }
    else
    {
      SigmoidDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Tanh(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Tanh");
    if(IsHost(input)==true)
    {
      TanhHost(input,output);
    }
    else
    {
      TanhDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Softmax(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Softmax");
    if(IsHost(input)==true)
    {
      SoftmaxHost(input,output);
    }
    else
    {
      SoftmaxDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::LeakyReLU(const CAIF_DeviceTensor &input,
               CAIF_DeviceTensor &output,
               float alpha)
{
  try
  {
    RequireSameLocation(input,output,"LeakyReLU");
    if(IsHost(input)==true)
    {
      LeakyReLUHost(input,output,alpha);
    }
    else
    {
      LeakyReLUDevice(input,output,alpha);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ELU(const CAIF_DeviceTensor &input,
         CAIF_DeviceTensor &output,
         float alpha)
{
  try
  {
    RequireSameLocation(input,output,"ELU");
    if(IsHost(input)==true)
    {
      ELUHost(input,output,alpha);
    }
    else
    {
      ELUDevice(input,output,alpha);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::GELU(const CAIF_DeviceTensor &input,
                    CAIF_DeviceTensor &output,
                    const CAIF_GELUApproximation::CAIF_GELUApproximation_e approx)
{
  try
  {
    RequireSameLocation(input,output,"GELU");
    if(IsHost(input)==true)
    {
      GELUHost(input,output,approx);
    }
    else
    {
      GELUDevice(input,output,approx);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Swish(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Swish");
    if(IsHost(input)==true)
    {
      SwishHost(input,output);
    }
    else
    {
      SwishDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Activations (backward)
//------------------------------------------------------------------------------

void CAIF_Ops::ReLUBackward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(grad_output,input,grad_input,"ReLUBackward");
    if(IsHost(grad_output)==true)
    {
      ReLUBackwardHost(grad_output,input,grad_input);
    }
    else
    {
      ReLUBackwardDevice(grad_output,input,grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SigmoidBackward(const CAIF_DeviceTensor &grad_output,
                     const CAIF_DeviceTensor &output,
                     CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(grad_output,output,grad_input,"SigmoidBackward");
    if(IsHost(grad_output)==true)
    {
      SigmoidBackwardHost(grad_output,output,grad_input);
    }
    else
    {
      SigmoidBackwardDevice(grad_output,output,grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::TanhBackward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &output,
                  CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(grad_output,output,grad_input,"TanhBackward");
    if(IsHost(grad_output)==true)
    {
      TanhBackwardHost(grad_output,output,grad_input);
    }
    else
    {
      TanhBackwardDevice(grad_output,output,grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SoftmaxBackward(const CAIF_DeviceTensor &grad_output,
                     const CAIF_DeviceTensor &output,
                     CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(grad_output,output,grad_input,"SoftmaxBackward");
    if(IsHost(grad_output)==true)
    {
      SoftmaxBackwardHost(grad_output,output,grad_input);
    }
    else
    {
      SoftmaxBackwardDevice(grad_output,output,grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::LeakyReLUBackward(const CAIF_DeviceTensor &grad_output,
                       const CAIF_DeviceTensor &input,
                       CAIF_DeviceTensor &grad_input,
                       float alpha)
{
  try
  {
    RequireSameLocation(grad_output,input,grad_input,"LeakyReLUBackward");
    if(IsHost(grad_output)==true)
    {
      LeakyReLUBackwardHost(grad_output,input,grad_input,alpha);
    }
    else
    {
      LeakyReLUBackwardDevice(grad_output,input,grad_input,alpha);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ELUBackward(const CAIF_DeviceTensor &grad_output,
                 const CAIF_DeviceTensor &input,
                 const CAIF_DeviceTensor &output,
                 CAIF_DeviceTensor &grad_input,
                 float alpha)
{
  try
  {
    RequireSameLocation(grad_output,input,grad_input,"ELUBackward");
    RequireSameLocation(grad_output,output,"ELUBackward");
    if(IsHost(grad_output)==true)
    {
      ELUBackwardHost(grad_output,input,output,grad_input,alpha);
    }
    else
    {
      ELUBackwardDevice(grad_output,input,output,grad_input,alpha);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::GELUBackward(const CAIF_DeviceTensor &grad_output,
                            const CAIF_DeviceTensor &input,
                            CAIF_DeviceTensor &grad_input,
                            const CAIF_GELUApproximation::CAIF_GELUApproximation_e approx)
{
  try
  {
    RequireSameLocation(grad_output,input,grad_input,"GELUBackward");
    if(IsHost(grad_output)==true)
    {
      GELUBackwardHost(grad_output,input,grad_input,approx);
    }
    else
    {
      GELUBackwardDevice(grad_output,input,grad_input,approx);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SwishBackward(const CAIF_DeviceTensor &grad_output,
                   const CAIF_DeviceTensor &input,
                   const CAIF_DeviceTensor &output,
                   CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(grad_output,input,grad_input,"SwishBackward");
    RequireSameLocation(grad_output,output,"SwishBackward");
    if(IsHost(grad_output)==true)
    {
      SwishBackwardHost(grad_output,input,output,grad_input);
    }
    else
    {
      SwishBackwardDevice(grad_output,input,output,grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Reductions (scalar return)
//------------------------------------------------------------------------------

float CAIF_Ops::ReduceSum(const CAIF_DeviceTensor &tensor)
{
  try
  {
    if(IsHost(tensor)==true)
    {
      return ReduceSumHost(tensor);
    }
    return ReduceSumDevice(tensor);
  }
  CAIF_CATCH_BLOCK();
  return 0.0f;
}

float CAIF_Ops::ReduceMean(const CAIF_DeviceTensor &tensor)
{
  try
  {
    if(IsHost(tensor)==true)
    {
      return ReduceMeanHost(tensor);
    }
    return ReduceMeanDevice(tensor);
  }
  CAIF_CATCH_BLOCK();
  return 0.0f;
}

//------------------------------------------------------------------------------
// Losses
//------------------------------------------------------------------------------

void CAIF_Ops::MSELoss(const CAIF_DeviceTensor &pred,
             const CAIF_DeviceTensor &target,
             CAIF_DeviceTensor &loss)
{
  try
  {
    RequireSameLocation(pred,target,loss,"MSELoss");
    if(IsHost(pred)==true)
    {
      MSELossHost(pred,target,loss);
    }
    else
    {
      MSELossDevice(pred,target,loss);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MSELossBackward(const CAIF_DeviceTensor &pred,
                     const CAIF_DeviceTensor &target,
                     CAIF_DeviceTensor &grad)
{
  try
  {
    RequireSameLocation(pred,target,grad,"MSELossBackward");
    if(IsHost(pred)==true)
    {
      MSELossBackwardHost(pred,target,grad);
    }
    else
    {
      MSELossBackwardDevice(pred,target,grad);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Optimizer
//------------------------------------------------------------------------------

void CAIF_Ops::AdamUpdate(CAIF_DeviceTensor &param,
                const CAIF_DeviceTensor &grad,
                CAIF_DeviceTensor &m,
                CAIF_DeviceTensor &v,
                float lr,
                float beta1,
                float beta2,
                float epsilon,
                float weight_decay,
                int t)
{
  try
  {
    RequireSameLocation(param,grad,m,"AdamUpdate");
    RequireSameLocation(param,v,"AdamUpdate");
    if(IsHost(param)==true)
    {
      AdamUpdateHost(param,grad,m,v,lr,beta1,beta2,epsilon,weight_decay,t);
    }
    else
    {
      AdamUpdateDevice(param,grad,m,v,lr,beta1,beta2,epsilon,weight_decay,t);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SgdUpdate(CAIF_DeviceTensor &param,
               const CAIF_DeviceTensor &grad,
               float lr,
               float weight_decay)
{
  try
  {
    RequireSameLocation(param,grad,"SgdUpdate");
    if(IsHost(param)==true)
    {
      SgdUpdateHost(param,grad,lr,weight_decay);
    }
    else
    {
      SgdUpdateDevice(param,grad,lr,weight_decay);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MomentumUpdate(CAIF_DeviceTensor &param,
                    const CAIF_DeviceTensor &grad,
                    CAIF_DeviceTensor &velocity,
                    float lr,
                    float momentum,
                    float weight_decay)
{
  try
  {
    RequireSameLocation(param,grad,velocity,"MomentumUpdate");
    if(IsHost(param)==true)
    {
      MomentumUpdateHost(param,grad,velocity,lr,momentum,weight_decay);
    }
    else
    {
      MomentumUpdateDevice(param,grad,velocity,lr,momentum,weight_decay);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::RmspropUpdate(CAIF_DeviceTensor &param,
                   const CAIF_DeviceTensor &grad,
                   CAIF_DeviceTensor &avg_sq,
                   float lr,
                   float alpha,
                   float epsilon,
                   float weight_decay)
{
  try
  {
    RequireSameLocation(param,grad,avg_sq,"RmspropUpdate");
    if(IsHost(param)==true)
    {
      RmspropUpdateHost(param,grad,avg_sq,lr,alpha,epsilon,weight_decay);
    }
    else
    {
      RmspropUpdateDevice(param,grad,avg_sq,lr,alpha,epsilon,weight_decay);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AdaGradUpdate(CAIF_DeviceTensor &param,
                   const CAIF_DeviceTensor &grad,
                   CAIF_DeviceTensor &accum,
                   float lr,
                   float epsilon,
                   float weight_decay)
{
  try
  {
    RequireSameLocation(param,grad,accum,"AdaGradUpdate");
    if(IsHost(param)==true)
    {
      AdaGradUpdateHost(param,grad,accum,lr,epsilon,weight_decay);
    }
    else
    {
      AdaGradUpdateDevice(param,grad,accum,lr,epsilon,weight_decay);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Additional element-wise
//------------------------------------------------------------------------------

void CAIF_Ops::Multiply(const CAIF_DeviceTensor &a,
              const CAIF_DeviceTensor &b,
              CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(a,b,output,"Multiply");
    if(IsHost(a)==true)
    {
      MultiplyHost(a,b,output);
    }
    else
    {
      MultiplyDevice(a,b,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SiLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"SiLU");
    if(IsHost(input)==true)
    {
      SiLUHost(input,output);
    }
    else
    {
      SiLUDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SiLUBackward(const CAIF_DeviceTensor &input,
                  const CAIF_DeviceTensor &grad_output,
                  CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(input,grad_output,grad_input,"SiLUBackward");
    if(IsHost(input)==true)
    {
      SiLUBackwardHost(input,grad_output,grad_input);
    }
    else
    {
      SiLUBackwardDevice(input,grad_output,grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddBias(const CAIF_DeviceTensor &input,
             const CAIF_DeviceTensor &bias,
             CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,bias,output,"AddBias");
    if(IsHost(input)==true)
    {
      AddBiasHost(input,bias,output);
    }
    else
    {
      AddBiasDevice(input,bias,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddScalar(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"AddScalar");
    if(IsHost(input)==true)
    {
      AddScalarHost(input,scalar,output);
    }
    else
    {
      AddScalarDevice(input,scalar,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Subtract(const CAIF_DeviceTensor &a,
              const CAIF_DeviceTensor &b,
              CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(a,b,output,"Subtract");
    if(IsHost(a)==true)
    {
      SubtractHost(a,b,output);
    }
    else
    {
      SubtractDevice(a,b,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SubtractScalar(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"SubtractScalar");
    if(IsHost(input)==true)
    {
      SubtractScalarHost(input,scalar,output);
    }
    else
    {
      SubtractScalarDevice(input,scalar,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Divide(const CAIF_DeviceTensor &a,
            const CAIF_DeviceTensor &b,
            CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(a,b,output,"Divide");
    if(IsHost(a)==true)
    {
      DivideHost(a,b,output);
    }
    else
    {
      DivideDevice(a,b,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::DivideScalar(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"DivideScalar");
    if(IsHost(input)==true)
    {
      DivideScalarHost(input,scalar,output);
    }
    else
    {
      DivideScalarDevice(input,scalar,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Sqrt(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Sqrt");
    if(IsHost(input)==true)
    {
      SqrtHost(input,output);
    }
    else
    {
      SqrtDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Reductions (tensor return)
//------------------------------------------------------------------------------

void CAIF_Ops::SumAxis(const CAIF_DeviceTensor &input,uint32_t axis,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"SumAxis");
    if(IsHost(input)==true)
    {
      SumAxisHost(input,axis,output);
    }
    else
    {
      SumAxisDevice(input,axis,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::Sum(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"Sum");
    if(IsHost(input)==true)
    {
      SumHost(input,output);
    }
    else
    {
      SumDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::LogSumExp(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"LogSumExp");
    if(IsHost(input)==true)
    {
      LogSumExpHost(input,output);
    }
    else
    {
      LogSumExpDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Top-k / scatter / normalize
//------------------------------------------------------------------------------

void CAIF_Ops::TopK(const CAIF_DeviceTensor &input,
          uint32_t k,
          CAIF_DeviceTensor &indices,
          CAIF_DeviceTensor &values)
{
  try
  {
    RequireSameLocation(input,indices,values,"TopK");
    if(IsHost(input)==true)
    {
      TopKHost(input,k,indices,values);
    }
    else
    {
      TopKDevice(input,k,indices,values);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::NormalizeRows(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(input,output,"NormalizeRows");
    if(IsHost(input)==true)
    {
      NormalizeRowsHost(input,output);
    }
    else
    {
      NormalizeRowsDevice(input,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::NormalizeRowsBackwardTopKGather(const CAIF_DeviceTensor &grad_w,
                                     const CAIF_DeviceTensor &probs,
                                     const CAIF_DeviceTensor &indices,
                                     CAIF_DeviceTensor &grad_p_topk)
{
  try
  {
    RequireSameLocation(grad_w,probs,grad_p_topk,"NormalizeRowsBackwardTopKGather");
    RequireSameLocation(grad_w,indices,"NormalizeRowsBackwardTopKGather");
    if(IsHost(grad_w)==true)
    {
      NormalizeRowsBackwardTopKGatherHost(grad_w,probs,indices,grad_p_topk);
    }
    else
    {
      NormalizeRowsBackwardTopKGatherDevice(grad_w,probs,indices,grad_p_topk);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::GatherTopKValues(const CAIF_DeviceTensor &scores,
                      const CAIF_DeviceTensor &indices,
                      CAIF_DeviceTensor &out)
{
  try
  {
    RequireSameLocation(scores,indices,out,"GatherTopKValues");
    if(IsHost(scores)==true)
    {
      GatherTopKValuesHost(scores,indices,out);
    }
    else
    {
      GatherTopKValuesDevice(scores,indices,out);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ScatterAdd(const CAIF_DeviceTensor &values,
                const CAIF_DeviceTensor &indices,
                CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(values,indices,output,"ScatterAdd");
    if(IsHost(values)==true)
    {
      ScatterAddHost(values,indices,output);
    }
    else
    {
      ScatterAddDevice(values,indices,output);
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// MoE
//------------------------------------------------------------------------------

void CAIF_Ops::MoEDispatch(const CAIF_DeviceTensor &input,
                 const CAIF_DeviceTensor &expert_indices,
                 uint32_t top_k,
                 const std::vector<uint32_t> &token_counts,
                 std::vector<CAIF_DeviceTensor> &expert_inputs)
{
  try
  {
    RequireSameLocation(input,expert_indices,"MoEDispatch");
    if(IsHost(input)==true)
    {
      MoEDispatchHost(input,expert_indices,top_k,token_counts,expert_inputs);
    }
    else
    {
      MoEDispatchDevice(input,expert_indices,top_k,token_counts,expert_inputs);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECombine(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                const CAIF_DeviceTensor &expert_indices,
                const CAIF_DeviceTensor &expert_weights,
                uint32_t top_k,
                const std::vector<uint32_t> &token_counts,
                CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(expert_indices,expert_weights,output,"MoECombine");
    if(IsHost(output)==true)
    {
      MoECombineHost(expert_outputs,
                     expert_indices,
                     expert_weights,
                     top_k,
                     token_counts,
                     output);
    }
    else
    {
      MoECombineDevice(expert_outputs,
                       expert_indices,
                       expert_weights,
                       top_k,
                       token_counts,
                       output);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECombineBackward(const CAIF_DeviceTensor &grad_output,
                        const std::vector<CAIF_DeviceTensor> &expert_outputs,
                        const CAIF_DeviceTensor &expert_indices,
                        const CAIF_DeviceTensor &expert_weights,
                        uint32_t top_k,
                        const std::vector<uint32_t> &token_counts,
                        std::vector<CAIF_DeviceTensor> &grad_expert_outputs,
                        CAIF_DeviceTensor &grad_weights)
{
  try
  {
    RequireSameLocation(grad_output,expert_indices,grad_weights,"MoECombineBackward");
    RequireSameLocation(grad_output,expert_weights,"MoECombineBackward");
    if(IsHost(grad_output)==true)
    {
      MoECombineBackwardHost(grad_output,
                             expert_outputs,
                             expert_indices,
                             expert_weights,
                             top_k,
                             token_counts,
                             grad_expert_outputs,
                             grad_weights);
    }
    else
    {
      MoECombineBackwardDevice(grad_output,
                               expert_outputs,
                               expert_indices,
                               expert_weights,
                               top_k,
                               token_counts,
                               grad_expert_outputs,
                               grad_weights);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEDispatchBackward(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
                         const CAIF_DeviceTensor &expert_indices,
                         uint32_t top_k,
                         const std::vector<uint32_t> &token_counts,
                         CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(expert_indices,grad_input,"MoEDispatchBackward");
    if(IsHost(grad_input)==true)
    {
      MoEDispatchBackwardHost(grad_expert_inputs,
                              expert_indices,
                              top_k,
                              token_counts,
                              grad_input);
    }
    else
    {
      MoEDispatchBackwardDevice(grad_expert_inputs,
                                expert_indices,
                                top_k,
                                token_counts,
                                grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoETopKGating(const CAIF_DeviceTensor &router_logits,
                   uint32_t num_experts,
                   uint32_t top_k,
                   CAIF_DeviceTensor &expert_indices,
                   CAIF_DeviceTensor &expert_weights,
                   CAIF_DeviceTensor &router_probs)
{
  try
  {
    RequireSameLocation(router_logits,expert_indices,expert_weights,"MoETopKGating");
    RequireSameLocation(router_logits,router_probs,"MoETopKGating");
    if(IsHost(router_logits)==true)
    {
      MoETopKGatingHost(router_logits,
                        num_experts,
                        top_k,
                        expert_indices,
                        expert_weights,
                        router_probs);
    }
    else
    {
      MoETopKGatingDevice(router_logits,
                          num_experts,
                          top_k,
                          expert_indices,
                          expert_weights,
                          router_probs);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECountPerExpert(const CAIF_DeviceTensor &expert_indices,
                       uint32_t num_experts,
                       uint32_t top_k,
                       CAIF_DeviceTensor &expert_counts)
{
  try
  {
    RequireSameLocation(expert_indices,expert_counts,"MoECountPerExpert");
    if(IsHost(expert_indices)==true)
    {
      MoECountPerExpertHost(expert_indices,num_experts,top_k,expert_counts);
    }
    else
    {
      MoECountPerExpertDevice(expert_indices,num_experts,top_k,expert_counts);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEZLossGradAdd(const CAIF_DeviceTensor &logsumexp_scaled,
                     const CAIF_DeviceTensor &probs,
                     CAIF_DeviceTensor &grad_logits)
{
  try
  {
    RequireSameLocation(logsumexp_scaled,probs,grad_logits,"MoEZLossGradAdd");
    if(IsHost(logsumexp_scaled)==true)
    {
      MoEZLossGradAddHost(logsumexp_scaled,probs,grad_logits);
    }
    else
    {
      MoEZLossGradAddDevice(logsumexp_scaled,probs,grad_logits);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEDispatchGPU(const CAIF_DeviceTensor &input,
                    const CAIF_DeviceTensor &expert_indices,
                    const CAIF_DeviceTensor &dispatch_map,
                    const CAIF_DeviceTensor &expert_offsets,
                    uint32_t top_k,
                    CAIF_DeviceTensor &expert_buffer)
{
  try
  {
    RequireSameLocation(input,expert_indices,expert_buffer,"MoEDispatchGPU");
    RequireSameLocation(input,dispatch_map,"MoEDispatchGPU");
    RequireSameLocation(input,expert_offsets,"MoEDispatchGPU");
    if(IsHost(input)==true)
    {
      MoEDispatchGPUHost(input,
                         expert_indices,
                         dispatch_map,
                         expert_offsets,
                         top_k,
                         expert_buffer);
    }
    else
    {
      MoEDispatchGPUDevice(input,
                           expert_indices,
                           dispatch_map,
                           expert_offsets,
                           top_k,
                           expert_buffer);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECombineGPU(const CAIF_DeviceTensor &expert_buffer,
                   const CAIF_DeviceTensor &expert_indices,
                   const CAIF_DeviceTensor &expert_weights,
                   const CAIF_DeviceTensor &dispatch_map,
                   const CAIF_DeviceTensor &expert_offsets,
                   uint32_t top_k,
                   CAIF_DeviceTensor &output)
{
  try
  {
    RequireSameLocation(expert_buffer,expert_indices,output,"MoECombineGPU");
    RequireSameLocation(expert_buffer,expert_weights,"MoECombineGPU");
    RequireSameLocation(expert_buffer,dispatch_map,"MoECombineGPU");
    RequireSameLocation(expert_buffer,expert_offsets,"MoECombineGPU");
    if(IsHost(expert_buffer)==true)
    {
      MoECombineGPUHost(expert_buffer,
                        expert_indices,
                        expert_weights,
                        dispatch_map,
                        expert_offsets,
                        top_k,
                        output);
    }
    else
    {
      MoECombineGPUDevice(expert_buffer,
                          expert_indices,
                          expert_weights,
                          dispatch_map,
                          expert_offsets,
                          top_k,
                          output);
    }
  }
  CAIF_CATCH_BLOCK();
}

uint32_t CAIF_Ops::MoEBuildDispatchMap(const CAIF_DeviceTensor &expert_indices,
                             uint32_t num_experts,
                             uint32_t top_k,
                             uint32_t capacity_per_expert,
                             CAIF_DeviceTensor &dispatch_map,
                             CAIF_DeviceTensor &expert_offsets)
{
  try
  {
    RequireSameLocation(expert_indices,dispatch_map,expert_offsets,"MoEBuildDispatchMap");
    if(IsHost(expert_indices)==true)
    {
      return MoEBuildDispatchMapHost(expert_indices,
                                     num_experts,
                                     top_k,
                                     capacity_per_expert,
                                     dispatch_map,
                                     expert_offsets);
    }
    return MoEBuildDispatchMapDevice(expert_indices,
                                     num_experts,
                                     top_k,
                                     capacity_per_expert,
                                     dispatch_map,
                                     expert_offsets);
  }
  CAIF_CATCH_BLOCK();
  return 0u;
}

void CAIF_Ops::MoECombineBackwardGPU(const CAIF_DeviceTensor &grad_output,
                           const CAIF_DeviceTensor &expert_buffer,
                           const CAIF_DeviceTensor &expert_indices,
                           const CAIF_DeviceTensor &expert_weights,
                           const CAIF_DeviceTensor &dispatch_map,
                           const CAIF_DeviceTensor &expert_offsets,
                           uint32_t top_k,
                           CAIF_DeviceTensor &grad_expert_buffer,
                           CAIF_DeviceTensor &grad_weights)
{
  try
  {
    RequireSameLocation(grad_output,expert_buffer,grad_expert_buffer,"MoECombineBackwardGPU");
    RequireSameLocation(grad_output,expert_indices,"MoECombineBackwardGPU");
    RequireSameLocation(grad_output,expert_weights,"MoECombineBackwardGPU");
    RequireSameLocation(grad_output,dispatch_map,"MoECombineBackwardGPU");
    RequireSameLocation(grad_output,expert_offsets,"MoECombineBackwardGPU");
    RequireSameLocation(grad_output,grad_weights,"MoECombineBackwardGPU");
    if(IsHost(grad_output)==true)
    {
      MoECombineBackwardGPUHost(grad_output,
                                expert_buffer,
                                expert_indices,
                                expert_weights,
                                dispatch_map,
                                expert_offsets,
                                top_k,
                                grad_expert_buffer,
                                grad_weights);
    }
    else
    {
      MoECombineBackwardGPUDevice(grad_output,
                                  expert_buffer,
                                  expert_indices,
                                  expert_weights,
                                  dispatch_map,
                                  expert_offsets,
                                  top_k,
                                  grad_expert_buffer,
                                  grad_weights);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEDispatchBackwardGPU(const CAIF_DeviceTensor &grad_expert_buffer,
                            const CAIF_DeviceTensor &expert_indices,
                            const CAIF_DeviceTensor &dispatch_map,
                            const CAIF_DeviceTensor &expert_offsets,
                            uint32_t top_k,
                            CAIF_DeviceTensor &grad_input)
{
  try
  {
    RequireSameLocation(grad_expert_buffer,expert_indices,grad_input,"MoEDispatchBackwardGPU");
    RequireSameLocation(grad_expert_buffer,dispatch_map,"MoEDispatchBackwardGPU");
    RequireSameLocation(grad_expert_buffer,expert_offsets,"MoEDispatchBackwardGPU");
    if(IsHost(grad_expert_buffer)==true)
    {
      MoEDispatchBackwardGPUHost(grad_expert_buffer,
                                 expert_indices,
                                 dispatch_map,
                                 expert_offsets,
                                 top_k,
                                 grad_input);
    }
    else
    {
      MoEDispatchBackwardGPUDevice(grad_expert_buffer,
                                   expert_indices,
                                   dispatch_map,
                                   expert_offsets,
                                   top_k,
                                   grad_input);
    }
  }
  CAIF_CATCH_BLOCK();
}


}//end instance namespace
