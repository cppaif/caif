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
// CAIF_Ops host backend (OpenBLAS / OpenMP).
//
// Every FooHost(...) here is a host-side implementation that operates on
// tensors with Location_e::Host_e. The public CAIF_Ops::Foo(...) dispatcher
// in caif_ops.cpp branches on tensor location and routes host tensors here.
//
// Primary dtype: Float32 (full BLAS + direct-loop path). Matrix ops
// up-cast Float16/BFloat16 inputs to Float32 for BLAS and down-cast the
// result. Element-wise and reductions accept Float32 directly; dtype
// dispatch throws for unsupported dtypes (documented per op).
//------------------------------------------------------------------------------
#include "caif_ops.h"
#include "caif_constants.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_exception.h"
#include "caif_host_fp_cast.h"
#include "caif_host_gemm.h"
#include "caif_host_slice_helper.h"
#include "caif_host_float_read_view.h"
#include "caif_host_float_write_view.h"
#include "caif_host_float_readwrite_view.h"
#include "caif_topk_descending_compare.h"
#include "openblas/cblas.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace instance
{

//------------------------------------------------------------------------------
// Matrix
//------------------------------------------------------------------------------

void CAIF_Ops::MatMulHost(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &output,
                CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    RequireMatchingDtype(a,b,output,g_caif_op_matmul);
    const auto &sa=a.Shape();
    const auto &sb=b.Shape();
    const auto &so=output.Shape();
    if(sa.size()!=2||sb.size()!=2||so.size()!=2)
    {
      THROW_CAIFE("MatMul host: 2D tensors required");
    }
    if(sa[1]!=sb[0])
    {
      THROW_CAIFE("MatMul host: A cols must equal B rows");
    }
    if(so[0]!=sa[0]||so[1]!=sb[1])
    {
      THROW_CAIFE("MatMul host: output shape mismatch");
    }
    CAIF_HostGemm::MatMul2DFloat(a,b,output,false,false);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MatMulTransposeAHost(const CAIF_DeviceTensor &a,
                          const CAIF_DeviceTensor &b,
                          CAIF_DeviceTensor &output,
                          CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    RequireMatchingDtype(a,b,output,g_caif_op_matmul_transpose_a);
    const auto &sa=a.Shape();
    const auto &sb=b.Shape();
    const auto &so=output.Shape();
    if(sa.size()!=2||sb.size()!=2||so.size()!=2)
    {
      THROW_CAIFE("MatMulTransposeA host: 2D tensors required");
    }
    if(sa[0]!=sb[0])
    {
      THROW_CAIFE("MatMulTransposeA host: A rows must equal B rows");
    }
    if(so[0]!=sa[1]||so[1]!=sb[1])
    {
      THROW_CAIFE("MatMulTransposeA host: output shape mismatch");
    }
    CAIF_HostGemm::MatMul2DFloat(a,b,output,true,false);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MatMulTransposeBHost(const CAIF_DeviceTensor &a,
                          const CAIF_DeviceTensor &b,
                          CAIF_DeviceTensor &output,
                          CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    RequireMatchingDtype(a,b,output,g_caif_op_matmul_transpose_b);
    const auto &sa=a.Shape();
    const auto &sb=b.Shape();
    const auto &so=output.Shape();
    if(sa.size()!=2||sb.size()!=2||so.size()!=2)
    {
      THROW_CAIFE("MatMulTransposeB host: 2D tensors required");
    }
    if(sa[1]!=sb[1])
    {
      THROW_CAIFE("MatMulTransposeB host: A cols must equal B cols");
    }
    if(so[0]!=sa[0]||so[1]!=sb[0])
    {
      THROW_CAIFE("MatMulTransposeB host: output shape mismatch");
    }
    CAIF_HostGemm::MatMul2DFloat(a,b,output,false,true);
  }
  CAIF_CATCH_BLOCK();
}


void CAIF_Ops::BatchedMatMulHost(const CAIF_DeviceTensor &a,
                       const CAIF_DeviceTensor &b,
                       CAIF_DeviceTensor &output,
                       int m,
                       int k,
                       int n,
                       int batch_count,
                       CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    RequireMatchingDtype(a,b,output,g_caif_op_batched_matmul);
    CAIF_HostGemm::BatchedMatMulFloatInternal(a,b,output,m,k,n,batch_count,false,false);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BatchedMatMulTransposeAHost(const CAIF_DeviceTensor &a,
                                 const CAIF_DeviceTensor &b,
                                 CAIF_DeviceTensor &output,
                                 int k,
                                 int m,
                                 int n,
                                 int batch_count,
                                 CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    RequireMatchingDtype(a,b,output,g_caif_op_batched_matmul_transpose_a);
    CAIF_HostGemm::BatchedMatMulFloatInternal(a,b,output,m,k,n,batch_count,true,false);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BatchedMatMulTransposeBHost(const CAIF_DeviceTensor &a,
                                 const CAIF_DeviceTensor &b,
                                 CAIF_DeviceTensor &output,
                                 int m,
                                 int k,
                                 int n,
                                 int batch_count,
                                 CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    RequireMatchingDtype(a,b,output,g_caif_op_batched_matmul_transpose_b);
    CAIF_HostGemm::BatchedMatMulFloatInternal(a,b,output,m,k,n,batch_count,false,true);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::TransposeHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const auto &si=input.Shape();
    if(si.size()!=2)
    {
      THROW_CAIFE("Transpose host: 2D tensor required");
    }
    const int rows=static_cast<int>(si[0]);
    const int cols=static_cast<int>(si[1]);
    if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32
       ||output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Transpose host: Float32 only");
    }
    // fp32 by gate above
    const float *in=static_cast<const float*>(input.DeviceDataRaw());
    // fp32 by gate above
    float *out=static_cast<float*>(output.DeviceDataRaw());
    #pragma omp parallel for
    for(int r=0;r<rows;++r)
    {
      for(int c=0;c<cols;++c)
      {
        out[static_cast<size_t>(c)*static_cast<size_t>(rows)+static_cast<size_t>(r)]=
          in[static_cast<size_t>(r)*static_cast<size_t>(cols)+static_cast<size_t>(c)];
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddHost(const CAIF_DeviceTensor &a,
             const CAIF_DeviceTensor &b,
             CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=a.TotalElements();
    if(b.TotalElements()!=n||output.TotalElements()!=n)
    {
      THROW_CAIFE("Add host: element count mismatch");
    }
    CAIF_HostFloatReadView av(a);
    CAIF_HostFloatReadView bv(b);
    CAIF_HostFloatWriteView ov(output);
    const float *ap=av.Data();
    const float *bp=bv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ap[i]+bp[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ScaleHost(CAIF_DeviceTensor &tensor,float scale)
{
  try
  {
    const size_t n=tensor.TotalElements();
    CAIF_HostFloatReadWriteView tv(tensor);
    float *tp=tv.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      tp[i]*=scale;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::UnscaleCheckInfHost(CAIF_DeviceTensor &grad,
                                   float inv_scale,
                                   CAIF_DeviceTensor &found_inf)
{
  try
  {
    if(found_inf.TotalElements()!=1)
    {
      THROW_CAIFE("UnscaleCheckInf host: found_inf must have exactly 1 element");
    }
    const size_t n=grad.TotalElements();
    CAIF_HostFloatReadWriteView gv(grad);
    float *gp=gv.Data();
    bool overflow=false;
    #pragma omp parallel for reduction(||:overflow)
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      gp[i]*=inv_scale;
      if(std::isfinite(gp[i])==false)
      {
        overflow=true;
      }
    }
    if(overflow==true)
    {
      CAIF_HostFloatReadWriteView fv(found_inf);
      fv.Data()[0]=1.0f;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ScaleHost(const CAIF_DeviceTensor &input,
               float scale,
               CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    if(output.TotalElements()!=n)
    {
      THROW_CAIFE("Scale host: element count mismatch");
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ip[i]*scale;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddScaledHost(CAIF_DeviceTensor &target,
                   const CAIF_DeviceTensor &source,
                   float scale)
{
  try
  {
    const size_t n=target.TotalElements();
    if(source.TotalElements()!=n)
    {
      THROW_CAIFE("AddScaled host: element count mismatch");
    }
    CAIF_HostFloatReadWriteView tv(target);
    CAIF_HostFloatReadView sv(source);
    float *tp=tv.Data();
    const float *sp=sv.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      tp[i]+=sp[i]*scale;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BiasAddHost(const CAIF_DeviceTensor &input,
                 const CAIF_DeviceTensor &bias,
                 CAIF_DeviceTensor &output)
{
  try
  {
    const auto &si=input.Shape();
    if(si.size()!=2)
    {
      THROW_CAIFE("BiasAdd host: 2D input required");
    }
    const int rows=static_cast<int>(si[0]);
    const int cols=static_cast<int>(si[1]);
    if(bias.TotalElements()!=static_cast<size_t>(cols))
    {
      THROW_CAIFE("BiasAdd host: bias length must equal input cols");
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatReadView bv(bias);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    const float *bp=bv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int r=0;r<rows;++r)
    {
      for(int c=0;c<cols;++c)
      {
        const size_t idx=static_cast<size_t>(r)*static_cast<size_t>(cols)+static_cast<size_t>(c);
        op[idx]=ip[idx]+bp[c];
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MatMulBiasHost(const CAIF_DeviceTensor &a,
                    const CAIF_DeviceTensor &b,
                    const CAIF_DeviceTensor &bias,
                    CAIF_DeviceTensor &output,
                    cudaStream_t stream,
                    CAIF_RunContext &ctx)
{
  try
  {
    (void)stream;
    MatMulHost(a,b,output,ctx);
    const auto &so=output.Shape();
    if(so.size()!=2)
    {
      THROW_CAIFE("MatMulBias host: 2D output required");
    }
    const int rows=static_cast<int>(so[0]);
    const int cols=static_cast<int>(so[1]);
    if(bias.TotalElements()!=static_cast<size_t>(cols))
    {
      THROW_CAIFE("MatMulBias host: bias length must equal output cols");
    }
    CAIF_HostFloatReadWriteView ov(output);
    CAIF_HostFloatReadView bv(bias);
    float *op=ov.Data();
    const float *bp=bv.Data();
    #pragma omp parallel for
    for(int r=0;r<rows;++r)
    {
      for(int c=0;c<cols;++c)
      {
        op[static_cast<size_t>(r)*static_cast<size_t>(cols)+static_cast<size_t>(c)]+=bp[c];
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::BiasGradientHost(const CAIF_DeviceTensor &grad,
                      CAIF_DeviceTensor &bias_grad)
{
  try
  {
    const auto &sg=grad.Shape();
    if(sg.size()!=2)
    {
      THROW_CAIFE("BiasGradient host: 2D grad required");
    }
    const int rows=static_cast<int>(sg[0]);
    const int cols=static_cast<int>(sg[1]);
    if(bias_grad.TotalElements()!=static_cast<size_t>(cols))
    {
      THROW_CAIFE("BiasGradient host: bias_grad length must equal grad cols");
    }
    CAIF_HostFloatReadView gv(grad);
    CAIF_HostFloatWriteView bv(bias_grad);
    const float *gp=gv.Data();
    float *bp=bv.Data();
    #pragma omp parallel for
    for(int c=0;c<cols;++c)
    {
      float acc=0.0f;
      for(int r=0;r<rows;++r)
      {
        acc+=gp[static_cast<size_t>(r)*static_cast<size_t>(cols)+static_cast<size_t>(c)];
      }
      bp[c]=acc;
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Positional encodings
//------------------------------------------------------------------------------

void CAIF_Ops::AddPositionalEncodingHost(const CAIF_DeviceTensor &input,
                               const CAIF_DeviceTensor &pe_table,
                               CAIF_DeviceTensor &output)
{
  try
  {
    const auto &si=input.Shape();
    const auto &sp=pe_table.Shape();
    if(si.size()!=3||sp.size()!=2)
    {
      THROW_CAIFE("AddPositionalEncoding host: input 3D, pe_table 2D");
    }
    const int batch=static_cast<int>(si[0]);
    const int seq=static_cast<int>(si[1]);
    const int dim=static_cast<int>(si[2]);
    if(sp[1]!=static_cast<uint32_t>(dim))
    {
      THROW_CAIFE("AddPositionalEncoding host: pe_table dim mismatch");
    }
    const int pe_rows=static_cast<int>(sp[0]);
    const float *ip=CAIF_HostFpCast::HostFp32(input,"AddPositionalEncoding");
    const float *pp=CAIF_HostFpCast::HostFp32(pe_table,"AddPositionalEncoding");
    float *op=CAIF_HostFpCast::HostFp32(output,"AddPositionalEncoding");
    #pragma omp parallel for collapse(2)
    for(int b=0;b<batch;++b)
    {
      for(int s=0;s<seq;++s)
      {
        const int pe_row=std::min(s,pe_rows-1);
        for(int d=0;d<dim;++d)
        {
          const size_t io=static_cast<size_t>(b)*static_cast<size_t>(seq)*static_cast<size_t>(dim)
                          +static_cast<size_t>(s)*static_cast<size_t>(dim)
                          +static_cast<size_t>(d);
          op[io]=ip[io]+pp[static_cast<size_t>(pe_row)*static_cast<size_t>(dim)+static_cast<size_t>(d)];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::PositionalEncodingBackwardHost(const CAIF_DeviceTensor &grad_output,
                                    CAIF_DeviceTensor &grad_table)
{
  try
  {
    const auto &sg=grad_output.Shape();
    const auto &st=grad_table.Shape();
    if(sg.size()!=3||st.size()!=2)
    {
      THROW_CAIFE("PositionalEncodingBackward host: grad 3D, table 2D");
    }
    const int batch=static_cast<int>(sg[0]);
    const int seq=static_cast<int>(sg[1]);
    const int dim=static_cast<int>(sg[2]);
    const int pe_rows=static_cast<int>(st[0]);
    if(st[1]!=static_cast<uint32_t>(dim))
    {
      THROW_CAIFE("PositionalEncodingBackward host: dim mismatch");
    }
    const float *gp=CAIF_HostFpCast::HostFp32(grad_output,"PositionalEncodingBackward");
    float *tp=CAIF_HostFpCast::HostFp32(grad_table,"PositionalEncodingBackward");
    std::memset(tp,0,st[0]*st[1]*sizeof(float));
    for(int b=0;b<batch;++b)
    {
      for(int s=0;s<seq;++s)
      {
        const int pe_row=std::min(s,pe_rows-1);
        for(int d=0;d<dim;++d)
        {
          const size_t go=static_cast<size_t>(b)*static_cast<size_t>(seq)*static_cast<size_t>(dim)
                          +static_cast<size_t>(s)*static_cast<size_t>(dim)
                          +static_cast<size_t>(d);
          tp[static_cast<size_t>(pe_row)*static_cast<size_t>(dim)+static_cast<size_t>(d)]+=gp[go];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK();
}


int32_t CAIF_Ops::RelativePositionBucket(int32_t relative_position,
                                         uint32_t max_distance,
                                         bool bidirectional,
                                         uint32_t num_buckets)
{
  int32_t ret=0;
  int32_t n=-relative_position;
  if(bidirectional==true)
  {
    num_buckets/=2;
    if(n<0)
    {
      ret+=static_cast<int32_t>(num_buckets);
      n=-n;
    }
  }
  else
  {
    if(n<0)
    {
      n=0;
    }
  }
  const int32_t half=static_cast<int32_t>(num_buckets)/2;
  bool is_small=false;
  if(n<half)
  {
    is_small=true;
  }
  if(is_small==true)
  {
    ret+=n;
  }
  else
  {
    const float ln_half=std::log(static_cast<float>(half));
    const float ln_ratio=std::log(static_cast<float>(max_distance)/static_cast<float>(half));
    const float scaled=(std::log(static_cast<float>(n))-ln_half)/ln_ratio;
    int32_t val=half+static_cast<int32_t>(scaled*static_cast<float>(half));
    if(val>=static_cast<int32_t>(num_buckets))
    {
      val=static_cast<int32_t>(num_buckets)-1;
    }
    ret+=val;
  }
  return ret;
}


void CAIF_Ops::ComputeRelativePositionBiasHost(const CAIF_DeviceTensor &embedding,
                                     CAIF_DeviceTensor &output,
                                     uint32_t max_distance,
                                     bool bidirectional)
{
  try
  {
    const auto &se=embedding.Shape();
    const auto &so=output.Shape();
    if(se.size()!=2||so.size()!=3)
    {
      THROW_CAIFE("ComputeRelativePositionBias host: embedding 2D, output 3D");
    }
    const int num_heads=static_cast<int>(se[0]);
    const int num_buckets=static_cast<int>(se[1]);
    const int q_len=static_cast<int>(so[1]);
    const int k_len=static_cast<int>(so[2]);
    if(static_cast<int>(so[0])!=num_heads)
    {
      THROW_CAIFE("ComputeRelativePositionBias host: head mismatch");
    }
    const float *ep=CAIF_HostFpCast::HostFp32(embedding,"ComputeRelativePositionBias");
    float *op=CAIF_HostFpCast::HostFp32(output,"ComputeRelativePositionBias");
    #pragma omp parallel for collapse(2)
    for(int h=0;h<num_heads;++h)
    {
      for(int q=0;q<q_len;++q)
      {
        for(int k=0;k<k_len;++k)
        {
          const int32_t bucket=RelativePositionBucket(k-q,
                                                     max_distance,
                                                     bidirectional,
                                                     static_cast<uint32_t>(num_buckets));
          op[static_cast<size_t>(h)*static_cast<size_t>(q_len)*static_cast<size_t>(k_len)
             +static_cast<size_t>(q)*static_cast<size_t>(k_len)
             +static_cast<size_t>(k)]=
            ep[static_cast<size_t>(h)*static_cast<size_t>(num_buckets)+static_cast<size_t>(bucket)];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AccumulateRelativePositionBiasGradientHost(const CAIF_DeviceTensor &grad_output,
                                                CAIF_DeviceTensor &grad_embedding,
                                                uint32_t max_distance,
                                                bool bidirectional)
{
  try
  {
    const auto &sg=grad_output.Shape();
    const auto &se=grad_embedding.Shape();
    if(sg.size()!=3||se.size()!=2)
    {
      THROW_CAIFE("RelativePositionBiasBackward host: grad 3D, emb 2D");
    }
    const int num_heads=static_cast<int>(sg[0]);
    const int q_len=static_cast<int>(sg[1]);
    const int k_len=static_cast<int>(sg[2]);
    const int num_buckets=static_cast<int>(se[1]);
    const float *gp=CAIF_HostFpCast::HostFp32(grad_output,"RelativePositionBiasBackward");
    float *ep=CAIF_HostFpCast::HostFp32(grad_embedding,"RelativePositionBiasBackward");
    for(int h=0;h<num_heads;++h)
    {
      for(int q=0;q<q_len;++q)
      {
        for(int k=0;k<k_len;++k)
        {
          const int32_t bucket=RelativePositionBucket(k-q,
                                                     max_distance,
                                                     bidirectional,
                                                     static_cast<uint32_t>(num_buckets));
          ep[static_cast<size_t>(h)*static_cast<size_t>(num_buckets)+static_cast<size_t>(bucket)]+=
            gp[static_cast<size_t>(h)*static_cast<size_t>(q_len)*static_cast<size_t>(k_len)
               +static_cast<size_t>(q)*static_cast<size_t>(k_len)
               +static_cast<size_t>(k)];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Slice / Concat (last dimension)
//------------------------------------------------------------------------------


void CAIF_Ops::CastHost(const CAIF_DeviceTensor &input,
              CAIF_DeviceTensor &output,
              CAIF_RunContext &ctx)
{
  try
  {
    (void)ctx;
    if(input.Shape()!=output.Shape())
    {
      THROW_CAIFE("Cast host: input/output shape mismatch");
    }
    if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32||
       output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
    {
      THROW_CAIFE("Cast host: host backend only supports fp32<->fp32");
    }
    const float *ip=CAIF_HostFpCast::HostFp32(input,"Cast");
    float *op=CAIF_HostFpCast::HostFp32(output,"Cast");
    std::memcpy(op,ip,input.TotalElements()*sizeof(float));
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SliceLastDimHost(const CAIF_DeviceTensor &input,
                      CAIF_DeviceTensor &output,
                      uint32_t col_start)
{
  try
  {
    size_t in_rows=0;
    size_t in_cols=0;
    size_t out_rows=0;
    size_t out_cols=0;
    CAIF_HostSliceHelper::SliceDimsForLastDim(input,in_rows,in_cols);
    CAIF_HostSliceHelper::SliceDimsForLastDim(output,out_rows,out_cols);
    if(in_rows!=out_rows)
    {
      THROW_CAIFE("SliceLastDim host: row count mismatch");
    }
    if(col_start+out_cols>in_cols)
    {
      THROW_CAIFE("SliceLastDim host: slice out of bounds");
    }
    const float *ip=CAIF_HostFpCast::HostFp32(input,"SliceLastDim");
    float *op=CAIF_HostFpCast::HostFp32(output,"SliceLastDim");
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(in_rows);++r)
    {
      std::memcpy(op+static_cast<size_t>(r)*out_cols,
                  ip+static_cast<size_t>(r)*in_cols+col_start,
                  out_cols*sizeof(float));
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SliceLastDimBackwardHost(const CAIF_DeviceTensor &grad_output,
                              CAIF_DeviceTensor &grad_input,
                              uint32_t col_start)
{
  try
  {
    size_t out_rows=0;
    size_t out_cols=0;
    size_t in_rows=0;
    size_t in_cols=0;
    CAIF_HostSliceHelper::SliceDimsForLastDim(grad_output,out_rows,out_cols);
    CAIF_HostSliceHelper::SliceDimsForLastDim(grad_input,in_rows,in_cols);
    if(in_rows!=out_rows)
    {
      THROW_CAIFE("SliceLastDimBackward host: row count mismatch");
    }
    if(col_start+out_cols>in_cols)
    {
      THROW_CAIFE("SliceLastDimBackward host: slice out of bounds");
    }
    const float *gp=CAIF_HostFpCast::HostFp32(grad_output,"SliceLastDimBackward");
    float *ip=CAIF_HostFpCast::HostFp32(grad_input,"SliceLastDimBackward");
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(in_rows);++r)
    {
      for(size_t c=0;c<out_cols;++c)
      {
        ip[static_cast<size_t>(r)*in_cols+col_start+c]+=
          gp[static_cast<size_t>(r)*out_cols+c];
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ConcatLastDimHost(const CAIF_DeviceTensor &a,
                       const CAIF_DeviceTensor &b,
                       CAIF_DeviceTensor &output)
{
  try
  {
    size_t a_rows=0;
    size_t a_cols=0;
    size_t b_rows=0;
    size_t b_cols=0;
    size_t o_rows=0;
    size_t o_cols=0;
    CAIF_HostSliceHelper::SliceDimsForLastDim(a,a_rows,a_cols);
    CAIF_HostSliceHelper::SliceDimsForLastDim(b,b_rows,b_cols);
    CAIF_HostSliceHelper::SliceDimsForLastDim(output,o_rows,o_cols);
    if(a_rows!=b_rows||a_rows!=o_rows||a_cols+b_cols!=o_cols)
    {
      THROW_CAIFE("ConcatLastDim host: shape mismatch");
    }
    const float *ap=CAIF_HostFpCast::HostFp32(a,"ConcatLastDim");
    const float *bp=CAIF_HostFpCast::HostFp32(b,"ConcatLastDim");
    float *op=CAIF_HostFpCast::HostFp32(output,"ConcatLastDim");
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(a_rows);++r)
    {
      std::memcpy(op+static_cast<size_t>(r)*o_cols,
                  ap+static_cast<size_t>(r)*a_cols,
                  a_cols*sizeof(float));
      std::memcpy(op+static_cast<size_t>(r)*o_cols+a_cols,
                  bp+static_cast<size_t>(r)*b_cols,
                  b_cols*sizeof(float));
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Activations — forward
//------------------------------------------------------------------------------

void CAIF_Ops::ReLUHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      float v=ip[i];
      if(v<0.0f)
      {
        v=0.0f;
      }
      op[i]=v;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SigmoidHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=1.0f/(1.0f+std::exp(-ip[i]));
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::TanhHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=std::tanh(ip[i]);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SoftmaxHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const auto &s=input.Shape();
    if(s.size()<2)
    {
      THROW_CAIFE("Softmax host: tensor must have at least 2 dims");
    }
    const size_t last=s.back();
    size_t rows=1;
    for(size_t i=0;i+1<s.size();++i)
    {
      rows*=s[i];
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(rows);++r)
    {
      const float *row_in=ip+static_cast<size_t>(r)*last;
      float *row_out=op+static_cast<size_t>(r)*last;
      float max_val=row_in[0];
      for(size_t c=1;c<last;++c)
      {
        if(row_in[c]>max_val)
        {
          max_val=row_in[c];
        }
      }
      float sum=0.0f;
      for(size_t c=0;c<last;++c)
      {
        const float e=std::exp(row_in[c]-max_val);
        row_out[c]=e;
        sum+=e;
      }
      const float inv=1.0f/sum;
      for(size_t c=0;c<last;++c)
      {
        row_out[c]*=inv;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::LeakyReLUHost(const CAIF_DeviceTensor &input,
                   CAIF_DeviceTensor &output,
                   float alpha)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float v=ip[i];
      if(v>=0.0f)
      {
        op[i]=v;
      }
      else
      {
        op[i]=v*alpha;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ELUHost(const CAIF_DeviceTensor &input,
             CAIF_DeviceTensor &output,
             float alpha)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float v=ip[i];
      if(v>=0.0f)
      {
        op[i]=v;
      }
      else
      {
        op[i]=alpha*(std::exp(v)-1.0f);
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::GELUHost(const CAIF_DeviceTensor &input,
                        CAIF_DeviceTensor &output,
                        const CAIF_GELUApproximation::CAIF_GELUApproximation_e approx)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    if(approx==CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact)
    {
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        const float x=ip[i];
        op[i]=0.5f*x*(1.0f+std::erf(x*g_caif_gelu_inv_sqrt2));
      }
    }
    else
    {
      const float k=g_caif_gelu_sqrt_2_over_pi;
      const float c=g_caif_gelu_coeff;
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        const float x=ip[i];
        const float inner=k*(x+c*x*x*x);
        op[i]=0.5f*x*(1.0f+std::tanh(inner));
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SwishHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float x=ip[i];
      op[i]=x/(1.0f+std::exp(-x));
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Activations — backward
//------------------------------------------------------------------------------

void CAIF_Ops::ReLUBackwardHost(const CAIF_DeviceTensor &grad_output,
                      const CAIF_DeviceTensor &input,
                      CAIF_DeviceTensor &grad_input)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      if(ip[i]>0.0f)
      {
        op[i]=gp[i];
      }
      else
      {
        op[i]=0.0f;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SigmoidBackwardHost(const CAIF_DeviceTensor &grad_output,
                         const CAIF_DeviceTensor &output,
                         CAIF_DeviceTensor &grad_input)
{
  try
  {
    const size_t n=output.TotalElements();
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView yv(output);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *yp=yv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=gp[i]*yp[i]*(1.0f-yp[i]);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::TanhBackwardHost(const CAIF_DeviceTensor &grad_output,
                      const CAIF_DeviceTensor &output,
                      CAIF_DeviceTensor &grad_input)
{
  try
  {
    const size_t n=output.TotalElements();
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView yv(output);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *yp=yv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=gp[i]*(1.0f-yp[i]*yp[i]);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SoftmaxBackwardHost(const CAIF_DeviceTensor &grad_output,
                         const CAIF_DeviceTensor &output,
                         CAIF_DeviceTensor &grad_input)
{
  try
  {
    const auto &s=output.Shape();
    if(s.size()<2)
    {
      THROW_CAIFE("SoftmaxBackward host: tensor must have at least 2 dims");
    }
    const size_t last=s.back();
    size_t rows=1;
    for(size_t i=0;i+1<s.size();++i)
    {
      rows*=s[i];
    }
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView yv(output);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *yp=yv.Data();
    float *ip=ov.Data();
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(rows);++r)
    {
      const float *grow=gp+static_cast<size_t>(r)*last;
      const float *yrow=yp+static_cast<size_t>(r)*last;
      float *irow=ip+static_cast<size_t>(r)*last;
      float dot=0.0f;
      for(size_t c=0;c<last;++c)
      {
        dot+=grow[c]*yrow[c];
      }
      for(size_t c=0;c<last;++c)
      {
        irow[c]=yrow[c]*(grow[c]-dot);
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::LeakyReLUBackwardHost(const CAIF_DeviceTensor &grad_output,
                           const CAIF_DeviceTensor &input,
                           CAIF_DeviceTensor &grad_input,
                           float alpha)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      if(ip[i]>=0.0f)
      {
        op[i]=gp[i];
      }
      else
      {
        op[i]=gp[i]*alpha;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ELUBackwardHost(const CAIF_DeviceTensor &grad_output,
                     const CAIF_DeviceTensor &input,
                     const CAIF_DeviceTensor &output,
                     CAIF_DeviceTensor &grad_input,
                     float alpha)
{
  try
  {
    (void)output;
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      if(ip[i]>=0.0f)
      {
        op[i]=gp[i];
      }
      else
      {
        op[i]=gp[i]*alpha*std::exp(ip[i]);
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::GELUBackwardHost(const CAIF_DeviceTensor &grad_output,
                                const CAIF_DeviceTensor &input,
                                CAIF_DeviceTensor &grad_input,
                                const CAIF_GELUApproximation::CAIF_GELUApproximation_e approx)
{
  try
  {
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *ip=iv.Data();
    float *op=ov.Data();
    if(approx==CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact)
    {
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        const float x=ip[i];
        const float cdf=0.5f*(1.0f+std::erf(x*g_caif_gelu_inv_sqrt2));
        const float pdf=g_caif_gelu_inv_sqrt2pi*std::exp(-0.5f*x*x);
        op[i]=gp[i]*(cdf+x*pdf);
      }
    }
    else
    {
      const float k=g_caif_gelu_sqrt_2_over_pi;
      const float c=g_caif_gelu_coeff;
      #pragma omp parallel for
      for(int64_t i=0;i<static_cast<int64_t>(n);++i)
      {
        const float x=ip[i];
        const float inner=k*(x+c*x*x*x);
        const float th=std::tanh(inner);
        const float d_inner=k*(1.0f+3.0f*c*x*x);
        const float d=0.5f*(1.0f+th)+0.5f*x*(1.0f-th*th)*d_inner;
        op[i]=gp[i]*d;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SwishBackwardHost(const CAIF_DeviceTensor &grad_output,
                       const CAIF_DeviceTensor &input,
                       const CAIF_DeviceTensor &output,
                       CAIF_DeviceTensor &grad_input)
{
  try
  {
    (void)output;
    const size_t n=input.TotalElements();
    CAIF_HostFloatReadView gv(grad_output);
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(grad_input);
    const float *gp=gv.Data();
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float x=ip[i];
      const float sig=1.0f/(1.0f+std::exp(-x));
      op[i]=gp[i]*(sig+x*sig*(1.0f-sig));
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// Reductions / misc
//------------------------------------------------------------------------------

float CAIF_Ops::ReduceSumHost(const CAIF_DeviceTensor &tensor)
{
  try
  {
    const size_t n=tensor.TotalElements();
    CAIF_HostFloatReadView tv(tensor);
    const float *tp=tv.Data();
    float total=0.0f;
    #pragma omp parallel for reduction(+:total)
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      total+=tp[i];
    }
    return total;
  }
  CAIF_CATCH_BLOCK();
  return 0.0f;
}

float CAIF_Ops::ReduceMeanHost(const CAIF_DeviceTensor &tensor)
{
  try
  {
    const size_t n=tensor.TotalElements();
    if(n==0)
    {
      return 0.0f;
    }
    const float sum=ReduceSumHost(tensor);
    return sum/static_cast<float>(n);
  }
  CAIF_CATCH_BLOCK();
  return 0.0f;
}

void CAIF_Ops::MSELossHost(const CAIF_DeviceTensor &pred,
                 const CAIF_DeviceTensor &target,
                 CAIF_DeviceTensor &loss)
{
  try
  {
    const size_t n=pred.TotalElements();
    if(target.TotalElements()!=n||loss.TotalElements()!=1)
    {
      THROW_CAIFE("MSELoss host: shape mismatch");
    }
    CAIF_HostFloatReadView pv(pred);
    CAIF_HostFloatReadView tv(target);
    CAIF_HostFloatWriteView lv(loss);
    const float *pp=pv.Data();
    const float *tp=tv.Data();
    float *lp=lv.Data();
    float total=0.0f;
    #pragma omp parallel for reduction(+:total)
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float d=pp[i]-tp[i];
      total+=d*d;
    }
    lp[0]=total/static_cast<float>(n);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MSELossBackwardHost(const CAIF_DeviceTensor &pred,
                         const CAIF_DeviceTensor &target,
                         CAIF_DeviceTensor &grad)
{
  try
  {
    const size_t n=pred.TotalElements();
    if(target.TotalElements()!=n||grad.TotalElements()!=n)
    {
      THROW_CAIFE("MSELossBackward host: shape mismatch");
    }
    CAIF_HostFloatReadView pv(pred);
    CAIF_HostFloatReadView tv(target);
    CAIF_HostFloatWriteView gv(grad);
    const float *pp=pv.Data();
    const float *tp=tv.Data();
    float *gp=gv.Data();
    const float inv=2.0f/static_cast<float>(n);
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      gp[i]=inv*(pp[i]-tp[i]);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AdamUpdateHost(CAIF_DeviceTensor &param,
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
    const size_t n=param.TotalElements();
    if(grad.TotalElements()!=n||m.TotalElements()!=n||v.TotalElements()!=n)
    {
      THROW_CAIFE("AdamUpdate host: shape mismatch");
    }
    float *pp=CAIF_HostFpCast::HostFp32(param,"AdamUpdate");
    const float *gp=CAIF_HostFpCast::HostFp32(grad,"AdamUpdate");
    float *mp=CAIF_HostFpCast::HostFp32(m,"AdamUpdate");
    float *vp=CAIF_HostFpCast::HostFp32(v,"AdamUpdate");
    const float bc1=1.0f-std::pow(beta1,static_cast<float>(t));
    const float bc2=1.0f-std::pow(beta2,static_cast<float>(t));
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float g=gp[i]+weight_decay*pp[i];
      mp[i]=beta1*mp[i]+(1.0f-beta1)*g;
      vp[i]=beta2*vp[i]+(1.0f-beta2)*g*g;
      const float m_hat=mp[i]/bc1;
      const float v_hat=vp[i]/bc2;
      pp[i]-=lr*m_hat/(std::sqrt(v_hat)+epsilon);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SgdUpdateHost(CAIF_DeviceTensor &param,
                   const CAIF_DeviceTensor &grad,
                   float lr,
                   float weight_decay)
{
  try
  {
    const size_t n=param.TotalElements();
    if(grad.TotalElements()!=n)
    {
      THROW_CAIFE("SgdUpdate host: shape mismatch");
    }
    float *pp=CAIF_HostFpCast::HostFp32(param,"SgdUpdate");
    const float *gp=CAIF_HostFpCast::HostFp32(grad,"SgdUpdate");
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float g=gp[i]+weight_decay*pp[i];
      pp[i]-=lr*g;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MomentumUpdateHost(CAIF_DeviceTensor &param,
                        const CAIF_DeviceTensor &grad,
                        CAIF_DeviceTensor &velocity,
                        float lr,
                        float momentum,
                        float weight_decay)
{
  try
  {
    const size_t n=param.TotalElements();
    if(grad.TotalElements()!=n||velocity.TotalElements()!=n)
    {
      THROW_CAIFE("MomentumUpdate host: shape mismatch");
    }
    float *pp=CAIF_HostFpCast::HostFp32(param,"MomentumUpdate");
    const float *gp=CAIF_HostFpCast::HostFp32(grad,"MomentumUpdate");
    float *vp=CAIF_HostFpCast::HostFp32(velocity,"MomentumUpdate");
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float g=gp[i]+weight_decay*pp[i];
      vp[i]=momentum*vp[i]+g;
      pp[i]-=lr*vp[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::RmspropUpdateHost(CAIF_DeviceTensor &param,
                       const CAIF_DeviceTensor &grad,
                       CAIF_DeviceTensor &avg_sq,
                       float lr,
                       float alpha,
                       float epsilon,
                       float weight_decay)
{
  try
  {
    const size_t n=param.TotalElements();
    if(grad.TotalElements()!=n||avg_sq.TotalElements()!=n)
    {
      THROW_CAIFE("RmspropUpdate host: shape mismatch");
    }
    float *pp=CAIF_HostFpCast::HostFp32(param,"RmspropUpdate");
    const float *gp=CAIF_HostFpCast::HostFp32(grad,"RmspropUpdate");
    float *ap=CAIF_HostFpCast::HostFp32(avg_sq,"RmspropUpdate");
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float g=gp[i]+weight_decay*pp[i];
      ap[i]=alpha*ap[i]+(1.0f-alpha)*g*g;
      pp[i]-=lr*g/(std::sqrt(ap[i])+epsilon);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AdaGradUpdateHost(CAIF_DeviceTensor &param,
                       const CAIF_DeviceTensor &grad,
                       CAIF_DeviceTensor &accum,
                       float lr,
                       float epsilon,
                       float weight_decay)
{
  try
  {
    const size_t n=param.TotalElements();
    if(grad.TotalElements()!=n||accum.TotalElements()!=n)
    {
      THROW_CAIFE("AdaGradUpdate host: shape mismatch");
    }
    float *pp=CAIF_HostFpCast::HostFp32(param,"AdaGradUpdate");
    const float *gp=CAIF_HostFpCast::HostFp32(grad,"AdaGradUpdate");
    float *ap=CAIF_HostFpCast::HostFp32(accum,"AdaGradUpdate");
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      const float g=gp[i]+weight_decay*pp[i];
      ap[i]+=g*g;
      pp[i]-=lr*g/(std::sqrt(ap[i])+epsilon);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MultiplyHost(const CAIF_DeviceTensor &a,
                  const CAIF_DeviceTensor &b,
                  CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=a.TotalElements();
    if(b.TotalElements()!=n||output.TotalElements()!=n)
    {
      THROW_CAIFE("Multiply host: shape mismatch");
    }
    CAIF_HostFloatReadView av(a);
    CAIF_HostFloatReadView bv(b);
    CAIF_HostFloatWriteView ov(output);
    const float *ap=av.Data();
    const float *bp=bv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ap[i]*bp[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SiLUHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    SwishHost(input,output);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SiLUBackwardHost(const CAIF_DeviceTensor &input,
                      const CAIF_DeviceTensor &grad_output,
                      CAIF_DeviceTensor &grad_input)
{
  try
  {
    SwishBackwardHost(grad_output,input,input,grad_input);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddBiasHost(const CAIF_DeviceTensor &input,
                 const CAIF_DeviceTensor &bias,
                 CAIF_DeviceTensor &output)
{
  try
  {
    BiasAddHost(input,bias,output);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::AddScalarHost(const CAIF_DeviceTensor &input,
                   float scalar,
                   CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    if(output.TotalElements()!=n)
    {
      THROW_CAIFE("AddScalar host: shape mismatch");
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ip[i]+scalar;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SubtractHost(const CAIF_DeviceTensor &a,
                  const CAIF_DeviceTensor &b,
                  CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=a.TotalElements();
    if(b.TotalElements()!=n||output.TotalElements()!=n)
    {
      THROW_CAIFE("Subtract host: shape mismatch");
    }
    CAIF_HostFloatReadView av(a);
    CAIF_HostFloatReadView bv(b);
    CAIF_HostFloatWriteView ov(output);
    const float *ap=av.Data();
    const float *bp=bv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ap[i]-bp[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SubtractScalarHost(const CAIF_DeviceTensor &input,
                        float scalar,
                        CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    if(output.TotalElements()!=n)
    {
      THROW_CAIFE("SubtractScalar host: shape mismatch");
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ip[i]-scalar;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::DivideHost(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=a.TotalElements();
    if(b.TotalElements()!=n||output.TotalElements()!=n)
    {
      THROW_CAIFE("Divide host: shape mismatch");
    }
    CAIF_HostFloatReadView av(a);
    CAIF_HostFloatReadView bv(b);
    CAIF_HostFloatWriteView ov(output);
    const float *ap=av.Data();
    const float *bp=bv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ap[i]/bp[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::DivideScalarHost(const CAIF_DeviceTensor &input,
                      float scalar,
                      CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    if(output.TotalElements()!=n)
    {
      THROW_CAIFE("DivideScalar host: shape mismatch");
    }
    const float inv=1.0f/scalar;
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=ip[i]*inv;
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SqrtHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=input.TotalElements();
    if(output.TotalElements()!=n)
    {
      THROW_CAIFE("Sqrt host: shape mismatch");
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t i=0;i<static_cast<int64_t>(n);++i)
    {
      op[i]=std::sqrt(ip[i]);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SumAxisHost(const CAIF_DeviceTensor &input,
                 uint32_t axis,
                 CAIF_DeviceTensor &output)
{
  try
  {
    const auto &s=input.Shape();
    if(axis>=s.size())
    {
      THROW_CAIFE("SumAxis host: axis out of range");
    }
    size_t outer=1;
    for(size_t i=0;i<axis;++i)
    {
      outer*=s[i];
    }
    const size_t mid=s[axis];
    size_t inner=1;
    for(size_t i=axis+1;i<s.size();++i)
    {
      inner*=s[i];
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    const size_t total=outer*inner;
    std::memset(op,0,total*sizeof(float));
    for(size_t o=0;o<outer;++o)
    {
      for(size_t m=0;m<mid;++m)
      {
        for(size_t in=0;in<inner;++in)
        {
          op[o*inner+in]+=ip[(o*mid+m)*inner+in];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::SumHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    if(output.TotalElements()!=1)
    {
      THROW_CAIFE("Sum host: output must be scalar");
    }
    CAIF_HostFloatWriteView ov(output);
    float *op=ov.Data();
    op[0]=ReduceSumHost(input);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::LogSumExpHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const auto &s=input.Shape();
    if(s.size()<2)
    {
      THROW_CAIFE("LogSumExp host: tensor must have at least 2 dims");
    }
    const size_t last=s.back();
    size_t rows=1;
    for(size_t i=0;i+1<s.size();++i)
    {
      rows*=s[i];
    }
    CAIF_HostFloatReadView iv(input);
    CAIF_HostFloatWriteView ov(output);
    const float *ip=iv.Data();
    float *op=ov.Data();
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(rows);++r)
    {
      const float *row=ip+static_cast<size_t>(r)*last;
      float max_val=row[0];
      for(size_t c=1;c<last;++c)
      {
        if(row[c]>max_val)
        {
          max_val=row[c];
        }
      }
      float sum=0.0f;
      for(size_t c=0;c<last;++c)
      {
        sum+=std::exp(row[c]-max_val);
      }
      op[static_cast<size_t>(r)]=max_val+std::log(sum);
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::TopKHost(const CAIF_DeviceTensor &input,
              uint32_t k,
              CAIF_DeviceTensor &indices,
              CAIF_DeviceTensor &values)
{
  try
  {
    const auto &s=input.Shape();
    if(s.size()<2)
    {
      THROW_CAIFE("TopK host: input must have at least 2 dims");
    }
    const size_t last=s.back();
    size_t rows=1;
    for(size_t i=0;i+1<s.size();++i)
    {
      rows*=s[i];
    }
    if(k==0||static_cast<size_t>(k)>last)
    {
      THROW_CAIFE("TopK host: k out of range");
    }
    const float *ip=CAIF_HostFpCast::HostFp32(input,"TopK");
    float *vp=CAIF_HostFpCast::HostFp32(values,"TopK");
    if(indices.Dtype()!=CAIF_DataType::CAIF_DataType_e::Int32
       &&indices.Dtype()!=CAIF_DataType::CAIF_DataType_e::UInt32)
    {
      THROW_CAIFE("TopK host: indices must be Int32 or UInt32");
    }
    uint32_t *idx=static_cast<uint32_t*>(indices.DeviceDataRaw());
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(rows);++r)
    {
      const float *row=ip+static_cast<size_t>(r)*last;
      std::vector<uint32_t> order(last);
      for(size_t c=0;c<last;++c)
      {
        order[c]=static_cast<uint32_t>(c);
      }
      CAIF_TopKDescendingCompare cmp(row);
      std::partial_sort(order.begin(),
                        order.begin()+k,
                        order.end(),
                        cmp);
      for(uint32_t j=0;j<k;++j)
      {
        const size_t dst=static_cast<size_t>(r)*static_cast<size_t>(k)+static_cast<size_t>(j);
        idx[dst]=order[j];
        vp[dst]=row[order[j]];
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::NormalizeRowsHost(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
  try
  {
    const auto &s=input.Shape();
    if(s.size()<2)
    {
      THROW_CAIFE("NormalizeRows host: input must have at least 2 dims");
    }
    const size_t last=s.back();
    size_t rows=1;
    for(size_t i=0;i+1<s.size();++i)
    {
      rows*=s[i];
    }
    const float *ip=CAIF_HostFpCast::HostFp32(input,"NormalizeRows");
    float *op=CAIF_HostFpCast::HostFp32(output,"NormalizeRows");
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(rows);++r)
    {
      const float *in=ip+static_cast<size_t>(r)*last;
      float *out=op+static_cast<size_t>(r)*last;
      float sum=0.0f;
      for(size_t c=0;c<last;++c)
      {
        sum+=in[c];
      }
      const float inv=1.0f/(sum+g_caif_division_epsilon);
      for(size_t c=0;c<last;++c)
      {
        out[c]=in[c]*inv;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::NormalizeRowsBackwardTopKGatherHost(const CAIF_DeviceTensor &grad_w,
                                         const CAIF_DeviceTensor &probs,
                                         const CAIF_DeviceTensor &indices,
                                         CAIF_DeviceTensor &grad_p_topk)
{
  try
  {
    const auto &sg=grad_w.Shape();
    if(sg.size()!=2)
    {
      THROW_CAIFE("NormalizeRowsBackwardTopKGather host: grad_w must be 2D");
    }
    const int rows=static_cast<int>(sg[0]);
    const int k=static_cast<int>(sg[1]);
    if(probs.Shape().size()!=2||static_cast<int>(probs.Shape()[0])!=rows)
    {
      THROW_CAIFE("NormalizeRowsBackwardTopKGather host: probs shape mismatch");
    }
    const int num_experts=static_cast<int>(probs.Shape()[1]);
    const float *gw=CAIF_HostFpCast::HostFp32(grad_w,"NormalizeRowsBackwardTopKGather");
    const float *pp=CAIF_HostFpCast::HostFp32(probs,"NormalizeRowsBackwardTopKGather");
    float *gp=CAIF_HostFpCast::HostFp32(grad_p_topk,"NormalizeRowsBackwardTopKGather");
    const uint32_t *idx=static_cast<const uint32_t*>(indices.DeviceDataRaw());
    #pragma omp parallel for
    for(int r=0;r<rows;++r)
    {
      float sum_p_topk=0.0f;
      for(int j=0;j<k;++j)
      {
        sum_p_topk+=pp[static_cast<size_t>(r)*static_cast<size_t>(num_experts)
                       +static_cast<size_t>(idx[static_cast<size_t>(r)*static_cast<size_t>(k)
                                               +static_cast<size_t>(j)])];
      }
      const float inv=1.0f/(sum_p_topk+g_caif_division_epsilon);
      const float inv2=inv*inv;
      float dot=0.0f;
      for(int j=0;j<k;++j)
      {
        const float p=pp[static_cast<size_t>(r)*static_cast<size_t>(num_experts)
                         +static_cast<size_t>(idx[static_cast<size_t>(r)*static_cast<size_t>(k)
                                                 +static_cast<size_t>(j)])];
        dot+=gw[static_cast<size_t>(r)*static_cast<size_t>(k)+static_cast<size_t>(j)]*p;
      }
      for(int j=0;j<k;++j)
      {
        const float g=gw[static_cast<size_t>(r)*static_cast<size_t>(k)+static_cast<size_t>(j)];
        gp[static_cast<size_t>(r)*static_cast<size_t>(k)+static_cast<size_t>(j)]=g*inv-dot*inv2;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::GatherTopKValuesHost(const CAIF_DeviceTensor &scores,
                          const CAIF_DeviceTensor &indices,
                          CAIF_DeviceTensor &out)
{
  try
  {
    const std::vector<uint32_t> &s_shape=scores.Shape();
    const std::vector<uint32_t> &i_shape=indices.Shape();
    const std::vector<uint32_t> &o_shape=out.Shape();
    if(s_shape.size()!=2||i_shape.size()!=2||o_shape.size()!=2||
       i_shape!=o_shape||i_shape[0]!=s_shape[0])
    {
      THROW_CAIFE("GatherTopKValues host: scores must be [N x E], indices/out [N x K]");
    }
    const size_t rows=s_shape[0];
    const size_t num_experts=s_shape[1];
    const size_t k=i_shape[1];
    const float *sp=CAIF_HostFpCast::HostFp32(scores,"GatherTopKValues");
    float *op=CAIF_HostFpCast::HostFp32(out,"GatherTopKValues");
    const uint32_t *idx=static_cast<const uint32_t*>(indices.DeviceDataRaw());
    #pragma omp parallel for
    for(size_t r=0;r<rows;++r)
    {
      for(size_t j=0;j<k;++j)
      {
        const size_t e=static_cast<size_t>(idx[r*k+j]);
        op[r*k+j]=sp[r*num_experts+e];
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::ScatterAddHost(const CAIF_DeviceTensor &values,
                    const CAIF_DeviceTensor &indices,
                    CAIF_DeviceTensor &output)
{
  try
  {
    const size_t n=values.TotalElements();
    if(indices.TotalElements()!=n)
    {
      THROW_CAIFE("ScatterAdd host: values/indices length mismatch");
    }
    const float *vp=CAIF_HostFpCast::HostFp32(values,"ScatterAdd");
    float *op=CAIF_HostFpCast::HostFp32(output,"ScatterAdd");
    const uint32_t *idx=static_cast<const uint32_t*>(indices.DeviceDataRaw());
    for(size_t i=0;i<n;++i)
    {
      op[idx[i]]+=vp[i];
    }
  }
  CAIF_CATCH_BLOCK();
}

//------------------------------------------------------------------------------
// MoE — dispatch / combine on CPU
//------------------------------------------------------------------------------

void CAIF_Ops::MoEDispatchHost(const CAIF_DeviceTensor &input,
                     const CAIF_DeviceTensor &expert_indices,
                     uint32_t top_k,
                     const std::vector<uint32_t> &token_counts,
                     std::vector<CAIF_DeviceTensor> &expert_inputs)
{
  try
  {
    const auto &si=input.Shape();
    if(si.size()!=2)
    {
      THROW_CAIFE("MoEDispatch host: input must be 2D [num_tokens,model_dim]");
    }
    const size_t num_tokens=si[0];
    const size_t dim=si[1];
    const float *ip=CAIF_HostFpCast::HostFp32(input,"MoEDispatch");
    const uint32_t *idx=static_cast<const uint32_t*>(expert_indices.DeviceDataRaw());
    std::vector<size_t> cursors(expert_inputs.size(),0);
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t e=idx[tok*top_k+k];
        if(e>=expert_inputs.size())
        {
          continue;
        }
        if(cursors[e]>=token_counts[e])
        {
          continue;
        }
        float *eo=CAIF_HostFpCast::HostFp32(expert_inputs[e],"MoEDispatch");
        std::memcpy(eo+cursors[e]*dim,ip+tok*dim,dim*sizeof(float));
        cursors[e]++;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECombineHost(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                    const CAIF_DeviceTensor &expert_indices,
                    const CAIF_DeviceTensor &expert_weights,
                    uint32_t top_k,
                    const std::vector<uint32_t> &token_counts,
                    CAIF_DeviceTensor &output)
{
  try
  {
    const auto &so=output.Shape();
    if(so.size()!=2)
    {
      THROW_CAIFE("MoECombine host: output must be 2D");
    }
    const size_t num_tokens=so[0];
    const size_t dim=so[1];
    float *op=CAIF_HostFpCast::HostFp32(output,"MoECombine");
    std::memset(op,0,num_tokens*dim*sizeof(float));
    const uint32_t *idx=static_cast<const uint32_t*>(expert_indices.DeviceDataRaw());
    const float *wp=CAIF_HostFpCast::HostFp32(expert_weights,"MoECombine");
    std::vector<size_t> cursors(expert_outputs.size(),0);
    (void)token_counts;
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t e=idx[tok*top_k+k];
        if(e>=expert_outputs.size())
        {
          continue;
        }
        const float w=wp[tok*top_k+k];
        const float *eo=CAIF_HostFpCast::HostFp32(expert_outputs[e],"MoECombine");
        for(size_t d=0;d<dim;++d)
        {
          op[tok*dim+d]+=w*eo[cursors[e]*dim+d];
        }
        cursors[e]++;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECombineBackwardHost(const CAIF_DeviceTensor &grad_output,
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
    const auto &sg=grad_output.Shape();
    if(sg.size()!=2)
    {
      THROW_CAIFE("MoECombineBackward host: grad must be 2D");
    }
    const size_t num_tokens=sg[0];
    const size_t dim=sg[1];
    const float *gp=CAIF_HostFpCast::HostFp32(grad_output,"MoECombineBackward");
    const uint32_t *idx=static_cast<const uint32_t*>(expert_indices.DeviceDataRaw());
    const float *wp=CAIF_HostFpCast::HostFp32(expert_weights,"MoECombineBackward");
    float *gwp=CAIF_HostFpCast::HostFp32(grad_weights,"MoECombineBackward");
    for(size_t i=0;i<grad_expert_outputs.size();++i)
    {
      float *p=CAIF_HostFpCast::HostFp32(grad_expert_outputs[i],"MoECombineBackward");
      std::memset(p,0,grad_expert_outputs[i].TotalElements()*sizeof(float));
    }
    std::memset(gwp,0,num_tokens*top_k*sizeof(float));
    std::vector<size_t> cursors(expert_outputs.size(),0);
    (void)token_counts;
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t e=idx[tok*top_k+k];
        if(e>=expert_outputs.size())
        {
          continue;
        }
        const float w=wp[tok*top_k+k];
        const float *eo=CAIF_HostFpCast::HostFp32(expert_outputs[e],"MoECombineBackward");
        float *geo=CAIF_HostFpCast::HostFp32(grad_expert_outputs[e],"MoECombineBackward");
        float dot=0.0f;
        for(size_t d=0;d<dim;++d)
        {
          geo[cursors[e]*dim+d]+=w*gp[tok*dim+d];
          dot+=gp[tok*dim+d]*eo[cursors[e]*dim+d];
        }
        gwp[tok*top_k+k]=dot;
        cursors[e]++;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEDispatchBackwardHost(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
                             const CAIF_DeviceTensor &expert_indices,
                             uint32_t top_k,
                             const std::vector<uint32_t> &token_counts,
                             CAIF_DeviceTensor &grad_input)
{
  try
  {
    const auto &sg=grad_input.Shape();
    if(sg.size()!=2)
    {
      THROW_CAIFE("MoEDispatchBackward host: grad_input must be 2D");
    }
    const size_t num_tokens=sg[0];
    const size_t dim=sg[1];
    float *gp=CAIF_HostFpCast::HostFp32(grad_input,"MoEDispatchBackward");
    std::memset(gp,0,num_tokens*dim*sizeof(float));
    const uint32_t *idx=static_cast<const uint32_t*>(expert_indices.DeviceDataRaw());
    std::vector<size_t> cursors(grad_expert_inputs.size(),0);
    (void)token_counts;
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t e=idx[tok*top_k+k];
        if(e>=grad_expert_inputs.size())
        {
          continue;
        }
        const float *ge=CAIF_HostFpCast::HostFp32(grad_expert_inputs[e],"MoEDispatchBackward");
        for(size_t d=0;d<dim;++d)
        {
          gp[tok*dim+d]+=ge[cursors[e]*dim+d];
        }
        cursors[e]++;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoETopKGatingHost(const CAIF_DeviceTensor &router_logits,
                       uint32_t num_experts,
                       uint32_t top_k,
                       CAIF_DeviceTensor &expert_indices,
                       CAIF_DeviceTensor &expert_weights,
                       CAIF_DeviceTensor &router_probs)
{
  try
  {
    const auto &s=router_logits.Shape();
    if(s.size()!=2||s[1]!=num_experts)
    {
      THROW_CAIFE("MoETopKGating host: router_logits shape mismatch");
    }
    const size_t rows=s[0];
    const float *lp=CAIF_HostFpCast::HostFp32(router_logits,"MoETopKGating");
    float *probs=CAIF_HostFpCast::HostFp32(router_probs,"MoETopKGating");
    float *wp=CAIF_HostFpCast::HostFp32(expert_weights,"MoETopKGating");
    uint32_t *idx=static_cast<uint32_t*>(expert_indices.DeviceDataRaw());
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(rows);++r)
    {
      const float *row=lp+static_cast<size_t>(r)*num_experts;
      float *pr=probs+static_cast<size_t>(r)*num_experts;
      float max_val=row[0];
      for(uint32_t e=1;e<num_experts;++e)
      {
        if(row[e]>max_val)
        {
          max_val=row[e];
        }
      }
      float sum=0.0f;
      for(uint32_t e=0;e<num_experts;++e)
      {
        const float ev=std::exp(row[e]-max_val);
        pr[e]=ev;
        sum+=ev;
      }
      const float inv=1.0f/sum;
      for(uint32_t e=0;e<num_experts;++e)
      {
        pr[e]*=inv;
      }
      std::vector<uint32_t> order(num_experts);
      for(uint32_t e=0;e<num_experts;++e)
      {
        order[e]=e;
      }
      CAIF_TopKDescendingCompare cmp(pr);
      std::partial_sort(order.begin(),
                        order.begin()+top_k,
                        order.end(),
                        cmp);
      float sum_topk=0.0f;
      for(uint32_t k=0;k<top_k;++k)
      {
        sum_topk+=pr[order[k]];
      }
      const float inv_topk=1.0f/(sum_topk+g_caif_division_epsilon);
      for(uint32_t k=0;k<top_k;++k)
      {
        idx[static_cast<size_t>(r)*top_k+k]=order[k];
        wp[static_cast<size_t>(r)*top_k+k]=pr[order[k]]*inv_topk;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECountPerExpertHost(const CAIF_DeviceTensor &expert_indices,
                           uint32_t num_experts,
                           uint32_t top_k,
                           CAIF_DeviceTensor &expert_counts)
{
  try
  {
    const size_t total=expert_indices.TotalElements();
    (void)top_k;
    const uint32_t *idx=static_cast<const uint32_t*>(expert_indices.DeviceDataRaw());
    if(expert_counts.Dtype()!=CAIF_DataType::CAIF_DataType_e::Int32
       &&expert_counts.Dtype()!=CAIF_DataType::CAIF_DataType_e::UInt32)
    {
      THROW_CAIFE("MoECountPerExpert host: expert_counts must be Int32/UInt32");
    }
    uint32_t *counts=static_cast<uint32_t*>(expert_counts.DeviceDataRaw());
    std::memset(counts,0,num_experts*sizeof(uint32_t));
    for(size_t i=0;i<total;++i)
    {
      const uint32_t e=idx[i];
      if(e<num_experts)
      {
        counts[e]++;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEZLossGradAddHost(const CAIF_DeviceTensor &logsumexp_scaled,
                         const CAIF_DeviceTensor &probs,
                         CAIF_DeviceTensor &grad_logits)
{
  try
  {
    const auto &sg=grad_logits.Shape();
    if(sg.size()!=2)
    {
      THROW_CAIFE("MoEZLossGradAdd host: grad_logits must be 2D");
    }
    const size_t rows=sg[0];
    const size_t num_experts=sg[1];
    const float *lse=CAIF_HostFpCast::HostFp32(logsumexp_scaled,"MoEZLossGradAdd");
    const float *pp=CAIF_HostFpCast::HostFp32(probs,"MoEZLossGradAdd");
    float *gp=CAIF_HostFpCast::HostFp32(grad_logits,"MoEZLossGradAdd");
    #pragma omp parallel for
    for(int64_t r=0;r<static_cast<int64_t>(rows);++r)
    {
      const float scale=lse[static_cast<size_t>(r)];
      for(size_t e=0;e<num_experts;++e)
      {
        gp[static_cast<size_t>(r)*num_experts+e]+=scale*pp[static_cast<size_t>(r)*num_experts+e];
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEDispatchGPUHost(const CAIF_DeviceTensor &input,
                        const CAIF_DeviceTensor &expert_indices,
                        const CAIF_DeviceTensor &dispatch_map,
                        const CAIF_DeviceTensor &expert_offsets,
                        uint32_t top_k,
                        CAIF_DeviceTensor &expert_buffer)
{
  try
  {
    (void)expert_indices;
    (void)expert_offsets;
    const auto &si=input.Shape();
    const auto &seb=expert_buffer.Shape();
    if(si.size()!=2||seb.size()!=2)
    {
      THROW_CAIFE("MoEDispatchGPU host: input/expert_buffer must be 2D");
    }
    const size_t num_tokens=si[0];
    const size_t dim=si[1];
    const float *ip=CAIF_HostFpCast::HostFp32(input,"MoEDispatchGPU");
    float *bp=CAIF_HostFpCast::HostFp32(expert_buffer,"MoEDispatchGPU");
    const uint32_t *dmap=static_cast<const uint32_t*>(dispatch_map.DeviceDataRaw());
    std::memset(bp,0,expert_buffer.TotalElements()*sizeof(float));
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t slot=dmap[tok*top_k+k];
        if(slot==UINT32_MAX)
        {
          continue;
        }
        std::memcpy(bp+static_cast<size_t>(slot)*dim,ip+tok*dim,dim*sizeof(float));
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoECombineGPUHost(const CAIF_DeviceTensor &expert_buffer,
                       const CAIF_DeviceTensor &expert_indices,
                       const CAIF_DeviceTensor &expert_weights,
                       const CAIF_DeviceTensor &dispatch_map,
                       const CAIF_DeviceTensor &expert_offsets,
                       uint32_t top_k,
                       CAIF_DeviceTensor &output)
{
  try
  {
    (void)expert_indices;
    (void)expert_offsets;
    const auto &so=output.Shape();
    if(so.size()!=2)
    {
      THROW_CAIFE("MoECombineGPU host: output must be 2D");
    }
    const size_t num_tokens=so[0];
    const size_t dim=so[1];
    const float *bp=CAIF_HostFpCast::HostFp32(expert_buffer,"MoECombineGPU");
    const float *wp=CAIF_HostFpCast::HostFp32(expert_weights,"MoECombineGPU");
    float *op=CAIF_HostFpCast::HostFp32(output,"MoECombineGPU");
    std::memset(op,0,num_tokens*dim*sizeof(float));
    const uint32_t *dmap=static_cast<const uint32_t*>(dispatch_map.DeviceDataRaw());
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t slot=dmap[tok*top_k+k];
        if(slot==UINT32_MAX)
        {
          continue;
        }
        const float w=wp[tok*top_k+k];
        for(size_t d=0;d<dim;++d)
        {
          op[tok*dim+d]+=w*bp[static_cast<size_t>(slot)*dim+d];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

uint32_t CAIF_Ops::MoEBuildDispatchMapHost(const CAIF_DeviceTensor &expert_indices,
                                 uint32_t num_experts,
                                 uint32_t top_k,
                                 uint32_t capacity_per_expert,
                                 CAIF_DeviceTensor &dispatch_map,
                                 CAIF_DeviceTensor &expert_offsets)
{
  try
  {
    const size_t num_tokens=expert_indices.TotalElements()/top_k;
    const uint32_t *idx=static_cast<const uint32_t*>(expert_indices.DeviceDataRaw());
    uint32_t *dmap=static_cast<uint32_t*>(dispatch_map.DeviceDataRaw());
    uint32_t *offs=static_cast<uint32_t*>(expert_offsets.DeviceDataRaw());
    std::vector<uint32_t> counts(num_experts,0);
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t e=idx[tok*top_k+k];
        uint32_t slot=UINT32_MAX;
        if(e<num_experts&&counts[e]<capacity_per_expert)
        {
          slot=e*capacity_per_expert+counts[e];
          counts[e]++;
        }
        dmap[tok*top_k+k]=slot;
      }
    }
    uint32_t total=0;
    for(uint32_t e=0;e<num_experts;++e)
    {
      offs[e]=e*capacity_per_expert;
      total+=counts[e];
    }
    return total;
  }
  CAIF_CATCH_BLOCK();
  return 0;
}

void CAIF_Ops::MoECombineBackwardGPUHost(const CAIF_DeviceTensor &grad_output,
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
    (void)expert_indices;
    (void)expert_offsets;
    const auto &sg=grad_output.Shape();
    if(sg.size()!=2)
    {
      THROW_CAIFE("MoECombineBackwardGPU host: grad_output must be 2D");
    }
    const size_t num_tokens=sg[0];
    const size_t dim=sg[1];
    const float *gp=CAIF_HostFpCast::HostFp32(grad_output,"MoECombineBackwardGPU");
    const float *bp=CAIF_HostFpCast::HostFp32(expert_buffer,"MoECombineBackwardGPU");
    const float *wp=CAIF_HostFpCast::HostFp32(expert_weights,"MoECombineBackwardGPU");
    float *gb=CAIF_HostFpCast::HostFp32(grad_expert_buffer,"MoECombineBackwardGPU");
    float *gw=CAIF_HostFpCast::HostFp32(grad_weights,"MoECombineBackwardGPU");
    const uint32_t *dmap=static_cast<const uint32_t*>(dispatch_map.DeviceDataRaw());
    std::memset(gb,0,grad_expert_buffer.TotalElements()*sizeof(float));
    std::memset(gw,0,num_tokens*top_k*sizeof(float));
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t slot=dmap[tok*top_k+k];
        if(slot==UINT32_MAX)
        {
          continue;
        }
        const float w=wp[tok*top_k+k];
        float dot=0.0f;
        for(size_t d=0;d<dim;++d)
        {
          gb[static_cast<size_t>(slot)*dim+d]+=w*gp[tok*dim+d];
          dot+=gp[tok*dim+d]*bp[static_cast<size_t>(slot)*dim+d];
        }
        gw[tok*top_k+k]=dot;
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_Ops::MoEDispatchBackwardGPUHost(const CAIF_DeviceTensor &grad_expert_buffer,
                                const CAIF_DeviceTensor &expert_indices,
                                const CAIF_DeviceTensor &dispatch_map,
                                const CAIF_DeviceTensor &expert_offsets,
                                uint32_t top_k,
                                CAIF_DeviceTensor &grad_input)
{
  try
  {
    (void)expert_indices;
    (void)expert_offsets;
    const auto &sg=grad_input.Shape();
    if(sg.size()!=2)
    {
      THROW_CAIFE("MoEDispatchBackwardGPU host: grad_input must be 2D");
    }
    const size_t num_tokens=sg[0];
    const size_t dim=sg[1];
    const float *gb=CAIF_HostFpCast::HostFp32(grad_expert_buffer,"MoEDispatchBackwardGPU");
    float *gp=CAIF_HostFpCast::HostFp32(grad_input,"MoEDispatchBackwardGPU");
    std::memset(gp,0,num_tokens*dim*sizeof(float));
    const uint32_t *dmap=static_cast<const uint32_t*>(dispatch_map.DeviceDataRaw());
    for(size_t tok=0;tok<num_tokens;++tok)
    {
      for(uint32_t k=0;k<top_k;++k)
      {
        const uint32_t slot=dmap[tok*top_k+k];
        if(slot==UINT32_MAX)
        {
          continue;
        }
        for(size_t d=0;d<dim;++d)
        {
          gp[tok*dim+d]+=gb[static_cast<size_t>(slot)*dim+d];
        }
      }
    }
  }
  CAIF_CATCH_BLOCK();
}


}//end instance namespace
