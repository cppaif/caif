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
// Direct tests for CAIF_DeviceCrossAttention<ComputeT, StorageT>. Used in
// encdec stacks; previously only exercised indirectly via training tests.
// This file gives it a direct forward+backward shape/finiteness gate plus
// an fp16 / bf16 dtype-sweep verifying the templated cells.
//------------------------------------------------------------------------------

#include "caif_device_cross_attention.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

#ifdef USE_CAIF_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_crossattn_test_batch=2;
constexpr uint32_t g_caif_crossattn_test_dec_seq=4;
constexpr uint32_t g_caif_crossattn_test_enc_seq=5;
constexpr uint32_t g_caif_crossattn_test_dim=16;
constexpr uint32_t g_caif_crossattn_test_heads=4;
constexpr uint32_t g_caif_crossattn_test_kv_heads=4;
constexpr uint32_t g_caif_crossattn_test_head_dim=4;
constexpr uint32_t g_caif_crossattn_test_seed_init=7;
constexpr int32_t g_caif_crossattn_test_data_mod=29;
constexpr float g_caif_crossattn_test_data_scale=0.11f;
constexpr float g_caif_crossattn_test_data_shift=1.5f;
constexpr float g_caif_crossattn_test_data_range=0.35f;
constexpr float g_caif_crossattn_test_grad_fill=0.5f;

//------------------------------------------------------------------------------
// Shape/finiteness and dtype-sweep tests for CAIF_DeviceCrossAttention.
//------------------------------------------------------------------------------
class CAIF_CrossAttentionTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeData(const size_t n,const int32_t seed);
    static bool IsFinite(const float *p,const size_t n);
    static CAIF_DeviceTensor MakeFp32Device(const std::vector<float> &data,
                                             const std::vector<uint32_t> &shape,
                                             CAIF_CudaStream &stream);

    template<typename StorageT>
    static bool RunCrossAttentionDtype(const CAIF_DataType::CAIF_DataType_e storage_dt);

    static void TestCrossAttentionForwardShape();
    static void TestCrossAttentionBackward();
    static void TestCrossAttentionDtypeSweep();
};

std::vector<float> CAIF_CrossAttentionTests::MakeData(const size_t n,const int32_t seed)
{
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    const float t=static_cast<float>((static_cast<int32_t>(i)+seed)%g_caif_crossattn_test_data_mod)*
                  g_caif_crossattn_test_data_scale;
    v[i]=(t-g_caif_crossattn_test_data_shift)*g_caif_crossattn_test_data_range;
  }
  return v;
}

bool CAIF_CrossAttentionTests::IsFinite(const float *p,const size_t n)
{
  for(size_t i=0;i<n;++i)
  {
    if(std::isfinite(p[i])==false)
    {
      return false;
    }
  }
  return true;
}

CAIF_DeviceTensor CAIF_CrossAttentionTests::MakeFp32Device(const std::vector<float> &data,
                                                             const std::vector<uint32_t> &shape,
                                                             CAIF_CudaStream &stream)
{
  return CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
}

template<typename StorageT>
bool CAIF_CrossAttentionTests::RunCrossAttentionDtype(
  const CAIF_DataType::CAIF_DataType_e storage_dt)
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  ctx.SetTraining(true);

  CAIF_DeviceCrossAttentionConfig cfg=
    {g_caif_crossattn_test_dim,
     g_caif_crossattn_test_dim,
     g_caif_crossattn_test_heads,
     g_caif_crossattn_test_kv_heads,
     g_caif_crossattn_test_head_dim};
  CAIF_DeviceCrossAttention<float,StorageT> layer(cfg,stream);
  layer.InitializeWeights(g_caif_crossattn_test_seed_init+2);

  const std::vector<float> dec_host=
    MakeData(g_caif_crossattn_test_batch*g_caif_crossattn_test_dec_seq*
             g_caif_crossattn_test_dim,31);
  const std::vector<float> enc_host=
    MakeData(g_caif_crossattn_test_batch*g_caif_crossattn_test_enc_seq*
             g_caif_crossattn_test_dim,32);
  CAIF_DeviceTensor dec_fp32=
    MakeFp32Device(dec_host,
                   {g_caif_crossattn_test_batch,
                    g_caif_crossattn_test_dec_seq,
                    g_caif_crossattn_test_dim},
                   stream);
  CAIF_DeviceTensor enc_fp32=
    MakeFp32Device(enc_host,
                   {g_caif_crossattn_test_batch,
                    g_caif_crossattn_test_enc_seq,
                    g_caif_crossattn_test_dim},
                   stream);
  CAIF_DeviceTensor dec=dec_fp32.To(storage_dt);
  CAIF_DeviceTensor enc=enc_fp32.To(storage_dt);

  CAIF_DeviceTensor out=layer.ForwardCross(dec,enc,ctx);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_HostTensor host_out=out_fp32.ToHost();

  return out.Shape().size()==3 &&
         out.Shape()[0]==g_caif_crossattn_test_batch &&
         out.Shape()[1]==g_caif_crossattn_test_dec_seq &&
         out.Shape()[2]==g_caif_crossattn_test_dim &&
         IsFinite(host_out.Data(),host_out.TotalElements());
}

void CAIF_CrossAttentionTests::TestCrossAttentionForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    CAIF_DeviceCrossAttentionConfig cfg=
      {g_caif_crossattn_test_dim,
       g_caif_crossattn_test_dim,
       g_caif_crossattn_test_heads,
       g_caif_crossattn_test_kv_heads,
       g_caif_crossattn_test_head_dim};
    CAIF_DeviceCrossAttention<float,float> layer(cfg,stream);
    layer.InitializeWeights(g_caif_crossattn_test_seed_init);

    const std::vector<float> dec_host=
      MakeData(g_caif_crossattn_test_batch*g_caif_crossattn_test_dec_seq*
               g_caif_crossattn_test_dim,11);
    const std::vector<float> enc_host=
      MakeData(g_caif_crossattn_test_batch*g_caif_crossattn_test_enc_seq*
               g_caif_crossattn_test_dim,12);
    CAIF_DeviceTensor dec=
      MakeFp32Device(dec_host,
                     {g_caif_crossattn_test_batch,
                      g_caif_crossattn_test_dec_seq,
                      g_caif_crossattn_test_dim},
                     stream);
    CAIF_DeviceTensor enc=
      MakeFp32Device(enc_host,
                     {g_caif_crossattn_test_batch,
                      g_caif_crossattn_test_enc_seq,
                      g_caif_crossattn_test_dim},
                     stream);

    CAIF_DeviceTensor out=layer.ForwardCross(dec,enc,ctx);
    CAIF_HostTensor host_out=out.ToHost();

    bool ok=out.Shape().size()==3 &&
            out.Shape()[0]==g_caif_crossattn_test_batch &&
            out.Shape()[1]==g_caif_crossattn_test_dec_seq &&
            out.Shape()[2]==g_caif_crossattn_test_dim;
    if(ok==true)
    {
      ok=IsFinite(host_out.Data(),host_out.TotalElements());
    }
    CAIF_TestHarness::Report("CrossAttention::Forward fp32 shape+finite",ok);
  }
  CAIF_TEST_CATCH_BLOCK("CrossAttention::Forward fp32")
}

void CAIF_CrossAttentionTests::TestCrossAttentionBackward()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    CAIF_DeviceCrossAttentionConfig cfg=
      {g_caif_crossattn_test_dim,
       g_caif_crossattn_test_dim,
       g_caif_crossattn_test_heads,
       g_caif_crossattn_test_kv_heads,
       g_caif_crossattn_test_head_dim};
    CAIF_DeviceCrossAttention<float,float> layer(cfg,stream);
    layer.InitializeWeights(g_caif_crossattn_test_seed_init+1);

    const std::vector<float> dec_host=
      MakeData(g_caif_crossattn_test_batch*g_caif_crossattn_test_dec_seq*
               g_caif_crossattn_test_dim,21);
    const std::vector<float> enc_host=
      MakeData(g_caif_crossattn_test_batch*g_caif_crossattn_test_enc_seq*
               g_caif_crossattn_test_dim,22);
    CAIF_DeviceTensor dec=
      MakeFp32Device(dec_host,
                     {g_caif_crossattn_test_batch,
                      g_caif_crossattn_test_dec_seq,
                      g_caif_crossattn_test_dim},
                     stream);
    CAIF_DeviceTensor enc=
      MakeFp32Device(enc_host,
                     {g_caif_crossattn_test_batch,
                      g_caif_crossattn_test_enc_seq,
                      g_caif_crossattn_test_dim},
                     stream);

    layer.ZeroGradients();
    CAIF_DeviceTensor out=layer.ForwardCross(dec,enc,ctx);

    std::vector<float> grad_host(g_caif_crossattn_test_batch*g_caif_crossattn_test_dec_seq*
                                 g_caif_crossattn_test_dim,g_caif_crossattn_test_grad_fill);
    CAIF_DeviceTensor grad_out=
      MakeFp32Device(grad_host,
                     {g_caif_crossattn_test_batch,
                      g_caif_crossattn_test_dec_seq,
                      g_caif_crossattn_test_dim},
                     stream);
    CAIF_DeviceTensor grad_enc=
      CAIF_DeviceTensor::Uninitialized({g_caif_crossattn_test_batch,
                                        g_caif_crossattn_test_enc_seq,
                                        g_caif_crossattn_test_dim},
                                       stream,
                                       CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_DeviceTensor grad_dec=layer.BackwardCross(grad_out,grad_enc,ctx);

    CAIF_HostTensor host_gd=grad_dec.ToHost();
    CAIF_HostTensor host_ge=grad_enc.ToHost();

    bool ok=
      grad_dec.Shape()==std::vector<uint32_t>{g_caif_crossattn_test_batch,
                                              g_caif_crossattn_test_dec_seq,
                                              g_caif_crossattn_test_dim} &&
      grad_enc.Shape()==std::vector<uint32_t>{g_caif_crossattn_test_batch,
                                              g_caif_crossattn_test_enc_seq,
                                              g_caif_crossattn_test_dim};
    if(ok==true)
    {
      ok=IsFinite(host_gd.Data(),host_gd.TotalElements()) &&
         IsFinite(host_ge.Data(),host_ge.TotalElements());
    }
    CAIF_TestHarness::Report("CrossAttention::Backward fp32 shape+finite",ok);
  }
  CAIF_TEST_CATCH_BLOCK("CrossAttention::Backward fp32")
}

void CAIF_CrossAttentionTests::TestCrossAttentionDtypeSweep()
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    CAIF_TestHarness::Report("CrossAttention::Forward fp16 storage",
                             RunCrossAttentionDtype<__half>(Dtype_e::Float16));
    CAIF_TestHarness::Report("CrossAttention::Forward bf16 storage",
                             RunCrossAttentionDtype<__nv_bfloat16>(Dtype_e::BFloat16));
  }
  CAIF_TEST_CATCH_BLOCK("CrossAttention::dtype-sweep")
}

void CAIF_CrossAttentionTests::RunAll()
{
  ISE_Out::Out()<<"Cross-Attention Tests\n";
  ISE_Out::Out()<<"=====================\n";
  CAIF_TestHarness::Reset();
  TestCrossAttentionForwardShape();
  TestCrossAttentionBackward();
  TestCrossAttentionDtypeSweep();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_CrossAttentionTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"USE_CAIF_CUDA off — cross-attention tests skipped."
                <<"\n";
  return 0;
#endif
}
