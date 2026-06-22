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
// Prefix-LM mask tests
//
// Covers:
//   - Direct kernel: launch_prefix_mask_fill and launch_prefix_mask_grad
//   - Flash path: MHA forward/backward with SetPrefixLengths
//   - Equivalence: prefix_len=0 degenerates to pure causal
//   - Equivalence: prefix_len=seq_len degenerates to no mask (full attention
//     over the prefix, causal beyond — since prefix covers all tokens,
//     every position is bidirectional)
//   - Heterogeneous batch: different prefix_len per row
//   - ClearPrefixLengths reverts to pure causal
//------------------------------------------------------------------------------

#include "caif_device_multi_head_attention.h"
#include "caif_test_harness.h"
#include "caif_cuda_kernels_attention_support.cuh"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_constants.h"
#include "caif_cpu_reference/caif_cpu_softmax.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr float g_caif_prefix_test_mask_value=-1e9f;
constexpr float g_caif_prefix_test_mask_tol=1.0f;
constexpr float g_caif_prefix_test_unmask_tol=1e-5f;
constexpr float g_caif_prefix_test_fill_tol=5e-4f;
constexpr float g_caif_prefix_test_hetero_tol=2e-3f;
constexpr float g_caif_prefix_test_clear_tol=5e-4f;
constexpr float g_caif_prefix_test_scores_init=1.0f;
constexpr float g_caif_prefix_test_grad_init=0.7f;
constexpr uint32_t g_caif_prefix_test_kernel_batch=2;
constexpr uint32_t g_caif_prefix_test_kernel_heads=2;
constexpr uint32_t g_caif_prefix_test_kernel_seq=4;
constexpr uint32_t g_caif_prefix_test_grad_batch=1;
constexpr uint32_t g_caif_prefix_test_grad_heads=1;
constexpr uint32_t g_caif_prefix_test_grad_seq=4;
constexpr uint32_t g_caif_prefix_test_flash_batch=2;
constexpr uint32_t g_caif_prefix_test_flash_seq=8;
constexpr uint32_t g_caif_prefix_test_flash_dim=128;
constexpr uint32_t g_caif_prefix_test_flash_heads=4;
constexpr uint32_t g_caif_prefix_test_hetero_batch=3;
constexpr uint32_t g_caif_prefix_test_hetero_seq=8;
constexpr uint32_t g_caif_prefix_test_hetero_dim=128;
constexpr uint32_t g_caif_prefix_test_hetero_heads=4;
constexpr uint32_t g_caif_prefix_test_bwd_batch=2;
constexpr uint32_t g_caif_prefix_test_bwd_seq=16;
constexpr uint32_t g_caif_prefix_test_bwd_dim=128;
constexpr uint32_t g_caif_prefix_test_bwd_heads=4;
constexpr uint32_t g_caif_prefix_test_clear_batch=1;
constexpr uint32_t g_caif_prefix_test_clear_seq=8;
constexpr uint32_t g_caif_prefix_test_clear_dim=64;
constexpr uint32_t g_caif_prefix_test_clear_heads=2;

//------------------------------------------------------------------------------
// Prefix-LM mask correctness tests.
//------------------------------------------------------------------------------
class CAIF_PrefixLMTests
{
  public:
    static void RunAll();

  protected:

  private:
    static bool FloatClose(float a,float b,float tol);

    static CAIF_DeviceTensor MakePrefixTensor(const std::vector<int32_t> &lens,
                                              CAIF_CudaStream &stream);

    static CAIF_DeviceMultiHeadAttentionConfig
    MakeConfig(uint32_t dim,uint32_t num_heads,bool causal);

    // CPU MHA with optional per-row prefix-LM mask.
    // prefix_lens: length batch, entry b = prefix length for row b. If empty,
    // pure causal is applied when causal==true.
    static void CpuMHAPrefix(const float *input,
                              const float *w_q,
                              const float *w_k,
                              const float *w_v,
                              const float *w_o,
                              float *output,
                              int batch,
                              int seq_len,
                              int dim,
                              int num_heads,
                              int head_dim,
                              bool causal,
                              const std::vector<int32_t> &prefix_lens);

    static void TestKernelPrefixMaskFill();
    static void TestKernelPrefixMaskGrad();
    static void TestFlashPrefixZeroEqualsCausal();
    static void TestFlashPrefixHeterogeneousVsCPU();
    static void TestFlashPrefixBackwardNoNaN();
    static void TestClearRevertsToCausal();
};

bool CAIF_PrefixLMTests::FloatClose(const float a,const float b,const float tol)
{
  return std::fabs(a-b)<tol;
}

CAIF_DeviceTensor CAIF_PrefixLMTests::MakePrefixTensor(const std::vector<int32_t> &lens,
                                                        CAIF_CudaStream &stream)
{
  std::vector<uint32_t> shape={static_cast<uint32_t>(lens.size())};
  // Prefix lengths are unsigned —
  // kernel signatures take `const uint32_t *`. Convert input int32 to
  // uint32 for upload (prefix lengths are always non-negative).
  std::vector<uint32_t> ulens(lens.size());
  for(size_t i=0;i<lens.size();++i)
  {
    ulens[i]=static_cast<uint32_t>(lens[i]);
  }
  return CAIF_DeviceTensor::FromHostRaw(ulens.data(),
                                        shape,
                                        stream,
                                        CAIF_DataType::CAIF_DataType_e::UInt32);
}

CAIF_DeviceMultiHeadAttentionConfig
CAIF_PrefixLMTests::MakeConfig(const uint32_t dim,
                                const uint32_t num_heads,
                                const bool causal)
{
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             num_heads,
                                             num_heads,
                                             dim/num_heads,
                                             causal,
                                             false,
                                             g_caif_rope_default_base,
                                             0.0f);
  return config;
}

void CAIF_PrefixLMTests::CpuMHAPrefix(const float *input,
                                       const float *w_q,
                                       const float *w_k,
                                       const float *w_v,
                                       const float *w_o,
                                       float *output,
                                       const int batch,
                                       const int seq_len,
                                       const int dim,
                                       const int num_heads,
                                       const int head_dim,
                                       const bool causal,
                                       const std::vector<int32_t> &prefix_lens)
{
  const int bs=batch*seq_len;
  const int qk_dim=num_heads*head_dim;

  std::vector<float> q(static_cast<size_t>(bs*qk_dim));
  std::vector<float> k(static_cast<size_t>(bs*qk_dim));
  std::vector<float> v(static_cast<size_t>(bs*qk_dim));
  for(int i=0;i<bs;++i)
  {
    for(int j=0;j<qk_dim;++j)
    {
      float sq=0.0f;
      float sk=0.0f;
      float sv=0.0f;
      for(int d=0;d<dim;++d)
      {
        sq+=input[i*dim+d]*w_q[d*qk_dim+j];
        sk+=input[i*dim+d]*w_k[d*qk_dim+j];
        sv+=input[i*dim+d]*w_v[d*qk_dim+j];
      }
      q[static_cast<size_t>(i*qk_dim+j)]=sq;
      k[static_cast<size_t>(i*qk_dim+j)]=sk;
      v[static_cast<size_t>(i*qk_dim+j)]=sv;
    }
  }

  std::vector<float> concat(static_cast<size_t>(bs*qk_dim),0.0f);
  const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));

  for(int b=0;b<batch;++b)
  {
    int pfx=0;
    if(prefix_lens.empty()==false)
    {
      pfx=prefix_lens[static_cast<size_t>(b)];
    }
    for(int h=0;h<num_heads;++h)
    {
      std::vector<float> qh(static_cast<size_t>(seq_len*head_dim));
      std::vector<float> kh(static_cast<size_t>(seq_len*head_dim));
      std::vector<float> vh(static_cast<size_t>(seq_len*head_dim));
      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          const int idx=(b*seq_len+s)*qk_dim+h*head_dim+d;
          qh[static_cast<size_t>(s*head_dim+d)]=q[static_cast<size_t>(idx)];
          kh[static_cast<size_t>(s*head_dim+d)]=k[static_cast<size_t>(idx)];
          vh[static_cast<size_t>(s*head_dim+d)]=v[static_cast<size_t>(idx)];
        }
      }

      std::vector<float> scores(static_cast<size_t>(seq_len*seq_len),0.0f);
      for(int i=0;i<seq_len;++i)
      {
        for(int j=0;j<seq_len;++j)
        {
          float s=0.0f;
          for(int d=0;d<head_dim;++d)
          {
            s+=qh[static_cast<size_t>(i*head_dim+d)]*kh[static_cast<size_t>(j*head_dim+d)];
          }
          scores[static_cast<size_t>(i*seq_len+j)]=s*scale;
        }
      }

      // Mask: prefix-LM if prefix_lens provided, else pure causal if requested.
      for(int i=0;i<seq_len;++i)
      {
        for(int j=0;j<seq_len;++j)
        {
          bool masked=false;
          if(prefix_lens.empty()==false)
          {
            if(j>i && j>=pfx)
            {
              masked=true;
            }
          }
          else if(causal==true)
          {
            if(j>i)
            {
              masked=true;
            }
          }
          if(masked==true)
          {
            scores[static_cast<size_t>(i*seq_len+j)]=g_caif_prefix_test_mask_value;
          }
        }
      }

      for(int i=0;i<seq_len;++i)
      {
        CAIF_CpuSoftmax::ApplyRow(scores.data()+i*seq_len,seq_len);
      }

      std::vector<float> ctx_out(static_cast<size_t>(seq_len*head_dim),0.0f);
      for(int i=0;i<seq_len;++i)
      {
        for(int d=0;d<head_dim;++d)
        {
          float s=0.0f;
          for(int j=0;j<seq_len;++j)
          {
            s+=scores[static_cast<size_t>(i*seq_len+j)]
               *vh[static_cast<size_t>(j*head_dim+d)];
          }
          ctx_out[static_cast<size_t>(i*head_dim+d)]=s;
        }
      }

      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          concat[static_cast<size_t>((b*seq_len+s)*qk_dim+h*head_dim+d)]=
            ctx_out[static_cast<size_t>(s*head_dim+d)];
        }
      }
    }
  }

  for(int i=0;i<bs;++i)
  {
    for(int j=0;j<dim;++j)
    {
      float s=0.0f;
      for(int p=0;p<qk_dim;++p)
      {
        s+=concat[static_cast<size_t>(i*qk_dim+p)]*w_o[static_cast<size_t>(p*dim+j)];
      }
      output[i*dim+j]=s;
    }
  }
}

//------------------------------------------------------------------------------
// Direct kernel: launch_prefix_mask_fill
//------------------------------------------------------------------------------
void CAIF_PrefixLMTests::TestKernelPrefixMaskFill()
{
  try
  {
    const int batch=static_cast<int>(g_caif_prefix_test_kernel_batch);
    const int num_heads=static_cast<int>(g_caif_prefix_test_kernel_heads);
    const int seq_len=static_cast<int>(g_caif_prefix_test_kernel_seq);
    const int bh=batch*num_heads;

    CAIF_CudaStream stream;

    // Initial scores: all 1.0f so masked cells become visibly -1e9 post-kernel.
    std::vector<float> host_scores(static_cast<size_t>(bh*seq_len*seq_len),
                                   g_caif_prefix_test_scores_init);
    CAIF_DeviceTensor scores=CAIF_DeviceTensor::FromHostData(
      host_scores.data(),
      {static_cast<uint32_t>(bh),
       static_cast<uint32_t>(seq_len),
       static_cast<uint32_t>(seq_len)},
      stream);

    std::vector<int32_t> prefix_lens={1,3};
    CAIF_DeviceTensor prefix=MakePrefixTensor(prefix_lens,stream);

    launch_prefix_mask_fill(scores.DevicePtr<float>(),
                            prefix.DevicePtr<uint32_t>(),
                            batch,
                            num_heads,
                            seq_len,
                            stream.Handle());
    stream.Synchronize();

    CAIF_HostTensor out=scores.ToHost();
    bool passed=true;
    for(int b=0;b<batch;++b)
    {
      const int pfx=prefix_lens[static_cast<size_t>(b)];
      for(int h=0;h<num_heads;++h)
      {
        for(int r=0;r<seq_len;++r)
        {
          for(int c=0;c<seq_len;++c)
          {
            const int idx=((b*num_heads+h)*seq_len+r)*seq_len+c;
            const float got=out.Data()[static_cast<size_t>(idx)];
            const bool should_mask=(c>r && c>=pfx);
            if(should_mask==true)
            {
              if(FloatClose(got,g_caif_prefix_test_mask_value,
                            g_caif_prefix_test_mask_tol)==false)
              {
                ISE_Out::Out()<<"  b="
                              <<b
                              <<" h="
                              <<h
                              <<" r="
                              <<r
                              <<" c="
                              <<c
                              <<" expected masked got "
                              <<got
                              <<"\n";
                passed=false;
              }
            }
            else
            {
              if(FloatClose(got,g_caif_prefix_test_scores_init,
                            g_caif_prefix_test_unmask_tol)==false)
              {
                ISE_Out::Out()<<"  b="
                              <<b
                              <<" h="
                              <<h
                              <<" r="
                              <<r
                              <<" c="
                              <<c
                              <<" expected unmodified got "
                              <<got
                              <<"\n";
                passed=false;
              }
            }
          }
        }
      }
    }
    CAIF_TestHarness::Report("Prefix::KernelMaskFill",passed);
  }
  CAIF_TEST_CATCH_BLOCK("Prefix::KernelMaskFill")
}

//------------------------------------------------------------------------------
// Direct kernel: launch_prefix_mask_grad zeros masked positions
//------------------------------------------------------------------------------
void CAIF_PrefixLMTests::TestKernelPrefixMaskGrad()
{
  try
  {
    const int batch=static_cast<int>(g_caif_prefix_test_grad_batch);
    const int num_heads=static_cast<int>(g_caif_prefix_test_grad_heads);
    const int seq_len=static_cast<int>(g_caif_prefix_test_grad_seq);
    const int bh=batch*num_heads;

    CAIF_CudaStream stream;
    std::vector<float> host_grad(static_cast<size_t>(bh*seq_len*seq_len),
                                 g_caif_prefix_test_grad_init);
    CAIF_DeviceTensor grad=CAIF_DeviceTensor::FromHostData(
      host_grad.data(),
      {static_cast<uint32_t>(bh),
       static_cast<uint32_t>(seq_len),
       static_cast<uint32_t>(seq_len)},
      stream);

    std::vector<int32_t> prefix_lens={2};
    CAIF_DeviceTensor prefix=MakePrefixTensor(prefix_lens,stream);

    launch_prefix_mask_grad(grad.DevicePtr<float>(),
                            prefix.DevicePtr<uint32_t>(),
                            batch,
                            num_heads,
                            seq_len,
                            stream.Handle());
    stream.Synchronize();

    CAIF_HostTensor out=grad.ToHost();
    bool passed=true;
    const int pfx=prefix_lens[0];
    for(int r=0;r<seq_len;++r)
    {
      for(int c=0;c<seq_len;++c)
      {
        const float got=out.Data()[static_cast<size_t>(r*seq_len+c)];
        const bool should_zero=(c>r && c>=pfx);
        float expected=0.0f;
        if(should_zero==false)
        {
          expected=g_caif_prefix_test_grad_init;
        }
        if(FloatClose(got,expected,g_caif_prefix_test_unmask_tol)==false)
        {
          ISE_Out::Out()<<"  r="
                        <<r
                        <<" c="
                        <<c
                        <<" expected "
                        <<expected
                        <<" got "
                        <<got
                        <<"\n";
          passed=false;
        }
      }
    }
    CAIF_TestHarness::Report("Prefix::KernelMaskGrad",passed);
  }
  CAIF_TEST_CATCH_BLOCK("Prefix::KernelMaskGrad")
}

//------------------------------------------------------------------------------
// Flash path: prefix_len=0 for all rows must match pure causal MHA bit-for-bit
// within fp32 tolerance.
//------------------------------------------------------------------------------
void CAIF_PrefixLMTests::TestFlashPrefixZeroEqualsCausal()
{
  try
  {
    const uint32_t batch=g_caif_prefix_test_flash_batch;
    const uint32_t seq_len=g_caif_prefix_test_flash_seq;
    const uint32_t dim=g_caif_prefix_test_flash_dim;
    const uint32_t num_heads=g_caif_prefix_test_flash_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceMultiHeadAttention<float,float> mha(MakeConfig(dim,num_heads,true),stream);

    std::vector<float> host_input(static_cast<size_t>(batch*seq_len*dim));
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%37)*0.013f-0.2f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    // Pure causal (no prefix)
    CAIF_DeviceTensor out_causal=mha.Forward(input,ctx);
    CAIF_HostTensor h_causal=out_causal.ToHost();

    // With prefix_lens=[0,0] — should be identical to pure causal
    std::vector<int32_t> zeros={0,0};
    CAIF_DeviceTensor prefix=MakePrefixTensor(zeros,stream);
    ctx.SetPrefixLengths(prefix);

    CAIF_DeviceTensor out_prefix=mha.Forward(input,ctx);
    CAIF_HostTensor h_prefix=out_prefix.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_causal.TotalElements();++i)
    {
      if(FloatClose(h_causal.Data()[i],h_prefix.Data()[i],g_caif_prefix_test_fill_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": causal="
                      <<h_causal.Data()[i]
                      <<" prefix0="
                      <<h_prefix.Data()[i]
                      <<"\n";
        passed=false;
        break;
      }
    }
    ctx.ClearPrefixLengths();
    CAIF_TestHarness::Report("Prefix::FlashZeroEqualsCausal",passed);
  }
  CAIF_TEST_CATCH_BLOCK("Prefix::FlashZeroEqualsCausal")
}

//------------------------------------------------------------------------------
// Flash path: heterogeneous prefix lengths, compare against CPU reference
//------------------------------------------------------------------------------
void CAIF_PrefixLMTests::TestFlashPrefixHeterogeneousVsCPU()
{
  try
  {
    const uint32_t batch=g_caif_prefix_test_hetero_batch;
    const uint32_t seq_len=g_caif_prefix_test_hetero_seq;
    const uint32_t dim=g_caif_prefix_test_hetero_dim;
    const uint32_t num_heads=g_caif_prefix_test_hetero_heads;
    const uint32_t head_dim=dim/num_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceMultiHeadAttention<float,float> mha(MakeConfig(dim,num_heads,true),stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    std::vector<float> host_input(static_cast<size_t>(batch*seq_len*dim));
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>((i*7+3)%41)*0.01f-0.2f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    std::vector<int32_t> prefix_lens={0,3,static_cast<int32_t>(seq_len)};
    CAIF_DeviceTensor prefix=MakePrefixTensor(prefix_lens,stream);
    ctx.SetPrefixLengths(prefix);

    CAIF_DeviceTensor gpu_out=mha.Forward(input,ctx);
    CAIF_HostTensor h_gpu=gpu_out.ToHost();

    std::vector<float> expected(static_cast<size_t>(batch*seq_len*dim));
    CpuMHAPrefix(host_input.data(),
                 h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                 expected.data(),
                 static_cast<int>(batch),
                 static_cast<int>(seq_len),
                 static_cast<int>(dim),
                 static_cast<int>(num_heads),
                 static_cast<int>(head_dim),
                 true,
                 prefix_lens);

    bool passed=true;
    float max_diff=0.0f;
    for(size_t i=0;i<expected.size();++i)
    {
      const float diff=std::fabs(h_gpu.Data()[i]-expected[i]);
      if(diff>max_diff)
      {
        max_diff=diff;
      }
      if(diff>g_caif_prefix_test_hetero_tol)
      {
        if(passed==true)
        {
          ISE_Out::Out()<<"  First mismatch at "
                        <<i
                        <<": gpu="
                        <<h_gpu.Data()[i]
                        <<" cpu="
                        <<expected[i]
                        <<" diff="
                        <<diff
                        <<"\n";
        }
        passed=false;
      }
    }
    ISE_Out::Out()<<"  max |gpu-cpu|="
                  <<max_diff
                  <<"\n";
    ctx.ClearPrefixLengths();
    CAIF_TestHarness::Report("Prefix::FlashHeterogeneousVsCPU",passed);
  }
  CAIF_TEST_CATCH_BLOCK("Prefix::FlashHeterogeneousVsCPU")
}

//------------------------------------------------------------------------------
// Flash backward: finite gradients, none NaN/Inf, with heterogeneous prefix
//------------------------------------------------------------------------------
void CAIF_PrefixLMTests::TestFlashPrefixBackwardNoNaN()
{
  try
  {
    const uint32_t batch=g_caif_prefix_test_bwd_batch;
    const uint32_t seq_len=g_caif_prefix_test_bwd_seq;
    const uint32_t dim=g_caif_prefix_test_bwd_dim;
    const uint32_t num_heads=g_caif_prefix_test_bwd_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DeviceMultiHeadAttention<float,float> mha(MakeConfig(dim,num_heads,true),stream);

    std::vector<float> host_input(static_cast<size_t>(batch*seq_len*dim));
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>((i*11+5)%53)*0.009f-0.2f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    std::vector<int32_t> prefix_lens={4,12};
    CAIF_DeviceTensor prefix=MakePrefixTensor(prefix_lens,stream);
    ctx.SetPrefixLengths(prefix);

    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor out=mha.Forward(input,ctx);
    CAIF_HostTensor h_out=out.ToHost();

    std::vector<float> grad_ones(static_cast<size_t>(batch*seq_len*dim),1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_in=mha.Backward(grad_out,ctx);
    CAIF_HostTensor h_grad_in=grad_in.ToHost();

    bool passed=true;

    for(size_t i=0;i<h_out.TotalElements();++i)
    {
      uint32_t bits=0;
      std::memcpy(&bits,&h_out.Data()[i],4);
      if((bits&0x7F800000)==0x7F800000)
      {
        ISE_Out::Out()<<"  Forward: NaN/Inf at "
                      <<i
                      <<"\n";
        passed=false;
        break;
      }
    }

    for(size_t i=0;i<h_grad_in.TotalElements();++i)
    {
      uint32_t bits=0;
      std::memcpy(&bits,&h_grad_in.Data()[i],4);
      if((bits&0x7F800000)==0x7F800000)
      {
        ISE_Out::Out()<<"  grad_input: NaN/Inf at "
                      <<i
                      <<"\n";
        passed=false;
        break;
      }
    }

    for(uint32_t p=0;p<mha.ParameterTensorCount();++p)
    {
      CAIF_HostTensor gp=mha.GradientTensor(p).ToHost();
      for(size_t i=0;i<gp.TotalElements();++i)
      {
        uint32_t bits=0;
        std::memcpy(&bits,&gp.Data()[i],4);
        if((bits&0x7F800000)==0x7F800000)
        {
          ISE_Out::Out()<<"  grad_param["
                        <<p
                        <<"]: NaN/Inf at "
                        <<i
                        <<"\n";
          passed=false;
          break;
        }
      }
    }

    ctx.ClearPrefixLengths();
    CAIF_TestHarness::Report("Prefix::FlashBackwardNoNaN",passed);
  }
  CAIF_TEST_CATCH_BLOCK("Prefix::FlashBackwardNoNaN")
}

//------------------------------------------------------------------------------
// ClearPrefixLengths reverts to pure causal
//------------------------------------------------------------------------------
void CAIF_PrefixLMTests::TestClearRevertsToCausal()
{
  try
  {
    const uint32_t batch=g_caif_prefix_test_clear_batch;
    const uint32_t seq_len=g_caif_prefix_test_clear_seq;
    const uint32_t dim=g_caif_prefix_test_clear_dim;
    const uint32_t num_heads=g_caif_prefix_test_clear_heads;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceMultiHeadAttention<float,float> mha(MakeConfig(dim,num_heads,true),stream);

    std::vector<float> host_input(static_cast<size_t>(batch*seq_len*dim));
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i%29)*0.02f-0.3f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    CAIF_DeviceTensor ref=mha.Forward(input,ctx);
    CAIF_HostTensor h_ref=ref.ToHost();

    std::vector<int32_t> prefix_lens={4};
    CAIF_DeviceTensor prefix=MakePrefixTensor(prefix_lens,stream);
    ctx.SetPrefixLengths(prefix);
    CAIF_DeviceTensor alt=mha.Forward(input,ctx);
    CAIF_HostTensor h_alt=alt.ToHost();

    ctx.ClearPrefixLengths();
    CAIF_DeviceTensor back=mha.Forward(input,ctx);
    CAIF_HostTensor h_back=back.ToHost();

    bool differed=false;
    for(size_t i=0;i<h_ref.TotalElements();++i)
    {
      if(FloatClose(h_ref.Data()[i],h_alt.Data()[i],g_caif_prefix_test_unmask_tol)==false)
      {
        differed=true;
        break;
      }
    }

    bool restored=true;
    for(size_t i=0;i<h_ref.TotalElements();++i)
    {
      if(FloatClose(h_ref.Data()[i],h_back.Data()[i],g_caif_prefix_test_clear_tol)==false)
      {
        ISE_Out::Out()<<"  Post-clear mismatch at "
                      <<i
                      <<": causal="
                      <<h_ref.Data()[i]
                      <<" after-clear="
                      <<h_back.Data()[i]
                      <<"\n";
        restored=false;
        break;
      }
    }

    if(differed==false)
    {
      ISE_Out::Out()<<"  Setting prefix produced identical output as causal — "
                    <<"prefix path may not be taking effect\n";
    }

    const bool passed=(differed==true && restored==true);
    CAIF_TestHarness::Report("Prefix::ClearRevertsToCausal",passed);
  }
  CAIF_TEST_CATCH_BLOCK("Prefix::ClearRevertsToCausal")
}

void CAIF_PrefixLMTests::RunAll()
{
  ISE_Out::Out()<<"=== Prefix-LM mask tests ==="
                <<"\n\n";
  TestKernelPrefixMaskFill();
  TestKernelPrefixMaskGrad();
  TestFlashPrefixZeroEqualsCausal();
  TestFlashPrefixHeterogeneousVsCPU();
  TestFlashPrefixBackwardNoNaN();
  TestClearRevertsToCausal();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_PrefixLMTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
