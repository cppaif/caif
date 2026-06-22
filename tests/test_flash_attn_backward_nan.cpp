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
// Test: Flash attention backward NaN/Inf checks across GQA configs.
//------------------------------------------------------------------------------
#include "caif_device_multi_head_attention.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_constants.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cstring>
#include <cmath>

namespace instance
{

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Bitwise NaN/Inf check safe with -ffast-math (host code)
//------------------------------------------------------------------------------
struct CAIF_NanCheckResult_t
{
  public:
    CAIF_NanCheckResult_t():_nan_count(0),
                            _inf_count(0),
                            _total(0),
                            _min_val(1e30f),
                            _max_val(-1e30f)
    {
    }

    uint32_t NanCount()const{return _nan_count;}
    uint32_t InfCount()const{return _inf_count;}
    uint32_t Total()const{return _total;}
    float MinVal()const{return _min_val;}
    float MaxVal()const{return _max_val;}
    void SetNanCount(const uint32_t v){_nan_count=v;}
    void SetInfCount(const uint32_t v){_inf_count=v;}
    void SetTotal(const uint32_t v){_total=v;}
    void SetMinVal(const float v){_min_val=v;}
    void SetMaxVal(const float v){_max_val=v;}
    void IncrNanCount(){++_nan_count;}
    void IncrInfCount(){++_inf_count;}

  private:
    uint32_t _nan_count;
    uint32_t _inf_count;
    uint32_t _total;
    float _min_val;
    float _max_val;
};

//------------------------------------------------------------------------------
// MHA backward NaN/Inf correctness tests.
//------------------------------------------------------------------------------
class CAIF_FlashAttnBackwardNanTests
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_NanCheckResult_t BitwiseNanCheck(const float *data,const uint32_t count);
    static void FillDeterministic(float *data,
                                  const uint32_t count,
                                  const float lo,
                                  const float hi,
                                  const uint32_t seed);
    static CAIF_DeviceMultiHeadAttentionConfig MakeQwenConfig(
                                                                           const uint32_t dim,
                                                                           const uint32_t num_heads,
                                                                           const uint32_t num_kv_heads,
                                                                           const uint32_t head_dim);
    static bool RunMHABackwardNanCheck(const char *test_label,
                                       const uint32_t dim,
                                       const uint32_t num_heads,
                                       const uint32_t num_kv_heads,
                                       const uint32_t head_dim,
                                       const uint32_t batch,
                                       const uint32_t seq_len,
                                       const float input_lo,
                                       const float input_hi,
                                       const uint32_t seed);

    static void TestSmallGQA();
    static void TestMediumGQA();
    static void TestQwenShortSeq();
    static void TestQwenFullSeq();
    static void TestQwenLargeInputs();
    static void TestQwenMultipleSeeds();
    static void TestMHAFullHeads();
};

CAIF_NanCheckResult_t CAIF_FlashAttnBackwardNanTests::BitwiseNanCheck(const float *data,
                                                                        const uint32_t count)
{
  CAIF_NanCheckResult_t result;
  result.SetTotal(count);

  for(uint32_t i=0;i<count;++i)
  {
    uint32_t bits=0;
    std::memcpy(&bits,&data[i],4);
    const uint32_t exponent=bits&0x7F800000;
    const uint32_t mantissa=bits&0x007FFFFF;

    if(exponent==0x7F800000)
    {
      if(mantissa!=0)
      {
        result.IncrNanCount();
      }
      else
      {
        result.IncrInfCount();
      }
    }
    else
    {
      if(data[i]<result.MinVal())
      {
        result.SetMinVal(data[i]);
      }
      if(data[i]>result.MaxVal())
      {
        result.SetMaxVal(data[i]);
      }
    }
  }

  return result;
}

void CAIF_FlashAttnBackwardNanTests::FillDeterministic(float *data,
                                                        const uint32_t count,
                                                        const float lo,
                                                        const float hi,
                                                        const uint32_t seed)
{
  uint32_t state=seed;
  const float range=hi-lo;

  for(uint32_t i=0;i<count;++i)
  {
    state=state*1664525u+1013904223u;
    const float t=static_cast<float>(state&0xFFFFu)/65535.0f;
    data[i]=lo+t*range;
  }
}

CAIF_DeviceMultiHeadAttentionConfig
CAIF_FlashAttnBackwardNanTests::MakeQwenConfig(const uint32_t dim,
                                                const uint32_t num_heads,
                                                const uint32_t num_kv_heads,
                                                const uint32_t head_dim)
{
  CAIF_DeviceMultiHeadAttentionConfig config(dim,
                                             num_heads,
                                             num_kv_heads,
                                             head_dim,
                                             true,
                                             true,
                                             g_caif_rope_default_base,
                                             0.0f);
  return config;
}

//------------------------------------------------------------------------------
// Run MHA forward+backward, check all gradients for NaN
// Returns true if NaN-free, false if NaN found
//------------------------------------------------------------------------------
bool CAIF_FlashAttnBackwardNanTests::RunMHABackwardNanCheck(const char *test_label,
                                                             const uint32_t dim,
                                                             const uint32_t num_heads,
                                                             const uint32_t num_kv_heads,
                                                             const uint32_t head_dim,
                                                             const uint32_t batch,
                                                             const uint32_t seq_len,
                                                             const float input_lo,
                                                             const float input_hi,
                                                             const uint32_t seed)
{
  CAIF_CudaStream stream;
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  auto config=MakeQwenConfig(dim,num_heads,num_kv_heads,head_dim);
  CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);

  ISE_Out::Out()<<"  "
                <<test_label
                <<": dim="
                <<dim
                <<" heads="
                <<num_heads
                <<" kv_heads="
                <<num_kv_heads
                <<" head_dim="
                <<head_dim
                <<" batch="
                <<batch
                <<" seq="
                <<seq_len
                <<" input=["
                <<input_lo
                <<","
                <<input_hi
                <<"]\n";

  // Create input data
  const uint32_t input_elems=batch*seq_len*dim;
  std::vector<float> host_input(input_elems);
  FillDeterministic(host_input.data(),input_elems,input_lo,input_hi,seed);

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {batch,seq_len,dim},
                                                          stream);

  // Forward with training=true
  ctx.SetTraining(true);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor output=mha.Forward(input,ctx);

  // Check forward output
  CAIF_HostTensor h_output=output.ToHost();
  CAIF_NanCheckResult_t fwd_check=BitwiseNanCheck(h_output.Data(),
                                                   h_output.TotalElements());
  if(fwd_check.NanCount()>0 || fwd_check.InfCount()>0)
  {
    ISE_Out::Out()<<"  Forward output: nan="
                  <<fwd_check.NanCount()
                  <<" inf="
                  <<fwd_check.InfCount()
                  <<" / "
                  <<fwd_check.Total()
                  <<" min="
                  <<fwd_check.MinVal()
                  <<" max="
                  <<fwd_check.MaxVal()
                  <<"\n";
  }
  else
  {
    ISE_Out::Out()<<"  Forward output OK: min="
                  <<fwd_check.MinVal()
                  <<" max="
                  <<fwd_check.MaxVal()
                  <<"\n";
  }

  // Backward with grad_output = ones
  std::vector<float> grad_ones(input_elems,1.0f);
  CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                             {batch,seq_len,dim},
                                                             stream);
  ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
  CAIF_DeviceTensor grad_input=mha.Backward(grad_out,ctx);

  // Check input gradient
  CAIF_HostTensor h_grad_input=grad_input.ToHost();
  CAIF_NanCheckResult_t dx_check=BitwiseNanCheck(h_grad_input.Data(),
                                                  h_grad_input.TotalElements());

  bool all_clean=true;

  if(dx_check.NanCount()>0 || dx_check.InfCount()>0)
  {
    ISE_Out::Out()<<"  grad_input: nan="
                  <<dx_check.NanCount()
                  <<" inf="
                  <<dx_check.InfCount()
                  <<" / "
                  <<dx_check.Total()
                  <<"\n";
    all_clean=false;
  }

  // Check all parameter gradients
  for(uint32_t p=0;p<mha.ParameterTensorCount();++p)
  {
    CAIF_HostTensor h_grad=mha.GradientTensor(p).ToHost();
    CAIF_NanCheckResult_t pg=BitwiseNanCheck(h_grad.Data(),h_grad.TotalElements());
    if(pg.NanCount()>0 || pg.InfCount()>0)
    {
      ISE_Out::Out()<<"  grad_param["
                    <<p
                    <<"]: nan="
                    <<pg.NanCount()
                    <<" inf="
                    <<pg.InfCount()
                    <<" / "
                    <<pg.Total()
                    <<" min="
                    <<pg.MinVal()
                    <<" max="
                    <<pg.MaxVal()
                    <<"\n";
      all_clean=false;
    }
  }

  if(all_clean==true)
  {
    ISE_Out::Out()<<"  All gradients NaN-free\n";
  }

  return all_clean;
}

//------------------------------------------------------------------------------
// Test 1: Small GQA config — quick sanity check
// heads=4, kv_heads=2, dim=128, head_dim=32
//------------------------------------------------------------------------------
void CAIF_FlashAttnBackwardNanTests::TestSmallGQA()
{
  bool passed=RunMHABackwardNanCheck("SmallGQA",128,4,2,32,1,16,-1.0f,1.0f,42);
  CAIF_TestHarness::Report("MHABackward::SmallGQA",passed);
}

//------------------------------------------------------------------------------
// Test 2: Medium GQA config
// heads=8, kv_heads=2, dim=512, head_dim=64
//------------------------------------------------------------------------------
void CAIF_FlashAttnBackwardNanTests::TestMediumGQA()
{
  bool passed=RunMHABackwardNanCheck("MediumGQA",512,8,2,64,2,64,-0.15f,0.14f,42);
  CAIF_TestHarness::Report("MHABackward::MediumGQA",passed);
}

//------------------------------------------------------------------------------
// Test 3: Qwen-like config with reduced seq_len
// heads=12, kv_heads=2, dim=1536, head_dim=128, seq=32
//------------------------------------------------------------------------------
void CAIF_FlashAttnBackwardNanTests::TestQwenShortSeq()
{
  bool passed=RunMHABackwardNanCheck("QwenShortSeq",1536,12,2,128,1,32,-0.15f,0.14f,42);
  CAIF_TestHarness::Report("MHABackward::QwenShortSeq",passed);
}

//------------------------------------------------------------------------------
// Test 4: Qwen-like config with training-length seq
// heads=12, kv_heads=2, dim=1536, head_dim=128, seq=512, batch=2
//------------------------------------------------------------------------------
void CAIF_FlashAttnBackwardNanTests::TestQwenFullSeq()
{
  bool passed=RunMHABackwardNanCheck("QwenFullSeq",1536,12,2,128,2,512,-0.15f,0.14f,42);
  CAIF_TestHarness::Report("MHABackward::QwenFullSeq",passed);
}

//------------------------------------------------------------------------------
// Test 5: Qwen config with larger input values
// Same as test 4 but inputs in [-5, 5]
//------------------------------------------------------------------------------
void CAIF_FlashAttnBackwardNanTests::TestQwenLargeInputs()
{
  bool passed=RunMHABackwardNanCheck("QwenLargeInputs",1536,12,2,128,2,512,-5.0f,5.0f,123);
  CAIF_TestHarness::Report("MHABackward::QwenLargeInputs",passed);
}

//------------------------------------------------------------------------------
// Test 6: Multiple seeds for Qwen config to catch non-deterministic NaN
//------------------------------------------------------------------------------
void CAIF_FlashAttnBackwardNanTests::TestQwenMultipleSeeds()
{
  const uint32_t num_runs=5;

  ISE_Out::Out()<<"  Running "
                <<num_runs
                <<" seeds with Qwen-like config (dim=1536, seq=512, batch=2)\n";

  bool all_passed=true;
  uint32_t nan_runs=0;

  for(uint32_t run=0;run<num_runs;++run)
  {
    const uint32_t seed=run*7919+31;
    bool clean=RunMHABackwardNanCheck("QwenSeed",1536,12,2,128,2,512,-0.15f,0.14f,seed);
    if(clean==false)
    {
      all_passed=false;
      ++nan_runs;
    }
  }

  if(nan_runs>0)
  {
    ISE_Out::Out()<<"  NaN in "
                  <<nan_runs
                  <<" / "
                  <<num_runs
                  <<" runs\n";
  }

  CAIF_TestHarness::Report("MHABackward::QwenMultipleSeeds",all_passed);
}

//------------------------------------------------------------------------------
// Test 7: Standard MHA (kv_heads==heads) with matching dims
// heads=12, kv_heads=12, dim=1536, head_dim=128
//------------------------------------------------------------------------------
void CAIF_FlashAttnBackwardNanTests::TestMHAFullHeads()
{
  bool passed=RunMHABackwardNanCheck("MHAFullHeads",1536,12,12,128,1,128,-0.15f,0.14f,42);
  CAIF_TestHarness::Report("MHABackward::MHAFullHeads",passed);
}

void CAIF_FlashAttnBackwardNanTests::RunAll()
{
  ISE_Out::Out()<<"=== MHA Backward NaN Tests ===\n\n";
  TestSmallGQA();
  TestMediumGQA();
  TestQwenShortSeq();
  TestQwenFullSeq();
  TestQwenLargeInputs();
  TestQwenMultipleSeeds();
  TestMHAFullHeads();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FlashAttnBackwardNanTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
