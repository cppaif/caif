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

#include "caif_device_multi_head_attention.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    std::cout<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    std::cout<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Bitwise NaN/Inf check safe with -ffast-math (host code)
//------------------------------------------------------------------------------
struct NanCheckResult_t
{
  uint32_t nan_count;
  uint32_t inf_count;
  uint32_t total;
  float min_val;
  float max_val;
};

static NanCheckResult_t BitwiseNanCheck(const float *data,const uint32_t count)
{
  NanCheckResult_t result;
  result.nan_count=0;
  result.inf_count=0;
  result.total=count;
  result.min_val=1e30f;
  result.max_val=-1e30f;

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
        ++result.nan_count;
      }
      else
      {
        ++result.inf_count;
      }
    }
    else
    {
      if(data[i]<result.min_val)
      {
        result.min_val=data[i];
      }
      if(data[i]>result.max_val)
      {
        result.max_val=data[i];
      }
    }
  }

  return result;
}

//------------------------------------------------------------------------------
// Fill buffer with deterministic pseudo-random data in [lo, hi]
//------------------------------------------------------------------------------
static void FillDeterministic(float *data,
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

//------------------------------------------------------------------------------
// Helper: create GQA config matching Qwen2.5-Coder-1.5B
//------------------------------------------------------------------------------
static CAIF_DeviceMultiHeadAttention::AttentionConfig_t MakeQwenConfig(
  const uint32_t dim,
  const uint32_t num_heads,
  const uint32_t num_kv_heads,
  const uint32_t head_dim)
{
  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=dim;
  config.num_heads=num_heads;
  config.num_kv_heads=num_kv_heads;
  config.head_dim=head_dim;
  config.causal=true;
  config.use_rope=true;
  config.rope_base=g_caif_rope_default_base;
  config.dropout_rate=0.0f;
  return config;
}

//------------------------------------------------------------------------------
// Run MHA forward+backward, check all gradients for NaN
// Returns true if NaN-free, false if NaN found
//------------------------------------------------------------------------------
static bool RunMHABackwardNanCheck(const char *test_label,
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
  auto config=MakeQwenConfig(dim,num_heads,num_kv_heads,head_dim);
  CAIF_DeviceMultiHeadAttention mha(config,stream);

  std::cout<<"  "<<test_label<<": dim="<<dim
           <<" heads="<<num_heads
           <<" kv_heads="<<num_kv_heads
           <<" head_dim="<<head_dim
           <<" batch="<<batch
           <<" seq="<<seq_len
           <<" input=["<<input_lo<<","<<input_hi<<"]\n";

  // Create input data
  const uint32_t input_elems=batch*seq_len*dim;
  std::vector<float> host_input(input_elems);
  FillDeterministic(host_input.data(),input_elems,input_lo,input_hi,seed);

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                        {batch,seq_len,dim},
                                                        stream);

  // Forward with training=true
  CAIF_DeviceTensor output=mha.Forward(input,true);

  // Check forward output
  CAIF_HostTensor h_output=output.ToHost();
  NanCheckResult_t fwd_check=BitwiseNanCheck(h_output.Data(),
                                              h_output.TotalElements());
  if(fwd_check.nan_count>0||fwd_check.inf_count>0)
  {
    std::cout<<"  Forward output: nan="<<fwd_check.nan_count
             <<" inf="<<fwd_check.inf_count
             <<" / "<<fwd_check.total
             <<" min="<<fwd_check.min_val
             <<" max="<<fwd_check.max_val<<"\n";
  }
  else
  {
    std::cout<<"  Forward output OK: min="<<fwd_check.min_val
             <<" max="<<fwd_check.max_val<<"\n";
  }

  // Backward with grad_output = ones
  std::vector<float> grad_ones(input_elems,1.0f);
  CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                           {batch,seq_len,dim},
                                                           stream);
  CAIF_DeviceTensor grad_input=mha.Backward(grad_out);

  // Check input gradient
  CAIF_HostTensor h_grad_input=grad_input.ToHost();
  NanCheckResult_t dx_check=BitwiseNanCheck(h_grad_input.Data(),
                                             h_grad_input.TotalElements());

  bool all_clean=true;

  if(dx_check.nan_count>0||dx_check.inf_count>0)
  {
    std::cout<<"  grad_input: nan="<<dx_check.nan_count
             <<" inf="<<dx_check.inf_count
             <<" / "<<dx_check.total<<"\n";
    all_clean=false;
  }

  // Check all parameter gradients
  for(uint32_t p=0;p<mha.ParameterTensorCount();++p)
  {
    CAIF_HostTensor h_grad=mha.GradientTensor(p).ToHost();
    NanCheckResult_t pg=BitwiseNanCheck(h_grad.Data(),h_grad.TotalElements());
    if(pg.nan_count>0||pg.inf_count>0)
    {
      std::cout<<"  grad_param["<<p<<"]: nan="<<pg.nan_count
               <<" inf="<<pg.inf_count
               <<" / "<<pg.total
               <<" min="<<pg.min_val
               <<" max="<<pg.max_val<<"\n";
      all_clean=false;
    }
  }

  if(all_clean==true)
  {
    std::cout<<"  All gradients NaN-free\n";
  }

  return all_clean;
}

//------------------------------------------------------------------------------
// Test 1: Small GQA config — quick sanity check
// heads=4, kv_heads=2, dim=128, head_dim=32
//------------------------------------------------------------------------------
static void TestSmallGQA()
{
  bool passed=RunMHABackwardNanCheck("SmallGQA",
                                     128,
                                     4,
                                     2,
                                     32,
                                     1,
                                     16,
                                     -1.0f,
                                     1.0f,
                                     42);
  ReportResult("MHABackward::SmallGQA",passed);
}

//------------------------------------------------------------------------------
// Test 2: Medium GQA config
// heads=8, kv_heads=2, dim=512, head_dim=64
//------------------------------------------------------------------------------
static void TestMediumGQA()
{
  bool passed=RunMHABackwardNanCheck("MediumGQA",
                                     512,
                                     8,
                                     2,
                                     64,
                                     2,
                                     64,
                                     -0.15f,
                                     0.14f,
                                     42);
  ReportResult("MHABackward::MediumGQA",passed);
}

//------------------------------------------------------------------------------
// Test 3: Qwen-like config with reduced seq_len
// heads=12, kv_heads=2, dim=1536, head_dim=128, seq=32
//------------------------------------------------------------------------------
static void TestQwenShortSeq()
{
  bool passed=RunMHABackwardNanCheck("QwenShortSeq",
                                     1536,
                                     12,
                                     2,
                                     128,
                                     1,
                                     32,
                                     -0.15f,
                                     0.14f,
                                     42);
  ReportResult("MHABackward::QwenShortSeq",passed);
}

//------------------------------------------------------------------------------
// Test 4: Qwen-like config with training-length seq
// heads=12, kv_heads=2, dim=1536, head_dim=128, seq=512, batch=2
//------------------------------------------------------------------------------
static void TestQwenFullSeq()
{
  bool passed=RunMHABackwardNanCheck("QwenFullSeq",
                                     1536,
                                     12,
                                     2,
                                     128,
                                     2,
                                     512,
                                     -0.15f,
                                     0.14f,
                                     42);
  ReportResult("MHABackward::QwenFullSeq",passed);
}

//------------------------------------------------------------------------------
// Test 5: Qwen config with larger input values
// Same as test 4 but inputs in [-5, 5]
//------------------------------------------------------------------------------
static void TestQwenLargeInputs()
{
  bool passed=RunMHABackwardNanCheck("QwenLargeInputs",
                                     1536,
                                     12,
                                     2,
                                     128,
                                     2,
                                     512,
                                     -5.0f,
                                     5.0f,
                                     123);
  ReportResult("MHABackward::QwenLargeInputs",passed);
}

//------------------------------------------------------------------------------
// Test 6: Multiple seeds for Qwen config to catch non-deterministic NaN
//------------------------------------------------------------------------------
static void TestQwenMultipleSeeds()
{
  const uint32_t num_runs=5;

  std::cout<<"  Running "<<num_runs
           <<" seeds with Qwen-like config (dim=1536, seq=512, batch=2)\n";

  bool all_passed=true;
  uint32_t nan_runs=0;

  for(uint32_t run=0;run<num_runs;++run)
  {
    const uint32_t seed=run*7919+31;
    bool clean=RunMHABackwardNanCheck("QwenSeed",
                                      1536,
                                      12,
                                      2,
                                      128,
                                      2,
                                      512,
                                      -0.15f,
                                      0.14f,
                                      seed);
    if(clean==false)
    {
      all_passed=false;
      ++nan_runs;
    }
  }

  if(nan_runs>0)
  {
    std::cout<<"  NaN in "<<nan_runs<<" / "<<num_runs<<" runs\n";
  }

  ReportResult("MHABackward::QwenMultipleSeeds",all_passed);
}

//------------------------------------------------------------------------------
// Test 7: Standard MHA (kv_heads==heads) with matching dims
// heads=12, kv_heads=12, dim=1536, head_dim=128
//------------------------------------------------------------------------------
static void TestMHAFullHeads()
{
  bool passed=RunMHABackwardNanCheck("MHAFullHeads",
                                     1536,
                                     12,
                                     12,
                                     128,
                                     1,
                                     128,
                                     -0.15f,
                                     0.14f,
                                     42);
  ReportResult("MHABackward::MHAFullHeads",passed);
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== MHA Backward NaN Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestSmallGQA();
  TestMediumGQA();
  TestQwenShortSeq();
  TestQwenFullSeq();
  TestQwenLargeInputs();
  TestQwenMultipleSeeds();
  TestMHAFullHeads();
#else
  std::cout<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  if(g_tests_failed>0)
  {
    return 1;
  }
  return 0;
}
