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
// KV-Cache test for Multi-Head Attention
//------------------------------------------------------------------------------
#include "caif_device_multi_head_attention.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>

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

static bool FloatEqual(float a,float b,float tolerance=1e-3f)
{
  return std::fabs(a-b)<tolerance;
}

static void FillRandom(std::vector<float> &data,float min_val,float max_val,
                       uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(min_val,max_val);
  for(size_t i=0;i<data.size();++i)
  {
    data[i]=dist(gen);
  }
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Test 1: Enable/Disable/Reset KV-Cache
//------------------------------------------------------------------------------
static void TestKVCacheManagement()
{
  CAIF_CudaStream stream;

  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=64;
  config.num_heads=4;
  config.num_kv_heads=4;
  config.head_dim=16;
  config.causal=true;
  config.use_rope=false;
  config.rope_base=10000.0f;
  config.dropout_rate=0.0f;

  CAIF_DeviceMultiHeadAttention mha(config,stream);
  mha.InitializeWeights(42);

  // Initially disabled
  bool test_pass=true;
  if(mha.IsKVCacheEnabled()!=false)
  {
    test_pass=false;
  }

  // Enable cache
  const uint32_t batch=2;
  const uint32_t max_seq=128;
  mha.EnableKVCache(batch,max_seq);
  if(mha.IsKVCacheEnabled()!=true)
  {
    test_pass=false;
  }
  if(mha.KVCacheLength()!=0)
  {
    test_pass=false;
  }

  // Disable cache
  mha.DisableKVCache();
  if(mha.IsKVCacheEnabled()!=false)
  {
    test_pass=false;
  }

  // Re-enable and reset
  mha.EnableKVCache(batch,max_seq);

  // Process some tokens
  const uint32_t seq_len=4;
  std::vector<float> input_data(batch*seq_len*config.dim);
  FillRandom(input_data,-1.0f,1.0f,123);
  CAIF_DeviceTensor d_input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,config.dim},stream);
  CAIF_DeviceTensor output=mha.ForwardCached(d_input);
  stream.Synchronize();

  if(mha.KVCacheLength()!=seq_len)
  {
    test_pass=false;
  }

  // Reset cache
  mha.ResetKVCache();
  if(mha.KVCacheLength()!=0)
  {
    test_pass=false;
  }
  if(mha.IsKVCacheEnabled()!=true)
  {
    test_pass=false;
  }

  ReportResult("TestKVCacheManagement",test_pass);
}

//------------------------------------------------------------------------------
// Test 2: Cached forward matches non-cached for full prompt
// When we process the entire sequence in one ForwardCached call, the output
// should match Forward() with the same input.
//------------------------------------------------------------------------------
static void TestCachedMatchesNonCached()
{
  CAIF_CudaStream stream;

  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=32;
  config.num_heads=2;
  config.num_kv_heads=2;
  config.head_dim=16;
  config.causal=true;
  config.use_rope=false;
  config.rope_base=10000.0f;
  config.dropout_rate=0.0f;

  CAIF_DeviceMultiHeadAttention mha(config,stream);
  mha.InitializeWeights(42);

  const uint32_t batch=1;
  const uint32_t seq_len=8;

  // Create input
  std::vector<float> input_data(batch*seq_len*config.dim);
  FillRandom(input_data,-1.0f,1.0f,999);
  CAIF_DeviceTensor d_input=CAIF_DeviceTensor::FromHostData(
                             input_data.data(),{batch,seq_len,config.dim},stream);

  // Non-cached forward
  CAIF_DeviceTensor output_nocache=mha.Forward(d_input,false);
  CAIF_HostTensor h_out_nocache=output_nocache.ToHost();

  // Cached forward (full prompt)
  mha.EnableKVCache(batch,seq_len);
  mha.ResetKVCache();
  CAIF_DeviceTensor output_cached=mha.ForwardCached(d_input);
  CAIF_HostTensor h_out_cached=output_cached.ToHost();

  // Compare outputs
  bool test_pass=true;
  const float tol=1e-4f;
  const size_t total=batch*seq_len*config.dim;
  for(size_t i=0;i<total;++i)
  {
    if(FloatEqual(h_out_nocache.Data()[i],h_out_cached.Data()[i],tol)==false)
    {
      std::cout<<"  Mismatch at "<<i<<": nocache="<<h_out_nocache.Data()[i]
               <<" cached="<<h_out_cached.Data()[i]<<"\n";
      test_pass=false;
      break;
    }
  }

  ReportResult("TestCachedMatchesNonCached",test_pass);
}

//------------------------------------------------------------------------------
// Test 3: Incremental decoding produces same output as full forward
// Process prompt, then decode one token at a time. The output at each position
// should match what we'd get from Forward() on the full sequence up to that point.
//------------------------------------------------------------------------------
static void TestIncrementalDecoding()
{
  CAIF_CudaStream stream;

  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=32;
  config.num_heads=2;
  config.num_kv_heads=2;
  config.head_dim=16;
  config.causal=true;
  config.use_rope=false;
  config.rope_base=10000.0f;
  config.dropout_rate=0.0f;

  CAIF_DeviceMultiHeadAttention mha(config,stream);
  mha.InitializeWeights(42);

  const uint32_t batch=1;
  const uint32_t prompt_len=4;
  const uint32_t decode_steps=3;
  const uint32_t total_len=prompt_len+decode_steps;

  // Create full input
  std::vector<float> full_input(batch*total_len*config.dim);
  FillRandom(full_input,-1.0f,1.0f,777);

  // Full forward for reference
  CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(
                            full_input.data(),{batch,total_len,config.dim},stream);
  CAIF_DeviceTensor full_output=mha.Forward(d_full,false);
  CAIF_HostTensor h_full_out=full_output.ToHost();

  // Incremental decoding
  mha.EnableKVCache(batch,total_len);
  mha.ResetKVCache();

  // Process prompt
  std::vector<float> prompt_data(batch*prompt_len*config.dim);
  for(uint32_t i=0;i<prompt_data.size();++i)
  {
    prompt_data[i]=full_input[i];
  }
  CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(
                              prompt_data.data(),{batch,prompt_len,config.dim},stream);
  CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt);
  CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

  bool test_pass=true;
  const float tol=1e-4f;

  // Check prompt output matches full output prefix
  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t s=0;s<prompt_len;++s)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        const size_t idx=b*total_len*config.dim+s*config.dim+d;
        const size_t prompt_idx=b*prompt_len*config.dim+s*config.dim+d;
        if(FloatEqual(h_full_out.Data()[idx],h_prompt_out.Data()[prompt_idx],tol)==false)
        {
          std::cout<<"  Prompt mismatch at b="<<b<<" s="<<s<<" d="<<d
                   <<": full="<<h_full_out.Data()[idx]
                   <<" incremental="<<h_prompt_out.Data()[prompt_idx]<<"\n";
          test_pass=false;
          break;
        }
      }
      if(test_pass==false)
      {
        break;
      }
    }
    if(test_pass==false)
    {
      break;
    }
  }

  // Decode one token at a time
  for(uint32_t step=0;step<decode_steps&&test_pass==true;++step)
  {
    const uint32_t pos=prompt_len+step;

    // Extract single token
    std::vector<float> token_data(batch*1*config.dim);
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        token_data[b*config.dim+d]=full_input[b*total_len*config.dim+pos*config.dim+d];
      }
    }
    CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(
                               token_data.data(),{batch,1,config.dim},stream);
    CAIF_DeviceTensor token_out=mha.ForwardCached(d_token);
    CAIF_HostTensor h_token_out=token_out.ToHost();

    // Check output matches full forward at this position
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        const size_t full_idx=b*total_len*config.dim+pos*config.dim+d;
        const size_t token_idx=b*config.dim+d;
        if(FloatEqual(h_full_out.Data()[full_idx],h_token_out.Data()[token_idx],tol)==false)
        {
          std::cout<<"  Step "<<step<<" mismatch at b="<<b<<" d="<<d
                   <<": full="<<h_full_out.Data()[full_idx]
                   <<" incremental="<<h_token_out.Data()[token_idx]<<"\n";
          test_pass=false;
          break;
        }
      }
      if(test_pass==false)
      {
        break;
      }
    }
  }

  ReportResult("TestIncrementalDecoding",test_pass);
}

//------------------------------------------------------------------------------
// Test 4: Incremental decoding with RoPE
// Same as Test 3 but with RoPE enabled. The position offsets must be correct.
//------------------------------------------------------------------------------
static void TestIncrementalDecodingWithRoPE()
{
  CAIF_CudaStream stream;

  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=32;
  config.num_heads=2;
  config.num_kv_heads=2;
  config.head_dim=16;
  config.causal=true;
  config.use_rope=true;
  config.rope_base=10000.0f;
  config.dropout_rate=0.0f;

  CAIF_DeviceMultiHeadAttention mha(config,stream);
  mha.InitializeWeights(42);

  const uint32_t batch=1;
  const uint32_t prompt_len=4;
  const uint32_t decode_steps=2;
  const uint32_t total_len=prompt_len+decode_steps;

  // Create full input
  std::vector<float> full_input(batch*total_len*config.dim);
  FillRandom(full_input,-1.0f,1.0f,888);

  // Full forward for reference
  CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(
                            full_input.data(),{batch,total_len,config.dim},stream);
  CAIF_DeviceTensor full_output=mha.Forward(d_full,false);
  CAIF_HostTensor h_full_out=full_output.ToHost();

  // Incremental decoding
  mha.EnableKVCache(batch,total_len);
  mha.ResetKVCache();

  // Process prompt
  std::vector<float> prompt_data(batch*prompt_len*config.dim);
  for(uint32_t i=0;i<prompt_data.size();++i)
  {
    prompt_data[i]=full_input[i];
  }
  CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(
                              prompt_data.data(),{batch,prompt_len,config.dim},stream);
  CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt);
  CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

  bool test_pass=true;
  const float tol=1e-4f;

  // Check prompt output matches full output prefix
  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t s=0;s<prompt_len;++s)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        const size_t idx=b*total_len*config.dim+s*config.dim+d;
        const size_t prompt_idx=b*prompt_len*config.dim+s*config.dim+d;
        if(FloatEqual(h_full_out.Data()[idx],h_prompt_out.Data()[prompt_idx],tol)==false)
        {
          std::cout<<"  Prompt mismatch at b="<<b<<" s="<<s<<" d="<<d
                   <<": full="<<h_full_out.Data()[idx]
                   <<" incremental="<<h_prompt_out.Data()[prompt_idx]<<"\n";
          test_pass=false;
          break;
        }
      }
      if(test_pass==false)
      {
        break;
      }
    }
    if(test_pass==false)
    {
      break;
    }
  }

  // Decode one token at a time
  for(uint32_t step=0;step<decode_steps&&test_pass==true;++step)
  {
    const uint32_t pos=prompt_len+step;

    // Extract single token
    std::vector<float> token_data(batch*1*config.dim);
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        token_data[b*config.dim+d]=full_input[b*total_len*config.dim+pos*config.dim+d];
      }
    }
    CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(
                               token_data.data(),{batch,1,config.dim},stream);
    CAIF_DeviceTensor token_out=mha.ForwardCached(d_token);
    CAIF_HostTensor h_token_out=token_out.ToHost();

    // Check output matches full forward at this position
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        const size_t full_idx=b*total_len*config.dim+pos*config.dim+d;
        const size_t token_idx=b*config.dim+d;
        if(FloatEqual(h_full_out.Data()[full_idx],h_token_out.Data()[token_idx],tol)==false)
        {
          std::cout<<"  RoPE step "<<step<<" mismatch at b="<<b<<" d="<<d
                   <<": full="<<h_full_out.Data()[full_idx]
                   <<" incremental="<<h_token_out.Data()[token_idx]<<"\n";
          test_pass=false;
          break;
        }
      }
      if(test_pass==false)
      {
        break;
      }
    }
  }

  ReportResult("TestIncrementalDecodingWithRoPE",test_pass);
}

//------------------------------------------------------------------------------
// Test 5: GQA with KV-Cache
// Test incremental decoding when num_kv_heads < num_heads (grouped-query attention)
//------------------------------------------------------------------------------
static void TestGQAWithKVCache()
{
  CAIF_CudaStream stream;

  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=64;
  config.num_heads=4;
  config.num_kv_heads=2;  // GQA: 4 query heads share 2 kv heads
  config.head_dim=16;
  config.causal=true;
  config.use_rope=false;
  config.rope_base=10000.0f;
  config.dropout_rate=0.0f;

  CAIF_DeviceMultiHeadAttention mha(config,stream);
  mha.InitializeWeights(42);

  const uint32_t batch=1;
  const uint32_t prompt_len=3;
  const uint32_t decode_steps=2;
  const uint32_t total_len=prompt_len+decode_steps;

  // Create full input
  std::vector<float> full_input(batch*total_len*config.dim);
  FillRandom(full_input,-1.0f,1.0f,555);

  // Full forward for reference
  CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(
                            full_input.data(),{batch,total_len,config.dim},stream);
  CAIF_DeviceTensor full_output=mha.Forward(d_full,false);
  CAIF_HostTensor h_full_out=full_output.ToHost();

  // Incremental decoding
  mha.EnableKVCache(batch,total_len);
  mha.ResetKVCache();

  // Process prompt
  std::vector<float> prompt_data(batch*prompt_len*config.dim);
  for(uint32_t i=0;i<prompt_data.size();++i)
  {
    prompt_data[i]=full_input[i];
  }
  CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(
                              prompt_data.data(),{batch,prompt_len,config.dim},stream);
  CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt);
  CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

  bool test_pass=true;
  const float tol=1e-4f;

  // Check prompt output
  for(uint32_t s=0;s<prompt_len&&test_pass==true;++s)
  {
    for(uint32_t d=0;d<config.dim;++d)
    {
      const size_t idx=s*config.dim+d;
      if(FloatEqual(h_full_out.Data()[idx],h_prompt_out.Data()[idx],tol)==false)
      {
        std::cout<<"  GQA prompt mismatch at s="<<s<<" d="<<d
                 <<": full="<<h_full_out.Data()[idx]
                 <<" cached="<<h_prompt_out.Data()[idx]<<"\n";
        test_pass=false;
        break;
      }
    }
  }

  // Decode tokens
  for(uint32_t step=0;step<decode_steps&&test_pass==true;++step)
  {
    const uint32_t pos=prompt_len+step;
    std::vector<float> token_data(batch*config.dim);
    for(uint32_t d=0;d<config.dim;++d)
    {
      token_data[d]=full_input[pos*config.dim+d];
    }
    CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(
                               token_data.data(),{batch,1,config.dim},stream);
    CAIF_DeviceTensor token_out=mha.ForwardCached(d_token);
    CAIF_HostTensor h_token_out=token_out.ToHost();

    for(uint32_t d=0;d<config.dim;++d)
    {
      const size_t full_idx=pos*config.dim+d;
      if(FloatEqual(h_full_out.Data()[full_idx],h_token_out.Data()[d],tol)==false)
      {
        std::cout<<"  GQA step "<<step<<" mismatch at d="<<d
                 <<": full="<<h_full_out.Data()[full_idx]
                 <<" cached="<<h_token_out.Data()[d]<<"\n";
        test_pass=false;
        break;
      }
    }
  }

  ReportResult("TestGQAWithKVCache",test_pass);
}

//------------------------------------------------------------------------------
// Test 6: Batch size > 1 with KV-Cache
//------------------------------------------------------------------------------
static void TestBatchedKVCache()
{
  CAIF_CudaStream stream;

  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=32;
  config.num_heads=2;
  config.num_kv_heads=2;
  config.head_dim=16;
  config.causal=true;
  config.use_rope=false;
  config.rope_base=10000.0f;
  config.dropout_rate=0.0f;

  CAIF_DeviceMultiHeadAttention mha(config,stream);
  mha.InitializeWeights(42);

  const uint32_t batch=3;
  const uint32_t prompt_len=4;
  const uint32_t decode_steps=2;
  const uint32_t total_len=prompt_len+decode_steps;

  // Create full input
  std::vector<float> full_input(batch*total_len*config.dim);
  FillRandom(full_input,-1.0f,1.0f,333);

  // Full forward for reference
  CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(
                            full_input.data(),{batch,total_len,config.dim},stream);
  CAIF_DeviceTensor full_output=mha.Forward(d_full,false);
  CAIF_HostTensor h_full_out=full_output.ToHost();

  // Incremental decoding
  mha.EnableKVCache(batch,total_len);
  mha.ResetKVCache();

  // Process prompt
  std::vector<float> prompt_data(batch*prompt_len*config.dim);
  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t s=0;s<prompt_len;++s)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        prompt_data[b*prompt_len*config.dim+s*config.dim+d]=
          full_input[b*total_len*config.dim+s*config.dim+d];
      }
    }
  }
  CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(
                              prompt_data.data(),{batch,prompt_len,config.dim},stream);
  CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt);
  CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

  bool test_pass=true;
  const float tol=1e-4f;

  // Check prompt output for all batches
  for(uint32_t b=0;b<batch&&test_pass==true;++b)
  {
    for(uint32_t s=0;s<prompt_len&&test_pass==true;++s)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        const size_t full_idx=b*total_len*config.dim+s*config.dim+d;
        const size_t prompt_idx=b*prompt_len*config.dim+s*config.dim+d;
        if(FloatEqual(h_full_out.Data()[full_idx],h_prompt_out.Data()[prompt_idx],tol)==false)
        {
          std::cout<<"  Batch prompt mismatch at b="<<b<<" s="<<s<<" d="<<d
                   <<": full="<<h_full_out.Data()[full_idx]
                   <<" cached="<<h_prompt_out.Data()[prompt_idx]<<"\n";
          test_pass=false;
          break;
        }
      }
    }
  }

  // Decode tokens
  for(uint32_t step=0;step<decode_steps&&test_pass==true;++step)
  {
    const uint32_t pos=prompt_len+step;
    std::vector<float> token_data(batch*config.dim);
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        token_data[b*config.dim+d]=
          full_input[b*total_len*config.dim+pos*config.dim+d];
      }
    }
    CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(
                               token_data.data(),{batch,1,config.dim},stream);
    CAIF_DeviceTensor token_out=mha.ForwardCached(d_token);
    CAIF_HostTensor h_token_out=token_out.ToHost();

    for(uint32_t b=0;b<batch&&test_pass==true;++b)
    {
      for(uint32_t d=0;d<config.dim;++d)
      {
        const size_t full_idx=b*total_len*config.dim+pos*config.dim+d;
        const size_t token_idx=b*config.dim+d;
        if(FloatEqual(h_full_out.Data()[full_idx],h_token_out.Data()[token_idx],tol)==false)
        {
          std::cout<<"  Batch step "<<step<<" mismatch at b="<<b<<" d="<<d
                   <<": full="<<h_full_out.Data()[full_idx]
                   <<" cached="<<h_token_out.Data()[token_idx]<<"\n";
          test_pass=false;
          break;
        }
      }
    }
  }

  ReportResult("TestBatchedKVCache",test_pass);
}

#endif  // USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main()
{
  std::cout<<"=== KV-Cache Tests ===\n";

#ifdef USE_CAIF_CUDA
  TestKVCacheManagement();
  TestCachedMatchesNonCached();
  TestIncrementalDecoding();
  TestIncrementalDecodingWithRoPE();
  TestGQAWithKVCache();
  TestBatchedKVCache();
#else
  std::cout<<"CUDA not enabled, skipping tests.\n";
#endif

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  if(g_tests_failed>0)
  {
    std::cout<<"SOME TESTS FAILED!\n";
    return 1;
  }

  std::cout<<"All tests passed!\n";
  return 0;
}
