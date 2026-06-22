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
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <random>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr float g_caif_kvcache_test_tol=5e-4f;
constexpr float g_caif_kvcache_test_min_val=-1.0f;
constexpr float g_caif_kvcache_test_max_val=1.0f;
constexpr uint32_t g_caif_kvcache_test_seed_1=123;
constexpr uint32_t g_caif_kvcache_test_seed_2=999;
constexpr uint32_t g_caif_kvcache_test_seed_3=777;
constexpr uint32_t g_caif_kvcache_test_seed_4=888;
constexpr uint32_t g_caif_kvcache_test_seed_5=555;
constexpr uint32_t g_caif_kvcache_test_seed_6=333;
constexpr float g_caif_kvcache_test_rope_base=10000.0f;
constexpr float g_caif_kvcache_test_dropout=0.0f;

//------------------------------------------------------------------------------
// KV-Cache correctness and management tests.
//------------------------------------------------------------------------------
class CAIF_KVCacheTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void FillRandom(std::vector<float> &data,
                           const float min_val,
                           const float max_val,
                           const uint32_t seed);

    static void TestKVCacheManagement();
    static void TestCachedMatchesNonCached();
    static void TestIncrementalDecoding();
    static void TestIncrementalDecodingWithRoPE();
    static void TestGQAWithKVCache();
    static void TestBatchedKVCache();
};

void CAIF_KVCacheTests::FillRandom(std::vector<float> &data,
                                   const float min_val,
                                   const float max_val,
                                   const uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(min_val,max_val);
  for(size_t i=0;i<data.size();++i)
  {
    data[i]=dist(gen);
  }
}

//------------------------------------------------------------------------------
// Test 1: Enable/Disable/Reset KV-Cache
//------------------------------------------------------------------------------
void CAIF_KVCacheTests::TestKVCacheManagement()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    CAIF_DeviceMultiHeadAttentionConfig config(64,
                                               4,
                                               4,
                                               16,
                                               true,
                                               false,
                                               g_caif_kvcache_test_rope_base,
                                               g_caif_kvcache_test_dropout);

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    mha.InitializeWeights(42);

    bool test_pass=true;

    // Initially disabled
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
    std::vector<float> input_data(batch*seq_len*config.Dim());
    FillRandom(input_data,g_caif_kvcache_test_min_val,g_caif_kvcache_test_max_val,
               g_caif_kvcache_test_seed_1);
    CAIF_DeviceTensor d_input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                               {batch,seq_len,config.Dim()},
                                                               stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=mha.ForwardCached(d_input,ctx);
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

    CAIF_TestHarness::Report("KVCache::Management",test_pass);
  }
  CAIF_TEST_CATCH_BLOCK("KVCache::Management")
}

//------------------------------------------------------------------------------
// Test 2: Cached forward matches non-cached for full prompt.
// When we process the entire sequence in one ForwardCached call, the output
// should match Forward() with the same input.
//------------------------------------------------------------------------------
void CAIF_KVCacheTests::TestCachedMatchesNonCached()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    CAIF_DeviceMultiHeadAttentionConfig config(32,
                                               2,
                                               2,
                                               16,
                                               true,
                                               false,
                                               g_caif_kvcache_test_rope_base,
                                               g_caif_kvcache_test_dropout);

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    mha.InitializeWeights(42);

    const uint32_t batch=1;
    const uint32_t seq_len=8;

    // Create input
    std::vector<float> input_data(batch*seq_len*config.Dim());
    FillRandom(input_data,g_caif_kvcache_test_min_val,g_caif_kvcache_test_max_val,
               g_caif_kvcache_test_seed_2);
    CAIF_DeviceTensor d_input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                               {batch,seq_len,config.Dim()},
                                                               stream);

    // Non-cached forward
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output_nocache=mha.Forward(d_input,ctx);
    CAIF_HostTensor h_out_nocache=output_nocache.ToHost();

    // Cached forward (full prompt)
    mha.EnableKVCache(batch,seq_len);
    mha.ResetKVCache();
    CAIF_DeviceTensor output_cached=mha.ForwardCached(d_input,ctx);
    CAIF_HostTensor h_out_cached=output_cached.ToHost();

    // Compare outputs
    bool test_pass=true;
    const size_t total=batch*seq_len*config.Dim();
    for(size_t i=0;i<total;++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_out_nocache.Data()[i],
                                       h_out_cached.Data()[i],
                                       g_caif_kvcache_test_tol)==false)
      {
        ISE_Out::Out()<<"  Mismatch at "
                      <<i
                      <<": nocache="
                      <<h_out_nocache.Data()[i]
                      <<" cached="
                      <<h_out_cached.Data()[i]
                      <<"\n";
        test_pass=false;
        break;
      }
    }

    CAIF_TestHarness::Report("KVCache::CachedMatchesNonCached",test_pass);
  }
  CAIF_TEST_CATCH_BLOCK("KVCache::CachedMatchesNonCached")
}

//------------------------------------------------------------------------------
// Test 3: Incremental decoding produces same output as full forward.
// Process prompt, then decode one token at a time. The output at each
// position should match what we'd get from Forward() on the full sequence
// up to that point.
//------------------------------------------------------------------------------
void CAIF_KVCacheTests::TestIncrementalDecoding()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    CAIF_DeviceMultiHeadAttentionConfig config(32,
                                               2,
                                               2,
                                               16,
                                               true,
                                               false,
                                               g_caif_kvcache_test_rope_base,
                                               g_caif_kvcache_test_dropout);

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    mha.InitializeWeights(42);

    const uint32_t batch=1;
    const uint32_t prompt_len=4;
    const uint32_t decode_steps=3;
    const uint32_t total_len=prompt_len+decode_steps;

    // Create full input
    std::vector<float> full_input(batch*total_len*config.Dim());
    FillRandom(full_input,g_caif_kvcache_test_min_val,g_caif_kvcache_test_max_val,
               g_caif_kvcache_test_seed_3);

    // Full forward for reference
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(full_input.data(),
                                                              {batch,total_len,config.Dim()},
                                                              stream);
    CAIF_DeviceTensor full_output=mha.Forward(d_full,ctx);
    CAIF_HostTensor h_full_out=full_output.ToHost();

    // Incremental decoding
    mha.EnableKVCache(batch,total_len);
    mha.ResetKVCache();

    // Process prompt
    std::vector<float> prompt_data(batch*prompt_len*config.Dim());
    for(uint32_t i=0;i<prompt_data.size();++i)
    {
      prompt_data[i]=full_input[i];
    }
    CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(prompt_data.data(),
                                                                {batch,prompt_len,config.Dim()},
                                                                stream);
    CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt,ctx);
    CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

    bool test_pass=true;

    // Check prompt output matches full output prefix
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t s=0;s<prompt_len;++s)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          const size_t idx=b*total_len*config.Dim()+s*config.Dim()+d;
          const size_t prompt_idx=b*prompt_len*config.Dim()+s*config.Dim()+d;
          if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[idx],
                                           h_prompt_out.Data()[prompt_idx],
                                           g_caif_kvcache_test_tol)==false)
          {
            ISE_Out::Out()<<"  Prompt mismatch at b="
                          <<b
                          <<" s="
                          <<s
                          <<" d="
                          <<d
                          <<": full="
                          <<h_full_out.Data()[idx]
                          <<" incremental="
                          <<h_prompt_out.Data()[prompt_idx]
                          <<"\n";
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
      std::vector<float> token_data(batch*config.Dim());
      for(uint32_t b=0;b<batch;++b)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          token_data[b*config.Dim()+d]=full_input[b*total_len*config.Dim()+pos*config.Dim()+d];
        }
      }
      CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(token_data.data(),
                                                                  {batch,1,config.Dim()},
                                                                  stream);
      CAIF_DeviceTensor token_out=mha.ForwardCached(d_token,ctx);
      CAIF_HostTensor h_token_out=token_out.ToHost();

      // Check output matches full forward at this position
      for(uint32_t b=0;b<batch;++b)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          const size_t full_idx=b*total_len*config.Dim()+pos*config.Dim()+d;
          const size_t token_idx=b*config.Dim()+d;
          if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[full_idx],
                                           h_token_out.Data()[token_idx],
                                           g_caif_kvcache_test_tol)==false)
          {
            ISE_Out::Out()<<"  Step "
                          <<step
                          <<" mismatch at b="
                          <<b
                          <<" d="
                          <<d
                          <<": full="
                          <<h_full_out.Data()[full_idx]
                          <<" incremental="
                          <<h_token_out.Data()[token_idx]
                          <<"\n";
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

    CAIF_TestHarness::Report("KVCache::IncrementalDecoding",test_pass);
  }
  CAIF_TEST_CATCH_BLOCK("KVCache::IncrementalDecoding")
}

//------------------------------------------------------------------------------
// Test 4: Incremental decoding with RoPE.
// Same as Test 3 but with RoPE enabled. Position offsets must be correct.
//------------------------------------------------------------------------------
void CAIF_KVCacheTests::TestIncrementalDecodingWithRoPE()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    CAIF_DeviceMultiHeadAttentionConfig config(32,
                                               2,
                                               2,
                                               16,
                                               true,
                                               true,
                                               g_caif_kvcache_test_rope_base,
                                               g_caif_kvcache_test_dropout);

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    mha.InitializeWeights(42);

    const uint32_t batch=1;
    const uint32_t prompt_len=4;
    const uint32_t decode_steps=2;
    const uint32_t total_len=prompt_len+decode_steps;

    // Create full input
    std::vector<float> full_input(batch*total_len*config.Dim());
    FillRandom(full_input,g_caif_kvcache_test_min_val,g_caif_kvcache_test_max_val,
               g_caif_kvcache_test_seed_4);

    // Full forward for reference
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(full_input.data(),
                                                              {batch,total_len,config.Dim()},
                                                              stream);
    CAIF_DeviceTensor full_output=mha.Forward(d_full,ctx);
    CAIF_HostTensor h_full_out=full_output.ToHost();

    // Incremental decoding
    mha.EnableKVCache(batch,total_len);
    mha.ResetKVCache();

    // Process prompt
    std::vector<float> prompt_data(batch*prompt_len*config.Dim());
    for(uint32_t i=0;i<prompt_data.size();++i)
    {
      prompt_data[i]=full_input[i];
    }
    CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(prompt_data.data(),
                                                                {batch,prompt_len,config.Dim()},
                                                                stream);
    CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt,ctx);
    CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

    bool test_pass=true;

    // Check prompt output matches full output prefix
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t s=0;s<prompt_len;++s)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          const size_t idx=b*total_len*config.Dim()+s*config.Dim()+d;
          const size_t prompt_idx=b*prompt_len*config.Dim()+s*config.Dim()+d;
          if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[idx],
                                           h_prompt_out.Data()[prompt_idx],
                                           g_caif_kvcache_test_tol)==false)
          {
            ISE_Out::Out()<<"  Prompt mismatch at b="
                          <<b
                          <<" s="
                          <<s
                          <<" d="
                          <<d
                          <<": full="
                          <<h_full_out.Data()[idx]
                          <<" incremental="
                          <<h_prompt_out.Data()[prompt_idx]
                          <<"\n";
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
      std::vector<float> token_data(batch*config.Dim());
      for(uint32_t b=0;b<batch;++b)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          token_data[b*config.Dim()+d]=full_input[b*total_len*config.Dim()+pos*config.Dim()+d];
        }
      }
      CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(token_data.data(),
                                                                  {batch,1,config.Dim()},
                                                                  stream);
      CAIF_DeviceTensor token_out=mha.ForwardCached(d_token,ctx);
      CAIF_HostTensor h_token_out=token_out.ToHost();

      // Check output matches full forward at this position
      for(uint32_t b=0;b<batch;++b)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          const size_t full_idx=b*total_len*config.Dim()+pos*config.Dim()+d;
          const size_t token_idx=b*config.Dim()+d;
          if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[full_idx],
                                           h_token_out.Data()[token_idx],
                                           g_caif_kvcache_test_tol)==false)
          {
            ISE_Out::Out()<<"  RoPE step "
                          <<step
                          <<" mismatch at b="
                          <<b
                          <<" d="
                          <<d
                          <<": full="
                          <<h_full_out.Data()[full_idx]
                          <<" incremental="
                          <<h_token_out.Data()[token_idx]
                          <<"\n";
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

    CAIF_TestHarness::Report("KVCache::IncrementalDecodingWithRoPE",test_pass);
  }
  CAIF_TEST_CATCH_BLOCK("KVCache::IncrementalDecodingWithRoPE")
}

//------------------------------------------------------------------------------
// Test 5: GQA with KV-Cache.
// Test incremental decoding when num_kv_heads < num_heads (grouped-query).
//------------------------------------------------------------------------------
void CAIF_KVCacheTests::TestGQAWithKVCache()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // GQA: 4 query heads share 2 kv heads
    CAIF_DeviceMultiHeadAttentionConfig config(64,
                                               4,
                                               2,
                                               16,
                                               true,
                                               false,
                                               g_caif_kvcache_test_rope_base,
                                               g_caif_kvcache_test_dropout);

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    mha.InitializeWeights(42);

    const uint32_t batch=1;
    const uint32_t prompt_len=3;
    const uint32_t decode_steps=2;
    const uint32_t total_len=prompt_len+decode_steps;

    // Create full input
    std::vector<float> full_input(batch*total_len*config.Dim());
    FillRandom(full_input,g_caif_kvcache_test_min_val,g_caif_kvcache_test_max_val,
               g_caif_kvcache_test_seed_5);

    // Full forward for reference
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(full_input.data(),
                                                              {batch,total_len,config.Dim()},
                                                              stream);
    CAIF_DeviceTensor full_output=mha.Forward(d_full,ctx);
    CAIF_HostTensor h_full_out=full_output.ToHost();

    // Incremental decoding
    mha.EnableKVCache(batch,total_len);
    mha.ResetKVCache();

    // Process prompt
    std::vector<float> prompt_data(batch*prompt_len*config.Dim());
    for(uint32_t i=0;i<prompt_data.size();++i)
    {
      prompt_data[i]=full_input[i];
    }
    CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(prompt_data.data(),
                                                                {batch,prompt_len,config.Dim()},
                                                                stream);
    CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt,ctx);
    CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

    bool test_pass=true;

    // Check prompt output
    for(uint32_t s=0;s<prompt_len&&test_pass==true;++s)
    {
      for(uint32_t d=0;d<config.Dim();++d)
      {
        const size_t idx=s*config.Dim()+d;
        if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[idx],
                                         h_prompt_out.Data()[idx],
                                         g_caif_kvcache_test_tol)==false)
        {
          ISE_Out::Out()<<"  GQA prompt mismatch at s="
                        <<s
                        <<" d="
                        <<d
                        <<": full="
                        <<h_full_out.Data()[idx]
                        <<" cached="
                        <<h_prompt_out.Data()[idx]
                        <<"\n";
          test_pass=false;
          break;
        }
      }
    }

    // Decode tokens
    for(uint32_t step=0;step<decode_steps&&test_pass==true;++step)
    {
      const uint32_t pos=prompt_len+step;
      std::vector<float> token_data(batch*config.Dim());
      for(uint32_t d=0;d<config.Dim();++d)
      {
        token_data[d]=full_input[pos*config.Dim()+d];
      }
      CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(token_data.data(),
                                                                  {batch,1,config.Dim()},
                                                                  stream);
      CAIF_DeviceTensor token_out=mha.ForwardCached(d_token,ctx);
      CAIF_HostTensor h_token_out=token_out.ToHost();

      for(uint32_t d=0;d<config.Dim();++d)
      {
        const size_t full_idx=pos*config.Dim()+d;
        if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[full_idx],
                                         h_token_out.Data()[d],
                                         g_caif_kvcache_test_tol)==false)
        {
          ISE_Out::Out()<<"  GQA step "
                        <<step
                        <<" mismatch at d="
                        <<d
                        <<": full="
                        <<h_full_out.Data()[full_idx]
                        <<" cached="
                        <<h_token_out.Data()[d]
                        <<"\n";
          test_pass=false;
          break;
        }
      }
    }

    CAIF_TestHarness::Report("KVCache::GQAWithKVCache",test_pass);
  }
  CAIF_TEST_CATCH_BLOCK("KVCache::GQAWithKVCache")
}

//------------------------------------------------------------------------------
// Test 6: Batch size > 1 with KV-Cache
//------------------------------------------------------------------------------
void CAIF_KVCacheTests::TestBatchedKVCache()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    CAIF_DeviceMultiHeadAttentionConfig config(32,
                                               2,
                                               2,
                                               16,
                                               true,
                                               false,
                                               g_caif_kvcache_test_rope_base,
                                               g_caif_kvcache_test_dropout);

    CAIF_DeviceMultiHeadAttention<float,float> mha(config,stream);
    mha.InitializeWeights(42);

    const uint32_t batch=3;
    const uint32_t prompt_len=4;
    const uint32_t decode_steps=2;
    const uint32_t total_len=prompt_len+decode_steps;

    // Create full input
    std::vector<float> full_input(batch*total_len*config.Dim());
    FillRandom(full_input,g_caif_kvcache_test_min_val,g_caif_kvcache_test_max_val,
               g_caif_kvcache_test_seed_6);

    // Full forward for reference
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor d_full=CAIF_DeviceTensor::FromHostData(full_input.data(),
                                                              {batch,total_len,config.Dim()},
                                                              stream);
    CAIF_DeviceTensor full_output=mha.Forward(d_full,ctx);
    CAIF_HostTensor h_full_out=full_output.ToHost();

    // Incremental decoding
    mha.EnableKVCache(batch,total_len);
    mha.ResetKVCache();

    // Process prompt
    std::vector<float> prompt_data(batch*prompt_len*config.Dim());
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t s=0;s<prompt_len;++s)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          prompt_data[b*prompt_len*config.Dim()+s*config.Dim()+d]=
            full_input[b*total_len*config.Dim()+s*config.Dim()+d];
        }
      }
    }
    CAIF_DeviceTensor d_prompt=CAIF_DeviceTensor::FromHostData(prompt_data.data(),
                                                                {batch,prompt_len,config.Dim()},
                                                                stream);
    CAIF_DeviceTensor prompt_out=mha.ForwardCached(d_prompt,ctx);
    CAIF_HostTensor h_prompt_out=prompt_out.ToHost();

    bool test_pass=true;

    // Check prompt output for all batches
    for(uint32_t b=0;b<batch&&test_pass==true;++b)
    {
      for(uint32_t s=0;s<prompt_len&&test_pass==true;++s)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          const size_t full_idx=b*total_len*config.Dim()+s*config.Dim()+d;
          const size_t prompt_idx=b*prompt_len*config.Dim()+s*config.Dim()+d;
          if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[full_idx],
                                           h_prompt_out.Data()[prompt_idx],
                                           g_caif_kvcache_test_tol)==false)
          {
            ISE_Out::Out()<<"  Batch prompt mismatch at b="
                          <<b
                          <<" s="
                          <<s
                          <<" d="
                          <<d
                          <<": full="
                          <<h_full_out.Data()[full_idx]
                          <<" cached="
                          <<h_prompt_out.Data()[prompt_idx]
                          <<"\n";
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
      std::vector<float> token_data(batch*config.Dim());
      for(uint32_t b=0;b<batch;++b)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          token_data[b*config.Dim()+d]=full_input[b*total_len*config.Dim()+pos*config.Dim()+d];
        }
      }
      CAIF_DeviceTensor d_token=CAIF_DeviceTensor::FromHostData(token_data.data(),
                                                                  {batch,1,config.Dim()},
                                                                  stream);
      CAIF_DeviceTensor token_out=mha.ForwardCached(d_token,ctx);
      CAIF_HostTensor h_token_out=token_out.ToHost();

      for(uint32_t b=0;b<batch&&test_pass==true;++b)
      {
        for(uint32_t d=0;d<config.Dim();++d)
        {
          const size_t full_idx=b*total_len*config.Dim()+pos*config.Dim()+d;
          const size_t token_idx=b*config.Dim()+d;
          if(CAIF_TestHarness::FloatEqual(h_full_out.Data()[full_idx],
                                           h_token_out.Data()[token_idx],
                                           g_caif_kvcache_test_tol)==false)
          {
            ISE_Out::Out()<<"  Batch step "
                          <<step
                          <<" mismatch at b="
                          <<b
                          <<" d="
                          <<d
                          <<": full="
                          <<h_full_out.Data()[full_idx]
                          <<" cached="
                          <<h_token_out.Data()[token_idx]
                          <<"\n";
            test_pass=false;
            break;
          }
        }
      }
    }

    CAIF_TestHarness::Report("KVCache::BatchedKVCache",test_pass);
  }
  CAIF_TEST_CATCH_BLOCK("KVCache::BatchedKVCache")
}

void CAIF_KVCacheTests::RunAll()
{
  ISE_Out::Out()<<"=== KV-Cache Tests ==="
                <<"\n\n";
  TestKVCacheManagement();
  TestCachedMatchesNonCached();
  TestIncrementalDecoding();
  TestIncrementalDecodingWithRoPE();
  TestGQAWithKVCache();
  TestBatchedKVCache();
  ISE_Out::Out()<<"\n=== Summary ===\n"
                <<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"\n"
                <<"Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_KVCacheTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  instance::ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
                           <<"\n";
  return 0;
#endif
}
