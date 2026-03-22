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

#include "caif_device_token_embedding.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include <iostream>
#include <vector>
#include <cmath>
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

static bool FloatEqual(float a,float b,float tolerance=1e-5f)
{
  return std::fabs(a-b)<tolerance;
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// CPU reference: gather embedding rows
//------------------------------------------------------------------------------
static void CpuEmbeddingLookup(const float *table,
                                const uint32_t *ids,
                                float *output,
                                int num_tokens,
                                int dim)
{
  for(int t=0;t<num_tokens;++t)
  {
    const uint32_t id=ids[t];
    for(int d=0;d<dim;++d)
    {
      output[t*dim+d]=table[id*dim+d];
    }
  }
}

//------------------------------------------------------------------------------
// Test 1: Forward output shape
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=3;

    CAIF_CudaStream stream;
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);

    // Create float token IDs
    std::vector<float> float_ids={0,1,2,3,4,5};
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             float_ids.data(),{batch,seq_len},stream);

    CAIF_DeviceTensor output=emb.Forward(input,false);
    const auto &shape=output.Shape();

    bool passed=(shape.size()==3&&
                 shape[0]==batch&&
                 shape[1]==seq_len&&
                 shape[2]==dim);
    if(passed==false)
    {
      std::cout<<"  Expected shape [2,3,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
    }

    ReportResult("TokenEmbedding::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward values match CPU reference
//------------------------------------------------------------------------------
static void TestForwardValues()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=3;
    constexpr uint32_t num_tokens=batch*seq_len;

    CAIF_CudaStream stream;
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);

    // Read embedding table
    CAIF_HostTensor host_table=emb.ParameterTensor(0).ToHost();

    // Token IDs
    std::vector<uint32_t> ids={0,5,10,3,7,15};
    std::vector<float> float_ids(num_tokens);
    for(uint32_t i=0;i<num_tokens;++i)
    {
      float_ids[i]=static_cast<float>(ids[i]);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             float_ids.data(),{batch,seq_len},stream);
    CAIF_DeviceTensor output=emb.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // CPU reference
    std::vector<float> expected(num_tokens*dim);
    CpuEmbeddingLookup(host_table.Data(),ids.data(),expected.data(),
                        num_tokens,dim);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("TokenEmbedding::ForwardValues",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::ForwardValues",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: ForwardFromIds matches Forward float path
//------------------------------------------------------------------------------
static void TestForwardFromIds()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=3;
    constexpr uint32_t num_tokens=batch*seq_len;

    CAIF_CudaStream stream;
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);

    std::vector<uint32_t> ids={0,5,10,3,7,15};
    std::vector<float> float_ids(num_tokens);
    for(uint32_t i=0;i<num_tokens;++i)
    {
      float_ids[i]=static_cast<float>(ids[i]);
    }

    // Float path
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             float_ids.data(),{batch,seq_len},stream);
    CAIF_DeviceTensor out_float=emb.Forward(input,false);
    CAIF_HostTensor host_float=out_float.ToHost();

    // uint32 path
    CAIF_DeviceTensor out_ids=emb.ForwardFromIds(ids.data(),batch,seq_len,false);
    CAIF_HostTensor host_ids=out_ids.ToHost();

    bool passed=true;
    for(size_t i=0;i<num_tokens*dim;++i)
    {
      if(FloatEqual(host_float.Data()[i],host_ids.Data()[i])==false)
      {
        std::cout<<"  Mismatch at "<<i<<": float="<<host_float.Data()[i]
                 <<" ids="<<host_ids.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("TokenEmbedding::ForwardFromIds",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::ForwardFromIds",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Forward bounds check (valid IDs produce correct output)
//------------------------------------------------------------------------------
static void TestForwardBoundsCheck()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);

    // Use first and last valid IDs
    std::vector<uint32_t> ids={0,1,14,15};
    CAIF_DeviceTensor output=emb.ForwardFromIds(ids.data(),batch,seq_len,false);
    CAIF_HostTensor host_output=output.ToHost();
    CAIF_HostTensor host_table=emb.ParameterTensor(0).ToHost();

    bool passed=true;
    for(uint32_t t=0;t<seq_len;++t)
    {
      for(uint32_t d=0;d<dim;++d)
      {
        const float expected=host_table.Data()[ids[t]*dim+d];
        const float actual=host_output.Data()[t*dim+d];
        if(FloatEqual(actual,expected)==false)
        {
          std::cout<<"  Mismatch at token "<<t<<" dim "<<d
                   <<": got "<<actual<<" expected "<<expected<<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    ReportResult("TokenEmbedding::ForwardBoundsCheck",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::ForwardBoundsCheck",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Backward weight gradient (finite-difference)
//------------------------------------------------------------------------------
static void TestBackwardWeightGrad()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=3;
    constexpr uint32_t num_tokens=batch*seq_len;
    constexpr float h=1e-3f;
    constexpr float grad_tol=5e-2f;

    CAIF_CudaStream stream;

    // Read initial embedding table
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);
    CAIF_HostTensor host_table=emb.ParameterTensor(0).ToHost();
    std::vector<float> table_data(host_table.Data(),
                                  host_table.Data()+vocab_size*dim);

    std::vector<uint32_t> ids={0,5,10,3,7,15};

    // Analytical backward
    CAIF_DeviceTensor out=emb.ForwardFromIds(ids.data(),batch,seq_len,true);
    std::vector<float> grad_ones(num_tokens*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),
                                {batch,seq_len,dim},stream);
    emb.Backward(grad_out);
    CAIF_HostTensor host_grad=emb.GradientTensor(0).ToHost();

    // Finite-difference for a subset of table entries
    bool passed=true;
    for(uint32_t token_idx=0;token_idx<num_tokens&&passed==true;++token_idx)
    {
      const uint32_t id=ids[token_idx];
      for(uint32_t d=0;d<dim&&passed==true;++d)
      {
        const size_t idx=id*dim+d;

        // f(x+h)
        std::vector<float> table_plus(table_data);
        table_plus[idx]+=h;
        CAIF_DeviceTokenEmbedding emb_p(config,stream);
        emb_p.ParameterTensor(0).CopyFromHost(table_plus.data(),vocab_size*dim);
        CAIF_DeviceTensor out_p=emb_p.ForwardFromIds(
                                 ids.data(),batch,seq_len,false);
        CAIF_HostTensor hout_p=out_p.ToHost();
        float sum_plus=0.0f;
        for(size_t j=0;j<num_tokens*dim;++j)
        {
          sum_plus+=hout_p.Data()[j];
        }

        // f(x-h)
        std::vector<float> table_minus(table_data);
        table_minus[idx]-=h;
        CAIF_DeviceTokenEmbedding emb_m(config,stream);
        emb_m.ParameterTensor(0).CopyFromHost(table_minus.data(),vocab_size*dim);
        CAIF_DeviceTensor out_m=emb_m.ForwardFromIds(
                                 ids.data(),batch,seq_len,false);
        CAIF_HostTensor hout_m=out_m.ToHost();
        float sum_minus=0.0f;
        for(size_t j=0;j<num_tokens*dim;++j)
        {
          sum_minus+=hout_m.Data()[j];
        }

        const float numerical=(sum_plus-sum_minus)/(2.0f*h);
        const float analytical=host_grad.Data()[idx];

        if(std::fabs(numerical-analytical)>grad_tol)
        {
          std::cout<<"  Grad mismatch at table["<<id<<"]["<<d
                   <<"]: analytical="<<analytical
                   <<" numerical="<<numerical<<"\n";
          passed=false;
        }
      }
    }

    ReportResult("TokenEmbedding::BackwardWeightGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::BackwardWeightGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Parameter count
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;

    CAIF_CudaStream stream;
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);

    bool passed=true;
    if(emb.ParameterTensorCount()!=1)
    {
      std::cout<<"  ParameterTensorCount expected 1, got "
               <<emb.ParameterTensorCount()<<"\n";
      passed=false;
    }
    if(emb.TotalParameterCount()!=vocab_size*dim)
    {
      std::cout<<"  TotalParameterCount expected "<<vocab_size*dim
               <<", got "<<emb.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("TokenEmbedding::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: ZeroGradients
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=1;
    constexpr uint32_t seq_len=3;

    CAIF_CudaStream stream;
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);

    // Run forward+backward to produce non-zero gradients
    std::vector<uint32_t> ids={0,1,2};
    CAIF_DeviceTensor out=emb.ForwardFromIds(ids.data(),batch,seq_len,true);
    std::vector<float> grad_ones(seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    emb.Backward(grad_out);

    // Zero gradients
    emb.ZeroGradients();

    CAIF_HostTensor host_grad=emb.GradientTensor(0).ToHost();
    bool passed=true;
    for(size_t i=0;i<vocab_size*dim;++i)
    {
      if(host_grad.Data()[i]!=0.0f)
      {
        std::cout<<"  Gradient not zeroed at "<<i<<": "
                 <<host_grad.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("TokenEmbedding::ZeroGradients",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::ZeroGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    constexpr uint32_t vocab_size=16;
    constexpr uint32_t dim=8;

    CAIF_CudaStream stream;
    CAIF_DeviceTokenEmbedding::Config_t config{vocab_size,dim};
    CAIF_DeviceTokenEmbedding emb(config,stream);

    const std::string desc=emb.Description();
    const std::string expected="TokenEmbedding(vocab=16,dim=8)";
    bool passed=(desc==expected);
    if(passed==false)
    {
      std::cout<<"  Expected '"<<expected<<"', got '"<<desc<<"'\n";
    }

    ReportResult("TokenEmbedding::Description",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TokenEmbedding::Description",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF_DeviceTokenEmbedding Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestForwardShape();
  TestForwardValues();
  TestForwardFromIds();
  TestForwardBoundsCheck();
  TestBackwardWeightGrad();
  TestParameterCount();
  TestZeroGradients();
  TestDescription();
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
