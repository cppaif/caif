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

static bool FloatEqual(float a,float b,float tolerance=1e-3f)
{
  return std::fabs(a-b)<tolerance;
}

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// CPU reference helpers
//------------------------------------------------------------------------------

static void CpuSoftmax(float *data,int rows,int cols)
{
  for(int r=0;r<rows;++r)
  {
    float *row=data+r*cols;
    float max_val=row[0];
    for(int c=1;c<cols;++c)
    {
      if(row[c]>max_val)
      {
        max_val=row[c];
      }
    }
    float sum=0.0f;
    for(int c=0;c<cols;++c)
    {
      row[c]=std::exp(row[c]-max_val);
      sum+=row[c];
    }
    for(int c=0;c<cols;++c)
    {
      row[c]/=sum;
    }
  }
}

static void CpuMatMul(const float *a,const float *b,float *c,
                       int m,int k,int n)
{
  for(int i=0;i<m;++i)
  {
    for(int j=0;j<n;++j)
    {
      float sum=0.0f;
      for(int p=0;p<k;++p)
      {
        sum+=a[i*k+p]*b[p*n+j];
      }
      c[i*n+j]=sum;
    }
  }
}

static void CpuMatMulTransposeB(const float *a,const float *b,float *c,
                                 int m,int k,int n)
{
  for(int i=0;i<m;++i)
  {
    for(int j=0;j<n;++j)
    {
      float sum=0.0f;
      for(int p=0;p<k;++p)
      {
        sum+=a[i*k+p]*b[j*k+p];
      }
      c[i*n+j]=sum;
    }
  }
}

/**
 * CPU reference MHA with GQA (num_kv_heads may differ from num_heads).
 * K/V are projected with kv_dim, then KV heads are repeated for attention.
 */
static void CpuMHAWithGQA(const float *input,
                            const float *w_q,
                            const float *w_k,
                            const float *w_v,
                            const float *w_o,
                            float *output,
                            int batch,
                            int seq_len,
                            int dim,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            bool causal)
{
  const int bs=batch*seq_len;
  const int qk_dim=num_heads*head_dim;
  const int kv_dim=num_kv_heads*head_dim;
  const int repeat_factor=num_heads/num_kv_heads;

  // Project Q: [bs, dim] @ [dim, qk_dim] -> [bs, qk_dim]
  std::vector<float> q_proj(bs*qk_dim);
  CpuMatMul(input,w_q,q_proj.data(),bs,dim,qk_dim);

  // Project K, V: [bs, dim] @ [dim, kv_dim] -> [bs, kv_dim]
  std::vector<float> k_proj(bs*kv_dim);
  std::vector<float> v_proj(bs*kv_dim);
  CpuMatMul(input,w_k,k_proj.data(),bs,dim,kv_dim);
  CpuMatMul(input,w_v,v_proj.data(),bs,dim,kv_dim);

  // Split heads and compute attention
  std::vector<float> concat(bs*qk_dim,0.0f);

  for(int b=0;b<batch;++b)
  {
    for(int h=0;h<num_heads;++h)
    {
      const int kv_h=h/repeat_factor;

      // Extract Q for this head: [seq_len, head_dim]
      std::vector<float> q_head(seq_len*head_dim);
      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          q_head[s*head_dim+d]=q_proj[(b*seq_len+s)*qk_dim+h*head_dim+d];
        }
      }

      // Extract K, V for the corresponding kv_head: [seq_len, head_dim]
      std::vector<float> k_head(seq_len*head_dim);
      std::vector<float> v_head(seq_len*head_dim);
      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          k_head[s*head_dim+d]=k_proj[(b*seq_len+s)*kv_dim+kv_h*head_dim+d];
          v_head[s*head_dim+d]=v_proj[(b*seq_len+s)*kv_dim+kv_h*head_dim+d];
        }
      }

      // scores = Q @ K^T -> [seq_len, seq_len]
      std::vector<float> scores(seq_len*seq_len);
      CpuMatMulTransposeB(q_head.data(),k_head.data(),scores.data(),
                           seq_len,head_dim,seq_len);

      // Scale
      const float scale=1.0f/std::sqrt(static_cast<float>(head_dim));
      for(int i=0;i<seq_len*seq_len;++i)
      {
        scores[i]*=scale;
      }

      // Causal mask
      if(causal==true)
      {
        for(int i=0;i<seq_len;++i)
        {
          for(int j=i+1;j<seq_len;++j)
          {
            scores[i*seq_len+j]=-1e9f;
          }
        }
      }

      // Softmax
      CpuSoftmax(scores.data(),seq_len,seq_len);

      // context = attn @ V -> [seq_len, head_dim]
      std::vector<float> ctx(seq_len*head_dim);
      CpuMatMul(scores.data(),v_head.data(),ctx.data(),
                 seq_len,seq_len,head_dim);

      // Write to concat
      for(int s=0;s<seq_len;++s)
      {
        for(int d=0;d<head_dim;++d)
        {
          concat[(b*seq_len+s)*qk_dim+h*head_dim+d]=ctx[s*head_dim+d];
        }
      }
    }
  }

  // Output projection: [bs, qk_dim] @ [qk_dim, dim] -> [bs, dim]
  CpuMatMul(concat.data(),w_o,output,bs,qk_dim,dim);
}

//------------------------------------------------------------------------------
// Helper: create GQA config
//------------------------------------------------------------------------------
static CAIF_DeviceMultiHeadAttention::AttentionConfig_t MakeGQAConfig(
  uint32_t dim,uint32_t num_heads,uint32_t num_kv_heads,uint32_t head_dim,
  bool causal)
{
  CAIF_DeviceMultiHeadAttention::AttentionConfig_t config;
  config.dim=dim;
  config.num_heads=num_heads;
  config.num_kv_heads=num_kv_heads;
  config.head_dim=head_dim;
  config.causal=causal;
  config.use_rope=false;
  config.rope_base=g_caif_rope_default_base;
  config.dropout_rate=0.0f;
  return config;
}

//------------------------------------------------------------------------------
// Test 1: GQA forward output shape correct
//------------------------------------------------------------------------------
static void TestGQAForwardShape()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,false);
    CAIF_DeviceMultiHeadAttention mha(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=mha.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      std::cout<<"  Shape mismatch\n";
      passed=false;
    }

    ReportResult("GQA::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: GQA forward vs CPU reference (heads=4, kv_heads=2)
//------------------------------------------------------------------------------
static void TestGQAForwardVsCPU()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,false);
    CAIF_DeviceMultiHeadAttention mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=mha.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    std::vector<float> expected(batch*seq_len*dim);
    CpuMHAWithGQA(host_input.data(),
                   h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                   expected.data(),
                   batch,seq_len,dim,num_heads,num_kv_heads,head_dim,false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i],1e-3f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]
                 <<" diff="<<std::fabs(host_output.Data()[i]-expected[i])<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("GQA::ForwardVsCPU",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::ForwardVsCPU",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: GQA Multi-Query Attention (kv_heads=1) matches CPU reference
//------------------------------------------------------------------------------
static void TestGQAForwardMQA()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=1;
    const uint32_t head_dim=2;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,false);
    CAIF_DeviceMultiHeadAttention mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=mha.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    std::vector<float> expected(batch*seq_len*dim);
    CpuMHAWithGQA(host_input.data(),
                   h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                   expected.data(),
                   batch,seq_len,dim,num_heads,num_kv_heads,head_dim,false);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i],1e-3f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]
                 <<" diff="<<std::fabs(host_output.Data()[i]-expected[i])<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("GQA::ForwardMQA",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::ForwardMQA",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: GQA + causal matches CPU reference
//------------------------------------------------------------------------------
static void TestGQAForwardCausal()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,true);
    CAIF_DeviceMultiHeadAttention mha(config,stream);

    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor output=mha.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    std::vector<float> expected(batch*seq_len*dim);
    CpuMHAWithGQA(host_input.data(),
                   h_wq.Data(),h_wk.Data(),h_wv.Data(),h_wo.Data(),
                   expected.data(),
                   batch,seq_len,dim,num_heads,num_kv_heads,head_dim,true);

    bool passed=true;
    for(size_t i=0;i<expected.size();++i)
    {
      if(FloatEqual(host_output.Data()[i],expected[i],1e-3f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": got "<<host_output.Data()[i]
                 <<" expected "<<expected[i]
                 <<" diff="<<std::fabs(host_output.Data()[i]-expected[i])<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("GQA::ForwardCausal",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::ForwardCausal",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: GQA backward input gradient (finite difference)
//------------------------------------------------------------------------------
static void TestGQABackwardInputGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,false);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    CAIF_DeviceMultiHeadAttention mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    mha.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor grad_input=mha.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      CAIF_DeviceMultiHeadAttention mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_p.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_plus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DeviceMultiHeadAttention mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_m.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_minus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dx mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical
                 <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    ReportResult("GQA::BackwardInputGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::BackwardInputGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: GQA backward weight gradient (finite difference on W_q)
//------------------------------------------------------------------------------
static void TestGQABackwardWeightGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,false);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    CAIF_DeviceMultiHeadAttention mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    mha.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    mha.Backward(grad_out);
    CAIF_HostTensor host_grad_wq=mha.GradientTensor(0).ToHost();

    bool passed=true;
    const size_t check_count=4;
    std::vector<float> wq_data(h_wq.Data(),h_wq.Data()+h_wq.TotalElements());

    for(size_t i=0;i<check_count&&passed==true;++i)
    {
      std::vector<float> wq_plus(wq_data);
      std::vector<float> wq_minus(wq_data);
      wq_plus[i]+=h;
      wq_minus[i]-=h;

      CAIF_DeviceMultiHeadAttention mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(wq_plus.data(),wq_plus.size());
      mha_p.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DeviceMultiHeadAttention mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(wq_minus.data(),wq_minus.size());
      mha_m.ParameterTensor(1).CopyFromHost(h_wk.Data(),h_wk.TotalElements());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_wq.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dW_q mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical
                 <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    ReportResult("GQA::BackwardWeightGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::BackwardWeightGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: GQA backward W_k gradient (critical: must sum over repeated groups)
//------------------------------------------------------------------------------
static void TestGQABackwardWeightGrad_K()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,false);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    CAIF_DeviceMultiHeadAttention mha(config,stream);
    CAIF_HostTensor h_wq=mha.ParameterTensor(0).ToHost();
    CAIF_HostTensor h_wk=mha.ParameterTensor(1).ToHost();
    CAIF_HostTensor h_wv=mha.ParameterTensor(2).ToHost();
    CAIF_HostTensor h_wo=mha.ParameterTensor(3).ToHost();

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    mha.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    mha.Backward(grad_out);
    CAIF_HostTensor host_grad_wk=mha.GradientTensor(1).ToHost();

    bool passed=true;
    const size_t check_count=4;
    std::vector<float> wk_data(h_wk.Data(),h_wk.Data()+h_wk.TotalElements());

    for(size_t i=0;i<check_count&&passed==true;++i)
    {
      std::vector<float> wk_plus(wk_data);
      std::vector<float> wk_minus(wk_data);
      wk_plus[i]+=h;
      wk_minus[i]-=h;

      CAIF_DeviceMultiHeadAttention mha_p(config,stream);
      mha_p.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_p.ParameterTensor(1).CopyFromHost(wk_plus.data(),wk_plus.size());
      mha_p.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_p.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=mha_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      CAIF_DeviceMultiHeadAttention mha_m(config,stream);
      mha_m.ParameterTensor(0).CopyFromHost(h_wq.Data(),h_wq.TotalElements());
      mha_m.ParameterTensor(1).CopyFromHost(wk_minus.data(),wk_minus.size());
      mha_m.ParameterTensor(2).CopyFromHost(h_wv.Data(),h_wv.TotalElements());
      mha_m.ParameterTensor(3).CopyFromHost(h_wo.Data(),h_wo.TotalElements());

      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=mha_m.Forward(inp_m,false);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad_wk.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol)
      {
        std::cout<<"  dW_k mismatch at "<<i<<": analytical="<<analytical
                 <<" numerical="<<numerical
                 <<" diff="<<std::fabs(numerical-analytical)<<"\n";
        passed=false;
      }
    }

    ReportResult("GQA::BackwardWeightGrad_K",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::BackwardWeightGrad_K",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: GQA parameter count
// total params = dim*qk_dim + 2*dim*kv_dim + qk_dim*dim
//------------------------------------------------------------------------------
static void TestGQAParameterCount()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,false);
    CAIF_DeviceMultiHeadAttention mha(config,stream);

    const uint32_t qk_dim=num_heads*head_dim;
    const uint32_t kv_dim=num_kv_heads*head_dim;
    const size_t expected_total=dim*qk_dim+2*dim*kv_dim+qk_dim*dim;

    bool passed=true;
    if(mha.TotalParameterCount()!=expected_total)
    {
      std::cout<<"  TotalParameterCount expected "<<expected_total
               <<", got "<<mha.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("GQA::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: GQA with kv_heads==heads produces same output as standard MHA
//------------------------------------------------------------------------------
static void TestGQAMHAEquivalence()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=3;
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t head_dim=2;

    CAIF_CudaStream stream;

    // Standard MHA (kv_heads == heads)
    auto config_mha=MakeGQAConfig(dim,num_heads,num_heads,head_dim,false);
    CAIF_DeviceMultiHeadAttention mha_std(config_mha,stream);

    // GQA with kv_heads==heads (should behave identically)
    auto config_gqa=MakeGQAConfig(dim,num_heads,num_heads,head_dim,false);
    CAIF_DeviceMultiHeadAttention mha_gqa(config_gqa,stream);

    // Copy weights
    for(size_t p=0;p<mha_std.ParameterTensorCount();++p)
    {
      CAIF_HostTensor hw=mha_std.ParameterTensor(p).ToHost();
      mha_gqa.ParameterTensor(p).CopyFromHost(hw.Data(),hw.TotalElements());
    }

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor out_std=mha_std.Forward(input,false);
    CAIF_DeviceTensor out_gqa=mha_gqa.Forward(input,false);

    CAIF_HostTensor h_std=out_std.ToHost();
    CAIF_HostTensor h_gqa=out_gqa.ToHost();

    bool passed=true;
    for(size_t i=0;i<h_std.TotalElements();++i)
    {
      if(FloatEqual(h_std.Data()[i],h_gqa.Data()[i],1e-5f)==false)
      {
        std::cout<<"  Mismatch at "<<i<<": std="<<h_std.Data()[i]
                 <<" gqa="<<h_gqa.Data()[i]<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("GQA::MHAEquivalence",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::MHAEquivalence",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Description includes kv_heads=N when GQA active
//------------------------------------------------------------------------------
static void TestGQADescriptionString()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t num_heads=4;
    const uint32_t num_kv_heads=2;
    const uint32_t head_dim=2;

    CAIF_CudaStream stream;
    auto config=MakeGQAConfig(dim,num_heads,num_kv_heads,head_dim,true);
    CAIF_DeviceMultiHeadAttention mha(config,stream);

    const std::string desc=mha.Description();
    const std::string expected=
      "MultiHeadAttention(dim=8,heads=4,head_dim=2,causal=true,kv_heads=2)";
    bool passed=(desc==expected);
    if(passed==false)
    {
      std::cout<<"  Expected '"<<expected<<"', got '"<<desc<<"'\n";
    }

    ReportResult("GQA::DescriptionString",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("GQA::DescriptionString",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF GQA Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestGQAForwardShape();
  TestGQAForwardVsCPU();
  TestGQAForwardMQA();
  TestGQAForwardCausal();
  TestGQABackwardInputGrad();
  TestGQABackwardWeightGrad();
  TestGQABackwardWeightGrad_K();
  TestGQAParameterCount();
  TestGQAMHAEquivalence();
  TestGQADescriptionString();
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
