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

#include "caif_device_transformer_block.h"
#include "caif_device_gated_activations.h"
#include "caif_device_pointwise_activations.h"
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

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// Test 1: Forward shape preserved
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block(config,stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      std::cout<<"  Shape mismatch: expected [2,4,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          std::cout<<",";
        }
        std::cout<<shape[i];
      }
      std::cout<<"]\n";
      passed=false;
    }

    ReportResult("TransformerBlock::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward residual - output differs from zero
//------------------------------------------------------------------------------
static void TestForwardResidual()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block(config,stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.1f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    // The residual connections mean the output should contain
    // the input signal plus attention/ffn modifications.
    // Check that output is not all zeros and differs from input.
    bool has_nonzero=false;
    bool differs_from_input=false;
    for(size_t i=0;i<host_input.size();++i)
    {
      if(host_output.Data()[i]!=0.0f)
      {
        has_nonzero=true;
      }
      if(std::fabs(host_output.Data()[i]-host_input[i])>1e-6f)
      {
        differs_from_input=true;
      }
    }

    bool passed=has_nonzero&&differs_from_input;
    if(has_nonzero==false)
    {
      std::cout<<"  Output is all zeros\n";
    }
    if(differs_from_input==false)
    {
      std::cout<<"  Output is identical to input (sub-layers had no effect)\n";
    }

    ReportResult("TransformerBlock::ForwardResidual",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::ForwardResidual",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Causal vs non-causal produces different output
//------------------------------------------------------------------------------
static void TestForwardCausal()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=4;
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;

    // Create deterministic input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.03f-0.2f;
    }

    // Non-causal block
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config_nc;
    config_nc.dim=dim;
    config_nc.num_heads=num_heads;
    config_nc.num_kv_heads=num_heads;
    config_nc.ffn_dim=ffn_dim;
    config_nc.dropout_rate=0.0f;
    config_nc.causal=false;
    config_nc.use_rope=false;
    config_nc.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block_nc(config_nc,stream);

    // Causal block with same weights
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config_c;
    config_c.dim=dim;
    config_c.num_heads=num_heads;
    config_c.num_kv_heads=num_heads;
    config_c.ffn_dim=ffn_dim;
    config_c.dropout_rate=0.0f;
    config_c.causal=true;
    config_c.use_rope=false;
    config_c.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block_c(config_c,stream);

    // Copy weights from non-causal to causal
    for(size_t p=0;p<block_nc.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_param=block_nc.ParameterTensor(p).ToHost();
      block_c.ParameterTensor(p).CopyFromHost(h_param.Data(),
                                               h_param.TotalElements());
    }

    CAIF_DeviceTensor input_nc=CAIF_DeviceTensor::FromHostData(
                                host_input.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor input_c=CAIF_DeviceTensor::FromHostData(
                               host_input.data(),{batch,seq_len,dim},stream);

    CAIF_DeviceTensor out_nc=block_nc.Forward(input_nc,false);
    CAIF_DeviceTensor out_c=block_c.Forward(input_c,false);

    CAIF_HostTensor host_nc=out_nc.ToHost();
    CAIF_HostTensor host_c=out_c.ToHost();

    bool differs=false;
    for(size_t i=0;i<host_input.size();++i)
    {
      if(std::fabs(host_nc.Data()[i]-host_c.Data()[i])>1e-6f)
      {
        differs=true;
        break;
      }
    }

    bool passed=differs;
    if(passed==false)
    {
      std::cout<<"  Causal and non-causal outputs are identical\n";
    }

    ReportResult("TransformerBlock::ForwardCausal",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::ForwardCausal",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Backward input gradient (finite difference)
//------------------------------------------------------------------------------
static void TestBackwardGrad()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=4;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=8;
    const float h=1e-3f;
    const float grad_tol=5e-2f;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    CAIF_DeviceTransformerBlock block(config,stream);

    // Save all weights
    const size_t num_params=block.ParameterTensorCount();
    std::vector<CAIF_HostTensor> saved_params;
    for(size_t p=0;p<num_params;++p)
    {
      saved_params.push_back(block.ParameterTensor(p).ToHost());
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    block.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    CAIF_DeviceTensor grad_input=block.Backward(grad_out);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      CAIF_DeviceTransformerBlock block_p(config,stream);
      for(size_t p=0;p<num_params;++p)
      {
        block_p.ParameterTensor(p).CopyFromHost(saved_params[p].Data(),
                                                 saved_params[p].TotalElements());
      }
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(
                               input_plus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_p=block_p.Forward(inp_p,false);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      CAIF_DeviceTransformerBlock block_m(config,stream);
      for(size_t p=0;p<num_params;++p)
      {
        block_m.ParameterTensor(p).CopyFromHost(saved_params[p].Data(),
                                                 saved_params[p].TotalElements());
      }
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(
                               input_minus.data(),{batch,seq_len,dim},stream);
      CAIF_DeviceTensor out_m=block_m.Forward(inp_m,false);
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

    ReportResult("TransformerBlock::BackwardGrad",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::BackwardGrad",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Parameter tensor count
//------------------------------------------------------------------------------
static void TestParameterTensorCount()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    bool passed=true;

    // SwiGLU (gated, default): norm1(1) + attn(4) + norm2(1) + ffn(3) = 9
    {
      CAIF_DeviceTransformerBlock block(config,stream);
      if(block.ParameterTensorCount()!=9)
      {
        std::cout<<"  SwiGLU ParameterTensorCount expected 9, got "
                 <<block.ParameterTensorCount()<<"\n";
        passed=false;
      }
    }

    // GELU (pointwise): norm1(1) + attn(4) + norm2(1) + ffn(2) = 8
    {
      auto activation=std::make_unique<CAIF_DeviceGELUActivation>();
      CAIF_DeviceTransformerBlock block(config,std::move(activation),stream);
      if(block.ParameterTensorCount()!=8)
      {
        std::cout<<"  GELU ParameterTensorCount expected 8, got "
                 <<block.ParameterTensorCount()<<"\n";
        passed=false;
      }
    }

    ReportResult("TransformerBlock::ParameterTensorCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::ParameterTensorCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Total parameter count
//------------------------------------------------------------------------------
static void TestTotalParameterCount()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t head_dim=dim/num_heads;  // 4
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block(config,stream);

    // norm1: gamma[dim] = 8
    // attn: W_q[dim, num_heads*head_dim] = 8*8 = 64
    //       W_k[dim, num_kv_heads*head_dim] = 8*8 = 64
    //       W_v[dim, num_kv_heads*head_dim] = 8*8 = 64
    //       W_o[num_heads*head_dim, dim] = 8*8 = 64
    // norm2: gamma[dim] = 8
    // ffn (SwiGLU): W_gate[dim, ffn_dim] = 8*16 = 128
    //               W_up[dim, ffn_dim] = 8*16 = 128
    //               W_down[ffn_dim, dim] = 16*8 = 128
    // Total = 8 + 64+64+64+64 + 8 + 128+128+128 = 656
    const size_t expected=dim+
                          dim*num_heads*head_dim+
                          dim*num_heads*head_dim+
                          dim*num_heads*head_dim+
                          num_heads*head_dim*dim+
                          dim+
                          dim*ffn_dim+
                          dim*ffn_dim+
                          ffn_dim*dim;

    bool passed=true;
    if(block.TotalParameterCount()!=expected)
    {
      std::cout<<"  Expected "<<expected<<", got "
               <<block.TotalParameterCount()<<"\n";
      passed=false;
    }

    ReportResult("TransformerBlock::TotalParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::TotalParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: ZeroGradients
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block(config,stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,dim},stream);
    block.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(
                                grad_ones.data(),{batch,seq_len,dim},stream);
    block.Backward(grad_out);

    // Zero gradients
    block.ZeroGradients();

    bool passed=true;
    for(size_t p=0;p<block.ParameterTensorCount();++p)
    {
      CAIF_HostTensor host_grad=block.GradientTensor(p).ToHost();
      for(size_t i=0;i<host_grad.TotalElements();++i)
      {
        if(host_grad.Data()[i]!=0.0f)
        {
          std::cout<<"  Gradient["<<p<<"] not zeroed at "<<i<<": "
                   <<host_grad.Data()[i]<<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    ReportResult("TransformerBlock::ZeroGradients",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::ZeroGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=true;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block(config,stream);
    const std::string desc=block.Description();

    bool passed=true;

    // Check that key components are present
    if(desc.find("TransformerBlock")==std::string::npos)
    {
      std::cout<<"  Missing 'TransformerBlock' in description\n";
      passed=false;
    }
    if(desc.find("dim=8")==std::string::npos)
    {
      std::cout<<"  Missing 'dim=8' in description\n";
      passed=false;
    }
    if(desc.find("heads=2")==std::string::npos)
    {
      std::cout<<"  Missing 'heads=2' in description\n";
      passed=false;
    }
    if(desc.find("kv_heads=2")==std::string::npos)
    {
      std::cout<<"  Missing 'kv_heads=2' in description\n";
      passed=false;
    }
    if(desc.find("ffn_dim=16")==std::string::npos)
    {
      std::cout<<"  Missing 'ffn_dim=16' in description\n";
      passed=false;
    }
    if(desc.find("causal=true")==std::string::npos)
    {
      std::cout<<"  Missing 'causal=true' in description\n";
      passed=false;
    }

    const std::string expected=
      "TransformerBlock(dim=8,heads=2,kv_heads=2,ffn_dim=16,causal=true)";
    if(desc!=expected)
    {
      std::cout<<"  Expected '"<<expected<<"', got '"<<desc<<"'\n";
      passed=false;
    }

    ReportResult("TransformerBlock::Description",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::Description",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: FFN dim auto-compute when ffn_dim=0
//------------------------------------------------------------------------------
static void TestFFNDimAutoCompute()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t num_heads=2;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=0;  // auto-compute
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    CAIF_DeviceTransformerBlock block(config,stream);

    // Expected: round_to(4 * 8 * 2/3, 256) = round_to(21.33, 256) = 256
    uint32_t raw=g_caif_ffn_multiplier_numerator*
                 dim*
                 g_caif_ffn_gated_numerator/
                 g_caif_ffn_gated_denominator;
    uint32_t expected=((raw+g_caif_ffn_alignment-1)/g_caif_ffn_alignment)*
                      g_caif_ffn_alignment;

    bool passed=true;
    if(block.EffectiveFFNDim()!=expected)
    {
      std::cout<<"  Expected ffn_dim="<<expected<<", got "
               <<block.EffectiveFFNDim()<<"\n";
      passed=false;
    }

    ReportResult("TransformerBlock::FFNDimAutoCompute",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::FFNDimAutoCompute",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Default activation (convenience constructor) produces SwiGLU
//------------------------------------------------------------------------------
static void TestDefaultActivation()
{
  try
  {
    const uint32_t dim=8;
    const uint32_t num_heads=2;
    const uint32_t ffn_dim=16;

    CAIF_CudaStream stream;
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t config;
    config.dim=dim;
    config.num_heads=num_heads;
    config.num_kv_heads=num_heads;
    config.ffn_dim=ffn_dim;
    config.dropout_rate=0.0f;
    config.causal=false;
    config.use_rope=false;
    config.rope_base=g_caif_rope_default_base;

    // Use convenience constructor (no activation arg -> SwiGLU)
    CAIF_DeviceTransformerBlock block(config,stream);

    // SwiGLU is gated -> 3 FFN tensors -> total 9 param tensors
    bool passed=true;
    if(block.ParameterTensorCount()!=9)
    {
      std::cout<<"  Expected 9 param tensors (SwiGLU), got "
               <<block.ParameterTensorCount()<<"\n";
      passed=false;
    }

    ReportResult("TransformerBlock::DefaultActivation",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("TransformerBlock::DefaultActivation",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  std::cout<<"=== CAIF_DeviceTransformerBlock Tests ===\n\n";

#ifdef USE_CAIF_CUDA
  TestForwardShape();
  TestForwardResidual();
  TestForwardCausal();
  TestBackwardGrad();
  TestParameterTensorCount();
  TestTotalParameterCount();
  TestZeroGradients();
  TestDescription();
  TestFFNDimAutoCompute();
  TestDefaultActivation();
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
