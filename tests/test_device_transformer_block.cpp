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
// Tests for CAIF_DeviceTransformerBlock<float,float>.
//------------------------------------------------------------------------------
#include "caif_device_transformer_block.h"
#include "caif_test_harness.h"
#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_device_gelu_activation.h"
#include "caif_device_network.h"
#include "caif_run_context.h"
#include "caif_settings.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include "caif_grad_mode.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_tb_test_batch_shape=2;
constexpr uint32_t g_caif_tb_test_seq_shape=4;
constexpr uint32_t g_caif_tb_test_dim_shape=8;
constexpr uint32_t g_caif_tb_test_heads_shape=2;
constexpr uint32_t g_caif_tb_test_ffn_shape=16;
constexpr uint32_t g_caif_tb_test_batch_residual=1;
constexpr uint32_t g_caif_tb_test_seq_residual=2;
constexpr uint32_t g_caif_tb_test_dim_residual=8;
constexpr uint32_t g_caif_tb_test_heads_residual=2;
constexpr uint32_t g_caif_tb_test_ffn_residual=16;
constexpr uint32_t g_caif_tb_test_batch_causal=1;
constexpr uint32_t g_caif_tb_test_seq_causal=4;
constexpr uint32_t g_caif_tb_test_batch_bwd=1;
constexpr uint32_t g_caif_tb_test_seq_bwd=2;
constexpr uint32_t g_caif_tb_test_dim_bwd=4;
constexpr uint32_t g_caif_tb_test_heads_bwd=2;
constexpr uint32_t g_caif_tb_test_ffn_bwd=8;
constexpr uint32_t g_caif_tb_test_dim_param=8;
constexpr uint32_t g_caif_tb_test_heads_param=2;
constexpr uint32_t g_caif_tb_test_ffn_param=16;
constexpr uint32_t g_caif_tb_test_batch_zero=1;
constexpr uint32_t g_caif_tb_test_seq_zero=2;
constexpr float g_caif_tb_test_residual_eps=1e-6f;

//------------------------------------------------------------------------------
// TransformerBlock forward and backward correctness tests.
//------------------------------------------------------------------------------
class CAIF_TransformerBlockTests
{
  public:
    static void RunAll();

  protected:

  private:
    typedef CAIF_DeviceTransformerBlockConfig BlockConfig_t;

    static void TestForwardShape();
    static void TestForwardResidual();
    static void TestForwardCausal();
    static void TestBackwardGrad(const GradMode_t &mode);
    static void TestParameterTensorCount();
    static void TestTotalParameterCount();
    static void TestZeroGradients();
    static void TestDescription();
    static void TestFFNDimAutoCompute();
    static void TestDefaultActivation();
};

//------------------------------------------------------------------------------
// Test 1: Forward shape preserved
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestForwardShape()
{
  try
  {
    const uint32_t batch=g_caif_tb_test_batch_shape;
    const uint32_t seq_len=g_caif_tb_test_seq_shape;
    const uint32_t dim=g_caif_tb_test_dim_shape;
    const uint32_t num_heads=g_caif_tb_test_heads_shape;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_shape;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=block.Forward(input,ctx);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=dim)
    {
      ISE_Out::Out()<<"  Shape mismatch: expected [2,4,8], got [";
      for(size_t i=0;i<shape.size();++i)
      {
        if(i>0)
        {
          ISE_Out::Out()<<",";
        }
        ISE_Out::Out()<<shape[i];
      }
      ISE_Out::Out()<<"]\n";
      passed=false;
    }

    CAIF_TestHarness::Report("TransformerBlock::ForwardShape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::ForwardShape")
}

//------------------------------------------------------------------------------
// Test 2: Forward residual - output differs from zero
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestForwardResidual()
{
  try
  {
    const uint32_t batch=g_caif_tb_test_batch_residual;
    const uint32_t seq_len=g_caif_tb_test_seq_residual;
    const uint32_t dim=g_caif_tb_test_dim_residual;
    const uint32_t num_heads=g_caif_tb_test_heads_residual;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_residual;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.1f;
    }
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=block.Forward(input,ctx);
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
      if(std::fabs(host_output.Data()[i]-host_input[i])>g_caif_tb_test_residual_eps)
      {
        differs_from_input=true;
      }
    }

    bool passed=has_nonzero&&differs_from_input;
    if(has_nonzero==false)
    {
      ISE_Out::Out()<<"  Output is all zeros\n";
    }
    if(differs_from_input==false)
    {
      ISE_Out::Out()<<"  Output is identical to input (sub-layers had no effect)\n";
    }

    CAIF_TestHarness::Report("TransformerBlock::ForwardResidual",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::ForwardResidual")
}

//------------------------------------------------------------------------------
// Test 3: Causal vs non-causal produces different output
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestForwardCausal()
{
  try
  {
    const uint32_t batch=g_caif_tb_test_batch_causal;
    const uint32_t seq_len=g_caif_tb_test_seq_causal;
    const uint32_t dim=g_caif_tb_test_dim_shape;
    const uint32_t num_heads=g_caif_tb_test_heads_shape;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_shape;

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Create deterministic input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.03f-0.2f;
    }

    // Non-causal block
    BlockConfig_t config_nc(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block_nc(config_nc,stream);

    // Causal block with same weights
    BlockConfig_t config_c(dim,num_heads,num_heads,ffn_dim,0.0f,true,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block_c(config_c,stream);

    // Copy weights from non-causal to causal
    for(size_t p=0;p<block_nc.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_param=block_nc.ParameterTensor(p).ToHost();
      block_c.ParameterTensor(p).CopyFromHost(h_param.Data(),h_param.TotalElements());
    }

    CAIF_DeviceTensor input_nc=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    CAIF_DeviceTensor input_c=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                              {batch,seq_len,dim},
                                                              stream);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor out_nc=block_nc.Forward(input_nc,ctx);
    CAIF_DeviceTensor out_c=block_c.Forward(input_c,ctx);

    CAIF_HostTensor host_nc=out_nc.ToHost();
    CAIF_HostTensor host_c=out_c.ToHost();

    bool differs=false;
    for(size_t i=0;i<host_input.size();++i)
    {
      if(std::fabs(host_nc.Data()[i]-host_c.Data()[i])>g_caif_tb_test_residual_eps)
      {
        differs=true;
        break;
      }
    }

    bool passed=differs;
    if(passed==false)
    {
      ISE_Out::Out()<<"  Causal and non-causal outputs are identical\n";
    }

    CAIF_TestHarness::Report("TransformerBlock::ForwardCausal",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::ForwardCausal")
}

//------------------------------------------------------------------------------
// Test 4: Backward input gradient (finite difference)
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestBackwardGrad(const GradMode_t &mode)
{
  const std::string test_name=std::string("TransformerBlock::BackwardGrad::")+mode.Label();
  const bool prev_precise=CAIF_Settings::PreciseGradients();
  try
  {
    CAIF_Settings::SetPreciseGradients(mode.Precise());

    const uint32_t batch=g_caif_tb_test_batch_bwd;
    const uint32_t seq_len=g_caif_tb_test_seq_bwd;
    const uint32_t dim=g_caif_tb_test_dim_bwd;
    const uint32_t num_heads=g_caif_tb_test_heads_bwd;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_bwd;
    const float h=mode.FdH();
    const float grad_tol=mode.Tol();

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    // Create base input
    std::vector<float> host_input(batch*seq_len*dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f-0.2f;
    }

    // Get analytical gradient
    CAIF_DeviceTransformerBlock<float,float> block(config,stream);

    // Save all weights
    const size_t num_params=block.ParameterTensorCount();
    std::vector<CAIF_HostTensor> saved_params;
    for(size_t p=0;p<num_params;++p)
    {
      saved_params.push_back(block.ParameterTensor(p).ToHost());
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    block.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=block.Backward(grad_out,ctx);
    CAIF_HostTensor host_grad=grad_input.ToHost();

    bool passed=true;

    for(size_t i=0;i<host_input.size()&&passed==true;++i)
    {
      std::vector<float> input_plus(host_input);
      std::vector<float> input_minus(host_input);
      input_plus[i]+=h;
      input_minus[i]-=h;

      // Forward with +h
      CAIF_DeviceTransformerBlock<float,float> block_p(config,stream);
      for(size_t p=0;p<num_params;++p)
      {
        block_p.ParameterTensor(p).CopyFromHost(saved_params[p].Data(),
                                                saved_params[p].TotalElements());
      }
      CAIF_DeviceTensor inp_p=CAIF_DeviceTensor::FromHostData(input_plus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      ctx.SetTraining(false);
      CAIF_DeviceTensor out_p=block_p.Forward(inp_p,ctx);
      CAIF_HostTensor hout_p=out_p.ToHost();
      float sum_plus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_plus+=hout_p.Data()[j];
      }

      // Forward with -h
      CAIF_DeviceTransformerBlock<float,float> block_m(config,stream);
      for(size_t p=0;p<num_params;++p)
      {
        block_m.ParameterTensor(p).CopyFromHost(saved_params[p].Data(),
                                                saved_params[p].TotalElements());
      }
      CAIF_DeviceTensor inp_m=CAIF_DeviceTensor::FromHostData(input_minus.data(),
                                                              {batch,seq_len,dim},
                                                              stream);
      CAIF_DeviceTensor out_m=block_m.Forward(inp_m,ctx);
      CAIF_HostTensor hout_m=out_m.ToHost();
      float sum_minus=0.0f;
      for(size_t j=0;j<batch*seq_len*dim;++j)
      {
        sum_minus+=hout_m.Data()[j];
      }

      const float numerical=(sum_plus-sum_minus)/(2.0f*h);
      const float analytical=host_grad.Data()[i];

      if(std::fabs(numerical-analytical)>grad_tol*std::max(1.0f,std::fabs(analytical)))
      {
        ISE_Out::Out()<<"  dx mismatch at "
                      <<i
                      <<": analytical="
                      <<analytical
                      <<" numerical="
                      <<numerical
                      <<" diff="
                      <<std::fabs(numerical-analytical)
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK("test_name.c_str()")
  CAIF_Settings::SetPreciseGradients(prev_precise);
}

//------------------------------------------------------------------------------
// Test 5: Parameter tensor count
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestParameterTensorCount()
{
  try
  {
    const uint32_t dim=g_caif_tb_test_dim_param;
    const uint32_t num_heads=g_caif_tb_test_heads_param;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_param;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    bool passed=true;

    // SwiGLU (gated, default): norm1(1) + attn(4) + norm2(1) + ffn(3) = 9
    {
      CAIF_DeviceTransformerBlock<float,float> block(config,stream);
      if(block.ParameterTensorCount()!=9)
      {
        ISE_Out::Out()<<"  SwiGLU ParameterTensorCount expected 9, got "
                      <<block.ParameterTensorCount()
                      <<"\n";
        passed=false;
      }
    }

    // GELU (pointwise): norm1(1) + attn(4) + norm2(1) + ffn(2) = 8
    {
      auto activation=std::make_unique<CAIF_DeviceGELUActivation<float,float>>();
      CAIF_DeviceTransformerBlock<float,float> block(config,std::move(activation),stream);
      if(block.ParameterTensorCount()!=8)
      {
        ISE_Out::Out()<<"  GELU ParameterTensorCount expected 8, got "
                      <<block.ParameterTensorCount()
                      <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("TransformerBlock::ParameterTensorCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::ParameterTensorCount")
}

//------------------------------------------------------------------------------
// Test 6: Total parameter count
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestTotalParameterCount()
{
  try
  {
    const uint32_t dim=g_caif_tb_test_dim_param;
    const uint32_t num_heads=g_caif_tb_test_heads_param;
    const uint32_t head_dim=dim/num_heads;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_param;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block(config,stream);

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
      ISE_Out::Out()<<"  Expected "
                    <<expected
                    <<", got "
                    <<block.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("TransformerBlock::TotalParameterCount",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::TotalParameterCount")
}

//------------------------------------------------------------------------------
// Test 7: ZeroGradients
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestZeroGradients()
{
  try
  {
    const uint32_t batch=g_caif_tb_test_batch_zero;
    const uint32_t seq_len=g_caif_tb_test_seq_zero;
    const uint32_t dim=g_caif_tb_test_dim_param;
    const uint32_t num_heads=g_caif_tb_test_heads_param;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_param;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block(config,stream);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    // Run forward + backward to produce non-zero gradients
    std::vector<float> host_input(batch*seq_len*dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                            {batch,seq_len,dim},
                                                            stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    block.Forward(input,ctx);

    std::vector<float> grad_ones(batch*seq_len*dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),
                                                               {batch,seq_len,dim},
                                                               stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    block.Backward(grad_out,ctx);

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
          ISE_Out::Out()<<"  Gradient["
                        <<p
                        <<"] not zeroed at "
                        <<i
                        <<": "
                        <<host_grad.Data()[i]
                        <<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    CAIF_TestHarness::Report("TransformerBlock::ZeroGradients",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::ZeroGradients")
}

//------------------------------------------------------------------------------
// Test 8: Description string
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestDescription()
{
  try
  {
    const uint32_t dim=g_caif_tb_test_dim_param;
    const uint32_t num_heads=g_caif_tb_test_heads_param;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_param;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,true,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block(config,stream);
    const std::string desc=block.Description();

    bool passed=true;

    // Check that key components are present
    if(desc.find("TransformerBlock")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'TransformerBlock' in description\n";
      passed=false;
    }
    if(desc.find("dim=8")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'dim=8' in description\n";
      passed=false;
    }
    if(desc.find("heads=2")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'heads=2' in description\n";
      passed=false;
    }
    if(desc.find("kv_heads=2")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'kv_heads=2' in description\n";
      passed=false;
    }
    if(desc.find("ffn_dim=16")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'ffn_dim=16' in description\n";
      passed=false;
    }
    if(desc.find("causal=true")==std::string::npos)
    {
      ISE_Out::Out()<<"  Missing 'causal=true' in description\n";
      passed=false;
    }

    const std::string expected=
      "TransformerBlock(dim=8,heads=2,kv_heads=2,ffn_dim=16,causal=true)";
    if(desc!=expected)
    {
      ISE_Out::Out()<<"  Expected '"
                    <<expected
                    <<"', got '"
                    <<desc
                    <<"'\n";
      passed=false;
    }

    CAIF_TestHarness::Report("TransformerBlock::Description",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::Description")
}

//------------------------------------------------------------------------------
// Test 9: FFN dim auto-compute when ffn_dim=0
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestFFNDimAutoCompute()
{
  try
  {
    const uint32_t dim=g_caif_tb_test_dim_param;
    const uint32_t num_heads=g_caif_tb_test_heads_param;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,0,0.0f,false,false,g_caif_rope_default_base);

    CAIF_DeviceTransformerBlock<float,float> block(config,stream);

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
      ISE_Out::Out()<<"  Expected ffn_dim="
                    <<expected
                    <<", got "
                    <<block.EffectiveFFNDim()
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("TransformerBlock::FFNDimAutoCompute",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::FFNDimAutoCompute")
}

//------------------------------------------------------------------------------
// Test 10: Default activation (convenience constructor) produces SwiGLU
//------------------------------------------------------------------------------
void CAIF_TransformerBlockTests::TestDefaultActivation()
{
  try
  {
    const uint32_t dim=g_caif_tb_test_dim_param;
    const uint32_t num_heads=g_caif_tb_test_heads_param;
    const uint32_t ffn_dim=g_caif_tb_test_ffn_param;

    CAIF_CudaStream stream;
    BlockConfig_t config(dim,num_heads,num_heads,ffn_dim,0.0f,false,false,g_caif_rope_default_base);

    // Use convenience constructor (no activation arg -> SwiGLU)
    CAIF_DeviceTransformerBlock<float,float> block(config,stream);

    // SwiGLU is gated -> 3 FFN tensors -> total 9 param tensors
    bool passed=true;
    if(block.ParameterTensorCount()!=9)
    {
      ISE_Out::Out()<<"  Expected 9 param tensors (SwiGLU), got "
                    <<block.ParameterTensorCount()
                    <<"\n";
      passed=false;
    }

    CAIF_TestHarness::Report("TransformerBlock::DefaultActivation",passed);
  }
  CAIF_TEST_CATCH_BLOCK("TransformerBlock::DefaultActivation")
}

void CAIF_TransformerBlockTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DeviceTransformerBlock<float,float> Tests ===\n\n";
  TestForwardShape();
  TestForwardResidual();
  TestForwardCausal();
  TestBackwardGrad(g_caif_grad_mode_precise);
  TestBackwardGrad(g_caif_grad_mode_tf32);
  TestParameterTensorCount();
  TestTotalParameterCount();
  TestZeroGradients();
  TestDescription();
  TestFFNDimAutoCompute();
  TestDefaultActivation();
}

#endif  // USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_TransformerBlockTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
