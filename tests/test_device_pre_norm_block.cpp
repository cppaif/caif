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
// AIF - AI Framework
// Test: CAIF_DevicePreNormBlock (Generic pre-norm residual block)
//------------------------------------------------------------------------------
#include "caif_device_pre_norm_block.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_ffn.h"
#include "caif_device_gated_activations.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>
#include <string>
#include <memory>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    ISE_Out::Out()<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    ISE_Out::Out()<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
}

#ifdef USE_CAIF_CUDA

static const uint32_t g_test_dim=8;
static const uint32_t g_test_ffn_dim=16;

//------------------------------------------------------------------------------
// Helper: make a single FFN sub-layer with RMSNorm
//------------------------------------------------------------------------------
static CAIF_DevicePreNormBlock::SubLayer_t MakeFFNSubLayer(const std::string &norm_prefix,
                                                          const std::string &ffn_prefix,
                                                          uint32_t dim,
                                                          uint32_t ffn_dim,
                                                          CAIF_CudaStream &stream)
{
  CAIF_DevicePreNormBlock::SubLayer_t stage;
  stage.norm_prefix=norm_prefix;
  stage.layer_prefix=ffn_prefix;
  stage.norm=std::make_unique<CAIF_DeviceRMSNorm>(dim,stream);

  CAIF_DeviceFFN::FFNConfig_t ffn_config;
  ffn_config.dim=dim;
  ffn_config.ffn_dim=ffn_dim;
  stage.layer=std::make_unique<CAIF_DeviceFFN>(ffn_config,
                                              std::make_unique<CAIF_DeviceSwiGLUActivation>(),
                                              stream);
  return stage;
}

//------------------------------------------------------------------------------
// Helper: build a single-stage PreNormBlock (RMSNorm + FFN)
//------------------------------------------------------------------------------
static CAIF_DevicePreNormBlock MakeSingleStageBlock(CAIF_CudaStream &stream)
{
  CAIF_DevicePreNormBlock::SubLayerVec_t sub_layers;
  sub_layers.push_back(MakeFFNSubLayer("norm.","ffn.",g_test_dim,g_test_ffn_dim,stream));
  return CAIF_DevicePreNormBlock(std::move(sub_layers),stream);
}

//------------------------------------------------------------------------------
// Helper: build a two-stage PreNormBlock
//------------------------------------------------------------------------------
static CAIF_DevicePreNormBlock MakeTwoStageBlock(CAIF_CudaStream &stream)
{
  CAIF_DevicePreNormBlock::SubLayerVec_t sub_layers;
  sub_layers.push_back(MakeFFNSubLayer("norm1.","attn.",g_test_dim,g_test_ffn_dim,stream));
  sub_layers.push_back(MakeFFNSubLayer("norm2.","ffn.",g_test_dim,g_test_ffn_dim,stream));
  return CAIF_DevicePreNormBlock(std::move(sub_layers),stream);
}

//------------------------------------------------------------------------------
// Test 1: Forward shape preserved
//------------------------------------------------------------------------------
static void TestForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeSingleStageBlock(stream);

    std::vector<float> host_input(batch*seq_len*g_test_dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(
                             host_input.data(),{batch,seq_len,g_test_dim},stream);
    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=g_test_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }

    ReportResult("PreNormBlock::ForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::ForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 2: Forward produces finite values
//------------------------------------------------------------------------------
static void TestForwardFinite()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;

    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeSingleStageBlock(stream);

    std::vector<float> host_input(batch*seq_len*g_test_dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,g_test_dim},stream);
    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_output.TotalElements();++i)
    {
      if(std::isfinite(host_output.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite value at index "<<i<<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("PreNormBlock::ForwardFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::ForwardFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 3: Residual connection (zero weights => output == input)
//------------------------------------------------------------------------------
static void TestResidualConnection()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;

    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeSingleStageBlock(stream);

    // Zero all FFN weights (indices 1,2,3 — index 0 is RMSNorm gamma)
    for(size_t p=1;p<block.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_w=block.ParameterTensor(p).ToHost();
      std::vector<float> zeros(h_w.TotalElements(),0.0f);
      block.ParameterTensor(p).CopyFromHost(zeros.data(),zeros.size());
    }

    std::vector<float> host_input(batch*seq_len*g_test_dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.1f+0.1f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,g_test_dim},stream);
    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    for(size_t i=0;i<host_output.TotalElements();++i)
    {
      if(std::fabs(host_output.Data()[i]-host_input[i])>1e-4f)
      {
        ISE_Out::Out()<<"  Residual mismatch at "
                      <<i
                      <<": expected "
                      <<host_input[i]
                      <<" got "
                      <<host_output.Data()[i]
                      <<"\n";
        passed=false;
        break;
      }
    }

    ReportResult("PreNormBlock::ResidualConnection",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::ResidualConnection",false);
  }
}

//------------------------------------------------------------------------------
// Test 4: Parameter count (single stage: RMSNorm[1] + FFN SwiGLU[3])
//------------------------------------------------------------------------------
static void TestParameterCount()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeSingleStageBlock(stream);

    // RMSNorm: 1 tensor (gamma [dim])
    // FFN SwiGLU: 3 tensors (gate [dim*ffn], up [dim*ffn], down [ffn*dim])
    bool passed=true;
    if(block.ParameterTensorCount()!=4)
    {
      ISE_Out::Out()<<"  Expected 4 param tensors, got "<<block.ParameterTensorCount()<<"\n";
      passed=false;
    }

    // Total scalar params: dim + 3*dim*ffn_dim
    const size_t expected_total=g_test_dim+3*g_test_dim*g_test_ffn_dim;
    if(block.TotalParameterCount()!=expected_total)
    {
      ISE_Out::Out()<<"  Expected "
                    <<expected_total
                    <<" total params, got "
                    <<block.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    ReportResult("PreNormBlock::ParameterCount",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::ParameterCount",false);
  }
}

//------------------------------------------------------------------------------
// Test 5: Parameter count for two-stage block
//------------------------------------------------------------------------------
static void TestParameterCountTwoStage()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeTwoStageBlock(stream);

    // 2 x (RMSNorm[1] + FFN[3]) = 8 param tensors
    bool passed=true;
    if(block.ParameterTensorCount()!=8)
    {
      ISE_Out::Out()<<"  Expected 8 param tensors, got "<<block.ParameterTensorCount()<<"\n";
      passed=false;
    }

    const size_t expected_total=2*(g_test_dim+3*g_test_dim*g_test_ffn_dim);
    if(block.TotalParameterCount()!=expected_total)
    {
      ISE_Out::Out()<<"  Expected "
                    <<expected_total
                    <<" total params, got "
                    <<block.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    if(block.SubLayerCount()!=2)
    {
      ISE_Out::Out()<<"  Expected 2 sub-layers, got "<<block.SubLayerCount()<<"\n";
      passed=false;
    }

    ReportResult("PreNormBlock::ParameterCountTwoStage",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::ParameterCountTwoStage",false);
  }
}

//------------------------------------------------------------------------------
// Test 6: Parameter names with prefix
//------------------------------------------------------------------------------
static void TestParameterNames()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeTwoStageBlock(stream);

    auto names=block.ParameterNames("block.");

    bool passed=true;
    if(names.size()!=8)
    {
      ISE_Out::Out()<<"  Expected 8 names, got "<<names.size()<<"\n";
      for(size_t i=0;i<names.size();++i)
      {
        ISE_Out::Out()<<"    ["<<i<<"]: "<<names[i]<<"\n";
      }
      passed=false;
    }
    else
    {
      for(size_t i=0;i<names.size();++i)
      {
        if(names[i].find("block.")==std::string::npos)
        {
          ISE_Out::Out()<<"  Name ["<<i<<"] missing 'block.' prefix: "<<names[i]<<"\n";
          passed=false;
        }
      }
    }

    ReportResult("PreNormBlock::ParameterNames",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::ParameterNames",false);
  }
}

//------------------------------------------------------------------------------
// Test 7: ZeroGradients clears all gradients
//------------------------------------------------------------------------------
static void TestZeroGradients()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;

    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeSingleStageBlock(stream);

    std::vector<float> host_input(batch*seq_len*g_test_dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,g_test_dim},stream);
    block.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*g_test_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),{batch,seq_len,g_test_dim},stream);
    block.Backward(grad_out);
    block.ZeroGradients();

    bool passed=true;
    for(size_t p=0;p<block.ParameterTensorCount();++p)
    {
      CAIF_HostTensor h_grad=block.GradientTensor(p).ToHost();
      for(size_t i=0;i<h_grad.TotalElements();++i)
      {
        if(h_grad.Data()[i]!=0.0f)
        {
          ISE_Out::Out()<<"  Gradient["<<p<<"]["<<i<<"] not zero: "<<h_grad.Data()[i]<<"\n";
          passed=false;
          break;
        }
      }
      if(passed==false)
      {
        break;
      }
    }

    ReportResult("PreNormBlock::ZeroGradients",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::ZeroGradients",false);
  }
}

//------------------------------------------------------------------------------
// Test 8: Backward produces finite gradients
//------------------------------------------------------------------------------
static void TestBackwardFinite()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;

    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeSingleStageBlock(stream);

    std::vector<float> host_input(batch*seq_len*g_test_dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,g_test_dim},stream);
    block.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*g_test_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),{batch,seq_len,g_test_dim},stream);
    CAIF_DeviceTensor grad_input=block.Backward(grad_out);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    const auto &shape=h_grad.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=g_test_dim)
    {
      ISE_Out::Out()<<"  Gradient shape mismatch\n";
      passed=false;
    }

    for(size_t i=0;i<h_grad.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(h_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite gradient at index "<<i<<"\n";
        passed=false;
      }
    }

    ReportResult("PreNormBlock::BackwardFinite",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::BackwardFinite",false);
  }
}

//------------------------------------------------------------------------------
// Test 9: Two-stage forward shape preserved
//------------------------------------------------------------------------------
static void TestTwoStageForwardShape()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=4;

    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeTwoStageBlock(stream);

    std::vector<float> host_input(batch*seq_len*g_test_dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,g_test_dim},stream);
    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    bool passed=true;
    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=g_test_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }

    for(size_t i=0;i<host_output.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(host_output.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite value at index "<<i<<"\n";
        passed=false;
      }
    }

    ReportResult("PreNormBlock::TwoStageForwardShape",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::TwoStageForwardShape",false);
  }
}

//------------------------------------------------------------------------------
// Test 10: Description string
//------------------------------------------------------------------------------
static void TestDescription()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeTwoStageBlock(stream);

    const std::string desc=block.Description();

    bool passed=true;
    if(desc.find("PreNormBlock")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'PreNormBlock': "<<desc<<"\n";
      passed=false;
    }
    if(desc.find("stages=2")==std::string::npos)
    {
      ISE_Out::Out()<<"  Description missing 'stages=2': "<<desc<<"\n";
      passed=false;
    }

    ReportResult("PreNormBlock::Description",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::Description",false);
  }
}

//------------------------------------------------------------------------------
// Test 11: Norm-less sub-layer (norm=nullptr, layer only)
//------------------------------------------------------------------------------
static void TestNormlessSubLayer()
{
  try
  {
    const uint32_t batch=1;
    const uint32_t seq_len=2;

    CAIF_CudaStream stream;

    CAIF_DevicePreNormBlock::SubLayerVec_t sub_layers;
    CAIF_DevicePreNormBlock::SubLayer_t stage;
    stage.norm_prefix="";
    stage.layer_prefix="ffn.";
    stage.norm=nullptr;

    CAIF_DeviceFFN::FFNConfig_t ffn_config;
    ffn_config.dim=g_test_dim;
    ffn_config.ffn_dim=g_test_ffn_dim;
    stage.layer=std::make_unique<CAIF_DeviceFFN>(ffn_config,
                                                std::make_unique<CAIF_DeviceSwiGLUActivation>(),
                                                stream);
    sub_layers.push_back(std::move(stage));
    CAIF_DevicePreNormBlock block(std::move(sub_layers),stream);

    // FFN only: 3 param tensors (no RMSNorm)
    bool passed=true;
    if(block.ParameterTensorCount()!=3)
    {
      ISE_Out::Out()<<"  Expected 3 param tensors, got "<<block.ParameterTensorCount()<<"\n";
      passed=false;
    }

    std::vector<float> host_input(batch*seq_len*g_test_dim,0.1f);
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,g_test_dim},stream);
    CAIF_DeviceTensor output=block.Forward(input,false);
    CAIF_HostTensor host_output=output.ToHost();

    const auto &shape=host_output.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=g_test_dim)
    {
      ISE_Out::Out()<<"  Shape mismatch\n";
      passed=false;
    }

    for(size_t i=0;i<host_output.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(host_output.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite value at index "<<i<<"\n";
        passed=false;
      }
    }

    ReportResult("PreNormBlock::NormlessSubLayer",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::NormlessSubLayer",false);
  }
}

//------------------------------------------------------------------------------
// Test 12: Two-stage backward with weight gradient check
//------------------------------------------------------------------------------
static void TestTwoStageBackward()
{
  try
  {
    const uint32_t batch=2;
    const uint32_t seq_len=3;

    CAIF_CudaStream stream;
    CAIF_DevicePreNormBlock block=MakeTwoStageBlock(stream);

    std::vector<float> host_input(batch*seq_len*g_test_dim);
    for(size_t i=0;i<host_input.size();++i)
    {
      host_input[i]=static_cast<float>(i)*0.05f-0.3f;
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),{batch,seq_len,g_test_dim},stream);
    block.Forward(input,true);

    std::vector<float> grad_ones(batch*seq_len*g_test_dim,1.0f);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::FromHostData(grad_ones.data(),{batch,seq_len,g_test_dim},stream);
    CAIF_DeviceTensor grad_input=block.Backward(grad_out);
    CAIF_HostTensor h_grad=grad_input.ToHost();

    bool passed=true;
    const auto &shape=h_grad.Shape();
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=seq_len||shape[2]!=g_test_dim)
    {
      ISE_Out::Out()<<"  Gradient shape mismatch\n";
      passed=false;
    }

    for(size_t i=0;i<h_grad.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(h_grad.Data()[i])==false)
      {
        ISE_Out::Out()<<"  Non-finite gradient at index "<<i<<"\n";
        passed=false;
      }
    }

    // Check weight gradients are non-zero for all param tensors
    for(size_t p=0;p<block.ParameterTensorCount()&&passed==true;++p)
    {
      CAIF_HostTensor h_wgrad=block.GradientTensor(p).ToHost();
      bool any_nonzero=false;
      for(size_t i=0;i<h_wgrad.TotalElements();++i)
      {
        if(h_wgrad.Data()[i]!=0.0f)
        {
          any_nonzero=true;
          break;
        }
      }
      if(any_nonzero==false)
      {
        ISE_Out::Out()<<"  Weight gradient["<<p<<"] is all zeros\n";
        passed=false;
      }
    }

    ReportResult("PreNormBlock::TwoStageBackward",passed);
  }
  catch(const std::exception &e)
  {
    ISE_Out::Out()<<"Exception: "<<e.what()<<"\n";
    ReportResult("PreNormBlock::TwoStageBackward",false);
  }
}

#endif  // USE_CAIF_CUDA

int main()
{
  try
  {
    ISE_Out::Out()<<"=== CAIF_DevicePreNormBlock Tests ===\n\n";

#ifdef USE_CAIF_CUDA
    TestForwardShape();
    TestForwardFinite();
    TestResidualConnection();
    TestParameterCount();
    TestParameterCountTwoStage();
    TestParameterNames();
    TestZeroGradients();
    TestBackwardFinite();
    TestTwoStageForwardShape();
    TestDescription();
    TestNormlessSubLayer();
    TestTwoStageBackward();
#else
    ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

    ISE_Out::Out()<<"\n=== Summary ===\n";
    ISE_Out::Out()<<"Passed: "<<g_tests_passed<<"\n";
    ISE_Out::Out()<<"Failed: "<<g_tests_failed<<"\n";

    if(g_tests_failed>0)
    {
      return 1;
    }
    return 0;
  }
  catch(const CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"AIF Exception: "<<e<<std::endl;
    return 1;
  }
  catch(const std::exception &e)
  {
    ISE_Out::ErrLog()<<"std::exception: "<<e.what()<<std::endl;
    return 1;
  }
}
