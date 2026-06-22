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
// Multi-tensor ("foreach") optimizer step: parity with the per-parameter path.
//
// A 3-parameter fp32 layer (sizes 16 / 8 / 15, so the prefix-sum + per-element
// tensor lookup is exercised) is optimized for several steps with
// CAIF_Settings::MultiTensorOptimizer ON (one kernel launch for all params) vs
// OFF (one launch per param). Both paths run identical fp32 math, so the
// resulting parameters must match for every optimizer.
//------------------------------------------------------------------------------
#include "caif_device_network.h"
#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_constants.h"
#include "caif_settings.h"
#include "caif_exception.h"
#include "caif_adam_optimizer.h"
#include "caif_sgd_optimizer.h"
#include "caif_momentum_optimizer.h"
#include "caif_rmsprop_optimizer.h"
#include "caif_adagrad_optimizer.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr float g_caif_mt_lr=0.01f;
constexpr float g_caif_mt_wd=0.001f;
constexpr int g_caif_mt_steps=3;
constexpr float g_caif_mt_tol=1e-5f;

//------------------------------------------------------------------------------
// Test-only layer: three fp32 parameter tensors of different sizes with fixed,
// deterministic params + grads. Forward/Backward are unused (the optimizer step
// is what is probed) and throw if called.
//------------------------------------------------------------------------------
class CAIF_MtTestLayer:public CAIF_DeviceLayer
{
  public:
    explicit CAIF_MtTestLayer(CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                       _p0(MakeParam(stream,{4,4},0)),
                                                       _p1(MakeParam(stream,{8},1)),
                                                       _p2(MakeParam(stream,{3,5},2)),
                                                       _g0(MakeGrad(stream,{4,4},0)),
                                                       _g1(MakeGrad(stream,{8},1)),
                                                       _g2(MakeGrad(stream,{3,5},2))
    {
    }

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_MtTestLayer::ForwardImpl not used in this test");
    }
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &,CAIF_RunContext &)override
    {
      THROW_CAIFE("CAIF_MtTestLayer::BackwardImpl not used in this test");
    }
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Dense_e;
    }
    void ZeroGradients()override
    {
      for(size_t i=0;i<ParameterTensorCount();++i)
      {
        GradientTensor(i).Fill(0.0f);
      }
    }
    size_t ParameterTensorCount()const override
    {
      return 3;
    }
    CAIF_DeviceTensor &ParameterTensor(size_t i)override
    {
      return ParamRef(i);
    }
    const CAIF_DeviceTensor &ParameterTensor(size_t i)const override
    {
      return ParamRef(i);
    }
    CAIF_DeviceTensor &GradientTensor(size_t i)override
    {
      return GradRef(i);
    }
    const CAIF_DeviceTensor &GradientTensor(size_t i)const override
    {
      return GradRef(i);
    }
    size_t TotalParameterCount()const override
    {
      return ParameterTensor(0).TotalElements()
             +ParameterTensor(1).TotalElements()
             +ParameterTensor(2).TotalElements();
    }
    CAIF_DataType::CAIF_DataType_e RuntimeStorageDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }
    CAIF_DataType::CAIF_DataType_e RuntimeComputeDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }
    std::string Description()const override
    {
      return "CAIF_MtTestLayer";
    }
    std::vector<std::string> ParameterNames(const std::string &prefix)const override
    {
      return {prefix+"p0",prefix+"p1",prefix+"p2"};
    }

  private:
    CAIF_DeviceTensor &ParamRef(size_t i)
    {
      if(i==0)
      {
        return _p0;
      }
      if(i==1)
      {
        return _p1;
      }
      return _p2;
    }
    const CAIF_DeviceTensor &ParamRef(size_t i)const
    {
      if(i==0)
      {
        return _p0;
      }
      if(i==1)
      {
        return _p1;
      }
      return _p2;
    }
    CAIF_DeviceTensor &GradRef(size_t i)
    {
      if(i==0)
      {
        return _g0;
      }
      if(i==1)
      {
        return _g1;
      }
      return _g2;
    }
    const CAIF_DeviceTensor &GradRef(size_t i)const
    {
      if(i==0)
      {
        return _g0;
      }
      if(i==1)
      {
        return _g1;
      }
      return _g2;
    }
    static CAIF_DeviceTensor MakeParam(CAIF_CudaStream &stream,
                                       const std::vector<uint32_t> &shape,
                                       const int which)
    {
      CAIF_DeviceTensor t=CAIF_DeviceTensor::Zeros(shape,stream,CAIF_DataType::CAIF_DataType_e::Float32);
      const size_t n=t.TotalElements();
      std::vector<float> host(n);
      for(size_t i=0;i<n;++i)
      {
        host[i]=0.5f+0.01f*static_cast<float>(i)+0.1f*static_cast<float>(which);
      }
      t.CopyFromHost(host.data(),n);
      return t;
    }
    static CAIF_DeviceTensor MakeGrad(CAIF_CudaStream &stream,
                                      const std::vector<uint32_t> &shape,
                                      const int which)
    {
      CAIF_DeviceTensor t=CAIF_DeviceTensor::Zeros(shape,stream,CAIF_DataType::CAIF_DataType_e::Float32);
      const size_t n=t.TotalElements();
      std::vector<float> host(n);
      for(size_t i=0;i<n;++i)
      {
        host[i]=0.1f-0.001f*static_cast<float>(i)+0.01f*static_cast<float>(which);
      }
      t.CopyFromHost(host.data(),n);
      return t;
    }

    CAIF_DeviceTensor _p0;
    CAIF_DeviceTensor _p1;
    CAIF_DeviceTensor _p2;
    CAIF_DeviceTensor _g0;
    CAIF_DeviceTensor _g1;
    CAIF_DeviceTensor _g2;
};

class CAIF_MultiTensorOptimizerTest
{
  public:
    static void RunAll();

  private:
    static std::unique_ptr<CAIF_Optimizer> MakeOptimizer(const int kind,
                                                         CAIF_CudaStream &stream);
    static std::vector<float> ReadAllParams(CAIF_DeviceNetwork &net);
    static std::vector<float> RunOptimizer(const int kind,const bool batched);
    static void TestParity(const int kind,const std::string &name);
};

std::unique_ptr<CAIF_Optimizer> CAIF_MultiTensorOptimizerTest::MakeOptimizer(
    const int kind,
    CAIF_CudaStream &stream)
{
  if(kind==0)
  {
    return std::make_unique<CAIF_AdamOptimizer>(g_caif_mt_lr,
                                                g_caif_default_beta1,
                                                g_caif_default_beta2,
                                                g_caif_adam_epsilon,
                                                g_caif_mt_wd,
                                                stream);
  }
  if(kind==1)
  {
    return std::make_unique<CAIF_SgdOptimizer>(g_caif_mt_lr,g_caif_mt_wd,stream);
  }
  if(kind==2)
  {
    return std::make_unique<CAIF_MomentumOptimizer>(g_caif_mt_lr,
                                                    g_caif_sgd_default_momentum,
                                                    g_caif_mt_wd,
                                                    stream);
  }
  if(kind==3)
  {
    return std::make_unique<CAIF_RmspropOptimizer>(g_caif_mt_lr,
                                                   g_caif_rmsprop_default_alpha,
                                                   g_caif_adam_epsilon,
                                                   g_caif_mt_wd,
                                                   stream);
  }
  return std::make_unique<CAIF_AdaGradOptimizer>(g_caif_mt_lr,
                                                 g_caif_adam_epsilon,
                                                 g_caif_mt_wd,
                                                 stream);
}

std::vector<float> CAIF_MultiTensorOptimizerTest::ReadAllParams(CAIF_DeviceNetwork &net)
{
  std::vector<float> out;
  CAIF_DeviceLayer &layer=net.Layer(0);
  for(size_t i=0;i<layer.ParameterTensorCount();++i)
  {
    const CAIF_DeviceTensor &p=layer.ParameterTensor(i);
    const size_t n=p.TotalElements();
    std::vector<float> host(n);
    p.CopyToHost(host.data());
    for(size_t j=0;j<n;++j)
    {
      out.push_back(host[j]);
    }
  }
  return out;
}

std::vector<float> CAIF_MultiTensorOptimizerTest::RunOptimizer(const int kind,const bool batched)
{
  CAIF_CudaStream stream;
  CAIF_DeviceNetwork net(stream);
  net.AddLayer(std::make_unique<CAIF_MtTestLayer>(stream));

  const bool prev=CAIF_Settings::MultiTensorOptimizer();
  CAIF_Settings::SetMultiTensorOptimizer(batched);

  std::unique_ptr<CAIF_Optimizer> opt=MakeOptimizer(kind,stream);
  opt->Initialize(net);
  for(int k=0;k<g_caif_mt_steps;++k)
  {
    opt->Step(net);
  }
  CAIF_Settings::SetMultiTensorOptimizer(prev);

  return ReadAllParams(net);
}

void CAIF_MultiTensorOptimizerTest::TestParity(const int kind,const std::string &name)
{
  const std::string full="MultiTensorOptimizer::Parity::"+name;
  try
  {
    const std::vector<float> batched=RunOptimizer(kind,true);
    const std::vector<float> perparam=RunOptimizer(kind,false);

    bool ok=(batched.size()==perparam.size());
    float maxdiff=0.0f;
    for(size_t i=0;ok==true&&i<batched.size();++i)
    {
      const float d=std::fabs(batched[i]-perparam[i]);
      if(d>maxdiff)
      {
        maxdiff=d;
      }
      if(d>g_caif_mt_tol)
      {
        ok=false;
      }
    }
    if(ok==false)
    {
      ISE_Out::Out()<<"  "<<name<<" maxdiff="<<maxdiff<<"\n";
    }
    CAIF_TestHarness::Report(full.c_str(),ok);
  }
  CAIF_TEST_CATCH_BLOCK(full.c_str())
}

void CAIF_MultiTensorOptimizerTest::RunAll()
{
  ISE_Out::Out()<<"=== Multi-tensor optimizer: batched vs per-param parity ==="
                <<"\n\n";
  TestParity(0,"Adam");
  TestParity(1,"SGD");
  TestParity(2,"Momentum");
  TestParity(3,"RMSprop");
  TestParity(4,"AdaGrad");
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_MultiTensorOptimizerTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
