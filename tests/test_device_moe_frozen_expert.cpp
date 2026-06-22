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
// MoEFrozenExpert tests.
// Full 9-cell <ComputeT, StorageT> coverage + ForwardImpl/ForwardInto parity.
//------------------------------------------------------------------------------
#include "caif_device_moe_frozen_expert.h"
#include "caif_device_frozen_linear.h"
#include "caif_test_harness.h"
#include "caif_test_constants.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <random>
#include <cmath>
#include <vector>
#include <memory>

namespace instance
{

#ifdef USE_CAIF_CUDA

//------------------------------------------------------------------------------
// MoEFrozenExpert full 9-cell type-dispatch tests.
//------------------------------------------------------------------------------
class CAIF_MoEFrozenExpertTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeRandom(size_t n,uint32_t seed);

    template<typename ComputeT,typename StorageT>
    static CAIF_DataType::CAIF_DataType_e DtypeFromCpp();

    template<typename ComputeT,typename StorageT>
    static std::unique_ptr<CAIF_DeviceFrozenLinear<ComputeT,StorageT>>
    MakeFrozenLinear(uint32_t in_dim,
                     uint32_t out_dim,
                     uint32_t seed,
                     CAIF_CudaStream &stream);

    // Promote `t` to fp32 if it isn't already. Returns a fresh owning fp32 tensor.
    static CAIF_DeviceTensor PromoteToFp32(const CAIF_DeviceTensor &t);

    // Cast an fp32 tensor down to `target` if needed. Returns a fresh owning tensor.
    static CAIF_DeviceTensor DemoteFromFp32(CAIF_DeviceTensor &fp32,
                                             CAIF_DataType::CAIF_DataType_e target);

    template<typename ComputeT,typename StorageT>
    static void TestFrozenExpertCell(const char *cell_name,uint32_t seed);

    template<typename ComputeT,typename StorageT>
    static void TestForwardIntoMatchesForwardImpl(const char *cell_name,uint32_t seed);
};

std::vector<float> CAIF_MoEFrozenExpertTests::MakeRandom(const size_t n,const uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(g_caif_moe_fexp_weight_init_lo,g_caif_moe_fexp_weight_init_hi);
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    v[i]=dist(gen);
  }
  return v;
}

template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<float,float>()
{
  return CAIF_DataType::CAIF_DataType_e::Float32;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<float,__half>()
{
  return CAIF_DataType::CAIF_DataType_e::Float16;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<float,__nv_bfloat16>()
{
  return CAIF_DataType::CAIF_DataType_e::BFloat16;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<__half,float>()
{
  return CAIF_DataType::CAIF_DataType_e::Float32;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<__half,__half>()
{
  return CAIF_DataType::CAIF_DataType_e::Float16;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<__half,__nv_bfloat16>()
{
  return CAIF_DataType::CAIF_DataType_e::BFloat16;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<__nv_bfloat16,float>()
{
  return CAIF_DataType::CAIF_DataType_e::Float32;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<__nv_bfloat16,__half>()
{
  return CAIF_DataType::CAIF_DataType_e::Float16;
}
template<>
CAIF_DataType::CAIF_DataType_e
CAIF_MoEFrozenExpertTests::DtypeFromCpp<__nv_bfloat16,__nv_bfloat16>()
{
  return CAIF_DataType::CAIF_DataType_e::BFloat16;
}

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceFrozenLinear<ComputeT,StorageT>>
CAIF_MoEFrozenExpertTests::MakeFrozenLinear(const uint32_t in_dim,
                                              const uint32_t out_dim,
                                              const uint32_t seed,
                                              CAIF_CudaStream &stream)
{
  auto layer=std::make_unique<CAIF_DeviceFrozenLinear<ComputeT,StorageT>>(in_dim,out_dim,stream);
  std::vector<float> w=MakeRandom(static_cast<size_t>(in_dim)*out_dim,seed);
  CAIF_DeviceTensor fp32_w=CAIF_DeviceTensor::FromHostData(w.data(),{in_dim,out_dim},stream);
  const CAIF_DataType::CAIF_DataType_e sd=DtypeFromCpp<ComputeT,StorageT>();
  if(sd==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    layer->LoadFromTensor(std::move(fp32_w));
  }
  else
  {
    CAIF_DeviceTensor storage_w=fp32_w.To(sd);
    layer->LoadFromTensor(std::move(storage_w));
  }
  return layer;
}

CAIF_DeviceTensor CAIF_MoEFrozenExpertTests::PromoteToFp32(const CAIF_DeviceTensor &t)
{
  if(t.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    return t.Clone();
  }
  return t.To(CAIF_DataType::CAIF_DataType_e::Float32);
}

CAIF_DeviceTensor CAIF_MoEFrozenExpertTests::DemoteFromFp32(CAIF_DeviceTensor &fp32,
                                                              const CAIF_DataType::CAIF_DataType_e target)
{
  if(target==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    return std::move(fp32);
  }
  return fp32.To(target);
}

template<typename ComputeT,typename StorageT>
void CAIF_MoEFrozenExpertTests::TestFrozenExpertCell(const char *cell_name,
                                                      const uint32_t seed)
{
  try
  {
    CAIF_CudaStream stream;
    typename CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::FrozenSubLayers_t subs;
    subs.gate=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_input_dim,
                                                    g_caif_moe_fexp_test_hidden_dim,
                                                    seed+g_caif_moe_fexp_seed_offset_gate,
                                                    stream);
    subs.up=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_input_dim,
                                                  g_caif_moe_fexp_test_hidden_dim,
                                                  seed+g_caif_moe_fexp_seed_offset_up,
                                                  stream);
    subs.down=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_hidden_dim,
                                                    g_caif_moe_fexp_test_input_dim,
                                                    seed+g_caif_moe_fexp_seed_offset_down,
                                                    stream);

    CAIF_DeviceMoEFrozenExpertConfig cfg(g_caif_moe_fexp_test_input_dim,g_caif_moe_fexp_test_hidden_dim,true);
    CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT> expert(cfg,std::move(subs),stream);

    bool passed=true;
    if(expert.ParameterTensorCount()!=0)
    {
      ISE_Out::Out()<<"  ["
                    <<cell_name
                    <<"] expected 0 parameters, got "
                    <<expert.ParameterTensorCount()
                    <<"\n";
      passed=false;
    }
    if(expert.TotalParameterCount()!=0)
    {
      ISE_Out::Out()<<"  ["
                    <<cell_name
                    <<"] expected 0 total params, got "
                    <<expert.TotalParameterCount()
                    <<"\n";
      passed=false;
    }

    const CAIF_DataType::CAIF_DataType_e sd=DtypeFromCpp<ComputeT,StorageT>();
    const uint32_t input_seed=seed*g_caif_moe_fexp_seed_offset_input_mul+
                               g_caif_moe_fexp_seed_offset_input_add;
    std::vector<float> host_input=MakeRandom(
      static_cast<size_t>(g_caif_moe_fexp_test_batch)*g_caif_moe_fexp_test_input_dim,
      input_seed);
    CAIF_DeviceTensor input_fp32=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_moe_fexp_test_batch,g_caif_moe_fexp_test_input_dim},
      stream);
    CAIF_DeviceTensor input=DemoteFromFp32(input_fp32,sd);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceTensor out=expert.Forward(input,ctx);
    CAIF_DeviceTensor out_fp32=PromoteToFp32(out);
    CAIF_HostTensor h_out=out_fp32.ToHost();
    if(h_out.Shape().size()!=2
       ||h_out.Shape()[0]!=g_caif_moe_fexp_test_batch
       ||h_out.Shape()[1]!=g_caif_moe_fexp_test_input_dim)
    {
      ISE_Out::Out()<<"  ["
                    <<cell_name
                    <<"] forward shape mismatch\n";
      passed=false;
    }
    for(size_t i=0;i<h_out.TotalElements()&&passed==true;++i)
    {
      if(std::isfinite(h_out.Data()[i])==false)
      {
        ISE_Out::Out()<<"  ["
                      <<cell_name
                      <<"] non-finite forward at "
                      <<i
                      <<"\n";
        passed=false;
      }
    }

    if(passed==true)
    {
      std::vector<float> grad_data(
        static_cast<size_t>(g_caif_moe_fexp_test_batch)*g_caif_moe_fexp_test_input_dim,1.0f);
      CAIF_DeviceTensor grad_fp32=CAIF_DeviceTensor::FromHostData(
        grad_data.data(),
        {g_caif_moe_fexp_test_batch,g_caif_moe_fexp_test_input_dim},
        stream);
      CAIF_DeviceTensor grad_out=DemoteFromFp32(grad_fp32,sd);
      ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
      CAIF_DeviceTensor grad_in=expert.Backward(grad_out,ctx);
      CAIF_DeviceTensor gi_fp32=PromoteToFp32(grad_in);
      CAIF_HostTensor h_gi=gi_fp32.ToHost();
      for(size_t i=0;i<h_gi.TotalElements();++i)
      {
        if(std::isfinite(h_gi.Data()[i])==false)
        {
          ISE_Out::Out()<<"  ["
                        <<cell_name
                        <<"] non-finite backward at "
                        <<i
                        <<"\n";
          passed=false;
          break;
        }
      }
    }

    std::string label=std::string("MoEFrozenExpert::Cell::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),passed);
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception :"
                     <<e
                     <<"\n";
    std::string label=std::string("MoEFrozenExpert::Cell::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
  catch(ISE_Exception &iseex)
  {
    ISE_Out::ErrLog()<<"ISE Exception :"
                     <<iseex
                     <<"\n";
    std::string label=std::string("MoEFrozenExpert::Cell::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
  catch(std::exception &stdex)
  {
    ISE_Out::ErrLog()<<"std Exception :"
                     <<stdex.what()
                     <<"\n";
    std::string label=std::string("MoEFrozenExpert::Cell::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
  catch(...)
  {
    ISE_Out::ErrLog()<<"UNKNOWN ERROR\n";
    std::string label=std::string("MoEFrozenExpert::Cell::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
}

template<typename ComputeT,typename StorageT>
void CAIF_MoEFrozenExpertTests::TestForwardIntoMatchesForwardImpl(const char *cell_name,
                                                                    const uint32_t seed)
{
  try
  {
    CAIF_CudaStream stream;
    typename CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::FrozenSubLayers_t subs1;
    subs1.gate=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_input_dim,
                                                     g_caif_moe_fexp_test_hidden_dim,
                                                     seed+g_caif_moe_fexp_seed_offset_gate,
                                                     stream);
    subs1.up=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_input_dim,
                                                   g_caif_moe_fexp_test_hidden_dim,
                                                   seed+g_caif_moe_fexp_seed_offset_up,
                                                   stream);
    subs1.down=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_hidden_dim,
                                                     g_caif_moe_fexp_test_input_dim,
                                                     seed+g_caif_moe_fexp_seed_offset_down,
                                                     stream);

    typename CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::FrozenSubLayers_t subs2;
    subs2.gate=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_input_dim,
                                                     g_caif_moe_fexp_test_hidden_dim,
                                                     seed+g_caif_moe_fexp_seed_offset_gate,
                                                     stream);
    subs2.up=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_input_dim,
                                                   g_caif_moe_fexp_test_hidden_dim,
                                                   seed+g_caif_moe_fexp_seed_offset_up,
                                                   stream);
    subs2.down=MakeFrozenLinear<ComputeT,StorageT>(g_caif_moe_fexp_test_hidden_dim,
                                                     g_caif_moe_fexp_test_input_dim,
                                                     seed+g_caif_moe_fexp_seed_offset_down,
                                                     stream);

    CAIF_DeviceMoEFrozenExpertConfig cfg(g_caif_moe_fexp_test_input_dim,g_caif_moe_fexp_test_hidden_dim,true);
    CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT> expert_a(cfg,std::move(subs1),stream);
    CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT> expert_b(cfg,std::move(subs2),stream);

    const CAIF_DataType::CAIF_DataType_e sd=DtypeFromCpp<ComputeT,StorageT>();
    const uint32_t input_seed=seed*g_caif_moe_fexp_seed_offset_parity_mul+
                               g_caif_moe_fexp_seed_offset_parity_add;
    std::vector<float> host_input=MakeRandom(
      static_cast<size_t>(g_caif_moe_fexp_test_batch)*g_caif_moe_fexp_test_input_dim,
      input_seed);
    CAIF_DeviceTensor input_fp32=CAIF_DeviceTensor::FromHostData(
      host_input.data(),
      {g_caif_moe_fexp_test_batch,g_caif_moe_fexp_test_input_dim},
      stream);
    CAIF_DeviceTensor input=DemoteFromFp32(input_fp32,sd);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    CAIF_DeviceTensor out_a=expert_a.Forward(input,ctx);
    CAIF_DeviceTensor out_b=CAIF_DeviceTensor::Uninitialized(
      {g_caif_moe_fexp_test_batch,g_caif_moe_fexp_test_input_dim},
      stream,
      sd);
    expert_b.ForwardInto(input,out_b,ctx);

    CAIF_DeviceTensor a_fp32=PromoteToFp32(out_a);
    CAIF_DeviceTensor b_fp32=PromoteToFp32(out_b);
    CAIF_HostTensor h_a=a_fp32.ToHost();
    CAIF_HostTensor h_b=b_fp32.ToHost();
    bool passed=true;
    for(size_t i=0;i<h_a.TotalElements();++i)
    {
      if(CAIF_TestHarness::FloatEqual(h_a.Data()[i],h_b.Data()[i],g_caif_moe_fexp_parity_tol)==false)
      {
        ISE_Out::Out()<<"  ["
                      <<cell_name
                      <<"] ForwardInto mismatch at "
                      <<i
                      <<" ForwardImpl="
                      <<h_a.Data()[i]
                      <<" ForwardInto="
                      <<h_b.Data()[i]
                      <<"\n";
        passed=false;
        break;
      }
    }
    std::string label=std::string("MoEFrozenExpert::ForwardIntoParity::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),passed);
  }
  catch(CAIF_Exception &e)
  {
    ISE_Out::ErrLog()<<"CAIF Exception :"
                     <<e
                     <<"\n";
    std::string label=std::string("MoEFrozenExpert::ForwardIntoParity::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
  catch(ISE_Exception &iseex)
  {
    ISE_Out::ErrLog()<<"ISE Exception :"
                     <<iseex
                     <<"\n";
    std::string label=std::string("MoEFrozenExpert::ForwardIntoParity::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
  catch(std::exception &stdex)
  {
    ISE_Out::ErrLog()<<"std Exception :"
                     <<stdex.what()
                     <<"\n";
    std::string label=std::string("MoEFrozenExpert::ForwardIntoParity::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
  catch(...)
  {
    ISE_Out::ErrLog()<<"UNKNOWN ERROR\n";
    std::string label=std::string("MoEFrozenExpert::ForwardIntoParity::")+cell_name;
    CAIF_TestHarness::Report(label.c_str(),false);
  }
}

void CAIF_MoEFrozenExpertTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_DeviceMoEFrozenExpert Tests ==="
                <<"\n\n";
  TestFrozenExpertCell<float,float>("float_float",g_caif_moe_fexp_seed_cell_ff);
  TestFrozenExpertCell<float,__half>("float_half",g_caif_moe_fexp_seed_cell_fh);
  TestFrozenExpertCell<float,__nv_bfloat16>("float_bfloat16",g_caif_moe_fexp_seed_cell_fb);
  TestFrozenExpertCell<__half,float>("half_float",g_caif_moe_fexp_seed_cell_hf);
  TestFrozenExpertCell<__half,__half>("half_half",g_caif_moe_fexp_seed_cell_hh);
  TestFrozenExpertCell<__half,__nv_bfloat16>("half_bfloat16",g_caif_moe_fexp_seed_cell_hb);
  TestFrozenExpertCell<__nv_bfloat16,float>("bfloat16_float",g_caif_moe_fexp_seed_cell_bf);
  TestFrozenExpertCell<__nv_bfloat16,__half>("bfloat16_half",g_caif_moe_fexp_seed_cell_bh);
  TestFrozenExpertCell<__nv_bfloat16,__nv_bfloat16>("bfloat16_bfloat16",
                                                      g_caif_moe_fexp_seed_cell_bb);
  TestForwardIntoMatchesForwardImpl<float,float>("float_float",g_caif_moe_fexp_seed_parity_ff);
  TestForwardIntoMatchesForwardImpl<float,__half>("float_half",g_caif_moe_fexp_seed_parity_fh);
  TestForwardIntoMatchesForwardImpl<__half,__half>("half_half",g_caif_moe_fexp_seed_parity_hh);
  TestForwardIntoMatchesForwardImpl<__nv_bfloat16,__nv_bfloat16>("bfloat16_bfloat16",
                                                                    g_caif_moe_fexp_seed_parity_bb);
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
  try
  {
#ifdef USE_CAIF_CUDA
    instance::CAIF_MoEFrozenExpertTests::RunAll();
#else
    instance::ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif
    return instance::CAIF_TestHarness::FinalExitCode();
  }
  catch(const instance::CAIF_Exception &e)
  {
    instance::ISE_Out::ErrLog()<<"CAIF Exception: "
                                <<e
                                <<"\n";
    return 1;
  }
  catch(const std::exception &e)
  {
    instance::ISE_Out::ErrLog()<<"std::exception: "
                                <<e.what()
                                <<"\n";
    return 1;
  }
}
