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
// Test: CAIF_BlockOffloadScheduler copy-stream overlap. A multi-stage
// PreNormBlock of HostPinned frozen sublayers must produce results
// identical to the GPU-resident block while the scheduler prefetches on
// its copy stream (lookahead=1), and every stage must be evicted again
// once its backward completes (read-before-free guard exercised on every
// stage exit).
//------------------------------------------------------------------------------
#include "caif_block_offload_scheduler.h"
#include "caif_device_pre_norm_block.h"
#include "caif_device_frozen_linear.h"
#include "caif_device_rmsnorm.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_test_harness.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_offload_cs_test_dim=8;
constexpr uint32_t g_caif_offload_cs_test_batch=2;
constexpr uint32_t g_caif_offload_cs_test_seq=3;
constexpr uint32_t g_caif_offload_cs_test_stages=3;
constexpr uint32_t g_caif_offload_cs_test_iterations=2;
constexpr uint32_t g_caif_offload_cs_test_weight_seed=77;
// offloaded and resident paths run identical kernels on identical weights;
// the tolerance only absorbs accumulation-order differences, if any
constexpr float g_caif_offload_cs_test_tol=1e-6f;

class CAIF_OffloadCopyStreamTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<std::vector<float>> MakeStageWeights();
    static CAIF_DevicePreNormBlock<float,float> MakeFrozenBlock(
        const std::vector<std::vector<float>> &stage_weights,
        const bool offloaded,
        CAIF_CudaStream &stream,
        std::vector<CAIF_DeviceFrozenLinearBase*> &frozen_out);
    static CAIF_DeviceTensor MakeInput(CAIF_CudaStream &stream,const uint32_t seed);
    static CAIF_DeviceTensor MakeGrad(CAIF_CudaStream &stream);
    static bool CompareTensors(const CAIF_HostTensor &ref,
                               const CAIF_HostTensor &got,
                               const char *label);
    static bool AllStagesEvicted(const std::vector<CAIF_DeviceFrozenLinearBase*> &frozen,
                                 const char *label);
    static bool RunParityIteration(CAIF_DevicePreNormBlock<float,float> &reference,
                                   CAIF_DevicePreNormBlock<float,float> &offloaded,
                                   CAIF_CudaStream &stream,
                                   CAIF_RunContext &ctx,
                                   const uint32_t iteration);

    static void TestOffloadedForwardBackwardParity();
    static void TestEvictAllClearsPendingPrefetch();
};

std::vector<std::vector<float>> CAIF_OffloadCopyStreamTests::MakeStageWeights()
{
  std::mt19937 gen(g_caif_offload_cs_test_weight_seed);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);
  const size_t count=static_cast<size_t>(g_caif_offload_cs_test_dim)*
                     g_caif_offload_cs_test_dim;
  std::vector<std::vector<float>> weights(g_caif_offload_cs_test_stages);
  for(size_t s=0;s<weights.size();++s)
  {
    weights[s].resize(count);
    for(size_t i=0;i<count;++i)
    {
      weights[s][i]=dist(gen);
    }
  }
  return weights;
}

CAIF_DevicePreNormBlock<float,float> CAIF_OffloadCopyStreamTests::MakeFrozenBlock(
    const std::vector<std::vector<float>> &stage_weights,
    const bool offloaded,
    CAIF_CudaStream &stream,
    std::vector<CAIF_DeviceFrozenLinearBase*> &frozen_out)
{
  CAIF_DevicePreNormBlock<float,float>::SubLayerVec_t sub_layers;
  for(size_t s=0;s<stage_weights.size();++s)
  {
    CAIF_DevicePreNormBlock<float,float>::SubLayer_t stage;
    stage.norm_prefix="norm"+std::to_string(s)+".";
    stage.layer_prefix="frozen"+std::to_string(s)+".";
    stage.norm=std::make_unique<CAIF_DeviceRMSNorm<float,float>>(g_caif_offload_cs_test_dim,
                                                                 stream);
    std::unique_ptr<CAIF_DeviceFrozenLinear<float,float>> frozen=
      std::make_unique<CAIF_DeviceFrozenLinear<float,float>>(g_caif_offload_cs_test_dim,
                                                             g_caif_offload_cs_test_dim,
                                                             stream);
    if(offloaded==true)
    {
      frozen->SetOffloadPolicy(CAIF_OffloadPolicy::CAIF_OffloadPolicy_e::HostPinned_e);
    }
    CAIF_DeviceTensor weight=CAIF_DeviceTensor::FromHostData(
                              stage_weights[s].data(),
                              {g_caif_offload_cs_test_dim,g_caif_offload_cs_test_dim},
                              stream);
    frozen->LoadFromTensor(std::move(weight));
    frozen_out.push_back(frozen.get());
    stage.layer=std::move(frozen);
    sub_layers.push_back(std::move(stage));
  }
  return CAIF_DevicePreNormBlock<float,float>(std::move(sub_layers),stream);
}

CAIF_DeviceTensor CAIF_OffloadCopyStreamTests::MakeInput(CAIF_CudaStream &stream,
                                                         const uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  std::vector<float> host(static_cast<size_t>(g_caif_offload_cs_test_batch)*
                          g_caif_offload_cs_test_seq*
                          g_caif_offload_cs_test_dim);
  for(size_t i=0;i<host.size();++i)
  {
    host[i]=dist(gen);
  }
  return CAIF_DeviceTensor::FromHostData(host.data(),
                                         {g_caif_offload_cs_test_batch,
                                          g_caif_offload_cs_test_seq,
                                          g_caif_offload_cs_test_dim},
                                         stream);
}

CAIF_DeviceTensor CAIF_OffloadCopyStreamTests::MakeGrad(CAIF_CudaStream &stream)
{
  std::vector<float> host(static_cast<size_t>(g_caif_offload_cs_test_batch)*
                          g_caif_offload_cs_test_seq*
                          g_caif_offload_cs_test_dim,
                          1.0f);
  return CAIF_DeviceTensor::FromHostData(host.data(),
                                         {g_caif_offload_cs_test_batch,
                                          g_caif_offload_cs_test_seq,
                                          g_caif_offload_cs_test_dim},
                                         stream);
}

bool CAIF_OffloadCopyStreamTests::CompareTensors(const CAIF_HostTensor &ref,
                                                 const CAIF_HostTensor &got,
                                                 const char *label)
{
  if(ref.TotalElements()!=got.TotalElements())
  {
    ISE_Out::Out()<<"  "
                  <<label
                  <<": element count mismatch\n";
    return false;
  }
  for(size_t i=0;i<ref.TotalElements();++i)
  {
    const float diff=std::fabs(ref.Data()[i]-got.Data()[i]);
    if(std::isfinite(got.Data()[i])==false || diff>g_caif_offload_cs_test_tol)
    {
      ISE_Out::Out()<<"  "
                    <<label
                    <<": mismatch at index "
                    <<i
                    <<" ref="
                    <<ref.Data()[i]
                    <<" got="
                    <<got.Data()[i]
                    <<"\n";
      return false;
    }
  }
  return true;
}

bool CAIF_OffloadCopyStreamTests::AllStagesEvicted(
    const std::vector<CAIF_DeviceFrozenLinearBase*> &frozen,
    const char *label)
{
  for(size_t s=0;s<frozen.size();++s)
  {
    if(frozen[s]->IsPrefetched()==true)
    {
      ISE_Out::Out()<<"  "
                    <<label
                    <<": stage "
                    <<s
                    <<" still prefetched\n";
      return false;
    }
  }
  return true;
}

bool CAIF_OffloadCopyStreamTests::RunParityIteration(
    CAIF_DevicePreNormBlock<float,float> &reference,
    CAIF_DevicePreNormBlock<float,float> &offloaded,
    CAIF_CudaStream &stream,
    CAIF_RunContext &ctx,
    const uint32_t iteration)
{
  CAIF_DeviceTensor input=MakeInput(stream,iteration);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
  CAIF_DeviceTensor ref_out=reference.Forward(input,ctx);
  CAIF_DeviceTensor off_out=offloaded.Forward(input,ctx);

  CAIF_DeviceTensor grad=MakeGrad(stream);
  ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
  CAIF_DeviceTensor ref_gin=reference.Backward(grad,ctx);
  CAIF_DeviceTensor off_gin=offloaded.Backward(grad,ctx);
  stream.Synchronize();

  bool passed=CompareTensors(ref_out.ToHost(),off_out.ToHost(),"forward output");
  passed=passed && CompareTensors(ref_gin.ToHost(),off_gin.ToHost(),"grad input");
  return passed;
}

//------------------------------------------------------------------------------
// Test 1: Offloaded multi-stage block matches the GPU-resident block across
// repeated forward+backward iterations, and every stage is evicted again
// after its backward exit hook.
//------------------------------------------------------------------------------
void CAIF_OffloadCopyStreamTests::TestOffloadedForwardBackwardParity()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    const std::vector<std::vector<float>> stage_weights=MakeStageWeights();
    std::vector<CAIF_DeviceFrozenLinearBase*> ref_frozen;
    std::vector<CAIF_DeviceFrozenLinearBase*> off_frozen;
    CAIF_DevicePreNormBlock<float,float> reference=
      MakeFrozenBlock(stage_weights,false,stream,ref_frozen);
    CAIF_DevicePreNormBlock<float,float> offloaded=
      MakeFrozenBlock(stage_weights,true,stream,off_frozen);
    for(size_t s=0;s<off_frozen.size();++s)
    {
      offloaded.OffloadSchedulerMut().RegisterAtStage(s,*off_frozen[s]);
    }

    bool passed=true;
    for(uint32_t it=0;it<g_caif_offload_cs_test_iterations && passed==true;++it)
    {
      passed=RunParityIteration(reference,offloaded,stream,ctx,it);
      passed=passed && AllStagesEvicted(off_frozen,"after backward");
    }

    CAIF_TestHarness::Report("OffloadCopyStream::ForwardBackwardParity",passed);
  }
  CAIF_TEST_CATCH_BLOCK("OffloadCopyStream::ForwardBackwardParity")
}

//------------------------------------------------------------------------------
// Test 2: EvictAll drops both the GPU scratch and a pending lookahead
// prefetch event. OnEnterForwardStage(0) awaits stage 0 and issues the
// stage-1 H2D; EvictAll right after must leave every stage evicted, and a
// following full pass must still match the reference (a stale event
// consumed against a freed weight would corrupt it).
//------------------------------------------------------------------------------
void CAIF_OffloadCopyStreamTests::TestEvictAllClearsPendingPrefetch()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    const std::vector<std::vector<float>> stage_weights=MakeStageWeights();
    std::vector<CAIF_DeviceFrozenLinearBase*> ref_frozen;
    std::vector<CAIF_DeviceFrozenLinearBase*> off_frozen;
    CAIF_DevicePreNormBlock<float,float> reference=
      MakeFrozenBlock(stage_weights,false,stream,ref_frozen);
    CAIF_DevicePreNormBlock<float,float> offloaded=
      MakeFrozenBlock(stage_weights,true,stream,off_frozen);
    for(size_t s=0;s<off_frozen.size();++s)
    {
      offloaded.OffloadSchedulerMut().RegisterAtStage(s,*off_frozen[s]);
    }

    offloaded.OffloadSchedulerMut().OnEnterForwardStage(0u,stream);
    offloaded.OffloadSchedulerMut().EvictAll();
    stream.Synchronize();

    bool passed=AllStagesEvicted(off_frozen,"after EvictAll");
    passed=passed && RunParityIteration(reference,offloaded,stream,ctx,0u);
    passed=passed && AllStagesEvicted(off_frozen,"after post-EvictAll pass");

    CAIF_TestHarness::Report("OffloadCopyStream::EvictAllClearsPendingPrefetch",passed);
  }
  CAIF_TEST_CATCH_BLOCK("OffloadCopyStream::EvictAllClearsPendingPrefetch")
}

void CAIF_OffloadCopyStreamTests::RunAll()
{
  ISE_Out::Out()<<"=== CAIF_BlockOffloadScheduler copy-stream overlap Tests ===\n\n";
  TestOffloadedForwardBackwardParity();
  TestEvictAllClearsPendingPrefetch();
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
  instance::CAIF_OffloadCopyStreamTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
