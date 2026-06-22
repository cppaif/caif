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
// Test: CAIF_Ops::GatherTopKValues
//
// Verifies the gather-by-row-indices kernel that backs SigmoidNoauxTc Phase 1b
// in CAIF_DeviceMoERouter::Route().  out[t,k] = scores[t, indices[t,k]].
// Test path is deliberately small and dependency-free: build a known scores
// tensor + a known indices tensor, run the device op, compare against a
// host-computed reference.  Three dtypes (fp32, fp16, bf16) are covered to
// match the kernel's explicit instantiation set.
//------------------------------------------------------------------------------
#include "caif_device_tensor.h"
#include "caif_ops.h"
#include "caif_cuda_stream.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace instance
{

constexpr uint32_t g_caif_gathertopk_test_num_tokens=32;
constexpr uint32_t g_caif_gathertopk_test_num_experts=16;
constexpr uint32_t g_caif_gathertopk_test_top_k=4;
constexpr float g_caif_gathertopk_test_fp16_tol=1e-3f;
constexpr float g_caif_gathertopk_test_bf16_tol=1e-2f;
constexpr float g_caif_gathertopk_test_fp32_tol=1e-6f;
constexpr float g_caif_gathertopk_test_scores_lo=-1.0f;
constexpr float g_caif_gathertopk_test_scores_hi=1.0f;

//------------------------------------------------------------------------------
// Analytic tests for CAIF_Ops::GatherTopKValues across fp32/fp16/bf16.
//------------------------------------------------------------------------------
class CAIF_GatherTopkValuesTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void GatherHost(const std::vector<float> &scores,
                           const std::vector<int32_t> &indices,
                           std::vector<float> &out);
    static bool Close(const float a,const float b,const float tol);
    static bool RunOneDtype(CAIF_DataType::CAIF_DataType_e dt,
                            const float tol,
                            const uint32_t seed);
    static void TestGatherTopKValuesFp32();
    static void TestGatherTopKValuesFp16();
    static void TestGatherTopKValuesBf16();
};

void CAIF_GatherTopkValuesTests::GatherHost(const std::vector<float> &scores,
                                             const std::vector<int32_t> &indices,
                                             std::vector<float> &out)
{
  for(uint32_t t=0;t<g_caif_gathertopk_test_num_tokens;++t)
  {
    for(uint32_t k=0;k<g_caif_gathertopk_test_top_k;++k)
    {
      const int32_t e=indices[t*g_caif_gathertopk_test_top_k+k];
      out[t*g_caif_gathertopk_test_top_k+k]=
        scores[t*g_caif_gathertopk_test_num_experts+static_cast<uint32_t>(e)];
    }
  }
}

bool CAIF_GatherTopkValuesTests::Close(const float a,const float b,const float tol)
{
  return std::fabs(a-b)<=tol;
}

bool CAIF_GatherTopkValuesTests::RunOneDtype(CAIF_DataType::CAIF_DataType_e dt,
                                              const float tol,
                                              const uint32_t seed)
{
  try
  {
    CAIF_CudaStream stream;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> sdist(g_caif_gathertopk_test_scores_lo,
                                                g_caif_gathertopk_test_scores_hi);
    std::uniform_int_distribution<int32_t> idist(
      0,
      static_cast<int32_t>(g_caif_gathertopk_test_num_experts)-1);

    std::vector<float> h_scores(g_caif_gathertopk_test_num_tokens*
                                g_caif_gathertopk_test_num_experts);
    for(size_t i=0;i<h_scores.size();++i)
    {
      h_scores[i]=sdist(gen);
    }

    std::vector<int32_t> h_indices(g_caif_gathertopk_test_num_tokens*
                                   g_caif_gathertopk_test_top_k);
    for(size_t i=0;i<h_indices.size();++i)
    {
      h_indices[i]=idist(gen);
    }

    std::vector<float> expected(g_caif_gathertopk_test_num_tokens*
                                g_caif_gathertopk_test_top_k,0.0f);
    GatherHost(h_scores,h_indices,expected);

    CAIF_DeviceTensor scores_fp32=
      CAIF_DeviceTensor::FromHostData(h_scores.data(),
                                      {g_caif_gathertopk_test_num_tokens,
                                       g_caif_gathertopk_test_num_experts},
                                      stream);
    CAIF_DeviceTensor scores=scores_fp32.To(dt);

    CAIF_DeviceTensor indices=
      CAIF_DeviceTensor::Uninitialized({g_caif_gathertopk_test_num_tokens,
                                        g_caif_gathertopk_test_top_k},
                                       stream,
                                       CAIF_DataType::CAIF_DataType_e::Int32);
    indices.CopyFromHostRaw(h_indices.data(),h_indices.size()*sizeof(int32_t));

    CAIF_DeviceTensor out=
      CAIF_DeviceTensor::Uninitialized({g_caif_gathertopk_test_num_tokens,
                                        g_caif_gathertopk_test_top_k},
                                       stream,dt);

    CAIF_Ops::GatherTopKValues(scores,indices,out);

    CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
    std::vector<float> got(g_caif_gathertopk_test_num_tokens*
                           g_caif_gathertopk_test_top_k,0.0f);
    out_fp32.CopyToHostRaw(got.data());

    stream.Synchronize();

    for(size_t i=0;i<expected.size();++i)
    {
      if(Close(expected[i],got[i],tol)==false)
      {
        ISE_Out::Out()<<"  mismatch at i="
                      <<i
                      <<": expected="
                      <<expected[i]
                      <<" got="
                      <<got[i]
                      <<" tol="
                      <<tol
                      <<"\n";
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK()
  return false;
}

void CAIF_GatherTopkValuesTests::TestGatherTopKValuesFp32()
{
  CAIF_TestHarness::Report("GatherTopKValues fp32",
                           RunOneDtype(CAIF_DataType::CAIF_DataType_e::Float32,
                                       g_caif_gathertopk_test_fp32_tol,
                                       71));
}

void CAIF_GatherTopkValuesTests::TestGatherTopKValuesFp16()
{
  CAIF_TestHarness::Report("GatherTopKValues fp16",
                           RunOneDtype(CAIF_DataType::CAIF_DataType_e::Float16,
                                       g_caif_gathertopk_test_fp16_tol,
                                       73));
}

void CAIF_GatherTopkValuesTests::TestGatherTopKValuesBf16()
{
  CAIF_TestHarness::Report("GatherTopKValues bf16",
                           RunOneDtype(CAIF_DataType::CAIF_DataType_e::BFloat16,
                                       g_caif_gathertopk_test_bf16_tol,
                                       79));
}

void CAIF_GatherTopkValuesTests::RunAll()
{
  ISE_Out::Out()<<"GatherTopKValues Tests\n";
  ISE_Out::Out()<<"======================\n";

  TestGatherTopKValuesFp32();
  TestGatherTopKValuesFp16();
  TestGatherTopKValuesBf16();

  ISE_Out::Out()<<"\n";
  ISE_Out::Out()<<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"  Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

}//end instance namespace

int main()
{
  instance::CAIF_GatherTopkValuesTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
