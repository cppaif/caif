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
// The MoE GPU dispatch path always applied a finite
// per-expert capacity and dropped over-capacity token-to-expert assignments,
// with no way to request HF-parity no-drop routing — ForwardImpl threw on any
// overflow strategy other than Drop. HF DeepSeek-V2 / Mixtral / GLM-MoE never
// drop at inference, so any routing imbalance silently diverged from the
// reference (and shrank the MoE branch below sum-of-weights == 1).
//
// The fix adds OverflowStrategy_e::NoDrop (capacity 0 => the dispatch kernel's
// unlimited path) and makes a non-positive capacity_factor mean no-drop too.
//
// Test config forces dropping under Drop: top_k=1, num_experts=4, num_tokens=32,
// capacity_factor=0.5 => per-expert capacity ceil(32/4 * 0.5 * 1) = 4, below the
// mean per-expert load of 8 (= num_tokens * top_k / num_experts), so Drop
// discards tokens no matter how routing distributes them. With top_k=1 a dropped
// token's combined output row is exactly zero (the combine sums no surviving
// expert), which is the observable.
//
//   - Drop   => at least some all-zero output rows (tokens were dropped).
//   - NoDrop => zero all-zero rows (every token reached its expert).
//
// Before the fix the NoDrop case throws ("only Drop overflow supported");
// after the fix it runs and drops nothing.
//------------------------------------------------------------------------------
#include "caif_device_moe_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cstdint>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_moe_nodrop_input_dim=16;
constexpr uint32_t g_moe_nodrop_hidden_dim=32;
constexpr uint32_t g_moe_nodrop_num_experts=4;
constexpr uint32_t g_moe_nodrop_top_k=1;
constexpr uint32_t g_moe_nodrop_num_tokens=32;
// 0.5 => per-expert capacity 4, below the mean per-expert load of 8, so the Drop
// strategy is guaranteed to discard token-to-expert assignments.
constexpr float g_moe_nodrop_capacity_factor=0.5f;

class CAIF_MoECapacityNoDropBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static uint32_t ForwardZeroRowCount(
      const CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e strategy);
    static void TestDropDiscardsTokens();
    static void TestNoDropKeepsAllTokens();
};

// Build a single-expert-per-token MoE layer with the given overflow strategy,
// run one inference forward over a fixed input, and return the number of
// all-zero output rows (each one is a token the combine produced no expert
// contribution for — i.e. a dropped token under top_k == 1).
uint32_t CAIF_MoECapacityNoDropBugTest::ForwardZeroRowCount(
  const CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e strategy)
{
  CAIF_CudaStream stream;
  CAIF_DeviceMoELayer<float,float> layer(g_moe_nodrop_input_dim,
                                         g_moe_nodrop_hidden_dim,
                                         g_moe_nodrop_num_experts,
                                         g_moe_nodrop_top_k,
                                         true,
                                         false,
                                         0,
                                         0,
                                         false,
                                         0.0f,
                                         g_moe_nodrop_capacity_factor,
                                         strategy,
                                         0.0f,
                                         0.0f,
                                         stream);

  const size_t total=static_cast<size_t>(g_moe_nodrop_num_tokens)*g_moe_nodrop_input_dim;
  std::vector<float> host_input(total);
  for(size_t i=0;i<total;++i)
  {
    host_input[i]=static_cast<float>((i%7)+1)*0.13f-0.4f;
  }
  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(host_input.data(),
                                                          {g_moe_nodrop_num_tokens,
                                                            g_moe_nodrop_input_dim},
                                                          stream);

  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  ctx.SetTraining(false);
  ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

  CAIF_DeviceTensor output=layer.Forward(input,ctx);

  std::vector<float> host_output(total);
  output.CopyToHost(host_output.data());

  uint32_t zero_rows=0;
  for(uint32_t t=0;t<g_moe_nodrop_num_tokens;++t)
  {
    bool all_zero=true;
    for(uint32_t d=0;d<g_moe_nodrop_input_dim;++d)
    {
      if(host_output[static_cast<size_t>(t)*g_moe_nodrop_input_dim+d]!=0.0f)
      {
        all_zero=false;
        break;
      }
    }
    if(all_zero==true)
    {
      ++zero_rows;
    }
  }
  return zero_rows;
}

// Characterization: the Drop strategy at a sub-mean capacity_factor really does
// discard tokens (some output rows are all-zero). Passes before and after the
// fix; it anchors the contrast with the NoDrop case below.
void CAIF_MoECapacityNoDropBugTest::TestDropDiscardsTokens()
{
  try
  {
    const uint32_t zero_rows=ForwardZeroRowCount(
      CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::Drop);
    const bool dropped=(zero_rows>0u);
    if(dropped==false)
    {
      ISE_Out::Out()<<"  Drop strategy dropped no tokens at capacity_factor 0.5"
                    <<" (expected some)\n";
    }
    CAIF_TestHarness::Report("BugC6::MoECapacity::DropDiscardsTokens",dropped);
  }
  CAIF_TEST_CATCH_BLOCK("BugC6::MoECapacity::DropDiscardsTokens")
}

// The fix: NoDrop routes every token to its expert even under forced overflow,
// so no output row is all-zero. Before the fix this throws because ForwardImpl
// rejects every strategy except Drop.
void CAIF_MoECapacityNoDropBugTest::TestNoDropKeepsAllTokens()
{
  try
  {
    const uint32_t zero_rows=ForwardZeroRowCount(
      CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e::NoDrop);
    const bool kept_all=(zero_rows==0u);
    if(kept_all==false)
    {
      ISE_Out::Out()<<"  NoDrop strategy dropped "
                    <<zero_rows
                    <<" of "
                    <<g_moe_nodrop_num_tokens
                    <<" tokens (expected none)\n";
    }
    CAIF_TestHarness::Report("BugC6::MoECapacity::NoDropKeepsAllTokens",kept_all);
  }
  CAIF_TEST_CATCH_BLOCK("BugC6::MoECapacity::NoDropKeepsAllTokens")
}

void CAIF_MoECapacityNoDropBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug C6: MoE capacity dropping / NoDrop parity path ==="
                <<"\n\n";
  TestDropDiscardsTokens();
  TestNoDropKeepsAllTokens();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_MoECapacityNoDropBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
