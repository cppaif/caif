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
// embedding_backward_kernel accumulates the gradient
// table with a bare atomicAdd in the storage dtype. For a bf16 embedding the
// running sum is held in bf16, so repeated tokens (ubiquitous in LM training)
// lose low-order contributions. Every other accumulating backward in the
// kernel file accumulates in fp32.
//
// This test points 300 tokens at the SAME id (0), each contributing a
// gradient of 1.0. The exact sum is 300. With bf16 accumulation the running
// sum stalls near 256 (256 + 1 rounds to 256 under round-half-to-even), far
// short of 300. The test asserts the row equals 300; it FAILS against the
// current bf16-accumulating kernel and PASSES once the gradient is
// accumulated in fp32.
//------------------------------------------------------------------------------
#include "caif_device_token_embedding.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_ops.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr uint32_t g_caif_embgrad_bug_vocab=2;
constexpr uint32_t g_caif_embgrad_bug_dim=2;
constexpr uint32_t g_caif_embgrad_bug_num_tokens=300;
constexpr uint32_t g_caif_embgrad_bug_seq_len=1;
constexpr float g_caif_embgrad_bug_grad_value=1.0f;
constexpr float g_caif_embgrad_bug_tol=1.0f;

class CAIF_EmbeddingGradPrecisionBugTest
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> ReadAsFp32(const CAIF_DeviceTensor &src,CAIF_CudaStream &stream);
    static void TestRepeatedTokenAccumulation();
};

std::vector<float> CAIF_EmbeddingGradPrecisionBugTest::ReadAsFp32(const CAIF_DeviceTensor &src,
                                                                  CAIF_CudaStream &stream)
{
  const size_t n=src.TotalElements();
  std::vector<float> out(n);
  if(src.Dtype()==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    src.CopyToHost(out.data());
    return out;
  }
  const std::vector<uint32_t> shape(src.Shape().begin(),src.Shape().end());
  CAIF_DeviceTensor scratch=CAIF_DeviceTensor::Uninitialized(shape,
                                                             stream,
                                                             CAIF_DataType::CAIF_DataType_e::Float32);
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_Ops::Cast(src,scratch,ctx);
  scratch.CopyToHost(out.data());
  return out;
}

void CAIF_EmbeddingGradPrecisionBugTest::TestRepeatedTokenAccumulation()
{
  try
  {
    CAIF_CudaStream stream;
    typedef CAIF_DeviceTokenEmbedding<float,__nv_bfloat16> Embedding_t;
    CAIF_DeviceTokenEmbeddingConfig config{g_caif_embgrad_bug_vocab,g_caif_embgrad_bug_dim};
    Embedding_t emb(config,stream);

    // Every token points at id 0, so the same gradient row receives
    // num_tokens independent atomicAdds of grad_value.
    std::vector<uint32_t> ids(g_caif_embgrad_bug_num_tokens,0u);
    CAIF_DeviceTensor out=emb.ForwardFromIds(ids.data(),
                                             g_caif_embgrad_bug_num_tokens,
                                             g_caif_embgrad_bug_seq_len,
                                             true);

    const std::vector<uint32_t> out_shape(out.Shape().begin(),out.Shape().end());
    const size_t grad_elems=
      static_cast<size_t>(g_caif_embgrad_bug_num_tokens)*g_caif_embgrad_bug_dim;
    std::vector<float> grad_host(grad_elems,g_caif_embgrad_bug_grad_value);
    CAIF_DeviceTensor grad_fp32=CAIF_DeviceTensor::FromHostData(grad_host.data(),
                                                               out_shape,
                                                               stream);
    CAIF_DeviceTensor grad_out=CAIF_DeviceTensor::Uninitialized(
                                 out_shape,
                                 stream,
                                 CAIF_DataType::CAIF_DataType_e::BFloat16);

    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_Ops::Cast(grad_fp32,grad_out,ctx);

    emb.Backward(grad_out,ctx);

    const std::vector<float> grad_table=ReadAsFp32(emb.EmbeddingTableGrad(),stream);

    const float expected=
      static_cast<float>(g_caif_embgrad_bug_num_tokens)*g_caif_embgrad_bug_grad_value;
    bool ok=true;
    for(uint32_t d=0;d<g_caif_embgrad_bug_dim;++d)
    {
      if(std::fabs(grad_table[d]-expected)>g_caif_embgrad_bug_tol)
      {
        ok=false;
      }
    }
    if(ok==false)
    {
      ISE_Out::Out()<<"  grad_table[0]="
                    <<grad_table[0]
                    <<" expected="
                    <<expected
                    <<" (bf16 accumulation lost precision)\n";
    }
    CAIF_TestHarness::Report("BugC4::Embedding::RepeatedTokenAccumulation",ok);
  }
  CAIF_TEST_CATCH_BLOCK("BugC4::Embedding::RepeatedTokenAccumulation")
}

void CAIF_EmbeddingGradPrecisionBugTest::RunAll()
{
  ISE_Out::Out()<<"=== Bug C4: embedding gradient bf16 accumulation precision ==="
                <<"\n\n";
  TestRepeatedTokenAccumulation();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_EmbeddingGradPrecisionBugTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
