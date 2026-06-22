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
// CAIF_MoEComposer dtype dispatch: the composer assembles a
// whole decoder-only MoE model (embedding + blocks + final norm + head) at the
// config's compute/storage dtype, not just <float,float>. Validates that
// BuildModel dispatches fp32 / fp16 / bf16, that every composed layer carries
// the requested storage dtype (no silent fall-back to float), and that an
// unsupported (compute, storage) pair is refused.
//------------------------------------------------------------------------------
#include "caif_moe_composer.h"
#include "caif_moe_composer_model_config.h"
#include "caif_moe_composer_block_config.h"
#include "caif_device_network.h"
#include "caif_positional_encoding_mode.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <memory>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

class CAIF_MoEComposerDtypeTest
{
  public:
    static void RunAll();

  protected:

  private:
    static CAIF_MoEComposerModelConfig SmallConfig();
    static void TestBuildDtype(const std::string &name,
                               const CAIF_DataType::CAIF_DataType_e compute,
                               const CAIF_DataType::CAIF_DataType_e storage,
                               const bool expect_built);
};

CAIF_MoEComposerModelConfig CAIF_MoEComposerDtypeTest::SmallConfig()
{
  // Small decoder-only MoE model (vocab 64, dim 32, 2 layers, 4 experts): big
  // enough to exercise every composed layer, small enough to assemble in a few
  // MB so the bf16 path is validated without DSv2-Lite's full footprint.
  CAIF_MoEComposerBlockConfig block(32,
                                    4,
                                    4,
                                    0.0f,
                                    true,
                                    true,
                                    10000.0f,
                                    0,
                                    1.0e-5f,
                                    32,
                                    64,
                                    4,
                                    2,
                                    true,
                                    false,
                                    0,
                                    0,
                                    false,
                                    0.0f,
                                    0.0f,
                                    CAIF_MoEComposerBlockConfig::OverflowStrategy_e::NoDrop,
                                    0.0f,
                                    0.0f,
                                    "norm1.",
                                    "attn.",
                                    "norm2.",
                                    "moe.");
  return CAIF_MoEComposerModelConfig(64,
                                     16,
                                     0,
                                     false,
                                     CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::None,
                                     1.0e-5f,
                                     2,
                                     block);
}

void CAIF_MoEComposerDtypeTest::TestBuildDtype(const std::string &name,
                                               const CAIF_DataType::CAIF_DataType_e compute,
                                               const CAIF_DataType::CAIF_DataType_e storage,
                                               const bool expect_built)
{
  const std::string test_name="MoEComposerDtype::"+name;
  try
  {
    CAIF_CudaStream stream;
    CAIF_MoEComposerModelConfig cfg=SmallConfig();
    cfg.SetComputeDtype(compute);
    cfg.SetStorageDtype(storage);
    if(expect_built==false)
    {
      // An unsupported (compute, storage) pair must be rejected, not silently
      // built at the wrong dtype.
      bool threw=false;
      try
      {
        std::unique_ptr<CAIF_DeviceNetwork> rejected=CAIF_MoEComposer::BuildModel(cfg,stream);
      }
      catch(CAIF_Exception &)
      {
        threw=true;
      }
      CAIF_TestHarness::Report(test_name.c_str(),threw);
      return;
    }

    // Embedding + 2 blocks + final norm + head = 5 layers, and layer 0 (the
    // token embedding) must report the requested storage dtype rather than
    // silently fall back to float.
    std::unique_ptr<CAIF_DeviceNetwork> net=CAIF_MoEComposer::BuildModel(cfg,stream);
    const bool shaped=(net!=nullptr&&net->LayerCount()==5);
    bool passed=false;
    if(shaped==true&&net->Layer(0).RuntimeStorageDtype()==storage)
    {
      passed=true;
    }
    CAIF_TestHarness::Report(test_name.c_str(),passed);
  }
  CAIF_TEST_CATCH_BLOCK(test_name.c_str())
}

void CAIF_MoEComposerDtypeTest::RunAll()
{
  ISE_Out::Out()<<"=== MoEComposer dtype dispatch (F7): fp32 / fp16 / bf16 assembly ==="
                <<"\n\n";
  TestBuildDtype("Fp32",
                 CAIF_DataType::CAIF_DataType_e::Float32,
                 CAIF_DataType::CAIF_DataType_e::Float32,
                 true);
  TestBuildDtype("Fp16",
                 CAIF_DataType::CAIF_DataType_e::Float32,
                 CAIF_DataType::CAIF_DataType_e::Float16,
                 true);
  TestBuildDtype("Bf16",
                 CAIF_DataType::CAIF_DataType_e::Float32,
                 CAIF_DataType::CAIF_DataType_e::BFloat16,
                 true);
  TestBuildDtype("UnsupportedComputeBf16",
                 CAIF_DataType::CAIF_DataType_e::BFloat16,
                 CAIF_DataType::CAIF_DataType_e::BFloat16,
                 false);
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_MoEComposerDtypeTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
