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
// Parity test for the fused MLA flash-prefill forward kernel.
// Drives launch_flash_attention_forward_mla against
// an independent fp64 CPU attention reference (scores -> scale -> offset-causal
// mask -> softmax -> *V) across both supported configs:
//   * (128,128) identity (D_qk == D_v),
//   * (192,128) decoupled (DSv2-Lite, D_qk = qk_nope+qk_rope),
// and the cases the offset/decoupling must get right: non-causal vs causal;
// whole-prompt (q_offset=0) vs chunked-into-warm-cache (q_offset>0, q_len<kv_len);
// multi-block (seq > BC), single sub-tile (seq < BC), and ragged (non-multiple
// of BR/BC) lengths; multiple batch-heads.
// The kernel is unconditionally TF32, so parity is checked at a TF32-class
// tolerance (plan deep-dive C). A wrong offset/boundary/D_v mask attends or
// writes the wrong elements and produces O(magnitude) error, far above it.
//------------------------------------------------------------------------------
#include "caif_cuda_kernels_flash_mla.cuh"
#include "caif_cuda_kernels_constants.cuh"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

// Supported head dims, mirrored from the kernel's constants so a drift fails to
// build rather than silently testing the wrong dims.
constexpr int g_mla_v_dim=g_cu_fa_mla_v_dim;
constexpr int g_mla_qk_identity=g_cu_fa_mla_qk_dim_identity;
constexpr int g_mla_qk_decoupled=g_cu_fa_mla_qk_dim_decoupled;
// TF32-class parity: fused TF32 tensor cores vs the fp64 reference.
constexpr float g_mla_parity_atol=5.0e-3f;
constexpr float g_mla_parity_rtol=2.0e-2f;
// Deterministic LCG (Numerical Recipes); fixed seeds, never date-derived.
constexpr uint32_t g_mla_lcg_mul=1664525u;
constexpr uint32_t g_mla_lcg_add=1013904223u;
constexpr uint32_t g_mla_seed_q=10001u;
constexpr uint32_t g_mla_seed_k=20002u;
constexpr uint32_t g_mla_seed_v=30003u;
// Test fixture lengths.
constexpr int g_mla_bh=2;          // batch-heads exercised together
constexpr int g_mla_seq=192;       // multi-block whole-prompt length (3 x BC)
constexpr int g_mla_chunk=64;      // new-token chunk for warm-cache prefill
constexpr int g_mla_cache=128;     // warm-cache length == q_offset
constexpr int g_mla_small=32;      // single sub-tile (< BC) length
constexpr int g_mla_ragged=80;     // non-multiple of BR/BC (exercises boundary)
// Unsupported dims used to confirm the predicate gates them out.
constexpr int g_mla_qk_unsupported=160;
constexpr int g_mla_v_unsupported=64;

class CAIF_FlashMlaParityTest
{
  public:
    static void RunAll();

  protected:

  private:
    static float NextRand(uint32_t &state);
    static std::vector<float> RandomBuffer(const int count,const uint32_t seed);
    static std::vector<uint32_t> Shape(const int a,const int b,const int c);
    static void ReferenceAttention(const std::vector<float> &q,
                                   const std::vector<float> &k,
                                   const std::vector<float> &v,
                                   std::vector<float> &out,
                                   const int bh,
                                   const int q_len,
                                   const int kv_len,
                                   const int q_offset,
                                   const int causal,
                                   const float scale,
                                   const int qk_dim,
                                   const int v_dim);
    static void RunCase(const char *name,
                        const int bh,
                        const int q_len,
                        const int kv_len,
                        const int q_offset,
                        const int causal,
                        const int qk_dim,
                        const int v_dim);
    static void TestAvailable();
    static void RunTileParity();
};

float CAIF_FlashMlaParityTest::NextRand(uint32_t &state)
{
  state=state*g_mla_lcg_mul+g_mla_lcg_add;
  const float u=static_cast<float>(state)/static_cast<float>(UINT32_MAX);
  return u+u-1.0f;
}

std::vector<float> CAIF_FlashMlaParityTest::RandomBuffer(const int count,const uint32_t seed)
{
  std::vector<float> buf(static_cast<size_t>(count));
  uint32_t state=seed;
  for(int i=0;i<count;++i)
  {
    buf[static_cast<size_t>(i)]=NextRand(state);
  }
  return buf;
}

std::vector<uint32_t> CAIF_FlashMlaParityTest::Shape(const int a,const int b,const int c)
{
  return std::vector<uint32_t>{static_cast<uint32_t>(a),
                               static_cast<uint32_t>(b),
                               static_cast<uint32_t>(c)};
}

// fp64 attention reference. Layout matches the kernel: Q/K [bh, seq, qk_dim] and
// V/O [bh, seq, v_dim], row-major; Q/O over q_len, K/V over kv_len. The score is
// a qk_dim dot; the output spans v_dim. Offset-causal: query i (absolute
// position q_offset+i) attends key j iff j <= q_offset+i.
void CAIF_FlashMlaParityTest::ReferenceAttention(const std::vector<float> &q,
                                                 const std::vector<float> &k,
                                                 const std::vector<float> &v,
                                                 std::vector<float> &out,
                                                 const int bh,
                                                 const int q_len,
                                                 const int kv_len,
                                                 const int q_offset,
                                                 const int causal,
                                                 const float scale,
                                                 const int qk_dim,
                                                 const int v_dim)
{
  const double neg_inf=-std::numeric_limits<double>::infinity();
  for(int b=0;b<bh;++b)
  {
    for(int i=0;i<q_len;++i)
    {
      const int q_base=(b*q_len+i)*qk_dim;
      std::vector<double> score(static_cast<size_t>(kv_len));
      double row_max=neg_inf;
      for(int j=0;j<kv_len;++j)
      {
        const int k_base=(b*kv_len+j)*qk_dim;
        double dot=0.0;
        for(int e=0;e<qk_dim;++e)
        {
          dot+=static_cast<double>(q[static_cast<size_t>(q_base+e)])*
               static_cast<double>(k[static_cast<size_t>(k_base+e)]);
        }
        double s=dot*static_cast<double>(scale);
        if(causal==1 && j>q_offset+i)
        {
          s=neg_inf;
        }
        score[static_cast<size_t>(j)]=s;
        if(s>row_max)
        {
          row_max=s;
        }
      }

      double denom=0.0;
      for(int j=0;j<kv_len;++j)
      {
        const double w=std::exp(score[static_cast<size_t>(j)]-row_max);
        score[static_cast<size_t>(j)]=w;
        denom+=w;
      }

      const int o_base=(b*q_len+i)*v_dim;
      for(int e=0;e<v_dim;++e)
      {
        double acc=0.0;
        if(denom>0.0)
        {
          for(int j=0;j<kv_len;++j)
          {
            const int v_base=(b*kv_len+j)*v_dim;
            acc+=score[static_cast<size_t>(j)]*
                 static_cast<double>(v[static_cast<size_t>(v_base+e)]);
          }
          acc/=denom;
        }
        out[static_cast<size_t>(o_base+e)]=static_cast<float>(acc);
      }
    }
  }
}

void CAIF_FlashMlaParityTest::RunCase(const char *name,
                                      const int bh,
                                      const int q_len,
                                      const int kv_len,
                                      const int q_offset,
                                      const int causal,
                                      const int qk_dim,
                                      const int v_dim)
{
  try
  {
    const float scale=1.0f/std::sqrt(static_cast<float>(qk_dim));
    const std::vector<float> q=RandomBuffer(bh*q_len*qk_dim,g_mla_seed_q);
    const std::vector<float> k=RandomBuffer(bh*kv_len*qk_dim,g_mla_seed_k);
    const std::vector<float> v=RandomBuffer(bh*kv_len*v_dim,g_mla_seed_v);

    CAIF_CudaStream stream;
    CAIF_DeviceTensor q_dev=CAIF_DeviceTensor::Zeros(Shape(bh,q_len,qk_dim),stream);
    CAIF_DeviceTensor k_dev=CAIF_DeviceTensor::Zeros(Shape(bh,kv_len,qk_dim),stream);
    CAIF_DeviceTensor v_dev=CAIF_DeviceTensor::Zeros(Shape(bh,kv_len,v_dim),stream);
    CAIF_DeviceTensor out_dev=CAIF_DeviceTensor::Zeros(Shape(bh,q_len,v_dim),stream);
    q_dev.CopyFromHost(q.data(),q.size());
    k_dev.CopyFromHost(k.data(),k.size());
    v_dev.CopyFromHost(v.data(),v.size());

    const bool launched=launch_flash_attention_forward_mla<float>(q_dev.DevicePtr(),
                                                                  k_dev.DevicePtr(),
                                                                  v_dev.DevicePtr(),
                                                                  out_dev.DevicePtr(),
                                                                  bh,
                                                                  q_len,
                                                                  kv_len,
                                                                  qk_dim,
                                                                  v_dim,
                                                                  scale,
                                                                  causal,
                                                                  q_offset,
                                                                  stream.Handle());
    stream.Synchronize();

    std::vector<float> got(static_cast<size_t>(bh*q_len*v_dim));
    out_dev.CopyToHost(got.data());

    std::vector<float> ref(static_cast<size_t>(bh*q_len*v_dim));
    ReferenceAttention(q,k,v,ref,bh,q_len,kv_len,q_offset,causal,scale,qk_dim,v_dim);

    bool ok=(launched==true);
    float worst_abs=0.0f;
    for(size_t i=0;i<got.size();++i)
    {
      const float diff=std::fabs(got[i]-ref[i]);
      const float tol=g_mla_parity_atol+g_mla_parity_rtol*std::fabs(ref[i]);
      if(diff>tol)
      {
        ok=false;
      }
      if(diff>worst_abs)
      {
        worst_abs=diff;
      }
    }

    if(ok==false)
    {
      ISE_Out::Out()<<"  "
                    <<name
                    <<": launched="
                    <<launched
                    <<" worst |got-ref|="
                    <<worst_abs
                    <<" (atol "
                    <<g_mla_parity_atol
                    <<", rtol "
                    <<g_mla_parity_rtol
                    <<")\n";
    }
    CAIF_TestHarness::Report(name,ok);
  }
  CAIF_TEST_CATCH_BLOCK(name)
}

void CAIF_FlashMlaParityTest::TestAvailable()
{
  try
  {
    int device_id=0;
    cudaGetDevice(&device_id);
    const bool identity_ok=mla_flash_prefill_available(g_mla_qk_identity,g_mla_v_dim,device_id);
    const bool decoupled_ok=mla_flash_prefill_available(g_mla_qk_decoupled,g_mla_v_dim,device_id);
    // Wrong qk_dim and wrong v_dim are gated out.
    const bool qk_gated=(mla_flash_prefill_available(g_mla_qk_unsupported,g_mla_v_dim,device_id)==false);
    const bool v_gated=(mla_flash_prefill_available(g_mla_qk_decoupled,g_mla_v_unsupported,device_id)==false);
    const bool ok=(identity_ok==true && decoupled_ok==true && qk_gated==true && v_gated==true);
    if(ok==false)
    {
      ISE_Out::Out()<<"  availability: (128,128)="
                    <<identity_ok
                    <<" (192,128)="
                    <<decoupled_ok
                    <<" qk-gated="
                    <<qk_gated
                    <<" v-gated="
                    <<v_gated
                    <<"\n";
    }
    CAIF_TestHarness::Report("FlashMLA::Parity::Available",ok);
  }
  CAIF_TEST_CATCH_BLOCK("FlashMLA::Parity::Available")
}

void CAIF_FlashMlaParityTest::RunTileParity()
{
  // Validate EVERY candidate tile in the g_cu_fa_mla_tile_* table (not just the
  // auto-selected one) against the fp64 reference, via the forced-tile launcher.
  // A tile with a bad fragment layout or warp grouping is caught HERE rather
  // than silently shipped: (192,128) causal multi-block exercises the phase-2
  // D_v path and the cross-warp reduce that the BR=64 tiles stress.
  try
  {
    constexpr int bh=g_mla_bh;
    constexpr int seq=g_mla_seq;
    constexpr int qk=g_mla_qk_decoupled;
    constexpr int v=g_mla_v_dim;
    constexpr int causal=1;
    const float scale=1.0f/std::sqrt(static_cast<float>(qk));
    const std::vector<float> q=RandomBuffer(bh*seq*qk,g_mla_seed_q);
    const std::vector<float> k=RandomBuffer(bh*seq*qk,g_mla_seed_k);
    const std::vector<float> vv=RandomBuffer(bh*seq*v,g_mla_seed_v);
    std::vector<float> ref(static_cast<size_t>(bh*seq*v));
    ReferenceAttention(q,k,vv,ref,bh,seq,seq,0,causal,scale,qk,v);

    CAIF_CudaStream stream;
    CAIF_DeviceTensor q_dev=CAIF_DeviceTensor::Zeros(Shape(bh,seq,qk),stream);
    CAIF_DeviceTensor k_dev=CAIF_DeviceTensor::Zeros(Shape(bh,seq,qk),stream);
    CAIF_DeviceTensor v_dev=CAIF_DeviceTensor::Zeros(Shape(bh,seq,v),stream);
    CAIF_DeviceTensor out_dev=CAIF_DeviceTensor::Zeros(Shape(bh,seq,v),stream);
    q_dev.CopyFromHost(q.data(),q.size());
    k_dev.CopyFromHost(k.data(),k.size());
    v_dev.CopyFromHost(vv.data(),vv.size());

    for(int t=0;t<g_cu_fa_mla_tile_count;++t)
    {
      const bool launched=launch_flash_attention_forward_mla_tile<float>(q_dev.DevicePtr(),
                                                                         k_dev.DevicePtr(),
                                                                         v_dev.DevicePtr(),
                                                                         out_dev.DevicePtr(),
                                                                         bh,
                                                                         seq,
                                                                         seq,
                                                                         qk,
                                                                         v,
                                                                         scale,
                                                                         causal,
                                                                         0,
                                                                         t,
                                                                         stream.Handle());
      stream.Synchronize();
      std::vector<float> got(static_cast<size_t>(bh*seq*v));
      out_dev.CopyToHost(got.data());

      bool ok=(launched==true);
      float worst_abs=0.0f;
      for(size_t i=0;i<got.size();++i)
      {
        const float diff=std::fabs(got[i]-ref[i]);
        const float tol=g_mla_parity_atol+g_mla_parity_rtol*std::fabs(ref[i]);
        if(diff>tol)
        {
          ok=false;
        }
        if(diff>worst_abs)
        {
          worst_abs=diff;
        }
      }

      std::string name="FlashMLA::Parity::Tile::";
      name+=std::to_string(g_cu_fa_mla_tile_br[t]);
      name+="x";
      name+=std::to_string(g_cu_fa_mla_tile_bc[t]);
      if(ok==false)
      {
        ISE_Out::Out()<<"  "
                      <<name
                      <<": launched="
                      <<launched
                      <<" worst |got-ref|="
                      <<worst_abs
                      <<" (atol "
                      <<g_mla_parity_atol
                      <<", rtol "
                      <<g_mla_parity_rtol
                      <<")\n";
      }
      CAIF_TestHarness::Report(name.c_str(),ok);
    }
  }
  CAIF_TEST_CATCH_BLOCK("FlashMLA::Parity::Tiles")
}

void CAIF_FlashMlaParityTest::RunAll()
{
  ISE_Out::Out()<<"=== Flash-MLA prefill parity (Steps 1-2, 128x128 + 192x128) ==="
                <<"\n\n";
  TestAvailable();
  RunTileParity();

  RunCase("FlashMLA::Parity::128x128::NonCausal::MultiBlock",
          g_mla_bh,
          g_mla_seq,
          g_mla_seq,
          0,
          0,
          g_mla_qk_identity,
          g_mla_v_dim);
  RunCase("FlashMLA::Parity::128x128::Causal::MultiBlock",
          g_mla_bh,
          g_mla_seq,
          g_mla_seq,
          0,
          1,
          g_mla_qk_identity,
          g_mla_v_dim);
  RunCase("FlashMLA::Parity::128x128::Causal::ChunkedWarmCache",
          g_mla_bh,
          g_mla_chunk,
          g_mla_cache+g_mla_chunk,
          g_mla_cache,
          1,
          g_mla_qk_identity,
          g_mla_v_dim);
  RunCase("FlashMLA::Parity::128x128::Causal::SingleSubTile",
          1,
          g_mla_small,
          g_mla_small,
          0,
          1,
          g_mla_qk_identity,
          g_mla_v_dim);
  RunCase("FlashMLA::Parity::128x128::NonCausal::Ragged",
          1,
          g_mla_ragged,
          g_mla_ragged,
          0,
          0,
          g_mla_qk_identity,
          g_mla_v_dim);

  RunCase("FlashMLA::Parity::192x128::NonCausal::MultiBlock",
          g_mla_bh,
          g_mla_seq,
          g_mla_seq,
          0,
          0,
          g_mla_qk_decoupled,
          g_mla_v_dim);
  RunCase("FlashMLA::Parity::192x128::Causal::MultiBlock",
          g_mla_bh,
          g_mla_seq,
          g_mla_seq,
          0,
          1,
          g_mla_qk_decoupled,
          g_mla_v_dim);
  RunCase("FlashMLA::Parity::192x128::Causal::ChunkedWarmCache",
          g_mla_bh,
          g_mla_chunk,
          g_mla_cache+g_mla_chunk,
          g_mla_cache,
          1,
          g_mla_qk_decoupled,
          g_mla_v_dim);
  RunCase("FlashMLA::Parity::192x128::Causal::Ragged",
          1,
          g_mla_ragged,
          g_mla_ragged,
          0,
          1,
          g_mla_qk_decoupled,
          g_mla_v_dim);
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_FlashMlaParityTest::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"Skipped (USE_CAIF_CUDA not defined)"
                <<"\n";
  return 0;
#endif
}
