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
// Test: dtype + compute-precision parity for every MatMul variant.
//
// Each of the 7 MatMul ops (MatMul, MatMulTransposeA, MatMulTransposeB,
// MatMulBias, BatchedMatMul, BatchedMatMulTransposeA, BatchedMatMulTransposeB)
// is exercised against an fp32-storage + fp32-compute reference under four
// configurations:
//   1. fp16 storage, fp32 compute  (input rounding dominates)
//   2. bf16 storage, fp32 compute
//   3. fp32 storage, bf16 compute  (CUBLAS_COMPUTE_32F_FAST_16BF path)
//   4. fp32 storage, fp16 compute  (CUBLAS_COMPUTE_32F_FAST_16F path)
// Cases 3 and 4 validate the new `compute_dtype` parameter added in Step 4.
//------------------------------------------------------------------------------
#include "caif_device_tensor.h"
#include "caif_ops.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr float g_caif_matmul_dtype_test_fp16_tol=5e-3f;
constexpr float g_caif_matmul_dtype_test_bf16_tol=2e-2f;
constexpr size_t g_caif_matmul_dtype_test_num_cases=4;

//------------------------------------------------------------------------------
// MatMul dtype + compute-precision parity test suite.
//------------------------------------------------------------------------------
class CAIF_MatMulDtypeTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeRandom(const size_t n,const uint32_t seed);
    static bool RelClose(const std::vector<float> &ref,
                         const std::vector<float> &got,
                         const float tol);
    static void ReportOne(const char *variant,const char *label,const bool ok);

    static std::vector<float> RunMatMul(const std::vector<float> &a_host,
                                        const std::vector<float> &b_host,
                                        const uint32_t m,
                                        const uint32_t k,
                                        const uint32_t n,
                                        const CAIF_DataType::CAIF_DataType_e storage,
                                        const CAIF_DataType::CAIF_DataType_e compute,
                                        CAIF_CudaStream &stream);

    static std::vector<float> RunMatMulTA(const std::vector<float> &a_host,
                                          const std::vector<float> &b_host,
                                          const uint32_t k,
                                          const uint32_t m,
                                          const uint32_t n,
                                          const CAIF_DataType::CAIF_DataType_e storage,
                                          const CAIF_DataType::CAIF_DataType_e compute,
                                          CAIF_CudaStream &stream);

    static std::vector<float> RunMatMulTB(const std::vector<float> &a_host,
                                          const std::vector<float> &b_host,
                                          const uint32_t m,
                                          const uint32_t k,
                                          const uint32_t n,
                                          const CAIF_DataType::CAIF_DataType_e storage,
                                          const CAIF_DataType::CAIF_DataType_e compute,
                                          CAIF_CudaStream &stream);

    static std::vector<float> RunMatMulBias(const std::vector<float> &a_host,
                                            const std::vector<float> &b_host,
                                            const std::vector<float> &bias_host,
                                            const uint32_t m,
                                            const uint32_t k,
                                            const uint32_t n,
                                            const CAIF_DataType::CAIF_DataType_e storage,
                                            const CAIF_DataType::CAIF_DataType_e compute,
                                            CAIF_CudaStream &stream);

    static std::vector<float> RunBatchedMatMul(const std::vector<float> &a_host,
                                               const std::vector<float> &b_host,
                                               const uint32_t batch,
                                               const uint32_t m,
                                               const uint32_t k,
                                               const uint32_t n,
                                               const CAIF_DataType::CAIF_DataType_e storage,
                                               const CAIF_DataType::CAIF_DataType_e compute,
                                               CAIF_CudaStream &stream);

    static std::vector<float> RunBatchedMatMulTA(const std::vector<float> &a_host,
                                                 const std::vector<float> &b_host,
                                                 const uint32_t batch,
                                                 const uint32_t k,
                                                 const uint32_t m,
                                                 const uint32_t n,
                                                 const CAIF_DataType::CAIF_DataType_e storage,
                                                 const CAIF_DataType::CAIF_DataType_e compute,
                                                 CAIF_CudaStream &stream);

    static std::vector<float> RunBatchedMatMulTB(const std::vector<float> &a_host,
                                                 const std::vector<float> &b_host,
                                                 const uint32_t batch,
                                                 const uint32_t m,
                                                 const uint32_t k,
                                                 const uint32_t n,
                                                 const CAIF_DataType::CAIF_DataType_e storage,
                                                 const CAIF_DataType::CAIF_DataType_e compute,
                                                 CAIF_CudaStream &stream);

    static void TestMatMulAll();
    static void TestMatMulTransposeAAll();
    static void TestMatMulTransposeBAll();
    static void TestMatMulBiasAll();
    static void TestBatchedMatMulAll();
    static void TestBatchedMatMulTransposeAAll();
    static void TestBatchedMatMulTransposeBAll();

    static const CAIF_DataType::CAIF_DataType_e s_case_storage[g_caif_matmul_dtype_test_num_cases];
    static const CAIF_DataType::CAIF_DataType_e s_case_compute[g_caif_matmul_dtype_test_num_cases];
    static const float s_case_tol[g_caif_matmul_dtype_test_num_cases];
    static const char *const s_case_label[g_caif_matmul_dtype_test_num_cases];
    static const CAIF_DataType::CAIF_DataType_e s_ref_storage;
    static const CAIF_DataType::CAIF_DataType_e s_ref_compute;
};

const CAIF_DataType::CAIF_DataType_e
CAIF_MatMulDtypeTests::s_case_storage[g_caif_matmul_dtype_test_num_cases]=
{
  CAIF_DataType::CAIF_DataType_e::Float16,
  CAIF_DataType::CAIF_DataType_e::BFloat16,
  CAIF_DataType::CAIF_DataType_e::Float32,
  CAIF_DataType::CAIF_DataType_e::Float32
};

const CAIF_DataType::CAIF_DataType_e
CAIF_MatMulDtypeTests::s_case_compute[g_caif_matmul_dtype_test_num_cases]=
{
  CAIF_DataType::CAIF_DataType_e::Float32,
  CAIF_DataType::CAIF_DataType_e::Float32,
  CAIF_DataType::CAIF_DataType_e::BFloat16,
  CAIF_DataType::CAIF_DataType_e::Float16
};

const float CAIF_MatMulDtypeTests::s_case_tol[g_caif_matmul_dtype_test_num_cases]=
{
  g_caif_matmul_dtype_test_fp16_tol,
  g_caif_matmul_dtype_test_bf16_tol,
  g_caif_matmul_dtype_test_bf16_tol,
  g_caif_matmul_dtype_test_fp16_tol
};

const char *const CAIF_MatMulDtypeTests::s_case_label[g_caif_matmul_dtype_test_num_cases]=
{
  "FP16storage",
  "BF16storage",
  "FP32s_BF16c",
  "FP32s_FP16c"
};

const CAIF_DataType::CAIF_DataType_e
CAIF_MatMulDtypeTests::s_ref_storage=CAIF_DataType::CAIF_DataType_e::Float32;

const CAIF_DataType::CAIF_DataType_e
CAIF_MatMulDtypeTests::s_ref_compute=CAIF_DataType::CAIF_DataType_e::Float32;

std::vector<float> CAIF_MatMulDtypeTests::MakeRandom(const size_t n,const uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    v[i]=dist(gen);
  }
  return v;
}

bool CAIF_MatMulDtypeTests::RelClose(const std::vector<float> &ref,
                                     const std::vector<float> &got,
                                     const float tol)
{
  if(ref.size()!=got.size())
  {
    return false;
  }
  float max_abs_ref=0.0f;
  for(float r:ref)
  {
    if(std::fabs(r)>max_abs_ref)
    {
      max_abs_ref=std::fabs(r);
    }
  }
  const float denom=std::max(max_abs_ref,1e-6f);
  for(size_t i=0;i<ref.size();++i)
  {
    const float rel=std::fabs(ref[i]-got[i])/denom;
    if(rel>tol)
    {
      ISE_Out::Out()<<"  mismatch i="
                    <<i
                    <<" ref="
                    <<ref[i]
                    <<" got="
                    <<got[i]
                    <<" rel="
                    <<rel
                    <<"\n";
      return false;
    }
  }
  return true;
}

void CAIF_MatMulDtypeTests::ReportOne(const char *variant,const char *label,const bool ok)
{
  char buf[128];
  std::snprintf(buf,sizeof(buf),"%s::%s",variant,label);
  CAIF_TestHarness::Report(buf,ok);
}

//------------------------------------------------------------------------------
// Per-variant runners. Each takes fp32 host inputs plus explicit storage and
// compute dtypes, and returns fp32 host output (with the op run at the given
// storage+compute dtype).
//------------------------------------------------------------------------------

std::vector<float> CAIF_MatMulDtypeTests::RunMatMul(const std::vector<float> &a_host,
                                                     const std::vector<float> &b_host,
                                                     const uint32_t m,
                                                     const uint32_t k,
                                                     const uint32_t n,
                                                     const CAIF_DataType::CAIF_DataType_e storage,
                                                     const CAIF_DataType::CAIF_DataType_e compute,
                                                     CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_DeviceTensor a_fp32=CAIF_DeviceTensor::FromHostData(a_host.data(),{m,k},stream);
  CAIF_DeviceTensor b_fp32=CAIF_DeviceTensor::FromHostData(b_host.data(),{k,n},stream);
  CAIF_DeviceTensor a=a_fp32.To(storage);
  CAIF_DeviceTensor b=b_fp32.To(storage);
  CAIF_DeviceTensor out=CAIF_DeviceTensor::Zeros({m,n},stream,storage);
  CAIF_Ops::MatMul(a,b,out,ctx,compute);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> got(static_cast<size_t>(m)*n);
  out_fp32.CopyToHost(got.data());
  return got;
}

std::vector<float> CAIF_MatMulDtypeTests::RunMatMulTA(const std::vector<float> &a_host,
                                                       const std::vector<float> &b_host,
                                                       const uint32_t k,
                                                       const uint32_t m,
                                                       const uint32_t n,
                                                       const CAIF_DataType::CAIF_DataType_e storage,
                                                       const CAIF_DataType::CAIF_DataType_e compute,
                                                       CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_DeviceTensor a_fp32=CAIF_DeviceTensor::FromHostData(a_host.data(),{k,m},stream);
  CAIF_DeviceTensor b_fp32=CAIF_DeviceTensor::FromHostData(b_host.data(),{k,n},stream);
  CAIF_DeviceTensor a=a_fp32.To(storage);
  CAIF_DeviceTensor b=b_fp32.To(storage);
  CAIF_DeviceTensor out=CAIF_DeviceTensor::Zeros({m,n},stream,storage);
  CAIF_Ops::MatMulTransposeA(a,b,out,ctx,compute);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> got(static_cast<size_t>(m)*n);
  out_fp32.CopyToHost(got.data());
  return got;
}

std::vector<float> CAIF_MatMulDtypeTests::RunMatMulTB(const std::vector<float> &a_host,
                                                       const std::vector<float> &b_host,
                                                       const uint32_t m,
                                                       const uint32_t k,
                                                       const uint32_t n,
                                                       const CAIF_DataType::CAIF_DataType_e storage,
                                                       const CAIF_DataType::CAIF_DataType_e compute,
                                                       CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_DeviceTensor a_fp32=CAIF_DeviceTensor::FromHostData(a_host.data(),{m,k},stream);
  CAIF_DeviceTensor b_fp32=CAIF_DeviceTensor::FromHostData(b_host.data(),{n,k},stream);
  CAIF_DeviceTensor a=a_fp32.To(storage);
  CAIF_DeviceTensor b=b_fp32.To(storage);
  CAIF_DeviceTensor out=CAIF_DeviceTensor::Zeros({m,n},stream,storage);
  CAIF_Ops::MatMulTransposeB(a,b,out,ctx,compute);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> got(static_cast<size_t>(m)*n);
  out_fp32.CopyToHost(got.data());
  return got;
}

std::vector<float> CAIF_MatMulDtypeTests::RunMatMulBias(const std::vector<float> &a_host,
                                                         const std::vector<float> &b_host,
                                                         const std::vector<float> &bias_host,
                                                         const uint32_t m,
                                                         const uint32_t k,
                                                         const uint32_t n,
                                                         const CAIF_DataType::CAIF_DataType_e storage,
                                                         const CAIF_DataType::CAIF_DataType_e compute,
                                                         CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_DeviceTensor a_fp32=CAIF_DeviceTensor::FromHostData(a_host.data(),{m,k},stream);
  CAIF_DeviceTensor b_fp32=CAIF_DeviceTensor::FromHostData(b_host.data(),{k,n},stream);
  CAIF_DeviceTensor bias_fp32=CAIF_DeviceTensor::FromHostData(bias_host.data(),{n},stream);
  CAIF_DeviceTensor a=a_fp32.To(storage);
  CAIF_DeviceTensor b=b_fp32.To(storage);
  CAIF_DeviceTensor bias=bias_fp32.To(storage);
  CAIF_DeviceTensor out=CAIF_DeviceTensor::Zeros({m,n},stream,storage);
  CAIF_Ops::MatMulBias(a,b,bias,out,stream.Handle(),ctx,compute);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> got(static_cast<size_t>(m)*n);
  out_fp32.CopyToHost(got.data());
  return got;
}

std::vector<float> CAIF_MatMulDtypeTests::RunBatchedMatMul(const std::vector<float> &a_host,
                                                            const std::vector<float> &b_host,
                                                            const uint32_t batch,
                                                            const uint32_t m,
                                                            const uint32_t k,
                                                            const uint32_t n,
                                                            const CAIF_DataType::CAIF_DataType_e storage,
                                                            const CAIF_DataType::CAIF_DataType_e compute,
                                                            CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_DeviceTensor a_fp32=CAIF_DeviceTensor::FromHostData(a_host.data(),{batch,m,k},stream);
  CAIF_DeviceTensor b_fp32=CAIF_DeviceTensor::FromHostData(b_host.data(),{batch,k,n},stream);
  CAIF_DeviceTensor a=a_fp32.To(storage);
  CAIF_DeviceTensor b=b_fp32.To(storage);
  CAIF_DeviceTensor out=CAIF_DeviceTensor::Zeros({batch,m,n},stream,storage);
  CAIF_Ops::BatchedMatMul(a,
                          b,
                          out,
                          static_cast<int>(m),
                          static_cast<int>(k),
                          static_cast<int>(n),
                          static_cast<int>(batch),
                          ctx,
                          compute);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> got(static_cast<size_t>(batch)*m*n);
  out_fp32.CopyToHost(got.data());
  return got;
}

std::vector<float> CAIF_MatMulDtypeTests::RunBatchedMatMulTA(const std::vector<float> &a_host,
                                                              const std::vector<float> &b_host,
                                                              const uint32_t batch,
                                                              const uint32_t k,
                                                              const uint32_t m,
                                                              const uint32_t n,
                                                              const CAIF_DataType::CAIF_DataType_e storage,
                                                              const CAIF_DataType::CAIF_DataType_e compute,
                                                              CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_DeviceTensor a_fp32=CAIF_DeviceTensor::FromHostData(a_host.data(),{batch,k,m},stream);
  CAIF_DeviceTensor b_fp32=CAIF_DeviceTensor::FromHostData(b_host.data(),{batch,k,n},stream);
  CAIF_DeviceTensor a=a_fp32.To(storage);
  CAIF_DeviceTensor b=b_fp32.To(storage);
  CAIF_DeviceTensor out=CAIF_DeviceTensor::Zeros({batch,m,n},stream,storage);
  CAIF_Ops::BatchedMatMulTransposeA(a,
                                    b,
                                    out,
                                    static_cast<int>(k),
                                    static_cast<int>(m),
                                    static_cast<int>(n),
                                    static_cast<int>(batch),
                                    ctx,
                                    compute);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> got(static_cast<size_t>(batch)*m*n);
  out_fp32.CopyToHost(got.data());
  return got;
}

std::vector<float> CAIF_MatMulDtypeTests::RunBatchedMatMulTB(const std::vector<float> &a_host,
                                                              const std::vector<float> &b_host,
                                                              const uint32_t batch,
                                                              const uint32_t m,
                                                              const uint32_t k,
                                                              const uint32_t n,
                                                              const CAIF_DataType::CAIF_DataType_e storage,
                                                              const CAIF_DataType::CAIF_DataType_e compute,
                                                              CAIF_CudaStream &stream)
{
  CAIF_RunContext ctx;
  ctx.SetStream(stream);
  CAIF_DeviceTensor a_fp32=CAIF_DeviceTensor::FromHostData(a_host.data(),{batch,m,k},stream);
  CAIF_DeviceTensor b_fp32=CAIF_DeviceTensor::FromHostData(b_host.data(),{batch,n,k},stream);
  CAIF_DeviceTensor a=a_fp32.To(storage);
  CAIF_DeviceTensor b=b_fp32.To(storage);
  CAIF_DeviceTensor out=CAIF_DeviceTensor::Zeros({batch,m,n},stream,storage);
  CAIF_Ops::BatchedMatMulTransposeB(a,
                                    b,
                                    out,
                                    static_cast<int>(m),
                                    static_cast<int>(k),
                                    static_cast<int>(n),
                                    static_cast<int>(batch),
                                    ctx,
                                    compute);
  CAIF_DeviceTensor out_fp32=out.To(CAIF_DataType::CAIF_DataType_e::Float32);
  std::vector<float> got(static_cast<size_t>(batch)*m*n);
  out_fp32.CopyToHost(got.data());
  return got;
}

//------------------------------------------------------------------------------
// Per-variant test drivers. Each runs an fp32/fp32 reference then compares
// against all four (storage,compute) configurations at the case's tolerance.
//------------------------------------------------------------------------------

void CAIF_MatMulDtypeTests::TestMatMulAll()
{
  const uint32_t m=16;
  const uint32_t k=32;
  const uint32_t n=24;
  CAIF_CudaStream stream;
  std::vector<float> a=MakeRandom(static_cast<size_t>(m)*k,11);
  std::vector<float> b=MakeRandom(static_cast<size_t>(k)*n,22);
  std::vector<float> ref=RunMatMul(a,b,m,k,n,s_ref_storage,s_ref_compute,stream);
  for(size_t i=0;i<g_caif_matmul_dtype_test_num_cases;++i)
  {
    try
    {
      std::vector<float> got=RunMatMul(a,b,m,k,n,s_case_storage[i],s_case_compute[i],stream);
      ReportOne("MatMul",s_case_label[i],RelClose(ref,got,s_case_tol[i]));
    }
    catch(const CAIF_Exception &e)
    {
      ISE_Out::Out()<<"MatMul::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e
                    <<"\n";
      ReportOne("MatMul",s_case_label[i],false);
    }
    catch(const std::exception &e)
    {
      ISE_Out::Out()<<"MatMul::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e.what()
                    <<"\n";
      ReportOne("MatMul",s_case_label[i],false);
    }
  }
}

void CAIF_MatMulDtypeTests::TestMatMulTransposeAAll()
{
  const uint32_t k=20;
  const uint32_t m=12;
  const uint32_t n=8;
  CAIF_CudaStream stream;
  std::vector<float> a=MakeRandom(static_cast<size_t>(k)*m,55);
  std::vector<float> b=MakeRandom(static_cast<size_t>(k)*n,66);
  std::vector<float> ref=RunMatMulTA(a,b,k,m,n,s_ref_storage,s_ref_compute,stream);
  for(size_t i=0;i<g_caif_matmul_dtype_test_num_cases;++i)
  {
    try
    {
      std::vector<float> got=RunMatMulTA(a,b,k,m,n,s_case_storage[i],s_case_compute[i],stream);
      ReportOne("MatMulTransposeA",s_case_label[i],RelClose(ref,got,s_case_tol[i]));
    }
    catch(const CAIF_Exception &e)
    {
      ISE_Out::Out()<<"MatMulTransposeA::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e
                    <<"\n";
      ReportOne("MatMulTransposeA",s_case_label[i],false);
    }
    catch(const std::exception &e)
    {
      ISE_Out::Out()<<"MatMulTransposeA::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e.what()
                    <<"\n";
      ReportOne("MatMulTransposeA",s_case_label[i],false);
    }
  }
}

void CAIF_MatMulDtypeTests::TestMatMulTransposeBAll()
{
  const uint32_t m=10;
  const uint32_t k=28;
  const uint32_t n=14;
  CAIF_CudaStream stream;
  std::vector<float> a=MakeRandom(static_cast<size_t>(m)*k,77);
  std::vector<float> b=MakeRandom(static_cast<size_t>(n)*k,88);
  std::vector<float> ref=RunMatMulTB(a,b,m,k,n,s_ref_storage,s_ref_compute,stream);
  for(size_t i=0;i<g_caif_matmul_dtype_test_num_cases;++i)
  {
    try
    {
      std::vector<float> got=RunMatMulTB(a,b,m,k,n,s_case_storage[i],s_case_compute[i],stream);
      ReportOne("MatMulTransposeB",s_case_label[i],RelClose(ref,got,s_case_tol[i]));
    }
    catch(const CAIF_Exception &e)
    {
      ISE_Out::Out()<<"MatMulTransposeB::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e
                    <<"\n";
      ReportOne("MatMulTransposeB",s_case_label[i],false);
    }
    catch(const std::exception &e)
    {
      ISE_Out::Out()<<"MatMulTransposeB::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e.what()
                    <<"\n";
      ReportOne("MatMulTransposeB",s_case_label[i],false);
    }
  }
}

void CAIF_MatMulDtypeTests::TestMatMulBiasAll()
{
  const uint32_t m=12;
  const uint32_t k=20;
  const uint32_t n=16;
  CAIF_CudaStream stream;
  std::vector<float> a=MakeRandom(static_cast<size_t>(m)*k,101);
  std::vector<float> b=MakeRandom(static_cast<size_t>(k)*n,102);
  std::vector<float> bias=MakeRandom(n,103);
  std::vector<float> ref=RunMatMulBias(a,b,bias,m,k,n,s_ref_storage,s_ref_compute,stream);
  for(size_t i=0;i<g_caif_matmul_dtype_test_num_cases;++i)
  {
    try
    {
      std::vector<float> got=RunMatMulBias(a,
                                           b,
                                           bias,
                                           m,
                                           k,
                                           n,
                                           s_case_storage[i],
                                           s_case_compute[i],
                                           stream);
      ReportOne("MatMulBias",s_case_label[i],RelClose(ref,got,s_case_tol[i]));
    }
    catch(const CAIF_Exception &e)
    {
      ISE_Out::Out()<<"MatMulBias::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e
                    <<"\n";
      ReportOne("MatMulBias",s_case_label[i],false);
    }
    catch(const std::exception &e)
    {
      ISE_Out::Out()<<"MatMulBias::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e.what()
                    <<"\n";
      ReportOne("MatMulBias",s_case_label[i],false);
    }
  }
}

void CAIF_MatMulDtypeTests::TestBatchedMatMulAll()
{
  const uint32_t batch=4;
  const uint32_t m=8;
  const uint32_t k=16;
  const uint32_t n=12;
  CAIF_CudaStream stream;
  std::vector<float> a=MakeRandom(static_cast<size_t>(batch)*m*k,91);
  std::vector<float> b=MakeRandom(static_cast<size_t>(batch)*k*n,92);
  std::vector<float> ref=RunBatchedMatMul(a,b,batch,m,k,n,s_ref_storage,s_ref_compute,stream);
  for(size_t i=0;i<g_caif_matmul_dtype_test_num_cases;++i)
  {
    try
    {
      std::vector<float> got=RunBatchedMatMul(a,
                                              b,
                                              batch,
                                              m,
                                              k,
                                              n,
                                              s_case_storage[i],
                                              s_case_compute[i],
                                              stream);
      ReportOne("BatchedMatMul",s_case_label[i],RelClose(ref,got,s_case_tol[i]));
    }
    catch(const CAIF_Exception &e)
    {
      ISE_Out::Out()<<"BatchedMatMul::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e
                    <<"\n";
      ReportOne("BatchedMatMul",s_case_label[i],false);
    }
    catch(const std::exception &e)
    {
      ISE_Out::Out()<<"BatchedMatMul::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e.what()
                    <<"\n";
      ReportOne("BatchedMatMul",s_case_label[i],false);
    }
  }
}

void CAIF_MatMulDtypeTests::TestBatchedMatMulTransposeAAll()
{
  const uint32_t batch=4;
  const uint32_t k=20;
  const uint32_t m=10;
  const uint32_t n=14;
  CAIF_CudaStream stream;
  std::vector<float> a=MakeRandom(static_cast<size_t>(batch)*k*m,201);
  std::vector<float> b=MakeRandom(static_cast<size_t>(batch)*k*n,202);
  std::vector<float> ref=RunBatchedMatMulTA(a,b,batch,k,m,n,s_ref_storage,s_ref_compute,stream);
  for(size_t i=0;i<g_caif_matmul_dtype_test_num_cases;++i)
  {
    try
    {
      std::vector<float> got=RunBatchedMatMulTA(a,
                                                b,
                                                batch,
                                                k,
                                                m,
                                                n,
                                                s_case_storage[i],
                                                s_case_compute[i],
                                                stream);
      ReportOne("BatchedMatMulTransposeA",s_case_label[i],RelClose(ref,got,s_case_tol[i]));
    }
    catch(const CAIF_Exception &e)
    {
      ISE_Out::Out()<<"BatchedMatMulTransposeA::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e
                    <<"\n";
      ReportOne("BatchedMatMulTransposeA",s_case_label[i],false);
    }
    catch(const std::exception &e)
    {
      ISE_Out::Out()<<"BatchedMatMulTransposeA::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e.what()
                    <<"\n";
      ReportOne("BatchedMatMulTransposeA",s_case_label[i],false);
    }
  }
}

void CAIF_MatMulDtypeTests::TestBatchedMatMulTransposeBAll()
{
  const uint32_t batch=4;
  const uint32_t m=10;
  const uint32_t k=16;
  const uint32_t n=12;
  CAIF_CudaStream stream;
  std::vector<float> a=MakeRandom(static_cast<size_t>(batch)*m*k,301);
  std::vector<float> b=MakeRandom(static_cast<size_t>(batch)*n*k,302);
  std::vector<float> ref=RunBatchedMatMulTB(a,b,batch,m,k,n,s_ref_storage,s_ref_compute,stream);
  for(size_t i=0;i<g_caif_matmul_dtype_test_num_cases;++i)
  {
    try
    {
      std::vector<float> got=RunBatchedMatMulTB(a,
                                                b,
                                                batch,
                                                m,
                                                k,
                                                n,
                                                s_case_storage[i],
                                                s_case_compute[i],
                                                stream);
      ReportOne("BatchedMatMulTransposeB",s_case_label[i],RelClose(ref,got,s_case_tol[i]));
    }
    catch(const CAIF_Exception &e)
    {
      ISE_Out::Out()<<"BatchedMatMulTransposeB::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e
                    <<"\n";
      ReportOne("BatchedMatMulTransposeB",s_case_label[i],false);
    }
    catch(const std::exception &e)
    {
      ISE_Out::Out()<<"BatchedMatMulTransposeB::"
                    <<s_case_label[i]
                    <<" threw: "
                    <<e.what()
                    <<"\n";
      ReportOne("BatchedMatMulTransposeB",s_case_label[i],false);
    }
  }
}

void CAIF_MatMulDtypeTests::RunAll()
{
  ISE_Out::Out()<<"=== MatMul Dtype + Compute-Precision Parity Tests ===\n\n";
  TestMatMulAll();
  TestMatMulTransposeAAll();
  TestMatMulTransposeBAll();
  TestMatMulBiasAll();
  TestBatchedMatMulAll();
  TestBatchedMatMulTransposeAAll();
  TestBatchedMatMulTransposeBAll();
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
  instance::CAIF_MatMulDtypeTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
  return 0;
#endif
}
