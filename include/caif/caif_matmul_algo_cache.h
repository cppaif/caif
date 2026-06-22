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
// MatMul algo cache for cublasLt. First call per (op_a, op_b, m, n, k,
// lda, ldb, ldc, matType, computeType, batch_count, strides) asks cuBLAS-Lt
// for its top-ranked algo via heuristic and stores it; subsequent calls reuse
// it directly. Beats cublasGemmEx(CUBLAS_GEMM_DEFAULT) on large GEMMs because
// Lt's algo catalog is broader and we skip the per-call heuristic lookup.
//
// Extracted from caif_ops_device.cpp for the one-class-per-file structural
// refactor; the cuBLAS-Lt dtype map and the shared probe/execute helpers moved
// in as methods since they exist only to serve this cache. All three —
// MatrixTypeFor, LtMatMulExecuteCached, ProbeAndCacheBestAlgo — are defined
// out-of-line in caif_matmul_algo_cache.cpp.
//------------------------------------------------------------------------------
#pragma once

#ifdef USE_CAIF_CUDA

#include "caif_base.h"
#include "caif_data_type.h"
#include "caif_constants.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace instance
{

class CAIF_MatMulAlgoCache:public CAIF_Base
{
  public:
    static constexpr int ProbeCandidates(){return _probe_candidates;}
    static constexpr int ProbeIters(){return _probe_iters;}

    struct Key_t
    {
      cublasOperation_t OpA()const{return _op_a;}
      void SetOpA(const cublasOperation_t v){_op_a=v;}
      cublasOperation_t OpB()const{return _op_b;}
      void SetOpB(const cublasOperation_t v){_op_b=v;}
      int M()const{return _m;}
      void SetM(const int v){_m=v;}
      int N()const{return _n;}
      void SetN(const int v){_n=v;}
      int K()const{return _k;}
      void SetK(const int v){_k=v;}
      int Lda()const{return _lda;}
      void SetLda(const int v){_lda=v;}
      int Ldb()const{return _ldb;}
      void SetLdb(const int v){_ldb=v;}
      int Ldc()const{return _ldc;}
      void SetLdc(const int v){_ldc=v;}
      cudaDataType_t MatType()const{return _mat_type;}
      void SetMatType(const cudaDataType_t v){_mat_type=v;}
      cublasComputeType_t ComputeType()const{return _compute_type;}
      void SetComputeType(const cublasComputeType_t v){_compute_type=v;}
      int BatchCount()const{return _batch_count;}
      void SetBatchCount(const int v){_batch_count=v;}
      long long StrideA()const{return _stride_a;}
      void SetStrideA(const long long v){_stride_a=v;}
      long long StrideB()const{return _stride_b;}
      void SetStrideB(const long long v){_stride_b=v;}
      long long StrideC()const{return _stride_c;}
      void SetStrideC(const long long v){_stride_c=v;}
      uint8_t EpilogueBias()const{return _epilogue_bias;}
      void SetEpilogueBias(const uint8_t v){_epilogue_bias=v;}

      bool operator==(const Key_t &o)const
      {
        return OpA()==o.OpA() &&
               OpB()==o.OpB() &&
               M()==o.M() &&
               N()==o.N() &&
               K()==o.K() &&
               Lda()==o.Lda() &&
               Ldb()==o.Ldb() &&
               Ldc()==o.Ldc() &&
               MatType()==o.MatType() &&
               ComputeType()==o.ComputeType() &&
               BatchCount()==o.BatchCount() &&
               StrideA()==o.StrideA() &&
               StrideB()==o.StrideB() &&
               StrideC()==o.StrideC() &&
               EpilogueBias()==o.EpilogueBias();
      }

      private:
        cublasOperation_t _op_a;
        cublasOperation_t _op_b;
        int _m;
        int _n;
        int _k;
        int _lda;
        int _ldb;
        int _ldc;
        cudaDataType_t _mat_type;
        cublasComputeType_t _compute_type;
        int _batch_count;
        long long _stride_a;
        long long _stride_b;
        long long _stride_c;
        uint8_t _epilogue_bias;
    };

    struct Hasher_t
    {
      size_t operator()(const Key_t &k)const
      {
        size_t h=static_cast<size_t>(k.OpA());
        h=h*1315423911u+static_cast<size_t>(k.OpB());
        h=h*1315423911u+static_cast<size_t>(k.M());
        h=h*1315423911u+static_cast<size_t>(k.N());
        h=h*1315423911u+static_cast<size_t>(k.K());
        h=h*1315423911u+static_cast<size_t>(k.Lda());
        h=h*1315423911u+static_cast<size_t>(k.Ldb());
        h=h*1315423911u+static_cast<size_t>(k.Ldc());
        h=h*1315423911u+static_cast<size_t>(k.MatType());
        h=h*1315423911u+static_cast<size_t>(k.ComputeType());
        h=h*1315423911u+static_cast<size_t>(k.BatchCount());
        h=h*1315423911u+static_cast<size_t>(k.StrideA());
        h=h*1315423911u+static_cast<size_t>(k.StrideB());
        h=h*1315423911u+static_cast<size_t>(k.StrideC());
        h=h*1315423911u+static_cast<size_t>(k.EpilogueBias());
        return h;
      }
    };

    typedef std::unordered_map<Key_t,cublasLtMatmulAlgo_t,Hasher_t> CacheMap_t;

    static CAIF_MatMulAlgoCache &Instance()
    {
      static CAIF_MatMulAlgoCache s_instance;
      return s_instance;
    }

    bool Lookup(const Key_t &k,cublasLtMatmulAlgo_t &out)
    {
      std::lock_guard<std::mutex> lk(Mutex());
      CacheMap_t::const_iterator it=Map().find(k);
      if(it==Map().end())
      {
        return false;
      }
      out=it->second;
      return true;
    }

    void Store(const Key_t &k,const cublasLtMatmulAlgo_t &algo)
    {
      std::lock_guard<std::mutex> lk(Mutex());
      Map()[k]=algo;
    }

    // Map CAIF dtype to cuBLAS matrix element type. Throws for dtypes that
    // do not belong on the gemm path (Int8/Int4 weight-only quant dequants
    // first; see CAIF int8/int4 kernel task).
    static cudaDataType_t MatrixTypeFor(CAIF_DataType::CAIF_DataType_e dt);

    // Shared cuBLAS-Lt execution helper: builds op+layout descriptors, looks
    // up the algo in the shape-keyed cache, probes on miss, executes. All
    // matmul variants (Mat, MatBias, MatTransposeA, MatTransposeB, Batched*)
    // share this path so the cache and algo-pick benefit applies uniformly.
    // Args are cuBLAS-Lt-side (caller has already applied the row-major
    // trick: for row-major C = op_A(A) * op_B(B), pass first_ptr=B_ptr with
    // op_a/layout encoding B^T's col-major view, second_ptr=A_ptr with
    // op_b/layout encoding A's col-major view). batch_count==0 means a
    // single gemm (strides ignored).
    static cublasStatus_t LtMatMulExecuteCached(cublasLtHandle_t lt_handle,
                                                cudaStream_t stream,
                                                cublasOperation_t op_a,
                                                cublasOperation_t op_b,
                                                int m,
                                                int n,
                                                int k,
                                                const float *alpha,
                                                const void *first_ptr,
                                                int first_rows,
                                                int first_cols,
                                                int lda,
                                                const void *second_ptr,
                                                int second_rows,
                                                int second_cols,
                                                int ldb,
                                                const float *beta,
                                                void *c_ptr,
                                                int ldc,
                                                int batch_count,
                                                long long stride_a,
                                                long long stride_b,
                                                long long stride_c,
                                                cudaDataType_t mat_type,
                                                cublasComputeType_t compute_type,
                                                const void *bias_ptr,
                                                void *workspace,
                                                size_t workspace_size);

  protected:

  private:
    CAIF_MatMulAlgoCache()=default;
    ~CAIF_MatMulAlgoCache()=default;
    CAIF_MatMulAlgoCache(const CAIF_MatMulAlgoCache &)=delete;
    CAIF_MatMulAlgoCache &operator=(const CAIF_MatMulAlgoCache &)=delete;

    // Probe cublasLt's top candidate algos for this op+layout combo, time
    // each on the current stream against real data pointers, and return the
    // fastest. Called once per cache-miss shape; subsequent calls hit the
    // cache and skip probing entirely. Requires beta==0 on the caller (the
    // probe overwrites output multiple times; beta==0 makes the last write
    // the correct answer so no scratch buffer is needed).
    static bool ProbeAndCacheBestAlgo(cublasLtHandle_t lt_handle,
                                      cublasLtMatmulDesc_t op_desc,
                                      cublasLtMatrixLayout_t a_desc,
                                      cublasLtMatrixLayout_t b_desc,
                                      cublasLtMatrixLayout_t c_desc,
                                      const void *alpha,
                                      const void *a_data,
                                      const void *b_data,
                                      const void *beta,
                                      void *c_data,
                                      void *workspace,
                                      size_t workspace_size,
                                      cudaStream_t stream,
                                      cublasLtMatmulAlgo_t &best_algo_out);

    static constexpr int _probe_candidates=g_caif_matmul_probe_candidates_default;
    static constexpr int _probe_iters=g_caif_matmul_probe_iters_default;

    // Internal accessors for the singleton's map + mutex.
    CacheMap_t &Map(){return _map;}
    std::mutex &Mutex(){return _mutex;}

    CacheMap_t _map;
    std::mutex _mutex;
};

}//end instance namespace

#endif // USE_CAIF_CUDA
