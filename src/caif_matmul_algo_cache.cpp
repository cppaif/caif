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
// CAIF_MatMulAlgoCache out-of-line methods: the cuBLAS-Lt dtype map
// (MatrixTypeFor), the shared cached-execute path (LtMatMulExecuteCached),
// and the heuristic probe run on a cache miss (ProbeAndCacheBestAlgo).
//------------------------------------------------------------------------------
#ifdef USE_CAIF_CUDA

#include "caif_matmul_algo_cache.h"
#include "caif_constants.h"
#include "caif_exception.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace instance
{

cudaDataType_t CAIF_MatMulAlgoCache::MatrixTypeFor(CAIF_DataType::CAIF_DataType_e dt)
{
  try
  {
    switch(dt)
    {
      case CAIF_DataType::CAIF_DataType_e::Float32:
        return CUDA_R_32F;
      case CAIF_DataType::CAIF_DataType_e::Float16:
        return CUDA_R_16F;
      case CAIF_DataType::CAIF_DataType_e::BFloat16:
        return CUDA_R_16BF;
      default:
        break;
    }
    THROW_CAIFE("cuBLAS matmul: unsupported dtype (expected fp32/fp16/bf16)");
  }
  CAIF_CATCH_BLOCK();
  return CUDA_R_32F;
}

bool CAIF_MatMulAlgoCache::ProbeAndCacheBestAlgo(cublasLtHandle_t lt_handle,
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
                                                 cublasLtMatmulAlgo_t &best_algo_out)
{
  try
  {
    cublasLtMatmulPreference_t pref=nullptr;
    cublasStatus_t s=cublasLtMatmulPreferenceCreate(&pref);
    if(s!=CUBLAS_STATUS_SUCCESS)
    {
      return false;
    }
    cublasLtMatmulPreferenceSetAttribute(pref,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace_size,
                                         sizeof(workspace_size));

    cublasLtMatmulHeuristicResult_t results[CAIF_MatMulAlgoCache::ProbeCandidates()];
    std::memset(&results,0,sizeof(results));
    int returned=0;
    s=cublasLtMatmulAlgoGetHeuristic(lt_handle,
                                     op_desc,
                                     a_desc,
                                     b_desc,
                                     c_desc,
                                     c_desc,
                                     pref,
                                     CAIF_MatMulAlgoCache::ProbeCandidates(),
                                     results,
                                     &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if(s!=CUBLAS_STATUS_SUCCESS || returned<1)
    {
      return false;
    }

    // Rank algos by MIN-of-iters, not SUM/MEAN. A single slow iter (cold
    // caches, TLB warmup, async scheduler hiccup) can otherwise mask an
    // algo whose steady-state is best. Matches the peak-of-N rule used by
    // the benchmark pipeline itself.
    cudaEvent_t ev_per_iter[2*CAIF_MatMulAlgoCache::ProbeIters()];
    for(int e=0;e<2*CAIF_MatMulAlgoCache::ProbeIters();++e)
    {
      cudaEventCreate(&ev_per_iter[e]);
    }

    int best_idx=-1;
    float best_ms=1.0e30f;
    for(int i=0;i<returned;++i)
    {
      if(results[i].state!=CUBLAS_STATUS_SUCCESS)
      {
        continue;
      }
      cublasStatus_t warm=cublasLtMatmul(lt_handle,
                                         op_desc,
                                         alpha,
                                         a_data,a_desc,
                                         b_data,b_desc,
                                         beta,
                                         c_data,c_desc,
                                         c_data,c_desc,
                                         &results[i].algo,
                                         workspace,workspace_size,
                                         stream);
      if(warm!=CUBLAS_STATUS_SUCCESS)
      {
        continue;
      }
      bool probe_ok=true;
      for(int iter=0;iter<CAIF_MatMulAlgoCache::ProbeIters();++iter)
      {
        cudaEventRecord(ev_per_iter[2*iter],stream);
        cublasStatus_t rs=cublasLtMatmul(lt_handle,
                                         op_desc,
                                         alpha,
                                         a_data,a_desc,
                                         b_data,b_desc,
                                         beta,
                                         c_data,c_desc,
                                         c_data,c_desc,
                                         &results[i].algo,
                                         workspace,workspace_size,
                                         stream);
        cudaEventRecord(ev_per_iter[2*iter+1],stream);
        if(rs!=CUBLAS_STATUS_SUCCESS)
        {
          probe_ok=false;
          break;
        }
      }
      cudaEventSynchronize(ev_per_iter[2*CAIF_MatMulAlgoCache::ProbeIters()-1]);
      if(probe_ok==false)
      {
        continue;
      }
      float min_ms=1.0e30f;
      for(int iter=0;iter<CAIF_MatMulAlgoCache::ProbeIters();++iter)
      {
        float iter_ms=0.0f;
        cudaEventElapsedTime(&iter_ms,ev_per_iter[2*iter],ev_per_iter[2*iter+1]);
        if(iter_ms<min_ms)
        {
          min_ms=iter_ms;
        }
      }
      if(min_ms<best_ms)
      {
        best_ms=min_ms;
        best_idx=i;
      }
    }

    for(int e=0;e<2*CAIF_MatMulAlgoCache::ProbeIters();++e)
    {
      cudaEventDestroy(ev_per_iter[e]);
    }

    if(best_idx<0)
    {
      return false;
    }
    best_algo_out=results[best_idx].algo;

    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

cublasStatus_t CAIF_MatMulAlgoCache::LtMatMulExecuteCached(cublasLtHandle_t lt_handle,
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
                                                           size_t workspace_size)
{
  try
  {
    cublasLtMatmulDesc_t op_desc=nullptr;
    cublasStatus_t s=cublasLtMatmulDescCreate(&op_desc,compute_type,CUDA_R_32F);
    if(s!=CUBLAS_STATUS_SUCCESS)
    {
      return s;
    }
    cublasLtMatmulDescSetAttribute(op_desc,
                                   CUBLASLT_MATMUL_DESC_TRANSA,
                                   &op_a,
                                   sizeof(op_a));
    cublasLtMatmulDescSetAttribute(op_desc,
                                   CUBLASLT_MATMUL_DESC_TRANSB,
                                   &op_b,
                                   sizeof(op_b));
    if(bias_ptr!=nullptr)
    {
      const cublasLtEpilogue_t epi=CUBLASLT_EPILOGUE_BIAS;
      cublasLtMatmulDescSetAttribute(op_desc,
                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     &epi,
                                     sizeof(epi));
      cublasLtMatmulDescSetAttribute(op_desc,
                                     CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                     &bias_ptr,
                                     sizeof(bias_ptr));
    }

    cublasLtMatrixLayout_t a_desc=nullptr;
    cublasLtMatrixLayout_t b_desc=nullptr;
    cublasLtMatrixLayout_t c_desc=nullptr;
    cublasLtMatrixLayoutCreate(&a_desc,mat_type,first_rows,first_cols,lda);
    cublasLtMatrixLayoutCreate(&b_desc,mat_type,second_rows,second_cols,ldb);
    cublasLtMatrixLayoutCreate(&c_desc,mat_type,m,n,ldc);

    if(batch_count>0)
    {
      const int32_t bc=batch_count;
      cublasLtMatrixLayoutSetAttribute(a_desc,
                                       CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &bc,
                                       sizeof(bc));
      cublasLtMatrixLayoutSetAttribute(b_desc,
                                       CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &bc,
                                       sizeof(bc));
      cublasLtMatrixLayoutSetAttribute(c_desc,
                                       CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &bc,
                                       sizeof(bc));
      cublasLtMatrixLayoutSetAttribute(a_desc,
                                       CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                       &stride_a,
                                       sizeof(stride_a));
      cublasLtMatrixLayoutSetAttribute(b_desc,
                                       CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                       &stride_b,
                                       sizeof(stride_b));
      cublasLtMatrixLayoutSetAttribute(c_desc,
                                       CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                       &stride_c,
                                       sizeof(stride_c));
    }

    CAIF_MatMulAlgoCache::Key_t key;
    key.SetOpA(op_a);
    key.SetOpB(op_b);
    key.SetM(m);
    key.SetN(n);
    key.SetK(k);
    key.SetLda(lda);
    key.SetLdb(ldb);
    key.SetLdc(ldc);
    key.SetMatType(mat_type);
    key.SetComputeType(compute_type);
    key.SetBatchCount(batch_count);
    key.SetStrideA(stride_a);
    key.SetStrideB(stride_b);
    key.SetStrideC(stride_c);
    uint8_t epilogue_bias=static_cast<uint8_t>(0);
    if(bias_ptr!=nullptr)
    {
      epilogue_bias=static_cast<uint8_t>(1);
    }
    key.SetEpilogueBias(epilogue_bias);

    cublasLtMatmulAlgo_t algo;
    CAIF_MatMulAlgoCache &cache=CAIF_MatMulAlgoCache::Instance();
    const bool have_cached=cache.Lookup(key,algo);

    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
    if(g_caif_matmul_skip_probe==true)
    {
//      cudaEvent_t ev_s=nullptr;
//      cudaEvent_t ev_e=nullptr;
//      if(g_caif_matmul_trace_enabled==true)
//      {
//        cudaEventCreate(&ev_s);
//        cudaEventCreate(&ev_e);
//        cudaEventRecord(ev_s,stream);
//      }
      status=cublasLtMatmul(lt_handle,
                            op_desc,
                            alpha,
                            first_ptr,a_desc,
                            second_ptr,b_desc,
                            beta,
                            c_ptr,c_desc,
                            c_ptr,c_desc,
                            nullptr,
                            workspace,workspace_size,
                            stream);
//      if(g_caif_matmul_trace_enabled==true)
//      {
//        cudaEventRecord(ev_e,stream);
//        cudaEventSynchronize(ev_e);
//        float ms=0.0f;
//        cudaEventElapsedTime(&ms,ev_s,ev_e);
//        const uint64_t flops=2ull*static_cast<uint64_t>(m)
//                                *static_cast<uint64_t>(n)
//                                *static_cast<uint64_t>(k);
//        const double tflops=(static_cast<double>(flops)*g_caif_matmul_tflops_per_flop)
//                            /(static_cast<double>(ms)*g_caif_matmul_seconds_per_ms);
//        std::fprintf(stderr,
//                     "[MM-NOPROBE] m=%d n=%d k=%d ct=%d ws=%zu "
//                     "ms=%.3f tflops=%.2f\n",
//                     m,
//                     n,
//                     k,
//                     static_cast<int>(compute_type),
//                     workspace_size,
//                     static_cast<double>(ms),
//                     tflops);
//        cudaEventDestroy(ev_s);
//        cudaEventDestroy(ev_e);
//      }
    }
    else if(have_cached==true)
    {
//      cudaEvent_t ev_s=nullptr;
//      cudaEvent_t ev_e=nullptr;
//      if(g_caif_matmul_trace_enabled==true)
//      {
//        cudaEventCreate(&ev_s);
//        cudaEventCreate(&ev_e);
//        cudaEventRecord(ev_s,stream);
//      }
      status=cublasLtMatmul(lt_handle,
                            op_desc,
                            alpha,
                            first_ptr,a_desc,
                            second_ptr,b_desc,
                            beta,
                            c_ptr,c_desc,
                            c_ptr,c_desc,
                            &algo,
                            workspace,workspace_size,
                            stream);
//      if(g_caif_matmul_trace_enabled==true)
//      {
//        cudaEventRecord(ev_e,stream);
//        cudaEventSynchronize(ev_e);
//        float ms=0.0f;
//        cudaEventElapsedTime(&ms,ev_s,ev_e);
//        const uint64_t flops=2ull*static_cast<uint64_t>(m)
//                                *static_cast<uint64_t>(n)
//                                *static_cast<uint64_t>(k);
//        const double tflops=(static_cast<double>(flops)*g_caif_matmul_tflops_per_flop)
//                            /(static_cast<double>(ms)*g_caif_matmul_seconds_per_ms);
//        const uintptr_t pa=reinterpret_cast<uintptr_t>(first_ptr);
//        const uintptr_t pb=reinterpret_cast<uintptr_t>(second_ptr);
//        const uintptr_t pc=reinterpret_cast<uintptr_t>(c_ptr);
//        const uintptr_t pw=reinterpret_cast<uintptr_t>(workspace);
//        const unsigned int amod=g_caif_matmul_trace_alignment_modulus;
//        std::fprintf(stderr,
//                     "[MM] m=%d n=%d k=%d ct=%d ws=%zu "
//                     "ms=%.3f tflops=%.2f "
//                     "a=%lx(mod%u=%lu) b=%lx(mod%u=%lu) "
//                     "c=%lx(mod%u=%lu) w=%lx(mod%u=%lu)\n",
//                     m,
//                     n,
//                     k,
//                     static_cast<int>(compute_type),
//                     workspace_size,
//                     static_cast<double>(ms),
//                     tflops,
//                     static_cast<unsigned long>(pa),
//                     amod,
//                     static_cast<unsigned long>(pa%amod),
//                     static_cast<unsigned long>(pb),
//                     amod,
//                     static_cast<unsigned long>(pb%amod),
//                     static_cast<unsigned long>(pc),
//                     amod,
//                     static_cast<unsigned long>(pc%amod),
//                     static_cast<unsigned long>(pw),
//                     amod,
//                     static_cast<unsigned long>(pw%amod));
//        cudaEventDestroy(ev_s);
//        cudaEventDestroy(ev_e);
//      }
    }
    else
    {
      const bool probed=ProbeAndCacheBestAlgo(lt_handle,
                                              op_desc,
                                              a_desc,b_desc,c_desc,
                                              alpha,
                                              first_ptr,second_ptr,
                                              beta,
                                              c_ptr,
                                              workspace,workspace_size,
                                              stream,
                                              algo);
      if(probed==true)
      {
        cache.Store(key,algo);
//        if(g_caif_matmul_trace_enabled==true)
//        {
//          const uint64_t *algo_bytes=reinterpret_cast<const uint64_t*>(algo.data);
//          std::fprintf(stderr,
//                       "[MM-PICK] m=%d n=%d k=%d ct=%d ws=%zu "
//                       "mat_type=%d batch=%d "
//                       "algo[0..3]=%016lx %016lx %016lx %016lx\n",
//                       m,
//                       n,
//                       k,
//                       static_cast<int>(compute_type),
//                       workspace_size,
//                       static_cast<int>(mat_type),
//                       batch_count,
//                       static_cast<unsigned long>(algo_bytes[0]),
//                       static_cast<unsigned long>(algo_bytes[1]),
//                       static_cast<unsigned long>(algo_bytes[2]),
//                       static_cast<unsigned long>(algo_bytes[3]));
//        }
      }
      else
      {
        // Fallback: no probe-ranked algo available; let cuBLAS-Lt default pick.
        status=cublasLtMatmul(lt_handle,
                              op_desc,
                              alpha,
                              first_ptr,a_desc,
                              second_ptr,b_desc,
                              beta,
                              c_ptr,c_desc,
                              c_ptr,c_desc,
                              nullptr,
                              workspace,workspace_size,
                              stream);
      }
    }

    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatmulDescDestroy(op_desc);
    return status;
  }
  CAIF_CATCH_BLOCK();
  return CUBLAS_STATUS_INTERNAL_ERROR;
}

}//end instance namespace

#endif // USE_CAIF_CUDA
