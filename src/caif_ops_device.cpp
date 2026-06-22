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
// CAIF_Ops device backend (cuBLAS / cuBLAS-Lt / cuDNN / custom CUDA kernels).
//
// Every public CAIF_Ops::Foo(...) entry in caif_ops.cpp dispatches on tensor
// location and forwards to either FooDevice(...) (here) or FooHost(...)
// (caif_ops_host.cpp).
//------------------------------------------------------------------------------
#include "caif_ops.h"
#include "caif_matmul_algo_cache.h"
#include "caif_constants.h"
#include "caif_device_context.h"
#include "caif_device_network.h"
#include "caif_run_context.h"
#include "caif_cuda_kernels_attention_support.cuh"
#include "caif_cuda_kernels_embeddings.cuh"
#include "caif_cuda_kernels_loss.cuh"
#include "caif_cuda_kernels_moe.cuh"
#include "caif_cuda_kernels_optimizers.cuh"
#include "caif_cuda_kernels_quant.cuh"
#include "caif_cuda_kernels_tensor_ops.cuh"
#include "caif_cuda_kernels_elementwise.cuh"
#include "caif_cuda_kernels_activations.cuh"
#include "caif_exception.h"
#include "caif_settings.h"
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#endif

namespace instance
{


//------------------------------------------------------------------------------
// Matrix Operations
//------------------------------------------------------------------------------

void CAIF_Ops::MatMulDevice(const CAIF_DeviceTensor &a,
                  const CAIF_DeviceTensor &b,
                  CAIF_DeviceTensor &output,
                  CAIF_RunContext &ctx,
                  const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
#ifdef USE_CAIF_CUDA
  const auto &shape_a=a.Shape();
  const auto &shape_b=b.Shape();
  const auto &shape_out=output.Shape();

  if(shape_a.size()!=2||shape_b.size()!=2||shape_out.size()!=2)
  {
    THROW_CAIFE("MatMul requires 2D tensors");
  }

  const int rows_a=static_cast<int>(shape_a[0]);
  const int cols_a=static_cast<int>(shape_a[1]);
  const int rows_b=static_cast<int>(shape_b[0]);
  const int cols_b=static_cast<int>(shape_b[1]);

  if(cols_a!=rows_b)
  {
    THROW_CAIFE("MatMul: A columns must equal B rows");
  }
  if(shape_out[0]!=shape_a[0]||shape_out[1]!=shape_b[1])
  {
    THROW_CAIFE("MatMul: output shape mismatch");
  }

  RequireMatchingDtype(a,b,output,g_caif_op_matmul);

  CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
  cublasLtHandle_t lt_handle=device_ctx.CublasLtHandle();
  const cudaStream_t stream=output.Stream().Handle();

  const cudaDataType_t mat_type=CAIF_MatMulAlgoCache::MatrixTypeFor(a.Dtype());
  const cublasComputeType_t compute_type=static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(a.Dtype(),
                                                                                             compute_dtype));

  void *use_ws=device_ctx.CublasLtWorkspace();
  size_t use_ws_size=device_ctx.CublasLtWorkspaceSize();

  // Row-major trick: C = A*B (row-major) ≡ C^T = B^T * A^T in cuBLAS
  // column-major terms. Pass B_ptr first with col-major-view-of-B = B^T (N,K,N),
  // A_ptr second with col-major-view-of-A = A^T (K,M,K), both ops N.
  const float alpha=1.0f;
  const float beta=0.0f;
  cublasStatus_t status=CAIF_MatMulAlgoCache::LtMatMulExecuteCached(lt_handle,
                                                                    stream,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    cols_b,rows_a,cols_a,
                                                                    &alpha,
                                                                    b.DeviceDataRaw(),
                                                                    cols_b,cols_a,cols_b,
                                                                    a.DeviceDataRaw(),
                                                                    cols_a,rows_a,cols_a,
                                                                    &beta,
                                                                    output.DeviceDataRaw(),
                                                                    cols_b,
                                                                    0,
                                                                    0,0,0,
                                                                    mat_type,
                                                                    compute_type,
                                                                    nullptr,
                                                                    use_ws,
                                                                    use_ws_size);
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cublasLt MatMul failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)ctx;
  (void)compute_dtype;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MatMulBiasDevice(const CAIF_DeviceTensor &a,
                      const CAIF_DeviceTensor &b,
                      const CAIF_DeviceTensor &bias,
                      CAIF_DeviceTensor &output,
                      cudaStream_t stream,
                      CAIF_RunContext &ctx,
                      const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
#ifdef USE_CAIF_CUDA
  const auto &shape_a=a.Shape();
  const auto &shape_b=b.Shape();

  RequireMatchingDtype(a,b,output,g_caif_op_matmul_bias);
  if(bias.Dtype()!=a.Dtype())
  {
    THROW_CAIFE("cuBLAS MatMulBias: bias dtype must match matrices");
  }

  const int rows_a=static_cast<int>(shape_a[0]);
  const int cols_a=static_cast<int>(shape_a[1]);
  const int cols_b=static_cast<int>(shape_b[1]);

  CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
  cublasLtHandle_t lt_handle=device_ctx.CublasLtHandle();

  const cudaDataType_t mat_type=CAIF_MatMulAlgoCache::MatrixTypeFor(a.Dtype());
  const cublasComputeType_t compute_type=static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(a.Dtype(),
                                                                                             compute_dtype));

  // Row-major trick with BIAS epilogue. Same layout as plain MatMul:
  // first=B with col-major (N,K,N), second=A with col-major (K,M,K), both N.
  const float alpha=1.0f;
  const float beta=0.0f;
  cublasStatus_t status=CAIF_MatMulAlgoCache::LtMatMulExecuteCached(lt_handle,
                                                                    stream,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    cols_b,rows_a,cols_a,
                                                                    &alpha,
                                                                    b.DeviceDataRaw(),
                                                                    cols_b,cols_a,cols_b,
                                                                    a.DeviceDataRaw(),
                                                                    cols_a,rows_a,cols_a,
                                                                    &beta,
                                                                    output.DeviceDataRaw(),
                                                                    cols_b,
                                                                    0,
                                                                    0,0,0,
                                                                    mat_type,
                                                                    compute_type,
                                                                    bias.DeviceDataRaw(),
                                                                    device_ctx.CublasLtWorkspace(),
                                                                    device_ctx.CublasLtWorkspaceSize());
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cublasLt MatMulBias failed");
  }
#else
  (void)a;
  (void)b;
  (void)bias;
  (void)output;
  (void)stream;
  (void)ctx;
  (void)compute_dtype;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MatMulTransposeADevice(const CAIF_DeviceTensor &a,
                            const CAIF_DeviceTensor &b,
                            CAIF_DeviceTensor &output,
                            CAIF_RunContext &ctx,
                            const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
#ifdef USE_CAIF_CUDA
  // A^T * B: A is [K x M], transposed to [M x K], B is [K x N], result is [M x N]
  const auto &shape_a=a.Shape();
  const auto &shape_b=b.Shape();
  const auto &shape_out=output.Shape();

  if(shape_a.size()!=2||shape_b.size()!=2||shape_out.size()!=2)
  {
    THROW_CAIFE("MatMulTransposeA requires 2D tensors");
  }

  const int a_rows=static_cast<int>(shape_a[0]);  // K (rows of physical A)
  const int a_cols=static_cast<int>(shape_a[1]);  // M (cols of physical A)
  const int b_rows=static_cast<int>(shape_b[0]);  // K
  const int b_cols=static_cast<int>(shape_b[1]);  // N

  // After transpose: A^T is [M x K]
  // Check: K (a_rows) must equal K (b_rows)
  if(a_rows!=b_rows)
  {
    THROW_CAIFE("MatMulTransposeA: A rows must equal B rows");
  }

  const int m=a_cols;  // Rows of A^T
  const int n=b_cols;  // Cols of B
  const int k=a_rows;  // Cols of A^T = Rows of B

  if(shape_out[0]!=static_cast<uint32_t>(m)||shape_out[1]!=static_cast<uint32_t>(n))
  {
    THROW_CAIFE("MatMulTransposeA: output shape mismatch");
  }

  RequireMatchingDtype(a,b,output,g_caif_op_matmul_transpose_a);

  CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
  const cudaStream_t stream=output.Stream().Handle();

  const float alpha=1.0f;
  const float beta=0.0f;
  const cudaDataType_t mat_type=CAIF_MatMulAlgoCache::MatrixTypeFor(a.Dtype());
  const cublasComputeType_t compute_type=static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(a.Dtype(),
                                                                                             compute_dtype));

  // Row-major: C = A^T * B. Lt-side: op_a=N on B (layout N,K,N), op_b=T on A
  // (layout M,K,M). Lt m,n,k = N,M,K. Matches the legacy cublasGemmEx(N,T,...)
  // formulation but routed through the cached-algo path.
  (void)a_cols;(void)b_cols;(void)a_rows;(void)b_rows;
  cublasStatus_t status=CAIF_MatMulAlgoCache::LtMatMulExecuteCached(device_ctx.CublasLtHandle(),
                                                                    stream,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_T,
                                                                    n,m,k,
                                                                    &alpha,
                                                                    b.DeviceDataRaw(),
                                                                    n,k,n,
                                                                    a.DeviceDataRaw(),
                                                                    m,k,m,
                                                                    &beta,
                                                                    output.DeviceDataRaw(),
                                                                    n,
                                                                    0,
                                                                    0,0,0,
                                                                    mat_type,
                                                                    compute_type,
                                                                    nullptr,
                                                                    device_ctx.CublasLtWorkspace(),
                                                                    device_ctx.CublasLtWorkspaceSize());
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cublasLt MatMulTransposeA failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)ctx;
  (void)compute_dtype;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MatMulTransposeBDevice(const CAIF_DeviceTensor &a,
                            const CAIF_DeviceTensor &b,
                            CAIF_DeviceTensor &output,
                            CAIF_RunContext &ctx,
                            const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
#ifdef USE_CAIF_CUDA
  // A * B^T: A is [M x K], B is [N x K], transposed to [K x N], result is [M x N]
  const auto &shape_a=a.Shape();
  const auto &shape_b=b.Shape();
  const auto &shape_out=output.Shape();

  if(shape_a.size()!=2||shape_b.size()!=2||shape_out.size()!=2)
  {
    THROW_CAIFE("MatMulTransposeB requires 2D tensors");
  }

  const int a_rows=static_cast<int>(shape_a[0]);  // M
  const int a_cols=static_cast<int>(shape_a[1]);  // K
  const int b_rows=static_cast<int>(shape_b[0]);  // N
  const int b_cols=static_cast<int>(shape_b[1]);  // K

  // After transpose: B^T is [K x N]
  // Check: K (a_cols) must equal K (b_cols)
  if(a_cols!=b_cols)
  {
    THROW_CAIFE("MatMulTransposeB: A cols must equal B cols");
  }

  const int m=a_rows;  // Rows of A
  const int n=b_rows;  // Rows of B (cols of B^T)
  const int k=a_cols;  // Cols of A

  if(shape_out[0]!=static_cast<uint32_t>(m)||shape_out[1]!=static_cast<uint32_t>(n))
  {
    THROW_CAIFE("MatMulTransposeB: output shape mismatch");
  }

  RequireMatchingDtype(a,b,output,g_caif_op_matmul_transpose_b);

  CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
  const cudaStream_t stream=output.Stream().Handle();

  const float alpha=1.0f;
  const float beta=0.0f;
  const cudaDataType_t mat_type=CAIF_MatMulAlgoCache::MatrixTypeFor(a.Dtype());
  const cublasComputeType_t compute_type=static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(a.Dtype(),
                                                                                             compute_dtype));

  // Row-major: C = A * B^T. Lt-side: op_a=T on B (layout K,N,K), op_b=N on A
  // (layout K,M,K). Lt m,n,k = N,M,K.
  (void)a_rows;(void)b_rows;(void)a_cols;(void)b_cols;
  cublasStatus_t status=CAIF_MatMulAlgoCache::LtMatMulExecuteCached(device_ctx.CublasLtHandle(),
                                                                    stream,
                                                                    CUBLAS_OP_T,
                                                                    CUBLAS_OP_N,
                                                                    n,m,k,
                                                                    &alpha,
                                                                    b.DeviceDataRaw(),
                                                                    k,n,k,
                                                                    a.DeviceDataRaw(),
                                                                    k,m,k,
                                                                    &beta,
                                                                    output.DeviceDataRaw(),
                                                                    n,
                                                                    0,
                                                                    0,0,0,
                                                                    mat_type,
                                                                    compute_type,
                                                                    nullptr,
                                                                    device_ctx.CublasLtWorkspace(),
                                                                    device_ctx.CublasLtWorkspaceSize());
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cublasLt MatMulTransposeB failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)ctx;
  (void)compute_dtype;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Batched Matrix Operations
//------------------------------------------------------------------------------

void CAIF_Ops::BatchedMatMulDevice(const CAIF_DeviceTensor &a,
                         const CAIF_DeviceTensor &b,
                         CAIF_DeviceTensor &output,
                         int m,
                         int k,
                         int n,
                         int batch_count,
                         CAIF_RunContext &ctx,
                         const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
#ifdef USE_CAIF_CUDA
  RequireMatchingDtype(a,b,output,g_caif_op_batched_matmul);

  CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
  const cudaStream_t stream=output.Stream().Handle();

  const float alpha=1.0f;
  const float beta=0.0f;
  const cudaDataType_t mat_type=CAIF_MatMulAlgoCache::MatrixTypeFor(a.Dtype());
  const cublasComputeType_t compute_type=static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(a.Dtype(),
                                                                                             compute_dtype));

  const long long int stride_a=static_cast<long long int>(m)*k;
  const long long int stride_b=static_cast<long long int>(k)*n;
  const long long int stride_c=static_cast<long long int>(m)*n;

  cublasStatus_t status=CAIF_MatMulAlgoCache::LtMatMulExecuteCached(device_ctx.CublasLtHandle(),
                                                                    stream,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    n,m,k,
                                                                    &alpha,
                                                                    b.DeviceDataRaw(),
                                                                    n,k,n,
                                                                    a.DeviceDataRaw(),
                                                                    k,m,k,
                                                                    &beta,
                                                                    output.DeviceDataRaw(),
                                                                    n,
                                                                    batch_count,
                                                                    stride_b,stride_a,stride_c,
                                                                    mat_type,
                                                                    compute_type,
                                                                    nullptr,
                                                                    device_ctx.CublasLtWorkspace(),
                                                                    device_ctx.CublasLtWorkspaceSize());
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cublasLt BatchedMatMul failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
  (void)batch_count;
  (void)ctx;
  (void)compute_dtype;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::BatchedMatMulTransposeADevice(const CAIF_DeviceTensor &a,
                                   const CAIF_DeviceTensor &b,
                                   CAIF_DeviceTensor &output,
                                   int k,
                                   int m,
                                   int n,
                                   int batch_count,
                                   CAIF_RunContext &ctx,
                                   const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
#ifdef USE_CAIF_CUDA
  // C = A^T * B per batch. A[K,M], B[K,N], C[M,N].
  RequireMatchingDtype(a,b,output,g_caif_op_batched_matmul_transpose_a);

  CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
  const cudaStream_t stream=output.Stream().Handle();

  const float alpha=1.0f;
  const float beta=0.0f;
  const cudaDataType_t mat_type=CAIF_MatMulAlgoCache::MatrixTypeFor(a.Dtype());
  const cublasComputeType_t compute_type=static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(a.Dtype(),
                                                                                             compute_dtype));

  const long long int stride_a=static_cast<long long int>(k)*m;
  const long long int stride_b=static_cast<long long int>(k)*n;
  const long long int stride_c=static_cast<long long int>(m)*n;

  cublasStatus_t status=CAIF_MatMulAlgoCache::LtMatMulExecuteCached(device_ctx.CublasLtHandle(),
                                                                    stream,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_T,
                                                                    n,m,k,
                                                                    &alpha,
                                                                    b.DeviceDataRaw(),
                                                                    n,k,n,
                                                                    a.DeviceDataRaw(),
                                                                    m,k,m,
                                                                    &beta,
                                                                    output.DeviceDataRaw(),
                                                                    n,
                                                                    batch_count,
                                                                    stride_b,stride_a,stride_c,
                                                                    mat_type,
                                                                    compute_type,
                                                                    nullptr,
                                                                    device_ctx.CublasLtWorkspace(),
                                                                    device_ctx.CublasLtWorkspaceSize());
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cublasLt BatchedMatMulTransposeA failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)k;
  (void)m;
  (void)n;
  (void)batch_count;
  (void)ctx;
  (void)compute_dtype;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::BatchedMatMulTransposeBDevice(const CAIF_DeviceTensor &a,
                                   const CAIF_DeviceTensor &b,
                                   CAIF_DeviceTensor &output,
                                   int m,
                                   int k,
                                   int n,
                                   int batch_count,
                                   CAIF_RunContext &ctx,
                                   const CAIF_DataType::CAIF_DataType_e compute_dtype)
{
#ifdef USE_CAIF_CUDA
  // C = A * B^T per batch. A[M,K], B[N,K], C[M,N].
  RequireMatchingDtype(a,b,output,g_caif_op_batched_matmul_transpose_b);

  CAIF_DeviceContext &device_ctx=CAIF_DeviceContext::Instance();
  const cudaStream_t stream=output.Stream().Handle();

  const float alpha=1.0f;
  const float beta=0.0f;
  const cudaDataType_t mat_type=CAIF_MatMulAlgoCache::MatrixTypeFor(a.Dtype());
  const cublasComputeType_t compute_type=static_cast<cublasComputeType_t>(ctx.ComputeTypeFor(a.Dtype(),
                                                                                             compute_dtype));

  const long long int stride_a=static_cast<long long int>(m)*k;
  const long long int stride_b=static_cast<long long int>(n)*k;
  const long long int stride_c=static_cast<long long int>(m)*n;

  cublasStatus_t status=CAIF_MatMulAlgoCache::LtMatMulExecuteCached(device_ctx.CublasLtHandle(),
                                                                    stream,
                                                                    CUBLAS_OP_T,
                                                                    CUBLAS_OP_N,
                                                                    n,m,k,
                                                                    &alpha,
                                                                    b.DeviceDataRaw(),
                                                                    k,n,k,
                                                                    a.DeviceDataRaw(),
                                                                    k,m,k,
                                                                    &beta,
                                                                    output.DeviceDataRaw(),
                                                                    n,
                                                                    batch_count,
                                                                    stride_b,stride_a,stride_c,
                                                                    mat_type,
                                                                    compute_type,
                                                                    nullptr,
                                                                    device_ctx.CublasLtWorkspace(),
                                                                    device_ctx.CublasLtWorkspaceSize());
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cublasLt BatchedMatMulTransposeB failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
  (void)batch_count;
  (void)ctx;
  (void)compute_dtype;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Tensor Manipulation
//------------------------------------------------------------------------------

void CAIF_Ops::TransposeDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape_in=input.Shape();
  const auto &shape_out=output.Shape();

  if(shape_in.size()!=2||shape_out.size()!=2)
  {
    THROW_CAIFE("Transpose requires 2D tensors");
  }

  const uint32_t rows=shape_in[0];
  const uint32_t cols=shape_in[1];

  if(shape_out[0]!=cols||shape_out[1]!=rows)
  {
    THROW_CAIFE("Transpose: output shape must be [cols, rows]");
  }

  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Transpose: input and output dtype must match");
  }

  const cudaStream_t stream=output.Stream().Handle();

  // Dtype dispatch: fp32 uses cublasSgeam (hand-tuned), fp16/bf16 use the
  // templated launch_transpose_0213 kernel modeling 2D [M,N]->[N,M] as a
  // 4D 0213 reshuffle with batch=1, dim2=1: [1,M,N,1] -> [1,N,M,1].
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      // Use cuBLAS geam for transpose: C = alpha*A^T + beta*B
      // With beta=0 and alpha=1, this is just C = A^T
      cublasHandle_t handle=CAIF_DeviceContext::Instance().CublasHandle();
      cublasSetStream(handle,stream);

      const float alpha=1.0f;
      const float beta=0.0f;

      // cuBLAS uses column-major, so we need to think of our row-major [M,N] as column-major [N,M]
      // Transposing column-major [N,M] gives column-major [M,N], which is row-major [N,M]
      // So we call geam with rows/cols swapped
      cublasStatus_t status=cublasSgeam(handle,
                                         CUBLAS_OP_T,       // Transpose A
                                         CUBLAS_OP_N,       // Don't touch B
                                         rows,              // rows of output (cols of input)
                                         cols,              // cols of output (rows of input)
                                         &alpha,
                                         input.DevicePtr<float>(),
                                         cols,              // lda = cols of input
                                         &beta,
                                         nullptr,           // B not used
                                         rows,              // ldb (not used but must be valid)
                                         output.DevicePtr<float>(),
                                         rows);             // ldc = rows of output (= cols of input)

      if(status!=CUBLAS_STATUS_SUCCESS)
      {
        THROW_CAIFE("Transpose: cublasSgeam failed");
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float16:
    {
      launch_transpose_0213<__half>(input.DevicePtr<__half>(),
                                    output.DevicePtr<__half>(),
                                    1,
                                    static_cast<int>(rows),
                                    static_cast<int>(cols),
                                    1,
                                    stream);
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
    {
      launch_transpose_0213<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                           output.DevicePtr<__nv_bfloat16>(),
                                           1,
                                           static_cast<int>(rows),
                                           static_cast<int>(cols),
                                           1,
                                           stream);
      break;
    }
    default:
      THROW_CAIFE("Transpose: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Element-wise Operations
//------------------------------------------------------------------------------

void CAIF_Ops::AddDevice(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(a.TotalElements()!=b.TotalElements()||a.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Add: tensor size mismatch");
  }
  if(a.Dtype()!=b.Dtype() || a.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Add: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(a.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_add<float>(a.DevicePtr<float>(),
                                    b.DevicePtr<float>(),
                                    output.DevicePtr<float>(),
                                    n,
                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_add<__half>(a.DevicePtr<__half>(),
                                     b.DevicePtr<__half>(),
                                     output.DevicePtr<__half>(),
                                     n,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_add<__nv_bfloat16>(a.DevicePtr<__nv_bfloat16>(),
                                            b.DevicePtr<__nv_bfloat16>(),
                                            output.DevicePtr<__nv_bfloat16>(),
                                            n,
                                            stream);
      break;
    default:
      THROW_CAIFE("Add: unsupported dtype");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::ScaleDevice(CAIF_DeviceTensor &tensor,float scale)
{
#ifdef USE_CAIF_CUDA
  const int64_t n=static_cast<int64_t>(tensor.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=tensor.Stream().Handle();
  switch(tensor.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_mul_scalar<float>(tensor.DevicePtr<float>(),
                                           scale,
                                           tensor.DevicePtr<float>(),
                                           n,
                                           stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_mul_scalar<__half>(tensor.DevicePtr<__half>(),
                                            scale,
                                            tensor.DevicePtr<__half>(),
                                            n,
                                            stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_mul_scalar<__nv_bfloat16>(tensor.DevicePtr<__nv_bfloat16>(),
                                                   scale,
                                                   tensor.DevicePtr<__nv_bfloat16>(),
                                                   n,
                                                   stream);
      break;
    default:
      THROW_CAIFE("Scale: unsupported dtype");
  }
#else
  (void)tensor;
  (void)scale;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::UnscaleCheckInfDevice(CAIF_DeviceTensor &grad,
                                     float inv_scale,
                                     CAIF_DeviceTensor &found_inf)
{
#ifdef USE_CAIF_CUDA
  const int64_t n=static_cast<int64_t>(grad.TotalElements());
  if(n==0)
  {
    return;
  }
  if(found_inf.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("UnscaleCheckInf: found_inf must be fp32");
  }
  if(found_inf.TotalElements()!=1)
  {
    THROW_CAIFE("UnscaleCheckInf: found_inf must have exactly 1 element");
  }
  const cudaStream_t stream=grad.Stream().Handle();
  float *flag=found_inf.DevicePtr<float>();
  switch(grad.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_unscale_check_inf<float>(grad.DevicePtr<float>(),inv_scale,flag,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_unscale_check_inf<__half>(grad.DevicePtr<__half>(),inv_scale,flag,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_unscale_check_inf<__nv_bfloat16>(grad.DevicePtr<__nv_bfloat16>(),inv_scale,flag,n,stream);
      break;
    default:
      THROW_CAIFE("UnscaleCheckInf: unsupported grad dtype");
  }
#else
  (void)grad;
  (void)inv_scale;
  (void)found_inf;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::AddScaledDevice(CAIF_DeviceTensor &target,const CAIF_DeviceTensor &source,float scale)
{
#ifdef USE_CAIF_CUDA
  if(target.TotalElements()!=source.TotalElements())
  {
    THROW_CAIFE("AddScaled: tensor size mismatch");
  }
  if(target.Dtype()!=source.Dtype())
  {
    THROW_CAIFE("AddScaled: target and source dtype must match");
  }

  const int64_t n=static_cast<int64_t>(target.TotalElements());
  if(n==0)
  {
    return;
  }

  // Dtype dispatch.
  // - fp32 uses cuBLAS SAXPY (single-kernel hardware-tuned axpy).
  // - fp16 / bf16 chain templated `launch_elementwise_mul_scalar<T>`
  //   into a stream-stack scratch buffer then templated
  //   `launch_elementwise_add<T>` to fold the scaled source into the
  //   target. Cost is two kernel launches + one scratch alloc per call;
  //   AddScaled has no internal callers today, so this is a correctness
  //   path rather than a perf-critical one.
  // No `default` throw: every dtype the layer system can construct
  // (fp32/fp16/bf16) is handled here. Storage in INT8/INT4 cannot reach
  // this op (FrozenLinear weights are not gradient targets).
  switch(target.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      // Use cuBLAS SAXPY: target = target + scale * source
      CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
      ctx.SetCublasStream(target.Stream().Handle());

      cublasStatus_t status=cublasSaxpy(ctx.CublasHandle(),
                                        n,
                                        &scale,
                                        source.DevicePtr<float>(),
                                        1,
                                        target.DevicePtr<float>(),
                                        1);
      if(status!=CUBLAS_STATUS_SUCCESS)
      {
        THROW_CAIFE("cuBLAS AddScaledDevice (saxpy) failed");
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float16:
    {
      CAIF_DeviceTensor scratch=CAIF_DeviceTensor::Uninitialized(target.Shape(),
                                                                  target.Stream(),
                                                                  CAIF_DataType::CAIF_DataType_e::Float16);
      launch_elementwise_mul_scalar<__half>(source.DevicePtr<__half>(),
                                            scale,
                                            scratch.DevicePtr<__half>(),
                                            n,
                                            target.Stream().Handle());
      launch_elementwise_add<__half>(target.DevicePtr<__half>(),
                                     scratch.DevicePtr<__half>(),
                                     target.DevicePtr<__half>(),
                                     n,
                                     target.Stream().Handle());
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
    {
      CAIF_DeviceTensor scratch=CAIF_DeviceTensor::Uninitialized(target.Shape(),
                                                                  target.Stream(),
                                                                  CAIF_DataType::CAIF_DataType_e::BFloat16);
      launch_elementwise_mul_scalar<__nv_bfloat16>(source.DevicePtr<__nv_bfloat16>(),
                                                   scale,
                                                   scratch.DevicePtr<__nv_bfloat16>(),
                                                   n,
                                                   target.Stream().Handle());
      launch_elementwise_add<__nv_bfloat16>(target.DevicePtr<__nv_bfloat16>(),
                                            scratch.DevicePtr<__nv_bfloat16>(),
                                            target.DevicePtr<__nv_bfloat16>(),
                                            n,
                                            target.Stream().Handle());
      break;
    }
    default:
      THROW_CAIFE("AddScaled: unsupported dtype");
  }
#else
  (void)target;
  (void)source;
  (void)scale;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Bias Operations
//------------------------------------------------------------------------------

void CAIF_Ops::BiasAddDevice(const CAIF_DeviceTensor &input,
                             const CAIF_DeviceTensor &bias,
                             CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("BiasAdd: input must be 2D [batch x units]");
  }
  if(bias.Shape().size()!=1)
  {
    THROW_CAIFE("BiasAdd: bias must be 1D [units]");
  }
  if(shape[1]!=bias.Shape()[0])
  {
    THROW_CAIFE("BiasAdd: bias size must match input units dimension");
  }
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("BiasAdd: input/output size mismatch");
  }

  if(input.Dtype()!=bias.Dtype() || input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("BiasAdd: tensor dtype mismatch");
  }

  const int batch_size=static_cast<int>(shape[0]);
  const int units=static_cast<int>(shape[1]);
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_bias_add_2d<float>(input.DevicePtr<float>(),
                                bias.DevicePtr<float>(),
                                output.DevicePtr<float>(),
                                batch_size,
                                units,
                                stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_bias_add_2d<__half>(input.DevicePtr<__half>(),
                                 bias.DevicePtr<__half>(),
                                 output.DevicePtr<__half>(),
                                 batch_size,
                                 units,
                                 stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_bias_add_2d<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                        bias.DevicePtr<__nv_bfloat16>(),
                                        output.DevicePtr<__nv_bfloat16>(),
                                        batch_size,
                                        units,
                                        stream);
      break;
    default:
      THROW_CAIFE("BiasAdd: unsupported dtype");
  }
#else
  (void)input;
  (void)bias;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::BiasGradientDevice(const CAIF_DeviceTensor &grad,CAIF_DeviceTensor &bias_grad)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=grad.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("BiasGradient: grad must be 2D [batch x units]");
  }
  if(bias_grad.Shape().size()!=1)
  {
    THROW_CAIFE("BiasGradient: bias_grad must be 1D [units]");
  }
  if(shape[1]!=bias_grad.Shape()[0])
  {
    THROW_CAIFE("BiasGradient: bias_grad size must match grad units dimension");
  }

  if(grad.Dtype()!=bias_grad.Dtype())
  {
    THROW_CAIFE("BiasGradient: tensor dtype mismatch");
  }

  const int batch_size=static_cast<int>(shape[0]);
  const int units=static_cast<int>(shape[1]);
  const cudaStream_t stream=bias_grad.Stream().Handle();
  switch(grad.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_bias_grad_2d<float>(grad.DevicePtr<float>(),
                                 bias_grad.DevicePtr<float>(),
                                 batch_size,
                                 units,
                                 stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_bias_grad_2d<__half>(grad.DevicePtr<__half>(),
                                  bias_grad.DevicePtr<__half>(),
                                  batch_size,
                                  units,
                                  stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_bias_grad_2d<__nv_bfloat16>(grad.DevicePtr<__nv_bfloat16>(),
                                         bias_grad.DevicePtr<__nv_bfloat16>(),
                                         batch_size,
                                         units,
                                         stream);
      break;
    default:
      THROW_CAIFE("BiasGradient: unsupported dtype");
  }
#else
  (void)grad;
  (void)bias_grad;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::AddPositionalEncodingDevice(const CAIF_DeviceTensor &input,
                           const CAIF_DeviceTensor &pe_table,
                           CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2 && shape.size()!=3)
  {
    THROW_CAIFE("AddPositionalEncoding: input must be 2D or 3D");
  }
  if(pe_table.Shape().size()!=2)
  {
    THROW_CAIFE("AddPositionalEncoding: pe_table must be 2D [seq_len, dim]");
  }
  if(output.Shape()!=shape)
  {
    THROW_CAIFE("AddPositionalEncoding: input/output shape mismatch");
  }
  if(input.Dtype()!=pe_table.Dtype() || input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("AddPositionalEncoding: tensor dtype mismatch");
  }

  int batch;
  int seq_len;
  if(shape.size()==3)
  {
    batch=static_cast<int>(shape[0]);
    seq_len=static_cast<int>(shape[1]);
  }
  else
  {
    batch=1;
    seq_len=static_cast<int>(shape[0]);
  }
  const int dim=static_cast<int>(shape.back());

  if(static_cast<int>(pe_table.Shape()[0])<seq_len ||
     static_cast<int>(pe_table.Shape()[1])!=dim)
  {
    THROW_CAIFE("AddPositionalEncoding: pe_table shape incompatible with input");
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_add_positional_encoding<float>(input.DevicePtr<float>(),
                                            pe_table.DevicePtr<float>(),
                                            output.DevicePtr<float>(),
                                            batch,
                                            seq_len,
                                            dim,
                                            stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_add_positional_encoding<__half>(input.DevicePtr<__half>(),
                                             pe_table.DevicePtr<__half>(),
                                             output.DevicePtr<__half>(),
                                             batch,
                                             seq_len,
                                             dim,
                                             stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_add_positional_encoding<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                                    pe_table.DevicePtr<__nv_bfloat16>(),
                                                    output.DevicePtr<__nv_bfloat16>(),
                                                    batch,
                                                    seq_len,
                                                    dim,
                                                    stream);
      break;
    default:
      THROW_CAIFE("AddPositionalEncoding: unsupported dtype");
  }
#else
  (void)input;
  (void)pe_table;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::PositionalEncodingBackwardDevice(const CAIF_DeviceTensor &grad_output,
                                CAIF_DeviceTensor &grad_table)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=grad_output.Shape();
  if(shape.size()!=3)
  {
    THROW_CAIFE("PositionalEncodingBackward: grad_output must be 3D [batch, seq_len, dim]");
  }
  if(grad_table.Shape().size()!=2)
  {
    THROW_CAIFE("PositionalEncodingBackward: grad_table must be 2D [seq_len, dim]");
  }
  if(grad_output.Dtype()!=grad_table.Dtype())
  {
    THROW_CAIFE("PositionalEncodingBackward: tensor dtype mismatch");
  }

  const int batch=static_cast<int>(shape[0]);
  const int seq_len=static_cast<int>(shape[1]);
  const int dim=static_cast<int>(shape[2]);

  if(static_cast<int>(grad_table.Shape()[0])<seq_len ||
     static_cast<int>(grad_table.Shape()[1])!=dim)
  {
    THROW_CAIFE("PositionalEncodingBackward: grad_table shape incompatible with grad_output");
  }

  const cudaStream_t stream=grad_table.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_pe_table_backward<float>(grad_output.DevicePtr<float>(),
                                      grad_table.DevicePtr<float>(),
                                      batch,
                                      seq_len,
                                      dim,
                                      stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_pe_table_backward<__half>(grad_output.DevicePtr<__half>(),
                                       grad_table.DevicePtr<__half>(),
                                       batch,
                                       seq_len,
                                       dim,
                                       stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_pe_table_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                              grad_table.DevicePtr<__nv_bfloat16>(),
                                              batch,
                                              seq_len,
                                              dim,
                                              stream);
      break;
    default:
      THROW_CAIFE("PositionalEncodingBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)grad_table;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::ComputeRelativePositionBiasDevice(const CAIF_DeviceTensor &embedding,
                                 CAIF_DeviceTensor &output,
                                 uint32_t max_distance,
                                 bool bidirectional)
{
#ifdef USE_CAIF_CUDA
  const auto &emb_shape=embedding.Shape();
  const auto &out_shape=output.Shape();
  if(emb_shape.size()!=2)
  {
    THROW_CAIFE("ComputeRelativePositionBias: embedding must be 2D [num_heads, num_buckets]");
  }
  if(out_shape.size()!=3)
  {
    THROW_CAIFE("ComputeRelativePositionBias: output must be 3D [num_heads, q_len, k_len]");
  }
  if(embedding.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("ComputeRelativePositionBias: embedding must be Float32");
  }
  if(emb_shape[0]!=out_shape[0])
  {
    THROW_CAIFE("ComputeRelativePositionBias: embedding/output num_heads mismatch");
  }

  const int num_heads=static_cast<int>(out_shape[0]);
  const int q_len=static_cast<int>(out_shape[1]);
  const int k_len=static_cast<int>(out_shape[2]);
  const int num_buckets=static_cast<int>(emb_shape[1]);
  int bidir_flag=0;
  if(bidirectional==true)
  {
    bidir_flag=1;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_relative_position_bias_forward<float>(embedding.DevicePtr<float>(),
                                                   output.DevicePtr<float>(),
                                                   num_heads,
                                                   q_len,
                                                   k_len,
                                                   num_buckets,
                                                   static_cast<int>(max_distance),
                                                   bidir_flag,
                                                   stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_relative_position_bias_forward<__half>(embedding.DevicePtr<float>(),
                                                    output.DevicePtr<__half>(),
                                                    num_heads,
                                                    q_len,
                                                    k_len,
                                                    num_buckets,
                                                    static_cast<int>(max_distance),
                                                    bidir_flag,
                                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_relative_position_bias_forward<__nv_bfloat16>(embedding.DevicePtr<float>(),
                                                           output.DevicePtr<__nv_bfloat16>(),
                                                           num_heads,
                                                           q_len,
                                                           k_len,
                                                           num_buckets,
                                                           static_cast<int>(max_distance),
                                                           bidir_flag,
                                                           stream);
      break;
    default:
      THROW_CAIFE("ComputeRelativePositionBias: unsupported output dtype");
  }
#else
  (void)embedding;
  (void)output;
  (void)max_distance;
  (void)bidirectional;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::AccumulateRelativePositionBiasGradientDevice(const CAIF_DeviceTensor &grad_output,
                                            CAIF_DeviceTensor &grad_embedding,
                                            uint32_t max_distance,
                                            bool bidirectional)
{
#ifdef USE_CAIF_CUDA
  const auto &go_shape=grad_output.Shape();
  const auto &ge_shape=grad_embedding.Shape();
  if(go_shape.size()!=3)
  {
    THROW_CAIFE("AccumulateRelativePositionBiasGradient: grad_output must be 3D [num_heads, q_len, k_len]");
  }
  if(ge_shape.size()!=2)
  {
    THROW_CAIFE("AccumulateRelativePositionBiasGradient: grad_embedding must be 2D [num_heads, num_buckets]");
  }
  if(grad_embedding.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("AccumulateRelativePositionBiasGradient: grad_embedding must be Float32");
  }
  if(go_shape[0]!=ge_shape[0])
  {
    THROW_CAIFE("AccumulateRelativePositionBiasGradient: grad_output/grad_embedding num_heads mismatch");
  }

  const int num_heads=static_cast<int>(go_shape[0]);
  const int q_len=static_cast<int>(go_shape[1]);
  const int k_len=static_cast<int>(go_shape[2]);
  const int num_buckets=static_cast<int>(ge_shape[1]);
  int bidir_flag=0;
  if(bidirectional==true)
  {
    bidir_flag=1;
  }

  const cudaStream_t stream=grad_embedding.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_relative_position_bias_backward<float>(grad_output.DevicePtr<float>(),
                                                    grad_embedding.DevicePtr<float>(),
                                                    num_heads,
                                                    q_len,
                                                    k_len,
                                                    num_buckets,
                                                    static_cast<int>(max_distance),
                                                    bidir_flag,
                                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_relative_position_bias_backward<__half>(grad_output.DevicePtr<__half>(),
                                                     grad_embedding.DevicePtr<float>(),
                                                     num_heads,
                                                     q_len,
                                                     k_len,
                                                     num_buckets,
                                                     static_cast<int>(max_distance),
                                                     bidir_flag,
                                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_relative_position_bias_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                                            grad_embedding.DevicePtr<float>(),
                                                            num_heads,
                                                            q_len,
                                                            k_len,
                                                            num_buckets,
                                                            static_cast<int>(max_distance),
                                                            bidir_flag,
                                                            stream);
      break;
    default:
      THROW_CAIFE("AccumulateRelativePositionBiasGradient: unsupported grad_output dtype");
  }
#else
  (void)grad_output;
  (void)grad_embedding;
  (void)max_distance;
  (void)bidirectional;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}


void CAIF_Ops::SliceValidateAndDims(const CAIF_DeviceTensor &input,
                                    const CAIF_DeviceTensor &output,
                                    uint32_t col_start,
                                    const std::string &op_name,
                                    int &rows,
                                    int &in_cols,
                                    int &out_cols)
{
  const auto &in_shape=input.Shape();
  const auto &out_shape=output.Shape();
  if(in_shape.size()<2)
  {
    THROW_CAIFE(std::string(op_name)+": input must be >=2D");
  }
  if(out_shape.size()!=in_shape.size())
  {
    THROW_CAIFE(std::string(op_name)+": input/output rank mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE(std::string(op_name)+": input/output dtype mismatch");
  }
  in_cols=static_cast<int>(in_shape.back());
  out_cols=static_cast<int>(out_shape.back());
  size_t r=1;
  for(size_t i=0;i<in_shape.size()-1;++i)
  {
    if(in_shape[i]!=out_shape[i])
    {
      THROW_CAIFE(std::string(op_name)+": leading dim mismatch");
    }
    r*=in_shape[i];
  }
  rows=static_cast<int>(r);
  if(static_cast<int>(col_start)+out_cols>in_cols)
  {
    THROW_CAIFE(std::string(op_name)+": slice range exceeds input width");
  }
}


void CAIF_Ops::CastDevice(const CAIF_DeviceTensor &input,
                CAIF_DeviceTensor &output,
                CAIF_RunContext &ctx)
{
#ifdef USE_CAIF_CUDA
  if(input.Shape()!=output.Shape())
  {
    THROW_CAIFE("Cast: input/output shape mismatch");
  }
  const CAIF_DataType::CAIF_DataType_e src=input.Dtype();
  const CAIF_DataType::CAIF_DataType_e dst=output.Dtype();

  if(src==CAIF_DataType::CAIF_DataType_e::Int8||
     src==CAIF_DataType::CAIF_DataType_e::Int4||
     dst==CAIF_DataType::CAIF_DataType_e::Int8||
     dst==CAIF_DataType::CAIF_DataType_e::Int4)
  {
    THROW_CAIFE("Cast: integer dtypes require Quantize/Dequantize, not Cast");
  }

  (void)ctx;
  cudaStream_t raw_stream=output.Stream().Handle();
  const int64_t n=static_cast<int64_t>(input.TotalElements());

  if(src==dst)
  {
    const size_t bytes=input.DtypeInfo().StorageSizeBytes(input.TotalElements());
    cudaError_t s=cudaMemcpyAsync(output.DeviceDataRaw(),
                                  input.DeviceDataRaw(),
                                  bytes,
                                  cudaMemcpyDeviceToDevice,
                                  raw_stream);
    if(s!=cudaSuccess)
    {
      THROW_CAIFE("Cast: identity memcpy failed");
    }
    return;
  }

  if(src==CAIF_DataType::CAIF_DataType_e::Float32&&
     dst==CAIF_DataType::CAIF_DataType_e::Float16)
  {
    launch_convert_fp32_to_fp16(input.DevicePtr<float>(),
                                output.DeviceDataRaw(),
                                n,
                                raw_stream);
    return;
  }
  if(src==CAIF_DataType::CAIF_DataType_e::Float16&&
     dst==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    launch_convert_fp16_to_fp32(input.DeviceDataRaw(),
                                output.DevicePtr<float>(),
                                n,
                                raw_stream);
    return;
  }
  if(src==CAIF_DataType::CAIF_DataType_e::Float32&&
     dst==CAIF_DataType::CAIF_DataType_e::BFloat16)
  {
    launch_convert_fp32_to_bf16(input.DevicePtr<float>(),
                                output.DeviceDataRaw(),
                                n,
                                raw_stream);
    return;
  }
  if(src==CAIF_DataType::CAIF_DataType_e::BFloat16&&
     dst==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    launch_convert_bf16_to_fp32(input.DeviceDataRaw(),
                                output.DevicePtr<float>(),
                                n,
                                raw_stream);
    return;
  }
  if(src==CAIF_DataType::CAIF_DataType_e::Float16&&
     dst==CAIF_DataType::CAIF_DataType_e::BFloat16)
  {
    CAIF_DeviceTensor tmp=CAIF_DeviceTensor::Uninitialized(input.Shape(),
                                                           output.Stream(),
                                                           CAIF_DataType::CAIF_DataType_e::Float32);
    launch_convert_fp16_to_fp32(input.DeviceDataRaw(),
                                tmp.DevicePtr<float>(),
                                n,
                                raw_stream);
    launch_convert_fp32_to_bf16(tmp.DevicePtr<float>(),
                                output.DeviceDataRaw(),
                                n,
                                raw_stream);
    return;
  }
  if(src==CAIF_DataType::CAIF_DataType_e::BFloat16&&
     dst==CAIF_DataType::CAIF_DataType_e::Float16)
  {
    CAIF_DeviceTensor tmp=CAIF_DeviceTensor::Uninitialized(input.Shape(),
                                                           output.Stream(),
                                                           CAIF_DataType::CAIF_DataType_e::Float32);
    launch_convert_bf16_to_fp32(input.DeviceDataRaw(),
                                tmp.DevicePtr<float>(),
                                n,
                                raw_stream);
    launch_convert_fp32_to_fp16(tmp.DevicePtr<float>(),
                                output.DeviceDataRaw(),
                                n,
                                raw_stream);
    return;
  }

  THROW_CAIFE("Cast: unsupported dtype pair");
#else
  (void)input;
  (void)output;
  (void)ctx;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::QuantizeInt8Device(const CAIF_DeviceTensor &input,
                        CAIF_DeviceTensor &output,
                        CAIF_DeviceTensor &scales,
                        CAIF_Ops::QuantScheme_e scheme,
                        CAIF_RunContext &ctx)
{
#ifdef USE_CAIF_CUDA
  (void)ctx;
  if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("QuantizeInt8: input must be fp32");
  }
  if(output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Int8)
  {
    THROW_CAIFE("QuantizeInt8: output must be Int8");
  }
  if(scales.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("QuantizeInt8: scales must be fp32");
  }
  if(input.Shape()!=output.Shape())
  {
    THROW_CAIFE("QuantizeInt8: input/output shape mismatch");
  }
  cudaStream_t stream=output.Stream().Handle();
  const int64_t n=static_cast<int64_t>(input.TotalElements());

  if(scheme==CAIF_Ops::QuantScheme_e::PerTensor_e)
  {
    if(scales.TotalElements()!=1)
    {
      THROW_CAIFE("QuantizeInt8 PerTensor: scales must have 1 element");
    }
    launch_quantize_int8_per_tensor(input.DevicePtr<float>(),
                                    output.DeviceDataRaw(),
                                    scales.DeviceDataRaw(),
                                    n,
                                    stream);
    return;
  }

  if(input.Shape().size()!=2)
  {
    THROW_CAIFE("QuantizeInt8 PerChannel: input must be 2D");
  }
  const int rows=static_cast<int>(input.Shape()[0]);
  const int cols=static_cast<int>(input.Shape()[1]);
  if(scales.TotalElements()!=static_cast<size_t>(cols))
  {
    THROW_CAIFE("QuantizeInt8 PerChannel: scales length must equal cols");
  }
  launch_quantize_int8_per_channel(input.DevicePtr<float>(),
                                   output.DeviceDataRaw(),
                                   scales.DeviceDataRaw(),
                                   rows,
                                   cols,
                                   stream);
#else
  (void)input;
  (void)output;
  (void)scales;
  (void)scheme;
  (void)ctx;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::DequantizeInt8Device(const CAIF_DeviceTensor &input,
                          CAIF_DeviceTensor &output,
                          const CAIF_DeviceTensor &scales,
                          CAIF_Ops::QuantScheme_e scheme,
                          CAIF_RunContext &ctx)
{
#ifdef USE_CAIF_CUDA
  (void)ctx;
  if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Int8)
  {
    THROW_CAIFE("DequantizeInt8: input must be Int8");
  }
  if(output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("DequantizeInt8: output must be fp32");
  }
  if(scales.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("DequantizeInt8: scales must be fp32");
  }
  if(input.Shape()!=output.Shape())
  {
    THROW_CAIFE("DequantizeInt8: input/output shape mismatch");
  }
  cudaStream_t stream=output.Stream().Handle();
  const int64_t n=static_cast<int64_t>(input.TotalElements());

  if(scheme==CAIF_Ops::QuantScheme_e::PerTensor_e)
  {
    if(scales.TotalElements()!=1)
    {
      THROW_CAIFE("DequantizeInt8 PerTensor: scales must have 1 element");
    }
    launch_dequantize_int8_per_tensor(input.DeviceDataRaw(),
                                      output.DevicePtr<float>(),
                                      scales.DeviceDataRaw(),
                                      n,
                                      stream);
    return;
  }

  if(input.Shape().size()!=2)
  {
    THROW_CAIFE("DequantizeInt8 PerChannel: input must be 2D");
  }
  const int rows=static_cast<int>(input.Shape()[0]);
  const int cols=static_cast<int>(input.Shape()[1]);
  if(scales.TotalElements()!=static_cast<size_t>(cols))
  {
    THROW_CAIFE("DequantizeInt8 PerChannel: scales length must equal cols");
  }
  launch_dequantize_int8_per_channel(input.DeviceDataRaw(),
                                     output.DevicePtr<float>(),
                                     scales.DeviceDataRaw(),
                                     rows,
                                     cols,
                                     stream);
#else
  (void)input;
  (void)output;
  (void)scales;
  (void)scheme;
  (void)ctx;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::QuantizeInt4PerGroupDevice(const CAIF_DeviceTensor &input,
                                CAIF_DeviceTensor &output,
                                CAIF_DeviceTensor &scales,
                                uint32_t group_size,
                                CAIF_RunContext &ctx)
{
#ifdef USE_CAIF_CUDA
  (void)ctx;
  if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("QuantizeInt4PerGroup: input must be fp32");
  }
  if(output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Int4)
  {
    THROW_CAIFE("QuantizeInt4PerGroup: output must be Int4");
  }
  if(scales.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float16)
  {
    THROW_CAIFE("QuantizeInt4PerGroup: scales must be fp16");
  }
  if(group_size==0)
  {
    THROW_CAIFE("QuantizeInt4PerGroup: group_size must be > 0");
  }
  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const size_t num_groups=(static_cast<size_t>(n)+group_size-1)/group_size;
  if(scales.TotalElements()!=num_groups)
  {
    THROW_CAIFE("QuantizeInt4PerGroup: scales length must equal num_groups");
  }
  cudaStream_t stream=output.Stream().Handle();
  launch_quantize_to_int4(input.DevicePtr<float>(),
                          output.DeviceDataRaw(),
                          scales.DeviceDataRaw(),
                          n,
                          static_cast<int>(group_size),
                          stream);
#else
  (void)input;
  (void)output;
  (void)scales;
  (void)group_size;
  (void)ctx;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::DequantizeInt4PerGroupDevice(const CAIF_DeviceTensor &input,
                                  CAIF_DeviceTensor &output,
                                  const CAIF_DeviceTensor &scales,
                                  uint32_t group_size,
                                  CAIF_RunContext &ctx)
{
#ifdef USE_CAIF_CUDA
  (void)ctx;
  if(input.Dtype()!=CAIF_DataType::CAIF_DataType_e::Int4)
  {
    THROW_CAIFE("DequantizeInt4PerGroup: input must be Int4");
  }
  if(output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32)
  {
    THROW_CAIFE("DequantizeInt4PerGroup: output must be fp32");
  }
  if(scales.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float16)
  {
    THROW_CAIFE("DequantizeInt4PerGroup: scales must be fp16");
  }
  if(group_size==0)
  {
    THROW_CAIFE("DequantizeInt4PerGroup: group_size must be > 0");
  }
  const int64_t n=static_cast<int64_t>(output.TotalElements());
  const size_t num_groups=(static_cast<size_t>(n)+group_size-1)/group_size;
  if(scales.TotalElements()!=num_groups)
  {
    THROW_CAIFE("DequantizeInt4PerGroup: scales length must equal num_groups");
  }
  cudaStream_t stream=output.Stream().Handle();
  launch_dequantize_int4(input.DeviceDataRaw(),
                         scales.DeviceDataRaw(),
                         output.DevicePtr<float>(),
                         n,
                         static_cast<int>(group_size),
                         stream);
#else
  (void)input;
  (void)output;
  (void)scales;
  (void)group_size;
  (void)ctx;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SliceLastDimDevice(const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &output,
                  uint32_t col_start)
{
#ifdef USE_CAIF_CUDA
  int rows=0;
  int in_cols=0;
  int out_cols=0;
  SliceValidateAndDims(input,output,col_start,"SliceLastDim",rows,in_cols,out_cols);
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_slice_last_dim<float>(input.DevicePtr<float>(),
                                   output.DevicePtr<float>(),
                                   rows,
                                   in_cols,
                                   static_cast<int>(col_start),
                                   out_cols,
                                   stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_slice_last_dim<__half>(input.DevicePtr<__half>(),
                                    output.DevicePtr<__half>(),
                                    rows,
                                    in_cols,
                                    static_cast<int>(col_start),
                                    out_cols,
                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_slice_last_dim<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                           output.DevicePtr<__nv_bfloat16>(),
                                           rows,
                                           in_cols,
                                           static_cast<int>(col_start),
                                           out_cols,
                                           stream);
      break;
    default:
      THROW_CAIFE("SliceLastDim: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  (void)col_start;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SliceLastDimBackwardDevice(const CAIF_DeviceTensor &grad_output,
                          CAIF_DeviceTensor &grad_input,
                          uint32_t col_start)
{
#ifdef USE_CAIF_CUDA
  int rows=0;
  int in_cols=0;
  int out_cols=0;
  SliceValidateAndDims(grad_input,grad_output,col_start,
                       "SliceLastDimBackward",rows,in_cols,out_cols);
  const cudaStream_t stream=grad_input.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_slice_last_dim_backward<float>(grad_output.DevicePtr<float>(),
                                            grad_input.DevicePtr<float>(),
                                            rows,
                                            in_cols,
                                            static_cast<int>(col_start),
                                            out_cols,
                                            stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_slice_last_dim_backward<__half>(grad_output.DevicePtr<__half>(),
                                             grad_input.DevicePtr<__half>(),
                                             rows,
                                             in_cols,
                                             static_cast<int>(col_start),
                                             out_cols,
                                             stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_slice_last_dim_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                                    grad_input.DevicePtr<__nv_bfloat16>(),
                                                    rows,
                                                    in_cols,
                                                    static_cast<int>(col_start),
                                                    out_cols,
                                                    stream);
      break;
    default:
      THROW_CAIFE("SliceLastDimBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)grad_input;
  (void)col_start;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::ConcatLastDimDevice(const CAIF_DeviceTensor &a,
                   const CAIF_DeviceTensor &b,
                   CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &a_shape=a.Shape();
  const auto &b_shape=b.Shape();
  const auto &out_shape=output.Shape();
  if(a_shape.size()<2||b_shape.size()!=a_shape.size()||out_shape.size()!=a_shape.size())
  {
    THROW_CAIFE("ConcatLastDim: inputs/output must share rank >=2");
  }
  if(a.Dtype()!=b.Dtype()||a.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("ConcatLastDim: dtype mismatch");
  }
  size_t r=1;
  for(size_t i=0;i<a_shape.size()-1;++i)
  {
    if(a_shape[i]!=b_shape[i]||a_shape[i]!=out_shape[i])
    {
      THROW_CAIFE("ConcatLastDim: leading dim mismatch");
    }
    r*=a_shape[i];
  }
  const int rows=static_cast<int>(r);
  const int cols_a=static_cast<int>(a_shape.back());
  const int cols_b=static_cast<int>(b_shape.back());
  if(static_cast<int>(out_shape.back())!=cols_a+cols_b)
  {
    THROW_CAIFE("ConcatLastDim: output last dim must equal cols_a+cols_b");
  }
  const cudaStream_t stream=output.Stream().Handle();
  switch(a.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_concat_last_dim<float>(a.DevicePtr<float>(),
                                    b.DevicePtr<float>(),
                                    output.DevicePtr<float>(),
                                    rows,
                                    cols_a,
                                    cols_b,
                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_concat_last_dim<__half>(a.DevicePtr<__half>(),
                                     b.DevicePtr<__half>(),
                                     output.DevicePtr<__half>(),
                                     rows,
                                     cols_a,
                                     cols_b,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_concat_last_dim<__nv_bfloat16>(a.DevicePtr<__nv_bfloat16>(),
                                            b.DevicePtr<__nv_bfloat16>(),
                                            output.DevicePtr<__nv_bfloat16>(),
                                            rows,
                                            cols_a,
                                            cols_b,
                                            stream);
      break;
    default:
      THROW_CAIFE("ConcatLastDim: unsupported dtype");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Activation Functions (Forward) - vectorized CUDA kernels
//------------------------------------------------------------------------------

void CAIF_Ops::ReLUDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("ReLU: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("ReLU: input/output dtype mismatch");
  }
  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_relu_forward<float>(input.DevicePtr<float>(),
                                 output.DevicePtr<float>(),
                                 n,
                                 stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_relu_forward<__half>(input.DevicePtr<__half>(),
                                  output.DevicePtr<__half>(),
                                  n,
                                  stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_relu_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                         output.DevicePtr<__nv_bfloat16>(),
                                         n,
                                         stream);
      break;
    default:
      THROW_CAIFE("ReLU: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SigmoidDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Sigmoid: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Sigmoid: input/output dtype mismatch");
  }
  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_sigmoid_forward<float>(input.DevicePtr<float>(),
                                    output.DevicePtr<float>(),
                                    n,
                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_sigmoid_forward<__half>(input.DevicePtr<__half>(),
                                     output.DevicePtr<__half>(),
                                     n,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_sigmoid_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                            output.DevicePtr<__nv_bfloat16>(),
                                            n,
                                            stream);
      break;
    default:
      THROW_CAIFE("Sigmoid: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::TanhDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Tanh: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Tanh: input/output dtype mismatch");
  }
  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_tanh_forward<float>(input.DevicePtr<float>(),
                                 output.DevicePtr<float>(),
                                 n,
                                 stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_tanh_forward<__half>(input.DevicePtr<__half>(),
                                  output.DevicePtr<__half>(),
                                  n,
                                  stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_tanh_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                         output.DevicePtr<__nv_bfloat16>(),
                                         n,
                                         stream);
      break;
    default:
      THROW_CAIFE("Tanh: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SoftmaxDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("Softmax: input must be 2D [batch x classes]");
  }
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Softmax: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Softmax: input/output dtype mismatch");
  }

  const int num_rows=static_cast<int>(shape[0]);
  const int row_len=static_cast<int>(shape[1]);
  const cudaStream_t stream=output.Stream().Handle();

  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_attention_softmax<float>(input.DevicePtr<float>(),
                                      output.DevicePtr<float>(),
                                      num_rows,
                                      row_len,
                                      stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_attention_softmax<__half>(input.DevicePtr<__half>(),
                                       output.DevicePtr<__half>(),
                                       num_rows,
                                       row_len,
                                       stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_attention_softmax<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                              output.DevicePtr<__nv_bfloat16>(),
                                              num_rows,
                                              row_len,
                                              stream);
      break;
    default:
      THROW_CAIFE("Softmax: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Activation Functions (Backward)
//------------------------------------------------------------------------------

void CAIF_Ops::ReLUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=input.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("ReLUBackward: tensor size mismatch");
  }

  if(grad_output.Dtype()!=input.Dtype()||grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("ReLUBackward: dtype mismatch");
  }
  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=grad_input.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_relu_backward<float>(grad_output.DevicePtr<float>(),
                                  input.DevicePtr<float>(),
                                  grad_input.DevicePtr<float>(),
                                  n,
                                  stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_relu_backward<__half>(grad_output.DevicePtr<__half>(),
                                   input.DevicePtr<__half>(),
                                   grad_input.DevicePtr<__half>(),
                                   n,
                                   stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_relu_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                          input.DevicePtr<__nv_bfloat16>(),
                                          grad_input.DevicePtr<__nv_bfloat16>(),
                                          n,
                                          stream);
      break;
    default:
      THROW_CAIFE("ReLUBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)input;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SigmoidBackwardDevice(const CAIF_DeviceTensor &grad_output,
                     const CAIF_DeviceTensor &output,
                     CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=output.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("SigmoidBackward: tensor size mismatch");
  }
  if(grad_output.Dtype()!=output.Dtype()||grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("SigmoidBackward: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  const cudaStream_t stream=grad_input.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_sigmoid_backward<float>(grad_output.DevicePtr<float>(),
                                     output.DevicePtr<float>(),
                                     grad_input.DevicePtr<float>(),
                                     n,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_sigmoid_backward<__half>(grad_output.DevicePtr<__half>(),
                                      output.DevicePtr<__half>(),
                                      grad_input.DevicePtr<__half>(),
                                      n,
                                      stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_sigmoid_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                             output.DevicePtr<__nv_bfloat16>(),
                                             grad_input.DevicePtr<__nv_bfloat16>(),
                                             n,
                                             stream);
      break;
    default:
      THROW_CAIFE("SigmoidBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::TanhBackwardDevice(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &output,
                  CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=output.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("TanhBackward: tensor size mismatch");
  }
  if(grad_output.Dtype()!=output.Dtype()||grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("TanhBackward: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  const cudaStream_t stream=grad_input.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_tanh_backward<float>(grad_output.DevicePtr<float>(),
                                  output.DevicePtr<float>(),
                                  grad_input.DevicePtr<float>(),
                                  n,
                                  stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_tanh_backward<__half>(grad_output.DevicePtr<__half>(),
                                   output.DevicePtr<__half>(),
                                   grad_input.DevicePtr<__half>(),
                                   n,
                                   stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_tanh_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                          output.DevicePtr<__nv_bfloat16>(),
                                          grad_input.DevicePtr<__nv_bfloat16>(),
                                          n,
                                          stream);
      break;
    default:
      THROW_CAIFE("TanhBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SoftmaxBackwardDevice(const CAIF_DeviceTensor &grad_output,
                     const CAIF_DeviceTensor &output,
                     CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=output.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("SoftmaxBackward: tensors must be 2D [batch x classes]");
  }
  if(grad_output.TotalElements()!=output.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("SoftmaxBackward: tensor size mismatch");
  }
  if(grad_output.Dtype()!=output.Dtype()||grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("SoftmaxBackward: tensor dtype mismatch");
  }

  const int num_rows=static_cast<int>(shape[0]);
  const int row_len=static_cast<int>(shape[1]);
  const cudaStream_t stream=grad_input.Stream().Handle();

  switch(output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_attention_softmax_backward<float>(grad_output.DevicePtr<float>(),
                                               output.DevicePtr<float>(),
                                               grad_input.DevicePtr<float>(),
                                               num_rows,
                                               row_len,
                                               stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_attention_softmax_backward<__half>(grad_output.DevicePtr<__half>(),
                                                output.DevicePtr<__half>(),
                                                grad_input.DevicePtr<__half>(),
                                                num_rows,
                                                row_len,
                                                stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_attention_softmax_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                                       output.DevicePtr<__nv_bfloat16>(),
                                                       grad_input.DevicePtr<__nv_bfloat16>(),
                                                       num_rows,
                                                       row_len,
                                                       stream);
      break;
    default:
      THROW_CAIFE("SoftmaxBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::LeakyReLUDevice(const CAIF_DeviceTensor &input,
               CAIF_DeviceTensor &output,
               float alpha)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("LeakyReLU: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("LeakyReLU: input/output dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_leaky_relu_forward<float>(input.DevicePtr<float>(),
                                       output.DevicePtr<float>(),
                                       alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_leaky_relu_forward<__half>(input.DevicePtr<__half>(),
                                        output.DevicePtr<__half>(),
                                        alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_leaky_relu_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                               output.DevicePtr<__nv_bfloat16>(),
                                               alpha,n,stream);
      break;
    default:
      THROW_CAIFE("LeakyReLU: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::ELUDevice(const CAIF_DeviceTensor &input,
         CAIF_DeviceTensor &output,
         float alpha)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("ELU: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("ELU: input/output dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elu_forward<float>(input.DevicePtr<float>(),
                                output.DevicePtr<float>(),
                                alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elu_forward<__half>(input.DevicePtr<__half>(),
                                 output.DevicePtr<__half>(),
                                 alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elu_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                        output.DevicePtr<__nv_bfloat16>(),
                                        alpha,n,stream);
      break;
    default:
      THROW_CAIFE("ELU: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::GELUDevice(const CAIF_DeviceTensor &input,
                          CAIF_DeviceTensor &output,
                          const CAIF_GELUApproximation::CAIF_GELUApproximation_e approx)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("GELU: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("GELU: input/output dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  const bool exact=(approx==CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      if(exact==true)
      {
        launch_gelu_forward_erf<float>(input.DevicePtr<float>(),
                                       output.DevicePtr<float>(),
                                       n,stream);
      }
      else
      {
        launch_gelu_forward<float>(input.DevicePtr<float>(),
                                   output.DevicePtr<float>(),
                                   n,stream);
      }
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      if(exact==true)
      {
        launch_gelu_forward_erf<__half>(input.DevicePtr<__half>(),
                                        output.DevicePtr<__half>(),
                                        n,stream);
      }
      else
      {
        launch_gelu_forward<__half>(input.DevicePtr<__half>(),
                                    output.DevicePtr<__half>(),
                                    n,stream);
      }
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      if(exact==true)
      {
        launch_gelu_forward_erf<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                               output.DevicePtr<__nv_bfloat16>(),
                                               n,stream);
      }
      else
      {
        launch_gelu_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                           output.DevicePtr<__nv_bfloat16>(),
                                           n,stream);
      }
      break;
    default:
      THROW_CAIFE("GELU: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  (void)approx;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SwishDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Swish: input/output size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Swish: input/output dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_swish_forward<float>(input.DevicePtr<float>(),
                                  output.DevicePtr<float>(),
                                  n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_swish_forward<__half>(input.DevicePtr<__half>(),
                                   output.DevicePtr<__half>(),
                                   n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_swish_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                          output.DevicePtr<__nv_bfloat16>(),
                                          n,stream);
      break;
    default:
      THROW_CAIFE("Swish: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::LeakyReLUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                       const CAIF_DeviceTensor &input,
                       CAIF_DeviceTensor &grad_input,
                       float alpha)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=input.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("LeakyReLUBackward: tensor size mismatch");
  }
  if(grad_output.Dtype()!=input.Dtype()||grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("LeakyReLUBackward: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=grad_input.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_leaky_relu_backward<float>(grad_output.DevicePtr<float>(),
                                        input.DevicePtr<float>(),
                                        grad_input.DevicePtr<float>(),
                                        alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_leaky_relu_backward<__half>(grad_output.DevicePtr<__half>(),
                                         input.DevicePtr<__half>(),
                                         grad_input.DevicePtr<__half>(),
                                         alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_leaky_relu_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                                input.DevicePtr<__nv_bfloat16>(),
                                                grad_input.DevicePtr<__nv_bfloat16>(),
                                                alpha,n,stream);
      break;
    default:
      THROW_CAIFE("LeakyReLUBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)input;
  (void)grad_input;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::ELUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                 const CAIF_DeviceTensor &input,
                 const CAIF_DeviceTensor &output,
                 CAIF_DeviceTensor &grad_input,
                 float alpha)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=input.TotalElements()||
     grad_output.TotalElements()!=output.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("ELUBackward: tensor size mismatch");
  }
  if(grad_output.Dtype()!=input.Dtype()||grad_output.Dtype()!=output.Dtype()||
     grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("ELUBackward: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=grad_input.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elu_backward<float>(grad_output.DevicePtr<float>(),
                                 input.DevicePtr<float>(),
                                 output.DevicePtr<float>(),
                                 grad_input.DevicePtr<float>(),
                                 alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elu_backward<__half>(grad_output.DevicePtr<__half>(),
                                  input.DevicePtr<__half>(),
                                  output.DevicePtr<__half>(),
                                  grad_input.DevicePtr<__half>(),
                                  alpha,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elu_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                         input.DevicePtr<__nv_bfloat16>(),
                                         output.DevicePtr<__nv_bfloat16>(),
                                         grad_input.DevicePtr<__nv_bfloat16>(),
                                         alpha,n,stream);
      break;
    default:
      THROW_CAIFE("ELUBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)input;
  (void)output;
  (void)grad_input;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::GELUBackwardDevice(const CAIF_DeviceTensor &grad_output,
                                  const CAIF_DeviceTensor &input,
                                  CAIF_DeviceTensor &grad_input,
                                  const CAIF_GELUApproximation::CAIF_GELUApproximation_e approx)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=input.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("GELUBackward: tensor size mismatch");
  }
  if(grad_output.Dtype()!=input.Dtype()||grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("GELUBackward: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=grad_input.Stream().Handle();
  const bool exact=(approx==CAIF_GELUApproximation::CAIF_GELUApproximation_e::Exact);
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      if(exact==true)
      {
        launch_gelu_backward_erf<float>(grad_output.DevicePtr<float>(),
                                        input.DevicePtr<float>(),
                                        grad_input.DevicePtr<float>(),
                                        n,stream);
      }
      else
      {
        launch_gelu_backward<float>(grad_output.DevicePtr<float>(),
                                    input.DevicePtr<float>(),
                                    grad_input.DevicePtr<float>(),
                                    n,stream);
      }
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      if(exact==true)
      {
        launch_gelu_backward_erf<__half>(grad_output.DevicePtr<__half>(),
                                         input.DevicePtr<__half>(),
                                         grad_input.DevicePtr<__half>(),
                                         n,stream);
      }
      else
      {
        launch_gelu_backward<__half>(grad_output.DevicePtr<__half>(),
                                     input.DevicePtr<__half>(),
                                     grad_input.DevicePtr<__half>(),
                                     n,stream);
      }
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      if(exact==true)
      {
        launch_gelu_backward_erf<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                                input.DevicePtr<__nv_bfloat16>(),
                                                grad_input.DevicePtr<__nv_bfloat16>(),
                                                n,stream);
      }
      else
      {
        launch_gelu_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                            input.DevicePtr<__nv_bfloat16>(),
                                            grad_input.DevicePtr<__nv_bfloat16>(),
                                            n,stream);
      }
      break;
    default:
      THROW_CAIFE("GELUBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)input;
  (void)grad_input;
  (void)approx;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SwishBackwardDevice(const CAIF_DeviceTensor &grad_output,
                   const CAIF_DeviceTensor &input,
                   const CAIF_DeviceTensor &output,
                   CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=input.TotalElements()||
     grad_output.TotalElements()!=output.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("SwishBackward: tensor size mismatch");
  }
  if(grad_output.Dtype()!=input.Dtype()||grad_output.Dtype()!=output.Dtype()||
     grad_output.Dtype()!=grad_input.Dtype())
  {
    THROW_CAIFE("SwishBackward: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  const cudaStream_t stream=grad_input.Stream().Handle();
  switch(grad_output.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_swish_backward<float>(grad_output.DevicePtr<float>(),
                                   input.DevicePtr<float>(),
                                   output.DevicePtr<float>(),
                                   grad_input.DevicePtr<float>(),
                                   n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_swish_backward<__half>(grad_output.DevicePtr<__half>(),
                                    input.DevicePtr<__half>(),
                                    output.DevicePtr<__half>(),
                                    grad_input.DevicePtr<__half>(),
                                    n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_swish_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                           input.DevicePtr<__nv_bfloat16>(),
                                           output.DevicePtr<__nv_bfloat16>(),
                                           grad_input.DevicePtr<__nv_bfloat16>(),
                                           n,stream);
      break;
    default:
      THROW_CAIFE("SwishBackward: unsupported dtype");
  }
#else
  (void)grad_output;
  (void)input;
  (void)output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Reduction Operations
//------------------------------------------------------------------------------

float CAIF_Ops::ReduceSumDevice(const CAIF_DeviceTensor &tensor)
{
#ifdef USE_CAIF_CUDA
  const int64_t n=static_cast<int64_t>(tensor.TotalElements());
  if(n==0)
  {
    return 0.0f;
  }

  // Allocate device memory for the result
  float *d_result=nullptr;
  cudaError_t err=cudaMalloc(reinterpret_cast<void**>(&d_result),sizeof(float));
  if(err!=cudaSuccess)
  {
    THROW_CAIFE("Failed to allocate device memory for reduction result");
  }

  // Reduction always accumulates into fp32 to avoid fp16/bf16 overflow.
  const cudaStream_t stream=tensor.Stream().Handle();
  switch(tensor.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_reduction_sum<float>(tensor.DevicePtr<float>(),d_result,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_reduction_sum<__half>(tensor.DevicePtr<__half>(),d_result,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_reduction_sum<__nv_bfloat16>(tensor.DevicePtr<__nv_bfloat16>(),d_result,n,stream);
      break;
    default:
      cudaFree(d_result);
      THROW_CAIFE("ReduceSum: unsupported dtype");
  }

  // Sync stream and copy result to host (this is the sync point)
  tensor.Stream().Synchronize();

  float result=0.0f;
  err=cudaMemcpy(&result,d_result,sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(d_result);

  if(err!=cudaSuccess)
  {
    THROW_CAIFE("Failed to copy reduction result to host");
  }

  return result;
#else
  (void)tensor;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

float CAIF_Ops::ReduceMeanDevice(const CAIF_DeviceTensor &tensor)
{
#ifdef USE_CAIF_CUDA
  const size_t n=tensor.TotalElements();
  if(n==0)
  {
    return 0.0f;
  }

  const float sum=ReduceSumDevice(tensor);
  return sum/static_cast<float>(n);
#else
  (void)tensor;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Loss Functions
//------------------------------------------------------------------------------

void CAIF_Ops::MSELossDevice(const CAIF_DeviceTensor &pred,
                             const CAIF_DeviceTensor &target,
                             CAIF_DeviceTensor &loss)
{
#ifdef USE_CAIF_CUDA
  if(pred.TotalElements()!=target.TotalElements())
  {
    THROW_CAIFE("MSELoss: prediction and target size mismatch");
  }
  if(loss.TotalElements()!=1)
  {
    THROW_CAIFE("MSELoss: loss tensor must have exactly 1 element");
  }
  if(pred.Dtype()!=target.Dtype())
  {
    THROW_CAIFE("MSELoss: pred and target dtype must match");
  }

  const int64_t n=static_cast<int64_t>(pred.TotalElements());
  // Loss scalar is always fp32; pred/target carry their declared storage.
  float *loss_ptr=loss.DevicePtr<float>();
  const cudaStream_t stream=loss.Stream().Handle();
  switch(pred.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_mse_loss<float>(pred.DevicePtr<float>(),
                             target.DevicePtr<float>(),
                             loss_ptr,
                             n,
                             stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_mse_loss<__half>(pred.DevicePtr<__half>(),
                              target.DevicePtr<__half>(),
                              loss_ptr,
                              n,
                              stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_mse_loss<__nv_bfloat16>(pred.DevicePtr<__nv_bfloat16>(),
                                     target.DevicePtr<__nv_bfloat16>(),
                                     loss_ptr,
                                     n,
                                     stream);
      break;
    default:
      THROW_CAIFE("MSELoss: unsupported dtype");
  }
#else
  (void)pred;
  (void)target;
  (void)loss;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MSELossBackwardDevice(const CAIF_DeviceTensor &pred,
                     const CAIF_DeviceTensor &target,
                     CAIF_DeviceTensor &grad)
{
#ifdef USE_CAIF_CUDA
  if(pred.TotalElements()!=target.TotalElements()||
     pred.TotalElements()!=grad.TotalElements())
  {
    THROW_CAIFE("MSELossBackward: tensor size mismatch");
  }
  if(pred.Dtype()!=target.Dtype()||pred.Dtype()!=grad.Dtype())
  {
    THROW_CAIFE("MSELossBackward: pred/target/grad dtype must match");
  }

  const int64_t n=static_cast<int64_t>(pred.TotalElements());
  const cudaStream_t stream=grad.Stream().Handle();
  switch(pred.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_mse_gradient<float>(pred.DevicePtr<float>(),
                                 target.DevicePtr<float>(),
                                 grad.DevicePtr<float>(),
                                 n,
                                 stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_mse_gradient<__half>(pred.DevicePtr<__half>(),
                                  target.DevicePtr<__half>(),
                                  grad.DevicePtr<__half>(),
                                  n,
                                  stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_mse_gradient<__nv_bfloat16>(pred.DevicePtr<__nv_bfloat16>(),
                                         target.DevicePtr<__nv_bfloat16>(),
                                         grad.DevicePtr<__nv_bfloat16>(),
                                         n,
                                         stream);
      break;
    default:
      THROW_CAIFE("MSELossBackward: unsupported dtype");
  }
#else
  (void)pred;
  (void)target;
  (void)grad;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Optimizer Operations
//------------------------------------------------------------------------------

void CAIF_Ops::AdamUpdateDevice(CAIF_DeviceTensor &param,
                const CAIF_DeviceTensor &grad,
                CAIF_DeviceTensor &m,
                CAIF_DeviceTensor &v,
                float lr,
                float beta1,
                float beta2,
                float epsilon,
                float weight_decay,
                int t)
{
#ifdef USE_CAIF_CUDA
  if(param.TotalElements()!=grad.TotalElements()||
     param.TotalElements()!=m.TotalElements()||
     param.TotalElements()!=v.TotalElements())
  {
    THROW_CAIFE("AdamUpdate: tensor size mismatch");
  }

  const int64_t n=static_cast<int64_t>(param.TotalElements());
  if(n==0)
  {
    return;
  }

  const float bias_correction1=1.0f-std::pow(beta1,static_cast<float>(t));
  const float bias_correction2=1.0f-std::pow(beta2,static_cast<float>(t));

  // m/v are fp32 master state regardless of param dtype.
  const CAIF_DataType::CAIF_DataType_e pdt=param.Dtype();
  if(pdt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    launch_fused_adam<float>(param.DevicePtr<float>(),
                             grad.DevicePtr<float>(),
                             m.DevicePtr<float>(),
                             v.DevicePtr<float>(),
                             lr,beta1,beta2,epsilon,weight_decay,
                             bias_correction1,bias_correction2,
                             n,param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::Float16)
  {
    launch_fused_adam<__half>(param.DevicePtr<__half>(),
                              grad.DevicePtr<__half>(),
                              m.DevicePtr<float>(),
                              v.DevicePtr<float>(),
                              lr,beta1,beta2,epsilon,weight_decay,
                              bias_correction1,bias_correction2,
                              n,param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::BFloat16)
  {
    launch_fused_adam<__nv_bfloat16>(param.DevicePtr<__nv_bfloat16>(),
                                     grad.DevicePtr<__nv_bfloat16>(),
                                     m.DevicePtr<float>(),
                                     v.DevicePtr<float>(),
                                     lr,beta1,beta2,epsilon,weight_decay,
                                     bias_correction1,bias_correction2,
                                     n,param.Stream().Handle());
  }
  else
  {
    THROW_CAIFE("AdamUpdate: unsupported param dtype");
  }
#else
  (void)param;
  (void)grad;
  (void)m;
  (void)v;
  (void)lr;
  (void)beta1;
  (void)beta2;
  (void)epsilon;
  (void)weight_decay;
  (void)t;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Additional Element-wise Operations
//------------------------------------------------------------------------------

void CAIF_Ops::MultiplyDevice(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(a.TotalElements()!=b.TotalElements()||a.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Multiply: tensor size mismatch");
  }
  if(a.Dtype()!=b.Dtype() || a.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Multiply: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  const cudaStream_t stream=output.Stream().Handle();
  switch(a.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_mul<float>(a.DevicePtr<float>(),
                                    b.DevicePtr<float>(),
                                    output.DevicePtr<float>(),
                                    n,
                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_mul<__half>(a.DevicePtr<__half>(),
                                     b.DevicePtr<__half>(),
                                     output.DevicePtr<__half>(),
                                     n,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_mul<__nv_bfloat16>(a.DevicePtr<__nv_bfloat16>(),
                                            b.DevicePtr<__nv_bfloat16>(),
                                            output.DevicePtr<__nv_bfloat16>(),
                                            n,
                                            stream);
      break;
    default:
      THROW_CAIFE("Multiply: unsupported dtype");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::ScaleDevice(const CAIF_DeviceTensor &input,float scale,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Scale: tensor size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Scale: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_mul_scalar<float>(input.DevicePtr<float>(),
                                           scale,
                                           output.DevicePtr<float>(),
                                           n,
                                           stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_mul_scalar<__half>(input.DevicePtr<__half>(),
                                            scale,
                                            output.DevicePtr<__half>(),
                                            n,
                                            stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_mul_scalar<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                                   scale,
                                                   output.DevicePtr<__nv_bfloat16>(),
                                                   n,
                                                   stream);
      break;
    default:
      THROW_CAIFE("Scale: unsupported dtype");
  }
#else
  (void)input;
  (void)scale;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SiLUDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  // SiLU is the same as Swish: x * sigmoid(x)
  SwishDevice(input,output);
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SiLUBackwardDevice(const CAIF_DeviceTensor &input,
                  const CAIF_DeviceTensor &grad_output,
                  CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=grad_output.TotalElements()||
     input.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("SiLUBackward: tensor size mismatch");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());

  // SiLU == Swish. The templated swish_backward needs the post-activation
  // output as its third argument; recompute it here via swish_forward into
  // a scratch tensor so the caller's contract (input + grad_output ->
  // grad_input) doesn't change.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=input.Dtype();
  cudaStream_t stream=grad_input.Stream().Handle();
  CAIF_DeviceTensor scratch_out=CAIF_DeviceTensor::Uninitialized(input.Shape(),
                                                                 grad_input.Stream(),
                                                                 dt);
  if(dt==Dtype_e::Float32)
  {
    launch_swish_forward<float>(input.DevicePtr<float>(),
                                scratch_out.DevicePtr<float>(),
                                n,stream);
    launch_swish_backward<float>(grad_output.DevicePtr<float>(),
                                 input.DevicePtr<float>(),
                                 scratch_out.DevicePtr<float>(),
                                 grad_input.DevicePtr<float>(),
                                 n,stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_swish_forward<__half>(input.DevicePtr<__half>(),
                                 scratch_out.DevicePtr<__half>(),
                                 n,stream);
    launch_swish_backward<__half>(grad_output.DevicePtr<__half>(),
                                  input.DevicePtr<__half>(),
                                  scratch_out.DevicePtr<__half>(),
                                  grad_input.DevicePtr<__half>(),
                                  n,stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_swish_forward<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                        scratch_out.DevicePtr<__nv_bfloat16>(),
                                        n,stream);
    launch_swish_backward<__nv_bfloat16>(grad_output.DevicePtr<__nv_bfloat16>(),
                                         input.DevicePtr<__nv_bfloat16>(),
                                         scratch_out.DevicePtr<__nv_bfloat16>(),
                                         grad_input.DevicePtr<__nv_bfloat16>(),
                                         n,stream);
  }
  else
  {
    THROW_CAIFE("SiLUBackward: unsupported input dtype");
  }
#else
  (void)input;
  (void)grad_output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::AddBiasDevice(const CAIF_DeviceTensor &input,
                             const CAIF_DeviceTensor &bias,
                             CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  BiasAddDevice(input,bias,output);
#else
  (void)input;
  (void)bias;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::AddScalarDevice(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("AddScalar: tensor size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("AddScalar: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_add_scalar<float>(input.DevicePtr<float>(),
                                           scalar,
                                           output.DevicePtr<float>(),
                                           n,
                                           stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_add_scalar<__half>(input.DevicePtr<__half>(),
                                            scalar,
                                            output.DevicePtr<__half>(),
                                            n,
                                            stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_add_scalar<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                                   scalar,
                                                   output.DevicePtr<__nv_bfloat16>(),
                                                   n,
                                                   stream);
      break;
    default:
      THROW_CAIFE("AddScalar: unsupported dtype");
  }
#else
  (void)input;
  (void)scalar;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SubtractDevice(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(a.TotalElements()!=b.TotalElements()||a.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Subtract: tensor size mismatch");
  }
  if(a.Dtype()!=b.Dtype() || a.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Subtract: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(a.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_sub<float>(a.DevicePtr<float>(),
                                    b.DevicePtr<float>(),
                                    output.DevicePtr<float>(),
                                    n,
                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_sub<__half>(a.DevicePtr<__half>(),
                                     b.DevicePtr<__half>(),
                                     output.DevicePtr<__half>(),
                                     n,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_sub<__nv_bfloat16>(a.DevicePtr<__nv_bfloat16>(),
                                            b.DevicePtr<__nv_bfloat16>(),
                                            output.DevicePtr<__nv_bfloat16>(),
                                            n,
                                            stream);
      break;
    default:
      THROW_CAIFE("Subtract: unsupported dtype");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SubtractScalarDevice(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("SubtractScalar: tensor size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("SubtractScalar: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_sub_scalar<float>(input.DevicePtr<float>(),
                                           scalar,
                                           output.DevicePtr<float>(),
                                           n,
                                           stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_sub_scalar<__half>(input.DevicePtr<__half>(),
                                            scalar,
                                            output.DevicePtr<__half>(),
                                            n,
                                            stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_sub_scalar<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                                   scalar,
                                                   output.DevicePtr<__nv_bfloat16>(),
                                                   n,
                                                   stream);
      break;
    default:
      THROW_CAIFE("SubtractScalar: unsupported dtype");
  }
#else
  (void)input;
  (void)scalar;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::DivideDevice(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(a.TotalElements()!=b.TotalElements()||a.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Divide: tensor size mismatch");
  }
  if(a.Dtype()!=b.Dtype() || a.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Divide: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(a.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_div<float>(a.DevicePtr<float>(),
                                    b.DevicePtr<float>(),
                                    output.DevicePtr<float>(),
                                    n,
                                    stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_div<__half>(a.DevicePtr<__half>(),
                                     b.DevicePtr<__half>(),
                                     output.DevicePtr<__half>(),
                                     n,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_div<__nv_bfloat16>(a.DevicePtr<__nv_bfloat16>(),
                                            b.DevicePtr<__nv_bfloat16>(),
                                            output.DevicePtr<__nv_bfloat16>(),
                                            n,
                                            stream);
      break;
    default:
      THROW_CAIFE("Divide: unsupported dtype");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::DivideScalarDevice(const CAIF_DeviceTensor &input,float scalar,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("DivideScalar: tensor size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("DivideScalar: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_div_scalar<float>(input.DevicePtr<float>(),
                                           scalar,
                                           output.DevicePtr<float>(),
                                           n,
                                           stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_div_scalar<__half>(input.DevicePtr<__half>(),
                                            scalar,
                                            output.DevicePtr<__half>(),
                                            n,
                                            stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_div_scalar<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                                   scalar,
                                                   output.DevicePtr<__nv_bfloat16>(),
                                                   n,
                                                   stream);
      break;
    default:
      THROW_CAIFE("DivideScalar: unsupported dtype");
  }
#else
  (void)input;
  (void)scalar;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SqrtDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Sqrt: tensor size mismatch");
  }
  if(input.Dtype()!=output.Dtype())
  {
    THROW_CAIFE("Sqrt: tensor dtype mismatch");
  }

  const int64_t n=static_cast<int64_t>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  const cudaStream_t stream=output.Stream().Handle();
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_elementwise_sqrt<float>(input.DevicePtr<float>(),
                                     output.DevicePtr<float>(),
                                     n,
                                     stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_elementwise_sqrt<__half>(input.DevicePtr<__half>(),
                                      output.DevicePtr<__half>(),
                                      n,
                                      stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_elementwise_sqrt<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                             output.DevicePtr<__nv_bfloat16>(),
                                             n,
                                             stream);
      break;
    default:
      THROW_CAIFE("Sqrt: unsupported dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Reduction Operations (tensor output)
//------------------------------------------------------------------------------

void CAIF_Ops::SumAxisDevice(const CAIF_DeviceTensor &input,uint32_t axis,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(axis>=shape.size())
  {
    THROW_CAIFE("SumAxis: axis out of range");
  }

  const int batch=static_cast<int>(shape[0]);
  const int dim=static_cast<int>(shape[1]);

  if(shape.size()!=2)
  {
    THROW_CAIFE("SumAxis: only 2D tensors supported currently");
  }

  // launch_sum_axis* always writes fp32 output regardless of input T —
  // the kernel keeps an fp32 accumulator and stores fp32. If the caller
  // passed an fp16/bf16 output tensor we sum into an fp32 staging buffer
  // here, then cast back to the output's storage dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e in_dt=input.Dtype();
  const Dtype_e out_dt=output.Dtype();
  cudaStream_t stream=output.Stream().Handle();
  const std::vector<uint32_t> out_shape=output.Shape();
  CAIF_DeviceTensor staging;
  CAIF_DeviceTensor *target=&output;
  if(out_dt!=Dtype_e::Float32)
  {
    staging=CAIF_DeviceTensor::Uninitialized(out_shape,output.Stream(),Dtype_e::Float32);
    target=&staging;
  }
  if(axis==0)
  {
    if(output.TotalElements()!=static_cast<size_t>(dim))
    {
      THROW_CAIFE("SumAxis: output size mismatch for axis=0");
    }
    if(in_dt==Dtype_e::Float32)
    {
      launch_sum_axis0<float>(input.DevicePtr<float>(),
                              target->DevicePtr<float>(),
                              batch,dim,stream);
    }
    else if(in_dt==Dtype_e::Float16)
    {
      launch_sum_axis0<__half>(input.DevicePtr<__half>(),
                               target->DevicePtr<float>(),
                               batch,dim,stream);
    }
    else if(in_dt==Dtype_e::BFloat16)
    {
      launch_sum_axis0<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                       target->DevicePtr<float>(),
                                       batch,dim,stream);
    }
    else
    {
      THROW_CAIFE("SumAxis: unsupported input dtype");
    }
  }
  else
  {
    if(output.TotalElements()!=static_cast<size_t>(batch))
    {
      THROW_CAIFE("SumAxis: output size mismatch for axis=1");
    }
    if(in_dt==Dtype_e::Float32)
    {
      launch_sum_axis1<float>(input.DevicePtr<float>(),
                              target->DevicePtr<float>(),
                              batch,dim,stream);
    }
    else if(in_dt==Dtype_e::Float16)
    {
      launch_sum_axis1<__half>(input.DevicePtr<__half>(),
                               target->DevicePtr<float>(),
                               batch,dim,stream);
    }
    else if(in_dt==Dtype_e::BFloat16)
    {
      launch_sum_axis1<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                       target->DevicePtr<float>(),
                                       batch,dim,stream);
    }
    else
    {
      THROW_CAIFE("SumAxis: unsupported input dtype");
    }
  }
  if(out_dt!=Dtype_e::Float32)
  {
    output=staging.To(out_dt);
  }
#else
  (void)input;
  (void)axis;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SumDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(output.TotalElements()!=1)
  {
    THROW_CAIFE("Sum: output must have exactly 1 element");
  }

  const int64_t n=static_cast<int64_t>(input.TotalElements());
  // Reduction always accumulates into fp32 to avoid fp16/bf16 overflow.
  // The kernel writes 4 bytes (fp32). If the caller's `output` tensor is
  // bf16 / fp16 (2 bytes), reinterpreting its buffer as `float*` overruns
  // adjacent memory and reading the buffer back as bf16/fp16 yields
  // garbage. Stage into a typed-fp32 buffer when the output dtype is
  // smaller, then cast to the caller's dtype before returning.
  const cudaStream_t stream=output.Stream().Handle();
  CAIF_DeviceTensor staging;
  float *out_ptr=nullptr;
  const bool needs_staging=(output.Dtype()!=CAIF_DataType::CAIF_DataType_e::Float32);
  if(needs_staging==true)
  {
    staging=CAIF_DeviceTensor::Uninitialized(output.Shape(),
                                              output.Stream(),
                                              CAIF_DataType::CAIF_DataType_e::Float32);
    out_ptr=staging.DevicePtr<float>();
  }
  else
  {
    out_ptr=output.DevicePtr<float>();
  }
  switch(input.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_reduction_sum<float>(input.DevicePtr<float>(),out_ptr,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_reduction_sum<__half>(input.DevicePtr<__half>(),out_ptr,n,stream);
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_reduction_sum<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),out_ptr,n,stream);
      break;
    default:
      THROW_CAIFE("Sum: unsupported dtype");
  }
  if(needs_staging==true)
  {
    output=staging.To(output.Dtype());
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::LogSumExpDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("LogSumExp: input must be 2D [batch x dim]");
  }
  if(output.TotalElements()!=shape[0])
  {
    THROW_CAIFE("LogSumExp: output must have batch elements");
  }

  const int batch=static_cast<int>(shape[0]);
  const int dim=static_cast<int>(shape[1]);

  // launch_logsumexp writes fp32 output regardless of input T. Stage into
  // an fp32 buffer when the caller asked for non-fp32 output and cast back.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e in_dt=input.Dtype();
  const Dtype_e out_dt=output.Dtype();
  cudaStream_t stream=output.Stream().Handle();
  CAIF_DeviceTensor staging;
  CAIF_DeviceTensor *target=&output;
  if(out_dt!=Dtype_e::Float32)
  {
    staging=CAIF_DeviceTensor::Uninitialized(output.Shape(),
                                              output.Stream(),
                                              Dtype_e::Float32);
    target=&staging;
  }
  if(in_dt==Dtype_e::Float32)
  {
    launch_logsumexp<float>(input.DevicePtr<float>(),
                            target->DevicePtr<float>(),
                            batch,dim,stream);
  }
  else if(in_dt==Dtype_e::Float16)
  {
    launch_logsumexp<__half>(input.DevicePtr<__half>(),
                             target->DevicePtr<float>(),
                             batch,dim,stream);
  }
  else if(in_dt==Dtype_e::BFloat16)
  {
    launch_logsumexp<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                     target->DevicePtr<float>(),
                                     batch,dim,stream);
  }
  else
  {
    THROW_CAIFE("LogSumExp: unsupported input dtype");
  }
  if(out_dt!=Dtype_e::Float32)
  {
    output=staging.To(out_dt);
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Top-K and Scatter Operations
//------------------------------------------------------------------------------

void CAIF_Ops::TopKDevice(const CAIF_DeviceTensor &input,
          uint32_t k,
          CAIF_DeviceTensor &indices,
          CAIF_DeviceTensor &values)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("TopK: input must be 2D [batch x dim]");
  }

  const uint32_t batch=shape[0];
  const uint32_t dim=shape[1];

  if(k>dim)
  {
    THROW_CAIFE("TopK: k must be <= dim");
  }
  if(indices.TotalElements()!=batch*k||values.TotalElements()!=batch*k)
  {
    THROW_CAIFE("TopK: output tensors must be [batch x k]");
  }

  // Dispatch by input dtype — input and values share dtype, indices are int32.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=input.Dtype();
  int32_t *idx_ptr=indices.DevicePtr<int32_t>();
  const int n_batch=static_cast<int>(batch);
  const int n_dim=static_cast<int>(dim);
  cudaStream_t stream=values.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_topk<float>(input.DevicePtr<float>(),
                       idx_ptr,
                       values.DevicePtr<float>(),
                       n_batch,n_dim,static_cast<int>(k),stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_topk<__half>(input.DevicePtr<__half>(),
                        idx_ptr,
                        values.DevicePtr<__half>(),
                        n_batch,n_dim,static_cast<int>(k),stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_topk<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                idx_ptr,
                                values.DevicePtr<__nv_bfloat16>(),
                                n_batch,n_dim,static_cast<int>(k),stream);
  }
  else
  {
    THROW_CAIFE("TopK: unsupported input dtype");
  }
#else
  (void)input;
  (void)k;
  (void)indices;
  (void)values;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::NormalizeRowsDevice(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("NormalizeRows: input must be 2D");
  }
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("NormalizeRows: input/output size mismatch");
  }

  const int batch=static_cast<int>(shape[0]);
  const int dim=static_cast<int>(shape[1]);

  // Dispatch by tensor dtype — input and output share dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=input.Dtype();
  cudaStream_t stream=output.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_normalize_rows<float>(input.DevicePtr<float>(),
                                 output.DevicePtr<float>(),
                                 batch,dim,stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_normalize_rows<__half>(input.DevicePtr<__half>(),
                                  output.DevicePtr<__half>(),
                                  batch,dim,stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_normalize_rows<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                          output.DevicePtr<__nv_bfloat16>(),
                                          batch,dim,stream);
  }
  else
  {
    THROW_CAIFE("NormalizeRows: unsupported input dtype");
  }
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::NormalizeRowsBackwardTopKGatherDevice(const CAIF_DeviceTensor &grad_w,
                                     const CAIF_DeviceTensor &probs,
                                     const CAIF_DeviceTensor &indices,
                                     CAIF_DeviceTensor &grad_p_topk)
{
#ifdef USE_CAIF_CUDA
  const auto &gw_shape=grad_w.Shape();
  if(gw_shape.size()!=2)
  {
    THROW_CAIFE("NormalizeRowsBackwardTopKGather: grad_w must be 2D [N x K]");
  }
  const uint32_t num_tokens=gw_shape[0];
  const uint32_t top_k=gw_shape[1];
  if(top_k>32)
  {
    THROW_CAIFE("NormalizeRowsBackwardTopKGather: top_k must be <= 32");
  }
  const auto &p_shape=probs.Shape();
  if(p_shape.size()!=2||p_shape[0]!=num_tokens)
  {
    THROW_CAIFE("NormalizeRowsBackwardTopKGather: probs must be [N x E]");
  }
  const uint32_t num_experts=p_shape[1];
  if(indices.Shape()!=gw_shape||grad_p_topk.Shape()!=gw_shape)
  {
    THROW_CAIFE("NormalizeRowsBackwardTopKGather: indices/grad_p_topk shape mismatch");
  }

  // Dispatch by tensor dtype — grad_w / probs / grad_p_topk share dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=grad_w.Dtype();
  const int32_t *idx_ptr=indices.DevicePtr<int32_t>();
  const int n_tok=static_cast<int>(num_tokens);
  const int n_exp=static_cast<int>(num_experts);
  const int n_topk=static_cast<int>(top_k);
  cudaStream_t stream=grad_p_topk.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_normalize_rows_backward_topk_gather<float>(grad_w.DevicePtr<float>(),
                                                      probs.DevicePtr<float>(),
                                                      idx_ptr,
                                                      grad_p_topk.DevicePtr<float>(),
                                                      n_tok,n_exp,n_topk,stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_normalize_rows_backward_topk_gather<__half>(grad_w.DevicePtr<__half>(),
                                                       probs.DevicePtr<__half>(),
                                                       idx_ptr,
                                                       grad_p_topk.DevicePtr<__half>(),
                                                       n_tok,n_exp,n_topk,stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_normalize_rows_backward_topk_gather<__nv_bfloat16>(
                                          grad_w.DevicePtr<__nv_bfloat16>(),
                                          probs.DevicePtr<__nv_bfloat16>(),
                                          idx_ptr,
                                          grad_p_topk.DevicePtr<__nv_bfloat16>(),
                                          n_tok,n_exp,n_topk,stream);
  }
  else
  {
    THROW_CAIFE("NormalizeRowsBackwardTopKGather: unsupported dtype");
  }
#else
  (void)grad_w;
  (void)probs;
  (void)indices;
  (void)grad_p_topk;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::GatherTopKValuesDevice(const CAIF_DeviceTensor &scores,
                            const CAIF_DeviceTensor &indices,
                            CAIF_DeviceTensor &out)
{
#ifdef USE_CAIF_CUDA
  const std::vector<uint32_t> &s_shape=scores.Shape();
  if(s_shape.size()!=2)
  {
    THROW_CAIFE("GatherTopKValues: scores must be 2D [N x E]");
  }
  const std::vector<uint32_t> &i_shape=indices.Shape();
  const std::vector<uint32_t> &o_shape=out.Shape();
  if(i_shape.size()!=2||o_shape.size()!=2||i_shape!=o_shape||i_shape[0]!=s_shape[0])
  {
    THROW_CAIFE("GatherTopKValues: indices/out must be [N x K], N must match scores");
  }
  const uint32_t num_tokens=s_shape[0];
  const uint32_t num_experts=s_shape[1];
  const uint32_t top_k=i_shape[1];
  if(top_k>32)
  {
    THROW_CAIFE("GatherTopKValues: top_k must be <= 32");
  }
  if(scores.Dtype()!=out.Dtype())
  {
    THROW_CAIFE("GatherTopKValues: scores and out must share dtype");
  }

  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=scores.Dtype();
  const int32_t *idx_ptr=indices.DevicePtr<int32_t>();
  const int n_tok=static_cast<int>(num_tokens);
  const int n_exp=static_cast<int>(num_experts);
  const int n_topk=static_cast<int>(top_k);
  cudaStream_t stream=out.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_gather_topk_values<float>(scores.DevicePtr<float>(),
                                     idx_ptr,
                                     out.DevicePtr<float>(),
                                     n_tok,
                                     n_exp,
                                     n_topk,
                                     stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_gather_topk_values<__half>(scores.DevicePtr<__half>(),
                                      idx_ptr,
                                      out.DevicePtr<__half>(),
                                      n_tok,
                                      n_exp,
                                      n_topk,
                                      stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_gather_topk_values<__nv_bfloat16>(scores.DevicePtr<__nv_bfloat16>(),
                                             idx_ptr,
                                             out.DevicePtr<__nv_bfloat16>(),
                                             n_tok,
                                             n_exp,
                                             n_topk,
                                             stream);
  }
  else
  {
    THROW_CAIFE("GatherTopKValues: unsupported dtype");
  }
#else
  (void)scores;
  (void)indices;
  (void)out;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::ScatterAddDevice(const CAIF_DeviceTensor &values,
                const CAIF_DeviceTensor &indices,
                CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &val_shape=values.Shape();
  const auto &idx_shape=indices.Shape();
  const auto &out_shape=output.Shape();

  if(val_shape.size()!=2||idx_shape.size()!=2||out_shape.size()!=2)
  {
    THROW_CAIFE("ScatterAdd: all tensors must be 2D");
  }
  if(val_shape[0]!=idx_shape[0]||val_shape[1]!=idx_shape[1])
  {
    THROW_CAIFE("ScatterAdd: values and indices shapes must match");
  }
  if(val_shape[0]!=out_shape[0])
  {
    THROW_CAIFE("ScatterAdd: batch dimensions must match");
  }

  const int batch=static_cast<int>(val_shape[0]);
  const int k=static_cast<int>(val_shape[1]);
  const int dim=static_cast<int>(out_shape[1]);

  // Dispatch by tensor dtype — values and output share dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=values.Dtype();
  const int32_t *idx_ptr=indices.DevicePtr<int32_t>();
  cudaStream_t stream=output.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_scatter_add<float>(values.DevicePtr<float>(),
                              idx_ptr,
                              output.DevicePtr<float>(),
                              batch,k,dim,stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_scatter_add<__half>(values.DevicePtr<__half>(),
                               idx_ptr,
                               output.DevicePtr<__half>(),
                               batch,k,dim,stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_scatter_add<__nv_bfloat16>(values.DevicePtr<__nv_bfloat16>(),
                                      idx_ptr,
                                      output.DevicePtr<__nv_bfloat16>(),
                                      batch,k,dim,stream);
  }
  else
  {
    THROW_CAIFE("ScatterAdd: unsupported values dtype");
  }
#else
  (void)values;
  (void)indices;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// MoE-specific Operations
//------------------------------------------------------------------------------

void CAIF_Ops::MoEDispatchDevice(const CAIF_DeviceTensor &input,
                 const CAIF_DeviceTensor &expert_indices,
                 uint32_t top_k,
                 const std::vector<uint32_t> &token_counts,
                 std::vector<CAIF_DeviceTensor> &expert_inputs)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoEDispatch: input must be 2D [num_tokens x dim]");
  }

  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];
  const uint32_t num_experts=static_cast<uint32_t>(token_counts.size());

  // Copy indices to host for dispatch logic (stored as floats, cast to int)
  std::vector<int32_t> indices_i32(num_tokens*top_k);
  expert_indices.CopyToHostRaw(indices_i32.data());

  // Track position within each expert's input
  std::vector<uint32_t> expert_positions(num_experts,0);

  // Copy input to host for dispatch (can be optimized with CUDA kernel later)
  std::vector<float> input_host(num_tokens*dim);
  input.CopyToHost(input_host.data());

  // Prepare host buffers for each expert
  std::vector<std::vector<float>> expert_buffers(num_experts);
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(token_counts[e]>0)
    {
      expert_buffers[e].resize(token_counts[e]*dim);
    }
  }

  // Dispatch tokens to experts
  for(uint32_t t=0;t<num_tokens;++t)
  {
    for(uint32_t k=0;k<top_k;++k)
    {
      const int32_t expert_idx=indices_i32[t*top_k+k];
      if(expert_idx>=0&&expert_idx<static_cast<int32_t>(num_experts))
      {
        const uint32_t pos=expert_positions[expert_idx];
        if(pos<token_counts[expert_idx])
        {
          // Copy token to expert buffer
          for(uint32_t d=0;d<dim;++d)
          {
            expert_buffers[expert_idx][pos*dim+d]=input_host[t*dim+d];
          }
          ++expert_positions[expert_idx];
        }
      }
    }
  }

  // Copy to device tensors
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(token_counts[e]>0&&expert_inputs[e].TotalElements()>0)
    {
      expert_inputs[e].CopyFromHost(expert_buffers[e].data(),expert_buffers[e].size());
    }
  }
#else
  (void)input;
  (void)expert_indices;
  (void)top_k;
  (void)token_counts;
  (void)expert_inputs;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoECombineDevice(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                const CAIF_DeviceTensor &expert_indices,
                const CAIF_DeviceTensor &expert_weights,
                uint32_t top_k,
                const std::vector<uint32_t> &token_counts,
                CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=output.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoECombine: output must be 2D [num_tokens x dim]");
  }

  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];
  const uint32_t num_experts=static_cast<uint32_t>(token_counts.size());

  // Copy indices and weights to host
  std::vector<int32_t> indices_i32(num_tokens*top_k);
  std::vector<float> weights_host(num_tokens*top_k);
  expert_indices.CopyToHostRaw(indices_i32.data());
  expert_weights.CopyToHost(weights_host.data());

  // Copy expert outputs to host
  std::vector<std::vector<float>> expert_buffers(num_experts);
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(token_counts[e]>0&&expert_outputs[e].TotalElements()>0)
    {
      expert_buffers[e].resize(token_counts[e]*dim);
      expert_outputs[e].CopyToHost(expert_buffers[e].data());
    }
  }

  // Track position within each expert's output
  std::vector<uint32_t> expert_positions(num_experts,0);

  // Combine outputs
  std::vector<float> output_host(num_tokens*dim,0.0f);

  for(uint32_t t=0;t<num_tokens;++t)
  {
    for(uint32_t k=0;k<top_k;++k)
    {
      const int32_t expert_idx=indices_i32[t*top_k+k];
      const float weight=weights_host[t*top_k+k];

      if(expert_idx>=0&&expert_idx<static_cast<int32_t>(num_experts))
      {
        const uint32_t pos=expert_positions[expert_idx];
        if(pos<token_counts[expert_idx]&&expert_buffers[expert_idx].size()>0)
        {
          // Add weighted expert output
          for(uint32_t d=0;d<dim;++d)
          {
            output_host[t*dim+d]+=weight*expert_buffers[expert_idx][pos*dim+d];
          }
          ++expert_positions[expert_idx];
        }
      }
    }
  }

  // Copy to device
  output.CopyFromHost(output_host.data(),output_host.size());
#else
  (void)expert_outputs;
  (void)expert_indices;
  (void)expert_weights;
  (void)top_k;
  (void)token_counts;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoECombineBackwardDevice(const CAIF_DeviceTensor &grad_output,
                        const std::vector<CAIF_DeviceTensor> &expert_outputs,
                        const CAIF_DeviceTensor &expert_indices,
                        const CAIF_DeviceTensor &expert_weights,
                        uint32_t top_k,
                        const std::vector<uint32_t> &token_counts,
                        std::vector<CAIF_DeviceTensor> &grad_expert_outputs,
                        CAIF_DeviceTensor &grad_weights)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=grad_output.Shape();
  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];
  const uint32_t num_experts=static_cast<uint32_t>(token_counts.size());

  // Copy to host (indices stored as floats)
  std::vector<float> grad_out_host(num_tokens*dim);
  std::vector<int32_t> indices_i32(num_tokens*top_k);
  std::vector<float> weights_host(num_tokens*top_k);

  grad_output.CopyToHost(grad_out_host.data());
  expert_indices.CopyToHostRaw(indices_i32.data());
  expert_weights.CopyToHost(weights_host.data());

  // Copy expert outputs to host for weight gradient
  std::vector<std::vector<float>> expert_out_host(num_experts);
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(token_counts[e]>0&&expert_outputs[e].TotalElements()>0)
    {
      expert_out_host[e].resize(token_counts[e]*dim);
      expert_outputs[e].CopyToHost(expert_out_host[e].data());
    }
  }

  // Prepare gradient buffers
  std::vector<std::vector<float>> grad_expert_host(num_experts);
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(token_counts[e]>0)
    {
      grad_expert_host[e].resize(token_counts[e]*dim,0.0f);
    }
  }
  std::vector<float> grad_weights_host(num_tokens*top_k,0.0f);

  // Track positions
  std::vector<uint32_t> expert_positions(num_experts,0);

  // Compute gradients
  for(uint32_t t=0;t<num_tokens;++t)
  {
    for(uint32_t k=0;k<top_k;++k)
    {
      const int32_t expert_idx=indices_i32[t*top_k+k];
      const float weight=weights_host[t*top_k+k];

      if(expert_idx>=0&&expert_idx<static_cast<int32_t>(num_experts))
      {
        const uint32_t pos=expert_positions[expert_idx];
        if(pos<token_counts[expert_idx])
        {
          // grad_expert = weight * grad_output
          for(uint32_t d=0;d<dim;++d)
          {
            grad_expert_host[expert_idx][pos*dim+d]+=weight*grad_out_host[t*dim+d];
          }

          // grad_weight = dot(grad_output, expert_output)
          float dot=0.0f;
          for(uint32_t d=0;d<dim;++d)
          {
            dot+=grad_out_host[t*dim+d]*expert_out_host[expert_idx][pos*dim+d];
          }
          grad_weights_host[t*top_k+k]=dot;

          ++expert_positions[expert_idx];
        }
      }
    }
  }

  // Copy back to device - use grad_output's stream (grad_weights is uninitialized)
  CAIF_CudaStream &stream=const_cast<CAIF_CudaStream&>(grad_output.Stream());
  grad_expert_outputs.resize(num_experts);
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(token_counts[e]>0)
    {
      grad_expert_outputs[e]=CAIF_DeviceTensor::Uninitialized({token_counts[e],dim},stream);
      grad_expert_outputs[e].CopyFromHost(grad_expert_host[e].data(),grad_expert_host[e].size());
    }
    else
    {
      grad_expert_outputs[e]=CAIF_DeviceTensor();
    }
  }

  CAIF_DeviceTensor temp=CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},stream);
  temp.CopyFromHost(grad_weights_host.data(),grad_weights_host.size());
  grad_weights=std::move(temp);
#else
  (void)grad_output;
  (void)expert_outputs;
  (void)expert_indices;
  (void)expert_weights;
  (void)top_k;
  (void)token_counts;
  (void)grad_expert_outputs;
  (void)grad_weights;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoEDispatchBackwardDevice(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
                         const CAIF_DeviceTensor &expert_indices,
                         uint32_t top_k,
                         const std::vector<uint32_t> &token_counts,
                         CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=grad_input.Shape();
  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];
  const uint32_t num_experts=static_cast<uint32_t>(token_counts.size());

  // Copy indices to host (indices stored as floats)
  std::vector<int32_t> indices_i32(num_tokens*top_k);
  expert_indices.CopyToHostRaw(indices_i32.data());

  // Copy expert gradients to host
  std::vector<std::vector<float>> grad_expert_host(num_experts);
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(token_counts[e]>0&&grad_expert_inputs[e].TotalElements()>0)
    {
      grad_expert_host[e].resize(token_counts[e]*dim);
      grad_expert_inputs[e].CopyToHost(grad_expert_host[e].data());
    }
  }

  // Track positions
  std::vector<uint32_t> expert_positions(num_experts,0);

  // Combine gradients back to input positions
  std::vector<float> grad_input_host(num_tokens*dim,0.0f);

  for(uint32_t t=0;t<num_tokens;++t)
  {
    for(uint32_t k=0;k<top_k;++k)
    {
      const int32_t expert_idx=indices_i32[t*top_k+k];

      if(expert_idx>=0&&expert_idx<static_cast<int32_t>(num_experts))
      {
        const uint32_t pos=expert_positions[expert_idx];
        if(pos<token_counts[expert_idx]&&grad_expert_host[expert_idx].size()>0)
        {
          // Add gradient from this expert
          for(uint32_t d=0;d<dim;++d)
          {
            grad_input_host[t*dim+d]+=grad_expert_host[expert_idx][pos*dim+d];
          }
          ++expert_positions[expert_idx];
        }
      }
    }
  }

  // Copy to device
  grad_input.CopyFromHost(grad_input_host.data(),grad_input_host.size());
#else
  (void)grad_expert_inputs;
  (void)expert_indices;
  (void)top_k;
  (void)token_counts;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// GPU-Optimized MoE Operations (Phase 6)
//------------------------------------------------------------------------------

void CAIF_Ops::MoETopKGatingDevice(const CAIF_DeviceTensor &router_logits,
                   uint32_t num_experts,
                   uint32_t top_k,
                   CAIF_DeviceTensor &expert_indices,
                   CAIF_DeviceTensor &expert_weights,
                   CAIF_DeviceTensor &router_probs)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=router_logits.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoETopKGating: router_logits must be 2D [num_tokens x num_experts]");
  }

  const uint32_t num_tokens=shape[0];
  if(shape[1]!=num_experts)
  {
    THROW_CAIFE("MoETopKGating: router_logits last dim must equal num_experts");
  }

  // launch_moe_topk_gating<T> templated on the router_logits dtype.
  // expert_indices is Int32; expert_weights / router_probs are Float32
  // (router-output contract — internal accumulation runs in fp32 for
  // numerical stability regardless of input dtype).
  switch(router_logits.Dtype())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      launch_moe_topk_gating<float>(router_logits.DevicePtr<float>(),
                                    expert_indices.DevicePtr<int32_t>(),
                                    expert_weights.DevicePtr<float>(),
                                    router_probs.DevicePtr<float>(),
                                    static_cast<int>(num_tokens),
                                    static_cast<int>(num_experts),
                                    static_cast<int>(top_k),
                                    expert_indices.Stream().Handle());
      break;
    case CAIF_DataType::CAIF_DataType_e::Float16:
      launch_moe_topk_gating<__half>(router_logits.DevicePtr<__half>(),
                                     expert_indices.DevicePtr<int32_t>(),
                                     expert_weights.DevicePtr<float>(),
                                     router_probs.DevicePtr<float>(),
                                     static_cast<int>(num_tokens),
                                     static_cast<int>(num_experts),
                                     static_cast<int>(top_k),
                                     expert_indices.Stream().Handle());
      break;
    case CAIF_DataType::CAIF_DataType_e::BFloat16:
      launch_moe_topk_gating<__nv_bfloat16>(router_logits.DevicePtr<__nv_bfloat16>(),
                                            expert_indices.DevicePtr<int32_t>(),
                                            expert_weights.DevicePtr<float>(),
                                            router_probs.DevicePtr<float>(),
                                            static_cast<int>(num_tokens),
                                            static_cast<int>(num_experts),
                                            static_cast<int>(top_k),
                                            expert_indices.Stream().Handle());
      break;
    default:
      THROW_CAIFE("MoETopKGating: unsupported dtype");
  }
#else
  (void)router_logits;
  (void)num_experts;
  (void)top_k;
  (void)expert_indices;
  (void)expert_weights;
  (void)router_probs;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoEZLossGradAddDevice(const CAIF_DeviceTensor &logsumexp_scaled,
                     const CAIF_DeviceTensor &probs,
                     CAIF_DeviceTensor &grad_logits)
{
#ifdef USE_CAIF_CUDA
  const auto &probs_shape=probs.Shape();
  if(probs_shape.size()!=2)
  {
    THROW_CAIFE("MoEZLossGradAdd: probs must be 2D [num_tokens x num_experts]");
  }
  if(logsumexp_scaled.Shape().size()!=1)
  {
    THROW_CAIFE("MoEZLossGradAdd: logsumexp_scaled must be 1D [num_tokens]");
  }
  if(logsumexp_scaled.Shape()[0]!=probs_shape[0])
  {
    THROW_CAIFE("MoEZLossGradAdd: logsumexp_scaled length must match probs rows");
  }
  if(grad_logits.Shape()!=probs_shape)
  {
    THROW_CAIFE("MoEZLossGradAdd: grad_logits shape must match probs");
  }

  const int num_tokens=static_cast<int>(probs_shape[0]);
  const int num_experts=static_cast<int>(probs_shape[1]);

  // Dispatch by activation dtype — all three tensors share dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=grad_logits.Dtype();
  cudaStream_t stream=grad_logits.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_moe_z_loss_grad<float>(logsumexp_scaled.DevicePtr<float>(),
                                   probs.DevicePtr<float>(),
                                   grad_logits.DevicePtr<float>(),
                                   num_tokens,
                                   num_experts,
                                   stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_moe_z_loss_grad<__half>(logsumexp_scaled.DevicePtr<__half>(),
                                    probs.DevicePtr<__half>(),
                                    grad_logits.DevicePtr<__half>(),
                                    num_tokens,
                                    num_experts,
                                    stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_moe_z_loss_grad<__nv_bfloat16>(logsumexp_scaled.DevicePtr<__nv_bfloat16>(),
                                           probs.DevicePtr<__nv_bfloat16>(),
                                           grad_logits.DevicePtr<__nv_bfloat16>(),
                                           num_tokens,
                                           num_experts,
                                           stream);
  }
  else
  {
    THROW_CAIFE("MoEZLossGradAdd: unsupported grad_logits dtype");
  }
#else
  (void)logsumexp_scaled;
  (void)probs;
  (void)grad_logits;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoEGroupMask(CAIF_DeviceTensor &selection,uint32_t n_group,uint32_t topk_group)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=selection.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoEGroupMask: selection must be 2D [num_tokens x num_experts]");
  }
  if(n_group==0||shape[1]%n_group!=0)
  {
    THROW_CAIFE("MoEGroupMask: num_experts must be divisible by n_group");
  }
  if(topk_group==0||topk_group>n_group)
  {
    THROW_CAIFE("MoEGroupMask: topk_group must be in [1, n_group]");
  }
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=selection.Dtype();
  const int num_tokens=static_cast<int>(shape[0]);
  const int num_experts=static_cast<int>(shape[1]);
  const int groups=static_cast<int>(n_group);
  const int top_groups=static_cast<int>(topk_group);
  cudaStream_t stream=selection.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_moe_group_mask<float>(selection.DevicePtr<float>(),num_tokens,num_experts,groups,top_groups,stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_moe_group_mask<__half>(selection.DevicePtr<__half>(),num_tokens,num_experts,groups,top_groups,stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_moe_group_mask<__nv_bfloat16>(selection.DevicePtr<__nv_bfloat16>(),
                                         num_tokens,
                                         num_experts,
                                         groups,
                                         top_groups,
                                         stream);
  }
  else
  {
    THROW_CAIFE("MoEGroupMask: unsupported selection dtype");
  }
#else
  (void)selection;
  (void)n_group;
  (void)topk_group;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoEBiasUpdate(CAIF_DeviceTensor &bias,const CAIF_DeviceTensor &expert_counts,float rate)
{
#ifdef USE_CAIF_CUDA
  if(expert_counts.TotalElements()!=bias.TotalElements())
  {
    THROW_CAIFE("MoEBiasUpdate: expert_counts must match bias length");
  }
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=bias.Dtype();
  const int num_experts=static_cast<int>(bias.TotalElements());
  const int *counts=expert_counts.DevicePtr<int32_t>();
  cudaStream_t stream=bias.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_moe_bias_update<float>(bias.DevicePtr<float>(),counts,num_experts,rate,stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_moe_bias_update<__half>(bias.DevicePtr<__half>(),counts,num_experts,rate,stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_moe_bias_update<__nv_bfloat16>(bias.DevicePtr<__nv_bfloat16>(),counts,num_experts,rate,stream);
  }
  else
  {
    THROW_CAIFE("MoEBiasUpdate: unsupported bias dtype");
  }
#else
  (void)bias;
  (void)expert_counts;
  (void)rate;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoECountPerExpertDevice(const CAIF_DeviceTensor &expert_indices,
                       uint32_t num_experts,
                       uint32_t top_k,
                       CAIF_DeviceTensor &expert_counts)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=expert_indices.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoECountPerExpert: expert_indices must be 2D [num_tokens x top_k]");
  }

  const uint32_t num_tokens=shape[0];

  // Zero the counts first
  cudaMemsetAsync(expert_counts.DeviceDataRaw(),0,
                  num_experts*sizeof(int32_t),
                  expert_counts.Stream().Handle());

  launch_moe_count_per_expert(expert_indices.DevicePtr<int32_t>(),
                              expert_counts.DevicePtr<int32_t>(),
                              static_cast<int>(num_tokens),
                              static_cast<int>(num_experts),
                              static_cast<int>(top_k),
                              0,  // No capacity limit in count phase
                              expert_counts.Stream().Handle());
#else
  (void)expert_indices;
  (void)num_experts;
  (void)top_k;
  (void)expert_counts;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoEDispatchGPUDevice(const CAIF_DeviceTensor &input,
                    const CAIF_DeviceTensor &expert_indices,
                    const CAIF_DeviceTensor &dispatch_map,
                    const CAIF_DeviceTensor &expert_offsets,
                    uint32_t top_k,
                    CAIF_DeviceTensor &expert_buffer)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoEDispatchGPU: input must be 2D [num_tokens x dim]");
  }

  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];

  // Dispatch by activation dtype — input and expert_buffer must share dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=input.Dtype();
  const int32_t *idx_ptr=expert_indices.DevicePtr<int32_t>();
  const int32_t *map_ptr=dispatch_map.DevicePtr<int32_t>();
  const int32_t *off_ptr=expert_offsets.DevicePtr<int32_t>();
  const int n_tok=static_cast<int>(num_tokens);
  const int n_dim=static_cast<int>(dim);
  const int n_topk=static_cast<int>(top_k);
  cudaStream_t stream=expert_buffer.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_moe_dispatch<float>(input.DevicePtr<float>(),
                               idx_ptr,
                               map_ptr,
                               expert_buffer.DevicePtr<float>(),
                               off_ptr,
                               n_tok,
                               n_dim,
                               n_topk,
                               stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_moe_dispatch<__half>(input.DevicePtr<__half>(),
                                idx_ptr,
                                map_ptr,
                                expert_buffer.DevicePtr<__half>(),
                                off_ptr,
                                n_tok,
                                n_dim,
                                n_topk,
                                stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_moe_dispatch<__nv_bfloat16>(input.DevicePtr<__nv_bfloat16>(),
                                       idx_ptr,
                                       map_ptr,
                                       expert_buffer.DevicePtr<__nv_bfloat16>(),
                                       off_ptr,
                                       n_tok,
                                       n_dim,
                                       n_topk,
                                       stream);
  }
  else
  {
    THROW_CAIFE("MoEDispatchGPU: unsupported input dtype");
  }
#else
  (void)input;
  (void)expert_indices;
  (void)dispatch_map;
  (void)expert_offsets;
  (void)top_k;
  (void)expert_buffer;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoECombineGPUDevice(const CAIF_DeviceTensor &expert_buffer,
                   const CAIF_DeviceTensor &expert_indices,
                   const CAIF_DeviceTensor &expert_weights,
                   const CAIF_DeviceTensor &dispatch_map,
                   const CAIF_DeviceTensor &expert_offsets,
                   uint32_t top_k,
                   CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=output.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoECombineGPU: output must be 2D [num_tokens x dim]");
  }

  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];

  // Dispatch by activation dtype — expert_buffer / expert_weights / output share dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=output.Dtype();
  const int32_t *idx_ptr=expert_indices.DevicePtr<int32_t>();
  const int32_t *map_ptr=dispatch_map.DevicePtr<int32_t>();
  const int32_t *off_ptr=expert_offsets.DevicePtr<int32_t>();
  const int n_tok=static_cast<int>(num_tokens);
  const int n_dim=static_cast<int>(dim);
  const int n_topk=static_cast<int>(top_k);
  cudaStream_t stream=output.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_moe_combine<float>(expert_buffer.DevicePtr<float>(),
                              idx_ptr,
                              expert_weights.DevicePtr<float>(),
                              map_ptr,
                              off_ptr,
                              output.DevicePtr<float>(),
                              n_tok,
                              n_dim,
                              n_topk,
                              stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_moe_combine<__half>(expert_buffer.DevicePtr<__half>(),
                               idx_ptr,
                               expert_weights.DevicePtr<__half>(),
                               map_ptr,
                               off_ptr,
                               output.DevicePtr<__half>(),
                               n_tok,
                               n_dim,
                               n_topk,
                               stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_moe_combine<__nv_bfloat16>(expert_buffer.DevicePtr<__nv_bfloat16>(),
                                      idx_ptr,
                                      expert_weights.DevicePtr<__nv_bfloat16>(),
                                      map_ptr,
                                      off_ptr,
                                      output.DevicePtr<__nv_bfloat16>(),
                                      n_tok,
                                      n_dim,
                                      n_topk,
                                      stream);
  }
  else
  {
    THROW_CAIFE("MoECombineGPU: unsupported output dtype");
  }
#else
  (void)expert_buffer;
  (void)expert_indices;
  (void)expert_weights;
  (void)dispatch_map;
  (void)expert_offsets;
  (void)top_k;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

uint32_t CAIF_Ops::MoEBuildDispatchMapDevice(const CAIF_DeviceTensor &expert_indices,
                             uint32_t num_experts,
                             uint32_t top_k,
                             uint32_t capacity_per_expert,
                             CAIF_DeviceTensor &dispatch_map,
                             CAIF_DeviceTensor &expert_offsets)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=expert_indices.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoEBuildDispatchMap: expert_indices must be 2D [num_tokens x top_k]");
  }

  const uint32_t num_tokens=shape[0];

  // Copy indices to host for dispatch map building
  std::vector<int32_t> indices_i32(num_tokens*top_k);
  expert_indices.CopyToHostRaw(indices_i32.data());

  // Count tokens per expert
  std::vector<uint32_t> counts(num_experts,0);
  for(uint32_t t=0;t<num_tokens;++t)
  {
    for(uint32_t k=0;k<top_k;++k)
    {
      const int32_t expert_idx=indices_i32[t*top_k+k];
      if(expert_idx>=0&&expert_idx<static_cast<int32_t>(num_experts))
      {
        ++counts[expert_idx];
      }
    }
  }

  // Apply capacity limit and compute offsets
  std::vector<int32_t> offsets(num_experts+1,0);
  for(uint32_t e=0;e<num_experts;++e)
  {
    if(capacity_per_expert>0&&counts[e]>capacity_per_expert)
    {
      counts[e]=capacity_per_expert;
    }
    offsets[e+1]=offsets[e]+static_cast<int32_t>(counts[e]);
  }
  const uint32_t total_assigned=static_cast<uint32_t>(offsets[num_experts]);

  // Build dispatch map with position tracking
  std::vector<int32_t> map_host(num_tokens*top_k,-1);
  std::vector<uint32_t> positions(num_experts,0);

  for(uint32_t t=0;t<num_tokens;++t)
  {
    for(uint32_t k=0;k<top_k;++k)
    {
      const int32_t expert_idx=indices_i32[t*top_k+k];
      if(expert_idx>=0&&expert_idx<static_cast<int32_t>(num_experts))
      {
        const uint32_t capacity=counts[expert_idx];
        if(positions[expert_idx]<capacity)
        {
          map_host[t*top_k+k]=static_cast<int32_t>(positions[expert_idx]);
          ++positions[expert_idx];
        }
        // else: token dropped due to capacity
      }
    }
  }

  // Copy to device
  dispatch_map.CopyFromHostRaw(map_host.data(),
                               map_host.size()*sizeof(int32_t));
  expert_offsets.CopyFromHostRaw(offsets.data(),
                                 offsets.size()*sizeof(int32_t));

  return total_assigned;
#else
  (void)expert_indices;
  (void)num_experts;
  (void)top_k;
  (void)capacity_per_expert;
  (void)dispatch_map;
  (void)expert_offsets;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoECombineBackwardGPUDevice(const CAIF_DeviceTensor &grad_output,
                           const CAIF_DeviceTensor &expert_buffer,
                           const CAIF_DeviceTensor &expert_indices,
                           const CAIF_DeviceTensor &expert_weights,
                           const CAIF_DeviceTensor &dispatch_map,
                           const CAIF_DeviceTensor &expert_offsets,
                           uint32_t top_k,
                           CAIF_DeviceTensor &grad_expert_buffer,
                           CAIF_DeviceTensor &grad_weights)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=grad_output.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoECombineBackwardGPU: grad_output must be 2D [num_tokens x dim]");
  }

  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];

  // Dispatch by activation dtype — grad_output / weights / buffer / grads share dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=grad_output.Dtype();
  const int32_t *idx_ptr=expert_indices.DevicePtr<int32_t>();
  const int32_t *map_ptr=dispatch_map.DevicePtr<int32_t>();
  const int32_t *off_ptr=expert_offsets.DevicePtr<int32_t>();
  const int n_tok=static_cast<int>(num_tokens);
  const int n_dim=static_cast<int>(dim);
  const int n_topk=static_cast<int>(top_k);
  cudaStream_t stream=grad_expert_buffer.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_moe_combine_backward_grad_expert<float>(grad_output.DevicePtr<float>(),
                                                   idx_ptr,
                                                   expert_weights.DevicePtr<float>(),
                                                   map_ptr,
                                                   off_ptr,
                                                   grad_expert_buffer.DevicePtr<float>(),
                                                   n_tok,
                                                   n_dim,
                                                   n_topk,
                                                   stream);
    launch_moe_combine_backward_grad_weights<float>(grad_output.DevicePtr<float>(),
                                                    expert_buffer.DevicePtr<float>(),
                                                    idx_ptr,
                                                    map_ptr,
                                                    off_ptr,
                                                    grad_weights.DevicePtr<float>(),
                                                    n_tok,
                                                    n_dim,
                                                    n_topk,
                                                    stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_moe_combine_backward_grad_expert<__half>(grad_output.DevicePtr<__half>(),
                                                    idx_ptr,
                                                    expert_weights.DevicePtr<__half>(),
                                                    map_ptr,
                                                    off_ptr,
                                                    grad_expert_buffer.DevicePtr<__half>(),
                                                    n_tok,
                                                    n_dim,
                                                    n_topk,
                                                    stream);
    launch_moe_combine_backward_grad_weights<__half>(grad_output.DevicePtr<__half>(),
                                                     expert_buffer.DevicePtr<__half>(),
                                                     idx_ptr,
                                                     map_ptr,
                                                     off_ptr,
                                                     grad_weights.DevicePtr<__half>(),
                                                     n_tok,
                                                     n_dim,
                                                     n_topk,
                                                     stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_moe_combine_backward_grad_expert<__nv_bfloat16>(
                                                grad_output.DevicePtr<__nv_bfloat16>(),
                                                idx_ptr,
                                                expert_weights.DevicePtr<__nv_bfloat16>(),
                                                map_ptr,
                                                off_ptr,
                                                grad_expert_buffer.DevicePtr<__nv_bfloat16>(),
                                                n_tok,
                                                n_dim,
                                                n_topk,
                                                stream);
    launch_moe_combine_backward_grad_weights<__nv_bfloat16>(
                                                grad_output.DevicePtr<__nv_bfloat16>(),
                                                expert_buffer.DevicePtr<__nv_bfloat16>(),
                                                idx_ptr,
                                                map_ptr,
                                                off_ptr,
                                                grad_weights.DevicePtr<__nv_bfloat16>(),
                                                n_tok,
                                                n_dim,
                                                n_topk,
                                                stream);
  }
  else
  {
    THROW_CAIFE("MoECombineBackwardGPU: unsupported grad_output dtype");
  }
#else
  (void)grad_output;
  (void)expert_buffer;
  (void)expert_indices;
  (void)expert_weights;
  (void)dispatch_map;
  (void)expert_offsets;
  (void)top_k;
  (void)grad_expert_buffer;
  (void)grad_weights;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MoEDispatchBackwardGPUDevice(const CAIF_DeviceTensor &grad_expert_buffer,
                            const CAIF_DeviceTensor &expert_indices,
                            const CAIF_DeviceTensor &dispatch_map,
                            const CAIF_DeviceTensor &expert_offsets,
                            uint32_t top_k,
                            CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  const auto &shape=grad_input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("MoEDispatchBackwardGPU: grad_input must be 2D [num_tokens x dim]");
  }

  const uint32_t num_tokens=shape[0];
  const uint32_t dim=shape[1];

  // Dispatch by activation dtype.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  const Dtype_e dt=grad_input.Dtype();
  const int32_t *idx_ptr=expert_indices.DevicePtr<int32_t>();
  const int32_t *map_ptr=dispatch_map.DevicePtr<int32_t>();
  const int32_t *off_ptr=expert_offsets.DevicePtr<int32_t>();
  const int n_tok=static_cast<int>(num_tokens);
  const int n_dim=static_cast<int>(dim);
  const int n_topk=static_cast<int>(top_k);
  cudaStream_t stream=grad_input.Stream().Handle();
  if(dt==Dtype_e::Float32)
  {
    launch_moe_dispatch_backward<float>(grad_expert_buffer.DevicePtr<float>(),
                                        idx_ptr,
                                        map_ptr,
                                        off_ptr,
                                        grad_input.DevicePtr<float>(),
                                        n_tok,
                                        n_dim,
                                        n_topk,
                                        stream);
  }
  else if(dt==Dtype_e::Float16)
  {
    launch_moe_dispatch_backward<__half>(grad_expert_buffer.DevicePtr<__half>(),
                                         idx_ptr,
                                         map_ptr,
                                         off_ptr,
                                         grad_input.DevicePtr<__half>(),
                                         n_tok,
                                         n_dim,
                                         n_topk,
                                         stream);
  }
  else if(dt==Dtype_e::BFloat16)
  {
    launch_moe_dispatch_backward<__nv_bfloat16>(grad_expert_buffer.DevicePtr<__nv_bfloat16>(),
                                                idx_ptr,
                                                map_ptr,
                                                off_ptr,
                                                grad_input.DevicePtr<__nv_bfloat16>(),
                                                n_tok,
                                                n_dim,
                                                n_topk,
                                                stream);
  }
  else
  {
    THROW_CAIFE("MoEDispatchBackwardGPU: unsupported grad_input dtype");
  }
#else
  (void)grad_expert_buffer;
  (void)expert_indices;
  (void)dispatch_map;
  (void)expert_offsets;
  (void)top_k;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::SgdUpdateDevice(CAIF_DeviceTensor &param,
                     const CAIF_DeviceTensor &grad,
                     const float lr,
                     const float weight_decay)
{
#ifdef USE_CAIF_CUDA
  if(param.TotalElements()!=grad.TotalElements())
  {
    THROW_CAIFE("SgdUpdate: tensor size mismatch");
  }
  const int64_t n=static_cast<int64_t>(param.TotalElements());
  if(n==0)
  {
    return;
  }
  const CAIF_DataType::CAIF_DataType_e pdt=param.Dtype();
  if(pdt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    launch_fused_sgd<float>(param.DevicePtr<float>(),
                            grad.DevicePtr<float>(),
                            lr,weight_decay,n,param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::Float16)
  {
    launch_fused_sgd<__half>(param.DevicePtr<__half>(),
                             grad.DevicePtr<__half>(),
                             lr,weight_decay,n,param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::BFloat16)
  {
    launch_fused_sgd<__nv_bfloat16>(param.DevicePtr<__nv_bfloat16>(),
                                    grad.DevicePtr<__nv_bfloat16>(),
                                    lr,weight_decay,n,param.Stream().Handle());
  }
  else
  {
    THROW_CAIFE("SgdUpdate: unsupported param dtype");
  }
#else
  (void)param;
  (void)grad;
  (void)lr;
  (void)weight_decay;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::MomentumUpdateDevice(CAIF_DeviceTensor &param,
                          const CAIF_DeviceTensor &grad,
                          CAIF_DeviceTensor &velocity,
                          const float lr,
                          const float momentum,
                          const float weight_decay)
{
#ifdef USE_CAIF_CUDA
  if(param.TotalElements()!=grad.TotalElements()||
     param.TotalElements()!=velocity.TotalElements())
  {
    THROW_CAIFE("MomentumUpdate: tensor size mismatch");
  }
  const int64_t n=static_cast<int64_t>(param.TotalElements());
  if(n==0)
  {
    return;
  }
  const CAIF_DataType::CAIF_DataType_e pdt=param.Dtype();
  if(pdt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    launch_fused_sgd_momentum<float>(param.DevicePtr<float>(),
                                     grad.DevicePtr<float>(),
                                     velocity.DevicePtr<float>(),
                                     lr,momentum,weight_decay,n,
                                     param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::Float16)
  {
    launch_fused_sgd_momentum<__half>(param.DevicePtr<__half>(),
                                      grad.DevicePtr<__half>(),
                                      velocity.DevicePtr<__half>(),
                                      lr,momentum,weight_decay,n,
                                      param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::BFloat16)
  {
    launch_fused_sgd_momentum<__nv_bfloat16>(param.DevicePtr<__nv_bfloat16>(),
                                             grad.DevicePtr<__nv_bfloat16>(),
                                             velocity.DevicePtr<__nv_bfloat16>(),
                                             lr,momentum,weight_decay,n,
                                             param.Stream().Handle());
  }
  else
  {
    THROW_CAIFE("MomentumUpdate: unsupported param dtype");
  }
#else
  (void)param;
  (void)grad;
  (void)velocity;
  (void)lr;
  (void)momentum;
  (void)weight_decay;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::RmspropUpdateDevice(CAIF_DeviceTensor &param,
                         const CAIF_DeviceTensor &grad,
                         CAIF_DeviceTensor &avg_sq,
                         const float lr,
                         const float alpha,
                         const float epsilon,
                         const float weight_decay)
{
#ifdef USE_CAIF_CUDA
  if(param.TotalElements()!=grad.TotalElements()||
     param.TotalElements()!=avg_sq.TotalElements())
  {
    THROW_CAIFE("RmspropUpdate: tensor size mismatch");
  }
  const int64_t n=static_cast<int64_t>(param.TotalElements());
  if(n==0)
  {
    return;
  }
  const CAIF_DataType::CAIF_DataType_e pdt=param.Dtype();
  if(pdt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    launch_fused_rmsprop<float>(param.DevicePtr<float>(),
                                grad.DevicePtr<float>(),
                                avg_sq.DevicePtr<float>(),
                                lr,alpha,epsilon,weight_decay,n,
                                param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::Float16)
  {
    launch_fused_rmsprop<__half>(param.DevicePtr<__half>(),
                                 grad.DevicePtr<__half>(),
                                 avg_sq.DevicePtr<__half>(),
                                 lr,alpha,epsilon,weight_decay,n,
                                 param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::BFloat16)
  {
    launch_fused_rmsprop<__nv_bfloat16>(param.DevicePtr<__nv_bfloat16>(),
                                        grad.DevicePtr<__nv_bfloat16>(),
                                        avg_sq.DevicePtr<__nv_bfloat16>(),
                                        lr,alpha,epsilon,weight_decay,n,
                                        param.Stream().Handle());
  }
  else
  {
    THROW_CAIFE("RmspropUpdate: unsupported param dtype");
  }
#else
  (void)param;
  (void)grad;
  (void)avg_sq;
  (void)lr;
  (void)alpha;
  (void)epsilon;
  (void)weight_decay;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void CAIF_Ops::AdaGradUpdateDevice(CAIF_DeviceTensor &param,
                         const CAIF_DeviceTensor &grad,
                         CAIF_DeviceTensor &accum,
                         const float lr,
                         const float epsilon,
                         const float weight_decay)
{
#ifdef USE_CAIF_CUDA
  if(param.TotalElements()!=grad.TotalElements()||
     param.TotalElements()!=accum.TotalElements())
  {
    THROW_CAIFE("AdaGradUpdate: tensor size mismatch");
  }
  const int64_t n=static_cast<int64_t>(param.TotalElements());
  if(n==0)
  {
    return;
  }
  const CAIF_DataType::CAIF_DataType_e pdt=param.Dtype();
  if(pdt==CAIF_DataType::CAIF_DataType_e::Float32)
  {
    launch_fused_adagrad<float>(param.DevicePtr<float>(),
                                grad.DevicePtr<float>(),
                                accum.DevicePtr<float>(),
                                lr,epsilon,weight_decay,n,
                                param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::Float16)
  {
    launch_fused_adagrad<__half>(param.DevicePtr<__half>(),
                                 grad.DevicePtr<__half>(),
                                 accum.DevicePtr<__half>(),
                                 lr,epsilon,weight_decay,n,
                                 param.Stream().Handle());
  }
  else if(pdt==CAIF_DataType::CAIF_DataType_e::BFloat16)
  {
    launch_fused_adagrad<__nv_bfloat16>(param.DevicePtr<__nv_bfloat16>(),
                                        grad.DevicePtr<__nv_bfloat16>(),
                                        accum.DevicePtr<__nv_bfloat16>(),
                                        lr,epsilon,weight_decay,n,
                                        param.Stream().Handle());
  }
  else
  {
    THROW_CAIFE("AdaGradUpdate: unsupported param dtype");
  }
#else
  (void)param;
  (void)grad;
  (void)accum;
  (void)lr;
  (void)epsilon;
  (void)weight_decay;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}


}//end instance namespace
