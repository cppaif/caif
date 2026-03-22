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

#include "caif_device_ops.h"
#include "caif_device_context.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <cmath>

#ifdef USE_CAIF_CUDA
#include "cuda/cuda_runtime_api.h"
#include "cuda/cublas_v2.h"
#include "cudnn/cudnn.h"
#endif

namespace instance
{

namespace CAIF_DeviceOps
{

//------------------------------------------------------------------------------
// Matrix Operations
//------------------------------------------------------------------------------

void MatMul(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  // Validate dimensions
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

  // Set stream for cuBLAS
  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCublasStream(output.Stream().Handle());

  const float alpha=1.0f;
  const float beta=0.0f;

  // cuBLAS uses column-major order, so for row-major data:
  // C = A * B becomes: C^T = B^T * A^T
  // We compute: Sgemm(B, A, C) with reversed dimensions
  cublasStatus_t status=cublasSgemm(ctx.CublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    cols_b,       // m = columns of B = columns of C
                                    rows_a,       // n = rows of A = rows of C
                                    cols_a,       // k = columns of A = rows of B
                                    &alpha,
                                    b.DevicePtr(),
                                    cols_b,       // ldb
                                    a.DevicePtr(),
                                    cols_a,       // lda
                                    &beta,
                                    output.DevicePtr(),
                                    cols_b);      // ldc
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuBLAS MatMul failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void MatMulBias(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                const CAIF_DeviceTensor &bias,
                CAIF_DeviceTensor &output,
                cudaStream_t stream)
{
#ifdef USE_CAIF_CUDA
  const auto &shape_a=a.Shape();
  const auto &shape_b=b.Shape();

  const int rows_a=static_cast<int>(shape_a[0]);
  const int cols_a=static_cast<int>(shape_a[1]);
  const int cols_b=static_cast<int>(shape_b[1]);

  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  cublasLtHandle_t lt_handle=ctx.CublasLtHandle();

  // Create operation descriptor
  cublasLtMatmulDesc_t op_desc=nullptr;
  cublasLtMatmulDescCreate(&op_desc,CUBLAS_COMPUTE_32F,CUDA_R_32F);

  // Set epilogue to fuse bias add
  cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(op_desc,
                                  CUBLASLT_MATMUL_DESC_EPILOGUE,
                                  &epilogue,
                                  sizeof(epilogue));

  // Set bias pointer
  const float *bias_ptr=bias.DevicePtr();
  cublasLtMatmulDescSetAttribute(op_desc,
                                  CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                  &bias_ptr,
                                  sizeof(bias_ptr));

  // Row-major trick: C = A*B becomes C^T = B^T * A^T in column-major
  // Matrix B (N x K in col-major)
  cublasLtMatrixLayout_t b_desc=nullptr;
  cublasLtMatrixLayoutCreate(&b_desc,CUDA_R_32F,cols_b,cols_a,cols_b);

  // Matrix A (K x M in col-major)
  cublasLtMatrixLayout_t a_desc=nullptr;
  cublasLtMatrixLayoutCreate(&a_desc,CUDA_R_32F,cols_a,rows_a,cols_a);

  // Matrix C (N x M in col-major)
  cublasLtMatrixLayout_t c_desc=nullptr;
  cublasLtMatrixLayoutCreate(&c_desc,CUDA_R_32F,cols_b,rows_a,cols_b);

  const float alpha=1.0f;
  const float beta=0.0f;

  cublasStatus_t status=cublasLtMatmul(lt_handle,
                                        op_desc,
                                        &alpha,
                                        b.DevicePtr(),
                                        b_desc,
                                        a.DevicePtr(),
                                        a_desc,
                                        &beta,
                                        output.DevicePtr(),
                                        c_desc,
                                        output.DevicePtr(),
                                        c_desc,
                                        nullptr,
                                        ctx.CublasLtWorkspace(),
                                        ctx.CublasLtWorkspaceSize(),
                                        stream);

  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatmulDescDestroy(op_desc);

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
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void MatMulTransposeA(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
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

  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCublasStream(output.Stream().Handle());

  const float alpha=1.0f;
  const float beta=0.0f;

  // For row-major: C = A^T * B
  // In cuBLAS (column-major): we need to compute C^T = B^T * A
  // So: Sgemm with op_b=N, op_a=T
  cublasStatus_t status=cublasSgemm(ctx.CublasHandle(),
                                    CUBLAS_OP_N,     // op_b: B not transposed
                                    CUBLAS_OP_T,     // op_a: A transposed
                                    n,               // m = cols of B
                                    m,               // n = cols of A (rows of A^T)
                                    k,               // k = rows of A
                                    &alpha,
                                    b.DevicePtr(),
                                    b_cols,          // ldb
                                    a.DevicePtr(),
                                    a_cols,          // lda
                                    &beta,
                                    output.DevicePtr(),
                                    n);              // ldc
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuBLAS MatMulTransposeA failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void MatMulTransposeB(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
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

  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCublasStream(output.Stream().Handle());

  const float alpha=1.0f;
  const float beta=0.0f;

  // For row-major: C = A * B^T
  // In cuBLAS (column-major): we need to compute C^T = (B^T)^T * A^T = B * A^T
  // So: Sgemm with op_b=T, op_a=N
  cublasStatus_t status=cublasSgemm(ctx.CublasHandle(),
                                    CUBLAS_OP_T,     // op_b: B transposed
                                    CUBLAS_OP_N,     // op_a: A not transposed
                                    n,               // m = rows of B (cols of B^T)
                                    m,               // n = rows of A
                                    k,               // k = cols of A
                                    &alpha,
                                    b.DevicePtr(),
                                    b_cols,          // ldb
                                    a.DevicePtr(),
                                    a_cols,          // lda
                                    &beta,
                                    output.DevicePtr(),
                                    n);              // ldc
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuBLAS MatMulTransposeB failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Batched Matrix Operations
//------------------------------------------------------------------------------

void BatchedMatMul(const CAIF_DeviceTensor &a,
                   const CAIF_DeviceTensor &b,
                   CAIF_DeviceTensor &output,
                   int m,
                   int k,
                   int n,
                   int batch_count)
{
#ifdef USE_CAIF_CUDA
  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCublasStream(output.Stream().Handle());

  const float alpha=1.0f;
  const float beta=0.0f;

  // Row-major trick: C = A * B becomes C^T = B^T * A^T
  // cublasSgemmStridedBatched(handle, op_b, op_a, n, m, k, ...)
  const long long int stride_a=static_cast<long long int>(m)*k;
  const long long int stride_b=static_cast<long long int>(k)*n;
  const long long int stride_c=static_cast<long long int>(m)*n;

  cublasStatus_t status=cublasSgemmStridedBatched(ctx.CublasHandle(),
                                                   CUBLAS_OP_N,
                                                   CUBLAS_OP_N,
                                                   n,m,k,
                                                   &alpha,
                                                   b.DevicePtr(),n,stride_b,
                                                   a.DevicePtr(),k,stride_a,
                                                   &beta,
                                                   output.DevicePtr(),n,stride_c,
                                                   batch_count);
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuBLAS BatchedMatMul failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
  (void)batch_count;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void BatchedMatMulTransposeA(const CAIF_DeviceTensor &a,
                             const CAIF_DeviceTensor &b,
                             CAIF_DeviceTensor &output,
                             int k,
                             int m,
                             int n,
                             int batch_count)
{
#ifdef USE_CAIF_CUDA
  // C = A^T * B where A is [K x M], B is [K x N], C is [M x N]
  // Row-major trick: C^T = B^T * A
  // op_b = N (B not transposed), op_a = T (A transposed)
  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCublasStream(output.Stream().Handle());

  const float alpha=1.0f;
  const float beta=0.0f;

  const long long int stride_a=static_cast<long long int>(k)*m;
  const long long int stride_b=static_cast<long long int>(k)*n;
  const long long int stride_c=static_cast<long long int>(m)*n;

  cublasStatus_t status=cublasSgemmStridedBatched(ctx.CublasHandle(),
                                                   CUBLAS_OP_N,
                                                   CUBLAS_OP_T,
                                                   n,m,k,
                                                   &alpha,
                                                   b.DevicePtr(),n,stride_b,
                                                   a.DevicePtr(),m,stride_a,
                                                   &beta,
                                                   output.DevicePtr(),n,stride_c,
                                                   batch_count);
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuBLAS BatchedMatMulTransposeA failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)k;
  (void)m;
  (void)n;
  (void)batch_count;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void BatchedMatMulTransposeB(const CAIF_DeviceTensor &a,
                             const CAIF_DeviceTensor &b,
                             CAIF_DeviceTensor &output,
                             int m,
                             int k,
                             int n,
                             int batch_count)
{
#ifdef USE_CAIF_CUDA
  // C = A * B^T where A is [M x K], B is [N x K], C is [M x N]
  // Row-major trick: C^T = (B^T)^T * A^T = B * A^T
  // op_b = T (B transposed), op_a = N (A not transposed)
  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCublasStream(output.Stream().Handle());

  const float alpha=1.0f;
  const float beta=0.0f;

  const long long int stride_a=static_cast<long long int>(m)*k;
  const long long int stride_b=static_cast<long long int>(n)*k;
  const long long int stride_c=static_cast<long long int>(m)*n;

  cublasStatus_t status=cublasSgemmStridedBatched(ctx.CublasHandle(),
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   n,m,k,
                                                   &alpha,
                                                   b.DevicePtr(),k,stride_b,
                                                   a.DevicePtr(),k,stride_a,
                                                   &beta,
                                                   output.DevicePtr(),n,stride_c,
                                                   batch_count);
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuBLAS BatchedMatMulTransposeB failed");
  }
#else
  (void)a;
  (void)b;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
  (void)batch_count;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Tensor Manipulation
//------------------------------------------------------------------------------

void Transpose(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
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

  // Use cuBLAS geam for transpose: C = alpha*A^T + beta*B
  // With beta=0 and alpha=1, this is just C = A^T
  cublasHandle_t handle=CAIF_DeviceContext::Instance().CublasHandle();
  cublasSetStream(handle,output.Stream().Handle());

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
                                     input.DevicePtr(),
                                     cols,              // lda = cols of input
                                     &beta,
                                     nullptr,           // B not used
                                     rows,              // ldb (not used but must be valid)
                                     output.DevicePtr(),
                                     rows);             // ldc = rows of output (= cols of input)

  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("Transpose: cublasSgeam failed");
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

void Add(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(a.TotalElements()!=b.TotalElements()||a.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Add: tensor size mismatch");
  }

  const int n=static_cast<int>(output.TotalElements());
  launch_elementwise_add(a.DevicePtr(),
                         b.DevicePtr(),
                         output.DevicePtr(),
                         n,
                         output.Stream().Handle());
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void Scale(CAIF_DeviceTensor &tensor,float scale)
{
#ifdef USE_CAIF_CUDA
  const int n=static_cast<int>(tensor.TotalElements());
  if(n==0)
  {
    return;
  }

  // Use in-place scaling: result = tensor * scale
  launch_elementwise_mul_scalar(tensor.DevicePtr(),
                                scale,
                                tensor.DevicePtr(),
                                n,
                                tensor.Stream().Handle());
#else
  (void)tensor;
  (void)scale;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void AddScaled(CAIF_DeviceTensor &target,const CAIF_DeviceTensor &source,float scale)
{
#ifdef USE_CAIF_CUDA
  if(target.TotalElements()!=source.TotalElements())
  {
    THROW_CAIFE("AddScaled: tensor size mismatch");
  }

  const int n=static_cast<int>(target.TotalElements());
  if(n==0)
  {
    return;
  }

  // Use cuBLAS SAXPY: target = target + scale * source
  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCublasStream(target.Stream().Handle());

  cublasStatus_t status=cublasSaxpy(ctx.CublasHandle(),
                                    n,
                                    &scale,
                                    source.DevicePtr(),
                                    1,
                                    target.DevicePtr(),
                                    1);
  if(status!=CUBLAS_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuBLAS AddScaled (saxpy) failed");
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

void BiasAdd(const CAIF_DeviceTensor &input,const CAIF_DeviceTensor &bias,CAIF_DeviceTensor &output)
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

  const int batch_size=static_cast<int>(shape[0]);
  const int units=static_cast<int>(shape[1]);

  launch_bias_add_2d(input.DevicePtr(),
                     bias.DevicePtr(),
                     output.DevicePtr(),
                     batch_size,
                     units,
                     output.Stream().Handle());
#else
  (void)input;
  (void)bias;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void BiasGradient(const CAIF_DeviceTensor &grad,CAIF_DeviceTensor &bias_grad)
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

  const int batch_size=static_cast<int>(shape[0]);
  const int units=static_cast<int>(shape[1]);

  launch_bias_grad_2d(grad.DevicePtr(),
                      bias_grad.DevicePtr(),
                      batch_size,
                      units,
                      bias_grad.Stream().Handle());
#else
  (void)grad;
  (void)bias_grad;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Activation Functions (Forward) - vectorized CUDA kernels
//------------------------------------------------------------------------------

void ReLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("ReLU: input/output size mismatch");
  }
  const int n=static_cast<int>(input.TotalElements());
  launch_relu_forward(input.DevicePtr(),
                      output.DevicePtr(),
                      n,
                      output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void Sigmoid(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Sigmoid: input/output size mismatch");
  }
  const int n=static_cast<int>(input.TotalElements());
  launch_sigmoid_forward(input.DevicePtr(),
                         output.DevicePtr(),
                         n,
                         output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void Tanh(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Tanh: input/output size mismatch");
  }
  const int n=static_cast<int>(input.TotalElements());
  launch_tanh_forward(input.DevicePtr(),
                      output.DevicePtr(),
                      n,
                      output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void Softmax(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
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

  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCudnnStream(output.Stream().Handle());

  const int batch=static_cast<int>(shape[0]);
  const int classes=static_cast<int>(shape[1]);

  cudnnTensorDescriptor_t tensor_desc=nullptr;
  cudnnStatus_t status;

  status=cudnnCreateTensorDescriptor(&tensor_desc);
  if(status!=CUDNN_STATUS_SUCCESS)
  {
    THROW_CAIFE("Failed to create softmax tensor descriptor");
  }

  // Set as 4D tensor [N,C,H,W] = [batch,classes,1,1]
  status=cudnnSetTensor4dDescriptor(tensor_desc,
                                    CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT,
                                    batch,
                                    classes,
                                    1,
                                    1);
  if(status!=CUDNN_STATUS_SUCCESS)
  {
    cudnnDestroyTensorDescriptor(tensor_desc);
    THROW_CAIFE("Failed to set softmax tensor descriptor");
  }

  const float alpha=1.0f;
  const float beta_val=0.0f;

  // Apply softmax over channel dimension (CUDNN_SOFTMAX_MODE_CHANNEL)
  status=cudnnSoftmaxForward(ctx.CudnnHandle(),
                             CUDNN_SOFTMAX_ACCURATE,
                             CUDNN_SOFTMAX_MODE_CHANNEL,
                             &alpha,
                             tensor_desc,
                             input.DevicePtr(),
                             &beta_val,
                             tensor_desc,
                             output.DevicePtr());

  cudnnDestroyTensorDescriptor(tensor_desc);

  if(status!=CUDNN_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuDNN softmax forward failed");
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

void ReLUBackward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=input.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("ReLUBackward: tensor size mismatch");
  }

  const int n=static_cast<int>(input.TotalElements());
  launch_relu_backward(grad_output.DevicePtr(),
                       input.DevicePtr(),
                       grad_input.DevicePtr(),
                       n,
                       grad_input.Stream().Handle());
#else
  (void)grad_output;
  (void)input;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void SigmoidBackward(const CAIF_DeviceTensor &grad_output,
                     const CAIF_DeviceTensor &output,
                     CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=output.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("SigmoidBackward: tensor size mismatch");
  }

  const int n=static_cast<int>(output.TotalElements());
  launch_sigmoid_backward(grad_output.DevicePtr(),
                          output.DevicePtr(),
                          grad_input.DevicePtr(),
                          n,
                          grad_input.Stream().Handle());
#else
  (void)grad_output;
  (void)output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void TanhBackward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &output,
                  CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=output.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("TanhBackward: tensor size mismatch");
  }

  const int n=static_cast<int>(output.TotalElements());
  launch_tanh_backward(grad_output.DevicePtr(),
                       output.DevicePtr(),
                       grad_input.DevicePtr(),
                       n,
                       grad_input.Stream().Handle());
#else
  (void)grad_output;
  (void)output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void SoftmaxBackward(const CAIF_DeviceTensor &grad_output,
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

  CAIF_DeviceContext &ctx=CAIF_DeviceContext::Instance();
  ctx.SetCudnnStream(grad_input.Stream().Handle());

  const int batch=static_cast<int>(shape[0]);
  const int classes=static_cast<int>(shape[1]);

  cudnnTensorDescriptor_t tensor_desc=nullptr;
  cudnnStatus_t status;

  status=cudnnCreateTensorDescriptor(&tensor_desc);
  if(status!=CUDNN_STATUS_SUCCESS)
  {
    THROW_CAIFE("Failed to create softmax backward tensor descriptor");
  }

  // Set as 4D tensor [N,C,H,W] = [batch,classes,1,1]
  status=cudnnSetTensor4dDescriptor(tensor_desc,
                                    CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT,
                                    batch,
                                    classes,
                                    1,
                                    1);
  if(status!=CUDNN_STATUS_SUCCESS)
  {
    cudnnDestroyTensorDescriptor(tensor_desc);
    THROW_CAIFE("Failed to set softmax backward tensor descriptor");
  }

  const float alpha=1.0f;
  const float beta_val=0.0f;

  status=cudnnSoftmaxBackward(ctx.CudnnHandle(),
                              CUDNN_SOFTMAX_ACCURATE,
                              CUDNN_SOFTMAX_MODE_CHANNEL,
                              &alpha,
                              tensor_desc,
                              output.DevicePtr(),       // y
                              tensor_desc,
                              grad_output.DevicePtr(),  // dy
                              &beta_val,
                              tensor_desc,
                              grad_input.DevicePtr());  // dx

  cudnnDestroyTensorDescriptor(tensor_desc);

  if(status!=CUDNN_STATUS_SUCCESS)
  {
    THROW_CAIFE("cuDNN softmax backward failed");
  }
#else
  (void)grad_output;
  (void)output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void LeakyReLU(const CAIF_DeviceTensor &input,
               CAIF_DeviceTensor &output,
               float alpha)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("LeakyReLU: input/output size mismatch");
  }

  const int n=static_cast<int>(input.TotalElements());
  launch_leaky_relu_forward(input.DevicePtr(),
                            output.DevicePtr(),
                            alpha,
                            n,
                            output.Stream().Handle());
#else
  (void)input;
  (void)output;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void ELU(const CAIF_DeviceTensor &input,
         CAIF_DeviceTensor &output,
         float alpha)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("ELU: input/output size mismatch");
  }

  const int n=static_cast<int>(input.TotalElements());
  launch_elu_forward(input.DevicePtr(),
                     output.DevicePtr(),
                     alpha,
                     n,
                     output.Stream().Handle());
#else
  (void)input;
  (void)output;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void GELU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("GELU: input/output size mismatch");
  }

  const int n=static_cast<int>(input.TotalElements());
  launch_gelu_forward(input.DevicePtr(),
                      output.DevicePtr(),
                      n,
                      output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void Swish(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Swish: input/output size mismatch");
  }

  const int n=static_cast<int>(input.TotalElements());
  launch_swish_forward(input.DevicePtr(),
                       output.DevicePtr(),
                       n,
                       output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void LeakyReLUBackward(const CAIF_DeviceTensor &grad_output,
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

  const int n=static_cast<int>(input.TotalElements());
  launch_leaky_relu_backward(grad_output.DevicePtr(),
                             input.DevicePtr(),
                             grad_input.DevicePtr(),
                             alpha,
                             n,
                             grad_input.Stream().Handle());
#else
  (void)grad_output;
  (void)input;
  (void)grad_input;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void ELUBackward(const CAIF_DeviceTensor &grad_output,
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

  const int n=static_cast<int>(input.TotalElements());
  launch_elu_backward(grad_output.DevicePtr(),
                      input.DevicePtr(),
                      output.DevicePtr(),
                      grad_input.DevicePtr(),
                      alpha,
                      n,
                      grad_input.Stream().Handle());
#else
  (void)grad_output;
  (void)input;
  (void)output;
  (void)grad_input;
  (void)alpha;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void GELUBackward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &input,
                  CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(grad_output.TotalElements()!=input.TotalElements()||
     grad_output.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("GELUBackward: tensor size mismatch");
  }

  const int n=static_cast<int>(input.TotalElements());
  launch_gelu_backward(grad_output.DevicePtr(),
                       input.DevicePtr(),
                       grad_input.DevicePtr(),
                       n,
                       grad_input.Stream().Handle());
#else
  (void)grad_output;
  (void)input;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void SwishBackward(const CAIF_DeviceTensor &grad_output,
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

  const int n=static_cast<int>(input.TotalElements());
  launch_swish_backward(grad_output.DevicePtr(),
                        input.DevicePtr(),
                        output.DevicePtr(),
                        grad_input.DevicePtr(),
                        n,
                        grad_input.Stream().Handle());
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

float ReduceSum(const CAIF_DeviceTensor &tensor)
{
#ifdef USE_CAIF_CUDA
  const int n=static_cast<int>(tensor.TotalElements());
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

  launch_reduction_sum(tensor.DevicePtr(),d_result,n,tensor.Stream().Handle());

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

float ReduceMean(const CAIF_DeviceTensor &tensor)
{
#ifdef USE_CAIF_CUDA
  const size_t n=tensor.TotalElements();
  if(n==0)
  {
    return 0.0f;
  }

  const float sum=ReduceSum(tensor);
  return sum/static_cast<float>(n);
#else
  (void)tensor;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Loss Functions
//------------------------------------------------------------------------------

void MSELoss(const CAIF_DeviceTensor &pred,const CAIF_DeviceTensor &target,CAIF_DeviceTensor &loss)
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

  const int n=static_cast<int>(pred.TotalElements());

  launch_mse_loss(pred.DevicePtr(),
                  target.DevicePtr(),
                  loss.DevicePtr(),
                  n,
                  loss.Stream().Handle());
#else
  (void)pred;
  (void)target;
  (void)loss;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void MSELossBackward(const CAIF_DeviceTensor &pred,
                     const CAIF_DeviceTensor &target,
                     CAIF_DeviceTensor &grad)
{
#ifdef USE_CAIF_CUDA
  if(pred.TotalElements()!=target.TotalElements()||
     pred.TotalElements()!=grad.TotalElements())
  {
    THROW_CAIFE("MSELossBackward: tensor size mismatch");
  }

  const int n=static_cast<int>(pred.TotalElements());

  launch_mse_gradient(pred.DevicePtr(),
                      target.DevicePtr(),
                      grad.DevicePtr(),
                      n,
                      grad.Stream().Handle());
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

void AdamUpdate(CAIF_DeviceTensor &param,
                const CAIF_DeviceTensor &grad,
                CAIF_DeviceTensor &m,
                CAIF_DeviceTensor &v,
                float lr,
                float beta1,
                float beta2,
                float epsilon,
                int t)
{
#ifdef USE_CAIF_CUDA
  if(param.TotalElements()!=grad.TotalElements()||
     param.TotalElements()!=m.TotalElements()||
     param.TotalElements()!=v.TotalElements())
  {
    THROW_CAIFE("AdamUpdate: tensor size mismatch");
  }

  const int n=static_cast<int>(param.TotalElements());
  if(n==0)
  {
    return;
  }

  // Compute bias correction factors
  const float bias_correction1=1.0f-std::pow(beta1,static_cast<float>(t));
  const float bias_correction2=1.0f-std::pow(beta2,static_cast<float>(t));

  // No weight decay for this simplified version
  const float weight_decay=0.0f;

  launch_fused_adam(param.DevicePtr(),
                    grad.DevicePtr(),
                    m.DevicePtr(),
                    v.DevicePtr(),
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                    bias_correction1,
                    bias_correction2,
                    n,
                    param.Stream().Handle());
#else
  (void)param;
  (void)grad;
  (void)m;
  (void)v;
  (void)lr;
  (void)beta1;
  (void)beta2;
  (void)epsilon;
  (void)t;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Additional Element-wise Operations
//------------------------------------------------------------------------------

void Multiply(const CAIF_DeviceTensor &a,const CAIF_DeviceTensor &b,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(a.TotalElements()!=b.TotalElements()||a.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Multiply: tensor size mismatch");
  }

  const int n=static_cast<int>(output.TotalElements());
  launch_elementwise_mul(a.DevicePtr(),
                         b.DevicePtr(),
                         output.DevicePtr(),
                         n,
                         output.Stream().Handle());
#else
  (void)a;
  (void)b;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void Scale(const CAIF_DeviceTensor &input,float scale,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=output.TotalElements())
  {
    THROW_CAIFE("Scale: tensor size mismatch");
  }

  const int n=static_cast<int>(output.TotalElements());
  if(n==0)
  {
    return;
  }

  launch_elementwise_mul_scalar(input.DevicePtr(),
                                scale,
                                output.DevicePtr(),
                                n,
                                output.Stream().Handle());
#else
  (void)input;
  (void)scale;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void SiLU(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  // SiLU is the same as Swish: x * sigmoid(x)
  Swish(input,output);
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void SiLUBackward(const CAIF_DeviceTensor &input,
                  const CAIF_DeviceTensor &grad_output,
                  CAIF_DeviceTensor &grad_input)
{
#ifdef USE_CAIF_CUDA
  if(input.TotalElements()!=grad_output.TotalElements()||
     input.TotalElements()!=grad_input.TotalElements())
  {
    THROW_CAIFE("SiLUBackward: tensor size mismatch");
  }

  const int n=static_cast<int>(input.TotalElements());

  // SiLU backward: grad_input = grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
  //              = grad_output * (sigmoid(x) * (1 + x * (1 - sigmoid(x))))
  launch_silu_backward(grad_output.DevicePtr(),
                       input.DevicePtr(),
                       grad_input.DevicePtr(),
                       n,
                       grad_input.Stream().Handle());
#else
  (void)input;
  (void)grad_output;
  (void)grad_input;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void AddBias(const CAIF_DeviceTensor &input,const CAIF_DeviceTensor &bias,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  BiasAdd(input,bias,output);
#else
  (void)input;
  (void)bias;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Reduction Operations (tensor output)
//------------------------------------------------------------------------------

void SumAxis(const CAIF_DeviceTensor &input,uint32_t axis,CAIF_DeviceTensor &output)
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

  if(axis==0)
  {
    // Sum over batch dimension: output is [dim]
    if(output.TotalElements()!=static_cast<size_t>(dim))
    {
      THROW_CAIFE("SumAxis: output size mismatch for axis=0");
    }
    launch_sum_axis0(input.DevicePtr(),
                     output.DevicePtr(),
                     batch,
                     dim,
                     output.Stream().Handle());
  }
  else
  {
    // Sum over dim dimension: output is [batch]
    if(output.TotalElements()!=static_cast<size_t>(batch))
    {
      THROW_CAIFE("SumAxis: output size mismatch for axis=1");
    }
    launch_sum_axis1(input.DevicePtr(),
                     output.DevicePtr(),
                     batch,
                     dim,
                     output.Stream().Handle());
  }
#else
  (void)input;
  (void)axis;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void Sum(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
{
#ifdef USE_CAIF_CUDA
  if(output.TotalElements()!=1)
  {
    THROW_CAIFE("Sum: output must have exactly 1 element");
  }

  const int n=static_cast<int>(input.TotalElements());
  launch_reduction_sum(input.DevicePtr(),
                       output.DevicePtr(),
                       n,
                       output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void LogSumExp(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
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

  launch_logsumexp(input.DevicePtr(),
                   output.DevicePtr(),
                   batch,
                   dim,
                   output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

//------------------------------------------------------------------------------
// Top-K and Scatter Operations
//------------------------------------------------------------------------------

void TopK(const CAIF_DeviceTensor &input,
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

  launch_topk(input.DevicePtr(),
              reinterpret_cast<int32_t*>(indices.DevicePtr()),
              values.DevicePtr(),
              static_cast<int>(batch),
              static_cast<int>(dim),
              static_cast<int>(k),
              values.Stream().Handle());
#else
  (void)input;
  (void)k;
  (void)indices;
  (void)values;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void NormalizeRows(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)
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

  launch_normalize_rows(input.DevicePtr(),
                        output.DevicePtr(),
                        batch,
                        dim,
                        output.Stream().Handle());
#else
  (void)input;
  (void)output;
  THROW_CAIFE("CUDA support not built (USE_CAIF_CUDA not defined)");
#endif
}

void ScatterAdd(const CAIF_DeviceTensor &values,
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

  launch_scatter_add(values.DevicePtr(),
                     reinterpret_cast<const int32_t*>(indices.DevicePtr()),
                     output.DevicePtr(),
                     batch,
                     k,
                     dim,
                     output.Stream().Handle());
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

void MoEDispatch(const CAIF_DeviceTensor &input,
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
  std::vector<float> indices_float(num_tokens*top_k);
  expert_indices.CopyToHost(indices_float.data());

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
      const int32_t expert_idx=static_cast<int32_t>(indices_float[t*top_k+k]);
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

void MoECombine(const std::vector<CAIF_DeviceTensor> &expert_outputs,
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

  // Copy indices and weights to host (indices stored as floats)
  std::vector<float> indices_float(num_tokens*top_k);
  std::vector<float> weights_host(num_tokens*top_k);
  expert_indices.CopyToHost(indices_float.data());
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
      const int32_t expert_idx=static_cast<int32_t>(indices_float[t*top_k+k]);
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

void MoECombineBackward(const CAIF_DeviceTensor &grad_output,
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
  std::vector<float> indices_float(num_tokens*top_k);
  std::vector<float> weights_host(num_tokens*top_k);

  grad_output.CopyToHost(grad_out_host.data());
  expert_indices.CopyToHost(indices_float.data());
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
      const int32_t expert_idx=static_cast<int32_t>(indices_float[t*top_k+k]);
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

void MoEDispatchBackward(const std::vector<CAIF_DeviceTensor> &grad_expert_inputs,
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
  std::vector<float> indices_float(num_tokens*top_k);
  expert_indices.CopyToHost(indices_float.data());

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
      const int32_t expert_idx=static_cast<int32_t>(indices_float[t*top_k+k]);

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

void MoETopKGating(const CAIF_DeviceTensor &router_logits,
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

  launch_moe_topk_gating(router_logits.DevicePtr(),
                         expert_indices.DevicePtr(),
                         expert_weights.DevicePtr(),
                         router_probs.DevicePtr(),
                         static_cast<int>(num_tokens),
                         static_cast<int>(num_experts),
                         static_cast<int>(top_k),
                         expert_indices.Stream().Handle());
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

void MoECountPerExpert(const CAIF_DeviceTensor &expert_indices,
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
  cudaMemsetAsync(expert_counts.DevicePtr(),0,
                  num_experts*sizeof(int32_t),
                  expert_counts.Stream().Handle());

  launch_moe_count_per_expert(expert_indices.DevicePtr(),
                              reinterpret_cast<int*>(expert_counts.DevicePtr()),
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

void MoEDispatchGPU(const CAIF_DeviceTensor &input,
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

  launch_moe_dispatch(input.DevicePtr(),
                      expert_indices.DevicePtr(),
                      reinterpret_cast<const int*>(dispatch_map.DevicePtr()),
                      expert_buffer.DevicePtr(),
                      reinterpret_cast<const int*>(expert_offsets.DevicePtr()),
                      static_cast<int>(num_tokens),
                      static_cast<int>(dim),
                      static_cast<int>(top_k),
                      expert_buffer.Stream().Handle());
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

void MoECombineGPU(const CAIF_DeviceTensor &expert_buffer,
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

  launch_moe_combine(expert_buffer.DevicePtr(),
                     expert_indices.DevicePtr(),
                     expert_weights.DevicePtr(),
                     reinterpret_cast<const int*>(dispatch_map.DevicePtr()),
                     reinterpret_cast<const int*>(expert_offsets.DevicePtr()),
                     output.DevicePtr(),
                     static_cast<int>(num_tokens),
                     static_cast<int>(dim),
                     static_cast<int>(top_k),
                     output.Stream().Handle());
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

uint32_t MoEBuildDispatchMap(const CAIF_DeviceTensor &expert_indices,
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
  std::vector<float> indices_float(num_tokens*top_k);
  expert_indices.CopyToHost(indices_float.data());

  // Count tokens per expert
  std::vector<uint32_t> counts(num_experts,0);
  for(uint32_t t=0;t<num_tokens;++t)
  {
    for(uint32_t k=0;k<top_k;++k)
    {
      const int32_t expert_idx=static_cast<int32_t>(indices_float[t*top_k+k]);
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
      const int32_t expert_idx=static_cast<int32_t>(indices_float[t*top_k+k]);
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
  dispatch_map.CopyFromHost(reinterpret_cast<const float*>(map_host.data()),
                            map_host.size());
  expert_offsets.CopyFromHost(reinterpret_cast<const float*>(offsets.data()),
                              offsets.size());

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

}//end CAIF_DeviceOps namespace

}//end instance namespace
