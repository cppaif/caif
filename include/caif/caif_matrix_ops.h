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

/**
 * @file aif_matrix_ops.h
 * @brief High-performance matrix operations
 * 
 * Matrix multiplication operations with direct data access.
 * Uses BLAS/cuBLAS for optimized implementations.
 */

#ifndef CAIF_MATRIX_OPS_H
#define CAIF_MATRIX_OPS_H

#include "caif_tensor.h"
#include "caif_tensor_backend.h"
#include "caif_exception.h"
#include <cstdint>

namespace instance
{

class CAIF_Framework;

/**
 * @brief Static class for high-performance matrix operations
 */
class CAIF_MatrixOps
{
  public:
    CAIF_MatrixOps()=delete;  // Static class - no instances
    
    //--------------------------------------------------------------------------
    // Direct pointer operations (for maximum performance)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Matrix multiply C = A * B using raw pointers
     * @param a Input matrix A [m x k] row-major
     * @param b Input matrix B [k x n] row-major
     * @param c Output matrix C [m x n] row-major
     * @param m Rows of A and C
     * @param k Cols of A, Rows of B
     * @param n Cols of B and C
     */
    static void Multiply(const float *a,
                         const float *b,
                         float *c,
                         const uint32_t m,
                         const uint32_t k,
                         const uint32_t n);
    
    /**
     * @brief Matrix multiply with transpose options C = op(A) * op(B)
     * @param a Input matrix A
     * @param b Input matrix B
     * @param c Output matrix C
     * @param m Rows of op(A) and C
     * @param k Cols of op(A), Rows of op(B)
     * @param n Cols of op(B) and C
     * @param lda Leading dimension of A
     * @param ldb Leading dimension of B
     * @param ldc Leading dimension of C
     * @param trans_a Transpose A flag
     * @param trans_b Transpose B flag
     */
    static void MultiplyEx(
                           const float *a,
                           const float *b,
                           float *c,
                           const uint32_t m,
                           const uint32_t k,
                           const uint32_t n,
                           const uint32_t lda,
                           const uint32_t ldb,
                           const uint32_t ldc,
                           const bool trans_a,
                           const bool trans_b
                          );
    
    //--------------------------------------------------------------------------
    // CAIF_Tensor operations (convenience wrappers)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Matrix multiply using tensors: C = A * B
     */
    static CAIF_Tensor Multiply(const CAIF_Tensor &a,const CAIF_Tensor &b);
    
    /**
     * @brief Matrix multiply with transpose options: C = op(A) * op(B)
     */
    static CAIF_Tensor MultiplyEx(
                                 const CAIF_Tensor &a,
                                 const CAIF_Tensor &b,
                                 const CAIF_TensorBackend::Transpose_e trans_a,
                                 const CAIF_TensorBackend::Transpose_e trans_b
                                );
    
    //--------------------------------------------------------------------------
    // Backend-aware operations (use GPU if available)
    //--------------------------------------------------------------------------
    
    /**
     * @brief Matrix multiply using framework's current backend
     */
    static CAIF_Tensor Multiply(CAIF_Framework &framework,const CAIF_Tensor &a,const CAIF_Tensor &b);
    
    /**
     * @brief Matrix multiply with transpose using framework's current backend
     */
    static CAIF_Tensor MultiplyEx(
                                 CAIF_Framework &framework,
                                 const CAIF_Tensor &a,
                                 const CAIF_Tensor &b,
                                 const CAIF_TensorBackend::Transpose_e trans_a,
                                 const CAIF_TensorBackend::Transpose_e trans_b
                                );
};

}//end instance namespace

#endif  // CAIF_MATRIX_OPS_H
