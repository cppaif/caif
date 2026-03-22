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

#include "caif_framework.h"
#include "caif_tensor.h"
#include "caif_settings.h"
#include <iostream>
#include <cmath>

using namespace instance;

int main()
{
  std::cout<<"=== MatrixMultiplyEx Verification Test ===\n";
  
  // Set CUDA backend
  CAIF_Settings::SetBackendOverride(CAIF_TensorBackend::BackendType_e::CUDA);
  
  // Test 1: Simple matmul without transpose (should match regular matmul)
  {
    CAIF_Framework framework;
    
    // A = [2, 3], B = [3, 2]
    CAIF_Tensor a(framework, {2, 3}, CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Tensor b(framework, {3, 2}, CAIF_DataType::CAIF_DataType_e::Float32);
    
    float *a_data=static_cast<float *>(a.Data());
    float *b_data=static_cast<float *>(b.Data());
    
    // A = [[1,2,3],[4,5,6]]
    a_data[0]=1; a_data[1]=2; a_data[2]=3;
    a_data[3]=4; a_data[4]=5; a_data[5]=6;
    
    // B = [[1,2],[3,4],[5,6]]  
    b_data[0]=1; b_data[1]=2;
    b_data[2]=3; b_data[3]=4;
    b_data[4]=5; b_data[5]=6;
    
    // Expected C = A * B = [[22,28],[49,64]]
    CAIF_Tensor c=framework.MatrixMultiplyEx(
                                            a,
                                            b,
                                            CAIF_TensorBackend::Transpose_e::NoTrans,
                                            CAIF_TensorBackend::Transpose_e::NoTrans
                                           );
    
    float *c_data=static_cast<float *>(c.Data());
    std::cout<<"Test 1 - No transpose:\n";
    std::cout<<"  C[0,0]="<<c_data[0]<<" (expected 22)\n";
    std::cout<<"  C[0,1]="<<c_data[1]<<" (expected 28)\n";
    std::cout<<"  C[1,0]="<<c_data[2]<<" (expected 49)\n";
    std::cout<<"  C[1,1]="<<c_data[3]<<" (expected 64)\n";
    
    bool pass=(std::abs(c_data[0]-22)<0.01f&&
               std::abs(c_data[1]-28)<0.01f&&
               std::abs(c_data[2]-49)<0.01f&&
               std::abs(c_data[3]-64)<0.01f);
    std::cout<<"  "<<(pass?"PASSED":"FAILED")<<"\n\n";
  }
  
  // Test 2: trans_a=Trans (used in weight gradient)
  {
    CAIF_Framework framework;
    
    // A = [3, 2] (will transpose to [2, 3]), B = [3, 2]
    CAIF_Tensor a(framework, {3, 2}, CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Tensor b(framework, {3, 2}, CAIF_DataType::CAIF_DataType_e::Float32);
    
    float *a_data=static_cast<float *>(a.Data());
    float *b_data=static_cast<float *>(b.Data());
    
    // A = [[1,4],[2,5],[3,6]]  -> A^T = [[1,2,3],[4,5,6]]
    a_data[0]=1; a_data[1]=4;
    a_data[2]=2; a_data[3]=5;
    a_data[4]=3; a_data[5]=6;
    
    // B = [[1,2],[3,4],[5,6]]
    b_data[0]=1; b_data[1]=2;
    b_data[2]=3; b_data[3]=4;
    b_data[4]=5; b_data[5]=6;
    
    // Expected C = A^T * B = [[22,28],[49,64]]
    CAIF_Tensor c=framework.MatrixMultiplyEx(
                                            a,
                                            b,
                                            CAIF_TensorBackend::Transpose_e::Trans,
                                            CAIF_TensorBackend::Transpose_e::NoTrans
                                           );
    
    float *c_data=static_cast<float *>(c.Data());
    std::cout<<"Test 2 - trans_a=Trans:\n";
    std::cout<<"  C[0,0]="<<c_data[0]<<" (expected 22)\n";
    std::cout<<"  C[0,1]="<<c_data[1]<<" (expected 28)\n";
    std::cout<<"  C[1,0]="<<c_data[2]<<" (expected 49)\n";
    std::cout<<"  C[1,1]="<<c_data[3]<<" (expected 64)\n";
    
    bool pass=(std::abs(c_data[0]-22)<0.01f&&
               std::abs(c_data[1]-28)<0.01f&&
               std::abs(c_data[2]-49)<0.01f&&
               std::abs(c_data[3]-64)<0.01f);
    std::cout<<"  "<<(pass?"PASSED":"FAILED")<<"\n\n";
  }
  
  // Test 3: trans_b=Trans (used in input gradient)
  {
    CAIF_Framework framework;
    
    // A = [2, 3], B = [2, 3] (will transpose to [3, 2])
    CAIF_Tensor a(framework, {2, 3}, CAIF_DataType::CAIF_DataType_e::Float32);
    CAIF_Tensor b(framework, {2, 3}, CAIF_DataType::CAIF_DataType_e::Float32);
    
    float *a_data=static_cast<float *>(a.Data());
    float *b_data=static_cast<float *>(b.Data());
    
    // A = [[1,2,3],[4,5,6]]
    a_data[0]=1; a_data[1]=2; a_data[2]=3;
    a_data[3]=4; a_data[4]=5; a_data[5]=6;
    
    // B = [[1,3,5],[2,4,6]] -> B^T = [[1,2],[3,4],[5,6]]
    b_data[0]=1; b_data[1]=3; b_data[2]=5;
    b_data[3]=2; b_data[4]=4; b_data[5]=6;
    
    // Expected C = A * B^T = [[22,28],[49,64]]
    CAIF_Tensor c=framework.MatrixMultiplyEx(
                                            a,
                                            b,
                                            CAIF_TensorBackend::Transpose_e::NoTrans,
                                            CAIF_TensorBackend::Transpose_e::Trans
                                           );
    
    float *c_data=static_cast<float *>(c.Data());
    std::cout<<"Test 3 - trans_b=Trans:\n";
    std::cout<<"  C[0,0]="<<c_data[0]<<" (expected 22)\n";
    std::cout<<"  C[0,1]="<<c_data[1]<<" (expected 28)\n";
    std::cout<<"  C[1,0]="<<c_data[2]<<" (expected 49)\n";
    std::cout<<"  C[1,1]="<<c_data[3]<<" (expected 64)\n";
    
    bool pass=(std::abs(c_data[0]-22)<0.01f&&
               std::abs(c_data[1]-28)<0.01f&&
               std::abs(c_data[2]-49)<0.01f&&
               std::abs(c_data[3]-64)<0.01f);
    std::cout<<"  "<<(pass?"PASSED":"FAILED")<<"\n\n";
  }
  
  return 0;
}

