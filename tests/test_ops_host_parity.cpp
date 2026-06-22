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
// Test: host-backend parity with device backend.
//
// For each op with a host implementation, build the same inputs twice — once
// as device tensors, once as host tensors — run CAIF_Ops::Foo through the
// public dispatch, then compare outputs within tolerance. Since the dispatch
// branches on tensor Location_e, this exercises both backends through the
// same surface.
//
// FP32 parity only — FP16/BF16 dtype parity is covered by
// test_device_matmul_dtype.cpp for matrix ops. On the host side the
// FP16/BF16 path up-casts to FP32 for BLAS, so FP32 parity plus device
// FP16/BF16 coverage is the correct matrix to fill.
//------------------------------------------------------------------------------
#include "caif_device_tensor.h"
#include "caif_ops.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "caif_tolerances.h"
#include "ise_lib/ise_out.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace instance
{

constexpr uint32_t g_caif_hostparity_test_binary_n=64;
constexpr uint32_t g_caif_hostparity_test_unary_n=64;
constexpr uint32_t g_caif_hostparity_test_softmax_rows=8;
constexpr uint32_t g_caif_hostparity_test_softmax_cols=32;
constexpr uint32_t g_caif_hostparity_test_biasadd_rows=16;
constexpr uint32_t g_caif_hostparity_test_biasadd_cols=24;
constexpr uint32_t g_caif_hostparity_test_matmul_m=8;
constexpr uint32_t g_caif_hostparity_test_matmul_k=12;
constexpr uint32_t g_caif_hostparity_test_matmul_n=16;
constexpr uint32_t g_caif_hostparity_test_reducesum_n=256;
constexpr float g_caif_hostparity_test_sqrt_offset=0.1f;
constexpr float g_caif_hostparity_test_denom_floor=1.0e-6f;

//------------------------------------------------------------------------------
// Abstract runner base classes for binary and unary ops.
// These avoid lambdas while keeping the parity loop generic.
//------------------------------------------------------------------------------
class CAIF_BinaryOpRunner
{
  public:
    virtual ~CAIF_BinaryOpRunner()=default;
    virtual void Invoke(const CAIF_DeviceTensor &a,
                        const CAIF_DeviceTensor &b,
                        CAIF_DeviceTensor &out)=0;

  protected:

  private:
};

class CAIF_AddOpRunner:public CAIF_BinaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &out)override
    {
      CAIF_Ops::Add(a,b,out);
    }

  protected:

  private:
};

class CAIF_MultiplyOpRunner:public CAIF_BinaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &out)override
    {
      CAIF_Ops::Multiply(a,b,out);
    }

  protected:

  private:
};

class CAIF_SubtractOpRunner:public CAIF_BinaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &out)override
    {
      CAIF_Ops::Subtract(a,b,out);
    }

  protected:

  private:
};

class CAIF_UnaryOpRunner
{
  public:
    virtual ~CAIF_UnaryOpRunner()=default;
    virtual void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)=0;

  protected:

  private:
};

class CAIF_ReLUOpRunner:public CAIF_UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::ReLU(input,output);
    }

  protected:

  private:
};

class CAIF_SigmoidOpRunner:public CAIF_UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Sigmoid(input,output);
    }

  protected:

  private:
};

class CAIF_TanhOpRunner:public CAIF_UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Tanh(input,output);
    }

  protected:

  private:
};

class CAIF_GELUOpRunner:public CAIF_UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::GELU(input,output,CAIF_GELUApproximation::CAIF_GELUApproximation_e::Tanh);
    }

  protected:

  private:
};

class CAIF_SwishOpRunner:public CAIF_UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Swish(input,output);
    }

  protected:

  private:
};

class CAIF_SqrtOpRunner:public CAIF_UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Sqrt(input,output);
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// Host/Device parity tests for CAIF_Ops.
//------------------------------------------------------------------------------
class CAIF_OpsHostParityTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeRandomVec(const size_t n,const uint32_t seed);
    static bool RelClose(const std::vector<float> &ref,
                         const std::vector<float> &got,
                         const float tol);
    static std::vector<float> ReadDeviceTensorFp32(const CAIF_DeviceTensor &t);
    static std::vector<float> ReadHostTensorFp32(const CAIF_DeviceTensor &t);
    static void WriteHostTensorFp32(CAIF_DeviceTensor &t,const std::vector<float> &data);

    static bool TestMatMulFp32();
    static bool TestBinary(CAIF_BinaryOpRunner &runner,const uint32_t seed);
    static bool TestUnary(CAIF_UnaryOpRunner &runner,
                          const uint32_t seed,
                          const bool positive_inputs);
    static bool TestSoftmax();
    static bool TestBiasAdd();
    static bool TestReduceSum();
};

std::vector<float> CAIF_OpsHostParityTests::MakeRandomVec(const size_t n,
                                                           const uint32_t seed)
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

bool CAIF_OpsHostParityTests::RelClose(const std::vector<float> &ref,
                                        const std::vector<float> &got,
                                        const float tol)
{
  if(ref.size()!=got.size())
  {
    return false;
  }
  float max_abs_ref=0.0f;
  for(const float r:ref)
  {
    const float a=std::fabs(r);
    if(a>max_abs_ref)
    {
      max_abs_ref=a;
    }
  }
  const float denom=std::max(max_abs_ref,g_caif_hostparity_test_denom_floor);
  for(size_t i=0;i<ref.size();++i)
  {
    const float rel=std::fabs(ref[i]-got[i])/denom;
    if(rel>tol)
    {
      return false;
    }
  }
  return true;
}

std::vector<float> CAIF_OpsHostParityTests::ReadDeviceTensorFp32(const CAIF_DeviceTensor &t)
{
  std::vector<float> out(t.TotalElements());
  t.CopyToHost(out.data());
  return out;
}

std::vector<float> CAIF_OpsHostParityTests::ReadHostTensorFp32(const CAIF_DeviceTensor &t)
{
  std::vector<float> out(t.TotalElements());
  std::memcpy(out.data(),t.DeviceDataRaw(),out.size()*sizeof(float));
  return out;
}

void CAIF_OpsHostParityTests::WriteHostTensorFp32(CAIF_DeviceTensor &t,
                                                   const std::vector<float> &data)
{
  std::memcpy(t.DeviceDataRaw(),data.data(),data.size()*sizeof(float));
}

//------------------------------------------------------------------------------
// MatMul FP32 parity
//------------------------------------------------------------------------------
bool CAIF_OpsHostParityTests::TestMatMulFp32()
{
  try
  {
    const std::vector<float> a_data=MakeRandomVec(g_caif_hostparity_test_matmul_m*
                                                   g_caif_hostparity_test_matmul_k,1);
    const std::vector<float> b_data=MakeRandomVec(g_caif_hostparity_test_matmul_k*
                                                   g_caif_hostparity_test_matmul_n,2);

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    CAIF_DeviceTensor a_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_matmul_m,
                                g_caif_hostparity_test_matmul_k},stream);
    CAIF_DeviceTensor b_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_matmul_k,
                                g_caif_hostparity_test_matmul_n},stream);
    CAIF_DeviceTensor c_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_matmul_m,
                                g_caif_hostparity_test_matmul_n},stream);
    a_dev.CopyFromHost(a_data.data(),a_data.size());
    b_dev.CopyFromHost(b_data.data(),b_data.size());
    CAIF_Ops::MatMul(a_dev,b_dev,c_dev,ctx);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(c_dev);

    CAIF_DeviceTensor a_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_matmul_m,
                                    g_caif_hostparity_test_matmul_k});
    CAIF_DeviceTensor b_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_matmul_k,
                                    g_caif_hostparity_test_matmul_n});
    CAIF_DeviceTensor c_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_matmul_m,
                                    g_caif_hostparity_test_matmul_n});
    WriteHostTensorFp32(a_host,a_data);
    WriteHostTensorFp32(b_host,b_data);
    CAIF_Ops::MatMul(a_host,b_host,c_host,ctx);
    const std::vector<float> host_out=ReadHostTensorFp32(c_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32MatmulCrossLoc());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

//------------------------------------------------------------------------------
// Generic element-wise parity helpers.
//------------------------------------------------------------------------------
bool CAIF_OpsHostParityTests::TestBinary(CAIF_BinaryOpRunner &runner,const uint32_t seed)
{
  try
  {
    const std::vector<float> a_data=
      MakeRandomVec(g_caif_hostparity_test_binary_n,seed);
    const std::vector<float> b_data=
      MakeRandomVec(g_caif_hostparity_test_binary_n,seed+1);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor a_dev=CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_binary_n},stream);
    CAIF_DeviceTensor b_dev=CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_binary_n},stream);
    CAIF_DeviceTensor o_dev=CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_binary_n},stream);
    a_dev.CopyFromHost(a_data.data(),g_caif_hostparity_test_binary_n);
    b_dev.CopyFromHost(b_data.data(),g_caif_hostparity_test_binary_n);
    runner.Invoke(a_dev,b_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor a_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_binary_n});
    CAIF_DeviceTensor b_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_binary_n});
    CAIF_DeviceTensor o_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_binary_n});
    WriteHostTensorFp32(a_host,a_data);
    WriteHostTensorFp32(b_host,b_data);
    runner.Invoke(a_host,b_host,o_host);
    const std::vector<float> host_out=ReadHostTensorFp32(o_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32Rel());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_OpsHostParityTests::TestUnary(CAIF_UnaryOpRunner &runner,
                                         const uint32_t seed,
                                         const bool positive_inputs)
{
  try
  {
    std::vector<float> in_data=MakeRandomVec(g_caif_hostparity_test_unary_n,seed);
    if(positive_inputs==true)
    {
      for(float &x:in_data)
      {
        x=std::fabs(x)+g_caif_hostparity_test_sqrt_offset;
      }
    }

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_unary_n},stream);
    CAIF_DeviceTensor o_dev=CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_unary_n},stream);
    i_dev.CopyFromHost(in_data.data(),g_caif_hostparity_test_unary_n);
    runner.Invoke(i_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor i_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_unary_n});
    CAIF_DeviceTensor o_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_unary_n});
    WriteHostTensorFp32(i_host,in_data);
    runner.Invoke(i_host,o_host);
    const std::vector<float> host_out=ReadHostTensorFp32(o_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32Rel());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

//------------------------------------------------------------------------------
// Per-op shape-specific parity tests.
//------------------------------------------------------------------------------
bool CAIF_OpsHostParityTests::TestSoftmax()
{
  try
  {
    const std::vector<float> in_data=
      MakeRandomVec(g_caif_hostparity_test_softmax_rows*g_caif_hostparity_test_softmax_cols,7);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_softmax_rows,
                                g_caif_hostparity_test_softmax_cols},stream);
    CAIF_DeviceTensor o_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_softmax_rows,
                                g_caif_hostparity_test_softmax_cols},stream);
    i_dev.CopyFromHost(in_data.data(),in_data.size());
    CAIF_Ops::Softmax(i_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor i_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_softmax_rows,
                                    g_caif_hostparity_test_softmax_cols});
    CAIF_DeviceTensor o_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_softmax_rows,
                                    g_caif_hostparity_test_softmax_cols});
    WriteHostTensorFp32(i_host,in_data);
    CAIF_Ops::Softmax(i_host,o_host);
    const std::vector<float> host_out=ReadHostTensorFp32(o_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32Softmax());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_OpsHostParityTests::TestBiasAdd()
{
  try
  {
    const std::vector<float> in_data=
      MakeRandomVec(g_caif_hostparity_test_biasadd_rows*g_caif_hostparity_test_biasadd_cols,11);
    const std::vector<float> bias_data=
      MakeRandomVec(g_caif_hostparity_test_biasadd_cols,12);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_biasadd_rows,
                                g_caif_hostparity_test_biasadd_cols},stream);
    CAIF_DeviceTensor b_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_biasadd_cols},stream);
    CAIF_DeviceTensor o_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_biasadd_rows,
                                g_caif_hostparity_test_biasadd_cols},stream);
    i_dev.CopyFromHost(in_data.data(),in_data.size());
    b_dev.CopyFromHost(bias_data.data(),bias_data.size());
    CAIF_Ops::BiasAdd(i_dev,b_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor i_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_biasadd_rows,
                                    g_caif_hostparity_test_biasadd_cols});
    CAIF_DeviceTensor b_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_biasadd_cols});
    CAIF_DeviceTensor o_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_biasadd_rows,
                                    g_caif_hostparity_test_biasadd_cols});
    WriteHostTensorFp32(i_host,in_data);
    WriteHostTensorFp32(b_host,bias_data);
    CAIF_Ops::BiasAdd(i_host,b_host,o_host);
    const std::vector<float> host_out=ReadHostTensorFp32(o_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32Rel());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_OpsHostParityTests::TestReduceSum()
{
  try
  {
    const std::vector<float> in_data=
      MakeRandomVec(g_caif_hostparity_test_reducesum_n,13);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=
      CAIF_DeviceTensor::Zeros({g_caif_hostparity_test_reducesum_n},stream);
    i_dev.CopyFromHost(in_data.data(),g_caif_hostparity_test_reducesum_n);
    const float dev_sum=CAIF_Ops::ReduceSum(i_dev);
    stream.Synchronize();

    CAIF_DeviceTensor i_host=
      CAIF_DeviceTensor::ZerosHost({g_caif_hostparity_test_reducesum_n});
    WriteHostTensorFp32(i_host,in_data);
    const float host_sum=CAIF_Ops::ReduceSum(i_host);

    const float denom=std::max(std::fabs(dev_sum),g_caif_hostparity_test_denom_floor);
    const float rel=std::fabs(dev_sum-host_sum)/denom;
    return rel<CAIF_Tolerances::Fp32Rel();
  }
  CAIF_CATCH_BLOCK();
  return false;
}

void CAIF_OpsHostParityTests::RunAll()
{
  ISE_Out::Out()<<"Host/Device Parity Tests\n";
  ISE_Out::Out()<<"========================\n";

  CAIF_TestHarness::Report("MatMul FP32 parity",TestMatMulFp32());

  CAIF_AddOpRunner add_runner;
  CAIF_TestHarness::Report("Add parity",TestBinary(add_runner,21));
  CAIF_MultiplyOpRunner mul_runner;
  CAIF_TestHarness::Report("Multiply parity",TestBinary(mul_runner,23));
  CAIF_SubtractOpRunner sub_runner;
  CAIF_TestHarness::Report("Subtract parity",TestBinary(sub_runner,25));

  CAIF_ReLUOpRunner relu_runner;
  CAIF_TestHarness::Report("ReLU parity",TestUnary(relu_runner,31,false));
  CAIF_SigmoidOpRunner sigmoid_runner;
  CAIF_TestHarness::Report("Sigmoid parity",TestUnary(sigmoid_runner,33,false));
  CAIF_TanhOpRunner tanh_runner;
  CAIF_TestHarness::Report("Tanh parity",TestUnary(tanh_runner,35,false));
  CAIF_GELUOpRunner gelu_runner;
  CAIF_TestHarness::Report("GELU parity",TestUnary(gelu_runner,37,false));
  CAIF_SwishOpRunner swish_runner;
  CAIF_TestHarness::Report("Swish parity",TestUnary(swish_runner,39,false));
  CAIF_SqrtOpRunner sqrt_runner;
  CAIF_TestHarness::Report("Sqrt parity",TestUnary(sqrt_runner,41,true));

  CAIF_TestHarness::Report("Softmax parity",TestSoftmax());
  CAIF_TestHarness::Report("BiasAdd parity",TestBiasAdd());
  CAIF_TestHarness::Report("ReduceSum parity",TestReduceSum());

  ISE_Out::Out()<<"\n";
  ISE_Out::Out()<<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"  Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
}

}//end instance namespace

int main()
{
  instance::CAIF_OpsHostParityTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
