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

using namespace instance;

namespace
{

void ReportResult(const char *name,bool ok)
{
  CAIF_TestHarness::Report(name,ok);
}


std::vector<float> MakeRandomVec(const size_t n,const uint32_t seed)
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

bool RelClose(const std::vector<float> &ref,
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
  const float denom=std::max(max_abs_ref,1.0e-6f);
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

std::vector<float> ReadDeviceTensorFp32(const CAIF_DeviceTensor &t)
{
  std::vector<float> out(t.TotalElements());
  t.CopyToHost(out.data());
  return out;
}

std::vector<float> ReadHostTensorFp32(const CAIF_DeviceTensor &t)
{
  std::vector<float> out(t.TotalElements());
  std::memcpy(out.data(),t.DeviceDataRaw(),out.size()*sizeof(float));
  return out;
}

void WriteHostTensorFp32(CAIF_DeviceTensor &t,const std::vector<float> &data)
{
  std::memcpy(t.DeviceDataRaw(),data.data(),data.size()*sizeof(float));
}

//------------------------------------------------------------------------------
// MatMul FP32 parity
//------------------------------------------------------------------------------

bool TestMatMulFp32()
{
  try
  {
    const uint32_t m=8,k=12,n=16;
    const std::vector<float> a_data=MakeRandomVec(m*k,1);
    const std::vector<float> b_data=MakeRandomVec(k*n,2);

    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    CAIF_DeviceTensor a_dev=CAIF_DeviceTensor::Zeros({m,k},stream);
    CAIF_DeviceTensor b_dev=CAIF_DeviceTensor::Zeros({k,n},stream);
    CAIF_DeviceTensor c_dev=CAIF_DeviceTensor::Zeros({m,n},stream);
    a_dev.CopyFromHost(a_data.data(),a_data.size());
    b_dev.CopyFromHost(b_data.data(),b_data.size());
    CAIF_Ops::MatMul(a_dev,b_dev,c_dev,ctx);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(c_dev);

    CAIF_DeviceTensor a_host=CAIF_DeviceTensor::ZerosHost({m,k});
    CAIF_DeviceTensor b_host=CAIF_DeviceTensor::ZerosHost({k,n});
    CAIF_DeviceTensor c_host=CAIF_DeviceTensor::ZerosHost({m,n});
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
// Generic element-wise parity helpers via functors (no lambdas).
//------------------------------------------------------------------------------

class BinaryOpRunner
{
  public:
    virtual ~BinaryOpRunner()=default;
    virtual void Invoke(const CAIF_DeviceTensor &a,
                        const CAIF_DeviceTensor &b,
                        CAIF_DeviceTensor &out)=0;
};

class AddOpRunner:public BinaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &out)override
    {
      CAIF_Ops::Add(a,b,out);
    }
};

class MultiplyOpRunner:public BinaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &out)override
    {
      CAIF_Ops::Multiply(a,b,out);
    }
};

class SubtractOpRunner:public BinaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &a,
                const CAIF_DeviceTensor &b,
                CAIF_DeviceTensor &out)override
    {
      CAIF_Ops::Subtract(a,b,out);
    }
};

bool TestBinary(BinaryOpRunner &runner,const uint32_t seed)
{
  try
  {
    const uint32_t n=64;
    const std::vector<float> a_data=MakeRandomVec(n,seed);
    const std::vector<float> b_data=MakeRandomVec(n,seed+1);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor a_dev=CAIF_DeviceTensor::Zeros({n},stream);
    CAIF_DeviceTensor b_dev=CAIF_DeviceTensor::Zeros({n},stream);
    CAIF_DeviceTensor o_dev=CAIF_DeviceTensor::Zeros({n},stream);
    a_dev.CopyFromHost(a_data.data(),n);
    b_dev.CopyFromHost(b_data.data(),n);
    runner.Invoke(a_dev,b_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor a_host=CAIF_DeviceTensor::ZerosHost({n});
    CAIF_DeviceTensor b_host=CAIF_DeviceTensor::ZerosHost({n});
    CAIF_DeviceTensor o_host=CAIF_DeviceTensor::ZerosHost({n});
    WriteHostTensorFp32(a_host,a_data);
    WriteHostTensorFp32(b_host,b_data);
    runner.Invoke(a_host,b_host,o_host);
    const std::vector<float> host_out=ReadHostTensorFp32(o_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32Rel());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

class UnaryOpRunner
{
  public:
    virtual ~UnaryOpRunner()=default;
    virtual void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)=0;
};

class ReLUOpRunner:public UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::ReLU(input,output);
    }
};

class SigmoidOpRunner:public UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Sigmoid(input,output);
    }
};

class TanhOpRunner:public UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Tanh(input,output);
    }
};

class GELUOpRunner:public UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::GELU(input,output);
    }
};

class SwishOpRunner:public UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Swish(input,output);
    }
};

class SqrtOpRunner:public UnaryOpRunner
{
  public:
    void Invoke(const CAIF_DeviceTensor &input,CAIF_DeviceTensor &output)override
    {
      CAIF_Ops::Sqrt(input,output);
    }
};

bool TestUnary(UnaryOpRunner &runner,const uint32_t seed,const bool positive_inputs)
{
  try
  {
    const uint32_t n=64;
    std::vector<float> in_data=MakeRandomVec(n,seed);
    if(positive_inputs==true)
    {
      for(float &x:in_data)
      {
        x=std::fabs(x)+0.1f;
      }
    }

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=CAIF_DeviceTensor::Zeros({n},stream);
    CAIF_DeviceTensor o_dev=CAIF_DeviceTensor::Zeros({n},stream);
    i_dev.CopyFromHost(in_data.data(),n);
    runner.Invoke(i_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor i_host=CAIF_DeviceTensor::ZerosHost({n});
    CAIF_DeviceTensor o_host=CAIF_DeviceTensor::ZerosHost({n});
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

bool TestSoftmax()
{
  try
  {
    const uint32_t rows=8;
    const uint32_t cols=32;
    const std::vector<float> in_data=MakeRandomVec(rows*cols,7);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=CAIF_DeviceTensor::Zeros({rows,cols},stream);
    CAIF_DeviceTensor o_dev=CAIF_DeviceTensor::Zeros({rows,cols},stream);
    i_dev.CopyFromHost(in_data.data(),in_data.size());
    CAIF_Ops::Softmax(i_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor i_host=CAIF_DeviceTensor::ZerosHost({rows,cols});
    CAIF_DeviceTensor o_host=CAIF_DeviceTensor::ZerosHost({rows,cols});
    WriteHostTensorFp32(i_host,in_data);
    CAIF_Ops::Softmax(i_host,o_host);
    const std::vector<float> host_out=ReadHostTensorFp32(o_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32Softmax());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool TestBiasAdd()
{
  try
  {
    const uint32_t rows=16;
    const uint32_t cols=24;
    const std::vector<float> in_data=MakeRandomVec(rows*cols,11);
    const std::vector<float> bias_data=MakeRandomVec(cols,12);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=CAIF_DeviceTensor::Zeros({rows,cols},stream);
    CAIF_DeviceTensor b_dev=CAIF_DeviceTensor::Zeros({cols},stream);
    CAIF_DeviceTensor o_dev=CAIF_DeviceTensor::Zeros({rows,cols},stream);
    i_dev.CopyFromHost(in_data.data(),in_data.size());
    b_dev.CopyFromHost(bias_data.data(),bias_data.size());
    CAIF_Ops::BiasAdd(i_dev,b_dev,o_dev);
    stream.Synchronize();
    const std::vector<float> dev_out=ReadDeviceTensorFp32(o_dev);

    CAIF_DeviceTensor i_host=CAIF_DeviceTensor::ZerosHost({rows,cols});
    CAIF_DeviceTensor b_host=CAIF_DeviceTensor::ZerosHost({cols});
    CAIF_DeviceTensor o_host=CAIF_DeviceTensor::ZerosHost({rows,cols});
    WriteHostTensorFp32(i_host,in_data);
    WriteHostTensorFp32(b_host,bias_data);
    CAIF_Ops::BiasAdd(i_host,b_host,o_host);
    const std::vector<float> host_out=ReadHostTensorFp32(o_host);

    return RelClose(dev_out,host_out,CAIF_Tolerances::Fp32Rel());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool TestReduceSum()
{
  try
  {
    const uint32_t n=256;
    const std::vector<float> in_data=MakeRandomVec(n,13);

    CAIF_CudaStream stream;

    CAIF_DeviceTensor i_dev=CAIF_DeviceTensor::Zeros({n},stream);
    i_dev.CopyFromHost(in_data.data(),n);
    const float dev_sum=CAIF_Ops::ReduceSum(i_dev);
    stream.Synchronize();

    CAIF_DeviceTensor i_host=CAIF_DeviceTensor::ZerosHost({n});
    WriteHostTensorFp32(i_host,in_data);
    const float host_sum=CAIF_Ops::ReduceSum(i_host);

    const float denom=std::max(std::fabs(dev_sum),1.0e-6f);
    const float rel=std::fabs(dev_sum-host_sum)/denom;
    return rel<CAIF_Tolerances::Fp32Rel();
  }
  CAIF_CATCH_BLOCK();
  return false;
}

}  // namespace

int main()
{
  ISE_Out::Out()<<"Host/Device Parity Tests\n";
  ISE_Out::Out()<<"========================\n";

  ReportResult("MatMul FP32 parity",TestMatMulFp32());

  AddOpRunner add_runner;
  ReportResult("Add parity",TestBinary(add_runner,21));
  MultiplyOpRunner mul_runner;
  ReportResult("Multiply parity",TestBinary(mul_runner,23));
  SubtractOpRunner sub_runner;
  ReportResult("Subtract parity",TestBinary(sub_runner,25));

  ReLUOpRunner relu_runner;
  ReportResult("ReLU parity",TestUnary(relu_runner,31,false));
  SigmoidOpRunner sigmoid_runner;
  ReportResult("Sigmoid parity",TestUnary(sigmoid_runner,33,false));
  TanhOpRunner tanh_runner;
  ReportResult("Tanh parity",TestUnary(tanh_runner,35,false));
  GELUOpRunner gelu_runner;
  ReportResult("GELU parity",TestUnary(gelu_runner,37,false));
  SwishOpRunner swish_runner;
  ReportResult("Swish parity",TestUnary(swish_runner,39,false));
  SqrtOpRunner sqrt_runner;
  ReportResult("Sqrt parity",TestUnary(sqrt_runner,41,true));

  ReportResult("Softmax parity",TestSoftmax());
  ReportResult("BiasAdd parity",TestBiasAdd());
  ReportResult("ReduceSum parity",TestReduceSum());

  ISE_Out::Out()<<"\n";
  ISE_Out::Out()<<"Passed: "
                <<CAIF_TestHarness::PassedCount()
                <<"  Failed: "
                <<CAIF_TestHarness::FailedCount()
                <<"\n";
  return CAIF_TestHarness::FinalExitCode();
}
