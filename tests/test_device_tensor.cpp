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
// Tests for CAIF_HostTensor, CAIF_DeviceTensor, CAIF_CudaStream, CAIF_CudaEvent.
//------------------------------------------------------------------------------
#include "caif_device_tensor.h"
#include "caif_test_harness.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cuda_event.h"
#include "ise_lib/ise_out.h"
#include <vector>
#include <cmath>

namespace instance
{

constexpr float g_caif_tensor_test_fill_val=3.14f;
constexpr float g_caif_tensor_test_fill_val2=7.0f;
constexpr float g_caif_tensor_test_fill_val3=5.0f;
constexpr float g_caif_tensor_test_clone_val=42.0f;
constexpr float g_caif_tensor_test_tol=1e-4f;
constexpr float g_caif_tensor_test_roundtrip_denom=256.0f;
constexpr size_t g_caif_tensor_test_from_host_n=100;

//------------------------------------------------------------------------------
// CAIF_DeviceTensorTests — host and device tensor test suite.
//------------------------------------------------------------------------------
class CAIF_DeviceTensorTests
{
  public:
    static void RunAll();

  protected:

  private:
    static bool FloatEqual(float a,float b,float tolerance=g_caif_tensor_test_tol);

    static void TestHostTensorZeros();
    static void TestHostTensorFromData();
    static void TestHostTensorFill();
    static void TestHostTensorReshape();
    static void TestHostTensorMoveSemantics();

#ifdef USE_CAIF_CUDA
    static void TestCudaStreamCreate();
    static void TestCudaStreamDefault();
    static void TestCudaEventCreate();
    static void TestCudaStreamRecordEvent();
    static void TestDeviceTensorZeros();
    static void TestDeviceTensorFromHost();
    static void TestDeviceTensorRoundTrip();
    static void TestDeviceTensorMoveSemantics();
    static void TestDeviceTensorClone();
    static void TestDeviceTensorFill();
    static void TestDeviceTensorNoDirtyFlags();
#endif
};

bool CAIF_DeviceTensorTests::FloatEqual(const float a,const float b,const float tolerance)
{
  return CAIF_TestHarness::FloatEqual(a,b,tolerance);
}

// ============================================================================
// CAIF_HostTensor Tests
// ============================================================================

void CAIF_DeviceTensorTests::TestHostTensorZeros()
{
  try
  {
    CAIF_HostTensor tensor=CAIF_HostTensor::Zeros({10,20});

    bool passed=true;
    if(tensor.Shape().size()!=2)
    {
      passed=false;
    }
    if(tensor.Shape()[0]!=10||tensor.Shape()[1]!=20)
    {
      passed=false;
    }
    if(tensor.TotalElements()!=200)
    {
      passed=false;
    }

    // Check all values are zero
    const float *data=tensor.Data();
    for(size_t i=0;i<tensor.TotalElements();++i)
    {
      if(data[i]!=0.0f)
      {
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("HostTensor::Zeros",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostTensor::Zeros")
}

void CAIF_DeviceTensorTests::TestHostTensorFromData()
{
  try
  {
    std::vector<float> source_data={1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    CAIF_HostTensor tensor=CAIF_HostTensor::FromData(source_data.data(),{2,3});

    bool passed=true;
    if(tensor.TotalElements()!=6)
    {
      passed=false;
    }

    // Check values match
    const float *data=tensor.Data();
    for(size_t i=0;i<6;++i)
    {
      if(data[i]!=source_data[i])
      {
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("HostTensor::FromData",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostTensor::FromData")
}

void CAIF_DeviceTensorTests::TestHostTensorFill()
{
  try
  {
    CAIF_HostTensor tensor=CAIF_HostTensor::Uninitialized({5,5});
    tensor.Fill(g_caif_tensor_test_fill_val);

    bool passed=true;
    const float *data=tensor.Data();
    for(size_t i=0;i<tensor.TotalElements();++i)
    {
      if(FloatEqual(data[i],g_caif_tensor_test_fill_val)==false)
      {
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("HostTensor::Fill",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostTensor::Fill")
}

void CAIF_DeviceTensorTests::TestHostTensorReshape()
{
  try
  {
    CAIF_HostTensor tensor=CAIF_HostTensor::Zeros({2,3,4});

    // Reshape to different dimensions with same total elements
    tensor.Reshape({4,6});

    bool passed=true;
    if(tensor.Shape().size()!=2)
    {
      passed=false;
    }
    if(tensor.Shape()[0]!=4||tensor.Shape()[1]!=6)
    {
      passed=false;
    }
    if(tensor.TotalElements()!=24)
    {
      passed=false;
    }

    CAIF_TestHarness::Report("HostTensor::Reshape",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostTensor::Reshape")
}

void CAIF_DeviceTensorTests::TestHostTensorMoveSemantics()
{
  try
  {
    CAIF_HostTensor tensor1=CAIF_HostTensor::Zeros({10,10});
    tensor1.Fill(g_caif_tensor_test_fill_val3);

    // Move construct
    CAIF_HostTensor tensor2=std::move(tensor1);

    bool passed=true;

    // tensor1 should be empty
    if(tensor1.IsEmpty()==false)
    {
      passed=false;
    }

    // tensor2 should have the data
    if(tensor2.TotalElements()!=100)
    {
      passed=false;
    }
    if(FloatEqual(tensor2.Data()[0],g_caif_tensor_test_fill_val3)==false)
    {
      passed=false;
    }

    CAIF_TestHarness::Report("HostTensor::MoveSemantics",passed);
  }
  CAIF_TEST_CATCH_BLOCK("HostTensor::MoveSemantics")
}

// ============================================================================
// CUDA Stream and Event Tests (only run with CUDA)
// ============================================================================

#ifdef USE_CAIF_CUDA

void CAIF_DeviceTensorTests::TestCudaStreamCreate()
{
  try
  {
    CAIF_CudaStream stream;

    bool passed=stream.IsValid();

    CAIF_TestHarness::Report("CudaStream::Create",passed);
  }
  CAIF_TEST_CATCH_BLOCK("CudaStream::Create")
}

void CAIF_DeviceTensorTests::TestCudaStreamDefault()
{
  try
  {
    CAIF_CudaStream &stream=CAIF_CudaStream::Default();

    bool passed=stream.IsValid();

    CAIF_TestHarness::Report("CudaStream::Default",passed);
  }
  CAIF_TEST_CATCH_BLOCK("CudaStream::Default")
}

void CAIF_DeviceTensorTests::TestCudaEventCreate()
{
  try
  {
    CAIF_CudaEvent event;

    bool passed=event.IsValid();

    CAIF_TestHarness::Report("CudaEvent::Create",passed);
  }
  CAIF_TEST_CATCH_BLOCK("CudaEvent::Create")
}

void CAIF_DeviceTensorTests::TestCudaStreamRecordEvent()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_CudaEvent event=stream.RecordEvent();

    bool passed=event.IsValid();

    // Event should complete (no actual work submitted)
    stream.Synchronize();
    if(event.IsComplete()==false)
    {
      passed=false;
    }

    CAIF_TestHarness::Report("CudaStream::RecordEvent",passed);
  }
  CAIF_TEST_CATCH_BLOCK("CudaStream::RecordEvent")
}

// ============================================================================
// CAIF_DeviceTensor Tests (only run with CUDA)
// ============================================================================

void CAIF_DeviceTensorTests::TestDeviceTensorZeros()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceTensor tensor=CAIF_DeviceTensor::Zeros({10,20},stream);

    bool passed=true;
    if(tensor.Shape().size()!=2)
    {
      passed=false;
    }
    if(tensor.Shape()[0]!=10||tensor.Shape()[1]!=20)
    {
      passed=false;
    }
    if(tensor.TotalElements()!=200)
    {
      passed=false;
    }
    if(tensor.IsAllocated()==false)
    {
      passed=false;
    }

    // Download and verify zeros
    CAIF_HostTensor host=tensor.ToHost();
    const float *data=host.Data();
    for(size_t i=0;i<host.TotalElements();++i)
    {
      if(data[i]!=0.0f)
      {
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("DeviceTensor::Zeros",passed);
  }
  CAIF_TEST_CATCH_BLOCK("DeviceTensor::Zeros")
}

void CAIF_DeviceTensorTests::TestDeviceTensorFromHost()
{
  try
  {
    // Create host tensor with data
    std::vector<float> source_data(g_caif_tensor_test_from_host_n);
    for(size_t i=0;i<g_caif_tensor_test_from_host_n;++i)
    {
      source_data[i]=static_cast<float>(i)*0.1f;
    }
    CAIF_HostTensor host=CAIF_HostTensor::FromData(source_data.data(),{10,10});

    // Upload to device
    CAIF_CudaStream stream;
    CAIF_DeviceTensor device=CAIF_DeviceTensor::FromHost(host,stream);

    bool passed=true;
    if(device.TotalElements()!=g_caif_tensor_test_from_host_n)
    {
      passed=false;
    }

    // Download and verify
    CAIF_HostTensor downloaded=device.ToHost();
    const float *data=downloaded.Data();
    for(size_t i=0;i<g_caif_tensor_test_from_host_n;++i)
    {
      if(FloatEqual(data[i],source_data[i])==false)
      {
        passed=false;
        ISE_Out::Out()<<"Mismatch at index "
                      <<i
                      <<": got "
                      <<data[i]
                      <<" expected "
                      <<source_data[i]
                      <<"\n";
        break;
      }
    }

    CAIF_TestHarness::Report("DeviceTensor::FromHost",passed);
  }
  CAIF_TEST_CATCH_BLOCK("DeviceTensor::FromHost")
}

void CAIF_DeviceTensorTests::TestDeviceTensorRoundTrip()
{
  try
  {
    // Create host data
    CAIF_HostTensor original=CAIF_HostTensor::Zeros({32,64});
    float *data=original.Data();
    for(size_t i=0;i<original.TotalElements();++i)
    {
      data[i]=static_cast<float>(i%256)/g_caif_tensor_test_roundtrip_denom;
    }

    // Upload to device
    CAIF_CudaStream stream;
    CAIF_DeviceTensor device=original.ToDevice(stream);

    // Download back
    CAIF_HostTensor roundtrip=device.ToHost();

    bool passed=true;
    const float *orig_data=original.Data();
    const float *rt_data=roundtrip.Data();
    for(size_t i=0;i<original.TotalElements();++i)
    {
      if(FloatEqual(orig_data[i],rt_data[i])==false)
      {
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("DeviceTensor::RoundTrip",passed);
  }
  CAIF_TEST_CATCH_BLOCK("DeviceTensor::RoundTrip")
}

void CAIF_DeviceTensorTests::TestDeviceTensorMoveSemantics()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceTensor tensor1=CAIF_DeviceTensor::Zeros({10,10},stream);

    // Fill with non-zero values by uploading
    CAIF_HostTensor host=CAIF_HostTensor::Uninitialized({10,10});
    host.Fill(g_caif_tensor_test_fill_val2);
    tensor1.CopyFromHost(host.Data(),host.TotalElements());

    // Move construct
    CAIF_DeviceTensor tensor2=std::move(tensor1);

    bool passed=true;

    // tensor1 should be empty
    if(tensor1.IsEmpty()==false||tensor1.IsAllocated()==true)
    {
      passed=false;
    }

    // tensor2 should have the data
    if(tensor2.TotalElements()!=100||tensor2.IsAllocated()==false)
    {
      passed=false;
    }

    // Verify data survived the move
    CAIF_HostTensor downloaded=tensor2.ToHost();
    if(FloatEqual(downloaded.Data()[0],g_caif_tensor_test_fill_val2)==false)
    {
      passed=false;
    }

    CAIF_TestHarness::Report("DeviceTensor::MoveSemantics",passed);
  }
  CAIF_TEST_CATCH_BLOCK("DeviceTensor::MoveSemantics")
}

void CAIF_DeviceTensorTests::TestDeviceTensorClone()
{
  try
  {
    // Create source tensor with data
    CAIF_HostTensor host=CAIF_HostTensor::Uninitialized({16,16});
    host.Fill(g_caif_tensor_test_clone_val);

    CAIF_CudaStream stream;
    CAIF_DeviceTensor original=host.ToDevice(stream);

    // Clone
    CAIF_DeviceTensor cloned=original.Clone();

    bool passed=true;

    // Both should be allocated
    if(original.IsAllocated()==false||cloned.IsAllocated()==false)
    {
      passed=false;
    }

    // They should have different device pointers
    if(original.DevicePtr()==cloned.DevicePtr())
    {
      passed=false;
    }

    // Cloned should have same data
    CAIF_HostTensor downloaded=cloned.ToHost();
    if(FloatEqual(downloaded.Data()[0],g_caif_tensor_test_clone_val)==false)
    {
      passed=false;
    }

    CAIF_TestHarness::Report("DeviceTensor::Clone",passed);
  }
  CAIF_TEST_CATCH_BLOCK("DeviceTensor::Clone")
}

void CAIF_DeviceTensorTests::TestDeviceTensorFill()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceTensor tensor=CAIF_DeviceTensor::Uninitialized({8,8},stream);

    // Fill with zeros (uses cudaMemset)
    tensor.Fill(0.0f);

    CAIF_HostTensor downloaded=tensor.ToHost();
    bool passed=true;
    const float *data=downloaded.Data();
    for(size_t i=0;i<64;++i)
    {
      if(data[i]!=0.0f)
      {
        passed=false;
        break;
      }
    }

    CAIF_TestHarness::Report("DeviceTensor::Fill",passed);
  }
  CAIF_TEST_CATCH_BLOCK("DeviceTensor::Fill")
}

void CAIF_DeviceTensorTests::TestDeviceTensorNoDirtyFlags()
{
  // This test verifies the design principle: no implicit synchronization.
  // The device tensor has NO dirty flags - what you see is what you get.
  try
  {
    CAIF_CudaStream stream;

    // Create device tensor
    CAIF_DeviceTensor device=CAIF_DeviceTensor::Zeros({10},stream);

    // The device pointer is always valid (no need to check dirty flags)
    float *ptr=device.DevicePtr();
    bool passed=(ptr!=nullptr);

    // No hidden state to manage
    if(device.IsAllocated()==false)
    {
      passed=false;
    }

    CAIF_TestHarness::Report("DeviceTensor::NoDirtyFlags",passed);
  }
  CAIF_TEST_CATCH_BLOCK("DeviceTensor::NoDirtyFlags")
}

#endif  // USE_CAIF_CUDA

void CAIF_DeviceTensorTests::RunAll()
{
  ISE_Out::Out()<<"=== Device-Resident Tensor Architecture Tests ===\n\n";

  ISE_Out::Out()<<"--- Host Tensor Tests ---\n";
  TestHostTensorZeros();
  TestHostTensorFromData();
  TestHostTensorFill();
  TestHostTensorReshape();
  TestHostTensorMoveSemantics();

#ifdef USE_CAIF_CUDA
  ISE_Out::Out()<<"\n--- CUDA Stream/Event Tests ---\n";
  TestCudaStreamCreate();
  TestCudaStreamDefault();
  TestCudaEventCreate();
  TestCudaStreamRecordEvent();

  ISE_Out::Out()<<"\n--- Device Tensor Tests ---\n";
  TestDeviceTensorZeros();
  TestDeviceTensorFromHost();
  TestDeviceTensorRoundTrip();
  TestDeviceTensorMoveSemantics();
  TestDeviceTensorClone();
  TestDeviceTensorFill();
  TestDeviceTensorNoDirtyFlags();
#else
  ISE_Out::Out()<<"\n[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif
}

}//end instance namespace

int main()
{
  instance::CAIF_DeviceTensorTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
