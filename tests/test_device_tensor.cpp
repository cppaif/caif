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

#include "caif_device_tensor.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_cuda_event.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportResult(const char *test_name,bool passed)
{
  if(passed==true)
  {
    std::cout<<"[PASS] "<<test_name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    std::cout<<"[FAIL] "<<test_name<<"\n";
    ++g_tests_failed;
  }
}

static bool FloatEqual(float a,float b,float tolerance=1e-5f)
{
  return std::fabs(a-b)<tolerance;
}

// ============================================================================
// CAIF_HostTensor Tests
// ============================================================================

static void TestHostTensorZeros()
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

    ReportResult("HostTensor::Zeros",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("HostTensor::Zeros",false);
  }
}

static void TestHostTensorFromData()
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

    ReportResult("HostTensor::FromData",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("HostTensor::FromData",false);
  }
}

static void TestHostTensorFill()
{
  try
  {
    CAIF_HostTensor tensor=CAIF_HostTensor::Uninitialized({5,5});
    tensor.Fill(3.14f);

    bool passed=true;
    const float *data=tensor.Data();
    for(size_t i=0;i<tensor.TotalElements();++i)
    {
      if(FloatEqual(data[i],3.14f)==false)
      {
        passed=false;
        break;
      }
    }

    ReportResult("HostTensor::Fill",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("HostTensor::Fill",false);
  }
}

static void TestHostTensorReshape()
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

    ReportResult("HostTensor::Reshape",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("HostTensor::Reshape",false);
  }
}

static void TestHostTensorMoveSemantics()
{
  try
  {
    CAIF_HostTensor tensor1=CAIF_HostTensor::Zeros({10,10});
    tensor1.Fill(5.0f);

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
    if(FloatEqual(tensor2.Data()[0],5.0f)==false)
    {
      passed=false;
    }

    ReportResult("HostTensor::MoveSemantics",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("HostTensor::MoveSemantics",false);
  }
}

// ============================================================================
// CUDA Stream and Event Tests (only run with CUDA)
// ============================================================================

#ifdef USE_CAIF_CUDA

static void TestCudaStreamCreate()
{
  try
  {
    CAIF_CudaStream stream;

    bool passed=stream.IsValid();

    ReportResult("CudaStream::Create",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("CudaStream::Create",false);
  }
}

static void TestCudaStreamDefault()
{
  try
  {
    CAIF_CudaStream &stream=CAIF_CudaStream::Default();

    bool passed=stream.IsValid();

    ReportResult("CudaStream::Default",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("CudaStream::Default",false);
  }
}

static void TestCudaEventCreate()
{
  try
  {
    CAIF_CudaEvent event;

    bool passed=event.IsValid();

    ReportResult("CudaEvent::Create",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("CudaEvent::Create",false);
  }
}

static void TestCudaStreamRecordEvent()
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

    ReportResult("CudaStream::RecordEvent",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("CudaStream::RecordEvent",false);
  }
}

// ============================================================================
// CAIF_DeviceTensor Tests (only run with CUDA)
// ============================================================================

static void TestDeviceTensorZeros()
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

    ReportResult("DeviceTensor::Zeros",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DeviceTensor::Zeros",false);
  }
}

static void TestDeviceTensorFromHost()
{
  try
  {
    // Create host tensor with data
    std::vector<float> source_data(100);
    for(size_t i=0;i<100;++i)
    {
      source_data[i]=static_cast<float>(i)*0.1f;
    }
    CAIF_HostTensor host=CAIF_HostTensor::FromData(source_data.data(),{10,10});

    // Upload to device
    CAIF_CudaStream stream;
    CAIF_DeviceTensor device=CAIF_DeviceTensor::FromHost(host,stream);

    bool passed=true;
    if(device.TotalElements()!=100)
    {
      passed=false;
    }

    // Download and verify
    CAIF_HostTensor downloaded=device.ToHost();
    const float *data=downloaded.Data();
    for(size_t i=0;i<100;++i)
    {
      if(FloatEqual(data[i],source_data[i])==false)
      {
        passed=false;
        std::cout<<"Mismatch at index "<<i<<": got "<<data[i]<<" expected "<<source_data[i]<<"\n";
        break;
      }
    }

    ReportResult("DeviceTensor::FromHost",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DeviceTensor::FromHost",false);
  }
}

static void TestDeviceTensorRoundTrip()
{
  try
  {
    // Create host data
    CAIF_HostTensor original=CAIF_HostTensor::Zeros({32,64});
    float *data=original.Data();
    for(size_t i=0;i<original.TotalElements();++i)
    {
      data[i]=static_cast<float>(i%256)/256.0f;
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

    ReportResult("DeviceTensor::RoundTrip",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DeviceTensor::RoundTrip",false);
  }
}

static void TestDeviceTensorMoveSemantics()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_DeviceTensor tensor1=CAIF_DeviceTensor::Zeros({10,10},stream);

    // Fill with non-zero values by uploading
    CAIF_HostTensor host=CAIF_HostTensor::Uninitialized({10,10});
    host.Fill(7.0f);
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
    if(FloatEqual(downloaded.Data()[0],7.0f)==false)
    {
      passed=false;
    }

    ReportResult("DeviceTensor::MoveSemantics",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DeviceTensor::MoveSemantics",false);
  }
}

static void TestDeviceTensorClone()
{
  try
  {
    // Create source tensor with data
    CAIF_HostTensor host=CAIF_HostTensor::Uninitialized({16,16});
    host.Fill(42.0f);

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
    if(FloatEqual(downloaded.Data()[0],42.0f)==false)
    {
      passed=false;
    }

    ReportResult("DeviceTensor::Clone",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DeviceTensor::Clone",false);
  }
}

static void TestDeviceTensorFill()
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

    ReportResult("DeviceTensor::Fill",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DeviceTensor::Fill",false);
  }
}

static void TestDeviceTensorNoDirtyFlags()
{
  // This test verifies the design principle: no implicit synchronization
  // The device tensor has NO dirty flags - what you see is what you get
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

    ReportResult("DeviceTensor::NoDirtyFlags",passed);
  }
  catch(const std::exception &e)
  {
    std::cout<<"Exception: "<<e.what()<<"\n";
    ReportResult("DeviceTensor::NoDirtyFlags",false);
  }
}

#endif  // USE_CAIF_CUDA

// ============================================================================
// Main
// ============================================================================

int main()
{
  std::cout<<"=== Device-Resident Tensor Architecture Tests ===\n\n";

  std::cout<<"--- Host Tensor Tests ---\n";
  TestHostTensorZeros();
  TestHostTensorFromData();
  TestHostTensorFill();
  TestHostTensorReshape();
  TestHostTensorMoveSemantics();

#ifdef USE_CAIF_CUDA
  std::cout<<"\n--- CUDA Stream/Event Tests ---\n";
  TestCudaStreamCreate();
  TestCudaStreamDefault();
  TestCudaEventCreate();
  TestCudaStreamRecordEvent();

  std::cout<<"\n--- Device Tensor Tests ---\n";
  TestDeviceTensorZeros();
  TestDeviceTensorFromHost();
  TestDeviceTensorRoundTrip();
  TestDeviceTensorMoveSemantics();
  TestDeviceTensorClone();
  TestDeviceTensorFill();
  TestDeviceTensorNoDirtyFlags();
#else
  std::cout<<"\n[SKIP] CUDA tests (USE_CAIF_CUDA not defined)\n";
#endif

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  if(g_tests_failed>0)
  {
    return 1;
  }
  return 0;
}
