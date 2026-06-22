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
// Cross-entropy loss tests.
//
// Tests cover loss shape (finite positive scalar), CPU reference parity,
// numerical stability with large logits, gradient shape, gradient values vs
// CPU reference, finite-difference gradient check, ignore-index masking,
// and fused loss+gradient vs separate computation.
//------------------------------------------------------------------------------
#include "caif_device_cross_entropy_loss.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_host_tensor.h"
#include "caif_test_harness.h"
#include "caif_cpu_reference/caif_cpu_cross_entropy.h"
#include "ise_lib/ise_out.h"

#include <vector>
#include <cmath>
#include <string>

namespace instance
{

#ifdef USE_CAIF_CUDA

constexpr int g_caif_ce_test_n_loss_shape=4;
constexpr int g_caif_ce_test_vocab_loss_shape=10;
constexpr int g_caif_ce_test_n_values=8;
constexpr int g_caif_ce_test_vocab_values=16;
constexpr int g_caif_ce_test_n_stability=4;
constexpr int g_caif_ce_test_vocab_stability=8;
constexpr int g_caif_ce_test_n_grad_shape=4;
constexpr int g_caif_ce_test_vocab_grad_shape=10;
constexpr int g_caif_ce_test_n_grad_vals=4;
constexpr int g_caif_ce_test_vocab_grad_vals=8;
constexpr int g_caif_ce_test_n_fd=2;
constexpr int g_caif_ce_test_vocab_fd=4;
constexpr int g_caif_ce_test_n_ignore=4;
constexpr int g_caif_ce_test_vocab_ignore=8;
constexpr int g_caif_ce_test_n_fused=4;
constexpr int g_caif_ce_test_vocab_fused=8;
constexpr int g_caif_ce_test_ignore_index=-100;
constexpr float g_caif_ce_test_values_tol=1e-4f;
constexpr float g_caif_ce_test_grad_tol=1e-4f;
constexpr float g_caif_ce_test_fd_tol=5e-2f;
constexpr float g_caif_ce_test_fused_tol=1e-5f;
constexpr float g_caif_ce_test_stability_base=100.0f;
constexpr float g_caif_ce_test_stability_scale=10.0f;
constexpr int g_caif_ce_test_fd_num_checks=4;

//------------------------------------------------------------------------------
// Cross-entropy loss tests.
//------------------------------------------------------------------------------
class CAIF_CrossEntropyTests
{
  public:
    static void RunAll();

  protected:

  private:
    static void TestLossShape();
    static void TestLossValues();
    static void TestLossNumericalStability();
    static void TestGradientShape();
    static void TestGradientValues();
    static void TestGradientFiniteDiff();
    static void TestIgnoreIndex();
    static void TestFusedMatchesSeparate();
};

//------------------------------------------------------------------------------
// Test 1: Loss output is scalar
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestLossShape()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_loss_shape;
  const int vocab_size=g_caif_ce_test_vocab_loss_shape;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>(i%vocab_size)*0.1f;
  }

  std::vector<float> targets_data={0.0f,1.0f,2.0f,3.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_data.data(),
    {static_cast<uint32_t>(n)},
    stream);

  const float loss=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLoss(logits,targets,stream);

  // Loss should be a finite positive number
  bool passed=(std::isfinite(loss)==true && loss>=0.0f);
  CAIF_TestHarness::Report("CrossEntropy::LossShape",passed);
}

//------------------------------------------------------------------------------
// Test 2: Loss values match CPU reference
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestLossValues()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_values;
  const int vocab_size=g_caif_ce_test_vocab_values;

  std::vector<float> logits_data(n*vocab_size);
  std::vector<int> targets_int(n);
  std::vector<float> targets_float(n);

  // Random-ish logits
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>((i*7+3)%100)/50.0f-1.0f;
  }

  // Targets
  for(int i=0;i<n;++i)
  {
    targets_int[i]=i%vocab_size;
    targets_float[i]=static_cast<float>(targets_int[i]);
  }

  // CPU reference
  const float cpu_loss=CAIF_CpuCrossEntropy::Loss(logits_data,
                                                   targets_int,
                                                   n,
                                                   vocab_size,
                                                   g_caif_ce_test_ignore_index);

  // GPU
  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_float.data(),
    {static_cast<uint32_t>(n)},
    stream);

  const float gpu_loss=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLoss(logits,targets,stream);

  const float diff=std::abs(gpu_loss-cpu_loss);
  const bool passed=(diff<g_caif_ce_test_values_tol);

  if(passed==false)
  {
    ISE_Out::Out()<<"    CPU loss: "
                 <<cpu_loss
                 <<", GPU loss: "
                 <<gpu_loss
                 <<", diff: "
                 <<diff
                 <<"\n";
  }

  CAIF_TestHarness::Report("CrossEntropy::LossValues",passed);
}

//------------------------------------------------------------------------------
// Test 3: Numerical stability with large logits
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestLossNumericalStability()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_stability;
  const int vocab_size=g_caif_ce_test_vocab_stability;

  // Large logits that would overflow naive exp
  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=g_caif_ce_test_stability_base+
                   static_cast<float>(i%vocab_size)*g_caif_ce_test_stability_scale;
  }

  std::vector<float> targets_data={0.0f,1.0f,2.0f,3.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_data.data(),
    {static_cast<uint32_t>(n)},
    stream);

  const float loss=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLoss(logits,targets,stream);

  // Loss should be finite (not NaN or Inf)
  const bool passed=(std::isfinite(loss)==true);

  if(passed==false)
  {
    ISE_Out::Out()<<"    Loss with large logits: "
                 <<loss
                 <<"\n";
  }

  CAIF_TestHarness::Report("CrossEntropy::LossNumericalStability",passed);
}

//------------------------------------------------------------------------------
// Test 4: Gradient shape matches logits
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestGradientShape()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_grad_shape;
  const int vocab_size=g_caif_ce_test_vocab_grad_shape;

  std::vector<float> logits_data(n*vocab_size,0.5f);
  std::vector<float> targets_data={0.0f,1.0f,2.0f,3.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_data.data(),
    {static_cast<uint32_t>(n)},
    stream);

  CAIF_DeviceTensor grad=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeGradient(logits,targets,stream);

  const auto &grad_shape=grad.Shape();
  const bool passed=(grad_shape.size()==2 &&
                     grad_shape[0]==static_cast<uint32_t>(n) &&
                     grad_shape[1]==static_cast<uint32_t>(vocab_size));

  CAIF_TestHarness::Report("CrossEntropy::GradientShape",passed);
}

//------------------------------------------------------------------------------
// Test 5: Gradient values match CPU reference
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestGradientValues()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_grad_vals;
  const int vocab_size=g_caif_ce_test_vocab_grad_vals;

  std::vector<float> logits_data(n*vocab_size);
  std::vector<int> targets_int(n);
  std::vector<float> targets_float(n);

  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>((i*3+1)%20)/10.0f-1.0f;
  }

  for(int i=0;i<n;++i)
  {
    targets_int[i]=(i*2)%vocab_size;
    targets_float[i]=static_cast<float>(targets_int[i]);
  }

  // CPU reference
  std::vector<float> cpu_grad;
  CAIF_CpuCrossEntropy::Gradient(logits_data,targets_int,cpu_grad,n,vocab_size,g_caif_ce_test_ignore_index);

  // GPU
  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_float.data(),
    {static_cast<uint32_t>(n)},
    stream);

  CAIF_DeviceTensor gpu_grad=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeGradient(logits,targets,stream);
  CAIF_HostTensor host_grad=gpu_grad.ToHost();

  // Compare
  float max_diff=0.0f;
  for(int i=0;i<n*vocab_size;++i)
  {
    const float diff=std::abs(host_grad.Data()[i]-cpu_grad[i]);
    if(diff>max_diff)
    {
      max_diff=diff;
    }
  }

  const bool passed=(max_diff<g_caif_ce_test_grad_tol);

  if(passed==false)
  {
    ISE_Out::Out()<<"    Max gradient diff: "
                 <<max_diff
                 <<"\n";
  }

  CAIF_TestHarness::Report("CrossEntropy::GradientValues",passed);
}

//------------------------------------------------------------------------------
// Test 6: Finite difference gradient check
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestGradientFiniteDiff()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_fd;
  const int vocab_size=g_caif_ce_test_vocab_fd;
  constexpr float h=1e-3f;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>(i)/static_cast<float>(n*vocab_size);
  }

  std::vector<float> targets_data={1.0f,2.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_data.data(),
    {static_cast<uint32_t>(n)},
    stream);

  // Compute analytical gradient
  CAIF_DeviceTensor grad=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeGradient(logits,targets,stream);
  CAIF_HostTensor host_grad=grad.ToHost();

  // Check a few elements with finite difference
  bool all_passed=true;

  for(int check=0;check<g_caif_ce_test_fd_num_checks;++check)
  {
    const int idx=check*(n*vocab_size)/g_caif_ce_test_fd_num_checks;

    // f(x+h)
    std::vector<float> logits_plus=logits_data;
    logits_plus[idx]+=h;
    CAIF_DeviceTensor logits_p=CAIF_DeviceTensor::FromHostData(
      logits_plus.data(),
      {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
      stream);
    const float loss_plus=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLoss(logits_p,targets,stream);

    // f(x-h)
    std::vector<float> logits_minus=logits_data;
    logits_minus[idx]-=h;
    CAIF_DeviceTensor logits_m=CAIF_DeviceTensor::FromHostData(
      logits_minus.data(),
      {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
      stream);
    const float loss_minus=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLoss(logits_m,targets,stream);

    const float numerical_grad=(loss_plus-loss_minus)/(2.0f*h);
    const float analytical_grad=host_grad.Data()[idx];
    const float diff=std::abs(numerical_grad-analytical_grad);

    if(diff>g_caif_ce_test_fd_tol)
    {
      ISE_Out::Out()<<"    idx "
                   <<idx
                   <<": numerical="
                   <<numerical_grad
                   <<", analytical="
                   <<analytical_grad
                   <<", diff="
                   <<diff
                   <<"\n";
      all_passed=false;
    }
  }

  CAIF_TestHarness::Report("CrossEntropy::GradientFiniteDiff",all_passed);
}

//------------------------------------------------------------------------------
// Test 7: Ignore index excludes positions
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestIgnoreIndex()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_ignore;
  const int vocab_size=g_caif_ce_test_vocab_ignore;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>(i%vocab_size)*0.1f;
  }

  // Two positions are ignored
  std::vector<float> targets_with_ignore={0.0f,
                                          static_cast<float>(g_caif_ce_test_ignore_index),
                                          2.0f,
                                          static_cast<float>(g_caif_ce_test_ignore_index)};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_with_ignore.data(),
    {static_cast<uint32_t>(n)},
    stream);

  // Loss should only consider positions 0 and 2
  const float loss_with_ignore=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLoss(
    logits,targets,stream,g_caif_ce_test_ignore_index);

  // Reference: compute loss for just positions 0 and 2
  std::vector<float> logits_subset(2*vocab_size);
  for(int v=0;v<vocab_size;++v)
  {
    logits_subset[0*vocab_size+v]=logits_data[0*vocab_size+v];
    logits_subset[1*vocab_size+v]=logits_data[2*vocab_size+v];
  }
  std::vector<int> targets_subset={0,2};
  const float loss_subset=CAIF_CpuCrossEntropy::Loss(logits_subset,
                                                      targets_subset,
                                                      2,
                                                      vocab_size,
                                                      -100);

  const float diff=std::abs(loss_with_ignore-loss_subset);
  const bool passed=(diff<g_caif_ce_test_values_tol);

  if(passed==false)
  {
    ISE_Out::Out()<<"    Loss with ignore: "
                 <<loss_with_ignore
                 <<", subset loss: "
                 <<loss_subset
                 <<"\n";
  }

  CAIF_TestHarness::Report("CrossEntropy::IgnoreIndex",passed);
}

//------------------------------------------------------------------------------
// Test 8: Fused matches separate loss + gradient
//------------------------------------------------------------------------------
void CAIF_CrossEntropyTests::TestFusedMatchesSeparate()
{
  CAIF_CudaStream stream;

  const int n=g_caif_ce_test_n_fused;
  const int vocab_size=g_caif_ce_test_vocab_fused;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>((i*5+2)%30)/15.0f-1.0f;
  }

  std::vector<float> targets_data={0.0f,3.0f,5.0f,7.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(
    logits_data.data(),
    {static_cast<uint32_t>(n),static_cast<uint32_t>(vocab_size)},
    stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(
    targets_data.data(),
    {static_cast<uint32_t>(n)},
    stream);

  // Separate computation
  const float loss_separate=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLoss(logits,targets,stream);
  CAIF_DeviceTensor grad_separate=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeGradient(logits,targets,stream);

  // Fused computation
  CAIF_DeviceTensor grad_fused;
  const float loss_fused=CAIF_DeviceCrossEntropyLoss<float,float>::ComputeLossAndGradient(
    logits,targets,grad_fused,stream);

  // Compare loss
  const float loss_diff=std::abs(loss_fused-loss_separate);

  // Compare gradient
  CAIF_HostTensor host_sep=grad_separate.ToHost();
  CAIF_HostTensor host_fused=grad_fused.ToHost();

  float max_grad_diff=0.0f;
  for(int i=0;i<n*vocab_size;++i)
  {
    const float diff=std::abs(host_sep.Data()[i]-host_fused.Data()[i]);
    if(diff>max_grad_diff)
    {
      max_grad_diff=diff;
    }
  }

  const bool passed=(loss_diff<g_caif_ce_test_fused_tol && max_grad_diff<g_caif_ce_test_fused_tol);

  if(passed==false)
  {
    ISE_Out::Out()<<"    Loss diff: "
                 <<loss_diff
                 <<", max grad diff: "
                 <<max_grad_diff
                 <<"\n";
  }

  CAIF_TestHarness::Report("CrossEntropy::FusedMatchesSeparate",passed);
}

void CAIF_CrossEntropyTests::RunAll()
{
  ISE_Out::Out()<<"=== Cross-Entropy Loss Tests ==="
               <<"\n";
  TestLossShape();
  TestLossValues();
  TestLossNumericalStability();
  TestGradientShape();
  TestGradientValues();
  TestGradientFiniteDiff();
  TestIgnoreIndex();
  TestFusedMatchesSeparate();
}

#endif// USE_CAIF_CUDA

}//end instance namespace

int main()
{
#ifdef USE_CAIF_CUDA
  instance::CAIF_CrossEntropyTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
#else
  ISE_Out::Out()<<"[SKIP] CUDA tests (USE_CAIF_CUDA not defined)"
               <<"\n";
  return 0;
#endif
}
