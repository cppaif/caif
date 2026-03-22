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
// Cross-entropy loss tests
//------------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include "caif_device_cross_entropy_loss.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_host_tensor.h"

using namespace instance;

static int g_tests_passed=0;
static int g_tests_failed=0;

static void ReportTest(const std::string &name,bool passed)
{
  if(passed==true)
  {
    std::cout<<"  PASS: "<<name<<"\n";
    ++g_tests_passed;
  }
  else
  {
    std::cout<<"  FAIL: "<<name<<"\n";
    ++g_tests_failed;
  }
}

//------------------------------------------------------------------------------
// CPU reference implementation
//------------------------------------------------------------------------------
static float CpuCrossEntropyLoss(const std::vector<float> &logits,
                                  const std::vector<int> &targets,
                                  int n,
                                  int vocab_size,
                                  int ignore_index)
{
  float total_loss=0.0f;
  int count=0;

  for(int i=0;i<n;++i)
  {
    if(targets[i]==ignore_index)
    {
      continue;
    }

    // Find max for stability
    float max_logit=-1e30f;
    for(int v=0;v<vocab_size;++v)
    {
      const float val=logits[i*vocab_size+v];
      if(val>max_logit)
      {
        max_logit=val;
      }
    }

    // Compute log-sum-exp
    float sum_exp=0.0f;
    for(int v=0;v<vocab_size;++v)
    {
      sum_exp+=std::exp(logits[i*vocab_size+v]-max_logit);
    }
    const float log_sum_exp=max_logit+std::log(sum_exp);

    // Loss for this position
    const float target_logit=logits[i*vocab_size+targets[i]];
    total_loss+=log_sum_exp-target_logit;
    ++count;
  }

  if(count==0)
  {
    return 0.0f;
  }
  return total_loss/static_cast<float>(count);
}

static void CpuCrossEntropyGradient(const std::vector<float> &logits,
                                     const std::vector<int> &targets,
                                     std::vector<float> &grad,
                                     int n,
                                     int vocab_size,
                                     int ignore_index)
{
  grad.resize(n*vocab_size);

  for(int i=0;i<n;++i)
  {
    if(targets[i]==ignore_index)
    {
      for(int v=0;v<vocab_size;++v)
      {
        grad[i*vocab_size+v]=0.0f;
      }
      continue;
    }

    // Find max for stability
    float max_logit=-1e30f;
    for(int v=0;v<vocab_size;++v)
    {
      const float val=logits[i*vocab_size+v];
      if(val>max_logit)
      {
        max_logit=val;
      }
    }

    // Compute sum of exp
    float sum_exp=0.0f;
    for(int v=0;v<vocab_size;++v)
    {
      sum_exp+=std::exp(logits[i*vocab_size+v]-max_logit);
    }

    // Gradient: softmax - one_hot
    for(int v=0;v<vocab_size;++v)
    {
      const float softmax_val=std::exp(logits[i*vocab_size+v]-max_logit)/sum_exp;
      float g=softmax_val;
      if(v==targets[i])
      {
        g-=1.0f;
      }
      grad[i*vocab_size+v]=g/static_cast<float>(n);
    }
  }
}

//------------------------------------------------------------------------------
// Test 1: Loss output is scalar
//------------------------------------------------------------------------------
static void TestLossShape()
{
  CAIF_CudaStream stream;

  constexpr int n=4;
  constexpr int vocab_size=10;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>(i%vocab_size)*0.1f;
  }

  std::vector<float> targets_data={0.0f,1.0f,2.0f,3.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_data.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  float loss=CAIF_DeviceCrossEntropyLoss::ComputeLoss(logits,targets,stream);

  // Loss should be a finite positive number
  bool passed=(std::isfinite(loss)==true&&loss>=0.0f);
  ReportTest("TestLossShape",passed);
}

//------------------------------------------------------------------------------
// Test 2: Loss values match CPU reference
//------------------------------------------------------------------------------
static void TestLossValues()
{
  CAIF_CudaStream stream;

  constexpr int n=8;
  constexpr int vocab_size=16;

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
  constexpr int ignore_index=-100;
  const float cpu_loss=CpuCrossEntropyLoss(logits_data,targets_int,n,vocab_size,ignore_index);

  // GPU
  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_float.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  const float gpu_loss=CAIF_DeviceCrossEntropyLoss::ComputeLoss(logits,targets,stream);

  const float diff=std::abs(gpu_loss-cpu_loss);
  constexpr float tol=1e-4f;
  const bool passed=(diff<tol);

  if(passed==false)
  {
    std::cout<<"    CPU loss: "<<cpu_loss<<", GPU loss: "<<gpu_loss<<", diff: "<<diff<<"\n";
  }

  ReportTest("TestLossValues",passed);
}

//------------------------------------------------------------------------------
// Test 3: Numerical stability with large logits
//------------------------------------------------------------------------------
static void TestLossNumericalStability()
{
  CAIF_CudaStream stream;

  constexpr int n=4;
  constexpr int vocab_size=8;

  // Large logits that would overflow naive exp
  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=100.0f+static_cast<float>(i%vocab_size)*10.0f;
  }

  std::vector<float> targets_data={0.0f,1.0f,2.0f,3.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_data.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  const float loss=CAIF_DeviceCrossEntropyLoss::ComputeLoss(logits,targets,stream);

  // Loss should be finite (not NaN or Inf)
  const bool passed=(std::isfinite(loss)==true);

  if(passed==false)
  {
    std::cout<<"    Loss with large logits: "<<loss<<"\n";
  }

  ReportTest("TestLossNumericalStability",passed);
}

//------------------------------------------------------------------------------
// Test 4: Gradient shape matches logits
//------------------------------------------------------------------------------
static void TestGradientShape()
{
  CAIF_CudaStream stream;

  constexpr int n=4;
  constexpr int vocab_size=10;

  std::vector<float> logits_data(n*vocab_size,0.5f);
  std::vector<float> targets_data={0.0f,1.0f,2.0f,3.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_data.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  CAIF_DeviceTensor grad=CAIF_DeviceCrossEntropyLoss::ComputeGradient(logits,targets,stream);

  const auto &grad_shape=grad.Shape();
  const bool passed=(grad_shape.size()==2&&
                     grad_shape[0]==static_cast<uint32_t>(n)&&
                     grad_shape[1]==static_cast<uint32_t>(vocab_size));

  ReportTest("TestGradientShape",passed);
}

//------------------------------------------------------------------------------
// Test 5: Gradient values match CPU reference
//------------------------------------------------------------------------------
static void TestGradientValues()
{
  CAIF_CudaStream stream;

  constexpr int n=4;
  constexpr int vocab_size=8;

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
  constexpr int ignore_index=-100;
  std::vector<float> cpu_grad;
  CpuCrossEntropyGradient(logits_data,targets_int,cpu_grad,n,vocab_size,ignore_index);

  // GPU
  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_float.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  CAIF_DeviceTensor gpu_grad=CAIF_DeviceCrossEntropyLoss::ComputeGradient(logits,targets,stream);
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

  constexpr float tol=1e-4f;
  const bool passed=(max_diff<tol);

  if(passed==false)
  {
    std::cout<<"    Max gradient diff: "<<max_diff<<"\n";
  }

  ReportTest("TestGradientValues",passed);
}

//------------------------------------------------------------------------------
// Test 6: Finite difference gradient check
//------------------------------------------------------------------------------
static void TestGradientFiniteDiff()
{
  CAIF_CudaStream stream;

  constexpr int n=2;
  constexpr int vocab_size=4;
  constexpr float h=1e-3f;
  constexpr float tol=5e-2f;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>(i)/static_cast<float>(n*vocab_size);
  }

  std::vector<float> targets_data={1.0f,2.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_data.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  // Compute analytical gradient
  CAIF_DeviceTensor grad=CAIF_DeviceCrossEntropyLoss::ComputeGradient(logits,targets,stream);
  CAIF_HostTensor host_grad=grad.ToHost();

  // Check a few elements with finite difference
  bool all_passed=true;
  constexpr int num_checks=4;

  for(int check=0;check<num_checks;++check)
  {
    const int idx=check*(n*vocab_size)/num_checks;

    // f(x+h)
    std::vector<float> logits_plus=logits_data;
    logits_plus[idx]+=h;
    CAIF_DeviceTensor logits_p=CAIF_DeviceTensor::FromHostData(logits_plus.data(),
                                                              {static_cast<uint32_t>(n),
                                                               static_cast<uint32_t>(vocab_size)},
                                                              stream);
    const float loss_plus=CAIF_DeviceCrossEntropyLoss::ComputeLoss(logits_p,targets,stream);

    // f(x-h)
    std::vector<float> logits_minus=logits_data;
    logits_minus[idx]-=h;
    CAIF_DeviceTensor logits_m=CAIF_DeviceTensor::FromHostData(logits_minus.data(),
                                                              {static_cast<uint32_t>(n),
                                                               static_cast<uint32_t>(vocab_size)},
                                                              stream);
    const float loss_minus=CAIF_DeviceCrossEntropyLoss::ComputeLoss(logits_m,targets,stream);

    const float numerical_grad=(loss_plus-loss_minus)/(2.0f*h);
    const float analytical_grad=host_grad.Data()[idx];
    const float diff=std::abs(numerical_grad-analytical_grad);

    if(diff>tol)
    {
      std::cout<<"    idx "<<idx<<": numerical="<<numerical_grad
               <<", analytical="<<analytical_grad<<", diff="<<diff<<"\n";
      all_passed=false;
    }
  }

  ReportTest("TestGradientFiniteDiff",all_passed);
}

//------------------------------------------------------------------------------
// Test 7: Ignore index excludes positions
//------------------------------------------------------------------------------
static void TestIgnoreIndex()
{
  CAIF_CudaStream stream;

  constexpr int n=4;
  constexpr int vocab_size=8;
  constexpr int ignore_index=-100;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>(i%vocab_size)*0.1f;
  }

  // Two positions are ignored
  std::vector<float> targets_with_ignore={0.0f,static_cast<float>(ignore_index),
                                           2.0f,static_cast<float>(ignore_index)};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_with_ignore.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  // Loss should only consider positions 0 and 2
  const float loss_with_ignore=CAIF_DeviceCrossEntropyLoss::ComputeLoss(
      logits,targets,stream,ignore_index);

  // Reference: compute loss for just positions 0 and 2
  std::vector<float> logits_subset(2*vocab_size);
  for(int v=0;v<vocab_size;++v)
  {
    logits_subset[0*vocab_size+v]=logits_data[0*vocab_size+v];
    logits_subset[1*vocab_size+v]=logits_data[2*vocab_size+v];
  }
  std::vector<int> targets_subset={0,2};
  const float loss_subset=CpuCrossEntropyLoss(logits_subset,targets_subset,2,vocab_size,-100);

  const float diff=std::abs(loss_with_ignore-loss_subset);
  constexpr float tol=1e-4f;
  const bool passed=(diff<tol);

  if(passed==false)
  {
    std::cout<<"    Loss with ignore: "<<loss_with_ignore<<", subset loss: "<<loss_subset<<"\n";
  }

  ReportTest("TestIgnoreIndex",passed);
}

//------------------------------------------------------------------------------
// Test 8: Fused matches separate loss + gradient
//------------------------------------------------------------------------------
static void TestFusedMatchesSeparate()
{
  CAIF_CudaStream stream;

  constexpr int n=4;
  constexpr int vocab_size=8;

  std::vector<float> logits_data(n*vocab_size);
  for(int i=0;i<n*vocab_size;++i)
  {
    logits_data[i]=static_cast<float>((i*5+2)%30)/15.0f-1.0f;
  }

  std::vector<float> targets_data={0.0f,3.0f,5.0f,7.0f};

  CAIF_DeviceTensor logits=CAIF_DeviceTensor::FromHostData(logits_data.data(),
                                                          {static_cast<uint32_t>(n),
                                                           static_cast<uint32_t>(vocab_size)},
                                                          stream);
  CAIF_DeviceTensor targets=CAIF_DeviceTensor::FromHostData(targets_data.data(),
                                                           {static_cast<uint32_t>(n)},
                                                           stream);

  // Separate computation
  const float loss_separate=CAIF_DeviceCrossEntropyLoss::ComputeLoss(logits,targets,stream);
  CAIF_DeviceTensor grad_separate=CAIF_DeviceCrossEntropyLoss::ComputeGradient(logits,targets,stream);

  // Fused computation
  CAIF_DeviceTensor grad_fused;
  const float loss_fused=CAIF_DeviceCrossEntropyLoss::ComputeLossAndGradient(
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

  constexpr float tol=1e-5f;
  const bool passed=(loss_diff<tol&&max_grad_diff<tol);

  if(passed==false)
  {
    std::cout<<"    Loss diff: "<<loss_diff<<", max grad diff: "<<max_grad_diff<<"\n";
  }

  ReportTest("TestFusedMatchesSeparate",passed);
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main()
{
  std::cout<<"=== Cross-Entropy Loss Tests ===\n";

  TestLossShape();
  TestLossValues();
  TestLossNumericalStability();
  TestGradientShape();
  TestGradientValues();
  TestGradientFiniteDiff();
  TestIgnoreIndex();
  TestFusedMatchesSeparate();

  std::cout<<"\n=== Summary ===\n";
  std::cout<<"Passed: "<<g_tests_passed<<"\n";
  std::cout<<"Failed: "<<g_tests_failed<<"\n";

  if(g_tests_failed==0)
  {
    std::cout<<"All tests passed!\n";
    return 0;
  }
  return 1;
}
