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

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include "caif_settings.h"
#include "caif_tensor_backend.h"

// Default timeout in seconds for each test
constexpr int g_default_test_timeout_seconds=30;

/**
 * @brief Structure to hold test result information
 */
struct TestResult
{
  std::string test_name;
  bool passed;
  bool timed_out;
  int exit_code;
  double execution_time;
};

static std::string g_backend_arg;
static int g_timeout_seconds=g_default_test_timeout_seconds;

/**
 * @brief Run a single test executable with timeout support
 */
TestResult RunTest(const std::string &test_name,
                   const std::string &executable_path,
                   const std::string &backend_arg)
{
  TestResult result;
  result.test_name=test_name;
  result.timed_out=false;
  
  std::cout<<"=== Running "<<test_name<<" ===\n";
  
  auto start_time=std::chrono::high_resolution_clock::now();
  
  // Execute the test with timeout using the timeout command
  // timeout returns 124 if the command times out
  std::string cmd="timeout "+std::to_string(g_timeout_seconds)+" "+executable_path;
  if(backend_arg.empty()==false)
  {
    cmd+=" --backend="+backend_arg;
  }
  int exit_code=std::system(cmd.c_str());
  
  // std::system returns the exit status shifted left by 8 bits on POSIX
  // WEXITSTATUS equivalent: (exit_code >> 8) & 0xFF
  const int actual_exit_code=(exit_code>>8)&0xFF;
  
  auto end_time=std::chrono::high_resolution_clock::now();
  auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
  
  result.exit_code=actual_exit_code;
  result.execution_time=duration.count()/1000.0;  // Convert to seconds
  
  // timeout command returns 124 when the command times out
  if(actual_exit_code==124)
  {
    result.timed_out=true;
    result.passed=false;
    std::cout<<"Result: ⏱️ TIMEOUT (exceeded "<<g_timeout_seconds<<"s)\n\n";
  }
  else
  {
    result.passed=(actual_exit_code==0);
    std::cout<<"Result: "<<(result.passed?"✅ PASSED":"❌ FAILED")<<" ("<<result.execution_time<<"s)\n\n";
  }
  
  return result;
}

static instance::CAIF_TensorBackend::BackendType_e BackendFromArg(const std::string &value)
{
  if(value=="cuda"||value=="CUDA")
  {
    return instance::CAIF_TensorBackend::BackendType_e::CUDA;
  }
  if(value=="cpu"||value=="CPU"||value=="blas"||value=="BLAS")
  {
    // BLAS is the default/preferred CPU backend (faster than Eigen)
    return instance::CAIF_TensorBackend::BackendType_e::BLAS;
  }
  if(value=="eigen"||value=="EIGEN")
  {
    return instance::CAIF_TensorBackend::BackendType_e::Eigen;
  }
  if(value=="vulkan"||value=="VULKAN")
  {
    return instance::CAIF_TensorBackend::BackendType_e::Vulkan;
  }
  return instance::CAIF_TensorBackend::BackendType_e::Auto;
}

static void ConfigureFromArgs(int argc,char *argv[])
{
  std::string backend_string;
  for(int i=1;i<argc;++i)
  {
    const std::string arg(argv[i]);
    const std::string backend_prefix="--backend=";
    const std::string timeout_prefix="--timeout=";
    
    if(arg.rfind(backend_prefix,0)==0)
    {
      const std::string value=arg.substr(backend_prefix.size());
      const auto backend=BackendFromArg(value);
      instance::CAIF_Settings::SetBackendOverride(backend);
      std::cout<<"Backend override set to "<<value<<"\n";
      backend_string=value;
    }
    else if(arg.rfind(timeout_prefix,0)==0)
    {
      const std::string value=arg.substr(timeout_prefix.size());
      g_timeout_seconds=std::stoi(value);
      std::cout<<"Timeout set to "<<g_timeout_seconds<<" seconds\n";
    }
  }
  if(backend_string.empty()==false)
  {
    g_backend_arg=backend_string;
    std::cout<<"Backend argument will be propagated to child tests\n";
  }
}

/**
 * @brief Print comprehensive test summary
 */
void PrintTestSummary(const std::vector<TestResult> &results)
{
  std::cout<<"========================================\n";
  std::cout<<"          TEST SUMMARY                  \n";
  std::cout<<"========================================\n\n";
  
  int passed_count=0;
  int failed_count=0;
  int timeout_count=0;
  double total_time=0.0;
  
  for(const auto &result:results)
  {
    std::cout<<result.test_name<<": ";
    if(result.passed==true)
    {
      std::cout<<"✅ PASSED";
      passed_count++;
    }
    else if(result.timed_out==true)
    {
      std::cout<<"⏱️ TIMEOUT";
      timeout_count++;
      failed_count++;
    }
    else
    {
      std::cout<<"❌ FAILED (exit code: "<<result.exit_code<<")";
      failed_count++;
    }
    std::cout<<" ("<<result.execution_time<<"s)\n";
    
    total_time+=result.execution_time;
  }
  
  std::cout<<"\n========================================\n";
  std::cout<<"Total Tests: "<<results.size()<<"\n";
  std::cout<<"Passed: "<<passed_count<<"\n";
  std::cout<<"Failed: "<<failed_count;
  if(timeout_count>0)
  {
    std::cout<<" ("<<timeout_count<<" timed out)";
  }
  std::cout<<"\n";
  std::cout<<"Success Rate: "
           <<(static_cast<double>(passed_count)/
              static_cast<double>(results.size())*100.0)
           <<"%\n";
  std::cout<<"Total Execution Time: "<<total_time<<"s\n";
  std::cout<<"Timeout per test: "<<g_timeout_seconds<<"s\n";
  std::cout<<"========================================\n";
  
  if(failed_count==0)
  {
    std::cout<<"\n✅ ALL TESTS PASSED! ✅\n";
    std::cout<<"The AIF neural network framework is fully functional!\n";
  }
  else
  {
    std::cout<<"\n❌ "<<failed_count<<" test(s) failed.\n";
    if(timeout_count>0)
    {
      std::cout<<"⏱️ "<<timeout_count<<" test(s) timed out (possible hang/deadlock).\n";
    }
    std::cout<<"Please review the failed tests and fix any issues.\n";
  }
}

/**
 * @brief Main test runner function
 */
int main(int argc,char *argv[])
{
  std::cout<<"===========================================\n";
  std::cout<<"  AIF Neural Network Framework Test Suite  \n";
  std::cout<<"===========================================\n\n";
  std::cout<<"Usage: ./run_all_tests [--backend=cuda|eigen|blas] [--timeout=SECONDS]\n\n";

  ConfigureFromArgs(argc,argv);
  
  // Define all test executables
  // Tests are tagged with whether they support backend override
  // Old CAIF_NeuralNetwork tests only support CPU backends (BLAS/Eigen)
  // Device-based tests use CUDA automatically when compiled with USE_CAIF_CUDA
  struct TestEntry
  {
    std::string name;
    std::string path;
    bool supports_cuda_backend;
  };

  std::vector<TestEntry> tests={
    // CPU-only tests (use CAIF_NeuralNetwork, no CUDA backend support)
    {"Tensor Tests", "./test_tensor", false},
    {"Shapes Tests", "./test_shapes", false},
    {"Activation Functions Tests", "./test_activation_functions", false},
    {"Loss Functions Tests", "./test_loss_functions", false},
    {"Optimizers Tests", "./test_optimizers", false},
    {"Dense Layer Tests", "./test_dense_layer", false},
    {"Convolution Layer Tests", "./test_conv_layer", false},
    {"Neural Network Tests", "./test_neural_network", false},
    {"Neural Network Training Tests", "./test_neural_network_training", false},
    {"Learning Process Tests", "./test_learning_process", false},
    {"Integration Tests", "./test_integration", false},
    {"Model Serializer Tests", "./test_model_serializer", false},
    {"Pythagorean Example Tests", "./test_pythagorean_example", false},
    {"Pythagorean Detailed Tests", "./test_pythagorean_detailed", false},
    // Device-based tests (use CUDA automatically via USE_CAIF_CUDA)
    {"Device RMSNorm Tests", "./test_device_rmsnorm", true},
    {"Device LayerNorm Tests", "./test_device_layernorm", true},
    {"Multi-Head Attention Tests", "./test_device_attention", true},
    {"FFN Tests", "./test_device_ffn", true},
    {"Transformer Block Tests", "./test_device_transformer_block", true},
    {"RoPE Tests", "./test_device_rope", true},
    {"GQA Tests", "./test_device_gqa", true},
    {"Token Embedding Tests", "./test_device_token_embedding", true},
    {"Patch Embedding Tests", "./test_device_patch_embedding", true},
    {"Positional Encoding Tests", "./test_device_positional_encoding", true},
    {"Linear Head Tests", "./test_device_linear_head", true},
    {"Transformer Model Tests", "./test_device_transformer_model", true},
    {"Cross-Entropy Loss Tests", "./test_device_cross_entropy", true},
    {"KV-Cache Tests", "./test_device_kv_cache", true},
    {"Multi-Modal Embedding Tests", "./test_device_multimodal_embedding", true},
    {"Transformer Training Tests", "./test_transformer_training", true}
  };

  std::vector<TestResult> results;

  // Run each test
  for(const auto &test:tests)
  {
    // Only pass backend arg to tests that don't support CUDA backend override
    // Device tests use CUDA automatically when compiled with USE_CAIF_CUDA
    std::string backend_for_test;
    if(test.supports_cuda_backend==false&&g_backend_arg!="cuda"&&g_backend_arg!="CUDA")
    {
      backend_for_test=g_backend_arg;
    }
    TestResult result=RunTest(test.name,test.path,backend_for_test);
    results.push_back(result);
  }
  
  // Print comprehensive summary
  PrintTestSummary(results);
  
  // Return appropriate exit code
  bool all_passed=true;
  for(const auto &result:results)
  {
    if(!result.passed)
    {
      all_passed=false;
      break;
    }
  }
  
  return all_passed?0:1;
} 
