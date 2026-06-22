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
// Directory-walk test runner: discovers every built test_* executable in the
// current directory that has a matching source file, runs each under a
// timeout, and prints a pass/fail/timeout summary.
//------------------------------------------------------------------------------
#include "ise_lib/ise_out.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace instance
{

constexpr int g_caif_runner_default_timeout_seconds=30;
constexpr int g_caif_runner_timeout_exit_code=124;
constexpr int g_caif_runner_exit_status_shift=8;
constexpr int g_caif_runner_exit_status_mask=0xFF;
constexpr double g_caif_runner_ms_per_second=1000.0;
constexpr double g_caif_runner_percent_scale=100.0;

//------------------------------------------------------------------------------
// One test executable's outcome.
//------------------------------------------------------------------------------
class CAIF_TestResult
{
  public:
    typedef std::vector<CAIF_TestResult> CAIF_TestResultVec_t;

    const std::string &TestName()const{return _test_name;}
    void SetTestName(const std::string &test_name){_test_name=test_name;}
    bool Passed()const{return _passed;}
    void SetPassed(const bool passed){_passed=passed;}
    bool TimedOut()const{return _timed_out;}
    void SetTimedOut(const bool timed_out){_timed_out=timed_out;}
    int ExitCode()const{return _exit_code;}
    void SetExitCode(const int exit_code){_exit_code=exit_code;}
    double ExecutionTime()const{return _execution_time;}
    void SetExecutionTime(const double execution_time){_execution_time=execution_time;}

  protected:

  private:
    std::string _test_name;
    bool _passed=false;
    bool _timed_out=false;
    int _exit_code=0;
    double _execution_time=0.0;
};

//------------------------------------------------------------------------------
// Discovers and runs the test executables.
//------------------------------------------------------------------------------
class CAIF_TestRunner
{
  public:
    static int Run(const int argc,char *argv[]);

  protected:

  private:
    static int TimeoutSeconds(){return _timeout_seconds;}
    static void SetTimeoutSeconds(const int timeout_seconds){_timeout_seconds=timeout_seconds;}

    static bool IsExecutable(const std::string &path);
    static bool HasBinaryExtension(const std::string &name);
    static bool IsTestBinaryCandidate(const std::string &name);
    static std::vector<std::string> DiscoverTestBinaries();
    static CAIF_TestResult RunOne(const std::string &executable_path);
    static void ConfigureFromArgs(const int argc,char *argv[]);
    static void PrintTestSummary(const CAIF_TestResult::CAIF_TestResultVec_t &results);

    static int _timeout_seconds;
};

int CAIF_TestRunner::_timeout_seconds=g_caif_runner_default_timeout_seconds;

bool CAIF_TestRunner::IsExecutable(const std::string &path)
{
  struct stat st;
  if(stat(path.c_str(),&st)!=0)
  {
    return false;
  }
  if((st.st_mode&S_IFREG)==0)
  {
    return false;
  }
  // Skip zero-byte stubs left behind by failed builds.
  if(st.st_size==0)
  {
    return false;
  }
  return (st.st_mode&(S_IXUSR|S_IXGRP|S_IXOTH))!=0;
}

bool CAIF_TestRunner::HasBinaryExtension(const std::string &name)
{
  const std::vector<std::string> skip_suffixes={".o",".d",".bin",".json",".cpp",".h",".py"};
  for(size_t i=0;i<skip_suffixes.size();++i)
  {
    const std::string &suffix=skip_suffixes[i];
    if(name.size()>=suffix.size() &&
       name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0)
    {
      return true;
    }
  }
  return false;
}

bool CAIF_TestRunner::IsTestBinaryCandidate(const std::string &name)
{
  const bool starts_with_test=(name.rfind("test_",0)==0);
  const bool is_model_dir=(name=="test_network_models" ||
                           name=="test_serializer_models");
  return starts_with_test==true &&
         is_model_dir==false &&
         HasBinaryExtension(name)==false;
}

std::vector<std::string> CAIF_TestRunner::DiscoverTestBinaries()
{
  std::vector<std::string> out;
  DIR *dir=opendir(".");
  if(dir==nullptr)
  {
    return out;
  }
  struct dirent *entry=nullptr;
  while((entry=readdir(dir))!=nullptr)
  {
    const std::string name(entry->d_name);
    const std::string path="./"+name;
    if(IsTestBinaryCandidate(name)==true &&
       IsExecutable(path)==true)
    {
      out.push_back(path);
    }
  }
  closedir(dir);
  std::sort(out.begin(),out.end());
  return out;
}

CAIF_TestResult CAIF_TestRunner::RunOne(const std::string &executable_path)
{
  CAIF_TestResult result;
  result.SetTestName(executable_path);
  result.SetTimedOut(false);

  ISE_Out::Out()<<"=== Running "
                <<executable_path
                <<" ===\n";

  const auto start_time=std::chrono::high_resolution_clock::now();
  const std::string cmd="timeout "+std::to_string(TimeoutSeconds())+" "+executable_path;
  const int raw_status=std::system(cmd.c_str());
  const int actual_exit_code=(raw_status>>g_caif_runner_exit_status_shift)&
                             g_caif_runner_exit_status_mask;
  const auto end_time=std::chrono::high_resolution_clock::now();
  const auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time-start_time);

  result.SetExitCode(actual_exit_code);
  result.SetExecutionTime(duration.count()/g_caif_runner_ms_per_second);

  if(actual_exit_code==g_caif_runner_timeout_exit_code)
  {
    result.SetTimedOut(true);
    result.SetPassed(false);
    ISE_Out::Out()<<"Result: TIMEOUT (exceeded "
                  <<TimeoutSeconds()
                  <<"s)\n\n";
    return result;
  }

  result.SetPassed(actual_exit_code==0);
  if(result.Passed()==true)
  {
    ISE_Out::Out()<<"Result: PASSED ("
                  <<result.ExecutionTime()
                  <<"s)\n\n";
  }
  else
  {
    ISE_Out::Out()<<"Result: FAILED (exit="
                  <<actual_exit_code
                  <<", "
                  <<result.ExecutionTime()
                  <<"s)\n\n";
  }
  return result;
}

void CAIF_TestRunner::ConfigureFromArgs(const int argc,char *argv[])
{
  for(int i=1;i<argc;++i)
  {
    const std::string arg(argv[i]);
    const std::string timeout_prefix="--timeout=";
    if(arg.rfind(timeout_prefix,0)==0)
    {
      const std::string value=arg.substr(timeout_prefix.size());
      SetTimeoutSeconds(std::stoi(value));
      ISE_Out::Out()<<"Timeout set to "
                    <<TimeoutSeconds()
                    <<" seconds\n";
    }
  }
}

void CAIF_TestRunner::PrintTestSummary(const CAIF_TestResult::CAIF_TestResultVec_t &results)
{
  ISE_Out::Out()<<"========================================\n"
                <<"          TEST SUMMARY                  \n"
                <<"========================================\n\n";

  int passed_count=0;
  int failed_count=0;
  int timeout_count=0;
  double total_time=0.0;

  for(size_t i=0;i<results.size();++i)
  {
    const CAIF_TestResult &result=results[i];
    ISE_Out::Out()<<result.TestName()
                  <<": ";
    if(result.Passed()==true)
    {
      ISE_Out::Out()<<"PASSED";
      ++passed_count;
    }
    else if(result.TimedOut()==true)
    {
      ISE_Out::Out()<<"TIMEOUT";
      ++timeout_count;
      ++failed_count;
    }
    else
    {
      ISE_Out::Out()<<"FAILED (exit="
                    <<result.ExitCode()
                    <<")";
      ++failed_count;
    }
    ISE_Out::Out()<<" ("
                  <<result.ExecutionTime()
                  <<"s)\n";
    total_time+=result.ExecutionTime();
  }

  ISE_Out::Out()<<"\n========================================\n"
                <<"Total Tests: "
                <<results.size()
                <<"\n"
                <<"Passed: "
                <<passed_count
                <<"\n"
                <<"Failed: "
                <<failed_count;
  if(timeout_count>0)
  {
    ISE_Out::Out()<<" ("
                  <<timeout_count
                  <<" timed out)";
  }
  ISE_Out::Out()<<"\n";
  if(results.empty()==false)
  {
    ISE_Out::Out()<<"Success Rate: "
                  <<(static_cast<double>(passed_count)/
                     static_cast<double>(results.size())*g_caif_runner_percent_scale)
                  <<"%\n";
  }
  ISE_Out::Out()<<"Total Execution Time: "
                <<total_time
                <<"s\n"
                <<"Timeout per test: "
                <<TimeoutSeconds()
                <<"s\n"
                <<"========================================\n";
}

int CAIF_TestRunner::Run(const int argc,char *argv[])
{
  ISE_Out::Out()<<"===========================================\n"
                <<"  CAIF Test Suite (directory-walk runner)  \n"
                <<"===========================================\n\n"
                <<"Usage: ./run_all_tests [--timeout=SECONDS]\n\n";

  ConfigureFromArgs(argc,argv);

  std::vector<std::string> paths=DiscoverTestBinaries();
  if(paths.empty()==true)
  {
    ISE_Out::Out()<<"No test_* executables found in current directory.\n";
    return 1;
  }

  ISE_Out::Out()<<"Discovered "
                <<paths.size()
                <<" test executable(s).\n\n";

  CAIF_TestResult::CAIF_TestResultVec_t results;
  for(size_t i=0;i<paths.size();++i)
  {
    results.push_back(RunOne(paths[i]));
  }

  PrintTestSummary(results);

  for(size_t i=0;i<results.size();++i)
  {
    if(results[i].Passed()==false)
    {
      return 1;
    }
  }
  return 0;
}

}//end instance namespace

int main(int argc,char *argv[])
{
  return instance::CAIF_TestRunner::Run(argc,argv);
}
