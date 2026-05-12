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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

constexpr int g_default_test_timeout_seconds=30;

struct TestResult
{
  std::string test_name;
  bool passed;
  bool timed_out;
  int exit_code;
  double execution_time;
};

static int g_timeout_seconds=g_default_test_timeout_seconds;

static bool IsExecutable(const std::string &path)
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

static bool HasBinaryExtension(const std::string &name)
{
  const char *const skip_suffixes[]={".o",".d",".bin",".json",".cpp",".h",".py"};
  const size_t count=sizeof(skip_suffixes)/sizeof(skip_suffixes[0]);
  for(size_t i=0;i<count;++i)
  {
    const std::string suffix(skip_suffixes[i]);
    if(name.size()>=suffix.size() &&
       name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0)
    {
      return true;
    }
  }
  return false;
}

static bool HasSourceFile(const std::string &binary_name)
{
  const std::string src=binary_name+".cpp";
  struct stat st;
  return stat(src.c_str(),&st)==0;
}

static bool IsTestBinaryCandidate(const std::string &name)
{
  const bool starts_with_test=(name.rfind("test_",0)==0);
  const bool is_model_dir=(name=="test_network_models" ||
                           name=="test_serializer_models");
  return starts_with_test==true &&
         is_model_dir==false &&
         HasBinaryExtension(name)==false;
}

static std::vector<std::string> DiscoverTestBinaries()
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
       IsExecutable(path)==true &&
       HasSourceFile(name)==true)
    {
      out.push_back(path);
    }
  }
  closedir(dir);
  std::sort(out.begin(),out.end());
  return out;
}

static TestResult RunTest(const std::string &executable_path)
{
  TestResult result;
  result.test_name=executable_path;
  result.timed_out=false;

  std::cout<<"=== Running "
           <<executable_path
           <<" ===\n";

  const auto start_time=std::chrono::high_resolution_clock::now();
  const std::string cmd="timeout "+std::to_string(g_timeout_seconds)+" "+
                        executable_path;
  const int raw_status=std::system(cmd.c_str());
  const int actual_exit_code=(raw_status>>8)&0xFF;
  const auto end_time=std::chrono::high_resolution_clock::now();
  const auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time-start_time);

  result.exit_code=actual_exit_code;
  result.execution_time=duration.count()/1000.0;

  if(actual_exit_code==124)
  {
    result.timed_out=true;
    result.passed=false;
    std::cout<<"Result: TIMEOUT (exceeded "
             <<g_timeout_seconds
             <<"s)\n\n";
  }
  else
  {
    result.passed=(actual_exit_code==0);
    if(result.passed==true)
    {
      std::cout<<"Result: PASSED ("
               <<result.execution_time
               <<"s)\n\n";
    }
    else
    {
      std::cout<<"Result: FAILED (exit="
               <<actual_exit_code
               <<", "
               <<result.execution_time
               <<"s)\n\n";
    }
  }
  return result;
}

static void ConfigureFromArgs(int argc,char *argv[])
{
  for(int i=1;i<argc;++i)
  {
    const std::string arg(argv[i]);
    const std::string timeout_prefix="--timeout=";
    if(arg.rfind(timeout_prefix,0)==0)
    {
      const std::string value=arg.substr(timeout_prefix.size());
      g_timeout_seconds=std::stoi(value);
      std::cout<<"Timeout set to "
               <<g_timeout_seconds
               <<" seconds\n";
    }
  }
}

static void PrintTestSummary(const std::vector<TestResult> &results)
{
  std::cout<<"========================================\n"
           <<"          TEST SUMMARY                  \n"
           <<"========================================\n\n";

  int passed_count=0;
  int failed_count=0;
  int timeout_count=0;
  double total_time=0.0;

  for(size_t i=0;i<results.size();++i)
  {
    const TestResult &result=results[i];
    std::cout<<result.test_name
             <<": ";
    if(result.passed==true)
    {
      std::cout<<"PASSED";
      ++passed_count;
    }
    else if(result.timed_out==true)
    {
      std::cout<<"TIMEOUT";
      ++timeout_count;
      ++failed_count;
    }
    else
    {
      std::cout<<"FAILED (exit="
               <<result.exit_code
               <<")";
      ++failed_count;
    }
    std::cout<<" ("
             <<result.execution_time
             <<"s)\n";
    total_time+=result.execution_time;
  }

  std::cout<<"\n========================================\n"
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
    std::cout<<" ("
             <<timeout_count
             <<" timed out)";
  }
  std::cout<<"\n";
  if(results.empty()==false)
  {
    std::cout<<"Success Rate: "
             <<(static_cast<double>(passed_count)/
                static_cast<double>(results.size())*100.0)
             <<"%\n";
  }
  std::cout<<"Total Execution Time: "
           <<total_time
           <<"s\n"
           <<"Timeout per test: "
           <<g_timeout_seconds
           <<"s\n"
           <<"========================================\n";
}

int main(int argc,char *argv[])
{
  std::cout<<"===========================================\n"
           <<"  CAIF Test Suite (directory-walk runner)  \n"
           <<"===========================================\n\n"
           <<"Usage: ./run_all_tests [--timeout=SECONDS]\n\n";

  ConfigureFromArgs(argc,argv);

  std::vector<std::string> paths=DiscoverTestBinaries();
  if(paths.empty()==true)
  {
    std::cout<<"No test_* executables found in current directory.\n";
    return 1;
  }

  std::cout<<"Discovered "
           <<paths.size()
           <<" test executable(s).\n\n";

  std::vector<TestResult> results;
  for(size_t i=0;i<paths.size();++i)
  {
    results.push_back(RunTest(paths[i]));
  }

  PrintTestSummary(results);

  for(size_t i=0;i<results.size();++i)
  {
    if(results[i].passed==false)
    {
      return 1;
    }
  }
  return 0;
}
