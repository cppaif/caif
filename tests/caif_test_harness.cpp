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
// Shared test-harness primitives implementation.
//------------------------------------------------------------------------------
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"
#include <cmath>

namespace instance
{

int CAIF_TestHarness::_passed_count=0;
int CAIF_TestHarness::_failed_count=0;

void CAIF_TestHarness::Reset()
{
  _passed_count=0;
  _failed_count=0;
}

void CAIF_TestHarness::Report(const char *test_name,const bool passed)
{
  if(passed==true)
  {
    ISE_Out::Out()<<"[PASS] "
                  <<test_name
                  <<"\n";
    ++_passed_count;
  }
  else
  {
    ISE_Out::Out()<<"[FAIL] "
                  <<test_name
                  <<"\n";
    ++_failed_count;
  }
}

int CAIF_TestHarness::PassedCount()
{
  return _passed_count;
}

int CAIF_TestHarness::FailedCount()
{
  return _failed_count;
}

int CAIF_TestHarness::FinalExitCode()
{
  ISE_Out::Out()<<"\nPassed: "
                <<_passed_count
                <<"  Failed: "
                <<_failed_count
                <<"\n";
  if(_failed_count==0)
  {
    return 0;
  }
  return 1;
}

bool CAIF_TestHarness::FloatEqual(const float a,const float b,const float tolerance)
{
  return std::fabs(a-b)<tolerance;
}

}//end instance namespace
