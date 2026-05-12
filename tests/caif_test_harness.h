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
// Shared test-harness primitives: result counters, a stream+context Setup(),
// and a FloatEqual helper. Replaces 55+ ad-hoc copies across tests/.
//------------------------------------------------------------------------------
#ifndef CAIF_TEST_HARNESS_H
#define CAIF_TEST_HARNESS_H

#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <cstdint>
#include <exception>

namespace instance
{

class CAIF_TestHarness
{
  public:
    static void Reset();
    static void Report(const char *test_name,const bool passed);
    static int PassedCount();
    static int FailedCount();
    static int FinalExitCode();

    static bool FloatEqual(const float a,const float b,const float tolerance);

  protected:
  private:
    static int _passed_count;
    static int _failed_count;
};

// Test-side analogue of CAIF_CATCH_BLOCK: logs the full call stack for
// CAIF_Exception / ISE_Exception via operator<<, reports the test as failed
// via CAIF_TestHarness::Report, and does NOT rethrow so the next test can
// continue. Usage:
//
//   try { ... assertion-carrying test body ... }
//   CAIF_TEST_CATCH_BLOCK("Suite::Case")
//
#define CAIF_TEST_CATCH_BLOCK(test_name)\
  catch(CAIF_Exception &e)\
  {\
    ISE_Out::ErrLog()<<"CAIF Exception :"<<e<<std::endl;\
    CAIF_TestHarness::Report(test_name,false);\
  }\
  catch(ISE_Exception &iseex)\
  {\
    ISE_Out::ErrLog()<<"ISE Exception :"<<iseex<<std::endl;\
    CAIF_TestHarness::Report(test_name,false);\
  }\
  catch(std::exception &stdex)\
  {\
    ISE_Out::ErrLog()<<"std Exception :"<<stdex.what()<<std::endl;\
    CAIF_TestHarness::Report(test_name,false);\
  }\
  catch(...)\
  {\
    ISE_Out::ErrLog()<<"UNKNOWN ERROR"<<std::endl;\
    CAIF_TestHarness::Report(test_name,false);\
  }

}//end instance namespace

#endif  // CAIF_TEST_HARNESS_H
