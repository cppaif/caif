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

/**
 * @file rtnr_exception.h
 * @brief Project-specific exception and catch macros for Retrainer
 */

#pragma once

#include "ise_lib/ise_exception.h"
#include "ise_lib/ise_string.h"

namespace instance
{

class RTNR_Exception:public ISE_Exception
{
  public:
    RTNR_Exception(const unsigned int line,
                   const ISE_String &file,
                   const ISE_String &function,
                   const ISE_String &desc):ISE_Exception(line,file,function,desc)
    {
    }

    RTNR_Exception(const unsigned int line,
                   const ISE_String &file,
                   const ISE_String &function,
                   const ISE_String &desc,
                   ...):ISE_Exception()
    {
      va_list vars;
      va_start(vars,desc);
      AddTraceVA_LIST(line,file,function,desc,&vars);
    }

    virtual ~RTNR_Exception(){}

  protected:

  private:
};

}

#define THROW_RTNRE(desc) \
{ \
  instance::RTNR_Exception ex(__LINE__,__FILE__,__FUNCTION__,desc,nullptr); \
  throw ex; \
}

#define RETHROW_RTNRE(e,desc) \
{ \
  e.AddTrace(__LINE__,__FILE__,__FUNCTION__,desc,nullptr); \
  throw e; \
}

#define RETHROW_RTNRES(e) \
{ \
  e.AddTrace(__LINE__,__FILE__,__FUNCTION__,"",""); \
  throw e; \
}

#define RTNR_CATCH_BLOCK(context) \
  catch(instance::RTNR_Exception &e) \
  { \
    ISE_Out::ErrLog()<<"RTNR Exception :"<<e<<std::endl; \
    RETHROW_RTNRE(e,context); \
  } \
  catch(ISE_Exception &iseex) \
  { \
    ISE_Out::ErrLog()<<"ISE Exception :"<<iseex<<std::endl; \
    RETHROW_ISEE(iseex,context); \
  } \
  catch(std::exception &stdex) \
  { \
    ISE_Out::ErrLog()<<"std Exception :"<<stdex.what()<<std::endl; \
    RETHROW_STDE(stdex,context); \
  } \
  catch(...) \
  { \
    ISE_Out::ErrLog()<<"UNKNOWN ERROR"<<std::endl; \
    THROW_RTNRE("Unknown RTNR_Exception"); \
  }
