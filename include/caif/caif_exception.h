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

#ifndef CAIF_EXCEPTION
#define CAIF_EXCEPTION

#include "ise_lib/ise_exception.h"

namespace instance
{

#define THROW_CAIFE(desc)\
{\
  CAIF_Exception ex(__LINE__,__FILE__,__FUNCTION__,desc,nullptr);\
  throw ex;\
}

#define RETHROW_CAIFE(e,desc)\
{\
  e.AddTrace(__LINE__,__FILE__,__FUNCTION__,desc,nullptr);\
  throw e;\
}

#define RETHROW_CAIFES(e)\
{\
  e.AddTrace(__LINE__,__FILE__,__FUNCTION__,"","");\
  throw e;\
}

#define CAIF_CATCH_BLOCK()\
  catch(CAIF_Exception &e)\
  {\
    ISE_Out::ErrLog()<<"CAIF Exception :"<<e<<std::endl;\
    RETHROW_CAIFE(e,"CAIF_Exception");\
  }\
  catch(ISE_Exception &iseex)\
  {\
    ISE_Out::ErrLog()<<"ISE Exception :"<<iseex<<std::endl;\
    RETHROW_ISEE(iseex,"ISE_Exception");\
  }\
  catch(std::exception &stdex)\
  {\
    ISE_Out::ErrLog()<<"std Exception :"<<stdex.what()<<std::endl;\
    RETHROW_STDE(stdex,"std::exception");\
  }\
  catch(...)\
  {\
    ISE_Out::ErrLog()<<"UNKNOWN ERROR"<<std::endl;\
    THROW_CAIFE("Unknown CAIF_Exception");\
  }\

class CAIF_Exception:public ISE_Exception
{
  public:

    /**
    * @brief Constructor
    *
    * @param line Line exception is thrown from (__LINE__)
    * @param file File exception is thrown from (__FILE__)
    * @param function Function exception is thrown from (__FUNCTION__)
    * @param desc Description of error.
    */
    CAIF_Exception(const unsigned int line,
                  const std::string &file,
                  const std::string &function,
                  const std::string &desc):ISE_Exception(line,file,function,desc)
    {
    }

    /**
    * @brief Constructor
    *
    * @param line Line exception is thrown from (__LINE__)
    * @param file File exception is thrown from (__FILE__)
    * @param function Function exception is thrown from (__FUNCTION__)
    * @param desc Description of error.
    * @param variable for replacements in desc
    */
    CAIF_Exception(const unsigned int line,
                  const std::string &file,
                  const std::string &function,
                  const std::string &desc,
                  ...):ISE_Exception()
    {
      va_list vars;
      va_start(vars,desc);
      AddTraceVA_LIST(line,file,function,desc,&vars);
    }

    /**
    * @brief Destructor
    */
    virtual ~CAIF_Exception(){}

  protected:

  private:

};//end CAIF_Exception
}//end instance namespace
#endif 