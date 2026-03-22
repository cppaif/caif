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

#ifndef CAIF_BASE
#define CAIF_BASE

#include "ise_lib/ise_base.h"
#include "caif_exception.h"

namespace instance
{

class CAIF_Base:public ISE_Base
{

  public:

    static void AIFDisableLogLogging()
    {
      SDbgLog()<<"Removing DB LogLevel:"<<_ll.Mask()<<std::endl;
      ISE_Out::RemoveLogLevel(_ll);
    }

    static void AIFEnableLogLogging()
    {
      ISE_Out::AddLogLevel(_ll);
      SDbgLog()<<"Added DB LogLogLevel:"<<_ll.Mask()<<std::endl;
    }

    static void AIFDisableErrorLogging()
    {
      SDbgLog()<<"Removing DB ErrorLogLevel:"<<_ell.Mask()<<std::endl;
      ISE_Out::RemoveLogLevel(_ell);
    }

    static void AIFEnableErrorLogging()
    {
      ISE_Out::AddLogLevel(_ell);
      SDbgLog()<<"Added DB ErroLogLevel:"<<_ell.Mask()<<std::endl;
    }

    static void AIFDisableDebugLogging()
    {
      SDbgLog()<<"Removing DB DbgLogLevel:"<<_dll.Mask()<<std::endl;
      ISE_Out::RemoveLogLevel(_dll);
    }

    static void AIFEnableDebugLogging()
    {
      ISE_Out::AddLogLevel(_dll);
      SDbgLog()<<"Added DB DbgLogLevel:"<<_dll.Mask()<<std::endl;
    }


  protected:

    /** General logging level
     */ 
    virtual const ISE_Out::ISE_LogLevel& LogLevel()const{return _ll;}

    /** Error log level mask
     */
    virtual const ISE_Out::ISE_LogLevel& ErrorLogLevel()const{return _ell;}

    /** Debug log level mask
     */
    virtual const ISE_Out::ISE_LogLevel& DebugLogLevel()const{return _dll;}

    /** Normal log level
     */
    static ISE_Out::ISE_LogLevel _ll;

    /** Error log level
     */
    static ISE_Out::ISE_LogLevel _ell;

    /** Debug log level
     */
    static ISE_Out::ISE_LogLevel _dll;

};//end CAIF_Base
}//end instance namespace
#endif 
