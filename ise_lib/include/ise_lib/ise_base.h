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

#ifndef ISE_BASE
#define ISE_BASE

#include "ise_out.h"
#include "ise_exception.h"

namespace instance
{
#if defined(WIN32)
  #if defined(CT_EXPORTS)
  #define ISE_EXPORTS __declspec(dllexport)
  #else
  #define ISE_EXPORTS __declspec(dllimport)
  #endif
#else
  #if defined(CT_EXPORTS)
  #define ISE_EXPORTS __attribute__((visibility("default")))
  #else
  #define ISE_EXPORTS
  #endif
#endif

/** @ISE_Base
  Simply a base class for all classes in the Terminal Bit libraries.
*/
class ISE_Base
{
  public:

    /** Constructor
    */
    ISE_Base(){}

    /** Destructor
    */
    virtual ~ISE_Base(){}

    /** Do some output
    */
    static void Output(char* str,...)
    {
      va_list vlst;
      va_start(vlst,str);
      ISE_Out::Output(str,vlst);
    }

    /*! \brief used to avoid compiler errors for unused variables.
    */
    //void DoNothing()const{};
    constexpr void NoOp()const{};


    virtual inline std::ostream& Out()const{return ISE_Out::Out(LogLevel());}
    virtual inline std::ostream& Out(){return ISE_Out::Out(LogLevel());}
    static inline std::ostream& SOut(){return ISE_Out::Out(_ll);}
    virtual inline std::ostream& Out(const ISE_Out::ISE_LogLevel &l)const{return ISE_Out::Out(l);}
    virtual inline std::ostream& Out(const ISE_Out::ISE_LogLevel &l){return ISE_Out::Out(l);}
    static inline std::ostream& SOut(const ISE_Out::ISE_LogLevel &l){return ISE_Out::Out(l);}

    virtual inline std::ostream& Log()const{return ISE_Out::Log(LogLevel());}
    virtual inline std::ostream& Log(){return ISE_Out::Log(LogLevel());}
    static inline std::ostream& SLog(){return ISE_Out::Log(_ll);}
    virtual inline std::ostream& Log(const ISE_Out::ISE_LogLevel &l)const{return ISE_Out::Out(l);}
    virtual inline std::ostream& Log(const ISE_Out::ISE_LogLevel &l){return ISE_Out::Out(l);}
    static inline std::ostream& SLog(const ISE_Out::ISE_LogLevel &l){return ISE_Out::Out(l);}

    virtual inline std::ostream& ErrorLog()const{return ISE_Out::ErrorLog(ErrorLogLevel());}
    virtual inline std::ostream& ErrorLog(){return ISE_Out::ErrorLog(ErrorLogLevel());}
    static inline std::ostream& SErrorLog(){return ISE_Out::ErrorLog(_ell);}
    virtual inline std::ostream& ErrorLog(const ISE_Out::ISE_LogLevel &l)const{return ISE_Out::ErrorLog(l);}
    virtual inline std::ostream& ErrorLog(const ISE_Out::ISE_LogLevel &l){return ISE_Out::ErrorLog(l);}
    static inline std::ostream& SErrorLog(const ISE_Out::ISE_LogLevel &l){return ISE_Out::ErrorLog(l);}

    virtual inline std::ostream& DbgLog()const{return ISE_Out::Out(DebugLogLevel());}
    virtual inline std::ostream& DbgLog(){return ISE_Out::Out(DebugLogLevel());}
    static inline std::ostream& SDbgLog(){return ISE_Out::Out(_dll);}
    virtual inline std::ostream& DbgLog(const ISE_Out::ISE_LogLevel &l)const{return ISE_Out::Out(l);}
    virtual inline std::ostream& DbgLog(const ISE_Out::ISE_LogLevel &l){return ISE_Out::Out(l);}
    static inline std::ostream& SDbgLog(const ISE_Out::ISE_LogLevel &l){return ISE_Out::Out(l);}

    //this is an explicit way to get rid of unused variable warnings on a per variable basis
    //its inline and the if(0) and everything inside of it should be compiled out by
    //compiler optimizations making this is a 0 op function in the end.
    template<class T>
    void Use(T t){if(0){if(&t!=nullptr){}}}

    virtual void DisableLogLogging()
    {
      //Log()<<"Removing:"<<LogLevel().Mask()<<std::endl;
      ISE_Out::RemoveLogLevel(LogLevel());
    }

    virtual void EnableLogLogging()
    {
      ISE_Out::AddLogLevel(LogLevel());
    }

    virtual void DisableErrorLogging()
    {
      //Log()<<"Removing:"<<ErrorLogLevel().Mask()<<std::endl;
      ISE_Out::RemoveLogLevel(ErrorLogLevel());
    }

    virtual void EnableErrorLogging()
    {
      ISE_Out::AddLogLevel(ErrorLogLevel());
    }

    virtual void DisableDebugLogging()
    {
      //Log()<<"Removing:"<<DebugLogLevel().Mask()<<std::endl;
      ISE_Out::RemoveLogLevel(DebugLogLevel());
    }

    virtual void EnableDebugLogging()
    {
      ISE_Out::AddLogLevel(DebugLogLevel());
    }

  protected:

    //These are none static virtual so that dervide bases of can simply override them
    //with their own static varialbes OR an instance of any class derived from this one
    //can have its own log level that overrides the default

    /** General logging level
     */
    virtual const ISE_Out::ISE_LogLevel& LogLevel()const{return _ll;}

    /** Error log level mask
     */
    virtual const ISE_Out::ISE_LogLevel& ErrorLogLevel()const{return _ell;}

    /** Debug log level mask
     */
    virtual const ISE_Out::ISE_LogLevel& DebugLogLevel()const{return _dll;}

  private:

    /** Normal log level
     */
    static ISE_Out::ISE_LogLevel _ll;

    /** Error log level
     */
    static ISE_Out::ISE_LogLevel _ell;

    /** Debug log level
     */
    static ISE_Out::ISE_LogLevel _dll;
};
}//end namespace instance
#endif
