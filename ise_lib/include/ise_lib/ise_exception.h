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

#ifndef ISE_EXCEPTION
#define ISE_EXCEPTION

#include "ise_out.h"

#include <string>
#include <stdarg.h>
#include <vector>
//as of c++20 the new way of creating a string with varialbes
//is std::fromat.  Please use that and the simple string only
//macros from here on out.
#include <format>

namespace instance
{
/** @ISE_Exception
  Simply a base class for all classes in the Terminal Bit libraries.
*/
class ISE_Exception
{
  public:

    /** Constructor
    */
    ISE_Exception(const int line,
                 const std::string &file,
                 const std::string &function,
                 const std::string &desc
                 )
    {
      AddTraceVA_LIST(line,file,function,desc,nullptr);
    }

    ISE_Exception(const int line,
                 const char *file,
                 const char *function,
                 const std::string &desc,
                 ...)
    {
      //ISE_Out::Output()<<"DESC="<<*desc<<std::endl;
      va_list vars;
      va_start(vars,desc);
      AddTraceVA_LIST(line,file,function,desc,&vars);
    }

    /** Destructor
    */
    virtual ~ISE_Exception(){}

    virtual void AddTrace(const int line,
                          const std::string &file,
                          const std::string &function,
                          const std::string &desc,
                          ...)
    {
      //ISE_Out::Output()<<"Add trace ..."<<std::endl;
      va_list vars;
      va_start(vars,desc);

      //ISE_Out::Output()<<"Calling Add trace va_list desc="<<*desc<<std::endl;
      AddTraceVA_LIST(line,file,function,std::string(desc),&vars);
    }

    /** adds a stack trace
    */
    virtual void AddTraceVA_LIST(const int line,
                          const std::string &file,
                          const std::string &function,
                          const std::string &desc,
                          va_list *vars)
    {
      //ISE_Out::Output()<<"Add trace va_list"<<std::endl;
      ISE_Stack_t st(file,function.c_str(),line,desc,vars);
      //ISE_Out::Output()<<"Stack created"<<std::endl;
      _stack.push_back(st);
    }

    /** stack trace struct
    */
    struct ISE_Stack_t
    {
      /** file name
      */
      std::string _file_name;

      /** function name
      */
      std::string _function_name;

      /** line number
      */
      int _line;

      /** Description for this trace
       */
      std::string _desc;

      /** destructor
      */
      ISE_Stack_t(const std::string &file,
                 const char* func,
                 int line,
                 const std::string &desc,
                 va_list *vars):_file_name(file),
                                _function_name(func),
                                _line(line)
      {
        BuildDescription(desc,vars);
      }

      void BuildDescription(const std::string &desc, va_list *vars)
      {
        if(vars==nullptr)
        {
          _desc=desc;
          return;
        }
        char* frmt=(char*)calloc(desc.size()+2048,sizeof(std::string::value_type));
        vsprintf(frmt,desc.c_str(),*vars);
        _desc=frmt;
        free(frmt);
        va_end(*vars);
      }

      /** assignment operator
      */
      ISE_Stack_t& operator=(const ISE_Stack_t &st)
      {
        _file_name=st._file_name;
        _function_name=st._function_name;
        _line=st._line;
        _desc=st._desc;
        return *this;
      }

    };

    typedef std::vector<ISE_Stack_t> ISE_StackVec_t;

    const ISE_StackVec_t& Stack()const{return _stack;}


  protected:

    /** Used by derived classes with varadic constructors
     */
    ISE_Exception(){}


    /** stack trace, list of string with lines, functions and files
    */
    ISE_StackVec_t _stack;

  private:

};


#define THROW_ISEEV(desc,...)\
{\
  ISE_Exception ex(__LINE__,__FILE__,__FUNCTION__,desc,__VA_ARGS__);\
  throw ex;\
}

#define RETHROW_ISEEV(lex,desc,...)\
{\
  ISE_Out::Output()<<"re throwing:"<<lex<<std::endl;\
  lex.AddTrace(__LINE__,__FILE__ ,__FUNCTION__,desc,__VA_ARGS__);\
  ISE_Out::Output()<<lex<<std::endl;\
  throw lex;\
}


#define RETHROW_STDEV(lex,desc,...)\
{\
  ISE_Out::Output()<<"re throwing std::exception:"<<lex<<std::endl;\
  ISE_Exception(__LINE__,__FILE__ ,__FUNCTION__,desc,__VA_ARGS__);\
  ISE_Out::Output()<<lex<<std::endl;\
  throw lex;\
}

#define THROW_ISEE(desc)\
{\
  ISE_Exception ex(__LINE__,__FILE__,__FUNCTION__,desc,nullptr);\
  throw ex;\
}


#define RETHROW_ISEE(lex, desc)\
{\
  lex.AddTraceVA_LIST(__LINE__ , __FILE__ ,__FUNCTION__,desc,nullptr);\
  ISE_Out::Output()<<lex<<std::endl;\
  throw lex;\
}

#define RETHROW_STDE(lex, desc)\
{\
  std::string d=desc;\
  d+=" ";\
  d+=lex.what();\
  ISE_Exception(__LINE__ , __FILE__ ,__FUNCTION__,d,nullptr);\
  ISE_Out::Output()<<d<<std::endl;\
  throw lex;\
}

#define RETHROW_ISEES(lex)\
{\
  lex.AddTraceVA_LIST(__LINE__ , __FILE__ ,__FUNCTION__,nullptr,nullptr);\
  ISE_Out::Output()<<lex<<std::endl;\
  throw lex;\
}
}//end ISE namespace

std::ostream& operator<<(std::ostream &o,const instance::ISE_Exception &e);
std::ostream& operator<<(std::ostream &o,const instance::ISE_Exception::ISE_Stack_t &s);
#endif
