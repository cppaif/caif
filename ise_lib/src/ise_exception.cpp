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

#include "ise_exception.h"

#include <ostream>

using namespace instance;

std::ostream& operator<<(std::ostream &o,const ISE_Exception &e)
{
  o<<"Exception:"<<std::endl;
  for(unsigned int i=0;i<e.Stack().size();++i)
  {
    //this isn't working for some reason
    //o<<e.Stack()[i];
    o<<"/t"<<e.Stack()[i]._file_name;
    o<<" - ";
    o<<e.Stack()[i]._function_name;
    o<<" - ";
    o<<e.Stack()[i]._line;
    o<<" - ";
    o<<e.Stack()[i]._desc;
    o<<std::endl;
  }
  return o;
}

std::ostream& operator<<(std::ostream &o,const ISE_Exception::ISE_Stack_t &s)
{
  o<<s._file_name;
  o<<" - ";
  o<<s._function_name;
  o<<" - ";
  o<<s._line;
  o<<" - ";
  o<<s._desc;
  return o;
}
