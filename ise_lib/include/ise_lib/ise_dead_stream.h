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

#ifndef ISE_DEAD_STREAM
#define ISE_DEAD_STREAM

#include <ios>
#include <string>

namespace instance
{
template<class CharT,class Traits=std::char_traits<CharT>>
class ISE_DeadStream: virtual public std::basic_ostream <CharT,Traits>
{
  public:
    // types (inherited from basic_ios)
    using char_type   = CharT;
    using int_type    = typename Traits::int_type;
    using pos_type    = typename Traits::pos_type;
    using off_type    = typename Traits::off_type;
    using traits_type = Traits;

    explicit ISE_DeadStream(std::basic_streambuf<char_type, Traits>* sb):std::basic_ostream<CharT,Traits>(sb){}
    virtual ~ISE_DeadStream(){}

    // prefix/suffix
    class sentry;

    // formatted output
    ISE_DeadStream<CharT, Traits>&
      operator<<(std::basic_ostream<CharT, Traits>& (*pf)(std::basic_ostream<CharT, Traits>&));
    ISE_DeadStream<CharT, Traits>&
      operator<<(std::basic_ios<CharT, Traits>& (*pf)(std::basic_ios<CharT, Traits>&));
    ISE_DeadStream<CharT, Traits>&
      operator<<(std::ios_base& (*pf)(std::ios_base&));

    ISE_DeadStream<CharT, Traits>& operator<<(bool n){}
    ISE_DeadStream<CharT, Traits>& operator<<(short n){}
    ISE_DeadStream<CharT, Traits>& operator<<(unsigned short n){}
    ISE_DeadStream<CharT, Traits>& operator<<(int n){}
    ISE_DeadStream<CharT, Traits>& operator<<(unsigned int n){}
    ISE_DeadStream<CharT, Traits>& operator<<(long n){}
    ISE_DeadStream<CharT, Traits>& operator<<(unsigned long n){}
    ISE_DeadStream<CharT, Traits>& operator<<(long long n){}
    ISE_DeadStream<CharT, Traits>& operator<<(unsigned long long n){}
    ISE_DeadStream<CharT, Traits>& operator<<(float f){}
    ISE_DeadStream<CharT, Traits>& operator<<(double f){}
    ISE_DeadStream<CharT, Traits>& operator<<(long double f){}

    ISE_DeadStream<CharT, Traits>& operator<<(const void* p){}
    ISE_DeadStream<CharT, Traits>& operator<<(nullptr_t){}
    ISE_DeadStream<CharT, Traits>& operator<<(std::basic_streambuf<char_type, Traits>* sb){}

    // unformatted output
    ISE_DeadStream<CharT, Traits>& put(char_type c){}
    ISE_DeadStream<CharT, Traits>& write(const char_type* s,std::streamsize n){}

    ISE_DeadStream<CharT, Traits>& flush(){}

    // seeks
    pos_type tellp(){}
    ISE_DeadStream<CharT, Traits>& seekp(pos_type){}
    ISE_DeadStream<CharT, Traits>& seekp(off_type,std::ios_base::seekdir){}

  protected:


  private:

};//end ISE_DeadStream
};//end instance namespace
#endif
