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

#include "ise_out.h"
#include "ise_exception.h"

using namespace instance;

ISE_Out::ISE_LogLevel::ISE_LogLevelVect_t& ISE_Out::LogLevels()
{
static ISE_Out::ISE_LogLevel::ISE_LogLevelVect_t log_levels={
                                                                  ISE_Out::ISE_LogLevel(1UL<<0,"Log",true),
                                                                  ISE_Out::ISE_LogLevel(1UL<<1,"Error",true),
                                                                  ISE_Out::ISE_LogLevel(1UL<<2,"Debug",true),
                                                                  ISE_Out::ISE_LogLevel(1UL<<3),
                                                                  ISE_Out::ISE_LogLevel(1UL<<4),
                                                                  ISE_Out::ISE_LogLevel(1UL<<5),
                                                                  ISE_Out::ISE_LogLevel(1UL<<6),
                                                                  ISE_Out::ISE_LogLevel(1UL<<7),
                                                                  ISE_Out::ISE_LogLevel(1UL<<8),
                                                                  ISE_Out::ISE_LogLevel(1UL<<9),
                                                                  ISE_Out::ISE_LogLevel(1UL<<10),
                                                                  ISE_Out::ISE_LogLevel(1UL<<11),
                                                                  ISE_Out::ISE_LogLevel(1UL<<12),
                                                                  ISE_Out::ISE_LogLevel(1UL<<13),
                                                                  ISE_Out::ISE_LogLevel(1UL<<14),
                                                                  ISE_Out::ISE_LogLevel(1UL<<15),
                                                                  ISE_Out::ISE_LogLevel(1UL<<16),
                                                                  ISE_Out::ISE_LogLevel(1UL<<17),
                                                                  ISE_Out::ISE_LogLevel(1UL<<18),
                                                                  ISE_Out::ISE_LogLevel(1UL<<19),
                                                                  ISE_Out::ISE_LogLevel(1UL<<20),
                                                                  ISE_Out::ISE_LogLevel(1UL<<21),
                                                                  ISE_Out::ISE_LogLevel(1UL<<22),
                                                                  ISE_Out::ISE_LogLevel(1UL<<23),
                                                                  ISE_Out::ISE_LogLevel(1UL<<24),
                                                                  ISE_Out::ISE_LogLevel(1UL<<25),
                                                                  ISE_Out::ISE_LogLevel(1UL<<26),
                                                                  ISE_Out::ISE_LogLevel(1UL<<27),
                                                                  ISE_Out::ISE_LogLevel(1UL<<28),
                                                                  ISE_Out::ISE_LogLevel(1UL<<29),
                                                                  ISE_Out::ISE_LogLevel(1UL<<30),
                                                                  ISE_Out::ISE_LogLevel(1UL<<31),
                                                                  ISE_Out::ISE_LogLevel(1UL<<32),
                                                                  ISE_Out::ISE_LogLevel(1UL<<33),
                                                                  ISE_Out::ISE_LogLevel(1UL<<34),
                                                                  ISE_Out::ISE_LogLevel(1UL<<35),
                                                                  ISE_Out::ISE_LogLevel(1UL<<36),
                                                                  ISE_Out::ISE_LogLevel(1UL<<37),
                                                                  ISE_Out::ISE_LogLevel(1UL<<38),
                                                                  ISE_Out::ISE_LogLevel(1UL<<39),
                                                                  ISE_Out::ISE_LogLevel(1UL<<40),
                                                                  ISE_Out::ISE_LogLevel(1UL<<41),
                                                                  ISE_Out::ISE_LogLevel(1UL<<42),
                                                                  ISE_Out::ISE_LogLevel(1UL<<43),
                                                                  ISE_Out::ISE_LogLevel(1UL<<44),
                                                                  ISE_Out::ISE_LogLevel(1UL<<45),
                                                                  ISE_Out::ISE_LogLevel(1UL<<46),
                                                                  ISE_Out::ISE_LogLevel(1UL<<47),
                                                                  ISE_Out::ISE_LogLevel(1UL<<48),
                                                                  ISE_Out::ISE_LogLevel(1UL<<49),
                                                                  ISE_Out::ISE_LogLevel(1UL<<50),
                                                                  ISE_Out::ISE_LogLevel(1UL<<51),
                                                                  ISE_Out::ISE_LogLevel(1UL<<52),
                                                                  ISE_Out::ISE_LogLevel(1UL<<53),
                                                                  ISE_Out::ISE_LogLevel(1UL<<54),
                                                                  ISE_Out::ISE_LogLevel(1UL<<55),
                                                                  ISE_Out::ISE_LogLevel(1UL<<56),
                                                                  ISE_Out::ISE_LogLevel(1UL<<57),
                                                                  ISE_Out::ISE_LogLevel(1UL<<58),
                                                                  ISE_Out::ISE_LogLevel(1UL<<59),
                                                                  ISE_Out::ISE_LogLevel(1UL<<60),
                                                                  ISE_Out::ISE_LogLevel(1UL<<61),
                                                                  ISE_Out::ISE_LogLevel(1UL<<62),
                                                                  ISE_Out::ISE_LogLevel(1UL<<63)
                                                                };

  return log_levels;
}

//To get the log level intialized with basic logging and error logging enabled we need to get
//the log levels from the list and initialize it
unsigned int ISE_Out::InitializeLogLevel()
{
  //call the LogLevels function to force proper order of static initialization
  AddLogLevel(LogLevels()[0]);
  AddLogLevel(LogLevels()[1]);
  //we are setting the level by calling AddLogLevel and then
  //overwrittint it with itself as part of controlling the static initializations
  return _level;
}

std::mutex ISE_Out::_mtx;
std::vector<ISE_Out::ISE_Warning> ISE_Out::_warnings;
std::ostream *ISE_Out::_error_stream=&std::cerr;
std::ostream *ISE_Out::_stream=&std::cout;
unsigned int ISE_Out::_level=ISE_Out::InitializeLogLevel();
ISE_DeadStream<char> ISE_Out::ds(nullptr);


ISE_Out::ISE_Out()
{
}

ISE_Out::~ISE_Out()
{

}

const ISE_Out::ISE_LogLevel& ISE_Out::ReserveLogLevel(const std::string &name,bool add/*=false*/)
{
  //throw 1;
  //*_stream<<"reserving "<<name<<" add="<<add<<std::endl;
  for(unsigned int i=0;i<LogLevels().size();++i)
  {
    if(LogLevels()[i].Reserved()==false)
    {
      LogLevels()[i].Reserve(name);
      //*_stream<<" mask="<<LogLevels()[i].Mask()<<" current level="<<_level;
      if(add==true)
      {
        //*_stream<<"adding "<<name<<" add="<<add<<std::endl;
        AddLogLevel(LogLevels()[i]);
      }

      //*_stream<<" new level="<<_level<<std::endl;
      return LogLevels()[i];
    }
  }
  THROW_ISEE("OUT OF LOG LEVELS????");
}

/** @file
  Output a warning and save the warning to a list.
*/
void ISE_Out::OutputWarning(bool save, char *fstr,va_list &vl,unsigned int l/*=ISE_Out::NORMAL*/)
{
  if((_level&l)==0){
     return;
  }
  try
  {
    char str[2048];
    {
      std::lock_guard<std::mutex> lg(_mtx);

      vsprintf(str,fstr,vl);

      if(save==true)
      {
        ISE_Warning sw;
        sw._warning=str;
        _warnings.push_back(sw);
      }
    }

    Output(str);
  }
  catch(ISE_Exception &ex)
  {
    RETHROW_ISEE(ex,"ISE_Exception")
  }
  catch(std::exception &stex)
  {
    RETHROW_STDE(stex,"std::exception");
  }
  catch(...)
  {
    THROW_ISEE("Unknown exception");
  }
}
