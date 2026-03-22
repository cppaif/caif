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

#ifndef ISE_OUTPUT
#define	ISE_OUTPUT

#include "ise_dead_stream.h"

#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <vector>
#include <mutex>
#include <bitset>

namespace instance
{

// Inlined from ise_constants.h — only constant CAIF needs
const std::string g_ise_default_log_level_name="[unset]";

class ISE_Out
{
  public:

    //these are really just generalizations that aren't used except to set the default
    //log level.  ISE_Base reserves the log levels it needs
    enum LogLevel_e:unsigned int
    {
      NONE=0,
      LOG=1<<0,
      ERROR=1<<1,
      DEBUG=1<<2,
      N_DEFAULT_LEVELS=3
    };

    /** class for available bit masks
     */
    class ISE_LogLevel
    {
      public:
        typedef std::vector<ISE_LogLevel> ISE_LogLevelVect_t;

        ISE_LogLevel(uint64_t mask=ISE_Out::LOG):_mask(mask),
                                                     _name("[unset]"),
                                                     _reserved(false)
        {

        }

        ISE_LogLevel(uint64_t mask,
                     const std::string &name,
                     bool r):_mask(mask),
                             _name(name),
                             _reserved(r)
        {

        }

        ISE_LogLevel(const ISE_Out::ISE_LogLevel &l):_mask(l.Mask()),
                                                     _name(l.Name()),
                                                     _reserved(l.Reserved())
        {

        }

        uint64_t Mask()const{return _mask;}
        const std::string& Name()const{return _name;}
        void SetName(const std::string &s){_name=s;}

        void Reserve(){_reserved=true;}
        void Reserve(const std::string &name){Reserve();SetName(name);}
        bool Reserved()const{return _reserved;}

        void operator=(const ISE_LogLevel &l)
        {
          _mask=l.Mask();
          _name=l.Name();
          _reserved=l.Reserved();
        }

        bool operator==(const ISE_LogLevel &l)const
        {
          return Mask()==l.Mask() &&
                 Name()==l.Name() &&
                 Reserved()==l.Reserved();
        }

      protected:

      private:

        //the bit mask
        uint64_t _mask;

        //optional name
        std::string _name;

        //indicates if this level has been reserved
        bool _reserved;
    };

    static const ISE_LogLevel& ReserveLogLevel(const std::string &name,bool add=false);
    static const ISE_LogLevel& ReserveLogLevel(const bool add)
    {
      return ReserveLogLevel(g_ise_default_log_level_name,add);
    }

    static const ISE_LogLevel& ReserveLogLevel(){return ReserveLogLevel(g_ise_default_log_level_name);}

    static inline std::ostream& Output(unsigned int l=(unsigned int)ISE_Out::LOG)
    {
      //*_stream<<"_level="<<std::hex<<_level<<" l="<<std::hex<<l<<std::endl;
      //It is import to understand that while code below works to disable to
      //the stream.  It will not work in a threaded environment.  There would be multiple
      //threads enabling and disabling the stream at the same time cause all sorts of
      //calimity in the output.  Instead a dead stream that does nothing has been created
      //when a log level not in use tries to log something it goes to the dummy stream that
      //is just a pass through

      /*if((_level&l)==0)
        _stream->setstate(std::ios_base::failbit);
      else
        _stream->clear();
      */
      if((_level&l)==0)
      {
        //*_stream<<"log level <<"<<std::bitset<32>(_level)<<" : "<<std::bitset<32>(l)<<" not enabled"<<std::endl;
        return ds;
      }
      else
        return *_stream;
    }

    static inline std::ostream& Out(const ISE_Out::ISE_LogLevel &l){return Output(l.Mask());}
    static inline std::ostream& Out(unsigned int l=(unsigned int)ISE_Out::LOG){return Output(l);}
    static inline std::ostream& Log(const ISE_Out::ISE_LogLevel &l){return Output(l.Mask());}
    static inline std::ostream& Log(unsigned int l=(unsigned int)ISE_Out::LOG){return Output(l);}

    static inline std::ostream& ErrLog(const ISE_Out::ISE_LogLevel &l){return ErrorLog(l.Mask());}
    static inline std::ostream& ErrorLog(const ISE_Out::ISE_LogLevel &l){return ErrorLog(l.Mask());}
    static inline std::ostream& ErrLog(unsigned int l=(unsigned int)ISE_Out::ERROR){return ErrorLog(l);}
    static inline std::ostream& ErrorLog(unsigned int l=(unsigned int)ISE_Out::ERROR)
    {
      //*_stream<<"_level="<<std::hex<<_level<<" l="<<std::hex<<l<<std::endl;
      //It is import to understand that while code below works to disable to
      //the stream.  It will not work in a threaded environment.  There would be multiple
      //threads enabling and disabling the stream at the same time cause all sorts of
      //calimity in the output.  Instead a dead stream that does nothing has been created
      //when a log level not in use tries to log something it goes to the dummy stream that
      //is just a pass through

      /*if((_level&l)==0)
        _stream->setstate(std::ios_base::failbit);
      else
        _stream->clear();
      */
      if((_level&l)==0)
      {
        //*_stream<<"log level <<"<<l<<"not enabled"<<std::endl;
        return ds;
      }
      else
        return *_stream;
    }

    static inline std::ostream& DbgLog(const ISE_Out::ISE_LogLevel &l){return Output(l.Mask());}
    static inline std::ostream& DbgLog(unsigned int l=(unsigned int)ISE_Out::DEBUG){return Output(l);}
    static inline std::ostream& DbgOut(const ISE_Out::ISE_LogLevel &l){return Output(l.Mask());}
    static inline std::ostream& DbgOut(unsigned int l=(unsigned int)ISE_Out::DEBUG){return Output(l);}

    /** Thread save output
    */
    static void Output(char * fstr,...)
    {
      std::lock_guard<std::mutex> lg(_mtx);
      fflush(stdout);
      va_list ap;
      va_start(ap,fstr);
      vprintf(fstr,ap);
      va_end(ap);
      fflush(stdout);
    }

    /** Created for custom logging functions
    */
    static void Output(char *fstr, va_list &vl,unsigned int l=ISE_Out::LOG)
    {
      if((_level&l)==0){
        return;
      }
      std::lock_guard<std::mutex> lg(_mtx);
      vprintf(fstr,vl);
    }

    /** These outputs are saved as warnings
    */
    static void OutputWarning(char *fstr,...)
    {
      va_list ap;
      va_start(ap,fstr);
      OutputWarning(true,fstr,ap);
    }
    /** output a warning
    */
    static void OutputWarning(bool save,char *fstr,...)
    {
      va_list ap;
      va_start(ap,fstr);
      OutputWarning(save,fstr,ap);
    }

    /** Set the stream to output to
    */
    static inline void SetStream(std::ostream& stream,unsigned int l=ISE_Out::LOG)
    {
      if((_level&l)==0){
        return;
      }
      std::lock_guard<std::mutex> lg(_mtx);
      _stream=&stream;
    }

    /** va_list warning output
    */
    static void OutputWarning(bool save, char *fstr,va_list &vl,unsigned int l=ISE_Out::LOG);

    /** add a log level
     */
    static void AddLogLevel(const ISE_Out::ISE_LogLevel &l)
    {
      //*_stream<<"Adding log level:"<<l.Mask()<<" "<<l.Name()<<std::endl;
      _level|=l.Mask();
      //*_stream<<"log level <<"<<std::bitset<32>(_level)<<" : "<<std::bitset<32>(l.Mask())<<" added"<<std::endl;
    }

    /** add a log level
     */
    static void RemoveLogLevel(const ISE_Out::ISE_LogLevel &l)
    {
      //*_stream<<"Removing log level:"<<l.Mask()<<" "<<l.Name()<<std::endl;
      _level=_level&~l.Mask();
      //*_stream<<"log level <<"<<std::bitset<32>(_level)<<" : "<<std::bitset<32>(l.Mask())<<" removed"<<std::endl;
    }

    /** Return the Log Log level this is hardcoded in the cpp file.
     */
    static const ISE_Out::ISE_LogLevel& LogLogLevel(){return LogLevels()[0];}

    /** Return the Error Log level this is hardcoded in the cpp file.
     */
    static const ISE_Out::ISE_LogLevel& ErrorLogLevel(){return LogLevels()[1];}

    /** Return the Debug Log level this is hardcoded in the cpp file.
     */
    static const ISE_Out::ISE_LogLevel& DebugLogLevel(){return LogLevels()[2];}


  protected:

    /**
      Constructor
    */
    ISE_Out();

    /** @ISE_Outputging
      Destructor
    */
    ~ISE_Out();

    /**
      Structure that holds errors or warnings.
    */
    struct ISE_Warning
    {
      /** The string that was output.
      */
      std::string _warning;

      ISE_Warning& operator=(const ISE_Warning &wrn)
      {
        _warning=wrn._warning;
        return *this;
      }

      bool operator==(const ISE_Warning &wrn)
      {
        return _warning==wrn._warning;
      }
    };

  private:

    /** add a log level
     */
    static void AddLogLevel(unsigned int l)
    {
      _level|=l;
    }

    /** add a log level
     */
    static void RemoveLogLevel(unsigned int l)
    {
      _level=_level&~l;
    }


    static unsigned int InitializeLogLevel();

    /** output  stream.
    */
    static std::ostream *_stream;

    /** Mutex for locking static memebers
    */
    static std::mutex _mtx;

    /** Disables outputs
    */
    static bool _silent;

    /** List of all saved warnings
    */
    static std::vector<ISE_Warning> _warnings;

    /** output error
    */
    static std::ostream *_error_stream;

    /** Log level .. there are enums but this
     * is just done the numerica value
     */
    static unsigned int _level;

    /** Dead stream .. DOES NOTHING used for unset log levels.
     */
    static ISE_DeadStream<char> ds;

    /** All available log levels and their reserve status
     */

    static ISE_Out::ISE_LogLevel::ISE_LogLevelVect_t& LogLevels();
};
}//end ISE namespace
#endif
