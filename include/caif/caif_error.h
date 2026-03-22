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
 * @file aif_error.h
 * @brief Lightweight error carrier for public API expected error values
 */

#pragma once

#include "caif_exception.h"
#include "caif_base.h"
#include <string>
#include <sstream>

namespace instance
{
  /**
   * @brief Error wrapper for public APIs
   *
   * Holds an optional pointer to an CAIF_Exception and a human-readable message.
   * Designed to be used as the error type for std::expected in public APIs.
   */
  class CAIF_Error:public CAIF_Base
  {
    public:
      CAIF_Error():_exception(nullptr),_message(){}

      explicit CAIF_Error(const std::string &message)
        :_exception(nullptr),_message(message)
      {
      }

      explicit CAIF_Error(const char *message)
        :_exception(nullptr),_message(message==nullptr?std::string():std::string(message))
      {
      }

      explicit CAIF_Error(CAIF_Exception &ex)
        :_exception(&ex)
      {
        std::ostringstream oss;
        oss<<static_cast<const ISE_Exception&>(ex);
        _message=oss.str();
      }

      CAIF_Error &operator=(const std::string &message)
      {
        _message=message;
        _exception=nullptr;
        return *this;
      }

      operator std::string()const
      {
        return _message;
      }

      void SetException(CAIF_Exception *ex)
      {
        _exception=ex;
        if(ex!=nullptr)
        {
          std::ostringstream oss;
          oss<<static_cast<const ISE_Exception&>(*ex);
          _message=oss.str();
        }
      }

      CAIF_Exception *Exception()const{return _exception;}
      const std::string &Message()const{return _message;}
      void SetMessage(const std::string &message){_message=message;}

      [[noreturn]] void Throw()const
      {
        if(_exception!=nullptr)
        {
          RETHROW_CAIFES((*_exception));
        }
        else
        {
          THROW_CAIFE(_message.c_str());
        }
      }

    private:
      CAIF_Exception *_exception;        ///< Optional pointer to originating exception
      std::string _message;              ///< Human-readable error message
  };
}//end instance namespace


