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

#include "ise_base.h"

using namespace instance;

//due to what can only be a bug gcc compliers std::string has to be explicitly created in these or
//calls of the strings evaluate to bools (yeah for real)
//
//FYI the defaul log level in ISE_OUT is 1 which should evaluate to Log here assuming it is
//the first one created.  But its more of a side effect and should be addressed in the future
//any strnageness reset and then explicity set the log levels at the start of the program
ISE_Out::ISE_LogLevel ISE_Base::_ll(ISE_Out::LogLogLevel());
ISE_Out::ISE_LogLevel ISE_Base::_ell(ISE_Out::ErrorLogLevel());
ISE_Out::ISE_LogLevel ISE_Base::_dll(ISE_Out::DebugLogLevel());
