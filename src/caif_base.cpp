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

#include "caif/caif_base.h"

namespace instance
{

// Initialize AIF-specific log levels
ISE_Out::ISE_LogLevel CAIF_Base::_ll=ISE_Out::ReserveLogLevel("CAIF_LOG",true);
ISE_Out::ISE_LogLevel CAIF_Base::_ell=ISE_Out::ReserveLogLevel("CAIF_ERROR",true);
ISE_Out::ISE_LogLevel CAIF_Base::_dll=ISE_Out::ReserveLogLevel("CAIF_DEBUG",false);

}//end instance namespace
