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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// Named RAII scope for CAIF_RunContext subsystem tags. Lambdas are forbidden
// by coding guidelines; this small class replaces the scope-guard idiom.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_run_context.h"

namespace instance
{

// Pushes a subsystem tag on construction, pops on destruction. The pop is
// wrapped in try{}catch(...){} because destructors must not throw during
// stack unwinding — if the stack is already broken, swallowing the pop
// failure is correct.
class CAIF_RunContextSubsystemScope:public CAIF_Base
{
  public:
    CAIF_RunContextSubsystemScope(CAIF_RunContext &ctx,
                                  const CAIF_RunContext::Subsystem_e s):_ctx(ctx)
    {
      _ctx.PushSubsystem(s);
    }

    ~CAIF_RunContextSubsystemScope()
    {
      try
      {
        _ctx.PopSubsystem();
      }
      catch(...)
      {
      }
    }

    CAIF_RunContextSubsystemScope(const CAIF_RunContextSubsystemScope &)=delete;
    CAIF_RunContextSubsystemScope &operator=(const CAIF_RunContextSubsystemScope &)=delete;
    CAIF_RunContextSubsystemScope(CAIF_RunContextSubsystemScope &&)=delete;
    CAIF_RunContextSubsystemScope &operator=(CAIF_RunContextSubsystemScope &&)=delete;

  protected:

  private:
    CAIF_RunContext &_ctx;
};

}//end instance namespace
