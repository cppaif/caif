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
// Named RAII scopes for CAIF_RunContext. Lambdas are forbidden by coding
// guidelines; these small classes replace the scope-guard idiom.
//------------------------------------------------------------------------------
#ifndef CAIF_RUN_CONTEXT_SCOPE_H
#define CAIF_RUN_CONTEXT_SCOPE_H

#include "caif_run_context.h"

namespace instance
{

// Pushes a subsystem tag on construction, pops on destruction. The pop is
// wrapped in try{}catch(...){} because destructors must not throw during
// stack unwinding — if the stack is already broken, swallowing the pop
// failure is correct.
class CAIF_RunContextSubsystemScope
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

// Sets the pass direction on construction, restores the previous direction
// on destruction. Used at the top-level Forward/Backward entry points and
// anywhere a nested pass flip is legitimate (e.g. gradcheck perturbation
// forwards that must match the analytical backward's precision).
class CAIF_RunContextPassScope
{
  public:
    CAIF_RunContextPassScope(CAIF_RunContext &ctx,
                             const CAIF_RunContext::Pass_e pass):_ctx(ctx),
                                                                 _previous(ctx.Pass())
    {
      _ctx.SetPass(pass);
    }

    ~CAIF_RunContextPassScope()
    {
      _ctx.SetPass(_previous);
    }

    CAIF_RunContextPassScope(const CAIF_RunContextPassScope &)=delete;
    CAIF_RunContextPassScope &operator=(const CAIF_RunContextPassScope &)=delete;
    CAIF_RunContextPassScope(CAIF_RunContextPassScope &&)=delete;
    CAIF_RunContextPassScope &operator=(CAIF_RunContextPassScope &&)=delete;

  protected:

  private:
    CAIF_RunContext &_ctx;
    CAIF_RunContext::Pass_e _previous;
};

}//end instance namespace

#endif  // CAIF_RUN_CONTEXT_SCOPE_H
