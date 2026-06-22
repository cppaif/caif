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

#pragma once

#include "caif_base.h"
#include "caif_param_role.h"
#include "caif_data_type.h"
#include <string>

namespace instance
{

// Per-role record carried by CAIF_RoleRegistry. One instance per
// Role_e value; the registry owns a vector of these indexed by the
// enum's underlying value. Every field is caif-domain — Name() is
// the only piece a caller can override via the registry's SetName /
// LoadNamesFromJSON surface. Family / Kind / DefaultTrainable /
// DefaultDtype are intrinsic to the role and immutable.
class CAIF_RoleInfo:public CAIF_Base
{
  public:
    CAIF_RoleInfo():_role(CAIF_ParamRole::Role_e::Unknown_e),
                    _name(),
                    _family(CAIF_ParamRole::Family_e::Unknown_e),
                    _kind(CAIF_ParamRole::Kind_e::Unknown_e),
                    _default_trainable(false),
                    _default_dtype(CAIF_DataType::CAIF_DataType_e::Float32)
    {
    }

    CAIF_RoleInfo(const CAIF_ParamRole::Role_e role,
                  const std::string &name,
                  const CAIF_ParamRole::Family_e family,
                  const CAIF_ParamRole::Kind_e kind,
                  const bool default_trainable,
                  const CAIF_DataType::CAIF_DataType_e default_dtype):_role(role),
                                                                     _name(name),
                                                                     _family(family),
                                                                     _kind(kind),
                                                                     _default_trainable(default_trainable),
                                                                     _default_dtype(default_dtype)
    {
    }

    ~CAIF_RoleInfo()=default;
    CAIF_RoleInfo(const CAIF_RoleInfo &other)=default;
    CAIF_RoleInfo &operator=(const CAIF_RoleInfo &other)=default;

    CAIF_ParamRole::Role_e Role()const{return _role;}
    const std::string &Name()const{return _name;}
    void SetName(const std::string &name){_name=name;}
    CAIF_ParamRole::Family_e Family()const{return _family;}
    CAIF_ParamRole::Kind_e Kind()const{return _kind;}
    bool DefaultTrainable()const{return _default_trainable;}
    CAIF_DataType::CAIF_DataType_e DefaultDtype()const{return _default_dtype;}

  protected:

  private:
    CAIF_ParamRole::Role_e _role;
    std::string _name;
    CAIF_ParamRole::Family_e _family;
    CAIF_ParamRole::Kind_e _kind;
    bool _default_trainable;
    CAIF_DataType::CAIF_DataType_e _default_dtype;
};

}//end instance namespace
