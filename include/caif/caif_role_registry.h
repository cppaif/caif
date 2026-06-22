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
#include "caif_role_info.h"
#include <map>
#include <string>
#include <vector>

namespace instance
{

// Process-global registry of every caif parameter role + its info
// record. Populated once at first Instance() call from a hand-written
// canonical table whose values come from g_caif_role_name_* constants
// in caif_constants.h. All fields except Name() are immutable after
// init; callers may override Name() via SetName / LoadNamesFromJSON to
// map caif's neutral default names onto an external format (HF
// safetensors, GGUF, etc). Layering: this registry holds NO external
// format vocabulary itself — the external strings come from
// caller-supplied JSON profiles.
//
// Lookup paths:
//   Info(role)            O(1) — primary vector lookup
//   Name(role)            O(1)
//   Family(role)          O(1)
//   Kind(role)            O(1)
//   RolesByFamily(f)      O(log N) — map<Family_e, vector<Role_e>>
//   RolesByKind(k)        O(log N) — map<Kind_e, vector<Role_e>>
//   RoleByName(name)      O(log N) — map<string, Role_e>; throws on miss
//   HasRoleForName(name)  O(log N) — same lookup, returns bool
class CAIF_RoleRegistry:public CAIF_Base
{
  public:
    typedef std::vector<CAIF_RoleInfo> RoleInfoVec_t;
    typedef std::vector<CAIF_ParamRole::Role_e> RoleVec_t;
    typedef std::map<CAIF_ParamRole::Family_e,RoleVec_t> FamilyToRolesMap_t;
    typedef std::map<CAIF_ParamRole::Kind_e,RoleVec_t> KindToRolesMap_t;
    typedef std::map<std::string,CAIF_ParamRole::Role_e> NameToRoleMap_t;

    static CAIF_RoleRegistry &Instance();

    const CAIF_RoleInfo &Info(const CAIF_ParamRole::Role_e role)const;
    const std::string &Name(const CAIF_ParamRole::Role_e role)const;
    CAIF_ParamRole::Family_e Family(const CAIF_ParamRole::Role_e role)const;
    CAIF_ParamRole::Kind_e Kind(const CAIF_ParamRole::Role_e role)const;
    bool DefaultTrainable(const CAIF_ParamRole::Role_e role)const;
    CAIF_DataType::CAIF_DataType_e DefaultDtype(const CAIF_ParamRole::Role_e role)const;

    const RoleVec_t &RolesByFamily(const CAIF_ParamRole::Family_e family)const;
    const RoleVec_t &RolesByKind(const CAIF_ParamRole::Kind_e kind)const;

    CAIF_ParamRole::Role_e RoleByName(const std::string &name)const;
    bool HasRoleForName(const std::string &name)const;

    // Override the Name() field for a single role. Rebuilds the
    // NameToRoleMap_t index so future RoleByName(new_name) succeeds
    // and the old name is no longer mapped.
    void SetName(const CAIF_ParamRole::Role_e role,const std::string &name);

  protected:

  private:
    CAIF_RoleRegistry();
    ~CAIF_RoleRegistry()=default;
    CAIF_RoleRegistry(const CAIF_RoleRegistry &)=delete;
    CAIF_RoleRegistry &operator=(const CAIF_RoleRegistry &)=delete;

    // Internal accessors. Direct _member access from method bodies
    // is forbidden per CODING_GUIDELINES.md §"Member Access via
    // Accessors"; mutating-through accessors named *Mut().
    const RoleInfoVec_t &InfoVec()const{return _info_by_role;}
    RoleInfoVec_t &InfoVecMut(){return _info_by_role;}
    const FamilyToRolesMap_t &ByFamily()const{return _by_family;}
    FamilyToRolesMap_t &ByFamilyMut(){return _by_family;}
    const KindToRolesMap_t &ByKind()const{return _by_kind;}
    KindToRolesMap_t &ByKindMut(){return _by_kind;}
    const NameToRoleMap_t &ByName()const{return _by_name;}
    NameToRoleMap_t &ByNameMut(){return _by_name;}

    // Populate _info_by_role from the canonical table. Called once
    // from the constructor; never re-runs. Values come from
    // g_caif_role_name_* constants in caif_constants.h.
    void PopulateCanonicalTable();

    // Insert one role's row into _info_by_role at its enum-indexed slot.
    // Used only from PopulateCanonicalTable. Range-checks the index.
    void SetRow(const CAIF_ParamRole::Role_e role,
                const std::string &name,
                const CAIF_ParamRole::Family_e family,
                const CAIF_ParamRole::Kind_e kind,
                const bool default_trainable,
                const CAIF_DataType::CAIF_DataType_e default_dtype);

    // Rebuild the secondary indexes (by_family, by_kind, by_name)
    // from _info_by_role. Called from the constructor and from any
    // SetName / LoadNamesFromJSON that changes a name.
    void RebuildIndexes();

    RoleInfoVec_t _info_by_role;
    FamilyToRolesMap_t _by_family;
    KindToRolesMap_t _by_kind;
    NameToRoleMap_t _by_name;
};

}//end instance namespace
