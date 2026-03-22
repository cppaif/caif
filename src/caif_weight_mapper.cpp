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
// Generic weight name mapper implementation
//------------------------------------------------------------------------------
#include "caif_weight_mapper.h"
#include <algorithm>

using namespace instance;

bool CAIF_WeightMapper::ComparePrefixByLength(const PrefixRule_t &a,const PrefixRule_t &b)
{
  return a.hf_prefix.size()>b.hf_prefix.size();
}

void CAIF_WeightMapper::AddPrefixRule(const std::string &hf_prefix,const std::string &aif_prefix)
{
  PrefixRule_t rule;
  rule.hf_prefix=hf_prefix;
  rule.aif_prefix=aif_prefix;
  _prefix_rules.push_back(rule);

  // Sort by longest hf_prefix first for longest-match semantics
  std::sort(_prefix_rules.begin(),_prefix_rules.end(),ComparePrefixByLength);
}

void CAIF_WeightMapper::AddAlias(const std::string &hf_name,const std::string &aif_name)
{
  _hf_to_aif_aliases[hf_name]=aif_name;
  _aif_to_hf_aliases[aif_name]=hf_name;
}

std::string CAIF_WeightMapper::HfToAif(const std::string &hf_name)const
{
  // Check aliases first (exact match)
  auto alias_it=_hf_to_aif_aliases.find(hf_name);
  if(alias_it!=_hf_to_aif_aliases.end())
  {
    return alias_it->second;
  }

  // Try prefix rules (already sorted by longest prefix first)
  for(const auto &rule:_prefix_rules)
  {
    if(hf_name.size()>=rule.hf_prefix.size()&&
       hf_name.compare(0,rule.hf_prefix.size(),rule.hf_prefix)==0)
    {
      return rule.aif_prefix+hf_name.substr(rule.hf_prefix.size());
    }
  }

  return "";
}

std::string CAIF_WeightMapper::AifToHf(const std::string &aif_name)const
{
  // Check reverse aliases first (exact match)
  auto alias_it=_aif_to_hf_aliases.find(aif_name);
  if(alias_it!=_aif_to_hf_aliases.end())
  {
    return alias_it->second;
  }

  // Try reverse prefix rules (sorted by longest aif_prefix first)
  // We need to try longest aif_prefix match, but rules are sorted by
  // hf_prefix length. Iterate all and pick longest aif_prefix match.
  const PrefixRule_t *best_rule=nullptr;
  for(const auto &rule:_prefix_rules)
  {
    if(aif_name.size()>=rule.aif_prefix.size()&&
       aif_name.compare(0,rule.aif_prefix.size(),rule.aif_prefix)==0)
    {
      if(best_rule==nullptr||
         rule.aif_prefix.size()>best_rule->aif_prefix.size())
      {
        best_rule=&rule;
      }
    }
  }

  if(best_rule!=nullptr)
  {
    return best_rule->hf_prefix+aif_name.substr(best_rule->aif_prefix.size());
  }

  return "";
}

std::vector<std::pair<std::string,std::string>>
CAIF_WeightMapper::RequiredHfNames(const std::vector<std::string> &aif_names)const
{
  std::vector<std::pair<std::string,std::string>> result;
  result.reserve(aif_names.size());

  for(const auto &aif_name:aif_names)
  {
    std::string hf_name=AifToHf(aif_name);
    if(hf_name.empty()==false)
    {
      result.push_back({hf_name,aif_name});
    }
  }

  return result;
}

std::vector<std::string> CAIF_WeightMapper::MissingNames(const std::vector<std::string> &aif_names,
                                                         const std::set<std::string> &available_hf_names)const
{
  std::vector<std::string> missing;

  for(const auto &aif_name:aif_names)
  {
    std::string hf_name=AifToHf(aif_name);
    if(hf_name.empty()==true||
       available_hf_names.find(hf_name)==available_hf_names.end())
    {
      missing.push_back(aif_name);
    }
  }

  return missing;
}

std::vector<std::string>
CAIF_WeightMapper::UnmappedHfNames(const std::set<std::string> &available_hf_names)const
{
  std::vector<std::string> unmapped;

  for(const auto &hf_name:available_hf_names)
  {
    std::string aif_name=HfToAif(hf_name);
    if(aif_name.empty()==true)
    {
      unmapped.push_back(hf_name);
    }
  }

  return unmapped;
}
