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
// AIF - AI Framework
// Generic weight name mapper between HuggingFace and AIF conventions
//------------------------------------------------------------------------------
#ifndef CAIF_WEIGHT_MAPPER_H
#define CAIF_WEIGHT_MAPPER_H

#include "caif_base.h"
#include <string>
#include <vector>
#include <map>
#include <set>

namespace instance
{

/**
 * @brief Generic weight name mapper for converting between HuggingFace
 * SafeTensors weight names and AIF parameter names.
 *
 * AIF layers already follow HF naming conventions (q_proj.weight, etc.)
 * and composite layers build hierarchical names via prefix propagation.
 * The main difference is typically a top-level prefix (e.g., HF uses
 * "model." while AIF uses no prefix) or occasional aliases (e.g., tied
 * weights where "lm_head.weight" == "embed_tokens.weight").
 *
 * Usage:
 *   CAIF_WeightMapper mapper;
 *   mapper.AddPrefixRule("model.", "");  // strip "model." prefix
 *   mapper.AddAlias("model.output.weight", "lm_head.weight");  // tied weights
 *
 *   std::string aif_name = mapper.HfToAif("model.layers.0.self_attn.q_proj.weight");
 *   // -> "layers.0.self_attn.q_proj.weight"
 */
class CAIF_WeightMapper:public CAIF_Base
{
  public:
    CAIF_WeightMapper()=default;
    ~CAIF_WeightMapper()=default;

    /**
     * @brief Add a prefix mapping rule.
     *
     * All HF names starting with hf_prefix will be mapped to AIF names
     * by replacing hf_prefix with aif_prefix. Rules are tried in order
     * of longest prefix match.
     *
     * @param hf_prefix The HuggingFace name prefix (e.g., "model.")
     * @param aif_prefix The AIF name prefix (e.g., "" for stripping)
     */
    void AddPrefixRule(const std::string &hf_prefix,const std::string &aif_prefix);

    /**
     * @brief Add a direct name alias (exact match).
     *
     * Use for special cases like tied weights or renamed parameters.
     *
     * @param hf_name The exact HF weight name
     * @param aif_name The corresponding AIF parameter name
     */
    void AddAlias(const std::string &hf_name,const std::string &aif_name);

    /**
     * @brief Convert an HF weight name to an AIF parameter name.
     *
     * Tries aliases first, then prefix rules (longest match wins).
     * Returns empty string if no rule matches.
     */
    std::string HfToAif(const std::string &hf_name)const;

    /**
     * @brief Convert an AIF parameter name to an HF weight name.
     *
     * Tries reverse aliases first, then reverse prefix rules.
     * Returns empty string if no rule matches.
     */
    std::string AifToHf(const std::string &aif_name)const;

    /**
     * @brief HF names needed for a list of AIF parameter names.
     * @return Vector of (hf_name,aif_name) pairs for all resolvable names
     */
    std::vector<std::pair<std::string,std::string>>
    RequiredHfNames(const std::vector<std::string> &aif_names)const;

    /**
     * @brief Find AIF parameter names that cannot be resolved to any
     * available HF name.
     *
     * @param aif_names Expected AIF parameter names
     * @param available_hf_names Set of HF names available in safetensors
     * @return List of AIF parameter names that have no matching HF weight
     */
    std::vector<std::string> MissingNames(const std::vector<std::string> &aif_names,
                                           const std::set<std::string> &available_hf_names)const;

    /**
     * @brief Find HF names in the available set that have no mapping rule.
     * Useful for detecting unexpected weights in a safetensors file.
     */
    std::vector<std::string>
    UnmappedHfNames(const std::set<std::string> &available_hf_names)const;

    /**
     * @brief Number of prefix rules configured.
     */
    size_t PrefixRuleCount()const{return _prefix_rules.size();}

    /**
     * @brief Number of aliases configured.
     */
    size_t AliasCount()const{return _hf_to_aif_aliases.size();}

  protected:

  private:
    struct PrefixRule_t
    {
      std::string hf_prefix;
      std::string aif_prefix;
    };

    static bool ComparePrefixByLength(const PrefixRule_t &a,const PrefixRule_t &b);

    // Prefix rules sorted by longest hf_prefix first (for longest-match)
    std::vector<PrefixRule_t> _prefix_rules;

    // Direct aliases: hf_name -> aif_name
    std::map<std::string,std::string> _hf_to_aif_aliases;

    // Reverse aliases: aif_name -> hf_name
    std::map<std::string,std::string> _aif_to_hf_aliases;
};

}//end instance namespace

#endif  // CAIF_WEIGHT_MAPPER_H
