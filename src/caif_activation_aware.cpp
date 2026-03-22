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

#include "caif_activation_aware.h"
#include <cmath>

namespace instance
{
bool CAIF_ActivationAware::UsesKaiming(const CAIF_ActivationType_e activation)
{
  switch(activation)
  {
    case CAIF_ActivationType_e::ReLU:
    case CAIF_ActivationType_e::LeakyReLU:
    case CAIF_ActivationType_e::ELU:
    case CAIF_ActivationType_e::GELU:
    case CAIF_ActivationType_e::Swish:
      return true;
    default:
      return false;
  }
}

float CAIF_ActivationAware::Gain(const CAIF_ActivationType_e activation)
{
  switch(activation)
  {
    case CAIF_ActivationType_e::Linear:
      return g_caif_aa_gain_linear;
    case CAIF_ActivationType_e::Sigmoid:
      return g_caif_aa_gain_sigmoid;
    case CAIF_ActivationType_e::Tanh:
      return g_caif_aa_gain_tanh;
    case CAIF_ActivationType_e::Softmax:
      return g_caif_aa_gain_softmax;
    case CAIF_ActivationType_e::ReLU:
      return g_caif_aa_gain_relu;
    case CAIF_ActivationType_e::LeakyReLU:
      return g_caif_aa_gain_leakyrelu;
    case CAIF_ActivationType_e::ELU:
      return g_caif_aa_gain_elu;
    case CAIF_ActivationType_e::GELU:
      return g_caif_aa_gain_gelu;
    case CAIF_ActivationType_e::Swish:
      return g_caif_aa_gain_swish;
    default:
      return g_caif_aa_gain_linear;
  }
}

float CAIF_ActivationAware::UniformBound(
                                        const CAIF_ActivationType_e activation,
                                        const uint32_t fan_in,
                                        const uint32_t fan_out,
                                        const bool activation_aware_enabled
                                      )
{
    const float fan_in_f=static_cast<float>(fan_in);
    const float fan_out_f=static_cast<float>(fan_out);
    float safe_fan_in=fan_in_f;
    if(fan_in_f<=0.0f)
    {
      safe_fan_in=1.0f;
    }

  if(activation_aware_enabled==false)
  {
    return 1.0f/std::sqrt(safe_fan_in);
  }

  const float gain=Gain(activation);
  if(UsesKaiming(activation)==true)
  {
    return gain*std::sqrt(3.0f/safe_fan_in);
  }

    const float fan_sum=fan_in_f+fan_out_f;
    float safe_fan_sum=fan_sum;
    if(fan_sum<=0.0f)
    {
      safe_fan_sum=1.0f;
    }
  return gain*std::sqrt(6.0f/safe_fan_sum);
}
}//end instance namespace


