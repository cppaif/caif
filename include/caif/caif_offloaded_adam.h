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
// CAIF - Adam optimizer with first/second-moment state offloaded to host
// pinned memory.
//
// Sibling to CAIF_AdamOptimizer. Subclasses CAIF_Optimizer directly so its
// per-parameter state (`_host_m[i]`, `_host_v[i]`) lives in pinned host RAM
// from construction; UpdateOne streams them onto the GPU per param at Step
// time, runs the existing CAIF_Ops::AdamUpdate kernel, writes the updated
// m/v back to host, and frees the transient GPU scratch.
//
// The base class's AMP master (CAIF_Optimizer::_master) is unaffected — for
// non-fp32 trainable params (e.g. bf16 add-MoE adapters) the fp32 master
// still lives on GPU as the existing flow expects. Offloading master
// requires base-class refactor and is post-MVP work; m + v alone are
// 2/3 of Adam's per-param state and the dominant chunk.
//
// Typical use: a trainer running with `--cpu-offload optimizer` (or
// `--cpu-offload both`) hands a CAIF_OffloadedAdam to its optimizer-init
// path instead of the default CAIF_AdamOptimizer.
//------------------------------------------------------------------------------
#pragma once

#include "caif_optimizer.h"
#include "caif_optimizer_type.h"
#include "caif_host_pinned_tensor.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace instance
{

class CAIF_OffloadedAdam:public CAIF_Optimizer
{
  public:
    typedef std::vector<std::unique_ptr<CAIF_HostPinnedTensor>> HostTensorPtrVec_t;

    CAIF_OffloadedAdam(const float lr,
                       const float beta1,
                       const float beta2,
                       const float epsilon,
                       const float weight_decay,
                       CAIF_CudaStream &stream);
    ~CAIF_OffloadedAdam()override=default;

    CAIF_OptimizerType::CAIF_OptimizerType_e Type()const override
    {
      return CAIF_OptimizerType::CAIF_OptimizerType_e::Adam;
    }

  protected:
    void AllocateState(const CAIF_DeviceTensor &param)override;
    void UpdateOne(CAIF_DeviceTensor &target,
                   const CAIF_DeviceTensor &grad,
                   const size_t idx)override;

  private:
    float Beta1()const{return _beta1;}
    float Beta2()const{return _beta2;}
    float Epsilon()const{return _epsilon;}
    const HostTensorPtrVec_t &HostM()const{return _host_m;}
    HostTensorPtrVec_t &HostMMut(){return _host_m;}
    const HostTensorPtrVec_t &HostV()const{return _host_v;}
    HostTensorPtrVec_t &HostVMut(){return _host_v;}

    float _beta1;
    float _beta2;
    float _epsilon;

    HostTensorPtrVec_t _host_m;
    HostTensorPtrVec_t _host_v;
};

}//end instance namespace