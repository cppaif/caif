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

#include "caif_constants.h"
#include "caif_device_tensor.h"

#include <cstddef>
#include <vector>

namespace instance
{

class CAIF_DeviceNetwork;

// Pure-virtual optimizer base. Holds the scaffolding every optimizer
// needs (learning rate, weight decay, step counter, AMP fp32-master +
// grad-fp32 buffers when the network's params are non-fp32). Concrete
// subclasses (Adam, SGD, Momentum, RMSprop, AdaGrad) own their own
// per-parameter state vectors and override AllocateState + UpdateOne.
class CAIF_Optimizer
{
  public:
    // Pair-of-indices addressing a single trainable param:
    // (top-level layer index in the network, parameter index within
    // that layer's flattened ParameterTensor walk). The optimizer
    // captures these at Initialize() time so it can iterate exactly
    // the trainable subset at Step() without re-querying
    // IsLayerTrainable / IsParameterTrainable on every step.
    struct ParamRef_t
    {
      public:
        ParamRef_t()=default;
        ParamRef_t(const size_t layer_idx,
                   const size_t param_idx):_layer_idx(layer_idx),
                                           _param_idx(param_idx){}

        size_t LayerIndex()const{return _layer_idx;}
        size_t ParamIndex()const{return _param_idx;}
        void SetLayerIndex(const size_t i){_layer_idx=i;}
        void SetParamIndex(const size_t i){_param_idx=i;}

      private:
        size_t _layer_idx=0;
        size_t _param_idx=0;
    };

    typedef std::vector<ParamRef_t> ParamRefVec_t;

    CAIF_Optimizer(const float lr,
                   const float weight_decay,
                   CAIF_CudaStream &stream);
    virtual ~CAIF_Optimizer()=default;

    // Walk the network's trainable layers and allocate per-parameter
    // state. Idempotent: calling Initialize a second time discards
    // previous state and re-allocates.
    void Initialize(CAIF_DeviceNetwork &network);

    // One optimizer step across every (param, grad) pair owned by the
    // network's trainable layers. Increments StepCount() and dispatches
    // to UpdateOne() per parameter.
    void Step(CAIF_DeviceNetwork &network);

    float LearningRate()const{return _lr;}
    void SetLearningRate(const float lr){_lr=lr;}

    int StepCount()const{return _t;}

    virtual CAIF_OptimizerType_e Type()const=0;

  protected:
    // Subclass extension point: allocate any per-parameter state the
    // subclass needs (m+v for Adam, velocity for Momentum, etc).
    // Called once per trainable parameter tensor by Initialize().
    virtual void AllocateState(const CAIF_DeviceTensor &param)=0;

    // Subclass extension point: do the parameter update for the idx-th
    // trainable param. `target` is the fp32 master copy when AMP is
    // active (param's native dtype != fp32), otherwise `target` aliases
    // the param itself. `grad` is fp32 in both cases (the caller upcasts
    // when needed). Subclasses call CAIF_Ops::<Optimizer>Update.
    virtual void UpdateOne(CAIF_DeviceTensor &target,
                           const CAIF_DeviceTensor &grad,
                           const size_t idx)=0;

    float WeightDecay()const{return _weight_decay;}
    CAIF_CudaStream &Stream()const{return *_stream;}

  private:
    void IncrementStep(){_t++;}
    void ResetStepCount(){_t=0;}
    const std::vector<CAIF_DeviceTensor> &Masters()const{return _master;}
    std::vector<CAIF_DeviceTensor> &MastersMut(){return _master;}
    const std::vector<CAIF_DeviceTensor> &GradFp32s()const{return _grad_fp32;}
    std::vector<CAIF_DeviceTensor> &GradFp32sMut(){return _grad_fp32;}
    const ParamRefVec_t &TrainableRefs()const{return _trainable_refs;}
    ParamRefVec_t &TrainableRefsMut(){return _trainable_refs;}

    float _lr;
    float _weight_decay;
    int _t;
    CAIF_CudaStream *_stream;

    // AMP master + grad-fp32 buffers — one per trainable param, in
    // network-walk order. Empty (Tensor::TotalElements()==0) when the
    // param is already fp32 (fast path, no upcast/downcast needed);
    // otherwise master holds the fp32 copy that Update* writes to and
    // grad_fp32 is the per-step upcast scratch.
    std::vector<CAIF_DeviceTensor> _master;
    std::vector<CAIF_DeviceTensor> _grad_fp32;

    // (layer, param) refs of every parameter the optimizer owns state
    // for. Populated by Initialize via IsLayerTrainable +
    // IsParameterTrainable on each candidate param; iterated 1:1 by
    // Step. Indices into _master / _grad_fp32 / per-subclass state
    // vectors are positional within this list — the optimizer never
    // re-queries the network's trainable flags after Initialize.
    ParamRefVec_t _trainable_refs;
};

}//end instance namespace
