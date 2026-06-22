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
#include "caif_optimizer_type.h"
#include "caif_constants.h"
#include "caif_device_tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace instance
{

class CAIF_DeviceNetwork;

// Pure-virtual optimizer base. Holds the scaffolding every optimizer
// needs (learning rate, weight decay, step counter, AMP fp32-master +
// grad-fp32 buffers when the network's params are non-fp32). Concrete
// subclasses (Adam, SGD, Momentum, RMSprop, AdaGrad) own their own
// per-parameter state vectors and override AllocateState + UpdateOne.
class CAIF_Optimizer:public CAIF_Base
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

    // Loss-scaler hook: unscale every trainable gradient in place by
    // inv_scale and OR overflow into found_inf (a 1-element fp32 tensor sharing
    // the optimizer's stream). Reuses the same TrainableRefs() walk as Step(),
    // so it touches exactly the parameters Step() would update. The caller
    // (CAIF_LossScaler) zeroes found_inf first and, if it ends up set, skips
    // Step() entirely — no partial update, and the step counter does not move.
    void UnscaleGradsCheckInf(CAIF_DeviceNetwork &network,
                              float inv_scale,
                              CAIF_DeviceTensor &found_inf);

    float LearningRate()const{return _lr;}
    void SetLearningRate(const float lr){_lr=lr;}

    int StepCount()const{return _t;}

    virtual CAIF_OptimizerType::CAIF_OptimizerType_e Type()const=0;

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

    // Multi-tensor ("foreach") batched step. Default returns false, so Step()
    // keeps the per-parameter UpdateOne loop. A subclass overrides this to
    // update EVERY parameter in a single kernel launch. Only the all-fp32 case
    // is batched: BuildSharedBatch returns false when any param carries an AMP
    // fp32 master, so AMP stays on the per-param path unchanged. Returns true
    // iff it performed the whole step.
    virtual bool BatchedStep(CAIF_DeviceNetwork &network)
    {
      (void)network;
      return false;
    }

    // Build + upload the shared batch context for an all-fp32 step: device
    // arrays of param (target) pointers and grad pointers, plus the
    // element-count prefix sum offsets[n+1]. Returns false (caller must fall
    // back to the per-param loop) when there are no trainable params or any
    // param is non-fp32. The per-subclass state pointer arrays are built by the
    // override, in the same trainable order (state vector index == ref index).
    bool BuildSharedBatch(CAIF_DeviceNetwork &network);

    int BatchNumTensors()const{return _batch_num_tensors;}
    int64_t BatchTotalElements()const{return _batch_total_elements;}
    const int64_t *BatchOffsetsDevice();
    float *const *BatchTargetsDevice();
    const float *const *BatchGradsDevice();

    // Upload `bytes` of host data into a device scratch tensor, (re)sizing it
    // as needed, async on `stream`. Used by batched overrides to stage their
    // state pointer arrays. (An Int8 tensor is used as a raw byte buffer.)
    static void UploadScratch(CAIF_DeviceTensor &scratch,
                              const void *host,
                              size_t bytes,
                              CAIF_CudaStream &stream);

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

    std::vector<float *> &HostTargetsMut(){return _h_targets;}
    std::vector<const float *> &HostGradsMut(){return _h_grads;}
    std::vector<int64_t> &HostOffsetsMut(){return _h_offsets;}
    const std::vector<int64_t> &HostOffsets()const{return _h_offsets;}
    CAIF_DeviceTensor &DeviceOffsetsMut(){return _d_offsets;}
    CAIF_DeviceTensor &DeviceTargetsMut(){return _d_targets;}
    CAIF_DeviceTensor &DeviceGradsMut(){return _d_grads;}
    void SetBatchNumTensors(const int n){_batch_num_tensors=n;}
    void SetBatchTotalElements(const int64_t n){_batch_total_elements=n;}

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

    // Multi-tensor batched-step scratch (all-fp32 path). Host staging arrays +
    // device buffers (Int8 tensors used as raw byte buffers, reinterpreted to
    // int64_t* / float**). Built each step by BuildSharedBatch; empty until the
    // first batched step. Param/grad pointers can move between steps (grad
    // buffers especially), so they are re-staged every step — still one launch
    // versus N.
    std::vector<float *> _h_targets;
    std::vector<const float *> _h_grads;
    std::vector<int64_t> _h_offsets;
    CAIF_DeviceTensor _d_offsets;
    CAIF_DeviceTensor _d_targets;
    CAIF_DeviceTensor _d_grads;
    int _batch_num_tensors;
    int64_t _batch_total_elements;
};

}//end instance namespace
