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
// CAIF_DeviceContainer — abstract polymorphic plumbing base for layers
// composed of other layers.
//
// Container itself has no dtype-specific internal buffers — its job is to
// own a polymorphic `vector<unique_ptr<CAIF_DeviceLayer>>` of sublayers and
// chain them sequentially. It deliberately stays NON-TEMPLATED so callers
// that need a polymorphic container handle can hold one across cells (e.g.
// a `Network<float, __half>` and a `Network<float, float>` both expose the
// same `CAIF_DeviceContainer&` interface for sublayer iteration).
//
// The 5 templated container subclasses derive from this base AND from
// CAIF_DeviceLayerTyped<C, S> via their own concrete template parameters:
//   - CAIF_DeviceNetwork<C, S>
//   - CAIF_DevicePreNormBlock<C, S>
//   - CAIF_DeviceTransformerBlock<C, S>
//   - CAIF_DeviceTransformerModel<C, S>
//   - CAIF_DeviceViTModel<C, S>
//
// Each subclass templates its own intermediate-buffer allocations on its
// <C, S>; the un-templated Container base handles the polymorphic sublayer
// list, parameter aggregation, and aux-loss summation.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace instance
{

class CAIF_DeviceContainer:public CAIF_DeviceLayer
{
  public:
    typedef std::vector<std::unique_ptr<CAIF_DeviceLayer>> SublayerVec_t;
    typedef std::vector<bool> TrainableVec_t;

    explicit CAIF_DeviceContainer(CAIF_CudaStream &stream);
    ~CAIF_DeviceContainer()override=default;

    CAIF_DeviceContainer(const CAIF_DeviceContainer &)=delete;
    CAIF_DeviceContainer &operator=(const CAIF_DeviceContainer &)=delete;
    CAIF_DeviceContainer(CAIF_DeviceContainer &&other);
    CAIF_DeviceContainer &operator=(CAIF_DeviceContainer &&other);

    void AddLayer(std::unique_ptr<CAIF_DeviceLayer> layer);

    // Replace the sublayer at `index` with `layer`. The previous
    // sublayer's unique_ptr is destroyed (its tensors released) and
    // `layer` takes its slot. Used for in-place model surgery — e.g.
    // swapping a dense FFN sublayer for a MoE sublayer during add-MoE
    // fine-tuning, or swapping a trainable layer for its frozen
    // equivalent. `index` must be < LayerCount(); mismatch throws.
    // The trainable flag for the slot is preserved.
    void ReplaceLayer(size_t index,std::unique_ptr<CAIF_DeviceLayer> layer);

    size_t LayerCount()const{return Sublayers().size();}

    // Internal accessors — single point of access for the sublayer
    // and trainable vectors. Used inside Container's own method bodies
    // and from subclasses (PreNormBlock etc.) per the protected-member
    // discipline.
    SublayerVec_t &Sublayers(){return _sublayers;}
    const SublayerVec_t &Sublayers()const{return _sublayers;}
    void SetSublayers(SublayerVec_t &&v){_sublayers=std::move(v);}
    TrainableVec_t &Trainable(){return _trainable;}
    const TrainableVec_t &Trainable()const{return _trainable;}
    void SetTrainable(TrainableVec_t &&v){_trainable=std::move(v);}

    CAIF_DeviceLayer &Layer(size_t index);
    const CAIF_DeviceLayer &Layer(size_t index)const;

    void SetLayerTrainable(size_t index,bool trainable);
    bool IsLayerTrainable(size_t index)const;

    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;

    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;

    void ZeroGradients()override;

    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    bool IsParameterTrainable(size_t index)const override;

    size_t TotalParameterCount()const override;

    std::string Description()const override;

    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    size_t FrozenTensorCount()const override;
    CAIF_DeviceTensor FrozenTensorFP32(size_t index)const override;
    std::vector<std::string> FrozenTensorNames(const std::string &prefix="")const override;

    float AuxLoss()const override;

    // Containers hold heterogeneous-dtype children, so they have no
    // single answer for either dtype. Return Float32 as a sentinel;
    // callers that need per-leaf dtypes recurse into Layer(i) and
    // call RuntimeStorageDtype()/RuntimeComputeDtype() on each child.
    CAIF_DataType::CAIF_DataType_e RuntimeStorageDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }
    CAIF_DataType::CAIF_DataType_e RuntimeComputeDtype()const override
    {
      return CAIF_DataType::CAIF_DataType_e::Float32;
    }

  protected:
    std::pair<CAIF_DeviceLayer *,size_t> ResolveParameterIndex(size_t index);
    std::pair<const CAIF_DeviceLayer *,size_t> ResolveParameterIndex(size_t index)const;

    SublayerVec_t _sublayers;
    TrainableVec_t _trainable;

  private:
};

}//end instance namespace
