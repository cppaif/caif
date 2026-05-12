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
// Abstract base class for device-resident layers
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_LAYER_H
#define CAIF_DEVICE_LAYER_H

#include "caif_base.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include "caif_run_context.h"
#include "caif_run_context_scope.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Abstract base class for all device-resident layers
 *
 * Defines the interface that CAIF_DeviceNetwork uses to manage
 * heterogeneous layer stacks. Every layer stores its parameters and
 * gradients as CAIF_DeviceTensor objects on the GPU.
 *
 * Subclasses implement ForwardImpl / BackwardImpl, parameter access,
 * and gradient zeroing. The active runtime stream is read from the
 * per-call CAIF_RunContext (ctx.Stream()) during Forward/Backward;
 * layers also retain a construction-time stream reference for tensor
 * allocation and non-runtime utility methods (weight loading,
 * checkpoint I/O, optimizer-state allocation). The ctx is the source
 * of truth during a pass; the layer field is a construction-/utility-
 * time convenience.
 */
class CAIF_DeviceLayer:public CAIF_Base
{
  public:
    virtual ~CAIF_DeviceLayer()=default;

    // Non-copyable (device tensors are move-only)
    CAIF_DeviceLayer(const CAIF_DeviceLayer &)=delete;
    CAIF_DeviceLayer &operator=(const CAIF_DeviceLayer &)=delete;

    // Movable
    CAIF_DeviceLayer(CAIF_DeviceLayer &&other):_stream(other._stream)
    {
      other._stream=nullptr;
    }

    CAIF_DeviceLayer &operator=(CAIF_DeviceLayer &&other)
    {
      if(this!=&other)
      {
        _stream=other._stream;
        other._stream=nullptr;
      }
      return *this;
    }

    /**
     * @brief Forward pass entry point.
     *
     * Non-virtual. Wraps ForwardImpl with a CAIF_RunContextSubsystemScope so
     * every subclass automatically pushes/pops its SubsystemTag() onto the
     * ctx stack. Subclasses implement ForwardImpl, not Forward.
     */
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,
                              CAIF_RunContext &ctx)
    {
      CAIF_RunContextSubsystemScope scope(ctx,SubsystemTag());
      return ForwardImpl(input,ctx);
    }

    /**
     * @brief Backward pass entry point.
     *
     * Non-virtual. Wraps BackwardImpl with a CAIF_RunContextSubsystemScope
     * so the subsystem stack is maintained through backward pass as well.
     */
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output,
                               CAIF_RunContext &ctx)
    {
      CAIF_RunContextSubsystemScope scope(ctx,SubsystemTag());
      return BackwardImpl(grad_output,ctx);
    }

    virtual CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                          CAIF_RunContext &ctx)=0;
    virtual CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                           CAIF_RunContext &ctx)=0;
    virtual CAIF_RunContext::Subsystem_e SubsystemTag()const=0;

    virtual float AuxLoss()const{return 0.0f;}

    virtual void ZeroGradients()=0;

    virtual size_t ParameterTensorCount()const=0;
    virtual CAIF_DeviceTensor &ParameterTensor(size_t index)=0;
    virtual const CAIF_DeviceTensor &ParameterTensor(size_t index)const=0;
    virtual CAIF_DeviceTensor &GradientTensor(size_t index)=0;
    virtual const CAIF_DeviceTensor &GradientTensor(size_t index)const=0;

    // Per-parameter trainable flag. Defaults to true on leaf layers; the
    // CAIF_DeviceContainer override consults its per-sublayer _trainable
    // vector and recurses, so callers can freeze a single sublayer
    // (e.g. attn / norm / embedding while the MoE sublayer trains, or a
    // base FFN whose LoRA adapter is the only trainable surface). The
    // optimizer Initialize/Step both honor this flag.
    virtual bool IsParameterTrainable(size_t index)const
    {
      static_cast<void>(index);
      return true;
    }

    virtual size_t TotalParameterCount()const=0;

    virtual std::string Description()const=0;

    virtual std::vector<std::string> ParameterNames(const std::string &prefix="")const=0;

    // Frozen (non-trainable) tensors held by this layer or its children.
    // Returned dequantized to fp32 by value so external harnesses can
    // round-trip base weights for parity init against a reference
    // implementation; not visited by the optimizer.
    virtual size_t FrozenTensorCount()const{return 0;}
    virtual CAIF_DeviceTensor FrozenTensorFP32(size_t index)const
    {
      static_cast<void>(index);
      THROW_CAIFE("layer has no frozen tensors");
    }
    virtual std::vector<std::string> FrozenTensorNames(const std::string &prefix="")const
    {
      static_cast<void>(prefix);
      return {};
    }

    // Polymorphic re-randomization of trainable weights. Default is no-op
    // because most layers (norms, embeddings, dropout) initialize their
    // parameters at construction. Layers that take a seed-dependent path
    // (MHA, MLA, CrossAttention, DenseLayer, etc.) override this to forward
    // the seed to their per-template InitializeWeights routine. Strategies
    // call this through the polymorphic CAIF_DeviceLayer pointer so they
    // don't need to know the (ComputeT,StorageT) cell.
    virtual void InitializeWeights(uint32_t seed=0)
    {
      static_cast<void>(seed);
    }

    /**
     * @brief Enable / disable activation (gradient) checkpointing on this
     * layer.
     *
     * Default: no-op. Layers that support checkpointing (today:
     * CAIF_DevicePreNormBlock) override this and route to their own
     * SetCheckpointed. Callers walk the un-templated layer list and call
     * this on every layer; non-block layers silently no-op.
     */
    virtual void SetCheckpointed(const bool b)
    {
      static_cast<void>(b);
    }
    virtual bool Checkpointed()const{return false;}

    /**
     * @brief True when the layer holds a valid construction stream.
     *
     * Returns false only after a move-from. Runtime Forward/Backward code
     * no longer checks this; it reads ctx.Stream() and trusts the
     * top-level container to have set a valid stream.
     */
    bool HasStream()const{return _stream!=nullptr;}

    /**
     * @brief Access the construction-time CUDA stream by reference.
     *
     * Non-runtime convenience: used for checkpoint I/O, optimizer-state
     * allocation, and other out-of-pass work that does not receive a ctx.
     * Inside ForwardImpl / BackwardImpl, read the active stream via
     * ctx.Stream() instead — that is the per-call source of truth.
     */
    CAIF_CudaStream &Stream()const
    {
      if(_stream==nullptr)
      {
        THROW_CAIFE("stream is null");
      }
      return *_stream;
    }

  protected:
    explicit CAIF_DeviceLayer(CAIF_CudaStream &stream):_stream(&stream){}

  private:
    CAIF_CudaStream *_stream;
};

}//end instance namespace

#endif  // CAIF_DEVICE_LAYER_H
