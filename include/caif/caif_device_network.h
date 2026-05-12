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
// Device-resident neural network using CAIF_DeviceTensor
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_NETWORK_H
#define CAIF_DEVICE_NETWORK_H

#include "caif_device_container.h"
#include "caif_device_tensor.h"
#include "caif_device_dense_layer.h"
#include "caif_model_format.h"
#include "caif_cuda_stream.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <string>

namespace instance
{

class CAIF_Optimizer;


/**
 * @brief Top-level device-resident neural network.
 *
 * A CAIF_DeviceContainer that also owns:
 *   - the training-loop facing Forward(input,training) / Backward(grad) entry
 *     points, which construct the per-call CAIF_RunContext and stash any
 *     pending sideband state onto it (prefix lengths, encoder context,
 *     position bias),
 *   - the active optimizer (Adam / SGD / Momentum / RMSprop / AdaGrad)
 *     selected by Initialize{Adam,Sgd,Momentum,Rmsprop,AdaGrad}(),
 *   - gradient norm clipping,
 *   - SafeTensors + legacy binary model I/O.
 *
 * Sublayer ownership, trainability, parameter/gradient iteration, aux-loss
 * summation, and LayerCount()/Layer(i) accessors all come from
 * CAIF_DeviceContainer.
 */
class CAIF_DeviceNetwork:public CAIF_DeviceContainer
{
  public:
    /**
     * @brief Construct a device network.
     *
     * @param stream CUDA stream for all operations.
     */
    explicit CAIF_DeviceNetwork(CAIF_CudaStream &stream);

    // Out-of-line so unique_ptr<CAIF_Optimizer> sees the full type.
    ~CAIF_DeviceNetwork()override;

    CAIF_DeviceNetwork(const CAIF_DeviceNetwork &)=delete;
    CAIF_DeviceNetwork &operator=(const CAIF_DeviceNetwork &)=delete;

    CAIF_DeviceNetwork(CAIF_DeviceNetwork &&other);
    CAIF_DeviceNetwork &operator=(CAIF_DeviceNetwork &&other);

    /**
     * @brief Add a dense layer to the network.
     *
     * Layers are added sequentially. The first layer's input_size must match
     * the data input size; subsequent layers auto-connect when input_size=0.
     */
    void AddDenseLayer(uint32_t input_size,
                       uint32_t output_size,
                       CAIF_DeviceActivation_e activation,
                       bool use_bias=true);

    /**
     * @brief Add a polymorphic layer (ownership transferred).
     *
     * Adam state is invalidated; InputSize/OutputSize are NOT updated (only
     * AddDenseLayer tracks the declarative dense-chain IO sizes).
     */
    void AddLayer(std::unique_ptr<CAIF_DeviceLayer> layer);

    /**
     * @brief Mark a sublayer as trainable / non-trainable.
     *
     * Delegates to CAIF_DeviceContainer and then invalidates Adam state so
     * the optimiser is rebuilt on the next InitializeAdam call.
     */
    void SetLayerTrainable(size_t index,bool trainable);

    /**
     * @brief Training-loop facing forward pass.
     *
     * Constructs a per-call CAIF_RunContext, sets stream/training/pass,
     * stashes any pending sideband state, then invokes the base-class
     * CAIF_DeviceLayer::Forward path (which pushes Network_e and chains
     * sublayers via CAIF_DeviceContainer::ForwardImpl).
     */
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training=false);

    /**
     * @brief Training-loop facing backward pass.
     *
     * Constructs the per-call CAIF_RunContext in Backward_e mode via
     * CAIF_RunContextPassScope, stashes pending sideband, then invokes
     * the inherited CAIF_DeviceLayer::Backward.
     */
    void Backward(const CAIF_DeviceTensor &grad_output);

    // CAIF_DeviceLayer hooks — only the tag remains overridden; Forward/
    // Backward IMPLs are inherited from CAIF_DeviceContainer (default
    // sequential chain).
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::Network_e;
    }
    std::string Description()const override;

    /**
     * @brief Stash pending sideband state copied into the per-call run context.
     *
     * Prefix-LM lengths, cross-attention encoder context, and T5 relative
     * position bias used to be distributed across per-layer virtual setters.
     * They now live on CAIF_RunContext, which Network constructs inside its
     * own Forward/Backward. Callers that need to drive these sidebands (the
     * trainer, encoder-decoder orchestration) set them here; the stashed
     * state is copied onto the ctx at the start of every Forward and
     * Backward. Call Clear<Name>() to remove a previously stashed sideband.
     *
     * Targets are held by non-owning reference — the caller keeps the
     * tensors alive across the Forward/Backward pair.
     */
    void SetPrefixLengths(const CAIF_DeviceTensor &t){_pending_prefix_lengths=&t;}
    void ClearPrefixLengths(){_pending_prefix_lengths=nullptr;}
    void SetEncoderContext(const CAIF_DeviceTensor &t){_pending_encoder_context=&t;}
    void ClearEncoderContext(){_pending_encoder_context=nullptr;}
    void SetGradEncoderContext(CAIF_DeviceTensor &t){_pending_grad_encoder_context=&t;}
    void ClearGradEncoderContext(){_pending_grad_encoder_context=nullptr;}
    void SetPositionBias(const CAIF_DeviceTensor &t){_pending_position_bias=&t;}
    void ClearPositionBias(){_pending_position_bias=nullptr;}
    void SetGradPositionBias(CAIF_DeviceTensor &t){_pending_grad_position_bias=&t;}
    void ClearGradPositionBias(){_pending_grad_position_bias=nullptr;}

    /**
     * @brief Initialize Adam optimizer state.
     *
     * Convenience shim — wraps the CAIF_AdamOptimizer behind the
     * polymorphic _optimizer member so OptimizerStep / SetLearningRate
     * work uniformly across optimizer choices.
     */
    void InitializeAdam(float lr=g_caif_default_learning_rate,
                        float beta1=g_caif_default_beta1,
                        float beta2=g_caif_default_beta2,
                        float epsilon=g_caif_adam_epsilon,
                        float weight_decay=0.0f);

    /**
     * @brief Initialize Adam with first/second-moment state offloaded to
     * pinned host RAM. Same kernel as InitializeAdam, just keeps m / v on
     * host between optimizer steps. See caif/CPU_OFFLOAD_DESIGN.md.
     */
    void InitializeOffloadedAdam(float lr=g_caif_default_learning_rate,
                                 float beta1=g_caif_default_beta1,
                                 float beta2=g_caif_default_beta2,
                                 float epsilon=g_caif_adam_epsilon,
                                 float weight_decay=0.0f);

    /**
     * @brief Initialize plain-SGD optimizer state (no momentum).
     */
    void InitializeSgd(float lr,
                       float weight_decay=0.0f);

    /**
     * @brief Initialize SGD-with-momentum optimizer state.
     */
    void InitializeMomentum(float lr,
                            float momentum=g_caif_sgd_default_momentum,
                            float weight_decay=0.0f);

    /**
     * @brief Initialize RMSprop optimizer state.
     */
    void InitializeRmsprop(float lr,
                           float alpha=g_caif_rmsprop_default_alpha,
                           float epsilon=g_caif_rmsprop_default_epsilon,
                           float weight_decay=0.0f);

    /**
     * @brief Initialize AdaGrad optimizer state.
     */
    void InitializeAdaGrad(float lr,
                           float epsilon=g_caif_adagrad_default_epsilon,
                           float weight_decay=0.0f);

    /**
     * @brief One optimizer step across every trainable parameter
     * tensor. Dispatches through the active CAIF_Optimizer subclass.
     */
    void OptimizerStep();

    /**
     * @brief Backwards-compatible alias for OptimizerStep().
     * @deprecated Use OptimizerStep().
     */
    void AdamStep(){OptimizerStep();}

    /**
     * @brief Set the active optimizer's learning rate (for LR
     * scheduling). Throws if no Initialize* has been called.
     */
    void SetLearningRate(float lr);

    /**
     * @brief Clip gradient norm across all trainable parameters.
     *
     * Computes the global L2 norm of all trainable gradients and scales
     * them so the total norm does not exceed max_norm. Returns the total
     * gradient norm before clipping.
     */
    float ClipGradientNorm(float max_norm);

    /**
     * @brief Sum of squares of all trainable gradients (no sqrt, no scale).
     *
     * Split out of ClipGradientNorm so callers that own multiple networks
     * (e.g. an encoder-decoder trainer) can accumulate this across
     * networks, take one sqrt to get a joint L2 norm, and apply one joint
     * scale via ScaleGradients. Per-network clip with a combined reduction
     * is the standard "global-norm clip" semantics; calling
     * ClipGradientNorm on each network separately gives a different scale
     * on each side and the combined post-clip norm can exceed max_norm
     * by up to sqrt(N).
     */
    float GradientNormSquared();

    /**
     * @brief Multiply every trainable gradient tensor by coef in place.
     *
     * Pairs with GradientNormSquared for multi-network joint gradient
     * clipping. No-op when coef == 1.0f. Skips frozen layers, matching
     * the same filter ClipGradientNorm uses.
     */
    void ScaleGradients(float coef);

    /**
     * @brief Get a dense layer by index (typed access). Throws if the layer
     * at the given index is not a dense layer.
     */
    CAIF_DeviceDenseLayer<float,float> &DenseLayer(size_t index);
    const CAIF_DeviceDenseLayer<float,float> &DenseLayer(size_t index)const;

    /**
     * @brief Declarative dense-chain input/output sizes.
     *
     * Tracked only by AddDenseLayer; polymorphic AddLayer calls do not
     * update these. Returns 0 when the dense chain is empty.
     */
    uint32_t InputSize()const{return _input_size;}
    uint32_t OutputSize()const{return _output_size;}

    /**
     * @brief Save network parameters to SafeTensors format.
     */
    void SaveSafeTensors(const std::string &path)const;

    /**
     * @brief Load network parameters from SafeTensors file. The network must
     * already have layers added that match the saved model.
     */
    void LoadSafeTensors(const std::string &path);

    /**
     * @brief Save network using specified format.
     */
    void Save(const std::string &path,const CAIF_ModelFormat &format)const;

    /**
     * @brief Load network parameters using specified format.
     */
    void Load(const std::string &path,const CAIF_ModelFormat &format);

    /**
     * @brief Save the network to legacy binary format (dense layers only).
     * @deprecated Use SaveSafeTensors instead.
     */
    void SaveModel(const std::string &filepath,bool save_optimizer_state=true)const;

    /**
     * @brief Load network from legacy binary format (dense layers only).
     * @deprecated Use LoadSafeTensors instead. The network must not have any
     * layers added before loading.
     */
    void LoadModel(const std::string &filepath,bool load_optimizer_state=true);

  protected:

  private:
    // Copy any stashed sideband (prefix lengths, encoder ctx, position bias)
    // onto the per-call run context. Called at the top of Forward and
    // Backward so both passes see the same sideband configuration.
    void StashSidebandIntoContext(CAIF_RunContext &ctx)const;

    // Declarative dense-chain IO sizes, tracked by AddDenseLayer.
    uint32_t _input_size;
    uint32_t _output_size;

    // Internal accessors + setters for _optimizer — methods access the
    // optimizer through these instead of touching `_optimizer` directly
    // (per coding guidelines: accessor-only access, even from within
    // the class's own methods).  Setters are defined out-of-line in
    // the .cpp so the unique_ptr destructor sees the full
    // CAIF_Optimizer type.
    bool HasOptimizer()const{return _optimizer!=nullptr;}
    CAIF_Optimizer &Optimizer(){return *_optimizer;}
    void ClearOptimizer();
    void SetOptimizer(std::unique_ptr<CAIF_Optimizer> optimizer);

    // Active optimizer.  Set by Initialize{Adam,Sgd,Momentum,Rmsprop,
    // AdaGrad}; null until then.  Owns its own state (m/v for Adam,
    // velocity for Momentum, etc.) plus the AMP fp32-master + grad-fp32
    // scaffolding that every optimizer needs when params are non-fp32.
    std::unique_ptr<CAIF_Optimizer> _optimizer;

    // Pending sideband state — copied onto the per-call run context at the
    // top of Forward/Backward. nullptr means no sideband.
    const CAIF_DeviceTensor *_pending_prefix_lengths=nullptr;
    const CAIF_DeviceTensor *_pending_encoder_context=nullptr;
    CAIF_DeviceTensor *_pending_grad_encoder_context=nullptr;
    const CAIF_DeviceTensor *_pending_position_bias=nullptr;
    CAIF_DeviceTensor *_pending_grad_position_bias=nullptr;
};

}//end instance namespace

#endif  // CAIF_DEVICE_NETWORK_H
