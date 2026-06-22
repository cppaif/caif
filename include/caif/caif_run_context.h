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
// Authoritative per-run state: pass direction, subsystem tag stack, sideband
// state (encoder context, position bias, prefix lengths, KV-cache is NOT
// here — it is per-attention-layer), stream, precision selection, RNG.
// Owned by the top-level container (CAIF_DeviceNetwork) and passed by
// non-const reference through every Forward/Backward call.
//------------------------------------------------------------------------------
#ifndef CAIF_RUN_CONTEXT_H
#define CAIF_RUN_CONTEXT_H

#include "caif_base.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"
#include <cstdint>
#include <vector>

namespace instance
{

class CAIF_DeviceTensor;

class CAIF_RunContext:public CAIF_Base
{
  public:
    enum class Pass_e:uint32_t
    {
      Forward_e=0,
      Backward_e=1
    };

    enum class Subsystem_e:uint32_t
    {
      None_e=0,
      // embeddings
      TokenEmbedding_e,
      PatchEmbedding_e,
      SpectrogramEmbedding_e,
      TabularEmbedding_e,
      PositionalEncoding_e,
      // blocks + containers
      Network_e,
      TransformerModel_e,
      ViTModel_e,
      TransformerBlock_e,
      PreNormBlock_e,
      // attention family
      MHA_e,
      MLA_e,
      GQA_e,
      CrossAttention_e,
      T5Attention_e,
      RoPE_e,
      RelativePositionBias_e,
      // feed-forward family
      FFN_e,
      FFNGate_e,
      FFNUp_e,
      FFNDown_e,
      MoE_e,
      MoERouter_e,
      MoEExpert_e,
      // norms
      RMSNorm_e,
      LayerNorm_e,
      // linear
      Dense_e,
      FrozenLinear_e,
      LoRAAdapter_e,
      LinearHead_e,
      // activation / regularisation / shape
      Activation_e,
      Dropout_e,
      BatchNorm_e,
      Flatten_e,
      Reshape_e,
      // vision primitives
      Conv2D_e,
      Pooling2D_e,
      // loss
      CrossEntropyLoss_e,
      MSELoss_e
    };

    typedef std::vector<Subsystem_e> SubsystemStackVec_t;

    CAIF_RunContext()=default;
    ~CAIF_RunContext()=default;

    // Non-copyable: a run context is a stack-allocated per-call thing. If
    // anything copies it the copies would diverge on state. Moves are also
    // disallowed because the ctx is pinned at the top-level Forward/Backward
    // stack frame.
    CAIF_RunContext(const CAIF_RunContext &)=delete;
    CAIF_RunContext &operator=(const CAIF_RunContext &)=delete;
    CAIF_RunContext(CAIF_RunContext &&)=delete;
    CAIF_RunContext &operator=(CAIF_RunContext &&)=delete;

    // Pass direction + training flag.
    Pass_e Pass()const{return _pass;}
    void SetPass(const Pass_e p){_pass=p;}
    bool Training()const{return _training;}
    void SetTraining(const bool t){_training=t;}

    // Subsystem tag stack. Pushed/popped by the CAIF_DeviceLayer base-class
    // scaffolding via RAII scopes; subclasses never touch this directly.
    Subsystem_e CurrentSubsystem()const;
    uint32_t SubsystemDepth()const{return static_cast<uint32_t>(SubsystemStack().size());}
    void PushSubsystem(const Subsystem_e s){SubsystemStack().push_back(s);}
    void PopSubsystem();
    const SubsystemStackVec_t &SubsystemStack()const{return _subsystem_stack;}
    SubsystemStackVec_t &SubsystemStack(){return _subsystem_stack;}

    // Stream hoisted off every layer. Ops read ctx.Stream() rather than
    // reaching through the layer that owns them.
    CAIF_CudaStream &Stream()const;
    void SetStream(CAIF_CudaStream &stream){_stream=&stream;}
    bool HasStream()const{return _stream!=nullptr;}

    // Cross-attention plumbing. Replaces SetContext / SetGradContext
    // sideband virtuals on CAIF_DeviceLayer. Accessors return references and
    // throw when unset; callers gate reads with HasEncoderContext() /
    // HasGradEncoderContext(). Raw pointers never leak out.
    bool HasEncoderContext()const{return _encoder_context!=nullptr;}
    const CAIF_DeviceTensor &EncoderContext()const;
    void SetEncoderContext(const CAIF_DeviceTensor &t){_encoder_context=&t;}
    void ClearEncoderContext(){_encoder_context=nullptr;}
    bool HasGradEncoderContext()const{return _grad_encoder_context!=nullptr;}
    CAIF_DeviceTensor &GradEncoderContext()const;
    void SetGradEncoderContext(CAIF_DeviceTensor &t){_grad_encoder_context=&t;}
    void ClearGradEncoderContext(){_grad_encoder_context=nullptr;}

    // Relative / absolute position bias. Replaces SetPositionBias /
    // SetGradPositionBias sideband virtuals.
    bool HasPositionBias()const{return _position_bias!=nullptr;}
    const CAIF_DeviceTensor &PositionBias()const;
    void SetPositionBias(const CAIF_DeviceTensor &t){_position_bias=&t;}
    void ClearPositionBias(){_position_bias=nullptr;}
    bool HasGradPositionBias()const{return _grad_position_bias!=nullptr;}
    CAIF_DeviceTensor &GradPositionBias()const;
    void SetGradPositionBias(CAIF_DeviceTensor &t){_grad_position_bias=&t;}
    void ClearGradPositionBias(){_grad_position_bias=nullptr;}

    // Prefix-LM lengths. Replaces SetPrefixLengths / ClearPrefixLengths /
    // HasPrefixLengths sideband virtuals. Absent means pure-causal masking.
    bool HasPrefixLengths()const{return _prefix_lengths!=nullptr;}
    const CAIF_DeviceTensor &PrefixLengths()const;
    void SetPrefixLengths(const CAIF_DeviceTensor &t){_prefix_lengths=&t;}
    void ClearPrefixLengths(){_prefix_lengths=nullptr;}

    // cuBLAS compute-type selection — the sole place TF32 vs FP32 is
    // decided. Returns the numeric cublasComputeType_t value cast to
    // int32_t so the public header does not depend on cuBLAS. Callers
    // cast back to cublasComputeType_t at the call site.
    // Symmetric in pass direction: forward and backward use the same
    // compute type. FP32 inputs use CUBLAS_COMPUTE_32F_FAST_TF32 in
    // Performance mode and CUBLAS_COMPUTE_32F in Accuracy mode. FP16/BF16
    // inputs always use CUBLAS_COMPUTE_32F (fp32 accumulate). Selection
    // comes from CAIF_Settings::MatmulMode().
    int32_t ComputeTypeFor(const CAIF_DataType::CAIF_DataType_e dt)const;

    // Two-argument form for per-layer compute precision. The layer passes
    // its own `compute_dtype` (Step 3's `_compute_dtype` field) and the
    // element dtype of the live tensor. Used by CAIF_Ops::MatMul family
    // to select cuBLAS reduced-precision compute types:
    //   fp32 storage + fp32 compute -> CUBLAS_COMPUTE_32F_FAST_TF32/32F
    //                                  (as the 1-arg form, global mode)
    //   fp32 storage + bf16 compute -> CUBLAS_COMPUTE_32F_FAST_16BF
    //   fp32 storage + fp16 compute -> CUBLAS_COMPUTE_32F_FAST_16F
    //   bf16/fp16 storage           -> CUBLAS_COMPUTE_32F (fp32 accumulate)
    // When `compute_dt == Float32` the result matches the 1-arg form, so
    // callers that pass Float32 as the default see no behavior change.
    int32_t ComputeTypeFor(const CAIF_DataType::CAIF_DataType_e input_dt,
                           const CAIF_DataType::CAIF_DataType_e compute_dt)const;

    // Deterministic RNG for Dropout and any future stochastic layer.
    // A layer reads the seed and a monotonic per-call counter; the counter
    // is advanced on every draw so runs are reproducible for a fixed seed.
    uint64_t RandomSeed()const{return _random_seed;}
    void SetRandomSeed(const uint64_t s)
    {
      _random_seed=s;
      _random_counter=0;
    }
    uint64_t NextRandomCounter(){return _random_counter++;}

    // Step bookkeeping — optional, for callers that want the ctx to carry
    // their optimiser step number.
    uint64_t Step()const{return _step;}
    void SetStep(const uint64_t s){_step=s;}

  protected:

  private:
    Pass_e _pass=Pass_e::Forward_e;
    bool _training=false;
    SubsystemStackVec_t _subsystem_stack;
    CAIF_CudaStream *_stream=nullptr;
    const CAIF_DeviceTensor *_encoder_context=nullptr;
    CAIF_DeviceTensor *_grad_encoder_context=nullptr;
    const CAIF_DeviceTensor *_position_bias=nullptr;
    CAIF_DeviceTensor *_grad_position_bias=nullptr;
    const CAIF_DeviceTensor *_prefix_lengths=nullptr;
    uint64_t _random_seed=0;
    uint64_t _random_counter=0;
    uint64_t _step=0;
};

}//end instance namespace

#endif  // CAIF_RUN_CONTEXT_H
