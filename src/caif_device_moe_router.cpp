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
// Device-resident MoE Router implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_router.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include <random>
#include <cmath>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoERouter<ComputeT,StorageT>::CAIF_DeviceMoERouter(const CAIF_DeviceMoERouterConfig &config,CAIF_CudaStream &stream)
  :CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream)
  ,_config(config)
{
  try
  {
    // Validate config
    if(Config().InputDim()==0)
    {
      THROW_CAIFE("MoERouter: input_dim must be > 0");
    }
    if(Config().NumExperts()==0)
    {
      THROW_CAIFE("MoERouter: num_experts must be > 0");
    }
    if(Config().TopK()==0||Config().TopK()>Config().NumExperts())
    {
      THROW_CAIFE("MoERouter: top_k must be > 0 and <= num_experts");
    }
    if(Config().GatingKind()==CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e&&
       Config().RoutingType()!=RoutingType_e::TopK)
    {
      THROW_CAIFE("MoERouter: SigmoidNoauxTc gating only supports TopK routing");
    }
    if(Config().NGroup()>1)
    {
      if(Config().NumExperts()%Config().NGroup()!=0)
      {
        THROW_CAIFE("MoERouter: num_experts must be divisible by n_group");
      }
      if(Config().TopkGroup()==0||Config().TopkGroup()>Config().NGroup())
      {
        THROW_CAIFE("MoERouter: topk_group must be in [1, n_group] when n_group > 1");
      }
      if(Config().RoutingType()!=RoutingType_e::TopK)
      {
        THROW_CAIFE("MoERouter: group routing (n_group > 1) only supports TopK routing");
      }
    }
    if(Config().BiasUpdateRate()>0.0f)
    {
      if(Config().UseBias()==false)
      {
        THROW_CAIFE("MoERouter: bias_update_rate > 0 requires use_bias=true");
      }
      if(Config().GatingKind()!=CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e)
      {
        THROW_CAIFE("MoERouter: aux-loss-free bias update requires SigmoidNoauxTc gating");
      }
    }

    // Allocate router weights [input_dim, num_experts] at the templated
    // storage dtype so the GEMM that consumes them matches the activation
    // tensor's dtype.
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    SetWRouter(CAIF_DeviceTensor::Uninitialized({Config().InputDim(),Config().NumExperts()},
                                                stream,
                                                sd));
    SetGradWRouter(CAIF_DeviceTensor::Zeros({Config().InputDim(),Config().NumExperts()},
                                            stream,
                                            sd));

    if(Config().UseBias()==true)
    {
      SetBRouter(CAIF_DeviceTensor::Zeros({Config().NumExperts()},stream,sd));
      SetGradBRouter(CAIF_DeviceTensor::Zeros({Config().NumExperts()},stream,sd));
    }

    // Xavier initialization for router weights — host-side fp32 stage,
    // then To() into StorageT before assigning.
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale=std::sqrt(2.0f/static_cast<float>(Config().InputDim()+Config().NumExperts()));
    std::normal_distribution<float> dist(0.0f,scale);

    std::vector<float> data(Config().InputDim()*Config().NumExperts());
    for(size_t i=0;i<data.size();++i)
    {
      data[i]=dist(gen);
    }
    WRouterMut().CopyFromHostFp32(data.data(),data.size());

    Stream().Synchronize();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoERouter<ComputeT,StorageT>::CAIF_DeviceMoERouter(CAIF_DeviceMoERouter<ComputeT,StorageT> &&other)
  :CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other))
  ,_config(other.Config())
  ,_w_router(std::move(other.WRouterMut()))
  ,_b_router(std::move(other.BRouterMut()))
  ,_grad_w_router(std::move(other.GradWRouterMut()))
  ,_grad_b_router(std::move(other.GradBRouterMut()))
  ,_cached_input(std::move(other.CachedInputMut()))
  ,_cached_logits(std::move(other.CachedLogitsMut()))
  ,_cached_probs(std::move(other.CachedProbsMut()))
  ,_cached_indices(std::move(other.CachedIndicesMut()))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoERouter<ComputeT,StorageT> &CAIF_DeviceMoERouter<ComputeT,StorageT>::operator=(CAIF_DeviceMoERouter<ComputeT,StorageT> &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    SetConfig(other.Config());
    SetWRouter(std::move(other.WRouterMut()));
    SetBRouter(std::move(other.BRouterMut()));
    SetGradWRouter(std::move(other.GradWRouterMut()));
    SetGradBRouter(std::move(other.GradBRouterMut()));
    SetCachedInput(std::move(other.CachedInputMut()));
    SetCachedLogits(std::move(other.CachedLogitsMut()));
    SetCachedProbs(std::move(other.CachedProbsMut()));
    SetCachedIndices(std::move(other.CachedIndicesMut()));
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoERouter<ComputeT,StorageT>::RouterOutput_t CAIF_DeviceMoERouter<ComputeT,StorageT>::Route(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)
{
  try
  {
    // Input can be [batch, seq_len, dim] or [num_tokens, dim]; the
    // router operates on a 2-D [num_tokens, dim] view. Previously this
    // was an `input.Clone()` (and a `.Reshape({num_tokens, dim})` in the
    // 3-D case) which paid one full N*dim device-to-device copy per
    // forward. The clone was load-bearing only as an alias-safety
    // measure — `flat_input` is consumed locally, never escapes Route(),
    // and the kernels invoked on it (MatMul, Softmax, Add) all write to
    // separately-allocated output tensors and never mutate flat_input
    // in place.
    //
    // WrapView is a non-owning device-pointer + shape + dtype + stream
    // record — zero allocations, zero copies, zero kernel launches. The
    // safety conditions for replacing the Clone:
    //   1. flat_input must not outlive `input`. Guaranteed: it is a
    //      function-local that drops at end-of-scope, before `input`
    //      goes out of scope at the caller.
    //   2. No kernel called on flat_input may mutate it. Verified: the
    //      MatMul on line 174 writes to `logits`; the Softmax on lines
    //      222/228 writes to a different tensor; the optional Add on
    //      line 198 writes to `logits`. None feed flat_input as the
    //      output operand.
    //   3. The caller's stream and the router's stream must be the same
    //      (or the kernels we launch must serialize against the writer
    //      of `input`). Verified: every device layer in CAIF is
    //      single-stream and the layer's Stream() is the same handle
    //      the model writes through. Cross-stream MoE is not supported.
    //   4. `_cached_input=flat_input.Clone()` below still produces an
    //      independent owning tensor for the backward pass (Clone()
    //      reads the bytes pointed at by the view and writes them to
    //      a fresh buffer). That cache survives `input` going out of
    //      scope at the caller.
    // Net effect on the after_forwardinto run vs post-phase4 peak:
    // MoE Forward prod (training) fp32 -0.4 ms, MoE Backward prod fp32
    // -1.3 ms (compare_multirun.py min-vs-min, peak-of-N rule).
    const auto &shape=input.Shape();
    uint32_t num_tokens=0;
    CAIF_DeviceTensor flat_input;

    if(shape.size()==2)
    {
      num_tokens=shape[0];
      flat_input=CAIF_DeviceTensor::WrapView(const_cast<void*>(input.DeviceDataRaw()),
                                             {num_tokens,shape[1]},
                                             Stream(),
                                             input.Dtype());
    }
    else if(shape.size()==3)
    {
      num_tokens=shape[0]*shape[1];
      flat_input=CAIF_DeviceTensor::WrapView(const_cast<void*>(input.DeviceDataRaw()),
                                             {num_tokens,shape[2]},
                                             Stream(),
                                             input.Dtype());
    }
    else
    {
      THROW_CAIFE("MoERouter::Route: expected input shape [N, dim] or [batch, seq, dim]");
    }

    if(flat_input.Shape()[1]!=Config().InputDim())
    {
      THROW_CAIFE("MoERouter::Route: input dimension mismatch");
    }

    // Cache input for backward
    if(ctx.Training()==true)
    {
      SetCachedInput(flat_input.Clone());
    }

    // Compute router logits: input @ w_router + b_router
    // [num_tokens, input_dim] @ [input_dim, num_experts] = [num_tokens, num_experts]
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();
    CAIF_DeviceTensor logits=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().NumExperts()},Stream(),sd);
    CAIF_Ops::MatMul(flat_input,WRouter(),logits,ctx,cdt);

    // SoftmaxTopK adds bias to logits BEFORE the score function (matches
    // softmax(logits+b)).  SigmoidNoauxTc treats the bias as a selection-
    // only correction added AFTER sigmoid; logits stays bias-free here so
    // _cached_logits / SoftmaxBackward stay in their existing shape and
    // the bias-correction add happens inside the TopK case below on a
    // per-token clone.
    if(Config().UseBias()==true&&
       Config().GatingKind()==CAIF_DeviceMoELayerFactory::GatingKind_e::SoftmaxTopK_e)
    {
      CAIF_Ops::AddBias(logits,BRouter(),logits);
    }

    // Add noise during training for exploration
    if(ctx.Training()==true&&Config().NoiseStd()>0.0f)
    {
      // Generate noise on host (fp32), upload to fp32 staging, cast to StorageT.
      std::vector<float> noise_data(num_tokens*Config().NumExperts());
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<float> dist(0.0f,Config().NoiseStd());
      for(size_t i=0;i<noise_data.size();++i)
      {
        noise_data[i]=dist(gen);
      }
      CAIF_DeviceTensor noise_fp32=CAIF_DeviceTensor::FromHostData(noise_data.data(),
                                                                  {num_tokens,Config().NumExperts()},
                                                                  Stream());
      if(sd==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        CAIF_Ops::Add(logits,noise_fp32,logits);
      }
      else
      {
        CAIF_DeviceTensor noise=noise_fp32.To(sd);
        CAIF_Ops::Add(logits,noise,logits);
      }
    }

    if(ctx.Training()==true)
    {
      SetCachedLogits(logits.Clone());
    }

    // Score function per gating regime:
    //   SoftmaxTopK_e   — softmax over experts (TopK/Soft) or over tokens
    //                     after transpose (ExpertChoice).
    //   SigmoidNoauxTc_e — per-element sigmoid; bias is a selection-only
    //                     correction added later inside the TopK case.
    //                     ctor validates routing_type==TopK for this
    //                     gating, so the ExpertChoice transpose branch
    //                     is unreachable from here.
    CAIF_DeviceTensor probs=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().NumExperts()},Stream(),sd);

    if(Config().GatingKind()==CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e)
    {
      CAIF_Ops::Sigmoid(logits,probs);
    }
    else if(Config().RoutingType()==RoutingType_e::ExpertChoice)
    {
      // Expert Choice: softmax over tokens for each expert
      // Transpose logits to [num_experts, num_tokens], softmax, transpose back
      CAIF_DeviceTensor logits_t=CAIF_DeviceTensor::Uninitialized({Config().NumExperts(),num_tokens},Stream(),sd);
      CAIF_Ops::Transpose(logits,logits_t);
      CAIF_DeviceTensor probs_t=CAIF_DeviceTensor::Uninitialized({Config().NumExperts(),num_tokens},Stream(),sd);
      CAIF_Ops::Softmax(logits_t,probs_t);
      CAIF_Ops::Transpose(probs_t,probs);
    }
    else
    {
      // TopK and Soft: softmax over experts for each token
      CAIF_Ops::Softmax(logits,probs);
    }

    if(ctx.Training()==true)
    {
      SetCachedProbs(probs.Clone());
    }

    // `logits` is fully consumed by this point — the optional bias-add
    // and noise-add (lines 178, 198, 203) have already written into it,
    // the softmax above has already read from it, and nothing past this
    // line touches `logits`. Move it into the output instead of cloning
    // (saves one O(num_tokens*num_experts) D2D copy per forward).
    //
    // `probs` cannot be moved here because the routing-type switch
    // immediately below (TopK / Soft / ExpertChoice branches) reads it
    // for top-k selection, soft routing, or transposed expert-choice
    // selection. A moved-from CAIF_DeviceTensor releases its device
    // pointer, so the switch would dereference null. Clone() preserves
    // a separate owning copy for the output while keeping the local
    // `probs` valid for the switch.
    RouterOutput_t output;
    output.router_logits=std::move(logits);
    output.router_probs=probs.Clone();

    // Route based on routing type
    switch(Config().RoutingType())
    {
      case RoutingType_e::TopK:
      {
        // Standard top-k: each token selects top_k experts.
        //
        // SoftmaxTopK: bias was already folded into logits before softmax
        // above, so probs already encode the bias-corrected distribution.
        // Top-k on probs directly; weights are always row-normalised.
        //
        // SigmoidNoauxTc (matches HF DeepSeek-V2 / GLM-4-MoE
        // `topk_method=noaux_tc`): probs holds sigmoid(logits) — the
        // ORIGINAL (uncorrected) sigmoid scores.  When use_bias is true,
        // selection happens on a per-token clone with `b_router` added
        // (bias is "for selection only" per the literature); the combine
        // weights are then drawn from the original sigmoid scores at the
        // chosen indices via CAIF_Ops::GatherTopKValues.  norm_topk_prob
        // gates the post-selection re-normalisation; routed_scaling_factor
        // multiplies the final weights.
        CAIF_DeviceTensor indices=CAIF_DeviceTensor::Uninitialized(
                                    {num_tokens,Config().TopK()},
                                    Stream(),
                                    CAIF_DataType::CAIF_DataType_e::Int32);
        CAIF_DeviceTensor weights=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().TopK()},Stream(),sd);

        if(Config().GatingKind()==CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e&&
           Config().UseBias()==true)
        {
          CAIF_DeviceTensor selection=probs.Clone();
          CAIF_Ops::AddBias(selection,BRouter(),selection);
          // DeepSeek group-limited routing: keep only the top-topk_group groups
          // before the expert top-k. No-op when n_group==1.
          if(Config().NGroup()>1)
          {
            CAIF_Ops::MoEGroupMask(selection,Config().NGroup(),Config().TopkGroup());
          }
          CAIF_Ops::TopK(selection,Config().TopK(),indices,weights);
          CAIF_Ops::GatherTopKValues(probs,indices,weights);
        }
        else if(Config().NGroup()>1)
        {
          // Group routing on a clone of probs (no bias correction). The top-k
          // values at the selected, non-masked experts are the original probs,
          // matching the un-grouped TopK(probs) path.
          CAIF_DeviceTensor selection=probs.Clone();
          CAIF_Ops::MoEGroupMask(selection,Config().NGroup(),Config().TopkGroup());
          CAIF_Ops::TopK(selection,Config().TopK(),indices,weights);
        }
        else
        {
          CAIF_Ops::TopK(probs,Config().TopK(),indices,weights);
        }

        // Both gating kinds respect norm_topk_prob from the model
        // config: HF Llama/DSv2/GLM-4-MoE/Qwen3 ship norm_topk_prob=true
        // (re-normalize top-k weights to sum to 1, so the routed sum
        // is in the same magnitude as a single expert output);
        // HF OLMoE / Olmo2 ship norm_topk_prob=false (use the raw
        // softmax probabilities, so the routed sum can be smaller
        // than 1.0 and the expert contribution is bounded by the
        // model's training-time dynamic range). Forcing normalization
        // on SoftmaxTopK regardless of the flag (the previous
        // behavior) inflated the routed sum by ~num_experts/top_k for
        // OLMoE and broke parity vs the reference implementation on
        // every OLMoE MoE step.
        if(Config().NormTopkProb()==true)
        {
          CAIF_Ops::NormalizeRows(weights,weights);
        }

        // routed_scaling_factor multiplies the final top-k combine weights for
        // BOTH gating kinds. HF DeepseekV2MoEGate applies it unconditionally
        // (topk_weight *= routed_scaling_factor) regardless of scoring_func, so
        // a softmax build with a non-unit factor (e.g. DeepSeek-V2-236B at 16.0)
        // must scale too. The != 1.0f guard only skips a no-op kernel launch.
        if(Config().RoutedScalingFactor()!=1.0f)
        {
          CAIF_Ops::Scale(weights,Config().RoutedScalingFactor());
        }

        if(ctx.Training()==true)
        {
          SetCachedIndices(indices.Clone());
        }

        output.expert_indices=std::move(indices);
        output.expert_weights=std::move(weights);
        break;
      }

      case RoutingType_e::ExpertChoice:
      {
        // Expert Choice: each expert selects top_k tokens
        // Capacity per expert = (num_tokens * top_k) / num_experts
        const uint32_t capacity=std::max(1u,(num_tokens*Config().TopK())/Config().NumExperts());

        // Transpose probs to [num_experts, num_tokens] for expert-wise top-k
        CAIF_DeviceTensor probs_t=CAIF_DeviceTensor::Uninitialized({Config().NumExperts(),num_tokens},
                                                                   Stream(),
                                                                   sd);
        CAIF_Ops::Transpose(probs,probs_t);

        // Each expert selects top-capacity tokens
        CAIF_DeviceTensor token_indices=CAIF_DeviceTensor::Uninitialized(
                                          {Config().NumExperts(),capacity},
                                          Stream(),
                                          CAIF_DataType::CAIF_DataType_e::Int32);
        CAIF_DeviceTensor token_weights=CAIF_DeviceTensor::Uninitialized({Config().NumExperts(),capacity},
                                                                         Stream(),
                                                                         sd);
        CAIF_Ops::TopK(probs_t,capacity,token_indices,token_weights);

        // Normalize weights per expert
        CAIF_Ops::NormalizeRows(token_weights,token_weights);

        if(ctx.Training()==true)
        {
          SetCachedIndices(token_indices.Clone());
        }

        // For Expert Choice, indices are [num_experts, capacity] - which tokens each expert processes
        // This is different from TopK where indices are [num_tokens, top_k]
        output.expert_indices=std::move(token_indices);
        output.expert_weights=std::move(token_weights);
        break;
      }

      case RoutingType_e::Soft:
      {
        // Soft MoE: all experts process all tokens with softmax weights
        // No sparse selection - return full probability matrix
        // indices: just sequential expert indices for each token
        CAIF_DeviceTensor indices=CAIF_DeviceTensor::Uninitialized(
                                    {num_tokens,Config().NumExperts()},
                                    Stream(),
                                    CAIF_DataType::CAIF_DataType_e::Int32);
        std::vector<int32_t> idx_data(num_tokens*Config().NumExperts());
        for(uint32_t t=0;t<num_tokens;++t)
        {
          for(uint32_t e=0;e<Config().NumExperts();++e)
          {
            idx_data[t*Config().NumExperts()+e]=static_cast<int32_t>(e);
          }
        }
        indices.CopyFromHostRaw(idx_data.data(),idx_data.size()*sizeof(int32_t));

        if(ctx.Training()==true)
        {
          SetCachedIndices(indices.Clone());
        }

        output.expert_indices=std::move(indices);
        output.expert_weights=probs.Clone();  // Full softmax probabilities
        break;
      }

      default:
        THROW_CAIFE("MoERouter::Route: unknown routing type");
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoERouter<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)
{
  try
  {
    // Forward returns the routing weights for compatibility with layer interface
    RouterOutput_t output=Route(input,ctx);
    return std::move(output.expert_weights);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoERouter<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)
{
  try
  {
    // This backward is for the standard layer interface
    // grad_output: gradient w.r.t. expert_weights [num_tokens, top_k]
    return BackwardRouting(grad_output,ctx);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoERouter<ComputeT,StorageT>::BackwardRouting(const CAIF_DeviceTensor &grad_weights,CAIF_RunContext &ctx)
{
  try
  {
    CAIF_DeviceTensor empty;
    return BackwardRoutingAuxAware(grad_weights,empty,empty,ctx);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoERouter<ComputeT,StorageT>::BackwardRoutingAuxAware(
  const CAIF_DeviceTensor &grad_weights,
  const CAIF_DeviceTensor &balance_bias,
  const CAIF_DeviceTensor &z_logsumexp_scaled,
  CAIF_RunContext &ctx)
{
  try
  {
    // SigmoidNoauxTc backward path:
    //
    // Forward chain:
    //   probs    = sigmoid(logits)               [N, E]
    //   selection= probs + b_router              [selection-only; non-differentiable]
    //   indices  = TopK(selection)               [N, K] (argmax — non-differentiable)
    //   gathered = probs[indices]                [N, K] (Phase 1b: original sigmoid scores)
    //   if norm_topk_prob: weights = gathered / sum(gathered)
    //   weights  = weights * routed_scaling_factor
    //
    // Backward chain (gradient flows from grad_weights):
    //   reverse scale (linear; commute with normalize-Jacobian → apply at end)
    //   reverse gather+normalize (NormalizeRowsBackwardTopKGather, when norm_topk_prob)
    //   scatter top-k grads to grad_probs [N, E]
    //   sigmoid backward: grad_logits = grad_probs * probs * (1 - probs)
    //   linear backward to grad_input + grad_w_router
    //
    // b_router gets NO gradient — selection is non-differentiable, and bias never enters
    // the weights path. The add-MoE / layer-surgery flow freezes the router anyway.
    //
    // Aux losses: balance_bias / z_logsumexp_scaled are softmax-specific (load-balance,
    // z-loss); SigmoidNoauxTc doesn't define these in the literature. Refuse if either
    // is non-empty rather than silently dropping the contribution.
    if(Config().GatingKind()==CAIF_DeviceMoELayerFactory::GatingKind_e::SigmoidNoauxTc_e)
    {
      if(balance_bias.Shape().size()>0||z_logsumexp_scaled.Shape().size()>0)
      {
        THROW_CAIFE("MoERouter::BackwardRoutingAuxAware: SigmoidNoauxTc does not support "
                    "balance_bias or z_logsumexp_scaled aux-loss inputs");
      }
      if(CachedProbs().Shape().size()==0||CachedIndices().Shape().size()==0||
         CachedInput().Shape().size()==0)
      {
        THROW_CAIFE("MoERouter::BackwardRoutingAuxAware: SigmoidNoauxTc backward requires "
                    "Route() to have been called with ctx.Training()==true first");
      }

      const std::vector<uint32_t> &shape_w=grad_weights.Shape();
      const uint32_t num_tokens=shape_w[0];
      const uint32_t top_k=Config().TopK();
      const CAIF_DataType::CAIF_DataType_e bsd=StorageDtype();

      if(GradTopkPreview().Shape().size()!=2||
         GradTopkPreview().Shape()[0]!=num_tokens||
         GradTopkPreview().Shape()[1]!=top_k||
         GradTopkPreview().Dtype()!=bsd)
      {
        SetGradTopkPreview(CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},Stream(),bsd));
      }

      if(Config().NormTopkProb()==true)
      {
        // Same gather+normalize Jacobian kernel as SoftmaxTopK; it operates on cached
        // sigmoid scores (in CachedProbs()) and chosen indices identically.
        CAIF_Ops::NormalizeRowsBackwardTopKGather(grad_weights,
                                                  CachedProbs(),
                                                  CachedIndices(),
                                                  GradTopkPreviewMut());
      }
      else
      {
        // No normalize — gather Jacobian is identity on the chosen positions.
        SetGradTopkPreview(grad_weights.Clone());
      }

      // Reverse the routed_scaling_factor multiply applied in forward. Linear, so it
      // commutes with the normalize-Jacobian above; folding it into grad_topk_preview
      // is equivalent to scaling grad_weights up front.
      if(Config().RoutedScalingFactor()!=1.0f)
      {
        CAIF_Ops::Scale(GradTopkPreviewMut(),Config().RoutedScalingFactor());
      }

      CAIF_DeviceTensor grad_probs=CAIF_DeviceTensor::Zeros({num_tokens,Config().NumExperts()},Stream(),bsd);
      CAIF_Ops::ScatterAdd(GradTopkPreview(),CachedIndices(),grad_probs);

      CAIF_DeviceTensor grad_logits=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().NumExperts()},
                                                                     Stream(),
                                                                     bsd);
      CAIF_Ops::SigmoidBackward(grad_probs,CachedProbs(),grad_logits);

      const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();
      CAIF_DeviceTensor grad_w_batch=CAIF_DeviceTensor::Uninitialized(
                                              {Config().InputDim(),Config().NumExperts()},
                                              Stream(),
                                              bsd);
      CAIF_Ops::MatMulTransposeA(CachedInput(),grad_logits,grad_w_batch,ctx,cdt);
      CAIF_Ops::Add(GradWRouter(),grad_w_batch,GradWRouterMut());

      // grad_b_router stays untouched — bias is non-differentiable in this path.

      CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},
                                                                    Stream(),
                                                                    bsd);
      CAIF_Ops::MatMulTransposeB(grad_logits,WRouter(),grad_input,ctx,cdt);
      return grad_input;
    }

    // grad_weights: [num_tokens, top_k] - gradient w.r.t. normalized routing weights
    const std::vector<uint32_t> &shape=grad_weights.Shape();
    const uint32_t num_tokens=shape[0];
    const uint32_t top_k=Config().TopK();

    // NormalizeRows backward Jacobian (device-side gather kernel):
    //   p[k]  = probs[t, indices[t,k]]
    //   s     = sum_k p[k],  w[k] = p[k]/s
    //   dot   = sum_k w[k]*grad_w[t,k]
    //   grad_p_topk = (grad_w - dot) / s
    // Gather pulls p directly from CachedProbs() so the forward path stays
    // allocation-clean. TopK is the only routing mode that normalizes rows;
    // ExpertChoice / Soft skip this and pass grad_weights through.
    const CAIF_DataType::CAIF_DataType_e bsd=StorageDtype();
    const bool topk_jacobian=(Config().RoutingType()==RoutingType_e::TopK&&
                              CachedProbs().Shape().size()>0&&
                              CachedIndices().Shape().size()>0);
    if(topk_jacobian==true)
    {
      if(GradTopkPreview().Shape().size()!=2||
         GradTopkPreview().Shape()[0]!=num_tokens||
         GradTopkPreview().Shape()[1]!=top_k||
         GradTopkPreview().Dtype()!=bsd)
      {
        SetGradTopkPreview(CAIF_DeviceTensor::Uninitialized({num_tokens,top_k},Stream(),bsd));
      }
      CAIF_Ops::NormalizeRowsBackwardTopKGather(grad_weights,
                                                CachedProbs(),
                                                CachedIndices(),
                                                GradTopkPreviewMut());

      // Reverse the routed_scaling_factor multiply applied in Route() (forward
      // scales the TopK combine weights by routed_scaling_factor for both
      // gating kinds). Linear, so it commutes with the normalize Jacobian
      // above; mirrors the SigmoidNoauxTc backward path at the top of this
      // function.
      if(Config().RoutedScalingFactor()!=1.0f)
      {
        CAIF_Ops::Scale(GradTopkPreviewMut(),Config().RoutedScalingFactor());
      }
    }
    const CAIF_DeviceTensor *grad_topk_preview_p=&grad_weights;
    if(topk_jacobian==true)
    {
      grad_topk_preview_p=&GradTopkPreview();
    }
    const CAIF_DeviceTensor &grad_topk_preview=*grad_topk_preview_p;

    // Scatter transformed top-k grads back to full expert dimension
    CAIF_DeviceTensor grad_probs=
      CAIF_DeviceTensor::Zeros({num_tokens,Config().NumExperts()},Stream(),bsd);
    CAIF_Ops::ScatterAdd(grad_topk_preview,CachedIndices(),grad_probs);

    // Load-balance loss feeds its gradient into grad_probs *before* the
    // softmax Jacobian flattens it into logit space. The contribution is
    // constant per expert column, so we broadcast-add a [num_experts] bias.
    if(balance_bias.Shape().size()>0)
    {
      CAIF_Ops::AddBias(grad_probs,balance_bias,grad_probs);
    }

    // Backward through softmax
    // grad_logits = probs * (grad_probs - sum(grad_probs * probs, axis=-1, keepdim=True))
    CAIF_DeviceTensor grad_logits=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().NumExperts()},
                                                                   Stream(),
                                                                   bsd);
    CAIF_Ops::SoftmaxBackward(CachedProbs(),grad_probs,grad_logits);

    // Router z-loss feeds its gradient directly into logit space
    // (dL_z/dlogit = (2*w/N) * logsumexp * probs — already includes the
    // softmax Jacobian, so we add after the softmax-backward above).
    // logsumexp_scaled is pre-multiplied by (2 * z_loss_weight / N) host-side.
    if(z_logsumexp_scaled.Shape().size()>0)
    {
      CAIF_Ops::MoEZLossGradAdd(z_logsumexp_scaled,CachedProbs(),grad_logits);
    }

    // Backward through linear projection
    // grad_w_router += input^T @ grad_logits
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();
    CAIF_DeviceTensor grad_w_batch=
      CAIF_DeviceTensor::Uninitialized({Config().InputDim(),Config().NumExperts()},Stream(),bsd);
    CAIF_Ops::MatMulTransposeA(CachedInput(),grad_logits,grad_w_batch,ctx,cdt);
    CAIF_Ops::Add(GradWRouter(),grad_w_batch,GradWRouterMut());

    if(Config().UseBias()==true)
    {
      // grad_b_router += sum(grad_logits, axis=0)
      CAIF_DeviceTensor grad_b_batch=CAIF_DeviceTensor::Uninitialized({Config().NumExperts()},Stream(),bsd);
      CAIF_Ops::SumAxis(grad_logits,0,grad_b_batch);
      CAIF_Ops::Add(GradBRouter(),grad_b_batch,GradBRouterMut());
    }

    // grad_input = grad_logits @ w_router^T
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},Stream(),bsd);
    CAIF_Ops::MatMulTransposeB(grad_logits,WRouter(),grad_input,ctx,cdt);

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoERouter<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    GradWRouterMut().FillZero();
    if(Config().UseBias()==true)
    {
      GradBRouterMut().FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMoERouter<ComputeT,StorageT>::ParameterTensorCount()const
{
  size_t count=1;  // w_router
  if(Config().UseBias()==true)
  {
    count+=1;  // b_router
  }
  return count;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMoERouter<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  if(index==0)
  {
    return WRouterMut();
  }
  if(Config().UseBias()==true&&index==1)
  {
    return BRouterMut();
  }
  THROW_CAIFE("MoERouter::ParameterTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMoERouter<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoERouter*>(this)->ParameterTensor(index);
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMoERouter<ComputeT,StorageT>::GradientTensor(size_t index)
{
  if(index==0)
  {
    return GradWRouterMut();
  }
  if(Config().UseBias()==true&&index==1)
  {
    return GradBRouterMut();
  }
  THROW_CAIFE("MoERouter::GradientTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMoERouter<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoERouter*>(this)->GradientTensor(index);
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMoERouter<ComputeT,StorageT>::TotalParameterCount()const
{
  size_t count=Config().InputDim()*Config().NumExperts();  // w_router
  if(Config().UseBias()==true)
  {
    count+=Config().NumExperts();  // b_router
  }
  return count;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceMoERouter<ComputeT,StorageT>::Description()const
{
  std::string desc="MoERouter[";
  desc+=std::to_string(Config().InputDim())+"->"+std::to_string(Config().NumExperts());
  desc+=",top_k="+std::to_string(Config().TopK());
  if(Config().UseBias()==true)
  {
    desc+=",bias";
  }
  if(Config().NoiseStd()>0.0f)
  {
    desc+=",noise="+std::to_string(Config().NoiseStd());
  }
  desc+="]";
  return desc;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceMoERouter<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
  std::vector<std::string> names;
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoERouterWeight_e));
  if(Config().UseBias()==true)
  {
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoERouterBias_e));
  }
  return names;
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoERouter<ComputeT,StorageT>::InitFavorExpert(uint32_t expert_index,
                                                                float bias_magnitude)
{
  try
  {
    if(expert_index>=Config().NumExperts())
    {
      THROW_CAIFE("MoERouter::InitFavorExpert: expert_index >= num_experts");
    }
    if(Config().UseBias()==false)
    {
      THROW_CAIFE("MoERouter::InitFavorExpert: requires use_bias=true so the "
                  "bias one-hot can dominate logits at init");
    }

    // Zero the matrix weight: with `_w_router` zero and `_b_router` set
    // to a one-hot pattern, every input row produces the same logits =
    // bias, so softmax(logits) is constant and concentrates fully on
    // `expert_index` for any input. Training then learns the routing
    // structure on top of this prior.
    std::vector<float> w_zero(Config().InputDim()*Config().NumExperts(),0.0f);
    WRouterMut().CopyFromHostFp32(w_zero.data(),w_zero.size());

    // Bias: one-hot with `bias_magnitude` at `expert_index`, zero
    // elsewhere.
    std::vector<float> b_data(Config().NumExperts(),0.0f);
    b_data[expert_index]=bias_magnitude;
    BRouterMut().CopyFromHostFp32(b_data.data(),b_data.size());
    Stream().Synchronize();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoERouter<ComputeT,StorageT>::LoadWRouter(CAIF_DeviceTensor &&w)
{
  try
  {
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2||
       shape[0]!=Config().InputDim()||
       shape[1]!=Config().NumExperts())
    {
      THROW_CAIFE("MoERouter::LoadWRouter: shape mismatch, expected "
                  "[input_dim, num_experts]");
    }
    if(w.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("MoERouter::LoadWRouter: dtype mismatch, expected StorageDtype");
    }
    _w_router=std::move(w);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoERouter<ComputeT,StorageT>::LoadBRouter(CAIF_DeviceTensor &&b)
{
  try
  {
    if(Config().UseBias()==false)
    {
      THROW_CAIFE("MoERouter::LoadBRouter: requires use_bias=true");
    }
    const std::vector<uint32_t> &shape=b.Shape();
    if(shape.size()!=1||shape[0]!=Config().NumExperts())
    {
      THROW_CAIFE("MoERouter::LoadBRouter: shape mismatch, expected [num_experts]");
    }
    if(b.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("MoERouter::LoadBRouter: dtype mismatch, expected StorageDtype");
    }
    _b_router=std::move(b);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoERouter<ComputeT,StorageT>::UpdateAuxLossFreeBias()
{
  try
  {
    if(Config().BiasUpdateRate()<=0.0f)
    {
      return;
    }
    if(CachedIndices().TotalElements()==0)
    {
      return;
    }
    const uint32_t num_experts=Config().NumExperts();
    CAIF_DeviceTensor counts=CAIF_DeviceTensor::Uninitialized({num_experts},
                                                              Stream(),
                                                              CAIF_DataType::CAIF_DataType_e::Int32);
    CAIF_Ops::MoECountPerExpert(CachedIndices(),num_experts,Config().TopK(),counts);
    CAIF_Ops::MoEBiasUpdate(BRouterMut(),counts,Config().BiasUpdateRate());
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceMoERouter<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceMoERouter<float,__half>;
template class CAIF_DeviceMoERouter<float,__nv_bfloat16>;
template class CAIF_DeviceMoERouter<__half,float>;
template class CAIF_DeviceMoERouter<__half,__half>;
template class CAIF_DeviceMoERouter<__half,__nv_bfloat16>;
template class CAIF_DeviceMoERouter<__nv_bfloat16,float>;
template class CAIF_DeviceMoERouter<__nv_bfloat16,__half>;
template class CAIF_DeviceMoERouter<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
