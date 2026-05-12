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
// Device-resident MoE Layer implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_layer.h"
#include "caif_ops.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#endif

namespace instance
{


// Build internal Router/Expert primitives from scalar args. Used by the
// primary constructor. Experts are always sized to _hidden_dim; if the
// caller wants a distinct shared-expert width they pass shared_hidden_dim.
template<typename ComputeT,typename StorageT>
void BuildPrimitives(uint32_t input_dim,
                     uint32_t hidden_dim,
                     uint32_t num_experts,
                     uint32_t top_k,
                     bool expert_use_gated,
                     bool expert_use_bias,
                     uint32_t num_shared_experts,
                     uint32_t shared_hidden_dim,
                     bool router_use_bias,
                     float router_noise_std,
                     CAIF_DeviceMoELayerFactory::GatingKind_e gating_kind,
                     bool norm_topk_prob,
                     float routed_scaling_factor,
                     CAIF_CudaStream &stream,
                     std::unique_ptr<CAIF_DeviceMoERouter<ComputeT,StorageT>> &out_router,
                     std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> &out_experts,
                     std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> &out_shared)
{
  typename CAIF_DeviceMoERouter<ComputeT,StorageT>::Config_t router_cfg;
  router_cfg.input_dim=input_dim;
  router_cfg.num_experts=num_experts;
  router_cfg.top_k=top_k;
  router_cfg.routing_type=CAIF_DeviceMoERouter<ComputeT,StorageT>::RoutingType_e::TopK;
  router_cfg.use_bias=router_use_bias;
  router_cfg.noise_std=router_noise_std;
  router_cfg.gating_kind=gating_kind;
  router_cfg.norm_topk_prob=norm_topk_prob;
  router_cfg.routed_scaling_factor=routed_scaling_factor;

  out_router=std::make_unique<CAIF_DeviceMoERouter<ComputeT,StorageT>>(router_cfg,stream);

  typename CAIF_DeviceMoEExpert<ComputeT,StorageT>::Config_t expert_cfg;
  expert_cfg.input_dim=input_dim;
  expert_cfg.hidden_dim=hidden_dim;
  expert_cfg.use_gated=expert_use_gated;
  expert_cfg.use_bias=expert_use_bias;

  out_experts.clear();
  out_experts.reserve(num_experts);
  for(uint32_t i=0;i<num_experts;++i)
  {
    out_experts.push_back(std::make_unique<CAIF_DeviceMoEExpert<ComputeT,StorageT>>(expert_cfg,stream));
  }

  out_shared.clear();
  if(num_shared_experts>0)
  {
    typename CAIF_DeviceMoEExpert<ComputeT,StorageT>::Config_t shared_cfg=expert_cfg;
    if(shared_hidden_dim>0)
    {
      shared_cfg.hidden_dim=shared_hidden_dim;
    }
    out_shared.reserve(num_shared_experts);
    for(uint32_t i=0;i<num_shared_experts;++i)
    {
      out_shared.push_back(std::make_unique<CAIF_DeviceMoEExpert<ComputeT,StorageT>>(shared_cfg,stream));
    }
  }
}

void ValidateCore(uint32_t input_dim,
                  uint32_t hidden_dim,
                  uint32_t num_experts,
                  uint32_t top_k)
{
  if(input_dim==0)
  {
    THROW_CAIFE("MoELayer: input_dim must be > 0");
  }
  if(hidden_dim==0)
  {
    THROW_CAIFE("MoELayer: hidden_dim must be > 0");
  }
  if(num_experts==0)
  {
    THROW_CAIFE("MoELayer: num_experts must be > 0");
  }
  if(top_k==0||top_k>num_experts)
  {
    THROW_CAIFE("MoELayer: top_k must be > 0 and <= num_experts");
  }
}


template<typename ComputeT,typename StorageT>
CAIF_DeviceMoELayer<ComputeT,StorageT>::CAIF_DeviceMoELayer(uint32_t input_dim,
                                         uint32_t hidden_dim,
                                         uint32_t num_experts,
                                         uint32_t top_k,
                                         bool expert_use_gated,
                                         bool expert_use_bias,
                                         uint32_t num_shared_experts,
                                         uint32_t shared_hidden_dim,
                                         bool router_use_bias,
                                         float router_noise_std,
                                         float capacity_factor,
                                         OverflowStrategy_e overflow_strategy,
                                         float balance_loss_weight,
                                         float z_loss_weight,
                                         CAIF_CudaStream &stream,
                                         CAIF_DeviceMoELayerFactory::GatingKind_e gating_kind,
                                         bool norm_topk_prob,
                                         float routed_scaling_factor):
                                           CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                           _input_dim(input_dim),
                                           _hidden_dim(hidden_dim),
                                           _top_k(top_k),
                                           _capacity_factor(capacity_factor),
                                           _overflow_strategy(overflow_strategy),
                                           _balance_loss_weight(balance_loss_weight),
                                           _z_loss_weight(z_loss_weight),
                                           _last_balance_loss(0.0f),
                                           _last_z_loss(0.0f)
{
  try
  {
    ValidateCore(input_dim,hidden_dim,num_experts,top_k);

    BuildPrimitives<ComputeT,StorageT>(input_dim,
                    hidden_dim,
                    num_experts,
                    top_k,
                    expert_use_gated,
                    expert_use_bias,
                    num_shared_experts,
                    shared_hidden_dim,
                    router_use_bias,
                    router_noise_std,
                    gating_kind,
                    norm_topk_prob,
                    routed_scaling_factor,
                    stream,
                    _router,
                    _experts,
                    _shared_experts);

    Stream().Synchronize();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoELayer<ComputeT,StorageT>::CAIF_DeviceMoELayer(uint32_t input_dim,
                                         uint32_t hidden_dim,
                                         uint32_t top_k,
                                         bool router_use_bias,
                                         float router_noise_std,
                                         float capacity_factor,
                                         OverflowStrategy_e overflow_strategy,
                                         float balance_loss_weight,
                                         float z_loss_weight,
                                         std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> routed_experts,
                                         std::vector<std::unique_ptr<CAIF_DeviceMoEExpertBase<ComputeT,StorageT>>> shared_experts,
                                         CAIF_CudaStream &stream,
                                         CAIF_DeviceMoELayerFactory::GatingKind_e gating_kind,
                                         bool norm_topk_prob,
                                         float routed_scaling_factor):CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                                                  _input_dim(input_dim),
                                                                  _hidden_dim(hidden_dim),
                                                                  _top_k(top_k),
                                                                  _capacity_factor(capacity_factor),
                                                                  _overflow_strategy(overflow_strategy),
                                                                  _balance_loss_weight(balance_loss_weight),
                                                                  _z_loss_weight(z_loss_weight),
                                                                  _experts(std::move(routed_experts)),
                                                                  _shared_experts(std::move(shared_experts)),
                                                                  _last_balance_loss(0.0f),
                                                                  _last_z_loss(0.0f)
{
  try
  {
    const uint32_t num_experts=static_cast<uint32_t>(_experts.size());
    ValidateCore(input_dim,hidden_dim,num_experts,top_k);

    typename CAIF_DeviceMoERouter<ComputeT,StorageT>::Config_t router_cfg;
    router_cfg.input_dim=input_dim;
    router_cfg.num_experts=num_experts;
    router_cfg.top_k=top_k;
    router_cfg.routing_type=CAIF_DeviceMoERouter<ComputeT,StorageT>::RoutingType_e::TopK;
    router_cfg.use_bias=router_use_bias;
    router_cfg.noise_std=router_noise_std;
    router_cfg.gating_kind=gating_kind;
    router_cfg.norm_topk_prob=norm_topk_prob;
    router_cfg.routed_scaling_factor=routed_scaling_factor;

    _router=std::make_unique<CAIF_DeviceMoERouter<ComputeT,StorageT>>(router_cfg,stream);

    Stream().Synchronize();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoELayer<ComputeT,StorageT>::CAIF_DeviceMoELayer(CAIF_DeviceMoELayer<ComputeT,StorageT> &&other):CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                                                                     _input_dim(other._input_dim),
                                                                     _hidden_dim(other._hidden_dim),
                                                                     _top_k(other._top_k),
                                                                     _capacity_factor(other._capacity_factor),
                                                                     _overflow_strategy(other._overflow_strategy),
                                                                     _balance_loss_weight(other._balance_loss_weight),
                                                                     _z_loss_weight(other._z_loss_weight),
                                                                     _router(std::move(other._router)),
                                                                     _experts(std::move(other._experts)),
                                                                     _shared_experts(std::move(other._shared_experts)),
                                                                     _cached_routing(std::move(other._cached_routing)),
                                                                     _cached_token_counts(std::move(other._cached_token_counts)),
                                                                     _cached_logsumexp(std::move(other._cached_logsumexp)),
                                                                     _last_balance_loss(other._last_balance_loss),
                                                                     _last_z_loss(other._last_z_loss),
                                                                     _overflow_tokens(std::move(other._overflow_tokens)),
                                                                     _ws_dispatch_map(std::move(other._ws_dispatch_map)),
                                                                     _ws_expert_offsets(std::move(other._ws_expert_offsets)),
                                                                     _ws_expert_input_buffer(std::move(other._ws_expert_input_buffer)),
                                                                     _ws_expert_output_buffer(std::move(other._ws_expert_output_buffer)),
                                                                     _ws_grad_expert_output_buffer(std::move(other._ws_grad_expert_output_buffer)),
                                                                     _ws_grad_expert_input_buffer(std::move(other._ws_grad_expert_input_buffer))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoELayer<ComputeT,StorageT> &CAIF_DeviceMoELayer<ComputeT,StorageT>::operator=(CAIF_DeviceMoELayer<ComputeT,StorageT> &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    _input_dim=other._input_dim;
    _hidden_dim=other._hidden_dim;
    _top_k=other._top_k;
    _capacity_factor=other._capacity_factor;
    _overflow_strategy=other._overflow_strategy;
    _balance_loss_weight=other._balance_loss_weight;
    _z_loss_weight=other._z_loss_weight;
    _router=std::move(other._router);
    _experts=std::move(other._experts);
    _shared_experts=std::move(other._shared_experts);
    _cached_routing=std::move(other._cached_routing);
    _cached_token_counts=std::move(other._cached_token_counts);
    _cached_logsumexp=std::move(other._cached_logsumexp);
    _last_balance_loss=other._last_balance_loss;
    _last_z_loss=other._last_z_loss;
    _overflow_tokens=std::move(other._overflow_tokens);
    _ws_dispatch_map=std::move(other._ws_dispatch_map);
    _ws_expert_offsets=std::move(other._ws_expert_offsets);
    _ws_expert_input_buffer=std::move(other._ws_expert_input_buffer);
    _ws_expert_output_buffer=std::move(other._ws_expert_output_buffer);
    _ws_grad_expert_output_buffer=std::move(other._ws_grad_expert_output_buffer);
    _ws_grad_expert_input_buffer=std::move(other._ws_grad_expert_input_buffer);
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoELayer<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)
{
  try
  {
    const auto &in_shape=input.Shape();
    uint32_t batch_size=1;
    uint32_t seq_len=1;
    uint32_t num_tokens=0;
    uint32_t flat_dim=0;

    if(in_shape.size()==2)
    {
      num_tokens=in_shape[0];
      flat_dim=in_shape[1];
    }
    else if(in_shape.size()==3)
    {
      batch_size=in_shape[0];
      seq_len=in_shape[1];
      num_tokens=batch_size*seq_len;
      flat_dim=in_shape[2];
    }
    else
    {
      THROW_CAIFE("MoELayer::Forward: expected input shape [N, dim] or [batch, seq, dim]");
    }

    if(flat_dim!=_input_dim)
    {
      THROW_CAIFE("MoELayer::Forward: input dimension mismatch");
    }

    // Non-owning view — downstream reads (router, dispatch, experts) are const,
    // so aliasing the caller's buffer is safe and avoids a full device clone.
    CAIF_DeviceTensor flat_input=CAIF_DeviceTensor::WrapView(
      const_cast<void*>(input.DeviceDataRaw()),
      {num_tokens,flat_dim},
      Stream(),
      input.Dtype());

#ifdef USE_CAIF_CUDA
    cudaEvent_t mt_ev_start=nullptr;
    cudaEvent_t mt_ev_router=nullptr;
    cudaEvent_t mt_ev_dispatch_map=nullptr;
    cudaEvent_t mt_ev_dispatch_gpu=nullptr;
    cudaEvent_t mt_ev_experts=nullptr;
    cudaEvent_t mt_ev_combine=nullptr;
    if(g_caif_moe_forward_trace_enabled==true)
    {
      cudaEventCreate(&mt_ev_start);
      cudaEventCreate(&mt_ev_router);
      cudaEventCreate(&mt_ev_dispatch_map);
      cudaEventCreate(&mt_ev_dispatch_gpu);
      cudaEventCreate(&mt_ev_experts);
      cudaEventCreate(&mt_ev_combine);
      cudaEventRecord(mt_ev_start,Stream().Handle());
    }
#endif

    // ---- Route once. Always route once per forward; cache for backward
    //      only in training mode. Dispatch, combine, and aux-loss grads all
    //      read from this single RouterOutput_t.
    typename CAIF_DeviceMoERouter<ComputeT,StorageT>::RouterOutput_t routing=_router->Route(flat_input,ctx);
    const typename CAIF_DeviceMoERouter<ComputeT,StorageT>::RouterOutput_t &r=routing;

#ifdef USE_CAIF_CUDA
    if(g_caif_moe_forward_trace_enabled==true)
    {
      cudaEventRecord(mt_ev_router,Stream().Handle());
    }
#endif

    // ---- Device-resident dispatch/combine path.
    //
    //   Old: MoEDispatch + MoECombine ran entirely on the host — CopyToHost
    //   for input tokens + indices, host loop rearranging into per-expert
    //   buffers, CopyFromHost back. That round-trip of N*dim floats per op
    //   dominated MoE Forward time (5-10 ms on a 4096x512 workload). The
    //   GPU variants (MoEDispatchGPU / MoECombineGPU) concatenate all expert
    //   inputs into a single [total_assigned, dim] buffer addressed via a
    //   dispatch_map (slot per token-expert assignment) + expert_offsets
    //   (start of each expert's block in the buffer).
    //
    //   MoEBuildDispatchMap still does one small host sync to read back the
    //   expert_indices (N*K int32 — tiny compared to the N*dim tensor data
    //   we used to round-trip). Only Drop overflow is supported on the GPU
    //   fast path; NoOp/Redistribute aren't benched or covered by MoE
    //   parity so we fail loudly if requested.
    if(_overflow_strategy!=OverflowStrategy_e::Drop)
    {
      THROW_CAIFE("MoELayer::Forward: only Drop overflow supported on GPU dispatch path");
    }

    const uint32_t num_experts=static_cast<uint32_t>(_experts.size());
    const uint32_t capacity=static_cast<uint32_t>(
      std::ceil(static_cast<float>(num_tokens)/num_experts*_capacity_factor*_top_k));

    if(_ws_dispatch_map.TotalElements()!=static_cast<size_t>(num_tokens)*_top_k)
    {
      _ws_dispatch_map=CAIF_DeviceTensor::Uninitialized({num_tokens,_top_k},
                                                        Stream(),
                                                        CAIF_DataType::CAIF_DataType_e::Int32);
    }
    if(_ws_expert_offsets.TotalElements()!=num_experts+1)
    {
      _ws_expert_offsets=CAIF_DeviceTensor::Uninitialized({num_experts+1},
                                                          Stream(),
                                                          CAIF_DataType::CAIF_DataType_e::Int32);
    }

    const uint32_t total_assigned=CAIF_Ops::MoEBuildDispatchMap(r.expert_indices,
                                                                      num_experts,
                                                                      _top_k,
                                                                      capacity,
                                                                      _ws_dispatch_map,
                                                                      _ws_expert_offsets);

    std::vector<int32_t> offsets_host(num_experts+1);
    _ws_expert_offsets.CopyToHostRaw(offsets_host.data());

    _cached_token_counts.assign(num_experts,0u);
    for(uint32_t e=0;e<num_experts;++e)
    {
      _cached_token_counts[e]=static_cast<uint32_t>(offsets_host[e+1]-offsets_host[e]);
    }

#ifdef USE_CAIF_CUDA
    if(g_caif_moe_forward_trace_enabled==true)
    {
      cudaEventRecord(mt_ev_dispatch_map,Stream().Handle());
    }
#endif

    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const size_t elem_bytes=CAIF_DataType(sd).ElementSizeBytes();

    const uint32_t buf_rows=std::max<uint32_t>(total_assigned,1u);
    if(_ws_expert_input_buffer.TotalElements()!=static_cast<size_t>(buf_rows)*_input_dim
       ||_ws_expert_input_buffer.Dtype()!=sd)
    {
      _ws_expert_input_buffer=CAIF_DeviceTensor::Uninitialized({buf_rows,_input_dim},Stream(),sd);
    }
    if(_ws_expert_output_buffer.TotalElements()!=static_cast<size_t>(buf_rows)*_input_dim
       ||_ws_expert_output_buffer.Dtype()!=sd)
    {
      _ws_expert_output_buffer=CAIF_DeviceTensor::Uninitialized({buf_rows,_input_dim},Stream(),sd);
    }

    if(total_assigned>0)
    {
      CAIF_Ops::MoEDispatchGPU(flat_input,
                                     r.expert_indices,
                                     _ws_dispatch_map,
                                     _ws_expert_offsets,
                                     _top_k,
                                     _ws_expert_input_buffer);
    }

#ifdef USE_CAIF_CUDA
    if(g_caif_moe_forward_trace_enabled==true)
    {
      cudaEventRecord(mt_ev_dispatch_gpu,Stream().Handle());
    }
#endif

    // Per-expert forward: write each expert's output directly into its
    // pre-assigned slice of the packed `_ws_expert_output_buffer`,
    // mirroring the same offset layout as `_ws_expert_input_buffer`.
    //
    // Why the in-place write matters:
    //   The pre-ForwardInto code did `out=_experts[i]->Forward(in_view)`
    //   (allocating a per-expert owning tensor) and then a separate
    //   `cudaMemcpyAsync(out_ptr_bytes, out.DeviceDataRaw(), ...)` to
    //   copy that tensor into the workspace slice. Per-expert that's
    //   1 alloc + 1 D2D copy of cnt*input_dim*elem_bytes per token group.
    //   With num_experts up to 64 in production MoE shapes, that adds
    //   up to a measurable hot path (compare_multirun.py min-vs-min
    //   confirms: MoE Backward prod fp32 -1.282 ms, MoE Forward prod
    //   training fp32 -0.406 ms vs the post-phase4 peak baseline).
    //
    // Why ForwardInto can't just be ForwardImpl with an output param:
    //   Some experts use the projection-pluggable path
    //   (`_use_projections==true`), where the down-projection layer
    //   allocates its own output tensor inside its Forward(). For those,
    //   ForwardInto still falls back to a D2D copy (see moe_expert.cpp).
    //   The non-projection path — which is the bench/perf-critical path
    //   — runs `MatMul(hidden, _w_down, output)` writing straight into
    //   the caller's slice, and *that* is what avoids the alloc + copy.
    //
    // Why the views are safe:
    //   - `in_view` and `out_view` reference distinct byte ranges in
    //     two different workspace tensors (_ws_expert_input_buffer vs
    //     _ws_expert_output_buffer), so the expert's forward never
    //     reads and writes the same bytes through different aliases.
    //   - The workspaces are member tensors of the layer; they live
    //     until the layer is destroyed, so the views are valid for the
    //     entire forward + backward window. Backward (BackwardImpl)
    //     reads `_ws_expert_output_buffer` directly via offsets, so the
    //     `expert_outputs` vector below is purely a forward-local
    //     bookkeeping list, not a backward dependency.
    //   - `Stream()` is the layer's stream, which is also the stream
    //     the workspaces were allocated against and the experts run on.
    //     No cross-stream race.
    //
    // expert_outputs[i] holds a non-owning view into the workspace slice
    // for expert i. It exists for two reasons:
    //   a) the empty `CAIF_DeviceTensor()` placeholder slot for
    //      cnt==0 experts keeps the vector index aligned with expert
    //      index, which the combine stage reads from indirectly via
    //      offsets. Moving an empty placeholder is cheap.
    //   b) some downstream code paths (shared experts, expert-choice
    //      routing variants in derived classes) iterate the expert
    //      outputs by index. Keeping the vector populated preserves
    //      that contract even though combine itself doesn't use it.
    std::vector<CAIF_DeviceTensor> expert_outputs;
    expert_outputs.reserve(num_experts);

    for(uint32_t i=0;i<num_experts;++i)
    {
      const uint32_t cnt=_cached_token_counts[i];
      if(cnt==0)
      {
        expert_outputs.push_back(CAIF_DeviceTensor());
        continue;
      }
      const uint32_t off=static_cast<uint32_t>(offsets_host[i]);
      uint8_t *in_ptr_bytes=static_cast<uint8_t*>(_ws_expert_input_buffer.DeviceDataRaw())
                            +static_cast<size_t>(off)*_input_dim*elem_bytes;
      uint8_t *out_ptr_bytes=static_cast<uint8_t*>(_ws_expert_output_buffer.DeviceDataRaw())
                             +static_cast<size_t>(off)*_input_dim*elem_bytes;
      CAIF_DeviceTensor in_view=CAIF_DeviceTensor::WrapView(in_ptr_bytes,
                                                            {cnt,_input_dim},
                                                            Stream(),
                                                            _ws_expert_input_buffer.Dtype());
      CAIF_DeviceTensor out_view=CAIF_DeviceTensor::WrapView(out_ptr_bytes,
                                                             {cnt,_input_dim},
                                                             Stream(),
                                                             _ws_expert_output_buffer.Dtype());
      _experts[i]->ForwardInto(in_view,out_view,ctx);
      expert_outputs.push_back(std::move(out_view));
    }

#ifdef USE_CAIF_CUDA
    if(g_caif_moe_forward_trace_enabled==true)
    {
      cudaEventRecord(mt_ev_experts,Stream().Handle());
    }
#endif

    CAIF_DeviceTensor combined=CAIF_DeviceTensor::Zeros({num_tokens,_input_dim},Stream(),sd);
    if(total_assigned>0)
    {
      CAIF_Ops::MoECombineGPU(_ws_expert_output_buffer,
                                    r.expert_indices,
                                    r.expert_weights,
                                    _ws_dispatch_map,
                                    _ws_expert_offsets,
                                    _top_k,
                                    combined);
    }

#ifdef USE_CAIF_CUDA
    if(g_caif_moe_forward_trace_enabled==true)
    {
      cudaEventRecord(mt_ev_combine,Stream().Handle());
      cudaEventSynchronize(mt_ev_combine);
      float ms_router=0.0f;
      float ms_dispatch_map=0.0f;
      float ms_dispatch_gpu=0.0f;
      float ms_experts=0.0f;
      float ms_combine=0.0f;
      cudaEventElapsedTime(&ms_router,mt_ev_start,mt_ev_router);
      cudaEventElapsedTime(&ms_dispatch_map,mt_ev_router,mt_ev_dispatch_map);
      cudaEventElapsedTime(&ms_dispatch_gpu,mt_ev_dispatch_map,mt_ev_dispatch_gpu);
      cudaEventElapsedTime(&ms_experts,mt_ev_dispatch_gpu,mt_ev_experts);
      cudaEventElapsedTime(&ms_combine,mt_ev_experts,mt_ev_combine);
      std::fprintf(stderr,
                   "[MoE-FWD] tokens=%u experts=%u "
                   "router=%.3f dispatch_map=%.3f dispatch_gpu=%.3f "
                   "experts=%.3f combine=%.3f\n",
                   num_tokens,
                   num_experts,
                   static_cast<double>(ms_router),
                   static_cast<double>(ms_dispatch_map),
                   static_cast<double>(ms_dispatch_gpu),
                   static_cast<double>(ms_experts),
                   static_cast<double>(ms_combine));
      cudaEventDestroy(mt_ev_start);
      cudaEventDestroy(mt_ev_router);
      cudaEventDestroy(mt_ev_dispatch_map);
      cudaEventDestroy(mt_ev_dispatch_gpu);
      cudaEventDestroy(mt_ev_experts);
      cudaEventDestroy(mt_ev_combine);
    }
#endif

    // ---- Shared experts: run on full flat_input, add to combined
    std::vector<CAIF_DeviceTensor> shared_outputs;
    if(_shared_experts.size()>0)
    {
      shared_outputs.reserve(_shared_experts.size());
      for(size_t i=0;i<_shared_experts.size();++i)
      {
        CAIF_DeviceTensor s=_shared_experts[i]->Forward(flat_input,ctx);
        CAIF_Ops::Add(combined,s,combined);
        shared_outputs.push_back(std::move(s));
      }
    }

    // ---- Aux losses (training only)
    _last_balance_loss=0.0f;
    _last_z_loss=0.0f;
    _cached_logsumexp=CAIF_DeviceTensor();

    if(ctx.Training()==true)
    {
      if(_balance_loss_weight>0.0f)
      {
        _last_balance_loss=ComputeBalanceLoss(r.router_probs);
      }
      if(_z_loss_weight>0.0f)
      {
        _last_z_loss=ComputeZLoss(r.router_logits);
      }
    }

    // ---- Commit caches for backward
    // BackwardImpl reads `_ws_expert_output_buffer` directly (the packed
    // workspace ForwardInto wrote into above) plus `_cached_routing`
    // and `_cached_token_counts`. The `expert_outputs` vector is a
    // forward-local list of non-owning views and is intentionally NOT
    // cached — caching the views would just duplicate pointers into the
    // same workspace, and dropping it lets the views' destructors run
    // (they don't free anything, they just zero out their device-pointer
    // fields, which is correct now that forward is done with them).
    if(ctx.Training()==true)
    {
      _cached_routing=std::move(routing);
    }
    (void)expert_outputs;

    if(in_shape.size()==3)
    {
      combined.Reshape({batch_size,seq_len,_input_dim});
    }
    return combined;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoELayer<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)
{
  try
  {
    const auto &g_shape=grad_output.Shape();
    uint32_t num_tokens=0;
    uint32_t flat_dim=0;

    if(g_shape.size()==2)
    {
      num_tokens=g_shape[0];
      flat_dim=g_shape[1];
    }
    else if(g_shape.size()==3)
    {
      num_tokens=g_shape[0]*g_shape[1];
      flat_dim=g_shape[2];
    }
    else
    {
      THROW_CAIFE("MoELayer::Backward: expected grad shape [N, dim] or [batch, seq, dim]");
    }

    CAIF_DeviceTensor flat_grad=CAIF_DeviceTensor::WrapView(
      const_cast<void*>(grad_output.DeviceDataRaw()),
      {num_tokens,flat_dim},
      Stream(),
      grad_output.Dtype());

    const uint32_t num_experts=static_cast<uint32_t>(_experts.size());
    const typename CAIF_DeviceMoERouter<ComputeT,StorageT>::RouterOutput_t &r=_cached_routing;

    // ---- GPU combine/dispatch backward using workspace buffers
    //
    //   Forward left _ws_expert_output_buffer populated with per-expert outputs
    //   concatenated by expert_offsets, and _ws_dispatch_map / _ws_expert_offsets
    //   describing the token-to-slot mapping. MoECombineBackwardGPU scatters
    //   grad_output into the grad_expert_output buffer and computes grad_weights
    //   in one kernel pair, fully on device. After per-expert Backward fills the
    //   grad_expert_input buffer, MoEDispatchBackwardGPU gathers per-token input
    //   gradients from it. This replaces the old host-roundtrip MoECombineBackward
    //   / MoEDispatchBackward path.
    uint32_t total_assigned=0;
    for(uint32_t e=0;e<num_experts;++e)
    {
      total_assigned+=_cached_token_counts[e];
    }
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const size_t elem_bytes=CAIF_DataType(sd).ElementSizeBytes();
    const uint32_t buf_rows=std::max<uint32_t>(total_assigned,1u);
    if(_ws_grad_expert_output_buffer.TotalElements()!=static_cast<size_t>(buf_rows)*_input_dim
       ||_ws_grad_expert_output_buffer.Dtype()!=sd)
    {
      _ws_grad_expert_output_buffer=CAIF_DeviceTensor::Uninitialized({buf_rows,_input_dim},Stream(),sd);
    }
    if(_ws_grad_expert_input_buffer.TotalElements()!=static_cast<size_t>(buf_rows)*_input_dim
       ||_ws_grad_expert_input_buffer.Dtype()!=sd)
    {
      _ws_grad_expert_input_buffer=CAIF_DeviceTensor::Uninitialized({buf_rows,_input_dim},Stream(),sd);
    }

    CAIF_DeviceTensor grad_weights=CAIF_DeviceTensor::Zeros({num_tokens,_top_k},Stream(),sd);

    if(total_assigned>0)
    {
      CAIF_Ops::MoECombineBackwardGPU(flat_grad,
                                            _ws_expert_output_buffer,
                                            r.expert_indices,
                                            r.expert_weights,
                                            _ws_dispatch_map,
                                            _ws_expert_offsets,
                                            _top_k,
                                            _ws_grad_expert_output_buffer,
                                            grad_weights);
    }

    // ---- Per-expert Backward: slice grad_output buffer, run expert Backward,
    //      copy result into grad_input buffer at the same offset.
    std::vector<int32_t> offsets_host(num_experts+1,0);
    for(uint32_t e=0;e<num_experts;++e)
    {
      offsets_host[e+1]=offsets_host[e]+static_cast<int32_t>(_cached_token_counts[e]);
    }

    for(uint32_t i=0;i<num_experts;++i)
    {
      const uint32_t cnt=_cached_token_counts[i];
      if(cnt==0)
      {
        continue;
      }
      const uint32_t off=static_cast<uint32_t>(offsets_host[i]);
      uint8_t *g_out_ptr_bytes=static_cast<uint8_t*>(_ws_grad_expert_output_buffer.DeviceDataRaw())
                               +static_cast<size_t>(off)*_input_dim*elem_bytes;
      CAIF_DeviceTensor g_out_view=CAIF_DeviceTensor::WrapView(g_out_ptr_bytes,
                                                                {cnt,_input_dim},
                                                                Stream(),
                                                                _ws_grad_expert_output_buffer.Dtype());
      CAIF_DeviceTensor g_in=_experts[i]->Backward(g_out_view,ctx);
#ifdef USE_CAIF_CUDA
      uint8_t *g_in_ptr_bytes=static_cast<uint8_t*>(_ws_grad_expert_input_buffer.DeviceDataRaw())
                              +static_cast<size_t>(off)*_input_dim*elem_bytes;
      cudaMemcpyAsync(g_in_ptr_bytes,
                      g_in.DeviceDataRaw(),
                      static_cast<size_t>(cnt)*_input_dim*elem_bytes,
                      cudaMemcpyDeviceToDevice,
                      Stream().Handle());
#endif
    }

    // ---- Dispatch backward -> grad_input_from_experts
    CAIF_DeviceTensor grad_input_from_experts=CAIF_DeviceTensor::Zeros({num_tokens,_input_dim},Stream(),sd);
    if(total_assigned>0)
    {
      CAIF_Ops::MoEDispatchBackwardGPU(_ws_grad_expert_input_buffer,
                                             r.expert_indices,
                                             _ws_dispatch_map,
                                             _ws_expert_offsets,
                                             _top_k,
                                             grad_input_from_experts);
    }

    // ---- Aux-loss gradient injection via router backward.
    //
    //   L_balance = bw * E * sum_e(f_e * P_e),  f_e = count[e] / (N*K)
    //   dL_balance / dprobs_{t,e} = bw * E * count[e] / (N^2 * K)
    //     — constant per expert column, uploaded as an [E]-shaped bias.
    //
    //   L_z = zw * (1/N) * sum_t (logsumexp_t)^2
    //   dL_z / dlogit_{t,e} = (2 * zw / N) * logsumexp_t * probs_{t,e}
    //     — row-broadcast multiply-add in the aux-aware router backward.
    CAIF_DeviceTensor balance_bias;
    CAIF_DeviceTensor z_logsumexp_scaled;

    if(_balance_loss_weight>0.0f)
    {
      const float denom=static_cast<float>(num_tokens)*
                         static_cast<float>(num_tokens)*
                         static_cast<float>(_top_k);
      const float scale=_balance_loss_weight*static_cast<float>(num_experts)/denom;
      std::vector<float> balance_bias_host(num_experts);
      for(uint32_t e=0;e<num_experts;++e)
      {
        balance_bias_host[e]=scale*static_cast<float>(_cached_token_counts[e]);
      }
      // Allocate the destination tensor at StorageT directly and upload
      // host-side fp32 with conversion at the boundary — no device-side
      // fp32 staging tensor.
      balance_bias=CAIF_DeviceTensor::Uninitialized({num_experts},Stream(),sd);
      balance_bias.CopyFromHostFp32(balance_bias_host.data(),balance_bias_host.size());
    }

    if(_z_loss_weight>0.0f&&_cached_logsumexp.Shape().size()>0)
    {
      const float scale=2.0f*_z_loss_weight/static_cast<float>(num_tokens);
      z_logsumexp_scaled=CAIF_DeviceTensor::Uninitialized({num_tokens},Stream(),sd);
      CAIF_Ops::Scale(_cached_logsumexp,scale,z_logsumexp_scaled);
    }

    CAIF_DeviceTensor grad_input_from_router=_router->BackwardRoutingAuxAware(grad_weights,
                                                                               balance_bias,
                                                                               z_logsumexp_scaled,
                                                                               ctx);

    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,_input_dim},Stream(),sd);
    CAIF_Ops::Add(grad_input_from_experts,grad_input_from_router,grad_input);

    // ---- Shared experts backward: they saw flat_grad as their grad_output.
    for(size_t i=0;i<_shared_experts.size();++i)
    {
      CAIF_DeviceTensor gs=_shared_experts[i]->Backward(flat_grad,ctx);
      CAIF_Ops::Add(grad_input,gs,grad_input);
    }

    if(g_shape.size()==3)
    {
      grad_input.Reshape({g_shape[0],g_shape[1],g_shape[2]});
    }
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
float CAIF_DeviceMoELayer<ComputeT,StorageT>::ComputeBalanceLoss(const CAIF_DeviceTensor &router_probs)
{
  try
  {
    const uint32_t num_tokens=router_probs.Shape()[0];
    const uint32_t num_experts=static_cast<uint32_t>(_experts.size());
    const float total_assignments=static_cast<float>(num_tokens)*static_cast<float>(_top_k);

    std::vector<float> fractions(num_experts,0.0f);
    for(uint32_t i=0;i<num_experts;++i)
    {
      fractions[i]=static_cast<float>(_cached_token_counts[i])/total_assignments;
    }

    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    CAIF_DeviceTensor sum_probs=CAIF_DeviceTensor::Uninitialized({num_experts},Stream(),sd);
    CAIF_Ops::SumAxis(router_probs,0,sum_probs);

    std::vector<float> sum_probs_host(num_experts);
    if(sd==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      sum_probs.CopyToHost(sum_probs_host.data());
    }
    else
    {
      CAIF_DeviceTensor sum_probs_fp32=sum_probs.To(CAIF_DataType::CAIF_DataType_e::Float32);
      sum_probs_fp32.CopyToHost(sum_probs_host.data());
    }

    float loss=0.0f;
    for(uint32_t i=0;i<num_experts;++i)
    {
      const float mean_p=sum_probs_host[i]/static_cast<float>(num_tokens);
      loss+=fractions[i]*mean_p;
    }
    return _balance_loss_weight*static_cast<float>(num_experts)*loss;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
float CAIF_DeviceMoELayer<ComputeT,StorageT>::ComputeZLoss(const CAIF_DeviceTensor &router_logits)
{
  try
  {
    const uint32_t num_tokens=router_logits.Shape()[0];
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();

    _cached_logsumexp=CAIF_DeviceTensor::Uninitialized({num_tokens},Stream(),sd);
    CAIF_Ops::LogSumExp(router_logits,_cached_logsumexp);

    CAIF_DeviceTensor logsumexp_sq=CAIF_DeviceTensor::Uninitialized({num_tokens},Stream(),sd);
    CAIF_Ops::Multiply(_cached_logsumexp,_cached_logsumexp,logsumexp_sq);

    // Sum is an atomicAdd accumulator — must start from zero, not
    // uninitialized memory. Uninitialized garbage was getting added
    // into the result, producing huge values (e.g. -2.4e24 at bf16
    // storage on layer 2 in the DSv2-Lite full-depth smoke).
    CAIF_DeviceTensor total=CAIF_DeviceTensor::Zeros({1},Stream(),sd);
    CAIF_Ops::Sum(logsumexp_sq,total);

    float total_host=0.0f;
    if(sd==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      total.CopyToHost(&total_host);
    }
    else
    {
      CAIF_DeviceTensor total_fp32=total.To(CAIF_DataType::CAIF_DataType_e::Float32);
      total_fp32.CopyToHost(&total_host);
    }

    return _z_loss_weight*total_host/static_cast<float>(num_tokens);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoELayer<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    _router->ZeroGradients();
    for(auto &e:_experts)
    {
      e->ZeroGradients();
    }
    for(auto &s:_shared_experts)
    {
      s->ZeroGradients();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMoELayer<ComputeT,StorageT>::ParameterTensorCount()const
{
  size_t count=_router->ParameterTensorCount();
  for(const auto &e:_experts)
  {
    count+=e->ParameterTensorCount();
  }
  for(const auto &s:_shared_experts)
  {
    count+=s->ParameterTensorCount();
  }
  return count;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMoELayer<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  size_t router_count=_router->ParameterTensorCount();
  if(index<router_count)
  {
    return _router->ParameterTensor(index);
  }
  index-=router_count;

  for(auto &e:_experts)
  {
    size_t n=e->ParameterTensorCount();
    if(index<n)
    {
      return e->ParameterTensor(index);
    }
    index-=n;
  }
  for(auto &s:_shared_experts)
  {
    size_t n=s->ParameterTensorCount();
    if(index<n)
    {
      return s->ParameterTensor(index);
    }
    index-=n;
  }
  THROW_CAIFE("MoELayer::ParameterTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMoELayer<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoELayer*>(this)->ParameterTensor(index);
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMoELayer<ComputeT,StorageT>::GradientTensor(size_t index)
{
  size_t router_count=_router->ParameterTensorCount();
  if(index<router_count)
  {
    return _router->GradientTensor(index);
  }
  index-=router_count;

  for(auto &e:_experts)
  {
    size_t n=e->ParameterTensorCount();
    if(index<n)
    {
      return e->GradientTensor(index);
    }
    index-=n;
  }
  for(auto &s:_shared_experts)
  {
    size_t n=s->ParameterTensorCount();
    if(index<n)
    {
      return s->GradientTensor(index);
    }
    index-=n;
  }
  THROW_CAIFE("MoELayer::GradientTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMoELayer<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoELayer*>(this)->GradientTensor(index);
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMoELayer<ComputeT,StorageT>::TotalParameterCount()const
{
  size_t count=_router->TotalParameterCount();
  for(const auto &e:_experts)
  {
    count+=e->TotalParameterCount();
  }
  for(const auto &s:_shared_experts)
  {
    count+=s->TotalParameterCount();
  }
  return count;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceMoELayer<ComputeT,StorageT>::Description()const
{
  std::string desc="MoELayer[";
  desc+=std::to_string(_input_dim)+"->"+std::to_string(_hidden_dim);
  desc+=",experts="+std::to_string(_experts.size());
  if(_shared_experts.size()>0)
  {
    desc+=",shared="+std::to_string(_shared_experts.size());
  }
  desc+=",top_k="+std::to_string(_top_k);
  desc+=",cap="+std::to_string(_capacity_factor);
  desc+="]";
  return desc;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceMoELayer<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  std::vector<std::string> names;

  std::vector<std::string> router_names=_router->ParameterNames(prefix+"router.");
  names.insert(names.end(),router_names.begin(),router_names.end());

  for(size_t i=0;i<_experts.size();++i)
  {
    std::string expert_prefix=prefix+"expert_"+std::to_string(i)+".";
    std::vector<std::string> en=_experts[i]->ParameterNames(expert_prefix);
    names.insert(names.end(),en.begin(),en.end());
  }

  for(size_t i=0;i<_shared_experts.size();++i)
  {
    std::string shared_prefix=prefix+"shared_expert_"+std::to_string(i)+".";
    std::vector<std::string> sn=_shared_experts[i]->ParameterNames(shared_prefix);
    names.insert(names.end(),sn.begin(),sn.end());
  }
  return names;
}
// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceMoELayer<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceMoELayer<float,__half>;
template class CAIF_DeviceMoELayer<float,__nv_bfloat16>;
template class CAIF_DeviceMoELayer<__half,float>;
template class CAIF_DeviceMoELayer<__half,__half>;
template class CAIF_DeviceMoELayer<__half,__nv_bfloat16>;
template class CAIF_DeviceMoELayer<__nv_bfloat16,float>;
template class CAIF_DeviceMoELayer<__nv_bfloat16,__half>;
template class CAIF_DeviceMoELayer<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
