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

#include "caif_device_moe_frozen_expert.h"
#include "caif_ops.h"
#include "caif_exception.h"
#ifdef USE_CAIF_CUDA
#include <cuda_runtime_api.h>
#endif

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::CAIF_DeviceMoEFrozenExpert(
    const CAIF_DeviceMoEFrozenExpertConfig &config,
    FrozenSubLayers_t sub_layers,
    CAIF_CudaStream &stream)
  :CAIF_DeviceMoEExpertBase<ComputeT,StorageT>(stream)
  ,_config(config)
  ,_sub_layers(std::move(sub_layers))
{
  try
  {
    if(Config().InputDim()==0)
    {
      THROW_CAIFE("MoEFrozenExpert: input_dim must be > 0");
    }
    if(Config().HiddenDim()==0)
    {
      THROW_CAIFE("MoEFrozenExpert: hidden_dim must be > 0");
    }
    if(SubLayers().up==nullptr)
    {
      THROW_CAIFE("MoEFrozenExpert: up sub-layer required");
    }
    if(SubLayers().down==nullptr)
    {
      THROW_CAIFE("MoEFrozenExpert: down sub-layer required");
    }
    if(Config().UseGated()==true&&SubLayers().gate==nullptr)
    {
      THROW_CAIFE("MoEFrozenExpert: gated config requires gate sub-layer");
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::CAIF_DeviceMoEFrozenExpert(
    CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT> &&other)
  :CAIF_DeviceMoEExpertBase<ComputeT,StorageT>(std::move(other))
  ,_config(other._config)
  ,_sub_layers(std::move(other._sub_layers))
  ,_cached_input(std::move(other._cached_input))
  ,_cached_gate_out(std::move(other._cached_gate_out))
  ,_cached_up_out(std::move(other._cached_up_out))
  ,_cached_hidden(std::move(other._cached_hidden))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT> &
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::operator=(
    CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT> &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceMoEExpertBase<ComputeT,StorageT>::operator=(std::move(other));
    SetConfig(other.Config());
    SetSubLayers(std::move(other.SubLayersMut()));
    SetCachedInput(std::move(other.CachedInput()));
    SetCachedGateOut(std::move(other.CachedGateOut()));
    SetCachedUpOut(std::move(other.CachedUpOut()));
    SetCachedHidden(std::move(other.CachedHidden()));
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::ForwardImpl(
    const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)
{
  try
  {
    const auto &shape=input.Shape();
    if(shape.size()!=2||shape[1]!=Config().InputDim())
    {
      THROW_CAIFE("MoEFrozenExpert::Forward: expected [N, input_dim]");
    }
    const uint32_t num_tokens=shape[0];
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();

    if(ctx.Training()==true)
    {
      SetCachedInput(input.Clone());
    }

    CAIF_DeviceLayer *up_layer=dynamic_cast<CAIF_DeviceLayer*>(SubLayers().up.get());
    CAIF_DeviceLayer *down_layer=dynamic_cast<CAIF_DeviceLayer*>(SubLayers().down.get());
    // FrozenLinear always emits its output at Float32 (its internal
    // cublasGemmEx writes CUDA_R_32F regardless of the cell). Cast to
    // StorageT here so the rest of the gated-FFN pipeline operates in
    // a consistent dtype (SiLU/Multiply/Down's input all want sd).
    CAIF_DeviceTensor up_out_fp32=up_layer->Forward(input,ctx);
    CAIF_DeviceTensor up_out;
    if(up_out_fp32.Dtype()==sd)
    {
      up_out=std::move(up_out_fp32);
    }
    else
    {
      up_out=up_out_fp32.To(sd);
    }

    CAIF_DeviceTensor hidden;
    if(Config().UseGated()==true)
    {
      CAIF_DeviceLayer *gate_layer=dynamic_cast<CAIF_DeviceLayer*>(SubLayers().gate.get());
      CAIF_DeviceTensor gate_out_fp32=gate_layer->Forward(input,ctx);
      CAIF_DeviceTensor gate_out;
      if(gate_out_fp32.Dtype()==sd)
      {
        gate_out=std::move(gate_out_fp32);
      }
      else
      {
        gate_out=gate_out_fp32.To(sd);
      }
      // SwiGLU: silu(gate) * up. Matches HF convention
      // `down_proj(act_fn(gate_proj(x)) * up_proj(x))`. The previous
      // `gate_out * silu(up_out)` pattern swapped the silu target and
      // broke parity vs the reference implementation on every loaded
      // MoE expert (the trained weights aren't symmetric in gate/up).
      // Same fix sits in CAIF_DeviceMoEExpert.
      CAIF_DeviceTensor gate_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},
                                                                        Stream(),
                                                                        sd);
      CAIF_Ops::SiLU(gate_out,gate_activated);
      hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::Multiply(gate_activated,up_out,hidden);
      if(ctx.Training()==true)
      {
        SetCachedGateOut(std::move(gate_out));
      }
    }
    else
    {
      hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::SiLU(up_out,hidden);
    }

    if(ctx.Training()==true)
    {
      SetCachedUpOut(std::move(up_out));
      SetCachedHidden(hidden.Clone());
    }

    CAIF_DeviceTensor down_out_fp32=down_layer->Forward(hidden,ctx);
    if(down_out_fp32.Dtype()==sd)
    {
      return down_out_fp32;
    }
    return down_out_fp32.To(sd);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::ForwardInto(
    const CAIF_DeviceTensor &input,
    CAIF_DeviceTensor &output,
    CAIF_RunContext &ctx)
{
  try
  {
    // FrozenLinear's Forward allocates its own output tensor — we cannot
    // pre-place the down-projection's output into the caller's slice.
    // Fall back to a single cudaMemcpyAsync from the down output into
    // the slice. This mirrors the trainable expert's `_use_projections`
    // ForwardInto fallback (the perf-critical path is the trainable
    // non-projections one; frozen experts on the per-expert hot loop
    // pay one extra D2D copy per forward, same as the projections
    // branch).
    const auto &out_shape=output.Shape();
    if(out_shape.size()!=2||out_shape[0]!=input.Shape()[0]||out_shape[1]!=Config().InputDim())
    {
      THROW_CAIFE("MoEFrozenExpert::ForwardInto: output shape must be [N, input_dim]");
    }
    if(output.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("MoEFrozenExpert::ForwardInto: output dtype must match StorageDtype()");
    }
    CAIF_DeviceTensor produced=ForwardImpl(input,ctx);
    // FrozenLinear's matmul output lands at ComputeDtype (fp32 for the
    // mixed-cell `<fp32, bf16>` / `<fp32, fp16>` paths), but the caller-
    // allocated `output` slice is at StorageDtype (sd). Cast before the
    // memcpy — otherwise we'd copy `bytes = num_elements * sd_size`
    // bytes from a fp32-laid-out buffer, leaking the upper-16 of every
    // fp32 lane into adjacent bf16/fp16 slots and producing garbage.
    if(produced.Dtype()!=output.Dtype())
    {
      produced=produced.To(output.Dtype());
    }
#ifdef USE_CAIF_CUDA
    const size_t bytes=
      static_cast<size_t>(out_shape[0])*out_shape[1]
      *CAIF_DataType(StorageDtype()).ElementSizeBytes();
    cudaMemcpyAsync(output.DeviceDataRaw(),
                    produced.DeviceDataRaw(),
                    bytes,
                    cudaMemcpyDeviceToDevice,
                    Stream().Handle());
#endif
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::BackwardImpl(
    const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)
{
  try
  {
    if(CachedInput().IsEmpty()==true)
    {
      THROW_CAIFE("MoEFrozenExpert::Backward: must call Forward(training=true) first");
    }

    const uint32_t num_tokens=CachedInput().Shape()[0];
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();

    CAIF_DeviceLayer *up_layer=dynamic_cast<CAIF_DeviceLayer*>(SubLayers().up.get());
    CAIF_DeviceLayer *down_layer=dynamic_cast<CAIF_DeviceLayer*>(SubLayers().down.get());

    // FrozenLinear::Backward returns the grad_input at Float32; cast each
    // grad to StorageT before composing with SiLU/Multiply (which both
    // want all operands at the same dtype).
    CAIF_DeviceTensor grad_hidden_fp32=down_layer->Backward(grad_output,ctx);
    CAIF_DeviceTensor grad_hidden;
    if(grad_hidden_fp32.Dtype()==sd)
    {
      grad_hidden=std::move(grad_hidden_fp32);
    }
    else
    {
      grad_hidden=grad_hidden_fp32.To(sd);
    }

    if(Config().UseGated()==true)
    {
      CAIF_DeviceLayer *gate_layer=dynamic_cast<CAIF_DeviceLayer*>(SubLayers().gate.get());

      // Backward through SwiGLU (silu(gate) * up). Mirrors the forward
      // semantics in the Forward path above:
      //   hidden = silu(gate_out) * up_out
      //   grad_up           = grad_hidden * silu(gate_out)
      //   grad_gate_activated = grad_hidden * up_out
      //   grad_gate         = SiLUBackward(gate_out, grad_gate_activated)
      // Re-materialize SiLU(_cached_gate_out) since we don't cache the
      // post-activation tensor (fp16/bf16 cache cost > recompute cost).
      CAIF_DeviceTensor gate_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},
                                                                        Stream(),
                                                                        sd);
      CAIF_Ops::SiLU(CachedGateOut(),gate_activated);

      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::Multiply(grad_hidden,gate_activated,grad_up);

      CAIF_DeviceTensor grad_gate_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},
                                                                             Stream(),
                                                                             sd);
      CAIF_Ops::Multiply(grad_hidden,CachedUpOut(),grad_gate_activated);

      CAIF_DeviceTensor grad_gate=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::SiLUBackward(CachedGateOut(),grad_gate_activated,grad_gate);

      // Push grad through gate + up FrozenLinear; both return grad_input
      // at Float32 (FrozenLinear::Backward contract). Cast back to sd
      // before summing with the StorageT-typed Add.
      CAIF_DeviceTensor gi_up_fp32=up_layer->Backward(grad_up,ctx);
      CAIF_DeviceTensor gi_gate_fp32=gate_layer->Backward(grad_gate,ctx);
      CAIF_DeviceTensor gi_up;
      CAIF_DeviceTensor gi_gate;
      if(gi_up_fp32.Dtype()==sd)
      {
        gi_up=std::move(gi_up_fp32);
      }
      else
      {
        gi_up=gi_up_fp32.To(sd);
      }
      if(gi_gate_fp32.Dtype()==sd)
      {
        gi_gate=std::move(gi_gate_fp32);
      }
      else
      {
        gi_gate=gi_gate_fp32.To(sd);
      }
      CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},Stream(),sd);
      CAIF_Ops::Add(gi_up,gi_gate,grad_input);
      return grad_input;
    }
    else
    {
      // grad_up = silu_backward(up_out, grad_hidden)
      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::SiLUBackward(CachedUpOut(),grad_hidden,grad_up);
      CAIF_DeviceTensor gi_fp32=up_layer->Backward(grad_up,ctx);
      if(gi_fp32.Dtype()==sd)
      {
        return gi_fp32;
      }
      return gi_fp32.To(sd);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::ParameterTensor(size_t)
{
  try
  {
    THROW_CAIFE("MoEFrozenExpert: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::ParameterTensor(size_t)const
{
  try
  {
    THROW_CAIFE("MoEFrozenExpert: no trainable parameters");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::GradientTensor(size_t)
{
  try
  {
    THROW_CAIFE("MoEFrozenExpert: no trainable parameters / gradients");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::GradientTensor(size_t)const
{
  try
  {
    THROW_CAIFE("MoEFrozenExpert: no trainable parameters / gradients");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::Description()const
{
  return std::string("CAIF_DeviceMoEFrozenExpert(input_dim=")
        +std::to_string(Config().InputDim())
        +",hidden_dim="
        +std::to_string(Config().HiddenDim())
        +",use_gated="
        +(Config().UseGated()?"true":"false")
        +",frozen=true)";
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceMoEFrozenExpert<ComputeT,StorageT>::ParameterNames(const std::string &)const
{
  return {};
}

#ifdef USE_CAIF_CUDA
template class CAIF_DeviceMoEFrozenExpert<float,float>;
template class CAIF_DeviceMoEFrozenExpert<float,__half>;
template class CAIF_DeviceMoEFrozenExpert<float,__nv_bfloat16>;
template class CAIF_DeviceMoEFrozenExpert<__half,float>;
template class CAIF_DeviceMoEFrozenExpert<__half,__half>;
template class CAIF_DeviceMoEFrozenExpert<__half,__nv_bfloat16>;
template class CAIF_DeviceMoEFrozenExpert<__nv_bfloat16,float>;
template class CAIF_DeviceMoEFrozenExpert<__nv_bfloat16,__half>;
template class CAIF_DeviceMoEFrozenExpert<__nv_bfloat16,__nv_bfloat16>;
#else
template class CAIF_DeviceMoEFrozenExpert<float,float>;
#endif

}//end instance namespace
