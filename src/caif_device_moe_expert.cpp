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
// Device-resident MoE Expert implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_expert.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include "caif_role_registry.h"
#include <random>
#include <cmath>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoEExpert<ComputeT,StorageT>::CAIF_DeviceMoEExpert(const CAIF_DeviceMoEExpertConfig &config,CAIF_CudaStream &stream)
  :CAIF_DeviceMoEExpertBase<ComputeT,StorageT>(stream)
  ,_config(config)
  ,_use_projections(false)
{
  try
  {
    // Validate config
    if(Config().InputDim()==0)
    {
      THROW_CAIFE("MoEExpert: input_dim must be > 0");
    }
    if(Config().HiddenDim()==0)
    {
      THROW_CAIFE("MoEExpert: hidden_dim must be > 0");
    }

    // Allocate weights at the templated storage dtype.
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    if(Config().UseGated()==true)
    {
      SetWGate(CAIF_DeviceTensor::Uninitialized({Config().InputDim(),Config().HiddenDim()},stream,sd));
      SetGradWGate(CAIF_DeviceTensor::Zeros({Config().InputDim(),Config().HiddenDim()},stream,sd));

      if(Config().UseBias()==true)
      {
        SetBGate(CAIF_DeviceTensor::Zeros({Config().HiddenDim()},stream,sd));
        SetGradBGate(CAIF_DeviceTensor::Zeros({Config().HiddenDim()},stream,sd));
      }
    }

    SetWUp(CAIF_DeviceTensor::Uninitialized({Config().InputDim(),Config().HiddenDim()},stream,sd));
    SetWDown(CAIF_DeviceTensor::Uninitialized({Config().HiddenDim(),Config().InputDim()},stream,sd));
    SetGradWUp(CAIF_DeviceTensor::Zeros({Config().InputDim(),Config().HiddenDim()},stream,sd));
    SetGradWDown(CAIF_DeviceTensor::Zeros({Config().HiddenDim(),Config().InputDim()},stream,sd));

    if(Config().UseBias()==true)
    {
      SetBUp(CAIF_DeviceTensor::Zeros({Config().HiddenDim()},stream,sd));
      SetBDown(CAIF_DeviceTensor::Zeros({Config().InputDim()},stream,sd));
      SetGradBUp(CAIF_DeviceTensor::Zeros({Config().HiddenDim()},stream,sd));
      SetGradBDown(CAIF_DeviceTensor::Zeros({Config().InputDim()},stream,sd));
    }

    // Xavier initialization — host fp32 stage, then To(sd) when needed.
    std::random_device rd;
    std::mt19937 gen(rd());

    float scale_up=std::sqrt(2.0f/static_cast<float>(Config().InputDim()+Config().HiddenDim()));
    float scale_down=std::sqrt(2.0f/static_cast<float>(Config().HiddenDim()+Config().InputDim()));

    std::normal_distribution<float> dist_up(0.0f,scale_up);
    std::normal_distribution<float> dist_down(0.0f,scale_down);

    // Initialize w_up — host-side fp32 sample, host-converted to sd at
    // upload (no device-side fp32 staging tensor).
    {
      std::vector<float> data(Config().InputDim()*Config().HiddenDim());
      for(size_t i=0;i<data.size();++i)
      {
        data[i]=dist_up(gen);
      }
      WUpMut().CopyFromHostFp32(data.data(),data.size());
    }

    // Initialize w_down
    {
      std::vector<float> data(Config().HiddenDim()*Config().InputDim());
      for(size_t i=0;i<data.size();++i)
      {
        data[i]=dist_down(gen);
      }
      WDownMut().CopyFromHostFp32(data.data(),data.size());
    }

    // Initialize w_gate if gated
    if(Config().UseGated()==true)
    {
      std::vector<float> data(Config().InputDim()*Config().HiddenDim());
      for(size_t i=0;i<data.size();++i)
      {
        data[i]=dist_up(gen);
      }
      WGateMut().CopyFromHostFp32(data.data(),data.size());
    }

    Stream().Synchronize();
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoEExpert<ComputeT,StorageT>::CAIF_DeviceMoEExpert(const CAIF_DeviceMoEExpertConfig &config,
                                         MoEExpertProjections_t projections,
                                         CAIF_CudaStream &stream):CAIF_DeviceMoEExpertBase<ComputeT,StorageT>(stream),
                                                                  _config(config),
                                                                  _projections(std::move(projections)),
                                                                  _use_projections(true)
{
  try
  {
    if(Config().InputDim()==0)
    {
      THROW_CAIFE("MoEExpert: input_dim must be > 0");
    }
    if(Config().HiddenDim()==0)
    {
      THROW_CAIFE("MoEExpert: hidden_dim must be > 0");
    }
    if(Projections().up==nullptr)
    {
      THROW_CAIFE("MoEExpert: up projection must not be null");
    }
    if(Projections().down==nullptr)
    {
      THROW_CAIFE("MoEExpert: down projection must not be null");
    }
    if(Config().UseGated()==true&&Projections().gate==nullptr)
    {
      THROW_CAIFE("MoEExpert: gate projection required for gated mode");
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoEExpert<ComputeT,StorageT>::CAIF_DeviceMoEExpert(CAIF_DeviceMoEExpert<ComputeT,StorageT> &&other)
  :CAIF_DeviceMoEExpertBase<ComputeT,StorageT>(std::move(other))
  ,_config(other._config)
  ,_projections(std::move(other._projections))
  ,_use_projections(other._use_projections)
  ,_w_gate(std::move(other._w_gate))
  ,_w_up(std::move(other._w_up))
  ,_w_down(std::move(other._w_down))
  ,_b_gate(std::move(other._b_gate))
  ,_b_up(std::move(other._b_up))
  ,_b_down(std::move(other._b_down))
  ,_grad_w_gate(std::move(other._grad_w_gate))
  ,_grad_w_up(std::move(other._grad_w_up))
  ,_grad_w_down(std::move(other._grad_w_down))
  ,_grad_b_gate(std::move(other._grad_b_gate))
  ,_grad_b_up(std::move(other._grad_b_up))
  ,_grad_b_down(std::move(other._grad_b_down))
  ,_cached_input(std::move(other._cached_input))
  ,_cached_gate_out(std::move(other._cached_gate_out))
  ,_cached_up_out(std::move(other._cached_up_out))
  ,_cached_hidden(std::move(other._cached_hidden))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceMoEExpert<ComputeT,StorageT> &CAIF_DeviceMoEExpert<ComputeT,StorageT>::operator=(CAIF_DeviceMoEExpert<ComputeT,StorageT> &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceMoEExpertBase<ComputeT,StorageT>::operator=(std::move(other));
    SetConfig(other.Config());
    SetProjections(std::move(other.ProjectionsMut()));
    SetUseProjections(other.UseProjections());
    SetWGate(std::move(other.WGateMut()));
    SetWUp(std::move(other.WUpMut()));
    SetWDown(std::move(other.WDownMut()));
    SetBGate(std::move(other.BGateMut()));
    SetBUp(std::move(other.BUpMut()));
    SetBDown(std::move(other.BDownMut()));
    SetGradWGate(std::move(other.GradWGateMut()));
    SetGradWUp(std::move(other.GradWUpMut()));
    SetGradWDown(std::move(other.GradWDownMut()));
    SetGradBGate(std::move(other.GradBGateMut()));
    SetGradBUp(std::move(other.GradBUpMut()));
    SetGradBDown(std::move(other.GradBDownMut()));
    SetCachedInput(std::move(other.CachedInputMut()));
    SetCachedGateOut(std::move(other.CachedGateOutMut()));
    SetCachedUpOut(std::move(other.CachedUpOutMut()));
    SetCachedHidden(std::move(other.CachedHiddenMut()));
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoEExpert<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,CAIF_RunContext &ctx)
{
  try
  {
    // Input: [num_tokens, input_dim]
    const auto &shape=input.Shape();
    if(shape.size()!=2||shape[1]!=Config().InputDim())
    {
      THROW_CAIFE("MoEExpert::Forward: expected input shape [N, input_dim]");
    }

    const uint32_t num_tokens=shape[0];
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    // Cache input for backward (only when not using projections)
    if(ctx.Training()==true&&UseProjections()==false)
    {
      SetCachedInput(input.Clone());
    }

    if(Config().UseGated()==true)
    {
      // Gated FFN (SwiGLU): output = (SiLU(gate) * up) @ down. This matches
      // HF's standard convention: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`
      // which is what every recent MoE (OLMoE / GLM-4-MoE / DSv2 / Qwen3MoE)
      // trains under and is what CAIF_DeviceFFN does via SwiGLUActivation::
      // Forward(gate, up, output). Swapping the silu target (silu(up) vs
      // silu(gate)) breaks parity vs the reference implementation on
      // every loaded MoE expert because the trained weights aren't
      // symmetric in gate/up.
      CAIF_DeviceTensor gate_out;
      if(UseProjections()==true)
      {
        gate_out=Projections().gate->Forward(input,ctx);
      }
      else
      {
        gate_out=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
        CAIF_Ops::MatMul(input,WGate(),gate_out,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(gate_out,BGate(),gate_out);
        }
      }

      CAIF_DeviceTensor up_out;
      if(UseProjections()==true)
      {
        up_out=Projections().up->Forward(input,ctx);
      }
      else
      {
        up_out=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
        CAIF_Ops::MatMul(input,WUp(),up_out,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(up_out,BUp(),up_out);
        }
      }

      // SwiGLU: silu(gate) * up. Standard HF convention.
      CAIF_DeviceTensor gate_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},
                                                                        Stream(),
                                                                        sd);
      CAIF_Ops::SiLU(gate_out,gate_activated);

      CAIF_DeviceTensor hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::Multiply(gate_activated,up_out,hidden);

      if(ctx.Training()==true)
      {
        SetCachedGateOut(std::move(gate_out));
        SetCachedUpOut(std::move(up_out));
        SetCachedHidden(hidden.Clone());
      }

      // Down projection
      CAIF_DeviceTensor output;
      if(UseProjections()==true)
      {
        output=Projections().down->Forward(hidden,ctx);
      }
      else
      {
        output=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},Stream(),sd);
        CAIF_Ops::MatMul(hidden,WDown(),output,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(output,BDown(),output);
        }
      }

      return output;
    }
    else
    {
      // Standard FFN: output = SiLU(input @ w_up + b_up) @ w_down + b_down
      CAIF_DeviceTensor up_out;
      if(UseProjections()==true)
      {
        up_out=Projections().up->Forward(input,ctx);
      }
      else
      {
        up_out=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
        CAIF_Ops::MatMul(input,WUp(),up_out,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(up_out,BUp(),up_out);
        }
      }

      CAIF_DeviceTensor hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::SiLU(up_out,hidden);

      if(ctx.Training()==true)
      {
        SetCachedUpOut(std::move(up_out));
        SetCachedHidden(hidden.Clone());
      }

      CAIF_DeviceTensor output;
      if(UseProjections()==true)
      {
        output=Projections().down->Forward(hidden,ctx);
      }
      else
      {
        output=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},Stream(),sd);
        CAIF_Ops::MatMul(hidden,WDown(),output,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(output,BDown(),output);
        }
      }

      return output;
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoEExpert<ComputeT,StorageT>::ForwardInto(const CAIF_DeviceTensor &input,
                                                          CAIF_DeviceTensor &output,
                                                          CAIF_RunContext &ctx)
{
  try
  {
    // Same flow as ForwardImpl, but the final down-projection writes into
    // `output` (caller-provided slice). Saves a per-expert allocation and
    // a per-expert D2D cudaMemcpyAsync in the MoE layer loop.
    const auto &shape=input.Shape();
    if(shape.size()!=2||shape[1]!=Config().InputDim())
    {
      THROW_CAIFE("MoEExpert::ForwardInto: expected input shape [N, input_dim]");
    }
    const auto &out_shape=output.Shape();
    if(out_shape.size()!=2||out_shape[0]!=shape[0]||out_shape[1]!=Config().InputDim())
    {
      THROW_CAIFE("MoEExpert::ForwardInto: output shape must be [N, input_dim]");
    }

    const uint32_t num_tokens=shape[0];
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    if(ctx.Training()==true&&UseProjections()==false)
    {
      SetCachedInput(input.Clone());
    }

    if(Config().UseGated()==true)
    {
      CAIF_DeviceTensor gate_out;
      if(UseProjections()==true)
      {
        gate_out=Projections().gate->Forward(input,ctx);
      }
      else
      {
        gate_out=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
        CAIF_Ops::MatMul(input,WGate(),gate_out,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(gate_out,BGate(),gate_out);
        }
      }

      CAIF_DeviceTensor up_out;
      if(UseProjections()==true)
      {
        up_out=Projections().up->Forward(input,ctx);
      }
      else
      {
        up_out=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
        CAIF_Ops::MatMul(input,WUp(),up_out,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(up_out,BUp(),up_out);
        }
      }

      // SwiGLU: silu(gate) * up. Same correction as the ForwardImpl path
      // above; see the comment there.
      CAIF_DeviceTensor gate_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},
                                                                        Stream(),
                                                                        sd);
      CAIF_Ops::SiLU(gate_out,gate_activated);

      CAIF_DeviceTensor hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::Multiply(gate_activated,up_out,hidden);

      if(ctx.Training()==true)
      {
        SetCachedGateOut(std::move(gate_out));
        SetCachedUpOut(std::move(up_out));
        SetCachedHidden(hidden.Clone());
      }

      // Down projection writes directly into the caller-provided output.
      // The projections branch can't take advantage (it allocates inside
      // its own Forward), so it falls back to a copy. The bench cells use
      // the non-projections constructor, so the fast path applies there.
      if(UseProjections()==true)
      {
        CAIF_DeviceTensor proj_out=Projections().down->Forward(hidden,ctx);
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(output.DeviceDataRaw(),
                        proj_out.DeviceDataRaw(),
                        static_cast<size_t>(num_tokens)*Config().InputDim()
                          *CAIF_DataType(sd).ElementSizeBytes(),
                        cudaMemcpyDeviceToDevice,
                        Stream().Handle());
#endif
      }
      else
      {
        CAIF_Ops::MatMul(hidden,WDown(),output,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(output,BDown(),output);
        }
      }
    }
    else
    {
      CAIF_DeviceTensor up_out;
      if(UseProjections()==true)
      {
        up_out=Projections().up->Forward(input,ctx);
      }
      else
      {
        up_out=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
        CAIF_Ops::MatMul(input,WUp(),up_out,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(up_out,BUp(),up_out);
        }
      }

      CAIF_DeviceTensor hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::SiLU(up_out,hidden);

      if(ctx.Training()==true)
      {
        SetCachedUpOut(std::move(up_out));
        SetCachedHidden(hidden.Clone());
      }

      if(UseProjections()==true)
      {
        CAIF_DeviceTensor proj_out=Projections().down->Forward(hidden,ctx);
#ifdef USE_CAIF_CUDA
        cudaMemcpyAsync(output.DeviceDataRaw(),
                        proj_out.DeviceDataRaw(),
                        static_cast<size_t>(num_tokens)*Config().InputDim()
                          *CAIF_DataType(sd).ElementSizeBytes(),
                        cudaMemcpyDeviceToDevice,
                        Stream().Handle());
#endif
      }
      else
      {
        CAIF_Ops::MatMul(hidden,WDown(),output,ctx,cdt);
        if(Config().UseBias()==true)
        {
          CAIF_Ops::AddBias(output,BDown(),output);
        }
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor CAIF_DeviceMoEExpert<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,CAIF_RunContext &ctx)
{
  try
  {
    // grad_output: [num_tokens, input_dim]
    const auto &shape=grad_output.Shape();
    const uint32_t num_tokens=shape[0];
    const CAIF_DataType::CAIF_DataType_e sd=StorageDtype();
    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();

    // Backward through down projection
    CAIF_DeviceTensor grad_hidden;
    if(UseProjections()==true)
    {
      grad_hidden=Projections().down->Backward(grad_output,ctx);
    }
    else
    {
      grad_hidden=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::MatMulTransposeB(grad_output,WDown(),grad_hidden,ctx,cdt);

      CAIF_DeviceTensor grad_w_down_batch=CAIF_DeviceTensor::Uninitialized({Config().HiddenDim(),
                                                                            Config().InputDim()},
                                                                          Stream(),sd);
      CAIF_Ops::MatMulTransposeA(CachedHidden(),grad_output,grad_w_down_batch,ctx,cdt);
      CAIF_Ops::Add(GradWDown(),grad_w_down_batch,GradWDownMut());

      if(Config().UseBias()==true)
      {
        CAIF_DeviceTensor grad_b_down_batch=CAIF_DeviceTensor::Uninitialized({Config().InputDim()},Stream(),sd);
        CAIF_Ops::SumAxis(grad_output,0,grad_b_down_batch);
        CAIF_Ops::Add(GradBDown(),grad_b_down_batch,GradBDownMut());
      }
    }

    CAIF_DeviceTensor grad_input;

    if(Config().UseGated()==true)
    {
      // Backward through SwiGLU (silu(gate) * up). Mirror the forward
      // semantics now in ForwardImpl / ForwardInto:
      //   hidden = silu(gate_out) * up_out
      //   grad_gate_act = grad_hidden * up_out
      //   grad_gate     = SiLUBackward(gate_out, grad_gate_act)
      //   grad_up       = grad_hidden * silu(gate_out)
      CAIF_DeviceTensor gate_activated=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},
                                                                        Stream(),
                                                                        sd);
      CAIF_Ops::SiLU(CachedGateOut(),gate_activated);

      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::Multiply(grad_hidden,gate_activated,grad_up);

      CAIF_DeviceTensor grad_gate_activated=
        CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::Multiply(grad_hidden,CachedUpOut(),grad_gate_activated);

      CAIF_DeviceTensor grad_gate=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::SiLUBackward(CachedGateOut(),grad_gate_activated,grad_gate);

      // Backward through gate and up projections
      if(UseProjections()==true)
      {
        CAIF_DeviceTensor gi_up=Projections().up->Backward(grad_up,ctx);
        CAIF_DeviceTensor gi_gate=Projections().gate->Backward(grad_gate,ctx);
        grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},Stream(),sd);
        CAIF_Ops::Add(gi_up,gi_gate,grad_input);
      }
      else
      {
        CAIF_DeviceTensor grad_w_up_batch=CAIF_DeviceTensor::Uninitialized({Config().InputDim(),
                                                                          Config().HiddenDim()},Stream(),sd);
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_up,grad_w_up_batch,ctx,cdt);
        CAIF_Ops::Add(GradWUp(),grad_w_up_batch,GradWUpMut());

        CAIF_DeviceTensor grad_w_gate_batch=CAIF_DeviceTensor::Uninitialized({Config().InputDim(),
                                                                              Config().HiddenDim()},
                                                                             Stream(),sd);
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_gate,grad_w_gate_batch,ctx,cdt);
        CAIF_Ops::Add(GradWGate(),grad_w_gate_batch,GradWGateMut());

        if(Config().UseBias()==true)
        {
          CAIF_DeviceTensor grad_b_up_batch=CAIF_DeviceTensor::Uninitialized({Config().HiddenDim()},Stream(),sd);
          CAIF_Ops::SumAxis(grad_up,0,grad_b_up_batch);
          CAIF_Ops::Add(GradBUp(),grad_b_up_batch,GradBUpMut());

          CAIF_DeviceTensor grad_b_gate_batch=CAIF_DeviceTensor::Uninitialized({Config().HiddenDim()},
                                                                               Stream(),
                                                                               sd);
          CAIF_Ops::SumAxis(grad_gate,0,grad_b_gate_batch);
          CAIF_Ops::Add(GradBGate(),grad_b_gate_batch,GradBGateMut());
        }

        CAIF_DeviceTensor grad_input_up=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},
                                                                         Stream(),
                                                                         sd);
        CAIF_Ops::MatMulTransposeB(grad_up,WUp(),grad_input_up,ctx,cdt);

        CAIF_DeviceTensor grad_input_gate=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},
                                                                           Stream(),
                                                                           sd);
        CAIF_Ops::MatMulTransposeB(grad_gate,WGate(),grad_input_gate,ctx,cdt);

        grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},Stream(),sd);
        CAIF_Ops::Add(grad_input_up,grad_input_gate,grad_input);
      }
    }
    else
    {
      // Backward through SiLU activation
      CAIF_DeviceTensor grad_up=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().HiddenDim()},Stream(),sd);
      CAIF_Ops::SiLUBackward(CachedUpOut(),grad_hidden,grad_up);

      if(UseProjections()==true)
      {
        grad_input=Projections().up->Backward(grad_up,ctx);
      }
      else
      {
        CAIF_DeviceTensor grad_w_up_batch=CAIF_DeviceTensor::Uninitialized({Config().InputDim(),
                                                                            Config().HiddenDim()},
                                                                          Stream(),sd);
        CAIF_Ops::MatMulTransposeA(CachedInput(),grad_up,grad_w_up_batch,ctx,cdt);
        CAIF_Ops::Add(GradWUp(),grad_w_up_batch,GradWUpMut());

        if(Config().UseBias()==true)
        {
          CAIF_DeviceTensor grad_b_up_batch=CAIF_DeviceTensor::Uninitialized({Config().HiddenDim()},Stream(),sd);
          CAIF_Ops::SumAxis(grad_up,0,grad_b_up_batch);
          CAIF_Ops::Add(GradBUp(),grad_b_up_batch,GradBUpMut());
        }

        grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,Config().InputDim()},Stream(),sd);
        CAIF_Ops::MatMulTransposeB(grad_up,WUp(),grad_input,ctx,cdt);
      }
    }

    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoEExpert<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    if(UseProjections()==true)
    {
      if(Config().UseGated()==true&&Projections().gate!=nullptr)
      {
        Projections().gate->ZeroGradients();
      }
      Projections().up->ZeroGradients();
      Projections().down->ZeroGradients();
    }
    else
    {
      if(Config().UseGated()==true)
      {
        GradWGateMut().FillZero();
        if(Config().UseBias()==true)
        {
          GradBGateMut().FillZero();
        }
      }

      GradWUpMut().FillZero();
      GradWDownMut().FillZero();

      if(Config().UseBias()==true)
      {
        GradBUpMut().FillZero();
        GradBDownMut().FillZero();
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMoEExpert<ComputeT,StorageT>::ParameterTensorCount()const
{
  if(UseProjections()==true)
  {
    size_t count=Projections().up->ParameterTensorCount()+
                 Projections().down->ParameterTensorCount();
    if(Config().UseGated()==true&&Projections().gate!=nullptr)
    {
      count+=Projections().gate->ParameterTensorCount();
    }
    return count;
  }
  size_t count=2;  // w_up, w_down
  if(Config().UseGated()==true)
  {
    count+=1;  // w_gate
  }
  if(Config().UseBias()==true)
  {
    count+=2;  // b_up, b_down
    if(Config().UseGated()==true)
    {
      count+=1;  // b_gate
    }
  }
  return count;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMoEExpert<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  if(UseProjections()==true)
  {
    size_t offset=0;
    if(Config().UseGated()==true&&Projections().gate!=nullptr)
    {
      const size_t count=Projections().gate->ParameterTensorCount();
      if(index<offset+count)
      {
        return Projections().gate->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=Projections().up->ParameterTensorCount();
      if(index<offset+count)
      {
        return Projections().up->ParameterTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=Projections().down->ParameterTensorCount();
      if(index<offset+count)
      {
        return Projections().down->ParameterTensor(index-offset);
      }
    }
    THROW_CAIFE("MoEExpert::ParameterTensor: index out of range");
  }
  size_t idx=0;
  if(Config().UseGated()==true)
  {
    if(index==idx)
    {
      return WGateMut();
    }
    ++idx;
  }
  if(index==idx)
  {
    return WUpMut();
  }
  ++idx;
  if(index==idx)
  {
    return WDownMut();
  }
  ++idx;
  if(Config().UseBias()==true)
  {
    if(Config().UseGated()==true)
    {
      if(index==idx)
      {
        return BGateMut();
      }
      ++idx;
    }
    if(index==idx)
    {
      return BUpMut();
    }
    ++idx;
    if(index==idx)
    {
      return BDownMut();
    }
  }
  THROW_CAIFE("MoEExpert::ParameterTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMoEExpert<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoEExpert*>(this)->ParameterTensor(index);
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &CAIF_DeviceMoEExpert<ComputeT,StorageT>::GradientTensor(size_t index)
{
  if(UseProjections()==true)
  {
    size_t offset=0;
    if(Config().UseGated()==true&&Projections().gate!=nullptr)
    {
      const size_t count=Projections().gate->ParameterTensorCount();
      if(index<offset+count)
      {
        return Projections().gate->GradientTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=Projections().up->ParameterTensorCount();
      if(index<offset+count)
      {
        return Projections().up->GradientTensor(index-offset);
      }
      offset+=count;
    }
    {
      const size_t count=Projections().down->ParameterTensorCount();
      if(index<offset+count)
      {
        return Projections().down->GradientTensor(index-offset);
      }
    }
    THROW_CAIFE("MoEExpert::GradientTensor: index out of range");
  }
  size_t idx=0;
  if(Config().UseGated()==true)
  {
    if(index==idx)
    {
      return GradWGateMut();
    }
    ++idx;
  }
  if(index==idx)
  {
    return GradWUpMut();
  }
  ++idx;
  if(index==idx)
  {
    return GradWDownMut();
  }
  ++idx;
  if(Config().UseBias()==true)
  {
    if(Config().UseGated()==true)
    {
      if(index==idx)
      {
        return GradBGateMut();
      }
      ++idx;
    }
    if(index==idx)
    {
      return GradBUpMut();
    }
    ++idx;
    if(index==idx)
    {
      return GradBDownMut();
    }
  }
  THROW_CAIFE("MoEExpert::GradientTensor: index out of range");
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &CAIF_DeviceMoEExpert<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoEExpert*>(this)->GradientTensor(index);
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceMoEExpert<ComputeT,StorageT>::TotalParameterCount()const
{
  if(UseProjections()==true)
  {
    size_t total=Projections().up->TotalParameterCount()+
                 Projections().down->TotalParameterCount();
    if(Config().UseGated()==true&&Projections().gate!=nullptr)
    {
      total+=Projections().gate->TotalParameterCount();
    }
    return total;
  }
  size_t count=0;
  count+=Config().InputDim()*Config().HiddenDim();  // w_up
  count+=Config().HiddenDim()*Config().InputDim();  // w_down
  if(Config().UseGated()==true)
  {
    count+=Config().InputDim()*Config().HiddenDim();  // w_gate
  }
  if(Config().UseBias()==true)
  {
    count+=Config().HiddenDim();  // b_up
    count+=Config().InputDim();   // b_down
    if(Config().UseGated()==true)
    {
      count+=Config().HiddenDim();  // b_gate
    }
  }
  return count;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceMoEExpert<ComputeT,StorageT>::Description()const
{
  std::string desc="MoEExpert[";
  desc+=std::to_string(Config().InputDim())+"->"+std::to_string(Config().HiddenDim());
  desc+="->"+std::to_string(Config().InputDim());
  if(Config().UseGated()==true)
  {
    desc+=",gated";
  }
  if(Config().UseBias()==true)
  {
    desc+=",bias";
  }
  desc+="]";
  return desc;
}

template<typename ComputeT,typename StorageT>
std::vector<std::string> CAIF_DeviceMoEExpert<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
  if(UseProjections()==true)
  {
    std::vector<std::string> names;
    std::vector<std::string> sub;
    if(Config().UseGated()==true&&Projections().gate!=nullptr)
    {
      sub=Projections().gate->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertGate_e)+".");
      names.insert(names.end(),sub.begin(),sub.end());
    }
    sub=Projections().up->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertUp_e)+".");
    names.insert(names.end(),sub.begin(),sub.end());
    sub=Projections().down->ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertDown_e)+".");
    names.insert(names.end(),sub.begin(),sub.end());
    return names;
  }
  std::vector<std::string> names;
  if(Config().UseGated()==true)
  {
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertGate_e));
  }
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertUp_e));
  names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertDown_e));
  if(Config().UseBias()==true)
  {
    if(Config().UseGated()==true)
    {
      names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertBiasGate_e));
    }
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertBiasUp_e));
    names.push_back(prefix+reg.Name(CAIF_ParamRole::Role_e::MoEExpertBiasDown_e));
  }
  return names;
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoEExpert<ComputeT,StorageT>::LoadWGate(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("MoEExpert::LoadWGate: not valid when using sub-projections");
    }
    if(Config().UseGated()==false)
    {
      THROW_CAIFE("MoEExpert::LoadWGate: requires use_gated=true");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2||
       shape[0]!=Config().InputDim()||
       shape[1]!=Config().HiddenDim())
    {
      THROW_CAIFE("MoEExpert::LoadWGate: shape mismatch, expected "
                  "[input_dim, hidden_dim]");
    }
    if(w.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("MoEExpert::LoadWGate: dtype mismatch, expected StorageDtype");
    }
    _w_gate=std::move(w);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoEExpert<ComputeT,StorageT>::LoadWUp(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("MoEExpert::LoadWUp: not valid when using sub-projections");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2||
       shape[0]!=Config().InputDim()||
       shape[1]!=Config().HiddenDim())
    {
      THROW_CAIFE("MoEExpert::LoadWUp: shape mismatch, expected "
                  "[input_dim, hidden_dim]");
    }
    if(w.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("MoEExpert::LoadWUp: dtype mismatch, expected StorageDtype");
    }
    _w_up=std::move(w);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceMoEExpert<ComputeT,StorageT>::LoadWDown(CAIF_DeviceTensor &&w)
{
  try
  {
    if(UseProjections()==true)
    {
      THROW_CAIFE("MoEExpert::LoadWDown: not valid when using sub-projections");
    }
    const std::vector<uint32_t> &shape=w.Shape();
    if(shape.size()!=2||
       shape[0]!=Config().HiddenDim()||
       shape[1]!=Config().InputDim())
    {
      THROW_CAIFE("MoEExpert::LoadWDown: shape mismatch, expected "
                  "[hidden_dim, input_dim]");
    }
    if(w.Dtype()!=StorageDtype())
    {
      THROW_CAIFE("MoEExpert::LoadWDown: dtype mismatch, expected StorageDtype");
    }
    _w_down=std::move(w);
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceMoEExpert<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceMoEExpert<float,__half>;
template class CAIF_DeviceMoEExpert<float,__nv_bfloat16>;
template class CAIF_DeviceMoEExpert<__half,float>;
template class CAIF_DeviceMoEExpert<__half,__half>;
template class CAIF_DeviceMoEExpert<__half,__nv_bfloat16>;
template class CAIF_DeviceMoEExpert<__nv_bfloat16,float>;
template class CAIF_DeviceMoEExpert<__nv_bfloat16,__half>;
template class CAIF_DeviceMoEExpert<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
